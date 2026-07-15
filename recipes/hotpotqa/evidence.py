"""Exact, official HotpotQA sentence evidence for the formal A0 contract.

The formal path never reconstructs sentence boundaries from paragraph text.
Sentence arrays and supporting-fact indices are read from a versioned SQLite
sidecar produced from the official HotpotQA distractor JSON files.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA_VERSION = "hotpotqa-official-sentence-v1"
EVIDENCE_SIDECAR_SCHEMA_VERSION = 1
EVIDENCE_SIDECAR_FILENAME = "hotpotqa_evidence_v1.sqlite3"
_EVIDENCE_ID_PREFIX = f"{EVIDENCE_SCHEMA_VERSION}:"


def normalize_title(value: Any) -> str:
    """Return the frozen title representation used only for audit joins."""

    title = unicodedata.normalize("NFKC", str(value or "")).replace("_", " ")
    return " ".join(title.split()).casefold()


def make_evidence_id(pid: Any, sentence_id: Any) -> str:
    """Serialize an existing global paragraph PID and official sentence index."""

    try:
        paragraph_id = int(pid)
        index = int(sentence_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evidence pid/sentence_id must be integers: {pid!r}, {sentence_id!r}") from exc
    if paragraph_id < 0 or index < 0:
        raise ValueError(f"Evidence pid/sentence_id must be non-negative: {paragraph_id}, {index}")
    return f"{_EVIDENCE_ID_PREFIX}{paragraph_id}:{index}"


def parse_evidence_id(evidence_id: str) -> tuple[int, int]:
    """Invert :func:`make_evidence_id` and reject non-canonical values."""

    value = str(evidence_id)
    if not value.startswith(_EVIDENCE_ID_PREFIX):
        raise ValueError(f"Unsupported HotpotQA evidence ID: {evidence_id!r}")
    raw_pid, separator, raw_sentence_id = value[len(_EVIDENCE_ID_PREFIX) :].partition(":")
    if not separator:
        raise ValueError(f"Malformed HotpotQA evidence ID: {evidence_id!r}")
    try:
        pid = int(raw_pid)
        sentence_id = int(raw_sentence_id)
    except ValueError as exc:
        raise ValueError(f"Malformed HotpotQA evidence ID: {evidence_id!r}") from exc
    if make_evidence_id(pid, sentence_id) != value:
        raise ValueError(f"Non-canonical HotpotQA evidence ID: {evidence_id!r}")
    return pid, sentence_id


def sentence_evidence_records(
    *,
    pid: Any,
    title: Any,
    sentences: Iterable[Any],
) -> list[dict[str, Any]]:
    """Create records from official source sentences without segmenting text."""

    paragraph_id = int(pid)
    clean_title = str(title or "")
    normalized_title = normalize_title(clean_title)
    return [
        {
            "evidence_id": make_evidence_id(paragraph_id, sentence_id),
            "pid": paragraph_id,
            "title": clean_title,
            "normalized_title": normalized_title,
            "sentence_id": sentence_id,
            "text": str(sentence),
        }
        for sentence_id, sentence in enumerate(sentences)
    ]


def is_sentence_evidence_record(record: Any) -> bool:
    """Return whether one runtime sentence record has a valid official ID."""

    if not isinstance(record, dict):
        return False
    try:
        expected = make_evidence_id(record.get("pid"), record.get("sentence_id"))
    except (TypeError, ValueError):
        return False
    return (
        record.get("evidence_id") == expected
        and record.get("normalized_title") == normalize_title(record.get("title"))
        and isinstance(record.get("text"), str)
    )


@dataclass(frozen=True)
class SampleEvidence:
    """Verifier/evaluation identity for one parquet row."""

    split: str
    row_index: int
    dataset_question_id: str
    official_qid: str
    question: str
    gold_evidence_ids: tuple[str, ...]
    unresolved_gold_facts: tuple[dict[str, Any], ...]

    @property
    def sample_key(self) -> str:
        return f"{self.split}:{self.row_index}"


class OfficialEvidenceStore:
    """Read-only access to the exact official sentence sidecar.

    A thread-local read-only SQLite connection keeps per-query lookup cheap and
    safe when multiple async trajectories share one retrieval process.
    """

    def __init__(self, path: str | Path, *, verify_integrity: bool = True):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"HotpotQA evidence sidecar not found: {self.path}")
        self._local = threading.local()
        connection = self._connect()
        try:
            if verify_integrity:
                result = connection.execute("PRAGMA integrity_check").fetchone()
                if result is None or result[0] != "ok":
                    raise ValueError(f"HotpotQA evidence sidecar integrity check failed: {result}")
            self.metadata = {
                str(key): str(value)
                for key, value in connection.execute("SELECT key, value FROM metadata")
            }
        finally:
            connection.close()
        schema_version = int(self.metadata.get("sidecar_schema_version", -1))
        evidence_version = self.metadata.get("evidence_schema_version")
        if schema_version != EVIDENCE_SIDECAR_SCHEMA_VERSION:
            raise ValueError(
                f"Evidence sidecar schema is {schema_version}; expected {EVIDENCE_SIDECAR_SCHEMA_VERSION}"
            )
        if evidence_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"Evidence ID schema is {evidence_version!r}; expected {EVIDENCE_SCHEMA_VERSION!r}"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30.0)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            self._local.connection = connection
        return connection

    @property
    def paragraph_count(self) -> int:
        return int(self.metadata["paragraph_count"])

    def sample_count(self, split: str) -> int:
        key = f"sample_count_{split}"
        if key in self.metadata:
            return int(self.metadata[key])
        row = self._connection().execute(
            "SELECT COUNT(*) FROM samples WHERE split = ?", (str(split),)
        ).fetchone()
        return int(row[0]) if row else 0

    def paragraph_evidence(
        self,
        pid: Any,
        *,
        expected_title: str | None = None,
        expected_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact source sentences for a corpus PID and verify alignment."""

        paragraph_id = int(pid)
        row = self._connection().execute(
            "SELECT title, normalized_title, sentences_json, joined_text "
            "FROM paragraphs WHERE pid = ?",
            (paragraph_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Evidence sidecar has no paragraph PID {paragraph_id}")
        title, normalized, sentences_json, joined_text = row
        if expected_title is not None and str(title) != str(expected_title):
            raise ValueError(
                f"Evidence title mismatch for PID {paragraph_id}: {title!r} != {expected_title!r}"
            )
        if expected_text is not None and str(joined_text) != str(expected_text):
            raise ValueError(f"Evidence paragraph text mismatch for PID {paragraph_id}")
        sentences = json.loads(sentences_json)
        records = sentence_evidence_records(pid=paragraph_id, title=title, sentences=sentences)
        if any(record["normalized_title"] != normalized for record in records):
            raise ValueError(f"Evidence normalized title mismatch for PID {paragraph_id}")
        return records

    def sample(self, split: str, row_index: Any) -> SampleEvidence:
        """Load the official qid and gold sentence IDs for one parquet row."""

        index = int(row_index)
        row = self._connection().execute(
            "SELECT dataset_question_id, official_qid, question, gold_evidence_json, "
            "unresolved_gold_json "
            "FROM samples WHERE split = ? AND row_index = ?",
            (str(split), index),
        ).fetchone()
        if row is None:
            raise KeyError(f"Evidence sidecar has no sample {split}:{index}")
        dataset_question_id, official_qid, question, gold_json, unresolved_json = row
        gold = tuple(str(value) for value in json.loads(gold_json))
        unresolved = tuple(dict(value) for value in json.loads(unresolved_json))
        if not gold and not unresolved:
            raise ValueError(f"Evidence sidecar sample {split}:{index} has no gold annotations")
        for evidence_id in gold:
            parse_evidence_id(evidence_id)
        return SampleEvidence(
            split=str(split),
            row_index=index,
            dataset_question_id=str(dataset_question_id),
            official_qid=str(official_qid),
            question=str(question),
            gold_evidence_ids=gold,
            unresolved_gold_facts=unresolved,
        )

    def list_samples(self, split: str, *, limit: int | None = None) -> list[SampleEvidence]:
        query = (
            "SELECT row_index, dataset_question_id, official_qid, question, gold_evidence_json, "
            "unresolved_gold_json "
            "FROM samples WHERE split = ? ORDER BY row_index"
        )
        parameters: list[Any] = [str(split)]
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(int(limit))
        rows = self._connection().execute(query, parameters).fetchall()
        return [
            SampleEvidence(
                split=str(split),
                row_index=int(row[0]),
                dataset_question_id=str(row[1]),
                official_qid=str(row[2]),
                question=str(row[3]),
                gold_evidence_ids=tuple(str(value) for value in json.loads(row[4])),
                unresolved_gold_facts=tuple(dict(value) for value in json.loads(row[5])),
            )
            for row in rows
        ]


def coerce_bool(value: Any, *, name: str = "value") -> bool:
    """Parse a bool without treating the string ``"false"`` as truthy."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")
