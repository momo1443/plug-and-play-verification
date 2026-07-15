#!/usr/bin/env python3
"""Build exact HotpotQA sentence evidence without rebuilding FAISS artifacts.

The builder aligns official distractor context paragraphs to the existing
``hpqa_corpus.jsonl`` by exact ``(title, joined source sentences)`` content.
It fails closed on any sample, paragraph, answer, or sentence-index mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ijson
import pyarrow.parquet as pq

from recipes.hotpotqa.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_SIDECAR_FILENAME,
    EVIDENCE_SIDECAR_SCHEMA_VERSION,
    make_evidence_id,
    normalize_title,
)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _paragraph_digest(title: str, text: str) -> bytes:
    payload = json.dumps([title, text], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _extract_question(parquet_row: Mapping[str, Any]) -> str:
    prompt = parquet_row.get("prompt") or []
    if isinstance(prompt, Mapping):
        prompt = [prompt]
    if isinstance(prompt, (list, tuple)):
        for message in reversed(prompt):
            if isinstance(message, Mapping) and message.get("content") is not None:
                if message.get("role") in (None, "user"):
                    return str(message["content"]).strip()
    raise ValueError("Parquet row has no user question")


def _extract_ground_truth(parquet_row: Mapping[str, Any]) -> str:
    reward_model = parquet_row.get("reward_model") or {}
    if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") is None:
        raise ValueError("Parquet row has no reward_model.ground_truth")
    return str(reward_model["ground_truth"])


def _parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["prompt", "reward_model", "extra_info"], batch_size=2048):
        yield from batch.to_pylist()


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(file_path for file_path in path.rglob("*") if file_path.is_file())
        if files:
            return files
    raise FileNotFoundError(f"Official HotpotQA source is missing or empty: {path}")


def _official_rows(path: Path) -> Iterator[dict[str, Any]]:
    for source_file in _source_files(path):
        if source_file.suffix == ".parquet":
            parquet = pq.ParquetFile(source_file)
            for batch in parquet.iter_batches(batch_size=1024):
                yield from batch.to_pylist()
        elif source_file.suffix == ".json":
            with source_file.open("rb") as handle:
                yield from ijson.items(handle, "item")
        else:
            raise ValueError(f"Unsupported official source file: {source_file}")


def _source_sha256(path: Path) -> tuple[str, list[dict[str, Any]]]:
    files = _source_files(path)
    if path.is_file():
        digest = sha256_file(path)
        return digest, [{"path": path.name, "sha256": digest, "bytes": path.stat().st_size}]
    tree_digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for source_file in files:
        relative = source_file.relative_to(path).as_posix()
        digest = sha256_file(source_file)
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
        records.append({"path": relative, "sha256": digest, "bytes": source_file.stat().st_size})
    return tree_digest.hexdigest(), records


def _context_rows(example: Mapping[str, Any]) -> Iterator[tuple[str, list[str]]]:
    context = example.get("context") or []
    if isinstance(context, Mapping) and "title" in context and "sentences" in context:
        for title, sentences in zip(context["title"], context["sentences"], strict=False):
            yield str(title).strip(), _canonical_source_sentences(sentences)
        return
    for item in context:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Malformed official context row: {item!r}")
        yield str(item[0]).strip(), _canonical_source_sentences(item[1])


def _canonical_source_sentences(sentences: Any) -> list[str]:
    """Mirror the corpus builder's paragraph-edge ``strip`` without re-splitting."""

    values = [str(sentence) for sentence in sentences]
    if values:
        values[0] = values[0].lstrip()
        values[-1] = values[-1].rstrip()
    return values


def _supporting_facts(example: Mapping[str, Any]) -> Iterator[tuple[str, int]]:
    facts = example.get("supporting_facts") or []
    if isinstance(facts, Mapping) and "title" in facts and "sent_id" in facts:
        facts = zip(facts["title"], facts["sent_id"], strict=False)
    for fact in facts:
        if not isinstance(fact, (list, tuple)) or len(fact) != 2:
            raise ValueError(f"Malformed supporting fact: {fact!r}")
        yield str(fact[0]).strip(), int(fact[1])


def _load_corpus_map(corpus_path: Path) -> tuple[dict[bytes, int], int]:
    digest_to_pid: dict[bytes, int] = {}
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Corpus contains a blank row at line {line_number}")
            record = json.loads(line)
            title = str(record.get("title", ""))
            text = str(record.get("text", ""))
            if not title or not text:
                raise ValueError(f"Corpus line {line_number} is missing title/text")
            pid = line_number - 1
            digest = _paragraph_digest(title, text)
            previous = digest_to_pid.setdefault(digest, pid)
            if previous != pid:
                raise ValueError(f"Corpus has a duplicate exact paragraph at PIDs {previous} and {pid}")
    if not digest_to_pid:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    return digest_to_pid, len(digest_to_pid)


def _create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        PRAGMA temp_store=FILE;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE paragraphs (
            pid INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            sentences_json TEXT NOT NULL,
            joined_text TEXT NOT NULL
        );
        CREATE TABLE samples (
            split TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            dataset_question_id TEXT NOT NULL,
            official_qid TEXT NOT NULL UNIQUE,
            question TEXT NOT NULL,
            gold_evidence_json TEXT NOT NULL,
            unresolved_gold_json TEXT NOT NULL,
            PRIMARY KEY (split, row_index)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _insert_paragraph(
    connection: sqlite3.Connection,
    *,
    pid: int,
    title: str,
    sentences: list[str],
    joined_text: str,
) -> None:
    sentences_json = json.dumps(sentences, ensure_ascii=False, separators=(",", ":"))
    connection.execute(
        "INSERT OR IGNORE INTO paragraphs "
        "(pid, title, normalized_title, sentences_json, joined_text) VALUES (?, ?, ?, ?, ?)",
        (pid, title, normalize_title(title), sentences_json, joined_text),
    )
    stored = connection.execute(
        "SELECT title, sentences_json, joined_text FROM paragraphs WHERE pid = ?", (pid,)
    ).fetchone()
    if stored != (title, sentences_json, joined_text):
        raise ValueError(f"Official sources disagree on sentence boundaries for corpus PID {pid}")


def _align_split(
    connection: sqlite3.Connection,
    *,
    split: str,
    official_source: Path,
    parquet_path: Path,
    digest_to_pid: dict[bytes, int],
) -> dict[str, int]:
    parquet_iterator = _parquet_rows(parquet_path)
    row_count = 0
    total_gold_facts = 0
    mapped_gold_facts = 0
    unresolved_gold_facts = 0
    samples_with_unresolved = 0
    for row_index, example in enumerate(_official_rows(official_source)):
        try:
            parquet_row = next(parquet_iterator)
        except StopIteration as exc:
            raise ValueError(f"Official {split} has more rows than {parquet_path}") from exc

        official_qid = str(example.get("_id") or example.get("id") or "").strip()
        if not official_qid:
            raise ValueError(f"Official {split}:{row_index} has no qid")
        question = str(example.get("question") or "").strip()
        answer = str(example.get("answer") or "")
        if question != _extract_question(parquet_row):
            raise ValueError(f"Question mismatch at {split}:{row_index} ({official_qid})")
        if answer != _extract_ground_truth(parquet_row):
            raise ValueError(f"Answer mismatch at {split}:{row_index} ({official_qid})")

        extra_info = parquet_row.get("extra_info") or {}
        if not isinstance(extra_info, Mapping):
            raise ValueError(f"Parquet extra_info is malformed at {split}:{row_index}")
        if int(extra_info.get("index", -1)) != row_index:
            raise ValueError(f"Parquet row index mismatch at {split}:{row_index}")
        if str(extra_info.get("split")) != split:
            raise ValueError(f"Parquet split mismatch at {split}:{row_index}")
        dataset_question_id = str(extra_info.get("question_id") or "")
        if not dataset_question_id:
            raise ValueError(f"Parquet question_id is empty at {split}:{row_index}")

        context_by_title: dict[str, tuple[int, int]] = {}
        for title, sentences in _context_rows(example):
            joined_text = " ".join(sentences).strip()
            if not title or not joined_text:
                raise ValueError(f"Empty official paragraph at {split}:{row_index}")
            digest = _paragraph_digest(title, joined_text)
            if digest not in digest_to_pid:
                raise ValueError(
                    f"Official paragraph does not exactly match the frozen corpus at "
                    f"{split}:{row_index}, title={title!r}"
                )
            pid = digest_to_pid[digest]
            _insert_paragraph(
                connection,
                pid=pid,
                title=title,
                sentences=sentences,
                joined_text=joined_text,
            )
            previous = context_by_title.setdefault(title, (pid, len(sentences)))
            if previous != (pid, len(sentences)):
                raise ValueError(f"Ambiguous duplicate context title {title!r} at {split}:{row_index}")

        gold_ids: list[str] = []
        unresolved: list[dict[str, Any]] = []
        for title, sentence_id in _supporting_facts(example):
            total_gold_facts += 1
            if title not in context_by_title:
                unresolved.append(
                    {
                        "title": title,
                        "sentence_id": sentence_id,
                        "reason": "official_title_missing_from_context",
                    }
                )
                unresolved_gold_facts += 1
                continue
            pid, sentence_count = context_by_title[title]
            if sentence_id < 0 or sentence_id >= sentence_count:
                unresolved.append(
                    {
                        "title": title,
                        "sentence_id": sentence_id,
                        "reason": "official_sentence_id_out_of_range",
                        "source_sentence_count": sentence_count,
                    }
                )
                unresolved_gold_facts += 1
                continue
            evidence_id = make_evidence_id(pid, sentence_id)
            if evidence_id not in gold_ids:
                gold_ids.append(evidence_id)
            mapped_gold_facts += 1
        if not gold_ids and not unresolved:
            raise ValueError(f"Official sample has no supporting facts at {split}:{row_index}")
        if unresolved:
            samples_with_unresolved += 1

        connection.execute(
            "INSERT INTO samples "
            "(split, row_index, dataset_question_id, official_qid, question, gold_evidence_json, "
            "unresolved_gold_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                split,
                row_index,
                dataset_question_id,
                official_qid,
                question,
                json.dumps(gold_ids, ensure_ascii=False, separators=(",", ":")),
                json.dumps(unresolved, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        row_count += 1
        if row_count % 5000 == 0:
            connection.commit()
            print(f"Aligned {split}: {row_count} samples", flush=True)

    try:
        next(parquet_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(f"Parquet {parquet_path} has more rows than official {split}")
    return {
        "samples": row_count,
        "total_gold_facts": total_gold_facts,
        "mapped_gold_facts": mapped_gold_facts,
        "unresolved_gold_facts": unresolved_gold_facts,
        "samples_with_unresolved": samples_with_unresolved,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_evidence_sidecar(
    *,
    train_source: Path,
    validation_source: Path,
    train_parquet: Path,
    validation_parquet: Path,
    corpus_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the sidecar and return its companion manifest."""

    inputs = {
        "official_train_source": train_source,
        "official_validation_source": validation_source,
        "train_parquet": train_parquet,
        "validation_parquet": validation_parquet,
        "corpus_jsonl": corpus_path,
    }
    for label, path in inputs.items():
        if label.startswith("official_"):
            _source_files(path)
        elif not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".manifest.json")
    if (output_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Evidence sidecar already exists: {output_path}. Pass --overwrite to rebuild it."
        )
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    print(f"Indexing frozen corpus digests: {corpus_path}", flush=True)
    digest_to_pid, corpus_count = _load_corpus_map(corpus_path)
    source_hashes: dict[str, str] = {}
    source_file_records: dict[str, list[dict[str, Any]]] = {}
    for label, path in inputs.items():
        if label.startswith("official_"):
            source_hashes[label], source_file_records[label] = _source_sha256(path)
        else:
            source_hashes[label] = sha256_file(path)
            source_file_records[label] = [
                {"path": path.name, "sha256": source_hashes[label], "bytes": path.stat().st_size}
            ]

    connection = _create_database(temporary)
    try:
        train_stats = _align_split(
            connection,
            split="train",
            official_source=train_source,
            parquet_path=train_parquet,
            digest_to_pid=digest_to_pid,
        )
        validation_stats = _align_split(
            connection,
            split="validation",
            official_source=validation_source,
            parquet_path=validation_parquet,
            digest_to_pid=digest_to_pid,
        )
        paragraph_count = int(connection.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0])
        if paragraph_count != corpus_count:
            raise ValueError(
                f"Official sources cover {paragraph_count} of {corpus_count} frozen corpus paragraphs"
            )

        metadata = {
            "sidecar_schema_version": str(EVIDENCE_SIDECAR_SCHEMA_VERSION),
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "paragraph_count": str(paragraph_count),
            "sample_count_train": str(train_stats["samples"]),
            "sample_count_validation": str(validation_stats["samples"]),
            "gold_fact_count_train": str(train_stats["total_gold_facts"]),
            "gold_fact_count_validation": str(validation_stats["total_gold_facts"]),
            "unresolved_gold_fact_count_train": str(train_stats["unresolved_gold_facts"]),
            "unresolved_gold_fact_count_validation": str(
                validation_stats["unresolved_gold_facts"]
            ),
            **{f"sha256_{key}": value for key, value in source_hashes.items()},
        }
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError(f"Built sidecar failed integrity check: {integrity}")
        connection.execute("VACUUM")
    except BaseException:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    connection.close()
    os.replace(temporary, output_path)

    manifest: dict[str, Any] = {
        "schema_version": EVIDENCE_SIDECAR_SCHEMA_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "recipes.hotpotqa.build_evidence_sidecar",
        "alignment": "exact title plus exact joined official sentences",
        "inputs": {
            label: {
                "path": str(path.resolve()),
                "sha256": source_hashes[label],
                "files": source_file_records[label],
            }
            for label, path in inputs.items()
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
        "counts": {
            "paragraphs": corpus_count,
            "train_samples": train_stats["samples"],
            "validation_samples": validation_stats["samples"],
            "train_gold_facts": train_stats,
            "validation_gold_facts": validation_stats,
            "gold_mapping_rate": (
                (train_stats["mapped_gold_facts"] + validation_stats["mapped_gold_facts"])
                / (train_stats["total_gold_facts"] + validation_stats["total_gold_facts"])
            ),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_source", type=Path, required=True)
    parser.add_argument("--validation_source", type=Path, required=True)
    parser.add_argument("--train_parquet", type=Path, required=True)
    parser.add_argument("--validation_parquet", type=Path, required=True)
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/corpus/hotpotqa_corpus") / EVIDENCE_SIDECAR_FILENAME,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_evidence_sidecar(
        train_source=args.train_source.expanduser().resolve(),
        validation_source=args.validation_source.expanduser().resolve(),
        train_parquet=args.train_parquet.expanduser().resolve(),
        validation_parquet=args.validation_parquet.expanduser().resolve(),
        corpus_path=args.corpus_path.expanduser().resolve(),
        output_path=args.output_path.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
