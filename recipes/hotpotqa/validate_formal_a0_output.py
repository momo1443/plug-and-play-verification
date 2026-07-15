#!/usr/bin/env python3
"""Strictly replay and validate a formal shared-AgentFlow A0 JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recipes.hotpotqa.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    OfficialEvidenceStore,
    parse_evidence_id,
)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _equal_number(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _replay_record(
    record: dict[str, Any],
    *,
    line_number: int,
    store: OfficialEvidenceStore | None,
) -> dict[str, Any]:
    if record.get("thinking_mode") != "disabled":
        raise ValueError(f"Line {line_number} does not use the frozen thinking-disabled contract")
    if record.get("force_first_search") is not False:
        raise ValueError(f"Line {line_number} used a forced first search")
    if record.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"Line {line_number} has the wrong evidence schema")
    thinking = record.get("qwen_thinking")
    if not isinstance(thinking, list) or not thinking:
        raise ValueError(f"Line {line_number} has no Qwen thinking-turn audit")
    if [turn.get("turn") for turn in thinking] != list(range(1, len(thinking) + 1)):
        raise ValueError(f"Line {line_number} has non-sequential thinking turns")

    sample_key = str(record.get("sample_key") or "")
    if not sample_key.startswith("validation:"):
        raise ValueError(f"Line {line_number} has invalid sample_key {sample_key!r}")
    try:
        row_index = int(sample_key.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Line {line_number} has invalid sample_key {sample_key!r}") from exc

    gold_ids = [str(value) for value in record.get("gold_evidence_ids") or []]
    unresolved = record.get("unresolved_gold_facts") or []
    if not isinstance(unresolved, list):
        raise ValueError(f"Line {line_number} unresolved_gold_facts is not a list")
    if store is not None:
        sample = store.sample("validation", row_index)
        if sample.sample_key != sample_key:
            raise ValueError(f"Line {line_number} sidecar sample key mismatch")
        if record.get("dataset_question_id") != sample.dataset_question_id:
            raise ValueError(f"Line {line_number} dataset question ID mismatch")
        if record.get("official_qid") != sample.official_qid:
            raise ValueError(f"Line {line_number} official qid mismatch")
        if gold_ids != list(sample.gold_evidence_ids):
            raise ValueError(f"Line {line_number} gold evidence IDs differ from sidecar")
        if unresolved != list(sample.unresolved_gold_facts):
            raise ValueError(f"Line {line_number} unresolved gold facts differ from sidecar")

    gold_set = set(gold_ids)
    for evidence_id in gold_ids:
        parse_evidence_id(evidence_id)
    eligible = not unresolved and bool(gold_set)
    covered: set[str] = set()
    all_returned: set[str] = set()
    search_steps = record.get("search_steps")
    if not isinstance(search_steps, list):
        raise ValueError(f"Line {line_number} search_steps is not a list")
    if len(search_steps) > 3:
        raise ValueError(f"Line {line_number} executed more than three searches")

    for expected_search_index, search_step in enumerate(search_steps, start=1):
        if search_step.get("search_index") != expected_search_index:
            raise ValueError(f"Line {line_number} has non-sequential search indices")
        if search_step.get("source") != "model":
            raise ValueError(f"Line {line_number} contains a non-model search")
        nested_ids: list[str] = []
        for paragraph in search_step.get("returned_evidence") or []:
            pid = int(paragraph.get("pid", -1))
            if pid < 0 or not str(paragraph.get("text") or ""):
                raise ValueError(f"Line {line_number} has malformed paragraph evidence")
            sentences = paragraph.get("sentence_evidence")
            if not isinstance(sentences, list):
                raise ValueError(f"Line {line_number} paragraph evidence has no sentence list")
            if store is not None:
                expected_sentences = store.paragraph_evidence(
                    pid, expected_title=str(paragraph.get("title") or "")
                )
                if sentences != expected_sentences:
                    raise ValueError(
                        f"Line {line_number} paragraph PID {pid} differs from official sidecar"
                    )
                joined_text = " ".join(sentence["text"] for sentence in expected_sentences).strip()
                expected_text = f"{paragraph.get('title', '')} {joined_text}".strip()
                if paragraph.get("text") != expected_text:
                    raise ValueError(
                        f"Line {line_number} paragraph PID {pid} model-visible text changed"
                    )
            for sentence in sentences:
                evidence_id = str(sentence.get("evidence_id") or "")
                parsed_pid, sentence_id = parse_evidence_id(evidence_id)
                if (
                    parsed_pid != pid
                    or int(sentence.get("pid", -1)) != pid
                    or int(sentence.get("sentence_id", -1)) != sentence_id
                ):
                    raise ValueError(f"Line {line_number} has malformed evidence {evidence_id}")
                nested_ids.append(evidence_id)

        returned = set(nested_ids)
        if search_step.get("returned_evidence_ids") != sorted(returned):
            raise ValueError(f"Line {line_number} returned evidence ID summary is wrong")
        new_gold = (returned & gold_set) - covered
        covered.update(returned & gold_set)
        if search_step.get("new_gold_evidence_ids") != sorted(new_gold):
            raise ValueError(f"Line {line_number} new-gold audit is wrong")
        if search_step.get("covered_gold_evidence_ids") != sorted(covered):
            raise ValueError(f"Line {line_number} covered-gold audit is wrong")
        expected_reward = len(new_gold) / len(gold_set) if eligible else None
        if not _equal_number(search_step.get("audit_process_reward"), expected_reward):
            raise ValueError(f"Line {line_number} audit process reward is wrong")
        all_returned.update(returned)

    expected_queries = [str(step.get("query") or "") for step in search_steps]
    if record.get("search_queries") != expected_queries:
        raise ValueError(f"Line {line_number} compact query list differs from search steps")

    hits = len(covered)
    expected_metrics = {
        "evidence_metrics_eligible": eligible,
        "gold_evidence_count": len(gold_set),
        "unresolved_gold_fact_count": len(unresolved),
        "retrieved_evidence_count": len(all_returned),
        "supporting_fact_hits": hits,
        "supporting_fact_recall": hits / len(gold_set) if eligible else None,
        "supporting_fact_precision": (
            hits / len(all_returned) if eligible and all_returned else (0.0 if eligible else None)
        ),
        "at_least_one_support": hits > 0 if eligible else None,
        "both_support": covered == gold_set if eligible else None,
        "complete_coverage_search_step": next(
            (
                int(step["search_index"])
                for step in search_steps
                if eligible and set(step.get("covered_gold_evidence_ids") or []) == gold_set
            ),
            None,
        ),
    }
    actual_metrics = record.get("evidence_metrics")
    if not isinstance(actual_metrics, dict):
        raise ValueError(f"Line {line_number} has no evidence_metrics object")
    for key, expected in expected_metrics.items():
        actual = actual_metrics.get(key)
        if isinstance(expected, float) or isinstance(actual, float):
            matches = _equal_number(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Line {line_number} evidence metric {key!r} is {actual!r}; expected {expected!r}"
            )
    return expected_metrics


def validate_formal_a0_output(
    path: Path,
    *,
    expected_rows: int | None = None,
    manifest_path: Path | None = None,
    mark_complete: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"A0 JSONL is missing: {path}")

    manifest: dict[str, Any] | None = None
    expected_samples: list[dict[str, Any]] | None = None
    store: OfficialEvidenceStore | None = None
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract_version") != "hotpotqa-formal-a0-shared-agentflow-v1":
            raise ValueError("Run manifest has the wrong formal A0 contract")
        if Path(manifest["output_jsonl"]).resolve() != path.resolve():
            raise ValueError("Run manifest points to a different JSONL")
        expected_rows = int(manifest["selection"]["expected_rows"])
        expected_samples = list(manifest["selection"]["samples"])
        artifact_lock = json.loads(
            Path(manifest["artifact_lock_path"]).read_text(encoding="utf-8")
        )
        store = OfficialEvidenceStore(
            Path(artifact_lock["artifacts"]["evidence_sidecar"]["path"])
        )

    rows = 0
    strict_matches = 0
    complete_thinking_rows = 0
    model_search_steps = 0
    retrieved_paragraphs = 0
    retrieved_sentence_ids: set[str] = set()
    eligible_rows = 0
    both_support_rows = 0
    at_least_one_rows = 0
    precision_sum = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"A0 JSONL has a blank line at {line_number}")
            record = json.loads(line)
            if int(record.get("sample_index", -1)) != rows:
                raise ValueError(f"Line {line_number} has a non-sequential sample_index")
            if expected_samples is not None:
                if rows >= len(expected_samples):
                    raise ValueError("A0 JSONL contains more rows than the manifest selection")
                expected_identity = expected_samples[rows]
                if (
                    record.get("sample_key") != expected_identity["sample_key"]
                    or record.get("official_qid") != expected_identity["official_qid"]
                ):
                    raise ValueError(f"Line {line_number} identity differs from run manifest")

            metrics = _replay_record(record, line_number=line_number, store=store)
            strict_matches += float(record.get("score", 0.0)) == 1.0
            complete_thinking_rows += all(
                bool(turn.get("complete")) for turn in record["qwen_thinking"]
            )
            search_steps = record["search_steps"]
            model_search_steps += len(search_steps)
            retrieved_paragraphs += sum(
                len(step.get("returned_evidence") or []) for step in search_steps
            )
            retrieved_sentence_ids.update(
                evidence_id
                for step in search_steps
                for evidence_id in step.get("returned_evidence_ids") or []
            )
            if metrics["evidence_metrics_eligible"]:
                eligible_rows += 1
                both_support_rows += bool(metrics["both_support"])
                at_least_one_rows += bool(metrics["at_least_one_support"])
                precision_sum += float(metrics["supporting_fact_precision"])
            rows += 1

    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"A0 JSONL has {rows} rows; expected {expected_rows}")
    if expected_samples is not None and rows != len(expected_samples):
        raise ValueError("A0 JSONL is incomplete relative to the manifest qid list")

    summary = {
        "status": "ok",
        "path": str(path),
        "sha256": _sha256(path),
        "rows": rows,
        "strict_matches": strict_matches,
        "strict_accuracy": strict_matches / rows if rows else 0.0,
        "complete_thinking_rows": complete_thinking_rows,
        "complete_thinking_rate": complete_thinking_rows / rows if rows else 0.0,
        "model_search_steps": model_search_steps,
        "mean_model_searches": model_search_steps / rows if rows else 0.0,
        "retrieved_paragraphs": retrieved_paragraphs,
        "unique_retrieved_sentence_ids": len(retrieved_sentence_ids),
        "evidence_metric_eligible_rows": eligible_rows,
        "both_support_rows": both_support_rows,
        "both_support_recall": both_support_rows / eligible_rows if eligible_rows else 0.0,
        "at_least_one_support_rows": at_least_one_rows,
        "at_least_one_support_recall": (
            at_least_one_rows / eligible_rows if eligible_rows else 0.0
        ),
        "mean_supporting_fact_precision": (
            precision_sum / eligible_rows if eligible_rows else 0.0
        ),
    }
    if mark_complete:
        if manifest is None or manifest_path is None:
            raise ValueError("--mark_complete requires --manifest")
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_validation"] = summary
        _write_json_atomic(manifest_path, manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--expected_rows", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mark_complete", action="store_true")
    args = parser.parse_args()
    summary = validate_formal_a0_output(
        args.jsonl.expanduser().resolve(),
        expected_rows=args.expected_rows,
        manifest_path=args.manifest.expanduser().resolve() if args.manifest else None,
        mark_complete=args.mark_complete,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
