#!/usr/bin/env python3
"""Prepare model-safe TACO A9 train/dev parquet files and a private test sidecar.

Reference solutions are read only to support optional offline validation outside
this script. They are never emitted into prompts, parquet rows, or the sidecar.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DATA_SOURCE = "taco_a9_python"
SPLIT_VERSION = "taco-a9-public-private-v1"


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _task_id(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    source = str(row.get("url") or row.get("source") or "").strip()
    return "taco:" + hashlib.sha256(f"{source}\n{question}".encode("utf-8")).hexdigest()[:20]


def _parse_cases(raw: Any) -> list[dict[str, str]] | None:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("fn_name"):
        return None
    inputs, outputs = value.get("inputs"), value.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return None
    cases = []
    for stdin, expected_stdout in zip(inputs, outputs, strict=True):
        if not isinstance(stdin, str) or not isinstance(expected_stdout, str):
            return None
        cases.append({"stdin": stdin, "expected_stdout": expected_stdout})
    return cases


def _split_cases(task_id: str, cases: list[dict[str, str]], developer_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(range(len(cases)))
    random.Random(_stable_int(task_id + ":tests")).shuffle(shuffled)
    developer_count = max(2, int(round(len(cases) * developer_fraction)))
    developer_count = min(developer_count, len(cases) - 2)
    developer_indices = set(shuffled[:developer_count])
    developer = [case for index, case in enumerate(cases) if index in developer_indices]
    private = [case for index, case in enumerate(cases) if index not in developer_indices]
    return developer, private


def _prompt(question: str, starter_code: str) -> list[dict[str, str]]:
    content = question.strip()
    if starter_code.strip():
        content += "\n\nStarter code:\n```python\n" + starter_code.strip() + "\n```"
    return [{"role": "user", "content": content}]


def _iter_rows(paths: Iterable[str]) -> Iterable[dict[str, Any]]:
    for path in paths:
        table = pq.read_table(path)
        yield from table.to_pylist()


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_paths = sorted(glob.glob(str(Path(args.taco_dir) / "ALL" / "train-*.parquet")))
    if not source_paths:
        raise FileNotFoundError("No TACO train parquet files found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_questions: set[str] = set()
    tasks: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for row in _iter_rows(source_paths):
        question = str(row.get("question") or "").strip()
        cases = _parse_cases(row.get("input_output"))
        if not question:
            skipped["empty_question"] = skipped.get("empty_question", 0) + 1
            continue
        if cases is None:
            skipped["unsupported_or_invalid_io"] = skipped.get("unsupported_or_invalid_io", 0) + 1
            continue
        if len(cases) < args.min_test_cases:
            skipped["too_few_tests"] = skipped.get("too_few_tests", 0) + 1
            continue
        normalized_question = " ".join(question.split())
        if normalized_question in seen_questions:
            skipped["duplicate_question"] = skipped.get("duplicate_question", 0) + 1
            continue
        seen_questions.add(normalized_question)
        task_id = _task_id(row)
        developer_cases, private_cases = _split_cases(task_id, cases, args.developer_fraction)
        if len(developer_cases) < 2 or len(private_cases) < 2:
            skipped["test_split_failed"] = skipped.get("test_split_failed", 0) + 1
            continue
        tasks.append(
            {
                "task_id": task_id,
                "question": question,
                "starter_code": str(row.get("starter_code") or ""),
                "difficulty": str(row.get("difficulty") or ""),
                "source": str(row.get("source") or ""),
                "developer_cases": developer_cases,
                "private_cases": private_cases,
            }
        )
        if args.max_tasks and len(tasks) >= args.max_tasks:
            break
    # Preserve eligible rows in source shard/row order for the paper prefix.
    dev_ids = {
        task["task_id"]
        for task in tasks
        if _stable_int(task["task_id"] + ":split") % 10_000 < round(args.validation_fraction * 10_000)
    }
    if not dev_ids:
        raise ValueError("Validation fraction selected no tasks")
    sidecar_path = output_dir / "taco_a9_sidecar.jsonl"
    with sidecar_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps({key: value for key, value in task.items() if key != "question"}, ensure_ascii=False) + "\n")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for index, task in enumerate(tasks):
        split = "validation" if task["task_id"] in dev_ids else "train"
        rows_by_split[split].append(
            {
                "data_source": DATA_SOURCE,
                "prompt": _prompt(task["question"], task["starter_code"]),
                "reward_model": {"ground_truth": "", "style": "rule"},
                "extra_info": {
                    "index": index,
                    "question_id": task["task_id"],
                    "split": split,
                    "difficulty": task["difficulty"],
                },
            }
        )
    train_before_trim = len(rows_by_split["train"])
    usable_train_rows = train_before_trim // args.train_batch_size * args.train_batch_size
    if usable_train_rows == 0:
        raise ValueError("No complete training batch remains after deterministic task split")
    rows_by_split["train"] = rows_by_split["train"][:usable_train_rows]
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"No rows for {split}")
        pq.write_table(pa.Table.from_pylist(rows), output_dir / f"{split}.parquet")
    manifest = {
        "training_row_order": "source_shard_then_row_after_filtering_and_validation_holdout",
        "contract_version": SPLIT_VERSION,
        "raw_taco_train_files": [str(Path(path).resolve()) for path in source_paths],
        "raw_row_count": sum(pq.ParquetFile(path).metadata.num_rows for path in source_paths),
        "eligible_tasks": len(tasks),
        "train_tasks": len(rows_by_split["train"]),
        "train_tasks_before_batch_trim": train_before_trim,
        "train_batch_size": args.train_batch_size,
        "validation_tasks": len(rows_by_split["validation"]),
        "min_test_cases": args.min_test_cases,
        "developer_fraction": args.developer_fraction,
        "validation_fraction": args.validation_fraction,
        "seed_contract": "sha256(task_id) deterministic split; no reference solutions emitted",
        "sidecar": str(sidecar_path.resolve()),
        "skipped": skipped,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-test-cases", type=int, default=6)
    parser.add_argument("--developer-fraction", type=float, default=0.4)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.developer_fraction < 1.0:
        raise ValueError("developer_fraction must be between zero and one")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if args.train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    print(json.dumps(build_dataset(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
