#!/usr/bin/env python3
"""Convert the exact TACO training prefix and create its runner-only test sidecar.

Reference solutions are not read or emitted into prompts, parquet rows, or the
sidecar. Only the fields needed by the agent and its test runner are converted.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DATA_SOURCE = "taco_a9_python"
SPLIT_VERSION = "taco-source-prefix-v2"


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _task_id(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    source = str(row.get("url") or row.get("source") or "").strip()
    return "taco:" + hashlib.sha256(f"{source}\n{question}".encode("utf-8")).hexdigest()[:20]


def _parse_cases(raw: Any) -> tuple[list[dict[str, str]], str | None]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("input_output must contain a JSON object")
    function_name = value.get("fn_name") or None
    if function_name is not None and (not isinstance(function_name, str) or not function_name.isidentifier()):
        raise ValueError("fn_name must identify a Python function")
    inputs, outputs = value.get("inputs"), value.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must be equally sized lists")
    cases = []
    for stdin, expected_stdout in zip(inputs, outputs, strict=True):
        if function_name is not None:
            if not isinstance(stdin, list):
                raise ValueError("Call-based inputs must contain a positional argument list per case")
            stdin = json.dumps(stdin, ensure_ascii=False, allow_nan=False)
            expected_stdout = json.dumps(expected_stdout, ensure_ascii=False, allow_nan=False)
        else:
            # TACO standard-input cases may be strings or lists of text lines.
            if isinstance(stdin, list) and all(isinstance(line, str) for line in stdin):
                stdin = "\n".join(stdin)
            if isinstance(expected_stdout, list) and all(isinstance(line, str) for line in expected_stdout):
                expected_stdout = "\n".join(expected_stdout)
            if not isinstance(stdin, str) or not isinstance(expected_stdout, str):
                raise ValueError("Standard-input cases require text inputs and outputs")
        cases.append({"stdin": stdin, "expected_stdout": expected_stdout})
    return cases, function_name


def _split_cases(task_id: str, cases: list[dict[str, str]], developer_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(range(len(cases)))
    random.Random(_stable_int(task_id + ":tests")).shuffle(shuffled)
    if len(cases) < 2:
        developer_count = 0
    elif len(cases) < 4:
        developer_count = 1
    else:
        developer_count = min(max(2, int(round(len(cases) * developer_fraction))), len(cases) - 2)
    developer_indices = set(shuffled[:developer_count])
    developer = [case for index, case in enumerate(cases) if index in developer_indices]
    private = [case for index, case in enumerate(cases) if index not in developer_indices]
    return developer, private


def _prompt(question: str, starter_code: str, function_name: str | None = None) -> list[dict[str, str]]:
    content = question
    if starter_code.strip():
        content += "\n\nStarter code:\n```python\n" + starter_code.strip() + "\n```"
    if function_name:
        content += f"\n\nUse call-based format: implement `{function_name}` as a function or a method of `Solution`."
    return [{"role": "user", "content": content}]


def _iter_rows(paths: Iterable[str]) -> Iterable[dict[str, Any]]:
    for path in paths:
        source = pq.ParquetFile(path)
        columns = [name for name in ("question", "input_output", "starter_code", "difficulty", "source", "url")
                   if name in source.schema_arrow.names]
        for batch in source.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def validate_source_prefix(path: Path, samples: int) -> None:
    """Require converted source-prefix data before a training run starts."""
    source = pq.ParquetFile(path)
    metadata = source.schema_arrow.metadata or {}
    if metadata.get(b"pnp.training_selection") != b"exact_source_training_prefix":
        raise ValueError("Prepare the source prefix with python -m recipes.taco_a9.prepare_data before training")
    if samples < 1 or source.metadata.num_rows < samples:
        raise ValueError(f"Training data must contain at least {samples} source-prefix rows")
    index = 0
    for batch in source.iter_batches(batch_size=256, columns=["extra_info"]):
        for row in batch.to_pylist():
            if row["extra_info"]["index"] != index:
                raise ValueError(f"Training source indices must be contiguous from zero; mismatch at row {index}")
            index += 1
            if index == samples:
                return


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_paths = sorted(glob.glob(str(Path(args.taco_dir) / "ALL" / "train-*.parquet")))
    if not source_paths:
        raise FileNotFoundError("No TACO train parquet files found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_samples = args.train_samples
    validation_size = args.validation_size
    if train_samples < 1 or validation_size < 0:
        raise ValueError("train_samples must be positive and validation_size non-negative")
    if not 0 < args.developer_fraction < 1:
        raise ValueError("developer_fraction must be between zero and one")
    required = train_samples + validation_size
    source_rows = list(islice(_iter_rows(source_paths), required))
    if len(source_rows) < required:
        raise ValueError(f"Need {required} source rows; found {len(source_rows)}")
    tasks: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Source row {index} requires a question; rows cannot be skipped or replaced")
        try:
            cases, function_name = _parse_cases(row.get("input_output"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Source row {index}: {exc}; rows cannot be skipped or replaced") from exc
        # The row index preserves distinct occurrences of repeated tasks.
        task_id = f"{_task_id(row)}:row:{index}"
        developer_cases, private_cases = _split_cases(_task_id(row), cases, args.developer_fraction)
        tasks.append(
            {
                "task_id": task_id,
                "question": question,
                "starter_code": str(row.get("starter_code") or ""),
                "difficulty": str(row.get("difficulty") or ""),
                "source": str(row.get("source") or ""),
                "source_index": index,
                "function_name": function_name,
                "developer_cases": developer_cases,
                "private_cases": private_cases,
            }
        )
    sidecar_path = output_dir / "taco_a9_sidecar.jsonl"
    with sidecar_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps({key: value for key, value in task.items() if key != "question"}, ensure_ascii=False) + "\n")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for index, task in enumerate(tasks):
        split = "train" if index < train_samples else "validation"
        rows_by_split[split].append(
            {
                "data_source": DATA_SOURCE,
                "prompt": _prompt(task["question"], task["starter_code"], task["function_name"]),
                "reward_model": {"ground_truth": "", "style": "rule"},
                "extra_info": {
                    "index": index,
                    "question_id": task["task_id"],
                    "split": split,
                    "difficulty": task["difficulty"],
                },
            }
        )
    schema = pa.Table.from_pylist(rows_by_split["train"]).schema.with_metadata({
        b"pnp.training_selection": b"exact_source_training_prefix",
    })
    for split, rows in rows_by_split.items():
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), output_dir / f"{split}.parquet")
    manifest = {
        "training_row_order": "exact_source_training_prefix",
        "contract_version": SPLIT_VERSION,
        "raw_taco_train_files": [str(Path(path).resolve()) for path in source_paths],
        "raw_row_count": sum(pq.ParquetFile(path).metadata.num_rows for path in source_paths),
        "train_tasks": len(rows_by_split["train"]),
        "train_source_indices": list(range(train_samples)),
        "validation_tasks": len(rows_by_split["validation"]),
        "validation_source_indices": list(range(train_samples, required)),
        "validation_purpose": "sequential local diagnostics after the training prefix",
        "developer_fraction": args.developer_fraction,
        "seed_contract": "sha256(task_id) test partition; source rows are never shuffled",
        "sidecar": str(sidecar_path.resolve()),
        "rows_filtered": 0,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--developer-fraction", type=float, default=0.4)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--validation-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.developer_fraction < 1.0:
        raise ValueError("developer_fraction must be between zero and one")
    print(json.dumps(build_dataset(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
