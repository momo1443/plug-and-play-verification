#!/usr/bin/env python3
"""Derive ToolEnv-compatible DeepScaleR parquet files without altering source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from recipes.deepscaler.prompts import build_agent_messages


def build_tool_dataset(source_path: Path, output_path: Path) -> tuple[int, int]:
    """Add the agent name and runner-only hidden answer to one parquet split.

    Rows without a reference answer cannot receive either tool feedback or a
    terminal reward, so they are excluded from the derived training data.
    """
    rows = pq.read_table(source_path).to_pylist()
    converted: list[dict[str, Any]] = []
    skipped_missing_ground_truth = 0
    for row in rows:
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is None or str(ground_truth).strip() == "":
            skipped_missing_ground_truth += 1
            continue
        raw_prompt = row.get("prompt") or []
        converted.append(
            {
                **row,
                "agent_name": "deepscaler_tool",
                "prompt": build_agent_messages(raw_prompt),
                "env_kwargs": json.dumps({"tools_kwargs": {"ground_truth": str(ground_truth)}}),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), output_path, compression="zstd")
    return len(converted), skipped_missing_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {args.source_dir}")
    for split in ("train", "validation"):
        source_path = args.source_dir / f"{split}.parquet"
        output_path = args.output_dir / f"{split}.parquet"
        if not source_path.is_file():
            raise FileNotFoundError(f"Source parquet does not exist: {source_path}")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        rows, skipped = build_tool_dataset(source_path, output_path)
        print(f"Wrote {rows} {split} rows to {output_path}; skipped_missing_ground_truth={skipped}")


if __name__ == "__main__":
    main()
