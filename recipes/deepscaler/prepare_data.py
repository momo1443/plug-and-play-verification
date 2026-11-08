"""Convert local DeepScaleR JSON into the parquet schema used by the paper flow.

Training uses the exact source prefix. Local diagnostic rows follow that prefix.
Only the problem enters the prompt; reference answers stay in reward_model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare_data(input_path: Path, output_dir: Path, validation_size: int = 2015,
                 train_samples: int = 10000) -> dict:
    import pandas as pd

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list in source training-split order")
    if train_samples < 1 or validation_size < 0:
        raise ValueError("train_samples must be positive and validation_size non-negative")
    required = train_samples + validation_size
    if len(records) < required:
        raise ValueError(f"Need {required} source rows; found {len(records)}")

    splits = {"train": [], "validation": []}
    for index, record in enumerate(records[:required]):
        problem = record.get("problem") if isinstance(record, dict) else None
        answer = record.get("answer") if isinstance(record, dict) else None
        if not isinstance(problem, str) or not problem.strip() or answer is None or not str(answer).strip():
            raise ValueError(f"Source row {index} requires a nonempty problem and answer; rows cannot be skipped or replaced")
        split = "train" if index < train_samples else "validation"
        splits[split].append({
            "data_source": "deepmath",
            "prompt": [{"role": "user", "content": record["problem"] +
                        "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(record["answer"])},
            "extra_info": {"index": index, "split": split, "source": "DeepScaleR-Preview-Dataset"},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(splits["train"][0])
    for split, rows in splits.items():
        pd.DataFrame(rows, columns=columns).to_parquet(output_dir / f"{split}.parquet", index=False)
    manifest = {
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_rows": len(records), "train_rows": len(splits["train"]),
        "selection": "exact_source_training_prefix",
        "train_source_indices": list(range(train_samples)),
        "labels_changed": False,
        "rows_filtered": 0,
        "validation_rows": len(splits["validation"]),
        "validation_source_indices": list(range(train_samples, required)),
        "validation_purpose": "sequential local diagnostics after the training prefix, not AIME evaluation",
    }
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path,
                        default=Path("data/raw/DeepScaleR-Preview-Dataset/deepscaler.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/corpus/deepscaler"))
    parser.add_argument("--validation-size", type=int, default=2015)
    parser.add_argument("--train-samples", type=int, default=10000)
    args = parser.parse_args()
    manifest = prepare_data(args.input_json, args.output_dir, args.validation_size, args.train_samples)
    print(f"Prepared {manifest['train_rows']} train / {manifest['validation_rows']} validation rows in {args.output_dir}")


if __name__ == "__main__":
    main()
