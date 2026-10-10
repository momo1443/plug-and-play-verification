"""Convert local DeepScaleR JSON into the parquet schema used by the paper flow.

The seeded validation split is a local diagnostic split, not AIME evaluation.
Only the problem enters the prompt; reference answers stay in reward_model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def prepare_data(input_path: Path, output_dir: Path, validation_size: int = 2015, seed: int = 42) -> dict:
    import pandas as pd

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not 0 < validation_size < len(records):
        raise ValueError("Input must be a JSON list with more rows than validation_size")
    problem_indices = defaultdict(list)
    invalid_indices = []
    for index, record in enumerate(records):
        problem = record.get("problem")
        answer = record.get("answer")
        if not isinstance(problem, str) or not problem.strip() or answer is None or not str(answer).strip():
            invalid_indices.append(index)
            continue
        problem_indices[problem.strip()].append(index)
    if validation_size >= len(records) - len(invalid_indices):
        raise ValueError("Not enough valid rows to create nonempty train and validation splits")

    # The source contains repeated questions, sometimes with differing labels.
    # Keep every source row, but assign an entire question group to one split.
    problems = list(problem_indices)
    random.Random(seed).shuffle(problems)
    validation_indices = set()
    for problem in problems:
        validation_indices.update(problem_indices[problem])
        if len(validation_indices) >= validation_size:
            break
    if len(validation_indices) == len(records) - len(invalid_indices):
        raise ValueError("Grouped validation split would leave no training rows")
    splits = {"train": [], "validation": []}
    invalid_set = set(invalid_indices)
    for index, record in enumerate(records):
        if index in invalid_set:
            continue
        split = "validation" if index in validation_indices else "train"
        splits[split].append({
            "data_source": "deepmath",
            "prompt": [{"role": "user", "content": record["problem"] +
                        "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(record["answer"])},
            "extra_info": {"index": index, "split": split, "source": "DeepScaleR-Preview-Dataset"},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet", index=False)
    manifest = {
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_rows": len(records), "train_rows": len(splits["train"]),
        "excluded_missing_problem_or_answer_indices": invalid_indices,
        "validation_rows": len(splits["validation"]), "seed": seed,
        "split_method": "seeded shuffle of question groups; all duplicates in one split; source order retained",
        "unique_problems": len(problem_indices),
        "duplicate_rows": len(records) - len(invalid_indices) - len(problem_indices),
        "conflicting_answer_groups": sum(
            len({str(records[i]['answer']) for i in indices}) > 1 for indices in problem_indices.values()
        ),
        "validation_target_rows": validation_size,
        "validation_source_indices": sorted(validation_indices),
        "validation_purpose": "local held-out DeepScaleR diagnostic, not AIME benchmark",
    }
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path,
                        default=Path("data/raw/DeepScaleR-Preview-Dataset/deepscaler.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/corpus/deepscaler"))
    parser.add_argument("--validation-size", type=int, default=2015)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_data(args.input_json, args.output_dir, args.validation_size, args.seed)
    print(f"Prepared {manifest['train_rows']} train / {manifest['validation_rows']} validation rows in {args.output_dir}")


if __name__ == "__main__":
    main()
