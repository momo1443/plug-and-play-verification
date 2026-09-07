#!/usr/bin/env python3
"""Convert DeepScaleR-Preview-Dataset JSON to Agent-R1 / verl RLHFDataset format.

Input schema:  problem, answer, solution
Output schema: data_source, prompt, reward_model, extra_info

The output format is identical to the DeepMath-103K pipeline so that
recipes/deepmath/reward_fn.py works without modification.

Usage:
    python recipes/deepmath/data_preprocess/process_deepscaler.py \
        --input_path data/corpus/deepscaler_preview/deepscaler.json \
        --output_dir data/corpus/deepscaler \
        --val_ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd


def _row_to_agent_r1(
    ex: dict[str, Any],
    split: str,
    row_index: int,
) -> dict[str, Any]:
    """Single DeepScaleR example -> verl RLHFDataset row."""
    question = str(ex["problem"]).strip()
    answer = str(ex["answer"]).strip()

    # Instruct the model to use \\boxed{} format for the final answer.
    # This is critical for math_reward to extract and verify the answer.
    prompt = [{
        "role": "user",
        "content": (
            f"{question}\n\n"
            "Solve the problem step by step. "
            "Put your final answer within \\boxed{{}}."
        ),
    }]

    reward_model = {
        "ground_truth": answer,
        "style": "rule",
    }

    extra_info: dict[str, Any] = {
        "index": row_index,
        "question_id": f"deepscaler_{split}_{row_index}",
        "split": split,
    }

    return {
        "data_source": "deepmath",  # reuse deepmath reward fn
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DeepScaleR-Preview-Dataset for Agent-R1 RL training."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to deepscaler.json",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for train/validation parquet",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.05,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading {args.input_path}...")
    with open(args.input_path) as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} examples")

    # Shuffle and split
    df = pd.DataFrame(raw_data)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(df) * args.val_ratio)
    n_train = len(df) - n_val

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]

    for split_name, split_df, parquet_name in [
        ("train", train_df, "train.parquet"),
        ("validation", val_df, "validation.parquet"),
    ]:
        rows = []
        for i, (_, raw_row) in enumerate(split_df.iterrows()):
            rows.append(_row_to_agent_r1(raw_row.to_dict(), split_name, i))
        out_df = pd.DataFrame(rows)
        path = os.path.join(out_dir, parquet_name)
        out_df.to_parquet(path, index=False)
        print(f"Wrote {len(rows)} rows -> {path}")

    print("Done!")


if __name__ == "__main__":
    main()
