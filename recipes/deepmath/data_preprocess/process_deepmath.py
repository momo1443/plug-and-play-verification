#!/usr/bin/env python3
"""Convert DeepMath-103K parquet to Agent-R1 / verl RLHFDataset format.

Input columns:  question, final_answer, difficulty, topic, r1_solution_1..3
Output schema:  data_source, prompt, reward_model, extra_info

Usage:
    python recipes/deepmath/data_preprocess/process_deepmath.py \
        --input_path data/corpus/deepmath/deepmath_103k.parquet \
        --output_dir data/corpus/deepmath \
        --val_ratio 0.05
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd


def _row_to_agent_r1(
    ex: dict[str, Any],
    split: str,
    row_index: int,
) -> dict[str, Any]:
    """Single DeepMath example -> verl RLHFDataset row."""
    question = str(ex["question"]).strip()
    answer = str(ex["final_answer"]).strip()

    # Instruct the model to use \boxed{} format for the final answer.
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
        "question_id": f"deepmath_{split}_{row_index}",
        "split": split,
        "difficulty": float(ex.get("difficulty", -1)),
        "topic": str(ex.get("topic", "")),
    }

    return {
        "data_source": "deepmath",
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DeepMath-103K for Agent-R1 RL training.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to deepmath_103k.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for train/validation parquet")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading {args.input_path}...")
    df = pd.read_parquet(args.input_path)
    print(f"Loaded {len(df)} examples")

    # Shuffle and split
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
