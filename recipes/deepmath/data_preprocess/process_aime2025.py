#!/usr/bin/env python3
"""Convert AIME 2025 dataset to Agent-R1 / verl RLHFDataset format.

Downloads from HuggingFace (opencompass/AIME2025) with configs AIME2025-I
and AIME2025-II, combining both into a single test parquet.

Output schema: data_source, prompt, reward_model, extra_info
Uses data_source="deepmath" so that recipes/deepmath/reward_fn.py works
without modification (\\boxed{} answer extraction + math equivalence).

Usage:
    python recipes/deepmath/data_preprocess/process_aime2025.py \
        --output_dir data/corpus/aime2025
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd


def _row_to_agent_r1(
    ex: dict[str, Any],
    row_index: int,
    source: str,
) -> dict[str, Any]:
    """Single AIME 2025 example -> verl RLHFDataset row."""
    question = str(ex["question"]).strip()
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
        "question_id": f"aime2025_{source}_{row_index}",
        "split": "test",
        "source": source,
    }

    return {
        "data_source": "deepmath",  # reuse deepmath reward fn
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare AIME 2025 for Agent-R1 evaluation."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/corpus/aime2025",
        help="Directory for test parquet",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    from datasets import load_dataset

    all_rows: list[dict[str, Any]] = []
    idx = 0

    for config_name, source_label in [
        ("AIME2025-I", "aime_i"),
        ("AIME2025-II", "aime_ii"),
    ]:
        print(f"Loading opencompass/AIME2025 config={config_name}...")
        ds = load_dataset("opencompass/AIME2025", config_name, split="test")
        print(f"  {len(ds)} examples")
        for ex in ds:
            all_rows.append(_row_to_agent_r1(ex, idx, source_label))
            idx += 1

    df = pd.DataFrame(all_rows)
    path = os.path.join(out_dir, "test.parquet")
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df)} rows -> {path}")
    print("Done!")


if __name__ == "__main__":
    main()
