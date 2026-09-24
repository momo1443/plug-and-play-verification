"""Convert the downloaded MathLLMs/MathVision Parquet into Agent-R1 input."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset

from recipes.deepscaler.prompts import DEEPSCALER_TOOL_SYSTEM_PROMPT


def build_row(row: pd.Series, root: Path, index: int) -> dict:
    question = re.sub(r"<image\d*>", "<image>", str(row["question"]), flags=re.IGNORECASE)
    raw_options = row["options"]
    options = [str(value) for value in raw_options] if raw_options is not None else []
    if options:
        question += "\nOptions: " + " ".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(options))
    image_bytes = row["decoded_image"]["bytes"]
    question_id = str(row["id"])
    answer = str(row["answer"])
    return {
        "prompt": [
            {"role": "system", "content": DEEPSCALER_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "images": [{"bytes": image_bytes}],
        "data_source": "mathvision",
        "answer": answer,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "env_kwargs": {"tools_kwargs": {"ground_truth": answer}},
        "extra_info": {
            "index": index,
            "question_id": question_id,
            "split": "test",
            "env_kwargs": {"tools_kwargs": {"ground_truth": answer}},
            "level": str(row["level"]),
            "subject": str(row["subject"]),
        },
        "index": index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataframe = pd.read_parquet(args.input)
    dataset = Dataset.from_list([build_row(row, args.input.parent.parent, i) for i, (_, row) in enumerate(dataframe.iterrows())])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output)
    print(f"wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
