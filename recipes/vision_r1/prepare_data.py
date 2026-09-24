"""Convert Osilly/Vision-R1-rl Parquet into Agent-R1 multimodal rows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset

MAX_IMAGE_PIXELS = 1_048_576


def _image_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Vision-R1 image must be a mapping, got {type(value)!r}")
    payload = dict(value)
    payload["max_pixels"] = MAX_IMAGE_PIXELS
    return payload


def build_row(row: pd.Series, index: int, split: str) -> dict[str, Any]:
    problem = str(row["problem"]).strip()
    images = [_image_payload(image) for image in row["images"]]
    image_placeholders = problem.count("<image>")
    if image_placeholders != len(images):
        raise ValueError(f"row {index} has {image_placeholders} image placeholders for {len(images)} images")
    answer = str(row["answer"]).strip()
    question_id = f"vision-r1-rl:{split}:{index}"
    return {
        "prompt": [{"role": "user", "content": problem}],
        "images": images,
        "data_source": "vision_r1_rl",
        "answer": answer,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": index,
            "question_id": question_id,
            "split": split,
        },
        "index": index,
    }


def convert(input_path: Path, output_path: Path, split: str) -> int:
    dataframe = pd.read_parquet(input_path)
    required = {"problem", "images", "answer"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Vision-R1 input is missing columns: {sorted(missing)}")
    rows = [build_row(row, index, split) for index, (_, row) in enumerate(dataframe.iterrows())]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(output_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "validation"), required=True)
    args = parser.parse_args()
    count = convert(args.input, args.output, args.split)
    print(f"wrote {count} {args.split} rows to {args.output}")


if __name__ == "__main__":
    main()
