#!/usr/bin/env python3
"""Build a task-id to byte-range index for a TACO test sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    offsets: dict[str, tuple[int, int]] = {}
    with args.sidecar.open("rb") as handle:
        while True:
            offset = handle.tell()
            payload = handle.readline()
            if not payload:
                break
            task_id = str(json.loads(payload)["task_id"])
            if task_id in offsets:
                raise ValueError(f"Duplicate task id: {task_id}")
            offsets[task_id] = (offset, len(payload))
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(offsets, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(f"indexed {len(offsets)} tasks")


if __name__ == "__main__":
    main()
