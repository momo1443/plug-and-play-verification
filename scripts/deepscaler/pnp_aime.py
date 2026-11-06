#!/usr/bin/env python3
"""Compatibility entry for the shared AIME2025/AIME2026 PNP evaluator.

Generation and answer-selection helpers remain importable for older callers.
Use run_pnp_eval.sh for the common benchmark command interface.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent_r1.evaluation.pnp_math import (
    generate_trajectories, selected_aime_integer, summarize_accuracy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "dataset", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    dataset_name = args.dataset_name.lower().replace("_", "").replace("-", "")
    years = [year for year, short in (("2025", "25"), ("2026", "26"))
             if year in args.dataset_label or any(name in dataset_name for name in (f"aime{year}", f"aime{short}"))]
    if len(years) != 1:
        parser.error("This wrapper requires exactly one AIME year (2025 or 2026) in "
                     "--dataset-name or --dataset-label")
    year = years[0]
    from agent_r1.evaluation.pnp import main as evaluate
    evaluate([f"aime{year}", "--model", args.model, "--checkpoint", args.checkpoint,
              "--dataset", args.dataset, "--dataset-version", year, "--split", "test",
              "--dataset-revision", args.dataset_revision, "--gpus", args.gpus,
              "--output", args.output])


if __name__ == "__main__":
    main()
