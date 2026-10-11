#!/usr/bin/env python3
"""Compatibility entry for the shared AIME2026 PNP evaluator.

Generation and answer-selection helpers remain importable for older callers.
Use run_pnp_eval.sh for the common four-benchmark command interface.
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
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    if "2025" in args.dataset_label or "aime25" in args.dataset_name.lower():
        parser.error("The unified PNP profile is AIME2026; use eval_aime2025.py for legacy AIME2025 evaluation")
    from agent_r1.evaluation.pnp import main as evaluate
    evaluate(["aime2026", "--model", args.model, "--checkpoint", args.checkpoint,
              "--dataset", args.dataset, "--dataset-version", "2026", "--split", "test",
              "--dataset-revision", args.dataset_revision, "--gpus", args.gpus,
              "--output", args.output])


if __name__ == "__main__":
    main()
