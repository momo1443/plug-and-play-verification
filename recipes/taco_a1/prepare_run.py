#!/usr/bin/env python3
"""Freeze an auditable manifest for the terminal-only TACO A1 run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


CODE_PATHS = (
    "agent_r1/agent_flow/agent_flow.py",
    "agent_r1/trainer/main_agent_grpo.py",
    "agent_r1/trainer/ppo/core_algos.py",
    "recipes/taco_a9/prepare_data.py",
    "recipes/taco_a9/agent_flow.py",
    "recipes/taco_a9/sandbox.py",
    "recipes/taco_a9/protocol.py",
    "recipes/taco_a1/base.yaml",
    "recipes/taco_a1/prepare_run.py",
    "examples/taco/run_a1_terminal.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    result = {}
    for package in ("ray", "torch", "transformers", "verl", "vllm", "pyarrow"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--sidecar-path", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--total-training-steps", type=int, required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="Record a reduced functional smoke run.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    paths = [Path(args.train_path).resolve(), Path(args.validation_path).resolve(), Path(args.sidecar_path).resolve(), Path(args.model_path).resolve() / "config.json"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required artifact missing: {missing}")
    allowed_batch_sizes = {4, 20} if args.smoke else {20}
    if args.num_gpus not in {4, 6, 8} or args.train_batch_size not in allowed_batch_sizes or args.rollout_n != 4 or args.seed != 42:
        raise ValueError("TACO A1 requires four, six, or eight actor GPUs, batch 20, rollout n=4, and seed 42")
    if args.total_training_steps <= 0 or args.total_epochs <= 0 or args.save_freq <= 0:
        raise ValueError("TACO A1 training parameters must be positive")
    missing_code = [relative for relative in CODE_PATHS if not (project_dir / relative).is_file()]
    if missing_code:
        raise FileNotFoundError(f"TACO A1 source missing: {missing_code}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "contract_version": "taco-a1-terminal-private-binary-v1",
        "status": "prepared",
        "mode": "smoke" if args.smoke else "main",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": "A1-code-terminal-private-binary",
        "model": {"path": str(Path(args.model_path).resolve()), "config_sha256": sha256(Path(args.model_path).resolve() / "config.json")},
        "data": {
            "train_path": str(paths[0]), "validation_path": str(paths[1]), "sidecar_path": str(paths[2]),
            "train_rows": pq.ParquetFile(paths[0]).metadata.num_rows,
            "validation_rows": pq.ParquetFile(paths[1]).metadata.num_rows,
            "reference_solutions_model_visible": False,
        },
        "training": {
            "algorithm": "GRPO", "credit_assignment": "step_causal", "num_gpus": args.num_gpus,
            "train_batch_size": args.train_batch_size, "rollout_n": args.rollout_n,
            "total_training_steps": args.total_training_steps, "total_epochs": args.total_epochs,
            "save_freq": args.save_freq, "seed": args.seed,
            "reference_kl": {"enabled": True, "coefficient": 0.001, "loss_type": "low_var_kl"},
        },
        "reward_contract": {
            "formula": "1.0 iff the submitted final code passes every private test; otherwise 0.0",
            "developer_test_reward": 0.0, "certificate_reward": 0.0, "validation_formula": "same terminal private-test binary score",
        },
        "executor": {"sandbox": "bubblewrap", "network": "unshared", "private_tests_model_visible": False},
        "code_sha256": {relative: sha256(project_dir / relative) for relative in CODE_PATHS},
        "package_versions": versions(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_dir / "run_manifest.json")


if __name__ == "__main__":
    main()
