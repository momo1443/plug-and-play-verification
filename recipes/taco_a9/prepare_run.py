#!/usr/bin/env python3
"""Write a manifest for a uniform A9 TACO coding run."""

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
    "agent_r1/verifier/reward.py",
    "agent_r1/trainer/main_agent_grpo.py",
    "agent_r1/trainer/ppo/core_algos.py",
    "recipes/reward_mixing.py",
    "recipes/taco_a9/prepare_data.py",
    "recipes/taco_a9/dsl.py",
    "recipes/taco_a9/sandbox.py",
    "recipes/taco_a9/verifier.py",
    "recipes/taco_a9/protocol.py",
    "recipes/taco_a9/prompts.py",
    "recipes/taco_a9/agent_flow.py",
    "recipes/taco_a9/base.yaml",
    "recipes/taco_a9/reward_fn.py",
    "recipes/taco_a9/prepare_run.py",
    "recipes/llm_judge/scoring.py",
    "recipes/hotpotqa/judge_server.py",
    "examples/taco/run_a9_uniform.sh",
    "examples/llm_judge/run_with_frozen_judge.sh",
    "examples/llm_judge/run_grpo_4b_judge_9b.sh",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
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
    parser.add_argument("--terminal-warmup-steps", type=int, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--reward-mode",
        choices=("uniform_certificate", "llm_judge"),
        default="uniform_certificate",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    train_path = Path(args.train_path).resolve()
    validation_path = Path(args.validation_path).resolve()
    sidecar_path = Path(args.sidecar_path).resolve()
    model_path = Path(args.model_path).resolve()
    for path in (train_path, validation_path, sidecar_path, model_path / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.train_batch_size != 20 or args.rollout_n != 4:
        raise ValueError("TACO A9 main run requires batch 20 and rollout n=4")
    if args.total_training_steps <= 0 or args.total_epochs <= 0:
        raise ValueError("TACO A9 total steps and epochs must both be positive")
    if args.terminal_warmup_steps < 0 or args.terminal_warmup_steps >= args.total_training_steps:
        raise ValueError("TACO A9 terminal warmup must be non-negative and shorter than total steps")
    if args.seed != 42:
        raise ValueError("TACO A9 main run requires seed 42")
    missing = [relative for relative in CODE_PATHS if not (project_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TACO A9 source files: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "contract_version": (
            "taco-qwen35-9b-llm-judge-v1"
            if args.reward_mode == "llm_judge"
            else "taco-a9-verified-process-group-uniform-v3"
        ),
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": "llm-judge" if args.reward_mode == "llm_judge" else "A9-code-uniform",
        "model": {"path": str(model_path), "config_sha256": _sha256(model_path / "config.json")},
        "data": {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "train_rows": pq.ParquetFile(train_path).metadata.num_rows,
            "validation_rows": pq.ParquetFile(validation_path).metadata.num_rows,
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": _sha256(sidecar_path),
            "reference_solutions_model_visible": False,
        },
        "training": {
            "algorithm": "GRPO",
            "credit_assignment": "step_causal",
            "train_batch_size": args.train_batch_size,
            "rollout_n": args.rollout_n,
            "ppo_micro_batch_size_per_gpu": 2,
            "total_training_steps": args.total_training_steps,
            "total_epochs": args.total_epochs,
            "save_freq": args.save_freq,
            "seed": args.seed,
            "terminal_warmup_steps": args.terminal_warmup_steps,
            "reference_kl": {"enabled": True, "coefficient": 0.001, "loss_type": "low_var_kl"},
        },
        "agent_contract": {
            "max_turns": 5,
            "actions": ["write_code", "run_developer_tests", "submit"],
            "developer_tests_visible_to_actor": False,
            "private_tests_visible_to_actor": False,
            "private_tests_used_after_submit": True,
            "sandbox": "bubblewrap",
            "certificate_verifier": "taco-a9-execution-replay-v1",
        },
        "reward_contract": ({
            "training_formula": "Qwen3.5-9B frozen LLM judge score",
            "validation_formula": "private_test_all_pass",
            "score_support": [0.0, 0.25, 0.5, 0.75, 1.0],
            "failure_policy": "abort_optimizer_update",
        } if args.reward_mode == "llm_judge" else {
            "weight_sampling": "uniform_0_1_per_prompt_group_after_terminal_warmup",
            "formula": "w * private_test_all_pass + (1-w) * verified_process_reward",
            "process_formula": "mean(replay_gated_developer_pass_rate_per_unique_artifact, certificate_score)",
            "outcome_formula": "1 iff all private tests pass, else 0",
            "validation_formula": "report private_test_all_pass and verified_process_reward separately",
            "terminal_warmup_steps": args.terminal_warmup_steps,
            "warmup_formula": "private_test_all_pass",
            "post_warmup_formula": "w * private_test_all_pass + (1-w) * verified_process_reward",
        }),
        "code_sha256": {relative: _sha256(project_dir / relative) for relative in CODE_PATHS},
        "package_versions": _versions(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_dir / "run_manifest.json")


if __name__ == "__main__":
    main()
