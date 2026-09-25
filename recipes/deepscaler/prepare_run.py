#!/usr/bin/env python3
"""Write a reproducibility manifest for a DeepScaleR A9 uniform run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_PATHS = (
    "examples/common/model_training.sh",
    "agent_r1/verifier/reward.py",
    "agent_r1/trainer/main_agent_grpo.py",
    "agent_r1/trainer/ppo/core_algos.py",
    "agent_r1/trainer/ppo/ray_trainer.py",
    "agent_r1/trainer/rollout_jsonl.py",
    "recipes/reward_mixing.py",
    "examples/deepmath/run_deepscaler_a9_uniform.sh",
    "recipes/deepscaler/prepare_run.py",
    "recipes/deepscaler/reward_fn.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("ray", "torch", "transformers", "verl", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model_path).resolve()
    train_path = Path(args.train_path).resolve()
    val_path = Path(args.val_path).resolve()
    resume_path = Path(args.resume_from_path).resolve() if args.resume_from_path else None

    missing = [relative for relative in CODE_PATHS if not (project_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing manifest code files: {missing}")
    for required in (model_path / "config.json", train_path, val_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    resume_global_step = None
    if resume_path is not None:
        if not resume_path.is_dir():
            raise FileNotFoundError(resume_path)
        prefix = "global_step_"
        if not resume_path.name.startswith(prefix) or not resume_path.name[len(prefix) :].isdigit():
            raise ValueError(f"Invalid checkpoint directory name: {resume_path.name}")
        resume_global_step = int(resume_path.name[len(prefix) :])

    code_hashes = {relative: _sha256(project_dir / relative) for relative in CODE_PATHS}
    model_identity_paths = (
        model_path / "config.json",
        model_path / "model.safetensors.index.json",
        model_path / "tokenizer_config.json",
    )

    return {
        "arm": "DeepScaleR-A9-uniform-process",
        "runtime_profile": "deepscaler_legacy",
        "contract_version": "deepscaler-a9-group-uniform-step-equations-v4-strict-em-warmup",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": args.status,
        "output_dir": str(output_dir),
        "artifact_paths": {
            "train_data": str(train_path),
            "validation_data": str(val_path),
            "resume_checkpoint": str(resume_path) if resume_path else None,
        },
        "code_sha256": code_hashes,
        "data_sha256": {
            "train_data": _sha256(train_path),
            "validation_data": _sha256(val_path),
        },
        "model": {
            "path": str(model_path),
            "identity_sha256": {path.name: _sha256(path) for path in model_identity_paths if path.is_file()},
        },
        "package_versions": _package_versions(),
        "resources": {
            "cuda_visible_devices": args.cuda_visible_devices.split(","),
            "num_gpus": args.num_gpus,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
            "vllm_max_num_seqs": args.vllm_max_num_seqs,
        },
        "training": {
            "algorithm": "GRPO",
            "credit_assignment": "step_causal",
            "train_max_samples": args.train_max_samples,
            "train_batch_size": args.train_batch_size,
            "rollout_n": args.rollout_n,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "total_training_steps": args.total_training_steps,
            "resume_global_step": resume_global_step,
            "save_freq": args.save_freq,
            "seed": args.seed,
            "reference_kl": {
                "enabled": True,
                "coefficient": 0.001,
                "loss_type": "low_var_kl",
            },
        },
        "reward_contract": {
            "schedule": {
                "warmup_steps": args.em_warmup_steps,
                "warmup": "strict_terminal_em",
                "post_warmup": "per_prompt_group_w_uniform_0_1",
                "formula": "w * terminal_em + (1 - w) * process_reward",
            },
            "process_reward": {
                "equation_score": "1 if nontrivial numeric equality verifies, else 0",
                "step_score": "verified_equations / extracted_equations; 0 when none extracted",
                "trajectory_score": "mean(step_score over equation-bearing reasoning steps); 0 when none",
                "unsupported_expressions": "fail_closed",
                "optimizer_placement": "completion_level_scalar_after_step_aggregation",
                "token_span_credit": False,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--resume-from-path", default="")
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--vllm-max-model-len", type=int, required=True)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--vllm-max-num-seqs", type=int, required=True)
    parser.add_argument("--train-max-samples", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument("--total-training-steps", type=int, required=True)
    parser.add_argument("--em-warmup-steps", type=int, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--status", default="prepared")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.output_dir).resolve() / "run_manifest.json"
    _write_json_atomic(manifest_path, build_manifest(args))
    print(manifest_path)


if __name__ == "__main__":
    main()
