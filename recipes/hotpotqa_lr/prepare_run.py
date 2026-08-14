#!/usr/bin/env python3
"""Fail-closed manifest preparation for A8-LR runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from recipes.hotpotqa_lr.dsl import DSL_VERSION, REASON_STEP_FORMAT
from recipes.hotpotqa_lr.prompts import (
    BOOTSTRAP_TOOL_SCHEMAS,
    FINISH_TOOL_SCHEMAS,
    LR_FINAL_TURN_PROMPT,
    LR_FINISH_AVAILABLE_PROMPT,
    LR_SYSTEM_PROMPT,
    LR_USER_PROMPT,
    SEARCH_OR_FINISH_TOOL_SCHEMAS,
)
from recipes.hotpotqa_lr.reward_contract import LR_CONTRACT_VERSION, LR_REWARD_ARM, PRIMARY_CONTRACT
from recipes.hotpotqa_lr.verifier import VERIFIER_VERSION

MANIFEST_VERSION = "hotpotqa-a8-lr-run-v1"


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected true or false, got {value!r}")
    return normalized == "true"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "transformers", "vllm", "verl", "ray"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


def _model_identity(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = {"config.json": _sha256(config_path)}
    for filename in ("model.safetensors.index.json", "tokenizer_config.json"):
        candidate = model_path / filename
        if candidate.is_file():
            identity[filename] = _sha256(candidate)
    return {
        "path": str(model_path.resolve()),
        "name": model_path.name,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "identity_sha256": identity,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--evidence-sidecar-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-max-samples", type=int, required=True)
    parser.add_argument("--val-max-samples", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--total-training-steps", type=int, required=True)
    parser.add_argument("--grpo-micro-batch-size", type=int, required=True)
    parser.add_argument("--data-shuffle", required=True)
    parser.add_argument("--actor-use-dynamic-bsz", required=True)
    parser.add_argument("--actor-max-token-len-per-gpu", type=int, required=True)
    parser.add_argument("--reference-kl-enabled", required=True)
    parser.add_argument("--reference-kl-loss-coef", type=float, required=True)
    parser.add_argument("--reference-kl-loss-type", required=True)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--max-actor-ckpt-to-keep", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--agent-workers", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_dir = Path(args.project_dir).resolve()
    train_path = Path(args.train_path).resolve()
    validation_path = Path(args.validation_path).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    sidecar_path = Path(args.evidence_sidecar_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (
        train_path,
        validation_path,
        corpus_dir / "index.bin",
        corpus_dir / "hpqa_corpus.jsonl",
        sidecar_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    train_rows = pq.ParquetFile(train_path).metadata.num_rows
    validation_rows = pq.ParquetFile(validation_path).metadata.num_rows
    if args.train_max_samples != 30_000 or train_rows < args.train_max_samples:
        raise ValueError("A8-LR main run requires the first 30,000 training rows")
    if args.val_max_samples != 7_405 or validation_rows != 7_405:
        raise ValueError("A8-LR requires all 7,405 validation rows")
    if args.train_batch_size != 20 or args.total_training_steps != 1_500:
        raise ValueError("A8-LR requires batch 20 and exactly 1,500 steps")
    if args.rollout_n != 4 or args.grpo_micro_batch_size != 2:
        raise ValueError("A8-LR requires rollout n=4 and micro-batch/GPU=2")
    if args.num_gpus != 5 or args.agent_workers != 5:
        raise ValueError("A8-LR-30 main run requires five GPUs and five agent workers")
    if _parse_bool(args.data_shuffle):
        raise ValueError("A8-LR train data must not be shuffled")
    if not _parse_bool(args.actor_use_dynamic_bsz):
        raise ValueError("A8-LR must retain dynamic actor batching")
    if not _parse_bool(args.reference_kl_enabled):
        raise ValueError("A8-LR must retain actor-loss reference KL")
    if args.reference_kl_loss_coef != 0.001 or args.reference_kl_loss_type != "low_var_kl":
        raise ValueError("A8-LR reference KL must remain low_var_kl at 0.001")
    if args.vllm_gpu_memory_utilization != 0.20:
        raise ValueError("A8-LR-30 requires vLLM utilization 0.20")
    if args.save_freq != 50 or args.max_actor_ckpt_to_keep != 2:
        raise ValueError("A8-LR must save every 50 steps and retain two actor checkpoints")
    if args.gamma != 1.0:
        raise ValueError("Trainer gamma must remain 1.0")

    code_paths = [
        "agent_r1/agent_flow/agent_flow.py",
        "agent_r1/trainer/main_agent_grpo.py",
        "agent_r1/trainer/ppo/core_algos.py",
        "agent_r1/trainer/ppo/ray_trainer.py",
        "agent_r1/trainer/rollout_jsonl.py",
        "recipes/hotpotqa/env/search_tool.py",
        "recipes/hotpotqa/evidence.py",
        "recipes/hotpotqa_lr/base.yaml",
        "recipes/hotpotqa_lr/dsl.py",
        "recipes/hotpotqa_lr/protocol.py",
        "recipes/hotpotqa_lr/prompts.py",
        "recipes/hotpotqa_lr/verifier.py",
        "recipes/hotpotqa_lr/reward_contract.py",
        "recipes/hotpotqa_lr/reward_fn.py",
        "recipes/hotpotqa_lr/agent_flow.py",
        "recipes/hotpotqa_lr/prepare_run.py",
        "examples/hotpotqa/run_rlvr.sh",
        "examples/hotpotqa_lr/run_lr.sh",
        "examples/hotpotqa_lr/run_lr_claim_source.sh",
    ]
    code_hashes = {}
    for relative in code_paths:
        path = project_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source missing: {path}")
        code_hashes[relative] = _sha256(path)

    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible_devices) != args.num_gpus:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES has {len(visible_devices)} entries, expected {args.num_gpus}"
        )
    prompt_hashes = {
        "system": _canonical_sha256(LR_SYSTEM_PROMPT),
        "user": _canonical_sha256(LR_USER_PROMPT),
        "finish_available": _canonical_sha256(LR_FINISH_AVAILABLE_PROMPT),
        "final_turn": _canonical_sha256(LR_FINAL_TURN_PROMPT),
    }
    schema_hashes = {
        "bootstrap": _canonical_sha256(BOOTSTRAP_TOOL_SCHEMAS),
        "search_or_finish": _canonical_sha256(SEARCH_OR_FINISH_TOOL_SCHEMAS),
        "finish": _canonical_sha256(FINISH_TOOL_SCHEMAS),
    }
    manifest = {
        "contract_version": MANIFEST_VERSION,
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": LR_REWARD_ARM.replace("_", "-"),
        "run_mode": "main",
        "output_dir": str(output_dir),
        "model": _model_identity(model_path),
        "selection": {
            "train": {"path": str(train_path), "source_rows": train_rows, "count": 30_000, "selection": "source_prefix"},
            "validation": {"path": str(validation_path), "source_rows": validation_rows, "count": 7_405},
        },
        "resources": {
            "cuda_visible_devices": visible_devices,
            "num_gpus": args.num_gpus,
            "agent_workers": args.agent_workers,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        },
        "training": {
            "algorithm": "GRPO",
            "credit_assignment": "step_causal",
            "train_batch_size": args.train_batch_size,
            "rollout_n": args.rollout_n,
            "grpo_micro_batch_size_per_gpu": args.grpo_micro_batch_size,
            "actor_max_token_len_per_gpu": args.actor_max_token_len_per_gpu,
            "actor_use_dynamic_bsz": True,
            "actor_lr": 1e-6,
            "data_shuffle": False,
            "total_training_steps": args.total_training_steps,
            "save_freq": args.save_freq,
            "max_actor_ckpt_to_keep": args.max_actor_ckpt_to_keep,
            "trainer_gamma": args.gamma,
            "reference_kl": {"enabled": True, "coefficient": 0.001, "loss_type": "low_var_kl", "placement": "actor_loss"},
        },
        "reward_contract": {
            "contract_id": LR_CONTRACT_VERSION,
            "formula": (
                f"{PRIMARY_CONTRACT.terminal_weight:.1f} * terminal_em + "
                f"{PRIMARY_CONTRACT.process_weight:.1f} * local_reward"
            ),
            "terminal_weight": PRIMARY_CONTRACT.terminal_weight,
            "process_weight": PRIMARY_CONTRACT.process_weight,
            "reward_horizon": PRIMARY_CONTRACT.reward_horizon,
            "dependency_taint_gamma": PRIMARY_CONTRACT.dependency_taint_gamma,
            "process_is_terminal_em_gated": False,
            "gold_answer_visible_to_verifier": False,
            "gold_evidence_visible_to_verifier": False,
            "verifier_version": VERIFIER_VERSION,
            "dsl_version": DSL_VERSION,
            "reason_step_format": REASON_STEP_FORMAT,
        },
        "actor_contract": {
            "first_search_is_uncredited_bootstrap": True,
            "max_searches": 3,
            "max_agent_flow_turns": 4,
            "reason_step_format": REASON_STEP_FORMAT,
            "prompt_sha256": prompt_hashes,
            "tool_schema_sha256": schema_hashes,
        },
        "artifact_paths": {
            "corpus_dir": str(corpus_dir),
            "evidence_sidecar": str(sidecar_path),
        },
        "code_sha256": code_hashes,
        "package_versions": _package_versions(),
    }
    _write_json_atomic(output_dir / "run_manifest.json", manifest)
    print(json.dumps({"status": "ok", "manifest": str(output_dir / "run_manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
