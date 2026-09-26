#!/usr/bin/env python3
"""Fail-closed manifest preparation for uniform A9 certificate-grounded runs."""

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

from recipes.hotpotqa.evidence import coerce_bool
from recipes.hotpotqa_a9.dsl import CERTIFICATE_SCHEMA_VERSION
from recipes.hotpotqa_a9.prompts import (
    BOOTSTRAP_TOOL_SCHEMAS,
    FINISH_TOOL_SCHEMAS,
    A9_FINAL_TURN_PROMPT,
    A9_FINISH_AVAILABLE_PROMPT,
    A9_SYSTEM_PROMPT,
    A9_USER_PROMPT,
    SEARCH_OR_FINISH_TOOL_SCHEMAS,
)
from recipes.hotpotqa_a9.reward_contract import (
    A9_CONTRACT_VERSION,
    CONTRACT_CERT_MIX,
    CONTRACT_FORMAT_STRICT,
)
from recipes.hotpotqa_a9.verifier import VERIFIER_VERSION

MANIFEST_VERSION = "hotpotqa-paper-matched-run-v3"


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
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-mode", choices=("main", "pilot64", "pilot2048"), default="main")
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
    parser.add_argument("--vllm-max-model-len", type=int, required=True)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--vllm-max-num-seqs", type=int, required=True)
    parser.add_argument("--vllm-kv-cache-memory-bytes", default="")
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--max-actor-ckpt-to-keep", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--agent-workers", type=int, required=True)
    parser.add_argument("--em-warmup-steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--allow-uncalibrated-launch", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, required=True)
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
    expected_rows = {"main": 10_000, "pilot64": 64, "pilot2048": 2048}[args.run_mode]
    scale = os.environ.get("AGENT_R1_MODEL_SCALE", "9b" if model_path.name == "Qwen3.5-9B" else "4b")
    expected_batch = (8 if scale == "9b" else 20) if args.run_mode == "main" else 16
    expected_steps = 500 if args.run_mode == "main" else expected_rows // expected_batch
    if args.train_max_samples != expected_rows or train_rows < args.train_max_samples:
        raise ValueError(f"{args.run_mode} requires the first {expected_rows} available training rows")
    if args.val_max_samples != 7_405 or validation_rows != 7_405:
        raise ValueError("Paper HotpotQA requires all 7,405 validation rows")
    if args.train_batch_size != expected_batch or args.total_training_steps != expected_steps:
        raise ValueError(f"{args.run_mode} requires batch {expected_batch} and {expected_steps} steps")
    if args.rollout_n != 4 or args.grpo_micro_batch_size not in (1, 2):
        raise ValueError("A9 requires rollout n=4 and micro-batch/GPU=1 or 2")
    if args.em_warmup_steps < 0:
        raise ValueError("A9 EM warmup steps must be non-negative")
    if args.gamma != 1.0:
        raise ValueError("Trainer gamma must remain 1.0")
    if args.seed != 42:
        raise ValueError("A9 requires frozen seed 42")

    code_paths = [
        "agent_r1/agent_flow/agent_flow.py",
        "agent_r1/verifier/reward.py",
        "agent_r1/evaluation/consistency.py",
        "recipes/hotpotqa_a9/reward_sources.py",
        "recipes/llm_judge/scoring.py",
        "agent_r1/trainer/main_agent_grpo.py",
        "agent_r1/trainer/ppo/core_algos.py",
        "agent_r1/trainer/ppo/ray_trainer.py",
        "agent_r1/trainer/rollout_jsonl.py",
        "recipes/reward_mixing.py",
        "recipes/hotpotqa/env/search_tool.py",
        "recipes/hotpotqa/evidence.py",
        "recipes/hotpotqa_a9/base.yaml",
        "recipes/hotpotqa_a9/dsl.py",
        "recipes/hotpotqa_a9/protocol.py",
        "recipes/hotpotqa_a9/prompts.py",
        "recipes/hotpotqa_a9/verifier.py",
        "recipes/hotpotqa_a9/reward_contract.py",
        "recipes/hotpotqa_a9/reward_fn.py",
        "recipes/hotpotqa_a9/agent_flow.py",
        "recipes/hotpotqa_a9/prepare_run.py",
        "scripts/hotpotqa/run_rlvr.sh",
        "scripts/hotpotqa/run_a9.sh",
    ]
    code_hashes = {}
    for relative in code_paths:
        path = project_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source missing: {path}")
        code_hashes[relative] = _sha256(path)

    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]

    prompt_hashes = {
        "system": _canonical_sha256(A9_SYSTEM_PROMPT),
        "user": _canonical_sha256(A9_USER_PROMPT),
        "finish_available": _canonical_sha256(A9_FINISH_AVAILABLE_PROMPT),
        "final_turn": _canonical_sha256(A9_FINAL_TURN_PROMPT),
    }
    schema_hashes = {
        "bootstrap": _canonical_sha256(BOOTSTRAP_TOOL_SCHEMAS),
        "search_or_finish": _canonical_sha256(SEARCH_OR_FINISH_TOOL_SCHEMAS),
        "finish": _canonical_sha256(FINISH_TOOL_SCHEMAS),
    }
    process_enabled = coerce_bool(os.environ.get("HOTPOTQA_A9_PROCESS_REWARD_ENABLED", "true"), name="HOTPOTQA_A9_PROCESS_REWARD_ENABLED")
    mode = {"A0": "terminal_only", "A1": "terminal_only", "A2": "gold_process_only",
            "A3": "gold_fixed_mix", "A6": "llm_process_uniform", "A7": "weak_fixed_mix"}.get(args.arm, "certificate_uniform")
    if not process_enabled:
        mode = "terminal_only"
    expected_terminal = 1.0 if mode == "terminal_only" else 0.0 if mode == "gold_process_only" else 0.5
    formula = {"terminal_only": "terminal_em", "gold_process_only": "gold_coverage",
               "gold_fixed_mix": "0.5 * terminal_em + 0.5 * gold_coverage",
               "weak_fixed_mix": "0.5 * terminal_em + 0.5 * observable_search_reward"}.get(
                   mode, "w * terminal_em + (1-w) * process_reward; one w per prompt group")
    manifest = {
        "contract_version": MANIFEST_VERSION,
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "run_mode": args.run_mode,
        "output_dir": str(output_dir),
        "model": _model_identity(model_path),
        "design_lineage": {
            "interaction_semantics": "terminal_trajectory_replay_with_step_credit_backfill",
            "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
            "verifier_version": VERIFIER_VERSION,
            "intentional_a9_differences": [
                "minimal_certificate_sidecar",
                "no_dependency_taint",
                "no_novelty_gating",
                "no_inference_verification",
                "no_ref_chain",
                "certificate_parse_failure_does_not_block_action",
            ],
        },
        "selection": {
            "train": {
                "path": str(train_path),
                "source_rows": train_rows,
                "count": args.train_max_samples,
                "selection": "source_prefix",
            },
            "validation": {"path": str(validation_path), "source_rows": validation_rows, "count": 7_405},
        },
        "resources": {
            "cuda_visible_devices": visible_devices,
            "num_gpus": args.num_gpus,
            "agent_workers": args.agent_workers,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
            "vllm_max_num_seqs": args.vllm_max_num_seqs,
        },
        "training": {
            "algorithm": os.environ.get("HOTPOTQA_OPTIMIZER", "grpo").upper(),
            "credit_assignment": "step_causal",
            "train_max_samples": args.train_max_samples,
            "sampled_prompt_count": args.train_batch_size * args.total_training_steps,
            "train_batch_size": args.train_batch_size,
            "rollout_n": args.rollout_n,
            "grpo_micro_batch_size_per_gpu": args.grpo_micro_batch_size,
            "actor_max_token_len_per_gpu": args.actor_max_token_len_per_gpu,
            "actor_use_dynamic_bsz": True,
            "actor_lr": 1e-6,
            "seed": args.seed,
            "data_shuffle": False,
            "total_training_steps": args.total_training_steps,
            "em_warmup_steps": args.em_warmup_steps,
            "save_freq": args.save_freq,
            "max_actor_ckpt_to_keep": args.max_actor_ckpt_to_keep,
            "trainer_gamma": args.gamma,
            "reference_kl": {
                "enabled": True,
                "coefficient": 0.001,
                "loss_type": "low_var_kl",
                "placement": "actor_loss",
            },
        },
        "reward_contract": {
            "contract_id": A9_CONTRACT_VERSION,
            "formula": formula,
            "terminal_weight": expected_terminal,
            "process_weight": 1.0 - expected_terminal,
            "weight_sampling": "uniform_0_1_per_prompt_group" if mode.endswith("uniform") else "fixed",
            "format_gate": True,
            "reward_source": mode,
            "reward_horizon": CONTRACT_CERT_MIX.reward_horizon,
            "process_is_terminal_em_gated": False,
            "gold_answer_visible_to_verifier": False,
            "gold_evidence_visible_to_verifier": args.arm in {"A2", "A3"},
            "verifier_version": VERIFIER_VERSION,
            "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
            "schedule": {
                "type": mode,
                "em_warmup_steps": args.em_warmup_steps,
                "warmup_formula": "terminal_em" if mode.endswith("uniform") else formula,
                "post_warmup_formula": formula,
            },
        },
        "actor_contract": {
            "first_search_is_uncredited_bootstrap": True,
            "max_searches": 3,
            "max_agent_flow_turns": 4,
            "verifier_timing": "terminal_trajectory_replay",
            "process_credit_application": "backfill_to_transition_steps",
            "certificate_parse_failure_blocks_action": bool(
                coerce_bool(os.environ.get("HOTPOTQA_A9_FORMAT_GATE", "0"), name="HOTPOTQA_A9_FORMAT_GATE")
            ),
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
