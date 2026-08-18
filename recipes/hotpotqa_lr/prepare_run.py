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
from recipes.hotpotqa_lr.reward_contract import (
    LR_CONTRACT_VERSION,
    LR_REWARD_ARM,
    PRIMARY_CONTRACT,
    REWARD_MODE_LR30,
    REWARD_MODE_TERMINAL_ONLY,
    contract_for_mode,
    expected_reward_arm,
    resolve_reward_mode,
)
from recipes.hotpotqa_lr.verifier import VERIFIER_VERSION

MANIFEST_VERSION = "hotpotqa-a8-lr-run-v2"
RESOURCE_CONTRACT_VERSION = "hotpotqa-a8-20260814-runtime-6gpu-auto-kv-v1"
INTERACTION_REFERENCE_RUN = "qwen35-4b_a8_lr30_main30k_n4_1500step_5gpu_vllm025_mlen8192_mseq20_save50_20260814-081225"
RUNTIME_REFERENCE_RUN = INTERACTION_REFERENCE_RUN


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
    parser.add_argument("--reward-mode", required=True)
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
    parser.add_argument("--em-warmup-steps", type=int, default=0)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--calibration-report", required=True)
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
    if args.train_max_samples != 30_000 or train_rows < args.train_max_samples:
        raise ValueError("A8-LR main run requires the first 30,000 training rows")
    if args.val_max_samples != 7_405 or validation_rows != 7_405:
        raise ValueError("A8-LR requires all 7,405 validation rows")
    if args.train_batch_size != 20 or args.total_training_steps != 1_500:
        raise ValueError("A8-LR requires batch 20 and exactly 1,500 steps")
    if args.rollout_n != 4 or args.grpo_micro_batch_size not in (1, 2):
        raise ValueError("A8-LR requires rollout n=4 and micro-batch/GPU=1 or 2")
    reward_mode = resolve_reward_mode(args.reward_mode)
    active_contract = contract_for_mode(reward_mode)
    expected_arm = expected_reward_arm(REASON_STEP_FORMAT, reward_mode)
    if args.arm != expected_arm or LR_REWARD_ARM != expected_arm:
        raise ValueError(f"A8-LR arm/reward contract mismatch: got {args.arm}, expected {expected_arm}")
    if args.num_gpus != 6 or args.agent_workers != 6:
        raise ValueError("A8-LR-v2 requires exactly six GPUs and six agent workers")
    if _parse_bool(args.data_shuffle):
        raise ValueError("A8-LR train data must not be shuffled")
    if not _parse_bool(args.actor_use_dynamic_bsz):
        raise ValueError("A8-LR must retain dynamic actor batching")
    if not _parse_bool(args.reference_kl_enabled):
        raise ValueError("A8-LR must retain actor-loss reference KL")
    if args.reference_kl_loss_coef != 0.001 or args.reference_kl_loss_type != "low_var_kl":
        raise ValueError("A8-LR reference KL must remain low_var_kl at 0.001")
    if args.vllm_gpu_memory_utilization != 0.25:
        raise ValueError("A8-LR-v2 requires vLLM utilization 0.25")
    if args.vllm_max_model_len != 8_192 or args.vllm_max_num_batched_tokens != 8_192 or args.vllm_max_num_seqs != 20:
        raise ValueError("A8-LR-v2 requires the 2026-08-14 A8 vLLM runtime shape 8192/8192/20")
    if str(args.vllm_kv_cache_memory_bytes).strip():
        raise ValueError("A8-LR-v2 requires automatic KV-cache sizing; explicit bytes are forbidden")
    if args.save_freq != 50 or args.max_actor_ckpt_to_keep != 2:
        raise ValueError("A8-LR must save every 50 steps and retain two actor checkpoints")
    if args.em_warmup_steps < 0:
        raise ValueError("A8-LR EM warmup steps must be non-negative")
    if reward_mode == REWARD_MODE_TERMINAL_ONLY and args.em_warmup_steps != 0:
        raise ValueError("A8-LR BASE cannot use an EM warmup schedule")
    if reward_mode == REWARD_MODE_LR30 and args.em_warmup_steps not in {0, 100}:
        raise ValueError("A8-LR-v2 treatment schedule must be constant LR30 or EM100 then LR30")
    if args.gamma != 1.0:
        raise ValueError("Trainer gamma must remain 1.0")
    if args.seed != 42:
        raise ValueError("A8-LR-v2 requires frozen seed 42")
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
        "examples/hotpotqa_lr/run_lr_em100.sh",
        "examples/hotpotqa_lr/run_lr_claim_source.sh",
        "examples/hotpotqa_lr/run_lr_claim_source_em100.sh",
        "examples/hotpotqa_lr/run_lr_base.sh",
        "examples/hotpotqa_lr/common_v2.sh",
    ]
    code_hashes = {}
    for relative in code_paths:
        path = project_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source missing: {path}")
        code_hashes[relative] = _sha256(path)

    calibration_report_path: Path | None = None
    if args.allow_uncalibrated_launch:
        if args.calibration_report.strip():
            raise ValueError("An uncalibrated launch cannot also provide a calibration report")
        calibration_metadata = {
            "status": "waived_by_user",
            "report": None,
            "reason": "user_requested_direct_1500_step_launch",
        }
    else:
        calibration_report_path = Path(args.calibration_report).resolve()
        if not calibration_report_path.is_file():
            raise FileNotFoundError(f"A8-LR calibration report missing: {calibration_report_path}")
        calibration_report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
        if calibration_report.get("contract_version") != "hotpotqa-a8-lr-calibration-report-v1":
            raise ValueError("A8-LR calibration report contract is unsupported")
        if calibration_report.get("passed") is not True:
            raise ValueError("A8-LR calibration gates did not pass")
        if calibration_report.get("seed") != args.seed:
            raise ValueError("A8-LR calibration seed does not match the training seed")
        if (
            calibration_report.get("reason_step_format") != REASON_STEP_FORMAT
            or calibration_report.get("dsl_version") != DSL_VERSION
            or calibration_report.get("verifier_version") != VERIFIER_VERSION
        ):
            raise ValueError("A8-LR calibration report does not match the active verifier contract")
        calibration_hashes = calibration_report.get("code_sha256")
        calibration_code_paths = (
            "recipes/hotpotqa_lr/agent_flow.py",
            "recipes/hotpotqa_lr/base.yaml",
            "recipes/hotpotqa_lr/dsl.py",
            "recipes/hotpotqa_lr/prompts.py",
            "recipes/hotpotqa_lr/protocol.py",
            "recipes/hotpotqa_lr/reward_contract.py",
            "recipes/hotpotqa_lr/reward_fn.py",
            "recipes/hotpotqa_lr/verifier.py",
        )
        if not isinstance(calibration_hashes, dict) or any(
            calibration_hashes.get(relative) != code_hashes[relative] for relative in calibration_code_paths
        ):
            raise ValueError("A8-LR code changed after calibration; rerun calibration")
        active_model_identity = _model_identity(model_path)
        calibrated_model = calibration_report.get("model")
        if not isinstance(calibrated_model, dict) or (
            calibrated_model.get("path") != active_model_identity["path"]
            or calibrated_model.get("identity_sha256") != active_model_identity["identity_sha256"]
        ):
            raise ValueError("A8-LR model path or identity does not match the calibrated policy")
        calibration_metadata = {
            "status": "passed",
            "report": str(calibration_report_path),
            "report_sha256": _sha256(calibration_report_path),
        }

    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible_devices) != args.num_gpus:
        raise ValueError(f"CUDA_VISIBLE_DEVICES has {len(visible_devices)} entries, expected {args.num_gpus}")
    if len(set(visible_devices)) != len(visible_devices):
        raise ValueError("CUDA_VISIBLE_DEVICES entries must be unique")
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
        "arm": args.arm.replace("_", "-"),
        "run_mode": "main",
        "output_dir": str(output_dir),
        "model": _model_identity(model_path),
        "calibration": calibration_metadata,
        "design_lineage": {
            "interaction_reference_run": INTERACTION_REFERENCE_RUN,
            "runtime_reference_run": RUNTIME_REFERENCE_RUN,
            "interaction_semantics": "terminal_trajectory_replay_with_step_credit_backfill",
            "runtime_compatibility": "six_gpu_with_20260814_a8_runtime_shape",
            "reward_compatibility": "lr30_or_terminal_only_with_optional_em100_schedule",
            "intentional_v2_differences": [
                "select_exact_span_not_creditable",
                "operation_specific_actor_schema",
                "novelty_gating",
                "fail_closed_calibration",
            ],
        },
        "selection": {
            "train": {
                "path": str(train_path),
                "source_rows": train_rows,
                "count": 30_000,
                "selection": "source_prefix",
            },
            "validation": {"path": str(validation_path), "source_rows": validation_rows, "count": 7_405},
        },
        "resources": {
            "contract_id": RESOURCE_CONTRACT_VERSION,
            "cuda_visible_devices": visible_devices,
            "num_gpus": args.num_gpus,
            "agent_workers": args.agent_workers,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
            "vllm_max_num_seqs": args.vllm_max_num_seqs,
            "vllm_kv_cache_memory_bytes": None,
            "kv_cache_sizing": "automatic",
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
            "contract_id": LR_CONTRACT_VERSION,
            "reward_mode": reward_mode,
            "formula": (
                f"{active_contract.terminal_weight:.1f} * terminal_em + "
                f"{active_contract.process_weight:.1f} * local_reward"
            ),
            "terminal_weight": active_contract.terminal_weight,
            "process_weight": active_contract.process_weight,
            "reward_horizon": PRIMARY_CONTRACT.reward_horizon,
            "dependency_taint_gamma": PRIMARY_CONTRACT.dependency_taint_gamma,
            "process_is_terminal_em_gated": False,
            "gold_answer_visible_to_verifier": False,
            "gold_evidence_visible_to_verifier": False,
            "verifier_version": VERIFIER_VERSION,
            "dsl_version": DSL_VERSION,
            "reason_step_format": REASON_STEP_FORMAT,
            "schedule": {
                "type": (
                    "terminal_only"
                    if reward_mode == REWARD_MODE_TERMINAL_ONLY
                    else "em_warmup_then_lr30"
                    if args.em_warmup_steps
                    else "constant_lr30"
                ),
                "em_warmup_steps": args.em_warmup_steps,
                "warmup_formula": "1.0 * terminal_em + 0.0 * local_reward",
                "post_warmup_formula": (
                    f"{active_contract.terminal_weight:.1f} * terminal_em + "
                    f"{active_contract.process_weight:.1f} * local_reward"
                ),
            },
        },
        "actor_contract": {
            "first_search_is_uncredited_bootstrap": True,
            "max_searches": 3,
            "max_agent_flow_turns": 4,
            "verifier_timing": "terminal_trajectory_replay",
            "process_credit_application": "backfill_to_transition_steps",
            "reason_step_format": REASON_STEP_FORMAT,
            "prompt_sha256": prompt_hashes,
            "tool_schema_sha256": schema_hashes,
        },
        "artifact_paths": {
            "corpus_dir": str(corpus_dir),
            "evidence_sidecar": str(sidecar_path),
            **(
                {
                    "calibration_report": str(calibration_report_path),
                    "calibration_report_sha256": _sha256(calibration_report_path),
                }
                if calibration_report_path is not None
                else {}
            ),
        },
        "code_sha256": code_hashes,
        "package_versions": _package_versions(),
    }
    _write_json_atomic(output_dir / "run_manifest.json", manifest)
    print(json.dumps({"status": "ok", "manifest": str(output_dir / "run_manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
