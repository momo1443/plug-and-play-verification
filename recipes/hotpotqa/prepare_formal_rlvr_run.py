#!/usr/bin/env python3
"""Fail-closed preflight for formal HotpotQA GRPO runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from recipes.hotpotqa.evidence import EVIDENCE_SCHEMA_VERSION, OfficialEvidenceStore
from recipes.hotpotqa.final_answer_protocol import resolve_final_answer_protocol
from recipes.hotpotqa.judge_prompts import JUDGE_VERSION as JUDGE_PROMPT_VERSION
from recipes.hotpotqa.a9_behavioral_verifier import (
    A9_ENTITY_LIBRARY_PATH,
    A9_PROBE_GENERATOR_VERSION,
    A9_VERIFIER_VERSION,
)
from recipes.hotpotqa.process_verifier import PROCESS_VERIFIER_VERSION, WEAK_EXECUTION_VERIFIER_VERSION
from recipes.hotpotqa.reward_arm import (
    RewardArm,
    TrainingRewardContract,
    parse_reward_arm,
    training_reward_contract,
)
from recipes.hotpotqa.validate_formal_a0_artifacts import (
    PREFLIGHT_CACHE_VERSION,
    fingerprint,
    load_cache,
    validate_formal_a0_artifacts,
    write_cache_atomic,
)

CONTRACT_VERSION = "hotpotqa-formal-rlvr-grpo-v14"
# Frozen read-only artifacts are re-validated on every launch; caching keyed on
# (size, mtime_ns) turns the ~11-min NAS re-hash/scan into stat() lookups.
PREFLIGHT_CACHE_FILENAME = "preflight_cache_v1.json"
FORMAL_TRAIN_ROWS = 90_447
FORMAL_TRAIN_SELECTION_ROWS = 30_000
FORMAL_VALIDATION_ROWS = 7_405
_RUN_MODE_TRAIN_ROWS = {
    "pilot64": 64,
    "pilot2048": 2_048,
    "main": FORMAL_TRAIN_SELECTION_ROWS,
}
_RUN_MODES = set(_RUN_MODE_TRAIN_ROWS)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected true or false, got {value!r}")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "transformers", "vllm", "verl", "ray"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


def _reward_description(contract: TrainingRewardContract, *, arm: RewardArm | None = None) -> str:
    """Describe optimizer-visible rewards from the frozen arm contract."""

    process_label = {
        RewardArm.A6: JUDGE_PROMPT_VERSION,
        RewardArm.A7: WEAK_EXECUTION_VERIFIER_VERSION,
        RewardArm.A9: A9_VERIFIER_VERSION,
    }.get(arm, PROCESS_VERIFIER_VERSION)
    components: list[str] = []
    for weight, label in (
        (contract.process_weight, process_label),
        (contract.terminal_weight, "terminal exact match"),
    ):
        if weight == 0.0:
            continue
        components.append(label if weight == 1.0 else f"{weight:g} * {label}")
    return " + ".join(components) if components else "none"


def _code_hashes(project_dir: Path, arm: RewardArm) -> dict[str, str]:
    paths = [
        "agent_r1/agent_flow/agent_flow.py",
        "agent_r1/config/agent_rl_trainer.yaml",
        "agent_r1/reward_loop/reward_loop.py",
        "agent_r1/trainer/main_agent_grpo.py",
        "agent_r1/trainer/main_agent_rl.py",
        "agent_r1/trainer/ppo/core_algos.py",
        "agent_r1/trainer/ppo/metric_utils.py",
        "agent_r1/trainer/ppo/ray_trainer.py",
        "agent_r1/trainer/ppo/trajectory_batching.py",
        "agent_r1/trainer/rollout_jsonl.py",
        "agent_r1/trainer/training_outputs.py",
        "agent_r1/workers/engine_workers.py",
        "recipes/hotpotqa/base.yaml",
        "recipes/hotpotqa/env/search_tool.py",
        "recipes/hotpotqa/evidence.py",
        "recipes/hotpotqa/hotpotqa_agent_flow.py",
        "recipes/hotpotqa/final_answer_protocol.py",
        "recipes/hotpotqa/output_parsing.py",
        "recipes/hotpotqa/prepare_formal_rlvr_run.py",
        "recipes/hotpotqa/process_verifier.py",
        "recipes/hotpotqa/prompts.py",
        "recipes/hotpotqa/reward_arm.py",
        "recipes/hotpotqa/reward_fn.py",
        "recipes/hotpotqa/validate_formal_a0_artifacts.py",
        "recipes/hotpotqa/judge_prompts.py",
        "recipes/hotpotqa/judge_server.py",
        "examples/hotpotqa/run_rlvr.sh",
        f"examples/hotpotqa/run_{arm.value.lower()}.sh",
    ]
    if arm is RewardArm.A9:
        paths.extend(
            [
                "recipes/hotpotqa/a9_behavioral_verifier.py",
                "recipes/hotpotqa/a9_entity_library.json",
            ]
        )
    hashes: dict[str, str] = {}
    for relative_path in paths:
        path = project_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required RLVR code file is missing: {path}")
        hashes[relative_path] = _sha256(path)
    return hashes


_MODEL_PARAM_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "hidden_act",
    "tie_word_embeddings",
    "dtype",
    "torch_dtype",
)


def _architecture_params(config: dict[str, Any]) -> dict[str, Any]:
    """Pull the readable transformer knobs from HF config (incl. Qwen3.5 text_config)."""

    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    params: dict[str, Any] = {}
    for key in _MODEL_PARAM_KEYS:
        if key in text:
            params[key] = text[key]
        elif key in config:
            params[key] = config[key]
    if "transformers_version" in config:
        params["transformers_version"] = config["transformers_version"]
    return params


def _model_identity(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    supported_model_types = {"qwen3_5", "qwen3"}
    if config.get("model_type") not in supported_model_types:
        raise ValueError(f"Expected model_type in {sorted(supported_model_types)}, got {config.get('model_type')!r}")
    identity_files = [config_path]
    for filename in ("model.safetensors.index.json", "tokenizer_config.json"):
        candidate = model_path / filename
        if candidate.is_file():
            identity_files.append(candidate)
    return {
        "path": str(model_path.resolve()),
        "name": model_path.name,
        "model_type": config["model_type"],
        "architectures": config.get("architectures", []),
        "params": _architecture_params(config),
        "identity_sha256": {path.name: _sha256(path) for path in identity_files},
    }


def _cached_selection_digest(
    store: OfficialEvidenceStore,
    split: str,
    limit: int,
    *,
    sidecar_path: Path,
    cache_path: Path | None,
    use_cache: bool,
) -> dict[str, Any]:
    """Selection digest that reuses a cached result when the sidecar is unchanged."""

    key = {
        "cache_version": PREFLIGHT_CACHE_VERSION,
        "split": split,
        "limit": limit,
        "sidecar_fingerprint": fingerprint(sidecar_path),
    }
    slot = f"{split}:{limit}"
    if use_cache:
        cached = (load_cache(cache_path).get("selection") or {}).get(slot)
        if isinstance(cached, dict) and cached.get("key") == key:
            return cached["digest"]
    digest = _selection_digest(store, split, limit)
    if use_cache and cache_path is not None:
        cache = load_cache(cache_path)
        cache.setdefault("selection", {})[slot] = {"key": key, "digest": digest}
        write_cache_atomic(cache_path, cache)
    return digest


def _selection_digest(store: OfficialEvidenceStore, split: str, limit: int) -> dict[str, Any]:
    samples = store.list_samples(split, limit=limit)
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.official_qid.encode("utf-8"))
        digest.update(b"\n")
    return {
        "split": split,
        "count": len(samples),
        "ordered_sample_qid_sha256": digest.hexdigest(),
        "first_sample_key": samples[0].sample_key if samples else None,
        "last_sample_key": samples[-1].sample_key if samples else None,
    }


def _judge_manifest_config(
    project_dir: Path,
    reward_arm: RewardArm,
) -> dict[str, Any] | None:
    """Return the frozen judge manifest for A6 only."""
    if reward_arm is not RewardArm.A6:
        return None
    workspace_dir = project_dir.parent
    default_model_name = "Qwen3-4B"
    judge_model_path = (
        Path(
            os.environ.get(
                "HOTPOTQA_JUDGE_MODEL",
                str(workspace_dir / "models" / default_model_name),
            )
        )
        .expanduser()
        .resolve()
    )
    return {
        "model": _model_identity(judge_model_path),
        "judge_version": JUDGE_PROMPT_VERSION,
        "verifier_version": PROCESS_VERIFIER_VERSION,
        "temperature": 0.0,
        "thinking": False,
        "precision": "bfloat16",
        "max_model_len": int(os.environ.get("HOTPOTQA_JUDGE_MAX_MODEL_LEN", "4096")),
        "completion_max_tokens": int(
            os.environ.get(
                "HOTPOTQA_JUDGE_COMPLETION_MAX_TOKENS",
                "256",
            )
        ),
        "max_retries": 3,
        "request_timeout_s": 10,
        "cache": "sha256-exact-input-success-only; separate prompt-version namespace",
        "placement": {
            "external_server": os.environ.get("HOTPOTQA_JUDGE_EXTERNAL", "").strip().lower()
            in {"1", "true", "yes", "on"},
            "gpu_id": int(os.environ.get("HOTPOTQA_JUDGE_GPU", "1")),
            "port": int(os.environ.get("HOTPOTQA_JUDGE_PORT", "29500")),
            "gpu_memory_utilization": float(os.environ.get("HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION", "0.50")),
        },
    }


def prepare_formal_rlvr_run(
    *,
    project_dir: Path,
    arm: str,
    run_mode: str,
    train_path: Path,
    validation_path: Path,
    corpus_dir: Path,
    evidence_sidecar_path: Path,
    model_path: Path,
    output_dir: Path,
    train_max_samples: int,
    val_max_samples: int,
    train_batch_size: int,
    rollout_n: int,
    total_training_steps: int,
    num_gpus: int,
    agent_workers: int,
    gamma: float,
    grpo_micro_batch_size: int = 1,
    data_shuffle: bool = False,
    actor_use_dynamic_bsz: bool = True,
    actor_max_token_len_per_gpu: int = 8_192,
    entropy_chunk_rows: int = 512,
    calculate_entropy: bool = False,
    reference_kl_enabled: bool = True,
    reference_kl_loss_coef: float = 0.001,
    reference_kl_loss_type: str = "low_var_kl",
    kl_in_reward: bool = False,
    use_fused_kernels: bool = True,
    fused_kernel_backend: str = "triton",
    actor_param_offload: bool = False,
    actor_optimizer_offload: bool = False,
    enable_gradient_checkpointing: bool = True,
    vllm_gpu_memory_utilization: float = 0.4,
    vllm_enable_sleep_mode: bool = False,
    vllm_free_cache_engine: bool = False,
    save_freq: int = 100,
    max_actor_ckpt_to_keep: int = 2,
    resume_mode: str = "disable",
    resume_from_path: str | None = None,
) -> dict[str, Any]:
    reward_arm = parse_reward_arm(arm)
    final_answer_protocol = resolve_final_answer_protocol()
    trainable_arms = {
        RewardArm.A1,
        RewardArm.A2,
        RewardArm.A3,
        RewardArm.A6,
        RewardArm.A7,
        RewardArm.A9,
    }
    if reward_arm not in trainable_arms:
        raise ValueError(
            "The RLVR training entrypoint only accepts "
            + ", ".join(arm.value for arm in sorted(trainable_arms, key=lambda arm: arm.value))
        )
    reward_contract = training_reward_contract(reward_arm)
    if run_mode not in _RUN_MODES:
        raise ValueError(f"run_mode must be one of {sorted(_RUN_MODES)}, got {run_mode!r}")
    if run_mode == "main" and num_gpus not in {4, 6, 7, 8}:
        raise ValueError(f"main runs require 4, 6, 7, or 8 GPUs, got {num_gpus}")
    for name, value in {
        "train_max_samples": train_max_samples,
        "val_max_samples": val_max_samples,
        "train_batch_size": train_batch_size,
        "rollout_n": rollout_n,
        "total_training_steps": total_training_steps,
        "num_gpus": num_gpus,
        "agent_workers": agent_workers,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if rollout_n < 2:
        raise ValueError("GRPO requires at least two rollouts per question")
    if grpo_micro_batch_size <= 0:
        raise ValueError("grpo_micro_batch_size must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if actor_max_token_len_per_gpu <= 0:
        raise ValueError("actor_max_token_len_per_gpu must be positive")
    if entropy_chunk_rows <= 0:
        raise ValueError("entropy_chunk_rows must be positive")
    if not reference_kl_enabled:
        raise ValueError("Formal trainable arms require verl/GRPO's built-in actor-loss reference KL")
    if not math.isfinite(reference_kl_loss_coef) or reference_kl_loss_coef <= 0.0:
        raise ValueError("reference_kl_loss_coef must be finite and positive")
    if reference_kl_loss_type != "low_var_kl":
        raise ValueError("Formal trainable arms require reference_kl_loss_type='low_var_kl'")
    if kl_in_reward:
        raise ValueError("Formal trainable arms keep reference KL out of the task reward")
    if fused_kernel_backend not in {"triton", "torch"}:
        raise ValueError("fused_kernel_backend must be 'triton' or 'torch'")
    if not 0.0 < vllm_gpu_memory_utilization < 1.0:
        raise ValueError("vllm_gpu_memory_utilization must be in (0, 1)")
    if save_freq <= 0:
        raise ValueError("save_freq must be positive")
    if max_actor_ckpt_to_keep <= 0:
        raise ValueError("max_actor_ckpt_to_keep must be positive")
    if resume_mode not in {"disable", "auto", "resume_path"}:
        raise ValueError("resume_mode must be 'disable', 'auto', or 'resume_path'")
    if resume_mode == "resume_path":
        if not resume_from_path:
            raise ValueError("resume_from_path is required for resume_mode='resume_path'")
        resume_checkpoint = Path(resume_from_path).expanduser().resolve()
        if not resume_checkpoint.is_dir() or not resume_checkpoint.name.startswith("global_step_"):
            raise ValueError(f"Invalid resume checkpoint directory: {resume_checkpoint}")
        resume_from_path = str(resume_checkpoint)
        resume_global_step = int(resume_checkpoint.name.removeprefix("global_step_"))
    else:
        if resume_from_path:
            raise ValueError("resume_from_path is only valid for resume_mode='resume_path'")
        resume_from_path = None
        resume_global_step = None

    for path in (train_path, validation_path, evidence_sidecar_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required RLVR artifact is missing: {path}")
    train_rows = int(pq.read_metadata(train_path).num_rows)
    val_rows = int(pq.read_metadata(validation_path).num_rows)
    if train_rows != FORMAL_TRAIN_ROWS or val_rows != FORMAL_VALIDATION_ROWS:
        raise ValueError(
            "Formal training data row counts do not match the frozen dataset; "
            f"expected train={FORMAL_TRAIN_ROWS}, "
            f"validation={FORMAL_VALIDATION_ROWS}, "
            f"got train={train_rows}, validation={val_rows}"
        )
    if train_max_samples > train_rows or val_max_samples > val_rows:
        raise ValueError("Requested sample count exceeds the available parquet rows")
    if val_max_samples != FORMAL_VALIDATION_ROWS:
        raise ValueError(
            "Formal training must retain the complete validation split: "
            f"expected {FORMAL_VALIDATION_ROWS}, got {val_max_samples}"
        )
    expected_train_rows = _RUN_MODE_TRAIN_ROWS[run_mode]
    if train_max_samples != expected_train_rows:
        raise ValueError(
            f"A {run_mode} run must use the first {expected_train_rows:,} rows of its training split"
        )
    if data_shuffle:
        raise ValueError(
            f"A {run_mode} run must set data_shuffle=false so max_samples selects the deterministic source prefix"
        )

    store = OfficialEvidenceStore(evidence_sidecar_path)
    if (
        store.sample_count("train") != FORMAL_TRAIN_ROWS
        or store.sample_count("validation") != FORMAL_VALIDATION_ROWS
    ):
        raise ValueError("Evidence sidecar sample counts do not match the frozen base dataset")
    for split in ("train", "validation"):
        total = int(store.metadata[f"gold_fact_count_{split}"])
        unresolved = int(store.metadata[f"unresolved_gold_fact_count_{split}"])
        mapping_rate = (total - unresolved) / total
        if mapping_rate < 0.99:
            raise ValueError(f"{split} gold mapping rate {mapping_rate:.6f} is below 0.99")

    use_cache = os.environ.get("HOTPOTQA_PREFLIGHT_NO_CACHE", "0") != "1"
    cache_path = corpus_dir / PREFLIGHT_CACHE_FILENAME
    artifact_preflight = validate_formal_a0_artifacts(
        validation_path=validation_path,
        corpus_dir=corpus_dir,
        evidence_sidecar_path=evidence_sidecar_path,
        cache_path=cache_path,
        use_cache=use_cache,
    )
    cuda_devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if cuda_devices and len(cuda_devices) != num_gpus:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES exposes {len(cuda_devices)} devices, expected {num_gpus}: {cuda_devices}"
        )
    judge_manifest = _judge_manifest_config(project_dir, reward_arm)
    a9_probe_seed = None
    if reward_arm is RewardArm.A9:
        try:
            a9_probe_seed = int(os.environ.get("HOTPOTQA_A9_PROBE_SEED", "42"))
        except ValueError as exc:
            raise ValueError("HOTPOTQA_A9_PROBE_SEED must be an integer") from exc

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": reward_arm.value,
        "run_mode": run_mode,
        "final_answer_protocol": final_answer_protocol,
        "output_dir": str(output_dir),
        "scientific_config": {
            "algorithm": "GRPO",
            "final_answer_protocol": final_answer_protocol,
            "grpo_credit_assignment": "step_causal",
            "gamma": gamma,
            "reward": _reward_description(reward_contract, arm=reward_arm),
            "process_reward": (
                JUDGE_PROMPT_VERSION
                if reward_arm == RewardArm.A6
                else WEAK_EXECUTION_VERIFIER_VERSION
                if reward_arm == RewardArm.A7
                else A9_VERIFIER_VERSION
                if reward_arm == RewardArm.A9
                else PROCESS_VERIFIER_VERSION
            ),
            "process_reward_weight": reward_contract.process_weight,
            "process_reward_max_executed_searches": 3 if reward_arm is RewardArm.A7 else None,
            "process_is_terminal_em_gated": False,
            "gold_answer_visible_to_process_verifier": False if reward_arm is RewardArm.A9 else None,
            "gold_evidence_visible_to_process_verifier": False if reward_arm is RewardArm.A9 else None,
            "a9_probe_generator_version": (
                A9_PROBE_GENERATOR_VERSION if reward_arm is RewardArm.A9 else None
            ),
            "a9_probe_seed": a9_probe_seed,
            "a9_entity_library_path": (
                str(A9_ENTITY_LIBRARY_PATH.resolve()) if reward_arm is RewardArm.A9 else None
            ),
            "a9_entity_library_sha256": (
                _sha256(A9_ENTITY_LIBRARY_PATH) if reward_arm is RewardArm.A9 else None
            ),
            "a9_probe_count_per_eligible_turn": 2 if reward_arm is RewardArm.A9 else None,
            "a9_process_score": (
                "binary: 1 if both pass, 0 if both fail, mask if partial/invalid"
                if reward_arm is RewardArm.A9
                else None
            ),
            "a9_verdict_classes": (
                {
                    "A": "valid; both probes pass; optimizer value 1",
                    "B": "valid; both probes fail; optimizer value 0",
                    "C": "probe construction or scoring invalid; process component masked",
                    "D": "valid partial result; raw diagnostic 0.5; process component masked",
                }
                if reward_arm is RewardArm.A9
                else None
            ),
            "a9_invalid_process_handling": (
                "mask partial or invalid local process component; keep trajectory and terminal EM"
                if reward_arm is RewardArm.A9
                else None
            ),
            "a9_validation_probe_enabled": False if reward_arm is RewardArm.A9 else None,
            "terminal_reward": "terminal exact match",
            "terminal_reward_weight": reward_contract.terminal_weight,
            "final_response_mask": reward_contract.final_response_mask,
            "validation_reward": "terminal exact match",
            "max_steps": 4,
            "max_model_searches": 3,
            "minimum_searches_before_finish": 0,
            "early_finish_after_search": False,
            "force_first_search": False,
            "thinking_mode": "disabled",
            "tool_parser": "hermes",
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "initialization": "configured-checkpoint",
            "schema_sft": False,
            "reference_kl": {
                "enabled": reference_kl_enabled,
                "placement": "actor_loss",
                "loss_type": reference_kl_loss_type,
                "coefficient": reference_kl_loss_coef,
                "in_reward": kl_in_reward,
            },
        },
        "training": {
            "source_train_rows": train_rows,
            "source_validation_rows": val_rows,
            "train_max_samples": train_max_samples,
            "train_selection": "source_prefix" if not data_shuffle else "seeded_random_subset",
            "data_shuffle": data_shuffle,
            "val_max_samples": val_max_samples,
            "train_batch_size": train_batch_size,
            "grpo_mini_batch_size": train_batch_size,
            "grpo_micro_batch_size_per_gpu": grpo_micro_batch_size,
            "actor_use_dynamic_bsz": actor_use_dynamic_bsz,
            "actor_max_token_len_per_gpu": actor_max_token_len_per_gpu,
            "entropy_chunk_rows": entropy_chunk_rows,
            "calculate_entropy": calculate_entropy,
            "entropy_coeff": 0.0,
            "use_fused_kernels": use_fused_kernels,
            "fused_kernel_backend": fused_kernel_backend,
            "use_remove_padding": False,
            "actor_param_offload": actor_param_offload,
            "actor_optimizer_offload": actor_optimizer_offload,
            "enable_gradient_checkpointing": enable_gradient_checkpointing,
            "rollout_n": rollout_n,
            "total_training_steps": total_training_steps,
            "save_freq": save_freq,
            "max_actor_ckpt_to_keep": max_actor_ckpt_to_keep,
            "resume_mode": resume_mode,
            "resume_from_path": resume_from_path,
            "resume_global_step": resume_global_step,
        },
        "resources": {
            "num_gpus": num_gpus,
            "agent_workers": agent_workers,
            "cuda_visible_devices": cuda_devices,
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            "vllm_enable_sleep_mode": vllm_enable_sleep_mode,
            "vllm_free_cache_engine": vllm_free_cache_engine,
        },
        "model": _model_identity(model_path),
        "judge": judge_manifest,
        "selection": {
            "train": _cached_selection_digest(
                store,
                "train",
                train_max_samples,
                sidecar_path=evidence_sidecar_path,
                cache_path=cache_path,
                use_cache=use_cache,
            ),
            "validation": _cached_selection_digest(
                store,
                "validation",
                val_max_samples,
                sidecar_path=evidence_sidecar_path,
                cache_path=cache_path,
                use_cache=use_cache,
            ),
        },
        "artifact_preflight": artifact_preflight,
        "code_sha256": _code_hashes(project_dir, reward_arm),
        "package_versions": _package_versions(),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Run manifest already exists: {manifest_path}")
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run_mode", choices=sorted(_RUN_MODES), required=True)
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--validation_path", type=Path, required=True)
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--evidence_sidecar_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_max_samples", type=int, required=True)
    parser.add_argument("--val_max_samples", type=int, required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--rollout_n", type=int, required=True)
    parser.add_argument("--total_training_steps", type=int, required=True)
    parser.add_argument("--grpo_micro_batch_size", type=int, default=1)
    parser.add_argument("--data_shuffle", type=_parse_bool, default=False)
    parser.add_argument("--actor_use_dynamic_bsz", type=_parse_bool, default=True)
    parser.add_argument("--actor_max_token_len_per_gpu", type=int, default=8_192)
    parser.add_argument("--entropy_chunk_rows", type=int, default=512)
    parser.add_argument("--calculate_entropy", type=_parse_bool, default=False)
    parser.add_argument("--reference_kl_enabled", type=_parse_bool, default=True)
    parser.add_argument("--reference_kl_loss_coef", type=float, default=0.001)
    parser.add_argument("--reference_kl_loss_type", choices=("low_var_kl",), default="low_var_kl")
    parser.add_argument("--kl_in_reward", type=_parse_bool, default=False)
    parser.add_argument("--use_fused_kernels", type=_parse_bool, default=True)
    parser.add_argument("--fused_kernel_backend", choices=("triton", "torch"), default="triton")
    parser.add_argument("--actor_param_offload", type=_parse_bool, default=False)
    parser.add_argument("--actor_optimizer_offload", type=_parse_bool, default=False)
    parser.add_argument("--enable_gradient_checkpointing", type=_parse_bool, default=True)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument("--vllm_enable_sleep_mode", type=_parse_bool, default=False)
    parser.add_argument("--vllm_free_cache_engine", type=_parse_bool, default=False)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--max_actor_ckpt_to_keep", type=int, default=2)
    parser.add_argument("--resume_mode", choices=("disable", "auto", "resume_path"), default="disable")
    parser.add_argument("--resume_from_path")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--agent_workers", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    args = parser.parse_args()
    kwargs = vars(args)
    if kwargs["resume_from_path"] in {None, "", "null"}:
        kwargs["resume_from_path"] = None
    for key in (
        "project_dir",
        "train_path",
        "validation_path",
        "corpus_dir",
        "evidence_sidecar_path",
        "model_path",
        "output_dir",
    ):
        kwargs[key] = kwargs[key].expanduser().resolve()
    manifest = prepare_formal_rlvr_run(**kwargs)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
