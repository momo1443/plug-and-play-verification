#!/usr/bin/env python3
"""Fail-closed CPU preflight and manifest builder for HotpotQA A1/A2 GRPO."""

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

from recipes.hotpotqa.evidence import EVIDENCE_SCHEMA_VERSION, OfficialEvidenceStore
from recipes.hotpotqa.process_verifier import PROCESS_VERIFIER_VERSION
from recipes.hotpotqa.reward_arm import RewardArm, parse_reward_arm
from recipes.hotpotqa.validate_formal_a0_artifacts import validate_formal_a0_artifacts

CONTRACT_VERSION = "hotpotqa-formal-rlvr-grpo-v1"
FORMAL_TRAIN_ROWS = 90_447
FORMAL_VALIDATION_ROWS = 7_405
_RUN_MODES = {"smoke", "main"}


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


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "transformers", "vllm", "verl", "ray"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "missing"
    return result


def _code_hashes(project_dir: Path) -> dict[str, str]:
    paths = (
        "agent_r1/agent_flow/agent_flow.py",
        "agent_r1/trainer/ppo/core_algos.py",
        "agent_r1/trainer/ppo/ray_trainer.py",
        "recipes/hotpotqa/base.yaml",
        "recipes/hotpotqa/hotpotqa_agent_flow.py",
        "recipes/hotpotqa/process_verifier.py",
        "recipes/hotpotqa/reward_arm.py",
        "recipes/hotpotqa/reward_fn.py",
        "examples/hotpotqa/run_rlvr.sh",
    )
    hashes: dict[str, str] = {}
    for relative_path in paths:
        path = project_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required RLVR code file is missing: {path}")
        hashes[relative_path] = _sha256(path)
    return hashes


def _model_identity(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"Expected Qwen3.5 model_type, got {config.get('model_type')!r}")
    identity_files = [config_path]
    for filename in ("model.safetensors.index.json", "tokenizer_config.json"):
        candidate = model_path / filename
        if candidate.is_file():
            identity_files.append(candidate)
    return {
        "path": str(model_path),
        "model_type": config["model_type"],
        "architectures": config.get("architectures", []),
        "identity_sha256": {
            path.name: _sha256(path) for path in identity_files
        },
    }


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
) -> dict[str, Any]:
    reward_arm = parse_reward_arm(arm)
    if reward_arm not in {RewardArm.A1, RewardArm.A2}:
        raise ValueError("The RLVR training entrypoint only accepts A1 or A2")
    if run_mode not in _RUN_MODES:
        raise ValueError(f"run_mode must be one of {sorted(_RUN_MODES)}, got {run_mode!r}")
    if run_mode == "main" and num_gpus not in {4, 8}:
        raise ValueError(f"main runs require 4 or 8 GPUs, got {num_gpus}")
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
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    for path in (train_path, validation_path, evidence_sidecar_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required RLVR artifact is missing: {path}")
    train_rows = int(pq.read_metadata(train_path).num_rows)
    val_rows = int(pq.read_metadata(validation_path).num_rows)
    if train_rows != FORMAL_TRAIN_ROWS or val_rows != FORMAL_VALIDATION_ROWS:
        raise ValueError(
            "Formal A1/A2 source data must contain exactly 90,447 train and 7,405 validation rows; "
            f"got train={train_rows}, validation={val_rows}"
        )
    if train_max_samples > train_rows or val_max_samples > val_rows:
        raise ValueError("Requested sample count exceeds the available parquet rows")
    if val_max_samples != FORMAL_VALIDATION_ROWS:
        raise ValueError("Formal A1/A2 always retains the complete 7,405-row validation split")
    if run_mode == "main" and train_max_samples != FORMAL_TRAIN_ROWS:
        raise ValueError("A main run must use the complete 90,447-row training split")
    if run_mode == "smoke" and (total_training_steps != 1 or train_max_samples > 64):
        raise ValueError("A smoke run is limited to one update and at most 64 training questions")

    store = OfficialEvidenceStore(evidence_sidecar_path)
    if store.sample_count("train") != train_rows or store.sample_count("validation") != val_rows:
        raise ValueError("Evidence sidecar sample counts do not match the train/validation parquet")
    for split in ("train", "validation"):
        total = int(store.metadata[f"gold_fact_count_{split}"])
        unresolved = int(store.metadata[f"unresolved_gold_fact_count_{split}"])
        mapping_rate = (total - unresolved) / total
        if mapping_rate < 0.99:
            raise ValueError(f"{split} gold mapping rate {mapping_rate:.6f} is below 0.99")

    artifact_preflight = validate_formal_a0_artifacts(
        validation_path=validation_path,
        corpus_dir=corpus_dir,
        evidence_sidecar_path=evidence_sidecar_path,
    )
    cuda_devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if cuda_devices and len(cuda_devices) != num_gpus:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES exposes {len(cuda_devices)} devices, expected {num_gpus}: {cuda_devices}"
        )

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": reward_arm.value,
        "run_mode": run_mode,
        "output_dir": str(output_dir),
        "scientific_config": {
            "algorithm": "GRPO",
            "grpo_credit_assignment": "step_causal",
            "gamma": gamma,
            "reward": "terminal exact match" if reward_arm is RewardArm.A1 else PROCESS_VERIFIER_VERSION,
            "a2_final_reward": 0.0 if reward_arm is RewardArm.A2 else None,
            "a2_final_response_mask": 0 if reward_arm is RewardArm.A2 else None,
            "validation_reward": "terminal exact match",
            "max_steps": 4,
            "max_model_searches": 3,
            "force_first_search": False,
            "thinking_mode": "disabled",
            "tool_parser": "hermes",
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        },
        "training": {
            "source_train_rows": train_rows,
            "source_validation_rows": val_rows,
            "train_max_samples": train_max_samples,
            "val_max_samples": val_max_samples,
            "train_batch_size": train_batch_size,
            "rollout_n": rollout_n,
            "total_training_steps": total_training_steps,
        },
        "resources": {
            "num_gpus": num_gpus,
            "agent_workers": agent_workers,
            "cuda_visible_devices": cuda_devices,
        },
        "model": _model_identity(model_path),
        "selection": {
            "train": _selection_digest(store, "train", train_max_samples),
            "validation": _selection_digest(store, "validation", val_max_samples),
        },
        "artifact_preflight": artifact_preflight,
        "code_sha256": _code_hashes(project_dir),
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
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--agent_workers", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    args = parser.parse_args()
    kwargs = vars(args)
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
