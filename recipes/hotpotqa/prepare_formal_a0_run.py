#!/usr/bin/env python3
"""Create the immutable artifact lock and per-run manifest for formal A0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recipes.hotpotqa.evidence import EVIDENCE_SCHEMA_VERSION, OfficialEvidenceStore
from recipes.hotpotqa.validate_formal_a0_artifacts import validate_formal_a0_artifacts

CONTRACT_VERSION = "hotpotqa-formal-a0-shared-agentflow-v1"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": (
            path.relative_to(relative_to).as_posix() if relative_to is not None else str(path)
        ),
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": _sha256(path),
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_file():
        return {"kind": "file", **_file_record(path)}
    if not path.is_dir():
        raise FileNotFoundError(f"Manifest artifact is missing: {path}")
    files = sorted(file_path for file_path in path.rglob("*") if file_path.is_file())
    if not files:
        raise ValueError(f"Manifest artifact directory is empty: {path}")
    records = [_file_record(file_path, relative_to=path) for file_path in files]
    tree = hashlib.sha256()
    for record in records:
        tree.update(record["path"].encode("utf-8"))
        tree.update(b"\0")
        tree.update(record["sha256"].encode("ascii"))
        tree.update(b"\n")
    return {
        "kind": "directory",
        "path": str(path),
        "sha256_tree": tree.hexdigest(),
        "bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }


def _validate_cached_artifact(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if record["kind"] == "file":
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Cached artifact lock no longer matches {path}")
        if path.stat().st_mtime_ns != int(record["mtime_ns"]):
            raise ValueError(f"Cached artifact mtime changed for {path}; rebuild artifact lock")
        return
    if not path.is_dir():
        raise ValueError(f"Cached artifact directory is missing: {path}")
    for file_record in record["files"]:
        file_path = path / file_record["path"]
        if (
            not file_path.is_file()
            or file_path.stat().st_size != int(file_record["bytes"])
            or file_path.stat().st_mtime_ns != int(file_record["mtime_ns"])
        ):
            raise ValueError(f"Cached artifact lock no longer matches {file_path}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _code_files(project_dir: Path) -> list[Path]:
    relative_paths = [
        "recipes/hotpotqa/base.yaml",
        "recipes/hotpotqa/evidence.py",
        "recipes/hotpotqa/hotpotqa_agent_flow.py",
        "recipes/hotpotqa/env/search_tool.py",
        "recipes/hotpotqa/prompts.py",
        "recipes/hotpotqa/reward_fn.py",
        "recipes/hotpotqa/validate_formal_a0_artifacts.py",
        "recipes/hotpotqa/validate_formal_a0_output.py",
        "agent_r1/trainer/compact_validation_record.py",
        "agent_r1/trainer/streaming_agent_validation.py",
        "agent_r1/trainer/streaming_jsonl.py",
        "examples/hotpotqa/run_a0.sh",
        "examples/hotpotqa/run_validation.sh",
    ]
    return [project_dir / relative_path for relative_path in relative_paths]


def prepare_formal_a0_run(
    *,
    project_dir: Path,
    validation_path: Path,
    corpus_dir: Path,
    evidence_sidecar_path: Path,
    model_path: Path,
    embedding_model_path: Path,
    output_dir: Path,
    artifact_lock_path: Path,
    val_max_samples: int,
    num_gpus: int,
    agent_workers: int,
) -> dict[str, Any]:
    preflight = validate_formal_a0_artifacts(
        validation_path=validation_path,
        corpus_dir=corpus_dir,
        evidence_sidecar_path=evidence_sidecar_path,
    )
    if num_gpus != 8:
        raise ValueError(f"Formal full A0 is frozen to 8 GPUs, got {num_gpus}")
    if agent_workers < 1:
        raise ValueError("Agent worker count must be positive")

    artifact_paths = {
        "validation_parquet": validation_path,
        "corpus_jsonl": corpus_dir / "hpqa_corpus.jsonl",
        "corpus_embeddings": corpus_dir / "hpqa_corpus.npy",
        "faiss_index": corpus_dir / "index.bin",
        "evidence_sidecar": evidence_sidecar_path,
        "evidence_sidecar_manifest": evidence_sidecar_path.with_suffix(".manifest.json"),
        "policy_model": model_path,
        "retrieval_model": embedding_model_path,
    }
    if artifact_lock_path.exists():
        artifact_lock = json.loads(artifact_lock_path.read_text(encoding="utf-8"))
        if artifact_lock.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("Existing artifact lock has the wrong contract version")
        for name, record in artifact_lock["artifacts"].items():
            if name not in artifact_paths or Path(record["path"]).resolve() != artifact_paths[name].resolve():
                raise ValueError(f"Existing artifact lock path mismatch for {name}")
            _validate_cached_artifact(record)
        for record in artifact_lock["code"].values():
            _validate_cached_artifact(record)
    else:
        artifacts = {
            name: _artifact_record(path) for name, path in artifact_paths.items()
        }
        code = {
            path.relative_to(project_dir).as_posix(): _artifact_record(path)
            for path in _code_files(project_dir)
        }
        artifact_lock = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts,
            "code": code,
        }
        artifact_lock["lock_sha256"] = hashlib.sha256(
            json.dumps(artifact_lock, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _write_json_atomic(artifact_lock_path, artifact_lock)

    store = OfficialEvidenceStore(evidence_sidecar_path)
    total_rows = store.sample_count("validation")
    expected_rows = total_rows if val_max_samples <= 0 else min(val_max_samples, total_rows)
    selected_samples = store.list_samples("validation", limit=expected_rows)
    scientific_config = {
        "arm": "A0",
        "framework": "agent_r1.agent_flow.HotpotQAAgentFlow",
        "validation_only": True,
        "optimizer_updates": 0,
        "process_reward_used_for_optimization": False,
        "terminal_reward": "normalized exact match (evaluation only)",
        "thinking_mode": "disabled",
        "verifier_reads_thinking": False,
        "force_first_search": False,
        "max_steps": 4,
        "max_model_searches": 3,
        "max_parallel_calls": 1,
        "retrieval_top_k": 5,
        "tool_parser": "hermes",
        "max_prompt_tokens_per_step": 8192,
        "max_response_tokens_per_step": 1024,
        "max_generated_tokens_per_trajectory": 4096,
        "engine_max_model_len": 12288,
        "validation_n": 1,
        "validation_do_sample": False,
        "validation_temperature": 0.0,
        "validation_top_p": 1.0,
        "validation_top_k": -1,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
    }
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "output_jsonl": str((output_dir / "0.jsonl").resolve()),
        "artifact_lock_path": str(artifact_lock_path.resolve()),
        "artifact_lock_sha256": artifact_lock["lock_sha256"],
        "scientific_config": scientific_config,
        "resources": {
            "num_gpus": num_gpus,
            "agent_workers": agent_workers,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "selection": {
            "split": "validation",
            "requested_max_samples": val_max_samples,
            "expected_rows": expected_rows,
            "samples": [
                {
                    "sample_key": sample.sample_key,
                    "official_qid": sample.official_qid,
                    "evidence_metrics_eligible": not sample.unresolved_gold_facts,
                }
                for sample in selected_samples
            ],
        },
        "preflight": preflight,
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Run manifest already exists: {manifest_path}")
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--validation_path", type=Path, required=True)
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--evidence_sidecar_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--embedding_model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--artifact_lock_path", type=Path, required=True)
    parser.add_argument("--val_max_samples", type=int, required=True)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--agent_workers", type=int, required=True)
    args = parser.parse_args()
    manifest = prepare_formal_a0_run(
        project_dir=args.project_dir.expanduser().resolve(),
        validation_path=args.validation_path.expanduser().resolve(),
        corpus_dir=args.corpus_dir.expanduser().resolve(),
        evidence_sidecar_path=args.evidence_sidecar_path.expanduser().resolve(),
        model_path=args.model_path.expanduser().resolve(),
        embedding_model_path=args.embedding_model_path.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        artifact_lock_path=args.artifact_lock_path.expanduser().resolve(),
        val_max_samples=args.val_max_samples,
        num_gpus=args.num_gpus,
        agent_workers=args.agent_workers,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
