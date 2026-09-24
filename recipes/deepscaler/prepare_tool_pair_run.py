#!/usr/bin/env python3
"""Write a frozen manifest for one paired DeepScaleR ToolEnv A1/A9 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reward_contract(arm: str, warmup_steps: int) -> dict[str, Any]:
    if arm == "A1":
        return {
            "formula": "terminal_EM(final_answer)",
            "optimizer_placement": "final_agent_step_only",
            "process_reward": "disabled",
        }
    if arm == "JUDGE":
        return {
            "training_formula": "Qwen3.5-9B frozen LLM judge score",
            "validation_formula": "terminal_EM(final_answer)",
            "score_support": [0.0, 0.25, 0.5, 0.75, 1.0],
            "optimizer_placement": "final_agent_step_only",
            "failure_policy": "abort_optimizer_update",
        }
    return {
        "warmup_steps": warmup_steps,
        "warmup_formula": "terminal_EM(final_answer)",
        "post_warmup_formula": "w * terminal_EM(final_answer) + (1 - w) * equation_process(trajectory)",
        "weight_sampling": "one w ~ Uniform(0, 1) per prompt group",
        "optimizer_placement": "process_credit_source_steps_terminal_credit_final_step",
        "validation_formula": "terminal_EM(final_answer)",
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model_path).resolve()
    train_path = Path(args.train_path).resolve()
    validation_path = Path(args.validation_path).resolve()
    required_paths = [model_path / "config.json", train_path, validation_path]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Paired ToolEnv run is missing required artifacts: {missing}")
    if args.arm not in {"A1", "A9", "JUDGE"}:
        raise ValueError("arm must be A1, A9, or JUDGE")
    if args.train_max_samples != args.train_batch_size * args.total_training_steps:
        raise ValueError("train_max_samples must equal train_batch_size * total_training_steps")
    if args.rollout_n != 4 or args.seed != 42 or args.max_steps != 5:
        raise ValueError("Paired ToolEnv contract requires rollout_n=4, seed=42, and max_steps=5")
    for path in (train_path, validation_path):
        if "agent_name" in pq.ParquetFile(path).schema.names:
            raise ValueError(f"Paired ToolEnv parquet must omit agent_name: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite paired run output: {output_dir}")

    code_paths = [
        "agent_r1/env/tool_format.py",
        "agent_r1/verifier/reward.py",
        "recipes/deepscaler/prompts.py",
        "recipes/deepscaler/tool_agent_flow.py",
        "recipes/deepscaler/trajectory_reward.py",
        "recipes/llm_judge/scoring.py",
        "recipes/llm_judge/deepscaler.yaml",
        "recipes/hotpotqa/judge_server.py",
        "recipes/reward_mixing.py",
        "recipes/deepscaler/tool_reward_arms.yaml",
        "recipes/deepscaler/tool.py",
        "recipes/deepscaler/prepare_tool_data.py",
        "recipes/deepscaler/prepare_tool_pair_run.py",
        "examples/deepmath/run_deepscaler_tool_pair.sh",
        "examples/llm_judge/run_with_frozen_judge.sh",
        "examples/llm_judge/run_grpo_4b_judge_9b.sh",
    ]
    missing_code = [relative for relative in code_paths if not (project_dir / relative).is_file()]
    if missing_code:
        raise FileNotFoundError(f"Paired ToolEnv source missing: {missing_code}")
    manifest = {
        "contract_version": "deepscaler-tool-paired-a1-a9-group-uniform-v2",
        "arm": args.arm,
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "reward_contract": _reward_contract(args.arm, args.em_warmup_steps),
        "shared_runtime": {
            "model_path": str(model_path),
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "train_batch_size": args.train_batch_size,
            "train_max_samples": args.train_max_samples,
            "total_training_steps": args.total_training_steps,
            "rollout_n": args.rollout_n,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "cuda_visible_devices": args.cuda_visible_devices,
            "num_gpus": args.num_gpus,
        },
        "data_sha256": {
            "train": _sha256(train_path),
            "validation": _sha256(validation_path),
        },
        "code_sha256": {relative: _sha256(project_dir / relative) for relative in code_paths},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--train-max-samples", type=int, required=True)
    parser.add_argument("--total-training-steps", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--em-warmup-steps", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    manifest = build_manifest(parser.parse_args())
    print(f"Wrote paired {manifest['arm']} manifest to {manifest['output_dir']}")


if __name__ == "__main__":
    main()
