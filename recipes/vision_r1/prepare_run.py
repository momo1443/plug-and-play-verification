"""Record the actual Vision-R1 paper profile before starting the trainer."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import pyarrow.parquet as pq


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-dir", "output-dir", "model-path", "train-path", "validation-path", "arm"):
        parser.add_argument("--" + name, required=True)
    for name in ("train-max-samples", "train-batch-size", "total-training-steps", "rollout-n",
                 "max-prompt-length", "max-response-length", "terminal-warmup-steps", "seed"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args()
    rows = pq.ParquetFile(args.train_path).metadata.num_rows
    if not 0 < args.train_max_samples <= rows:
        raise ValueError("Training prefix must be positive and available in the prepared split")
    root = Path(args.project_dir)
    code_paths = ["recipes/vision_r1/agent_flow.py", "recipes/vision_r1/verifier.py", "recipes/vision_r1/base.yaml",
                  "recipes/vision_r1/prepare_data.py", "recipes/vision_r1/prepare_run.py",
                  "agent_r1/verifier/reward.py", "agent_r1/evaluation/consistency.py",
                  "agent_r1/trainer/ppo/core_algos.py", "examples/common/model_training.sh",
                  "examples/vision_r1/run_visual_agent.sh"]
    manifest = {
        "contract_version": "vision-r1-paper-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm, "model": file_identity(Path(args.model_path) / "config.json"),
        "selection": {"train": file_identity(args.train_path), "validation": file_identity(args.validation_path),
                      "source_rows": rows, "count": args.train_max_samples, "selection": "source_prefix"},
        "training": {key: getattr(args, key) for key in ("train_batch_size", "total_training_steps", "rollout_n",
                     "max_prompt_length", "max_response_length", "terminal_warmup_steps", "seed")},
        "code_sha256": {path: file_identity(root / path)["sha256"] for path in code_paths},
        "evaluation": {"training_validation": "Vision-R1 test split", "paper_benchmark": "MATH-Vision; separate evaluation required"},
    }
    manifest["training"].update({"algorithm": os.environ.get("AGENT_R1_OPTIMIZER", "grpo").upper(),
        "sampled_prompt_count": args.train_batch_size * args.total_training_steps, "max_agent_steps": 3,
        "actor_lr": 1e-6, "reference_kl_coefficient": 0.001, "terminal_gate": "valid_final_submission"})
    output = Path(args.output_dir) / "run_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
