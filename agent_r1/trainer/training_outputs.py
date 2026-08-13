"""Shared output policy for Agent-R1 training runs."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, open_dict


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _training_run_dir(default_local_dir: str) -> Path:
    """Infer a run root while keeping the standard checkpoint layout intact."""

    checkpoint_dir = _absolute_path(default_local_dir)
    if checkpoint_dir.name == "checkpoints":
        return checkpoint_dir.parent
    return checkpoint_dir


def configure_training_outputs(config: DictConfig) -> None:
    """Enable TensorBoard and one rollout JSONL file for every training run."""

    trainer = config.trainer
    if bool(trainer.get("val_only", False)):
        return

    configured_backends = trainer.get("logger", ["console"])
    if isinstance(configured_backends, str):
        backends = [configured_backends]
    else:
        backends = list(configured_backends)
    if "tensorboard" not in backends:
        backends.append("tensorboard")

    run_dir = _training_run_dir(str(trainer.default_local_dir))
    tensorboard_dir = trainer.get("tensorboard_dir") or os.environ.get("TENSORBOARD_DIR")
    if tensorboard_dir:
        tensorboard_path = _absolute_path(str(tensorboard_dir))
        tensorboard_logdir = tensorboard_path
    else:
        tensorboard_logdir = run_dir / "tensorboard"
        experiment_name = str(trainer.get("experiment_name") or run_dir.name)
        safe_experiment_name = experiment_name.replace("/", "_").replace("\\", "_")
        tensorboard_path = tensorboard_logdir / safe_experiment_name

    rollout_data_file = trainer.get("rollout_data_file")
    if rollout_data_file:
        rollout_path = _absolute_path(str(rollout_data_file))
    else:
        legacy_rollout_dir = trainer.get("rollout_data_dir")
        rollout_path = (
            _absolute_path(str(legacy_rollout_dir)) / "rollouts.jsonl"
            if legacy_rollout_dir
            else run_dir / "rollouts.jsonl"
        )

    with open_dict(trainer):
        trainer.logger = backends
        trainer.tensorboard_dir = str(tensorboard_path)
        trainer.rollout_data_file = str(rollout_path)

    # verl's TensorBoard adapter reads this environment variable inside the Ray
    # task that owns the trainer. The shared RL runtime also forwards it explicitly in
    # Ray's runtime_env so remote workers see the same per-experiment location.
    os.environ["TENSORBOARD_DIR"] = str(tensorboard_path)

    print(f"Training visualization: tensorboard --logdir {tensorboard_logdir}")
    print(f"TensorBoard run name: {tensorboard_path.name}")
    print(f"Consolidated rollout output: {rollout_path}")
