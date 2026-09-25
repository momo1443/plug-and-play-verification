"""Reward schedule and deterministic outcome scoring for Vision-R1."""

from __future__ import annotations

from typing import Any

from agent_r1.verifier import UniformRewardSchedule, uniform_reward_schedule
from verl.utils.reward_score.math_reward import is_equiv

_GROUP_WEIGHT_NAMESPACE = b"vision-r1-visual-certificate-group-weight-v1"


def outcome_reward(answer: str | None, ground_truth: Any) -> float:
    if not isinstance(answer, str) or not answer.strip() or ground_truth is None:
        return 0.0
    prediction = answer.strip().strip("$ ")
    target = str(ground_truth).strip().strip("$ ")
    if prediction.casefold() == target.casefold():
        return 1.0
    return float(is_equiv(prediction, target))


def reward_schedule(
    *,
    reward_mode: str,
    global_step: int,
    is_validation: bool,
    terminal_warmup_steps: int,
    prompt_group_key: str,
) -> UniformRewardSchedule:
    if reward_mode == "terminal_only":
        return UniformRewardSchedule(1.0, 0.0, "terminal_only", "fixed", None)
    if reward_mode not in {"uniform_visual_certificate", "llm_judge"}:
        raise ValueError(f"Unsupported Vision-R1 reward mode: {reward_mode}")
    return uniform_reward_schedule(
        global_step=global_step,
        is_validation=is_validation,
        warmup_steps=terminal_warmup_steps,
        prompt_group_key=prompt_group_key,
        namespace=_GROUP_WEIGHT_NAMESPACE,
        warmup_phase="terminal_warmup",
        validation_phase="validation_terminal_only",
        mixed_phase="llm_process_uniform" if reward_mode == "llm_judge" else "visual_certificate_uniform",
    )
