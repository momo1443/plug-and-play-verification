"""Frozen A9-Code reward semantics for the 300-step pilot."""

from __future__ import annotations

from agent_r1.verifier import UniformRewardSchedule, uniform_reward_schedule


_GROUP_WEIGHT_NAMESPACE = b"taco-a9-group-weight-v1"


CONTRACT_VERSION = "taco-a9-terminal50-verified-process-group-uniform-v3"


def sample_weights(
    *,
    global_step: int,
    is_validation: bool,
    terminal_warmup_steps: int,
    prompt_group_key: str,
) -> tuple[float, float, str]:
    """Compatibility tuple for the shared verifier reward schedule."""
    schedule = reward_schedule(
        global_step=global_step,
        is_validation=is_validation,
        terminal_warmup_steps=terminal_warmup_steps,
        prompt_group_key=prompt_group_key,
    )
    return schedule.terminal_weight, schedule.process_weight, schedule.phase


def reward_schedule(
    *,
    global_step: int,
    is_validation: bool,
    terminal_warmup_steps: int,
    prompt_group_key: str,
) -> UniformRewardSchedule:
    return uniform_reward_schedule(
        global_step=global_step,
        is_validation=is_validation,
        warmup_steps=terminal_warmup_steps,
        prompt_group_key=prompt_group_key,
        namespace=_GROUP_WEIGHT_NAMESPACE,
        warmup_phase="terminal_private_warmup",
        validation_phase="validation_terminal_private_tests",
        mixed_phase="certificate_uniform",
    )
