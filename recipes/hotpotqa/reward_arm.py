"""Frozen reward-arm semantics for matched HotpotQA RLVR experiments."""

from __future__ import annotations

from enum import Enum
from typing import Any


class RewardArm(str, Enum):
    """Reward arms supported by the shared HotpotQA AgentFlow."""

    A0 = "A0"
    A1 = "A1"
    A2 = "A2"


def parse_reward_arm(value: Any) -> RewardArm:
    normalized = str(value).strip().upper()
    try:
        return RewardArm(normalized)
    except ValueError as exc:
        supported = ", ".join(arm.value for arm in RewardArm)
        raise ValueError(f"Unsupported HotpotQA reward arm {value!r}; expected one of {supported}") from exc


def resolve_reward_arm(value: Any | None, *, formal_a0: bool) -> RewardArm:
    """Resolve the explicit arm while preserving the formal-A0 entrypoint."""

    if formal_a0:
        if value is not None and parse_reward_arm(value) is not RewardArm.A0:
            raise ValueError("HOTPOTQA_FORMAL_A0=1 is only compatible with HOTPOTQA_REWARD_ARM=A0")
        return RewardArm.A0
    return parse_reward_arm(RewardArm.A1.value if value is None else value)


def search_step_reward(
    arm: RewardArm,
    process_reward: float | None,
    *,
    is_validation: bool,
) -> float:
    """Return the optimizer-visible reward for one generated search action."""

    if is_validation or arm is not RewardArm.A2:
        return 0.0
    return float(process_reward) if process_reward is not None else 0.0


def final_step_reward(arm: RewardArm, *, is_validation: bool) -> float | None:
    """Return ``None`` when terminal EM must be computed, otherwise zero."""

    if is_validation or arm in {RewardArm.A0, RewardArm.A1}:
        return None
    return 0.0


def final_step_response_mask(
    arm: RewardArm,
    response_length: int,
    *,
    is_validation: bool,
) -> list[int] | None:
    """Mask A2 training final-answer tokens out of process-policy loss."""

    if arm is RewardArm.A2 and not is_validation:
        return [0] * int(response_length)
    return None
