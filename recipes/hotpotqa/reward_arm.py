"""Frozen reward-arm semantics for matched HotpotQA RLVR experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RewardArm(str, Enum):
    """Reward arms supported by the shared HotpotQA AgentFlow."""

    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A6 = "A6"
    A7 = "A7"
    A9 = "A9"
    A9_CERT_MIX = "A9_CERT_MIX"


@dataclass(frozen=True)
class TrainingRewardContract:
    """Optimizer-visible reward weights and final-action loss mask for one arm."""

    process_weight: float
    terminal_weight: float
    final_response_mask: int


_TRAINING_REWARD_CONTRACTS = {
    RewardArm.A0: TrainingRewardContract(0.0, 0.0, 1),
    RewardArm.A1: TrainingRewardContract(0.0, 1.0, 1),
    RewardArm.A2: TrainingRewardContract(1.0, 0.0, 0),
    RewardArm.A3: TrainingRewardContract(0.5, 0.5, 1),
    RewardArm.A6: TrainingRewardContract(0.5, 0.5, 1),
    RewardArm.A7: TrainingRewardContract(0.5, 0.5, 1),
    RewardArm.A9: TrainingRewardContract(0.2, 0.8, 1),  # expected values; actual weights sampled U(0,1) per trajectory in hotpotqa_a9/
    RewardArm.A9_CERT_MIX: TrainingRewardContract(0.2, 0.8, 1),  # expected values; actual weights sampled U(0,1) per trajectory in hotpotqa_a9/
}

def training_reward_contract(arm: RewardArm) -> TrainingRewardContract:
    """Return the single frozen source of truth for train-time arm semantics."""

    return _TRAINING_REWARD_CONTRACTS[arm]


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

    if is_validation:
        return 0.0
    reward = float(process_reward) if process_reward is not None else 0.0
    return training_reward_contract(arm).process_weight * reward


def final_step_reward(arm: RewardArm, *, is_validation: bool) -> float | None:
    """Return ``None`` when terminal EM must be computed, otherwise zero."""

    if is_validation or training_reward_contract(arm).terminal_weight > 0.0:
        return None
    return 0.0


def scale_terminal_reward(
    arm: RewardArm,
    terminal_reward: float,
    *,
    is_validation: bool,
) -> float:
    """Scale a computed terminal EM for training without changing validation EM."""

    if is_validation:
        return float(terminal_reward)
    return training_reward_contract(arm).terminal_weight * float(terminal_reward)


def final_step_response_mask(
    arm: RewardArm,
    response_length: int,
    *,
    is_validation: bool,
) -> list[int] | None:
    """Apply the frozen final-token loss mask; validation always scores all tokens."""

    if not is_validation and training_reward_contract(arm).final_response_mask == 0:
        return [0] * int(response_length)
    return None
