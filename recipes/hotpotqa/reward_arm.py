"""Legacy raw-completion rewards; paper arms use hotpotqa_a9.reward_sources."""

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
    # Expected weights; hotpotqa_a9 shares each realized U(0,1) w per prompt group.
    RewardArm.A9: TrainingRewardContract(0.5, 0.5, 1),
    RewardArm.A9_CERT_MIX: TrainingRewardContract(0.5, 0.5, 1),
}

def training_reward_contract(arm: RewardArm) -> TrainingRewardContract:
    """Return fixed legacy weights, not the current paper arm schedule."""

    return _TRAINING_REWARD_CONTRACTS[arm]


def parse_reward_arm(value: Any) -> RewardArm:
    normalized = str(value).strip().upper()
    try:
        return RewardArm(normalized)
    except ValueError as exc:
        supported = ", ".join(arm.value for arm in RewardArm)
        raise ValueError(f"Unsupported HotpotQA reward arm {value!r}; expected one of {supported}") from exc


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


