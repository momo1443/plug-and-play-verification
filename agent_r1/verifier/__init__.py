"""Domain-agnostic interfaces for deterministic process verifiers."""

from .aggregation import VerificationGroup, normalize_verification_groups
from .reward import (
    ComposedVerificationReward,
    UniformRewardSchedule,
    VerificationCredit,
    VerificationResult,
    apply_composed_reward,
    compose_verification_reward,
    prompt_group_uniform_weight,
    uniform_reward_schedule,
)

__all__ = [
    "ComposedVerificationReward",
    "UniformRewardSchedule",
    "VerificationCredit",
    "VerificationGroup",
    "VerificationResult",
    "apply_composed_reward",
    "compose_verification_reward",
    "normalize_verification_groups",
    "prompt_group_uniform_weight",
    "uniform_reward_schedule",
]
