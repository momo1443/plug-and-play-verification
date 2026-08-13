"""Frozen optimizer reward contract for the primary A8-LR experiment."""

from __future__ import annotations

from dataclasses import dataclass

LR_CONTRACT_VERSION = "a8-lr-50-v1"


@dataclass(frozen=True)
class LocalReasoningRewardContract:
    terminal_weight: float = 0.5
    process_weight: float = 0.5
    reward_horizon: int = 3
    dependency_taint_gamma: float = 0.3
    final_response_mask: int = 1


PRIMARY_CONTRACT = LocalReasoningRewardContract()

if PRIMARY_CONTRACT.terminal_weight + PRIMARY_CONTRACT.process_weight != 1.0:
    raise RuntimeError("A8-LR reward weights must sum to one")
if PRIMARY_CONTRACT.reward_horizon != 3:
    raise RuntimeError("The frozen A8-LR reward horizon must remain H=3")
if not 0.0 <= PRIMARY_CONTRACT.dependency_taint_gamma <= 1.0:
    raise RuntimeError("dependency_taint_gamma must be in [0, 1]")


def compose_reward(terminal_em: float, local_reward: float) -> float:
    """Compose independent terminal and process signals without EM gating."""

    if not 0.0 <= float(terminal_em) <= 1.0:
        raise ValueError("terminal_em must be in [0, 1]")
    if not 0.0 <= float(local_reward) <= 1.0:
        raise ValueError("local_reward must be in [0, 1]")
    return (
        PRIMARY_CONTRACT.terminal_weight * float(terminal_em)
        + PRIMARY_CONTRACT.process_weight * float(local_reward)
    )
