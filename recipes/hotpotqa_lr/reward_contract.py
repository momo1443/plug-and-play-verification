"""Frozen optimizer reward contracts for corrected A8-LR experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass

from recipes.hotpotqa_lr.dsl import REASON_STEP_FORMAT, REASON_STEP_FORMAT_CLAIM_SOURCE

REWARD_MODE_ENV = "HOTPOTQA_LR_REWARD_MODE"
REWARD_MODE_LR30 = "lr30"
REWARD_MODE_TERMINAL_ONLY = "terminal_only"
SUPPORTED_REWARD_MODES = frozenset({REWARD_MODE_LR30, REWARD_MODE_TERMINAL_ONLY})


def resolve_reward_mode(value: str | None = None) -> str:
    mode = value if value is not None else os.environ.get(REWARD_MODE_ENV, REWARD_MODE_LR30)
    mode = str(mode).strip().lower()
    if mode not in SUPPORTED_REWARD_MODES:
        supported = ", ".join(sorted(SUPPORTED_REWARD_MODES))
        raise ValueError(f"{REWARD_MODE_ENV} must be one of {supported}, got {mode!r}")
    return mode


def expected_reward_arm(reason_step_format: str, reward_mode: str) -> str:
    suffix = "_CS" if reason_step_format == REASON_STEP_FORMAT_CLAIM_SOURCE else ""
    return f"A8_LR_BASE{suffix}" if reward_mode == REWARD_MODE_TERMINAL_ONLY else f"A8_LR30{suffix}"


@dataclass(frozen=True)
class LocalReasoningRewardContract:
    terminal_weight: float = 0.7
    process_weight: float = 0.3
    reward_horizon: int = 3
    dependency_taint_gamma: float = 0.3
    final_response_mask: int = 1


PRIMARY_CONTRACT = LocalReasoningRewardContract()
BASE_CONTRACT = LocalReasoningRewardContract(terminal_weight=1.0, process_weight=0.0)
LR_REWARD_MODE = resolve_reward_mode()
LR_REWARD_ARM = expected_reward_arm(REASON_STEP_FORMAT, LR_REWARD_MODE)
LR_CONTRACT_VERSION = (
    "a8-lr-claim-source-v2" if REASON_STEP_FORMAT == REASON_STEP_FORMAT_CLAIM_SOURCE else "a8-lr-dsl-v2"
)

if PRIMARY_CONTRACT.terminal_weight + PRIMARY_CONTRACT.process_weight != 1.0:
    raise RuntimeError("A8-LR reward weights must sum to one")
if PRIMARY_CONTRACT.reward_horizon != 3:
    raise RuntimeError("The frozen A8-LR reward horizon must remain H=3")
if not 0.0 <= PRIMARY_CONTRACT.dependency_taint_gamma <= 1.0:
    raise RuntimeError("dependency_taint_gamma must be in [0, 1]")


def contract_for_mode(reward_mode: str) -> LocalReasoningRewardContract:
    return BASE_CONTRACT if resolve_reward_mode(reward_mode) == REWARD_MODE_TERMINAL_ONLY else PRIMARY_CONTRACT


def compose_reward(terminal_em: float, local_reward: float) -> float:
    """Compose independent terminal and process signals without EM gating."""

    if not 0.0 <= float(terminal_em) <= 1.0:
        raise ValueError("terminal_em must be in [0, 1]")
    if not 0.0 <= float(local_reward) <= 1.0:
        raise ValueError("local_reward must be in [0, 1]")
    return PRIMARY_CONTRACT.terminal_weight * float(terminal_em) + PRIMARY_CONTRACT.process_weight * float(local_reward)
