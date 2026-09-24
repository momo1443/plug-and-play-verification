"""Shared schedule, composition, and placement for process-verifier rewards.

Task code owns artifact extraction and verification.  This module owns the
optimizer-facing contract: terminal warm-up, prompt-group-shared uniform
mixing, bounded credit composition, and causal placement on rollout steps.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_EPSILON = 1e-8


def prompt_group_uniform_weight(
    *,
    global_step: int,
    prompt_group_key: str,
    namespace: bytes,
) -> float:
    """Map one optimizer-step/prompt group to a reproducible U[0, 1) weight."""

    normalized_key = str(prompt_group_key).strip()
    if not normalized_key:
        raise ValueError("prompt_group_key is required for group-shared reward mixing")
    if not namespace:
        raise ValueError("namespace is required for group-shared reward mixing")
    payload = f"{int(global_step)}\0{normalized_key}".encode("utf-8")
    digest = hashlib.sha256(namespace + b"\0" + payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


@dataclass(frozen=True)
class VerificationCredit:
    """One bounded verifier score attributable to a one-indexed rollout step."""

    step_index: int
    score: float
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.step_index < 1:
            raise ValueError("Verification credits require a one-indexed positive step")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("Verification credit must be in [0, 1]")


@dataclass(frozen=True)
class VerificationResult:
    """The complete output of a task-specific verifier plugin.

    Credits must sum to at most one.  A plugin may split a trajectory-level
    score across several action steps, but cannot inflate its total reward by
    emitting more artifacts.
    """

    credits: tuple[VerificationCredit, ...]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.process_reward > 1.0 + _EPSILON:
            raise ValueError("Verification credits must sum to a bounded process reward")

    @property
    def process_reward(self) -> float:
        return sum(float(credit.score) for credit in self.credits)

    def components_by_step(self) -> dict[int, float]:
        components: dict[int, float] = {}
        for credit in self.credits:
            components[credit.step_index] = components.get(credit.step_index, 0.0) + float(credit.score)
        return components

    def audits_by_step(self) -> dict[int, list[Mapping[str, Any]]]:
        audits: dict[int, list[Mapping[str, Any]]] = {}
        for credit in self.credits:
            audits.setdefault(credit.step_index, []).append(credit.audit)
        return audits


@dataclass(frozen=True)
class UniformRewardSchedule:
    """One prompt group's terminal/process mixture for an optimizer update."""

    terminal_weight: float
    process_weight: float
    phase: str
    weight_sampling: str
    prompt_group_key: str | None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.terminal_weight) <= 1.0:
            raise ValueError("Terminal weight must be in [0, 1]")
        if not 0.0 <= float(self.process_weight) <= 1.0:
            raise ValueError("Process weight must be in [0, 1]")
        if abs(float(self.terminal_weight) + float(self.process_weight) - 1.0) > _EPSILON:
            raise ValueError("Terminal and process weights must sum to one")


def uniform_reward_schedule(
    *,
    global_step: int,
    is_validation: bool,
    warmup_steps: int,
    prompt_group_key: str,
    namespace: bytes,
    warmup_phase: str,
    validation_phase: str,
    mixed_phase: str,
) -> UniformRewardSchedule:
    """Use terminal reward during warm-up, then one uniform weight per group."""

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if is_validation:
        return UniformRewardSchedule(1.0, 0.0, validation_phase, "fixed", None)
    if global_step <= warmup_steps:
        return UniformRewardSchedule(1.0, 0.0, warmup_phase, "fixed", None)
    terminal_weight = prompt_group_uniform_weight(
        global_step=global_step,
        prompt_group_key=prompt_group_key,
        namespace=namespace,
    )
    return UniformRewardSchedule(
        terminal_weight,
        1.0 - terminal_weight,
        mixed_phase,
        "uniform_0_1_per_prompt_group",
        prompt_group_key,
    )


@dataclass(frozen=True)
class ComposedVerificationReward:
    """Optimizer-visible reward obtained from one verifier result."""

    score: float
    terminal_reward: float
    process_reward: float
    terminal_component: float
    process_step_components: Mapping[int, float]
    schedule: UniformRewardSchedule
    terminal_gate_passed: bool
    verifier_audit: Mapping[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "optimizer_total_reward": self.score,
            "terminal_reward": self.terminal_reward,
            "process_reward": self.process_reward,
            "optimizer_terminal_component": self.terminal_component,
            "optimizer_process_total": sum(self.process_step_components.values()),
            "optimizer_process_step_components": dict(self.process_step_components),
            "optimizer_terminal_weight": self.schedule.terminal_weight,
            "optimizer_process_weight": self.schedule.process_weight,
            "optimizer_reward_phase": self.schedule.phase,
            "weight_sampling": self.schedule.weight_sampling,
            "optimizer_weight_group_key": self.schedule.prompt_group_key,
            "terminal_gate_passed": self.terminal_gate_passed,
            "verification_audit": dict(self.verifier_audit),
        }


def compose_verification_reward(
    *,
    terminal_reward: float,
    verification: VerificationResult,
    schedule: UniformRewardSchedule,
    terminal_gate_passed: bool = True,
) -> ComposedVerificationReward:
    """Combine a bounded verifier result with the shared schedule."""

    if not 0.0 <= float(terminal_reward) <= 1.0:
        raise ValueError("Terminal reward must be in [0, 1]")
    gate = bool(terminal_gate_passed)
    process_step_components = {
        step_index: (schedule.process_weight * score if gate else 0.0)
        for step_index, score in verification.components_by_step().items()
    }
    terminal_component = schedule.terminal_weight * float(terminal_reward) if gate else 0.0
    score = terminal_component + sum(process_step_components.values())
    if score > 1.0 + _EPSILON:
        raise RuntimeError("Composed terminal and process reward must remain bounded")
    return ComposedVerificationReward(
        score=score,
        terminal_reward=float(terminal_reward),
        process_reward=verification.process_reward,
        terminal_component=terminal_component,
        process_step_components=process_step_components,
        schedule=schedule,
        terminal_gate_passed=gate,
        verifier_audit={
            **verification.audit,
            "credits_by_step": verification.audits_by_step(),
        },
    )


def apply_composed_reward(
    steps: Sequence[Any],
    composed: ComposedVerificationReward,
    *,
    final_step_index: int | None = None,
    extra_final_info: Mapping[str, Any] | None = None,
) -> None:
    """Place verifier credits at their source steps and terminal credit last.

    `steps` only need the AgentFlowStep reward_score and extra_fields surface,
    keeping the adapter independent from task environments and optimizer code.
    """

    if not steps:
        raise ValueError("Cannot apply a verifier reward to an empty trajectory")
    final_index = len(steps) if final_step_index is None else int(final_step_index)
    if not 1 <= final_index <= len(steps):
        raise ValueError("final_step_index must identify a rollout step")
    base_reward = sum(float(step.reward_score or 0.0) for step in steps)
    audits_by_step = composed.verifier_audit.get("credits_by_step", {})
    for step_index, component in composed.process_step_components.items():
        if not 1 <= step_index <= len(steps):
            raise ValueError(f"Verifier credit step {step_index} is outside trajectory length {len(steps)}")
        step = steps[step_index - 1]
        step.reward_score = float(step.reward_score or 0.0) + float(component)
        info = step.extra_fields.get("reward_extra_info", {})
        info.update(
            {
                "optimizer_process_component": float(component),
                "process_component_valid": float(component) > 0.0,
                "optimizer_reward_phase": composed.schedule.phase,
                "optimizer_terminal_weight": composed.schedule.terminal_weight,
                "optimizer_process_weight": composed.schedule.process_weight,
                "weight_sampling": composed.schedule.weight_sampling,
                "optimizer_weight_group_key": composed.schedule.prompt_group_key,
                "process_verifier_audit": audits_by_step.get(str(step_index), audits_by_step.get(step_index)),
            }
        )
        step.extra_fields["reward_extra_info"] = info

    final_step = steps[final_index - 1]
    final_step.reward_score = float(final_step.reward_score or 0.0) + composed.terminal_component
    final_info = final_step.extra_fields.get("reward_extra_info", {})
    final_info.update(composed.record())
    final_info.update(
        {
            "optimizer_base_reward": base_reward,
            "optimizer_assigned_total_reward": base_reward + composed.score,
        }
    )
    if extra_final_info:
        final_info.update(extra_final_info)
    final_step.extra_fields["reward_extra_info"] = final_info

    assigned = sum(float(step.reward_score or 0.0) for step in steps)
    expected = base_reward + composed.score
    if abs(assigned - expected) > _EPSILON:
        raise RuntimeError(
            "Step rewards do not sum to the composed verifier reward: "
            f"{assigned} != {expected}"
        )
