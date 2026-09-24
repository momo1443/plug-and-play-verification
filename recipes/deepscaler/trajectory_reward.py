"""Trajectory-level reward contracts for paired DeepScaleR ToolEnv arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from agent_r1.verifier import (
    ComposedVerificationReward,
    UniformRewardSchedule,
    VerificationCredit,
    VerificationResult,
    compose_verification_reward,
    uniform_reward_schedule,
)
from recipes.deepscaler.reward_fn import _process_reward_audit
from recipes.deepscaler.reward_fn_tool_terminal import compute_score as compute_terminal_em

_GROUP_WEIGHT_NAMESPACE = b"deepscaler-a9-group-weight-v1"


@dataclass(frozen=True)
class TrajectoryReward:
    """Causal step rewards and audit data for a completed agent trajectory."""

    score: float
    terminal_em: float
    process_reward: float
    terminal_component: float
    process_step_components: dict[int, float]
    terminal_weight: float
    process_weight: float
    phase: str
    weight_sampling: str
    prompt_group_key: str | None
    has_final_answer: bool
    process_segment_audits: list[dict[str, Any]]
    composed: ComposedVerificationReward

    def record(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "terminal_em": self.terminal_em,
            "process_reward": self.process_reward,
            "optimizer_terminal_component": self.terminal_component,
            "optimizer_process_total": sum(self.process_step_components.values()),
            "optimizer_process_step_components": self.process_step_components,
            "optimizer_terminal_weight": self.terminal_weight,
            "optimizer_process_weight": self.process_weight,
            "optimizer_reward_phase": self.phase,
            "weight_sampling": self.weight_sampling,
            "optimizer_weight_group_key": self.prompt_group_key,
            "has_final_answer": self.has_final_answer,
            "process_segment_audits": self.process_segment_audits,
        }


def verify_process(reasoning_segments: Iterable[tuple[int, str]]) -> VerificationResult:
    """DeepScaleR plugin: turn equation audits into normalized step credits."""
    records: list[dict[str, Any]] = []
    for turn, text in reasoning_segments:
        audit = _process_reward_audit(text)
        record = {
            "turn": turn,
            "step_count": int(audit["step_count"]),
            "graded_step_count": int(audit["graded_step_count"]),
            "equation_count": int(audit["equation_count"]),
            "verified_equation_count": int(audit["verified_equation_count"]),
            "process_reward": float(audit["process_reward"]),
            "steps": audit["steps"],
        }
        records.append(record)
    graded = [record for record in records if record["graded_step_count"] > 0]
    denominator = len(graded)
    credits = tuple(
        VerificationCredit(
            step_index=int(record["turn"]),
            score=float(record["process_reward"]) / denominator,
            audit=record,
        )
        for record in graded
    )
    return VerificationResult(
        credits=credits,
        audit={
            "verifier": "deepscaler_numeric_equations",
            "process_segment_audits": records,
            "credits_by_step": {
                int(record["turn"]): record for record in graded
            },
        },
    )


def _terminal_schedule(phase: str) -> UniformRewardSchedule:
    return UniformRewardSchedule(1.0, 0.0, phase, "fixed", None)


def _as_trajectory_reward(
    *,
    composed: ComposedVerificationReward,
    terminal_em: float,
    has_final_answer: bool,
    process_segment_audits: list[dict[str, Any]],
) -> TrajectoryReward:
    schedule = composed.schedule
    return TrajectoryReward(
        score=composed.score,
        terminal_em=terminal_em,
        process_reward=composed.process_reward,
        terminal_component=composed.terminal_component,
        process_step_components=dict(composed.process_step_components),
        terminal_weight=schedule.terminal_weight,
        process_weight=schedule.process_weight,
        phase=schedule.phase,
        weight_sampling=schedule.weight_sampling,
        prompt_group_key=schedule.prompt_group_key,
        has_final_answer=has_final_answer,
        process_segment_audits=process_segment_audits,
        composed=composed,
    )


def compute_tool_trajectory_reward(
    *,
    reward_mode: str,
    final_response: str | None,
    ground_truth: Any,
    reasoning_segments: Iterable[tuple[int, str]],
    global_step: int,
    is_validation: bool,
    em_warmup_steps: int,
    prompt_group_key: str | None = None,
) -> TrajectoryReward:
    """Score one complete DeepScaleR ToolEnv trajectory.

    A1 always uses strict terminal EM. A9 has the same terminal-only warmup,
    then samples one mixture weight shared by all rollouts for a prompt. Tool
    XML and observations must be excluded before passing reasoning text.
    """
    if reward_mode not in {"terminal_only", "uniform_equation_process"}:
        raise ValueError(f"Unsupported DeepScaleR ToolEnv reward mode: {reward_mode}")

    has_final_answer = bool(final_response and final_response.strip())
    terminal_em = (
        float(compute_terminal_em("deepmath", final_response, ground_truth)) if has_final_answer else 0.0
    )

    if reward_mode == "terminal_only":
        schedule = _terminal_schedule("terminal_only")
    else:
        schedule = uniform_reward_schedule(
            global_step=global_step,
            is_validation=is_validation,
            warmup_steps=em_warmup_steps,
            prompt_group_key=prompt_group_key or "unused-before-uniform-phase",
            namespace=_GROUP_WEIGHT_NAMESPACE,
            warmup_phase="em_warmup",
            validation_phase="validation_terminal_em",
            mixed_phase="uniform_equation_process",
        )

    should_verify = reward_mode == "uniform_equation_process" and schedule.process_weight > 0.0
    verification = (
        verify_process(reasoning_segments)
        if should_verify
        else VerificationResult(credits=(), audit={"verifier": "disabled", "credits_by_step": {}})
    )
    composed = compose_verification_reward(
        terminal_reward=terminal_em,
        verification=verification,
        schedule=schedule,
        terminal_gate_passed=has_final_answer,
    )
    process_segment_audits = list(verification.audit.get("process_segment_audits", []))
    return _as_trajectory_reward(
        composed=composed,
        terminal_em=terminal_em,
        has_final_answer=has_final_answer,
        process_segment_audits=process_segment_audits,
    )
