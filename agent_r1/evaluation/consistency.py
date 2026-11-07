"""Equation (9), independent of reward weights and the training reward source."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from agent_r1.verifier import VerificationResult

PROTOCOL_VERSION = "paper-local-consistency-v2"


def consistency_record(
    verification: VerificationResult, terminal_reward: float, *,
    eligible: bool, extra_checks: Iterable[bool] = (),
) -> dict[str, Any]:
    """Require a nonempty set of applicable checks, all of which pass.

    Final-submission eligibility is recorded separately and does not gate C_d.
    This deliberately does not threshold the normalized process reward. A
    shorter valid trace can have p < 1; a repaired trace can still contain an
    earlier failed check. Old dumps without check metadata cannot recover C_d.
    """
    checks = verification.audit.get("applicable_checks")
    if checks is None:
        raise ValueError("Deterministic verifier must record all applicable_checks")
    scores = [float(value) for value in checks] + [float(bool(value)) for value in extra_checks]
    if any(not 0 <= value <= 1 for value in scores):
        raise ValueError("Applicable verification checks must lie in [0, 1]")
    consistent = int(bool(scores) and all(value == 1 for value in scores))
    return {
        "verification_protocol": PROTOCOL_VERSION,
        "verification_checks": scores,
        "verification_check_count": len(scores),
        "verification_consistency": consistent,
        "verified_success": float(terminal_reward) * consistent,
        "terminal_eligible": bool(eligible),
    }


def finalize_evaluation_record(record: dict[str, Any]) -> dict[str, Any]:
    """Bind deterministic checks to the separately graded final outcome.

    AgentFlows receive no benchmark reference during generation. Their outcome
    placeholders must therefore be replaced before reporting Equation (9).
    """
    if not isinstance(record.get("is_correct"), bool):
        raise ValueError("Final evaluation requires a binary graded outcome")
    eligible = record["terminal_eligible"]
    if record["is_correct"] and not eligible:
        raise ValueError("An ineligible submission cannot be correct")
    verification = dict(record["verification"])
    if verification.get("verification_protocol") != PROTOCOL_VERSION:
        raise ValueError(f"Evaluation requires {PROTOCOL_VERSION} check metadata")
    deterministic = VerificationResult(
        credits=(), audit={"applicable_checks": verification["verification_checks"]},
    )
    fields = consistency_record(deterministic, float(record["is_correct"]), eligible=eligible)
    verification.update(fields)
    return {**record, **fields, "verification": verification}
