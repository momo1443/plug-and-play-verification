"""Simplified certificate-grounded verifier for A9 transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agent_r1.verifier import VerificationCredit, VerificationResult
from recipes.hotpotqa_a9.dsl import (
    CERTIFICATE_SCHEMA_VERSION,
    FinishCertificate,
    SearchCertificate,
    parse_finish_certificate,
    parse_search_certificate,
)
from recipes.hotpotqa_lr.reward_fn import normalize_answer
from recipes.hotpotqa_lr.verifier import artifact_content_sha256

VERIFIER_VERSION = "hotpotqa-a9-certificate-v1"


@dataclass(frozen=True)
class CertificateStepAudit:
    """Audit record for one certificate-grounded transition."""

    transition_index: int
    action_type: str
    action_value: str
    certificate_raw: Any
    certificate_parsed: dict[str, Any] | None
    grounding_valid: int
    coupling_valid: int
    own_valid: int
    raw_local_credit: float
    errors: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        return {
            "verifier_version": VERIFIER_VERSION,
            "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
            "transition_index": self.transition_index,
            "action_type": self.action_type,
            "action_value": self.action_value,
            "certificate_raw": self.certificate_raw,
            "certificate_parsed": self.certificate_parsed,
            "grounding_valid": self.grounding_valid,
            "coupling_valid": self.coupling_valid,
            "own_valid": self.own_valid,
            "raw_local_credit": self.raw_local_credit,
            "errors": list(self.errors),
        }


def verify_grounding(
    certificate: SearchCertificate | FinishCertificate,
    available_artifacts: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Check that support_span is an exact substring of the source artifact text."""
    artifact = available_artifacts.get(certificate.source_id)
    if artifact is None:
        return False
    text = artifact.get("text")
    digest = artifact.get("content_sha256")
    if not isinstance(text, str) or digest != artifact_content_sha256(text):
        return False
    return certificate.support_span in text


def verify_coupling(
    action_type: str,
    action_value: str,
    certificate: SearchCertificate | FinishCertificate,
) -> bool:
    """Check that the certificate content relates to the action taken.

    For search: target must appear in the query.
    For finish: answer_span must match the answer (normalized).
    """
    if isinstance(certificate, SearchCertificate):
        normalized_target = normalize_answer(certificate.target)
        normalized_query = normalize_answer(action_value)
        if not normalized_target or not normalized_query:
            return False
        return f" {normalized_target} " in f" {normalized_query} "
    if isinstance(certificate, FinishCertificate):
        return normalize_answer(certificate.answer_span) == normalize_answer(action_value)
    return False


def verify_trajectory(
    transitions: Sequence[Mapping[str, Any]],
    *,
    reward_horizon: int = 3,
) -> tuple[CertificateStepAudit, ...]:
    """Verify a trajectory of certificate-grounded transitions.

    For each transition:
    1. Parse the certificate (best-effort; failure yields 0 credit).
    2. Check grounding: does support_span appear in the source artifact?
    3. Check coupling: does the certificate relate to the action taken?
    4. Compute credit: own_valid / horizon if within horizon, else 0.

    No dependency taint, no novelty gating, no ref chains — all removed
    from A8 to keep the verifier minimal and focused on what it can
    deterministically verify.
    """
    if reward_horizon <= 0:
        raise ValueError("reward_horizon must be positive")

    audits: list[CertificateStepAudit] = []
    for index, transition in enumerate(transitions, start=1):
        raw_cert = transition.get("certificate")
        action_type = str(transition.get("action_type") or "")
        action_value = str(transition.get("action_value") or "")
        artifacts = transition.get("available_artifacts")
        if not isinstance(artifacts, Mapping):
            artifacts = {}

        certificate = None
        parsed_record: dict[str, Any] | None = None
        errors: list[str] = []
        grounding_valid = 0
        coupling_valid = 0

        # Parse certificate based on action type
        if action_type == "search":
            certificate, cert_errors = parse_search_certificate(raw_cert)
        elif action_type == "finish":
            certificate, cert_errors = parse_finish_certificate(raw_cert)
        else:
            cert_errors = ("unknown_action_type",)

        errors.extend(cert_errors)

        if certificate is not None:
            parsed_record = certificate.record()
            grounding_valid = int(verify_grounding(certificate, artifacts))
            coupling_valid = int(verify_coupling(action_type, action_value, certificate))
            if not grounding_valid:
                errors.append("grounding_failed")
            if not coupling_valid:
                errors.append("coupling_failed")

        own_valid = grounding_valid * coupling_valid
        horizon_ok = index <= reward_horizon

        credit = float(own_valid) / float(reward_horizon) if horizon_ok else 0.0
        if not horizon_ok:
            errors.append("reward_horizon_exceeded")

        audits.append(
            CertificateStepAudit(
                transition_index=index,
                action_type=action_type,
                action_value=action_value,
                certificate_raw=raw_cert,
                certificate_parsed=parsed_record,
                grounding_valid=grounding_valid,
                coupling_valid=coupling_valid,
                own_valid=own_valid,
                raw_local_credit=credit,
                errors=tuple(errors),
            )
        )

    return tuple(audits)


def trajectory_audit_record(audits: Sequence[CertificateStepAudit]) -> dict[str, Any]:
    """Compute summary statistics for a trajectory of certificate audits."""
    count = len(audits)
    local_reward = sum(audit.raw_local_credit for audit in audits)
    if not 0.0 <= local_reward <= 1.0 + 1e-12:
        raise RuntimeError(f"Local reward escaped [0, 1]: {local_reward}")
    return {
        "verifier_version": VERIFIER_VERSION,
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "parsed_certificate_count": sum(
            audit.certificate_parsed is not None for audit in audits
        ),
        "credited_transition_count": sum(audit.raw_local_credit > 0 for audit in audits),
        "local_reward": float(local_reward),
        "mean_grounding": (
            sum(audit.grounding_valid for audit in audits) / count if count else 0.0
        ),
        "mean_coupling": (
            sum(audit.coupling_valid for audit in audits) / count if count else 0.0
        ),
        "mean_own_valid": (
            sum(audit.own_valid for audit in audits) / count if count else 0.0
        ),
        "horizon_exceeded": any(
            "reward_horizon_exceeded" in audit.errors for audit in audits
        ),
        "steps": [audit.record() for audit in audits],
    }


def build_verification_result(
    audits: Sequence[CertificateStepAudit],
    transition_flow_step_indices: Sequence[int],
) -> VerificationResult:
    """HotpotQA plugin: attach each certificate audit to its source action."""

    if len(audits) != len(transition_flow_step_indices):
        raise ValueError("Every HotpotQA transition audit requires a rollout step")
    summary = trajectory_audit_record(audits)
    credits: list[VerificationCredit] = []
    credits_by_step: dict[int, list[dict[str, Any]]] = {}
    for audit, zero_based_step_index in zip(audits, transition_flow_step_indices, strict=True):
        step_index = int(zero_based_step_index) + 1
        record = audit.record()
        credits.append(
            VerificationCredit(
                step_index=step_index,
                score=float(audit.raw_local_credit),
                audit=record,
            )
        )
        credits_by_step.setdefault(step_index, []).append(record)
    return VerificationResult(
        credits=tuple(credits),
        audit={
            "verifier": VERIFIER_VERSION,
            "certificate_audit": summary,
            "applicable_checks": [audit.own_valid for audit in audits],
            "credits_by_step": credits_by_step,
        },
    )


def verify_process(
    transitions: Sequence[Mapping[str, Any]],
    transition_flow_step_indices: Sequence[int],
    *,
    reward_horizon: int = 3,
) -> VerificationResult:
    """Public HotpotQA verifier plugin entry point."""

    audits = verify_trajectory(transitions, reward_horizon=reward_horizon)
    return build_verification_result(audits, transition_flow_step_indices)
