"""Gold-free deterministic verifier for local reasoning transitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from recipes.hotpotqa_lr.dsl import DSL_VERSION, ClaimSourceStep, ReasonStep, parse_reason_step
from recipes.hotpotqa_lr.reward_fn import normalize_answer

VERIFIER_VERSION = "hotpotqa-local-reasoning-verifier-v1"


def artifact_content_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def artifact_id_for_passage(passage_id: Any) -> str:
    return f"passage:{passage_id}"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _output_value(step: ReasonStep) -> tuple[str | None, tuple[str, ...]]:
    output_keys = {
        "select_exact_span": "text",
        "extract_bridge_entity": "bridge_entity",
        "extract_answer_candidate": "answer_candidate",
        "normalize_answer": "normalized_answer",
        "compare_equal": "equal",
    }
    key = output_keys.get(step.op)
    if key is None or set(step.output) != {key}:
        return None, ("output_fields_invalid",)
    value = step.output.get(key)
    if not isinstance(value, str) or not value.strip():
        return None, ("output_value_invalid",)
    value = value.strip()
    if step.op == "compare_equal" and value not in {"yes", "no"}:
        return None, ("compare_output_invalid",)
    return value, ()


def _action_contains(action_type: str, action_value: str, conclusion: str) -> bool:
    normalized_conclusion = normalize_answer(conclusion)
    normalized_action = normalize_answer(action_value)
    if not normalized_conclusion or not normalized_action:
        return False
    if action_type == "finish":
        return normalized_conclusion == normalized_action
    return f" {normalized_conclusion} " in f" {normalized_action} "


def _claim_source_finish_contains_answer(action_type: str, action_value: str, claim: str) -> bool:
    if action_type != "finish":
        return False
    normalized_claim = normalize_answer(claim)
    normalized_answer = normalize_answer(action_value)
    return bool(
        normalized_claim
        and normalized_answer
        and f" {normalized_answer} " in f" {normalized_claim} "
    )


@dataclass(frozen=True)
class LocalStepAudit:
    transition_index: int
    action_type: str
    action_value: str
    raw_reason_step: Any
    canonical_reason_step: dict[str, Any] | None
    ref: str | None
    input_refs: tuple[str, ...]
    ancestor_refs: tuple[str, ...]
    conclusion: str | None
    grounding_valid: int
    inference_valid: int
    own_valid: int
    invalid_ancestor_count: int
    dependency_factor: float
    action_coupled: int
    raw_local_credit: float
    errors: tuple[str, ...]
    canonical_digest_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "verifier_version": VERIFIER_VERSION,
            "dsl_version": DSL_VERSION,
            "transition_index": self.transition_index,
            "action_type": self.action_type,
            "action_value": self.action_value,
            "raw_reason_step": self.raw_reason_step,
            "canonical_reason_step": self.canonical_reason_step,
            "ref": self.ref,
            "input_refs": list(self.input_refs),
            "ancestor_refs": list(self.ancestor_refs),
            "conclusion": self.conclusion,
            "grounding_valid": self.grounding_valid,
            "inference_valid": self.inference_valid,
            "own_valid": self.own_valid,
            "invalid_ancestor_count": self.invalid_ancestor_count,
            "dependency_factor": self.dependency_factor,
            "action_coupled": self.action_coupled,
            "raw_local_credit": self.raw_local_credit,
            "errors": list(self.errors),
            "canonical_digest_sha256": self.canonical_digest_sha256,
        }


def _inference_errors(
    step: ReasonStep,
    conclusion: str | None,
    prior_by_ref: Mapping[str, LocalStepAudit],
    question: str,
) -> tuple[str, ...]:
    if conclusion is None:
        return ("conclusion_unavailable",)
    spans = [premise.span for premise in step.premises]
    input_values = [prior_by_ref[ref].conclusion for ref in step.inputs if ref in prior_by_ref]

    if step.op == "select_exact_span":
        return () if spans and any(conclusion in span for span in spans) else ("span_selection_invalid",)
    if step.op == "extract_bridge_entity":
        if not spans or not any(conclusion in span for span in spans):
            return ("bridge_not_in_premise",)
        normalized = normalize_answer(conclusion)
        if normalized and f" {normalized} " in f" {normalize_answer(question)} ":
            return ("bridge_already_in_question",)
        return ()
    if step.op == "extract_answer_candidate":
        from_span = any(conclusion in span for span in spans)
        from_input = any(conclusion == value for value in input_values if value is not None)
        return () if from_span or from_input else ("answer_candidate_not_derived",)
    if step.op == "normalize_answer":
        if len(step.inputs) != 1 or step.premises:
            return ("normalize_sources_invalid",)
        source = input_values[0] if len(input_values) == 1 else None
        return () if source is not None and conclusion == normalize_answer(source) else ("normalization_invalid",)
    if step.op == "compare_equal":
        if len(step.inputs) != 2 or step.premises:
            return ("compare_sources_invalid",)
        if len(input_values) != 2 or any(value is None for value in input_values):
            return ("compare_inputs_unavailable",)
        expected = "yes" if normalize_answer(input_values[0]) == normalize_answer(input_values[1]) else "no"
        return () if conclusion == expected else ("comparison_invalid",)
    return ("op_unverifiable",)


def _base_audit(
    *,
    transition_index: int,
    raw_reason_step: Any,
    action_type: str,
    action_value: str,
    available_artifacts: Mapping[str, Mapping[str, Any]],
    prior_audits: Sequence[LocalStepAudit],
    question: str,
    dependency_taint_gamma: float,
) -> LocalStepAudit:
    parsed, parse_errors = parse_reason_step(raw_reason_step)
    prior_by_ref = {audit.ref: audit for audit in prior_audits if audit.ref}
    errors = list(parse_errors)
    ref: str | None = None
    inputs: tuple[str, ...] = ()
    ancestors: set[str] = set()
    conclusion: str | None = None
    canonical: dict[str, Any] | None = None
    grounding_valid = 0
    inference_valid = 0

    if parsed is not None:
        canonical = parsed.record()
        if isinstance(parsed, ClaimSourceStep):
            ref = f"cs{transition_index}"
            conclusion = parsed.claim
            artifact = available_artifacts.get(parsed.source)
            if artifact is None:
                errors.append(f"artifact_unknown:{parsed.source}")
            else:
                text = artifact.get("text")
                digest = artifact.get("content_sha256")
                if not isinstance(text, str) or digest != artifact_content_sha256(text):
                    errors.append(f"artifact_snapshot_invalid:{parsed.source}")
                elif parsed.claim not in text:
                    errors.append(f"claim_not_exact:{parsed.source}")
            grounding_valid = int(not errors)
            inference_valid = grounding_valid
        else:
            ref = parsed.ref
            inputs = parsed.inputs
            if ref in prior_by_ref:
                errors.append("ref_duplicate")
            for input_ref in inputs:
                ancestor = prior_by_ref.get(input_ref)
                if ancestor is None:
                    errors.append(f"input_unknown:{input_ref}")
                    continue
                ancestors.add(input_ref)
                ancestors.update(ancestor.ancestor_refs)

            for premise in parsed.premises:
                artifact = available_artifacts.get(premise.artifact_id)
                if artifact is None:
                    errors.append(f"artifact_unknown:{premise.artifact_id}")
                    continue
                text = artifact.get("text")
                digest = artifact.get("content_sha256")
                if not isinstance(text, str) or digest != artifact_content_sha256(text):
                    errors.append(f"artifact_snapshot_invalid:{premise.artifact_id}")
                elif premise.span not in text:
                    errors.append(f"span_not_exact:{premise.artifact_id}")

            conclusion, output_errors = _output_value(parsed)
            errors.extend(output_errors)
            grounding_valid = int(not errors)
            inference_errors = _inference_errors(parsed, conclusion, prior_by_ref, question)
            if inference_errors:
                errors.extend(inference_errors)
            else:
                inference_valid = 1

    invalid_ancestors = sum(
        1 for ancestor_ref in ancestors if prior_by_ref[ancestor_ref].own_valid == 0
    )
    dependency_factor = float(dependency_taint_gamma) ** invalid_ancestors
    own_valid = grounding_valid * inference_valid
    seed_record = {
        "transition_index": transition_index,
        "action_type": action_type,
        "action_value": action_value,
        "canonical_reason_step": canonical,
        "ancestor_refs": sorted(ancestors),
        "grounding_valid": grounding_valid,
        "inference_valid": inference_valid,
        "invalid_ancestor_count": invalid_ancestors,
        "dependency_factor": dependency_factor,
        "errors": errors,
    }
    return LocalStepAudit(
        transition_index=transition_index,
        action_type=action_type,
        action_value=action_value,
        raw_reason_step=raw_reason_step,
        canonical_reason_step=canonical,
        ref=ref,
        input_refs=inputs,
        ancestor_refs=tuple(sorted(ancestors)),
        conclusion=conclusion,
        grounding_valid=grounding_valid,
        inference_valid=inference_valid,
        own_valid=own_valid,
        invalid_ancestor_count=invalid_ancestors,
        dependency_factor=dependency_factor,
        action_coupled=0,
        raw_local_credit=0.0,
        errors=tuple(errors),
        canonical_digest_sha256=_canonical_sha256(seed_record),
    )


def verify_trajectory(
    transitions: Sequence[Mapping[str, Any]],
    *,
    question: str,
    reward_horizon: int = 3,
    dependency_taint_gamma: float = 0.3,
) -> tuple[LocalStepAudit, ...]:
    """Verify a trajectory and resolve direct/indirect action coupling backward."""

    if reward_horizon <= 0:
        raise ValueError("reward_horizon must be positive")
    if not 0.0 <= dependency_taint_gamma <= 1.0:
        raise ValueError("dependency_taint_gamma must be in [0, 1]")

    audits: list[LocalStepAudit] = []
    for index, transition in enumerate(transitions, start=1):
        artifacts = transition.get("available_artifacts")
        if not isinstance(artifacts, Mapping):
            artifacts = {}
        audits.append(
            _base_audit(
                transition_index=index,
                raw_reason_step=transition.get("reason_step"),
                action_type=str(transition.get("action_type") or ""),
                action_value=str(transition.get("action_value") or ""),
                available_artifacts=artifacts,
                prior_audits=audits,
                question=question,
                dependency_taint_gamma=dependency_taint_gamma,
            )
        )

    action_sinks = [(audit.action_type, audit.action_value) for audit in audits]
    coupled_refs = {
        audit.ref
        for audit in audits
        if audit.ref
        and audit.conclusion
        and any(
            _action_contains(action_type, action_value, audit.conclusion)
            or (
                isinstance(audit.canonical_reason_step, Mapping)
                and set(audit.canonical_reason_step) == {"claim", "source"}
                and _claim_source_finish_contains_answer(
                    action_type, action_value, audit.conclusion
                )
            )
            for action_type, action_value in action_sinks
        )
    }
    changed = True
    while changed:
        changed = False
        for audit in audits:
            if audit.ref in coupled_refs:
                for input_ref in audit.input_refs:
                    if input_ref not in coupled_refs:
                        coupled_refs.add(input_ref)
                        changed = True

    finalized: list[LocalStepAudit] = []
    for audit in audits:
        action_coupled = int(bool(audit.ref and audit.ref in coupled_refs))
        horizon_ok = audit.transition_index <= reward_horizon
        credit = (
            audit.own_valid
            * audit.dependency_factor
            * action_coupled
            / float(reward_horizon)
            if horizon_ok
            else 0.0
        )
        errors = list(audit.errors)
        if not horizon_ok:
            errors.append("reward_horizon_exceeded")
        record = audit.record()
        record.update(
            {
                "action_coupled": action_coupled,
                "raw_local_credit": credit,
                "errors": errors,
            }
        )
        finalized.append(
            replace(
                audit,
                action_coupled=action_coupled,
                raw_local_credit=float(credit),
                errors=tuple(errors),
                canonical_digest_sha256=_canonical_sha256(record),
            )
        )
    return tuple(finalized)


def trajectory_audit_record(audits: Sequence[LocalStepAudit]) -> dict[str, Any]:
    count = len(audits)
    local_reward = sum(audit.raw_local_credit for audit in audits)
    if not 0.0 <= local_reward <= 1.0 + 1e-12:
        raise RuntimeError(f"Local reward escaped [0, 1]: {local_reward}")
    return {
        "verifier_version": VERIFIER_VERSION,
        "dsl_version": DSL_VERSION,
        "parsed_reason_step_count": sum(audit.canonical_reason_step is not None for audit in audits),
        "credited_transition_count": sum(audit.raw_local_credit > 0 for audit in audits),
        "local_reward": float(local_reward),
        "mean_grounding": (sum(audit.grounding_valid for audit in audits) / count if count else 0.0),
        "mean_inference": (sum(audit.inference_valid for audit in audits) / count if count else 0.0),
        "mean_dependency_factor": (
            sum(audit.dependency_factor for audit in audits) / count if count else 0.0
        ),
        "mean_action_coupling": (
            sum(audit.action_coupled for audit in audits) / count if count else 0.0
        ),
        "mean_invalid_ancestor_count": (
            sum(audit.invalid_ancestor_count for audit in audits) / count if count else 0.0
        ),
        "horizon_exceeded": any("reward_horizon_exceeded" in audit.errors for audit in audits),
        "self_correction": any(
            audit.transition_index > 1
            and audit.invalid_ancestor_count == 0
            and audit.own_valid == 1
            and any(prior.own_valid == 0 for prior in audits[: audit.transition_index - 1])
            for audit in audits
        ),
        "steps": [audit.record() for audit in audits],
    }
