"""Deterministic replay verifier for Vision-R1 visual artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from agent_r1.verifier import VerificationCredit, VerificationGroup, VerificationResult, normalize_verification_groups
from recipes.vision_r1.artifacts import (
    VisualArtifact,
    image_sha256,
    parse_visual_certificate,
)


def _answer_coupled(answer: str, claim: str) -> bool:
    normalized_answer = "".join(str(answer).split()).casefold()
    normalized_claim = "".join(str(claim).split()).casefold()
    return bool(normalized_answer) and normalized_answer in normalized_claim


def _replay_artifact(
    artifact: VisualArtifact,
    artifacts: Mapping[str, VisualArtifact],
) -> tuple[bool, dict[str, Any]]:
    errors: list[str] = []
    parent = artifacts.get(artifact.parent_artifact_id or "")
    if parent is None:
        errors.append("parent_missing")
    elif parent.sha256 != artifact.parent_sha256:
        errors.append("parent_hash_mismatch")
    if artifact.bbox_2d is None:
        errors.append("bbox_missing")
    if errors:
        return False, {"artifact_id": artifact.artifact_id, "errors": errors}
    replay = parent.image.crop(artifact.bbox_2d).convert("RGB")
    replay_sha256 = image_sha256(replay)
    if replay_sha256 != artifact.sha256:
        errors.append("replay_hash_mismatch")
    if replay.size != (artifact.width, artifact.height):
        errors.append("replay_dimensions_mismatch")
    return not errors, {
        "artifact_id": artifact.artifact_id,
        "parent_artifact_id": artifact.parent_artifact_id,
        "bbox_2d": list(artifact.bbox_2d),
        "recorded_sha256": artifact.sha256,
        "replay_sha256": replay_sha256,
        "created_step": artifact.created_step,
        "errors": errors,
    }


def verify_process(
    *,
    artifacts: Mapping[str, VisualArtifact],
    raw_certificate: Any,
    submitted_answer: str,
    submission_attempted: bool | None = None,
) -> VerificationResult:
    """Verify crop evidence independently of final-answer eligibility.

    A readable certificate selects its required evidence; missing or invalid
    certificates fall back to the recorded crops. A zero certificate slot is
    retained only when a submission or certificate actually triggers that
    check. Missing required crops stay in the denominator. Answer coupling
    is an audit check and never suppresses independently replayed crop credits.
    """

    all_crop_checks = [_replay_artifact(artifact, artifacts)[0]
                       for artifact in artifacts.values() if artifact.parent_artifact_id is not None]
    certificate_applicable = (bool(submission_attempted) or raw_certificate is not None
                              or bool(submitted_answer.strip()))
    certificate, certificate_errors = parse_visual_certificate(raw_certificate)
    if certificate is None:
        crops = [artifact for artifact in artifacts.values() if artifact.parent_artifact_id is not None]
        artifact_audits = []
        crop_scores = []
        crop_ids = {artifact.artifact_id for artifact in crops}
        declared_ids = raw_certificate.get("evidence_artifact_ids", []) if isinstance(raw_certificate, Mapping) else []
        missing_evidence_slots = sum(
            not isinstance(item, str) or item not in crop_ids for item in declared_ids
        ) if isinstance(declared_ids, list) else 1
        # Even malformed declarations retain extractable missing evidence slots.
        missing_certificate_slots = int(certificate_applicable)
        for artifact in crops:
            valid, audit = _replay_artifact(artifact, artifacts)
            artifact_audits.append(audit)
            crop_scores.append(float(valid))
        normalization = normalize_verification_groups(tuple(
            VerificationGroup(
                group_id=f"vision:crop:{artifact.artifact_id}", kind="singleton_artifact",
                scores=(score,),
            ) for artifact, score in zip(crops, crop_scores, strict=True)
        ) + tuple(VerificationGroup(
            group_id=f"vision:missing_evidence:{index}", kind="singleton_artifact", scores=(0.0,),
        ) for index in range(missing_evidence_slots)) + (VerificationGroup(
            group_id="vision:submission_certificate", kind="singleton_artifact", scores=(0.0,),
            applicable=(certificate_applicable,),
        ),))
        denominator = normalization["normalizer"]
        credits = tuple(VerificationCredit(
            step_index=artifact.created_step, score=group["normalized_credit"],
            audit={"kind": "visual_crop_replay", "artifact_id": artifact.artifact_id,
                   "sha256": artifact.sha256, "normalization_group": group},
        ) for artifact, group in zip(crops, normalization["groups"][:len(crops)], strict=True)
            if group["group_score"] > 0)
        return VerificationResult(
            credits=tuple(credits),
            audit={
                "verifier": "vision_r1_visual_replay",
                "schema_valid": False,
                "normalization": "mean_over_applicable_crop_and_required_evidence_checks_including_failures",
                "empty_step_policy": "zero_credit_excluded_from_denominator",
                "applicable_checks": [0] * (missing_certificate_slots + missing_evidence_slots) + [int(value) for value in all_crop_checks],
                "certificate_check_applicable": certificate_applicable,
                "answer_coupled": False,
                "certificate_errors": list(certificate_errors) if certificate_applicable else [],
                "artifact_audits": artifact_audits,
                "process_denominator": denominator,
                "normalization_groups": normalization["groups"],
                "missing_certificate_slots": missing_certificate_slots,
                "missing_evidence_slots": missing_evidence_slots,
                "replay_valid_artifact_count": len(credits),
            },
        )

    answer_coupled = _answer_coupled(submitted_answer, certificate.claim)
    artifact_audits: list[dict[str, Any]] = []
    cited_artifacts: list[VisualArtifact | None] = []
    cited_scores = []
    for artifact_id in certificate.evidence_artifact_ids:
        artifact = artifacts.get(artifact_id)
        cited_artifacts.append(artifact)
        if artifact is None or artifact.parent_artifact_id is None:
            artifact_audits.append({"artifact_id": artifact_id, "errors": ["artifact_missing"]})
            cited_scores.append(0.0)
            continue
        replay_valid, audit = _replay_artifact(artifact, artifacts)
        artifact_audits.append(audit)
        cited_scores.append(float(replay_valid))

    normalization = normalize_verification_groups(tuple(
        VerificationGroup(group_id=f"vision:cited:{artifact_id}", kind="singleton_artifact", scores=(score,))
        for artifact_id, score in zip(certificate.evidence_artifact_ids, cited_scores, strict=True)
    ))
    denominator = normalization["normalizer"]
    credits = tuple(
            VerificationCredit(
                step_index=artifact.created_step,
                score=group["normalized_credit"],
                audit={
                    "kind": "visual_crop_replay",
                    "artifact_id": artifact.artifact_id,
                    "sha256": artifact.sha256,
                    "normalization_group": group,
                },
            )
            for artifact, group in zip(cited_artifacts, normalization["groups"], strict=True)
            if artifact is not None and group["group_score"] > 0
    )
    return VerificationResult(
        credits=credits,
        audit={
            "verifier": "vision_r1_visual_replay",
            "schema_valid": True,
            "normalization": "mean_over_applicable_cited_crop_checks_including_failures",
            "empty_step_policy": "zero_credit_excluded_from_denominator",
            "certificate_check_applicable": True,
            "applicable_checks": [1, int(answer_coupled)] + [int(not item["errors"]) for item in artifact_audits] + [int(value) for value in all_crop_checks],
            "answer_coupled": answer_coupled,
            "certificate_errors": [],
            "artifact_audits": artifact_audits,
            "cited_artifact_count": denominator,
            "process_denominator": denominator,
            "normalization_groups": normalization["groups"],
            "missing_certificate_slots": 0,
            "replay_valid_artifact_count": len(credits),
        },
    )
