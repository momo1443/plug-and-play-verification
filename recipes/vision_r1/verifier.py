"""Deterministic replay verifier for Vision-R1 visual artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from agent_r1.verifier import VerificationCredit, VerificationResult
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
) -> VerificationResult:
    """Verify certificate schema, answer coupling, lineage, and crop replay."""

    certificate, certificate_errors = parse_visual_certificate(raw_certificate)
    if certificate is None:
        return VerificationResult(
            credits=(),
            audit={
                "verifier": "vision_r1_visual_replay",
                "schema_valid": False,
                "answer_coupled": False,
                "certificate_errors": list(certificate_errors),
                "artifact_audits": [],
            },
        )

    answer_coupled = _answer_coupled(submitted_answer, certificate.claim)
    artifact_audits: list[dict[str, Any]] = []
    valid_artifacts: list[VisualArtifact] = []
    for artifact_id in certificate.evidence_artifact_ids:
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact.parent_artifact_id is None:
            artifact_audits.append({"artifact_id": artifact_id, "errors": ["artifact_missing"]})
            continue
        replay_valid, audit = _replay_artifact(artifact, artifacts)
        artifact_audits.append(audit)
        if replay_valid:
            valid_artifacts.append(artifact)

    denominator = len(certificate.evidence_artifact_ids)
    credits = ()
    if answer_coupled and denominator:
        credits = tuple(
            VerificationCredit(
                step_index=artifact.created_step,
                score=1.0 / denominator,
                audit={
                    "kind": "visual_crop_replay",
                    "artifact_id": artifact.artifact_id,
                    "sha256": artifact.sha256,
                },
            )
            for artifact in valid_artifacts
        )
    return VerificationResult(
        credits=credits,
        audit={
            "verifier": "vision_r1_visual_replay",
            "schema_valid": True,
            "answer_coupled": answer_coupled,
            "certificate_errors": [],
            "artifact_audits": artifact_audits,
            "cited_artifact_count": denominator,
            "replay_valid_artifact_count": len(valid_artifacts),
        },
    )
