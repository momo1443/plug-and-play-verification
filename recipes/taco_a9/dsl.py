"""Immutable artifacts and model-authored certificates for TACO A9."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


CERTIFICATE_SCHEMA_VERSION = "taco-a9-execution-certificate-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CodeArtifact:
    artifact_id: str
    code: str
    sha256: str

    @classmethod
    def create(cls, ordinal: int, code: str) -> "CodeArtifact":
        digest = sha256_text(code)
        return cls(artifact_id=f"code:{ordinal}:{digest[:12]}", code=code, sha256=digest)

    def record(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "sha256": self.sha256}


@dataclass(frozen=True)
class ExecutionRecord:
    run_id: str
    code_artifact_id: str
    code_sha256: str
    suite_id: str
    suite_sha256: str
    result_sha256: str
    passed: int
    total: int
    all_passed: bool
    timed_out: bool

    def record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "code_artifact_id": self.code_artifact_id,
            "code_sha256": self.code_sha256,
            "suite_id": self.suite_id,
            "suite_sha256": self.suite_sha256,
            "result_sha256": self.result_sha256,
            "passed": self.passed,
            "total": self.total,
            "all_passed": self.all_passed,
            "timed_out": self.timed_out,
        }


@dataclass(frozen=True)
class SubmissionCertificate:
    final_code_artifact_id: str
    final_code_sha256: str
    evidence_run_ids: tuple[str, ...]
    claim: str

    def record(self) -> dict[str, Any]:
        return {
            "final_code_artifact_id": self.final_code_artifact_id,
            "final_code_sha256": self.final_code_sha256,
            "evidence_run_ids": list(self.evidence_run_ids),
            "claim": self.claim,
        }


def parse_submission_certificate(value: Any) -> tuple[SubmissionCertificate | None, tuple[str, ...]]:
    if not isinstance(value, dict):
        return None, ("certificate_not_object",)
    required = {"final_code_artifact_id", "final_code_sha256", "evidence_run_ids", "claim"}
    if set(value) != required:
        return None, ("certificate_fields_invalid",)
    artifact_id = value.get("final_code_artifact_id")
    code_sha256 = value.get("final_code_sha256")
    run_ids = value.get("evidence_run_ids")
    claim = value.get("claim")
    errors: list[str] = []
    if not isinstance(artifact_id, str) or not artifact_id.startswith("code:"):
        errors.append("final_code_artifact_id_invalid")
    if not isinstance(code_sha256, str) or len(code_sha256) != 64:
        errors.append("final_code_sha256_invalid")
    if not isinstance(run_ids, list) or not run_ids or not all(isinstance(item, str) and item.startswith("run:") for item in run_ids):
        errors.append("evidence_run_ids_invalid")
    if not isinstance(claim, str) or not claim.strip():
        errors.append("claim_invalid")
    if errors:
        return None, tuple(errors)
    return SubmissionCertificate(
        final_code_artifact_id=artifact_id,
        final_code_sha256=code_sha256,
        evidence_run_ids=tuple(run_ids),
        claim=claim.strip(),
    ), ()
