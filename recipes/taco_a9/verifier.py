"""Replay and lineage checks for TACO A9 execution certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent_r1.verifier import VerificationCredit, VerificationResult
from recipes.taco_a9.dsl import CodeArtifact, ExecutionRecord, SubmissionCertificate, parse_submission_certificate
from recipes.taco_a9.sandbox import BubblewrapPythonExecutor, TestSuite


VERIFIER_VERSION = "taco-a9-execution-replay-v1"


@dataclass(frozen=True)
class CertificateAudit:
    schema_valid: int
    lineage_valid: int
    replay_valid: int
    own_valid: int
    errors: tuple[str, ...]

    @property
    def raw_local_credit(self) -> float:
        return (self.schema_valid + self.lineage_valid + self.replay_valid) / 3.0

    def record(self) -> dict[str, Any]:
        return {
            "schema_valid": self.schema_valid,
            "lineage_valid": self.lineage_valid,
            "replay_valid": self.replay_valid,
            "own_valid": self.own_valid,
            "raw_local_credit": self.raw_local_credit,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class CodeArtifactAudit:
    artifact_id: str
    code_sha256: str
    tested: int
    run_id: str | None
    exact_lineage: int
    replay_valid: int
    passed: int
    total: int
    developer_pass_rate: float
    verified_score: float
    errors: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "code_sha256": self.code_sha256,
            "tested": self.tested,
            "run_id": self.run_id,
            "exact_lineage": self.exact_lineage,
            "replay_valid": self.replay_valid,
            "passed": self.passed,
            "total": self.total,
            "developer_pass_rate": self.developer_pass_rate,
            "verified_score": self.verified_score,
            "errors": list(self.errors),
        }


def build_verification_result(
    artifact_audits: list[CodeArtifactAudit],
    certificate_audit: CertificateAudit,
    *,
    run_step_indices: Mapping[str, int],
    final_step_index: int,
) -> VerificationResult:
    """TACO plugin: map replay audits and the submission certificate to credits."""

    denominator = len(artifact_audits) + 1
    credits: list[VerificationCredit] = []
    credits_by_step: dict[int, list[dict[str, Any]]] = {}
    for artifact_audit in artifact_audits:
        if artifact_audit.run_id is None:
            continue
        step_index = run_step_indices.get(artifact_audit.run_id)
        if step_index is None:
            continue
        record = artifact_audit.record()
        credits.append(
            VerificationCredit(
                step_index=step_index,
                score=float(artifact_audit.verified_score) / denominator,
                audit=record,
            )
        )
        credits_by_step.setdefault(step_index, []).append(record)

    certificate_record = certificate_audit.record()
    credits.append(
        VerificationCredit(
            step_index=final_step_index,
            score=float(certificate_audit.raw_local_credit) / denominator,
            audit=certificate_record,
        )
    )
    credits_by_step.setdefault(final_step_index, []).append(certificate_record)
    return VerificationResult(
        credits=tuple(credits),
        audit={
            "verifier": VERIFIER_VERSION,
            "artifact_process_audits": [item.record() for item in artifact_audits],
            "certificate_audit": certificate_record,
            "credits_by_step": credits_by_step,
        },
    )


def audit_code_artifacts(
    artifacts: Mapping[str, CodeArtifact],
    records: Mapping[str, ExecutionRecord],
    *,
    developer_suite: TestSuite,
    executor: BubblewrapPythonExecutor,
) -> tuple[list[CodeArtifactAudit], set[str]]:
    """Score every immutable code version using exact, replayable test evidence."""
    audits: list[CodeArtifactAudit] = []
    replay_valid_run_ids: set[str] = set()
    for artifact in artifacts.values():
        candidate_records = [
            record
            for record in records.values()
            if record.code_artifact_id == artifact.artifact_id
        ]
        exact_records = [
            record
            for record in candidate_records
            if record.code_sha256 == artifact.sha256
            and record.suite_id == developer_suite.suite_id
            and record.suite_sha256 == developer_suite.sha256
        ]
        errors: list[str] = []
        if not candidate_records:
            errors.append("artifact_not_tested")
        elif not exact_records:
            errors.append("execution_lineage_mismatch")

        replay_record = None
        matching_records: list[ExecutionRecord] = []
        if exact_records:
            replay_record, _ = executor.run_suite(artifact, developer_suite, run_ordinal=0)
            matching_records = [
                record
                for record in exact_records
                if record.result_sha256 == replay_record.result_sha256
            ]
            replay_valid_run_ids.update(record.run_id for record in matching_records)
            if not matching_records:
                errors.append("replay_result_mismatch")

        selected = matching_records[0] if matching_records else (exact_records[0] if exact_records else None)
        passed = int(selected.passed) if selected is not None else 0
        total = int(selected.total) if selected is not None else 0
        pass_rate = passed / total if total else 0.0
        exact_lineage = int(bool(exact_records))
        replay_valid = int(bool(matching_records))
        audits.append(
            CodeArtifactAudit(
                artifact_id=artifact.artifact_id,
                code_sha256=artifact.sha256,
                tested=int(bool(candidate_records)),
                run_id=selected.run_id if selected is not None else None,
                exact_lineage=exact_lineage,
                replay_valid=replay_valid,
                passed=passed,
                total=total,
                developer_pass_rate=pass_rate,
                verified_score=float(exact_lineage * replay_valid) * pass_rate,
                errors=tuple(errors),
            )
        )
    return audits, replay_valid_run_ids


def verify_submission_certificate(
    raw_certificate: Any,
    *,
    submitted_artifact: CodeArtifact | None,
    artifacts: Mapping[str, CodeArtifact],
    records: Mapping[str, ExecutionRecord],
    developer_suite: TestSuite,
    executor: BubblewrapPythonExecutor | None = None,
    replay_valid_run_ids: set[str] | None = None,
) -> tuple[CertificateAudit, SubmissionCertificate | None]:
    certificate, parse_errors = parse_submission_certificate(raw_certificate)
    if certificate is None:
        return CertificateAudit(0, 0, 0, 0, parse_errors), None
    errors: list[str] = []
    schema_valid = 1
    lineage_valid = 0
    replay_valid = 0
    cited_records = [records.get(run_id) for run_id in certificate.evidence_run_ids]
    if submitted_artifact is None:
        errors.append("submitted_artifact_missing")
    elif certificate.final_code_artifact_id != submitted_artifact.artifact_id:
        errors.append("submitted_artifact_id_mismatch")
    elif certificate.final_code_sha256 != submitted_artifact.sha256:
        errors.append("submitted_code_sha256_mismatch")
    elif artifacts.get(certificate.final_code_artifact_id) != submitted_artifact:
        errors.append("artifact_ledger_mismatch")
    elif any(record is None for record in cited_records):
        errors.append("evidence_run_missing")
    elif not any(
        record.code_sha256 == submitted_artifact.sha256
        and record.suite_id == developer_suite.suite_id
        and record.all_passed
        for record in cited_records
        if record is not None
    ):
        errors.append("final_code_not_successfully_tested")
    else:
        lineage_valid = 1

    if lineage_valid and replay_valid_run_ids is not None:
        if any(record is not None and record.run_id in replay_valid_run_ids for record in cited_records):
            replay_valid = 1
        else:
            errors.append("replay_result_mismatch")
    elif lineage_valid and executor is not None and submitted_artifact is not None:
        replay_record, _ = executor.run_suite(submitted_artifact, developer_suite, run_ordinal=0)
        matching = [record for record in cited_records if record is not None and record.code_sha256 == submitted_artifact.sha256]
        if any(
            replay_record.suite_sha256 == record.suite_sha256
            and replay_record.result_sha256 == record.result_sha256
            for record in matching
        ):
            replay_valid = 1
        else:
            errors.append("replay_result_mismatch")

    own_valid = int(
        schema_valid
        and lineage_valid
        and (replay_valid or (executor is None and replay_valid_run_ids is None))
    )
    return CertificateAudit(schema_valid, lineage_valid, replay_valid, own_valid, tuple(errors)), certificate


def verify_process(
    *,
    artifacts: Mapping[str, CodeArtifact],
    records: Mapping[str, ExecutionRecord],
    raw_certificate: Any,
    submitted_artifact: CodeArtifact | None,
    developer_suite: TestSuite,
    executor: BubblewrapPythonExecutor,
    run_step_indices: Mapping[str, int],
    final_step_index: int,
) -> VerificationResult:
    """Public TACO verifier plugin entry point."""

    artifact_audits, replay_valid_run_ids = audit_code_artifacts(
        artifacts,
        records,
        developer_suite=developer_suite,
        executor=executor,
    )
    certificate_audit, _ = verify_submission_certificate(
        raw_certificate,
        submitted_artifact=submitted_artifact,
        artifacts=artifacts,
        records=records,
        developer_suite=developer_suite,
        replay_valid_run_ids=replay_valid_run_ids,
    )
    return build_verification_result(
        artifact_audits,
        certificate_audit,
        run_step_indices=run_step_indices,
        final_step_index=final_step_index,
    )
