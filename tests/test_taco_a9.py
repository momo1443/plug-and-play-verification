"""Focused tests for TACO A9 test isolation and execution certificates."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recipes.taco_a9.dsl import CodeArtifact, ExecutionRecord, parse_submission_certificate
from recipes.taco_a9.prepare_data import build_dataset
from recipes.taco_a9.reward_contract import sample_weights
from recipes.taco_a9.sandbox import TestCase as SandboxTestCase, TestSuite as SandboxTestSuite
from recipes.taco_a9.verifier import (
    CertificateAudit,
    CodeArtifactAudit,
    audit_code_artifacts,
    build_verification_result,
    verify_process,
    verify_submission_certificate,
)


class TacoA9Test(unittest.TestCase):
    def _artifact(self, ordinal: int = 1) -> CodeArtifact:
        return CodeArtifact.create(ordinal, "print('ok')\n")

    def _record(self, artifact: CodeArtifact) -> ExecutionRecord:
        return ExecutionRecord(
            run_id="run:1:abc",
            code_artifact_id=artifact.artifact_id,
            code_sha256=artifact.sha256,
            suite_id="developer_v1",
            suite_sha256="a" * 64,
            result_sha256="b" * 64,
            passed=2,
            total=2,
            all_passed=True,
            timed_out=False,
        )

    def test_submission_certificate_requires_exact_fields(self):
        certificate, errors = parse_submission_certificate({"claim": "tested"})
        self.assertIsNone(certificate)
        self.assertIn("certificate_fields_invalid", errors)

    def test_lineage_accepts_final_code_with_real_passing_run(self):
        artifact = self._artifact()
        record = self._record(artifact)
        suite = SandboxTestSuite("developer_v1", (SandboxTestCase("", ""),))
        raw = {
            "final_code_artifact_id": artifact.artifact_id,
            "final_code_sha256": artifact.sha256,
            "evidence_run_ids": [record.run_id],
            "claim": "developer suite passed",
        }
        audit, certificate = verify_submission_certificate(
            raw,
            submitted_artifact=artifact,
            artifacts={artifact.artifact_id: artifact},
            records={record.run_id: record},
            developer_suite=suite,
        )
        self.assertIsNotNone(certificate)
        self.assertEqual(audit.schema_valid, 1)
        self.assertEqual(audit.lineage_valid, 1)
        self.assertEqual(audit.own_valid, 1)

    def test_lineage_rejects_test_record_for_different_code(self):
        artifact = self._artifact()
        other = CodeArtifact.create(2, "print('different')\n")
        record = self._record(other)
        suite = SandboxTestSuite("developer_v1", (SandboxTestCase("", ""),))
        raw = {
            "final_code_artifact_id": artifact.artifact_id,
            "final_code_sha256": artifact.sha256,
            "evidence_run_ids": [record.run_id],
            "claim": "developer suite passed",
        }
        audit, _ = verify_submission_certificate(
            raw,
            submitted_artifact=artifact,
            artifacts={artifact.artifact_id: artifact, other.artifact_id: other},
            records={record.run_id: record},
            developer_suite=suite,
        )
        self.assertEqual(audit.lineage_valid, 0)
        self.assertIn("final_code_not_successfully_tested", audit.errors)

    def test_uniform_schedule_is_shared_within_prompt_group(self):
        terminal_weights = [
            sample_weights(
                global_step=51,
                is_validation=False,
                terminal_warmup_steps=50,
                prompt_group_key="task:7",
            )[0]
            for _ in range(4)
        ]
        self.assertTrue(all(0.0 <= weight < 1.0 for weight in terminal_weights))
        self.assertEqual(len(set(terminal_weights)), 1)
        self.assertNotEqual(
            terminal_weights[0],
            sample_weights(
                global_step=51,
                is_validation=False,
                terminal_warmup_steps=50,
                prompt_group_key="task:8",
            )[0],
        )
        self.assertNotEqual(
            terminal_weights[0],
            sample_weights(
                global_step=52,
                is_validation=False,
                terminal_warmup_steps=50,
                prompt_group_key="task:7",
            )[0],
        )
        self.assertEqual(
            sample_weights(
                global_step=1,
                is_validation=False,
                terminal_warmup_steps=50,
                prompt_group_key="task:7",
            ),
            (1.0, 0.0, "terminal_private_warmup"),
        )
        self.assertEqual(
            sample_weights(
                global_step=1,
                is_validation=True,
                terminal_warmup_steps=50,
                prompt_group_key="task:7",
            ),
            (1.0, 0.0, "validation_terminal_private_tests"),
        )
        self.assertEqual(
            sample_weights(
                global_step=0,
                is_validation=False,
                terminal_warmup_steps=50,
                prompt_group_key="task:7",
            ),
            (1.0, 0.0, "terminal_private_warmup"),
        )

    def test_verifier_result_averages_artifacts_and_certificate(self):
        artifact_audits = [
            CodeArtifactAudit("code:1", "a" * 64, 1, "run:1", 1, 1, 2, 5, 0.4, 0.4, ()),
            CodeArtifactAudit("code:2", "b" * 64, 1, "run:2", 1, 1, 2, 2, 1.0, 1.0, ()),
        ]
        result = build_verification_result(
            artifact_audits,
            CertificateAudit(1, 1, 1, 1, ()),
            run_step_indices={"run:1": 2, "run:2": 4},
            final_step_index=5,
        )
        self.assertAlmostEqual(result.process_reward, 0.8)
        self.assertEqual(result.components_by_step(), {2: 0.4 / 3, 4: 1 / 3, 5: 1 / 3})
        self.assertEqual(CertificateAudit(1, 1, 1, 1, ()).raw_local_credit, 1.0)
        self.assertEqual(CertificateAudit(1, 1, 0, 0, ()).raw_local_credit, 2 / 3)

    def test_artifact_process_requires_exact_replayable_evidence(self):
        artifact = self._artifact()
        suite = SandboxTestSuite("developer_v1", (SandboxTestCase("", "ok\n"),))

        class FakeExecutor:
            def run_suite(self, replay_artifact, replay_suite, *, run_ordinal):
                self.assertion = (replay_artifact, replay_suite, run_ordinal)
                record = ExecutionRecord(
                    run_id="run:0:replay",
                    code_artifact_id=artifact.artifact_id,
                    code_sha256=artifact.sha256,
                    suite_id=suite.suite_id,
                    suite_sha256=suite.sha256,
                    result_sha256="c" * 64,
                    passed=1,
                    total=1,
                    all_passed=True,
                    timed_out=False,
                )
                return record, {}

        record = ExecutionRecord(
            run_id="run:1:abc",
            code_artifact_id=artifact.artifact_id,
            code_sha256=artifact.sha256,
            suite_id=suite.suite_id,
            suite_sha256=suite.sha256,
            result_sha256="c" * 64,
            passed=1,
            total=1,
            all_passed=True,
            timed_out=False,
        )
        audits, replayed = audit_code_artifacts(
            {artifact.artifact_id: artifact},
            {record.run_id: record},
            developer_suite=suite,
            executor=FakeExecutor(),
        )
        self.assertEqual(audits[0].verified_score, 1.0)
        self.assertEqual(replayed, {record.run_id})

        result = verify_process(
            artifacts={artifact.artifact_id: artifact},
            records={record.run_id: record},
            raw_certificate={
                "final_code_artifact_id": artifact.artifact_id,
                "final_code_sha256": artifact.sha256,
                "evidence_run_ids": [record.run_id],
                "claim": "developer suite passed",
            },
            submitted_artifact=artifact,
            developer_suite=suite,
            executor=FakeExecutor(),
            run_step_indices={record.run_id: 2},
            final_step_index=3,
        )
        self.assertEqual(result.process_reward, 1.0)
        self.assertEqual(result.components_by_step(), {2: 0.5, 3: 0.5})

    def test_untested_versions_and_untriggered_submission_checks_are_excluded(self):
        tested = CodeArtifactAudit("code:1", "a" * 64, 1, "run:1", 1, 1, 2, 2, 1, 1, ())
        untested = CodeArtifactAudit("code:2", "b" * 64, 0, None, 0, 0, 0, 0, 0, 0, ("artifact_not_tested",))
        result = build_verification_result(
            [tested, untested], CertificateAudit(0, 0, 0, 0, ()),
            run_step_indices={"run:1": 2}, final_step_index=3,
            certificate_applicable=False,
        )
        self.assertEqual(result.process_reward, 1)
        self.assertEqual(result.audit["process_denominator"], 1)
        self.assertEqual(result.components_by_step(), {2: 1})
        self.assertEqual(result.audit["applicable_checks"], [1])
        self.assertFalse(result.audit["artifact_process_audits"][1]["check_applicable"])

    def test_failed_execution_and_required_missing_certificate_are_retained(self):
        passed = CodeArtifactAudit("code:1", "a" * 64, 1, "run:1", 1, 1, 2, 2, 1, 1, ())
        failed = replace(passed, artifact_id="code:2", run_id="run:2",
                         passed=0, developer_pass_rate=0, verified_score=0)
        result = build_verification_result(
            [passed, failed], CertificateAudit(0, 0, 0, 0, ("certificate_not_object",)),
            run_step_indices={"run:1": 2, "run:2": 4}, final_step_index=5,
        )
        self.assertAlmostEqual(result.process_reward, 1 / 3)
        self.assertEqual(result.audit["process_denominator"], 3)
        self.assertEqual(result.audit["applicable_checks"], [0, 1, 0])

    def test_no_applicable_checks_receive_zero_process_reward(self):
        result = build_verification_result(
            [], CertificateAudit(0, 0, 0, 0, ()),
            run_step_indices={}, final_step_index=1, certificate_applicable=False,
        )
        self.assertEqual(result.process_reward, 0)
        self.assertEqual(result.audit["process_denominator"], 0)
        self.assertEqual(result.audit["applicable_checks"], [])

    def test_lineage_failure_is_retained_at_the_attempted_execution_step(self):
        artifact = self._artifact()
        record = replace(self._record(artifact), code_sha256="0" * 64)

        class NoReplayExecutor:
            def run_suite(self, *args, **kwargs):
                raise AssertionError("Lineage mismatch must not be replayed")

        result = verify_process(
            artifacts={artifact.artifact_id: artifact}, records={record.run_id: record},
            raw_certificate=None, submitted_artifact=None,
            developer_suite=SandboxTestSuite("developer_v1", (SandboxTestCase("", ""),)),
            executor=NoReplayExecutor(), run_step_indices={record.run_id: 2}, final_step_index=3,
        )
        self.assertEqual(result.process_reward, 0)
        self.assertEqual(result.audit["process_denominator"], 1)
        self.assertEqual(result.components_by_step(), {2: 0})
        self.assertIn("execution_lineage_mismatch", result.audit["artifact_process_audits"][0]["errors"])

    def test_tested_artifact_cannot_silently_lose_its_source_step(self):
        tested = CodeArtifactAudit("code:1", "a" * 64, 1, "run:1", 1, 1, 2, 2, 1, 1, ())
        with self.assertRaisesRegex(ValueError, "source rollout step"):
            build_verification_result(
                [tested], CertificateAudit(0, 0, 0, 0, ()),
                run_step_indices={}, final_step_index=1, certificate_applicable=False,
            )

    def test_preparation_excludes_solutions_and_partitions_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            taco_dir = root / "taco" / "ALL"
            taco_dir.mkdir(parents=True)
            rows = []
            for index in range(4):
                cases = {"inputs": [f"{n}\n" for n in range(6)], "outputs": [f"{n}\n" for n in range(6)]}
                rows.append(
                    {
                        "question": f"Task {index}",
                        "solutions": json.dumps(["print('reference solution')"]),
                        "starter_code": "",
                        "input_output": json.dumps(cases),
                        "difficulty": "EASY",
                        "source": "unit-test",
                        "url": f"https://example.test/{index}",
                    }
                )
            pq.write_table(pa.Table.from_pylist(rows), taco_dir / "train-00000-of-00001.parquet")
            output_dir = root / "prepared"
            manifest = build_dataset(
                argparse.Namespace(
                    taco_dir=str(root / "taco"),
                    output_dir=str(output_dir),
                    min_test_cases=6,
                    developer_fraction=0.5,
                    validation_fraction=0.5,
                    max_tasks=0,
                    train_batch_size=1,
                )
            )
            self.assertEqual(manifest["eligible_tasks"], 4)
            train_row = pq.read_table(output_dir / "train.parquet").to_pylist()[0]
            self.assertNotIn("solutions", train_row)
            self.assertNotIn("input_output", train_row)
            self.assertEqual(train_row["data_source"], "taco_a9_python")
            sidecar = json.loads((output_dir / "taco_a9_sidecar.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("solutions", sidecar)
            self.assertGreaterEqual(len(sidecar["developer_cases"]), 2)
            self.assertGreaterEqual(len(sidecar["private_cases"]), 2)


if __name__ == "__main__":
    unittest.main()
