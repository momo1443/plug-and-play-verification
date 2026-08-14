from __future__ import annotations

import unittest

from recipes.hotpotqa_lr.verifier import (
    artifact_content_sha256,
    trajectory_audit_record,
    verify_trajectory,
)


def _artifact(artifact_id: str, text: str) -> dict[str, dict[str, str]]:
    return {
        artifact_id: {
            "artifact_id": artifact_id,
            "text": text,
            "content_sha256": artifact_content_sha256(text),
        }
    }


class LocalReasoningVerifierTest(unittest.TestCase):
    def test_claim_source_reason_step_receives_grounded_credit(self):
        first_text = "Henry Miller married June Miller in 1924."
        second_text = "June Miller was an American writer."
        audits = verify_trajectory(
            [
                {
                    "reason_step": {
                        "claim": "June Miller",
                        "source": "passage:1",
                    },
                    "action_type": "search",
                    "action_value": "June Miller nationality",
                    "available_artifacts": _artifact("passage:1", first_text),
                },
                {
                    "reason_step": {
                        "claim": "June Miller was an American writer.",
                        "source": "passage:2",
                    },
                    "action_type": "finish",
                    "action_value": "American",
                    "available_artifacts": {
                        **_artifact("passage:1", first_text),
                        **_artifact("passage:2", second_text),
                    },
                },
            ],
            question="What was the nationality of Henry Miller's spouse?",
        )
        self.assertEqual([audit.grounding_valid for audit in audits], [1, 1])
        self.assertEqual([audit.inference_valid for audit in audits], [1, 1])
        self.assertEqual([audit.action_coupled for audit in audits], [1, 1])
        self.assertEqual([audit.novelty_valid for audit in audits], [1, 1])
        self.assertEqual(audits[0].canonical_reason_step, {"claim": "June Miller", "source": "passage:1"})
        self.assertAlmostEqual(trajectory_audit_record(audits)["local_reward"], 2 / 3)

    def test_claim_source_repeated_search_query_gets_no_local_credit(self):
        text = "Henry Miller married June Miller in 1924."
        audits = verify_trajectory(
            [
                {
                    "reason_step": {
                        "claim": "June Miller",
                        "source": "passage:1",
                    },
                    "action_type": "search",
                    "action_value": "June Miller nationality",
                    "available_artifacts": _artifact("passage:1", text),
                },
                {
                    "reason_step": {
                        "claim": "June Miller",
                        "source": "passage:1",
                    },
                    "action_type": "search",
                    "action_value": "June Miller nationality",
                    "available_artifacts": _artifact("passage:1", text),
                },
            ],
            question="What was the nationality of Henry Miller's spouse?",
        )
        self.assertEqual([audit.own_valid for audit in audits], [1, 1])
        self.assertEqual([audit.action_coupled for audit in audits], [1, 1])
        self.assertEqual([audit.novelty_valid for audit in audits], [1, 0])
        self.assertIn("query_repeated", audits[1].errors)
        self.assertIn("claim_repeated", audits[1].errors)
        self.assertIn("source_repeated", audits[1].errors)
        self.assertAlmostEqual(audits[0].raw_local_credit, 1 / 3)
        self.assertEqual(audits[1].raw_local_credit, 0.0)
        self.assertAlmostEqual(trajectory_audit_record(audits)["local_reward"], 1 / 3)

    def test_claim_source_repeated_source_gets_no_second_search_credit(self):
        text = "June Miller was an American writer and a photographer."
        audits = verify_trajectory(
            [
                {
                    "reason_step": {
                        "claim": "June Miller was an American writer",
                        "source": "passage:2",
                    },
                    "action_type": "search",
                    "action_value": "June Miller was an American writer",
                    "available_artifacts": _artifact("passage:2", text),
                },
                {
                    "reason_step": {
                        "claim": "June Miller was an American writer and a photographer",
                        "source": "passage:2",
                    },
                    "action_type": "search",
                    "action_value": "June Miller was an American writer and a photographer",
                    "available_artifacts": _artifact("passage:2", text),
                },
            ],
            question="What nationality was June Miller?",
        )
        self.assertEqual([audit.own_valid for audit in audits], [1, 1])
        self.assertEqual([audit.action_coupled for audit in audits], [1, 1])
        self.assertEqual([audit.novelty_valid for audit in audits], [1, 0])
        self.assertNotIn("claim_repeated", audits[1].errors)
        self.assertIn("source_repeated", audits[1].errors)
        self.assertAlmostEqual(audits[0].raw_local_credit, 1 / 3)
        self.assertEqual(audits[1].raw_local_credit, 0.0)
        self.assertAlmostEqual(trajectory_audit_record(audits)["local_reward"], 1 / 3)

    def test_claim_source_finish_can_reuse_search_evidence(self):
        text = "June Miller was an American writer and a photographer."
        audits = verify_trajectory(
            [
                {
                    "reason_step": {
                        "claim": "June Miller was an American writer",
                        "source": "passage:2",
                    },
                    "action_type": "search",
                    "action_value": "June Miller was an American writer",
                    "available_artifacts": _artifact("passage:2", text),
                },
                {
                    "reason_step": {
                        "claim": "June Miller was an American writer",
                        "source": "passage:2",
                    },
                    "action_type": "finish",
                    "action_value": "American",
                    "available_artifacts": _artifact("passage:2", text),
                },
            ],
            question="What nationality was June Miller?",
        )
        self.assertEqual([audit.own_valid for audit in audits], [1, 1])
        self.assertEqual([audit.action_coupled for audit in audits], [1, 1])
        self.assertEqual([audit.novelty_valid for audit in audits], [1, 1])
        self.assertAlmostEqual(trajectory_audit_record(audits)["local_reward"], 2 / 3)

    def test_claim_source_fails_closed_when_source_or_claim_is_invalid(self):
        text = "June Miller was an American writer."
        missing_source = verify_trajectory(
            [
                {
                    "reason_step": {
                        "claim": "June Miller was an American writer.",
                        "source": "passage:missing",
                    },
                    "action_type": "finish",
                    "action_value": "American",
                    "available_artifacts": _artifact("passage:2", text),
                }
            ],
            question="q",
        )[0]
        absent_claim = verify_trajectory(
            [
                {
                    "reason_step": {
                        "claim": "June Miller was a French writer.",
                        "source": "passage:2",
                    },
                    "action_type": "finish",
                    "action_value": "French",
                    "available_artifacts": _artifact("passage:2", text),
                }
            ],
            question="q",
        )[0]
        malformed = verify_trajectory(
            [
                {
                    "reason_step": {"claim": "June Miller was an American writer."},
                    "action_type": "finish",
                    "action_value": "American",
                    "available_artifacts": _artifact("passage:2", text),
                }
            ],
            question="q",
        )[0]
        self.assertEqual(missing_source.own_valid, 0)
        self.assertIn("artifact_unknown:passage:missing", missing_source.errors)
        self.assertEqual(absent_claim.own_valid, 0)
        self.assertIn("claim_not_exact:passage:2", absent_claim.errors)
        self.assertEqual(malformed.own_valid, 0)
        self.assertIn("reason_step_fields_invalid", malformed.errors)
        self.assertIn("source_invalid", malformed.errors)

    def test_valid_two_hop_reward_is_not_terminal_gated(self):
        first_text = "Henry Miller married June Miller in 1924."
        second_text = "June Miller was an American writer."
        audits = verify_trajectory(
            [
                {
                    "reason_step": {
                        "ref": "r1",
                        "op": "extract_bridge_entity",
                        "premises": [
                            {"artifact_id": "passage:1", "span": first_text}
                        ],
                        "inputs": [],
                        "output": {"bridge_entity": "June Miller"},
                    },
                    "action_type": "search",
                    "action_value": "June Miller nationality",
                    "available_artifacts": _artifact("passage:1", first_text),
                },
                {
                    "reason_step": {
                        "ref": "r2",
                        "op": "extract_answer_candidate",
                        "premises": [
                            {"artifact_id": "passage:2", "span": second_text}
                        ],
                        "inputs": ["r1"],
                        "output": {"answer_candidate": "American"},
                    },
                    "action_type": "finish",
                    "action_value": "American",
                    "available_artifacts": {
                        **_artifact("passage:1", first_text),
                        **_artifact("passage:2", second_text),
                    },
                },
            ],
            question="What was the nationality of Henry Miller's spouse?",
        )
        self.assertEqual([audit.grounding_valid for audit in audits], [1, 1])
        self.assertEqual([audit.inference_valid for audit in audits], [1, 1])
        self.assertEqual([audit.action_coupled for audit in audits], [1, 1])
        self.assertAlmostEqual(trajectory_audit_record(audits)["local_reward"], 2 / 3)

    def test_invalid_ancestor_is_tainted_but_independent_branch_recovers(self):
        text = "June Miller was an American writer."
        artifacts = _artifact("passage:2", text)
        audits = verify_trajectory(
            [
                {
                    "reason_step": {
                        "ref": "r1",
                        "op": "extract_answer_candidate",
                        "premises": [{"artifact_id": "passage:2", "span": text}],
                        "inputs": [],
                        "output": {"answer_candidate": "French"},
                    },
                    "action_type": "search",
                    "action_value": "French biography",
                    "available_artifacts": artifacts,
                },
                {
                    "reason_step": {
                        "ref": "r2",
                        "op": "extract_answer_candidate",
                        "premises": [{"artifact_id": "passage:2", "span": text}],
                        "inputs": ["r1"],
                        "output": {"answer_candidate": "American"},
                    },
                    "action_type": "finish",
                    "action_value": "American",
                    "available_artifacts": artifacts,
                },
                {
                    "reason_step": {
                        "ref": "r3",
                        "op": "extract_answer_candidate",
                        "premises": [{"artifact_id": "passage:2", "span": text}],
                        "inputs": [],
                        "output": {"answer_candidate": "American"},
                    },
                    "action_type": "finish",
                    "action_value": "American",
                    "available_artifacts": artifacts,
                },
            ],
            question="What nationality was June Miller?",
        )
        self.assertEqual(audits[0].own_valid, 0)
        self.assertEqual(audits[1].invalid_ancestor_count, 1)
        self.assertAlmostEqual(audits[1].dependency_factor, 0.3)
        self.assertEqual(audits[2].invalid_ancestor_count, 0)
        self.assertTrue(trajectory_audit_record(audits)["self_correction"])

    def test_unknown_artifact_and_sha_mismatch_fail_closed(self):
        raw = {
            "ref": "r1",
            "op": "select_exact_span",
            "premises": [{"artifact_id": "passage:1", "span": "answer"}],
            "inputs": [],
            "output": {"text": "answer"},
        }
        unknown = verify_trajectory(
            [
                {
                    "reason_step": raw,
                    "action_type": "finish",
                    "action_value": "answer",
                    "available_artifacts": {},
                }
            ],
            question="q",
        )[0]
        mismatch = verify_trajectory(
            [
                {
                    "reason_step": raw,
                    "action_type": "finish",
                    "action_value": "answer",
                    "available_artifacts": {
                        "passage:1": {"text": "answer", "content_sha256": "bad"}
                    },
                }
            ],
            question="q",
        )[0]
        self.assertEqual(unknown.grounding_valid, 0)
        self.assertEqual(mismatch.grounding_valid, 0)

    def test_direct_coupling_can_target_a_later_sink(self):
        text = "The bridge entity is June Miller."
        audits = verify_trajectory(
            [
                {
                    "reason_step": {
                        "ref": "r1",
                        "op": "extract_bridge_entity",
                        "premises": [{"artifact_id": "passage:1", "span": text}],
                        "inputs": [],
                        "output": {"bridge_entity": "June Miller"},
                    },
                    "action_type": "search",
                    "action_value": "spouse details",
                    "available_artifacts": _artifact("passage:1", text),
                },
                {
                    "reason_step": None,
                    "action_type": "search",
                    "action_value": "June Miller nationality",
                    "available_artifacts": _artifact("passage:1", text),
                },
            ],
            question="Who was the spouse?",
        )
        self.assertEqual(audits[0].action_coupled, 1)
        self.assertGreater(audits[0].raw_local_credit, 0.0)

    def test_fixed_horizon_zeros_fourth_transition(self):
        text = "alpha"
        transitions = []
        for index in range(1, 5):
            transitions.append(
                {
                    "reason_step": {
                        "ref": f"r{index}",
                        "op": "select_exact_span",
                        "premises": [{"artifact_id": "passage:1", "span": text}],
                        "inputs": [],
                        "output": {"text": text},
                    },
                    "action_type": "search",
                    "action_value": f"alpha {index}",
                    "available_artifacts": _artifact("passage:1", text),
                }
            )
        audits = verify_trajectory(transitions, question="q")
        self.assertAlmostEqual(sum(audit.raw_local_credit for audit in audits), 1.0)
        self.assertEqual(audits[-1].raw_local_credit, 0.0)
        self.assertIn("reward_horizon_exceeded", audits[-1].errors)


if __name__ == "__main__":
    unittest.main()
