"""Equation (9) over standalone evaluation records and independent final grades."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from agent_r1.evaluation.consistency import consistency_record, finalize_evaluation_record
from agent_r1.evaluation.pnp import math_records
from agent_r1.evaluation.pnp_datasets import EvaluationTask
from agent_r1.evaluation.pnp_protocol import PROFILES, summarize_records
from agent_r1.evaluation.summarize import summarize
from agent_r1.verifier import VerificationResult
from test_aime_agent_eval import CharacterTokenizer, ScriptedEngine


def record(question_id, checks, correct, eligible=True):
    verification = consistency_record(
        VerificationResult((), {"applicable_checks": checks}), 0.0, eligible=eligible,
    )
    return finalize_evaluation_record({
        "question_id": question_id, "is_correct": correct, "terminal_eligible": eligible,
        "termination_reason": "final_submission" if eligible else "token_budget",
        "generated_output_tokens": 10, "steps": [{"response": "recorded policy action"}],
        "verification": {**verification, "verified_process_reward": 1.0},
    })


class EvaluationConsistencyTest(unittest.TestCase):
    def test_independent_final_grade_replaces_flow_placeholder_outcome(self):
        row = record("q1", [1, 1], True)
        self.assertEqual(row["verified_success"], 1)
        self.assertEqual(row["verification"]["verified_success"], 1)
        self.assertEqual(summarize([row])["M_d"], 1)

    def test_correct_repaired_and_unsubmitted_attempts_share_one_denominator(self):
        rows = [record("q1", [1], True), record("q2", [0, 1], True),
                record("q3", [], False), record("q4", [1], False, eligible=False)]
        result = summarize_records(rows, [row["question_id"] for row in rows], PROFILES["aime2025"])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["C_d_mean"], 0.5)
        self.assertEqual(result["M_d"], 0.25)
        self.assertEqual(result["verified_success_count"], 1)

    def test_checks_are_required_even_when_process_reward_is_one(self):
        row = record("q1", [1], True)
        row["verification"].pop("verification_checks")
        with self.assertRaises(KeyError):
            finalize_evaluation_record(row)
        row.pop("verification_protocol")
        with self.assertRaises(ValueError):
            summarize_records([row], ["q1"], PROFILES["aime2025"])

    def test_invalid_success_and_stale_joint_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            record("q1", [1], True, eligible=False)
        row = record("q1", [0, 1], True)
        row["verified_success"] = 1
        with self.assertRaises(ValueError):
            summarize_records([row], ["q1"], PROFILES["aime2025"])

    def test_math_evaluation_exports_checks_and_joint_metrics(self):
        task = EvaluationTask("q1", {"problem": "Compute 1+1."}, {"answer": "2"})
        engine = ScriptedEngine(["1+1=2."] * 4 + ["2}"])
        rows = math_records(engine, CharacterTokenizer(), SimpleNamespace, [task], PROFILES["aime2025"])
        self.assertTrue(rows[0]["is_correct"])
        self.assertEqual(rows[0]["verified_success"], 1)
        self.assertEqual(summarize_records(rows, ["q1"], PROFILES["aime2025"])["M_d"], 1)

    def test_truncated_math_tail_uses_the_training_check_coverage(self):
        # The incomplete final equality is excluded; the earlier complete
        # equality is retained, exactly as in the training verifier call.
        text = "1+1=2.\n\n2+3=50"
        task = EvaluationTask("q1", {"problem": "Compute 1+1."}, {"answer": "2"})
        profile = replace(PROFILES["aime2025"], max_steps=1,
                          total_output_tokens=len(text), per_step_tokens=len(text))
        rows = math_records(ScriptedEngine([text]), CharacterTokenizer(), SimpleNamespace, [task], profile)
        self.assertTrue(rows[0]["steps"][0]["generation_truncated"])
        self.assertEqual(rows[0]["verification_checks"], [1.0])
        self.assertEqual(rows[0]["verification_consistency"], 1)
        self.assertEqual(rows[0]["verified_success"], 0)


if __name__ == "__main__":
    unittest.main()
