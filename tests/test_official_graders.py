"""Check original grader files and real official scoring at the adapter boundary."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from agent_r1.evaluation.official import WORKER, VENDOR, grader_python, grade_code, sandbox_command
from agent_r1.evaluation.pnp_protocol import mathvision_score, qa_scores
from recipes.livecodebench_a0.evaluate_a0 import grade_stdio, grade_functional

# Load the real legacy scoring module without its package's GPU agent imports.
_reward_spec = importlib.util.spec_from_file_location(
    "legacy_mathvision_reward", Path(__file__).resolve().parents[1] / "recipes/mathvision/reward_fn.py")
_reward_module = importlib.util.module_from_spec(_reward_spec)
_reward_spec.loader.exec_module(_reward_module)
compute_score = _reward_module.compute_score


class OfficialGradersTest(unittest.TestCase):
    def trusted_code_fixture(self, code, tests, function_name=None):
        # Only these test-authored fixtures execute here. Model submissions use
        # grade_code's bubblewrap process, never this direct fixture command.
        payload = {"code": code, "tests": tests, "function_name": function_name, "timeout_seconds": 1}
        completed = subprocess.run([grader_python(), "-I", str(WORKER), "livecodebench", "--once"],
                                   input=json.dumps(payload) + "\n", capture_output=True, text=True,
                                   check=True, timeout=30)
        reply = json.loads(completed.stdout)
        self.assertNotIn("error", reply)
        return reply["result"]

    def test_upstream_sources_and_licenses_match_recorded_checksums(self):
        for benchmark, identity in json.loads((VENDOR / "manifest.json").read_text()).items():
            self.assertEqual(len(identity["revision"]), 40)
            for relative, expected in identity["files_sha256"].items():
                self.assertEqual(hashlib.sha256((VENDOR / benchmark / relative).read_bytes()).hexdigest(), expected)

    def test_hotpotqa_uses_official_em_f1_and_yes_no_rules(self):
        self.assertEqual(qa_scores("The Eiffel Tower!", ["Eiffel Tower"])["answer_em"], 1)
        self.assertAlmostEqual(qa_scores("Eiffel", ["Eiffel Tower"])["answer_f1"], 2 / 3)
        self.assertEqual(qa_scores("yes", ["yes please"])["answer_f1"], 0)
        self.assertEqual(qa_scores("other", ["wrong", "other"])["answer_em"], 1)
        self.assertFalse(qa_scores(None, ["other"])["is_correct"])

    def test_mathvision_uses_original_symbolic_checker_and_choice_values(self):
        self.assertTrue(mathvision_score(r"\sin(\pi/6)", "0.5", []))
        # Keep upstream extraction semantics, including its indexed-root limitation.
        self.assertFalse(mathvision_score(r"\sqrt[3]{8}", "2", []))
        self.assertTrue(mathvision_score(r"\boxed{\frac{1}{2}}", "B", ["1", "0.5"]))
        self.assertFalse(mathvision_score("A", "B", ["1", "0.5"]))
        self.assertFalse(mathvision_score(None, "B", ["1", "0.5"]))

    def test_legacy_mathvision_validation_delegates_to_same_official_rules(self):
        self.assertEqual(compute_score("mathvision", r"\boxed{\sin(\pi/6)}", "0.5"), 1)
        self.assertEqual(compute_score("mathvision", "0.5", "B", {"options": ["1", "0.5"]}), 1)

    def test_livecodebench_accepts_equivalent_decimal_stdio_output(self):
        result = self.trusted_code_fixture("print('1.000')", [{"input": "", "output": "1"}])
        self.assertTrue(result["passed"])
        self.assertEqual(result["passed_cases"], 1)

    def test_livecodebench_preserves_large_integer_distinctions(self):
        result = self.trusted_code_fixture("print(50000000000000000)",
                                           [{"input": "", "output": "50000000000000001"}])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "wrong_answer")

    def test_livecodebench_functional_tuple_and_runtime_error_rules(self):
        code = "class Solution:\n    def solve(self, a, b):\n        return (a, b)"
        result = self.trusted_code_fixture(code, [{"input": "1\n2", "output": "[1, 2]"}], "solve")
        self.assertTrue(result["passed"])
        result = self.trusted_code_fixture("print(1/0)", [{"input": "", "output": "1"}])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "runtime_error")

    def test_existing_livecodebench_entrypoints_forward_to_the_official_adapter(self):
        tests = [{"input": "1", "output": "2"}]
        expected = {"passed": True, "reason": "accepted", "passed_cases": 1, "total_cases": 1}
        with patch("agent_r1.evaluation.official.grade_code", return_value=expected) as scorer:
            self.assertEqual(grade_stdio("print(2)", tests, 6), expected)
            scorer.assert_called_with("print(2)", tests, None, 6)
            self.assertEqual(grade_functional("code", tests, "solve", 6), expected)
            scorer.assert_called_with("code", tests, "solve", 6)

    def test_submitted_code_sandbox_has_no_project_or_dataset_mount(self):
        with patch("agent_r1.evaluation.official.shutil.which", return_value="bwrap"):
            command = sandbox_command()
        self.assertIn("--unshare-all", command)
        self.assertIn("--clearenv", command)
        self.assertIn("/work/official_worker.py", command)
        self.assertNotIn(str(Path(__file__).resolve().parents[1]), command)
        self.assertNotIn(str(Path(__file__).resolve().parents[1] / "data"), command)

    def test_real_livecodebench_sandbox_when_supported(self):
        try:
            result = grade_code("print('1.000')", [{"input": "", "output": "1"}], None, 1)
        except RuntimeError as exc:
            if "bubblewrap" in str(exc) or "Creating new namespace failed" in str(exc):
                self.skipTest(str(exc))
            raise
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
