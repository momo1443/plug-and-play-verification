"""Evaluation denominator and training-compatible submission boundaries."""
from types import SimpleNamespace
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "aime_agent_eval", Path(__file__).resolve().parents[1] / "scripts/deepscaler/pnp_aime.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
generate_trajectories, summarize_accuracy = module.generate_trajectories, module.summarize_accuracy
selected_aime_integer = module.selected_aime_integer


class CharacterTokenizer:
    eos_token_id = -1

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(item) for item in ids)

    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(message["content"] for message in messages)
        if kwargs.get("enable_thinking", True):
            text += "<think>"
        return self.encode(text)


class ScriptedEngine:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def generate(self, inputs, params, **kwargs):
        self.inputs.extend(inputs)
        response = next(self.responses)
        assert len(response) <= params.max_tokens
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[ord(c) for c in response])])]


class AimeAgentEvalTest(unittest.TestCase):
    def test_compatibility_wrapper_forwards_aime2026_and_frozen_revision(self):
        command = ["pnp_aime.py", "--model", "/model", "--checkpoint", "/checkpoint",
                   "--dataset", "/aime2026/test.parquet", "--output", "/run",
                   "--dataset-name", "MathArena/aime_2026", "--dataset-label", "I + II",
                   "--dataset-revision", "frozen-snapshot", "--gpus", "0"]
        with patch("sys.argv", command), patch("agent_r1.evaluation.pnp.main") as evaluate:
            module.main()
        forwarded = evaluate.call_args.args[0]
        self.assertEqual(forwarded[0], "aime2026")
        self.assertEqual(forwarded[forwarded.index("--dataset-version") + 1], "2026")
        self.assertEqual(forwarded[forwarded.index("--dataset-revision") + 1], "frozen-snapshot")
        self.assertEqual(forwarded[forwarded.index("--gpus") + 1], "0")

    def test_compatibility_wrapper_rejects_another_year_before_evaluation(self):
        command = ["pnp_aime.py", "--model", "/model", "--checkpoint", "/checkpoint",
                   "--dataset", "/dataset", "--output", "/run", "--dataset-name", "MathArena/aime_2024",
                   "--dataset-label", "AIME 2024", "--dataset-revision", "frozen-snapshot"]
        with patch("sys.argv", command), patch("sys.stderr", new=io.StringIO()), \
             patch("agent_r1.evaluation.pnp.main") as evaluate:
            with self.assertRaises(SystemExit) as error:
                module.main()
        self.assertEqual(error.exception.code, 2)
        evaluate.assert_not_called()

    def test_whole_numeric_answer_is_normalized_without_consulting_gold(self):
        for answer, expected in (("252+25", 277), (r"\frac{554}{2}", 277), ("007", 7),
                                 ("178 - 146", 32), ("x + 277", None), ("277 or 278", None),
                                 ("100, 110, ..., 999", None), ("2020", None),
                                 ("2.5", None), (None, None)):
            with self.subTest(answer=answer):
                self.assertEqual(selected_aime_integer(answer), expected)

    def test_missing_invalid_and_wrong_submissions_stay_in_denominator(self):
        records = [
            {"problem_idx": 1, "is_correct": True, "predicted_integer": 7},
            {"problem_idx": 2, "is_correct": False, "predicted_integer": None},
            {"problem_idx": 3, "is_correct": False, "predicted_integer": 3},
            {"problem_idx": 4, "is_correct": False, "predicted_integer": None},
        ]
        summary = summarize_accuracy(records, [1, 2, 3, 4])
        self.assertEqual(summary["accuracy"], 0.25)
        self.assertEqual(summary["missing_or_invalid_submissions"], 2)

    def test_incomplete_or_duplicate_coverage_cannot_publish_accuracy(self):
        for records in ([], [{"problem_idx": 1}, {"problem_idx": 1}]):
            with self.assertRaises(ValueError):
                summarize_accuracy(records, [1, 2])

    def test_restarted_final_box_uses_training_selector_and_discards_suffix(self):
        tokenizer = CharacterTokenizer()
        engine = ScriptedEngine(["Working."] * 4 + [r"\boxed{007} ignored suffix"])
        state = generate_trajectories(engine, tokenizer, SimpleNamespace, ["Solve this problem"])[0]
        self.assertEqual(state["selected_answer"], "007")
        self.assertEqual(state["termination_reason"], "final_answer")
        self.assertEqual(len(state["steps"]), 5)
        self.assertGreater(state["discarded_tokens"], 0)
        self.assertLessEqual(state["response_tokens"], 4096)
        self.assertNotIn("007", tokenizer.decode(engine.inputs[-1]["prompt_token_ids"]))

    def test_unsubmitted_agent_attempt_is_retained_after_all_turns(self):
        state = generate_trajectories(ScriptedEngine(["Working."] * 5), CharacterTokenizer(),
                                      SimpleNamespace, ["Solve this problem"])[0]
        self.assertIsNone(state["selected_answer"])
        self.assertEqual(state["termination_reason"], "max_steps")
        self.assertEqual(len(state["steps"]), 5)


if __name__ == "__main__":
    unittest.main()
