import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recipes.deepscaler.prepare_data import prepare_data


class DeepScaleRPrepareDataTest(unittest.TestCase):
    def test_duplicate_questions_stay_together_and_references_are_not_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(json.dumps([
                {"problem": "Question A", "answer": "private answer A", "solution": "private solution"},
                {"problem": "Question A", "answer": "alternate private answer"},
                {"problem": "Question B", "answer": "private answer B"},
                {"problem": "Question C", "answer": "private answer C"},
                {"problem": "Question D", "answer": ""},
            ]))
            first = prepare_data(source, root / "first", validation_size=1)
            second = prepare_data(source, root / "second", validation_size=1)
            self.assertEqual(first, second)
            train = pd.read_parquet(root / "first/train.parquet")
            validation = pd.read_parquet(root / "first/validation.parquet")
            prompts = lambda frame: {row[0]["content"] for row in frame.prompt}
            self.assertFalse(prompts(train) & prompts(validation))
            self.assertEqual(len(train) + len(validation), 4)
            self.assertEqual(first["excluded_missing_problem_or_answer_indices"], [4])
            self.assertEqual(first["conflicting_answer_groups"], 1)
            for frame in (train, validation):
                for prompt in frame.prompt:
                    self.assertNotIn("private", prompt[0]["content"])
                self.assertTrue(all(row["ground_truth"] for row in frame.reward_model))

    def test_single_question_group_cannot_leave_training_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(json.dumps([
                {"problem": "Same question", "answer": "1"},
                {"problem": "Same question", "answer": "2"},
                {"problem": "Bad row", "answer": ""},
            ]))
            with self.assertRaisesRegex(ValueError, "leave no training rows"):
                prepare_data(source, root / "output", validation_size=1)


if __name__ == "__main__":
    unittest.main()
