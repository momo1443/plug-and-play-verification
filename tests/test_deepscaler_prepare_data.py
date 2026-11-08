import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recipes.deepscaler.prepare_data import prepare_data


class DeepScaleRPrepareDataTest(unittest.TestCase):
    def test_source_prefix_keeps_duplicates_and_labels_in_original_order(self):
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
            first = prepare_data(source, root / "first", validation_size=1, train_samples=3)
            train = pd.read_parquet(root / "first/train.parquet")
            validation = pd.read_parquet(root / "first/validation.parquet")
            self.assertEqual([row["index"] for row in train.extra_info], [0, 1, 2])
            self.assertEqual([row["index"] for row in validation.extra_info], [3])
            self.assertEqual([row["ground_truth"] for row in train.reward_model],
                             ["private answer A", "alternate private answer", "private answer B"])
            self.assertEqual(first["train_source_indices"], [0, 1, 2])
            self.assertEqual(first["rows_filtered"], 0)
            self.assertFalse(first["labels_changed"])
            for frame in (train, validation):
                for prompt in frame.prompt:
                    self.assertNotIn("private", prompt[0]["content"])
                self.assertTrue(all(row["ground_truth"] for row in frame.reward_model))

    def test_missing_answer_in_prefix_is_an_error_not_a_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(json.dumps([
                {"problem": "First", "answer": "1"},
                {"problem": "Missing answer", "answer": ""},
                {"problem": "Later valid row", "answer": "2"},
            ]))
            with self.assertRaisesRegex(ValueError, "Source row 1.*cannot be skipped or replaced"):
                prepare_data(source, root / "output", validation_size=0, train_samples=2)
            self.assertFalse((root / "output/train.parquet").exists())

    def test_default_selects_exactly_first_ten_thousand_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(json.dumps([{"problem": f"Q{i}", "answer": str(i)} for i in range(10002)]))
            manifest = prepare_data(source, root / "output", validation_size=2)
            train = pd.read_parquet(root / "output/train.parquet")
            self.assertEqual(len(train), 10000)
            self.assertEqual([row["index"] for row in train.extra_info], list(range(10000)))
            self.assertEqual(manifest["validation_source_indices"], [10000, 10001])

    def test_short_source_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text('[{"problem":"Q","answer":"1"}]')
            with self.assertRaisesRegex(ValueError, "Need 2 source rows"):
                prepare_data(source, root / "output", validation_size=0, train_samples=2)


if __name__ == "__main__":
    unittest.main()
