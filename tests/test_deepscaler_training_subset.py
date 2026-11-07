import unittest

from recipes.deepscaler.prepare_training_subset import select_prompt_indices


class Tokenizer:
    def apply_chat_template(self, prompt, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": True, "return_dict": False}
        return range(prompt[0]["tokens"])


class TrainingSubsetTest(unittest.TestCase):
    def test_prefix_preserves_source_order_and_labels(self):
        rows = [{"prompt": [{"tokens": n}], "answer": str(i)}
                for i, n in enumerate([2048, 10, 20, 2049])]
        selected, excluded = select_prompt_indices(rows, Tokenizer(), 3, 2048)
        self.assertEqual(selected, [0, 1, 2])
        self.assertEqual(excluded, [])
        self.assertEqual([rows[i]["answer"] for i in selected], ["0", "1", "2"])

    def test_overlong_prefix_row_is_an_error_instead_of_backfilling(self):
        rows = [{"prompt": [{"tokens": n}], "answer": str(i)}
                for i, n in enumerate([2048, 2049, 10, 20, 30])]
        with self.assertRaisesRegex(ValueError, "prefix row 1 has 2049 prompt tokens"):
            select_prompt_indices(rows, Tokenizer(), 3, 2048)

    def test_insufficient_source_rows_cannot_silently_shorten_training(self):
        with self.assertRaisesRegex(ValueError, "Only 1 source rows; need 2"):
            select_prompt_indices([{"prompt": [{"tokens": 10}]}], Tokenizer(), 2, 2048)
