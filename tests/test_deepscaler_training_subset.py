import unittest

from recipes.deepscaler.prepare_training_subset import select_prompt_indices


class Tokenizer:
    def apply_chat_template(self, prompt, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": True, "return_dict": False}
        return range(prompt[0]["tokens"])


class TrainingSubsetTest(unittest.TestCase):
    def test_backfills_overlong_rows_without_reordering_or_changing_labels(self):
        rows = [{"prompt": [{"tokens": n}], "answer": str(i)}
                for i, n in enumerate([2048, 2049, 10, 20, 30])]
        selected, excluded = select_prompt_indices(rows, Tokenizer(), 3, 2048)
        self.assertEqual(selected, [0, 2, 3])
        self.assertEqual(excluded, [{"row": 1, "tokens": 2049}])
        self.assertEqual([rows[i]["answer"] for i in selected], ["0", "2", "3"])

    def test_insufficient_eligible_rows_cannot_silently_shorten_training(self):
        with self.assertRaisesRegex(ValueError, "Only 1 eligible rows; need 2"):
            select_prompt_indices([{"prompt": [{"tokens": 10}]}], Tokenizer(), 2, 2048)
