import json
import tempfile
import unittest
from pathlib import Path

from agent_r1.trainer.streaming_jsonl import StreamingJsonlWriter


def make_record(sample_index: int) -> dict:
    return {
        "sample_index": sample_index,
        "sample_key": f"validation:{sample_index}",
        "question": f"question-{sample_index}",
        "search_queries": [f"query-{sample_index}"],
        "answer": f"answer-{sample_index}",
        "ground_truth": f"answer-{sample_index}",
        "score": float(sample_index % 2),
    }


class StreamingJsonlWriterTest(unittest.TestCase):
    def test_appends_one_valid_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "0.jsonl"
            with StreamingJsonlWriter(path, resume=False) as writer:
                self.assertTrue(writer.append(make_record(0)))
                self.assertTrue(writer.append(make_record(1)))
                self.assertFalse(writer.append(make_record(1)))
                self.assertEqual(writer.count, 2)

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["sample_index"] for record in records], [0, 1])
            self.assertEqual(records[0]["sample_key"], "validation:0")

    def test_existing_file_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "0.jsonl"
            with StreamingJsonlWriter(path, resume=False) as writer:
                writer.append(make_record(0))

            with self.assertRaises(FileExistsError):
                StreamingJsonlWriter(path, resume=False)

    def test_resume_loads_ids_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "0.jsonl"
            with StreamingJsonlWriter(path, resume=False) as writer:
                writer.append(make_record(0))

            with StreamingJsonlWriter(path, resume=True) as writer:
                self.assertEqual(writer.count, 1)
                self.assertFalse(writer.append(make_record(0)))
                self.assertTrue(writer.append(make_record(1)))

            self.assertEqual(len(path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
