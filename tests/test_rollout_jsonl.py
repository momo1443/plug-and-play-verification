import json
import tempfile
import unittest
from pathlib import Path

from agent_r1.trainer.rollout_jsonl import append_jsonl_records


class RolloutJsonlTest(unittest.TestCase):
    def test_multiple_steps_append_to_one_jsonl_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"

            self.assertEqual(
                append_jsonl_records(output_path, [{"global_step": 1, "response": "第一步"}], fsync=False),
                1,
            )
            self.assertEqual(
                append_jsonl_records(
                    output_path,
                    [
                        {"global_step": 2, "response": "second-a"},
                        {"global_step": 2, "response": "second-b"},
                    ],
                    fsync=False,
                ),
                2,
            )

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["global_step"] for record in records], [1, 2, 2])
            self.assertEqual([path.name for path in output_path.parent.glob("*.jsonl")], ["rollouts.jsonl"])


if __name__ == "__main__":
    unittest.main()
