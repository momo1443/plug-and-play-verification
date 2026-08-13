import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from agent_r1.trainer.ppo.ray_trainer import RayAgentTrainer, fail_on_judge_invalid
from verl import DataProto


class _Tokenizer:
    def batch_decode(self, rows, *, skip_special_tokens):
        del skip_special_tokens
        return [str(int(row[0])) for row in rows]


class RolloutDumpPaddingTest(unittest.TestCase):
    def test_judge_invalid_aborts_the_whole_update(self):
        batch = DataProto(
            batch=TensorDict(
                {"prompts": torch.tensor([[1], [2]])},
                batch_size=2,
            ),
            non_tensor_batch={
                "judge_invalid": np.array([False, True], dtype=object),
                "trajectory_uids": np.array(["ok", "bad"], dtype=object),
            },
            meta_info={},
        )
        with self.assertRaisesRegex(RuntimeError, "bad"):
            fail_on_judge_invalid(batch)

    def test_all_valid_judge_records_pass(self):
        batch = DataProto(
            batch=TensorDict(
                {"prompts": torch.tensor([[1], [2]])},
                batch_size=2,
            ),
            non_tensor_batch={
                "judge_invalid": np.array([False, False], dtype=object),
            },
            meta_info={},
        )
        fail_on_judge_invalid(batch)

    def test_masked_world_size_padding_is_not_persisted(self):
        batch = DataProto(
            batch=TensorDict(
                {
                    "prompts": torch.tensor([[11], [22], [11]]),
                    "responses": torch.tensor([[111], [222], [111]]),
                    "token_level_scores": torch.tensor([[0.5], [1.0], [0.5]]),
                    "sample_mask": torch.tensor([True, True, False]),
                },
                batch_size=3,
            ),
            non_tensor_batch={
                "reward_model": np.array(
                    [
                        {"ground_truth": "a"},
                        {"ground_truth": "b"},
                        {"ground_truth": "a"},
                    ],
                    dtype=object,
                ),
                "trajectory_uids": np.array(["uid-a", "uid-b", "uid-a"], dtype=object),
                "step_indices": np.array([0, 0, 0]),
                "acc": np.array([1.0, 0.0, 1.0]),
            },
            meta_info={},
        )
        trainer = object.__new__(RayAgentTrainer)
        trainer.tokenizer = _Tokenizer()
        trainer.global_steps = 7

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"
            trainer._log_rollout_data(
                batch,
                reward_extra_infos_dict={"acc": [1.0, 0.0, 1.0]},
                timing_raw={},
                rollout_data_file=str(output_path),
            )
            records = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(len(records), 2)
        self.assertEqual([record["trajectory_uid"] for record in records], ["uid-a", "uid-b"])
        self.assertEqual([record["num_steps"] for record in records], [1, 1])
        self.assertEqual([record["score"] for record in records], [0.5, 1.0])
        self.assertEqual([record["steps"][0]["acc"] for record in records], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
