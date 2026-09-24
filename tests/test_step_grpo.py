import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from agent_r1.trainer.ppo.core_algos import AgentAdvantageEstimator, compute_step_grpo_advantage
from agent_r1.trainer.ppo.ray_trainer import compute_advantage
from verl import DataProto


def _reward_rows(values: list[float]) -> torch.Tensor:
    rewards = torch.zeros((len(values), 2), dtype=torch.float32)
    rewards[:, 1] = torch.tensor(values, dtype=torch.float32)
    return rewards


class StepGrpoTest(unittest.TestCase):
    def setUp(self):
        self.prompt_groups = np.array(["q", "q", "q", "q", "q", "q"], dtype=object)
        self.trajectory_uids = np.array(["r1", "r1", "r1", "r2", "r2", "r2"], dtype=object)
        self.step_indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        self.response_mask = torch.ones((6, 2), dtype=torch.float32)

    def test_process_reward_affects_generating_and_earlier_not_later_steps(self):
        rewards = _reward_rows([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        advantages, returns = compute_step_grpo_advantage(
            token_level_rewards=rewards,
            response_mask=self.response_mask,
            index=self.prompt_groups,
            trajectory_uids=self.trajectory_uids,
            step_indices=self.step_indices,
            gamma=1.0,
        )

        self.assertTrue(torch.allclose(returns[:, 0], torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])))
        self.assertGreater(float(advantages[0, 0]), 0.0)
        self.assertGreater(float(advantages[1, 0]), 0.0)
        self.assertEqual(float(advantages[2].abs().sum()), 0.0)
        self.assertLess(float(advantages[3, 0]), 0.0)
        self.assertLess(float(advantages[4, 0]), 0.0)
        self.assertEqual(float(advantages[5].abs().sum()), 0.0)

    def test_terminal_reward_propagates_causally_for_a1(self):
        rewards = _reward_rows([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        advantages, returns = compute_step_grpo_advantage(
            token_level_rewards=rewards,
            response_mask=self.response_mask,
            index=self.prompt_groups,
            trajectory_uids=self.trajectory_uids,
            step_indices=self.step_indices,
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
        )
        self.assertTrue(torch.allclose(returns[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(advantages[:, 0], torch.tensor([0.5, 0.5, 0.5, -0.5, -0.5, -0.5])))

    def test_fully_masked_a2_final_reward_cannot_leak_backward(self):
        rewards = _reward_rows([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        response_mask = self.response_mask.clone()
        response_mask[2] = 0.0
        response_mask[5] = 0.0
        advantages, returns = compute_step_grpo_advantage(
            token_level_rewards=rewards,
            response_mask=response_mask,
            index=self.prompt_groups,
            trajectory_uids=self.trajectory_uids,
            step_indices=self.step_indices,
            gamma=1.0,
        )
        self.assertEqual(float(advantages.abs().sum()), 0.0)
        self.assertEqual(float(returns.abs().sum()), 0.0)

    def test_a3_combined_rewards_reach_search_and_final_actions(self):
        # r_process=1.0 is weighted to 0.5 at step 1 and terminal EM=1.0 is
        # weighted to 0.5 at step 2. The final row remains unmasked, so both
        # components enter causal returns while later actions never affect earlier time.
        rewards = _reward_rows([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
        advantages, returns = compute_step_grpo_advantage(
            token_level_rewards=rewards,
            response_mask=self.response_mask,
            index=self.prompt_groups,
            trajectory_uids=self.trajectory_uids,
            step_indices=self.step_indices,
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
        )

        self.assertTrue(
            torch.allclose(
                returns[:, 0],
                torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]),
            )
        )
        self.assertGreater(float(advantages[0, 0]), 0.0)
        self.assertGreater(float(advantages[1, 0]), 0.0)
        self.assertGreater(float(advantages[2, 0]), 0.0)
        self.assertLess(float(advantages[3, 0]), 0.0)
        self.assertLess(float(advantages[4, 0]), 0.0)
        self.assertLess(float(advantages[5, 0]), 0.0)

    def test_singleton_step_keeps_existing_zero_baseline_behavior(self):
        advantages, returns = compute_step_grpo_advantage(
            token_level_rewards=_reward_rows([0.5]),
            response_mask=torch.ones((1, 2), dtype=torch.float32),
            index=np.array(["q"], dtype=object),
            trajectory_uids=np.array(["r"], dtype=object),
            step_indices=np.array([0], dtype=np.int64),
            norm_adv_by_std_in_grpo=False,
        )

        self.assertAlmostEqual(float(returns[0, 0]), 0.5)
        self.assertAlmostEqual(float(advantages[0, 0]), 0.5)

    def test_mixed_reward_is_normalized_once_across_all_rollouts(self):
        advantages, returns = compute_step_grpo_advantage(
            token_level_rewards=_reward_rows([0.5, 0.0, 0.0, 0.0]),
            response_mask=torch.ones((4, 2), dtype=torch.float32),
            index=np.array(["q"] * 4, dtype=object),
            trajectory_uids=np.array(["a", "b", "c", "d"], dtype=object),
            step_indices=np.zeros(4, dtype=np.int64),
            norm_adv_by_std_in_grpo=False,
        )

        self.assertTrue(torch.allclose(returns[:, 0], torch.tensor([0.5, 0.0, 0.0, 0.0])))
        self.assertTrue(
            torch.allclose(advantages[:, 0], torch.tensor([0.375, -0.125, -0.125, -0.125]))
        )

    def test_component_metadata_is_debug_only_in_trainer_wiring(self):
        rewards = _reward_rows([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        data = DataProto(
            batch=TensorDict(
                {
                    "token_level_rewards": rewards,
                    "response_mask": torch.ones((8, 2), dtype=torch.float32),
                },
                batch_size=8,
            ),
            non_tensor_batch={
                "uid": np.array(["q"] * 8, dtype=object),
                "trajectory_uids": np.array(
                    ["a", "a", "b", "b", "c", "c", "d", "d"],
                    dtype=object,
                ),
                "step_indices": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
                "a9_process_component_valid": np.array(
                    ["debug-only"] * 8,
                    dtype=object,
                ),
                "optimizer_process_component": np.array(
                    [999.0] * 8,
                    dtype=object,
                ),
            },
            meta_info={},
        )
        result = compute_advantage(
            data,
            adv_estimator=AgentAdvantageEstimator.GRPO,
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
            config=OmegaConf.create({"grpo": {"credit_assignment": "step_causal"}}),
        )
        expected_advantages, expected_returns = compute_step_grpo_advantage(
            token_level_rewards=rewards,
            response_mask=torch.ones((8, 2), dtype=torch.float32),
            index=np.array(["q"] * 8, dtype=object),
            trajectory_uids=np.array(
                ["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object
            ),
            step_indices=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
        )

        self.assertTrue(torch.equal(result.batch["advantages"], expected_advantages))
        self.assertTrue(torch.equal(result.batch["returns"], expected_returns))

    def test_rejects_misaligned_identity_arrays(self):
        with self.assertRaises(ValueError):
            compute_step_grpo_advantage(
                token_level_rewards=torch.zeros((1, 1)),
                response_mask=torch.ones((1, 1)),
                index=np.array(["q"], dtype=object),
                trajectory_uids=np.array(["r"], dtype=object),
                step_indices=np.array([], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
