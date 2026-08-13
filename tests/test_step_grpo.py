import math
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto

from agent_r1.trainer.ppo.core_algos import AgentAdvantageEstimator, compute_step_grpo_advantage
from agent_r1.trainer.ppo.ray_trainer import compute_advantage


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

    def test_a9_unknown_process_is_masked_but_valid_zero_is_negative_evidence(self):
        # Four trajectories, each with search step 0 and finish step 1:
        # A=valid/full (0.5 weighted), B=valid/fail (0),
        # C=invalid/unknown (MASK), D=valid/partial and therefore also MASK.
        prompt_groups = np.array(["q"] * 8, dtype=object)
        trajectory_uids = np.array(
            ["a", "a", "b", "b", "c", "c", "d", "d"],
            dtype=object,
        )
        step_indices = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        response_mask = torch.ones((8, 2), dtype=torch.float32)
        rewards = _reward_rows([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        process_rewards = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        process_mask = torch.tensor([True, False, True, False, False, False, False, False])

        advantages, returns = compute_step_grpo_advantage(
            token_level_rewards=rewards,
            response_mask=response_mask,
            index=prompt_groups,
            trajectory_uids=trajectory_uids,
            step_indices=step_indices,
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
            process_component_rewards=process_rewards,
            process_component_mask=process_mask,
        )

        self.assertTrue(torch.allclose(returns[:, 0], torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
        self.assertAlmostEqual(float(advantages[0, 0]), 0.25)
        self.assertAlmostEqual(float(advantages[2, 0]), -0.25)
        self.assertAlmostEqual(float(advantages[4, 0]), 0.0)
        self.assertAlmostEqual(float(advantages[6, 0]), 0.0)
        self.assertEqual(float(advantages[[1, 3, 5, 7]].abs().sum()), 0.0)

    def test_a9_invalid_process_trajectory_still_trains_on_terminal_em(self):
        prompt_groups = np.array(["q"] * 8, dtype=object)
        trajectory_uids = np.array(
            ["a", "a", "b", "b", "c", "c", "d", "d"],
            dtype=object,
        )
        step_indices = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        response_mask = torch.ones((8, 2), dtype=torch.float32)
        # C has unknown local process but terminal EM=1 -> weighted terminal 0.5.
        rewards = _reward_rows([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        process_rewards = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        process_mask = torch.tensor([True, False, True, False, False, False, False, False])

        advantages, returns = compute_step_grpo_advantage(
            token_level_rewards=rewards,
            response_mask=response_mask,
            index=prompt_groups,
            trajectory_uids=trajectory_uids,
            step_indices=step_indices,
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
            process_component_rewards=process_rewards,
            process_component_mask=process_mask,
        )

        self.assertAlmostEqual(float(returns[4, 0]), 0.5)
        self.assertAlmostEqual(float(returns[5, 0]), 0.5)
        self.assertGreater(float(advantages[4, 0]), 0.0)
        self.assertGreater(float(advantages[5, 0]), 0.0)

    def test_a9_masked_rows_do_not_enter_process_std(self):
        advantages, _ = compute_step_grpo_advantage(
            token_level_rewards=_reward_rows([0.5, 0.0, 0.0, 0.0]),
            response_mask=torch.ones((4, 2), dtype=torch.float32),
            index=np.array(["q"] * 4, dtype=object),
            trajectory_uids=np.array(["a", "b", "c", "d"], dtype=object),
            step_indices=np.zeros(4, dtype=np.int64),
            process_component_rewards=torch.tensor([0.5, 0.0, 0.0, 0.0]),
            process_component_mask=torch.tensor([True, True, False, False]),
        )

        expected = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(float(advantages[0, 0]), expected, places=5)
        self.assertAlmostEqual(float(advantages[1, 0]), -expected, places=5)
        self.assertEqual(float(advantages[2].abs().sum()), 0.0)
        self.assertEqual(float(advantages[3].abs().sum()), 0.0)

    def test_a9_mask_requires_zero_numeric_placeholder(self):
        with self.assertRaisesRegex(ValueError, "masked process components"):
            compute_step_grpo_advantage(
                token_level_rewards=torch.zeros((1, 2)),
                response_mask=torch.ones((1, 2)),
                index=np.array(["q"], dtype=object),
                trajectory_uids=np.array(["a"], dtype=object),
                step_indices=np.array([0], dtype=np.int64),
                process_component_rewards=torch.tensor([0.5]),
                process_component_mask=torch.tensor([False]),
            )

    def test_a9_component_mask_survives_dataproto_trainer_wiring(self):
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
                    [np.bool_(True), None, True, None, False, None, False, None],
                    dtype=object,
                ),
                "optimizer_process_component": np.array(
                    [0.5, None, 0.0, None, 0.0, None, 0.0, None],
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

        self.assertGreater(float(result.batch["advantages"][4, 0]), 0.0)
        self.assertGreater(float(result.batch["advantages"][5, 0]), 0.0)
        self.assertAlmostEqual(float(result.batch["returns"][4, 0]), 0.5)
        self.assertAlmostEqual(float(result.batch["returns"][5, 0]), 0.5)

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
