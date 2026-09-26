"""Contract tests for the domain-agnostic verifier reward adapter."""

from __future__ import annotations

import unittest

from types import SimpleNamespace as AgentFlowStep
from agent_r1.verifier import (
    VerificationCredit,
    VerificationResult,
    apply_composed_reward,
    compose_verification_reward,
    uniform_reward_schedule,
)


class VerifierRewardTest(unittest.TestCase):
    def test_schedule_keeps_warmup_and_group_shared_uniform_weight(self) -> None:
        kwargs = {
            "is_validation": False,
            "warmup_steps": 50,
            "prompt_group_key": "train:7",
            "namespace": b"test-verifier",
            "warmup_phase": "warmup",
            "validation_phase": "validation",
            "mixed_phase": "mixed",
        }
        warmup = uniform_reward_schedule(global_step=50, **kwargs)
        first = uniform_reward_schedule(global_step=51, **kwargs)
        second = uniform_reward_schedule(global_step=51, **kwargs)

        self.assertEqual((warmup.terminal_weight, warmup.process_weight), (1.0, 0.0))
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.terminal_weight + first.process_weight, 1.0)
        self.assertEqual(first.weight_sampling, "uniform_0_1_per_prompt_group")

    def test_adapter_places_credits_and_terminal_reward(self) -> None:
        schedule = uniform_reward_schedule(
            global_step=51,
            is_validation=False,
            warmup_steps=50,
            prompt_group_key="train:7",
            namespace=b"test-verifier",
            warmup_phase="warmup",
            validation_phase="validation",
            mixed_phase="mixed",
        )
        verification = VerificationResult(
            credits=(
                VerificationCredit(1, 0.25, {"kind": "first"}),
                VerificationCredit(2, 0.75, {"kind": "second"}),
            ),
            audit={
                "verifier": "unit-test",
                "credits_by_step": {
                    1: [{"kind": "first"}],
                    2: [{"kind": "second"}],
                },
            },
        )
        composed = compose_verification_reward(
            terminal_reward=1.0,
            verification=verification,
            schedule=schedule,
        )
        steps = [
            AgentFlowStep(prompt_ids=[1], response_ids=[1], reward_score=0.0, extra_fields={}),
            AgentFlowStep(prompt_ids=[2], response_ids=[2], reward_score=0.0, extra_fields={}),
        ]
        apply_composed_reward(steps, composed)

        self.assertAlmostEqual(steps[0].reward_score, schedule.process_weight * 0.25)
        self.assertAlmostEqual(
            steps[1].reward_score,
            schedule.terminal_weight + schedule.process_weight * 0.75,
        )
        self.assertAlmostEqual(sum(step.reward_score for step in steps), composed.score)
        self.assertEqual(
            steps[1].extra_fields["reward_extra_info"]["optimizer_total_reward"],
            composed.score,
        )

    def test_result_rejects_unbounded_total_credit(self) -> None:
        with self.assertRaises(ValueError):
            VerificationResult(
                credits=(
                    VerificationCredit(1, 0.6, {}),
                    VerificationCredit(2, 0.6, {}),
                ),
                audit={},
            )


if __name__ == "__main__":
    unittest.main()
