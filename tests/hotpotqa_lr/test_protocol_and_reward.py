from __future__ import annotations

import unittest
import os
import subprocess
import sys

from recipes.hotpotqa_lr.protocol import extract_tool_calls, parse_finish
from recipes.hotpotqa_lr.reward_contract import LR_CONTRACT_VERSION, compose_reward
from recipes.hotpotqa_lr.reward_fn import compute_score
from recipes.hotpotqa_lr.agent_flow import optimizer_reward_schedule


class ProtocolAndRewardTest(unittest.TestCase):
    def test_hermes_nested_reason_step_is_parsed(self):
        completion = """<tool_call>
<function=finish>
<parameter=status>answer</parameter>
<parameter=answer>American</parameter>
<parameter=reason_step>{"ref":"r1","op":"select_exact_span","premises":[],"inputs":[],"output":{"text":"American"}}</parameter>
</function>
</tool_call>"""
        calls = extract_tool_calls(completion)
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0]["arguments"]["reason_step"], dict)
        self.assertTrue(parse_finish(completion).envelope_valid)
        self.assertEqual(
            compute_score("hotpotqa_distractor", completion, "American"),
            1.0,
        )

    def test_terminal_and_local_rewards_are_independent(self):
        self.assertAlmostEqual(compose_reward(0.0, 2 / 3), 0.2)
        self.assertEqual(compose_reward(1.0, 0.0), 0.7)
        self.assertEqual(compose_reward(1.0, 1.0), 1.0)

    def test_em_warmup_reward_schedule(self):
        self.assertEqual(
            optimizer_reward_schedule(global_step=1, is_validation=False, em_warmup_steps=100),
            (1.0, 0.0, "em_warmup"),
        )
        self.assertEqual(
            optimizer_reward_schedule(global_step=100, is_validation=False, em_warmup_steps=100),
            (1.0, 0.0, "em_warmup"),
        )
        self.assertEqual(
            optimizer_reward_schedule(global_step=101, is_validation=False, em_warmup_steps=100),
            (0.7, 0.3, "lr30"),
        )
        self.assertEqual(
            optimizer_reward_schedule(global_step=50, is_validation=True, em_warmup_steps=100),
            (1.0, 0.0, "validation_terminal_em"),
        )

    def test_lr30_contract_is_current_a8_lr_contract(self):
        self.assertEqual(LR_CONTRACT_VERSION, "a8-lr-30-v1")

    def test_claim_source_contract_is_opt_in(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(__import__("pathlib").Path(__file__).resolve().parents[2])
        env["HOTPOTQA_LR_REASON_STEP_FORMAT"] = "claim_source"
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "from recipes.hotpotqa_lr.reward_contract import "
                    "LR_CONTRACT_VERSION,LR_REWARD_ARM;"
                    "from recipes.hotpotqa_lr.dsl import DSL_VERSION,REASON_STEP_FORMAT;"
                    "print(LR_CONTRACT_VERSION, LR_REWARD_ARM, DSL_VERSION, REASON_STEP_FORMAT)"
                ),
            ],
            env=env,
            text=True,
        ).strip()
        self.assertEqual(
            output,
            "a8-lr-30-claim-source-v2 A8_LR30_CS hotpotqa-local-reasoning-claim-source-v1 claim_source",
        )

    def test_non_finish_completion_scores_zero(self):
        self.assertEqual(
            compute_score("hotpotqa_distractor", "<answer>American</answer>", "American"),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
