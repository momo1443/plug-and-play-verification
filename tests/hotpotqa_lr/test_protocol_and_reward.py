from __future__ import annotations

import os
import subprocess
import sys
import unittest

from recipes.hotpotqa_lr.agent_flow import optimizer_reward_schedule
from recipes.hotpotqa_lr.protocol import extract_tool_calls, parse_finish
from recipes.hotpotqa_lr.reward_contract import (
    LR_CONTRACT_VERSION,
    REWARD_MODE_TERMINAL_ONLY,
    compose_reward,
)
from recipes.hotpotqa_lr.reward_fn import compute_score


class ProtocolAndRewardTest(unittest.TestCase):
    def test_hermes_nested_reason_step_is_parsed(self):
        completion = (
            "<tool_call>\n"
            '<function=finish>\n'
            '<parameter=status>answer</parameter>\n'
            '<parameter=answer>American</parameter>\n'
            '<parameter=reason_step>{"ref":"r1","op":"select_exact_span","premises":[],"inputs":[],"output":{"text":"American"}}</parameter>\n'
            '</function>\n'
            '</tool_call>'
        )
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
        self.assertEqual(
            optimizer_reward_schedule(
                global_step=500,
                is_validation=False,
                em_warmup_steps=0,
                reward_mode=REWARD_MODE_TERMINAL_ONLY,
            ),
            (1.0, 0.0, "terminal_only"),
        )

    def test_lr30_contract_is_current_a8_lr_contract(self):
        self.assertEqual(LR_CONTRACT_VERSION, "a8-lr-dsl-v2")

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
            "a8-lr-claim-source-v2 A8_LR30_CS hotpotqa-local-reasoning-claim-source-v1 claim_source",
        )

    def test_non_finish_completion_scores_zero(self):
        self.assertEqual(
            compute_score("hotpotqa_distractor", "<answer>American</answer>", "American"),
            0.0,
        )

    # -- Validation-time lenient fallback --

    def test_validation_lenient_fallback_answer_tag(self):
        """When finish tool call is missing but <answer> tag is present,
        validation-time scoring should extract the answer leniently."""
        completion = "<answer>American</answer>"
        extra_info = {"_agent_r1_is_validation": True}
        self.assertEqual(
            compute_score("hotpotqa_distractor", completion, "American", extra_info=extra_info),
            1.0,
        )

    def test_validation_lenient_fallback_invalid_envelope(self):
        """When finish tool call has an invalid envelope but <answer> tag
        is present, validation should still score it."""
        # Missing reason_step -> envelope_valid=False
        _TOOL_OPEN = chr(60) + "tool_call" + chr(62)
        _TOOL_CLOSE = chr(60) + "/tool_call" + chr(62)
        completion = (
            f"{_TOOL_OPEN}\n"
            '{"name": "finish", "arguments": {"status": "answer", "answer": "American"}}\n'
            f"{_TOOL_CLOSE}\n"
            "<answer>American</answer>"
        )
        self.assertFalse(parse_finish(completion).envelope_valid)
        extra_info = {"_agent_r1_is_validation": True}
        self.assertEqual(
            compute_score("hotpotqa_distractor", completion, "American", extra_info=extra_info),
            1.0,
        )

    def test_training_stays_strict_without_validation_flag(self):
        """Without _agent_r1_is_validation=True, the strict behaviour is
        preserved: a missing or invalid finish envelope scores 0."""
        completion = "<answer>American</answer>"
        # No extra_info -> training mode
        self.assertEqual(
            compute_score("hotpotqa_distractor", completion, "American"),
            0.0,
        )
        # Explicit is_validation=False
        self.assertEqual(
            compute_score(
                "hotpotqa_distractor",
                completion,
                "American",
                extra_info={"_agent_r1_is_validation": False},
            ),
            0.0,
        )

    def test_validation_prefers_finish_answer_over_tag(self):
        """When a valid finish tool call is present, the answer inside it
        should be used even during validation (not the <answer> tag)."""
        _TOOL_OPEN = chr(60) + "tool_call" + chr(62)
        _TOOL_CLOSE = chr(60) + "/tool_call" + chr(62)
        completion = (
            f"{_TOOL_OPEN}\n"
            '{"name": "finish", "arguments": {"status": "answer", "answer": "American", '
            '"reason_step": {"ref": "r1", "op": "extract_answer_candidate", '
            '"premises": [], "inputs": [], "output": {"answer_candidate": "American"}}}}\n'
            f"{_TOOL_CLOSE}"
        )
        extra_info = {"_agent_r1_is_validation": True}
        self.assertEqual(
            compute_score("hotpotqa_distractor", completion, "American", extra_info=extra_info),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
