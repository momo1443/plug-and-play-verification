from __future__ import annotations

import unittest

from recipes.hotpotqa_lr.protocol import extract_tool_calls, parse_finish
from recipes.hotpotqa_lr.reward_contract import compose_reward
from recipes.hotpotqa_lr.reward_fn import compute_score


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
        self.assertEqual(compose_reward(0.0, 2 / 3), 1 / 3)
        self.assertEqual(compose_reward(1.0, 0.0), 0.5)
        self.assertEqual(compose_reward(1.0, 1.0), 1.0)

    def test_non_finish_completion_scores_zero(self):
        self.assertEqual(
            compute_score("hotpotqa_distractor", "<answer>American</answer>", "American"),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
