from __future__ import annotations

import unittest

from recipes.deepscaler import reward_fn


class DeepScalerRewardTest(unittest.TestCase):
    def test_every_step_contributes_to_process_reward(self) -> None:
        solution = """Step 1: Check 2 + 3 = 5 and 4 + 4 = 9.
Step 2: Continue with 6 / 2 = 3."""

        score, fully_verified_steps, step_count = reward_fn.compute_process_reward(solution)

        self.assertAlmostEqual(score, 0.75)
        self.assertEqual(fully_verified_steps, 1)
        self.assertEqual(step_count, 2)

    def test_step_without_numeric_equation_is_excluded_from_average(self) -> None:
        solution = """Step 1: This paragraph gives an unsupported verbal conclusion.
Step 2: Check the arithmetic $8 - 3 = 5$."""

        score, fully_verified_steps, step_count = reward_fn.compute_process_reward(solution)

        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(fully_verified_steps, 1)
        self.assertEqual(step_count, 1)

    def test_trivial_literal_identity_does_not_earn_credit(self) -> None:
        self.assertEqual(reward_fn.compute_process_reward("1 = 1"), (0.0, 0, 1))

    def test_binomial_coefficient_uses_combinatorial_semantics(self) -> None:
        self.assertTrue(reward_fn._verify_equation_pair(r"\binom{10}{2}", "45"))
        self.assertFalse(reward_fn._verify_equation_pair(r"\binom{10}{2}", "20"))

    def test_unsupported_repeating_decimal_fails_closed(self) -> None:
        self.assertFalse(reward_fn._verify_equation_pair(r"0.\overline{6}", r"\frac{2}{3}"))

    def test_post_warmup_reward_uses_uniform_weight(self) -> None:
        solution = """Step 1: Check the arithmetic $2 + 3 = 5$.
Step 2: Therefore the final answer is $\boxed{0}$."""

        extra_info = {
            "_agent_r1_global_step": reward_fn.EM_WARMUP_STEPS + 1,
            "index": 17,
            "split": "train",
        }
        result = reward_fn.compute_score("deepmath", solution, "999", extra_info)
        same_group = reward_fn.compute_score("deepmath", solution, "999", extra_info)
        different_group = reward_fn.compute_score(
            "deepmath",
            solution,
            "999",
            {**extra_info, "index": 18},
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["optimizer_reward_phase"], "uniform")
        self.assertEqual(result["weight_sampling"], "uniform_0_1_per_prompt_group")
        self.assertEqual(result["optimizer_weight_group_key"], "deepmath:train:index:17")
        self.assertEqual(result["optimizer_terminal_weight"], same_group["optimizer_terminal_weight"])
        self.assertNotEqual(result["optimizer_terminal_weight"], different_group["optimizer_terminal_weight"])
        self.assertAlmostEqual(
            result["optimizer_terminal_weight"] + result["optimizer_process_weight"],
            1.0,
        )
        self.assertAlmostEqual(result["process_reward"], 1.0)
        self.assertAlmostEqual(result["score"], result["optimizer_process_weight"])
        self.assertEqual(result["num_extracted_equations"], 1)


if __name__ == "__main__":
    unittest.main()
