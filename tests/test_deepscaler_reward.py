from __future__ import annotations

import unittest

from recipes.deepscaler import reward_fn


class DeepScalerRewardTest(unittest.TestCase):
    def test_symbolic_identifiers_are_not_truncated_into_numbers(self) -> None:
        for equation in (
            "d1 = 2", "d3 = 1", "2d3 = 0", "s(200) = 2",
            "$s(200) = 2$", "- For d1=2: (1,1) -> 1 pair",
            "Let a + 2 = 4", "f(2) + 3 = 5", "x - 2 = 0",
            "x * 2 = 2", "x / 2 = 2", "x^2 = 2", "x+2=2",
            r"$3d_3 = 0$", r"$\operatorname{digit_sum}(200)=2$",
        ):
            with self.subTest(equation=equation):
                audit = reward_fn._process_reward_audit(equation)
                self.assertEqual(audit["equation_count"], 0)
                self.assertEqual(audit["process_reward"], 0)

    def test_incomplete_or_unsupported_expressions_are_not_graded(self) -> None:
        for equation in ("1 = 2: (1", "2 + = 3", r"\overline{12} = 12"):
            with self.subTest(equation=equation):
                self.assertEqual(reward_fn._process_reward_audit(equation)["equation_count"], 0)

    def test_supported_functions_retain_their_semantics(self) -> None:
        for equation in ("sqrt(4) = 2", r"\sqrt{4} = 2", "pi * 2 = 2 * pi",
                         r"\lfloor2.9\rfloor = 2", r"\lceil2.1\rceil = 3",
                         "1e-6 * 10^6 = 1", "1E-6 * 10^6 = 1"):
            with self.subTest(equation=equation):
                audit = reward_fn._process_reward_audit(equation)
                self.assertEqual(audit["equation_count"], 1)
                self.assertEqual(audit["verified_equation_count"], 1)
        self.assertEqual(reward_fn._process_reward_audit("sqrt(4) = 4")["verified_equation_count"], 0)
        self.assertFalse(reward_fn._verify_equation_pair("1e-6", "e-6"))
        audit = reward_fn._process_reward_audit("1 / 0 = 1")
        self.assertEqual(audit["equation_count"], 1)
        self.assertEqual(audit["verified_equation_count"], 0)

    def test_later_failure_in_chain_is_kept(self) -> None:
        audit = reward_fn._process_reward_audit("2 + 3 = 5 = 6")
        self.assertEqual(audit["equation_count"], 2)
        self.assertEqual(audit["verified_equation_count"], 1)

    def test_symbolic_steps_do_not_dilute_numeric_process_credit(self) -> None:
        solution = "Step 1: d1=2 and s(200)=2.\nStep 2: 45/216=5/24."
        self.assertEqual(reward_fn.compute_process_reward(solution), (1.0, 1, 1))
        solution = "Step 1: d1=2.\nStep 2: 2+3=6.\nStep 3: 2+3=5."
        self.assertEqual(reward_fn.compute_process_reward(solution), (0.5, 1, 2))

    def test_numeric_prefixes_survive_equation_extraction(self) -> None:
        for equation in (
            "0.8 * 100 = 80", "1.2 + 3 = 4.2", ".5 + .5 = 1",
            "(28 / 35) * 100 = 80", "-2 + 3 = 1",
            "Step 1: (28 / 35) * 100 = 80", "1. -2 + 3 = 1",
            "Percentage calculation: (28 / 35) * 100 = 80",
        ):
            with self.subTest(equation=equation):
                audit = reward_fn._process_reward_audit(equation)
                self.assertEqual(audit["equation_count"], 1)
                self.assertEqual(audit["verified_equation_count"], 1)

    def test_prefix_cleanup_cannot_turn_incorrect_equations_into_correct_ones(self) -> None:
        for equation in ("0.8 * 100 = 800", "1.2 + 3 = 5", "-2 + 3 = 5"):
            with self.subTest(equation=equation):
                audit = reward_fn._process_reward_audit(equation)
                self.assertEqual(audit["equation_count"], 1)
                self.assertEqual(audit["verified_equation_count"], 0)

    def test_every_step_contributes_to_process_reward(self) -> None:
        solution = r"""Step 1: Check 2 + 3 = 5 and 4 + 4 = 9.
Step 2: Continue with 6 / 2 = 3."""

        score, fully_verified_steps, step_count = reward_fn.compute_process_reward(solution)

        self.assertAlmostEqual(score, 0.75)
        self.assertEqual(fully_verified_steps, 1)
        self.assertEqual(step_count, 2)

    def test_step_without_numeric_equation_is_excluded_from_average(self) -> None:
        solution = r"""Step 1: This paragraph gives an unsupported verbal conclusion.
Step 2: Check the arithmetic $8 - 3 = 5$."""

        score, fully_verified_steps, step_count = reward_fn.compute_process_reward(solution)

        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(fully_verified_steps, 1)
        self.assertEqual(step_count, 1)

    def test_trivial_literal_identity_does_not_earn_credit(self) -> None:
        self.assertEqual(reward_fn.compute_process_reward("1 = 1"), (0.0, 0, 0))
        self.assertEqual(reward_fn.compute_process_reward("1 = 2"), (0.0, 0, 1))

    def test_binomial_coefficient_uses_combinatorial_semantics(self) -> None:
        self.assertTrue(reward_fn._verify_equation_pair(r"\binom{10}{2}", "45"))
        self.assertFalse(reward_fn._verify_equation_pair(r"\binom{10}{2}", "20"))

    def test_explicit_repeating_decimal_is_exact_and_ellipsis_is_excluded(self) -> None:
        self.assertTrue(reward_fn._verify_equation_pair(r"0.\overline{6}", r"\frac{2}{3}"))
        self.assertEqual(reward_fn._process_reward_audit("100/6 = 16.666...")["equation_count"], 0)

    def test_observed_thousands_roots_mixed_numbers_and_nested_fractions(self):
        for expression in (
            r"4 \times 45000 = 180,000", r"180,000 + 1 = 180,001",
            r"4 \times 45000 = 180,000.", r"$2+3=5,4+4=8$",
            r"1000 \times 45 \times 4 = 45,000 \times 4 = 180,000",
            r"\sqrt[3]{4} = 4^{1/3}", r"\sqrt[4]{16} = 2",
            r"\frac{1}{\frac{1}{2}} = 2", r"\frac{24}5 = 4.8",
            r"35\frac{7}{17} = \frac{602}{17}", r"-7\frac{1}{2} = -7.5",
            r"\binom{6}{1}\binom{5}{5} = 6 \times 1",
            r"2019 = 5613_7", r"(1000)_8 = 512",
            r"20\% \times 80 = 16", r"\cos(\frac{3\pi}{2}) = 0",
            r"\sin(30^{\circ}) = 0.5", r"\ln(1) = 0", r"\log(10) = 1",
            r"3^{4014} - 3^{4010} = 3^{4010}(3^4 - 1)",
        ):
            with self.subTest(expression=expression):
                audit = reward_fn._process_reward_audit(expression)
                self.assertGreater(audit['equation_count'], 0)
                self.assertEqual(audit['verified_equation_count'], audit['equation_count'])

    def test_unsupported_context_never_becomes_false_arithmetic(self):
        for expression in ("e = 3", "3e = 6", r"\sin(30) = 0.5",
                           r"$h_{stool}+150+46=230$", r"$\pi(1)=2$", "+7=42",
                           "1+1=0 (mod 2)", r"2019 \div 7 = 288 with remainder 3"):
            with self.subTest(expression=expression):
                self.assertEqual(reward_fn._process_reward_audit(expression)['equation_count'], 0)
        audit = reward_fn._process_reward_audit("3*5 = 4", question="If A*B means (A+B)/2, find (3*5)*8.")
        self.assertEqual(audit['equation_count'], 0)
        self.assertEqual(reward_fn._process_reward_audit("3*5=4")['verified_equation_count'], 0)

    def test_wrong_numeric_steps_and_nonfinite_values_still_fail(self):
        for expression in ("20 * 0.0145 = 2.9", "4 + 1.2 + 0.36 = 6.76",
                           "1/0=1", "sqrt(-1)=1", "2+3=5=6"):
            with self.subTest(expression=expression):
                audit = reward_fn._process_reward_audit(expression)
                self.assertGreater(audit['equation_count'], audit['verified_equation_count'])

    def test_terminal_equivalence_only_uses_submitted_answer(self):
        for answer, gold in ((r"\dfrac{24}{5}", r"$\frac{24}5$"),
                             ("180,001", "180001"), (r"22.5^{\circ}", r"22\frac{1}{2}^{\circ}"),
                             (r"\frac{\sqrt{2}}2", r"\frac{1}{\sqrt{2}}")):
            with self.subTest(answer=answer):
                self.assertEqual(reward_fn.compute_terminal_em(r"\boxed{" + answer + "}", gold), 1)
        for answer, gold in (("0.33", "1/3"), ("0.333333", "1/3"), ("66", "54"), ("5", "5 or 6")):
            self.assertEqual(reward_fn.compute_terminal_em(r"\boxed{" + answer + "}", gold), 0)
        self.assertEqual(reward_fn.compute_terminal_em(r"\boxed{(1,234)}", "1234"), 0)
        self.assertEqual(reward_fn.compute_terminal_em(r"\boxed{2004!}", "2004!"), 1)

    def test_symbolic_answers_match_in_async_actor_thread(self):
        from concurrent.futures import ThreadPoolExecutor
        from recipes.deepscaler.math_expressions import _parsed_answer, answers_equal
        # Ray's async actor loop is not the interpreter main thread. Clear the
        # parse cache so a previous main-thread comparison cannot hide it.
        _parsed_answer.cache_clear()
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertTrue(pool.submit(answers_equal, r"\left(1,2\right)", "(1,2)").result())
            self.assertFalse(pool.submit(answers_equal, r"\left(1,3\right)", "(1,2)").result())
            self.assertTrue(pool.submit(answers_equal, "x+x", "2x").result())
            self.assertFalse(pool.submit(answers_equal, "x+x", "3x").result())

    def test_observed_context_is_not_discarded_at_inline_math_boundaries(self):
        for equation in (
            r"30\% of $120 = 0.30 \times 120 = 36$",
            r"Half of $36 = 18$", r"Square of 3 = 9",
            r"Deficit = $135 - $105 = $30. Tom needs to pay $30.",
            r"After Double 3 = 0 + 40 = 40", r"- 15 + 12 = 27.",
            r"$- 2 + 3 = 1$",
        ):
            with self.subTest(equation=equation):
                audit = reward_fn._process_reward_audit(equation)
                self.assertGreater(audit['equation_count'], 0)
                self.assertEqual(audit['verified_equation_count'], audit['equation_count'])
        for equation in (
            "Day 2 = 3", "Numbers with no 2 = 648", "Investment 1 = 2700",
            "Sum of digits of 120 = 1 + 2 + 0 = 3", "1,2,0,0 = 3",
        ):
            with self.subTest(equation=equation):
                checks = reward_fn._process_reward_audit(equation)['steps'][0]['equation_checks']
                self.assertTrue(all(check['valid'] for check in checks))
        audit = reward_fn._process_reward_audit(r"20\% of $100 = 30$")
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 0)

    def test_running_totals_and_numeric_labels_do_not_create_failures(self):
        audit = reward_fn._process_reward_audit(
            "Numerator: 10-9=1, +8=9, -7=2, +6=8, -5=3, +4=7, -3=4."
        )
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 1)
        self.assertEqual(reward_fn._process_reward_audit("-7=2")['verified_equation_count'], 0)
        self.assertEqual(reward_fn._process_reward_audit(r"$- 7 = 2$")['verified_equation_count'], 0)
        audit = reward_fn._process_reward_audit("$10-9=1$\n$+8=9$\n$-7=2$\n$+6=8$\n$-5=3$")
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 1)
        self.assertEqual(reward_fn._process_reward_audit("-7=2")['equation_count'], 1)
        audit = reward_fn._process_reward_audit("Total ways for $D=5 = 12 + 24 = 36$.")
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 1)
        audit = reward_fn._process_reward_audit("$D=5=6$")
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 0)

    def test_units_digit_displays_require_context_and_a_correct_digit(self):
        audit = reward_fn._process_reward_audit("Possible ends for 3T:\n$3*4=2, 3*9=7$")
        self.assertEqual(audit['equation_count'], 0)
        self.assertEqual(len(audit['steps'][-1]['excluded_equations']), 2)
        self.assertEqual(reward_fn._process_reward_audit("3*4=2")['equation_count'], 1)
        audit = reward_fn._process_reward_audit("Units digit:\n3*9=8")
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 0)

    def test_generalized_binomial_and_multinomial_evidence(self):
        for equation in (
            r"\binom{-\frac{1}{2}}{100}=\frac{200!}{2^{200}(100!)^2}",
            r"\binom{12}{2,2,2,2,2,2}=\frac{12!}{(2!)^6}",
            r"\binom{6}{3,2,1}=60",
        ):
            audit = reward_fn._process_reward_audit(equation)
            self.assertEqual(audit['equation_count'], 1)
            self.assertEqual(audit['verified_equation_count'], 1)
        audit = reward_fn._process_reward_audit(r"\binom{6}{3,2,1}=61")
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 0)

    def test_ambiguous_rounding_and_notation_earn_no_exact_credit(self):
        for equation in (
            "100/3=33.33", r"30-2=28\%", r"2\pi / 0.25\pi=8",
            r"11(2)1_6=265", r"-2 \frac{8}{5}=-\frac{16}{5}",
        ):
            self.assertEqual(reward_fn._process_reward_audit(equation)['equation_count'], 0)
        self.assertEqual(reward_fn._process_reward_audit("100/3=34.33")['verified_equation_count'], 0)
        audit = reward_fn._process_reward_audit(r"Probability $\pi=15/6^3=15/216$.")
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 1)

    def test_truncation_preserves_finished_equations_and_real_failures(self):
        audit = reward_fn._process_reward_audit("Check $2+3=6$.\n$37*40=1", truncated=True)
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 0)
        self.assertEqual(audit['steps'][-1]['excluded_equations'][0]['reason'],
                         'generation_truncated_before_equation_end')
        self.assertEqual(reward_fn._process_reward_audit("37*40=1")['equation_count'], 1)
        audit = reward_fn._process_reward_audit("2+3=5=6", truncated=True)
        self.assertEqual(audit['equation_count'], 1)
        self.assertEqual(audit['verified_equation_count'], 1)
        audit = reward_fn._process_reward_audit(
            "Compute 200+3=203\nCompute 300+4=304\n$37*40=1", truncated=True)
        self.assertEqual(audit['equation_count'], 2)
        self.assertEqual(audit['verified_equation_count'], 2)

    def test_negative_exponents_are_exact_without_a_symbolic_subprocess(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        from recipes.deepscaler.math_expressions import answers_equal
        with patch('recipes.deepscaler.math_expressions._isolated_symbolic_equal',
                   side_effect=AssertionError('unexpected symbolic worker')):
            with ThreadPoolExecutor(max_workers=4) as pool:
                for prediction in (r"x^{-1}y^{-1}", r"(xy)^{-1}", r"\frac{1}{xy}"):
                    self.assertTrue(pool.submit(answers_equal, prediction, r"x^{ - 1}y^{ - 1}").result())
                self.assertFalse(pool.submit(answers_equal, r"(xy)^{-2}", r"x^{-1}y^{-1}").result())

    def test_equation_and_its_roots_have_distinct_answer_forms(self):
        from recipes.deepscaler.math_expressions import answers_equal
        self.assertFalse(answers_equal(r"-5\pm4\sqrt{2}", r"$y^2+10y-7=0$"))
        self.assertTrue(answers_equal(r"y^2+10y-7=0", r"$y^{2}+10y-7=0$"))
        self.assertTrue(answers_equal("3a=2b", "$a=2b/3$"))
        self.assertFalse(answers_equal("3a=3b", "$a=2b/3$"))

    def test_plain_numeric_units_do_not_become_symbolic_products(self):
        from recipes.deepscaler.math_expressions import answers_equal
        self.assertTrue(answers_equal(r"50\text{ cm}", "50 cm"))
        self.assertTrue(answers_equal("50", "50 cm"))
        self.assertFalse(answers_equal("50", "51 cm"))
        self.assertFalse(answers_equal("50", "50m"))

    def test_irrational_binomial_argument_is_unsupported_not_a_crash(self):
        from recipes.deepscaler.math_expressions import evaluation
        self.assertEqual(evaluation("comb(sqrt(2),3)")[1], "limit")

    def test_symbolic_worker_execution_failure_is_not_cached(self):
        import subprocess
        from unittest.mock import patch
        from recipes.deepscaler.math_expressions import _isolated_symbolic_equal
        _isolated_symbolic_equal.cache_clear()
        with patch('subprocess.run', side_effect=[
                subprocess.TimeoutExpired('symbolic', 8),
                subprocess.CompletedProcess('symbolic', 0, stdout='true', stderr='')]) as run:
            with self.assertRaises(RuntimeError):
                _isolated_symbolic_equal('x+x', '2x')
            self.assertTrue(_isolated_symbolic_equal('x+x', '2x'))
            self.assertEqual(run.call_count, 2)

    def test_post_warmup_reward_uses_uniform_weight(self) -> None:
        solution = r"""Step 1: Check the arithmetic $2 + 3 = 5$.
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
        self.assertEqual(result["optimizer_reward_phase"], "uniform_equation_process")
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
