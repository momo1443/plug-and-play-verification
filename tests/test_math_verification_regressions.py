"""Observed parser failures with nearby counterexamples; CPU only."""
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from recipes.deepscaler.math_expressions import answers_equal, _parsed_answer
from recipes.deepscaler.reward_fn import _process_reward_audit


def checks(text, **kwargs):
    return [check for step in _process_reward_audit(text, **kwargs)['steps']
            for check in step['equation_checks']]


class ProcessExtractionRegressionTest(unittest.TestCase):
    def test_sentence_math_and_prose_parentheses_keep_whole_numbers(self):
        for text, count in (
            (r'16 - 48 = -32$. $-32 + 34 = 2', 2),
            ('100 * -1 = -100. -100/200 = -0.5.', 2),
            (r'Calculated total chunks ($60\times512=30,720$)', 1),
            (r'60\times512=30,\!720', 1),
            ('125/12=10 and 5/12', 1),
            (r'15/11=1 \frac{4}{11}', 1),
        ):
            with self.subTest(text=text):
                evidence = checks(text)
                self.assertEqual(len(evidence), count)
                self.assertTrue(all(c['valid'] for c in evidence))
        self.assertFalse(checks('125/12=10 and 4/12')[0]['valid'])
        self.assertFalse(checks(r'Calculated total chunks ($60\times512=30,721$)')[0]['valid'])
        self.assertFalse(answers_equal('(1,234)', '1234'))

    def test_context_labels_do_not_become_numeric_equalities(self):
        for text, count in (
            ('Total duration: $14:12 - 8:00 = 6$ hours 12 minutes.', 0),
            ('Total minutes from 12:34 to 1:18 = 26 + 18 = 44', 1),
            (r'Width = distance between $x=-2$ and $x=1$ = $1-(-2)=3$', 1),
            (r'$164 / 60 = 2$ R $44$.', 0),
            ('In binary, $11000 = 24$.', 0),
            ('Binary subtraction: 11000 - 1 = 10111', 0),
            (r'Percent Increase: $(12/200)\times100\% = 12/2 = 6\%$', 0),
            (r'Marked Price = $100 - 30\% = 70$', 0),
            (r'$60 = 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60$.', 0),
            ('Cumulative: 5(2)=2, 6(3)=5, 7(4)=9', 0),
            ('Minutes: 78 (18+60) - 34 = 44', 0),
            (r'$20\% 2=0$', 0),
            ('1/10 / 1/2 = 1/5', 0),
            ('12/30 = 0.40 (40%)', 0),
            ('[1.1]=1', 0),
        ):
            with self.subTest(text=text):
                evidence = checks(text)
                self.assertEqual(len(evidence), count)
                self.assertTrue(all(c['valid'] for c in evidence))
        self.assertFalse(checks('11000 - 1 = 10111')[0]['valid'])
        self.assertFalse(checks('2 + 3 = 6')[0]['valid'])
        self.assertTrue(checks('Binary value: 11000_2 = 24')[0]['valid'])

    def test_real_odd_roots_and_explicit_fraction_division(self):
        for text in (r'(-125)^{1/3}=-5', r'(-125)^{2/3}=25',
                     r'(-1/125)^{-2/3}=25', r'(-8)^{-1/3}=-1/2',
                     r'1/10 \div 1/2 = 1/5', r'5/2 \div 1/4 = 10', '5/2 ÷ 1/4 = 10'):
            with self.subTest(text=text):
                self.assertTrue(all(c['valid'] for c in checks(text)))
                self.assertTrue(checks(text))
        for text in ('(-125)^(1/3)=5', '(-4)^(1/2)=2', '1/0=1'):
            self.assertFalse(checks(text)[0]['valid'])

    def test_truncated_number_and_explicitly_rejected_root(self):
        self.assertEqual(checks(r'$143,818,402 \times 10 = 1,', truncated=True), [])
        self.assertEqual(checks(r'$143,818,402 \times 10 = 1,'), [])
        self.assertFalse(checks(r'$143,818,402 \times 10 = 1$')[0]['valid'])
        text = (r'Original equation: $\sqrt{5-(-1)} = (-1)\sqrt{5-(-1)}$' + '\n'
                + r'LHS $\neq$ RHS.' + '\nSo x = -1 is NOT a solution.')
        self.assertEqual(checks(text), [])
        self.assertEqual(checks('Check third eq: $0+0=1$? No.'), [])
        self.assertFalse(checks('0+0=1')[0]['valid'])
        evidence = checks('Step 1: 2+3=6.\nStep 2: Correct that mistake: 2+3=5.')
        self.assertEqual([c['valid'] for c in evidence], [False, True])

    def test_rounded_subexpressions_are_excluded_not_awarded_credit(self):
        for text in ('-4/3*(1.8-3)=-1.333*(-1.2)', '2^(64/7)=2^9.14',
                     'sqrt(.25+.866^2)=1', '25/72=25*.0138',
                     '1.5-2.666=-7/6', '52^10=(3.8*10^8)^2', '10!=3.6*10^6',
                     '200/600=33.3%'):
            with self.subTest(text=text):
                self.assertEqual(checks(text), [])
                self.assertEqual(_process_reward_audit(text)['process_reward'], 0)
        for text in ('20*.0145=2.9', '4+1.2+.36=6.76', '2+3=6'):
            self.assertFalse(checks(text)[0]['valid'])


class TerminalEquivalenceRegressionTest(unittest.TestCase):
    def test_observed_equivalences_need_no_symbolic_worker(self):
        cases = [('17, 18', '17 or 18', True), ('18, 17', '17 or 18', True),
                 ('17', '17 or 18', False), ('17, 19', '17 or 18', False),
                 ("DR' + R", "$R+R'D$", True), ("R'D + R", "$R+R'D$", True),
                 ("DR + R", "$R+R'D$", False),
                 ('4:3', '4 : 3', True), ('4:3', '3:4', False),
                 ('1/4', '1:4', True), ('19:8', '19/8', True),
                 ('10:25 PM', '10:25 PM', True),
                 ('5', 'y=5', True), ('6', 'y=5', False)]
        with patch('recipes.deepscaler.math_expressions._isolated_symbolic_equal',
                   side_effect=AssertionError('unnecessary symbolic subprocess')):
            with ThreadPoolExecutor(max_workers=1) as pool:
                for a, b, expected in cases:
                    with self.subTest(a=a, b=b):
                        self.assertEqual(pool.submit(answers_equal, a, b).result(), expected)

    def test_empty_and_timed_out_parses_are_not_cached(self):
        from math_verify.errors import TimeoutException
        for failure in ([], TimeoutException('deadline')):
            _parsed_answer.cache_clear()
            with patch('math_verify.parse', side_effect=[failure, [5]]) as parse:
                with self.assertRaises(RuntimeError):
                    _parsed_answer('5')
                self.assertEqual(_parsed_answer('5'), [5])
                self.assertEqual(parse.call_count, 2)
        _parsed_answer.cache_clear()
