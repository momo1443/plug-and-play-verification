import dataclasses
import unittest

from recipes.hotpotqa.judge_prompts import (
    ALLOWED_JUDGE_OUTPUTS,
    JUDGE_VERSION,
    format_gold_facts_for_prompt,
    parse_judge_fact_ids,
    parse_judge_score,
)
from recipes.hotpotqa.process_verifier import (
    PROCESS_VERIFIER_VERSION,
    verify_new_evidence,
)
from recipes.hotpotqa.reward_arm import (
    RewardArm,
    final_step_response_mask,
    final_step_reward,
    parse_reward_arm,
    resolve_reward_arm,
    scale_terminal_reward,
    search_step_reward,
    training_reward_contract,
)


class ProcessVerifierTest(unittest.TestCase):
    def test_new_coverage_reward_is_bounded_and_duplicates_are_zero(self):
        first = verify_new_evidence(
            returned_evidence_ids=["gold-a", "distractor"],
            gold_evidence_ids=["gold-a", "gold-b"],
            covered_gold_evidence_ids=[],
            evidence_metrics_eligible=True,
        )
        self.assertEqual(first.reward, 0.5)
        self.assertEqual(first.new_gold_evidence_ids, ("gold-a",))
        self.assertEqual(first.covered_gold_evidence_ids, ("gold-a",))

        duplicate = verify_new_evidence(
            returned_evidence_ids=["gold-a", "distractor"],
            gold_evidence_ids=["gold-a", "gold-b"],
            covered_gold_evidence_ids=first.covered_gold_evidence_ids,
            evidence_metrics_eligible=True,
        )
        self.assertEqual(duplicate.reward, 0.0)
        self.assertEqual(duplicate.new_gold_evidence_ids, ())

        second = verify_new_evidence(
            returned_evidence_ids=["gold-b"],
            gold_evidence_ids=["gold-a", "gold-b"],
            covered_gold_evidence_ids=duplicate.covered_gold_evidence_ids,
            evidence_metrics_eligible=True,
        )
        self.assertEqual(second.reward, 0.5)
        self.assertEqual(second.covered_gold_evidence_ids, ("gold-a", "gold-b"))
        self.assertEqual(sum(item.reward for item in (first, duplicate, second)), 1.0)

    def test_one_search_can_cover_all_gold_evidence(self):
        result = verify_new_evidence(
            returned_evidence_ids=["gold-b", "gold-a", "gold-a"],
            gold_evidence_ids=["gold-a", "gold-b"],
            covered_gold_evidence_ids=[],
            evidence_metrics_eligible=True,
        )
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.new_gold_evidence_ids, ("gold-a", "gold-b"))

    def test_ineligible_annotation_has_no_process_reward(self):
        result = verify_new_evidence(
            returned_evidence_ids=["gold-a"],
            gold_evidence_ids=["gold-a"],
            covered_gold_evidence_ids=[],
            evidence_metrics_eligible=False,
        )
        self.assertIsNone(result.reward)

    def test_verifier_cannot_accept_answer_or_mutate_result(self):
        with self.assertRaises(TypeError):
            verify_new_evidence(
                returned_evidence_ids=[],
                gold_evidence_ids=[],
                covered_gold_evidence_ids=[],
                evidence_metrics_eligible=True,
                answer="hidden answer",
            )
        result = verify_new_evidence(
            returned_evidence_ids=[],
            gold_evidence_ids=[],
            covered_gold_evidence_ids=[],
            evidence_metrics_eligible=True,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.reward = 1.0
        self.assertEqual(PROCESS_VERIFIER_VERSION, "hotpotqa-new-evidence-v1")


class RewardArmTest(unittest.TestCase):
    def test_arm_parsing_and_formal_a0_compatibility(self):
        self.assertIs(parse_reward_arm("a2"), RewardArm.A2)
        self.assertIs(parse_reward_arm("a3"), RewardArm.A3)
        self.assertIs(parse_reward_arm("a6"), RewardArm.A6)
        self.assertIs(parse_reward_arm("a9"), RewardArm.A9)
        self.assertIs(resolve_reward_arm(None, formal_a0=False), RewardArm.A1)
        self.assertIs(resolve_reward_arm(None, formal_a0=True), RewardArm.A0)
        with self.assertRaises(ValueError):
            resolve_reward_arm("A2", formal_a0=True)
        with self.assertRaises(ValueError):
            parse_reward_arm("A99")

    def test_a1_is_terminal_only_in_training(self):
        self.assertEqual(search_step_reward(RewardArm.A1, 1.0, is_validation=False), 0.0)
        self.assertIsNone(final_step_reward(RewardArm.A1, is_validation=False))
        self.assertIsNone(final_step_response_mask(RewardArm.A1, 3, is_validation=False))

    def test_a2_training_uses_process_only_and_masks_final_answer(self):
        self.assertEqual(search_step_reward(RewardArm.A2, 0.5, is_validation=False), 0.5)
        self.assertEqual(final_step_reward(RewardArm.A2, is_validation=False), 0.0)
        self.assertEqual(
            final_step_response_mask(RewardArm.A2, 3, is_validation=False),
            [0, 0, 0],
        )

    def test_a2_validation_reports_terminal_em_not_process_return(self):
        self.assertEqual(search_step_reward(RewardArm.A2, 1.0, is_validation=True), 0.0)
        self.assertIsNone(final_step_reward(RewardArm.A2, is_validation=True))
        self.assertIsNone(final_step_response_mask(RewardArm.A2, 3, is_validation=True))

    def test_a3_combines_half_process_and_half_terminal_with_final_tokens(self):
        contract = training_reward_contract(RewardArm.A3)
        self.assertEqual(contract.process_weight, 0.5)
        self.assertEqual(contract.terminal_weight, 0.5)
        self.assertEqual(contract.final_response_mask, 1)
        self.assertEqual(search_step_reward(RewardArm.A3, 0.5, is_validation=False), 0.25)
        self.assertIsNone(final_step_reward(RewardArm.A3, is_validation=False))
        self.assertEqual(
            scale_terminal_reward(RewardArm.A3, 1.0, is_validation=False),
            0.5,
        )
        self.assertIsNone(final_step_response_mask(RewardArm.A3, 3, is_validation=False))

    def test_a3_validation_reports_unscaled_terminal_em(self):
        self.assertEqual(search_step_reward(RewardArm.A3, 1.0, is_validation=True), 0.0)
        self.assertEqual(scale_terminal_reward(RewardArm.A3, 1.0, is_validation=True), 1.0)
        self.assertIsNone(final_step_response_mask(RewardArm.A3, 3, is_validation=True))

    def test_a6_combines_half_llm_judge_and_half_terminal(self):
        contract = training_reward_contract(RewardArm.A6)
        self.assertEqual(contract.process_weight, 0.5)
        self.assertEqual(contract.terminal_weight, 0.5)
        self.assertEqual(contract.final_response_mask, 1)
        self.assertEqual(search_step_reward(RewardArm.A6, 0.5, is_validation=False), 0.25)
        self.assertIsNone(final_step_reward(RewardArm.A6, is_validation=False))
        self.assertEqual(
            scale_terminal_reward(RewardArm.A6, 1.0, is_validation=False),
            0.5,
        )
        self.assertIsNone(final_step_response_mask(RewardArm.A6, 3, is_validation=False))

    def test_a6_validation_reports_unscaled_terminal_em(self):
        self.assertEqual(search_step_reward(RewardArm.A6, 1.0, is_validation=True), 0.0)
        self.assertEqual(scale_terminal_reward(RewardArm.A6, 1.0, is_validation=True), 1.0)
        self.assertIsNone(final_step_response_mask(RewardArm.A6, 3, is_validation=True))

    def test_a6_factid_cumulative_process_increment_is_bounded(self):
        """Any sequence of judge fact-ID results must produce cumulative
        increment ≤ 1, matching A3's bound."""
        gold = {"fact-1", "fact-2", "fact-3", "fact-4"}
        # Simulate 3 search steps with judge returning subsets of gold
        judge_results = [
            {"fact-1", "fact-2"},      # step 1: 2 new
            {"fact-1", "fact-3"},      # step 2: 1 new (fact-1 already covered)
            {"fact-2", "fact-3", "fact-4"},  # step 3: 1 new (fact-4)
        ]
        covered: set[str] = set()
        cumulative = 0.0
        for judge_ids in judge_results:
            gold_hits = judge_ids & gold
            new_covered = gold_hits - covered
            increment = len(new_covered) / len(gold) if gold else 0.0
            covered.update(gold_hits)
            cumulative += increment
        self.assertLessEqual(cumulative, 1.0 + 1e-9)

    def test_a6_factid_non_increasing_coverage_produces_zero_increment(self):
        """If judge doesn't find any new gold facts, increment must be 0."""
        gold = {"fact-1", "fact-2"}
        covered: set[str] = set()

        # Step 1: judge finds fact-1
        judge_ids_1 = {"fact-1"}
        new_1 = (judge_ids_1 & gold) - covered
        inc_1 = len(new_1) / len(gold)
        covered.update(judge_ids_1 & gold)

        # Step 2: judge finds fact-1 again (no new)
        judge_ids_2 = {"fact-1"}
        new_2 = (judge_ids_2 & gold) - covered
        inc_2 = len(new_2) / len(gold)

        # Step 3: judge finds nothing new
        judge_ids_3: set[str] = set()
        new_3 = (judge_ids_3 & gold) - covered
        inc_3 = len(new_3) / len(gold)

        self.assertEqual(inc_1, 0.5)
        self.assertEqual(inc_2, 0.0)
        self.assertEqual(inc_3, 0.0)

    def test_a6_factid_matches_a3_when_judge_agrees_with_id_matching(self):
        """When the judge's fact-ID coverage exactly matches A3's deterministic
        ID matching, the process increments must be identical."""
        gold = ("gold-a", "gold-b", "gold-c")

        # A3 deterministic steps
        a3_step1 = verify_new_evidence(
            returned_evidence_ids=["gold-a", "distractor"],
            gold_evidence_ids=list(gold),
            covered_gold_evidence_ids=[],
            evidence_metrics_eligible=True,
        )
        a3_step2 = verify_new_evidence(
            returned_evidence_ids=["gold-b"],
            gold_evidence_ids=list(gold),
            covered_gold_evidence_ids=a3_step1.covered_gold_evidence_ids,
            evidence_metrics_eligible=True,
        )
        a3_step3 = verify_new_evidence(
            returned_evidence_ids=["gold-c", "gold-a"],
            gold_evidence_ids=list(gold),
            covered_gold_evidence_ids=a3_step2.covered_gold_evidence_ids,
            evidence_metrics_eligible=True,
        )

        # A6 fact-ID steps: judge returns same IDs as A3
        a6_covered: set[str] = set()
        a6_increments: list[float] = []

        # Step 1: judge says gold-a is covered
        judge_result_1 = {"gold-a"}
        new_1 = (judge_result_1 & set(gold)) - a6_covered
        a6_increments.append(len(new_1) / len(gold))
        a6_covered.update(judge_result_1 & set(gold))

        # Step 2: judge says gold-b is covered
        judge_result_2 = {"gold-b"}
        new_2 = (judge_result_2 & set(gold)) - a6_covered
        a6_increments.append(len(new_2) / len(gold))
        a6_covered.update(judge_result_2 & set(gold))

        # Step 3: judge says gold-c is covered
        judge_result_3 = {"gold-c"}
        new_3 = (judge_result_3 & set(gold)) - a6_covered
        a6_increments.append(len(new_3) / len(gold))
        a6_covered.update(judge_result_3 & set(gold))

        # Increments must match A3
        self.assertAlmostEqual(a6_increments[0], a3_step1.reward)
        self.assertAlmostEqual(a6_increments[1], a3_step2.reward)
        self.assertAlmostEqual(a6_increments[2], a3_step3.reward)
        # Cumulative must be 1.0
        self.assertAlmostEqual(sum(a6_increments), 1.0)

    def test_a6_factid_covered_set_resets_between_trajectories(self):
        """Each trajectory must start with an empty covered set."""
        gold = {"fact-1", "fact-2"}

        # First trajectory
        covered_1: set[str] = set()
        new_1 = ({"fact-1"} & gold) - covered_1
        inc_1 = len(new_1) / len(gold)
        covered_1.update({"fact-1"} & gold)

        # Second trajectory (independent)
        covered_2: set[str] = set()
        new_2 = ({"fact-2"} & gold) - covered_2
        inc_2 = len(new_2) / len(gold)

        self.assertEqual(inc_1, 0.5)
        self.assertEqual(inc_2, 0.5)  # Not affected by first trajectory

class JudgeScoreParserTest(unittest.TestCase):
    def test_valid_anchored_scores_are_parsed(self):
        for text, expected in ALLOWED_JUDGE_OUTPUTS.items():
            self.assertEqual(parse_judge_score(text), expected)

    def test_invalid_outputs_return_none(self):
        self.assertIsNone(parse_judge_score("0.3"))
        self.assertIsNone(parse_judge_score("0.6"))
        self.assertIsNone(parse_judge_score("maybe"))
        self.assertIsNone(parse_judge_score("1.5"))
        self.assertIsNone(parse_judge_score(""))
        self.assertIsNone(parse_judge_score("0.15"))
        self.assertIsNone(parse_judge_score("0.100"))

    def test_whitespace_is_stripped(self):
        self.assertEqual(parse_judge_score(" 0.5 "), 0.5)
        self.assertEqual(parse_judge_score(" 1.0\n"), 1.0)

    def test_judge_version_is_set(self):
        self.assertEqual(JUDGE_VERSION, "hotpotqa-llm-judge-factid-matched-v4")


class JudgeFactIdParserTest(unittest.TestCase):
    def test_valid_json_array_returns_set(self):
        allowed = {"fact-1", "fact-2", "fact-3"}
        result = parse_judge_fact_ids('["fact-1", "fact-3"]', allowed)
        self.assertEqual(result, {"fact-1", "fact-3"})

    def test_empty_array_returns_empty_set(self):
        allowed = {"fact-1", "fact-2"}
        result = parse_judge_fact_ids("[]", allowed)
        self.assertEqual(result, set())

    def test_invalid_ids_are_dropped_not_error(self):
        allowed = {"fact-1", "fact-2"}
        result = parse_judge_fact_ids('["fact-1", "fact-999"]', allowed)
        self.assertEqual(result, {"fact-1"})

    def test_non_json_returns_none(self):
        allowed = {"fact-1"}
        self.assertIsNone(parse_judge_fact_ids("not json", allowed))

    def test_empty_string_returns_none(self):
        allowed = {"fact-1"}
        self.assertIsNone(parse_judge_fact_ids("", allowed))

    def test_non_array_json_returns_none(self):
        allowed = {"fact-1"}
        self.assertIsNone(parse_judge_fact_ids('"fact-1"', allowed))
        self.assertIsNone(parse_judge_fact_ids("{}", allowed))

    def test_non_string_items_in_array(self):
        allowed = {"fact-1"}
        result = parse_judge_fact_ids('["fact-1", 123]', allowed)
        self.assertEqual(result, {"fact-1"})

    def test_json_in_markdown_code_block(self):
        allowed = {"fact-1", "fact-2"}
        text = '```json\n["fact-1", "fact-2"]\n```'
        result = parse_judge_fact_ids(text, allowed)
        self.assertEqual(result, {"fact-1", "fact-2"})

    def test_json_with_extra_text(self):
        allowed = {"fact-1"}
        text = 'The supported facts are ["fact-1"].'
        result = parse_judge_fact_ids(text, allowed)
        self.assertEqual(result, {"fact-1"})

    def test_whitespace_is_stripped(self):
        allowed = {"fact-1"}
        result = parse_judge_fact_ids(' ["fact-1"] ', allowed)
        self.assertEqual(result, {"fact-1"})


class FormatGoldFactsTest(unittest.TestCase):
    def test_empty_input_returns_none(self):
        self.assertEqual(format_gold_facts_for_prompt([]), "None")

    def test_single_fact(self):
        result = format_gold_facts_for_prompt([
            ("hotpotqa-official-sentence-v1:123:0", "[Washington] He was born in Virginia."),
        ])
        self.assertIn("hotpotqa-official-sentence-v1:123:0", result)
        self.assertIn("[Washington] He was born in Virginia.", result)

    def test_multiple_facts(self):
        facts = [
            ("fact-1", "[Title A] Text A."),
            ("fact-2", "[Title B] Text B."),
        ]
        result = format_gold_facts_for_prompt(facts)
        self.assertIn("[fact-1]", result)
        self.assertIn("[fact-2]", result)
        self.assertIn("Text A.", result)
        self.assertIn("Text B.", result)


if __name__ == "__main__":
    unittest.main()
