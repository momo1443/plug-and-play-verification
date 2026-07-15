import dataclasses
import unittest

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
    search_step_reward,
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
        self.assertIs(resolve_reward_arm(None, formal_a0=False), RewardArm.A1)
        self.assertIs(resolve_reward_arm(None, formal_a0=True), RewardArm.A0)
        with self.assertRaises(ValueError):
            resolve_reward_arm("A2", formal_a0=True)
        with self.assertRaises(ValueError):
            parse_reward_arm("A3")

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


if __name__ == "__main__":
    unittest.main()
