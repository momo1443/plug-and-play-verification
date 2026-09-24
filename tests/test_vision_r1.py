"""Tests for Vision-R1 visual artifacts and verifier integration."""

from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd
from PIL import Image

from recipes.vision_r1.agent_flow import _one_tool_call
from recipes.vision_r1.artifacts import CERTIFICATE_SCHEMA_VERSION, VisualArtifact, normalize_bbox
from recipes.vision_r1.prepare_data import MAX_IMAGE_PIXELS, build_row
from recipes.vision_r1.reward_contract import outcome_reward, reward_schedule
from recipes.vision_r1.verifier import verify_process


class VisionR1ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = VisualArtifact.root(Image.new("RGB", (100, 100), color=(10, 20, 30)))
        self.crop = VisualArtifact.crop(
            ordinal=1,
            parent=self.root,
            bbox_2d=[10, 10, 70, 80],
            created_step=1,
        )
        self.certificate = {
            "schema_version": CERTIFICATE_SCHEMA_VERSION,
            "evidence_artifact_ids": [self.crop.artifact_id],
            "claim": "The cited diagram region supports answer 11.",
        }

    def test_crop_is_deterministic_and_replayable(self) -> None:
        repeated = VisualArtifact.crop(
            ordinal=1,
            parent=self.root,
            bbox_2d=[10, 10, 70, 80],
            created_step=1,
        )
        self.assertEqual(self.crop.sha256, repeated.sha256)
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop},
            raw_certificate=self.certificate,
            submitted_answer="11",
        )
        self.assertEqual(result.process_reward, 1.0)
        self.assertEqual(result.credits[0].step_index, 1)

    def test_tampered_artifact_fails_replay(self) -> None:
        tampered = replace(self.crop, sha256="0" * 64)
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, tampered.artifact_id: tampered},
            raw_certificate=self.certificate,
            submitted_answer="11",
        )
        self.assertEqual(result.process_reward, 0.0)
        self.assertIn("replay_hash_mismatch", result.audit["artifact_audits"][0]["errors"])

    def test_certificate_must_couple_claim_to_answer(self) -> None:
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop},
            raw_certificate=self.certificate,
            submitted_answer="12",
        )
        self.assertEqual(result.process_reward, 0.0)
        self.assertFalse(result.audit["answer_coupled"])

    def test_rejects_tiny_and_full_image_crops(self) -> None:
        with self.assertRaises(ValueError):
            normalize_bbox([0, 0, 10, 10], width=100, height=100)
        with self.assertRaises(ValueError):
            normalize_bbox([0, 0, 100, 100], width=100, height=100)


class VisionR1ContractTest(unittest.TestCase):
    def test_hermes_tool_call_uses_shared_parser(self) -> None:
        name, arguments = _one_tool_call('<tool_call>{"name":"submit_answer","arguments":{"answer":"11"}}</tool_call>')
        self.assertEqual(name, "submit_answer")
        self.assertEqual(arguments, {"answer": "11"})

    def test_warmup_and_group_uniform_schedule(self) -> None:
        warmup = reward_schedule(
            reward_mode="uniform_visual_certificate",
            global_step=50,
            is_validation=False,
            terminal_warmup_steps=50,
            prompt_group_key="vision_r1_rl:7",
        )
        first = reward_schedule(
            reward_mode="uniform_visual_certificate",
            global_step=51,
            is_validation=False,
            terminal_warmup_steps=50,
            prompt_group_key="vision_r1_rl:7",
        )
        second = reward_schedule(
            reward_mode="uniform_visual_certificate",
            global_step=51,
            is_validation=False,
            terminal_warmup_steps=50,
            prompt_group_key="vision_r1_rl:7",
        )
        self.assertEqual((warmup.terminal_weight, warmup.process_weight), (1.0, 0.0))
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.terminal_weight + first.process_weight, 1.0)

    def test_outcome_reward_handles_exact_and_normalized_math(self) -> None:
        self.assertEqual(outcome_reward("C", "c"), 1.0)
        self.assertEqual(outcome_reward("0.5", r"\frac{1}{2}"), 1.0)
        self.assertEqual(outcome_reward("12", "11"), 0.0)

    def test_prepared_row_preserves_image_and_stable_id(self) -> None:
        source = pd.Series(
            {
                "problem": "<image>Find x.",
                "images": [{"bytes": b"fake"}],
                "answer": "11",
            }
        )
        row = build_row(source, 3, "train")
        self.assertEqual(row["extra_info"]["question_id"], "vision-r1-rl:train:3")
        self.assertEqual(row["images"][0]["max_pixels"], MAX_IMAGE_PIXELS)
        self.assertEqual(row["reward_model"]["ground_truth"], "11")


if __name__ == "__main__":
    unittest.main()
