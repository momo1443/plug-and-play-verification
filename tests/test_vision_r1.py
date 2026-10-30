"""Tests for Vision-R1 visual artifacts and verifier integration."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch
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

    def test_answer_coupling_does_not_erase_replayed_crop_reward(self) -> None:
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop},
            raw_certificate=self.certificate,
            submitted_answer="12",
        )
        self.assertEqual(result.process_reward, 1.0)
        self.assertFalse(result.audit["answer_coupled"])

    def test_missing_or_malformed_certificate_retains_verified_crops(self) -> None:
        for certificate in (None, {"bad": "field"}):
            with self.subTest(certificate=certificate):
                result = verify_process(
                    artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop},
                    raw_certificate=certificate, submitted_answer="",
                )
                expected_denominator = 1 if certificate is None else 2
                self.assertEqual(result.process_reward, 1 / expected_denominator)
                self.assertEqual(result.audit["process_denominator"], expected_denominator)
                self.assertFalse(result.audit["schema_valid"])

    def test_missing_required_certificate_on_attempted_submission_stays_in_denominator(self) -> None:
        for answer, attempted in (("11", None), ("", True)):
            with self.subTest(answer=answer, attempted=attempted):
                result = verify_process(
                    artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop},
                    raw_certificate=None, submitted_answer=answer,
                    submission_attempted=attempted,
                )
                self.assertEqual(result.process_reward, 0.5)
                self.assertEqual(result.audit["process_denominator"], 2)
                self.assertEqual(result.audit["missing_certificate_slots"], 1)
                self.assertIn(0, result.audit["applicable_checks"])

    def test_failed_crop_is_retained_without_submission(self) -> None:
        failed = replace(self.crop, artifact_id="image:crop:failed", sha256="0" * 64)
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop,
                       failed.artifact_id: failed},
            raw_certificate=None, submitted_answer="",
        )
        self.assertEqual(result.process_reward, 0.5)
        self.assertEqual(result.audit["process_denominator"], 2)
        self.assertEqual(result.audit["applicable_checks"], [1, 0])

    def test_no_applicable_checks_receive_zero_process_reward(self) -> None:
        result = verify_process(
            artifacts={self.root.artifact_id: self.root}, raw_certificate=None, submitted_answer="",
        )
        self.assertEqual(result.process_reward, 0)
        self.assertEqual(result.audit["process_denominator"], 0)
        self.assertEqual(result.audit["applicable_checks"], [])
        self.assertEqual(result.audit["missing_certificate_slots"], 0)

    def test_missing_required_crop_stays_in_denominator_without_submission(self) -> None:
        certificate = {**self.certificate,
                       "evidence_artifact_ids": [self.crop.artifact_id, "image:crop:missing"]}
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop},
            raw_certificate=certificate, submitted_answer="",
        )
        self.assertEqual(result.process_reward, 0.5)
        self.assertEqual(result.audit["process_denominator"], 2)

    def test_without_valid_crop_missing_submission_remains_zero(self) -> None:
        tampered = replace(self.crop, sha256="0" * 64)
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, tampered.artifact_id: tampered},
            raw_certificate=None, submitted_answer="",
        )
        self.assertEqual(result.process_reward, 0)

    def test_malformed_certificate_keeps_missing_required_evidence_slots(self) -> None:
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop},
            raw_certificate={"evidence_artifact_ids": [self.crop.artifact_id, "image:crop:missing", None]},
            submitted_answer="",
        )
        self.assertEqual(result.audit["process_denominator"], 4)
        self.assertEqual(result.process_reward, 0.25)

    def test_unextractable_evidence_does_not_crash_or_discard_required_slot(self) -> None:
        result = verify_process(
            artifacts={self.root.artifact_id: self.root, self.crop.artifact_id: self.crop},
            raw_certificate={**self.certificate, "evidence_artifact_ids": [self.crop.artifact_id, {}]},
            submitted_answer="",
        )
        self.assertFalse(result.audit["schema_valid"])
        self.assertEqual(result.audit["process_denominator"], 3)
        self.assertAlmostEqual(result.process_reward, 1 / 3)

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


class VisionToolCallFormatTest(unittest.TestCase):
    def test_qwen35_native_xml_tool_calls_are_parsed(self):
        from recipes.vision_r1.agent_flow import _one_tool_call
        xml = ("<tool_call>\n<function=inspect_region>\n<parameter=source_artifact_id>\nimage:root:abc\n</parameter>\n"
               "<parameter=bbox_2d>\n[1, 2, 30, 40]\n</parameter>\n<parameter=purpose>\nread digits\n</parameter>\n"
               "</function>\n</tool_call>")
        self.assertEqual(_one_tool_call(xml),
                         ("inspect_region", {"source_artifact_id": "image:root:abc", "bbox_2d": [1, 2, 30, 40],
                                             "purpose": "read digits"}))
        json_call = '<tool_call>{"name": "submit_answer", "arguments": {"answer": "7"}}</tool_call>'
        self.assertEqual(_one_tool_call(json_call), ("submit_answer", {"answer": "7"}))
        self.assertEqual(_one_tool_call(xml + json_call), (None, {}))

    def test_only_vision_allows_reasoning_before_its_tool_call(self):
        from recipes.hotpotqa_a9.prompts import A9_SYSTEM_PROMPT
        from recipes.taco_a9.prompts import SYSTEM_PROMPT as TACO_SYSTEM_PROMPT
        from recipes.vision_r1.prompts import FINAL_TURN_PROMPT, NO_PROSE_SYSTEM_PROMPT, SYSTEM_PROMPT
        no_prose = "Output exactly one tool call per turn and no prose."
        for prompt in (A9_SYSTEM_PROMPT, TACO_SYSTEM_PROMPT, NO_PROSE_SYSTEM_PROMPT):
            self.assertIn(no_prose, prompt)
        self.assertNotIn(no_prose, SYSTEM_PROMPT)
        self.assertIn("reason briefly in plain text, then output", SYSTEM_PROMPT)
        self.assertEqual(SYSTEM_PROMPT.split("\nYou may inspect")[1], NO_PROSE_SYSTEM_PROMPT.split("\nYou may inspect")[1])
        self.assertIn("do not reason", FINAL_TURN_PROMPT)
        self.assertIn("not a resized copy", SYSTEM_PROMPT)

    def test_reasoning_flag_selects_the_system_prompt_and_rejects_other_values(self):
        from recipes.vision_r1.prompts import NO_PROSE_SYSTEM_PROMPT, SYSTEM_PROMPT, system_prompt_template
        with patch.dict(os.environ):
            os.environ.pop("VISION_R1_ALLOW_REASONING", None)
            self.assertIs(system_prompt_template(), SYSTEM_PROMPT)
        for value, expected in (("1", SYSTEM_PROMPT), ("0", NO_PROSE_SYSTEM_PROMPT)):
            with self.subTest(value=value), patch.dict(os.environ, {"VISION_R1_ALLOW_REASONING": value}):
                self.assertIs(system_prompt_template(), expected)
        with patch.dict(os.environ, {"VISION_R1_ALLOW_REASONING": "false"}), self.assertRaises(ValueError):
            system_prompt_template()

    def test_qwen35_tool_env_format_also_accepts_json_bodies(self):
        from agent_r1.env.tool_format import ToolFormatWrapper
        wrapper = ToolFormatWrapper.from_name("qwen35")
        xml = "<tool_call>\n<function=check>\n<parameter=answer>\n7\n</parameter>\n</function>\n</tool_call>"
        json_call = '<tool_call>{"name": "check", "arguments": {"answer": "7"}}</tool_call>'
        string_arguments = '<tool_call>{"name": "check", "arguments": "{\\"answer\\": \\"7\\"}"}</tool_call>'
        for text in (xml, json_call, string_arguments):
            _, calls = wrapper.parse_response(text)
            self.assertEqual([(call.name, call.arguments) for call in calls], [("check", {"answer": "7"})])
        for malformed in ("<tool_call>not json</tool_call>", '<tool_call>{"name": "check"}</tool_call>',
                          '<tool_call>["check"]</tool_call>'):
            self.assertEqual(wrapper.parse_response(malformed)[1], [])


if __name__ == "__main__":
    unittest.main()
