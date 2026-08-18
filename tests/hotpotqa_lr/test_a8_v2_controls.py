from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recipes.hotpotqa_lr.analyze_calibration import main as analyze_calibration
from recipes.hotpotqa_lr.prompts import REASON_STEP_SCHEMA
from recipes.hotpotqa_lr.verifier import (
    artifact_content_sha256,
    trajectory_audit_record,
    verify_trajectory,
)


class A8V2ControlsTest(unittest.TestCase):
    def test_dsl_actor_schema_excludes_noncreditable_select_and_closes_outputs(self):
        branches = REASON_STEP_SCHEMA["oneOf"]
        operations = {branch["properties"]["op"]["const"] for branch in branches}
        self.assertNotIn("select_exact_span", operations)
        self.assertEqual(
            operations,
            {
                "extract_bridge_entity",
                "extract_answer_candidate",
                "normalize_answer",
                "compare_equal",
            },
        )
        for branch in branches:
            output = branch["properties"]["output"]
            self.assertFalse(output["additionalProperties"])
            self.assertEqual(set(output["properties"]), set(output["required"]))

    def test_terminal_only_mode_selects_same_interface_base_arm(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        env["HOTPOTQA_LR_REASON_STEP_FORMAT"] = "dsl"
        env["HOTPOTQA_LR_REWARD_MODE"] = "terminal_only"
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from recipes.hotpotqa_lr.reward_contract import LR_REWARD_ARM; print(LR_REWARD_ARM)",
            ],
            env=env,
            text=True,
        ).strip()
        self.assertEqual(output, "A8_LR_BASE")

    def test_calibration_replays_one_generation_set_for_lr30_and_base(self):
        text = "June Miller was an American writer."
        artifact = {
            "passage:2": {
                "artifact_id": "passage:2",
                "text": text,
                "content_sha256": artifact_content_sha256(text),
            }
        }
        records = []
        for group_index in range(64):
            for rollout_index in range(8):
                if rollout_index < 4:
                    reason_step = {
                        "ref": "r1",
                        "op": "extract_answer_candidate",
                        "premises": [{"artifact_id": "passage:2", "span": text}],
                        "inputs": [],
                        "output": {"answer_candidate": "American"},
                    }
                else:
                    reason_step = {
                        "ref": "r1",
                        "op": "select_exact_span",
                        "premises": [{"artifact_id": "passage:2", "span": text}],
                        "inputs": [],
                        "output": {"text": "American"},
                    }
                transitions = [
                    {
                        "reason_step": reason_step,
                        "action_type": "finish",
                        "action_value": "American",
                        "available_artifacts": artifact,
                    }
                ]
                audit = trajectory_audit_record(
                    verify_trajectory(transitions, question="What nationality was June Miller?")
                )
                records.append(
                    {
                        "sample_index": len(records),
                        "sample_key": f"train:{group_index}",
                        "question": "What nationality was June Miller?",
                        "answer": "American",
                        "score": 0.0,
                        "num_turns": 1,
                        "search_queries": [],
                        "local_reasoning_transitions": transitions,
                        "local_reasoning_audit": audit,
                    }
                )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rollouts = root / "0.jsonl"
            manifest = root / "run_manifest.json"
            report = root / "calibration_report.json"
            rollouts.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "run_mode": "frozen_policy_calibration",
                        "scientific_config": {"calibration_seed": 42},
                        "model": {"identity_sha256": {"config.json": "model-hash"}},
                        "code_sha256": {},
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "analyze_calibration",
                "--rollouts",
                str(rollouts),
                "--manifest",
                str(manifest),
                "--output",
                str(report),
            ]
            with mock.patch.object(sys, "argv", argv):
                analyze_calibration()
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertFalse(result["offline_reward_replay"]["duplicate_base_generation_required"])
            self.assertEqual(result["metrics"]["online_offline_replay_equality"], 1.0)


if __name__ == "__main__":
    unittest.main()
