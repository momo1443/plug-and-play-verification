"""Regression tests for the DeepScaleR ToolEnv adapter."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Keep this test runnable both through pytest and as ``python tests/...py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recipes.deepscaler.prepare_tool_data import build_tool_dataset
from recipes.deepscaler.reward_fn_tool_terminal import compute_score
from recipes.deepscaler.tool import DeepScalerAnswerCheckTool
from agent_r1.env.base import Action
from agent_r1.env.envs.tool import ToolEnv


class DeepScalerToolTest(unittest.TestCase):
    def test_answer_check_does_not_expose_hidden_answer(self) -> None:
        tool = DeepScalerAnswerCheckTool()
        response, reward, extra = asyncio.run(
            tool.run({"answer": "3/7"}, tools_kwargs={"ground_truth": "\\frac{3}{7}"})
        )
        self.assertEqual(response.text, "Candidate answer is correct.")
        self.assertIsNone(reward)
        self.assertEqual(extra, {"candidate_correct": True})
        self.assertNotIn("frac", response.text)

    def test_tool_env_returns_feedback_to_a_follow_up_turn(self) -> None:
        env = ToolEnv(
            tools=["check_deepscaler_answer"],
            tools_kwargs={"ground_truth": "\\frac{3}{7}"},
        )
        env.reset(raw_prompt=[{"role": "user", "content": "Solve the problem."}])
        observation, reward, done, _ = asyncio.run(
            env.step(
                Action(
                    text='<tool_call>{"name":"check_deepscaler_answer","arguments":{"answer":"3/7"}}</tool_call>'
                )
            )
        )
        self.assertIsNone(reward)
        self.assertFalse(done)
        self.assertIn("Candidate answer is correct.", observation.messages[-1]["content"])
        _, _, final_done, _ = asyncio.run(env.step(Action(text="The final answer is \\boxed{3/7}.")))
        self.assertTrue(final_done)

    def test_terminal_reward_is_strict_exact_match(self) -> None:
        self.assertEqual(compute_score("deepmath", "\\boxed{3/7}", "\\frac{3}{7}"), 1.0)
        self.assertEqual(compute_score("deepmath", "\\boxed{2/7}", "\\frac{3}{7}"), 0.0)

    def test_preparation_keeps_truth_out_of_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.parquet"
            output = root / "tool.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "data_source": "deepmath",
                            "prompt": [
                                {"role": "system", "content": "old instruction"},
                                {"role": "user", "content": "Solve x + 1 = 2."},
                            ],
                            "reward_model": {"ground_truth": "SECRET-GT", "style": "rule"},
                            "extra_info": {"index": 0},
                        }
                    ]
                ),
                source,
            )
            self.assertEqual(build_tool_dataset(source, output), (1, 0))
            row = pq.read_table(output).to_pylist()[0]
            self.assertEqual(row["agent_name"], "deepscaler_tool")
            self.assertNotIn("SECRET-GT", json.dumps(row["prompt"]))
            self.assertEqual(json.loads(row["env_kwargs"])["tools_kwargs"]["ground_truth"], "SECRET-GT")


if __name__ == "__main__":
    unittest.main()
