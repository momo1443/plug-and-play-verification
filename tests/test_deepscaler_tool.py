"""Regression tests for the DeepScaleR ToolEnv adapter."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

# Keep this test runnable both through pytest and as ``python tests/...py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_r1.agent_flow.agent_flow import AgentFlowStep
from agent_r1.env.base import Action
from agent_r1.env.envs.tool import ToolEnv
from agent_r1.env.tool_format import ToolFormatWrapper
from recipes.deepscaler import reward_fn as single_turn_a9_reward
from recipes.deepscaler.prepare_tool_data import build_tool_dataset
from recipes.deepscaler.prepare_tool_pair_run import build_manifest
from recipes.deepscaler.reward_fn_tool_terminal import compute_score
from recipes.deepscaler.tool import DeepScalerAnswerCheckTool
from recipes.deepscaler.tool_agent_flow import _apply_trajectory_reward
from recipes.deepscaler.trajectory_reward import compute_tool_trajectory_reward


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
                Action(text='<tool_call>{"name":"check_deepscaler_answer","arguments":{"answer":"3/7"}}</tool_call>')
            )
        )
        self.assertIsNone(reward)
        self.assertFalse(done)
        self.assertIn("Candidate answer is correct.", observation.messages[-1]["content"])
        _, _, final_done, _ = asyncio.run(env.step(Action(text="The final answer is \\boxed{3/7}.")))
        self.assertTrue(final_done)

    def test_qwen35_native_tool_call_returns_feedback(self) -> None:
        env = ToolEnv(
            tools=["check_deepscaler_answer"],
            tool_format="qwen35",
            tools_kwargs={"ground_truth": "\\frac{3}{7}"},
        )
        env.reset(raw_prompt=[{"role": "user", "content": "Solve the problem."}])
        native_call = """<tool_call>
<function=check_deepscaler_answer>
<parameter=answer>
3/7
</parameter>
</function>
</tool_call>"""
        observation, reward, done, _ = asyncio.run(env.step(Action(text=native_call)))
        self.assertIsNone(reward)
        self.assertFalse(done)
        self.assertIn("Candidate answer is correct.", observation.messages[-1]["content"])

    def test_qwen35_parser_fails_closed_on_hybrid_tool_call(self) -> None:
        wrapper = ToolFormatWrapper.from_name("qwen35")
        _, calls = wrapper.parse_response(
            '<tool_call><function=check_deepscaler_answer", "arguments": '
            '{"answer": "3/7"}}</tool_call>'
        )
        self.assertEqual(calls, [])

    def test_terminal_reward_is_strict_exact_match(self) -> None:
        self.assertEqual(compute_score("deepmath", "\\boxed{3/7}", "\\frac{3}{7}"), 1.0)
        self.assertEqual(compute_score("deepmath", "\\boxed{2/7}", "\\frac{3}{7}"), 0.0)

    def test_single_turn_a9_warmup_does_not_reward_wrong_boxed_answers(self) -> None:
        result = single_turn_a9_reward.compute_score(
            "deepmath",
            "\\boxed{8}",
            "7",
            {"_agent_r1_global_step": single_turn_a9_reward.EM_WARMUP_STEPS},
        )
        self.assertEqual(result["optimizer_reward_phase"], "em_warmup")
        self.assertEqual(result["terminal_em"], 0.0)
        self.assertEqual(result["score"], 0.0)

    def test_paired_a1_and_a9_reward_contracts(self) -> None:
        final = "The answer is \\boxed{7}."
        reasoning = [(1, "We compute 3+4=7.")]
        a1 = compute_tool_trajectory_reward(
            reward_mode="terminal_only",
            final_response=final,
            ground_truth="7",
            reasoning_segments=reasoning,
            global_step=100,
            is_validation=False,
            em_warmup_steps=50,
        )
        self.assertEqual(a1.score, 1.0)
        self.assertEqual(a1.process_weight, 0.0)

        warmup = compute_tool_trajectory_reward(
            reward_mode="uniform_equation_process",
            final_response=final,
            ground_truth="7",
            reasoning_segments=reasoning,
            global_step=50,
            is_validation=False,
            em_warmup_steps=50,
        )
        expected_warmup = a1.record()
        expected_warmup["optimizer_reward_phase"] = "em_warmup"
        self.assertEqual(warmup.record(), expected_warmup)

        mixed = compute_tool_trajectory_reward(
            reward_mode="uniform_equation_process",
            final_response="\\boxed{8}",
            ground_truth="7",
            reasoning_segments=reasoning,
            global_step=51,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="deepmath:train:index:5",
        )
        self.assertEqual(mixed.terminal_em, 0.0)
        self.assertEqual(mixed.process_reward, 1.0)
        self.assertEqual(mixed.score, mixed.process_weight)
        self.assertEqual(mixed.terminal_component, 0.0)
        self.assertEqual(mixed.process_step_components, {1: mixed.process_weight})
        self.assertEqual(len(mixed.process_segment_audits), 1)

    def test_a9_requires_a_final_answer_and_validation_is_terminal_only(self) -> None:
        no_final = compute_tool_trajectory_reward(
            reward_mode="uniform_equation_process",
            final_response=None,
            ground_truth="7",
            reasoning_segments=[(1, "3+4=7")],
            global_step=51,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="deepmath:train:index:6",
        )
        self.assertEqual(no_final.score, 0.0)
        self.assertEqual(no_final.process_reward, 1.0)
        self.assertEqual(no_final.process_step_components, {1: 0.0})

        validation = compute_tool_trajectory_reward(
            reward_mode="uniform_equation_process",
            final_response="\\boxed{8}",
            ground_truth="7",
            reasoning_segments=[(1, "3+4=7")],
            global_step=51,
            is_validation=True,
            em_warmup_steps=50,
        )
        self.assertEqual(validation.score, 0.0)
        self.assertEqual(validation.process_weight, 0.0)

    def test_a9_process_reward_is_backfilled_to_its_reasoning_turns(self) -> None:
        reward = compute_tool_trajectory_reward(
            reward_mode="uniform_equation_process",
            final_response="The answer is \\boxed{4}.",
            ground_truth="4",
            reasoning_segments=[(1, "We compute 1+1=2."), (3, "Then 2+2=4.")],
            global_step=51,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="deepmath:train:index:causal",
        )

        self.assertEqual(set(reward.process_step_components), {1, 3})
        self.assertAlmostEqual(reward.process_step_components[1], reward.process_weight / 2)
        self.assertAlmostEqual(reward.process_step_components[3], reward.process_weight / 2)
        self.assertAlmostEqual(reward.terminal_component, reward.terminal_weight)
        self.assertAlmostEqual(
            reward.score,
            reward.terminal_component + sum(reward.process_step_components.values()),
        )

        steps = [
            AgentFlowStep(prompt_ids=[turn], response_ids=[turn], reward_score=0.0)
            for turn in range(1, 4)
        ]
        _apply_trajectory_reward(steps, reward, tool_call_count=1, max_steps=5)

        self.assertAlmostEqual(steps[0].reward_score, reward.process_weight / 2)
        self.assertEqual(steps[1].reward_score, 0.0)
        self.assertAlmostEqual(
            steps[2].reward_score,
            reward.terminal_weight + reward.process_weight / 2,
        )
        self.assertAlmostEqual(sum(step.reward_score for step in steps), reward.score)
        self.assertEqual(
            steps[0].extra_fields["reward_extra_info"]["optimizer_process_component"],
            steps[0].reward_score,
        )
        self.assertAlmostEqual(
            steps[2].extra_fields["reward_extra_info"]["optimizer_process_component"],
            reward.process_weight / 2,
        )

    def test_a9_weight_is_shared_within_prompt_group(self) -> None:
        kwargs = {
            "reward_mode": "uniform_equation_process",
            "final_response": "The answer is \\boxed{7}.",
            "ground_truth": "7",
            "reasoning_segments": [(1, "We compute 3+4=7.")],
            "global_step": 51,
            "is_validation": False,
            "em_warmup_steps": 50,
        }
        first = compute_tool_trajectory_reward(**kwargs, prompt_group_key="deepmath:train:index:7")
        second = compute_tool_trajectory_reward(**kwargs, prompt_group_key="deepmath:train:index:7")
        other = compute_tool_trajectory_reward(**kwargs, prompt_group_key="deepmath:train:index:8")

        self.assertEqual(first.terminal_weight, second.terminal_weight)
        self.assertNotEqual(first.terminal_weight, other.terminal_weight)
        self.assertEqual(first.weight_sampling, "uniform_0_1_per_prompt_group")
        self.assertEqual(first.prompt_group_key, "deepmath:train:index:7")

    def test_paired_agent_config_has_two_explicit_reward_arms(self) -> None:
        configs = OmegaConf.load(Path(__file__).resolve().parents[1] / "recipes/deepscaler/tool_reward_arms.yaml")
        self.assertEqual([config.name for config in configs], ["deepscaler_tool_a1", "deepscaler_tool_a9"])
        self.assertEqual(configs[0].reward_mode, "terminal_only")
        self.assertEqual(configs[1].reward_mode, "uniform_equation_process")

    def test_paired_manifest_locks_the_common_runtime_and_reward_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            parquet_row = {
                "data_source": "deepmath",
                "prompt": [{"role": "user", "content": "Solve."}],
                "reward_model": {"ground_truth": "1", "style": "rule"},
            }
            train_path = root / "train.parquet"
            validation_path = root / "validation.parquet"
            pq.write_table(pa.Table.from_pylist([parquet_row]), train_path)
            pq.write_table(pa.Table.from_pylist([parquet_row]), validation_path)
            manifest = build_manifest(
                argparse.Namespace(
                    project_dir=str(Path(__file__).resolve().parents[1]),
                    output_dir=str(root / "output"),
                    run_id="paired-test",
                    arm="A9",
                    model_path=str(model),
                    train_path=str(train_path),
                    validation_path=str(validation_path),
                    train_batch_size=20,
                    train_max_samples=30000,
                    total_training_steps=1500,
                    rollout_n=4,
                    seed=42,
                    max_steps=5,
                    em_warmup_steps=50,
                    max_prompt_length=4096,
                    max_response_length=2048,
                    cuda_visible_devices="0,1,4,5,6,7",
                    num_gpus=6,
                )
            )
            self.assertEqual(manifest["reward_contract"]["warmup_steps"], 50)
            self.assertEqual(
                manifest["reward_contract"]["optimizer_placement"],
                "process_credit_source_steps_terminal_credit_final_step",
            )
            self.assertTrue((root / "output" / "run_manifest.json").is_file())

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

    def test_preparation_can_defer_agent_selection_to_the_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.parquet"
            output = root / "tool.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "data_source": "deepmath",
                            "prompt": [{"role": "user", "content": "Solve."}],
                            "reward_model": {"ground_truth": "1", "style": "rule"},
                        }
                    ]
                ),
                source,
            )
            build_tool_dataset(source, output, agent_name=None)
            self.assertNotIn("agent_name", pq.read_table(output).column_names)


if __name__ == "__main__":
    unittest.main()
