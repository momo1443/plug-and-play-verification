"""Execute actual flow methods with mocked GPU/runtime boundaries.

AST loading omits only imports so these wiring tests run without Ray/veRL on
CPU. Generation, environment execution and final-answer equivalence are test
doubles; process judging, scheduling, reward composition and run() are real.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import sys
import unittest
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

from PIL import Image

from agent_r1.verifier import (
    ComposedVerificationReward,
    UniformRewardSchedule,
    VerificationCredit,
    VerificationResult,
    apply_composed_reward,
    compose_verification_reward,
    uniform_reward_schedule,
)
from recipes.llm_judge.scoring import (
    ProcessStep,
    image_data_url,
    judge_reward_info,
    question_from_raw_prompt,
    verify_process_steps,
)
from recipes.reward_mixing import prompt_group_key_from_extra_info
from recipes.taco_a9.dsl import CodeArtifact, ExecutionRecord
from recipes.taco_a9.protocol import parse_tool_call
from recipes.taco_a9.reward_contract import reward_schedule as taco_schedule
from recipes.taco_a9.sandbox import TestCase, TestSuite

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def simple_timer(*args):
    yield


def register(name):
    return lambda cls: cls


class RuntimeBoundary:
    async def _postprocess(self, step, **kwargs):
        return step

    async def apply_chat_template(self, *args, **kwargs):
        return [1]

    async def _obs_to_prompt(self, *args, **kwargs):
        return [1]

    async def process_vision_info(self, messages):
        return {"images": [self.test_image]}


def load_source(relative, **injections):
    """Compile production definitions unchanged, replacing external imports."""
    name = "_process_judge_test_" + relative.replace("/", "_").replace(".", "_")
    module = ModuleType(name)
    module.__dict__.update(globals())
    module.__dict__.update(
        {
            "hashlib": hashlib,
            "logging": logging,
            "os": os,
            "Mapping": Mapping,
            "Sequence": Sequence,
            "dataclass": dataclass,
            "Any": Any,
            "uuid4": uuid4,
            "ComposedVerificationReward": ComposedVerificationReward,
            "UniformRewardSchedule": UniformRewardSchedule,
            "VerificationCredit": VerificationCredit,
            "VerificationResult": VerificationResult,
            "apply_composed_reward": apply_composed_reward,
            "compose_verification_reward": compose_verification_reward,
            "uniform_reward_schedule": uniform_reward_schedule,
            "ProcessStep": ProcessStep,
            "image_data_url": image_data_url,
            "judge_reward_info": judge_reward_info,
            "question_from_raw_prompt": question_from_raw_prompt,
            "verify_process_steps": verify_process_steps,
            "prompt_group_key_from_extra_info": prompt_group_key_from_extra_info,
            "ExecutionRecord": ExecutionRecord,
            "parse_tool_call": parse_tool_call,
            "TestCase": TestCase,
        }
    )
    module.__dict__.update(
        {
            "__name__": name,
            "AgentFlowBase": RuntimeBoundary,
            "AgentEnvLoop": RuntimeBoundary,
            "AgentFlowStep": SimpleNamespace,
            "AgentFlowOutput": SimpleNamespace,
            "Action": SimpleNamespace,
        }
    )
    module.__dict__.update(injections)
    sys.modules[name] = module
    tree = ast.parse((ROOT / relative).read_text(), filename=relative)
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    exec(compile(ast.fix_missing_locations(tree), relative, "exec"), module.__dict__)
    return module


tool_format = load_source(
    "agent_r1/env/tool_format.py",
    **{
        "ABC": __import__("abc").ABC,
        "abstractmethod": __import__("abc").abstractmethod,
        "re": __import__("re"),
    },
)
visual_artifacts = load_source("recipes/vision_r1/artifacts.py")
vision_prompts = load_source(
    "recipes/vision_r1/prompts.py", CERTIFICATE_SCHEMA_VERSION=visual_artifacts.CERTIFICATE_SCHEMA_VERSION
)
taco_prompts = load_source("recipes/taco_a9/prompts.py")
vision_contract = load_source("recipes/vision_r1/reward_contract.py", is_equiv=lambda a, b: a == b)
math_contract = load_source(
    "recipes/deepscaler/trajectory_reward.py",
    compute_terminal_em=lambda source, response, target: float(response == f"FINAL {target}"),
    _process_reward_audit=lambda text: {},
)


class FakeJudge:
    def __init__(self, score=1.0):
        self.score = score
        self.calls = []

    async def judge_structured(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.score is None:
            return None
        return SimpleNamespace(value=self.score, input_hash="hash", cache_hit=False, attempts=1, latency_s=0.0)


class FakeGeneration:
    def __init__(self, texts):
        self.texts = texts
        self.index = 0

    async def generate(self, **kwargs):
        index = self.index
        self.index += 1
        return SimpleNamespace(token_ids=[index], log_probs=None)

    def decode(self, ids, **kwargs):
        return self.texts[ids[0]]


def call(name, **arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


class ProcessJudgeFlowTest(unittest.IsolatedAsyncioTestCase):
    async def run_domain(
        self,
        domain,
        *,
        correct=True,
        validation=False,
        update=51,
        incomplete=False,
        failure=False,
        direct=False,
        developer_test=False,
    ):
        judge = FakeJudge(None if failure else 1.0)
        if domain == "deepmath":
            module = load_source(
                "recipes/deepscaler/tool_agent_flow.py",
                TrajectoryReward=math_contract.TrajectoryReward,
                compute_tool_trajectory_reward=math_contract.compute_tool_trajectory_reward,
            )
            flow = object.__new__(module.DeepScalerToolRewardAgentFlow)
            texts = ["Intermediate deduction", "FINAL correct" if correct else "FINAL wrong"]
            if incomplete:
                texts = ["Intermediate deduction", "Another intermediate deduction"]

            class Env:
                tool_schemas = []
                format_wrapper = SimpleNamespace(parse_response=lambda text: (text, []))

                def reset(self, **kwargs):
                    return SimpleNamespace(messages=[{"content": "task"}])

                async def step(self, action):
                    return (
                        SimpleNamespace(messages=[{"content": "observed feedback"}]),
                        None,
                        action.text.startswith("FINAL"),
                        {},
                    )

            flow._create_env = lambda **kwargs: Env()
            flow.em_warmup_steps = 50
            flow.skip_special_tokens = True
        elif domain == "taco":
            module = load_source(
                "recipes/taco_a9/agent_flow.py",
                reward_schedule=taco_schedule,
                SYSTEM_PROMPT=taco_prompts.SYSTEM_PROMPT,
                USER_PROMPT=taco_prompts.USER_PROMPT,
                FINAL_TURN_PROMPT=taco_prompts.FINAL_TURN_PROMPT,
                SUBMIT_ONLY_TOOLS=taco_prompts.SUBMIT_ONLY_TOOLS,
                WRITE_ONLY_TOOLS=taco_prompts.WRITE_ONLY_TOOLS,
                WRITE_OR_RUN_OR_SUBMIT_TOOLS=taco_prompts.WRITE_OR_RUN_OR_SUBMIT_TOOLS,
            )
            flow = object.__new__(module.TacoA9CodeAgentFlow)
            artifact = CodeArtifact.create(1, "print(1)")
            texts = [call("write_code", code="print(1)"), call("submit", code_artifact_id=artifact.artifact_id)]
            if incomplete:
                texts[1] = call("write_code", code="print(2)")
            flow.store = SimpleNamespace(suites=lambda task: (TestSuite("dev", ()), TestSuite("private", ())))
            if developer_test:
                texts.insert(1, call("run_developer_tests", code_artifact_id=artifact.artifact_id))

            async def run_suite(code, suite, ordinal, metrics):
                record = ExecutionRecord(
                    run_id=f"run:{ordinal}",
                    code_artifact_id=code.artifact_id,
                    code_sha256=code.sha256,
                    suite_id=suite.suite_id,
                    suite_sha256="suite",
                    result_sha256="result",
                    passed=int(correct),
                    total=1,
                    all_passed=correct,
                    timed_out=False,
                )
                hidden = "PRIVATE-SECRET" if suite.suite_id == "private" else "developer-example"
                return record, {"outcomes": [{"passed": correct, "case_index": 0, "stdin": hidden}]}

            flow._run_suite = run_suite
            flow.max_code_prompt_chars = 12000
            flow.terminal_warmup_steps = 50
        else:
            module = load_source(
                "recipes/vision_r1/agent_flow.py",
                ToolFormatWrapper=tool_format.ToolFormatWrapper,
                VisualArtifact=visual_artifacts.VisualArtifact,
                outcome_reward=vision_contract.outcome_reward,
                reward_schedule=vision_contract.reward_schedule,
                SYSTEM_PROMPT=vision_prompts.SYSTEM_PROMPT,
                FINAL_TURN_PROMPT=vision_prompts.FINAL_TURN_PROMPT,
                INSPECT_OR_SUBMIT_TOOLS=vision_prompts.INSPECT_OR_SUBMIT_TOOLS,
                SUBMIT_ONLY_TOOLS=vision_prompts.SUBMIT_ONLY_TOOLS,
            )
            flow = object.__new__(module.VisionR1VisualAgentFlow)
            flow.test_image = Image.new("RGB", (100, 100), "blue")
            root = visual_artifacts.VisualArtifact.root(flow.test_image)
            texts = [
                call(
                    "inspect_region",
                    source_artifact_id=root.artifact_id,
                    purpose="Read relevant labels",
                    bbox_2d=[0, 0, 60, 60],
                ),
                call("submit_answer", answer="correct" if correct else "wrong"),
            ]
            if incomplete:
                texts[1] = "invalid action"
            flow.terminal_warmup_steps = 50
        if direct:
            texts = texts[-1:]
        flow.max_steps = max(2, len(texts))
        flow.prompt_length = 10000
        flow.response_length = 1000
        flow.reward_mode = "llm_judge"
        flow.judge_server = judge
        flow.loop = asyncio.get_running_loop()
        generation = FakeGeneration(texts)
        flow.server_manager = generation
        flow.tokenizer = generation
        output = await flow.run(
            {},
            raw_prompt=[{"role": "user", "content": "Task without hidden answer"}],
            reward_model={"ground_truth": "correct"},
            extra_info={"question_id": "q1"},
            data_source=domain,
            _agent_r1_global_step=update,
            _agent_r1_is_validation=validation,
        )
        return output, judge

    async def test_three_domains_preserve_binary_outcomes_and_place_process_rewards(self):
        for domain in ("deepmath", "taco", "vision_r1"):
            for correct in (False, True):
                with self.subTest(domain=domain, correct=correct):
                    output, judge = await self.run_domain(domain, correct=correct)
                    first, final = output.steps
                    info = final.extra_fields["reward_extra_info"]
                    self.assertEqual(info["terminal_reward"], float(correct))
                    self.assertEqual(info["acc"], float(correct))
                    self.assertGreater(first.reward_score, 0.0)
                    self.assertEqual(final.reward_score, info["optimizer_terminal_weight"] * float(correct))
                    self.assertAlmostEqual(sum(s.reward_score for s in output.steps), info["optimizer_total_reward"])
                    self.assertEqual(len(judge.calls), 1)
                    self.assertNotIn("FINAL", judge.calls[0][0])
                    self.assertNotIn("submit_answer", judge.calls[0][0])
                    self.assertNotIn("private", judge.calls[0][0].split("### Task")[1])
                    if domain == "vision_r1":
                        self.assertEqual(len(judge.calls[0][1]["image_urls"]), 2)

    async def test_warmup_and_validation_skip_judge_and_keep_binary_outcome(self):
        for domain in ("deepmath", "taco", "vision_r1"):
            for settings in ({"update": 50}, {"validation": True}):
                output, judge = await self.run_domain(domain, **settings)
                self.assertEqual(judge.calls, [])
                self.assertEqual([s.reward_score for s in output.steps], [0.0, 1.0])

    async def test_incomplete_trajectories_keep_intermediate_credit_and_zero_outcome(self):
        for domain in ("deepmath", "taco", "vision_r1"):
            output, judge = await self.run_domain(domain, incomplete=True)
            self.assertGreater(output.steps[0].reward_score, 0.0)
            self.assertEqual(output.steps[-1].extra_fields["reward_extra_info"]["terminal_reward"], 0.0)
            self.assertTrue(judge.calls)

    async def test_direct_submission_has_no_process_reward(self):
        for domain in ("deepmath", "taco", "vision_r1"):
            output, judge = await self.run_domain(domain, direct=True)
            info = output.steps[-1].extra_fields["reward_extra_info"]
            self.assertEqual(judge.calls, [])
            self.assertEqual(info["process_reward"], 0.0)
            # Direct code submission refers to a nonexistent artifact and must fail.
            self.assertEqual(info["terminal_reward"], 0.0 if domain == "taco" else 1.0)

    async def test_developer_test_credit_uses_observed_feedback_without_private_tests(self):
        output, judge = await self.run_domain("taco", correct=False, developer_test=True)
        self.assertEqual(len(judge.calls), 2)
        self.assertIn("developer-example", judge.calls[1][0])
        self.assertNotIn("PRIVATE-SECRET", str(judge.calls))
        self.assertNotIn("developer-example", judge.calls[0][0])
        self.assertGreater(output.steps[1].reward_score, 0.0)
        self.assertEqual(output.steps[-1].reward_score, 0.0)

    async def test_judge_failure_is_reported_to_existing_trainer_gate(self):
        for domain in ("deepmath", "taco", "vision_r1"):
            output, _ = await self.run_domain(domain, failure=True)
            info = output.steps[-1].extra_fields["reward_extra_info"]
            self.assertTrue(info["judge_invalid"])
            self.assertEqual(info["terminal_reward"], 1.0)
            self.assertEqual(info["process_reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
