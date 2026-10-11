"""Execute actual flow methods with mocked GPU/runtime boundaries.

AST loading omits only imports so these wiring tests run without Ray/veRL on
CPU. Generation, environment execution and final-answer equivalence are test
doubles; deterministic verification, scheduling and reward composition are real.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

from agent_r1.verifier import (
    ComposedVerificationReward,
    UniformRewardSchedule,
    VerificationCredit,
    VerificationGroup,
    VerificationResult,
    apply_composed_reward,
    compose_verification_reward,
    normalize_verification_groups,
    uniform_reward_schedule,
)
from recipes.llm_judge.scoring import question_from_raw_prompt
from recipes.reward_mixing import prompt_group_key_from_extra_info
from agent_r1.evaluation.consistency import consistency_record
from recipes.taco_a9.protocol import parse_tool_call
from recipes.taco_a9.dsl import CodeArtifact, ExecutionRecord
from recipes.taco_a9.sandbox import TestCase

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
    name = "_pnp_test_" + relative.replace("/", "_").replace(".", "_")
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
            "VerificationGroup": VerificationGroup,
            "VerificationResult": VerificationResult,
            "apply_composed_reward": apply_composed_reward,
            "compose_verification_reward": compose_verification_reward,
            "normalize_verification_groups": normalize_verification_groups,
            "uniform_reward_schedule": uniform_reward_schedule,
            "question_from_raw_prompt": question_from_raw_prompt,
            "prompt_group_key_from_extra_info": prompt_group_key_from_extra_info,
            "ExecutionRecord": ExecutionRecord,
            "parse_tool_call": parse_tool_call,
            "TestCase": TestCase,
        }
    )
    module.__dict__.update(
        {
            "__name__": name,
            "__file__": str(ROOT / relative),
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
