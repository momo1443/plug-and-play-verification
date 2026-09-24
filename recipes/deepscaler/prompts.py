"""Prompts for DeepScaleR's multi-turn answer-checking environment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


DEEPSCALER_TOOL_SYSTEM_PROMPT = """You are a mathematical problem solver. Reason carefully and put your final answer in \\boxed{...}.
Before your final response, call check_deepscaler_answer at least once with only your candidate final answer. The tool reports whether that candidate is correct, but never reveals the answer. If it reports an error, recheck your derivation and try another candidate. To call it, emit exactly:
<tool_call>
<function=check_deepscaler_answer>
<parameter=answer>
your candidate
</parameter>
</function>
</tool_call>
After tool feedback, answer the original problem normally and include one final \\boxed{...}."""


def build_agent_messages(raw_prompt: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Preserve the problem while replacing its generic system instruction."""
    messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in raw_prompt
        if str(message.get("role", "")) != "system"
    ]
    return [{"role": "system", "content": DEEPSCALER_TOOL_SYSTEM_PROMPT}, *messages]
