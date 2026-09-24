"""Parse one TACO A9 tool call without trusting malformed model output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recipes.hotpotqa_lr.protocol import extract_tool_calls


@dataclass(frozen=True)
class ToolCallAudit:
    name: str | None
    arguments: dict[str, Any]
    errors: tuple[str, ...]


def parse_tool_call(text: str) -> ToolCallAudit:
    calls = extract_tool_calls(text)
    if len(calls) != 1 or not isinstance(calls[0], dict):
        return ToolCallAudit(None, {}, ("expected_exactly_one_tool_call",))
    name = calls[0].get("name")
    arguments = calls[0].get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return ToolCallAudit(None, {}, ("tool_call_shape_invalid",))
    return ToolCallAudit(name, arguments, ())
