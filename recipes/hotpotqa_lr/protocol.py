"""Tool-call parsing for the standalone LR actor contract."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

LR_FINISH_PROTOCOL = "hotpotqa-local-reasoning-finish-v1"

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_HERMES_FUNCTION = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL)
_HERMES_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def visible_completion(text: str) -> str:
    value = str(text or "")
    lowered = value.lower()
    close = lowered.rfind("</think>")
    if close >= 0:
        value = value[close + len("</think>") :]
    return value.strip()


def _normalize_literal(obj: Any) -> Any:
    """Recursively convert sets from ast.literal_eval to sorted lists."""
    if isinstance(obj, set):
        return sorted((_normalize_literal(item) for item in obj), key=str)
    if isinstance(obj, dict):
        return {str(k): _normalize_literal(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_literal(item) for item in obj]
    return obj


def _structured(value: str) -> Any:
    raw = value.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        pass
    if len(raw) > 16_384 or "__" in raw:
        return None
    try:
        result = ast.literal_eval(raw)
        return _normalize_literal(result)
    except (SyntaxError, ValueError):
        return None


def decode_arguments(arguments: Any) -> dict[str, Any] | None:
    value = _structured(arguments) if isinstance(arguments, str) else arguments
    if isinstance(value, str):
        value = _structured(value)
    return dict(value) if isinstance(value, Mapping) else None


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Recover JSON and Hermes XML calls while preserving nested JSON values."""

    calls: list[dict[str, Any]] = []
    for body in _TOOL_CALL_BLOCK.findall(visible_completion(text)):
        payload = _structured(body)
        if isinstance(payload, Mapping):
            arguments = decode_arguments(payload.get("arguments"))
            if isinstance(payload.get("name"), str) and arguments is not None:
                calls.append({"name": payload["name"], "arguments": arguments})
                continue
        for match in _HERMES_FUNCTION.finditer(body):
            arguments: dict[str, Any] = {}
            for parameter in _HERMES_PARAMETER.finditer(match.group(2)):
                name = parameter.group(1).strip()
                raw_value = parameter.group(2).strip()
                parsed = _structured(raw_value)
                arguments[name] = raw_value if parsed is None else parsed
            calls.append({"name": match.group(1).strip(), "arguments": arguments})
    return calls


@dataclass(frozen=True)
class FinishAudit:
    envelope_valid: bool
    answer: str | None
    reason_step: Any
    errors: tuple[str, ...]


def parse_finish(text: str) -> FinishAudit:
    """Parse one exact finish call; reason-step validity is audited separately."""

    visible = visible_completion(text)
    envelope = _TOOL_CALL_BLOCK.fullmatch(visible)
    if envelope is None:
        return FinishAudit(False, None, None, ("not_exactly_one_tool_call",))
    calls = extract_tool_calls(visible)
    if len(calls) != 1 or calls[0].get("name") != "finish":
        return FinishAudit(False, None, None, ("finish_call_unparseable",))
    arguments = calls[0]["arguments"]
    errors: list[str] = []
    if set(arguments) != {"status", "answer", "reason_step"}:
        errors.append("finish_argument_fields_invalid")
    if arguments.get("status") != "answer":
        errors.append("finish_status_invalid")
    answer = arguments.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("answer_not_nonempty_string")
    return FinishAudit(
        envelope_valid=not errors,
        answer=answer.strip() if isinstance(answer, str) and answer.strip() else None,
        reason_step=arguments.get("reason_step"),
        errors=tuple(errors),
    )
