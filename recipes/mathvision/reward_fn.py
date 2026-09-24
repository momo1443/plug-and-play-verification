"""A0 exact-answer reward for MATH-Vision validation."""

from __future__ import annotations

import re
from typing import Any


_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", re.IGNORECASE)
_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*([A-E]|[-+]?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def _normal(value: Any) -> str:
    text = str(value).strip().strip("$ ").replace("\\,", "")
    return text.casefold()


def _prediction(text: str) -> str | None:
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return _normal(boxed[-1])
    answers = _ANSWER_RE.findall(text)
    if answers:
        return _normal(answers[-1])
    # Permit a bare one-token final response while avoiding matches from long
    # chain-of-thought text.
    tail = text.strip().splitlines()[-1].strip(" .:()[]")
    if re.fullmatch(r"[A-Ea-e]|[-+]?\d+(?:\.\d+)?", tail):
        return _normal(tail)
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    if data_source != "mathvision" or not solution_str or ground_truth is None:
        return 0.0
    if "<tool_call>" in solution_str or "<tool_response>" in solution_str:
        return 0.0
    return float(_prediction(solution_str) == _normal(ground_truth))
