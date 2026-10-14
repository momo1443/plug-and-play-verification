"""Official MATH-Vision outcome grading for the legacy validation entrypoint."""

from __future__ import annotations

from typing import Any

from agent_r1.evaluation.pnp_protocol import mathvision_score


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
    return float(mathvision_score(solution_str, ground_truth, (extra_info or {}).get("options", [])))
