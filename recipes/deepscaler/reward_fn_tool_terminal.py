"""Strict terminal answer reward for DeepScaleR ToolEnv training."""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score, math_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Award one only when the final model response matches the math answer."""
    if data_source != "deepmath":
        return float(default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs))
    if not solution_str or ground_truth is None:
        return 0.0
    return float(math_reward.compute_score(solution_str, str(ground_truth)) > 0)
