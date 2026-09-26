"""Strict terminal answer reward for DeepScaleR ToolEnv training."""

from __future__ import annotations

from typing import Any

from recipes.deepscaler.reward_fn import compute_terminal_em


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Award one only when the final model response matches the math answer."""
    if data_source != "deepmath":
        from verl.utils.reward_score import default_compute_score
        return float(default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs))
    if not solution_str or ground_truth is None:
        return 0.0
    return compute_terminal_em(solution_str, str(ground_truth))
