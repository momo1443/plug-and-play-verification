"""Fallback scorer; the agent flow owns Vision-R1 trajectory rewards."""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    if data_source == "vision_r1_rl":
        return 0.0
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
