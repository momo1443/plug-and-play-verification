"""Fallback scorer for TACO A9.

The agent flow computes private-test terminal rewards directly because private
tests must remain absent from model-visible parquet records. This scorer exists
only to keep the standard custom-reward configuration explicit and fails closed
for malformed fallback use.
"""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None, **kwargs) -> float:
    if data_source == "taco_a9_python":
        return 0.0
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
