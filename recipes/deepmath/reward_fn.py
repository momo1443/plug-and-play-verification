"""Reward function for DeepMath-103K: terminal exact match via math equivalence.

Uses verl's built-in math_reward which extracts \\boxed{} answers and
performs LaTeX normalization + symbolic comparison.

Fallback: if no \\boxed{} is found, attempts to extract the last number or
expression from the output and compare directly.

Format reward: a small bonus is given when the model outputs \\boxed{}
even if the answer is wrong.  This accelerates format learning so that
GRPO does not waste gradient updates on rollouts that score zero purely
because they omitted the required answer format.

Reward structure (data_source == "deepmath"):
  - \\boxed{} present + correct answer  → 1.0
  - \\boxed{} present + wrong answer    → FORMAT_REWARD (default 0.1)
  - no \\boxed{} + fallback match       → 1.0 (but no format bonus)
  - no \\boxed{} + no match             → 0.0
"""

from __future__ import annotations

import re
from typing import Any

from verl.utils.reward_score import math_reward


_DEEPMATH_DATA_SOURCES = {"deepmath"}

# Small reward for producing \\boxed{} even if the answer is wrong.
# This gives the model a gradient signal for format compliance.
FORMAT_REWARD = 0.1


def _fallback_extract_last_answer(text: str) -> str | None:
    """Try to extract an answer when \\boxed{} is absent.

    Looks for common patterns:
    - "The answer is ..." / "answer is ..."
    - "Therefore, ..." at the end
    - Last standalone number/expression
    """
    if not text:
        return None

    # Try "The answer is ..." / "answer is ..."
    m = re.search(r'(?:[Tt]he )?[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$)', text)
    if m:
        return m.group(1).strip()

    # Try "So the answer is ..." near the end
    m = re.search(r'[Ss]o\s+(?:the )?[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$)', text)
    if m:
        return m.group(1).strip()

    return None


def _has_boxed_answer(text: str) -> bool:
    """Return True if text contains a \\boxed{} expression."""
    return bool(re.search(r'\\boxed\s*\{', text))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """
    Custom reward function for DeepMath.

    - If data_source is "deepmath": use math_reward (LaTeX boxed answer EM).
      Falls back to extracting from "The answer is ..." if no \\boxed{}.
      A format reward is added when \\boxed{} is present but the answer
      is wrong, to accelerate format learning.
    - Otherwise, fall back to verl's default_compute_score.
    """
    if data_source not in _DEEPMATH_DATA_SOURCES:
        from verl.utils.reward_score import default_compute_score
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)

    if not ground_truth or not solution_str:
        return 0.0

    gt_str = str(ground_truth)
    has_boxed = _has_boxed_answer(solution_str)

    # Primary: try math_reward (looks for \boxed{})
    score = math_reward.compute_score(solution_str, gt_str)
    if score > 0:
        return float(score)

    # Fallback: if no \boxed{} found, try "The answer is ..." pattern
    fallback_answer = _fallback_extract_last_answer(solution_str)
    if fallback_answer is not None:
        try:
            if math_reward.is_equiv(fallback_answer, gt_str):
                return 1.0
        except Exception:
            pass
        # Direct string comparison as last resort
        if fallback_answer.strip() == gt_str.strip():
            return 1.0

    # Format reward: small bonus for producing \boxed{} even if wrong.
    # This gives GRPO a positive signal for format compliance so the
    # model learns to write \boxed{} faster instead of being stuck at
    # zero reward on ~80% of rollouts.
    if has_boxed:
        return FORMAT_REWARD

    return 0.0
