"""Terminal exact-match scorer for the A9 certificate-grounded finish protocol."""

from __future__ import annotations

from typing import Any

from recipes.hotpotqa_a9.protocol import parse_finish_call
from recipes.hotpotqa_lr.reward_fn import normalize_answer

_LOCAL_EM_DATA_SOURCES = {
    "hotpotqa_distractor",
    "2wikimultihopqa",
    "musique",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
}


def _ground_truths(ground_truth: Any, extra_info: dict | None) -> list[str]:
    values: list[Any] = []
    if isinstance(ground_truth, (list, tuple, set)):
        values.extend(ground_truth)
    elif ground_truth is not None:
        values.append(ground_truth)
    if isinstance(extra_info, dict):
        aliases = extra_info.get("answers")
        if isinstance(aliases, (list, tuple, set)):
            values.extend(aliases)
        elif aliases is not None:
            values.append(aliases)
    return [str(value).strip() for value in values if str(value).strip()]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    if data_source not in _LOCAL_EM_DATA_SOURCES:
        from verl.utils.reward_score import default_compute_score

        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    finish = parse_finish_call(solution_str or "")
    if finish.answer is None:
        return 0.0
    candidates = _ground_truths(ground_truth, extra_info)
    if not candidates:
        return 0.0
    prediction = normalize_answer(finish.answer)
    return float(prediction in {normalize_answer(candidate) for candidate in candidates})
