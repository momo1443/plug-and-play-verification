"""Terminal exact-match scorer for the A8-LR finish protocol."""

from __future__ import annotations

import re
import string
from typing import Any

from recipes.hotpotqa.reward_fn import _extract_answer_from_solution
from recipes.hotpotqa_lr.protocol import parse_finish

_LOCAL_EM_DATA_SOURCES = {
    "hotpotqa_distractor",
    "2wikimultihopqa",
    "musique",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
}


def normalize_answer(value: str) -> str:
    lowered = str(value).lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


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
    candidates = _ground_truths(ground_truth, extra_info)
    if not candidates:
        return 0.0
    finish = parse_finish(solution_str or "")
    if not finish.envelope_valid or finish.answer is None:
        # Validation-time lenient fallback: if the finish tool call is
        # malformed or missing but the model produced an <answer> tag or
        # other recognisable answer text, score that instead so that
        # validation EM is not penalised by protocol formatting failures.
        is_validation = bool((extra_info or {}).get("_agent_r1_is_validation", False))
        if is_validation:
            answer = _extract_answer_from_solution(solution_str or "")
            if answer:
                prediction = normalize_answer(answer)
                return float(prediction in {normalize_answer(candidate) for candidate in candidates})
        return 0.0
    prediction = normalize_answer(finish.answer)
    return float(prediction in {normalize_answer(candidate) for candidate in candidates})
