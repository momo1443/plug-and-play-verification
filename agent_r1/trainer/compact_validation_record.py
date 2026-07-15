"""Extract the useful, non-repeated fields from a HotpotQA trajectory."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_USER_QUERY_BLOCK = re.compile(
    r"(?:^|\n)### User Query\s*\n(.*?)(?=\n\n### |\Z)", re.DOTALL
)
_SEARCH_FUNCTION_BLOCK = re.compile(
    r"<function=search>(.*?)</function>", re.DOTALL | re.IGNORECASE
)
_QUERY_PARAMETER_BLOCK = re.compile(
    r"<parameter=query>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE
)


def extract_question(raw_prompt: Any, decoded_input: str = "") -> str:
    """Get the original user question without persisting the rendered prompt."""

    if isinstance(raw_prompt, dict):
        content = raw_prompt.get("content")
        if content is not None:
            return str(content).strip()

    if isinstance(raw_prompt, (list, tuple)):
        for message in reversed(raw_prompt):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user" and message.get("content") is not None:
                return str(message["content"]).strip()

    match = _USER_QUERY_BLOCK.search(decoded_input)
    if match:
        return match.group(1).strip()
    return ""


def extract_search_queries(output_text: str) -> list[str]:
    """Keep generated search actions while dropping reasoning and passages."""

    queries: list[str] = []
    for function_body in _SEARCH_FUNCTION_BLOCK.findall(output_text or ""):
        match = _QUERY_PARAMETER_BLOCK.search(function_body)
        if match:
            queries.append(match.group(1).strip())
    return queries


def extract_final_answer(output_text: str) -> str | None:
    """Return the final complete answer tag, or ``None`` for a format failure."""

    text = output_text or ""
    if "</think>" in text.lower():
        close_start = text.lower().rfind("</think>")
        text = text[close_start + len("</think>") :]
    lowered = text.lower()
    close_start = lowered.rfind("</answer>")
    if close_start < 0:
        return None
    open_start = lowered.rfind("<answer>", 0, close_start)
    if open_start < 0:
        return None
    answer_start = open_start + len("<answer>")
    return text[answer_start:close_start].strip()


def _to_builtin(value: Any) -> Any:
    """Convert NumPy/Ray container values into JSON-serializable builtins."""

    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _to_builtin(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return _to_builtin(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def build_validation_record(
    *,
    sample_index: int,
    raw_prompt: Any,
    decoded_input: str,
    output_text: str,
    ground_truth: Any,
    score: Any,
    extra_info: Any,
    thinking_mode: Any,
    thinking_steps: Any,
    force_first_search: Any,
    evidence_schema_version: Any,
    search_steps: Any,
    executed_queries: Any = None,
    num_turns: Any = None,
    sample_key: Any = None,
    dataset_question_id: Any = None,
    official_qid: Any = None,
    gold_evidence_ids: Any = None,
    unresolved_gold_facts: Any = None,
    evidence_metrics: Any = None,
) -> dict[str, Any]:
    """Build one self-contained formal A0 evaluation record."""

    normalized_search_steps = _to_builtin(search_steps)
    if not isinstance(normalized_search_steps, list):
        normalized_search_steps = []
    normalized_thinking = _to_builtin(thinking_steps)
    if not isinstance(normalized_thinking, list):
        normalized_thinking = []

    normalized_queries = _to_builtin(executed_queries)
    if not isinstance(normalized_queries, list):
        normalized_queries = [
            str(step.get("query", ""))
            for step in normalized_search_steps
            if isinstance(step, dict) and step.get("source") == "model" and step.get("query")
        ]
    if not normalized_queries:
        normalized_queries = extract_search_queries(output_text)

    normalized_extra_info = _to_builtin(extra_info)
    if not isinstance(normalized_extra_info, dict):
        normalized_extra_info = {}
    split = str(normalized_extra_info.get("split") or "validation")
    row_index = normalized_extra_info.get("index")
    normalized_sample_key = _to_builtin(sample_key)
    if not normalized_sample_key and row_index is not None:
        normalized_sample_key = f"{split}:{int(row_index)}"
    normalized_dataset_qid = _to_builtin(dataset_question_id)
    if not normalized_dataset_qid:
        normalized_dataset_qid = normalized_extra_info.get("question_id")
    normalized_official_qid = _to_builtin(official_qid)
    if not normalized_official_qid:
        normalized_official_qid = normalized_dataset_qid
    normalized_gold = _to_builtin(gold_evidence_ids)
    if not isinstance(normalized_gold, list):
        normalized_gold = []
    normalized_unresolved = _to_builtin(unresolved_gold_facts)
    if not isinstance(normalized_unresolved, list):
        normalized_unresolved = []
    normalized_metrics = _to_builtin(evidence_metrics)
    if not isinstance(normalized_metrics, dict):
        normalized_metrics = {}

    record: dict[str, Any] = {
        "sample_index": int(sample_index),
        "sample_key": str(normalized_sample_key or ""),
        "dataset_question_id": str(normalized_dataset_qid or ""),
        "official_qid": str(normalized_official_qid or ""),
        "question": extract_question(raw_prompt, decoded_input),
        "search_queries": normalized_queries,
        "answer": extract_final_answer(output_text),
        "ground_truth": _to_builtin(ground_truth),
        "score": float(score),
    }
    if num_turns is not None:
        record["num_turns"] = int(_to_builtin(num_turns))
    record.update(
        {
            "thinking_mode": str(_to_builtin(thinking_mode)),
            "qwen_thinking": normalized_thinking,
            "force_first_search": bool(_to_builtin(force_first_search)),
            "evidence_schema_version": str(_to_builtin(evidence_schema_version)),
            "gold_evidence_ids": normalized_gold,
            "unresolved_gold_facts": normalized_unresolved,
            "evidence_metrics": normalized_metrics,
            "search_steps": normalized_search_steps,
        }
    )
    return record
