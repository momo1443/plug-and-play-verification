"""Extract the useful, non-repeated fields from a HotpotQA trajectory."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from recipes.hotpotqa.final_answer_protocol import RAW_FINAL_ANSWER_PROTOCOL
from recipes.hotpotqa_a9.protocol import A9_FINISH_PROTOCOL
from recipes.hotpotqa_lr.protocol import LR_FINISH_PROTOCOL, parse_finish

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


def _lenient_answer_extract(output_text: str) -> str | None:
    """Extract answer from <answer> tags or return full text as fallback.

    Used when the A8/A9 finish tool call fails to parse but the model
    may still have produced a recognisable answer in free text.
    """
    text = output_text or ""
    if "```" in text.lower():
        close_start = text.lower().rfind("```")
        text = text[close_start + len("```") :]
    lowered = text.lower()
    close_start = lowered.rfind("</answer>")
    if close_start >= 0:
        open_start = lowered.rfind("<answer>", 0, close_start)
        if open_start >= 0:
            answer_start = open_start + len("<answer>")
            return text[answer_start:close_start].strip()
    # Fallback: return the full (post-thinking) completion so that raw answers
    # without <answer> tags are still scored by EM / F1.
    return text.strip() or None


def extract_final_answer(
    output_text: str,
    final_answer_protocol: str = RAW_FINAL_ANSWER_PROTOCOL,
) -> str | None:
    """Return the final complete answer tag, or the full completion as fallback.

    Mirrors the fallback logic in ``summarize_test_accuracy._extract_answer_from_completion``
    so that models that do not wrap their answer in ``<answer>`` tags still get scored.
    When *final_answer_protocol* is empty (non-QA environments),
    returns the full text without attempting tag extraction.

    For A8-LR and A9 protocols, falls back to lenient <answer>-tag extraction
    when the finish tool call fails to parse, ensuring validation EM is not
    penalised by protocol formatting failures.
    """
    if not final_answer_protocol:
        return (output_text or "").strip() or None

    if final_answer_protocol == LR_FINISH_PROTOCOL:
        finish = parse_finish(output_text)
        if finish.envelope_valid:
            return finish.answer
        # Lenient fallback: try <answer> tags when finish envelope is invalid.
        return _lenient_answer_extract(output_text)

    if final_answer_protocol == A9_FINISH_PROTOCOL:
        from recipes.hotpotqa_a9.protocol import parse_finish_call as a9_parse_finish

        finish = a9_parse_finish(output_text)
        if finish.answer is not None:
            return finish.answer
        # Lenient fallback: try <answer> tags when no finish answer found.
        return _lenient_answer_extract(output_text)

    return _lenient_answer_extract(output_text)


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
    final_answer_protocol: Any,
    evidence_schema_version: Any,
    search_steps: Any,
    local_reasoning_transitions: Any = None,
    local_reasoning_audit: Any = None,
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

    normalized_protocol = str(_to_builtin(final_answer_protocol))
    supported_protocols = {
        RAW_FINAL_ANSWER_PROTOCOL,
        LR_FINISH_PROTOCOL,
        A9_FINISH_PROTOCOL,
    }
    # Allow None / empty string for non-QA environments
    # that do not use a final-answer protocol.
    if normalized_protocol in {"None", ""}:
        normalized_protocol = ""
    if normalized_protocol and normalized_protocol not in supported_protocols:
        raise ValueError(
            f"Unexpected final-answer protocol {normalized_protocol!r}; "
            f"expected one of {sorted(supported_protocols)!r}"
        )

    normalized_search_steps = _to_builtin(search_steps)
    if not isinstance(normalized_search_steps, list):
        normalized_search_steps = []
    normalized_thinking = _to_builtin(thinking_steps)
    if not isinstance(normalized_thinking, list):
        normalized_thinking = []
    normalized_local_transitions = _to_builtin(local_reasoning_transitions)
    if not isinstance(normalized_local_transitions, list):
        normalized_local_transitions = []
    normalized_local_audit = _to_builtin(local_reasoning_audit)
    if not isinstance(normalized_local_audit, dict):
        normalized_local_audit = {}

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
        "answer": extract_final_answer(output_text, normalized_protocol),
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
            "final_answer_protocol": normalized_protocol,
            "evidence_schema_version": str(_to_builtin(evidence_schema_version)),
            "gold_evidence_ids": normalized_gold,
            "unresolved_gold_facts": normalized_unresolved,
            "evidence_metrics": normalized_metrics,
            "search_steps": normalized_search_steps,
            "local_reasoning_transitions": normalized_local_transitions,
            "local_reasoning_audit": normalized_local_audit,
        }
    )
    return record
