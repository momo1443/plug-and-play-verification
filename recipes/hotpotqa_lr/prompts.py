"""Actor-visible prompts and tool schemas for A8-LR."""

from __future__ import annotations

import copy

LR_SYSTEM_PROMPT = (
    "You are a research agent. Answer the user query using Wikipedia search evidence and the "
    "local reasoning tool protocol. Output exactly one tool call per turn and no prose."
)

LR_USER_PROMPT = """### User Query
{user_query}

### Prior Searches
{history_actions}

### Retrieved Artifacts
{passage_list}

### Reasoning Ledger
{reasoning_ledger}

### Recent Tool Format Issues
{tool_feedback}

Use a new search when evidence is missing. The first search is a bootstrap query. Every later
search must declare one reason_step grounded in an exact span from a visible artifact. A reason
step has a unique ref, one of the allowed operations, exact artifact premises, prior ref inputs,
and the operation's exact output object. Do not report verifier results or reasoning prose.
When the evidence supports the minimal answer, call finish with status `answer`, the answer, and
one grounded reason_step. Avoid repeated queries.
"""

LR_FINISH_AVAILABLE_PROMPT = """You may call search again, or call finish now. Output exactly one
tool call and no surrounding text."""

LR_FINAL_TURN_PROMPT = """FINAL TURN: Do not search. Call finish with status `answer`, the smallest
answer span, and one grounded reason_step. Output exactly one tool call."""

_PREMISE_SCHEMA = {
    "type": "object",
    "properties": {
        "artifact_id": {"type": "string", "minLength": 1},
        "span": {"type": "string", "minLength": 1},
    },
    "required": ["artifact_id", "span"],
    "additionalProperties": False,
}

REASON_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "ref": {"type": "string", "pattern": "^r[1-9][0-9]*$"},
        "op": {
            "type": "string",
            "enum": [
                "select_exact_span",
                "extract_bridge_entity",
                "extract_answer_candidate",
                "normalize_answer",
                "compare_equal",
            ],
        },
        "premises": {"type": "array", "items": _PREMISE_SCHEMA, "maxItems": 8},
        "inputs": {
            "type": "array",
            "items": {"type": "string", "pattern": "^r[1-9][0-9]*$"},
            "maxItems": 8,
        },
        "output": {
            "type": "object",
            "description": (
                "Use exactly one key: text, bridge_entity, answer_candidate, "
                "normalized_answer, or equal, as required by op."
            ),
        },
    },
    "required": ["ref", "op", "premises", "inputs", "output"],
    "additionalProperties": False,
}

BOOTSTRAP_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Run the first Wikipedia retrieval query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

REASONED_SEARCH_SCHEMA = copy.deepcopy(BOOTSTRAP_SEARCH_SCHEMA)
REASONED_SEARCH_SCHEMA["function"]["description"] = (
    "Search using a conclusion derived from prior visible evidence."
)
REASONED_SEARCH_SCHEMA["function"]["parameters"]["properties"]["reason_step"] = REASON_STEP_SCHEMA
REASONED_SEARCH_SCHEMA["function"]["parameters"]["required"] = ["query", "reason_step"]

FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Submit the minimal answer and its final local reasoning transition.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "const": "answer"},
                "answer": {"type": "string", "minLength": 1},
                "reason_step": REASON_STEP_SCHEMA,
            },
            "required": ["status", "answer", "reason_step"],
            "additionalProperties": False,
        },
    },
}

BOOTSTRAP_TOOL_SCHEMAS = [BOOTSTRAP_SEARCH_SCHEMA]
SEARCH_OR_FINISH_TOOL_SCHEMAS = [REASONED_SEARCH_SCHEMA, FINISH_SCHEMA]
FINISH_TOOL_SCHEMAS = [FINISH_SCHEMA]
