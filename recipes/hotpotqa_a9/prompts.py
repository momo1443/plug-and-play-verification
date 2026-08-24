"""Actor-visible prompts and tool schemas for A9 certificate-grounded RLVR."""

from __future__ import annotations

A9_SYSTEM_PROMPT = (
    "You are a research agent. Answer the user query using Wikipedia search evidence. "
    "Output exactly one tool call per turn and no prose. "
    "Each tool call after the first search must include a certificate that cites an exact "
    "text span from a visible artifact and shows how it connects to your action."
)

A9_USER_PROMPT = """### User Query
{user_query}

### Prior Searches
{history_actions}

### Retrieved Artifacts
{passage_list}

### Certificate Ledger
{certificate_ledger}

### Recent Tool Format Issues
{tool_feedback}

Use a new search when evidence is missing. The first search is a bootstrap query — no
certificate needed. Every later search must include a certificate with three fields:
source_id (e.g. passage:17), support_span (exact text from that artifact), and target
(the entity you are searching for next).
When the evidence supports the answer, call finish with status `answer`, the answer, and
a certificate citing the source_id, support_span, and answer_span. Avoid repeated queries.
"""

A9_FINISH_AVAILABLE_PROMPT = """You may call search again, or call finish now. Output exactly one
tool call and no surrounding text."""

A9_FINAL_TURN_PROMPT = """FINAL TURN: Do not search. Call finish with status `answer`, the smallest
answer span, and a certificate citing the supporting artifact. Output exactly one tool call."""

# ── Certificate JSON Schemas ──────────────────────────────────────────────

SEARCH_CERTIFICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {
            "type": "string",
            "pattern": "^passage:.+$",
            "description": "Visible artifact id such as passage:37.",
        },
        "support_span": {
            "type": "string",
            "minLength": 1,
            "description": "Exact text span copied from the source artifact.",
        },
        "target": {
            "type": "string",
            "minLength": 1,
            "description": "The entity or term you are searching for next.",
        },
    },
    "required": ["source_id", "support_span", "target"],
    "additionalProperties": False,
}

FINISH_CERTIFICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {
            "type": "string",
            "pattern": "^passage:.+$",
            "description": "Visible artifact id such as passage:37.",
        },
        "support_span": {
            "type": "string",
            "minLength": 1,
            "description": "Exact text span copied from the source artifact.",
        },
        "answer_span": {
            "type": "string",
            "minLength": 1,
            "description": "The answer text found in or derived from the source.",
        },
    },
    "required": ["source_id", "support_span", "answer_span"],
    "additionalProperties": False,
}

# ── Tool Schemas ──────────────────────────────────────────────────────────

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

REASONED_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search using a target derived from a visible artifact citation.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "certificate": SEARCH_CERTIFICATE_SCHEMA,
            },
            "required": ["query", "certificate"],
            "additionalProperties": False,
        },
    },
}

FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Submit the answer with a certificate citing supporting evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "const": "answer"},
                "answer": {"type": "string", "minLength": 1},
                "certificate": FINISH_CERTIFICATE_SCHEMA,
            },
            "required": ["status", "answer", "certificate"],
            "additionalProperties": False,
        },
    },
}

BOOTSTRAP_TOOL_SCHEMAS = [BOOTSTRAP_SEARCH_SCHEMA]
SEARCH_OR_FINISH_TOOL_SCHEMAS = [REASONED_SEARCH_SCHEMA, FINISH_SCHEMA]
FINISH_TOOL_SCHEMAS = [FINISH_SCHEMA]
