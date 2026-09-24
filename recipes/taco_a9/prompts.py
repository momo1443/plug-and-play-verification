"""Actor-visible prompt and tool schemas for TACO A9."""

from __future__ import annotations


SYSTEM_PROMPT = """You are a Python programming agent. Output exactly one tool call per turn and no prose.
Use write_code to create a complete program, run_developer_tests to inspect a code artifact,
and submit to provide the final code artifact with a certificate. Never claim that a test run
occurred unless you cite a visible run_id returned by the tool."""

USER_PROMPT = """### Programming task
{question}

### Current code artifact
{current_code}

### Execution ledger
{execution_ledger}

### Tool feedback
{feedback}
"""

FINAL_TURN_PROMPT = """FINAL TURN: call submit. Its certificate must cite a developer test run for the
exact final code artifact. Output exactly one tool call."""

WRITE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_code",
        "description": "Create or replace a complete Python solution.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "minLength": 1}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

RUN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_developer_tests",
        "description": "Run the visible developer suite against one prior code artifact.",
        "parameters": {
            "type": "object",
            "properties": {"code_artifact_id": {"type": "string", "pattern": "^code:.+"}},
            "required": ["code_artifact_id"],
            "additionalProperties": False,
        },
    },
}

SUBMISSION_CERTIFICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_code_artifact_id": {"type": "string", "pattern": "^code:.+"},
        "final_code_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
        "evidence_run_ids": {"type": "array", "minItems": 1, "items": {"type": "string", "pattern": "^run:.+"}},
        "claim": {"type": "string", "minLength": 1},
    },
    "required": ["final_code_artifact_id", "final_code_sha256", "evidence_run_ids", "claim"],
    "additionalProperties": False,
}

SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit a final code artifact and execution certificate.",
        "parameters": {
            "type": "object",
            "properties": {
                "code_artifact_id": {"type": "string", "pattern": "^code:.+"},
                "certificate": SUBMISSION_CERTIFICATE_SCHEMA,
            },
            "required": ["code_artifact_id", "certificate"],
            "additionalProperties": False,
        },
    },
}

WRITE_ONLY_TOOLS = [WRITE_SCHEMA]
WRITE_OR_RUN_OR_SUBMIT_TOOLS = [WRITE_SCHEMA, RUN_SCHEMA, SUBMIT_SCHEMA]
SUBMIT_ONLY_TOOLS = [SUBMIT_SCHEMA]
