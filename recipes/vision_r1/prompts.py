"""Actor-visible protocol for the Vision-R1 visual agent."""

from __future__ import annotations

from recipes.vision_r1.artifacts import CERTIFICATE_SCHEMA_VERSION

SYSTEM_PROMPT = """You are a visual mathematics agent. Output exactly one tool call per turn and no prose.
You may inspect a useful root-image region with inspect_region, or finish with submit_answer.
Only cite crop artifact IDs that the tool returned. A crop is optional; do not crop when the original image is
already sufficient. The root image is {root_artifact_id}, size {width}x{height} pixels. Bounding boxes use
source-image pixels as [left, top, right, bottom].
On submission, the certificate claim must state how the cited visual evidence supports the submitted answer."""

FINAL_TURN_PROMPT = "FINAL TURN: call submit_answer now."

INSPECT_REGION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "inspect_region",
        "description": "Crop and return a non-trivial region from the root image.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "pattern": "^image:.+"},
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "purpose": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            "required": ["source_artifact_id", "bbox_2d", "purpose"],
            "additionalProperties": False,
        },
    },
}

VISUAL_CERTIFICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "enum": [CERTIFICATE_SCHEMA_VERSION]},
        "evidence_artifact_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {"type": "string", "pattern": "^image:crop:.+"},
        },
        "claim": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["schema_version", "evidence_artifact_ids", "claim"],
    "additionalProperties": False,
}

SUBMIT_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final answer and, when a crop was used, its visual evidence certificate.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "minLength": 1, "maxLength": 200},
                "certificate": VISUAL_CERTIFICATE_SCHEMA,
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

INSPECT_OR_SUBMIT_TOOLS = [INSPECT_REGION_SCHEMA, SUBMIT_ANSWER_SCHEMA]
SUBMIT_ONLY_TOOLS = [SUBMIT_ANSWER_SCHEMA]
