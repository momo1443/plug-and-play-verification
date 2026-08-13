"""Closed data model for the first HotpotQA local-reasoning DSL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

DSL_VERSION = "hotpotqa-local-reasoning-dsl-v1"
REASON_REF_PATTERN = re.compile(r"r[1-9][0-9]*")
OPERATIONS = frozenset(
    {
        "select_exact_span",
        "extract_bridge_entity",
        "extract_answer_candidate",
        "normalize_answer",
        "compare_equal",
    }
)


@dataclass(frozen=True)
class Premise:
    artifact_id: str
    span: str

    def record(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "span": self.span}


@dataclass(frozen=True)
class ReasonStep:
    ref: str
    op: str
    premises: tuple[Premise, ...]
    inputs: tuple[str, ...]
    output: Mapping[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "op": self.op,
            "premises": [premise.record() for premise in self.premises],
            "inputs": list(self.inputs),
            "output": dict(self.output),
        }


def parse_reason_step(value: Any) -> tuple[ReasonStep | None, tuple[str, ...]]:
    """Parse the closed schema without repairing malformed actor output."""

    errors: list[str] = []
    if not isinstance(value, Mapping):
        return None, ("reason_step_not_object",)
    required = {"ref", "op", "premises", "inputs", "output"}
    if set(value) != required:
        errors.append("reason_step_fields_invalid")

    ref = value.get("ref")
    if not isinstance(ref, str) or REASON_REF_PATTERN.fullmatch(ref) is None:
        errors.append("ref_invalid")
        ref = str(ref) if ref is not None else ""

    op = value.get("op")
    if op not in OPERATIONS:
        errors.append("op_invalid")
        op = str(op) if op is not None else ""

    raw_premises = value.get("premises")
    premises: list[Premise] = []
    if not isinstance(raw_premises, list) or len(raw_premises) > 8:
        errors.append("premises_invalid")
    else:
        for index, raw in enumerate(raw_premises):
            if not isinstance(raw, Mapping) or set(raw) != {"artifact_id", "span"}:
                errors.append(f"premise_{index}_fields_invalid")
                continue
            artifact_id = raw.get("artifact_id")
            span = raw.get("span")
            if not isinstance(artifact_id, str) or not artifact_id:
                errors.append(f"premise_{index}_artifact_id_invalid")
                continue
            if not isinstance(span, str) or not span:
                errors.append(f"premise_{index}_span_invalid")
                continue
            premises.append(Premise(artifact_id=artifact_id, span=span))

    raw_inputs = value.get("inputs")
    inputs: list[str] = []
    if not isinstance(raw_inputs, list) or len(raw_inputs) > 8:
        errors.append("inputs_invalid")
    else:
        for index, item in enumerate(raw_inputs):
            if not isinstance(item, str) or REASON_REF_PATTERN.fullmatch(item) is None:
                errors.append(f"input_{index}_invalid")
                continue
            inputs.append(item)
        if len(inputs) != len(set(inputs)):
            errors.append("inputs_duplicate")

    output = value.get("output")
    if not isinstance(output, Mapping):
        errors.append("output_not_object")
        output = {}

    step = ReasonStep(
        ref=ref,
        op=op,
        premises=tuple(premises),
        inputs=tuple(inputs),
        output=dict(output),
    )
    return (step if not errors else None), tuple(errors)
