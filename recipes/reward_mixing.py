"""Shared deterministic reward mixing for prompt-group GRPO comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_r1.verifier import prompt_group_uniform_weight


def prompt_group_key_from_extra_info(
    extra_info: Mapping[str, Any],
    *,
    data_source: str,
    identity_fields: Sequence[str] = ("question_id", "index"),
) -> str:
    """Build a stable prompt identity from dataset metadata shared by all rollouts."""
    if not isinstance(extra_info, Mapping):
        raise ValueError("extra_info must be a mapping for group-shared reward mixing")
    source = str(data_source).strip()
    if not source:
        raise ValueError("data_source is required for group-shared reward mixing")
    split = str(extra_info.get("split") or "train").strip()
    for field in identity_fields:
        value = extra_info.get(field)
        if value is not None and str(value).strip():
            return f"{source}:{split}:{field}:{str(value).strip()}"
    fields = ", ".join(identity_fields)
    raise ValueError(f"extra_info must contain one of ({fields}) for group-shared reward mixing")
