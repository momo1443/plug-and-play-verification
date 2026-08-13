"""Raw final-answer protocol retained by the non-LR HotpotQA arms."""

from __future__ import annotations

import os
from collections.abc import Mapping

RAW_FINAL_ANSWER_PROTOCOL = "raw_model_completion_v1"
_REMOVED_FORCE_FINAL_ENV = "HOTPOTQA_FORCE_FINAL_ANSWER"


def require_raw_final_only(environ: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environ is None else environ
    if _REMOVED_FORCE_FINAL_ENV in environment:
        raise RuntimeError(
            f"{_REMOVED_FORCE_FINAL_ENV} has been removed; unset it. "
            "Formal HotpotQA runs preserve the model's raw final completion."
        )
    return RAW_FINAL_ANSWER_PROTOCOL


def resolve_final_answer_protocol() -> str:
    return require_raw_final_only()
