"""Pure helpers for separating Qwen native thinking from visible output."""

from __future__ import annotations

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def split_native_thinking(response_text: str) -> tuple[str, str, bool]:
    """Split one Qwen generation into thinking and visible content.

    Qwen's chat template owns the opening ``<think>`` token, so the generated
    completion commonly starts with reasoning text and contains only the
    closing tag.  An optional generated opening tag is tolerated.
    """

    text = response_text or ""
    lowered = text.lower()
    close_start = lowered.find(_THINK_CLOSE)
    complete = close_start >= 0
    if complete:
        thinking = text[:close_start]
        visible = text[close_start + len(_THINK_CLOSE) :]
    else:
        thinking = text
        visible = ""

    stripped = thinking.lstrip()
    if stripped.lower().startswith(_THINK_OPEN):
        thinking = stripped[len(_THINK_OPEN) :]

    return thinking.strip(), visible.strip(), complete
