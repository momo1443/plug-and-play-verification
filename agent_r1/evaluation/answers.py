"""Extract one submitted answer before consulting any evaluation reference."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

AIME_EXTRACTION_VERSION = "final-answer-v3"


def final_answer_record(text: str) -> dict[str, str] | None:
    """Select the last explicit answer; never search the narrative for a gold match.

    Balanced boxed expressions, answer tags and explicit answer statements are
    accepted. A bare scalar is accepted only when it is the entire last line.
    A malformed last explicit answer fails closed rather than falling back to
    an earlier answer. The returned prefix is usable by a process-only judge.
    """
    candidates: list[tuple[int, int, str | None, str]] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        end = match.end()
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        answer = text[match.end():end - 1].strip() if depth == 0 else None
        candidates.append((match.start(), end, answer, "boxed"))
    for match in re.finditer(r"<answer>", text, re.IGNORECASE):
        closing = re.search(r"</answer>", text[match.end():], re.IGNORECASE)
        end = match.end() + closing.end() if closing else len(text)
        answer = text[match.end():match.end() + closing.start()].strip() if closing else None
        candidates.append((match.start(), end, answer, "answer_tag"))
    for match in re.finditer(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*([^\n]+)",
        text, re.IGNORECASE,
    ):
        candidates.append((match.start(), match.end(), match.group(1).strip().rstrip("."), "verbal"))
    stripped = text.rstrip()
    scalar = re.search(r"(?m)^[ \t]*([+-]?\d+(?:\.\d+)?)[ \t]*\Z", stripped)
    if scalar:
        candidates.append((scalar.start(), scalar.end(), scalar.group(1), "final_line"))
    if candidates:
        start, _, answer, method = max(candidates, key=lambda item: item[0])
        if not answer:
            return None
        # Exclude the complete final-answer sentence from judge input when the
        # selected box/tag is inside an explicit answer statement.
        prefix_end = min((a for a, b, _, _ in candidates if a <= start < b), default=start)
        return {"answer": answer, "method": method, "reasoning": text[:prefix_end].strip()}
    return None


def aime_integer(answer: str) -> int | None:
    """Accept an exact integer in [0, 999], including integral decimal notation."""
    value = str(answer).strip()
    wrapper = re.fullmatch(r"\\(?:text|mathrm)\{([^{}]+)\}", value)
    if wrapper:
        value = wrapper.group(1).strip()
    if not re.fullmatch(r"\+?\d+(?:\.0+)?", value):
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return int(number) if number.is_finite() and 0 <= number <= 999 else None


def score_aime(text: str, gold: str) -> dict:
    prediction = final_answer_record(text)
    predicted = aime_integer(prediction["answer"]) if prediction else None
    target = aime_integer(gold)
    if target is None:
        raise ValueError(f"Invalid AIME reference answer: {gold!r}")
    return {
        "predicted_answer": prediction["answer"] if prediction else None,
        "match_method": prediction["method"] if prediction else None,
        "is_correct": predicted is not None and predicted == target,
        "extraction_version": AIME_EXTRACTION_VERSION,
    }
