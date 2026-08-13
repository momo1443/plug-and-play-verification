"""LLM-as-judge prompt templates for A6 process reward evaluation.

The judge evaluates which gold supporting facts are semantically covered by
the accumulated retrieved passages, outputting a JSON array of covered
fact IDs.  The process increment is then computed using the same formula
as A3's deterministic evidence-ID verifier: |new_llm_covered| / |G_i|.

Version history:
- v3: scalar 0–1 anchored scores with high-watermark increment (deprecated)
- v4: fact-ID level coverage selection matching A3's formula
"""

from __future__ import annotations

import json
import logging
from typing import Collection

logger = logging.getLogger(__name__)

JUDGE_VERSION = "hotpotqa-llm-judge-factid-matched-v4"

JUDGE_SYSTEM_PROMPT = (
    "You are an evidence-coverage evaluator for a multi-hop question-answering "
    "search agent. Your task is to determine which reference supporting facts "
    "are semantically supported by the retrieved passages. "
    "Evaluate only text under the two retrieved-passages headings. "
    "Reference supporting facts are a matching rubric, not retrieved evidence, "
    "and must never count as coverage by themselves. Do not use outside knowledge. "
    "Hard rule: when both retrieved-passages sections are None, output []."
)

JUDGE_USER_PROMPT = """### Question
{question}

### Reference supporting facts
Each fact has a unique ID in brackets. Determine which of these facts are
semantically supported by the retrieved passages. The facts are a rubric only
and are not retrieved evidence.
{gold_supporting_fact_texts}

### Previous search queries
{previous_queries}

### Current search query
{current_query}

### Previously accumulated retrieved passages
{previous_passages}

### Newly retrieved passages
{current_passages}

### Scoring task
Look only at text under "Previously accumulated retrieved passages" and
"Newly retrieved passages". For each reference fact listed above, determine
whether the accumulated passages (previous + new) semantically support it.
A fact is supported if the passages contain information that clearly
establishes or confirms that fact, even if the wording differs.

Output the IDs of ALL supported facts as a JSON array of strings.
- If facts [fact-1] and [fact-3] are supported but [fact-2] is not, output: ["fact-1", "fact-3"]
- If no facts are supported, output: []
- Only include fact IDs that appear in the reference section above.
- Do NOT include any other text, explanation, or formatting.

The reference section itself never counts as support. If both retrieved
passage sections are None, the output must be []."""


# --- Deprecated v3 scalar-score helpers (kept for backward compatibility) ---

ALLOWED_JUDGE_OUTPUTS: dict[str, float] = {
    "0": 0.0,
    "0.0": 0.0,
    "0.25": 0.25,
    "0.5": 0.5,
    "0.50": 0.5,
    "0.75": 0.75,
    "1": 1.0,
    "1.0": 1.0,
}


def parse_judge_score(text: str) -> float | None:
    """Return the anchored score if *text* matches a registered output, else ``None``.

    .. deprecated:: v4
        Use :func:`parse_judge_fact_ids` for the fact-ID level judge.
    """
    return ALLOWED_JUDGE_OUTPUTS.get(text.strip())


# --- v4 fact-ID level helpers ---


def parse_judge_fact_ids(
    text: str,
    allowed_ids: Collection[str],
) -> set[str] | None:
    """Parse the judge's JSON array output and validate against allowed IDs.

    Parameters
    ----------
    text:
        Raw output from the judge model.  Expected to be a JSON array of
        fact-ID strings, e.g. ``["fact-1", "fact-3"]``.
    allowed_ids:
        The set of valid fact IDs that the judge may reference.  Any ID not
        in this set is silently dropped (not treated as an error), because
        small models occasionally hallucinate IDs.  However, if the output
        cannot be parsed as valid JSON at all, ``None`` is returned to signal
        a ``judge_invalid`` trajectory.

    Returns
    -------
    A set of validated fact IDs (subset of *allowed_ids*), or ``None`` if
    the output cannot be parsed as JSON.  Returning ``None`` signals an
    invalid judge response that marks the entire trajectory as excluded
    from the actor policy update.

    This function must not silently default to a non-empty set.
    """
    stripped = text.strip()
    if not stripped:
        logger.warning("Judge returned empty output.")
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks or extra text
        # Some models wrap output in ```json ... ``` or add explanatory text
        json_start = stripped.find("[")
        json_end = stripped.rfind("]")
        if json_start >= 0 and json_end > json_start:
            try:
                parsed = json.loads(stripped[json_start : json_end + 1])
            except json.JSONDecodeError:
                logger.warning("Judge output is not valid JSON: %r", stripped[:200])
                return None
        else:
            logger.warning("Judge output is not valid JSON: %r", stripped[:200])
            return None

    if not isinstance(parsed, list):
        logger.warning(
            "Judge output is not a JSON array (got %s): %r",
            type(parsed).__name__,
            stripped[:200],
        )
        return None

    allowed_set = set(allowed_ids)
    result: set[str] = set()
    rejected: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            if item in allowed_set:
                result.add(item)
            else:
                rejected.append(item)
        else:
            logger.warning(
                "Judge array contains non-string item: %r", item
            )

    if rejected:
        logger.warning(
            "Judge returned %d ID(s) not in allowed set: %r",
            len(rejected),
            rejected[:10],
        )

    return result


def format_gold_facts_for_prompt(
    fact_ids_and_texts: Collection[tuple[str, str]],
) -> str:
    """Format gold fact IDs and texts for the judge prompt.

    Parameters
    ----------
    fact_ids_and_texts:
        Pairs of (fact_id, fact_text) to include in the prompt.

    Returns
    -------
    Formatted string with one line per fact, e.g.
    ``[hotpotqa-official-sentence-v1:123:0] [Title] Sentence text.``
    """
    if not fact_ids_and_texts:
        return "None"
    lines: list[str] = []
    for fact_id, fact_text in fact_ids_and_texts:
        lines.append(f"[{fact_id}] {fact_text}")
    return "\n".join(lines)
