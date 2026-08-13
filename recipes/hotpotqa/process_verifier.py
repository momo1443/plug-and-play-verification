"""Stateless deterministic verifier for HotpotQA process rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PROCESS_VERIFIER_VERSION = "hotpotqa-new-evidence-v1"
WEAK_EXECUTION_VERIFIER_VERSION = "hotpotqa-weak-execution-v1"


@dataclass(frozen=True)
class ProcessVerification:
    returned_evidence_ids: tuple[str, ...]
    new_gold_evidence_ids: tuple[str, ...]
    covered_gold_evidence_ids: tuple[str, ...]
    reward: float | None


def verify_new_evidence(
    *,
    returned_evidence_ids: Iterable[str],
    gold_evidence_ids: Iterable[str],
    covered_gold_evidence_ids: Iterable[str],
    evidence_metrics_eligible: bool,
) -> ProcessVerification:
    """Compute ``|new gold evidence| / |all gold evidence|`` as a pure function.

    Its deliberately narrow signature accepts evidence identities only: no
    answer, ground-truth text, thinking, model, optimizer, or mutable global
    state can enter the verifier.
    """

    returned = {str(value) for value in returned_evidence_ids}
    gold = {str(value) for value in gold_evidence_ids}
    covered_before = {str(value) for value in covered_gold_evidence_ids}
    current_hits = returned & gold
    new_gold = current_hits - covered_before
    covered_after = covered_before | current_hits
    reward = len(new_gold) / len(gold) if evidence_metrics_eligible and gold else None
    return ProcessVerification(
        returned_evidence_ids=tuple(sorted(returned)),
        new_gold_evidence_ids=tuple(sorted(new_gold)),
        covered_gold_evidence_ids=tuple(sorted(covered_after)),
        reward=reward,
    )


# ------------------------------------------------------------------
# A7 weak execution verifier
# ------------------------------------------------------------------


@dataclass(frozen=True)
class WeakExecutionVerification:
    """Audit record for the A7 weak execution verifier."""

    model_generated: bool
    schema_and_parser_valid: bool
    nonempty_query: bool
    tool_execution_succeeded: bool
    nonempty_observation: bool
    raw_weak_process: float
    max_executed_searches: int


def verify_execution(
    *,
    model_generated: bool,
    schema_and_parser_valid: bool,
    query: str,
    tool_execution_succeeded: bool,
    returned_fact_records: Iterable[dict],
    max_executed_searches: int = 3,
) -> WeakExecutionVerification:
    """Compute the A7 weak execution process reward.

    The verifier checks only whether a model-generated search action produced
    a valid, observable environment transition.  It does **not** accept qid,
    gold evidence IDs, answer, covered set, or any semantic relevance signal.

    Each verified search step earns ``1 / max_executed_searches``; otherwise
    the raw reward is 0.  The caller (AgentFlow) multiplies by the arm's
    process weight (0.5 for A7) to obtain the optimizer-visible step reward.

    Parameters
    ----------
    model_generated:
        True when the search action originated from the model (not from
        ``force_first_search``, environment auto-retry, or cached replay).
    schema_and_parser_valid:
        True when the tool call passed Hermes parser and schema validation.
    query:
        The query string from the parsed tool-call arguments.
    tool_execution_succeeded:
        True when the search tool returned without error/timeout.
    returned_fact_records:
        An iterable of dicts, each with at least ``pid`` and ``text`` keys,
        representing the observation returned by the search tool.
    max_executed_searches:
        Normalisation denominator (H).  Defaults to 3 per the experiment
        manual: at most 3 model searches per trajectory.
    """
    nonempty_query = bool(query and query.strip())

    nonempty_observation = any(
        bool(record.get("pid") is not None and record.get("text") and str(record["text"]).strip())
        for record in returned_fact_records
    )

    execution_verified = (
        model_generated
        and schema_and_parser_valid
        and nonempty_query
        and tool_execution_succeeded
        and nonempty_observation
    )

    raw_weak_process = (1.0 / max_executed_searches) if execution_verified else 0.0

    return WeakExecutionVerification(
        model_generated=model_generated,
        schema_and_parser_valid=schema_and_parser_valid,
        nonempty_query=nonempty_query,
        tool_execution_succeeded=tool_execution_succeeded,
        nonempty_observation=nonempty_observation,
        raw_weak_process=raw_weak_process,
        max_executed_searches=max_executed_searches,
    )
