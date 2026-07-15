"""Stateless deterministic verifier for HotpotQA process rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PROCESS_VERIFIER_VERSION = "hotpotqa-new-evidence-v1"


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
