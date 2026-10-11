"""Verification groups and domain-specific normalization weights.

These groups contain artifact scores, independently of GRPO rollout groups.
Applicability is supplied by the domain protocol, never inferred from whether
an artifact passes verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class VerificationGroup:
    """Artifacts averaged within one interaction step or scored individually.

    ``base_weights`` allow domain-specific normalization within a step, such
    as equal weighting of mathematical reasoning segments. An individually
    scored artifact uses ``kind='singleton_artifact'`` and exactly one slot.
    Missing required evidence occupies an applicable slot with score zero.
    """

    group_id: str
    scores: tuple[float, ...]
    applicable: tuple[bool, ...] | None = None
    base_weights: tuple[float, ...] | None = None
    kind: str = "interaction_step"


def normalize_verification_groups(groups: Sequence[VerificationGroup]) -> dict[str, Any]:
    """Return bounded credits and the weights that actually compute them.

    Empty/inapplicable groups have weight zero and do not enter the outer
    denominator. Applicable failures retain positive weights and zero scores.
    The remaining groups receive equal outer weights; singleton groups use
    an inner weight of one. An empty applicable set yields process reward zero.
    """

    records: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group.group_id, str) or not group.group_id or group.group_id in group_ids:
            raise ValueError("Verification group IDs must be nonempty and unique")
        group_ids.add(group.group_id)
        if group.kind not in {"interaction_step", "singleton_artifact"}:
            raise ValueError("Unsupported verification group kind")
        scores = tuple(float(score) for score in group.scores)
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError("Artifact scores must be finite and in [0, 1]")
        if group.kind == "singleton_artifact" and len(scores) != 1:
            raise ValueError("Individually scored artifacts require singleton groups")
        applicable = group.applicable if group.applicable is not None else (True,) * len(scores)
        base_weights = group.base_weights if group.base_weights is not None else (1.0,) * len(scores)
        if len(applicable) != len(scores) or len(base_weights) != len(scores):
            raise ValueError("Every artifact slot requires an applicability flag and weight")
        if any(not isinstance(flag, bool) for flag in applicable):
            raise ValueError("Check applicability must be an explicit boolean")
        base_weights = tuple(float(weight) for weight in base_weights)
        if any(not math.isfinite(weight) or weight < 0 for weight in base_weights):
            raise ValueError("Verification weights must be finite and nonnegative")
        if any(flag and weight <= 0 for flag, weight in zip(applicable, base_weights, strict=True)):
            raise ValueError("Applicable checks must retain positive normalization weights")
        normalizer = math.fsum(weight for flag, weight in zip(applicable, base_weights, strict=True) if flag)
        within_weights = [weight / normalizer if flag else 0.0
                          for flag, weight in zip(applicable, base_weights, strict=True)] if normalizer else [0.0] * len(scores)
        records.append({
            "group_id": group.group_id,
            "group_kind": group.kind,
            "group_size": len(scores),
            "artifact_scores": list(scores),
            "check_applicability": list(applicable),
            "applicable_check_count": sum(applicable),
            "within_group_normalizer": normalizer,
            "within_group_weights": within_weights,
            "group_score": math.fsum(weight * score for weight, score in zip(within_weights, scores, strict=True)),
        })

    denominator = sum(record["applicable_check_count"] > 0 for record in records)
    for record in records:
        weight = 1.0 / denominator if record["applicable_check_count"] else 0.0
        record["group_weight"] = weight
        record["effective_weights"] = [weight * inner for inner in record["within_group_weights"]]
        record["normalized_credit"] = math.fsum(
            artifact_weight * score
            for artifact_weight, score in zip(record["effective_weights"], record["artifact_scores"], strict=True)
        )
    return {
        "normalizer": denominator,
        "process_reward": math.fsum(record["normalized_credit"] for record in records),
        "groups": records,
    }
