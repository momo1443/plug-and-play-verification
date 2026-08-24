"""Data models for A9 certificate sidecar records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CERTIFICATE_SCHEMA_VERSION = "hotpotqa-a9-certificate-v1"

SEARCH_CERT_FIELDS = frozenset({"source_id", "support_span", "target"})
FINISH_CERT_FIELDS = frozenset({"source_id", "support_span", "answer_span"})


@dataclass(frozen=True)
class SearchCertificate:
    """Certificate attached to a search action."""

    source_id: str
    support_span: str
    target: str

    def record(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "support_span": self.support_span,
            "target": self.target,
        }


@dataclass(frozen=True)
class FinishCertificate:
    """Certificate attached to a finish action."""

    source_id: str
    support_span: str
    answer_span: str

    def record(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "support_span": self.support_span,
            "answer_span": self.answer_span,
        }


def parse_search_certificate(value: Any) -> tuple[SearchCertificate | None, tuple[str, ...]]:
    """Parse a search certificate without repairing malformed actor output.

    Returns (None, errors) on any validation failure; never raises.
    """
    if not isinstance(value, Mapping):
        return None, ("certificate_not_object",)

    errors: list[str] = []
    if set(value) != SEARCH_CERT_FIELDS:
        errors.append("certificate_fields_invalid")

    source_id = value.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id_invalid")
    else:
        source_id = source_id.strip()

    support_span = value.get("support_span")
    if not isinstance(support_span, str) or not support_span.strip():
        errors.append("support_span_invalid")
    else:
        support_span = " ".join(support_span.split())

    target = value.get("target")
    if not isinstance(target, str) or not target.strip():
        errors.append("target_invalid")
    else:
        target = " ".join(target.split())

    cert = SearchCertificate(
        source_id=str(source_id or ""),
        support_span=str(support_span or ""),
        target=str(target or ""),
    )
    return (cert if not errors else None), tuple(errors)


def parse_finish_certificate(value: Any) -> tuple[FinishCertificate | None, tuple[str, ...]]:
    """Parse a finish certificate without repairing malformed actor output.

    Returns (None, errors) on any validation failure; never raises.
    """
    if not isinstance(value, Mapping):
        return None, ("certificate_not_object",)

    errors: list[str] = []
    if set(value) != FINISH_CERT_FIELDS:
        errors.append("certificate_fields_invalid")

    source_id = value.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id_invalid")
    else:
        source_id = source_id.strip()

    support_span = value.get("support_span")
    if not isinstance(support_span, str) or not support_span.strip():
        errors.append("support_span_invalid")
    else:
        support_span = " ".join(support_span.split())

    answer_span = value.get("answer_span")
    if not isinstance(answer_span, str) or not answer_span.strip():
        errors.append("answer_span_invalid")
    else:
        answer_span = " ".join(answer_span.split())

    cert = FinishCertificate(
        source_id=str(source_id or ""),
        support_span=str(support_span or ""),
        answer_span=str(answer_span or ""),
    )
    return (cert if not errors else None), tuple(errors)
