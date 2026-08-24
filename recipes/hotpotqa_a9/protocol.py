"""Tool-call parsing for the A9 certificate-grounded actor contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recipes.hotpotqa_a9.dsl import (
    FinishCertificate,
    SearchCertificate,
    parse_finish_certificate,
    parse_search_certificate,
)
from recipes.hotpotqa_lr.protocol import extract_tool_calls

A9_FINISH_PROTOCOL = "hotpotqa-a9-certificate-finish-v1"


@dataclass(frozen=True)
class SearchCallAudit:
    """Audit record for a parsed search tool call.

    The key invariant: *query* is always extracted when possible,
    regardless of whether the certificate parses successfully.
    """

    envelope_valid: bool
    query: str | None
    certificate_raw: Any
    certificate: SearchCertificate | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class FinishCallAudit:
    """Audit record for a parsed finish tool call.

    The key invariant: *answer* is always extracted when possible,
    regardless of whether the certificate parses successfully.
    """

    envelope_valid: bool
    answer: str | None
    certificate_raw: Any
    certificate: FinishCertificate | None
    errors: tuple[str, ...]


def parse_search_call(text: str) -> SearchCallAudit:
    """Parse one search tool call from model output.

    Certificate is best-effort: query is always extracted if present.
    """
    calls = extract_tool_calls(text)
    if len(calls) != 1 or calls[0].get("name") != "search":
        return SearchCallAudit(False, None, None, None, ("search_call_unparseable",))

    arguments = calls[0]["arguments"]
    errors: list[str] = []

    # Extract query — always, even if certificate is broken
    query = None
    raw_query = arguments.get("query")
    if isinstance(raw_query, str) and raw_query.strip():
        query = raw_query.strip()
    else:
        errors.append("query_not_nonempty_string")

    # Certificate is best-effort
    certificate_raw = arguments.get("certificate")
    certificate = None
    if certificate_raw is not None:
        certificate, cert_errors = parse_search_certificate(certificate_raw)
        errors.extend(cert_errors)

    envelope_valid = len(errors) == 0 or (query is not None and certificate is not None)
    return SearchCallAudit(
        envelope_valid=envelope_valid,
        query=query,
        certificate_raw=certificate_raw,
        certificate=certificate,
        errors=tuple(errors),
    )


def parse_finish_call(text: str) -> FinishCallAudit:
    """Parse one finish tool call from model output.

    Certificate is best-effort: answer is always extracted if present.
    """
    calls = extract_tool_calls(text)
    if len(calls) != 1 or calls[0].get("name") != "finish":
        return FinishCallAudit(False, None, None, None, ("finish_call_unparseable",))

    arguments = calls[0]["arguments"]
    errors: list[str] = []

    # Extract answer — always, even if certificate is broken
    answer = None
    if arguments.get("status") != "answer":
        errors.append("finish_status_invalid")
    raw_answer = arguments.get("answer")
    if isinstance(raw_answer, str) and raw_answer.strip():
        answer = raw_answer.strip()
    else:
        errors.append("answer_not_nonempty_string")

    # Certificate is best-effort
    certificate_raw = arguments.get("certificate")
    certificate = None
    if certificate_raw is not None:
        certificate, cert_errors = parse_finish_certificate(certificate_raw)
        errors.extend(cert_errors)

    envelope_valid = len(errors) == 0 or (answer is not None and certificate is not None)
    return FinishCallAudit(
        envelope_valid=envelope_valid,
        answer=answer,
        certificate_raw=certificate_raw,
        certificate=certificate,
        errors=tuple(errors),
    )
