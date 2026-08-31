"""Data models for A9 certificate sidecar records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

CERTIFICATE_SCHEMA_VERSION = "hotpotqa-a9-certificate-v1"

SEARCH_CERT_FIELDS = frozenset({"source_id", "support_span", "target"})
FINISH_CERT_FIELDS = frozenset({"source_id", "support_span", "answer_span"})

# ---------------------------------------------------------------------------
# Fault-tolerant certificate repair
# ---------------------------------------------------------------------------
# During EM warmup (step 1-100), cert reward=0 so the model degrades its JSON
# format.  By step 88, 100% of certificates fail json.loads().  The three
# required fields are still present, just in non-standard patterns:
#
#   Pattern A (majority):  "key}"  with value after the closing brace
#     e.g.  {"source_id": "passage:5", "support_span": "text...", "answer_span}
#           London, England
#
#   Pattern B (mixed):  commas replace colons for support_span, braces for last key
#     e.g.  {"source_id": "passage:5", "support_span", "text...", "target}
#           value text
#
# In ~70% of broken certs, the model omits the VALUE for the last key
# ("target} or "answer_span}) entirely.  We still return a repaired certificate
# in this case with the last field set to "" — the verifier will flag
# coupling_failed, but grounding can still be checked.

# Precompiled patterns for performance on hot paths.
_RE_SOURCE_ID = re.compile(r'"source_id"\s*[:,]\s*"([^"]+)"')
_RE_SOURCE_ID_UNQUOTED = re.compile(r'"source_id"\s*[:,]\s*(passage:\d+)')
_RE_SUPPORT_SPAN_COLON = re.compile(
    r'"support_span"\s*:\s*"((?:[^"\\]|\\.)*)"'
)
_RE_SUPPORT_SPAN_COMMA = re.compile(
    r'"support_span"\s*,\s*"((?:[^"\\]|\\.)*)"'
)
_RE_SUPPORT_SPAN_BRACE = re.compile(r'"support_span"?\s*\}')
_RE_LAST_KEY_BRACE = re.compile(r'"(answer_span|target)"?\s*[\}:,]')
_RE_ANSWER_COLON = re.compile(
    r'"answer_span"\s*:\s*"((?:[^"\\]|\\.)*)"'
)
_RE_TARGET_COLON = re.compile(
    r'"target"\s*:\s*"((?:[^"\\]|\\.)*)"'
)
_RE_ANSWER_COMMA = re.compile(
    r'"answer_span"\s*,\s*"((?:[^"\\]|\\.)*)"'
)
_RE_TARGET_COMMA = re.compile(
    r'"target"\s*,\s*"((?:[^"\\]|\\.)*)"'
)


def _repair_certificate(raw: Any) -> tuple[dict[str, str] | None, bool]:
    """Attempt to extract certificate fields from malformed output.

    Returns (fields_dict, was_repaired).
      - fields_dict: dict with extracted string fields, or None if extraction
        fails for the essential fields (source_id + support_span).
      - was_repaired: True if the fallback regex path was used (i.e. the input
        was not a valid Mapping or parseable JSON).

    The function tries, in order:
      1. If *raw* is already a Mapping → return it as a plain dict.
      2. If *raw* is a string → try json.loads().
      3. Regex-based fallback extraction for the degradation patterns above.
    """
    # Fast path: already a valid mapping.
    if isinstance(raw, Mapping):
        return dict(raw), False

    if not isinstance(raw, str) or not raw.strip():
        return None, False

    text = raw.strip()

    # Try standard JSON parse.
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, False
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Fallback: regex extraction ---
    fields: dict[str, str] = {}

    # source_id — always present and well-formed in most patterns.
    m = _RE_SOURCE_ID.search(text)
    if m:
        fields["source_id"] = m.group(1)
    else:
        # Unquoted source_id value, e.g. "source_id": passage:16817
        m = _RE_SOURCE_ID_UNQUOTED.search(text)
        if m:
            fields["source_id"] = m.group(1)
        else:
            # Without source_id the certificate is unusable.
            return None, True

    # support_span — three sub-patterns.
    m = _RE_SUPPORT_SPAN_COLON.search(text)
    if m:
        fields["support_span"] = m.group(1)
    else:
        m = _RE_SUPPORT_SPAN_COMMA.search(text)
        if m:
            fields["support_span"] = m.group(1)
        else:
            # Pattern A: "support_span"} followed by the span text.
            m = _RE_SUPPORT_SPAN_BRACE.search(text)
            if m:
                # The span text starts right after the `}` and continues until
                # the next key-like pattern, e.g. "target} or "answer_span}
                rest = text[m.end():].strip()
                stop = _RE_LAST_KEY_BRACE.search(rest)
                span_text = rest[:stop.start()].strip() if stop else rest.strip()
                if span_text:
                    fields["support_span"] = span_text
                else:
                    return None, True  # support_span required but empty
            else:
                return None, True  # support_span required but not found

    # answer_span / target — the last field, most commonly broken.
    # Try standard colon-pattern first (in case only support_span was broken).
    m = _RE_ANSWER_COLON.search(text)
    if m:
        fields["answer_span"] = m.group(1)
    else:
        m = _RE_TARGET_COLON.search(text)
        if m:
            fields["target"] = m.group(1)
        else:
            # Comma pattern: "answer_span", "value"
            m = _RE_ANSWER_COMMA.search(text)
            if m:
                fields["answer_span"] = m.group(1)
            else:
                m = _RE_TARGET_COMMA.search(text)
                if m:
                    fields["target"] = m.group(1)
                else:
                    # Pattern A: "answer_span}" or "target}" — value is after `}`
                    m = _RE_LAST_KEY_BRACE.search(text)
                    if m:
                        field_name = m.group(1)
                        rest = text[m.end():].strip()
                        # The value runs to end of string.  Strip trailing
                        # braces, quotes, commas that might linger.
                        value = rest.split("\n")[0].strip().rstrip("}\"\n,")
                        # Even if empty, record the key so the parser can
                        # distinguish "field missing" from "field empty".
                        fields[field_name] = value

    return fields, True


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parsers (with fault-tolerant fallback)
# ---------------------------------------------------------------------------
# Key design principle: if source_id + support_span are recoverable, return a
# certificate object even when the last field (target / answer_span) is empty
# or missing.  The verifier will flag coupling_failed for those, but grounding
# can still be checked — which is the whole point of the process reward.


def parse_search_certificate(value: Any) -> tuple[SearchCertificate | None, tuple[str, ...]]:
    """Parse a search certificate with fault-tolerant fallback.

    Returns (cert, errors).  cert is None only when source_id or support_span
    cannot be recovered.  When the last field (target) is missing or empty,
    cert is still returned with target="" and errors includes "target_invalid".
    """
    repaired, was_repaired = _repair_certificate(value)
    if repaired is None:
        return None, ("certificate_not_object",)

    errors: list[str] = []

    # Field-presence check — lenient for repaired certs (extra keys ok,
    # missing last field is ok — we'll flag it as invalid but still parse).
    required = SEARCH_CERT_FIELDS
    if was_repaired:
        missing = required - set(repaired)
        if missing:
            errors.append("certificate_fields_invalid")
    else:
        if set(repaired) != required:
            errors.append("certificate_fields_invalid")

    source_id = repaired.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id_invalid")
    else:
        source_id = source_id.strip()

    support_span = repaired.get("support_span")
    if not isinstance(support_span, str) or not support_span.strip():
        errors.append("support_span_invalid")
    else:
        support_span = " ".join(support_span.split())

    target = repaired.get("target")
    if not isinstance(target, str) or not target.strip():
        errors.append("target_invalid")
    else:
        target = " ".join(target.split())

    if was_repaired:
        errors.append("certificate_repaired")

    cert = SearchCertificate(
        source_id=str(source_id or ""),
        support_span=str(support_span or ""),
        target=str(target or ""),
    )
    # Return cert as long as source_id + support_span are valid.
    # Missing target → coupling check will fail, but grounding can still pass.
    core_valid = isinstance(source_id, str) and source_id and isinstance(support_span, str) and support_span
    return (cert if core_valid else None), tuple(errors)


def parse_finish_certificate(value: Any) -> tuple[FinishCertificate | None, tuple[str, ...]]:
    """Parse a finish certificate with fault-tolerant fallback.

    Returns (cert, errors).  cert is None only when source_id or support_span
    cannot be recovered.  When the last field (answer_span) is missing or empty,
    cert is still returned with answer_span="" and errors includes
    "answer_span_invalid".
    """
    repaired, was_repaired = _repair_certificate(value)
    if repaired is None:
        return None, ("certificate_not_object",)

    errors: list[str] = []

    # Field-presence check — lenient for repaired certs.
    required = FINISH_CERT_FIELDS
    if was_repaired:
        missing = required - set(repaired)
        if missing:
            errors.append("certificate_fields_invalid")
    else:
        if set(repaired) != required:
            errors.append("certificate_fields_invalid")

    source_id = repaired.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id_invalid")
    else:
        source_id = source_id.strip()

    support_span = repaired.get("support_span")
    if not isinstance(support_span, str) or not support_span.strip():
        errors.append("support_span_invalid")
    else:
        support_span = " ".join(support_span.split())

    answer_span = repaired.get("answer_span")
    if not isinstance(answer_span, str) or not answer_span.strip():
        errors.append("answer_span_invalid")
    else:
        answer_span = " ".join(answer_span.split())

    if was_repaired:
        errors.append("certificate_repaired")

    cert = FinishCertificate(
        source_id=str(source_id or ""),
        support_span=str(support_span or ""),
        answer_span=str(answer_span or ""),
    )
    # Return cert as long as source_id + support_span are valid.
    core_valid = isinstance(source_id, str) and source_id and isinstance(support_span, str) and support_span
    return (cert if core_valid else None), tuple(errors)
