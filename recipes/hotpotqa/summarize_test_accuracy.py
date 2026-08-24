"""Summarize HotpotQA test accuracy from per-sample JSONL results.

Usage (invoked by evaluate_hotpotqa_full_validation.sh):

    python -m recipes.hotpotqa.summarize_test_accuracy \
        <result_jsonl> \
        --expected-rows 7405 \
        --split validation \
        [--checkpoint /path/to/checkpoint] \
        [--output accuracy.json]

Reads the per-sample JSONL produced by streaming validation, recomputes
every score from scratch using the official EM normalization, and writes
a structured accuracy.json matching schema ``hotpotqa-test-accuracy-v2``.

v2 additions (diagnostic only, official EM remains the primary metric):
  - token_f1:           mean token-level F1 (HotpotQA official style)
  - norm_em:            EM after normalize() — same as official for this dataset
  - entity_aware_em:    EM with alias-containment check for entity answers
  - error_decomposition: bucketing of EM=0 errors into
        answer_realization vs reasoning/search
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Normalization – reuse the canonical HotpotQA EM logic from the recipe.
# ---------------------------------------------------------------------------

def _normalize_answer(value: str) -> str:
    """Canonical HotpotQA exact-match normalization."""
    lowered = str(value).lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


# ---------------------------------------------------------------------------
# Token F1 (HotpotQA official style)
# ---------------------------------------------------------------------------

def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 using Counter-based overlap."""
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Entity-aware EM with alias containment check
# ---------------------------------------------------------------------------

def _is_alias_match(pred: str, gold: str) -> bool:
    """Conservative alias check: one string contained in the other after
    normalization, AND the shorter one has >= 2 tokens (avoid trivial
    matches like "Lee" ⊂ "Lee Hazlewood").
    """
    pn = _normalize_answer(pred)
    gn = _normalize_answer(gold)
    if pn == gn:
        return True
    # Must share at least one non-trivial token
    pn_tokens = set(pn.split())
    gn_tokens = set(gn.split())
    if not (pn_tokens & gn_tokens):
        return False
    # Containment check
    if gn in pn or pn in gn:
        shorter = min(len(pn.split()), len(gn.split()))
        return shorter >= 2
    return False


# ---------------------------------------------------------------------------
# Token overlap ratio (for error bucketing)
# ---------------------------------------------------------------------------

def _token_overlap_ratio(pred: str, gold: str) -> float:
    """Fraction of gold tokens present in prediction."""
    pn = set(_normalize_answer(pred).split())
    gn = set(_normalize_answer(gold).split())
    if not gn:
        return 0.0
    return len(pn & gn) / len(gn)


# ---------------------------------------------------------------------------
# Answer extraction – mirrors reward_fn._extract_answer_from_solution
# ---------------------------------------------------------------------------

def _extract_answer_from_completion(completion: str) -> str:
    """Extract the predicted answer from a model completion string.

    Tries in order:
    1. A8-LR finish tool envelope
    2. Last <answer>...</answer> pair
    3. Full string fallback
    """
    # --- A8-LR finish envelope ---
    try:
        from recipes.hotpotqa_lr.protocol import parse_finish

        finish = parse_finish(completion)
        if finish.envelope_valid:
            return str(finish.answer) if finish.answer is not None else ""
    except ImportError:
        pass

    # --- <answer>...</answer> tags ---
    import re as _re

    lowered = completion.lower()
    think_close = lowered.rfind("</think>")
    if think_close >= 0:
        completion = completion[think_close + len("</think>"):]
        lowered = completion.lower()
    close_start = lowered.rfind("</answer>")
    if close_start >= 0:
        open_start = lowered.rfind("<answer>", 0, close_start)
        if open_start >= 0:
            answer_start = open_start + len("<answer>")
            return completion[answer_start:close_start].strip()
    return completion.strip()


# ---------------------------------------------------------------------------
# EM verification
# ---------------------------------------------------------------------------

def _exact_match_score(prediction: str, ground_truths: list[str]) -> float:
    """Return 1.0 if the normalized prediction matches any normalized ground truth.

    Handles edge cases where normalization strips everything (e.g. "The The"
    or "!!!") by falling back to raw string equality.
    """
    norm_pred = _normalize_answer(prediction)
    norm_gts = {_normalize_answer(gt) for gt in ground_truths if gt}
    if norm_pred and norm_pred in norm_gts:
        return 1.0
    # Fall back: if normalization produced empty strings on both sides,
    # compare the raw stripped strings instead.
    raw_pred = prediction.strip()
    if raw_pred and raw_pred in {gt.strip() for gt in ground_truths if gt.strip()}:
        return 1.0
    return 0.0


def _collect_ground_truths(payload: Any) -> list[str]:
    """Convert a ground-truth field into a list of answer strings."""
    if payload is None:
        return []
    if isinstance(payload, str):
        s = payload.strip()
        return [s] if s else []
    if isinstance(payload, (list, tuple, set)):
        return [str(item).strip() for item in payload if str(item).strip()]
    s = str(payload).strip()
    return [s] if s else []


# ---------------------------------------------------------------------------
# Core summarization
# ---------------------------------------------------------------------------

def summarize(
    jsonl_path: str,
    expected_rows: int,
    split: str = "validation",
    checkpoint: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Read *jsonl_path*, recompute every EM score, and return a summary dict."""

    # Per-record accumulators
    correct = 0
    incorrect = 0
    total = 0
    stored_verified = 0
    stored_mismatch = 0
    sample_indices: list[int] = []
    official_qids: set[str] = set()
    sample_keys: set[str] = set()

    # Diagnostic accumulators
    f1_sum = 0.0
    norm_em_correct = 0
    entity_aware_em_correct = 0

    # Error decomposition for entity answers where official EM = 0
    # Buckets:
    #   norm_em_fixable:     normalize() alone fixes it (official_em=0, norm_em=1)
    #   alias_only:          alias containment fixes it (official_em=0, norm_em=0, alias=1)
    #   partial_overlap:     token_overlap >= 0.5 but not alias — possibly related entity
    #   low_overlap:         token_overlap 0.1–0.5 — likely wrong entity
    #   no_overlap:          token_overlap < 0.1 — completely different answer
    entity_wrong_norm_fixable = 0
    entity_wrong_alias_only = 0
    entity_wrong_partial_overlap = 0
    entity_wrong_low_overlap = 0
    entity_wrong_no_overlap = 0

    entity_total = 0
    entity_correct = 0
    yesno_total = 0
    yesno_correct = 0

    # Collect alias-error examples for spot-check
    alias_examples: list[dict[str, str]] = []
    norm_fixable_examples: list[dict[str, str]] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total += 1

            # --- recompute score ---
            answer = str(d.get("answer", "") or "")
            ground_truth = d.get("ground_truth")
            gts = _collect_ground_truths(ground_truth)
            recomputed = _exact_match_score(answer, gts)

            # --- verify against stored score ---
            stored = float(d.get("score", -1))
            if abs(stored - recomputed) < 1e-6:
                stored_verified += 1
            else:
                stored_mismatch += 1

            if recomputed > 0.5:
                correct += 1
            else:
                incorrect += 1

            # --- diagnostic metrics ---
            gold_str = gts[0] if gts else ""
            f1_sum += _token_f1(answer, gold_str)

            norm_em = int(_normalize_answer(answer) == _normalize_answer(gold_str))
            norm_em_correct += norm_em

            is_alias = int(_is_alias_match(answer, gold_str))
            entity_aware_em_correct += (1 if recomputed > 0.5 else is_alias)

            # --- classify yes/no vs entity ---
            qtype = "yesno" if gold_str.lower() in ("yes", "no") else "entity"
            if qtype == "yesno":
                yesno_total += 1
                yesno_correct += (1 if recomputed > 0.5 else 0)
            else:
                entity_total += 1
                entity_correct += (1 if recomputed > 0.5 else 0)

                if recomputed < 0.5:
                    overlap = _token_overlap_ratio(answer, gold_str)
                    if norm_em == 1:
                        entity_wrong_norm_fixable += 1
                        if len(norm_fixable_examples) < 15:
                            norm_fixable_examples.append({
                                "question": d.get("question", "")[:100],
                                "prediction": answer,
                                "ground_truth": gold_str,
                            })
                    elif is_alias == 1:
                        entity_wrong_alias_only += 1
                        if len(alias_examples) < 15:
                            alias_examples.append({
                                "question": d.get("question", "")[:100],
                                "prediction": answer,
                                "ground_truth": gold_str,
                                "overlap": f"{overlap:.2f}",
                            })
                    elif overlap >= 0.5:
                        entity_wrong_partial_overlap += 1
                    elif overlap >= 0.1:
                        entity_wrong_low_overlap += 1
                    else:
                        entity_wrong_no_overlap += 1

            # --- collect identity info ---
            si = d.get("sample_index")
            if si is not None:
                sample_indices.append(int(si))
            oqid = d.get("official_qid")
            if oqid is not None:
                official_qids.add(str(oqid))
            sk = d.get("sample_key")
            if sk is not None:
                sample_keys.add(str(sk))

    # --- file metadata ---
    jsonl_bytes = os.path.getsize(jsonl_path)
    sha256 = hashlib.sha256()
    with open(jsonl_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            sha256.update(chunk)

    contiguous = sample_indices == list(range(total))

    accuracy = correct / total if total > 0 else 0.0
    mean_f1 = f1_sum / total if total > 0 else 0.0

    # Error decomposition summary
    answer_realization_errors = entity_wrong_norm_fixable + entity_wrong_alias_only
    reasoning_search_errors = (entity_wrong_partial_overlap
                               + entity_wrong_low_overlap
                               + entity_wrong_no_overlap)
    entity_wrong_total = entity_total - entity_correct

    result: dict[str, Any] = {
        # ── Primary metric (unchanged) ──────────────────────────────────
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "correct": correct,
        "incorrect": incorrect,
        "metric": "normalized_exact_match",

        # ── Diagnostic metrics ──────────────────────────────────────────
        "token_f1": round(mean_f1, 6),
        "token_f1_percent": round(mean_f1 * 100, 2),
        "norm_em_correct": norm_em_correct,
        "norm_em_percent": round(norm_em_correct / total * 100, 2) if total else 0.0,
        "entity_aware_em_correct": entity_aware_em_correct,
        "entity_aware_em_percent": round(entity_aware_em_correct / total * 100, 2) if total else 0.0,

        # ── Error decomposition (entity answers only) ──────────────────
        "error_decomposition": {
            "entity_total": entity_total,
            "entity_correct": entity_correct,
            "entity_wrong_total": entity_wrong_total,
            "answer_realization_errors": answer_realization_errors,
            "reasoning_search_errors": reasoning_search_errors,
            "buckets": {
                "norm_em_fixable": entity_wrong_norm_fixable,
                "alias_only": entity_wrong_alias_only,
                "partial_overlap_ge0.5": entity_wrong_partial_overlap,
                "low_overlap_0.1_to_0.5": entity_wrong_low_overlap,
                "no_overlap_lt0.1": entity_wrong_no_overlap,
            },
            "if_answer_realization_fixed": {
                "entity_em_percent": round(
                    (entity_correct + answer_realization_errors) / entity_total * 100, 2
                ) if entity_total else 0.0,
                "overall_em_percent": round(
                    (yesno_correct + entity_correct + answer_realization_errors) / total * 100, 2
                ) if total else 0.0,
            },
        },

        # ── Yes/No breakdown ───────────────────────────────────────────
        "yesno_total": yesno_total,
        "yesno_correct": yesno_correct,
        "yesno_accuracy_percent": round(yesno_correct / yesno_total * 100, 2) if yesno_total else 0.0,
        "entity_accuracy_percent": round(entity_correct / entity_total * 100, 2) if entity_total else 0.0,

        # ── Alias-error examples (for manual spot-check) ──────────────
        "alias_error_examples": alias_examples,
        "norm_fixable_examples": norm_fixable_examples,

        # ── Metadata (unchanged from v1) ──────────────────────────────
        "all_stored_scores_verified": stored_mismatch == 0,
        "checkpoint": checkpoint or "",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "hotpotqa",
        "expected_rows": expected_rows,
        "rows": total,
        "sample_indices_contiguous": contiguous,
        "schema_version": "hotpotqa-test-accuracy-v2",
        "scores_recomputed": True,
        "source_jsonl": {
            "bytes": jsonl_bytes,
            "path": os.path.abspath(jsonl_path),
            "sha256": sha256.hexdigest(),
        },
        "split": split,
        "status": "ok" if total == expected_rows else f"row_count_mismatch({total}/{expected_rows})",
        "unique_official_qids": len(official_qids),
        "unique_sample_keys": len(sample_keys),
    }

    if stored_mismatch > 0:
        result["_warnings"] = [
            f"{stored_mismatch}/{total} stored scores did not match recomputed EM"
        ]

    # --- write output ---
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"Wrote {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize HotpotQA test accuracy from per-sample JSONL"
    )
    parser.add_argument(
        "result_jsonl",
        help="Path to the per-sample result JSONL (e.g. 0.jsonl)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        required=True,
        help="Expected number of rows in the JSONL",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint directory path for the report",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write accuracy.json (default: print to stdout only)",
    )
    args = parser.parse_args()

    result = summarize(
        jsonl_path=args.result_jsonl,
        expected_rows=args.expected_rows,
        split=args.split,
        checkpoint=args.checkpoint,
        output_path=args.output,
    )

    # Always print the structured result to stdout (captured by tee in the
    # shell script to produce accuracy_validation.log).
    print(json.dumps(result, indent=2))

    # Exit with non-zero if row count mismatches or scores couldn't be verified.
    if result["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
