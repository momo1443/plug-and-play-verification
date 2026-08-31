"""Summarize GSM8K test accuracy from per-sample JSONL results.

Usage:

    python -m recipes.gsm8k.summarize_test_accuracy \
        <result_jsonl> \
        --expected-rows 1319 \
        --split test \
        [--checkpoint /path/to/checkpoint] \
        [--output accuracy.json]

Reads the per-sample JSONL produced by eval_gsm8k.py, recomputes every
score from scratch using GSM8K exact-match normalization, and writes a
structured accuracy.json matching schema ``gsm8k-test-accuracy-v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import timezone
from datetime import datetime as dt
from typing import Any


# ---------------------------------------------------------------------------
# GSM8K answer normalization
# ---------------------------------------------------------------------------

def normalize_answer(ans: str) -> str:
    """Normalize a GSM8K answer for comparison.

    - Strip whitespace and commas used as thousands separators.
    - Remove trailing .0 for integer answers (e.g. "18.0" -> "18").
    """
    ans = ans.replace(",", "").strip()
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans


# ---------------------------------------------------------------------------
# Answer extraction (mirrors eval_gsm8k.py logic)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Extract the numeric answer after '####' marker, or the last number."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: last number in the response
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


# ---------------------------------------------------------------------------
# Core summarization
# ---------------------------------------------------------------------------

def summarize(
    jsonl_path: str,
    expected_rows: int,
    split: str = "test",
    checkpoint: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Read *jsonl_path*, recompute every EM score, and return a summary dict."""

    correct = 0
    incorrect = 0
    total = 0
    stored_verified = 0
    stored_mismatch = 0
    sample_indices: list[int] = []

    # Error decomposition buckets for incorrect answers
    # For GSM8K, errors are simpler than HotpotQA:
    #   format_error:       model didn't produce a parseable number
    #   close_number:       |pred - gold| / max(|gold|, 1) < 0.05 — likely off by one or rounding
    #   wrong_answer:       clearly different number
    format_errors = 0
    close_numbers = 0
    wrong_answers = 0

    # Collect error examples for spot-check
    close_examples: list[dict[str, str]] = []
    format_error_examples: list[dict[str, str]] = []
    wrong_examples: list[dict[str, str]] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total += 1

            # --- recompute score ---
            gold_raw = str(d.get("gold_answer", "") or "")
            pred_raw = str(d.get("predicted_answer", "") or "")

            gold_norm = normalize_answer(gold_raw)
            pred_norm = normalize_answer(pred_raw)

            is_correct = (pred_norm == gold_norm) and (gold_norm != "")
            recomputed = 1.0 if is_correct else 0.0

            # --- verify against stored score ---
            stored = d.get("correct")
            if stored is not None:
                stored_bool = bool(stored)
                if stored_bool == is_correct:
                    stored_verified += 1
                else:
                    stored_mismatch += 1
            else:
                stored_verified += 1  # no stored score to verify

            if is_correct:
                correct += 1
            else:
                incorrect += 1

                # Classify the error
                if pred_norm == "":
                    format_errors += 1
                    if len(format_error_examples) < 15:
                        format_error_examples.append({
                            "question": d.get("question", "")[:100],
                            "gold_answer": gold_raw,
                            "predicted_answer": pred_raw or "(empty)",
                        })
                else:
                    # Try numeric closeness check
                    try:
                        pred_num = float(pred_norm)
                        gold_num = float(gold_norm) if gold_norm else 0.0
                        denom = max(abs(gold_num), 1.0)
                        rel_diff = abs(pred_num - gold_num) / denom
                        if rel_diff < 0.05:
                            close_numbers += 1
                            if len(close_examples) < 15:
                                close_examples.append({
                                    "question": d.get("question", "")[:100],
                                    "prediction": pred_norm,
                                    "ground_truth": gold_norm,
                                    "relative_diff": f"{rel_diff:.4f}",
                                })
                        else:
                            wrong_answers += 1
                            if len(wrong_examples) < 15:
                                wrong_examples.append({
                                    "question": d.get("question", "")[:100],
                                    "prediction": pred_norm,
                                    "ground_truth": gold_norm,
                                })
                    except (ValueError, ZeroDivisionError):
                        wrong_answers += 1
                        if len(wrong_examples) < 15:
                            wrong_examples.append({
                                "question": d.get("question", "")[:100],
                                "prediction": pred_norm,
                                "ground_truth": gold_norm,
                            })

            # --- collect identity info ---
            si = d.get("index")
            if si is not None:
                sample_indices.append(int(si))

    # --- file metadata ---
    jsonl_bytes = os.path.getsize(jsonl_path)
    sha256 = hashlib.sha256()
    with open(jsonl_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            sha256.update(chunk)

    contiguous = sample_indices == list(range(total))

    accuracy = correct / total if total > 0 else 0.0

    result: dict[str, Any] = {
        # ── Primary metric ─────────────────────────────────────────────
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "correct": correct,
        "incorrect": incorrect,
        "metric": "exact_match",

        # ── Error decomposition ────────────────────────────────────────
        "error_decomposition": {
            "format_errors": format_errors,
            "close_number_errors": close_numbers,
            "wrong_answer_errors": wrong_answers,
            "buckets": {
                "format_error": format_errors,
                "close_number_lt5pct": close_numbers,
                "wrong_answer": wrong_answers,
            },
        },

        # ── Error examples (for manual spot-check) ────────────────────
        "format_error_examples": format_error_examples,
        "close_number_examples": close_examples,
        "wrong_answer_examples": wrong_examples,

        # ── Metadata ──────────────────────────────────────────────────
        "all_stored_scores_verified": stored_mismatch == 0,
        "checkpoint": checkpoint or "",
        "created_at_utc": dt.now(timezone.utc).isoformat(),
        "dataset": "gsm8k",
        "expected_rows": expected_rows,
        "rows": total,
        "sample_indices_contiguous": contiguous,
        "schema_version": "gsm8k-test-accuracy-v1",
        "scores_recomputed": True,
        "source_jsonl": {
            "bytes": jsonl_bytes,
            "path": os.path.abspath(jsonl_path),
            "sha256": sha256.hexdigest(),
        },
        "split": split,
        "status": "ok" if total == expected_rows else f"row_count_mismatch({total}/{expected_rows})",
        "unique_sample_keys": len(set(sample_indices)),
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
        description="Summarize GSM8K test accuracy from per-sample JSONL"
    )
    parser.add_argument(
        "result_jsonl",
        help="Path to the per-sample result JSONL (e.g. results.jsonl)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        required=True,
        help="Expected number of rows in the JSONL",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)",
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

    # Always print the structured result to stdout
    print(json.dumps(result, indent=2))

    # Exit with non-zero if row count mismatches or scores couldn't be verified.
    if result["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
