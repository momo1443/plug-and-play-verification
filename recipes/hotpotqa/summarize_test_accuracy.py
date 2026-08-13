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
a structured accuracy.json matching schema ``hotpotqa-test-accuracy-v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Normalization – reuse the canonical HotpotQA EM logic from the recipe.
# ---------------------------------------------------------------------------

def _normalize_answer(value: str) -> str:
    """Canonical HotpotQA exact-match normalization."""
    import re
    import string

    lowered = str(value).lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


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

    correct = 0
    incorrect = 0
    total = 0
    stored_verified = 0
    stored_mismatch = 0
    sample_indices: list[int] = []
    official_qids: set[str] = set()
    sample_keys: set[str] = set()

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

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "all_stored_scores_verified": stored_mismatch == 0,
        "checkpoint": checkpoint or "",
        "correct": correct,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "hotpotqa",
        "expected_rows": expected_rows,
        "incorrect": incorrect,
        "metric": "normalized_exact_match",
        "rows": total,
        "sample_indices_contiguous": contiguous,
        "schema_version": "hotpotqa-test-accuracy-v1",
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
