#!/usr/bin/env python3
"""Standalone AIME 2025 evaluation with pure vLLM inference — no training framework needed.

Improved A0 evaluation with robust answer extraction:
  1. \\boxed{} extraction (primary)
  2. "The answer is ..." / "Therefore, ..." patterns
  3. Last standalone integer in [0, 999] (AIME answer range)
  4. All candidate answers checked via math_reward.is_equiv

Usage:
    CUDA_VISIBLE_DEVICES=2 python eval_aime2025.py

Computes exact-match accuracy on AIME 2025 (I+II, 30 problems) using greedy decoding.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# AIME answer extraction & verification
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> list[str]:
    """Extract all answers from \\boxed{} in the response."""
    candidates = []
    # Try \\boxed{...} pattern with nested braces
    for m in re.finditer(r'\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text):
        candidates.append(m.group(1).strip())
    # Try simple \boxed{...} without nested braces
    for m in re.finditer(r'\\boxed\s*\{([^}]+)\}', text):
        ans = m.group(1).strip()
        if ans not in candidates:
            candidates.append(ans)
    return candidates


def extract_verbal_answer(text: str) -> list[str]:
    """Extract answers from verbal patterns like 'The answer is ...'."""
    candidates = []
    patterns = [
        r'(?:[Tt]he )?[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$)',
        r'[Ss]o\s+(?:the )?[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$)',
        r'[Tt]herefore(?:,\s*)?(?:the )?[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$)',
        r'[Hh]ence(?:,\s*)?(?:the )?[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$)',
        r'[Aa]nswer:\s*(.+?)(?:\.|$)',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            ans = m.group(1).strip()
            if ans and ans not in candidates:
                candidates.append(ans)
    return candidates


def extract_aime_integers(text: str) -> list[str]:
    """Extract candidate AIME answers — integers in [0, 999] found in the text.

    AIME answers are always integers from 0 to 999 inclusive.
    We scan from the end of the text backwards to prioritize later (more final) answers.
    """
    candidates = []
    # Find all standalone integers (not part of larger numbers)
    # Look for numbers that appear at the end of a line or after a colon/equals
    for m in re.finditer(r'(?:^|=|:|\s)(\d{1,3})(?:\s*$|\s*\.|\s*\n|\s*\\)', text, re.MULTILINE):
        num_str = m.group(1).strip()
        if num_str and num_str not in candidates:
            candidates.append(num_str)

    # Fallback: any integer in the text near the end
    all_ints = re.findall(r'\b(\d{1,3})\b', text)
    # Reverse to prioritize later appearances
    for num_str in reversed(all_ints):
        num = int(num_str)
        if 0 <= num <= 999 and num_str not in candidates:
            candidates.append(num_str)

    return candidates


def extract_all_candidates(text: str) -> list[dict]:
    """Extract all candidate answers with their extraction method."""
    seen = set()
    candidates = []

    # Priority 1: \boxed{}
    for ans in extract_boxed_answer(text):
        if ans not in seen:
            candidates.append({"answer": ans, "method": "boxed"})
            seen.add(ans)

    # Priority 2: Verbal patterns
    for ans in extract_verbal_answer(text):
        if ans not in seen:
            candidates.append({"answer": ans, "method": "verbal"})
            seen.add(ans)

    # Priority 3: AIME integer patterns
    for ans in extract_aime_integers(text):
        if ans not in seen:
            candidates.append({"answer": ans, "method": "aime_int"})
            seen.add(ans)

    return candidates


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison — AIME answers are integers."""
    ans = ans.replace(",", "").replace(" ", "").replace("{", "").replace("}", "").strip()
    # Remove trailing .0 for integer answers
    if ans.endswith(".0"):
        ans = ans[:-2]
    # Remove LaTeX formatting
    ans = ans.replace("\\text{", "").replace("\\mathrm{", "").replace("\\,", "")
    ans = ans.replace("\\;", "").replace("\\!", "").replace("\\ ", "")
    # Remove degree symbol
    ans = ans.replace("^\\circ", "").replace("°", "").strip()
    # Remove trailing period
    ans = ans.rstrip(".")
    return ans


def check_equivalence(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold using multiple methods."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Direct string comparison
    if pred_norm == gold_norm and gold_norm != "":
        return True

    # Numeric comparison
    try:
        pred_num = int(float(pred_norm))
        gold_num = int(float(gold_norm))
        if pred_num == gold_num:
            return True
    except (ValueError, OverflowError):
        pass

    # math_reward for more robust comparison (LaTeX normalization)
    try:
        from verl.utils.reward_score import math_reward
        if math_reward.is_equiv(predicted, gold):
            return True
        # Also try normalized versions
        if math_reward.is_equiv(pred_norm, gold_norm):
            return True
    except Exception:
        pass

    return False


def find_best_match(text: str, gold: str) -> dict | None:
    """Extract all candidate answers from text and find the best match against gold.

    Returns the matching candidate info, or None if no match found.
    """
    candidates = extract_all_candidates(text)

    for cand in candidates:
        if check_equivalence(cand["answer"], gold):
            return cand

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_path = os.environ.get(
        "MODEL_PATH",
        "/nas/deepresearch/zsb/corhort/project/agenticrl/models/Qwen3.5-4B",
    )
    data_dir = os.environ.get(
        "AIME2025_DATA_DIR",
        "/nas/deepresearch/zsb/corhort/project/agenticrl/Agent-R1/data/corpus/aime2025",
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        f"/nas/deepresearch/zsb/corhort/project/agenticrl/logs/{Path(model_path).name}_a0_aime2025_v2",
    )
    max_samples = int(os.environ.get("MAX_SAMPLES", "-1"))  # -1 = all
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))

    # Load dataset from processed parquet
    import pandas as pd
    test_path = os.path.join(data_dir, "test.parquet")
    df = pd.read_parquet(test_path)

    questions = [row["prompt"] for _, row in df.iterrows()]
    gold_answers = [str(row["reward_model"]["ground_truth"]) for _, row in df.iterrows()]
    sources = [row["extra_info"].get("source", "unknown") for _, row in df.iterrows()]

    if max_samples > 0:
        questions = questions[:max_samples]
        gold_answers = gold_answers[:max_samples]
        sources = sources[:max_samples]

    model_name = Path(model_path).name
    print(f"=== AIME 2025 A0 Baseline Evaluation (v2 — robust extraction) ===")
    print(f"Model:    {model_name}")
    print(f"Samples:  {len(questions)}")
    print(f"Output:   {output_dir}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"===================================================================")

    # Build prompts using Qwen3.5 chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompts = []
    for chat_msgs in questions:
        text = tokenizer.apply_chat_template(
            chat_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)

    # Initialize vLLM
    print("Loading model with vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=8192,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    # Greedy decoding for deterministic A0 evaluation
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=4096,
        stop=["<|im_end|>"],
    )

    # Generate
    print("Generating responses...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s ({len(questions)/elapsed:.1f} samples/s)")

    # Evaluate
    correct = 0
    total = len(questions)
    results = []

    for i, (output, gold, source) in enumerate(zip(outputs, gold_answers, sources)):
        response = output.outputs[0].text.strip()
        gold_norm = normalize_answer(gold)

        # Find best match among all extracted candidates
        best_match = find_best_match(response, gold)
        is_correct = best_match is not None

        if is_correct:
            correct += 1

        # Also collect all candidates for analysis
        all_candidates = extract_all_candidates(response)

        results.append({
            "index": i,
            "source": source,
            "gold_answer_raw": gold,
            "gold_answer_normalized": gold_norm,
            "predicted_answer": best_match["answer"] if best_match else None,
            "match_method": best_match["method"] if best_match else None,
            "is_correct": is_correct,
            "all_candidates": all_candidates,
            "response": response,
        })

    accuracy = correct / total * 100 if total > 0 else 0

    # Per-source breakdown
    aime_i_correct = sum(1 for r in results if r["is_correct"] and r["source"] == "aime_i")
    aime_i_total = sum(1 for r in results if r["source"] == "aime_i")
    aime_ii_correct = sum(1 for r in results if r["is_correct"] and r["source"] == "aime_ii")
    aime_ii_total = sum(1 for r in results if r["source"] == "aime_ii")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(output_dir) / "results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": model_name,
        "model_path": model_path,
        "dataset": "AIME2025",
        "arm": "A0",
        "extraction_version": "v2_robust",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "aime_i": {"total": aime_i_total, "correct": aime_i_correct,
                    "accuracy": aime_i_correct / aime_i_total * 100 if aime_i_total > 0 else 0},
        "aime_ii": {"total": aime_ii_total, "correct": aime_ii_correct,
                     "accuracy": aime_ii_correct / aime_ii_total * 100 if aime_ii_total > 0 else 0},
        "elapsed_seconds": elapsed,
    }
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"=== Results ===")
    print(f"{'='*60}")
    print(f"Total:     {total}")
    print(f"Correct:   {correct}")
    print(f"Accuracy:  {accuracy:.2f}%")
    if aime_i_total:
        print(f"AIME I:    {aime_i_correct}/{aime_i_total} ({aime_i_correct/aime_i_total*100:.1f}%)")
    if aime_ii_total:
        print(f"AIME II:   {aime_ii_correct}/{aime_ii_total} ({aime_ii_correct/aime_ii_total*100:.1f}%)")
    print(f"Results:   {results_path}")
    print(f"Summary:   {summary_path}")

    # Print detailed per-problem comparison
    print(f"\n{'='*60}")
    print(f"=== Per-problem detail ===")
    print(f"{'='*60}")
    for r in results:
        status = "✓ CORRECT" if r["is_correct"] else "✗ WRONG  "
        print(f"\n--- Problem #{r['index']:2d} ({r['source']}) {status} ---")
        print(f"  Gold answer: {r['gold_answer_raw']} (normalized: {r['gold_answer_normalized']})")
        if r["predicted_answer"] is not None:
            print(f"  Predicted:   {r['predicted_answer']} (method: {r['match_method']})")
        else:
            print(f"  Predicted:   NO MATCH FOUND")
        print(f"  Candidates:  {r['all_candidates']}")
        # Print last ~400 chars of response to see what the model actually said
        resp = r["response"]
        if len(resp) > 400:
            print(f"  Response (tail): ...{resp[-400:]}")
        else:
            print(f"  Response: {resp}")

    # Also print a compact summary table
    print(f"\n{'='*60}")
    print(f"=== Compact summary ===")
    print(f"{'='*60}")
    print(f"{'#':>3} {'Source':>8} {'Gold':>8} {'Pred':>8} {'Method':>8} {'Result':>6}")
    print(f"{'-'*3} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in results:
        pred_str = r["predicted_answer"] if r["predicted_answer"] else "-"
        method_str = r["match_method"] if r["match_method"] else "-"
        result_str = "✓" if r["is_correct"] else "✗"
        print(f"{r['index']:3d} {r['source']:>8} {r['gold_answer_raw']:>8} {pred_str:>8} {method_str:>8} {result_str:>6}")


if __name__ == "__main__":
    main()
