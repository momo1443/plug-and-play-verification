#!/usr/bin/env python3
"""Standalone GSM8K evaluation with pure vLLM inference — no training framework needed.

Usage:
    CUDA_VISIBLE_DEVICES=7 python eval_gsm8k.py

Computes exact-match accuracy on GSM8K test set using greedy decoding.
"""

import json
import re
import os
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# GSM8K answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Extract the numeric answer after '####' marker, or the last number."""
    # Standard GSM8K format: answer after ####
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: last number in the response
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = ans.replace(",", "").strip()
    # Remove trailing .0 for integer answers
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_path = os.environ.get(
        "MODEL_PATH",
        "/nas/deepresearch/zsb/corhort/project/agenticrl/models/Qwen3.5-4B",
    )
    data_dir = os.environ.get(
        "GSM8K_DATA_DIR",
        "/nas/deepresearch/zsb/corhort/project/agenticrl/Agent-R1/data/corpus/gsm8k",
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        "/nas/deepresearch/zsb/corhort/project/agenticrl/logs/qwen35-4b_a0_gsm8k_standalone",
    )
    max_samples = int(os.environ.get("MAX_SAMPLES", "-1"))  # -1 = all

    # Load dataset from HuggingFace (parquet was deleted)
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    questions = [item["question"] for item in ds]
    answers_raw = [item["answer"] for item in ds]

    # Extract gold answers from #### markers
    gold_answers = []
    for a in answers_raw:
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", a)
        gold_answers.append(normalize_answer(match.group(1)) if match else "")

    if max_samples > 0:
        questions = questions[:max_samples]
        gold_answers = gold_answers[:max_samples]

    print(f"=== GSM8K Standalone Evaluation ===")
    print(f"Model:    {model_path}")
    print(f"Samples:  {len(questions)}")
    print(f"Output:   {output_dir}")
    print(f"===================================")

    # Build prompts — Qwen3.5 chat template for CoT
    prompts = []
    for q in questions:
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful math problem solver. "
            "Solve the problem step by step, then write the final numeric answer after ####.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{q}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)

    # Initialize vLLM
    print("\nLoading model with vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    # Greedy decoding
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=1024,
        stop=["<|im_end|>", "```"],
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

    for i, (output, gold) in enumerate(zip(outputs, gold_answers)):
        response = output.outputs[0].text.strip()
        predicted = normalize_answer(extract_answer(response))
        is_correct = predicted == gold and gold != ""
        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "question": questions[i],
            "gold_answer": gold,
            "predicted_answer": predicted,
            "response": response,
            "correct": is_correct,
        })

    accuracy = correct / total * 100 if total > 0 else 0

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(output_dir) / "results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": model_path,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
    }
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Total:     {total}")
    print(f"Correct:   {correct}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Results:   {results_path}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
