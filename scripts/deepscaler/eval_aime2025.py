#!/usr/bin/env python3
"""Standalone AIME 2025 evaluation with pure vLLM inference — no training framework needed.

Runs the same bounded reasoning-continuation protocol as math training. Only
the submitted final response is scored; the reference is never model feedback.

Usage:
    CUDA_VISIBLE_DEVICES=2 python eval_aime2025.py

Computes exact-match accuracy on AIME 2025 (I+II, 30 problems) using greedy decoding.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent_r1.evaluation.answers import AIME_EXTRACTION_VERSION, final_answer_record, score_aime
from agent_r1.evaluation.consistency import consistency_record
from agent_r1.evaluation.summarize import summarize
from recipes.deepscaler.trajectory_reward import verify_process
from recipes.deepscaler.prompts import build_math_messages, math_continuation_message


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_math_rollouts(llm, tokenizer, sampling_params_factory, questions, *,
                           max_steps=5, max_tokens=4096, max_model_length=8192,
                           max_prompt_length=2048):
    """Batch active trajectories; share the training prompt and total token budget."""
    if max_steps < 1 or max_tokens < 1:
        raise ValueError("Math turn count and total token budget must be positive")
    turn_token_limit = (max_tokens + max_steps - 1) // max_steps
    states = []
    for question in questions:
        initial_ids = tokenizer.apply_chat_template(question, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if len(initial_ids) > max_prompt_length:
            raise ValueError("Math evaluation prompt exceeds the configured input limit")
        states.append({"messages": build_math_messages(question, max_steps), "responses": [],
                       "reasoning_segments": [], "final_response": None, "response_tokens": 0,
                       "termination_reason": "max_steps"})
    active = list(range(len(states)))
    for turn in range(1, max_steps + 1):
        groups = {}
        for index in active:
            state = states[index]
            prompt = tokenizer.apply_chat_template(state["messages"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            prompt_ids = tokenizer.apply_chat_template(state["messages"], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            remaining = max_tokens - state["response_tokens"]
            limit = min(remaining, turn_token_limit, max_model_length - len(prompt_ids))
            if limit <= 0:
                state["termination_reason"] = "token_budget" if remaining <= 0 else "context_limit"
                continue
            groups.setdefault(limit, []).append((index, prompt))
        next_active = []
        for limit, requests in groups.items():
            params = sampling_params_factory(temperature=0, top_p=1.0, max_tokens=limit,
                                            stop=["<|im_end|>"])
            outputs = llm.generate([prompt for _, prompt in requests], params)
            if len(outputs) != len(requests):
                raise RuntimeError("Generation count differs from the evaluation denominator")
            for (index, _), output in zip(requests, outputs):
                state = states[index]
                candidate = output.outputs[0]
                ids = list(candidate.token_ids[:limit])
                if not ids:
                    state["termination_reason"] = "empty_generation"
                    continue
                state["response_tokens"] += len(ids)
                response = tokenizer.decode(ids, skip_special_tokens=True)
                final = final_answer_record(response)
                state["responses"].append(response)
                state["reasoning_segments"].append((turn, str(final["reasoning"]) if final else response))
                if final is not None:
                    state["final_response"] = response
                    state["termination_reason"] = "final_answer"
                    continue
                state["messages"].append({"role": "assistant", "content": response})
                state["messages"].append(math_continuation_message(final_turn=turn + 1 == max_steps))
                next_active.append(index)
        active = next_active
        if not active:
            break
    return states


def main():
    from vllm import LLM, SamplingParams

    project_dir = Path(__file__).resolve().parents[2]
    workspace_dir = Path(os.environ.get("WORKSPACE_DIR", str(project_dir.parent)))
    model_path = os.environ.get("MODEL_PATH", str(workspace_dir / "models/Qwen3.5-4B"))
    data_dir = os.environ.get(
        "AIME2025_DATA_DIR",
        str(project_dir / "data/corpus/aime2025"),
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        str(workspace_dir / "logs" / f"{Path(model_path).name}_a0_aime2025_v3"),
    )
    max_samples = int(os.environ.get("MAX_SAMPLES", "-1"))  # -1 = all
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    scale = os.environ.get("AGENT_R1_MODEL_SCALE", "9b" if Path(model_path).name == "Qwen3.5-9B" else "4b")
    max_tokens = int(os.environ.get("DEEPSCALER_MAX_RESPONSE_LENGTH",
        os.environ.get("MAX_RESPONSE_LENGTH", "5120" if scale == "9b" else "4096")))
    max_steps = int(os.environ.get("DEEPSCALER_MAX_STEPS", "5"))
    max_model_length = int(os.environ.get("DEEPSCALER_MAX_MODEL_LENGTH",
        str(max(8192, 2048 + max_tokens + 1024))))

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
    print(f"=== AIME 2025 A0 Baseline Evaluation (v3 — final answer only) ===")
    print(f"Model:    {model_name}")
    print(f"Samples:  {len(questions)}")
    print(f"Output:   {output_dir}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"===================================================================")

    # Build prompts using Qwen3.5 chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Loading model with vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_length,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    print(f"Generating up to {max_steps} turns with {max_tokens} total response tokens...")
    t0 = time.time()
    trajectories = generate_math_rollouts(llm, tokenizer, SamplingParams, questions,
        max_steps=max_steps, max_tokens=max_tokens, max_model_length=max_model_length)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s")

    # Evaluate
    correct = 0
    total = len(questions)
    results = []

    for i, (trajectory, gold, source) in enumerate(zip(trajectories, gold_answers, sources)):
        response = trajectory["final_response"] or ""
        gold_norm = str(gold).strip()

        scored = score_aime(response, gold)
        is_correct = scored["is_correct"]
        correct += int(is_correct)

        final = final_answer_record(response)
        verification = verify_process(trajectory["reasoning_segments"])
        results.append({
            **consistency_record(verification, float(is_correct), eligible=final is not None),
            "index": i,
            "source": source,
            "gold_answer_raw": gold,
            "gold_answer_normalized": gold_norm,
            "predicted_answer": scored["predicted_answer"],
            "match_method": scored["match_method"],
            "is_correct": is_correct,
            "response": "\n\n".join(trajectory["responses"]),
            "final_response": trajectory["final_response"],
            "policy_step_count": len(trajectory["responses"]),
            "generated_response_tokens": trajectory["response_tokens"],
            "termination_reason": trajectory["termination_reason"],
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
        "verification_metrics": summarize(results),
        "max_response_length": max_tokens,
        "response_budget_scope": "whole_trajectory",
        "max_agent_steps": max_steps,
        "max_model_length": max_model_length,
        "interaction": "reasoning_continuation_without_correctness_feedback",
        "verification_timing": "after_rollout",
        "model": model_name,
        "model_path": model_path,
        "dataset": "AIME2025",
        "arm": "A0",
        "extraction_version": AIME_EXTRACTION_VERSION,
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
