#!/usr/bin/env python3
"""Checkpoint evaluation: one bounded math-agent attempt per AIME problem."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent_r1.evaluation.answers import aime_integer, final_answer_record
from recipes.deepscaler.prompts import build_math_messages, math_continuation_message
from recipes.deepscaler.math_expressions import canonical_answer, evaluate, latex_to_python
from recipes.deepscaler.submission import (
    FINAL_ANSWER_PREFIX, SUBMISSION_CHUNK_TOKENS, completed_prefix_tokens, prefilled_final_record,
)


def selected_aime_integer(answer):
    """Canonicalize one submitted numeric expression without a reference.

    Full expressions such as 252+25 and integral fractions are acceptable.
    Unknown symbols, ranges, alternatives and noninteger values fail closed.
    No number is selected from the narrative or matched against the gold.
    """
    if answer is None:
        return None
    literal = aime_integer(answer)
    if literal is not None:
        return literal
    expression = latex_to_python(canonical_answer(answer))
    if expression is None:
        return None
    value = evaluate(expression)
    if value is None or not 0 <= value <= 999 or value != int(value):
        return None
    return int(value)


def generate_trajectories(engine, tokenizer, sampling_factory, problems, *, max_steps=5,
                          total_tokens=4096, context_limit=8192, initial_limit=2048):
    """Match the training continuation protocol without rewards or gold feedback."""
    states = []
    turn_limit = (total_tokens + max_steps - 1) // max_steps
    for problem in problems:
        raw = [{"role": "user", "content": problem +
                "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        ids = tokenizer.apply_chat_template(raw, tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(ids) > initial_limit:
            raise ValueError("AIME initial prompt exceeds the configured limit")
        states.append({"messages": build_math_messages(raw, max_steps), "responses": [],
                       "steps": [], "selected_answer": None, "final_response": None,
                       "response_tokens": 0, "discarded_tokens": 0,
                       "termination_reason": "max_steps"})
    active = list(range(len(states)))
    for turn in range(1, max_steps + 1):
        submission = max_steps > 1 and turn == max_steps
        groups = {}
        for index in active:
            state = states[index]
            options = {"enable_thinking": False} if submission else {}
            prompt_ids = tokenizer.apply_chat_template(state["messages"], tokenize=True,
                                                       add_generation_prompt=True, return_dict=False, **options)
            if submission:
                prompt_ids += tokenizer.encode(FINAL_ANSWER_PREFIX, add_special_tokens=False)
            if len(prompt_ids) > context_limit:
                state["termination_reason"] = "prompt_limit"
                continue
            remaining = total_tokens - state["response_tokens"]
            limit = min(remaining, turn_limit, context_limit - len(prompt_ids))
            if limit <= 0:
                state["termination_reason"] = "token_budget" if remaining <= 0 else "context_limit"
                continue
            groups.setdefault(limit, []).append((index, list(prompt_ids)))
        next_active = []
        for limit, requests in groups.items():
            partial = {index: {"ids": [], "consumed": 0, "requests": 0}
                       for index, _ in requests}
            pending = requests
            while pending:
                cap_groups = {}
                for index, prompt_ids in pending:
                    item = partial[index]
                    cap = min(SUBMISSION_CHUNK_TOKENS if submission else limit,
                              limit - item["consumed"])
                    cap_groups.setdefault(cap, []).append((index, prompt_ids))
                next_pending = []
                for cap, batch in cap_groups.items():
                    params = sampling_factory(temperature=0.0, top_p=1.0, top_k=-1,
                                              max_tokens=cap)
                    inputs = [{"prompt_token_ids": prompt_ids + partial[index]["ids"]}
                              for index, prompt_ids in batch]
                    outputs = engine.generate(inputs, params, use_tqdm=False)
                    if len(outputs) != len(batch):
                        raise RuntimeError("vLLM returned an incomplete evaluation batch")
                    for (index, prompt_ids), output in zip(batch, outputs, strict=True):
                        item = partial[index]
                        chunk = list(output.outputs[0].token_ids[:cap])
                        item["requests"] += 1
                        item["consumed"] += len(chunk)
                        item["ids"].extend(chunk)
                        kept = completed_prefix_tokens(tokenizer, item["ids"]) if submission else None
                        if kept is not None:
                            item["ids"] = item["ids"][:kept]
                        elif submission and len(chunk) == cap and item["consumed"] < limit and chunk[-1] != tokenizer.eos_token_id:
                            next_pending.append((index, prompt_ids))
                pending = next_pending
            for index, prompt_ids in requests:
                state, item = states[index], partial[index]
                ids = item["ids"]
                state["response_tokens"] += item["consumed"]
                state["discarded_tokens"] += item["consumed"] - len(ids)
                if not ids:
                    state["termination_reason"] = "empty_generation"
                    continue
                response = tokenizer.decode(ids, skip_special_tokens=True)
                tail = tokenizer.decode(prompt_ids[-32:], skip_special_tokens=False)
                final = (prefilled_final_record(response) if submission else
                         final_answer_record(response, require_think_close=bool(re.search(r"<think>\s*$", tail))))
                recorded = FINAL_ANSWER_PREFIX + response if submission else response
                state["responses"].append(recorded)
                state["steps"].append({"turn": turn, "mode": "submission" if submission else "reasoning",
                                       "response": recorded, "response_token_ids": ids,
                                       "generated_tokens": item["consumed"],
                                       "generation_requests": item["requests"]})
                if final is not None:
                    state["selected_answer"] = final["answer"]
                    state["final_response"] = recorded
                    state["termination_reason"] = "final_answer"
                else:
                    state["messages"].append({"role": "assistant", "content": response})
                    state["messages"].append(math_continuation_message(
                        final_turn=turn + 1 == max_steps, submission_only=True,
                        max_output_tokens=min(turn_limit, total_tokens - state["response_tokens"]),
                        prepare_submission=turn + 1 == max_steps - 1,
                    ))
                    next_active.append(index)
        active = next_active
        print(f"turn={turn} finished={len(states)-len(active)}/{len(states)} active={len(active)}", flush=True)
        if not active:
            break
    return states


def summarize_accuracy(records, expected_ids):
    ids = [record["problem_idx"] for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("Evaluation results must cover every dataset question exactly once")
    correct = sum(bool(record["is_correct"]) for record in records)
    submitted = sum(record["predicted_integer"] is not None for record in records)
    return {"total": len(expected_ids), "correct": correct, "accuracy": correct / len(expected_ids),
            "accuracy_percent": 100 * correct / len(expected_ids), "valid_integer_submissions": submitted,
            "missing_or_invalid_submissions": len(expected_ids) - submitted,
            "attempts_per_problem": 1, "decoding": "greedy", "metric": "agent_accuracy_pass_at_1"}


def read_dataset(path):
    import pandas as pd
    frame = pd.read_parquet(path)
    rows = [{"problem_idx": int(row.problem_idx), "problem": str(row.problem),
             "answer": str(row.answer)} for row in frame.itertuples(index=False)]
    if len(rows) != 30 or len({row["problem_idx"] for row in rows}) != 30:
        raise ValueError("Expected all 30 distinct AIME questions")
    if any(aime_integer(row["answer"]) is None for row in rows):
        raise ValueError("Dataset contains an invalid AIME reference")
    return rows


def worker(args, rows):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    shard = rows[args.worker_rank::len(args.gpus.split(","))]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16", enforce_eager=True,
                 max_model_len=8192, gpu_memory_utilization=0.3, max_num_seqs=16,
                 max_num_batched_tokens=8192, language_model_only=True, mm_processor_cache_gb=0,
                 generation_config="vllm", seed=42)
    states = generate_trajectories(engine, tokenizer, SamplingParams, [row["problem"] for row in shard])
    with (Path(args.output) / f"results_gpu{args.worker_rank}.jsonl").open("w") as stream:
        for row, state in zip(shard, states, strict=True):
            predicted = selected_aime_integer(state["selected_answer"])
            record = {"problem_idx": row["problem_idx"], "gold_answer": row["answer"],
                      "predicted_answer": state["selected_answer"], "predicted_integer": predicted,
                      "is_correct": predicted is not None and predicted == aime_integer(row["answer"]),
                      "policy_step_count": len(state["steps"]),
                      "generated_response_tokens": state["response_tokens"],
                      "discarded_after_final_answer_tokens": state["discarded_tokens"],
                      "termination_reason": state["termination_reason"], "steps": state["steps"],
                      "final_response": state["final_response"]}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"problem={row['problem_idx']} predicted={predicted} correct={record['is_correct']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "dataset", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--worker-rank", type=int)
    args = parser.parse_args()
    rows = read_dataset(args.dataset)
    if args.worker_rank is not None:
        worker(args, rows)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {**vars(args),
                "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                "checkpoint_global_step": int(match[1]) if (match := re.search(r"global_step_(\d+)$", args.checkpoint.rstrip("/"))) else 0,
                "model_status": "finetuned_checkpoint" if match else "original_pretrained",
                "answer_scoring": "reference-independent normalization of the whole submitted numeric expression to an integer in [0,999]",
                "initial_prompt_limit": 2048, "context_limit": 8192,
                "completion_budget": 4096, "per_generation_cap": 820, "max_interaction_steps": 5,
                "temperature": 0.0, "attempts_per_problem": 1, "seed": 42,
                "gpu_memory_utilization": 0.3,
                "parallelism": f"{len(args.gpus.split(','))} independent single-GPU replicas",
                "verification_rewards_computed": False, "gold_feedback_provided": False,
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("scripts/deepscaler/eval_aime_agent.py", "recipes/deepscaler/prompts.py",
                                             "recipes/deepscaler/submission.py", "agent_r1/evaluation/answers.py",
                                             "recipes/deepscaler/math_expressions.py")}}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    processes = []
    started = time.monotonic()
    try:
        for rank, gpu in enumerate(args.gpus.split(",")):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                   "VLLM_LOGGING_LEVEL": "WARN", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
            command = [sys.executable, str(Path(__file__).resolve()), "--model", args.model,
                       "--checkpoint", args.checkpoint, "--dataset", args.dataset, "--output", args.output,
                       "--dataset-name", args.dataset_name, "--dataset-revision", args.dataset_revision,
                       "--dataset-label", args.dataset_label,
                       "--gpus", args.gpus, "--worker-rank", str(rank)]
            with (output / f"gpu{gpu}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            processes.append(process)
            print(f"Started GPU {gpu}: pid={process.pid}, problems={len(rows[rank::len(args.gpus.split(','))])}", flush=True)
        (output / "worker_pids.json").write_text(json.dumps([p.pid for p in processes]) + "\n")
        while any(process.poll() is None for process in processes):
            failed = [p for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Evaluation worker failed: {[(p.pid, p.returncode) for p in failed]}")
            time.sleep(5)
        if any(process.returncode for process in processes):
            raise RuntimeError("Evaluation workers did not all complete successfully")
        records = []
        for rank in range(len(processes)):
            records.extend(json.loads(line) for line in (output / f"results_gpu{rank}.jsonl").read_text().splitlines())
        records.sort(key=lambda row: row["problem_idx"])
        summary = {**summarize_accuracy(records, [row["problem_idx"] for row in rows]),
                   "checkpoint": args.checkpoint, "dataset": args.dataset_label,
                   "elapsed_seconds": time.monotonic() - started}
        (output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    main()
