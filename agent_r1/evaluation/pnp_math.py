"""Training-compatible bounded math generation shared by PNP evaluators."""
from __future__ import annotations
import re
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
                          total_tokens=4096, context_limit=8192, initial_limit=2048,
                          per_step_tokens=None, temperature=0.0):
    """Match the training continuation protocol without rewards or gold feedback."""
    states = []
    if min(max_steps, total_tokens, context_limit, initial_limit) < 1:
        raise ValueError("Math evaluation budgets must be positive")
    turn_limit = per_step_tokens if per_step_tokens is not None else (total_tokens + max_steps - 1) // max_steps
    if turn_limit < 1:
        raise ValueError("Per-step generation budget must be positive")
    for problem in problems:
        raw = [{"role": "user", "content": problem +
                "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        ids = tokenizer.apply_chat_template(raw, tokenize=True, add_generation_prompt=True, return_dict=False)
        states.append({"messages": build_math_messages(raw, max_steps), "responses": [],
                       "steps": [], "selected_answer": None, "final_response": None,
                       "response_tokens": 0, "discarded_tokens": 0,
                       "termination_reason": "prompt_limit" if len(ids) > initial_limit else "max_steps"})
    active = [index for index, state in enumerate(states) if state["termination_reason"] != "prompt_limit"]
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
                    params = sampling_factory(temperature=temperature, top_p=1.0, top_k=-1,
                                              n=1, max_tokens=cap)
                    inputs = [{"prompt_token_ids": prompt_ids + partial[index]["ids"]}
                              for index, prompt_ids in batch]
                    outputs = engine.generate(inputs, params, use_tqdm=False)
                    if len(outputs) != len(batch):
                        raise RuntimeError("vLLM returned an incomplete evaluation batch")
                    for (index, prompt_ids), output in zip(batch, outputs, strict=True):
                        item = partial[index]
                        if len(output.outputs) != 1:
                            raise RuntimeError("Math evaluation requires exactly one completion")
                        chunk = list(output.outputs[0].token_ids)
                        if len(chunk) > cap:
                            raise RuntimeError("Inference backend exceeded the requested token budget")
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
    for state in states:
        if state["selected_answer"] is None and state["response_tokens"] >= total_tokens:
            state["termination_reason"] = "token_budget"
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
