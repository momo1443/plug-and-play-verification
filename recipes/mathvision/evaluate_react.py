"""Evaluate MATH-Vision with the Agent-R1 A0 ReAct protocol over an OpenAI API."""

from __future__ import annotations

import argparse
import base64
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from recipes.mathvision.reward_fn import _normal, _prediction


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_deepscaler_answer",
        "description": "Checks whether a proposed final mathematical answer is correct.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string", "description": "Your proposed final answer."}},
            "required": ["answer"],
        },
    },
}

QWEN_MATHVISION_SYSTEM_PROMPT = """Solve the following mathematics problem, analyzing the accompanying image.
If answer options are provided, choose the single best one. Please reason step by step,
and put your final answer within \\boxed{}."""

A0_REACT_SYSTEM_PROMPT = """You are a mathematical problem solver. Analyze the accompanying image with an explicit
observation-reasoning-action loop: identify the relevant visual facts, derive the result from them,
and check that the final answer follows from the derivation. Do not use answer-checking tools or
assume access to a reference answer. Then answer the original problem concisely and put the final
answer in \\boxed{...}."""


def extract_text_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for payload in TOOL_CALL_RE.findall(text):
        try:
            call = json.loads(payload)
            if call.get("name") == "check_deepscaler_answer" and isinstance(call.get("arguments"), dict):
                calls.append(call)
        except json.JSONDecodeError:
            continue
    return calls


def image_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Accept either Agent-R1-prepared rows or the original MATH-Vision parquet."""
    if "prompt" in row:
        return row

    question = re.sub(r"<image\d*>", "<image>", str(row["question"]), flags=re.IGNORECASE)
    raw_options = row.get("options")
    options = [str(option) for option in raw_options] if raw_options is not None else []
    if options:
        question += "\nOptions: " + " ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
    decoded_image = row["decoded_image"]
    return {
        "index": index,
        "prompt": [{"role": "user", "content": question}],
        "images": [{"bytes": decoded_image["bytes"]}],
        "answer": str(row["answer"]),
        "extra_info": {"question_id": str(row["id"])},
    }


def make_messages(row: dict[str, Any], protocol: str) -> list[dict[str, Any]]:
    prompt = row["prompt"]
    question = next(message["content"] for message in prompt if message["role"] == "user")
    system_prompt = QWEN_MATHVISION_SYSTEM_PROMPT if protocol == "qwen-baseline" else A0_REACT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url(row["images"][0]["bytes"])}},
            ],
        },
    ]


def request_completion(
    session: requests.Session,
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    use_tools: bool,
    request_timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    if use_tools:
        payload.update({"tools": [TOOL_SCHEMA], "tool_choice": "auto"})
    response = session.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json=payload,
        timeout=request_timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]


def evaluate_one(
    row: dict[str, Any],
    api_base: str,
    model: str,
    max_steps: int,
    max_tokens: int,
    protocol: str,
    request_timeout: int,
) -> dict[str, Any]:
    session = requests.Session()
    messages = make_messages(row, protocol)
    trajectory: list[dict[str, Any]] = []
    final_text = ""
    ground_truth = str(row["answer"])

    try:
        for turn in range(1, max_steps + 1):
            choice = request_completion(
                session, api_base, model, messages, max_tokens, use_tools=False, request_timeout=request_timeout
            )
            completion = choice["message"]
            text = str(completion.get("content") or "")
            native_calls = completion.get("tool_calls") or []
            calls = []
            for call in native_calls:
                try:
                    calls.append(
                        {
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "arguments": json.loads(call["function"]["arguments"]),
                        }
                    )
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
            if not calls:
                calls = extract_text_tool_calls(text)
            trajectory.append(
                {
                    "turn": turn,
                    "response": text,
                    "tool_calls": calls,
                    "finish_reason": choice.get("finish_reason"),
                }
            )
            if native_calls:
                messages.append({"role": "assistant", "content": text, "tool_calls": native_calls})
            else:
                messages.append({"role": "assistant", "content": text})
            if protocol == "qwen-baseline" or not calls:
                final_text = text
                break

            observations = []
            for call in calls:
                candidate = str(call["arguments"].get("answer", "")).strip()
                correct = _normal(candidate) == _normal(ground_truth)
                observation = "Candidate answer is correct." if correct else "Candidate answer is not correct. Recheck the derivation."
                if call.get("id"):
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
                else:
                    observations.append(f"<tool_response>\n{observation}\n</tool_response>")
            if observations:
                messages.append({"role": "user", "content": "\n".join(observations)})
        else:
            final_text = trajectory[-1]["response"] if trajectory else ""

        prediction = _prediction(final_text)
        return {
            "index": int(row["index"]),
            "question_id": row["extra_info"]["question_id"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": prediction == _normal(ground_truth),
            "num_turns": len(trajectory),
            "trajectory": trajectory,
        }
    except Exception as error:
        return {
            "index": int(row["index"]),
            "question_id": row["extra_info"]["question_id"],
            "ground_truth": ground_truth,
            "prediction": None,
            "correct": False,
            "num_turns": len(trajectory),
            "trajectory": trajectory,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18004/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", choices=("qwen-baseline", "a0-react"), default="a0-react")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    args = parser.parse_args()

    rows = [normalize_row(row, index) for index, row in enumerate(pd.read_parquet(args.data).to_dict("records"))]
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.end_index >= 0 and args.end_index < args.start_index:
        parser.error("--end-index must be greater than or equal to --start-index")
    rows = [row for row in rows if int(row["index"]) >= args.start_index]
    if args.end_index >= 0:
        rows = [row for row in rows if int(row["index"]) < args.end_index]
    if args.max_samples >= 0:
        rows = rows[: args.max_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed: set[int] = set()
    if args.output.exists():
        with args.output.open() as existing:
            for line in existing:
                completed.add(int(json.loads(line)["index"]))
    rows = [row for row in rows if int(row["index"]) not in completed]

    lock = threading.Lock()
    total = len(rows) + len(completed)
    scored = errors = 0
    with args.output.open("a") as output, ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                evaluate_one,
                row,
                args.api_base,
                args.model,
                args.max_steps,
                args.max_tokens,
                args.protocol,
                args.request_timeout,
            )
            for row in rows
        ]
        for finished, future in enumerate(as_completed(futures), start=len(completed) + 1):
            result = future.result()
            with lock:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
            scored += int(result["correct"])
            errors += int("error" in result)
            print(f"completed={finished}/{total} correct_new={scored} errors_new={errors}", flush=True)


if __name__ == "__main__":
    main()
