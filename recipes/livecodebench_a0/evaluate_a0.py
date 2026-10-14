#!/usr/bin/env python3
"""Frozen LiveCodeBench release_v6 A0 evaluation with bubblewrap execution."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import pickle
import shutil
import time
import zlib
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a problem "
    "specification and must return a correct Python program that passes all tests."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=float, default=6.0)
    parser.add_argument("--evaluation-workers", type=int, default=8)
    parser.add_argument("--max-problems", type=int, default=None)
    return parser.parse_args()


def load_problems(dataset_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(dataset_dir.glob("test*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    if len(rows) != 1055 or len({row["question_id"] for row in rows}) != 1055:
        raise RuntimeError(f"Expected 1,055 unique release_v6 problems, found {len(rows)}")
    return sorted(rows, key=lambda row: str(row["question_id"]))


def decode_tests(value: str) -> list[dict[str, Any]]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(value))))


def make_prompt(row: dict[str, Any]) -> str:
    question = row["question_content"]
    starter_code = row["starter_code"]
    if starter_code:
        format_instruction = (
            "Use the following starter code to write the solution. "
            "Enclose the completed code within a Python code block.\n\n"
            f"```python\n{starter_code}\n```"
        )
    else:
        format_instruction = (
            "Read inputs from stdin, write outputs to stdout, and do not directly test "
            "on the sample inputs. Enclose the program within a Python code block."
        )
    return f"### Question:\n{question}\n\n### Format:\n{format_instruction}\n\n### Answer:\n"


def extract_code(text: str) -> str:
    lines = text.splitlines()
    fences = [index for index, line in enumerate(lines) if line.strip().startswith("```")]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def grade_stdio(code: str, tests: list[dict[str, Any]], timeout_seconds: float) -> dict[str, Any]:
    """Delegate to the original official checker inside bubblewrap."""
    from agent_r1.evaluation.official import grade_code

    return grade_code(code, tests, None, timeout_seconds)


def grade_functional(
    code: str, tests: list[dict[str, Any]], fn_name: str | None, timeout_seconds: float
) -> dict[str, Any]:
    """Delegate call-based grading to the same unmodified official checker."""
    if not fn_name:
        return {"passed": False, "reason": "missing_function_name", "passed_cases": 0, "total_cases": len(tests)}
    from agent_r1.evaluation.official import grade_code

    return grade_code(code, tests, fn_name, timeout_seconds)


def grade_problem(record: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    row = record["problem"]
    tests = decode_tests(row["public_test_cases"]) + decode_tests(row["private_test_cases"])
    test_types = {test["testtype"] for test in tests}
    if test_types == {"stdin"}:
        result = grade_stdio(record["code"], tests, timeout_seconds)
        return {"question_id": row["question_id"], **result}
    if test_types == {"functional"}:
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        fn_name = metadata.get("func_name") if isinstance(metadata, dict) else None
        result = grade_functional(record["code"], tests, fn_name, timeout_seconds)
        return {"question_id": row["question_id"], **result}
    if test_types:
        return {
            "question_id": row["question_id"],
            "passed": False,
            "reason": f"unsupported_test_types:{sorted(test_types)}",
            "passed_cases": 0, "total_cases": len(tests),
        }
    return {
        "question_id": row["question_id"],
        "passed": False,
        "reason": "empty_tests",
        "passed_cases": 0,
        "total_cases": 0,
    }


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args = parse_args()
    if not shutil.which("bwrap"):
        raise RuntimeError("bubblewrap is required; refusing to execute generated code unsandboxed")
    if not args.model_path.is_dir() or not args.dataset_dir.is_dir():
        raise RuntimeError("The model path and frozen dataset directory must exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    problems = load_problems(args.dataset_dir)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]
    manifest = {
        "benchmark": "livecodebench/code_generation_lite",
        "release_version": "release_v6",
        "dataset_files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.dataset_dir.glob("test*.jsonl"))
        },
        "model_path": str(args.model_path),
        "generation": {"n": 1, "temperature": 0, "top_p": 1.0, "max_tokens": args.max_tokens, "enable_thinking": False},
        "tensor_parallel_size": args.tensor_parallel_size,
        "executor": {"name": "bubblewrap", "timeout_seconds": args.timeout_seconds},
        "problem_count": len(problems),
    }
    write_json(args.output_dir / "manifest.json", manifest)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": make_prompt(row)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for row in problems
    ]
    started = time.monotonic()
    llm = LLM(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(n=1, temperature=0, top_p=1.0, max_tokens=args.max_tokens),
    )
    generation_seconds = time.monotonic() - started
    records = [
        {"problem": row, "output": output.outputs[0].text, "code": extract_code(output.outputs[0].text)}
        for row, output in zip(problems, outputs)
    ]
    del outputs
    del llm
    write_json(
        args.output_dir / "predictions.json",
        [
            {
                "question_id": item["problem"]["question_id"],
                "output": item["output"],
                "code": item["code"],
            }
            for item in records
        ],
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.evaluation_workers) as executor:
        results = list(executor.map(lambda item: grade_problem(item, args.timeout_seconds), records))
    passed = sum(result["passed"] for result in results)
    write_json(args.output_dir / "per_problem.json", results)
    write_json(args.output_dir / "summary.json", {
        "problems": len(results), "passed": passed, "pass_at_1": passed / len(results) if results else 0.0,
        "generation_seconds": generation_seconds,
        "reason_counts": {
            reason: sum(result["reason"] == reason for result in results)
            for reason in sorted({result["reason"] for result in results})
        },
        "grading_protocol": "bubblewrap-isolated stdio/call-based execution; release_v6 code_generation_lite",
    })


if __name__ == "__main__":
    main()
