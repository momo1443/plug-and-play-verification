"""JSON adapter around unmodified, pinned benchmark scripts."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_grader(benchmark, root):
    identity = json.loads((root / "manifest.json").read_text())[benchmark]
    for relative, expected in identity["files_sha256"].items():
        if hashlib.sha256((root / benchmark / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Official grader was modified: {benchmark}/{relative}")
    if benchmark == "hotpotqa":
        return load_module("official_hotpotqa", root / benchmark / "hotpot_evaluate_v1.py")
    if benchmark == "livecodebench":
        return load_module("official_livecodebench", root / benchmark / "lcb_runner/evaluation/testing_util.py")
    # The upstream evaluation module loads its dataset at import time.
    # Supply the same expected file layout, then populate its task map per job.
    temporary = tempfile.TemporaryDirectory(prefix="pnp-mathvision-grader-")
    os.chdir(temporary.name)
    Path("data").mkdir()
    Path("data/test.jsonl").write_text("")
    sys.path.insert(0, str(root / benchmark / "evaluation"))
    module = load_module("official_mathvision", root / benchmark / "evaluation/evaluate.py")
    module._adapter_temporary = temporary
    return module


def grade(benchmark, module, job):
    if job.get("operation") == "probe":
        packages = ("numpy", "sympy", "antlr4-python3-runtime", "latex2sympy2", "ujson", "tqdm", "fire")
        return {"packages": {name: version(name) for name in packages}}
    if benchmark == "hotpotqa":
        if job.get("operation") == "normalize":
            return module.normalize_answer(job["answer"])
        answer, references = job["answer"], job["references"]
        em = max((float(module.exact_match_score(answer, gold)) for gold in references), default=0.0)
        f1 = max((float(module.f1_score(answer, gold)[0]) for gold in references), default=0.0)
        return {"is_correct": bool(em), "answer_em": em, "answer_f1": f1}
    if benchmark == "mathvision":
        if job.get("operation") == "equal":
            return bool(module.is_equal(job["left"], job["right"]))
        module.id_raw = {"pnp": {"id": "pnp", "answer": job["reference"], "options": job["options"]}}
        answer_file = Path("answer.jsonl")
        answer_file.write_text(json.dumps({"id": "pnp", "response": job["answer"]}) + "\n")
        module.evaluate(str(answer_file), regen_answer=True)
        record = json.loads(answer_file.read_text())
        return {"is_correct": bool(record["correct"]), "model_answer": record["model_answer"]}
    tests = job["tests"]
    sample = {"input_output": json.dumps({"inputs": [test["input"] for test in tests],
                                         "outputs": [test["output"] for test in tests],
                                         "fn_name": job.get("function_name")})}
    results, metadata = module.run_test(sample, test=job["code"], timeout=max(1, math.ceil(job["timeout_seconds"])))
    passed_cases = sum(float(value) > 0 for value in results)
    accepted = len(results) == len(tests) and bool(tests) and passed_cases == len(tests)
    reason = "accepted" if accepted else {-2: "wrong_answer", -3: "timeout", -4: "runtime_error"}.get(
        metadata.get("error_code"), "runtime_error")
    return {"passed": accepted, "reason": reason, "passed_cases": passed_cases, "total_cases": len(tests),
            "official_results": [bool(value) if float(value) > 0 else int(value) for value in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("hotpotqa", "mathvision", "livecodebench"))
    parser.add_argument("--vendor-root", type=Path, default=Path(__file__).resolve().parents[2] / "third_party/official_graders")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    module = None
    for line in sys.stdin:
        try:
            # LiveCodeBench's faulthandler requires stderr to have a real FD.
            with open(os.devnull, "w") as quiet_stderr, redirect_stdout(io.StringIO()), redirect_stderr(quiet_stderr):
                if module is None:
                    module = load_grader(args.benchmark, args.vendor_root.resolve())
                result = grade(args.benchmark, module, json.loads(line))
            reply = {"result": result}
        except BaseException as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply), flush=True)
        if args.once:
            break


if __name__ == "__main__":
    main()
