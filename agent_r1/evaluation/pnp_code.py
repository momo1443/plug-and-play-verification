"""LiveCodeBench public-test tools and separate final sandbox grading."""
from __future__ import annotations

from dataclasses import dataclass
import json

from recipes.livecodebench_a0.evaluate_a0 import decode_tests, grade_functional, grade_stdio
from recipes.taco_a9.dsl import ExecutionRecord, sha256_text
from recipes.taco_a9.sandbox import BubblewrapPythonExecutor, TestCase, TestSuite


def problem_tests(row):
    def decode(value):
        return value if isinstance(value, list) else decode_tests(value)
    public, private = decode(row["public_test_cases"]), decode(row["private_test_cases"])
    kinds = {case["testtype"] for case in public + private}
    if kinds not in ({"stdin"}, {"functional"}) or not private:
        raise ValueError("LiveCodeBench requires supported test types and held-out tests")
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    function_name = metadata.get("func_name") if kinds == {"functional"} else None
    if kinds == {"functional"} and not function_name:
        raise ValueError("LiveCodeBench functional problem is missing func_name")
    return public, private, function_name


@dataclass(frozen=True)
class LiveCodeSuite(TestSuite):
    function_name: str | None = None

    @property
    def sha256(self):
        return sha256_text(json.dumps({"cases": [vars(case) for case in self.cases],
                                     "function_name": self.function_name},
                                    sort_keys=True, ensure_ascii=False, separators=(",", ":")))


class PublicTaskStore:
    """The PNP flow can access public tests; its private suite is always empty."""
    def __init__(self, task):
        self.task_id = task.question_id
        inputs = task.actor_inputs
        self.public = LiveCodeSuite("developer_v1", tuple(
            TestCase(str(case["input"]), str(case["output"])) for case in inputs["public_tests"]),
            inputs["function_name"])

    def suites(self, task_id):
        if task_id != self.task_id:
            raise ValueError("LiveCodeBench task ID mismatch")
        return self.public, LiveCodeSuite("private_v1", (), self.public.function_name)


class LiveCodeExecutor(BubblewrapPythonExecutor):
    """Keep the training execution-record contract for stdio and call-based tasks."""
    def run_suite(self, artifact, suite, *, run_ordinal):
        outcomes = []
        grader = grade_functional if suite.function_name else grade_stdio
        for index, case in enumerate(suite.cases):
            test = {"input": case.stdin, "output": case.expected_stdout}
            args = (artifact.code, [test], suite.function_name, self.timeout_seconds) if suite.function_name else (
                artifact.code, [test], self.timeout_seconds)
            result = grader(*args)
            outcomes.append({"case_index": index, "passed": bool(result["passed"]),
                             "stdin": case.stdin[:4096], "expected_stdout": case.expected_stdout[:4096],
                             "stdout": str(result.get("stdout", ""))[:4096],
                             "stderr": str(result.get("reason", ""))[:4096],
                             "timed_out": result["reason"] == "timeout"})
        payload = {"suite_id": suite.suite_id, "outcomes": outcomes}
        digest = sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        passed = sum(item["passed"] for item in outcomes)
        return ExecutionRecord(
            f"run:{run_ordinal}:{artifact.sha256[:12]}", artifact.artifact_id, artifact.sha256,
            suite.suite_id, suite.sha256, digest, passed, len(outcomes),
            bool(outcomes) and passed == len(outcomes), any(item["timed_out"] for item in outcomes)), payload


def grade_submission(code, reference, timeout_seconds):
    if not code:
        return {"is_correct": False, "grading_reason": "missing_submission"}
    tests = reference["public_tests"] + reference["private_tests"]
    if reference["function_name"]:
        result = grade_functional(code, tests, reference["function_name"], timeout_seconds)
    else:
        result = grade_stdio(code, tests, timeout_seconds)
    return {"is_correct": bool(result["passed"]), "grading_reason": result["reason"],
            "passed_test_cases": result["passed_cases"], "total_test_cases": result["total_cases"]}
