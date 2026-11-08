"""Deterministic, bubblewrap-backed execution for TACO A9 code artifacts.

The executor deliberately fails closed when bubblewrap is unavailable. Model
code must never execute in the Ray worker's host Python process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from recipes.taco_a9.dsl import CodeArtifact, ExecutionRecord, sha256_text


@dataclass(frozen=True)
class TestCase:
    stdin: str
    expected_stdout: str


@dataclass(frozen=True)
class TestSuite:
    suite_id: str
    cases: tuple[TestCase, ...]
    function_name: str | None = None

    @property
    def sha256(self) -> str:
        payload = [{"stdin": case.stdin, "expected_stdout": case.expected_stdout} for case in self.cases]
        if self.function_name is not None:
            payload = {"function_name": self.function_name, "cases": payload}
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def normalize_stdout(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines()).strip()


def functional_outputs_equal(actual: str, expected: str) -> bool:
    """Compare TACO call-based results, including its wrapped-output convention."""
    try:
        result, reference = json.loads(actual), json.loads(expected)
    except (TypeError, ValueError):
        return False
    return result == reference or (isinstance(reference, list) and bool(reference) and result == reference[0])


class BubblewrapPythonExecutor:
    """Run standard-library Python programs with no network or host worktree mount."""

    def __init__(self, *, timeout_seconds: float = 2.0, python_path: str = "/usr/bin/python3") -> None:
        if not shutil.which("bwrap"):
            raise RuntimeError("TACO A9 requires bubblewrap; refusing to execute model code unsandboxed")
        if not Path(python_path).is_file():
            raise RuntimeError(f"Sandbox Python executable is unavailable: {python_path}")
        self.timeout_seconds = float(timeout_seconds)
        self.python_path = python_path

    def _command(self, code_path: Path, input_path: Path, *, functional: bool = False) -> list[str]:
        required_roots = ["/usr", "/bin", "/lib", "/lib64"]
        command = [
            "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/work",
            "--ro-bind",
            str(code_path),
            "/work/main.py",
            "--ro-bind",
            str(input_path),
            "/work/input.txt",
            "--chdir",
            "/work",
        ]
        for root in required_roots:
            if Path(root).exists():
                command.extend(["--ro-bind", root, root])
        if functional:
            command.extend(["--ro-bind", str(Path(__file__).with_name("functional_runner.py")), "/work/functional_runner.py"])
            command.extend([self.python_path, "-I", "/work/functional_runner.py", "/work/main.py"])
        else:
            command.extend([self.python_path, "-I", "/work/main.py"])
        return command

    def run_suite(self, artifact: CodeArtifact, suite: TestSuite, *, run_ordinal: int) -> tuple[ExecutionRecord, dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="taco-a9-") as temporary:
            root = Path(temporary)
            code_path = root / "main.py"
            code_path.write_text(artifact.code, encoding="utf-8")
            for case_index, case in enumerate(suite.cases):
                input_path = root / f"input-{case_index}.txt"
                stdin = (json.dumps({"function_name": suite.function_name, "arguments": json.loads(case.stdin)},
                                    ensure_ascii=False) if suite.function_name else case.stdin)
                input_path.write_text(stdin, encoding="utf-8")
                try:
                    completed = subprocess.run(
                        self._command(code_path, input_path, functional=bool(suite.function_name)),
                        input=stdin,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout_seconds,
                        check=False,
                        env={},
                    )
                    actual = normalize_stdout(completed.stdout)
                    expected = normalize_stdout(case.expected_stdout)
                    matched = (functional_outputs_equal(completed.stdout, case.expected_stdout)
                               if suite.function_name else actual == expected)
                    outcomes.append(
                        {
                            "case_index": case_index,
                            "passed": completed.returncode == 0 and matched,
                            "returncode": completed.returncode,
                            "stdin": case.stdin[:4096],
                            "expected_stdout": expected[:4096],
                            "stdout": actual[:4096],
                            "stderr": normalize_stdout(completed.stderr)[:4096],
                            "timed_out": False,
                        }
                    )
                except subprocess.TimeoutExpired:
                    outcomes.append(
                        {
                            "case_index": case_index,
                            "passed": False,
                            "returncode": None,
                            "stdin": case.stdin[:4096],
                            "expected_stdout": normalize_stdout(case.expected_stdout)[:4096],
                            "stdout": "",
                            "stderr": "timeout",
                            "timed_out": True,
                        }
                    )
        passed = sum(bool(outcome["passed"]) for outcome in outcomes)
        timed_out = any(bool(outcome["timed_out"]) for outcome in outcomes)
        result_payload = {"suite_id": suite.suite_id, "outcomes": outcomes}
        result_digest = sha256_text(json.dumps(result_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        run_id = f"run:{run_ordinal}:{artifact.sha256[:12]}"
        record = ExecutionRecord(
            run_id=run_id,
            code_artifact_id=artifact.artifact_id,
            code_sha256=artifact.sha256,
            suite_id=suite.suite_id,
            suite_sha256=suite.sha256,
            result_sha256=result_digest,
            passed=passed,
            total=len(suite.cases),
            all_passed=passed == len(suite.cases) and bool(suite.cases),
            timed_out=timed_out,
        )
        return record, result_payload
