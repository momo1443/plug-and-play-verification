"""Call upstream graders, keeping their scoring rules and versions intact."""
from __future__ import annotations

import atexit
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party/official_graders"
WORKER = Path(__file__).with_name("official_worker.py")
_workers = {}
_worker_lock = threading.Lock()


def grader_python():
    default = ROOT / ".venv-official-graders/bin/python"
    path = Path(os.environ.get("PNP_GRADER_PYTHON", str(default) if default.is_file() else sys.executable)).absolute()
    if not path.is_file():
        raise RuntimeError("Official grader Python is missing; run bash scripts/evaluation/setup_official_graders.sh")
    return str(path)


class GraderWorker:
    def __init__(self, benchmark):
        self.process = subprocess.Popen([grader_python(), "-I", str(WORKER), benchmark],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, bufsize=1)
        self.lock = threading.Lock()

    def request(self, payload):
        with self.lock:
            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("Official grader worker exited") from exc
            if not select.select([self.process.stdout], [], [], 30)[0]:
                self.close()
                raise RuntimeError("Official grader did not respond; evaluation was aborted")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Official grader worker exited without a result")
            reply = json.loads(line)
            if "error" in reply:
                raise RuntimeError("Official grader failed: " + reply["error"] +
                                   "; check scripts/evaluation/setup_official_graders.sh")
            return reply["result"]

    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


def request(benchmark, payload):
    key = (os.getpid(), benchmark)
    with _worker_lock:
        if key not in _workers:
            _workers[key] = GraderWorker(benchmark)
        worker = _workers[key]
    return worker.request(payload)


@atexit.register
def close_workers():
    for (pid, _), worker in _workers.items():
        if pid == os.getpid():
            worker.close()


@lru_cache
def grader_identity(benchmark):
    if benchmark == "aime2026":
        return {"metric": "exact_integer_equality", "reference": "official dataset integer answer"}
    identity = json.loads((VENDOR / "manifest.json").read_text())[benchmark]
    if benchmark == "livecodebench":
        # Probe imports in an isolated CPU worker; model code is only run by grade_code below.
        completed = subprocess.run([grader_python(), "-I", str(WORKER), benchmark, "--once"],
                                   input=json.dumps({"operation": "probe"}) + "\n", text=True,
                                   capture_output=True, timeout=30, check=True)
        reply = json.loads(completed.stdout)
        if "error" in reply:
            raise RuntimeError("Official grader failed: " + reply["error"])
        dependencies = reply["result"]
    else:
        dependencies = request(benchmark, {"operation": "probe"})
    return {**identity, "python": grader_python(), **dependencies}


@lru_cache
def python_roots():
    completed = subprocess.run([grader_python(), "-I", "-c",
                                "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))"],
                               capture_output=True, text=True, check=True, timeout=10)
    return list(dict.fromkeys(json.loads(completed.stdout)))


def sandbox_command():
    if not shutil.which("bwrap"):
        raise RuntimeError("LiveCodeBench requires bubblewrap (bwrap)")
    command = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--clearenv",
               "--setenv", "PATH", "/usr/bin:/bin", "--proc", "/proc", "--dev", "/dev",
               "--tmpfs", "/tmp", "--dir", "/work"]
    for root in dict.fromkeys(["/usr", "/bin", "/lib", "/lib64", *python_roots()]):
        if Path(root).exists():
            command.extend(["--ro-bind", root, root])
    command.extend(["--ro-bind", str(VENDOR), "/work/official_graders", "--ro-bind", str(WORKER),
                    "/work/official_worker.py", "--chdir", "/work", grader_python(), "-I",
                    "/work/official_worker.py", "livecodebench", "--vendor-root", "/work/official_graders", "--once"])
    return command


def grade_code(code, tests, function_name, timeout_seconds):
    if not code:
        return {"passed": False, "reason": "no_code_block", "passed_cases": 0, "total_cases": len(tests)}
    payload = {"code": code, "tests": tests, "function_name": function_name, "timeout_seconds": timeout_seconds}
    try:
        completed = subprocess.run(sandbox_command(), input=json.dumps(payload) + "\n", capture_output=True,
                                   text=True, env={}, timeout=(math.ceil(timeout_seconds) + 1) * len(tests) + 30,
                                   check=False)
    except subprocess.TimeoutExpired:
        return {"passed": False, "reason": "timeout", "passed_cases": 0, "total_cases": len(tests)}
    if "bwrap:" in completed.stderr:
        raise RuntimeError("LiveCodeBench sandbox failed: " + completed.stderr.strip())
    if completed.returncode != 0:
        return {"passed": False, "reason": "runtime_error", "passed_cases": 0, "total_cases": len(tests)}
    try:
        reply = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"passed": False, "reason": "runtime_error", "passed_cases": 0, "total_cases": len(tests)}
    if "error" in reply:
        raise RuntimeError("Official LiveCodeBench grader failed: " + reply["error"])
    return reply["result"]
