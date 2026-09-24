#!/usr/bin/env python3
"""Bubblewrap-side LiveCodeBench call-based runner for LeetCode problems."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

# These imports match the official LiveCodeBench call-based checker.
BASE_IMPORTS = """
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
from typing import *
sys.setrecursionlimit(50000)
"""


class FunctionalTimeout(Exception):
    pass


def alarm_handler(signum, frame):
    raise FunctionalTimeout("functional test timed out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--fn-name", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tests = json.loads(args.tests.read_text(encoding="utf-8"))
    try:
        namespace: dict[str, object] = {}
        exec(BASE_IMPORTS + "\n" + args.solution.read_text(encoding="utf-8"), namespace)
        solution_type = namespace.get("Solution")
        if solution_type is None:
            raise RuntimeError("Solution class is missing")
        solution = solution_type()  # type: ignore[operator]
        method = getattr(solution, args.fn_name)
    except BaseException as exc:
        print(json.dumps({"reason": "runtime_error", "passed_cases": 0, "error": repr(exc)}))
        return

    signal.signal(signal.SIGALRM, alarm_handler)
    passed_cases = 0
    try:
        for test in tests:
            inputs = [json.loads(line) for line in test["input"].split("\n")]
            expected = json.loads(test["output"])
            signal.alarm(max(1, int(args.timeout_seconds)))
            prediction = method(*inputs)
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            if prediction != expected:
                print(json.dumps({"reason": "wrong_answer", "passed_cases": passed_cases}))
                return
            passed_cases += 1
    except FunctionalTimeout:
        signal.alarm(0)
        print(json.dumps({"reason": "timeout", "passed_cases": passed_cases}))
        return
    except BaseException as exc:
        signal.alarm(0)
        print(json.dumps({"reason": "runtime_error", "passed_cases": passed_cases, "error": repr(exc)}))
        return
    signal.alarm(0)
    print(json.dumps({"reason": "accepted", "passed_cases": passed_cases}))


if __name__ == "__main__":
    main()
