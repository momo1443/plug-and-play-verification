"""Raw-prefix selection, sparse tests, and call-based TACO execution."""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from recipes.taco_a9.prepare_data import build_dataset, validate_source_prefix
from recipes.taco_a9.sandbox import BubblewrapPythonExecutor, TestCase, TestSuite, functional_outputs_equal
from recipes.taco_a9.dsl import CodeArtifact


class TacoSourcePrefixTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "raw/ALL").mkdir(parents=True)

    def row(self, index, cases=None):
        return {"question": f"Task {index}", "url": f"task:{index}", "starter_code": "",
                "input_output": json.dumps(cases if cases is not None else {"inputs": ["1\n"], "outputs": ["1\n"]}),
                "solutions": "reference solution must not enter prompts or sidecars"}

    def build(self, rows, samples=3, validation_size=1):
        pq.write_table(pa.Table.from_pylist(rows), self.root / "raw/ALL/train-00000.parquet")
        return build_dataset(argparse.Namespace(taco_dir=str(self.root / "raw"),
            output_dir=str(self.root / "out"), train_samples=samples,
            validation_size=validation_size, developer_fraction=0.4))

    def test_prefix_keeps_duplicate_functional_and_single_test_tasks(self):
        first = self.row(0)
        functional = self.row(2, {"fn_name": "add", "inputs": [[1, 2], [3, 4]], "outputs": [3, 7]})
        manifest = self.build([first, first, functional, self.row(3), self.row(4)])
        rows = pq.read_table(self.root / "out/train.parquet").to_pylist()
        self.assertEqual([row["extra_info"]["index"] for row in rows], [0, 1, 2])
        self.assertEqual(len({row["extra_info"]["question_id"] for row in rows}), 3)
        self.assertEqual(rows[0]["prompt"], rows[1]["prompt"])
        self.assertEqual(manifest["validation_source_indices"], [3])
        self.assertEqual(manifest["rows_filtered"], 0)
        validate_source_prefix(self.root / "out/train.parquet", 3)
        sidecar = [json.loads(line) for line in (self.root / "out/taco_a9_sidecar.jsonl").read_text().splitlines()]
        self.assertEqual(sidecar[0]["developer_cases"], [])
        self.assertEqual(len(sidecar[0]["private_cases"]), 1)
        self.assertEqual(sidecar[0]["private_cases"], sidecar[1]["private_cases"])
        self.assertEqual(sidecar[2]["function_name"], "add")
        self.assertEqual(len(sidecar[2]["developer_cases"]), 1)
        self.assertEqual(len(sidecar[2]["private_cases"]), 1)
        self.assertNotIn("reference solution", json.dumps(rows) + json.dumps(sidecar))
        # Load the real sidecar reader without importing its GPU dependencies.
        flow_path = Path(__file__).resolve().parents[1] / "recipes/taco_a9/agent_flow.py"
        tree = ast.parse(flow_path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TacoTaskStore"]
        namespace = {"Path": Path, "json": json, "Any": Any, "TestSuite": TestSuite, "TestCase": TestCase}
        exec(compile(tree, str(flow_path), "exec"), namespace)
        sidecar_path = self.root / "out/taco_a9_sidecar.jsonl"
        offsets = {}
        with sidecar_path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                offsets[json.loads(line)["task_id"]] = (offset, len(line))
        index_path = self.root / "out/index.json"
        index_path.write_text(json.dumps(offsets))
        store = namespace["TacoTaskStore"](str(sidecar_path), str(index_path))
        developer, private = store.suites(rows[2]["extra_info"]["question_id"])
        self.assertEqual((developer.function_name, private.function_name), ("add", "add"))

    def test_bad_prefix_row_is_not_replaced_by_a_later_row(self):
        invalid = self.row(1)
        invalid["input_output"] = "{}"
        with self.assertRaisesRegex(ValueError, "Source row 1.*cannot be skipped or replaced"):
            self.build([self.row(0), invalid, self.row(2)], samples=2, validation_size=0)
        self.assertFalse((self.root / "out/train.parquet").exists())

    def test_no_test_case_count_filter_or_batch_trimming(self):
        rows = [self.row(0, {"inputs": [], "outputs": []}), self.row(1), self.row(2)]
        self.build(rows, samples=3, validation_size=0)
        self.assertEqual(pq.ParquetFile(self.root / "out/train.parquet").metadata.num_rows, 3)
        self.assertEqual(pq.ParquetFile(self.root / "out/validation.parquet").metadata.num_rows, 0)

    def test_old_prepared_data_must_be_regenerated(self):
        path = self.root / "old.parquet"
        pq.write_table(pa.Table.from_pylist([{"extra_info": {"index": 0}}]), path)
        with self.assertRaisesRegex(ValueError, "Prepare the source prefix"):
            validate_source_prefix(path, 1)

    def test_default_prefix_crosses_shards_without_reading_later_bad_rows(self):
        rows = [self.row(i) for i in range(10000)] + [{**self.row(10000), "input_output": "{}"}]
        pq.write_table(pa.Table.from_pylist(rows[:5000]), self.root / "raw/ALL/train-00000.parquet")
        pq.write_table(pa.Table.from_pylist(rows[5000:]), self.root / "raw/ALL/train-00001.parquet")
        manifest = build_dataset(argparse.Namespace(taco_dir=str(self.root / "raw"),
            output_dir=str(self.root / "out"), train_samples=10000, validation_size=0, developer_fraction=0.4))
        self.assertEqual(manifest["train_source_indices"], list(range(10000)))
        validate_source_prefix(self.root / "out/train.parquet", 10000)


class TacoFunctionalTest(unittest.TestCase):
    def test_function_and_solution_method_runner(self):
        runner = Path(__file__).resolve().parents[1] / "recipes/taco_a9/functional_runner.py"
        for code in ("def add(a,b):\n print('diagnostic')\n return a+b\n",
                     "class Solution:\n def add(self,a,b): return a+b\n"):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "candidate.py"
                path.write_text(code)
                result = subprocess.run([sys.executable, "-I", str(runner), str(path)],
                    input=json.dumps({"function_name": "add", "arguments": [1, 2]}),
                    text=True, capture_output=True, check=True)
                self.assertEqual(json.loads(result.stdout), 3)

    def test_outputs_and_function_identity_are_part_of_the_contract(self):
        self.assertTrue(functional_outputs_equal('[1,2]', '[[1,2]]'))
        self.assertTrue(functional_outputs_equal('3', '[3]'))
        self.assertFalse(functional_outputs_equal('4', '[3]'))
        self.assertFalse(functional_outputs_equal('not JSON', '3'))
        cases = (TestCase('[1,2]', '3'),)
        self.assertNotEqual(TestSuite("dev", cases, "add").sha256, TestSuite("dev", cases, "multiply").sha256)

    def test_functional_executor_passes_only_arguments_to_sandbox(self):
        executor = object.__new__(BubblewrapPythonExecutor)
        executor.timeout_seconds, executor.python_path = 2.0, "/usr/bin/python3"
        def run(command, **kwargs):
            self.assertIn("bwrap", command)
            self.assertIn("/work/functional_runner.py", command)
            self.assertEqual(json.loads(kwargs["input"]), {"function_name": "add", "arguments": [1, 2]})
            self.assertNotIn("expected_stdout", kwargs["input"])
            return subprocess.CompletedProcess(command, 0, stdout='3', stderr='')
        with patch("recipes.taco_a9.sandbox.subprocess.run", side_effect=run):
            record, _ = executor.run_suite(CodeArtifact.create(1, "def add(a,b): return a+b"),
                TestSuite("dev", (TestCase('[1,2]', '3'),), "add"), run_ordinal=1)
        self.assertTrue(record.all_passed)

    def test_wrong_output_runtime_failure_and_timeout_never_pass(self):
        executor = object.__new__(BubblewrapPythonExecutor)
        executor.timeout_seconds, executor.python_path = 2.0, "/usr/bin/python3"
        suite = TestSuite("dev", (TestCase('[1,2]', '3'),), "add")
        for result in (subprocess.CompletedProcess([], 0, stdout='4', stderr=''),
                       subprocess.CompletedProcess([], 1, stdout='3', stderr='error'),
                       subprocess.TimeoutExpired([], 2)):
            with self.subTest(result=result), patch("recipes.taco_a9.sandbox.subprocess.run") as run:
                if isinstance(result, Exception):
                    run.side_effect = result
                else:
                    run.return_value = result
                record, _ = executor.run_suite(CodeArtifact.create(1, "def add(a,b): return 4"),
                    suite, run_ordinal=1)
                self.assertFalse(record.all_passed)
                self.assertEqual(record.passed, 0)
                self.assertEqual(record.timed_out, isinstance(result, subprocess.TimeoutExpired))
        record, _ = executor.run_suite(CodeArtifact.create(1, "pass"), TestSuite("dev", (), "add"), run_ordinal=1)
        self.assertFalse(record.all_passed)


if __name__ == "__main__":
    unittest.main()
