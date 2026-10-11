"""Shared evaluation budgets, denominator, reference isolation and real flows."""
from dataclasses import replace
import copy
import asyncio
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from agent_r1.evaluation.pnp import gpu_groups, grade_record, math_records, preflight, write_tables
from agent_r1.evaluation.pnp_datasets import EvaluationTask, load_tasks
from agent_r1.evaluation.pnp_protocol import (
    PROFILES, EvaluationStop, TrajectoryBudget, mathvision_score, qa_scores, summarize_records,
)
from agent_r1.evaluation.pnp_runtime import BudgetedServer, audit_partial_flow, generate_task, select_submission
from agent_r1.evaluation.pnp_code import PublicTaskStore, LiveCodeExecutor, LiveCodeSuite, grade_submission, problem_tests
from recipes.taco_a9.dsl import CodeArtifact
from recipes.taco_a9.sandbox import TestCase as SandboxCase
from test_aime_agent_eval import CharacterTokenizer, ScriptedEngine
from pnp_test_utils import load_source, vision_contract, vision_prompts, visual_artifacts, tool_format, call
from recipes.vision_r1.verifier import verify_process as verify_vision_process


class PnpEvaluationTest(unittest.TestCase):
    def test_common_decoding_and_task_training_interfaces(self):
        self.assertEqual([p.max_steps for p in PROFILES.values()], [5, 4, 5, 3])
        for profile in PROFILES.values():
            self.assertEqual(profile.sampling()["n"], 1)
            self.assertEqual((profile.temperature, profile.top_p, profile.top_k), (0.0, 1.0, -1))
            with self.assertRaises(ValueError):
                replace(profile, temperature=0.7)
        self.assertEqual(PROFILES["mathvision"].total_output_tokens, 6144)

    def test_budget_is_cumulative_and_keeps_observations_out_of_output_count(self):
        budget = TrajectoryBudget(replace(PROFILES["hotpotqa"], total_output_tokens=5, per_step_tokens=3))
        budget.consume(3, budget.next_cap(100))
        self.assertEqual(budget.next_cap(5000), 2)
        budget.consume(2, 2)
        with self.assertRaisesRegex(EvaluationStop, "token_budget"):
            budget.next_cap(1)
        with self.assertRaisesRegex(EvaluationStop, "prompt_limit"):
            TrajectoryBudget(PROFILES["mathvision"]).next_cap(8193)

    def test_inference_budget_violations_are_errors_not_incorrect_answers(self):
        with self.assertRaises(RuntimeError):
            TrajectoryBudget(PROFILES["hotpotqa"]).consume(4, 3)

    def test_missing_and_budget_stopped_attempts_remain_in_denominator(self):
        rows = [{"question_id": str(i), "terminal_eligible": i == 0, "is_correct": i == 0,
                 "termination_reason": reason, "generated_output_tokens": 3,
                 "verified_process_reward": 1000}
                for i, reason in enumerate(("final_submission", "token_budget", "max_steps", "empty_generation"))]
        result = summarize_records(rows, ["0", "1", "2", "3"], PROFILES["aime2026"])
        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["missing_or_invalid_submissions"], 3)
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                summarize_records(invalid, ["0", "1", "2", "3"], PROFILES["aime2026"])

    def test_qa_em_f1_and_boolean_answers(self):
        self.assertEqual(qa_scores("The Eiffel Tower", ["Eiffel Tower"])["answer_em"], 1)
        self.assertEqual(qa_scores("Eiffel", ["Eiffel Tower"])["answer_f1"], 2 / 3)
        self.assertEqual(qa_scores("yes", ["yes please"])["answer_f1"], 0)
        self.assertFalse(qa_scores(None, ["secret"])["is_correct"])

    def test_mathvision_choice_values_numeric_equivalence_and_no_narrative_search(self):
        for answer in ("B", "(b)", "0.5", r"\frac{1}{2}", r"\boxed{B}"):
            self.assertTrue(mathvision_score(answer, "B", ["1", "0.5"]))
        self.assertTrue(mathvision_score("(1/2, 2)", "(0.5, 2)", []))
        for answer in (None, "", "A or B", "Maybe the hidden answer is B but I choose A"):
            self.assertFalse(mathvision_score(answer, "B", ["1", "0.5"]))

    def test_public_store_never_receives_hidden_tests(self):
        task = EvaluationTask("q1", {"public_tests": [{"input": "1", "output": "2"}],
                                    "function_name": None}, {"private_tests": [{"input": "hidden", "output": "secret"}]})
        public, private = PublicTaskStore(task).suites("q1")
        self.assertEqual(len(public.cases), 1)
        self.assertEqual(private.cases, ())
        self.assertNotIn("secret", repr(public))
        self.assertNotEqual(LiveCodeSuite("dev", (SandboxCase("1", "2"),), "f").sha256,
                            LiveCodeSuite("dev", (SandboxCase("1", "2"),), "g").sha256)

    def test_hidden_tests_are_used_only_in_final_grading(self):
        reference = {"public_tests": [{"input": "public", "output": "1"}],
                     "private_tests": [{"input": "hidden", "output": "2"}], "function_name": None}
        with patch("agent_r1.evaluation.pnp_code.grade_stdio",
                   return_value={"passed": False, "reason": "wrong_answer", "passed_cases": 1, "total_cases": 2}) as grader:
            result = grade_submission("print(1)", reference, 6)
        self.assertFalse(result["is_correct"])
        self.assertEqual(grader.call_args.args[1], reference["public_tests"] + reference["private_tests"])

    def test_functional_and_stdio_problem_adapters_fail_closed(self):
        row = {"public_test_cases": [{"testtype": "functional", "input": "1", "output": "2"}],
               "private_test_cases": [{"testtype": "functional", "input": "3", "output": "4"}],
               "metadata": json.dumps({"func_name": "solve"})}
        self.assertEqual(problem_tests(row)[2], "solve")
        row["metadata"] = "{}"
        with self.assertRaises(ValueError):
            problem_tests(row)

    def test_select_only_submitted_code_artifact_and_require_qa_search(self):
        tokenizer = CharacterTokenizer()
        def step(text, kind):
            return SimpleNamespace(response_ids=tokenizer.encode(text), extra_fields={"step_kind": kind})
        artifact = CodeArtifact.create(1, "print(2)")
        written = step(call("write_code", code=artifact.code), "write_code")
        self.assertIsNone(select_submission("livecodebench", [written], tokenizer))
        submitted = step(call("submit", code_artifact_id=artifact.artifact_id), "submit")
        self.assertEqual(select_submission("livecodebench", [written, submitted], tokenizer), artifact.code)
        finished = step(call("finish", status="answer", answer="secret"), "final")
        self.assertIsNone(select_submission("hotpotqa", [finished], tokenizer))

    def test_math_uses_final_submission_and_keeps_unsubmitted_verified_process(self):
        task = EvaluationTask("q1", {"problem": "Compute 1+1."}, {"answer": "2"})
        engine = ScriptedEngine(["1+1=2."] * 4 + ["2}"])
        record = math_records(engine, CharacterTokenizer(), SimpleNamespace, [task], PROFILES["aime2026"])[0]
        self.assertTrue(record["is_correct"])
        self.assertEqual(record["predicted_integer"], 2)
        record = math_records(ScriptedEngine(["1+1=2."] * 5), CharacterTokenizer(),
                              SimpleNamespace, [task], PROFILES["aime2026"])[0]
        self.assertFalse(record["is_correct"])
        self.assertGreater(record["verification"]["verified_process_reward"], 0)

    def test_math_initial_prompt_overflow_is_an_incorrect_retained_attempt(self):
        task = EvaluationTask("q1", {"problem": "A very long question"}, {"answer": "2"})
        profile = replace(PROFILES["aime2026"], prompt_limit=1)
        row = math_records(ScriptedEngine([]), CharacterTokenizer(), SimpleNamespace, [task], profile)[0]
        self.assertFalse(row["is_correct"])
        self.assertEqual(row["termination_reason"], "prompt_limit")

    def test_math_backend_cap_violation_cannot_be_silently_truncated(self):
        task = EvaluationTask("q1", {"problem": "Compute 1+1."}, {"answer": "2"})
        engine = SimpleNamespace(generate=lambda inputs, params, **kwargs: [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1] * (params.max_tokens + 1))])])
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            math_records(engine, CharacterTokenizer(), SimpleNamespace, [task], PROFILES["aime2026"])

    def test_aime_local_source_preserves_ids_and_isolates_answers(self):
        root = Path(__file__).resolve().parents[1]
        tasks, count = load_tasks("aime2026", root / "data/raw/aime26/aime2026.jsonl",
                                 split="test", version="2026", max_problems=2)
        self.assertEqual((count, len(tasks)), (30, 2))
        self.assertEqual(set(tasks[0].actor_inputs), {"problem"})
        self.assertEqual(tasks[0].question_id, "1")

    def test_mathvision_loader_preserves_options_and_image(self):
        image = Image.new("RGB", (100, 100), "white")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image.save(root / "image.png")
            path = root / "test.jsonl"
            path.write_text(json.dumps({"id": "q1", "question": "<image> Select a value",
                                       "image": "image.png", "answer": "B", "options": ["1", "2"]}) + "\n")
            tasks, _ = load_tasks("mathvision", path, split="test", version="full")
        self.assertEqual(tasks[0].reference["options"], ["1", "2"])
        self.assertNotIn("answer", tasks[0].actor_inputs)
        self.assertEqual(tasks[0].actor_inputs["raw_prompt"][0]["content"][0]["image"].size, (100, 100))

    def test_dataset_duplicates_outside_smoke_subset_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.jsonl"
            path.write_text('\n'.join(json.dumps({"id": qid}) for qid in ("a", "b", "b")))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_tasks("mathvision", path, split="test", version="full", max_problems=1, load_images=False)

    def test_visual_metadata_load_does_not_decode_other_workers_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.jsonl"
            path.write_text(json.dumps({"id": "a", "question": "Q", "answer": "B", "image": "missing.png"}))
            tasks, _ = load_tasks("mathvision", path, split="test", version="full", load_images=False)
            self.assertIsNone(tasks[0].actor_inputs["raw_prompt"][0]["content"][0]["image"])
            tasks, count = load_tasks("mathvision", path, split="test", version="full", task_slice=slice(1, None, 4))
            self.assertEqual((tasks, count), ([], 1))

    def test_unusable_code_sandbox_stops_before_gpu_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").touch()
            (model / "model.safetensors").touch()
            args = SimpleNamespace(benchmark="livecodebench", model=model, timeout_seconds=6)
            with patch("agent_r1.evaluation.pnp.importlib.util.find_spec", return_value=object()), \
                 patch("agent_r1.evaluation.pnp.shutil.which", return_value="/usr/bin/bwrap"), \
                 patch("agent_r1.evaluation.pnp_code.grade_submission", return_value={
                     "is_correct": False, "grading_reason": "runtime_error"}):
                with self.assertRaisesRegex(RuntimeError, "sandbox preflight"):
                    preflight(args)

    def test_legacy_visual_answer_check_observations_do_not_depend_on_reference(self):
        from recipes.mathvision.evaluate_react import evaluate_one
        seen = []
        def completion(session, api_base, model, messages, max_tokens, **kwargs):
            self.assertFalse(kwargs["use_tools"])
            seen.append(copy.deepcopy(messages))
            text = call("check_deepscaler_answer", answer="1") if len(messages) == 2 else r"\boxed{1}"
            return {"message": {"content": text}, "finish_reason": "stop"}
        row = {"index": 0, "prompt": [{"role": "user", "content": "Solve the image"}],
               "images": [{"bytes": b"image"}], "extra_info": {"question_id": "q1"}}
        with patch("recipes.mathvision.evaluate_react.request_completion", side_effect=completion):
            first = evaluate_one({**row, "answer": "1"}, "local", "qwen", 3, 100, "a0-react", 5)
            second = evaluate_one({**row, "answer": "998"}, "local", "qwen", 3, 100, "a0-react", 5)
        self.assertEqual(seen[:2], seen[2:])
        self.assertTrue(first["correct"])
        self.assertFalse(second["correct"])
        self.assertIn("unavailable", seen[1][-1]["content"])

    def test_hotpotqa_compatibility_wrapper_uses_standalone_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            stub, captured = Path(temporary) / "python", Path(temporary) / "args.json"
            stub.write_text("#!" + sys.executable + "\nimport json, os, sys\n"
                            "open(os.environ['PNP_CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n")
            stub.chmod(0o755)
            env = {**os.environ, "PYTHON_BIN": str(stub), "PNP_CAPTURE": str(captured),
                   "HOTPOTQA_MODEL_PATH": "/model", "HOTPOTQA_DATASET_REVISION": "frozen-snapshot",
                   "HOTPOTQA_VAL_TEMPERATURE": "0.0", "HOTPOTQA_VAL_TOP_P": "1.0",
                   "HOTPOTQA_VAL_DO_SAMPLE": "False", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
            result = subprocess.run(["bash", str(root / "scripts/hotpotqa/pnp_hotpotqa_eval.sh")],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(captured.read_text())
            self.assertEqual(args[:3], ["-m", "agent_r1.evaluation.pnp", "hotpotqa"])
            self.assertEqual(args[args.index("--gpus") + 1], "0,1,2,3")
            captured.unlink()
            result = subprocess.run(["bash", str(root / "scripts/hotpotqa/pnp_hotpotqa_eval.sh")],
                                    env={**env, "HOTPOTQA_VAL_N": "4"}, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(captured.exists())

    def test_tables_come_from_the_same_defaults_as_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.md"
            write_tables(path)
            text = path.read_text()
        self.assertEqual(text.count(r"\begin{table}"), 4)
        self.assertIn("Total completion budget & 10240 tokens", text)
        self.assertNotIn("32768", text)
        self.assertNotIn("A0", text)

    def test_replica_groups_and_invalid_gpu_specification(self):
        self.assertEqual(gpu_groups("0,1,2,3", 1), [["0"], ["1"], ["2"], ["3"]])
        self.assertEqual(gpu_groups("0,1,2,3", 2), [["0", "1"], ["2", "3"]])
        for devices, tp in (("0,0", 1), ("0;echo", 1), ("0,1,2", 2), ("0", 0)):
            with self.assertRaises(ValueError):
                gpu_groups(devices, tp)


class PnpEvaluationRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_enforces_eval_sampling_and_total_budget(self):
        profile = replace(PROFILES["hotpotqa"], total_output_tokens=5, per_step_tokens=3)
        engine = ScriptedEngine(["abc", "de"])
        server = BudgetedServer(engine, SimpleNamespace, profile)
        await server.generate(prompt_ids=[1], sampling_params={"temperature": 1, "n": 4})
        await server.generate(prompt_ids=[1] * 100, sampling_params={})
        with self.assertRaisesRegex(EvaluationStop, "token_budget"):
            await server.generate(prompt_ids=[1], sampling_params={})
        self.assertEqual(server.budget.generated_tokens, 5)

    async def test_partial_code_audit_replays_public_evidence_without_submission(self):
        task = EvaluationTask("q1", {"public_tests": [{"input": "1", "output": "2"}],
                                     "function_name": None}, {"private_tests": [{"input": "secret", "output": "gold"}]})
        tokenizer = CharacterTokenizer()
        store = PublicTaskStore(task)
        with patch("recipes.taco_a9.sandbox.shutil.which", return_value="/test/bwrap"):
            executor = LiveCodeExecutor(timeout_seconds=6)
        artifact = CodeArtifact.create(1, "print(2)")
        with patch("agent_r1.evaluation.pnp_code.grade_stdio", return_value={
            "passed": True, "reason": "accepted", "passed_cases": 1, "total_cases": 1}) as grader:
            record, _ = executor.run_suite(artifact, store.public, run_ordinal=1)
            def step(text, records):
                return SimpleNamespace(response_ids=SimpleNamespace(tolist=lambda: tokenizer.encode(text)),
                                       extra_fields={"execution_ledger": records})
            steps = [step(call("write_code", code=artifact.code), []),
                     step(call("run_developer_tests", code_artifact_id=artifact.artifact_id), [record.record()])]
            audit = await audit_partial_flow("livecodebench", task, SimpleNamespace(
                tokenizer=tokenizer, store=store, executor=executor), steps)
        self.assertEqual(audit["verified_process_reward"], 1)
        self.assertEqual(audit["audit"]["process_denominator"], 1)
        self.assertTrue(all("secret" not in repr(request) for request in grader.call_args_list))

    async def test_partial_qa_audit_keeps_transition_step_indices(self):
        from recipes.hotpotqa_a9.verifier import verify_process
        transitions = [{"action_type": "search", "action_value": "June Miller nationality",
                        "certificate_parsed": None, "artifacts_before": {}}]
        steps = [SimpleNamespace(extra_fields={"certificate_transitions": []}),
                 SimpleNamespace(extra_fields={"certificate_transitions": transitions})]
        expected = verify_process(transitions, [1], reward_horizon=3)
        actual = await audit_partial_flow("hotpotqa", None, SimpleNamespace(reward_horizon=3), steps)
        self.assertEqual(actual["audit"], expected.audit)
        self.assertEqual(actual["verified_process_reward"], expected.process_reward)

    def visual_flow(self):
        module = load_source(
            "recipes/vision_r1/agent_flow.py", ToolFormatWrapper=tool_format.ToolFormatWrapper,
            VisualArtifact=visual_artifacts.VisualArtifact, reward_schedule=vision_contract.reward_schedule,
            outcome_reward=vision_contract.outcome_reward, verify_process=verify_vision_process,
            SYSTEM_PROMPT=vision_prompts.SYSTEM_PROMPT, FINAL_TURN_PROMPT=vision_prompts.FINAL_TURN_PROMPT,
            INSPECT_OR_SUBMIT_TOOLS=vision_prompts.INSPECT_OR_SUBMIT_TOOLS,
            SUBMIT_ONLY_TOOLS=vision_prompts.SUBMIT_ONLY_TOOLS)
        flow = object.__new__(module.VisionR1VisualAgentFlow)
        flow.max_steps, flow.prompt_length, flow.response_length = 3, 8192, 2048
        flow.reward_mode, flow.terminal_warmup_steps = "uniform_visual_certificate", 50
        flow.test_image = Image.new("RGB", (100, 100), "white")
        flow.evaluation_steps = []
        flow.tokenizer = CharacterTokenizer()
        async def record(step, **kwargs):
            step.response_ids = SimpleNamespace(tolist=lambda ids=step.response_ids: ids)
            flow.evaluation_steps.append(step)
            return step
        flow._postprocess = record
        return flow

    async def test_real_visual_flow_crop_submit_and_independent_scoring(self):
        flow = self.visual_flow()
        root = visual_artifacts.VisualArtifact.root(flow.test_image)
        responses = [call("inspect_region", source_artifact_id=root.artifact_id,
                          bbox_2d=[10, 10, 70, 70], purpose="Read the diagram"),
                     call("submit_answer", answer="B")]
        server = BudgetedServer(ScriptedEngine(responses), SimpleNamespace, PROFILES["mathvision"])
        flow.server_manager = server
        task = EvaluationTask("q1", {"raw_prompt": [{"role": "user", "content": "Select from the diagram"}],
                                    "extra_info": {"question_id": "q1"}}, {"answer": "B", "options": ["1", "2"]})
        with patch("agent_r1.evaluation.pnp_runtime.create_flow", return_value=flow):
            result = await generate_task("mathvision", task, flow.tokenizer, None, server,
                                         PROFILES["mathvision"], {})
        self.assertTrue(result["terminal_eligible"])
        self.assertEqual(result["policy_step_count"], 2)
        self.assertTrue(grade_record("mathvision", result, task.reference, 6)["is_correct"])
        self.assertEqual(result["verification"]["verified_process_reward"], 0.5)

    async def test_visual_token_budget_stop_keeps_verified_crop_and_zero_outcome(self):
        flow = self.visual_flow()
        root = visual_artifacts.VisualArtifact.root(flow.test_image)
        response = call("inspect_region", source_artifact_id=root.artifact_id,
                        bbox_2d=[10, 10, 70, 70], purpose="Read diagram")
        profile = replace(PROFILES["mathvision"], total_output_tokens=len(response))
        server = BudgetedServer(ScriptedEngine([response]), SimpleNamespace, profile)
        flow.server_manager = server
        task = EvaluationTask("q1", {"raw_prompt": [{"role": "user", "content": "Select from diagram"}],
                                     "extra_info": {"question_id": "q1"}}, {"answer": "B", "options": []})
        with patch("agent_r1.evaluation.pnp_runtime.create_flow", return_value=flow):
            result = await generate_task("mathvision", task, flow.tokenizer, None, server, profile, {})
        self.assertEqual(result["termination_reason"], "token_budget")
        self.assertFalse(result["terminal_eligible"])
        self.assertFalse(grade_record("mathvision", result, task.reference, 6)["is_correct"])
        self.assertEqual(result["verification"]["verified_process_reward"], 1)


if __name__ == "__main__":
    unittest.main()
