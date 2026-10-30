"""Run the current QA/code/vision PNP flows with CPU runtime boundaries."""
import asyncio
import unittest
from types import SimpleNamespace

from PIL import Image

from recipes.hotpotqa_a9.protocol import parse_finish_call, parse_search_call
from recipes.hotpotqa_a9.reward_sources import arm_schedule
from recipes.hotpotqa_a9.verifier import verify_process as verify_qa_process
from recipes.hotpotqa_lr.protocol import extract_tool_calls
from recipes.hotpotqa_lr.verifier import artifact_content_sha256
from recipes.taco_a9.dsl import CodeArtifact, ExecutionRecord
from recipes.taco_a9.reward_contract import reward_schedule as taco_schedule
from recipes.taco_a9.sandbox import TestSuite as SandboxTestSuite
from recipes.taco_a9.verifier import verify_process as verify_taco_process
from pnp_test_utils import (
    FakeGeneration,
    call,
    load_source,
    taco_prompts,
    tool_format,
    vision_contract,
    vision_prompts,
    visual_artifacts,
)


class PnpFlowTest(unittest.IsolatedAsyncioTestCase):
    async def run_domain(
        self,
        domain,
        *,
        correct=True,
        validation=False,
        update=51,
        incomplete=False,
        developer_test=False,
        reward_mode=None,
        malformed=False,
    ):
        if domain == "hotpotqa":
            module = load_source(
                "recipes/hotpotqa_a9/agent_flow.py",
                arm_schedule=arm_schedule, verify_process=verify_qa_process,
                extract_tool_calls=extract_tool_calls,
                parse_finish_call=parse_finish_call, parse_search_call=parse_search_call,
            )
            flow = object.__new__(module.HotpotQACertificateAgentFlow)
            flow.thinking_mode = "disabled"
            flow.reward_arm = "A9"
            flow.process_reward_enabled = True
            flow.em_warmup_steps = 50
            flow.reward_horizon = 3
            flow.enable_tool_parse_feedback = True
            flow.search_tool = SimpleNamespace(
                require_sentence_evidence=True,
                evidence_store=SimpleNamespace(sample=lambda split, index: SimpleNamespace(
                    question="Task without hidden answer", dataset_question_id="q1", official_qid="q1",
                    sample_key="train:1", gold_evidence_ids=("fact:1",), unresolved_gold_facts=(),
                )),
            )
            evidence = {"artifact_id": "passage:1", "text": "Alpha is relevant evidence.",
                        "content_sha256": artifact_content_sha256("Alpha is relevant evidence.")}
            certificate = {"source_id": "passage:1", "support_span": "Alpha", "target": "Alpha"}
            texts = [call("search", query="Alpha", certificate=certificate)] * 3
            texts += ["invalid action" if incomplete else
                      call("finish", status="answer", answer="correct" if correct else "wrong")]
            if malformed:
                texts[-1] = call("finish", status="answer", answer="")
            flow._prompt_ids_within_budget = lambda **kwargs: ([1], "question", {"passage:1": evidence})
            flow._do_search = lambda query, assistant_turn: {
                "success": True, "assistant_turn": assistant_turn, "returned_evidence": [evidence]}
            def attach_gold(step, gold, covered, **kwargs):
                step["new_gold_evidence_ids"] = list(gold - covered)
                covered.update(gold)
            flow._attach_gold_audit = attach_gold
            flow._ingest_search_step = lambda *args: None
            flow._extra_fields = lambda **kwargs: {"reward_extra_info": {}}

            async def qa_postprocess(step, **kwargs):
                if step.reward_score is None:
                    finish = parse_finish_call(flow.tokenizer.decode(step.response_ids))
                    step.reward_score = float(finish.answer == "correct")
                return step
            flow._postprocess = qa_postprocess
        elif domain == "taco":
            module = load_source(
                "recipes/taco_a9/agent_flow.py",
                reward_schedule=taco_schedule,
                verify_process=verify_taco_process,
                SYSTEM_PROMPT=taco_prompts.SYSTEM_PROMPT,
                USER_PROMPT=taco_prompts.USER_PROMPT,
                FINAL_TURN_PROMPT=taco_prompts.FINAL_TURN_PROMPT,
                SUBMIT_ONLY_TOOLS=taco_prompts.SUBMIT_ONLY_TOOLS,
                WRITE_ONLY_TOOLS=taco_prompts.WRITE_ONLY_TOOLS,
                WRITE_OR_RUN_OR_SUBMIT_TOOLS=taco_prompts.WRITE_OR_RUN_OR_SUBMIT_TOOLS,
            )
            flow = object.__new__(module.TacoA9CodeAgentFlow)
            artifact = CodeArtifact.create(1, "print(1)")
            texts = [call("write_code", code="print(1)"), call("submit", code_artifact_id=artifact.artifact_id)]
            if incomplete:
                texts[1] = call("write_code", code="print(2)")
            if malformed:
                texts[1] = call("submit", code_artifact_id="missing")
            flow.store = SimpleNamespace(suites=lambda task: (
                SandboxTestSuite("dev", ()), SandboxTestSuite("private", ())))
            if developer_test:
                texts.insert(1, call("run_developer_tests", code_artifact_id=artifact.artifact_id))

            def make_record(code, suite, ordinal):
                passed = correct if suite.suite_id == "private" else True
                record = ExecutionRecord(
                    run_id=f"run:{ordinal}",
                    code_artifact_id=code.artifact_id,
                    code_sha256=code.sha256,
                    suite_id=suite.suite_id,
                    suite_sha256=suite.sha256,
                    result_sha256="result",
                    passed=int(passed),
                    total=1,
                    all_passed=passed,
                    timed_out=False,
                )
                return record

            async def run_suite(code, suite, ordinal, metrics):
                record = make_record(code, suite, ordinal)
                hidden = "PRIVATE-SECRET" if suite.suite_id == "private" else "developer-example"
                return record, {"outcomes": [{"passed": correct, "case_index": 0, "stdin": hidden}]}

            flow._run_suite = run_suite
            flow.executor = SimpleNamespace(run_suite=lambda code, suite, run_ordinal:
                                            (make_record(code, suite, run_ordinal), {}))
            flow.max_code_prompt_chars = 12000
            flow.terminal_warmup_steps = 50
        else:
            module = load_source(
                "recipes/vision_r1/agent_flow.py",
                ToolFormatWrapper=tool_format.ToolFormatWrapper,
                VisualArtifact=visual_artifacts.VisualArtifact,
                verify_process=load_source("recipes/vision_r1/verifier.py",
                    parse_visual_certificate=visual_artifacts.parse_visual_certificate,
                    image_sha256=visual_artifacts.image_sha256).verify_process,
                outcome_reward=vision_contract.outcome_reward,
                reward_schedule=vision_contract.reward_schedule,
                system_prompt_template=vision_prompts.system_prompt_template,
                FINAL_TURN_PROMPT=vision_prompts.FINAL_TURN_PROMPT,
                INSPECT_OR_SUBMIT_TOOLS=vision_prompts.INSPECT_OR_SUBMIT_TOOLS,
                SUBMIT_ONLY_TOOLS=vision_prompts.SUBMIT_ONLY_TOOLS,
            )
            flow = object.__new__(module.VisionR1VisualAgentFlow)
            flow.test_image = Image.new("RGB", (100, 100), "blue")
            root = visual_artifacts.VisualArtifact.root(flow.test_image)
            texts = [
                call(
                    "inspect_region",
                    source_artifact_id=root.artifact_id,
                    purpose="Read relevant labels",
                    bbox_2d=[0, 0, 60, 60],
                ),
                call("submit_answer", answer="correct" if correct else "wrong"),
            ]
            if incomplete:
                texts[1] = "invalid action"
            if malformed:
                texts[1] = call("submit_answer", answer="")
            flow.terminal_warmup_steps = 50
        flow.max_steps = max(2, len(texts))
        flow.prompt_length = 10000
        flow.response_length = 1000
        flow.reward_mode = reward_mode or {
            "hotpotqa": "uniform_certificate", "taco": "uniform_certificate",
            "vision_r1": "uniform_visual_certificate",
        }[domain]
        flow.judge_server = None
        flow.loop = asyncio.get_running_loop()
        generation = FakeGeneration(texts)
        flow.server_manager = generation
        flow.tokenizer = generation
        output = await flow.run(
            {},
            raw_prompt=[{"role": "user", "content": "Task without hidden answer"}],
            reward_model={"ground_truth": "correct"},
            extra_info={"question_id": "q1", "index": 1, "split": "train"},
            data_source=domain,
            _agent_r1_global_step=update,
            _agent_r1_is_validation=validation,
        )
        return output

    async def test_qa_preserves_search_credit_for_wrong_and_missing_answers(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                output = await self.run_domain("hotpotqa", correct=False, incomplete=incomplete)
                info = output.steps[-1].extra_fields["reward_extra_info"]
                self.assertEqual(info["optimizer_terminal_component"], 0)
                self.assertEqual(info["acc"], 0)
                self.assertGreater(sum(step.reward_score for step in output.steps), 0)
                self.assertFalse(info["process_requires_terminal_submission"])
                self.assertEqual(info["terminal_gate_passed"], not incomplete)
                self.assertAlmostEqual(sum(step.reward_score for step in output.steps), info["optimizer_process_total"])

    async def test_deterministic_code_and_visual_evidence_survives_missing_submission(self):
        for domain, mode in (("taco", "uniform_certificate"), ("vision_r1", "uniform_visual_certificate")):
            output = await self.run_domain(domain, incomplete=True, reward_mode=mode,
                                             developer_test=domain == "taco")
            info = output.steps[-1].extra_fields["reward_extra_info"]
            self.assertEqual(info["terminal_reward"], 0)
            self.assertEqual(info["optimizer_terminal_component"], 0)
            self.assertFalse(info["terminal_gate_passed"])
            self.assertGreater(info["optimizer_process_total"], 0)
            self.assertAlmostEqual(info["process_reward"], 1)
            self.assertAlmostEqual(sum(step.reward_score for step in output.steps), info["optimizer_process_total"])

    async def test_deterministic_rewards_keep_valid_process_when_outcome_wrong(self):
        for domain, mode in (("taco", "uniform_certificate"), ("vision_r1", "uniform_visual_certificate")):
            output = await self.run_domain(domain, correct=False, reward_mode=mode,
                                             developer_test=domain == "taco")
            info = output.steps[-1].extra_fields["reward_extra_info"]
            self.assertEqual(info["acc"], 0)
            self.assertEqual(info["optimizer_terminal_component"], 0)
            self.assertGreater(info["optimizer_process_total"], 0)

    async def test_malformed_submission_keeps_process_and_fails_outcome_for_all_domains(self):
        for domain in ("hotpotqa", "taco", "vision_r1"):
            with self.subTest(domain=domain):
                output = await self.run_domain(domain, malformed=True, developer_test=domain == "taco")
                info = output.steps[-1].extra_fields["reward_extra_info"]
                self.assertEqual(info["terminal_reward"], 0)
                self.assertEqual(info["optimizer_terminal_component"], 0)
                self.assertFalse(info["terminal_gate_passed"])
                self.assertGreater(info["optimizer_process_total"], 0)
                self.assertAlmostEqual(sum(step.reward_score for step in output.steps), info["optimizer_process_total"])

    async def test_same_prefix_check_scores_ignore_outcomes_but_normalization_counts_applicable_checks(self):
        for domain, mode in (("hotpotqa", "uniform_certificate"), ("vision_r1", "uniform_visual_certificate")):
            credits = []
            raw_credits = []
            denominators = []
            for options in ({"correct": True}, {"correct": False}, {"incomplete": True}, {"malformed": True}):
                output = await self.run_domain(domain, reward_mode=mode, **options)
                denominator = output.steps[-1].extra_fields["reward_extra_info"]["verification_audit"]["process_denominator"]
                prefix = [step.reward_score for step in output.steps[:-1]]
                credits.append(prefix)
                raw_credits.append([value * denominator for value in prefix])
                denominators.append(denominator)
            self.assertEqual(credits[0], credits[1])
            self.assertEqual(denominators, [3, 3, 2, 3] if domain == "hotpotqa" else [2, 2, 1, 2])
            for raw_credit in raw_credits[1:]:
                for actual, expected in zip(raw_credit, raw_credits[0], strict=True):
                    self.assertAlmostEqual(actual, expected)

    async def test_no_submission_does_not_change_terminal_only_phases(self):
        for domain in ("hotpotqa", "taco", "vision_r1"):
            for options in ({"update": 50}, {"validation": True}):
                output = await self.run_domain(domain, incomplete=True, **options)
                info = output.steps[-1].extra_fields["reward_extra_info"]
                self.assertEqual(info["optimizer_process_weight"], 0)
                self.assertEqual(sum(step.reward_score for step in output.steps), 0)


if __name__ == "__main__":
    unittest.main()
