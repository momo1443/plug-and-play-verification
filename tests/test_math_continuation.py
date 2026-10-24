"""Actual math flow/evaluation methods, with only GPU generation boundaries mocked."""
from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_r1.evaluation.answers import final_answer_record, score_aime
from agent_r1.evaluation.consistency import consistency_record
from agent_r1.evaluation.summarize import summarize
from recipes.deepscaler.prepare_run import build_manifest
from recipes.deepscaler.prompts import build_math_messages, math_continuation_message
from recipes.deepscaler.submission import FINAL_ANSWER_PREFIX, generate_prefilled_submission, prefilled_final_record, prefilled_box_end
from verl.utils.tokenizer import normalize_token_ids
from recipes.deepscaler.trajectory_reward import compute_tool_trajectory_reward, verify_process
from pnp_test_utils import load_source

ROOT = Path(__file__).resolve().parents[1]
QUESTION = [{"role": "user", "content": "Compute the sum of two and three."}]
REASONING = "Check the arithmetic: $2 + 3 = 5$. Continue reasoning."


class Tokenizer:
    def __init__(self):
        self.responses = {}
        self.histories = []
        self.template_options = []

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        self.histories.append(copy.deepcopy(messages))
        self.template_options.append(dict(kwargs))
        if not tokenize:
            return repr(messages)
        return list(range(10 + 5 * len(messages)))

    def decode(self, ids, **kwargs):
        if ids[0] not in self.responses:
            if self.template_options[-1].get('enable_thinking') is False:
                return '</think>\n\n'
            return getattr(self, 'prompt_tail', '')
        return self.responses[ids[0]]

    def encode(self, text, **kwargs):
        assert text == FINAL_ANSWER_PREFIX
        return [9000, 9001, 9002]


class Generator:
    def __init__(self, tokenizer, texts, sizes=None):
        self.tokenizer = tokenizer
        self.texts = texts
        self.sizes = sizes or [1] * len(texts)
        self.calls = []

    async def generate(self, **kwargs):
        index = len(self.calls)
        self.calls.append(copy.deepcopy(kwargs))
        source_index = min(index, len(self.texts) - 1)
        text = self.texts[source_index]
        if 9000 in kwargs['prompt_ids'] and text.startswith(FINAL_ANSWER_PREFIX):
            text = text[len(FINAL_ANSWER_PREFIX):]
        self.tokenizer.responses[index + 100] = text
        size = min(self.sizes[source_index], kwargs["sampling_params"]["max_tokens"])
        return SimpleNamespace(token_ids=[index + 100] * size,
                               log_probs=[-0.25] * size, routed_experts=None)


class MathContinuationTest(unittest.IsolatedAsyncioTestCase):
    async def run_flow(self, texts, *, turns=5,
                       budget=4096, sizes=None, context=8192, update=51,
                       validation=False, gold="5", implicit_thinking=False,
                       final_turn_submission=True):
        module = load_source("recipes/deepscaler/agent_flow.py",
            final_answer_record=final_answer_record, verify_process=verify_process,
            compute_tool_trajectory_reward=compute_tool_trajectory_reward,
            build_math_messages=build_math_messages, math_continuation_message=math_continuation_message,
            FINAL_ANSWER_PREFIX=FINAL_ANSWER_PREFIX, generate_prefilled_submission=generate_prefilled_submission,
            prefilled_final_record=prefilled_final_record, normalize_token_ids=normalize_token_ids)
        flow = object.__new__(module.DeepScaleRPaperAgentFlow)
        tokenizer = Tokenizer()
        tokenizer.prompt_tail = '<think>\n' if implicit_thinking else ''
        generator = Generator(tokenizer, texts, sizes)
        generator.verification_call_counts = []
        def verify_after_rollout(segments, **kwargs):
            generator.verification_call_counts.append(len(generator.calls))
            return verify_process(segments, **kwargs)
        module.verify_process = verify_after_rollout
        flow.server_manager = generator
        flow.tokenizer = tokenizer
        flow.reward_mode = "uniform_equation_process"
        flow.max_steps = turns
        flow.response_length = budget
        flow.prompt_length = context if turns > 1 else 2048
        flow.initial_prompt_length = 2048
        flow.context_length = context
        flow.em_warmup_steps = 50
        flow.final_turn_submission = final_turn_submission
        flow.apply_chat_template_kwargs = {}
        flow.judge_server = None

        async def template(messages):
            return tokenizer.apply_chat_template(messages, **flow.apply_chat_template_kwargs)
        flow.apply_chat_template = template
        output = await flow.run({"temperature": .7}, raw_prompt=QUESTION,
            reward_model={"ground_truth": gold}, extra_info={"index": 1, "split": "train"},
            _agent_r1_global_step=update, _agent_r1_is_validation=validation)
        return output, generator, tokenizer

    async def test_open_prompt_thinking_requires_visible_final_submission(self):
        output, gen, _ = await self.run_flow([
            r"Check $2+3=5$. I should write `\boxed{...}`. The answer is 5.",
            r"Rechecked $2+3=5$.</think>\boxed{5}",
        ], implicit_thinking=True)
        self.assertEqual(len(gen.calls), 2)
        info = output.steps[-1].extra_fields['reward_extra_info']
        self.assertEqual(info['acc'], 1)
        self.assertEqual(info['termination_reason'], 'final_answer')
        self.assertEqual(info['verification_checks'], [1, 1])

    async def test_five_turns_backfill_source_credits_and_final_outcome(self):
        output, gen, tokenizer = await self.run_flow([REASONING] * 4 + [r"\boxed{5}"])
        self.assertEqual(len(output.steps), 5)
        self.assertEqual([step.num_turns for step in output.steps], [1, 2, 3, 4, 5])
        self.assertEqual([call["sampling_params"]["max_tokens"] for call in gen.calls], [820] * 4 + [64])
        self.assertEqual(gen.verification_call_counts, [5])
        self.assertEqual(output.metrics["step_tool_calls"], [0.0] * 5)
        self.assertEqual(output.metrics["tool_calls"], 0.0)
        info = output.steps[-1].extra_fields["reward_extra_info"]
        self.assertEqual(info["termination_reason"], "final_answer")
        self.assertEqual(info["generated_response_tokens"], 5)
        self.assertEqual(info["acc"], 1)
        for step in output.steps[:4]:
            self.assertAlmostEqual(step.reward_score, info["optimizer_process_weight"] / 4)
        self.assertAlmostEqual(output.steps[-1].reward_score, info["optimizer_terminal_weight"])
        self.assertAlmostEqual(sum(s.reward_score for s in output.steps), 1)
        self.assertIn("last available", tokenizer.histories[-1][-1]["content"])

    async def test_final_submission_turn_closes_thinking_and_keeps_budget(self):
        output, gen, tokenizer = await self.run_flow(
            [REASONING] * 4 + [r"\boxed{5}"], implicit_thinking=True,
            sizes=[820] * 5,
        )
        info = output.steps[-1].extra_fields['reward_extra_info']
        self.assertEqual(info['terminal_em'], 1)
        self.assertEqual(info['generated_response_tokens'], 3280 + 64)
        self.assertEqual([c['sampling_params']['max_tokens'] for c in gen.calls], [820]*4+[64])
        self.assertEqual(info['final_answer_prefill_step'], 5)
        self.assertEqual(gen.calls[-1]['prompt_ids'][-3:], [9000, 9001, 9002])
        self.assertNotIn(9000, output.steps[-1].response_ids)
        self.assertEqual(len(output.steps[-1].response_ids), len(output.steps[-1].response_logprobs))
        self.assertIn('next turn is reserved', tokenizer.histories[-2][-1]['content'])
        self.assertEqual(tokenizer.template_options[-1], {'enable_thinking': False})
        self.assertTrue(all('enable_thinking' not in c for c in tokenizer.template_options[:-1]))
        self.assertIn('816 output tokens', tokenizer.histories[-1][-1]['content'])
        self.assertIn('Respond ONLY', tokenizer.histories[-1][-1]['content'])

    async def test_full_thinking_protocol_remains_selectable(self):
        output, _, tokenizer = await self.run_flow(
            [REASONING] * 4 + [r"\boxed{5}"], implicit_thinking=True,
            final_turn_submission=False,
        )
        self.assertFalse(output.steps[-1].extra_fields['reward_extra_info']['has_final_answer'])
        self.assertTrue(all('enable_thinking' not in c for c in tokenizer.template_options))

    async def test_restarted_submission_uses_same_selected_answer_as_replay(self):
        completion = r'I have settled on the result. \boxed{5}'
        output, gen, _ = await self.run_flow([REASONING] * 4 + [completion])
        info = output.steps[-1].extra_fields['reward_extra_info']
        self.assertEqual(len(gen.calls), 5)
        self.assertEqual(info['terminal_em'], 1)
        self.assertTrue(info['has_final_answer'])
        self.assertEqual(prefilled_final_record(completion)['answer'], '5')
        self.assertEqual(info['deterministic_verification']['participating_step_indices'], [1, 2, 3, 4])

    async def test_final_prefill_does_not_grade_the_answer_as_process(self):
        output, gen, _ = await self.run_flow([REASONING] * 4 + ['5}'])
        self.assertEqual(len(gen.calls), 5)
        info = output.steps[-1].extra_fields['reward_extra_info']
        self.assertEqual(info['terminal_em'], 1)
        self.assertEqual(info['deterministic_verification']['participating_step_indices'], [1, 2, 3, 4])
        self.assertEqual(info['final_answer_prefill'], FINAL_ANSWER_PREFIX)
        self.assertEqual(output.steps[-1].response_ids, [104])

    async def test_wrong_final_stops_generation_and_preserves_process_credit(self):
        output, gen, _ = await self.run_flow([REASONING, r"\boxed{7}"])
        self.assertEqual(len(gen.calls), 2)
        info = output.steps[-1].extra_fields["reward_extra_info"]
        self.assertTrue(info["terminal_eligible"])
        self.assertEqual(info["acc"], 0)
        self.assertGreater(output.steps[0].reward_score, 0)
        self.assertEqual(output.steps[-1].reward_score, 0)

    async def test_incomplete_retains_process_credits_with_zero_terminal_reward(self):
        output, _, _ = await self.run_flow([REASONING] * 5)
        self.assertEqual(len(output.steps), 5)
        self.assertTrue(all(step.response_ids for step in output.steps))
        info = output.steps[-1].extra_fields["reward_extra_info"]
        self.assertFalse(info["terminal_eligible"])
        self.assertFalse(info["terminal_gate_passed"])
        self.assertFalse(info["process_requires_terminal_submission"])
        self.assertEqual(info["terminal_em"], 0)
        self.assertEqual(info["optimizer_terminal_component"], 0)
        self.assertEqual(info["verified_success"], 0)
        self.assertEqual(output.steps[-1].reward_score, 0)
        self.assertTrue(all(step.reward_score > 0 for step in output.steps[:4]))
        self.assertAlmostEqual(sum(step.reward_score for step in output.steps), info["optimizer_process_total"])
        self.assertAlmostEqual(info["optimizer_process_total"], info["optimizer_process_weight"])
        self.assertEqual(info["termination_reason"], "max_steps")
        self.assertEqual(info["deterministic_verification"]["applicable_checks"], [1] * 4)
        self.assertEqual(info["verification_consistency"], 1)
        summary = summarize([{"steps": [info]}])
        self.assertEqual(summary["C_d_mean"], 1)
        self.assertEqual(summary["accuracy"], 0)
        self.assertEqual(summary["M_d"], 0)

    async def test_incomplete_without_verified_progress_earns_no_reward(self):
        output, _, _ = await self.run_flow(["Check $2+3=6$."] * 5)
        self.assertTrue(all(step.reward_score == 0 for step in output.steps))
        info = output.steps[-1].extra_fields["reward_extra_info"]
        self.assertEqual(info["terminal_em"], 0)
        self.assertEqual(info["deterministic_process_reward"], 0)

    async def test_unsubmitted_progress_keeps_outcome_only_warmup_and_evaluation(self):
        for settings in ({"update":50}, {"validation":True}):
            output, _, _ = await self.run_flow([REASONING] * 5, **settings)
            self.assertTrue(all(step.reward_score == 0 for step in output.steps))
            info = output.steps[-1].extra_fields["reward_extra_info"]
            self.assertEqual(info["optimizer_process_weight"], 0)
            self.assertEqual(info["terminal_em"], 0)
            self.assertEqual(info["verification_consistency"], 1)

    async def test_corrected_equation_gets_new_credit_without_erasing_failed_history(self):
        output, _, _ = await self.run_flow([
            "Check $2 + 3 = 6$.", "Corrected: $2 + 3 = 5$.", r"\boxed{5}"
        ])
        info = output.steps[-1].extra_fields["reward_extra_info"]
        self.assertEqual(info["acc"], 1)
        self.assertEqual(info["verification_checks"], [0, 1])
        self.assertEqual(info["verification_passed_check_count"], 1)
        self.assertEqual(info["verification_failed_check_count"], 1)
        self.assertEqual(info["deterministic_process_reward"], 0.5)
        self.assertEqual(info["verification_consistency"], 0)
        self.assertEqual(output.steps[0].reward_score, 0)
        self.assertAlmostEqual(output.steps[1].reward_score, info["optimizer_process_weight"] / 2)
        self.assertAlmostEqual(output.steps[2].reward_score, info["optimizer_terminal_weight"])
        audit = info["deterministic_verification"]
        self.assertEqual(audit["participating_step_indices"], [1, 2])
        self.assertEqual(audit["normalizer"], 2)
        self.assertEqual(audit["process_segment_audits"][0]["steps"][0]["equation_checks"][0]["source_step"], 1)

    async def test_length_capped_equation_tail_is_not_a_failed_assertion(self):
        output, _, _ = await self.run_flow(
            ["Check $2+3=6$.\n$37*40=1", r"\boxed{5}"], sizes=[820, 1],
        )
        info = output.steps[-1].extra_fields['reward_extra_info']
        self.assertEqual(info['truncated_policy_steps'], [1])
        self.assertEqual(info['verification_checks'], [0])
        self.assertEqual(info['deterministic_verification']['evidence_parser_version'], 3)

    async def test_five_full_turns_do_not_multiply_the_total_token_budget(self):
        for budget, cap, last_cap in ((4096, 820, 816), (5120, 1024, 1024)):
            output, gen, _ = await self.run_flow([REASONING] * 5,
                budget=budget, sizes=[cap] * 5)
            self.assertEqual([c["sampling_params"]["max_tokens"] for c in gen.calls[:4]], [cap] * 4)
            self.assertEqual(sum(c["sampling_params"]["max_tokens"] for c in gen.calls[4:]), last_cap)
            self.assertEqual(sum(len(step.response_ids) for step in output.steps), budget)

    async def test_warmup_and_validation_use_only_final_reward(self):
        for update, validation in ((50, False), (51, True)):
            output, _, _ = await self.run_flow([REASONING, r"\boxed{5}"], update=update, validation=validation)
            self.assertEqual([step.reward_score for step in output.steps], [0, 1])
            info = output.steps[-1].extra_fields["reward_extra_info"]
            self.assertEqual(info["process_reward"], 0)
            self.assertEqual(info["deterministic_process_reward"], 1)
            self.assertEqual(info["verification_passed_check_count"], 1)
            self.assertEqual(info["verification_failed_check_count"], 0)

    async def test_one_and_eight_turn_limits(self):
        output, gen, tokenizer = await self.run_flow([r"\boxed{5}"], turns=1)
        self.assertEqual(len(output.steps), 1)
        self.assertEqual(gen.calls[0]["sampling_params"]["max_tokens"], 4096)
        self.assertEqual(tokenizer.histories, [QUESTION])
        output, gen, _ = await self.run_flow([REASONING] * 7 + [r"\boxed{5}"], turns=8)
        self.assertEqual(len(output.steps), 8)
        self.assertTrue(all(c["sampling_params"]["max_tokens"] == 512 for c in gen.calls[:7]))
        self.assertEqual(gen.calls[-1]['sampling_params']['max_tokens'], 64)

    async def test_rounded_partition_never_exceeds_whole_trajectory_budget(self):
        output, gen, _ = await self.run_flow([REASONING] * 4, turns=4, budget=10, sizes=[3] * 4)
        self.assertEqual([c["sampling_params"]["max_tokens"] for c in gen.calls], [3, 3, 3, 1])
        self.assertEqual(sum(len(s.response_ids) for s in output.steps), 10)
        self.assertGreater(sum(s.reward_score for s in output.steps), 0)
        self.assertEqual(output.steps[-1].extra_fields["reward_extra_info"]["optimizer_terminal_component"], 0)

    async def test_context_exhaustion_retains_failed_trajectory(self):
        output, gen, _ = await self.run_flow([REASONING], context=30)
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(gen.calls[0]["sampling_params"]["max_tokens"], 10)
        self.assertEqual(output.steps[-1].extra_fields["reward_extra_info"]["termination_reason"], "context_limit")
        self.assertGreater(output.steps[0].reward_score, 0)
        self.assertEqual(output.steps[-1].extra_fields["reward_extra_info"]["optimizer_terminal_component"], 0)

    async def test_changing_gold_never_changes_policy_context(self):
        histories = []
        for gold in ("5", "7"):
            _, _, tokenizer = await self.run_flow([REASONING, r"\boxed{5}"], gold=gold)
            histories.append(tokenizer.histories)
        self.assertEqual(*histories)


class MathEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.module = load_source("scripts/deepscaler/eval_aime2026.py",
            final_answer_record=final_answer_record, build_math_messages=build_math_messages,
            math_continuation_message=math_continuation_message, AIME_EXTRACTION_VERSION="test",
            FINAL_ANSWER_PREFIX=FINAL_ANSWER_PREFIX, prefilled_final_record=prefilled_final_record,
            prefilled_box_end=prefilled_box_end)

    def test_evaluation_active_batch_keeps_incomplete_denominator_and_total_budget(self):
        tokenizer = Tokenizer()
        histories = []
        limits = []
        texts = [r"\boxed{7}"] + [REASONING] * 5

        class LLM:
            def generate(self, prompts, params):
                limits.append(params.max_tokens)
                outputs = []
                for prompt in prompts:
                    index = len(histories)
                    histories.append(prompt)
                    tokenizer.responses[index + 100] = texts[index]
                    outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=[index + 100] * params.max_tokens)]))
                return outputs
        states = self.module.generate_math_rollouts(LLM(), tokenizer, SimpleNamespace,
            [QUESTION, QUESTION], max_tokens=13)
        self.assertEqual(len(states), 2)
        self.assertEqual(limits, [3, 3, 3, 3, 1])
        self.assertEqual(states[0]["final_response"], r"\boxed{7}")
        self.assertEqual(len(states[0]["responses"]), 1)
        self.assertFalse(score_aime(states[0]["final_response"], "5")["is_correct"])
        self.assertIsNone(states[1]["final_response"])
        self.assertEqual(states[1]["response_tokens"], 13)
        self.assertEqual(len(states[1]["responses"]), 5)
        self.assertEqual(states[1]["termination_reason"], "max_steps")
        self.assertIn("last available", histories[-1])

    def test_evaluation_generation_count_mismatch_is_explicit_failure(self):
        llm = SimpleNamespace(generate=lambda *args: [])
        with self.assertRaisesRegex(RuntimeError, "denominator"):
            self.module.generate_math_rollouts(llm, Tokenizer(), SimpleNamespace, [QUESTION])

    def test_evaluation_matches_training_thinking_and_submission_boundaries(self):
        class QwenTokenizer(Tokenizer):
            def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
                value = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
                if tokenize:
                    return value
                return ('assistant\n<think>\n\n</think>\n\n'
                        if kwargs.get('enable_thinking') is False else 'assistant\n<think>\n')
        tokenizer = QwenTokenizer()
        calls = []
        class LLM:
            def generate(self, prompts, params):
                i = len(calls); calls.append(prompts)
                tokenizer.responses[i+100] = ("5}" if i==4 else
                    "The answer is 5. Still thinking through the calculation.")
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[i+100]*params.max_tokens)])]
        states = self.module.generate_math_rollouts(LLM(), tokenizer, SimpleNamespace, [QUESTION])
        self.assertEqual(len(calls), 5)
        self.assertEqual(states[0]['final_response'], r"\boxed{5}")
        self.assertEqual(states[0]['response_tokens'], 4096)
        self.assertTrue(all(prompt[0].endswith('<think>\n') for prompt in calls[:4]))
        self.assertTrue(calls[4][0].endswith('</think>\n\n' + FINAL_ANSWER_PREFIX))


class MathManifestTest(unittest.TestCase):
    def test_manifest_records_turn_caps_total_budget_and_prompt_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "train.parquet").write_bytes(b"train")
            (root / "val.parquet").write_bytes(b"val")
            args = argparse.Namespace(project_dir=str(ROOT), output_dir=str(root / "out"),
                model_path=str(root), train_path=str(root / "train.parquet"), val_path=str(root / "val.parquet"),
                resume_from_path="", status="prepared", cuda_visible_devices="0,1,2,3,4,5,6,7", num_gpus=8,
                vllm_gpu_memory_utilization=.25, vllm_max_model_len=8192, vllm_max_num_batched_tokens=8192,
                vllm_max_num_seqs=20, train_max_samples=10000, train_batch_size=20, rollout_n=4,
                max_prompt_length=2048, max_response_length=4096, max_agent_steps=5, rollout_prompt_length=8192,
                total_training_steps=500, save_freq=50, seed=42, em_warmup_steps=50)
            manifest = build_manifest(args)
            config = manifest["training"]
            self.assertEqual(config["max_agent_steps"], 5)
            self.assertEqual(config["max_tokens_per_turn"], 820)
            self.assertEqual(config["sampled_prompt_count"], 10000)
            self.assertEqual(config["response_budget_scope"], "whole_trajectory")
            self.assertIn("recipes/deepscaler/prompts.py", manifest["code_sha256"])


class MathNormalizationTest(unittest.TestCase):
    def test_failed_turns_count_in_denominator_and_empty_turns_have_zero_credit(self):
        result = verify_process([(1, "Check $2 + 3 = 6$."),
            (2, "Corrected: $2 + 3 = 5$."), (3, "Continue reasoning without an equation.")])
        self.assertEqual(result.components_by_step(), {1: 0, 2: .5})
        self.assertEqual(result.audit["normalizer"], 2)
        self.assertEqual(result.audit["applicable_checks"], [0, 1])
        self.assertEqual(result.audit["process_segment_audits"][2]["normalized_credit"], 0)
        self.assertFalse(result.audit["process_segment_audits"][2]["participates_in_process_reward"])
        self.assertEqual(consistency_record(result, 1, eligible=True)["verification_consistency"], 0)

    def test_internal_segment_average_is_preserved_and_five_turn_total_is_bounded(self):
        text = "Step 1: $2 + 3 = 5$ and $4 + 4 = 9$.\nStep 2: $6 / 2 = 3$."
        result = verify_process([(turn, text) for turn in range(1, 6)])
        self.assertAlmostEqual(result.process_reward, .75)
        self.assertEqual(result.audit["normalizer"], 5)
        for turn, credit in result.components_by_step().items():
            self.assertAlmostEqual(credit, .15)
            checks = result.audit["credits_by_step"][turn]["steps"][0]["equation_checks"]
            self.assertTrue(all(check["source_step"] == turn for check in checks))

    def test_no_equations_has_zero_credit_and_no_applicable_checks(self):
        result = verify_process([(1, "Use a geometric argument."), (2, "Conclude the derivation.")])
        self.assertEqual(result.process_reward, 0)
        self.assertEqual(result.audit["normalizer"], 0)
        self.assertEqual(result.audit["applicable_checks"], [])

    def test_missing_records_or_duplicate_source_steps_are_explicit_failures(self):
        for segments in ([(1, None)], [(1, REASONING), (1, REASONING)], [(0, REASONING)]):
            with self.assertRaises(ValueError):
                verify_process(segments)


if __name__ == "__main__":
    unittest.main()
