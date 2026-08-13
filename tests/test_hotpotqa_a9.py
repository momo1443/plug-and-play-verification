import asyncio
import dataclasses
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from recipes.hotpotqa.a9_behavioral_verifier import (
    A9_VERIFIER_VERSION,
    CandidateScore,
    build_probe_plan,
    encode_action_candidate,
    extract_candidate_score,
    load_entity_library,
    parse_target_slot,
    replay_behavioral_artifact,
    verify_behavioral_scores,
    visible_probe_inputs_valid,
)
from recipes.hotpotqa.hotpotqa_agent_flow import HotpotQAAgentFlow, _visible_passages
from recipes.hotpotqa.prepare_formal_rlvr_run import prepare_formal_rlvr_run
from recipes.hotpotqa.reward_arm import (
    RewardArm,
    parse_reward_arm,
    scale_terminal_reward,
    search_step_reward,
    training_reward_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _passages():
    return [
        {
            "pid": 10,
            "title": "Henry Miller",
            "text": "Henry Miller married June Miller in 1924. She was an American writer.",
            "score": 1.0,
            "sentence_evidence": [],
        }
    ]


class A9FormalPreparationTest(unittest.TestCase):
    def test_formal_preflight_accepts_a9_as_a_trainable_arm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "Required RLVR artifact"):
                prepare_formal_rlvr_run(
                    project_dir=PROJECT_ROOT,
                    arm="A9",
                    run_mode="main",
                    train_path=temp_root / "train.parquet",
                    validation_path=temp_root / "validation.parquet",
                    corpus_dir=temp_root / "corpus",
                    evidence_sidecar_path=temp_root / "evidence.sqlite",
                    model_path=temp_root / "model",
                    output_dir=temp_root / "output",
                    train_max_samples=30_000,
                    val_max_samples=7_405,
                    train_batch_size=20,
                    rollout_n=4,
                    total_training_steps=1_500,
                    num_gpus=8,
                    agent_workers=8,
                    gamma=1.0,
                )


class A9ProbeConstructionTest(unittest.TestCase):
    def test_target_is_observation_grounded_and_typed(self):
        target = parse_target_slot("June Miller nationality", _passages())
        self.assertIsNotNone(target)
        self.assertEqual(target.surface, "June Miller")
        self.assertEqual(target.entity_type, "person")
        self.assertEqual(target.suffix, "nationality")

    def test_first_search_without_observation_is_ineligible(self):
        self.assertIsNone(parse_target_slot("June Miller nationality", []))

    def test_title_metadata_does_not_ground_a_target(self):
        passages = [{**_passages()[0], "text": "An American writer was born in 1902."}]
        self.assertIsNone(parse_target_slot("June Miller nationality", passages))

    def test_prompt_visibility_uses_the_same_passage_budget(self):
        from recipes.hotpotqa.env.search_tool import Passage

        passages = [
            ("q1", Passage(pid=1, title="Hidden title", text="Visible One", score=1.0)),
            ("q2", Passage(pid=2, title="Hidden title", text="Visible Two", score=1.0)),
        ]
        visible = _visible_passages(passages, max_chars=40)
        self.assertEqual([item[1].text for item in visible], ["Visible One"])

    def test_probe_plan_is_deterministic_and_collision_free(self):
        kwargs = dict(
            query="June Miller nationality",
            question="What nationality was Henry Miller's wife?",
            history_actions=["Henry Miller wife"],
            passages=_passages(),
            sample_key="train:17",
            turn_index=2,
            token_length=lambda value: len(value.split()),
            probe_seed=42,
        )
        first = build_probe_plan(**kwargs)
        second = build_probe_plan(**kwargs)
        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        sensitivity_text = json.dumps(first.sensitivity_passages)
        invariance_text = json.dumps(first.invariance_passages)
        self.assertNotIn("June Miller", sensitivity_text)
        self.assertIn(first.sensitivity_candidate, sensitivity_text)
        self.assertIn("June Miller", invariance_text)
        self.assertIn(first.distractor_candidate, invariance_text)
        self.assertNotEqual(first.sensitivity_candidate, first.distractor_candidate)
        self.assertTrue(
            visible_probe_inputs_valid(
                first,
                sensitivity_passages=first.sensitivity_passages,
                invariance_passages=first.invariance_passages,
            )
        )

    def test_visible_probe_check_rejects_truncated_target(self):
        plan = build_probe_plan(
            query="June Miller nationality",
            question="What nationality was Henry Miller's wife?",
            history_actions=["Henry Miller wife"],
            passages=_passages(),
            sample_key="train:17",
            turn_index=2,
            token_length=lambda value: len(value.split()),
        )
        self.assertIsNotNone(plan)
        truncated_invariance = [dict(record) for record in plan.invariance_passages]
        truncated_invariance[0]["text"] = plan.distractor_sentence
        self.assertFalse(
            visible_probe_inputs_valid(
                plan,
                sensitivity_passages=plan.sensitivity_passages,
                invariance_passages=truncated_invariance,
            )
        )

    def test_entity_library_is_versioned_and_complete(self):
        version, library = load_entity_library()
        self.assertEqual(version, "hotpotqa-a9-entity-library-v1")
        self.assertEqual(set(library), {"person", "place", "organization", "work"})
        self.assertTrue(all(len(values) >= 4 for values in library.values()))

    def test_gold_arguments_are_rejected_by_probe_api(self):
        with self.assertRaises(TypeError):
            build_probe_plan(
                query="June Miller nationality",
                question="Question",
                history_actions=[],
                passages=_passages(),
                sample_key="train:1",
                turn_index=2,
                token_length=lambda value: len(value.split()),
                gold_evidence_ids=["forbidden"],
            )


class A9ScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            WORKSPACE_ROOT / "models/Qwen3.5-4B",
            local_files_only=True,
        )

    def test_action_candidate_encoding_covers_only_target_tokens(self):
        target = parse_target_slot("June Miller nationality", _passages())
        self.assertIsNotNone(target)
        response = '<tool_call>{"name":"search","arguments":{"query":"June Miller nationality"}}</tool_call>'
        encoding = encode_action_candidate(
            self.tokenizer,
            response_text=response,
            query="June Miller nationality",
            target=target,
            candidate="Eleanor Hartley",
        )
        self.assertIsNotNone(encoding)
        target_ids = encoding.token_ids[encoding.target_token_start : encoding.target_token_end]
        self.assertEqual(
            self.tokenizer.decode(target_ids),
            "Eleanor Hartley",
        )

    def test_prompt_logprob_alignment_and_average(self):
        full_ids = [11, 12, 13, 14, 15, 16]
        prompt_ids = [[12], [13], [14], [15], [16], [0]]
        prompt_logprobs = [[-0.1], [-0.2], [-0.3], [-0.4], [-0.6], [0.0]]
        score = extract_candidate_score(
            candidate="candidate",
            full_prompt_token_ids=full_ids,
            absolute_target_start=3,
            absolute_target_end=5,
            prompt_logprob_ids=prompt_ids,
            prompt_logprobs=prompt_logprobs,
            policy_snapshot_id="step-8",
        )
        self.assertIsNotNone(score)
        self.assertEqual(score.token_ids, (14, 15))
        self.assertEqual(score.token_logprobs, (-0.3, -0.4))
        self.assertAlmostEqual(score.token_average_logprob, -0.35)

    def test_prompt_logprob_mismatch_fails_closed(self):
        score = extract_candidate_score(
            candidate="candidate",
            full_prompt_token_ids=[11, 12, 13],
            absolute_target_start=1,
            absolute_target_end=2,
            prompt_logprob_ids=[[99], [13], [0]],
            prompt_logprobs=[[-0.1], [-0.2], [0.0]],
            policy_snapshot_id="step-8",
        )
        self.assertIsNone(score)

    def test_agent_flow_candidate_scoring_uses_verl_prompt_logprob_contract(self):
        plan = build_probe_plan(
            query="June Miller nationality",
            question="What nationality was Henry Miller's wife?",
            history_actions=["Henry Miller wife"],
            passages=_passages(),
            sample_key="train:17",
            turn_index=2,
            token_length=lambda value: len(self.tokenizer.encode(value, add_special_tokens=False)),
        )
        self.assertIsNotNone(plan)
        response = '<tool_call>{"name":"search","arguments":{"query":"June Miller nationality"}}</tool_call>'

        class FakeServer:
            sampling_params = None

            async def generate(self, *, request_id, prompt_ids, sampling_params):
                self.sampling_params = sampling_params
                return SimpleNamespace(
                    extra_fields={
                        "prompt_ids": [[token_id] for token_id in prompt_ids[1:]] + [[0]],
                        "prompt_logprobs": [[-0.25] for _ in prompt_ids[1:]] + [[0.0]],
                        "global_steps": 7,
                    }
                )

        flow = object.__new__(HotpotQAAgentFlow)
        flow.tokenizer = self.tokenizer
        flow.server_manager = FakeServer()
        prompt_ids = self.tokenizer.encode("prefix", add_special_tokens=False)
        score = asyncio.run(
            flow._a9_score_candidate(
                prompt_ids=prompt_ids,
                response_text=response,
                query="June Miller nationality",
                plan=plan,
                candidate=plan.sensitivity_candidate,
            )
        )
        self.assertIsNotNone(score)
        self.assertEqual(score.policy_snapshot_id, "7")
        self.assertEqual(score.token_average_logprob, -0.25)
        self.assertEqual(flow.server_manager.sampling_params["prompt_logprobs"], 0)
        self.assertEqual(flow.server_manager.sampling_params["max_tokens"], 1)

    def test_behavioral_reward_has_three_levels_and_replay_hash(self):
        plan = build_probe_plan(
            query="June Miller nationality",
            question="What nationality was Henry Miller's wife?",
            history_actions=["Henry Miller wife"],
            passages=_passages(),
            sample_key="train:17",
            turn_index=2,
            token_length=lambda value: len(value.split()),
        )
        self.assertIsNotNone(plan)

        def score(candidate, value):
            return CandidateScore(candidate, (1,), (value,), value, "step-9")

        partial = verify_behavioral_scores(
            plan=plan,
            sensitivity_new=score(plan.sensitivity_candidate, -0.1),
            sensitivity_old=score(plan.target.surface, -1.0),
            invariance_target=score(plan.target.surface, -1.0),
            invariance_distractor=score(plan.distractor_candidate, -0.1),
            source_state_hash="state",
            expected_policy_snapshot_id="step-9",
        )
        self.assertEqual(partial.raw_process_reward, 0.5)
        self.assertFalse(partial.joint_pass)
        self.assertFalse(partial.process_component_valid)
        self.assertEqual(partial.optimizer_process_value, 0.0)
        self.assertEqual(partial.verdict_class, "D")
        self.assertEqual(partial.artifact["schema_version"], A9_VERIFIER_VERSION)
        self.assertEqual(len(partial.artifact["replay_hash"]), 64)
        self.assertFalse(partial.artifact["validity_checks"]["gold_inputs_used"])
        self.assertTrue(replay_behavioral_artifact(partial.artifact))

        tampered = json.loads(json.dumps(partial.artifact))
        tampered["raw_process_reward"] = 1.0
        self.assertFalse(replay_behavioral_artifact(tampered))

        full = verify_behavioral_scores(
            plan=plan,
            sensitivity_new=score(plan.sensitivity_candidate, -0.1),
            sensitivity_old=score(plan.target.surface, -1.0),
            invariance_target=score(plan.target.surface, -0.1),
            invariance_distractor=score(plan.distractor_candidate, -1.0),
            source_state_hash="state",
            expected_policy_snapshot_id="step-9",
        )
        self.assertEqual(full.raw_process_reward, 1.0)
        self.assertTrue(full.joint_pass)
        self.assertTrue(full.process_component_valid)
        self.assertEqual(full.optimizer_process_value, 1.0)
        self.assertEqual(full.verdict_class, "A")

        failed = verify_behavioral_scores(
            plan=plan,
            sensitivity_new=score(plan.sensitivity_candidate, -1.0),
            sensitivity_old=score(plan.target.surface, -0.1),
            invariance_target=score(plan.target.surface, -1.0),
            invariance_distractor=score(plan.distractor_candidate, -0.1),
            source_state_hash="state",
            expected_policy_snapshot_id="step-9",
        )
        self.assertEqual(failed.raw_process_reward, 0.0)
        self.assertTrue(failed.process_component_valid)
        self.assertEqual(failed.optimizer_process_value, 0.0)
        self.assertEqual(failed.verdict_class, "B")

    def test_snapshot_mismatch_fails_closed(self):
        plan = build_probe_plan(
            query="June Miller nationality",
            question="Question",
            history_actions=[],
            passages=_passages(),
            sample_key="train:1",
            turn_index=2,
            token_length=lambda value: len(value.split()),
        )
        self.assertIsNotNone(plan)
        left = CandidateScore("left", (1,), (-0.1,), -0.1, "step-1")
        right = CandidateScore("right", (2,), (-0.2,), -0.2, "step-2")
        result = verify_behavioral_scores(
            plan=plan,
            sensitivity_new=left,
            sensitivity_old=right,
            invariance_target=left,
            invariance_distractor=left,
            source_state_hash="state",
            expected_policy_snapshot_id="step-1",
        )
        self.assertFalse(result.eligible)
        self.assertIsNone(result.raw_process_reward)
        self.assertFalse(result.process_component_valid)
        self.assertEqual(result.optimizer_process_value, 0.0)
        self.assertEqual(result.verdict_class, "C")

    def test_sampled_action_snapshot_mismatch_fails_closed(self):
        plan = build_probe_plan(
            query="June Miller nationality",
            question="Question",
            history_actions=[],
            passages=_passages(),
            sample_key="train:1",
            turn_index=2,
            token_length=lambda value: len(value.split()),
        )
        self.assertIsNotNone(plan)
        score = CandidateScore("candidate", (1,), (-0.1,), -0.1, "step-2")
        result = verify_behavioral_scores(
            plan=plan,
            sensitivity_new=score,
            sensitivity_old=score,
            invariance_target=score,
            invariance_distractor=score,
            source_state_hash="state",
            expected_policy_snapshot_id="step-1",
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.artifact["reason"], "sampled_action_snapshot_mismatch")


class A9RewardAndLauncherTest(unittest.TestCase):
    def test_a9_reward_contract_is_outcome_independent_50_50(self):
        self.assertIs(parse_reward_arm("a9"), RewardArm.A9)
        contract = training_reward_contract(RewardArm.A9)
        self.assertEqual(dataclasses.asdict(contract), {
            "process_weight": 0.5,
            "terminal_weight": 0.5,
            "final_response_mask": 1,
        })
        self.assertEqual(search_step_reward(RewardArm.A9, 0.0, is_validation=False), 0.0)
        with self.assertRaisesRegex(ValueError, "partial and invalid probes must be masked"):
            search_step_reward(RewardArm.A9, 0.5, is_validation=False)
        self.assertEqual(search_step_reward(RewardArm.A9, 1.0, is_validation=False), 0.5)
        self.assertEqual(scale_terminal_reward(RewardArm.A9, 1.0, is_validation=False), 0.5)

    def test_launcher_and_manifest_contain_a9_without_gated_arm(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_a9.sh").read_text(encoding="utf-8")
        shared = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")
        preflight = (PROJECT_ROOT / "recipes/hotpotqa/prepare_formal_rlvr_run.py").read_text(
            encoding="utf-8"
        )
        reward_arm = (PROJECT_ROOT / "recipes/hotpotqa/reward_arm.py").read_text(encoding="utf-8")
        self.assertIn("HOTPOTQA_REWARD_ARM=A9", launcher)
        self.assertIn("A1|A2|A3|A6|A7|A9|A8_LR50", shared)
        self.assertIn('RewardArm.A9: A9_VERIFIER_VERSION', preflight)
        self.assertIn('"process_is_terminal_em_gated": False', preflight)
        self.assertIn('"a9_probe_seed": a9_probe_seed', preflight)
        self.assertIn('"a9_verdict_classes":', preflight)
        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3,4,5,6,7}"', launcher)
        self.assertIn("HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false", launcher)
        self.assertIn("HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false", launcher)
        self.assertNotIn("HOTPOTQA_NUM_GPUS", launcher)
        self.assertNotIn("HOTPOTQA_AGENT_WORKERS", launcher)
        self.assertNotIn("HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION", launcher)
        self.assertNotIn("A9_GATED", reward_arm + launcher + preflight)

    def test_a9_launcher_rejects_a_separate_pilot_experiment(self):
        launcher = PROJECT_ROOT / "examples/hotpotqa/run_a9.sh"
        result = subprocess.run(
            ["bash", str(launcher)],
            env={**os.environ, "HOTPOTQA_RUN_MODE": "pilot64"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be main", result.stderr)


if __name__ == "__main__":
    unittest.main()
