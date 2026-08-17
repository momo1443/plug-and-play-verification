import inspect
import unittest
from pathlib import Path

from recipes.hotpotqa.prepare_formal_rlvr_run import (
    _reward_description,
    prepare_formal_rlvr_run,
)
from recipes.hotpotqa.reward_arm import RewardArm, training_reward_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HotpotQALauncherDefaultsTest(unittest.TestCase):
    def test_preflight_function_uses_optimized_actor_defaults(self):
        parameters = inspect.signature(prepare_formal_rlvr_run).parameters

        self.assertIs(parameters["actor_use_dynamic_bsz"].default, True)
        self.assertEqual(parameters["actor_max_token_len_per_gpu"].default, 8_192)
        self.assertIs(parameters["actor_param_offload"].default, False)
        self.assertIs(parameters["actor_optimizer_offload"].default, False)

    def test_shell_and_cli_defaults_use_optimized_actor_settings(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")
        preflight = (PROJECT_ROOT / "recipes/hotpotqa/prepare_formal_rlvr_run.py").read_text(encoding="utf-8")

        self.assertIn("HOTPOTQA_ACTOR_USE_DYNAMIC_BSZ:-true", launcher)
        self.assertIn("HOTPOTQA_ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192", launcher)
        self.assertIn("HOTPOTQA_ACTOR_PARAM_OFFLOAD:-false", launcher)
        self.assertIn("HOTPOTQA_ACTOR_OPTIMIZER_OFFLOAD:-false", launcher)
        self.assertIn('"--actor_use_dynamic_bsz", type=_parse_bool, default=True', preflight)
        self.assertIn('"--actor_max_token_len_per_gpu", type=int, default=8_192', preflight)
        self.assertIn('"--actor_param_offload", type=_parse_bool, default=False', preflight)
        self.assertIn('"--actor_optimizer_offload", type=_parse_bool, default=False', preflight)

    def test_all_training_arms_share_actor_loss_reference_kl(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")
        preflight = (PROJECT_ROOT / "recipes/hotpotqa/prepare_formal_rlvr_run.py").read_text(encoding="utf-8")
        parameters = inspect.signature(prepare_formal_rlvr_run).parameters

        self.assertIs(parameters["reference_kl_enabled"].default, True)
        self.assertEqual(parameters["reference_kl_loss_coef"].default, 0.001)
        self.assertEqual(parameters["reference_kl_loss_type"].default, "low_var_kl")
        self.assertIs(parameters["kl_in_reward"].default, False)
        self.assertIn("REFERENCE_KL_ENABLED=true", launcher)
        self.assertIn("REFERENCE_KL_LOSS_COEF=0.001", launcher)
        self.assertIn("REFERENCE_KL_LOSS_TYPE=low_var_kl", launcher)
        self.assertIn("KL_IN_REWARD=false", launcher)
        self.assertIn('actor_rollout_ref.actor.use_kl_loss="$REFERENCE_KL_ENABLED"', launcher)
        self.assertIn('actor_rollout_ref.actor.kl_loss_coef="$REFERENCE_KL_LOSS_COEF"', launcher)
        self.assertIn('actor_rollout_ref.actor.kl_loss_type="$REFERENCE_KL_LOSS_TYPE"', launcher)
        self.assertIn('algorithm.use_kl_in_reward="$KL_IN_REWARD"', launcher)
        self.assertIn('"placement": "actor_loss"', preflight)
        self.assertIn('"coefficient": reference_kl_loss_coef', preflight)

    def test_formal_surface_uses_grpo_names(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")
        preflight = (PROJECT_ROOT / "recipes/hotpotqa/prepare_formal_rlvr_run.py").read_text(encoding="utf-8")
        parameters = inspect.signature(prepare_formal_rlvr_run).parameters

        self.assertIn("grpo_micro_batch_size", parameters)
        self.assertNotIn("ppo_micro_batch_size", parameters)
        self.assertIn("HOTPOTQA_GRPO_MICRO_BATCH_SIZE", launcher)
        self.assertIn("agent_r1.trainer.main_agent_grpo", launcher)
        self.assertNotIn("HOTPOTQA_PPO_MICRO_BATCH_SIZE", launcher)
        self.assertIn('"grpo_mini_batch_size": train_batch_size', preflight)
        self.assertIn('"grpo_micro_batch_size_per_gpu": grpo_micro_batch_size', preflight)

    def test_launcher_supports_validated_resume(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")

        self.assertIn("HOTPOTQA_RESUME_MODE:-disable", launcher)
        self.assertIn("HOTPOTQA_RESUME_FROM_PATH", launcher)
        self.assertIn('trainer.resume_mode="$RESUME_MODE"', launcher)
        self.assertIn('trainer.resume_from_path="$RESUME_FROM_CONFIG"', launcher)

    def test_ray_sessions_use_project_storage_through_a_short_link(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")

        self.assertIn("HOTPOTQA_RAY_STORAGE_ROOT:-$WORKSPACE_DIR/.ray_tmp", launcher)
        self.assertIn("/tmp/ar1r-${UID}", launcher)
        self.assertIn('export RAY_TMPDIR="$RAY_TMP_LINK/', launcher)

    def test_algorithm_entrypoints_share_one_neutral_trainer_config(self):
        config_dir = PROJECT_ROOT / "agent_r1/config"

        self.assertTrue((config_dir / "agent_rl_trainer.yaml").is_file())
        self.assertFalse((config_dir / "agent_grpo_trainer.yaml").exists())
        self.assertFalse((config_dir / "agent_ppo_trainer.yaml").exists())

    def test_formal_launcher_and_manifest_support_a3_combined(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")
        a3_launcher = (PROJECT_ROOT / "examples/hotpotqa/run_a3.sh").read_text(encoding="utf-8")
        preflight = (PROJECT_ROOT / "recipes/hotpotqa/prepare_formal_rlvr_run.py").read_text(encoding="utf-8")

        self.assertIn("A1|A2|A3", launcher)
        self.assertIn("HOTPOTQA_REWARD_ARM=A3", a3_launcher)
        self.assertIn("RewardArm.A3", preflight)
        self.assertIn('"process_reward_weight": reward_contract.process_weight', preflight)
        self.assertIn('"terminal_reward_weight": reward_contract.terminal_weight', preflight)
        self.assertIn('"final_response_mask": reward_contract.final_response_mask', preflight)

    def test_a6_uses_one_external_judge_and_last_six_training_gpus(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_a6.sh").read_text(encoding="utf-8")
        shared_launcher = (PROJECT_ROOT / "examples/hotpotqa/run_with_judge.sh").read_text(encoding="utf-8")
        flow = (PROJECT_ROOT / "recipes/hotpotqa/hotpotqa_agent_flow.py").read_text(encoding="utf-8")

        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"', launcher)
        self.assertIn("HOTPOTQA_JUDGE_EXTERNAL=1", launcher)
        self.assertIn('exec bash "$HERE/run_with_judge.sh"', launcher)
        self.assertIn("vllm.entrypoints.openai.api_server", shared_launcher)
        self.assertIn("HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.40", launcher)
        self.assertIn("create_judge_from_env", flow)

    def test_a8_lr_has_standalone_source_and_previous_hyperparameters(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa_lr/run_lr.sh").read_text(encoding="utf-8")
        em100_dsl_launcher = (
            PROJECT_ROOT / "examples/hotpotqa_lr/run_lr_em100.sh"
        ).read_text(encoding="utf-8")
        claim_source_launcher = (
            PROJECT_ROOT / "examples/hotpotqa_lr/run_lr_claim_source.sh"
        ).read_text(encoding="utf-8")
        em100_launcher = (
            PROJECT_ROOT / "examples/hotpotqa_lr/run_lr_claim_source_em100.sh"
        ).read_text(encoding="utf-8")
        generic_launcher = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")
        preflight = (PROJECT_ROOT / "recipes/hotpotqa_lr/prepare_run.py").read_text(encoding="utf-8")

        self.assertFalse((PROJECT_ROOT / "examples/hotpotqa/run_a8.sh").exists())
        self.assertFalse((PROJECT_ROOT / "recipes/hotpotqa/answer_certificate_verifier.py").exists())
        self.assertFalse((PROJECT_ROOT / "recipes/hotpotqa/certificate_forest.py").exists())
        self.assertFalse((PROJECT_ROOT / "recipes/hotpotqa/task_sufficiency_verifier.py").exists())
        self.assertIn("HOTPOTQA_REWARD_ARM=A8_LR30", launcher)
        self.assertIn("HOTPOTQA_REWARD_ARM=A8_LR30", em100_dsl_launcher)
        self.assertIn("HOTPOTQA_LR_REASON_STEP_FORMAT=dsl", em100_dsl_launcher)
        self.assertIn("HOTPOTQA_LR_EM_WARMUP_STEPS", em100_dsl_launcher)
        self.assertIn("a8_lr30_em100", em100_dsl_launcher)
        self.assertIn('HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-6}"', em100_dsl_launcher)
        self.assertIn('HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-5}"', launcher)
        self.assertIn('HOTPOTQA_AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-5}"', launcher)
        self.assertIn(
            'HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.20}"',
            launcher,
        )
        self.assertIn('HOTPOTQA_ROLLOUT_N="${HOTPOTQA_ROLLOUT_N:-4}"', launcher)
        self.assertIn("HOTPOTQA_TOTAL_TRAINING_STEPS=1500", launcher)
        self.assertIn("HOTPOTQA_SAVE_FREQ=50", launcher)
        self.assertIn("A1|A2|A3|A6|A7|A9|A8_LR30", generic_launcher)
        self.assertIn("A8_LR30_CS", generic_launcher)
        self.assertIn("HOTPOTQA_REWARD_ARM=A8_LR30_CS", claim_source_launcher)
        self.assertIn("HOTPOTQA_LR_REASON_STEP_FORMAT=claim_source", claim_source_launcher)
        self.assertIn("HOTPOTQA_REWARD_ARM=A8_LR30_CS", em100_launcher)
        self.assertIn("HOTPOTQA_LR_EM_WARMUP_STEPS", em100_launcher)
        self.assertIn("claimsource_em100", em100_launcher)
        self.assertIn("--em-warmup-steps", generic_launcher)
        self.assertIn('"contract_id": LR_CONTRACT_VERSION', preflight)
        self.assertIn('f"{PRIMARY_CONTRACT.terminal_weight:.1f} * terminal_em + "', preflight)
        self.assertIn('"em_warmup_steps": args.em_warmup_steps', preflight)
        self.assertIn('"warmup_formula": "1.0 * terminal_em + 0.0 * local_reward"', preflight)
        self.assertIn('"reason_step_format": REASON_STEP_FORMAT', preflight)
        self.assertIn('"process_is_terminal_em_gated": False', preflight)
        self.assertIn('"gold_answer_visible_to_verifier": False', preflight)
        self.assertIn('"gold_evidence_visible_to_verifier": False', preflight)
        self.assertIn('"first_search_is_uncredited_bootstrap": True', preflight)

    def test_manifest_uses_one_reward_contract_and_hashes_the_actual_arm(self):
        preflight = (PROJECT_ROOT / "recipes/hotpotqa/prepare_formal_rlvr_run.py").read_text(encoding="utf-8")

        self.assertIn('CONTRACT_VERSION = "hotpotqa-formal-rlvr-grpo-v14"', preflight)
        self.assertIn(
            '"reward": _reward_description(reward_contract, arm=reward_arm)',
            preflight,
        )
        self.assertNotIn('"a2_final_reward"', preflight)
        self.assertNotIn('"a2_final_response_mask"', preflight)
        self.assertIn('f"examples/hotpotqa/run_{arm.value.lower()}.sh"', preflight)
        self.assertIn('"recipes/hotpotqa/prepare_formal_rlvr_run.py"', preflight)
        self.assertIn('"recipes/hotpotqa/validate_formal_a0_artifacts.py"', preflight)

        self.assertEqual(
            _reward_description(training_reward_contract(RewardArm.A1)),
            "terminal exact match",
        )
        self.assertEqual(
            _reward_description(training_reward_contract(RewardArm.A2)),
            "hotpotqa-new-evidence-v1",
        )
        self.assertEqual(
            _reward_description(training_reward_contract(RewardArm.A3)),
            "0.5 * hotpotqa-new-evidence-v1 + 0.5 * terminal exact match",
        )

    def test_a0_validation_sampling_and_gpu_count_are_explicitly_configurable(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_a0.sh").read_text(encoding="utf-8")
        evaluator = (PROJECT_ROOT.parent / "scripts/evaluate_hotpotqa_full_validation.sh").read_text(encoding="utf-8")

        self.assertIn("HOTPOTQA_NUM_GPUS", launcher)
        self.assertIn("HOTPOTQA_TRAIN_BATCH_SIZE", launcher)
        self.assertIn("(8 + NUM_GPUS - 1) / NUM_GPUS * NUM_GPUS", launcher)
        self.assertIn("HOTPOTQA_VAL_DO_SAMPLE:-false", launcher)
        self.assertIn("HOTPOTQA_VAL_TEMPERATURE:-0", launcher)
        self.assertIn('validation_do_sample": os.environ["VAL_DO_SAMPLE"]', launcher)
        self.assertIn('validation_temperature": float(os.environ["VAL_TEMPERATURE"])', launcher)
        self.assertIn("HOTPOTQA_VAL_N=1", evaluator)
        self.assertIn('HOTPOTQA_AGENT_WORKERS="${NUM_GPUS}"', evaluator)
        self.assertIn('HOTPOTQA_VAL_BATCH_SIZE="${HOTPOTQA_VAL_BATCH_SIZE:-8}"', evaluator)
        self.assertIn('HOTPOTQA_VLLM_MAX_NUM_SEQS="${HOTPOTQA_VLLM_MAX_NUM_SEQS:-8}"', evaluator)
        self.assertIn("HOTPOTQA_GPU_IDLE_MAX_USED_MIB:-100", evaluator)
        self.assertIn("--interface raw", evaluator)
        self.assertIn('HOTPOTQA_VALIDATION_INTERFACE="${VALIDATION_INTERFACE}"', evaluator)
        self.assertIn('"validation_interface": validation_interface', launcher)
        self.assertIn("VALIDATION_INTERFACE=raw", launcher)

    def test_lr_native_validation_uses_its_training_interface(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa_lr/run_validation.sh").read_text(encoding="utf-8")
        evaluator = (PROJECT_ROOT.parent / "scripts/evaluate_hotpotqa_full_validation.sh").read_text(encoding="utf-8")

        self.assertIn("HOTPOTQA_REWARD_ARM=A8_LR30", launcher)
        self.assertIn("HOTPOTQA_REWARD_ARM=A8_LR30_CS", launcher)
        self.assertIn("HOTPOTQA_LR_REASON_STEP_FORMAT", launcher)
        self.assertIn("recipes/hotpotqa_lr/base.yaml", launcher)
        self.assertIn("recipes/hotpotqa_lr/reward_fn.py", launcher)
        self.assertIn("default_agent_flow=hotpotqa_local_reasoning_agent", launcher)
        self.assertIn("HOTPOTQA_STREAMING_METRIC_KEYS=acc,terminal_em", launcher)
        self.assertIn("actor_rollout_ref.rollout.val_kwargs.n=1", launcher)
        self.assertIn("actor_rollout_ref.rollout.val_kwargs.do_sample=false", launcher)
        self.assertIn("actor_rollout_ref.rollout.val_kwargs.temperature=0", launcher)
        self.assertIn("actor_rollout_ref.rollout.load_format=auto", launcher)
        self.assertIn('"initial_weight_sync": "skipped_source_loaded"', launcher)
        self.assertIn('HOTPOTQA_VLLM_ENABLE_SLEEP_MODE:-false', launcher)
        self.assertIn('HOTPOTQA_VLLM_FREE_CACHE_ENGINE:-false', launcher)
        self.assertIn("--interface raw|lr", evaluator)
        self.assertIn("examples/hotpotqa_lr/run_validation.sh", evaluator)


if __name__ == "__main__":
    unittest.main()
