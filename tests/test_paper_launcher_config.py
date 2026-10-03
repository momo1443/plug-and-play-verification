"""Exercise actual shell argument propagation without starting data jobs or GPUs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = "[q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj]"


class PaperLauncherConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.model = self.root / "Qwen3.5-4B"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        for name in ("train.parquet", "validation.parquet", "test.parquet",
                     "taco_a9_sidecar.jsonl", "taco_a9_sidecar.index.json"):
            (self.root / name).touch()
        self.calls = self.root / "calls.jsonl"
        fake_python = self.root / "python-stub"
        fake_python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['LAUNCHER_TEST_CALLS'], 'a') as f:\n"
            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1] == '-': print(12000)\n"
        )
        fake_python.chmod(0o755)
        bwrap = self.root / "bwrap"
        bwrap.write_text("#!/bin/sh\nexit 0\n")
        bwrap.chmod(0o755)
        prefixes = ("AGENT_R1_", "HOTPOTQA_", "DEEPSCALER_", "TACO_A1_", "TACO_A9_", "VISION_R1_")
        self.env = {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}
        self.env.update({
            "PYTHON_BIN": str(fake_python), "LAUNCHER_TEST_CALLS": str(self.calls),
            "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
            "RUN_ID": "launcher-test", "RAY_TMPDIR": str(self.root / "ray"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "HOTPOTQA_OUTPUT_DIR": str(self.output), "HOTPOTQA_SKIP_PREFLIGHT": "1",
            # Nonempty optional arrays also allow testing on macOS Bash 3.2.
            "HOTPOTQA_HYDRA_CONFIG_ONLY": "1", "HOTPOTQA_VLLM_KV_CACHE_MEMORY_BYTES": "1024",
            "DEEPSCALER_TOOL_OUTPUT_DIR": str(self.output),
            "DEEPSCALER_TOOL_PAIR_DATA_DIR": str(self.root),
            "TACO_A9_OUTPUT_DIR": str(self.output), "TACO_A9_DATA_ROOT": str(self.root),
            "TACO_A1_OUTPUT_DIR": str(self.output), "TACO_A1_OUTPUT_ROOT": str(self.root),
            "TACO_A1_DATA_ROOT": str(self.root), "TACO_A1_SMOKE": "true",
            "VISION_R1_OUTPUT_DIR": str(self.output), "VISION_R1_RAW_DATA_ROOT": str(self.root),
            "VISION_R1_PREPARED_ROOT": str(self.root), "VISION_R1_MODEL_PATH": str(self.model),
        })

    def launch(self, relative, *, arguments=(), **overrides):
        env = dict(self.env, **overrides)
        self.calls.write_text("")
        result = subprocess.run(["bash", str(PROJECT_ROOT / relative), *arguments], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        trainer = [args for args in calls if len(args) > 1 and args[1] in {"agent_r1.trainer.main_agent_grpo", "agent_r1.trainer.main_agent_ppo"}][-1]
        options = {}
        for arg in trainer[2:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                self.assertNotIn(key, options, f"Duplicate trainer override: {key}")
                options[key] = value
        return options

    def assert_adaptation(self, options, rank):
        self.assertEqual(options["actor_rollout_ref.model.lora_rank"], str(rank))
        self.assertEqual(options["actor_rollout_ref.model.lora_alpha"], "64")
        self.assertEqual(options["actor_rollout_ref.model.target_modules"], TARGETS)

    def test_9b_wrappers_pass_lora_to_trainer_even_for_renamed_model(self):
        for launcher, key in (
            ("hotpotqa/run_a9_uniform_9b.sh", "HOTPOTQA_MODEL_PATH"),
            ("taco/run_a9_uniform_9b.sh", "TACO_A9_MODEL_PATH"),
        ):
            with self.subTest(launcher=launcher):
                options = self.launch("scripts/" + launcher, **{key: str(self.root / "renamed-checkpoint")})
                self.assert_adaptation(options, 64)

    def test_4b_domains_preserve_full_finetuning(self):
        for launcher in ("hotpotqa/run_a9_uniform.sh", "taco/run_a9_uniform.sh",
                         "vision_r1/run_a9_uniform.sh", "extensions/deepscaler/run_deepscaler_tool_a9.sh"):
            with self.subTest(launcher=launcher):
                self.assert_adaptation(self.launch("scripts/" + launcher), 0)

    def test_deepscaler_profiles_keep_distinct_lengths(self):
        for launcher, expected in (("deepscaler/run_deepscaler_a9_uniform.sh", ("2048", "4096")),
                                   ("extensions/deepscaler/run_deepscaler_tool_a9.sh", ("4096", "2048"))):
            with self.subTest(launcher=launcher):
                options = self.launch("scripts/" + launcher)
                self.assertEqual((options["data.max_prompt_length"], options["data.max_response_length"]), expected)
                self.assertEqual(options["actor_rollout_ref.rollout.prompt_length"],
                                 "8192" if launcher.startswith("deepscaler/") else expected[0])
                self.assertEqual(options["actor_rollout_ref.rollout.response_length"], expected[1])

    def test_toolenv_length_override_reaches_data_and_rollout(self):
        options = self.launch("scripts/extensions/deepscaler/run_deepscaler_tool_a9.sh",
                              DEEPSCALER_TOOL_MAX_PROMPT_LENGTH="3072",
                              DEEPSCALER_TOOL_MAX_RESPONSE_LENGTH="1024")
        self.assertEqual(options["data.max_prompt_length"], "3072")
        self.assertEqual(options["data.max_response_length"], "1024")
        self.assertEqual(options["actor_rollout_ref.rollout.prompt_length"], "3072")
        self.assertEqual(options["actor_rollout_ref.rollout.response_length"], "1024")

    def test_named_9b_model_enables_lora_in_toolenv_and_vision(self):
        model = self.root / "Qwen3.5-9B"
        model.mkdir()
        (model / "config.json").write_text("{}")
        for launcher, key in (("extensions/deepscaler/run_deepscaler_tool_a9.sh", "DEEPSCALER_TOOL_MODEL_PATH"),
                              ("vision_r1/run_a9_uniform.sh", "VISION_R1_MODEL_PATH")):
            with self.subTest(launcher=launcher):
                self.assert_adaptation(self.launch("scripts/" + launcher, **{key: str(model)}), 64)

    def test_explicit_lora_settings_and_full_finetuning_opt_out(self):
        options = self.launch("scripts/hotpotqa/run_a9_uniform_9b.sh", AGENT_R1_LORA_RANK="0")
        self.assert_adaptation(options, 0)
        options = self.launch("scripts/taco/run_a9_uniform_9b.sh", AGENT_R1_LORA_RANK="32",
                              AGENT_R1_LORA_ALPHA="16", AGENT_R1_LORA_TARGET_MODULES="[q_proj,v_proj]")
        self.assertEqual(options["actor_rollout_ref.model.lora_rank"], "32")
        self.assertEqual(options["actor_rollout_ref.model.lora_alpha"], "16")
        self.assertEqual(options["actor_rollout_ref.model.target_modules"], "[q_proj,v_proj]")

    def test_outcome_only_baselines_share_model_adaptation(self):
        model9b = self.root / "Qwen3.5-9B"
        model9b.mkdir()
        (model9b / "config.json").write_text("{}")
        for model, rank in ((self.model, 0), (model9b, 64)):
            for launcher, key in (
                ("hotpotqa/run_a1.sh", "HOTPOTQA_MODEL_PATH"),
                ("taco/run_a1_terminal.sh", "TACO_A1_MODEL_PATH"),
                ("vision_r1/run_a1_terminal.sh", "VISION_R1_MODEL_PATH"),
                ("extensions/deepscaler/run_deepscaler_tool_a1.sh", "DEEPSCALER_TOOL_MODEL_PATH"),
            ):
                with self.subTest(launcher=launcher, rank=rank):
                    self.assert_adaptation(self.launch("scripts/" + launcher, **{key: str(model)}), rank)

    def test_invalid_lora_rank_fails_before_trainer(self):
        result = subprocess.run(["bash", str(PROJECT_ROOT / "scripts/extensions/deepscaler/run_deepscaler_tool_a9.sh")],
                                env=dict(self.env, AGENT_R1_LORA_RANK="-1"), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("AGENT_R1_LORA_RANK", result.stderr)
        self.assertFalse(self.calls.exists())

class PaperBudgetIntegrationTest(unittest.TestCase):
    setUp = PaperLauncherConfigTest.setUp
    launch = PaperLauncherConfigTest.launch
    assert_adaptation = PaperLauncherConfigTest.assert_adaptation
    def test_canonical_training_profiles_have_paper_budget(self):
        for launcher, batch in (("deepscaler/run_deepscaler_a9_uniform.sh", "20"),
                                ("deepscaler/run_deepscaler_a1.sh", "20"),
                                ("hotpotqa/run_a1.sh", "20"),
                                ("hotpotqa/run_a9_uniform.sh", "20"),
                                ("hotpotqa/run_a9_uniform_9b.sh", "8"),
                                ("taco/run_a1_terminal.sh", "20"),
                                ("taco/run_a9_uniform.sh", "20"),
                                ("vision_r1/run_a1_terminal.sh", "20"),
                                ("vision_r1/run_a9_uniform.sh", "20")):
            with self.subTest(launcher=launcher):
                options = self.launch("scripts/" + launcher)
                self.assertEqual(options["data.train_max_samples"], "10000")
                self.assertEqual(options["trainer.total_training_steps"], "500")
                self.assertEqual(options["data.train_batch_size"], batch)
                self.assertEqual(options["actor_rollout_ref.rollout.n"], "4")

    def test_matched_hotpot_interfaces_for_all_reward_sources(self):
        for arm in ("A1", "A2", "A3", "A6", "A7", "A9"):
            options = self.launch("scripts/hotpotqa/run_rlvr.sh", HOTPOTQA_REWARD_ARM=arm)
            self.assertEqual(options["actor_rollout_ref.rollout.agent.default_agent_flow"], "hotpotqa_certificate_agent")
            self.assertTrue(options["custom_reward_function.path"].endswith("hotpotqa_a9/reward_fn.py"))

    def test_9b_math_uses_paper_completion_length_and_shared_flow(self):
        options = self.launch("scripts/deepscaler/run_deepscaler_a9_uniform.sh", AGENT_R1_MODEL_SCALE="9b")
        self.assertEqual(options["data.max_response_length"], "5120")
        self.assertEqual(options["actor_rollout_ref.rollout.response_length"], "5120")
        self.assertEqual(options["actor_rollout_ref.rollout.agent.default_agent_flow"], "deepscaler_paper_agent")
        self.assert_adaptation(options, 64)

    def test_every_domain_uses_batch_eight_for_9b_actor(self):
        model = self.root / "Qwen3.5-9B"
        model.mkdir()
        (model / "config.json").write_text("{}")
        for launcher, model_key in (
            ("deepscaler/run_deepscaler_a1.sh", "DEEPSCALER_MODEL_PATH"),
            ("deepscaler/run_deepscaler_a9_uniform.sh", "DEEPSCALER_MODEL_PATH"),
            ("hotpotqa/run_a1.sh", "HOTPOTQA_MODEL_PATH"),
            ("hotpotqa/run_a9_uniform.sh", "HOTPOTQA_MODEL_PATH"),
            ("taco/run_a1_terminal.sh", "TACO_A1_MODEL_PATH"),
            ("taco/run_a9_uniform.sh", "TACO_A9_MODEL_PATH"),
            ("taco/run_a9_uniform_9b.sh", "TACO_A9_MODEL_PATH"),
            ("vision_r1/run_a1_terminal.sh", "VISION_R1_MODEL_PATH"),
            ("vision_r1/run_a9_uniform.sh", "VISION_R1_MODEL_PATH"),
        ):
            with self.subTest(launcher=launcher):
                options = self.launch("scripts/" + launcher, **{model_key: str(model)})
                self.assertEqual(options["data.train_batch_size"], "8")
                self.assertEqual(options["actor_rollout_ref.actor.ppo_mini_batch_size"], "8")
                self.assertEqual(options["trainer.total_training_steps"], "500")
                self.assertEqual(options["actor_rollout_ref.rollout.n"], "4")
                self.assert_adaptation(options, 64)

    def test_9b_ppo_and_explicit_batch_override(self):
        options = self.launch("scripts/deepscaler/run_deepscaler_a1.sh",
                              AGENT_R1_MODEL_SCALE="9b", AGENT_R1_OPTIMIZER="ppo")
        self.assertEqual(options["data.train_batch_size"], "8")
        self.assertEqual(options["algorithm.adv_estimator"], "gae")
        self.assertEqual(options["actor_rollout_ref.rollout.agent.max_steps"], "5")
        options = self.launch("scripts/taco/run_a9_uniform.sh", AGENT_R1_MODEL_SCALE="9b",
                              TACO_A9_TRAIN_BATCH_SIZE="4")
        self.assertEqual(options["data.train_batch_size"], "4")

    def test_judge_batch_follows_actor_scale(self):
        for scale, batch in (("4b", "20"), ("9b", "8")):
            for domain in ("deepscaler", "hotpotqa", "taco", "vision"):
                with self.subTest(scale=scale, domain=domain):
                    options = self.launch("scripts/llm_judge/run_grpo_4b_judge_9b.sh",
                        arguments=(domain,), AGENT_R1_MODEL_SCALE=scale, AGENT_R1_JUDGE_API_KEY="test")
                    self.assertEqual(options["data.train_batch_size"], batch)

    def test_math_turn_override_and_single_turn_compatibility(self):
        for turns in ("1", "5", "8"):
            for launcher in ("deepscaler/run_deepscaler_a1.sh", "deepscaler/run_deepscaler_a9_uniform.sh"):
                with self.subTest(turns=turns, launcher=launcher):
                    options = self.launch("scripts/" + launcher, DEEPSCALER_MAX_STEPS=turns)
                    self.assertEqual(options["actor_rollout_ref.rollout.agent.max_steps"], turns)
                    self.assertEqual(options["actor_rollout_ref.rollout.multi_turn.enable"],
                                     "False" if turns == "1" else "True")
                    self.assertEqual(options["actor_rollout_ref.rollout.prompt_length"],
                                     "2048" if turns == "1" else "8192")
                    self.assertEqual(options["data.max_response_length"], "4096")

    def test_math_defaults_to_five_turns_in_both_training_arms(self):
        for launcher in ("deepscaler/run_deepscaler_a1.sh", "deepscaler/run_deepscaler_a9_uniform.sh"):
            options = self.launch("scripts/" + launcher)
            self.assertEqual(options["actor_rollout_ref.rollout.agent.max_steps"], "5")
            self.assertEqual(options["actor_rollout_ref.rollout.multi_turn.enable"], "True")
            self.assertEqual(options["data.max_prompt_length"], "2048")
            self.assertEqual(options["actor_rollout_ref.rollout.response_length"], "4096")

    def test_invalid_math_turn_count_fails_before_trainer(self):
        for turns in ("0", "-1", "4.5", "invalid"):
            result = subprocess.run(["bash", str(PROJECT_ROOT / "scripts/deepscaler/run_deepscaler_a9_uniform.sh")],
                                    env=dict(self.env, DEEPSCALER_MAX_STEPS=turns), capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("DEEPSCALER_MAX_STEPS", result.stderr)

    def test_ppo_baselines_enable_existing_critic_with_same_budget(self):
        for launcher in ("deepscaler/run_deepscaler_a1.sh", "taco/run_a1_terminal.sh", "vision_r1/run_a1_terminal.sh"):
            options = self.launch("scripts/" + launcher, AGENT_R1_OPTIMIZER="ppo")
            self.assertEqual(options["algorithm.adv_estimator"], "gae")
            self.assertEqual(options["critic.enable"], "True")
            self.assertEqual(options["trainer.total_training_steps"], "500")
            self.assertEqual(options["data.train_max_samples"], "10000")


if __name__ == "__main__":
    unittest.main()
