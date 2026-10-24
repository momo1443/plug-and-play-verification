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


class PnpConfigTest(unittest.TestCase):
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
        prefixes = ("PNP_", "AGENT_R1_", "HOTPOTQA_", "DEEPSCALER_", "TACO_A1_", "TACO_A9_", "VISION_R1_")
        self.env = {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}
        self.env.update({
            "PYTHON_BIN": str(fake_python), "LAUNCHER_TEST_CALLS": str(self.calls),
            "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
            "RUN_ID": "launcher-test", "RAY_TMPDIR": str(self.root / "ray"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "HOTPOTQA_OUTPUT_DIR": str(self.output), "HOTPOTQA_SKIP_PREFLIGHT": "1",
            # Nonempty optional arrays also allow testing on macOS Bash 3.2.
            "HOTPOTQA_HYDRA_CONFIG_ONLY": "1", "HOTPOTQA_VLLM_KV_CACHE_MEMORY_BYTES": "1024",
            "DEEPSCALER_OUTPUT_DIR": str(self.output),
            "DEEPSCALER_TRAIN_PATH": str(self.root / "train.parquet"),
            "DEEPSCALER_VAL_PATH": str(self.root / "validation.parquet"),
            "TACO_A9_OUTPUT_DIR": str(self.output), "TACO_A9_DATA_ROOT": str(self.root),
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
        for domain, key in (
            ("hotpotqa", "HOTPOTQA_MODEL_PATH"),
            ("taco", "TACO_A9_MODEL_PATH"),
        ):
            with self.subTest(domain=domain):
                options = self.launch("scripts/models/Qwen3.5-9B.sh", arguments=(domain,),
                                      **{key: str(self.root / "renamed-checkpoint")})
                self.assert_adaptation(options, 64)

    def test_4b_domains_preserve_full_finetuning(self):
        for launcher in ("hotpotqa/pnp_hotpotqa.sh", "taco/pnp_taco.sh",
                         "vision_r1/pnp_vision_r1.sh", "deepscaler/pnp_deepscaler.sh"):
            with self.subTest(launcher=launcher):
                self.assert_adaptation(self.launch("scripts/" + launcher), 0)

    def test_deepscaler_model_profiles_keep_their_completion_budgets(self):
        for scale, expected_response in (("4b", "10240"), ("9b", "10240")):
            with self.subTest(scale=scale):
                options = self.launch("scripts/deepscaler/pnp_deepscaler.sh", AGENT_R1_MODEL_SCALE=scale)
                self.assertEqual(options["data.max_prompt_length"], "2048")
                self.assertEqual(options["data.max_response_length"], expected_response)
                self.assertEqual(options["actor_rollout_ref.rollout.prompt_length"], "13312")
                self.assertEqual(options["actor_rollout_ref.rollout.response_length"], expected_response)


    def test_named_9b_model_enables_lora_in_math_and_vision(self):
        model = self.root / "Qwen3.5-9B"
        model.mkdir()
        (model / "config.json").write_text("{}")
        for launcher, key in (("deepscaler/pnp_deepscaler.sh", "DEEPSCALER_MODEL_PATH"),
                              ("vision_r1/pnp_vision_r1.sh", "VISION_R1_MODEL_PATH")):
            with self.subTest(launcher=launcher):
                self.assert_adaptation(self.launch("scripts/" + launcher, **{key: str(model)}), 64)

    def test_explicit_lora_settings_and_full_finetuning_opt_out(self):
        options = self.launch("scripts/models/Qwen3.5-9B.sh", arguments=("hotpotqa",), AGENT_R1_LORA_RANK="0")
        self.assert_adaptation(options, 0)
        options = self.launch("scripts/models/Qwen3.5-9B.sh", arguments=("taco",), AGENT_R1_LORA_RANK="32",
                              AGENT_R1_LORA_ALPHA="16", AGENT_R1_LORA_TARGET_MODULES="[q_proj,v_proj]")
        self.assertEqual(options["actor_rollout_ref.model.lora_rank"], "32")
        self.assertEqual(options["actor_rollout_ref.model.lora_alpha"], "16")
        self.assertEqual(options["actor_rollout_ref.model.target_modules"], "[q_proj,v_proj]")


    def test_invalid_lora_rank_fails_before_trainer(self):
        result = subprocess.run(["bash", str(PROJECT_ROOT / "scripts/deepscaler/pnp_deepscaler.sh")],
                                env=dict(self.env, AGENT_R1_LORA_RANK="-1"), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("AGENT_R1_LORA_RANK", result.stderr)
        self.assertFalse(self.calls.exists())

class PnpBudgetIntegrationTest(unittest.TestCase):
    setUp = PnpConfigTest.setUp
    launch = PnpConfigTest.launch
    assert_adaptation = PnpConfigTest.assert_adaptation

    def test_hotpot_a6_reserves_one_of_eight_gpus_for_judge(self):
        options = self.launch("scripts/hotpotqa/run_a6.sh", CUDA_VISIBLE_DEVICES="")
        self.assertEqual(options["trainer.n_gpus_per_node"], "7")
        self.assertEqual(options["trainer.total_training_steps"], "500")
        self.assertEqual(options["actor_rollout_ref.rollout.agent.default_agent_flow"], "hotpotqa_certificate_agent")

    def test_hotpot_a6_rejects_shared_actor_and_judge_gpu(self):
        result = subprocess.run(["bash", str(PROJECT_ROOT / "scripts/hotpotqa/run_a6.sh")],
                                env=dict(self.env, HOTPOTQA_JUDGE_GPU="7"), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be separate", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_math_smoke_repeats_small_dataset_to_reach_500_updates(self):
        options = self.launch("scripts/deepscaler/pnp_deepscaler_smoke.sh", CUDA_VISIBLE_DEVICES="")
        self.assertEqual(options["trainer.n_gpus_per_node"], "8")
        self.assertEqual(options["trainer.total_training_steps"], "500")
        self.assertEqual(options["data.train_max_samples"], "4")
        self.assertEqual(options["data.train_batch_size"], "4")
        self.assertEqual(options["trainer.total_epochs"], "500")

    def test_canonical_training_profiles_have_paper_budget(self):
        for launcher, batch in (("deepscaler/pnp_deepscaler.sh", "20"),
                                ("hotpotqa/pnp_hotpotqa.sh", "20"),
                                ("taco/pnp_taco.sh", "20"),
                                ("vision_r1/pnp_vision_r1.sh", "20")):
            with self.subTest(launcher=launcher):
                options = self.launch("scripts/" + launcher)
                self.assertEqual(options["data.train_max_samples"], "10000")
                self.assertEqual(options["trainer.total_training_steps"], "500")
                self.assertEqual(options["data.train_batch_size"], batch)
                self.assertEqual(options["actor_rollout_ref.rollout.n"], "4")
        options = self.launch("scripts/models/Qwen3.5-9B.sh", arguments=("hotpotqa",))
        self.assertEqual(options["data.train_batch_size"], "8")
        self.assertEqual(options["trainer.total_training_steps"], "500")


    def test_9b_math_uses_paper_completion_length_and_shared_flow(self):
        options = self.launch("scripts/deepscaler/pnp_deepscaler.sh", AGENT_R1_MODEL_SCALE="9b")
        self.assertEqual(options["data.max_response_length"], "10240")
        self.assertEqual(options["actor_rollout_ref.rollout.response_length"], "10240")
        self.assertEqual(options["actor_rollout_ref.rollout.agent.default_agent_flow"], "deepscaler_paper_agent")
        self.assert_adaptation(options, 64)

    def test_every_domain_uses_batch_eight_for_9b_actor(self):
        model = self.root / "Qwen3.5-9B"
        model.mkdir()
        (model / "config.json").write_text("{}")
        for launcher, model_key in (
            ("deepscaler/pnp_deepscaler.sh", "DEEPSCALER_MODEL_PATH"),
            ("hotpotqa/pnp_hotpotqa.sh", "HOTPOTQA_MODEL_PATH"),
            ("taco/pnp_taco.sh", "TACO_A9_MODEL_PATH"),
            ("vision_r1/pnp_vision_r1.sh", "VISION_R1_MODEL_PATH"),
        ):
            with self.subTest(launcher=launcher):
                options = self.launch("scripts/" + launcher, **{model_key: str(model)})
                self.assertEqual(options["data.train_batch_size"], "8")
                self.assertEqual(options["actor_rollout_ref.actor.ppo_mini_batch_size"], "8")
                self.assertEqual(options["trainer.total_training_steps"], "500")
                self.assertEqual(options["actor_rollout_ref.rollout.n"], "4")
                self.assert_adaptation(options, 64)

    def test_9b_ppo_and_explicit_batch_override(self):
        options = self.launch("scripts/deepscaler/pnp_deepscaler.sh",
                              AGENT_R1_MODEL_SCALE="9b", AGENT_R1_OPTIMIZER="ppo")
        self.assertEqual(options["data.train_batch_size"], "8")
        self.assertEqual(options["algorithm.adv_estimator"], "gae")
        self.assertEqual(options["actor_rollout_ref.rollout.agent.max_steps"], "5")
        options = self.launch("scripts/taco/pnp_taco.sh", AGENT_R1_MODEL_SCALE="9b",
                              TACO_A9_TRAIN_BATCH_SIZE="4")
        self.assertEqual(options["data.train_batch_size"], "4")


    def test_math_turn_override_and_single_turn_compatibility(self):
        for turns in ("1", "5", "8"):
            for launcher in ("deepscaler/pnp_deepscaler.sh",):
                with self.subTest(turns=turns, launcher=launcher):
                    options = self.launch("scripts/" + launcher, DEEPSCALER_MAX_STEPS=turns)
                    self.assertEqual(options["actor_rollout_ref.rollout.agent.max_steps"], turns)
                    self.assertEqual(options["actor_rollout_ref.rollout.multi_turn.enable"],
                                     "False" if turns == "1" else "True")
                    self.assertEqual(options["actor_rollout_ref.rollout.prompt_length"],
                                     "2048" if turns == "1" else "13312")
                    self.assertEqual(options["data.max_response_length"], "10240")

    def test_math_defaults_to_five_turns(self):
        for launcher in ("deepscaler/pnp_deepscaler.sh",):
            options = self.launch("scripts/" + launcher)
            self.assertEqual(options["actor_rollout_ref.rollout.agent.max_steps"], "5")
            self.assertEqual(options["actor_rollout_ref.rollout.multi_turn.enable"], "True")
            self.assertEqual(options["data.max_prompt_length"], "2048")
            self.assertEqual(options["actor_rollout_ref.rollout.response_length"], "10240")

    def test_invalid_math_turn_count_fails_before_trainer(self):
        for turns in ("0", "-1", "4.5", "invalid"):
            result = subprocess.run(["bash", str(PROJECT_ROOT / "scripts/deepscaler/pnp_deepscaler.sh")],
                                    env=dict(self.env, DEEPSCALER_MAX_STEPS=turns), capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("DEEPSCALER_MAX_STEPS", result.stderr)

    def test_pnp_ppo_enables_critic_with_same_budget(self):
        for launcher in ("deepscaler/pnp_deepscaler.sh", "taco/pnp_taco.sh", "vision_r1/pnp_vision_r1.sh"):
            options = self.launch("scripts/" + launcher, AGENT_R1_OPTIMIZER="ppo")
            self.assertEqual(options["algorithm.adv_estimator"], "gae")
            self.assertEqual(options["critic.enable"], "True")
            self.assertEqual(options["trainer.total_training_steps"], "500")
            self.assertEqual(options["data.train_max_samples"], "10000")


if __name__ == "__main__":
    unittest.main()
