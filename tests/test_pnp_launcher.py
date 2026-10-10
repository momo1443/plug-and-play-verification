"""Check PNP dispatch, model profiles and forwarding through real shell entrypoints."""

import subprocess
import unittest

import test_paper_launcher_config as launcher_harness

PROJECT_ROOT = launcher_harness.PROJECT_ROOT


class PnpLauncherTest(unittest.TestCase):
    setUp = launcher_harness.PaperLauncherConfigTest.setUp
    launch = launcher_harness.PaperLauncherConfigTest.launch
    assert_adaptation = launcher_harness.PaperLauncherConfigTest.assert_adaptation

    def test_four_gpu_math_profile_matches_current_experiment(self):
        options = self.launch(
            "run_pnp.sh", arguments=("deepscaler", "--model", "Qwen3.5-4B", "--gpus", "0,1,2,3"),
            DEEPSCALER_OUTPUT_DIR=str(self.output), DEEPSCALER_TRAIN_PATH=str(self.root / "train.parquet"),
            DEEPSCALER_VAL_PATH=str(self.root / "validation.parquet"),
        )
        expected = {
            "trainer.n_gpus_per_node": "4", "trainer.total_training_steps": "500",
            "trainer.resume_mode": "disable", "trainer.test_freq": "-1", "trainer.save_freq": "10",
            "data.train_batch_size": "20", "data.train_max_samples": "10000",
            "data.max_prompt_length": "2048", "data.max_response_length": "4096",
            "actor_rollout_ref.rollout.n": "4", "actor_rollout_ref.rollout.max_model_len": "8192",
            "actor_rollout_ref.rollout.gpu_memory_utilization": "0.30",
            "actor_rollout_ref.rollout.max_num_seqs": "32", "actor_rollout_ref.rollout.agent.num_workers": "8",
            "actor_rollout_ref.rollout.agent.max_steps": "5", "actor_rollout_ref.actor.optim.lr": "1e-6",
            "actor_rollout_ref.actor.kl_loss_coef": "0.001", "ray_kwargs.ray_init.num_cpus": "16",
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(options[key], value)
        self.assert_adaptation(options, 0)
        self.assertTrue(options["actor_rollout_ref.rollout.agent.agent_flow_config_path"].endswith("pnp_deepscaler.yaml"))

    def test_dispatch_all_domains_and_named_model_path(self):
        for domain in ("deepscaler", "hotpotqa", "taco", "vision_r1"):
            for model, rank in (("Qwen3.5-4B", 0), ("Qwen3.5-9B", 64)):
                with self.subTest(domain=domain, model=model):
                    options = self.launch(
                        "run_pnp.sh", arguments=(domain, "--model", model, "--model-path", str(self.model)),
                        DEEPSCALER_OUTPUT_DIR=str(self.output), DEEPSCALER_TRAIN_PATH=str(self.root / "train.parquet"),
                        DEEPSCALER_VAL_PATH=str(self.root / "validation.parquet"),
                    )
                    self.assertEqual(options["actor_rollout_ref.model.path"], str(self.model))
                    self.assert_adaptation(options, rank)
                    self.assertEqual(options["data.train_batch_size"], "20" if rank == 0 else "8")
                    self.assertEqual(options["trainer.total_training_steps"], "500")

    def test_hotpot_gpu_selection_and_environment_settings_are_forwarded(self):
        options = self.launch("run_pnp.sh", arguments=(
            "hotpotqa", "--gpus", "0,1,2,3",
        ), HOTPOTQA_TOTAL_TRAINING_STEPS="80")
        self.assertEqual(options["trainer.n_gpus_per_node"], "4")
        self.assertEqual(options["trainer.total_training_steps"], "80")

    def test_math_hydra_argument_is_forwarded_without_shell_evaluation(self):
        key = "++ray_kwargs.ray_init._system_config.object_spilling_threshold"
        options = self.launch("run_pnp.sh", arguments=("deepscaler", "--", key + "=0.7"),
                              DEEPSCALER_OUTPUT_DIR=str(self.output),
                              DEEPSCALER_TRAIN_PATH=str(self.root / "train.parquet"),
                              DEEPSCALER_VAL_PATH=str(self.root / "validation.parquet"))
        self.assertEqual(options[key], "0.7")

    def test_bad_options_fail_before_any_python_work(self):
        for arguments in (("unknown",), ("deepscaler", "--model", "unknown"),
                          ("deepscaler", "--gpus", "0;echo"), ("deepscaler", "--model")):
            with self.subTest(arguments=arguments):
                self.calls.write_text("")
                result = subprocess.run(["bash", str(PROJECT_ROOT / "run_pnp.sh"), *arguments],
                                        env=self.env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self.calls.read_text(), "")


if __name__ == "__main__":
    unittest.main()
