import unittest

from omegaconf import OmegaConf

from agent_r1.trainer.ppo.ray_trainer import should_sleep_rollout_replicas, should_sync_initial_rollout_weights


class InitialRolloutSyncTest(unittest.TestCase):
    @staticmethod
    def make_config(*, val_only: bool, load_format: str, resume_mode: str = "disable"):
        return OmegaConf.create(
            {
                "trainer": {"val_only": val_only, "resume_mode": resume_mode},
                "actor_rollout_ref": {"rollout": {"load_format": load_format}},
            }
        )

    def test_source_loaded_validation_skips_redundant_sync(self):
        config = self.make_config(val_only=True, load_format="auto")

        self.assertFalse(should_sync_initial_rollout_weights(config))

    def test_training_always_syncs_actor_weights(self):
        config = self.make_config(val_only=False, load_format="auto")

        self.assertTrue(should_sync_initial_rollout_weights(config))

    def test_dummy_loaded_validation_requires_sync(self):
        config = self.make_config(val_only=True, load_format="dummy")

        self.assertTrue(should_sync_initial_rollout_weights(config))

    def test_resumed_validation_requires_sync(self):
        config = self.make_config(val_only=True, load_format="auto", resume_mode="resume_path")

        self.assertTrue(should_sync_initial_rollout_weights(config))


class RolloutSleepModeTest(unittest.TestCase):
    @staticmethod
    def make_config(enable_sleep_mode):
        return OmegaConf.create({"actor_rollout_ref": {"rollout": {"enable_sleep_mode": enable_sleep_mode}}})

    def test_disabled_sleep_mode_skips_sleep_call(self):
        self.assertFalse(should_sleep_rollout_replicas(self.make_config(False)))

    def test_enabled_sleep_mode_allows_sleep_call(self):
        self.assertTrue(should_sleep_rollout_replicas(self.make_config(True)))


if __name__ == "__main__":
    unittest.main()
