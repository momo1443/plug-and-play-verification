import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from agent_r1.trainer.training_outputs import configure_training_outputs


class TrainingOutputsTest(unittest.TestCase):
    def make_config(self, default_local_dir: Path, *, val_only: bool = False):
        return OmegaConf.create(
            {
                "trainer": {
                    "logger": ["console"],
                    "experiment_name": "a2-named-run",
                    "val_only": val_only,
                    "default_local_dir": str(default_local_dir),
                    "rollout_data_dir": None,
                    "rollout_data_file": None,
                    "tensorboard_dir": None,
                }
            }
        )

    def test_training_enables_tensorboard_and_single_rollout_file(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            run_dir = Path(temp_dir) / "run"
            config = self.make_config(run_dir / "checkpoints")

            configure_training_outputs(config)

            self.assertEqual(list(config.trainer.logger), ["console", "tensorboard"])
            self.assertEqual(
                config.trainer.tensorboard_dir,
                str(run_dir / "tensorboard" / "a2-named-run"),
            )
            self.assertEqual(config.trainer.rollout_data_file, str(run_dir / "rollouts.jsonl"))
            self.assertEqual(
                os.environ["TENSORBOARD_DIR"],
                str(run_dir / "tensorboard" / "a2-named-run"),
            )

    def test_legacy_rollout_directory_resolves_to_one_named_file(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            root = Path(temp_dir)
            config = self.make_config(root / "checkpoints")
            config.trainer.rollout_data_dir = str(root / "rollouts")

            configure_training_outputs(config)

            self.assertEqual(config.trainer.rollout_data_file, str(root / "rollouts" / "rollouts.jsonl"))

    def test_validation_only_does_not_enable_training_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ, {}, clear=True):
            config = self.make_config(Path(temp_dir) / "checkpoints", val_only=True)

            configure_training_outputs(config)

            self.assertEqual(list(config.trainer.logger), ["console"])
            self.assertIsNone(config.trainer.rollout_data_file)
            self.assertNotIn("TENSORBOARD_DIR", os.environ)


if __name__ == "__main__":
    unittest.main()
