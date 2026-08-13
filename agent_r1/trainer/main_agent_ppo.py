"""PPO-compatible entrypoint retained for PPO and StepPO recipes."""

import hydra

from agent_r1.trainer.main_agent_rl import run_agent_rl
from agent_r1.trainer.training_outputs import configure_training_outputs
from verl.utils.device import auto_set_device


@hydra.main(config_path="../config", config_name="agent_rl_trainer", version_base=None)
def main(config):
    """Launch the estimator selected by a PPO-compatible recipe."""
    auto_set_device(config)
    configure_training_outputs(config)
    run_agent_rl(config)


if __name__ == "__main__":
    main()
