"""GRPO-named entrypoint for formal critic-free Agent-R1 training."""

import hydra

from agent_r1.trainer.main_agent_rl import run_agent_rl
from agent_r1.trainer.training_outputs import configure_training_outputs
from verl.utils.device import auto_set_device


@hydra.main(config_path="../config", config_name="agent_rl_trainer", version_base=None)
def main(config):
    """Launch GRPO and reject configurations selecting another estimator."""
    if str(config.algorithm.adv_estimator).strip().lower() != "grpo":
        raise ValueError(
            "agent_r1.trainer.main_agent_grpo requires algorithm.adv_estimator=grpo"
        )
    auto_set_device(config)
    configure_training_outputs(config)
    run_agent_rl(config)


if __name__ == "__main__":
    main()
