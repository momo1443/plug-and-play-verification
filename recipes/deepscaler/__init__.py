"""DeepScaleR recipes, including the multi-turn answer-checking agent."""

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop

# Importing this module registers the recipe-local tool before ToolEnv creates
# an instance from ``recipes/deepscaler/base.yaml``.
from . import tool as _tool  # noqa: F401

__all__ = ["AgentEnvLoop"]
