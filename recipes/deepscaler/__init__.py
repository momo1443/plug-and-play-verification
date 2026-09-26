"""DeepScaleR recipes; numerical verification can be imported without the GPU runtime."""

__all__ = ["AgentEnvLoop"]


def __getattr__(name):
    if name == "AgentEnvLoop":
        from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop
        from . import tool  # Register the ToolEnv extension's answer checker.
        return AgentEnvLoop
    raise AttributeError(name)
