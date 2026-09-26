"""Vision-R1 multimodal recipe."""
__all__ = ["VisionR1VisualAgentFlow"]


def __getattr__(name):
    if name == "VisionR1VisualAgentFlow":
        from .agent_flow import VisionR1VisualAgentFlow
        return VisionR1VisualAgentFlow
    raise AttributeError(name)
