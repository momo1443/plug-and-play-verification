"""A hidden-answer checker used by the DeepScaleR ToolEnv recipe."""

from __future__ import annotations

from typing import Any

from agent_r1.tool import BaseTool, ToolResponse
from verl.utils.reward_score import math_reward


def _candidate_solution(answer: Any) -> str:
    candidate = str(answer).strip()
    if "\\boxed" in candidate:
        return candidate
    return f"\\boxed{{{candidate}}}"


@BaseTool.register("check_deepscaler_answer")
class DeepScalerAnswerCheckTool(BaseTool):
    """Check a proposed answer without exposing the hidden reference answer."""

    name = "check_deepscaler_answer"
    description = "Checks whether a proposed final mathematical answer is correct."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Your proposed final answer, without explanation.",
            }
        },
        "required": ["answer"],
    }

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        tools_kwargs = kwargs.get("tools_kwargs") or {}
        ground_truth = tools_kwargs.get("ground_truth")
        if ground_truth is None:
            raise ValueError("ground_truth is required for check_deepscaler_answer")

        candidate = _candidate_solution(args.get("answer", ""))
        is_correct = bool(math_reward.compute_score(candidate, str(ground_truth)) > 0)
        response = "Candidate answer is correct." if is_correct else "Candidate answer is not correct. Recheck the derivation."
        return ToolResponse(text=response), None, {"candidate_correct": is_correct}
