"""Prompts for math continuation and the separate answer-checking extension."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


MATH_CONTINUATION_INSTRUCTION = (
    "Solve the problem across successive reasoning turns if needed. Each turn may "
    "advance the existing derivation without restarting it. Recheck a calculation "
    "only if you identify a specific inconsistency. Put a final answer in \\boxed{...} "
    "only when ready to submit; this ends the trajectory. Before submission, do "
    "not use an answer tag, a final-answer statement, or a bare numeric final line. "
    "No correctness feedback or reference answer will be provided."
)


def build_math_messages(raw_prompt: Iterable[Mapping[str, Any]], max_steps: int) -> list[dict]:
    """Keep the original problem and apply identical instructions to every arm."""
    messages = [dict(message) for message in raw_prompt]
    if max_steps > 1:
        messages.append({"role": "user", "content": MATH_CONTINUATION_INSTRUCTION})
    return messages


def math_continuation_message(*, final_turn: bool, submission_only: bool = False,
                              max_output_tokens: int | None = None,
                              prepare_submission: bool = False) -> dict[str, str]:
    if final_turn and submission_only:
        budget = (f" You have at most {max_output_tokens} output tokens and no further turns."
                  if max_output_tokens is not None else " No further turns are available.")
        return {"role": "user", "content": (
            "This is the last available turn, reserved for final submission. Use your previous work and submit now. "
            "Respond ONLY with your final answer in \\boxed{...}; do not explain or restart the derivation."
            + budget + " If uncertain, submit your best candidate answer. "
            "The assistant response already starts with \\boxed{; continue with the answer "
            "and close the box. Do not repeat the opening prefix."
        )}
    instruction = (
        "This is the last available reasoning turn. Use the existing derivation and "
        "submit the final answer in \\boxed{...}."
        if final_turn else
        "Continue from the progress already made and resolve the remaining steps. "
        "Do not restart the solution or repeat completed calculations without a specific inconsistency. "
        "If the solution is complete, submit the final answer in \\boxed{...} now."
    )
    if prepare_submission:
        instruction += (" The next turn is reserved for final submission. "
                        "Finish the current approach and settle on your best candidate answer in this turn.")
    return {"role": "user", "content": instruction}


DEEPSCALER_TOOL_SYSTEM_PROMPT = """You are a mathematical problem solver. Reason carefully and put your final answer in \\boxed{...}.
Before your final response, call check_deepscaler_answer at least once with only your candidate final answer. The tool reports whether that candidate is correct, but never reveals the answer. If it reports an error, recheck your derivation and try another candidate. To call it, emit exactly:
<tool_call>
<function=check_deepscaler_answer>
<parameter=answer>
your candidate
</parameter>
</function>
</tool_call>
After tool feedback, answer the original problem normally and include one final \\boxed{...}."""


def build_agent_messages(raw_prompt: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Preserve the problem while replacing its generic system instruction."""
    messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in raw_prompt
        if str(message.get("role", "")) != "system"
    ]
    return [{"role": "system", "content": DEEPSCALER_TOOL_SYSTEM_PROMPT}, *messages]
