"""Paired terminal-only and process-mixed AgentFlows for DeepScaleR ToolEnv."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop
from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowStep, register
from agent_r1.env.base import Action
from agent_r1.verifier import apply_composed_reward
from recipes.deepscaler.trajectory_reward import TrajectoryReward, compute_tool_trajectory_reward
from recipes.llm_judge.scoring import ProcessStep, judge_reward_info, question_from_raw_prompt, verify_process_steps
from recipes.reward_mixing import prompt_group_key_from_extra_info
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _apply_trajectory_reward(
    steps: list[AgentFlowStep],
    trajectory_reward: TrajectoryReward,
    *,
    tool_call_count: int,
    max_steps: int,
) -> None:
    """Compatibility wrapper around the shared verifier credit adapter."""
    apply_composed_reward(
        steps,
        trajectory_reward.composed,
        extra_final_info={
            "terminal_em": trajectory_reward.terminal_em,
            "has_final_answer": trajectory_reward.has_final_answer,
            "process_segment_audits": trajectory_reward.process_segment_audits,
            "process_credit_application": "backfill_to_reasoning_turns",
            "tool_call_count": tool_call_count,
            "max_steps": max_steps,
        },
    )


@register("deepscaler_tool_a1")
@register("deepscaler_tool_a9")
@register("deepscaler_tool_llm_judge")
class DeepScalerToolRewardAgentFlow(AgentEnvLoop):
    """ToolEnv loop with terminal credit and causal process-credit backfill."""

    def __init__(self, *args, reward_mode: str, em_warmup_steps: int = 50, **kwargs) -> None:
        self.reward_mode = str(reward_mode)
        self.em_warmup_steps = int(em_warmup_steps)
        self.judge_server = kwargs.pop("shared_judge_server", None)
        if self.reward_mode == "llm_judge" and self.judge_server is None:
            raise ValueError("DeepScaleR llm_judge mode requires a shared Judge backend")
        super().__init__(*args, **kwargs)

    @staticmethod
    def _ground_truth(kwargs: dict[str, Any]) -> Any:
        reward_model = kwargs.get("reward_model") or {}
        if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") is None:
            raise ValueError("DeepScaleR ToolEnv requires reward_model.ground_truth")
        return reward_model["ground_truth"]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        global_step = int(kwargs.get("_agent_r1_global_step", -1))
        is_validation = bool(kwargs.get("_agent_r1_is_validation", False))
        prompt_group_key = None
        if self.reward_mode in {"uniform_equation_process", "llm_judge"} and not is_validation and global_step > self.em_warmup_steps:
            prompt_group_key = prompt_group_key_from_extra_info(
                kwargs.get("extra_info") or {},
                data_source=str(kwargs.get("data_source") or "deepmath"),
            )
        env = self._create_env(**kwargs)
        observation = env.reset(**kwargs)
        tools = getattr(env, "tool_schemas", None)
        steps = []
        reasoning_segments: list[tuple[int, str]] = []
        judge_steps: list[ProcessStep] = []
        metrics: dict[str, Any] = {}
        tool_call_count = 0
        final_response: str | None = None

        for turn in range(1, self.max_steps + 1):
            prompt_ids = await self._obs_to_prompt(observation, tools=tools)
            if len(prompt_ids) > self.prompt_length:
                logger.warning(
                    "DeepScaleR ToolEnv prompt exceeds limit at turn %d; marking trajectory incomplete.", turn
                )
                break

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
            response_ids = output.token_ids[: self.response_length]
            if not response_ids:
                raise RuntimeError("DeepScaleR ToolEnv received an empty generation")
            response_text = await self.loop.run_in_executor(
                None,
                lambda ids=response_ids: self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens),
            )

            # Retain model reasoning, but never score tool-call XML or observations.
            content, tool_calls = env.format_wrapper.parse_response(response_text)
            if content.strip():
                reasoning_segments.append((turn, content.strip()))
            tool_call_count += len(tool_calls)

            with simple_timer("tool_calls", metrics):
                next_observation, tool_reward, done, _ = await env.step(
                    Action(text=response_text, token_ids=response_ids)
                )
            if tool_reward is not None:
                raise RuntimeError("DeepScaleR answer-check tool must not emit a direct training reward")

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=(output.log_probs[: self.response_length] if output.log_probs else None),
                reward_score=0.0,
                num_turns=turn,
                extra_fields={
                    "deepscaler_tool_turn": turn,
                    "deepscaler_tool_call_count": len(tool_calls),
                    "deepscaler_tool_final": done,
                },
            )
            steps.append(await self._postprocess(step, **kwargs))
            if done:
                final_response = response_text
                break
            if self.reward_mode == "llm_judge" and not is_validation and global_step > self.em_warmup_steps:
                judge_steps.append(ProcessStep(
                    step_index=turn,
                    action=response_text,
                    observation=next_observation.messages[-1].get("content") if next_observation.messages else None,
                    valid=bool(content.strip() or tool_calls),
                ))
            observation = next_observation

        if not steps:
            raise RuntimeError("DeepScaleR ToolEnv produced no agent steps")

        verification = None
        if self.reward_mode == "llm_judge" and not is_validation and global_step > self.em_warmup_steps:
            verification = await verify_process_steps(
                self.judge_server, domain="deepmath",
                question=question_from_raw_prompt(kwargs.get("raw_prompt")),
                steps=judge_steps, max_steps=self.max_steps,
            )
        trajectory_reward = compute_tool_trajectory_reward(
            reward_mode=self.reward_mode,
            final_response=final_response,
            ground_truth=self._ground_truth(kwargs),
            reasoning_segments=reasoning_segments,
            global_step=global_step,
            is_validation=is_validation,
            em_warmup_steps=self.em_warmup_steps,
            prompt_group_key=prompt_group_key,
            process_verification=verification,
        )
        _apply_trajectory_reward(
            steps, trajectory_reward, tool_call_count=tool_call_count, max_steps=self.max_steps,
        )
        if self.reward_mode == "llm_judge":
            info = steps[-1].extra_fields.setdefault("reward_extra_info", {})
            info.update({"acc": trajectory_reward.terminal_em, "reward_mode": "llm_judge",
                         "judge_invalid": False})
            if verification is not None:
                info.update(judge_reward_info(verification))
        return AgentFlowOutput(steps=steps, metrics=metrics)
