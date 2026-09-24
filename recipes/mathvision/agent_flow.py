"""Multimodal ReAct AgentFlow used by the MATH-Vision A0 evaluation."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop
from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowStep, register
from agent_r1.env.base import Action
from verl.utils.profiler import simple_timer

# Importing the existing tool module registers the hidden-answer checker used
# by the ReAct environment. The checker returns only correctness feedback.
from recipes.deepscaler import tool as _deepscaler_tool  # noqa: F401


@register("mathvision_react")
class MathVisionReactAgentFlow(AgentEnvLoop):
    """Run a multimodal conversation with Hermes tool calls and image payloads."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        env = self._create_env(**kwargs)
        observation = env.reset(**kwargs)
        tools = getattr(env, "tool_schemas", None)
        steps: list[AgentFlowStep] = []
        metrics: dict[str, Any] = {}

        for turn in range(1, self.max_steps + 1):
            if observation.messages is not None:
                multi_modal_data = await self.process_vision_info(observation.messages)
                images = multi_modal_data.get("images")
                videos = multi_modal_data.get("videos")
                prompt_ids = await self.apply_chat_template(
                    observation.messages,
                    tools=tools,
                    images=images,
                    videos=videos,
                )
            else:
                multi_modal_data = {}
                images = videos = None
                prompt_ids = await self._obs_to_prompt(observation, tools=tools)

            if len(prompt_ids) > self.prompt_length:
                break

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                )

            response_ids = list(output.token_ids[: self.response_length])
            response_text = await self.loop.run_in_executor(
                None,
                lambda ids=response_ids: self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens),
            )

            with simple_timer("tool_calls", metrics):
                next_observation, _, done, _ = await env.step(
                    Action(text=response_text, token_ids=response_ids)
                )

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=(output.log_probs[: self.response_length] if output.log_probs else None),
                multi_modal_data=multi_modal_data,
                reward_score=0.0,
                num_turns=turn,
                extra_fields={"mathvision_react_turn": turn, "mathvision_react_done": done},
            )
            steps.append(await self._postprocess(step, **kwargs))
            if done:
                break
            observation = next_observation

        if not steps:
            raise RuntimeError("MATH-Vision ReAct produced no agent steps")
        return AgentFlowOutput(steps=steps, metrics=metrics)
