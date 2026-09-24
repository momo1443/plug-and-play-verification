"""Multimodal crop-and-submit AgentFlow for Vision-R1-rl."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.env.tool_format import ToolFormatWrapper
from agent_r1.verifier import (
    UniformRewardSchedule,
    VerificationResult,
    apply_composed_reward,
    compose_verification_reward,
)
from recipes.llm_judge.scoring import question_from_raw_prompt, score_candidate
from recipes.vision_r1.artifacts import VisualArtifact
from recipes.vision_r1.prompts import (
    FINAL_TURN_PROMPT,
    INSPECT_OR_SUBMIT_TOOLS,
    SUBMIT_ONLY_TOOLS,
    SYSTEM_PROMPT,
)
from recipes.vision_r1.reward_contract import outcome_reward, reward_schedule
from recipes.vision_r1.verifier import verify_process
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
_HERMES_TOOL_FORMAT = ToolFormatWrapper.from_name("hermes")


def _one_tool_call(text: str) -> tuple[str | None, dict[str, Any]]:
    _, calls = _HERMES_TOOL_FORMAT.parse_response(text)
    if len(calls) != 1:
        return None, {}
    name = calls[0].name
    arguments = calls[0].arguments
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None, {}
    return name, arguments


def _ground_truth(kwargs: Mapping[str, Any]) -> Any:
    reward_model = kwargs.get("reward_model") or {}
    if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") is None:
        raise ValueError("Vision-R1 requires reward_model.ground_truth")
    return reward_model["ground_truth"]


def _group_key(kwargs: Mapping[str, Any]) -> str:
    extra_info = kwargs.get("extra_info") or {}
    if isinstance(extra_info, Mapping):
        for key in ("question_id", "index"):
            value = extra_info.get(key)
            if value is not None and str(value).strip():
                return f"vision_r1_rl:{value}"
    raise ValueError("Vision-R1 requires a stable question_id or index")


def _artifact_feedback(artifact: VisualArtifact) -> str:
    record = artifact.record()
    return (
        "Visual artifact created and shown above. "
        f"artifact_id={record['artifact_id']}; sha256={record['sha256']}; "
        f"size={record['width']}x{record['height']}; "
        f"parent_artifact_id={record['parent_artifact_id']}; bbox_2d={record['bbox_2d']}."
    )


@register("vision_r1_visual_agent")
@register("vision_r1_llm_judge_visual_agent")
class VisionR1VisualAgentFlow(AgentFlowBase):
    """Interleave image crops with reasoning and verify cited crops by replay."""

    def __init__(
        self,
        *args,
        reward_mode: str,
        terminal_warmup_steps: int = 50,
        max_steps: int = 3,
        **kwargs,
    ) -> None:
        self.judge_server = kwargs.pop("shared_judge_server", None)
        super().__init__(*args, **kwargs)
        self.reward_mode = str(reward_mode)
        self.terminal_warmup_steps = int(os.environ.get("VISION_R1_TERMINAL_WARMUP_STEPS", terminal_warmup_steps))
        self.max_steps = int(max_steps)
        self.prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        self.response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        if self.reward_mode == "llm_judge" and self.judge_server is None:
            raise ValueError("Vision-R1 llm_judge mode requires a shared Judge backend")
        if self.reward_mode not in {"terminal_only", "uniform_visual_certificate", "llm_judge"}:
            raise ValueError(f"Unsupported Vision-R1 reward mode: {self.reward_mode}")
        if self.max_steps < 2:
            raise ValueError("Vision-R1 visual agent requires at least two turns")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        initial_mm = await self.process_vision_info(raw_prompt)
        original_images = initial_mm.get("images") or []
        if len(original_images) != 1:
            raise ValueError(f"Vision-R1 visual agent requires exactly one image, got {len(original_images)}")

        root = VisualArtifact.root(original_images[0])
        artifacts: dict[str, VisualArtifact] = {root.artifact_id: root}
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    root_artifact_id=root.artifact_id,
                    width=root.width,
                    height=root.height,
                ),
            },
            *(message for message in raw_prompt if message.get("role") != "system"),
        ]
        global_step = int(kwargs.get("_agent_r1_global_step", -1))
        is_validation = bool(kwargs.get("_agent_r1_is_validation", False))
        prompt_group_key = _group_key(kwargs)
        schedule = (
            UniformRewardSchedule(1.0, 0.0, "llm_judge_terminal", "fixed", prompt_group_key)
            if self.reward_mode == "llm_judge"
            else reward_schedule(
                reward_mode=self.reward_mode,
                global_step=global_step,
                is_validation=is_validation,
                terminal_warmup_steps=self.terminal_warmup_steps,
                prompt_group_key=prompt_group_key,
            )
        )
        steps: list[AgentFlowStep] = []
        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "step_generate_sequences": [],
            "step_tool_calls": [],
        }
        submitted_answer: str | None = None
        raw_certificate: Any = None
        terminal_applied = False

        for turn in range(1, self.max_steps + 1):
            if turn == self.max_steps:
                messages.append({"role": "user", "content": FINAL_TURN_PROMPT})
            tools = SUBMIT_ONLY_TOOLS if turn == self.max_steps else INSPECT_OR_SUBMIT_TOOLS
            multi_modal_data = await self.process_vision_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            prompt_ids = await self.apply_chat_template(
                messages,
                tools=tools,
                images=images,
                videos=videos,
            )
            if len(prompt_ids) > self.prompt_length:
                logger.warning("Vision-R1 prompt exceeded limit at turn %d", turn)
                break

            step_metrics: dict[str, float] = {}
            with simple_timer("generate_sequences", step_metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                )
            response_ids = list(output.token_ids[: self.response_length])
            if not response_ids:
                raise RuntimeError("Vision-R1 visual agent received an empty generation")
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            action_name, arguments = _one_tool_call(response_text)
            step_kind = action_name or "invalid_tool_call"
            feedback: str | None = None
            feedback_image = None

            with simple_timer("tool_calls", step_metrics):
                if action_name == "inspect_region" and turn < self.max_steps:
                    source_id = arguments.get("source_artifact_id")
                    purpose = arguments.get("purpose")
                    parent = root if source_id == root.artifact_id else None
                    if parent is None:
                        feedback = "Invalid inspect_region: source_artifact_id must identify the root image."
                        step_kind = "invalid_inspect_region"
                    elif not isinstance(purpose, str) or not purpose.strip():
                        feedback = "Invalid inspect_region: purpose must be a non-empty string."
                        step_kind = "invalid_inspect_region"
                    else:
                        try:
                            artifact = VisualArtifact.crop(
                                ordinal=len(artifacts),
                                parent=parent,
                                bbox_2d=arguments.get("bbox_2d"),
                                created_step=turn,
                            )
                        except ValueError as error:
                            feedback = f"Invalid inspect_region: {error}."
                            step_kind = "invalid_inspect_region"
                        else:
                            artifacts[artifact.artifact_id] = artifact
                            feedback = _artifact_feedback(artifact)
                            feedback_image = artifact.image
                elif action_name == "submit_answer":
                    answer = arguments.get("answer")
                    if isinstance(answer, str) and answer.strip():
                        submitted_answer = answer.strip()
                        raw_certificate = arguments.get("certificate")
                        step_kind = "submit_answer"
                    else:
                        feedback = "Invalid submit_answer: answer must be a non-empty string."
                        step_kind = "invalid_submit_answer"
                else:
                    feedback = "Output exactly one permitted tool call for this turn."

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                multi_modal_data=multi_modal_data,
                reward_score=0.0,
                num_turns=turn,
                extra_fields={
                    "vision_r1_turn": turn,
                    "step_kind": step_kind,
                    "visual_artifacts": [artifact.record() for artifact in artifacts.values()],
                    "reward_extra_info": {
                        "optimizer_reward_phase": schedule.phase,
                        "optimizer_terminal_weight": schedule.terminal_weight,
                        "optimizer_process_weight": schedule.process_weight,
                        "weight_sampling": schedule.weight_sampling,
                        "optimizer_weight_group_key": schedule.prompt_group_key,
                        "training_global_step": global_step,
                    },
                },
            )
            steps.append(await self._postprocess(step, **kwargs))
            metrics["generate_sequences"] += step_metrics.get("generate_sequences", 0.0)
            metrics["tool_calls"] += step_metrics.get("tool_calls", 0.0)
            metrics["step_generate_sequences"].append(step_metrics.get("generate_sequences", 0.0))
            metrics["step_tool_calls"].append(step_metrics.get("tool_calls", 0.0))

            if submitted_answer is not None:
                terminal_score = outcome_reward(submitted_answer, _ground_truth(kwargs))
                judge_result = None
                judge_invalid = False
                optimizer_terminal_score = terminal_score
                if self.reward_mode == "llm_judge" and not is_validation:
                    judge_result = await score_candidate(
                        self.judge_server,
                        domain="vision_r1",
                        question=question_from_raw_prompt(raw_prompt),
                        candidate=submitted_answer,
                        reference=_ground_truth(kwargs),
                        auxiliary={"certificate": raw_certificate},
                    )
                    judge_invalid = judge_result is None
                    optimizer_terminal_score = 0.0 if judge_invalid else judge_result.score
                verification = (
                    verify_process(
                        artifacts=artifacts,
                        raw_certificate=raw_certificate,
                        submitted_answer=submitted_answer,
                    )
                    if self.reward_mode == "uniform_visual_certificate"
                    else VerificationResult(
                        credits=(),
                        audit={"verifier": "disabled_terminal_only"},
                    )
                )
                composed = compose_verification_reward(
                    terminal_reward=optimizer_terminal_score,
                    verification=verification,
                    schedule=schedule,
                )
                apply_composed_reward(
                    steps,
                    composed,
                    extra_final_info={
                        "acc": terminal_score,
                        "terminal_exact_math_match": terminal_score,
                        "verified_process_reward": verification.process_reward,
                        "reward_mode": self.reward_mode,
                        "verifier_timing": "terminal_visual_replay",
                        "llm_judge_score": None if judge_result is None else judge_result.score,
                        "llm_judge_input_hash": None if judge_result is None else judge_result.input_hash,
                        "llm_judge_cache_hit": None if judge_result is None else judge_result.cache_hit,
                        "llm_judge_attempts": None if judge_result is None else judge_result.attempts,
                        "llm_judge_latency_s": None if judge_result is None else judge_result.latency_s,
                        "judge_invalid": judge_invalid,
                        "optimizer_total_reward": optimizer_terminal_score,
                    },
                )
                terminal_applied = True
                break

            messages.append({"role": "assistant", "content": response_text})
            if feedback_image is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": feedback_image},
                            {"type": "text", "text": feedback or ""},
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": feedback or "Invalid action."})

        if not steps:
            raise RuntimeError("Vision-R1 visual agent produced no steps")
        if not terminal_applied:
            composed = compose_verification_reward(
                terminal_reward=0.0,
                verification=VerificationResult(
                    credits=(),
                    audit={"verifier": "not_run_without_valid_submission"},
                ),
                schedule=schedule,
            )
            apply_composed_reward(
                steps,
                composed,
                extra_final_info={
                    "acc": 0.0,
                    "terminal_exact_math_match": 0.0,
                    "verified_process_reward": 0.0,
                    "reward_mode": self.reward_mode,
                    "verifier_timing": "not_run_without_valid_submission",
                    "llm_judge_score": None,
                    "judge_invalid": False,
                    "optimizer_total_reward": 0.0,
                },
            )
        return AgentFlowOutput(steps=steps, metrics=metrics)
