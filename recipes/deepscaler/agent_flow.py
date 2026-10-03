"""Bounded math reasoning turns, shared by outcome/verifier/frozen-judge arms."""
from __future__ import annotations
from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.evaluation.answers import final_answer_record
from agent_r1.evaluation.consistency import consistency_record
from agent_r1.verifier import VerificationResult, apply_composed_reward
from recipes.deepscaler.prompts import build_math_messages, math_continuation_message
from recipes.deepscaler.trajectory_reward import compute_tool_trajectory_reward, verify_process
from recipes.llm_judge.scoring import ProcessStep, judge_reward_info, question_from_raw_prompt, verify_process_steps
from recipes.reward_mixing import prompt_group_key_from_extra_info
from verl.utils.profiler import simple_timer


@register("deepscaler_paper_agent")
class DeepScaleRPaperAgentFlow(AgentFlowBase):
    def __init__(self, *args, reward_mode="uniform_equation_process", em_warmup_steps=50,
                 max_steps=5, **kwargs):
        self.judge_server = kwargs.pop("shared_judge_server", None)
        super().__init__(*args, **kwargs)
        if reward_mode not in {"terminal_only", "uniform_equation_process", "llm_judge"}:
            raise ValueError(f"Unsupported math reward mode: {reward_mode}")
        if reward_mode == "llm_judge" and self.judge_server is None:
            raise ValueError("Math llm_judge requires a shared frozen Judge backend")
        self.reward_mode = reward_mode
        self.em_warmup_steps = int(em_warmup_steps)
        self.max_steps = int(max_steps)
        if isinstance(max_steps, bool) or str(self.max_steps) != str(max_steps) or self.max_steps < 1:
            raise ValueError("Math max_steps must be a positive integer")
        rollout = self.config.actor_rollout_ref.rollout
        self.response_length = int(rollout.response_length)
        if self.response_length < 1:
            raise ValueError("Math response token budget must be positive")
        self.prompt_length = int(rollout.prompt_length)
        self.initial_prompt_length = int(self.dataset_config.max_prompt_length)
        self.context_length = int(rollout.max_model_len or (self.prompt_length + self.response_length))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = kwargs["raw_prompt"]
        initial_ids = await self.apply_chat_template(list(raw_prompt))
        if len(initial_ids) > self.initial_prompt_length:
            raise ValueError("Math initial prompt exceeds the configured input limit")
        messages = build_math_messages(raw_prompt, self.max_steps)
        metrics = {"generate_sequences": 0.0, "step_generate_sequences": []}
        steps = []
        reasoning_segments = []
        final_response = None
        response_tokens = 0
        termination_reason = "max_steps"
        turn_response_length = (self.response_length + self.max_steps - 1) // self.max_steps

        for turn in range(1, self.max_steps + 1):
            prompt_ids = initial_ids if self.max_steps == 1 else await self.apply_chat_template(messages)
            if len(prompt_ids) > self.prompt_length:
                termination_reason = "prompt_limit"
                break
            remaining = self.response_length - response_tokens
            token_limit = min(remaining, turn_response_length, self.context_length - len(prompt_ids),
                              int(sampling_params.get("max_tokens", self.response_length)))
            if token_limit <= 0:
                termination_reason = "token_budget" if remaining <= 0 else "context_limit"
                break
            step_params = dict(sampling_params, max_tokens=token_limit)
            step_metrics = {}
            with simple_timer("generate_sequences", step_metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=step_params,
                )
            duration = step_metrics.get("generate_sequences", 0.0)
            metrics["generate_sequences"] += duration
            response_ids = list(output.token_ids[:token_limit])
            if not response_ids:
                termination_reason = "empty_generation"
                break
            metrics["step_generate_sequences"].append(duration)
            response_tokens += len(response_ids)
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            final = final_answer_record(text)
            reasoning_segments.append((turn, str(final["reasoning"]) if final else text))
            step = AgentFlowStep(
                prompt_ids=prompt_ids, response_ids=response_ids,
                response_logprobs=output.log_probs[:len(response_ids)] if output.log_probs else None,
                routed_experts=(output.routed_experts[:len(prompt_ids) + len(response_ids)]
                                if getattr(output, "routed_experts", None) is not None else None),
                reward_score=0.0, num_turns=turn,
                extra_fields={"step_kind": "final" if final else "reasoning", "math_turn": turn},
            )
            steps.append(await self._postprocess(step, **kwargs))
            if final is not None:
                final_response = text
                termination_reason = "final_answer"
                break
            messages.append({"role": "assistant", "content": text})
            messages.append(math_continuation_message(final_turn=turn + 1 == self.max_steps))

        if not steps:
            raise RuntimeError(f"Math trajectory produced no policy steps ({termination_reason})")
        deterministic = verify_process(reasoning_segments)
        global_step = int(kwargs.get("_agent_r1_global_step", 0))
        validation = bool(kwargs.get("_agent_r1_is_validation", False))
        judge_verification = VerificationResult(credits=(), audit={})
        if self.reward_mode == "llm_judge" and not validation and global_step > self.em_warmup_steps:
            judge_verification = await verify_process_steps(
                self.judge_server, domain="deepmath", question=question_from_raw_prompt(raw_prompt),
                steps=[ProcessStep(turn, reasoning) for turn, reasoning in reasoning_segments if reasoning.strip()],
                max_steps=self.max_steps,
            )
        reward = compute_tool_trajectory_reward(
            reward_mode=self.reward_mode, final_response=final_response,
            ground_truth=kwargs["reward_model"]["ground_truth"],
            reasoning_segments=reasoning_segments, global_step=global_step,
            is_validation=validation, em_warmup_steps=self.em_warmup_steps,
            prompt_group_key=prompt_group_key_from_extra_info(kwargs.get("extra_info") or {}, data_source="deepmath"),
            process_verification=judge_verification if self.reward_mode == "llm_judge" else deterministic,
        )
        info = {
            **reward.record(), **consistency_record(deterministic, reward.terminal_em,
                                                    eligible=final_response is not None),
            "acc": reward.terminal_em, "reward_mode": self.reward_mode,
            "deterministic_verification": deterministic.audit,
            "training_global_step": global_step, "max_steps": self.max_steps,
            "policy_step_count": len(steps), "generated_response_tokens": response_tokens,
            "total_response_token_budget": self.response_length, "termination_reason": termination_reason,
            "process_credit_application": "backfill_to_reasoning_turns",
            "verification_timing": "after_rollout",
            "observation_source": "fixed_continuation_prompt_without_verification_feedback",
        }
        if self.reward_mode == "llm_judge":
            info.update(judge_reward_info(judge_verification))
        apply_composed_reward(steps, reward.composed, extra_final_info=info)
        return AgentFlowOutput(steps=steps, metrics=metrics)
