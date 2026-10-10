"""Bounded math reasoning turns, shared by outcome/verifier/frozen-judge arms."""
from __future__ import annotations
from typing import Any
import re
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.evaluation.answers import final_answer_record
from agent_r1.evaluation.consistency import consistency_record
from agent_r1.verifier import VerificationResult, apply_composed_reward
from recipes.deepscaler.prompts import build_math_messages, math_continuation_message
from recipes.deepscaler.submission import FINAL_ANSWER_PREFIX, generate_prefilled_submission, prefilled_final_record
from recipes.deepscaler.trajectory_reward import compute_tool_trajectory_reward, verify_process
from recipes.llm_judge.scoring import ProcessStep, judge_reward_info, question_from_raw_prompt, verify_process_steps
from recipes.reward_mixing import prompt_group_key_from_extra_info
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids


@register("deepscaler_paper_agent")
class DeepScaleRPaperAgentFlow(AgentFlowBase):
    def __init__(self, *args, reward_mode="uniform_equation_process", em_warmup_steps=50,
                 max_steps=5, final_turn_submission=True, **kwargs):
        self.judge_server = kwargs.pop("shared_judge_server", None)
        super().__init__(*args, **kwargs)
        if reward_mode not in {"terminal_only", "uniform_equation_process", "llm_judge"}:
            raise ValueError(f"Unsupported math reward mode: {reward_mode}")
        if reward_mode == "llm_judge" and self.judge_server is None:
            raise ValueError("Math llm_judge requires a shared frozen Judge backend")
        self.reward_mode = reward_mode
        self.em_warmup_steps = int(em_warmup_steps)
        self.max_steps = int(max_steps)
        if str(final_turn_submission).lower() not in {"true", "false"}:
            raise ValueError("Math final_turn_submission must be a boolean")
        self.final_turn_submission = str(final_turn_submission).lower() == "true"
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
        initial_ids = normalize_token_ids(await self.apply_chat_template(list(raw_prompt)))
        if len(initial_ids) > self.initial_prompt_length:
            raise ValueError("Math initial prompt exceeds the configured input limit")
        messages = build_math_messages(raw_prompt, self.max_steps)
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0,
                   "step_generate_sequences": [], "step_tool_calls": []}
        steps = []
        reasoning_segments = []
        truncated_steps = []
        final_response = None
        selected_answer = None
        response_tokens = 0
        prefill_step = None
        submission_requests = 0
        discarded_tokens = 0
        termination_reason = "max_steps"
        turn_response_length = (self.response_length + self.max_steps - 1) // self.max_steps

        for turn in range(1, self.max_steps + 1):
            submission_turn = (getattr(self, "final_turn_submission", True)
                               and self.max_steps > 1 and turn == self.max_steps)
            if submission_turn:
                # Each trajectory owns its flow instance. Replace its options
                # temporarily; never mutate the shared dataset configuration.
                previous_options = getattr(self, "apply_chat_template_kwargs", {})
                self.apply_chat_template_kwargs = {**previous_options, "enable_thinking": False}
                try:
                    prompt_ids = normalize_token_ids(await self.apply_chat_template(messages))
                finally:
                    self.apply_chat_template_kwargs = previous_options
                prompt_ids = list(prompt_ids) + self.tokenizer.encode(FINAL_ANSWER_PREFIX, add_special_tokens=False)
                prefill_step = turn
            else:
                prompt_ids = initial_ids if self.max_steps == 1 else normalize_token_ids(await self.apply_chat_template(messages))
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
                if submission_turn:
                    output = await generate_prefilled_submission(
                        self.server_manager, self.tokenizer, prompt_ids=prompt_ids,
                        sampling_params=step_params, token_limit=token_limit,
                    )
                    submission_requests = output.generation_requests
                    discarded_tokens = output.discarded_tokens
                else:
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
            metrics["step_tool_calls"].append(0.0)
            response_tokens += getattr(output, "generated_tokens", len(response_ids))
            truncated = (len(response_ids) == token_limit
                         and response_ids[-1] != getattr(self.tokenizer, "eos_token_id", None))
            if truncated:
                truncated_steps.append(turn)
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            try:
                prompt_tail = self.tokenizer.decode(prompt_ids[-32:], skip_special_tokens=False)
            except (KeyError, IndexError):
                prompt_tail = ""  # Minimal tokenizers used by CPU flow tests.
            starts_in_thinking = bool(re.search(r"<think>\s*$", prompt_tail))
            final = (prefilled_final_record(text) if submission_turn else
                     final_answer_record(text, require_think_close=starts_in_thinking))
            reasoning_segments.append((turn, "" if submission_turn else str(final["reasoning"]) if final else text))
            step = AgentFlowStep(
                prompt_ids=prompt_ids, response_ids=response_ids,
                response_logprobs=output.log_probs[:len(response_ids)] if output.log_probs else None,
                routed_experts=(output.routed_experts[:len(prompt_ids) + len(response_ids)]
                                if getattr(output, "routed_experts", None) is not None else None),
                reward_score=0.0, num_turns=turn,
                extra_fields={"step_kind": "final" if final else "reasoning", "math_turn": turn,
                              "math_generation_mode": "submission" if submission_turn else "reasoning"},
            )
            steps.append(await self._postprocess(step, **kwargs))
            if final is not None:
                # The parser has found a complete submission; the selected
                # reasoning prefix ends there, rather than at the length cap.
                if truncated and truncated_steps[-1] == turn:
                    truncated_steps.pop()
                final_response = FINAL_ANSWER_PREFIX + text if submission_turn else text
                selected_answer = final["answer"]
                termination_reason = "final_answer"
                break
            messages.append({"role": "assistant", "content": text})
            messages.append(math_continuation_message(
                final_turn=turn + 1 == self.max_steps,
                submission_only=getattr(self, "final_turn_submission", True),
                max_output_tokens=min(turn_response_length, self.response_length - response_tokens),
                prepare_submission=(getattr(self, "final_turn_submission", True)
                                    and turn + 1 == self.max_steps - 1),
            ))

        if not steps:
            raise RuntimeError(f"Math trajectory produced no policy steps ({termination_reason})")
        deterministic = verify_process(reasoning_segments, question=question_from_raw_prompt(raw_prompt),
                                       truncated_steps=truncated_steps)
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
            selected_answer=selected_answer,
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
            "deterministic_process_reward": sum(credit.score for credit in deterministic.credits),
            "verification_passed_check_count": sum(bool(check) for check in deterministic.audit["applicable_checks"]),
            "verification_failed_check_count": sum(not bool(check) for check in deterministic.audit["applicable_checks"]),
            "training_global_step": global_step, "max_steps": self.max_steps,
            "policy_step_count": len(steps), "generated_response_tokens": response_tokens,
            "total_response_token_budget": self.response_length, "termination_reason": termination_reason,
            "process_credit_application": "backfill_to_reasoning_turns",
            "verification_timing": "after_rollout",
            "final_turn_submission": getattr(self, "final_turn_submission", True),
            "final_answer_prefill": FINAL_ANSWER_PREFIX if prefill_step else "",
            "final_answer_prefill_step": prefill_step,
            "final_submission_generation_requests": submission_requests,
            "discarded_after_final_answer_tokens": discarded_tokens,
            "policy_response_tokens": response_tokens - discarded_tokens,
            "truncated_policy_steps": truncated_steps,
            "observation_source": "fixed_continuation_prompt_without_verification_feedback",
        }
        if self.reward_mode == "llm_judge":
            info.update(judge_reward_info(judge_verification))
        apply_composed_reward(steps, reward.composed, extra_final_info=info)
        return AgentFlowOutput(steps=steps, metrics=metrics)
