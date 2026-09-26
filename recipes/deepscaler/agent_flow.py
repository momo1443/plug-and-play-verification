"""Paper single-turn math flow, shared by terminal/verifier/frozen-judge arms."""
from __future__ import annotations
from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.evaluation.answers import final_answer_record
from agent_r1.evaluation.consistency import consistency_record
from agent_r1.verifier import VerificationResult, apply_composed_reward
from recipes.deepscaler.trajectory_reward import compute_tool_trajectory_reward, verify_process
from recipes.llm_judge.scoring import ProcessStep, judge_reward_info, question_from_raw_prompt, verify_process_steps
from recipes.reward_mixing import prompt_group_key_from_extra_info
from verl.utils.profiler import simple_timer


@register("deepscaler_paper_agent")
class DeepScaleRPaperAgentFlow(AgentFlowBase):
    def __init__(self, *args, reward_mode="uniform_equation_process", em_warmup_steps=50, **kwargs):
        self.judge_server = kwargs.pop("shared_judge_server", None)
        super().__init__(*args, **kwargs)
        if reward_mode not in {"terminal_only", "uniform_equation_process", "llm_judge"}:
            raise ValueError(f"Unsupported math reward mode: {reward_mode}")
        if reward_mode == "llm_judge" and self.judge_server is None:
            raise ValueError("Math llm_judge requires a shared frozen Judge backend")
        self.reward_mode = reward_mode
        self.em_warmup_steps = int(em_warmup_steps)
        self.response_length = int(self.config.actor_rollout_ref.rollout.response_length)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params,
            )
        response_ids = list(output.token_ids[:self.response_length])
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        final = final_answer_record(text)
        reasoning = str(final["reasoning"]) if final else text
        deterministic = verify_process([(1, reasoning)])
        global_step = int(kwargs.get("_agent_r1_global_step", 0))
        validation = bool(kwargs.get("_agent_r1_is_validation", False))
        judge_verification = VerificationResult(credits=(), audit={})
        if self.reward_mode == "llm_judge" and not validation and global_step > self.em_warmup_steps:
            judge_verification = await verify_process_steps(
                self.judge_server, domain="deepmath", question=question_from_raw_prompt(messages),
                steps=[ProcessStep(1, reasoning)] if reasoning.strip() else [], max_steps=1,
            )
        reward = compute_tool_trajectory_reward(
            reward_mode=self.reward_mode, final_response=text if final else None,
            ground_truth=kwargs["reward_model"]["ground_truth"],
            reasoning_segments=[(1, reasoning)], global_step=global_step,
            is_validation=validation, em_warmup_steps=self.em_warmup_steps,
            prompt_group_key=prompt_group_key_from_extra_info(kwargs.get("extra_info") or {}, data_source="deepmath"),
            process_verification=judge_verification if self.reward_mode == "llm_judge" else deterministic,
        )
        step = AgentFlowStep(
            prompt_ids=prompt_ids, response_ids=response_ids,
            response_logprobs=output.log_probs[:self.response_length] if output.log_probs else None,
            routed_experts=(output.routed_experts[:len(prompt_ids) + self.response_length]
                            if getattr(output, "routed_experts", None) is not None else None),
            reward_score=0.0, extra_fields={"step_kind": "final", "raw_prompt": messages},
        )
        step = await self._postprocess(step, **kwargs)
        info = {
            **reward.record(), **consistency_record(deterministic, reward.terminal_em, eligible=final is not None),
            "acc": reward.terminal_em, "reward_mode": self.reward_mode,
            "deterministic_verification": deterministic.audit,
            "training_global_step": global_step,
        }
        if self.reward_mode == "llm_judge":
            info.update(judge_reward_info(judge_verification))
        apply_composed_reward([step], reward.composed, extra_final_info=info)
        return AgentFlowOutput(steps=[step], metrics=metrics)
