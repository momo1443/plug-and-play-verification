"""Shared HotpotQA AgentFlow for formal reward-arm experiments.

Every arm uses this exact execution path. A0 is validation-only; A1 consumes
terminal EM; A2 consumes process reward with final tokens masked; A3 consumes
the frozen 0.5/0.5 combination while keeping final tokens trainable; A6
replaces the deterministic process verifier with an LLM semantic judge
(Qwen3-4B) that evaluates which gold supporting facts are semantically
covered by accumulated passages, then applies the same |new_llm_covered| /
|G_i| formula as A3; A7 replaces the process verifier with a weak execution
check that rewards each model-generated search producing a non-empty
observation at 1/3 per step, while keeping the same 0.5/0.5 combined-reward
    contract and terminal EM.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any, Sequence
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.reward_loop.reward_loop import RewardLoopWorker
from recipes.hotpotqa.a9_behavioral_verifier import (
    CandidateScore,
    ProbePlan,
    build_probe_plan,
    encode_action_candidate,
    extract_candidate_score,
    invalid_verification,
    verify_behavioral_scores,
    visible_probe_inputs_valid,
)
from recipes.hotpotqa.env.search_tool import (
    DEFAULT_HOTPOTQA_EMBEDDING_MODEL,
    HotpotQASearchToolLegacy,
    Passage,
    parse_legacy_tool_result,
    resolve_hotpotqa_embedding_devices,
)
from recipes.hotpotqa.evidence import EVIDENCE_SCHEMA_VERSION, coerce_bool
from recipes.hotpotqa.final_answer_protocol import resolve_final_answer_protocol
from recipes.hotpotqa.judge_prompts import JUDGE_USER_PROMPT, format_gold_facts_for_prompt
from recipes.hotpotqa.judge_server import (
    JudgeServerManager,
    RemoteJudgeClient,
    create_judge_from_env,
)
from recipes.hotpotqa.output_parsing import split_native_thinking
from recipes.hotpotqa.process_verifier import verify_execution, verify_new_evidence
from recipes.hotpotqa.prompts import (
    HOTPOTQA_FINAL_TURN_PROMPT,
    HOTPOTQA_SYSTEM_PROMPT,
    HOTPOTQA_TOOL_SCHEMAS,
    HOTPOTQA_USER_PROMPT,
)
from recipes.hotpotqa.reward_arm import (
    RewardArm,
    final_step_response_mask,
    final_step_reward,
    resolve_reward_arm,
    scale_terminal_reward,
    search_step_reward,
)
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_RETRIEVAL_TOOL_NAMES = frozenset({"search", "wiki_search"})
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_HERMES_FUNCTION_BLOCK = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_HERMES_PARAMETER_BLOCK = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def _json_or_python_dict(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        pass
    if len(value) > 8192 or "__" in value:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def _normalize_tool_call_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or "name" not in value:
        return None
    arguments = value.get("arguments")
    if isinstance(arguments, str):
        arguments = _json_or_python_dict(arguments)
    if not isinstance(arguments, dict):
        return None
    return {"name": str(value["name"]), "arguments": arguments}


def _recover_tool_calls_from_text(text: str) -> list[FunctionCall]:
    """Recover both JSON and Qwen3.5 Hermes XML tool calls."""

    calls: list[FunctionCall] = []
    for raw in _TOOL_CALL_BLOCK.findall(text):
        normalized = _normalize_tool_call_dict(_json_or_python_dict(raw))
        if normalized is not None:
            calls.append(
                FunctionCall(
                    name=normalized["name"],
                    arguments=json.dumps(normalized["arguments"], ensure_ascii=False),
                )
            )
            continue
        for function_name, function_body in _HERMES_FUNCTION_BLOCK.findall(raw):
            arguments: dict[str, Any] = {}
            for parameter_name, parameter_value in _HERMES_PARAMETER_BLOCK.findall(function_body):
                parameter_name = parameter_name.strip()
                parameter_value = parameter_value.strip()
                if parameter_name == "bindings":
                    try:
                        arguments[parameter_name] = json.loads(parameter_value)
                    except (TypeError, ValueError):
                        arguments[parameter_name] = parameter_value
                else:
                    arguments[parameter_name] = parameter_value
            if arguments:
                calls.append(
                    FunctionCall(
                        name=function_name.strip(),
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    )
                )
    return calls


def _decode_tool_arguments(arguments: str) -> dict[str, Any] | None:
    try:
        value: Any = json.loads(arguments)
    except Exception:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    return value if isinstance(value, dict) else None


def _visible_passages(
    passages: list[tuple[str, Passage]], max_chars: int
) -> list[tuple[str, Passage]]:
    """Return the exact passage texts admitted by the shared prompt budget."""

    visible: list[tuple[str, Passage]] = []
    total = 0
    for index, (query, passage) in enumerate(passages, start=1):
        visible_text = passage.text[:1200].replace("\n", " ")
        line = f"[{index}] (query: {query}) {visible_text}"
        if total + len(line) > max_chars:
            break
        visible.append(
            (
                query,
                Passage(
                    pid=passage.pid,
                    title=passage.title,
                    text=visible_text,
                    score=passage.score,
                    sentence_evidence=passage.sentence_evidence,
                ),
            )
        )
        total += len(line)
    return visible


def _format_passage_list(passages: list[tuple[str, Passage]], max_chars: int) -> str:
    """Render accumulated passages for the raw-answer arms."""

    if not passages:
        return "None"
    visible = _visible_passages(passages, max_chars)
    lines = [
        f"[{index}] (query: {query}) {passage.text}"
        for index, (query, passage) in enumerate(visible, start=1)
    ]
    if len(visible) < len(passages):
        lines.append(f"... ({len(passages) - len(visible)} more passages truncated)")
    return "\n".join(lines)


def _format_history_actions(actions: list[str], *, include_indices: bool = False) -> str:
    if not actions:
        return "None"
    if include_indices:
        return "\n".join(
            f"[Search {search_index}] {query}"
            for search_index, query in enumerate(actions, start=1)
        )
    return "\n".join(f"[Search] {query}" for query in actions)


@register("hotpotqa_agent")
class HotpotQAAgentFlow(AgentFlowBase):
    """One shared step-preserving flow for matched HotpotQA reward arms."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls,
        dataset_config,
        shared_judge_server: JudgeServerManager | RemoteJudgeClient | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            trainer_config,
            server_manager,
            reward_loop_worker,
            tokenizer,
            processor,
            dataset_cls,
            dataset_config,
            **kwargs,
        )

        self.max_steps = int(kwargs.get("max_steps", 4))
        self.max_parallel_calls = int(kwargs.get("max_parallel_calls", 1))
        force_first_value = os.environ.get("HOTPOTQA_FORCE_FIRST_SEARCH", kwargs.get("force_first_search", False))
        self.force_first_search = coerce_bool(force_first_value, name="HOTPOTQA_FORCE_FIRST_SEARCH")
        if self.force_first_search:
            raise ValueError("The shared formal HotpotQA AgentFlow does not permit forced searches")

        self.formal_a0 = coerce_bool(os.environ.get("HOTPOTQA_FORMAL_A0", False), name="HOTPOTQA_FORMAL_A0")
        self.formal_experiment = coerce_bool(
            os.environ.get("HOTPOTQA_FORMAL_EXPERIMENT", self.formal_a0),
            name="HOTPOTQA_FORMAL_EXPERIMENT",
        )
        # Reward arm decides optimizer-visible rewards and final-answer masking.
        # A0/A1 leave search rewards at zero and score the terminal answer (EM);
        # A2 pays the full deterministic evidence reward and masks the final
        # answer out of the policy loss. A3 pays 0.5 process + 0.5 terminal EM
        # and keeps final-answer tokens in the loss. A6 replaces the
        # deterministic evidence verifier with an LLM semantic judge (Qwen3-4B)
        # that evaluates cumulative coverage against gold supporting facts,
        # rewarding only the high-watermark increment. A7 replaces the process
        # verifier with a weak execution check that rewards each model-generated
        # search producing a non-empty observation at 1/3 per step, while
        # keeping the same 0.5/0.5 combined-reward contract and terminal EM.
        # A9 replaces the optimizer-visible process score with a gold-free
        # behavioral sensitivity/invariance verifier, keeping the same 0.5/0.5
        # reward weights and raw search/final protocol.
        # Validation always reports EM.
        self.reward_arm = resolve_reward_arm(os.environ.get("HOTPOTQA_REWARD_ARM"), formal_a0=self.formal_a0)
        # A6: frozen LLM judge configuration. The server is lazily started on the
        # first call to run() because __init__ is synchronous.
        # If a shared_judge_server was passed from the AgentFlowWorker,
        # all trajectories on this worker will reuse the same aiohttp
        # session / connector, avoiding fd-conflict crashes under uvloop.
        self.judge_server: JudgeServerManager | RemoteJudgeClient | None = shared_judge_server
        if self.reward_arm is RewardArm.A6:
            # If a shared_judge_server was passed by the caller, use it.
            # Otherwise, lazily create one from environment variables on the
            # first run() call.  When HOTPOTQA_JUDGE_API_KEY is set, a
            # RemoteJudgeClient is created (no local GPU needed); otherwise
            # a local vLLM JudgeServerManager is used.
            self._needs_judge_init = shared_judge_server is None
            self.judge_completion_max_tokens = int(
                os.environ.get(
                    "HOTPOTQA_JUDGE_COMPLETION_MAX_TOKENS",
                    "256",
                )
            )
        self.prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        self.response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        self.enable_tool_parse_feedback = bool(kwargs.get("enable_tool_parse_feedback", True))
        self.tool_parser_name = str(self.config.actor_rollout_ref.rollout.multi_turn.format)
        self.tool_parser = ToolParser.get_tool_parser(self.tool_parser_name, self.tokenizer)
        self.final_answer_protocol = resolve_final_answer_protocol()

        embedding_devices = resolve_hotpotqa_embedding_devices(
            kwargs.get("embedding_devices"), kwargs.get("agent_flow_worker_index")
        )
        self.search_tool = HotpotQASearchToolLegacy(
            embedding_model_name=kwargs.get("embedding_model_name", DEFAULT_HOTPOTQA_EMBEDDING_MODEL),
            embedding_devices=embedding_devices,
            corpus_data_dir=kwargs.get("corpus_data_dir"),
            require_sentence_evidence=kwargs.get("require_sentence_evidence", False),
            evidence_sidecar_path=kwargs.get("evidence_sidecar_path"),
        )
        enable_thinking = self.apply_chat_template_kwargs.get("enable_thinking", False)
        self.thinking_mode = "native" if coerce_bool(enable_thinking, name="enable_thinking") else "disabled"
        if self.formal_experiment:
            failures: list[str] = []
            if self.max_steps != 4:
                failures.append(f"max_steps={self.max_steps}, expected 4")
            if self.max_parallel_calls != 1:
                failures.append(f"max_parallel_calls={self.max_parallel_calls}, expected 1")
            if self.thinking_mode != "disabled":
                failures.append("thinking must be disabled for the formal HotpotQA contract")
            if not self.search_tool.require_sentence_evidence:
                failures.append("official sentence evidence must be required")
            if self.search_tool.evidence_schema_version != EVIDENCE_SCHEMA_VERSION:
                failures.append("official evidence schema is not active")
            if failures:
                raise ValueError("Formal HotpotQA invariant failure: " + "; ".join(failures))

    @staticmethod
    def _a9_passage_records(passages: list[tuple[str, Passage]]) -> list[dict[str, Any]]:
        """Expose only observation fields to the gold-free A9 verifier."""

        return [
            {"retrieval_query": query, "text": passage.text}
            for query, passage in passages
        ]

    @staticmethod
    def _passages_from_records(
        original: list[tuple[str, Passage]],
        records: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, Passage]]:
        if len(original) != len(records):
            raise ValueError("A9 counterfactual passage count changed")
        result: list[tuple[str, Passage]] = []
        for (query, original_passage), record in zip(original, records):
            result.append(
                (
                    query,
                    Passage(
                        pid=original_passage.pid,
                        title=original_passage.title,
                        text=str(record.get("text", "")),
                        score=original_passage.score,
                        sentence_evidence=[],
                    ),
                )
            )
        return result

    def _a9_token_length(self, value: str) -> int:
        return len(self.tokenizer.encode(value, add_special_tokens=False))

    async def _a9_score_candidate(
        self,
        *,
        prompt_ids: list[int],
        response_text: str,
        query: str,
        plan: ProbePlan,
        candidate: str,
    ) -> CandidateScore | None:
        encoding = encode_action_candidate(
            self.tokenizer,
            response_text=response_text,
            query=query,
            target=plan.target,
            candidate=candidate,
        )
        if encoding is None:
            return None
        full_prompt_ids = [*prompt_ids, *encoding.token_ids]
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=full_prompt_ids,
            sampling_params={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 1,
                "prompt_logprobs": 0,
                "logprobs": False,
            },
        )
        prompt_logprob_ids = output.extra_fields.get("prompt_ids")
        prompt_logprobs = output.extra_fields.get("prompt_logprobs")
        policy_snapshot_id = output.extra_fields.get("global_steps")
        if prompt_logprob_ids is None or prompt_logprobs is None or policy_snapshot_id is None:
            return None
        absolute_start = len(prompt_ids) + encoding.target_token_start
        absolute_end = len(prompt_ids) + encoding.target_token_end
        return extract_candidate_score(
            candidate=candidate,
            full_prompt_token_ids=full_prompt_ids,
            absolute_target_start=absolute_start,
            absolute_target_end=absolute_end,
            prompt_logprob_ids=prompt_logprob_ids,
            prompt_logprobs=prompt_logprobs,
            policy_snapshot_id=str(policy_snapshot_id),
        )

    async def _verify_a9_behavior(
        self,
        *,
        question: str,
        query: str,
        response_text: str,
        passages: list[tuple[str, Passage]],
        history_actions: list[str],
        feedback: str,
        sample_key: str,
        turn_index: int,
        expected_policy_snapshot_id: str,
    ):
        records = self._a9_passage_records(passages)
        plan = build_probe_plan(
            query=query,
            question=question,
            history_actions=history_actions,
            passages=records,
            sample_key=sample_key,
            turn_index=turn_index,
            token_length=self._a9_token_length,
            probe_seed=int(os.environ.get("HOTPOTQA_A9_PROBE_SEED", "42")),
        )
        if plan is None:
            return invalid_verification("no_valid_probe_plan")

        sensitivity_prompt, _, sensitivity_visible = self._prompt_ids_within_budget(
            question,
            self._passages_from_records(passages, plan.sensitivity_passages),
            history_actions,
            feedback,
            final_turn=False,
            finish_allowed=False,
        )
        invariance_prompt, _, invariance_visible = self._prompt_ids_within_budget(
            question,
            self._passages_from_records(passages, plan.invariance_passages),
            history_actions,
            feedback,
            final_turn=False,
            finish_allowed=False,
        )
        if not visible_probe_inputs_valid(
            plan,
            sensitivity_passages=self._a9_passage_records(sensitivity_visible),
            invariance_passages=self._a9_passage_records(invariance_visible),
        ):
            return invalid_verification("counterfactual_prompt_truncation_changed")
        scored = await asyncio.gather(
            self._a9_score_candidate(
                prompt_ids=sensitivity_prompt,
                response_text=response_text,
                query=query,
                plan=plan,
                candidate=plan.sensitivity_candidate,
            ),
            self._a9_score_candidate(
                prompt_ids=sensitivity_prompt,
                response_text=response_text,
                query=query,
                plan=plan,
                candidate=plan.target.surface,
            ),
            self._a9_score_candidate(
                prompt_ids=invariance_prompt,
                response_text=response_text,
                query=query,
                plan=plan,
                candidate=plan.target.surface,
            ),
            self._a9_score_candidate(
                prompt_ids=invariance_prompt,
                response_text=response_text,
                query=query,
                plan=plan,
                candidate=plan.distractor_candidate,
            ),
        )
        if any(score is None for score in scored):
            return invalid_verification("candidate_scoring_failed")
        source_state_hash = hashlib.sha256(
            json.dumps(
                {
                    "question": question,
                    "history_actions": history_actions,
                    "passages": records,
                    "query": query,
                    "feedback": feedback,
                    "turn_index": turn_index,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return verify_behavioral_scores(
            plan=plan,
            sensitivity_new=scored[0],
            sensitivity_old=scored[1],
            invariance_target=scored[2],
            invariance_distractor=scored[3],
            source_state_hash=source_state_hash,
            expected_policy_snapshot_id=expected_policy_snapshot_id,
        )

    def _build_messages(
        self,
        question: str,
        passages: list[tuple[str, Passage]],
        actions: list[str],
        tool_feedback: str,
        *,
        final_turn: bool,
        finish_allowed: bool,
        max_passage_chars: int,
    ) -> list[dict[str, str]]:
        feedback = tool_feedback.strip() or "None"
        user_content = HOTPOTQA_USER_PROMPT.format(
            user_query=question,
            passage_list=_format_passage_list(passages, max_passage_chars),
            history_actions=_format_history_actions(actions),
            tool_feedback=feedback,
        )
        if final_turn:
            user_content += f"\n\n{HOTPOTQA_FINAL_TURN_PROMPT}"
        system_prompt = HOTPOTQA_SYSTEM_PROMPT
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _tool_schemas(self, *, final_turn: bool, finish_allowed: bool) -> list[dict[str, Any]]:
        return HOTPOTQA_TOOL_SCHEMAS

    def _apply_text_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        final_turn: bool,
        finish_allowed: bool,
    ) -> list[int]:
        """Use the synchronous text path that is stable for Qwen3.5 templates."""

        return normalize_token_ids(
            apply_chat_template(
                self.tokenizer,
                messages,
                tools=self._tool_schemas(
                    final_turn=final_turn,
                    finish_allowed=finish_allowed,
                ),
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
        )

    def _prompt_ids_within_budget(
        self,
        question: str,
        passages: list[tuple[str, Passage]],
        actions: list[str],
        feedback: str,
        *,
        final_turn: bool,
        finish_allowed: bool,
    ) -> tuple[list[int], str, list[tuple[str, Passage]]]:
        max_chars = self.prompt_length * 3
        min_chars = 400
        while True:
            messages = self._build_messages(
                question,
                passages,
                actions,
                feedback,
                final_turn=final_turn,
                finish_allowed=finish_allowed,
                max_passage_chars=max_chars,
            )
            prompt_ids = self._apply_text_chat_template(
                messages,
                final_turn=final_turn,
                finish_allowed=finish_allowed,
            )
            if len(prompt_ids) <= self.prompt_length:
                return prompt_ids, messages[1]["content"], _visible_passages(passages, max_chars)
            if max_chars <= min_chars:
                raise ValueError(
                    f"HotpotQA prompt has {len(prompt_ids)} tokens after passage truncation; "
                    f"limit is {self.prompt_length}"
                )
            max_chars = max(min_chars, int(max_chars * 0.72))

    @staticmethod
    def _returned_evidence_ids(search_step: Mapping[str, Any]) -> list[str]:
        return [
            str(sentence.get("evidence_id"))
            for paragraph in search_step.get("returned_evidence") or []
            for sentence in paragraph.get("sentence_evidence") or []
            if sentence.get("evidence_id")
        ]

    @classmethod
    def _attach_evidence_audit(
        cls,
        search_step: dict[str, Any],
        gold_evidence_ids: set[str],
        covered_gold_evidence_ids: set[str],
        *,
        evidence_metrics_eligible: bool,
    ) -> None:
        verification = verify_new_evidence(
            returned_evidence_ids=cls._returned_evidence_ids(search_step),
            gold_evidence_ids=gold_evidence_ids,
            covered_gold_evidence_ids=covered_gold_evidence_ids,
            evidence_metrics_eligible=evidence_metrics_eligible,
        )
        covered_gold_evidence_ids.clear()
        covered_gold_evidence_ids.update(verification.covered_gold_evidence_ids)
        search_step["returned_evidence_ids"] = list(verification.returned_evidence_ids)
        search_step["new_gold_evidence_ids"] = list(verification.new_gold_evidence_ids)
        search_step["covered_gold_evidence_ids"] = list(verification.covered_gold_evidence_ids)
        search_step["audit_process_reward"] = verification.reward

    def _evidence_metrics(
        self,
        search_steps: list[dict[str, Any]],
        gold_evidence_ids: set[str],
        covered_gold_evidence_ids: set[str],
        unresolved_gold_facts: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        returned = {evidence_id for step in search_steps for evidence_id in self._returned_evidence_ids(step)}
        hits = len(covered_gold_evidence_ids)
        eligible = not unresolved_gold_facts and bool(gold_evidence_ids)
        return {
            "evidence_metrics_eligible": eligible,
            "gold_evidence_count": len(gold_evidence_ids),
            "unresolved_gold_fact_count": len(unresolved_gold_facts),
            "retrieved_evidence_count": len(returned),
            "supporting_fact_hits": hits,
            "supporting_fact_recall": (hits / len(gold_evidence_ids) if eligible else None),
            "supporting_fact_precision": (
                hits / len(returned) if eligible and returned else (0.0 if eligible else None)
            ),
            "at_least_one_support": hits > 0 if eligible else None,
            "both_support": (covered_gold_evidence_ids == gold_evidence_ids if eligible else None),
            "complete_coverage_search_step": next(
                (
                    int(step["search_index"])
                    for step in search_steps
                    if eligible and set(step.get("covered_gold_evidence_ids") or []) == gold_evidence_ids
                ),
                None,
            ),
        }

    def _make_extra_fields(
        self,
        *,
        anchor_obs: str,
        history_actions: list[str],
        search_steps: list[dict[str, Any]],
        qwen_thinking: list[dict[str, Any]],
        sample_key: str,
        base_sample_key: str,
        dataset_question_id: str,
        official_qid: str,
        gold_evidence_ids: tuple[str, ...],
        unresolved_gold_facts: tuple[dict[str, Any], ...],
        covered_gold_evidence_ids: set[str],
        step_kind: str,
        acc: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "anchor_obs": anchor_obs,
            "step_kind": step_kind,
            "executed_search_queries": list(history_actions),
            "search_steps": list(search_steps),
            "qwen_thinking": list(qwen_thinking),
            "thinking_mode": self.thinking_mode,
            "force_first_search": False,
            "final_answer_protocol": self.final_answer_protocol,
            "evidence_schema_version": self.search_tool.evidence_schema_version,
            "sample_key": sample_key,
            "base_sample_key": base_sample_key,
            "dataset_question_id": dataset_question_id,
            "official_qid": official_qid,
            "gold_evidence_ids": list(gold_evidence_ids),
            "unresolved_gold_facts": list(unresolved_gold_facts),
            "evidence_metrics": self._evidence_metrics(
                search_steps,
                set(gold_evidence_ids),
                covered_gold_evidence_ids,
                unresolved_gold_facts,
            ),
            "reward_extra_info": {
                "num_tool_steps": len(history_actions),
                "acc": acc,
                "judge_invalid": False,
            },
        }

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        question = str(raw_prompt[0]["content"]).strip()
        is_validation = bool(kwargs.get("_agent_r1_is_validation", False))
        extra_info = kwargs.get("extra_info") or {}
        if not isinstance(extra_info, Mapping):
            raise ValueError("HotpotQA extra_info must be a mapping")
        split = str(extra_info.get("split") or "validation")
        try:
            row_index = int(extra_info.get("index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("HotpotQA extra_info.index is required") from exc
        source_row_index = row_index

        dataset_question_id = str(extra_info.get("question_id") or "")
        official_qid = dataset_question_id
        sample_key = f"{split}:{row_index}"
        base_sample_key = f"{split}:{source_row_index}"
        gold_evidence_ids: tuple[str, ...] = ()
        unresolved_gold_facts: tuple[dict[str, Any], ...] = ()
        if self.search_tool.require_sentence_evidence:
            sample = self.search_tool.evidence_store.sample(split, source_row_index)
            if sample.question != question:
                raise ValueError(f"Evidence sidecar question mismatch for {sample.sample_key}")
            if dataset_question_id and sample.dataset_question_id != dataset_question_id:
                raise ValueError(f"Evidence sidecar dataset question ID mismatch for {sample.sample_key}")
            dataset_question_id = sample.dataset_question_id
            official_qid = sample.official_qid
            base_sample_key = sample.sample_key
            sample_key = sample.sample_key
            gold_evidence_ids = sample.gold_evidence_ids
            unresolved_gold_facts = sample.unresolved_gold_facts
        elif self.formal_experiment:
            raise RuntimeError("Formal HotpotQA experiments require the official evidence sidecar")

        judge_arms = {RewardArm.A6}
        # A6: lazily start the frozen Judge server on the first run() call.
        if self.reward_arm == RewardArm.A6 and self.judge_server is None:
            self.judge_server = create_judge_from_env(self.reward_arm)
            await self.judge_server.start()

        # A6: trajectory-local state for the fact-ID level LLM judge.
        # judge_covered_gold_evidence_ids_llm tracks which gold fact IDs the
        # judge has determined are covered across all search steps so far,
        # analogous to A3's covered_gold_evidence_ids.
        judge_covered_gold_evidence_ids_llm: set[str] = set()
        judge_gold_facts_for_prompt: list[tuple[str, str]] = []  # (fact_id, text)
        trajectory_judge_invalid: bool = False
        if self.reward_arm == RewardArm.A6 and not is_validation:
            judge_gold_facts_for_prompt = self._get_gold_supporting_fact_ids_and_texts(
                self.search_tool.evidence_store,
                gold_evidence_ids,
            )

        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "step_generate_sequences": [],
            "step_tool_calls": [],
        }
        steps: list[AgentFlowStep] = []
        passages: list[tuple[str, Passage]] = []
        history_actions: list[str] = []
        search_steps: list[dict[str, Any]] = []
        qwen_thinking: list[dict[str, Any]] = []
        covered_gold_evidence_ids: set[str] = set()
        feedback_lines: list[str] = []

        for step_number in range(1, self.max_steps + 1):
            step_metrics: dict[str, float] = {}
            final_turn = step_number == self.max_steps
            finish_allowed = False
            feedback = "\n".join(feedback_lines[-3:]) if self.enable_tool_parse_feedback else ""
            prompt_ids, anchor_obs, actor_visible_passages = self._prompt_ids_within_budget(
                question,
                passages,
                history_actions,
                feedback,
                final_turn=final_turn,
                finish_allowed=finish_allowed,
            )
            step_sampling_params = dict(sampling_params)
            step_sampling_params["max_tokens"] = min(
                int(step_sampling_params.get("max_tokens", self.response_length)),
                self.response_length,
            )
            with simple_timer("generate_sequences", step_metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=step_sampling_params,
                )

            response_ids = list(output.token_ids[: self.response_length])
            if not response_ids:
                raise RuntimeError(
                    "vLLM returned an empty generation "
                    f"(stop_reason={output.stop_reason!r}, extra_fields={output.extra_fields!r})"
                )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            if self.thinking_mode == "native":
                thinking_text, visible_text, thinking_complete = split_native_thinking(response_text)
            else:
                # With enable_thinking=false Qwen's chat template already places
                # an empty <think></think> block in the prompt.  The completion is
                # therefore visible output from its first token onward.
                thinking_text, visible_text, thinking_complete = "", response_text.strip(), True
            qwen_thinking.append(
                {
                    "turn": step_number,
                    "content": thinking_text,
                    "complete": thinking_complete,
                }
            )

            tool_calls: list[FunctionCall] = []
            if thinking_complete:
                if self.tool_parser_name == "hermes":
                    tool_calls = _recover_tool_calls_from_text(visible_text)
                else:
                    visible_ids = self.tokenizer.encode(visible_text, add_special_tokens=False)
                    _, tool_calls = await self.tool_parser.extract_tool_calls(visible_ids)
                    if not tool_calls:
                        tool_calls = _recover_tool_calls_from_text(visible_text)

            valid_queries: list[str] = []
            for tool_call in tool_calls[: self.max_parallel_calls]:
                if tool_call.name not in _RETRIEVAL_TOOL_NAMES:
                    continue
                arguments = _decode_tool_arguments(tool_call.arguments)
                query = arguments.get("query") if arguments else None
                if query:
                    valid_queries.append(str(query))

            if final_turn or (not tool_calls and "<tool_call>" not in visible_text):
                final_reward = final_step_reward(self.reward_arm, is_validation=is_validation)
                final_mask = final_step_response_mask(
                    self.reward_arm,
                    len(response_ids),
                    is_validation=is_validation,
                )
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=(
                        output.log_probs[: self.response_length] if output.log_probs else None
                    ),
                    reward_score=final_reward,
                    response_mask=final_mask,
                    extra_fields=self._make_extra_fields(
                        anchor_obs=anchor_obs,
                        history_actions=history_actions,
                        search_steps=search_steps,
                        qwen_thinking=qwen_thinking,
                        sample_key=sample_key,
                        base_sample_key=base_sample_key,
                        dataset_question_id=dataset_question_id,
                        official_qid=official_qid,
                        gold_evidence_ids=gold_evidence_ids,
                        unresolved_gold_facts=unresolved_gold_facts,
                        covered_gold_evidence_ids=covered_gold_evidence_ids,
                        step_kind="final",
                    ),
                )
                step = await self._postprocess(step, **kwargs)
                raw_terminal_reward = float(step.reward_score)
                terminal_component = scale_terminal_reward(
                    self.reward_arm,
                    raw_terminal_reward,
                    is_validation=is_validation,
                )
                if trajectory_judge_invalid and self.reward_arm in judge_arms:
                    step.reward_score = 0.0
                else:
                    step.reward_score = terminal_component
                reward_info = step.extra_fields.get("reward_extra_info", {})
                reward_info.update(
                    {
                        "acc": float(reward_info.get("acc", raw_terminal_reward)),
                        "terminal_em": raw_terminal_reward,
                        "optimizer_terminal_component": terminal_component,
                        "optimizer_total_reward": float(step.reward_score),
                        "judge_invalid": trajectory_judge_invalid,
                    }
                )
                step.extra_fields["reward_extra_info"] = reward_info
                steps.append(step)
                self._record_step_timing(metrics, step_metrics)
                break

            if valid_queries:
                query = valid_queries[0]
                a9_verification = None
                if self.reward_arm is RewardArm.A9 and not is_validation:
                    sampled_policy_snapshot = output.extra_fields.get("global_steps")
                    if sampled_policy_snapshot is None:
                        a9_verification = invalid_verification("sampled_action_snapshot_missing")
                    else:
                        a9_verification = await self._verify_a9_behavior(
                            question=question,
                            query=query,
                            response_text=response_text,
                            passages=actor_visible_passages,
                            history_actions=history_actions,
                            feedback=feedback,
                            sample_key=sample_key,
                            turn_index=step_number,
                            expected_policy_snapshot_id=str(sampled_policy_snapshot),
                        )
                with simple_timer("tool_calls", step_metrics):
                    search_step = self._do_search(
                        query,
                        assistant_turn=step_number,
                    )
                search_step["search_index"] = len(search_steps) + 1
                # Always run the deterministic audit for logging/comparison.
                self._attach_evidence_audit(
                    search_step,
                    set(gold_evidence_ids),
                    covered_gold_evidence_ids,
                    evidence_metrics_eligible=not unresolved_gold_facts,
                )
                search_step["deterministic_process_reward"] = search_step["audit_process_reward"]
                if self.reward_arm is RewardArm.A9:
                    if a9_verification is None:
                        a9_verification = invalid_verification("validation_probe_disabled")
                    search_step["a9_behavioral_audit"] = a9_verification.artifact
                    search_step["a9_eligible_turn"] = a9_verification.eligible
                    search_step["a9_no_valid_probe"] = a9_verification.no_valid_probe
                    search_step["a9_sensitivity_pass"] = a9_verification.sensitivity_pass
                    search_step["a9_invariance_pass"] = a9_verification.invariance_pass
                    search_step["a9_joint_pass"] = a9_verification.joint_pass
                    search_step["a9_verdict_class"] = a9_verification.verdict_class
                    search_step["a9_raw_process_reward"] = a9_verification.raw_process_reward
                    search_step["a9_process_component_valid"] = (
                        a9_verification.process_component_valid
                    )
                    search_step["audit_process_reward"] = (
                        a9_verification.optimizer_process_value
                    )
                # A6 training: override process reward with LLM judge
                # fact-ID level coverage, using the same formula as A3.
                if self.reward_arm == RewardArm.A6 and not is_validation and not trajectory_judge_invalid:
                    current_passages = self._format_judge_passages(search_step)
                    gold_facts_formatted = format_gold_facts_for_prompt(judge_gold_facts_for_prompt)
                    judge_prompt = self._build_judge_prompt(
                        question=question,
                        gold_supporting_fact_texts=gold_facts_formatted,
                        previous_queries=list(history_actions),
                        current_query=query,
                        previous_passages=self._format_judge_previous_passages([(q, p) for q, p in passages]),
                        current_passages=current_passages,
                    )
                    # Pass allowed fact IDs so judge() validates the output
                    allowed_ids = set(gold_evidence_ids)
                    judge_result = await self.judge_server.judge(judge_prompt, allowed_fact_ids=allowed_ids)

                    if judge_result is None:
                        trajectory_judge_invalid = True
                        search_step["llm_judge_covered_ids"] = None
                        search_step["llm_process_increment"] = None
                    else:
                        # Apply same formula as A3: |new_llm_covered| / |G_i|
                        judge_gold_hits = judge_result & set(gold_evidence_ids)
                        new_llm_covered = judge_gold_hits - judge_covered_gold_evidence_ids_llm
                        if gold_evidence_ids:
                            process_increment = len(new_llm_covered) / len(gold_evidence_ids)
                        else:
                            process_increment = 0.0
                        judge_covered_gold_evidence_ids_llm.update(judge_gold_hits)
                        search_step["audit_process_reward"] = process_increment
                        search_step["llm_judge_covered_ids"] = sorted(judge_result)
                        search_step["llm_judge_gold_hits"] = sorted(judge_gold_hits)
                        search_step["llm_new_covered_ids"] = sorted(new_llm_covered)
                        search_step["llm_process_increment"] = process_increment
                elif self.reward_arm == RewardArm.A6 and is_validation:
                    # Validation: compute judge fact-ID coverage for logging only, not reward.
                    current_passages = self._format_judge_passages(search_step)
                    gold_facts_for_logging = self._get_gold_supporting_fact_ids_and_texts(
                        self.search_tool.evidence_store,
                        gold_evidence_ids,
                    )
                    gold_facts_formatted = format_gold_facts_for_prompt(gold_facts_for_logging)
                    judge_prompt = self._build_judge_prompt(
                        question=question,
                        gold_supporting_fact_texts=gold_facts_formatted,
                        previous_queries=list(history_actions),
                        current_query=query,
                        previous_passages=self._format_judge_previous_passages([(q, p) for q, p in passages]),
                        current_passages=current_passages,
                    )
                    allowed_ids = set(gold_evidence_ids)
                    judge_result = await self.judge_server.judge(judge_prompt, allowed_fact_ids=allowed_ids)
                    search_step["llm_judge_covered_ids"] = sorted(judge_result) if judge_result is not None else None

                search_steps.append(search_step)
                history_actions.append(query)
                self._ingest_search_step(query, search_step, passages)
                step_kind = "search"

                # A7 uses the same verifier in train and validation; only the
                # train path replaces optimizer-visible process reward.
                if self.reward_arm == RewardArm.A7:
                    weak_verification = verify_execution(
                        model_generated=True,  # source=="model" guaranteed above
                        schema_and_parser_valid=True,  # valid_queries non-empty
                        query=query,
                        tool_execution_succeeded=bool(search_step.get("success", False)),
                        returned_fact_records=search_step.get("returned_evidence") or [],
                        max_executed_searches=3,
                    )
                    search_step["weak_execution_audit"] = {
                        "model_generated": weak_verification.model_generated,
                        "schema_and_parser_valid": weak_verification.schema_and_parser_valid,
                        "nonempty_query": weak_verification.nonempty_query,
                        "tool_execution_succeeded": weak_verification.tool_execution_succeeded,
                        "nonempty_observation": weak_verification.nonempty_observation,
                        "raw_weak_process": weak_verification.raw_weak_process,
                        "max_executed_searches": weak_verification.max_executed_searches,
                    }
                    if not is_validation:
                        search_step["audit_process_reward"] = weak_verification.raw_weak_process

            else:
                feedback_lines.append(
                    "The previous tool call was invalid. Use exactly one search call with a non-empty "
                    "string parameter named query."
                )
                step_kind = "invalid_tool_call"

            # Process arms pay their frozen weighted reward on the search
            # action that produced it; A0/A1 and every
            # validation pass keep search-step rewards at zero. Judge-invalid
            # A6 trajectories are zeroed here and rejected by the trainer
            # before any optimizer update.
            if step_kind == "search":
                if self.reward_arm in judge_arms and trajectory_judge_invalid:
                    step_reward = 0.0
                else:
                    process_reward = search_steps[-1].get("audit_process_reward")
                    step_reward = search_step_reward(self.reward_arm, process_reward, is_validation=is_validation)
            else:
                step_reward = 0.0

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=(output.log_probs[: self.response_length] if output.log_probs else None),
                reward_score=step_reward,
                extra_fields=self._make_extra_fields(
                    anchor_obs=anchor_obs,
                    history_actions=history_actions,
                    search_steps=search_steps,
                    qwen_thinking=qwen_thinking,
                    sample_key=sample_key,
                    base_sample_key=base_sample_key,
                    dataset_question_id=dataset_question_id,
                    official_qid=official_qid,
                    gold_evidence_ids=gold_evidence_ids,
                    unresolved_gold_facts=unresolved_gold_facts,
                    covered_gold_evidence_ids=covered_gold_evidence_ids,
                    step_kind=step_kind,
                ),
            )
            # Mark judge_invalid on every step of this trajectory so the
            # training loop can exclude the entire trajectory from the
            # actor policy update.
            if trajectory_judge_invalid:
                reward_info = step.extra_fields.get("reward_extra_info", {})
                reward_info["judge_invalid"] = True
                step.extra_fields["reward_extra_info"] = reward_info
            # A7: surface weak execution audit in reward_extra_info for
            # easy access in training logs / offline replay.
            if step_kind == "search" and self.reward_arm == RewardArm.A7:
                weak_audit = search_steps[-1].get("weak_execution_audit")
                if weak_audit:
                    reward_info = step.extra_fields.get("reward_extra_info", {})
                    reward_info["weak_execution"] = weak_audit
                    reward_info["weighted_process"] = float(step_reward)
                    step.extra_fields["reward_extra_info"] = reward_info
            if step_kind == "search" and self.reward_arm == RewardArm.A9:
                a9_audit = search_steps[-1].get("a9_behavioral_audit")
                reward_info = step.extra_fields.get("reward_extra_info", {})
                reward_info["a9_behavioral"] = a9_audit
                reward_info["a9_raw_process_reward"] = search_steps[-1].get(
                    "a9_raw_process_reward"
                )
                reward_info["optimizer_process_component"] = float(step_reward)
                reward_info["a9_process_component_valid"] = bool(
                    search_steps[-1].get("a9_process_component_valid", False)
                )
                reward_info["a9_verdict_class"] = search_steps[-1].get(
                    "a9_verdict_class"
                )
                step.extra_fields["reward_extra_info"] = reward_info
            steps.append(await self._postprocess(step, **kwargs))
            self._record_step_timing(metrics, step_metrics)

        return AgentFlowOutput(steps=steps, metrics=metrics)

    @staticmethod
    def _record_step_timing(metrics: dict[str, Any], step_metrics: Mapping[str, float]) -> None:
        """Store one raw timing sample per emitted step and maintain legacy totals."""
        for name in ("generate_sequences", "tool_calls"):
            elapsed = float(step_metrics.get(name, 0.0))
            metrics[name] += elapsed
            metrics[f"step_{name}"].append(elapsed)

    def _do_search(
        self,
        query: str,
        *,
        assistant_turn: int,
    ) -> dict[str, Any]:
        results = self.search_tool.batch_execute([{"query": query}])
        item = results[0] if results else {"success": False, "content": "missing result"}
        passages = parse_legacy_tool_result(str(item.get("content", ""))) if item.get("success", False) else []
        return {
            "search_index": None,
            "assistant_turn": assistant_turn,
            "source": "model",
            "query": query,
            "success": bool(item.get("success", False)),
            "returned_evidence": [passage.evidence_record() for passage in passages],
        }

    @staticmethod
    def _ingest_search_step(
        query: str,
        search_step: Mapping[str, Any],
        passages: list[tuple[str, Passage]],
    ) -> None:
        for record in search_step.get("returned_evidence") or []:
            passage = Passage(
                pid=int(record["pid"]),
                title=str(record.get("title", "")),
                text=str(record.get("text", "")),
                score=float(record.get("score", 0.0)),
                sentence_evidence=list(record.get("sentence_evidence") or []),
            )
            if not any(existing.pid == passage.pid for _, existing in passages):
                passages.append((query, passage))

    # ------------------------------------------------------------------
    # A6 LLM judge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_judge_passages(
        search_step: Mapping[str, Any],
        max_chars_per_passage: int = 800,
    ) -> list[str]:
        """Extract and format passage texts from a search step for the judge prompt."""
        lines: list[str] = []
        for record in search_step.get("returned_evidence") or []:
            title = str(record.get("title", ""))
            text = str(record.get("text", ""))[:max_chars_per_passage].replace("\n", " ")
            lines.append(f"[{title}] {text}")
        return lines

    @staticmethod
    def _format_judge_previous_passages(
        passages: list[tuple[str, Passage]],
        max_chars_per_passage: int = 800,
        max_total_chars: int = 6000,
    ) -> str:
        """Format previously accumulated passages for the judge prompt."""
        if not passages:
            return "None"
        lines: list[str] = []
        total = 0
        for query, passage in passages:
            line = f"(query: {query}) [{passage.title}] {passage.text[:max_chars_per_passage].replace(chr(10), ' ')}"
            if total + len(line) > max_total_chars:
                remaining = len(passages) - len(lines)
                if remaining > 0:
                    lines.append(f"... ({remaining} more passages truncated)")
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    @staticmethod
    def _get_gold_supporting_fact_ids_and_texts(
        evidence_store: Any,
        gold_evidence_ids: tuple[str, ...],
    ) -> list[tuple[str, str]]:
        """Retrieve fact IDs and their text for the v4 judge prompt.

        Returns a list of ``(fact_id, fact_text)`` pairs where *fact_id* is
        the official evidence ID (e.g.
        ``hotpotqa-official-sentence-v1:123:0``) and *fact_text* is the
        formatted ``[Title] Sentence text`` string.
        """
        if not gold_evidence_ids:
            return []
        result: list[tuple[str, str]] = []
        for evidence_id in gold_evidence_ids:
            try:
                from recipes.hotpotqa.evidence import parse_evidence_id

                pid, sentence_id = parse_evidence_id(evidence_id)
                records = evidence_store.paragraph_evidence(pid)
                for record in records:
                    if record.get("sentence_id") == sentence_id:
                        title = record.get("title", "")
                        text = record.get("text", "")
                        result.append((evidence_id, f"[{title}] {text}"))
                        break
                else:
                    result.append((evidence_id, f"[ID: {evidence_id}]"))
            except Exception:
                result.append((evidence_id, f"[ID: {evidence_id}]"))
        return result

    def _build_judge_prompt(
        self,
        *,
        question: str,
        gold_supporting_fact_texts: str,
        previous_queries: list[str],
        current_query: str,
        previous_passages: str,
        current_passages: list[str],
    ) -> str:
        """Build the judge evaluation prompt using the frozen template."""
        current_passages_str = "\n".join(current_passages) if current_passages else "None"
        previous_queries_str = "\n".join(f"[Search] {q}" for q in previous_queries) if previous_queries else "None"
        return JUDGE_USER_PROMPT.format(
            question=question,
            gold_supporting_fact_texts=gold_supporting_fact_texts,
            previous_queries=previous_queries_str,
            current_query=current_query,
            previous_passages=previous_passages,
            current_passages=current_passages_str,
        )
