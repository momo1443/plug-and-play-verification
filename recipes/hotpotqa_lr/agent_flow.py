"""Standalone AgentFlow for outcome-independent local reasoning RLVR."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.reward_loop.reward_loop import RewardLoopWorker
from recipes.hotpotqa.env.search_tool import (
    DEFAULT_HOTPOTQA_EMBEDDING_MODEL,
    HotpotQASearchToolLegacy,
    Passage,
    parse_legacy_tool_result,
    resolve_hotpotqa_embedding_devices,
)
from recipes.hotpotqa.evidence import EVIDENCE_SCHEMA_VERSION, coerce_bool
from recipes.hotpotqa.output_parsing import split_native_thinking
from recipes.hotpotqa_lr.prompts import (
    BOOTSTRAP_TOOL_SCHEMAS,
    FINISH_TOOL_SCHEMAS,
    LR_FINAL_TURN_PROMPT,
    LR_FINISH_AVAILABLE_PROMPT,
    LR_SYSTEM_PROMPT,
    LR_USER_PROMPT,
    SEARCH_OR_FINISH_TOOL_SCHEMAS,
)
from recipes.hotpotqa_lr.protocol import LR_FINISH_PROTOCOL, extract_tool_calls, parse_finish
from recipes.hotpotqa_lr.reward_contract import LR_CONTRACT_VERSION, PRIMARY_CONTRACT
from recipes.hotpotqa_lr.verifier import (
    artifact_content_sha256,
    artifact_id_for_passage,
    trajectory_audit_record,
    verify_trajectory,
)
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _format_history(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Search {index}] {query}" for index, query in enumerate(actions, start=1))


def _format_ledger(transitions: list[dict[str, Any]]) -> str:
    if not transitions:
        return "None"
    lines = []
    for transition in transitions:
        lines.append(
            json.dumps(
                {
                    "reason_step": transition.get("reason_step"),
                    "action": {
                        "type": transition.get("action_type"),
                        "value": transition.get("action_value"),
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(lines)


def _format_passages(
    passages: list[tuple[str, Passage]],
    max_chars: int,
) -> tuple[str, dict[str, dict[str, Any]]]:
    if not passages:
        return "None", {}
    lines: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    total = 0
    for query, passage in passages:
        artifact_id = artifact_id_for_passage(passage.pid)
        if artifact_id in artifacts:
            continue
        text = passage.text[:1200].replace("\n", " ")
        digest = artifact_content_sha256(text)
        line = f"[artifact_id={artifact_id}, sha256={digest}] (query: {query}) {text}"
        if total + len(line) > max_chars:
            lines.append("... (remaining artifacts truncated)")
            break
        lines.append(line)
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "passage_id": str(passage.pid),
            "title": passage.title,
            "text": text,
            "content_sha256": digest,
        }
        total += len(line)
    return "\n".join(lines), artifacts


@register("hotpotqa_local_reasoning_agent")
class HotpotQALocalReasoningAgentFlow(AgentFlowBase):
    """Four-turn HotpotQA flow with a fixed three-transition reward horizon."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls,
        dataset_config,
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
        self.reward_horizon = int(kwargs.get("reward_horizon", PRIMARY_CONTRACT.reward_horizon))
        self.dependency_taint_gamma = float(
            kwargs.get("dependency_taint_gamma", PRIMARY_CONTRACT.dependency_taint_gamma)
        )
        self.enable_tool_parse_feedback = bool(kwargs.get("enable_tool_parse_feedback", True))
        self.formal_experiment = coerce_bool(
            os.environ.get("HOTPOTQA_FORMAL_EXPERIMENT", True),
            name="HOTPOTQA_FORMAL_EXPERIMENT",
        )
        self.prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        self.response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        embedding_devices = resolve_hotpotqa_embedding_devices(
            kwargs.get("embedding_devices"), kwargs.get("agent_flow_worker_index")
        )
        self.search_tool = HotpotQASearchToolLegacy(
            embedding_model_name=kwargs.get(
                "embedding_model_name", DEFAULT_HOTPOTQA_EMBEDDING_MODEL
            ),
            embedding_devices=embedding_devices,
            corpus_data_dir=kwargs.get("corpus_data_dir"),
            require_sentence_evidence=kwargs.get("require_sentence_evidence", False),
            evidence_sidecar_path=kwargs.get("evidence_sidecar_path"),
        )
        enable_thinking = self.apply_chat_template_kwargs.get("enable_thinking", False)
        self.thinking_mode = (
            "native" if coerce_bool(enable_thinking, name="enable_thinking") else "disabled"
        )
        failures: list[str] = []
        if self.max_steps != 4:
            failures.append(f"max_steps={self.max_steps}, expected 4")
        if self.max_parallel_calls != 1:
            failures.append(f"max_parallel_calls={self.max_parallel_calls}, expected 1")
        if self.reward_horizon != PRIMARY_CONTRACT.reward_horizon:
            failures.append(f"reward_horizon={self.reward_horizon}, expected 3")
        if self.dependency_taint_gamma != PRIMARY_CONTRACT.dependency_taint_gamma:
            failures.append(
                f"dependency_taint_gamma={self.dependency_taint_gamma}, expected 0.3"
            )
        if self.thinking_mode != "disabled":
            failures.append("thinking must be disabled")
        if not self.search_tool.require_sentence_evidence:
            failures.append("official sentence evidence must be required")
        if self.search_tool.evidence_schema_version != EVIDENCE_SCHEMA_VERSION:
            failures.append("official evidence schema is not active")
        if failures:
            raise ValueError("A8-LR invariant failure: " + "; ".join(failures))

    def _tool_schemas(self, *, bootstrap: bool, final_turn: bool) -> list[dict[str, Any]]:
        if final_turn:
            return FINISH_TOOL_SCHEMAS
        if bootstrap:
            return BOOTSTRAP_TOOL_SCHEMAS
        return SEARCH_OR_FINISH_TOOL_SCHEMAS

    def _prompt_ids_within_budget(
        self,
        *,
        question: str,
        passages: list[tuple[str, Passage]],
        actions: list[str],
        transitions: list[dict[str, Any]],
        feedback: str,
        final_turn: bool,
    ) -> tuple[list[int], str, dict[str, dict[str, Any]]]:
        max_chars = self.prompt_length * 3
        while True:
            passage_text, artifacts = _format_passages(passages, max_chars)
            user_content = LR_USER_PROMPT.format(
                user_query=question,
                history_actions=_format_history(actions),
                passage_list=passage_text,
                reasoning_ledger=_format_ledger(transitions),
                tool_feedback=feedback.strip() or "None",
            )
            if final_turn:
                user_content += f"\n\n{LR_FINAL_TURN_PROMPT}"
            elif actions:
                user_content += f"\n\n{LR_FINISH_AVAILABLE_PROMPT}"
            messages = [
                {"role": "system", "content": LR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt_ids = normalize_token_ids(
                apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=self._tool_schemas(bootstrap=not actions, final_turn=final_turn),
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                )
            )
            if len(prompt_ids) <= self.prompt_length:
                return prompt_ids, user_content, artifacts
            if max_chars <= 400:
                raise ValueError(
                    f"A8-LR prompt has {len(prompt_ids)} tokens; limit is {self.prompt_length}"
                )
            max_chars = max(400, int(max_chars * 0.72))

    @staticmethod
    def _returned_evidence_ids(search_step: Mapping[str, Any]) -> list[str]:
        return [
            str(sentence["evidence_id"])
            for paragraph in search_step.get("returned_evidence") or []
            for sentence in paragraph.get("sentence_evidence") or []
            if sentence.get("evidence_id")
        ]

    @classmethod
    def _attach_gold_audit(
        cls,
        search_step: dict[str, Any],
        gold_evidence_ids: set[str],
        covered: set[str],
        *,
        eligible: bool,
    ) -> None:
        returned = set(cls._returned_evidence_ids(search_step))
        new_gold = returned & gold_evidence_ids - covered if eligible else set()
        covered.update(new_gold)
        search_step["returned_evidence_ids"] = sorted(returned)
        search_step["new_gold_evidence_ids"] = sorted(new_gold)
        search_step["covered_gold_evidence_ids"] = sorted(covered)

    @classmethod
    def _evidence_metrics(
        cls,
        search_steps: list[dict[str, Any]],
        gold_evidence_ids: tuple[str, ...],
        unresolved_gold_facts: tuple[dict[str, Any], ...],
        covered: set[str],
    ) -> dict[str, Any]:
        returned = {
            evidence_id
            for step in search_steps
            for evidence_id in cls._returned_evidence_ids(step)
        }
        gold = set(gold_evidence_ids)
        eligible = not unresolved_gold_facts and bool(gold)
        hits = len(covered)
        return {
            "evidence_metrics_eligible": eligible,
            "gold_evidence_count": len(gold),
            "retrieved_evidence_count": len(returned),
            "supporting_fact_hits": hits,
            "supporting_fact_recall": hits / len(gold) if eligible else None,
            "supporting_fact_precision": (
                hits / len(returned) if eligible and returned else (0.0 if eligible else None)
            ),
            "both_support": covered == gold if eligible else None,
        }

    def _extra_fields(
        self,
        *,
        anchor_obs: str,
        step_kind: str,
        actions: list[str],
        search_steps: list[dict[str, Any]],
        transitions: list[dict[str, Any]],
        qwen_thinking: list[dict[str, Any]],
        sample_key: str,
        dataset_question_id: str,
        official_qid: str,
        gold_evidence_ids: tuple[str, ...],
        unresolved_gold_facts: tuple[dict[str, Any], ...],
        covered: set[str],
    ) -> dict[str, Any]:
        return {
            "anchor_obs": anchor_obs,
            "step_kind": step_kind,
            "executed_search_queries": list(actions),
            "search_steps": list(search_steps),
            "local_reasoning_transitions": list(transitions),
            "qwen_thinking": list(qwen_thinking),
            "thinking_mode": self.thinking_mode,
            "force_first_search": False,
            "final_answer_protocol": LR_FINISH_PROTOCOL,
            "evidence_schema_version": self.search_tool.evidence_schema_version,
            "sample_key": sample_key,
            "base_sample_key": sample_key,
            "dataset_question_id": dataset_question_id,
            "official_qid": official_qid,
            "gold_evidence_ids": list(gold_evidence_ids),
            "unresolved_gold_facts": list(unresolved_gold_facts),
            "evidence_metrics": self._evidence_metrics(
                search_steps, gold_evidence_ids, unresolved_gold_facts, covered
            ),
            "reward_extra_info": {
                "num_tool_steps": len(actions),
                "lr_contract_id": LR_CONTRACT_VERSION,
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
        sample_key = f"{split}:{row_index}"
        dataset_question_id = str(extra_info.get("question_id") or "")
        official_qid = dataset_question_id
        gold_evidence_ids: tuple[str, ...] = ()
        unresolved_gold_facts: tuple[dict[str, Any], ...] = ()
        if self.search_tool.require_sentence_evidence:
            sample = self.search_tool.evidence_store.sample(split, row_index)
            if sample.question != question:
                raise ValueError(f"Evidence sidecar question mismatch for {sample.sample_key}")
            dataset_question_id = sample.dataset_question_id
            official_qid = sample.official_qid
            sample_key = sample.sample_key
            gold_evidence_ids = sample.gold_evidence_ids
            unresolved_gold_facts = sample.unresolved_gold_facts

        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "step_generate_sequences": [],
            "step_tool_calls": [],
        }
        steps = []
        passages: list[tuple[str, Passage]] = []
        actions: list[str] = []
        search_steps: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        transition_flow_step_indices: list[int] = []
        qwen_thinking: list[dict[str, Any]] = []
        covered_gold: set[str] = set()
        feedback_lines: list[str] = []

        for turn in range(1, self.max_steps + 1):
            final_turn = turn == self.max_steps
            step_metrics: dict[str, float] = {}
            prompt_ids, anchor_obs, available_artifacts = self._prompt_ids_within_budget(
                question=question,
                passages=passages,
                actions=actions,
                transitions=transitions,
                feedback=("\n".join(feedback_lines[-3:]) if self.enable_tool_parse_feedback else ""),
                final_turn=final_turn,
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
                raise RuntimeError("vLLM returned an empty A8-LR generation")
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            if self.thinking_mode == "native":
                thinking, visible_text, complete = split_native_thinking(response_text)
            else:
                thinking, visible_text, complete = "", response_text.strip(), True
            qwen_thinking.append({"turn": turn, "content": thinking, "complete": complete})
            calls = extract_tool_calls(visible_text) if complete else []
            call = calls[0] if len(calls) == 1 else None

            finish = parse_finish(visible_text) if actions else None
            terminal = final_turn or (finish is not None and finish.envelope_valid)
            if terminal:
                answer = finish.answer if finish is not None and finish.envelope_valid else None
                if answer is not None:
                    transitions.append(
                        {
                            "reason_step": finish.reason_step,
                            "action_type": "finish",
                            "action_value": answer,
                            "available_artifacts": available_artifacts,
                        }
                    )
                    transition_flow_step_indices.append(len(steps))
                final_reward: float | None = None if answer is not None and actions else 0.0
                final_step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=(
                        output.log_probs[: self.response_length] if output.log_probs else None
                    ),
                    reward_score=final_reward,
                    extra_fields=self._extra_fields(
                        anchor_obs=anchor_obs,
                        step_kind="final",
                        actions=actions,
                        search_steps=search_steps,
                        transitions=transitions,
                        qwen_thinking=qwen_thinking,
                        sample_key=sample_key,
                        dataset_question_id=dataset_question_id,
                        official_qid=official_qid,
                        gold_evidence_ids=gold_evidence_ids,
                        unresolved_gold_facts=unresolved_gold_facts,
                        covered=covered_gold,
                    ),
                )
                final_step = await self._postprocess(final_step, **kwargs)
                terminal_em = float(final_step.reward_score)
                audits = verify_trajectory(
                    transitions,
                    question=question,
                    reward_horizon=self.reward_horizon,
                    dependency_taint_gamma=self.dependency_taint_gamma,
                )
                trajectory_audit = trajectory_audit_record(audits)
                local_reward = float(trajectory_audit["local_reward"])
                weighted_local = 0.0 if is_validation else PRIMARY_CONTRACT.process_weight * local_reward
                for audit, flow_step_index in zip(audits, transition_flow_step_indices, strict=True):
                    weighted_credit = (
                        0.0
                        if is_validation
                        else PRIMARY_CONTRACT.process_weight * audit.raw_local_credit
                    )
                    if flow_step_index < len(steps):
                        steps[flow_step_index].reward_score = weighted_credit
                        step_info = steps[flow_step_index].extra_fields.get("reward_extra_info", {})
                        step_info.update(
                            {
                                "local_reasoning_step": audit.record(),
                                "raw_local_credit": audit.raw_local_credit,
                                "weighted_local_credit": weighted_credit,
                            }
                        )
                        steps[flow_step_index].extra_fields["reward_extra_info"] = step_info
                terminal_component = (
                    terminal_em
                    if is_validation
                    else PRIMARY_CONTRACT.terminal_weight * terminal_em
                )
                finish_weighted_credit = 0.0
                if audits and transition_flow_step_indices[-1] == len(steps):
                    finish_weighted_credit = (
                        0.0
                        if is_validation
                        else PRIMARY_CONTRACT.process_weight * audits[-1].raw_local_credit
                    )
                final_step.reward_score = terminal_component + finish_weighted_credit
                reward_info = final_step.extra_fields.get("reward_extra_info", {})
                reward_info.update(
                    {
                        "acc": terminal_em,
                        "terminal_em": terminal_em,
                        "optimizer_terminal_component": terminal_component,
                        "local_reward": local_reward,
                        "optimizer_local_component": weighted_local,
                        "optimizer_total_reward": terminal_component + weighted_local,
                        "local_reasoning_audit": trajectory_audit,
                        "finish_protocol_valid": bool(finish and finish.envelope_valid),
                        "minimum_search_requirement_met": bool(actions),
                    }
                )
                final_step.extra_fields["reward_extra_info"] = reward_info
                steps.append(final_step)
                self._record_step_timing(metrics, step_metrics)
                break

            query = None
            reason_step: Any = None
            if call is not None and call.get("name") == "search":
                arguments = call.get("arguments")
                if isinstance(arguments, Mapping):
                    raw_query = arguments.get("query")
                    if isinstance(raw_query, str) and raw_query.strip():
                        query = raw_query.strip()
                        reason_step = arguments.get("reason_step")
            if query is not None:
                with simple_timer("tool_calls", step_metrics):
                    search_step = self._do_search(query, assistant_turn=turn)
                search_step["search_index"] = len(search_steps) + 1
                self._attach_gold_audit(
                    search_step,
                    set(gold_evidence_ids),
                    covered_gold,
                    eligible=not unresolved_gold_facts,
                )
                search_steps.append(search_step)
                actions.append(query)
                self._ingest_search_step(query, search_step, passages)
                if len(actions) > 1:
                    transitions.append(
                        {
                            "reason_step": reason_step,
                            "action_type": "search",
                            "action_value": query,
                            "available_artifacts": available_artifacts,
                        }
                    )
                    transition_flow_step_indices.append(len(steps))
                step_kind = "search"
            else:
                feedback_lines.append(
                    "The previous output was not exactly one valid search call with a non-empty query."
                )
                step_kind = "invalid_tool_call"

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=(
                    output.log_probs[: self.response_length] if output.log_probs else None
                ),
                reward_score=0.0,
                extra_fields=self._extra_fields(
                    anchor_obs=anchor_obs,
                    step_kind=step_kind,
                    actions=actions,
                    search_steps=search_steps,
                    transitions=transitions,
                    qwen_thinking=qwen_thinking,
                    sample_key=sample_key,
                    dataset_question_id=dataset_question_id,
                    official_qid=official_qid,
                    gold_evidence_ids=gold_evidence_ids,
                    unresolved_gold_facts=unresolved_gold_facts,
                    covered=covered_gold,
                ),
            )
            steps.append(await self._postprocess(step, **kwargs))
            self._record_step_timing(metrics, step_metrics)

        return AgentFlowOutput(steps=steps, metrics=metrics)

    @staticmethod
    def _record_step_timing(metrics: dict[str, Any], step_metrics: Mapping[str, float]) -> None:
        for name in ("generate_sequences", "tool_calls"):
            elapsed = float(step_metrics.get(name, 0.0))
            metrics[name] += elapsed
            metrics[f"step_{name}"].append(elapsed)

    def _do_search(self, query: str, *, assistant_turn: int) -> dict[str, Any]:
        results = self.search_tool.batch_execute([{"query": query}])
        item = results[0] if results else {"success": False, "content": "missing result"}
        passages = (
            parse_legacy_tool_result(str(item.get("content", "")))
            if item.get("success", False)
            else []
        )
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
            passages.append(
                (
                    query,
                    Passage(
                        pid=int(record["pid"]),
                        title=str(record.get("title", "")),
                        text=str(record.get("text", "")),
                        score=float(record.get("score", 0.0)),
                        sentence_evidence=list(record.get("sentence_evidence") or []),
                    ),
                )
            )
