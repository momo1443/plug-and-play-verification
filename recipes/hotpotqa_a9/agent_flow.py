"""Certificate-grounded AgentFlow for A9 HotpotQA experiments."""

from __future__ import annotations

import json
import logging
import os
import random
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
from recipes.hotpotqa_a9.prompts import (
    A9_FINAL_TURN_PROMPT,
    A9_FINISH_AVAILABLE_PROMPT,
    A9_SYSTEM_PROMPT,
    A9_USER_PROMPT,
    BOOTSTRAP_TOOL_SCHEMAS,
    FINISH_TOOL_SCHEMAS,
    SEARCH_OR_FINISH_TOOL_SCHEMAS,
)
from recipes.hotpotqa_a9.protocol import (
    A9_FINISH_PROTOCOL,
    parse_finish_call,
    parse_search_call,
)
from recipes.hotpotqa_a9.reward_contract import (
    A9_CONTRACT_VERSION,
    CONTRACT_CERT_MIX,
    CONTRACT_FORMAT_STRICT,
)
from recipes.hotpotqa_a9.verifier import (
    trajectory_audit_record,
    verify_trajectory,
)
from recipes.hotpotqa_lr.protocol import extract_tool_calls
from recipes.hotpotqa_lr.verifier import (
    artifact_content_sha256,
    artifact_id_for_passage,
)
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _coerce_nonnegative_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return parsed


def optimizer_reward_schedule(
    *,
    global_step: int,
    is_validation: bool,
    em_warmup_steps: int,
    terminal_weight: float,
    process_weight: float,
    format_gate: bool = False,
) -> tuple[float, float, str]:
    """Return (terminal_weight, process_weight, phase_label) for this step.

    Post-warmup behaviour depends on format_gate:
    - format_gate=False (cert_mix): uses the fixed weights passed via
      terminal_weight / process_weight (typically 0.4/0.6).
    - format_gate=True (format_strict): samples w ~ U(0,1) per trajectory,
      giving terminal_weight=w and process_weight=1-w.  This provides
      variance that encourages the model to value both signals, while the
      format gate ensures protocol compliance.
    """
    if is_validation:
        return 1.0, 0.0, "validation_terminal_em"
    if em_warmup_steps > 0 and 0 < global_step <= em_warmup_steps:
        return 1.0, 0.0, "em_warmup"
    if format_gate:
        w = random.random()
        return w, 1.0 - w, "certificate_uniform"
    return terminal_weight, process_weight, "certificate_fixed"


def _format_history(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Search {index}] {query}" for index, query in enumerate(actions, start=1))


def _format_certificate_ledger(transitions: list[dict[str, Any]]) -> str:
    """Format previously parsed certificates for the prompt."""
    if not transitions:
        return "None"
    lines = []
    for transition in transitions:
        cert = transition.get("certificate_parsed")
        action_type = transition.get("action_type", "?")
        action_value = transition.get("action_value", "?")
        if cert is not None:
            lines.append(
                json.dumps(
                    {"action": {"type": action_type, "value": action_value}, "certificate": cert},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            lines.append(
                json.dumps(
                    {"action": {"type": action_type, "value": action_value}, "certificate": None},
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


@register("hotpotqa_certificate_agent")
class HotpotQACertificateAgentFlow(AgentFlowBase):
    """Four-turn HotpotQA flow with certificate-grounded process reward."""

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
        self.reward_horizon = int(kwargs.get("reward_horizon", CONTRACT_CERT_MIX.reward_horizon))
        self.contract = CONTRACT_CERT_MIX
        self.em_warmup_steps = _coerce_nonnegative_int(
            kwargs.get("em_warmup_steps", os.environ.get("HOTPOTQA_A9_EM_WARMUP_STEPS", 50)),
            name="em_warmup_steps",
        )
        self.format_penalty = float(
            kwargs.get("format_penalty",
                       os.environ.get("HOTPOTQA_A9_FORMAT_PENALTY", 0.1))
        )
        self.contract = CONTRACT_FORMAT_STRICT if coerce_bool(
            os.environ.get("HOTPOTQA_A9_FORMAT_GATE", "0"),
            name="HOTPOTQA_A9_FORMAT_GATE",
        ) else CONTRACT_CERT_MIX
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
        if self.reward_horizon != CONTRACT_CERT_MIX.reward_horizon:
            failures.append(f"reward_horizon={self.reward_horizon}, expected 3")
        if self.thinking_mode != "disabled":
            failures.append("thinking must be disabled")
        if not self.search_tool.require_sentence_evidence:
            failures.append("official sentence evidence must be required")
        if self.search_tool.evidence_schema_version != EVIDENCE_SCHEMA_VERSION:
            failures.append("official evidence schema is not active")
        if failures:
            raise ValueError("A9 invariant failure: " + "; ".join(failures))

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
            user_content = A9_USER_PROMPT.format(
                user_query=question,
                history_actions=_format_history(actions),
                passage_list=passage_text,
                certificate_ledger=_format_certificate_ledger(transitions),
                tool_feedback=feedback.strip() or "None",
            )
            if final_turn:
                user_content += f"\n\n{A9_FINAL_TURN_PROMPT}"
            elif actions:
                user_content += f"\n\n{A9_FINISH_AVAILABLE_PROMPT}"
            messages = [
                {"role": "system", "content": A9_SYSTEM_PROMPT},
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
                    f"A9 prompt has {len(prompt_ids)} tokens; limit is {self.prompt_length}"
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
        is_validation: bool = False,
    ) -> dict[str, Any]:
        return {
            "anchor_obs": anchor_obs,
            "step_kind": step_kind,
            "executed_search_queries": list(actions),
            "search_steps": list(search_steps),
            "certificate_transitions": list(transitions),
            "qwen_thinking": list(qwen_thinking),
            "thinking_mode": self.thinking_mode,
            "force_first_search": False,
            "final_answer_protocol": A9_FINISH_PROTOCOL,
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
            "_agent_r1_is_validation": is_validation,
            "reward_extra_info": {
                "num_tool_steps": len(actions),
                "a9_contract_id": A9_CONTRACT_VERSION,
                "judge_invalid": False,
                "a9_format_invalid": False,
            },
        }

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        question = str(raw_prompt[0]["content"]).strip()
        is_validation = bool(kwargs.get("_agent_r1_is_validation", False))
        global_step = int(kwargs.get("_agent_r1_global_step", -1))
        terminal_weight, process_weight, reward_phase = optimizer_reward_schedule(
            global_step=global_step,
            is_validation=is_validation,
            em_warmup_steps=self.em_warmup_steps,
            terminal_weight=self.contract.terminal_weight,
            process_weight=self.contract.process_weight,
            format_gate=self.contract.format_gate,
        )
        weight_sampling = "uniform_0_1" if reward_phase == "certificate_uniform" else "fixed"
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
                raise RuntimeError("vLLM returned an empty A9 generation")
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            if self.thinking_mode == "native":
                thinking, visible_text, complete = split_native_thinking(response_text)
            else:
                thinking, visible_text, complete = "", response_text.strip(), True
            qwen_thinking.append({"turn": turn, "content": thinking, "complete": complete})
            calls = extract_tool_calls(visible_text) if complete else []
            call = calls[0] if len(calls) == 1 else None

            # ── Finish detection ───────────────────────────────────────────
            # Parse the finish call if the model has searched at least once.
            # The key A9 invariant: the answer is always extracted regardless
            # of whether the certificate parses successfully.
            finish = parse_finish_call(visible_text) if actions else None
            terminal = final_turn or (finish is not None and finish.answer is not None)
            if terminal:
                answer = finish.answer if finish is not None else None
                if answer is not None:
                    transitions.append(
                        {
                            "certificate": finish.certificate_raw if finish else None,
                            "certificate_parsed": (
                                finish.certificate.record() if finish and finish.certificate else None
                            ),
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
                        is_validation=is_validation,
                    ),
                )
                final_step = await self._postprocess(final_step, **kwargs)
                terminal_em = float(final_step.reward_score)
                # Verify trajectory and backfill process credit to transition steps.
                audits = verify_trajectory(
                    transitions,
                    reward_horizon=self.reward_horizon,
                )
                trajectory_audit = trajectory_audit_record(audits)
                local_reward = float(trajectory_audit["local_reward"])
                weighted_local = process_weight * local_reward
                for audit, flow_step_index in zip(audits, transition_flow_step_indices, strict=True):
                    weighted_credit = process_weight * audit.raw_local_credit
                    if flow_step_index < len(steps):
                        steps[flow_step_index].reward_score = weighted_credit
                        step_info = steps[flow_step_index].extra_fields.get("reward_extra_info", {})
                        step_info.update(
                            {
                                "certificate_step_audit": audit.record(),
                                "raw_local_credit": audit.raw_local_credit,
                                "weighted_local_credit": weighted_credit,
                                "optimizer_reward_phase": reward_phase,
                                "optimizer_process_weight": process_weight,
                                "optimizer_terminal_weight": terminal_weight,
                                "weight_sampling": weight_sampling,
                                "training_global_step": global_step,
                                "a9_process_component_valid": audit.own_valid > 0,
                                "optimizer_process_component": weighted_credit,
                            }
                        )
                        steps[flow_step_index].extra_fields["reward_extra_info"] = step_info
                terminal_component = terminal_weight * terminal_em
                finish_weighted_credit = 0.0
                if audits and transition_flow_step_indices[-1] == len(steps):
                    finish_weighted_credit = process_weight * audits[-1].raw_local_credit
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
                        "optimizer_reward_phase": reward_phase,
                        "verifier_timing": "terminal_trajectory_replay",
                        "process_credit_application": "backfill_to_transition_steps",
                        "optimizer_terminal_weight": terminal_weight,
                        "optimizer_process_weight": process_weight,
                        "weight_sampling": weight_sampling,
                        "a9_em_warmup_steps": self.em_warmup_steps,
                        "training_global_step": global_step,
                        "certificate_audit": trajectory_audit,
                        "finish_protocol_valid": bool(finish and finish.answer is not None),
                        "minimum_search_requirement_met": bool(actions),
                        "a9_process_component_valid": any(
                            a.own_valid > 0 for a in audits
                        ),
                        "optimizer_process_component": weighted_local,
                    }
                )
                final_step.extra_fields["reward_extra_info"] = reward_info
                # ── format_gate: zero all reward when finish is missing ────
                # If format_gate is enabled and the model failed to produce a
                # valid finish tool call (answer is None), zero out reward for
                # every step in the trajectory so the model learns that it MUST
                # write a compliant finish call to earn any credit.  Suppressed
                # during EM warmup to avoid cold-start deadlock.
                format_gate_active = (
                    self.contract.format_gate
                    and reward_phase != "em_warmup"
                    and (finish is None or finish.answer is None)
                )
                if format_gate_active:
                    final_step.reward_score = 0.0
                    for step in steps:
                        step.reward_score = 0.0
                    reward_info["format_gate_triggered"] = True
                steps.append(final_step)
                self._record_step_timing(metrics, step_metrics)
                break

            # ── Search detection ───────────────────────────────────────────
            # The key A9 invariant: the query is always extracted regardless
            # of whether the certificate parses successfully.
            query = None
            search_cert_raw: Any = None
            search_cert_parsed: dict[str, Any] | None = None
            if call is not None and call.get("name") == "search":
                search_audit = parse_search_call(visible_text)
                query = search_audit.query
                search_cert_raw = search_audit.certificate_raw
                if search_audit.certificate is not None:
                    search_cert_parsed = search_audit.certificate.record()
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
                            "certificate": search_cert_raw,
                            "certificate_parsed": search_cert_parsed,
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
                reward_score=-self.format_penalty if step_kind == "invalid_tool_call" else 0.0,
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
                    is_validation=is_validation,
                ),
            )
            if step_kind == "invalid_tool_call":
                step.extra_fields["reward_extra_info"]["a9_format_invalid"] = True
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
        existing_pids = {p.pid for _, p in passages}
        for record in search_step.get("returned_evidence") or []:
            pid = int(record["pid"])
            if pid in existing_pids:
                continue
            existing_pids.add(pid)
            passages.append(
                (
                    query,
                    Passage(
                        pid=pid,
                        title=str(record.get("title", "")),
                        text=str(record.get("text", "")),
                        score=float(record.get("score", 0.0)),
                        sentence_evidence=list(record.get("sentence_evidence") or []),
                    ),
                )
            )
