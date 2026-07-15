"""Shared HotpotQA AgentFlow for formal A0 and future A2.

A0 and A2 must use this exact execution path.  A0 is validation-only and keeps
search-step rewards at zero; the deterministic evidence reward stored here is
audit-only.  A2 can later consume the same per-step audit value without changing
prompting, retrieval, trajectory boundaries, or evidence identities.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
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
from recipes.hotpotqa.prompts import (
    HOTPOTQA_FINAL_TURN_PROMPT,
    HOTPOTQA_SYSTEM_PROMPT,
    HOTPOTQA_TOOL_SCHEMAS,
    HOTPOTQA_USER_PROMPT,
)
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.workers.rollout.llm_server import LLMServerClient
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids

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
            arguments = {
                parameter_name: parameter_value.strip()
                for parameter_name, parameter_value in _HERMES_PARAMETER_BLOCK.findall(function_body)
            }
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


def _format_passage_list(passages: list[tuple[str, Passage]], max_chars: int) -> str:
    """Render only query/title/text; verifier IDs never enter the model prompt."""

    if not passages:
        return "None"
    lines: list[str] = []
    total = 0
    for index, (query, passage) in enumerate(passages, start=1):
        line = f"[{index}] (query: {query}) {passage.text[:1200].replace(chr(10), ' ')}"
        if total + len(line) > max_chars:
            lines.append(f"... ({len(passages) - index + 1} more passages truncated)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _format_history_actions(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Search] {query}" for query in actions)


@register("hotpotqa_agent")
class HotpotQAAgentFlow(AgentFlowBase):
    """One shared, step-preserving search flow for matched A0/A2 arms."""

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
        force_first_value = os.environ.get(
            "HOTPOTQA_FORCE_FIRST_SEARCH", kwargs.get("force_first_search", False)
        )
        self.force_first_search = coerce_bool(
            force_first_value, name="HOTPOTQA_FORCE_FIRST_SEARCH"
        )
        if self.force_first_search:
            raise ValueError("The shared formal HotpotQA AgentFlow does not permit forced searches")

        self.formal_a0 = coerce_bool(
            os.environ.get("HOTPOTQA_FORMAL_A0", False), name="HOTPOTQA_FORMAL_A0"
        )
        self.prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        self.response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        self.enable_tool_parse_feedback = bool(kwargs.get("enable_tool_parse_feedback", True))
        self.tool_parser_name = str(self.config.actor_rollout_ref.rollout.multi_turn.format)
        self.tool_parser = ToolParser.get_tool_parser(self.tool_parser_name, self.tokenizer)
        self.tool_schemas = HOTPOTQA_TOOL_SCHEMAS

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
        if self.formal_a0:
            failures: list[str] = []
            if self.max_steps != 4:
                failures.append(f"max_steps={self.max_steps}, expected 4")
            if self.max_parallel_calls != 1:
                failures.append(f"max_parallel_calls={self.max_parallel_calls}, expected 1")
            if self.thinking_mode != "disabled":
                failures.append("thinking must be disabled for the formal A0/A2 contract")
            if not self.search_tool.require_sentence_evidence:
                failures.append("official sentence evidence must be required")
            if self.search_tool.evidence_schema_version != EVIDENCE_SCHEMA_VERSION:
                failures.append("official evidence schema is not active")
            if failures:
                raise ValueError("Formal A0 invariant failure: " + "; ".join(failures))

    def _build_messages(
        self,
        question: str,
        passages: list[tuple[str, Passage]],
        actions: list[str],
        tool_feedback: str,
        *,
        final_turn: bool,
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
        return [
            {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _apply_text_chat_template(self, messages: list[dict[str, str]]) -> list[int]:
        """Use the synchronous text path that is stable for Qwen3.5 templates."""

        return normalize_token_ids(
            apply_chat_template(
                self.tokenizer,
                messages,
                tools=self.tool_schemas,
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
    ) -> tuple[list[int], str]:
        max_chars = self.prompt_length * 3
        min_chars = 400
        while True:
            messages = self._build_messages(
                question,
                passages,
                actions,
                feedback,
                final_turn=final_turn,
                max_passage_chars=max_chars,
            )
            prompt_ids = self._apply_text_chat_template(messages)
            if len(prompt_ids) <= self.prompt_length:
                return prompt_ids, messages[1]["content"]
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
        returned = set(cls._returned_evidence_ids(search_step))
        new_gold = (returned & gold_evidence_ids) - covered_gold_evidence_ids
        covered_gold_evidence_ids.update(returned & gold_evidence_ids)
        search_step["returned_evidence_ids"] = sorted(returned)
        search_step["new_gold_evidence_ids"] = sorted(new_gold)
        search_step["covered_gold_evidence_ids"] = sorted(covered_gold_evidence_ids)
        search_step["audit_process_reward"] = (
            len(new_gold) / len(gold_evidence_ids)
            if evidence_metrics_eligible and gold_evidence_ids
            else None
        )

    def _evidence_metrics(
        self,
        search_steps: list[dict[str, Any]],
        gold_evidence_ids: set[str],
        covered_gold_evidence_ids: set[str],
        unresolved_gold_facts: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        returned = {
            evidence_id
            for step in search_steps
            for evidence_id in self._returned_evidence_ids(step)
        }
        hits = len(covered_gold_evidence_ids)
        eligible = not unresolved_gold_facts and bool(gold_evidence_ids)
        return {
            "evidence_metrics_eligible": eligible,
            "gold_evidence_count": len(gold_evidence_ids),
            "unresolved_gold_fact_count": len(unresolved_gold_facts),
            "retrieved_evidence_count": len(returned),
            "supporting_fact_hits": hits,
            "supporting_fact_recall": (
                hits / len(gold_evidence_ids) if eligible else None
            ),
            "supporting_fact_precision": (
                hits / len(returned) if eligible and returned else (0.0 if eligible else None)
            ),
            "at_least_one_support": hits > 0 if eligible else None,
            "both_support": (
                covered_gold_evidence_ids == gold_evidence_ids if eligible else None
            ),
            "complete_coverage_search_step": next(
                (
                    int(step["search_index"])
                    for step in search_steps
                    if eligible
                    and set(step.get("covered_gold_evidence_ids") or []) == gold_evidence_ids
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
            "evidence_schema_version": self.search_tool.evidence_schema_version,
            "sample_key": sample_key,
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
            },
        }

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        question = str(raw_prompt[0]["content"]).strip()
        extra_info = kwargs.get("extra_info") or {}
        if not isinstance(extra_info, Mapping):
            raise ValueError("HotpotQA extra_info must be a mapping")
        split = str(extra_info.get("split") or "validation")
        try:
            row_index = int(extra_info.get("index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("HotpotQA extra_info.index is required") from exc

        dataset_question_id = str(extra_info.get("question_id") or "")
        official_qid = dataset_question_id
        sample_key = f"{split}:{row_index}"
        gold_evidence_ids: tuple[str, ...] = ()
        unresolved_gold_facts: tuple[dict[str, Any], ...] = ()
        if self.search_tool.require_sentence_evidence:
            sample = self.search_tool.evidence_store.sample(split, row_index)
            if sample.question != question:
                raise ValueError(f"Evidence sidecar question mismatch for {sample.sample_key}")
            if dataset_question_id and sample.dataset_question_id != dataset_question_id:
                raise ValueError(
                    f"Evidence sidecar dataset question ID mismatch for {sample.sample_key}"
                )
            dataset_question_id = sample.dataset_question_id
            official_qid = sample.official_qid
            sample_key = sample.sample_key
            gold_evidence_ids = sample.gold_evidence_ids
            unresolved_gold_facts = sample.unresolved_gold_facts
        elif self.formal_a0:
            raise RuntimeError("Formal A0 requires the official evidence sidecar")

        metrics: dict[str, Any] = {}
        steps: list[AgentFlowStep] = []
        passages: list[tuple[str, Passage]] = []
        history_actions: list[str] = []
        search_steps: list[dict[str, Any]] = []
        qwen_thinking: list[dict[str, Any]] = []
        covered_gold_evidence_ids: set[str] = set()
        feedback_lines: list[str] = []

        for step_number in range(1, self.max_steps + 1):
            final_turn = step_number == self.max_steps
            feedback = "\n".join(feedback_lines[-3:]) if self.enable_tool_parse_feedback else ""
            prompt_ids, anchor_obs = self._prompt_ids_within_budget(
                question,
                passages,
                history_actions,
                feedback,
                final_turn=final_turn,
            )

            step_sampling_params = dict(sampling_params)
            step_sampling_params["max_tokens"] = min(
                int(step_sampling_params.get("max_tokens", self.response_length)),
                self.response_length,
            )
            with simple_timer("generate_sequences", metrics):
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

            # The fourth turn is always terminal.  A tool call here is retained
            # in the response for failure analysis but is never executed.
            if final_turn or (not tool_calls and "<tool_call>" not in visible_text):
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=(
                        output.log_probs[: self.response_length] if output.log_probs else None
                    ),
                    reward_score=None,
                    extra_fields=self._make_extra_fields(
                        anchor_obs=anchor_obs,
                        history_actions=history_actions,
                        search_steps=search_steps,
                        qwen_thinking=qwen_thinking,
                        sample_key=sample_key,
                        dataset_question_id=dataset_question_id,
                        official_qid=official_qid,
                        gold_evidence_ids=gold_evidence_ids,
                        unresolved_gold_facts=unresolved_gold_facts,
                        covered_gold_evidence_ids=covered_gold_evidence_ids,
                        step_kind="final",
                    ),
                )
                step = await self._postprocess(step, **kwargs)
                reward_info = step.extra_fields.get("reward_extra_info", {})
                step.extra_fields["reward_extra_info"] = {
                    "num_tool_steps": len(history_actions),
                    "acc": float(reward_info.get("acc", step.reward_score or 0.0)),
                }
                steps.append(step)
                break

            if valid_queries:
                query = valid_queries[0]
                with simple_timer("tool_calls", metrics):
                    search_step = self._do_search(query, assistant_turn=step_number)
                search_step["search_index"] = len(search_steps) + 1
                self._attach_evidence_audit(
                    search_step,
                    set(gold_evidence_ids),
                    covered_gold_evidence_ids,
                    evidence_metrics_eligible=not unresolved_gold_facts,
                )
                search_steps.append(search_step)
                history_actions.append(query)
                self._ingest_search_step(query, search_step, passages)
                step_kind = "search"
            else:
                feedback_lines.append(
                    "The previous tool call was invalid. Use exactly one search call with a non-empty "
                    "string parameter named query."
                )
                step_kind = "invalid_tool_call"

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=(
                    output.log_probs[: self.response_length] if output.log_probs else None
                ),
                reward_score=0.0,
                extra_fields=self._make_extra_fields(
                    anchor_obs=anchor_obs,
                    history_actions=history_actions,
                    search_steps=search_steps,
                    qwen_thinking=qwen_thinking,
                    sample_key=sample_key,
                    dataset_question_id=dataset_question_id,
                    official_qid=official_qid,
                    gold_evidence_ids=gold_evidence_ids,
                    unresolved_gold_facts=unresolved_gold_facts,
                    covered_gold_evidence_ids=covered_gold_evidence_ids,
                    step_kind=step_kind,
                ),
            )
            steps.append(await self._postprocess(step, **kwargs))

        return AgentFlowOutput(steps=steps, metrics=metrics)

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
            passage = Passage(
                pid=int(record["pid"]),
                title=str(record.get("title", "")),
                text=str(record.get("text", "")),
                score=float(record.get("score", 0.0)),
                sentence_evidence=list(record.get("sentence_evidence") or []),
            )
            if not any(existing.pid == passage.pid for _, existing in passages):
                passages.append((query, passage))
