"""Five-turn write/test/revise/test/submit A9 agent flow for TACO."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from agent_r1.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from agent_r1.reward_loop.reward_loop import RewardLoopWorker
from agent_r1.verifier import (
    UniformRewardSchedule,
    VerificationResult,
    apply_composed_reward,
    compose_verification_reward,
)
from agent_r1.evaluation.consistency import consistency_record
from recipes.llm_judge.scoring import ProcessStep, judge_reward_info, verify_process_steps
from recipes.taco_a9.dsl import CodeArtifact, ExecutionRecord
from recipes.taco_a9.protocol import parse_tool_call
from recipes.taco_a9.prompts import (
    FINAL_TURN_PROMPT,
    SUBMIT_ONLY_TOOLS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    WRITE_ONLY_TOOLS,
    WRITE_OR_RUN_OR_SUBMIT_TOOLS,
)
from recipes.taco_a9.sandbox import BubblewrapPythonExecutor, TestCase, TestSuite
from recipes.taco_a9.reward_contract import reward_schedule
from recipes.taco_a9.verifier import verify_process
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.utils.profiler import simple_timer
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TacoTaskStore:
    """Runner-only test sidecar. It is never inserted into an actor prompt."""

    def __init__(self, sidecar_path: str, sidecar_index_path: str) -> None:
        path = Path(sidecar_path)
        index_path = Path(sidecar_index_path)
        if not path.is_file():
            raise FileNotFoundError(f"TACO A9 sidecar is unavailable: {path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"TACO A9 sidecar index is unavailable: {index_path}")
        self._path = path
        self._offsets: dict[str, tuple[int, int]] = json.loads(index_path.read_text(encoding="utf-8"))
        if not self._offsets:
            raise ValueError("TACO A9 sidecar index is empty")

    def _task(self, task_id: str) -> dict[str, Any]:
        location = self._offsets.get(task_id)
        if location is None:
            raise KeyError(f"TACO A9 task missing from sidecar: {task_id}")
        offset, length = location
        with self._path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
        task = json.loads(payload)
        if task.get("task_id") != task_id:
            raise ValueError(f"TACO A9 sidecar index mismatch for task: {task_id}")
        return task

    def suites(self, task_id: str) -> tuple[TestSuite, TestSuite]:
        task = self._task(task_id)
        def build(suite_id: str, cases: list[dict[str, str]]) -> TestSuite:
            return TestSuite(
                suite_id=suite_id,
                cases=tuple(TestCase(stdin=str(case["stdin"]), expected_stdout=str(case["expected_stdout"])) for case in cases),
            )
        return build("developer_v1", task["developer_cases"]), build("private_v1", task["private_cases"])


def _tool_schemas(turn: int, max_steps: int) -> list[dict[str, Any]]:
    if turn == 1:
        return WRITE_ONLY_TOOLS
    if turn == max_steps:
        return SUBMIT_ONLY_TOOLS
    return WRITE_OR_RUN_OR_SUBMIT_TOOLS


def _ledger_text(records: dict[str, ExecutionRecord]) -> str:
    if not records:
        return "None"
    return "\n".join(json.dumps(record.record(), sort_keys=True) for record in records.values())


def _artifact_text(artifact: CodeArtifact | None, max_chars: int) -> str:
    if artifact is None:
        return "None"
    code = artifact.code
    if len(code) > max_chars:
        code = code[:max_chars] + "\n# ... code truncated in prompt ..."
    return f"artifact_id={artifact.artifact_id}\nsha256={artifact.sha256}\n```python\n{code}\n```"


def _feedback_for_record(record: ExecutionRecord, payload: Mapping[str, Any]) -> str:
    failed = next((item for item in payload.get("outcomes", []) if not item.get("passed")), None)
    suffix = "all developer tests passed" if failed is None else (
        "first failure "
        f"case={failed.get('case_index')}, input={str(failed.get('stdin', ''))[:500]!r}, "
        f"expected={str(failed.get('expected_stdout', ''))[:500]!r}, "
        f"actual={str(failed.get('stdout', ''))[:500]!r}, stderr={str(failed.get('stderr', ''))[:500]!r}"
    )
    return f"{record.run_id}: passed {record.passed}/{record.total}; {suffix}"


@register("taco_a9_code_agent")
@register("taco_llm_judge_code_agent")
class TacoA9CodeAgentFlow(AgentFlowBase):
    """A9 coding flow with prompt-group-shared terminal/certificate weights."""

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
        self.judge_server = kwargs.pop("shared_judge_server", None)
        super().__init__(
            trainer_config,
            server_manager,
            reward_loop_worker,
            tokenizer,
            processor,
            dataset_cls,
            dataset_config,
        )
        self.max_steps = int(kwargs.get("max_steps", 5))
        self.prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        self.response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        self.max_code_prompt_chars = int(kwargs.get("max_code_prompt_chars", 12000))
        self.terminal_warmup_steps = int(kwargs.get("terminal_warmup_steps", 50))
        self.reward_mode = str(kwargs.get("reward_mode", "uniform_certificate"))
        self.store = TacoTaskStore(
            str(kwargs.get("sidecar_path") or os.environ.get("TACO_A9_SIDECAR", "")),
            str(kwargs.get("sidecar_index_path") or os.environ.get("TACO_A9_SIDECAR_INDEX", "")),
        )
        self.executor = BubblewrapPythonExecutor(timeout_seconds=float(kwargs.get("timeout_seconds", 2.0)))
        if self.reward_mode == "uniform_certificate" and self.max_steps != 5:
            raise ValueError("TACO A9 requires exactly five agent turns")
        if self.reward_mode == "terminal_private_binary" and self.max_steps < 4:
            raise ValueError("TACO A1 requires at least four agent turns")
        if self.reward_mode == "llm_judge" and self.judge_server is None:
            raise ValueError("TACO llm_judge mode requires a shared Judge backend")
        if self.reward_mode not in {"uniform_certificate", "terminal_private_binary", "llm_judge"}:
            raise ValueError(f"Unsupported TACO reward mode: {self.reward_mode}")

    async def _generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], metrics: dict[str, float]):
        with simple_timer("generate_sequences", metrics):
            return await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

    async def _run_suite(self, artifact: CodeArtifact, suite: TestSuite, ordinal: int, metrics: dict[str, float]):
        with simple_timer("tool_calls", metrics):
            return await self.loop.run_in_executor(None, lambda: self.executor.run_suite(artifact, suite, run_ordinal=ordinal))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        question = str(raw_prompt[-1]["content"]).strip()
        extra_info = kwargs.get("extra_info") or {}
        if not isinstance(extra_info, Mapping):
            raise ValueError("TACO A9 extra_info must be a mapping")
        task_id = str(extra_info.get("question_id") or "")
        developer_suite, private_suite = self.store.suites(task_id)
        is_validation = bool(kwargs.get("_agent_r1_is_validation", False))
        global_step = int(kwargs.get("_agent_r1_global_step", -1))
        if self.reward_mode == "terminal_private_binary":
            schedule = UniformRewardSchedule(
                1.0,
                0.0,
                "terminal_private_binary_only",
                "fixed",
                None,
            )
        else:
            schedule = reward_schedule(
                global_step=global_step,
                is_validation=is_validation,
                terminal_warmup_steps=self.terminal_warmup_steps,
                prompt_group_key=task_id,
            )
        terminal_weight = schedule.terminal_weight
        process_weight = schedule.process_weight
        phase = schedule.phase
        weight_sampling = schedule.weight_sampling

        artifacts: dict[str, CodeArtifact] = {}
        artifact_step_indices: dict[str, int] = {}
        records: dict[str, ExecutionRecord] = {}
        record_payloads: dict[str, dict[str, Any]] = {}
        current_artifact: CodeArtifact | None = None
        submitted_artifact: CodeArtifact | None = None
        submission_certificate: Any = None
        feedback: list[str] = []
        steps: list[AgentFlowStep] = []
        judge_steps: list[ProcessStep] = []
        action_checks: list[bool] = []
        terminal_info: dict[str, Any] = {}
        outcome_reward = 0.0
        run_step_indices: dict[str, int] = {}
        metrics: dict[str, Any] = {"generate_sequences": 0.0, "tool_calls": 0.0, "step_generate_sequences": [], "step_tool_calls": []}

        for turn in range(1, self.max_steps + 1):
            step_metrics: dict[str, float] = {}
            user_content = USER_PROMPT.format(
                question=question,
                current_code=_artifact_text(current_artifact, self.max_code_prompt_chars),
                execution_ledger=_ledger_text(records),
                feedback="\n".join(feedback[-3:]) or "None",
            )
            if turn == self.max_steps:
                user_content += "\n\n" + FINAL_TURN_PROMPT
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
            prompt_ids = await self.apply_chat_template(messages, tools=_tool_schemas(turn, self.max_steps))
            step_sampling = dict(sampling_params)
            step_sampling["max_tokens"] = min(int(step_sampling.get("max_tokens", self.response_length)), self.response_length)
            output = await self._generate(prompt_ids, step_sampling, step_metrics)
            response_ids = list(output.token_ids[: self.response_length])
            if not response_ids:
                raise RuntimeError("TACO A9 received an empty generation")
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            call = parse_tool_call(response_text)
            action_name = call.name or "invalid"
            action_args = call.arguments
            step_kind = action_name

            if action_name == "write_code" and isinstance(action_args.get("code"), str) and action_args["code"].strip():
                current_artifact = CodeArtifact.create(len(artifacts) + 1, action_args["code"])
                artifacts[current_artifact.artifact_id] = current_artifact
                artifact_step_indices[current_artifact.artifact_id] = turn
                feedback.append(f"Created {current_artifact.artifact_id} with sha256={current_artifact.sha256}.")
                reward_score = 0.0
            elif action_name == "run_developer_tests" and isinstance(action_args.get("code_artifact_id"), str):
                artifact = artifacts.get(action_args["code_artifact_id"])
                if artifact is None:
                    feedback.append("Unknown code artifact id. Write code before testing it.")
                    reward_score = 0.0
                    step_kind = "invalid_run"
                else:
                    record, payload = await self._run_suite(artifact, developer_suite, len(records) + 1, step_metrics)
                    records[record.run_id] = record
                    record_payloads[record.run_id] = payload
                    run_step_indices[record.run_id] = len(steps) + 1
                    feedback.append(_feedback_for_record(record, payload))
                    reward_score = 0.0
            elif action_name == "submit":
                artifact_id = action_args.get("code_artifact_id")
                submitted_artifact = artifacts.get(artifact_id) if isinstance(artifact_id, str) else None
                submission_certificate = action_args.get("certificate") if isinstance(action_args, dict) else None
                private_case_pass_rate = 0.0
                outcome_reward = 0.0
                if submitted_artifact is not None:
                    private_record, _ = await self._run_suite(submitted_artifact, private_suite, len(records) + 1, step_metrics)
                    private_case_pass_rate = (
                        private_record.passed / private_record.total if private_record.total else 0.0
                    )
                    outcome_reward = float(private_record.all_passed)

                process_credit = 0.0
                certificate_score = 0.0
                verification = VerificationResult(
                    credits=(), audit={"verifier": "disabled", "credits_by_step": {}}
                )
                should_audit_process = True  # Fixed deterministic evaluation for every reward arm.
                if should_audit_process:
                    verification = await self.loop.run_in_executor(
                        None,
                        lambda: verify_process(
                            artifacts=artifacts,
                            records=records,
                            raw_certificate=submission_certificate,
                            submitted_artifact=submitted_artifact,
                            developer_suite=developer_suite,
                            executor=self.executor,
                            run_step_indices=run_step_indices,
                            final_step_index=len(steps) + 1,
                        ),
                    )
                    certificate_score = float(
                        verification.audit["certificate_audit"]["raw_local_credit"]
                    )
                    process_credit = verification.process_reward
                composed_reward = compose_verification_reward(
                    terminal_reward=outcome_reward,
                    verification=verification,
                    schedule=schedule,
                    terminal_gate_passed=submitted_artifact is not None,
                )
                reward_score = 0.0
                step_kind = "submit"
                terminal_info = {
                    **consistency_record(verification, outcome_reward, eligible=submitted_artifact is not None, extra_checks=action_checks),
                    "acc": outcome_reward,
                    "terminal_private_all_pass": outcome_reward,
                    "terminal_private_pass_rate": private_case_pass_rate,
                    "verified_process_reward": process_credit,
                    "artifact_process_audits": verification.audit.get(
                        "artifact_process_audits", []
                    ),
                    "certificate_audit": verification.audit.get("certificate_audit"),
                    "certificate_score": certificate_score,
                    "reward_mode": self.reward_mode,
                    "verifier_timing": "terminal_trajectory_replay",
                    "training_global_step": global_step,
                }
            else:
                feedback.append("Output exactly one valid tool call allowed for this turn.")
                reward_score = 0.0
                step_kind = "invalid_tool_call"

            if self.reward_mode == "llm_judge" and process_weight > 0.0 and action_name != "submit":
                judge_steps.append(ProcessStep(
                    step_index=turn, action=response_text,
                    observation={"feedback": feedback[-1] if feedback else None,
                                 "code": _artifact_text(
                                     artifact if step_kind == "run_developer_tests" else current_artifact,
                                     self.max_code_prompt_chars,
                                 )},
                    valid=step_kind in {"write_code", "run_developer_tests"},
                ))

            action_checks.append(step_kind in {"write_code", "run_developer_tests", "submit"})
            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=reward_score,
                num_turns=turn,
                extra_fields={
                    "anchor_obs": user_content,
                    "raw_prompt": kwargs["raw_prompt"],
                    "taco_task_id": task_id,
                    "code_artifacts": [artifact.record() for artifact in artifacts.values()],
                    "execution_ledger": [record.record() for record in records.values()],
                    "step_kind": step_kind,
                    "_agent_r1_is_validation": is_validation,
                    "reward_extra_info": terminal_info if action_name == "submit" else {
                        "optimizer_reward_phase": phase,
                        "optimizer_terminal_weight": terminal_weight,
                        "optimizer_process_weight": process_weight,
                        "weight_sampling": weight_sampling,
                        "optimizer_weight_group_key": task_id,
                        "training_global_step": global_step,
                    },
                },
            )
            steps.append(await self._postprocess(step, **kwargs))
            if action_name == "submit" and self.reward_mode != "llm_judge":
                apply_composed_reward(
                    steps,
                    composed_reward,
                    extra_final_info=terminal_info,
                )
            metrics["generate_sequences"] += step_metrics.get("generate_sequences", 0.0)
            metrics["tool_calls"] += step_metrics.get("tool_calls", 0.0)
            metrics["step_generate_sequences"].append(step_metrics.get("generate_sequences", 0.0))
            metrics["step_tool_calls"].append(step_metrics.get("tool_calls", 0.0))
            if action_name == "submit":
                break

        if not steps:
            raise RuntimeError("TACO A9 produced no agent steps")
        if self.reward_mode == "llm_judge":
            verification = (
                await verify_process_steps(
                    self.judge_server, domain="taco", question=question,
                    steps=judge_steps, max_steps=self.max_steps,
                ) if process_weight > 0.0 else VerificationResult(credits=(), audit={})
            )
            composed = compose_verification_reward(
                terminal_reward=outcome_reward, verification=verification, schedule=schedule,
                terminal_gate_passed=submitted_artifact is not None,
            )
            terminal_info.update({"acc": outcome_reward, "terminal_private_all_pass": outcome_reward,
                                  "reward_mode": "llm_judge",
                                  "verified_process_reward": verification.process_reward,
                                  "verifier_timing": "causal_prefix_backfill",
                                  **judge_reward_info(verification)})
            apply_composed_reward(steps, composed, extra_final_info=terminal_info)
        # Failed/no-submit traces remain in the Eq. (9) denominator.
        final_info = steps[-1].extra_fields.setdefault("reward_extra_info", {})
        if "verification_protocol" not in final_info:
            final_info.update(consistency_record(
                VerificationResult(credits=(), audit={"applicable_checks": [0]}),
                0.0, eligible=False,
            ))
        final_info.update({
            "revision_count": max(0, len(artifacts) - 1),
            "developer_test_count": len(records),
            "retest_count": sum(max(0, sum(r.code_artifact_id == aid for r in records.values()) - 1) for aid in artifacts),
            "revision_after_failed_test": any(
                not record.all_passed and any(
                    artifact_step_indices[aid] > run_step_indices[record.run_id]
                    and artifacts[aid].sha256 != record.code_sha256 for aid in artifacts
                ) for record in records.values()
            ),
        })
        return AgentFlowOutput(steps=steps, metrics=metrics)
