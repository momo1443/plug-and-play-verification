"""Run existing PNP AgentFlows without a trainer, critic, or Ray cluster."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_r1.evaluation.pnp_protocol import EvaluationStop, TrajectoryBudget


class BudgetedServer:
    """Adapter from the AgentFlow generation interface to offline vLLM."""
    def __init__(self, engine, sampling_factory, profile):
        self.engine, self.sampling_factory = engine, sampling_factory
        self.budget = TrajectoryBudget(profile)
        self.generations = []

    async def generate(self, *, prompt_ids, sampling_params, image_data=None, video_data=None, **kwargs):
        cap = self.budget.next_cap(len(prompt_ids), sampling_params.get("max_tokens"))
        # Set the contract explicitly instead of inheriting model generation_config.
        params = {**self.budget.profile.sampling(), "max_tokens": cap}
        model_input = {"prompt_token_ids": list(prompt_ids)}
        if image_data or video_data:
            model_input["multi_modal_data"] = {}
            if image_data:
                model_input["multi_modal_data"]["image"] = image_data
            if video_data:
                model_input["multi_modal_data"]["video"] = video_data
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(None, lambda: self.engine.generate(
            [model_input], self.sampling_factory(**params), use_tqdm=False))
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("Evaluation requires exactly one completion for each request")
        candidate = outputs[0].outputs[0]
        ids = list(candidate.token_ids)
        self.budget.consume(len(ids), cap)
        self.generations.append({"prompt_tokens": len(prompt_ids), "generated_tokens": len(ids),
                                 "response_token_ids": ids,
                                 "finish_reason": getattr(candidate, "finish_reason", None)})
        return SimpleNamespace(token_ids=ids, log_probs=None)


class EvaluationRuntime:
    """Keep policy/tool logic unchanged; replace only trainer postprocessing."""
    async def apply_chat_template(self, *args, **kwargs):
        ids = await super().apply_chat_template(*args, **kwargs)
        self.server_manager.budget.next_cap(len(ids))
        return ids

    def _prompt_ids_within_budget(self, **kwargs):
        try:
            return super()._prompt_ids_within_budget(**kwargs)
        except ValueError as exc:
            if str(exc).startswith("A9 prompt has "):
                raise EvaluationStop("prompt_limit") from exc
            raise

    async def _postprocess(self, step, **kwargs):
        import torch
        from agent_r1.agent_flow.agent_flow import _InternalAgentFlowStep
        # The flow never receives a benchmark answer. Its training reward slots
        # are placeholders; accuracy is computed by the separate final grader.
        step.reward_score = 0.0
        ids = list(step.prompt_ids) + list(step.response_ids)
        processed = _InternalAgentFlowStep(
            prompt_ids=torch.tensor(step.prompt_ids), response_ids=torch.tensor(step.response_ids),
            input_ids=torch.tensor(ids), attention_mask=torch.ones(len(ids), dtype=torch.long),
            position_ids=torch.arange(len(ids)), response_mask=torch.ones(len(step.response_ids), dtype=torch.long),
            reward_score=0.0, num_turns=step.num_turns, extra_fields=step.extra_fields)
        self.evaluation_steps.append(processed)
        return processed


def create_flow(benchmark, task, tokenizer, processor, server, profile, resources):
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset
    trainer = OmegaConf.create({"actor_rollout_ref": {"rollout": {
        "prompt_length": profile.prompt_limit, "response_length": profile.per_step_tokens}}})
    dataset_config = OmegaConf.create({"apply_chat_template_kwargs": {"enable_thinking": False}})
    common = dict(trainer_config=DictConfigWrap(trainer), server_manager=server, reward_loop_worker=None,
                  tokenizer=tokenizer, processor=processor, dataset_cls=RLHFDataset, dataset_config=dataset_config)
    if benchmark == "hotpotqa":
        from recipes.hotpotqa_a9.agent_flow import HotpotQACertificateAgentFlow
        cls = HotpotQACertificateAgentFlow
        options = dict(max_steps=4, require_sentence_evidence=True, embedding_devices="cpu",
                       corpus_data_dir=resources["corpus_dir"], evidence_sidecar_path=resources["evidence_sidecar"],
                       embedding_model_name=resources["embedding_model"])
    elif benchmark == "livecodebench":
        from recipes.taco_a9.agent_flow import TacoA9CodeAgentFlow
        from agent_r1.evaluation.pnp_code import PublicTaskStore, LiveCodeExecutor
        cls = TacoA9CodeAgentFlow
        options = dict(max_steps=5, task_store=PublicTaskStore(task),
                       executor=LiveCodeExecutor(timeout_seconds=resources["timeout_seconds"]))
    else:
        from recipes.vision_r1.agent_flow import VisionR1VisualAgentFlow
        cls = VisionR1VisualAgentFlow
        options = dict(max_steps=3, reward_mode="uniform_visual_certificate")
    flow = type("PnpEvaluation" + cls.__name__, (EvaluationRuntime, cls), {})(**common, **options)
    flow.evaluation_steps = []
    return flow


def select_submission(benchmark, steps, tokenizer):
    """Select the actual submitted artifact, independently of all references."""
    from recipes.taco_a9.protocol import parse_tool_call
    from recipes.taco_a9.dsl import CodeArtifact
    from recipes.hotpotqa_a9.protocol import parse_finish_call, parse_search_call
    artifacts = {}
    searched = False
    for step in steps:
        text = tokenizer.decode(step.response_ids.tolist() if hasattr(step.response_ids, "tolist")
                                else step.response_ids, skip_special_tokens=True)
        if benchmark == "hotpotqa":
            # Search eligibility follows actual execution, not a guessed call.
            searched |= step.extra_fields.get("step_kind") == "search"
            finish = parse_finish_call(text) if searched else None
            if finish is not None and finish.answer is not None:
                return finish.answer
        else:
            call = parse_tool_call(text)
            if benchmark == "livecodebench":
                if call.name == "write_code" and isinstance(call.arguments.get("code"), str) and call.arguments["code"].strip():
                    artifact = CodeArtifact.create(len(artifacts) + 1, call.arguments["code"])
                    artifacts[artifact.artifact_id] = artifact.code
                elif call.name == "submit":
                    artifact_id = call.arguments.get("code_artifact_id")
                    return artifacts.get(artifact_id) if isinstance(artifact_id, str) else None
            elif call.name == "submit_answer":
                answer = call.arguments.get("answer")
                return answer.strip() if isinstance(answer, str) and answer.strip() else None
    return None


async def audit_partial_flow(benchmark, task, flow, steps):
    """Retain observable process checks even when a budget prevents submission."""
    if benchmark == "hotpotqa":
        from recipes.hotpotqa_a9.verifier import verify_process
        transitions, indices = [], []
        for index, step in enumerate(steps):
            current = step.extra_fields.get("certificate_transitions", [])
            if len(current) > len(transitions):
                indices.extend([index] * (len(current) - len(transitions)))
                transitions = current
        result = verify_process(transitions, indices, reward_horizon=flow.reward_horizon)
    elif benchmark == "livecodebench":
        from recipes.taco_a9.dsl import CodeArtifact, ExecutionRecord
        from recipes.taco_a9.protocol import parse_tool_call
        from recipes.taco_a9.verifier import verify_process
        artifacts, records, run_steps = {}, {}, {}
        for index, step in enumerate(steps, 1):
            ids = step.response_ids.tolist()
            call = parse_tool_call(flow.tokenizer.decode(ids, skip_special_tokens=True))
            if call.name == "write_code" and isinstance(call.arguments.get("code"), str) and call.arguments["code"].strip():
                artifact = CodeArtifact.create(len(artifacts) + 1, call.arguments["code"])
                artifacts[artifact.artifact_id] = artifact
            for record in step.extra_fields.get("execution_ledger", []):
                if record["run_id"] not in records:
                    records[record["run_id"]] = ExecutionRecord(**record)
                    run_steps[record["run_id"]] = index
        developer, _ = flow.store.suites(task.question_id)
        result = verify_process(artifacts=artifacts, records=records, raw_certificate=None,
                                submitted_artifact=None, developer_suite=developer, executor=flow.executor,
                                final_step_index=max(1, len(steps)), run_step_indices=run_steps,
                                submission_attempted=False)
    else:
        from recipes.vision_r1.artifacts import VisualArtifact
        from recipes.vision_r1.verifier import verify_process
        images = (await flow.process_vision_info(task.actor_inputs["raw_prompt"]))["images"]
        root = VisualArtifact.root(images[0])
        artifacts = {root.artifact_id: root}
        recorded = steps[-1].extra_fields.get("visual_artifacts", []) if steps else []
        for record in recorded:
            if record["bbox_2d"] is not None:
                artifact = VisualArtifact.crop(
                    ordinal=int(record["artifact_id"].split(":")[2]),
                    parent=artifacts[record["parent_artifact_id"]],
                    bbox_2d=record["bbox_2d"], created_step=record["created_step"])
                artifacts[artifact.artifact_id] = artifact
        result = verify_process(artifacts=artifacts, raw_certificate=None, submitted_answer="",
                                submission_attempted=False)
    return {"verified_process_reward": result.process_reward, "audit_completed": True, "audit": result.audit}


async def generate_task(benchmark, task, tokenizer, processor, server, profile, resources):
    flow = create_flow(benchmark, task, tokenizer, processor, server, profile, resources)
    termination = "max_steps"
    # Only public task fields enter the flow. In particular, private_tests and
    # reference answers stay in the caller's post-generation grading data.
    inputs = {key: value for key, value in task.actor_inputs.items() if key in {"raw_prompt", "extra_info"}}
    inputs.update(reward_model={"ground_truth": ""}, _agent_r1_is_validation=True, _agent_r1_global_step=-1)
    try:
        await flow.run(profile.sampling(), **inputs)
    except EvaluationStop as exc:
        termination = str(exc)
    steps = flow.evaluation_steps
    answer = select_submission(benchmark, steps, tokenizer)
    if answer is not None:
        termination = "final_submission"
    elif benchmark == "mathvision" and len(steps) < profile.max_steps and termination == "max_steps":
        termination = "prompt_limit"
    recorded_steps = []
    for index, step in enumerate(steps, 1):
        ids = step.response_ids.tolist()
        recorded_steps.append({"turn": index, "response": tokenizer.decode(ids, skip_special_tokens=True),
                               "generated_tokens": len(ids), "step_kind": step.extra_fields.get("step_kind"),
                               "artifacts": step.extra_fields.get("code_artifacts", step.extra_fields.get("visual_artifacts")),
                               "execution_ledger": step.extra_fields.get("execution_ledger")})
    info = steps[-1].extra_fields.get("reward_extra_info", {}) if steps else {}
    verification = {key: info[key] for key in (
        "verified_process_reward", "verification_checks", "verification_consistency",
        "certificate_audit", "artifact_process_audits", "verifier_timing") if key in info}
    if "local_reward" in info:
        verification["verified_process_reward"] = info["local_reward"]
    verification["audit_completed"] = "verified_process_reward" in verification
    if not verification["audit_completed"]:
        verification = await audit_partial_flow(benchmark, task, flow, steps)
    return {"question_id": task.question_id, "submitted_answer": answer,
            "terminal_eligible": answer is not None, "termination_reason": termination,
            "policy_step_count": len(steps), "generated_output_tokens": server.budget.generated_tokens,
            "steps": recorded_steps, "verification": verification}
