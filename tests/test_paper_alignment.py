"""Regression cases for paper final answers, eligibility and independent audits."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_r1.evaluation.answers import final_answer_record, score_aime
from agent_r1.evaluation.consistency import consistency_record
from agent_r1.evaluation.summarize import summarize
from agent_r1.verifier import VerificationCredit, VerificationResult
from recipes.deepscaler.trajectory_reward import compute_tool_trajectory_reward, verify_process as verify_math
from recipes.deepscaler.reward_fn import compute_terminal_em
from recipes.hotpotqa_a9.protocol import parse_finish_call, parse_search_call
from recipes.hotpotqa_a9.reward_contract import CONTRACT_CERT_MIX
from recipes.hotpotqa_a9.reward_sources import ARMS, arm_schedule, gold_evidence_verification, observable_search_verification
from recipes.hotpotqa_a9.verifier import verify_process as verify_hotpot
from recipes.hotpotqa_a9.reward_fn import compute_score as score_hotpot
from recipes.hotpotqa_lr.protocol import extract_tool_calls
from recipes.hotpotqa_lr.verifier import artifact_content_sha256
from test_process_judge_flows import load_source, FakeGeneration, FakeJudge, call


@pytest.mark.parametrize("response,expected", [
    (r"Intermediate result 42; final answer is 7", False),
    (r"Earlier \boxed{42}, corrected final \boxed{7}", False),
    (r"\boxed{42.9}", False), (r"\boxed{42.000}", True),
    (r"\boxed{042}", True), ("Final answer is 42.", True),
    (r"\boxed{42}, final \boxed{", False),
    (r"\boxed{42}, final <answer>", False),
    ("Earlier \\boxed{42}\n7", False),
    ("We used 42 in our calculations but did not finish.", False),
])
def test_aime_scores_only_the_selected_final_answer(response, expected):
    assert score_aime(response, "42")["is_correct"] is expected


def test_answer_selection_does_not_change_with_reference():
    response = r"42 was intermediate. Final answer is \boxed{7}."
    assert score_aime(response, "42")["predicted_answer"] == score_aime(response, "7")["predicted_answer"] == "7"
    assert "Final answer" not in final_answer_record(response)["reasoning"]
    assert compute_terminal_em(response, "42") == 0
    assert compute_terminal_em(r"\boxed{0}", "0") == 1


def test_consistency_is_not_normalized_process_reward_and_keeps_old_failures():
    short = VerificationResult((VerificationCredit(2, 1/3, {}),), {"applicable_checks": [1]})
    assert consistency_record(short, 1, eligible=True)["verified_success"] == 1
    failed_then_repaired = VerificationResult((VerificationCredit(3, 1, {}),), {"applicable_checks": [0, 1]})
    info = consistency_record(failed_then_repaired, 1, eligible=True)
    assert info["verified_success"] == 0
    assert consistency_record(short, 1, eligible=False)["verified_success"] == 0
    assert consistency_record(VerificationResult((), {"applicable_checks": []}), 1, eligible=True)["verified_success"] == 0
    with pytest.raises(ValueError, match="applicable_checks"):
        consistency_record(VerificationResult((), {}), 1, eligible=True)


def test_summary_uses_every_trajectory_and_rejects_missing_audits():
    rows = []
    for checks, correct in (([1], 1), ([0, 1], 1), ([], 0)):
        info = consistency_record(VerificationResult((), {"applicable_checks": checks}), correct, eligible=True)
        rows.append({"steps": [{"acc": correct, **info}]})
    result = summarize(rows)
    assert result["trajectory_count"] == 3
    assert result["accuracy"] == 2/3
    assert result["M_d"] == result["C_d_mean"] == 1/3
    with pytest.raises(ValueError, match="re-evaluate"):
        summarize([{"process_reward": 1, "acc": 1}])


def test_math_failed_equations_survive_and_incomplete_trace_is_zero():
    verification = verify_math([(1, "Step 1: 2 + 3 = 6\nStep 2: 2 + 3 = 5")])
    assert verification.audit["applicable_checks"] == [0, 1]
    for mode in ("terminal_only", "uniform_equation_process", "llm_judge"):
        reward = compute_tool_trajectory_reward(reward_mode=mode, final_response=None, ground_truth="5",
            reasoning_segments=[(1, "2 + 3 = 5")], global_step=51, is_validation=False, em_warmup_steps=50,
            prompt_group_key="train:1", process_verification=verification)
        assert reward.score == 0


def test_gold_credits_are_marginal_bounded_and_source_attributed():
    verification = gold_evidence_verification([
        {"assistant_turn": 1, "new_gold_evidence_ids": ["a"]},
        {"assistant_turn": 2, "new_gold_evidence_ids": ["b"]},
        {"assistant_turn": 3, "new_gold_evidence_ids": []}], ("a", "b"), ())
    assert verification.process_reward == 1
    assert verification.components_by_step() == {1: .5, 2: .5, 3: 0}


async def _hotpot(arm, *, complete=True, validation=False, invalid_certificate=False):
    module = load_source("recipes/hotpotqa_a9/agent_flow.py", ARMS=ARMS, arm_schedule=arm_schedule,
        gold_evidence_verification=gold_evidence_verification, observable_search_verification=observable_search_verification, verify_process=verify_hotpot,
        parse_finish_call=parse_finish_call, parse_search_call=parse_search_call, extract_tool_calls=extract_tool_calls)
    flow = object.__new__(module.HotpotQACertificateAgentFlow)
    source = "The answer is American."
    artifact = {"artifact_id": "passage:1", "text": source, "content_sha256": artifact_content_sha256(source)}
    finish = call("finish", status="answer", answer="American", certificate=None if invalid_certificate else
                  {"source_id": "passage:1", "support_span": source, "answer_span": "American"})
    texts = [call("search", query="nationality"), finish] if complete else [call("search", query="nationality")] + ["invalid"] * 3
    generator = FakeGeneration(texts)
    flow.server_manager = flow.tokenizer = generator
    flow.reward_arm = arm; flow.process_reward_enabled = True; flow.em_warmup_steps = 50
    flow.max_steps = 4; flow.reward_horizon = 3; flow.response_length = 1024
    flow.thinking_mode = "disabled"; flow.enable_tool_parse_feedback = True; flow.contract = CONTRACT_CERT_MIX
    flow.judge_server = FakeJudge()
    flow.search_tool = SimpleNamespace(require_sentence_evidence=False)
    prompts = []
    def prompt(**kwargs):
        prompts.append(kwargs)
        return [1], "observation", {"passage:1": artifact} if kwargs["actions"] else {}
    flow._prompt_ids_within_budget = prompt
    flow._extra_fields = lambda **kwargs: {"reward_extra_info": {}, "step_kind": kwargs["step_kind"]}
    flow._do_search = lambda query, assistant_turn: {"assistant_turn": assistant_turn, "success": True,
        "query": query, "returned_evidence": [{"pid": 1, "title": "Biography", "text": source}]}
    flow._ingest_search_step = lambda *args: None
    async def postprocess(step, **kwargs):
        if step.reward_score is None:
            step.reward_score = score_hotpot("hotpotqa_distractor", generator.decode(step.response_ids), "American")
        return step
    flow._postprocess = postprocess
    output = await flow.run({}, raw_prompt=[{"role": "user", "content": "Nationality?"}],
        extra_info={"split": "train", "index": 0}, _agent_r1_global_step=51, _agent_r1_is_validation=validation)
    return output, prompts, flow.judge_server


def test_hotpot_matched_interaction_and_independent_consistency_for_all_arms():
    baseline_prompts = None
    for arm in ("A0", "A1", "A2", "A3", "A6", "A7", "A9"):
        output, prompts, judge = asyncio.run(_hotpot(arm))
        if baseline_prompts is None:
            baseline_prompts = prompts
        assert prompts == baseline_prompts
        info = output.steps[-1].extra_fields["reward_extra_info"]
        assert info["acc"] == info["verified_success"] == 1
        assert sum(step.reward_score for step in output.steps) == pytest.approx(info["optimizer_total_reward"])
        if arm == "A6":
            assert len(judge.calls) == 1
            assert "finish" not in judge.calls[0][0]
            assert "gold_evidence" not in judge.calls[0][0]


def test_hotpot_invalid_certificate_does_not_gate_correct_final_answer():
    output, _, _ = asyncio.run(_hotpot("A9", invalid_certificate=True))
    info = output.steps[-1].extra_fields["reward_extra_info"]
    assert info["terminal_eligible"] and info["acc"] == 1
    assert info["verified_success"] == 0
    assert info["optimizer_terminal_component"] > 0


def test_hotpot_missing_finish_zero_for_every_arm_and_validation_is_outcome_only():
    for arm in ("A1", "A2", "A3", "A6", "A7", "A9"):
        output, _, _ = asyncio.run(_hotpot(arm, complete=False))
        assert all(step.reward_score == 0 for step in output.steps)
        assert output.steps[-1].extra_fields["reward_extra_info"]["verified_success"] == 0
        output, _, judge = asyncio.run(_hotpot(arm, validation=True))
        assert [step.reward_score for step in output.steps] == [0, 1]
        assert not judge.calls


def test_paper_math_actual_flow_shares_single_generation_and_excludes_final_from_judge():
    from recipes.llm_judge.scoring import question_from_raw_prompt
    from recipes.deepscaler.prompts import build_math_messages, math_continuation_message
    module = load_source("recipes/deepscaler/agent_flow.py", final_answer_record=final_answer_record,
        verify_process=verify_math, compute_tool_trajectory_reward=compute_tool_trajectory_reward,
        question_from_raw_prompt=question_from_raw_prompt, build_math_messages=build_math_messages,
        math_continuation_message=math_continuation_message)
    async def run(mode):
        flow = object.__new__(module.DeepScaleRPaperAgentFlow)
        flow.reward_mode = mode; flow.em_warmup_steps = 50; flow.response_length = 4096
        flow.max_steps = 1; flow.prompt_length = flow.initial_prompt_length = 2048; flow.context_length = 8192
        flow.judge_server = FakeJudge()
        flow.tokenizer = flow.server_manager = FakeGeneration([r"2 + 3 = 5. Final answer is \boxed{5}."])
        result = await flow.run({}, raw_prompt=[{"role": "user", "content": "2+3?"}],
            reward_model={"ground_truth": "5"}, extra_info={"index": 0, "split": "train"}, _agent_r1_global_step=51)
        return result, flow.judge_server
    for mode in ("terminal_only", "uniform_equation_process", "llm_judge"):
        result, judge = asyncio.run(run(mode))
        assert len(result.steps) == 1
        info = result.steps[0].extra_fields["reward_extra_info"]
        assert info["acc"] == info["verified_success"] == 1
        if mode == "llm_judge":
            assert "boxed" not in judge.calls[0][0]
