"""Shared, frozen PNP evaluation contract (no training dependencies)."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from .official import request

# v3: avg@k over k independent trajectories (k=4 for AIME2026/LiveCodeBench) and the
# five-step MATH-Vision budget. v2 summaries are single-trajectory and not comparable.
PROTOCOL_VERSION = "pnp-agent-evaluation-v3"


@dataclass(frozen=True)
class EvaluationProfile:
    benchmark: str
    max_steps: int
    total_output_tokens: int
    per_step_tokens: int
    prompt_limit: int
    context_limit: int
    metric: str
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = -1
    trajectories_per_problem: int = 1
    seed: int = 42
    precision: str = "bfloat16"
    inference_engine: str = "vllm"
    mode: str = "offline_python_api"

    def __post_init__(self):
        if any(value < 1 for value in (self.max_steps, self.total_output_tokens,
                                      self.per_step_tokens, self.prompt_limit, self.context_limit)):
            raise ValueError("Evaluation budgets must be positive")
        if self.prompt_limit >= self.context_limit:
            raise ValueError("Context limit must leave room after the prompt limit")
        if self.temperature < 0 or (self.top_p, self.top_k) != (1.0, -1) or self.trajectories_per_problem < 1:
            raise ValueError("Main evaluation requires top_p=1, top_k=-1 and at least one trajectory per problem")
        if self.trajectories_per_problem > 1 and self.temperature == 0:
            raise ValueError("Repeated trajectories require sampling (temperature > 0)")

    def sampling(self):
        return {"temperature": self.temperature, "top_p": self.top_p,
                "top_k": self.top_k, "n": 1, "max_tokens": self.per_step_tokens}

    def record(self):
        return {"protocol_version": PROTOCOL_VERSION, **asdict(self),
                "prompt_limit_scope": "initial_prompt" if self.benchmark == "aime2026" else "each_policy_step",
                "output_budget_unit": "all_generated_tokens_across_the_trajectory",
                "references_visible_to_policy": False, "hidden_test_feedback": False,
                "accuracy_denominator": "all_selected_questions",
                "accuracy_aggregation": f"avg@{self.trajectories_per_problem}",
                "process_reward_used_as_accuracy": False}


PROFILES = {
    "aime2026": EvaluationProfile("aime2026", 5, 10240, 2048, 2048, 13312, "integer_accuracy",
                                  trajectories_per_problem=4),
    "hotpotqa": EvaluationProfile("hotpotqa", 4, 4096, 1024, 8192, 12288, "answer_em"),
    "livecodebench": EvaluationProfile("livecodebench", 5, 10240, 2048, 8192, 12288, "pass_at_1",
                                       trajectories_per_problem=4),
    "mathvision": EvaluationProfile("mathvision", 5, 10240, 2048, 12288, 16384, "answer_accuracy"),
}


def evaluation_profile(benchmark, **overrides):
    """Overrides are explicit and recorded, and never change training profiles."""
    return replace(PROFILES[benchmark], **{key: value for key, value in overrides.items() if value is not None})


class EvaluationStop(Exception):
    """A model attempt ended at a protocol boundary; keep it in the denominator."""


@dataclass
class TrajectoryBudget:
    profile: EvaluationProfile
    generated_tokens: int = 0

    def next_cap(self, prompt_tokens, requested_tokens=None):
        remaining = self.profile.total_output_tokens - self.generated_tokens
        if remaining <= 0:
            raise EvaluationStop("token_budget")
        if prompt_tokens > self.profile.prompt_limit:
            raise EvaluationStop("prompt_limit")
        room = self.profile.context_limit - prompt_tokens
        if room <= 0:
            raise EvaluationStop("context_limit")
        return min(remaining, room, self.profile.per_step_tokens,
                   requested_tokens if requested_tokens is not None else self.profile.per_step_tokens)

    def consume(self, token_count, cap):
        if not 0 <= token_count <= cap:
            raise RuntimeError("Inference backend exceeded the requested token budget")
        self.generated_tokens += token_count
        if not token_count:
            raise EvaluationStop("empty_generation")


def qa_scores(answer, references):
    """HotpotQA answer EM and token F1; absent submissions score zero."""
    if not isinstance(answer, str) or not answer.strip():
        return {"is_correct": False, "answer_em": 0.0, "answer_f1": 0.0}
    return request("hotpotqa", {"answer": answer, "references": [str(value) for value in references]})


def mathvision_score(answer, reference, options):
    """Run the original MATH-V extraction and equivalence on the submission."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    return request("mathvision", {"answer": answer, "reference": str(reference), "options": list(options)})["is_correct"]


def summarize_records(records, expected_ids, profile):
    k = profile.trajectories_per_problem
    keys = [(row["question_id"], row.get("sample_index", 0)) for row in records]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Evaluation requires nonempty, unique dataset question IDs")
    if len(keys) != len(set(keys)) or set(keys) != {(qid, s) for qid in expected_ids for s in range(k)}:
        raise ValueError("Results must cover every selected question and sample exactly once")
    for row in records:
        if not isinstance(row["is_correct"], bool):
            raise ValueError("Accuracy requires a binary final outcome")
        if row["is_correct"] and not row["terminal_eligible"]:
            raise ValueError("An ineligible submission cannot be correct")
    n, trajectories = len(expected_ids), len(records)
    correct = sum(row["is_correct"] for row in records)
    # avg@k: mean accuracy over all k x n trajectories. "total" counts questions;
    # "correct" counts correct trajectories out of "total_trajectories".
    result = {"protocol_version": PROTOCOL_VERSION, "benchmark": profile.benchmark,
              "metric": profile.metric, "aggregation": f"avg@{k}", "total": n,
              "total_trajectories": trajectories, "correct": correct,
              "accuracy": correct / trajectories, "accuracy_percent": 100 * correct / trajectories,
              "trajectories_per_problem": k,
              "valid_submissions": sum(row["terminal_eligible"] for row in records),
              "missing_or_invalid_submissions": sum(not row["terminal_eligible"] for row in records),
              "termination_counts": dict(Counter(row["termination_reason"] for row in records)),
              "mean_generated_output_tokens": sum(row["generated_output_tokens"] for row in records) / trajectories}
    if k > 1:
        solved = {row["question_id"] for row in records if row["is_correct"]}
        per_sample = [sum(row["is_correct"] for row in records if row.get("sample_index", 0) == s) / n
                      for s in range(k)]
        # Spread of the k single-sample accuracies; avg@k is their mean.
        std = (sum((value - result["accuracy"]) ** 2 for value in per_sample) / (k - 1)) ** 0.5
        result.update(
            per_sample_accuracy=per_sample,
            per_sample_accuracy_std=std, accuracy_standard_error=std / k ** 0.5,
            **{f"pass_at_{k}": len(solved) / n})
    if profile.benchmark == "hotpotqa":
        result.update(answer_em=result["accuracy"],
                      answer_f1=sum(row["answer_f1"] for row in records) / trajectories)
    return result
