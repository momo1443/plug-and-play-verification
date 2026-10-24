"""Shared, single-attempt PNP evaluation contract (no training dependencies)."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from .official import request

PROTOCOL_VERSION = "pnp-agent-evaluation-v2"


@dataclass(frozen=True)
class EvaluationProfile:
    benchmark: str
    max_steps: int
    total_output_tokens: int
    per_step_tokens: int
    prompt_limit: int
    context_limit: int
    metric: str
    temperature: float = 0.0
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
        if (self.temperature, self.top_p, self.top_k, self.trajectories_per_problem) != (0.0, 1.0, -1, 1):
            raise ValueError("Main evaluation requires greedy decoding and one trajectory per problem")

    def sampling(self):
        return {"temperature": self.temperature, "top_p": self.top_p,
                "top_k": self.top_k, "n": 1, "max_tokens": self.per_step_tokens}

    def record(self):
        return {"protocol_version": PROTOCOL_VERSION, **asdict(self),
                "prompt_limit_scope": "initial_prompt" if self.benchmark == "aime2026" else "each_policy_step",
                "output_budget_unit": "all_generated_tokens_across_the_trajectory",
                "references_visible_to_policy": False, "hidden_test_feedback": False,
                "accuracy_denominator": "all_selected_questions",
                "process_reward_used_as_accuracy": False}


PROFILES = {
    "aime2026": EvaluationProfile("aime2026", 5, 32768, 6554, 2048, 40960, "integer_accuracy"),
    "hotpotqa": EvaluationProfile("hotpotqa", 4, 4096, 1024, 8192, 12288, "answer_em"),
    "livecodebench": EvaluationProfile("livecodebench", 5, 10240, 2048, 8192, 12288, "pass_at_1"),
    "mathvision": EvaluationProfile("mathvision", 3, 6144, 2048, 8192, 12288, "answer_accuracy"),
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


def normalize_qa_answer(value):
    return request("hotpotqa", {"operation": "normalize", "answer": str(value)})


def qa_scores(answer, references):
    """HotpotQA answer EM and token F1; absent submissions score zero."""
    if not isinstance(answer, str) or not answer.strip():
        return {"is_correct": False, "answer_em": 0.0, "answer_f1": 0.0}
    return request("hotpotqa", {"answer": answer, "references": [str(value) for value in references]})


def mathvision_equal(left, right):
    """Use the pinned MATH-V is_equal implementation without local rules."""
    return request("mathvision", {"operation": "equal", "left": str(left), "right": str(right)})


def mathvision_score(answer, reference, options):
    """Run the original MATH-V extraction and equivalence on the submission."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    return request("mathvision", {"answer": answer, "reference": str(reference), "options": list(options)})["is_correct"]


def summarize_records(records, expected_ids, profile):
    ids = [row["question_id"] for row in records]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Evaluation requires nonempty, unique dataset question IDs")
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("Results must cover every selected question exactly once")
    for row in records:
        if not isinstance(row["is_correct"], bool):
            raise ValueError("Accuracy requires a binary final outcome")
        if row["is_correct"] and not row["terminal_eligible"]:
            raise ValueError("An ineligible submission cannot be correct")
    n = len(expected_ids)
    correct = sum(row["is_correct"] for row in records)
    result = {"protocol_version": PROTOCOL_VERSION, "benchmark": profile.benchmark,
              "metric": profile.metric, "total": n, "correct": correct,
              "accuracy": correct / n, "accuracy_percent": 100 * correct / n,
              "trajectories_per_problem": 1,
              "valid_submissions": sum(row["terminal_eligible"] for row in records),
              "missing_or_invalid_submissions": sum(not row["terminal_eligible"] for row in records),
              "termination_counts": dict(Counter(row["termination_reason"] for row in records)),
              "mean_generated_output_tokens": sum(row["generated_output_tokens"] for row in records) / n}
    if profile.benchmark == "hotpotqa":
        result.update(answer_em=result["accuracy"],
                      answer_f1=sum(row["answer_f1"] for row in records) / n)
    return result
