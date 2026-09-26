"""Reward-only switches for the matched paper HotpotQA interaction."""
from agent_r1.verifier import UniformRewardSchedule, VerificationCredit, VerificationResult, uniform_reward_schedule

ARMS = {"A0", "A1", "A2", "A3", "A6", "A7", "A9", "A9_CERT_MIX"}


def arm_schedule(*, arm, global_step, is_validation, warmup_steps, prompt_group_key, process_enabled=True):
    if arm not in ARMS:
        raise ValueError(f"Unsupported paper arm: {arm}")
    if is_validation or arm in {"A0", "A1"} or not process_enabled:
        return UniformRewardSchedule(1.0, 0.0, "validation_terminal_em" if is_validation else "terminal_only", "fixed", None)
    if arm == "A2":
        return UniformRewardSchedule(0.0, 1.0, "gold_process_only", "fixed", None)
    if arm in {"A3", "A7"}:
        return UniformRewardSchedule(0.5, 0.5, "gold_fixed_mix" if arm == "A3" else "weak_fixed_mix", "fixed", None)
    return uniform_reward_schedule(
        global_step=global_step, is_validation=False, warmup_steps=warmup_steps,
        prompt_group_key=prompt_group_key, namespace=b"hotpotqa-a9-group-weight-v1",
        warmup_phase="em_warmup", validation_phase="validation_terminal_em", mixed_phase="process_uniform",
    )


def gold_evidence_verification(search_steps, gold_evidence_ids, unresolved_gold_facts):
    """Marginal gold-support coverage; privileged supervision only for gold arms."""
    denominator = len(set(gold_evidence_ids))
    credits = () if not denominator or unresolved_gold_facts else tuple(
        VerificationCredit(int(step["assistant_turn"]), len(step["new_gold_evidence_ids"]) / denominator, {})
        for step in search_steps
    )
    return VerificationResult(credits=credits, audit={"verifier": "gold_support_coverage"})


_GROUP_WEIGHT_NAMESPACE = b"hotpotqa-a9-group-weight-v1"

def reward_schedule(
    *,
    global_step: int,
    is_validation: bool,
    em_warmup_steps: int,
    prompt_group_key: str,
    process_reward_enabled: bool = True,
) -> UniformRewardSchedule:
    if not process_reward_enabled:
        return UniformRewardSchedule(1.0, 0.0, "terminal_only", "disabled", None)
    return uniform_reward_schedule(
        global_step=global_step,
        is_validation=is_validation,
        warmup_steps=em_warmup_steps,
        prompt_group_key=prompt_group_key,
        namespace=_GROUP_WEIGHT_NAMESPACE,
        warmup_phase="em_warmup",
        validation_phase="validation_terminal_em",
        mixed_phase="certificate_uniform",
    )


def optimizer_reward_schedule(
    *,
    global_step: int,
    is_validation: bool,
    em_warmup_steps: int,
    prompt_group_key: str,
) -> tuple[float, float, str]:
    """Return (terminal_weight, process_weight, phase_label) for this step.

    Post-warmup A9 deterministically samples one w ~ U(0, 1) from the optimizer
    step and prompt-group key. All rollouts for that prompt therefore share the
    same scalarization while different prompt groups retain varied mixtures.
    """
    schedule = reward_schedule(
        global_step=global_step,
        is_validation=is_validation,
        em_warmup_steps=em_warmup_steps,
        prompt_group_key=prompt_group_key,
    )
    return schedule.terminal_weight, schedule.process_weight, schedule.phase




def observable_search_verification(search_steps, horizon=3):
    """Retained A7 weak-execution control; separate from paper gold ablations."""
    return VerificationResult(tuple(
        VerificationCredit(int(step["assistant_turn"]), float(bool(step.get("success"))) / horizon, {})
        for step in search_steps[:horizon]
    ), {"verifier": "observable_search_execution"})
