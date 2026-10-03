#!/usr/bin/env bash
# Shared model adaptation settings for the paper launchers. Source this file,
# then call agent_r1_model_overrides with the actual actor model path.

agent_r1_model_overrides() {
    local model_path="${1%/}"
    AGENT_R1_MODEL_NAME="${model_path##*/}"
    local default_rank=0
    case "${model_path##*/}" in
        Qwen3.5-9B) default_rank=64 ;;
    esac
    [[ "${AGENT_R1_MODEL_SCALE:-}" == 9b ]] && default_rank=64
    local rank="${AGENT_R1_LORA_RANK:-$default_rank}"
    local alpha="${AGENT_R1_LORA_ALPHA:-64}"
    local targets="${AGENT_R1_LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj]}"
    if ! [[ "$rank" =~ ^(0|[1-9][0-9]*)$ && "$alpha" =~ ^[1-9][0-9]*$ ]]; then
        echo "AGENT_R1_LORA_RANK must be a nonnegative integer; AGENT_R1_LORA_ALPHA must be positive" >&2
        return 2
    fi
    # Keep the whole Hydra list as one argument; never eval user overrides.
    AGENT_R1_MODEL_OVERRIDES=(
        "actor_rollout_ref.model.lora_rank=$rank"
        "actor_rollout_ref.model.lora_alpha=$alpha"
        "actor_rollout_ref.model.target_modules=$targets"
    )
    if [[ "$rank" == 0 ]]; then
        echo "Actor adaptation: full fine-tuning (lora_rank=0)"
    else
        echo "Actor adaptation: LoRA (rank=$rank, alpha=$alpha, target_modules=$targets)"
    fi
}

# Appendix C defaults, with the user-requested 9B batch of 8 for eight A40 GPUs.
agent_r1_paper_profile() {
    local domain="$1"
    local model_path="${2%/}"
    local inferred_scale=4b
    [[ "${model_path##*/}" == Qwen3.5-9B ]] && inferred_scale=9b
    PAPER_MODEL_SCALE="${AGENT_R1_MODEL_SCALE:-$inferred_scale}"
    case "$PAPER_MODEL_SCALE" in 4b|9b) ;; *) echo "AGENT_R1_MODEL_SCALE must be 4b or 9b" >&2; return 2 ;; esac
    PAPER_TRAIN_SAMPLES=10000
    PAPER_TRAIN_STEPS=500
    PAPER_BATCH_SIZE=20
    [[ "$PAPER_MODEL_SCALE" == 9b ]] && PAPER_BATCH_SIZE=8
    PAPER_PROMPT_LENGTH=8192
    PAPER_RESPONSE_LENGTH=2048
    case "$domain" in
        deepscaler)
            PAPER_PROMPT_LENGTH=2048
            PAPER_RESPONSE_LENGTH=4096
            [[ "$PAPER_MODEL_SCALE" == 9b ]] && PAPER_RESPONSE_LENGTH=5120
            PAPER_MAX_STEPS=5 ;;
        hotpotqa)
            PAPER_RESPONSE_LENGTH=1024
            PAPER_MAX_STEPS=4 ;;
        taco) PAPER_MAX_STEPS=5 ;;
        vision_r1) PAPER_MAX_STEPS=3 ;;
        *) echo "Unknown paper domain: $domain" >&2; return 2 ;;
    esac
    return 0
}

# Reuse the repository's existing step-level PPO critic path for paper baselines.
agent_r1_optimizer_overrides() {
    local model_path="$1"
    AGENT_R1_TRAINER_MODULE=agent_r1.trainer.main_agent_grpo
    AGENT_R1_ALGORITHM_ARGS=(algorithm.adv_estimator=grpo ++algorithm.grpo.credit_assignment=step_causal algorithm.norm_adv_by_std_in_grpo=True)
    AGENT_R1_CRITIC_ARGS=(critic.enable=False)
    case "${AGENT_R1_OPTIMIZER:-grpo}" in
        grpo) ;;
        ppo)
            AGENT_R1_TRAINER_MODULE=agent_r1.trainer.main_agent_ppo
            AGENT_R1_ALGORITHM_ARGS=(algorithm.adv_estimator=gae algorithm.lam=1.0)
            AGENT_R1_CRITIC_ARGS=(critic.enable=True "critic.model.path=$model_path" critic.optim.lr=1e-5
                critic.model.enable_gradient_checkpointing=True critic.fsdp.param_offload=False
                critic.fsdp.optimizer_offload=False critic.ppo_micro_batch_size=1
                critic.ppo_micro_batch_size_per_gpu=1 trainer.critic_warmup=0 trainer.max_critic_ckpt_to_keep=2)
            ;;
        *) echo "AGENT_R1_OPTIMIZER must be grpo or ppo" >&2; return 2 ;;
    esac
}
