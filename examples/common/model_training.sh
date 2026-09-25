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
