#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "common_v2.sh must be sourced by an A8-LR launcher" >&2
    exit 2
fi

export HOTPOTQA_RUN_MODE=main
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3,4,5,6,7}"
export HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-6}"
export HOTPOTQA_AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-6}"
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
export HOTPOTQA_VLLM_MAX_MODEL_LEN="${HOTPOTQA_VLLM_MAX_MODEL_LEN:-8192}"
export HOTPOTQA_VLLM_MAX_NUM_BATCHED_TOKENS="${HOTPOTQA_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
export HOTPOTQA_VLLM_MAX_NUM_SEQS="${HOTPOTQA_VLLM_MAX_NUM_SEQS:-20}"
export HOTPOTQA_RAY_TMP_LINK="${HOTPOTQA_RAY_TMP_LINK:-/tmp/a8v2-${UID}}"
export HOTPOTQA_ROLLOUT_N="${HOTPOTQA_ROLLOUT_N:-4}"
export HOTPOTQA_TRAIN_MAX_SAMPLES=30000
export HOTPOTQA_TRAIN_BATCH_SIZE=20
export HOTPOTQA_GRPO_MICRO_BATCH_SIZE="${HOTPOTQA_GRPO_MICRO_BATCH_SIZE:-2}"
export HOTPOTQA_TOTAL_TRAINING_STEPS=1500
export HOTPOTQA_DATA_SHUFFLE=false
export HOTPOTQA_SAVE_FREQ=50
export HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP=2
export HOTPOTQA_SEED="${HOTPOTQA_SEED:-42}"
export HOTPOTQA_LR_ALLOW_UNCALIBRATED_LAUNCH="${HOTPOTQA_LR_ALLOW_UNCALIBRATED_LAUNCH:-0}"

case "$HOTPOTQA_LR_ALLOW_UNCALIBRATED_LAUNCH" in
    0|1) ;;
    *) echo "HOTPOTQA_LR_ALLOW_UNCALIBRATED_LAUNCH must be 0 or 1" >&2; exit 2 ;;
esac

if [[ "${HOTPOTQA_HYDRA_CONFIG_ONLY:-0}" != "1" \
    && -z "${HOTPOTQA_LR_CALIBRATION_REPORT:-}" \
    && "$HOTPOTQA_LR_ALLOW_UNCALIBRATED_LAUNCH" != "1" ]]; then
    echo "A8-LR-v2 requires HOTPOTQA_LR_CALIBRATION_REPORT from a passed calibration" >&2
    exit 2
fi
if [[ "$HOTPOTQA_LR_ALLOW_UNCALIBRATED_LAUNCH" == "1" \
    && -n "${HOTPOTQA_LR_CALIBRATION_REPORT:-}" ]]; then
    echo "An uncalibrated launch cannot also provide HOTPOTQA_LR_CALIBRATION_REPORT" >&2
    exit 2
fi

if [[ "$HOTPOTQA_NUM_GPUS" != "6" || "$HOTPOTQA_AGENT_WORKERS" != "6" ]]; then
    echo "A8-LR-v2 requires HOTPOTQA_NUM_GPUS=6 and HOTPOTQA_AGENT_WORKERS=6" >&2
    exit 2
fi
IFS=',' read -r -a _A8_VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
if (( ${#_A8_VISIBLE_GPUS[@]} != 6 )); then
    echo "A8-LR-v2 requires exactly six CUDA_VISIBLE_DEVICES entries" >&2
    exit 2
fi
declare -A _A8_SEEN_GPUS=()
for _a8_gpu in "${_A8_VISIBLE_GPUS[@]}"; do
    if [[ ! "$_a8_gpu" =~ ^[0-9]+$ || -n "${_A8_SEEN_GPUS[$_a8_gpu]:-}" ]]; then
        echo "A8-LR-v2 CUDA_VISIBLE_DEVICES entries must be unique non-negative integers" >&2
        exit 2
    fi
    _A8_SEEN_GPUS["$_a8_gpu"]=1
done
unset _a8_gpu _A8_VISIBLE_GPUS _A8_SEEN_GPUS
if [[ "$HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION" != "0.25" \
    || "$HOTPOTQA_VLLM_MAX_MODEL_LEN" != "8192" \
    || "$HOTPOTQA_VLLM_MAX_NUM_BATCHED_TOKENS" != "8192" \
    || "$HOTPOTQA_VLLM_MAX_NUM_SEQS" != "20" ]]; then
    echo "A8-LR-v2 requires the 2026-08-14 A8 runtime shape 0.25/8192/8192/20" >&2
    exit 2
fi
if [[ -n "${HOTPOTQA_VLLM_KV_CACHE_MEMORY_BYTES:-}" ]]; then
    echo "A8-LR-v2 forbids fixed KV-cache bytes; vLLM must size the cache automatically" >&2
    exit 2
fi
unset HOTPOTQA_VLLM_KV_CACHE_MEMORY_BYTES
if [[ "${HOTPOTQA_SKIP_PREFLIGHT:-0}" != "0" ]]; then
    echo "A8-LR-v2 preflight cannot be skipped" >&2
    exit 2
fi
export HOTPOTQA_SKIP_PREFLIGHT=0
