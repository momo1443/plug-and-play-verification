#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B TACO A9 verified-process training. The underlying 5-turn agent,
# replay verifier, and reward contract are shared with the 4B launcher.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TACO_A9_MODEL_PATH="${TACO_A9_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-9B}"
export TACO_A9_TENSOR_PARALLEL_SIZE="${TACO_A9_TENSOR_PARALLEL_SIZE:-8}"
export TACO_A9_TRAIN_BATCH_SIZE="${TACO_A9_TRAIN_BATCH_SIZE:-20}"
export TACO_A9_ROLLOUT_N="${TACO_A9_ROLLOUT_N:-4}"
export TACO_A9_MAX_PROMPT_LENGTH="${TACO_A9_MAX_PROMPT_LENGTH:-8192}"
export TACO_A9_MAX_RESPONSE_LENGTH="${TACO_A9_MAX_RESPONSE_LENGTH:-2048}"
export TACO_A9_MAX_MODEL_LENGTH="${TACO_A9_MAX_MODEL_LENGTH:-12288}"
export TACO_A9_MAX_BATCHED_TOKENS="${TACO_A9_MAX_BATCHED_TOKENS:-12288}"
export TACO_A9_MAX_NUM_SEQS="${TACO_A9_MAX_NUM_SEQS:-40}"
export TACO_A9_VLLM_MEMORY_UTILIZATION="${TACO_A9_VLLM_MEMORY_UTILIZATION:-0.80}"
export TACO_A9_VLLM_KV_CACHE_BYTES="${TACO_A9_VLLM_KV_CACHE_BYTES:-4294967296}"
export TACO_A9_ACTOR_PARAM_OFFLOAD="${TACO_A9_ACTOR_PARAM_OFFLOAD:-true}"
export TACO_A9_ACTOR_OPTIMIZER_OFFLOAD="${TACO_A9_ACTOR_OPTIMIZER_OFFLOAD:-true}"
export TACO_A9_SAVE_FREQ="${TACO_A9_SAVE_FREQ:-50}"

export RUN_ID="${RUN_ID:-qwen35-9b_taco_a9_verified_process_5turn_terminal50_n4_300step_8gpu_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_a9_uniform.sh" "$@"
