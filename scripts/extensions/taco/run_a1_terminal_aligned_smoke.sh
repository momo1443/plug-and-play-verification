#!/usr/bin/env bash
set -euo pipefail

# TACO smoke profile using the shared 500-step training budget.
export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7
export TACO_A1_TENSOR_PARALLEL_SIZE=1
export TACO_A1_TOTAL_TRAINING_STEPS=500
export TACO_A1_TRAIN_MAX_SAMPLES=20
export TACO_A1_SAVE_FREQ=1
export TACO_A1_MAX_PROMPT_LENGTH=2048
export TACO_A1_MAX_RESPONSE_LENGTH=4096
export TACO_A1_MAX_MODEL_LENGTH=8192
export TACO_A1_MAX_BATCHED_TOKENS=8192
export TACO_A1_MAX_NUM_SEQS=20
export TACO_A1_VLLM_MEMORY_UTILIZATION=0.25
export TACO_A1_VLLM_KV_CACHE_BYTES=0
export TACO_A1_ACTOR_PARAM_OFFLOAD=false
export TACO_A1_ACTOR_OPTIMIZER_OFFLOAD=false
export RUN_ID="qwen35-4b_taco_a1_terminal_runtime_aligned_smoke500_6gpu_$(date +%Y%m%d-%H%M%S)"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../taco/run_a1_terminal.sh"
