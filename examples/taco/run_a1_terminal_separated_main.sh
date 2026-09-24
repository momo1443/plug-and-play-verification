#!/usr/bin/env bash
set -euo pipefail

# Full A1 after the separated smoke has verified model load, agent flow,
# private-test reward, actor update, and checkpoint creation.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TACO_A1_TRAINER_GPUS=4
export TACO_A1_STANDALONE_ROLLOUT=true
export TACO_A1_ROLLOUT_GPUS=4
export TACO_A1_ROLLOUT_NNODES=1
export TACO_A1_TENSOR_PARALLEL_SIZE=1
export TACO_A1_CHECKPOINT_ENGINE_BACKEND=nccl
export TACO_A1_CHECKPOINT_BUCKET_MB=128
export TACO_A1_TOTAL_TRAINING_STEPS=300
export TACO_A1_TRAIN_MAX_SAMPLES=6000
export TACO_A1_SAVE_FREQ=50
export TACO_A1_MAX_PROMPT_LENGTH=8192
export TACO_A1_MAX_RESPONSE_LENGTH=2048
export TACO_A1_MAX_MODEL_LENGTH=12288
export TACO_A1_MAX_BATCHED_TOKENS=12288
export TACO_A1_MAX_NUM_SEQS=20
export TACO_A1_VLLM_MEMORY_UTILIZATION=0.50
export TACO_A1_VLLM_KV_CACHE_BYTES=0
export TACO_A1_ACTOR_PARAM_OFFLOAD=false
export TACO_A1_ACTOR_OPTIMIZER_OFFLOAD=false
export RUN_ID="qwen35-4b_taco_a1_terminal_separated_main300_actor4_rollout4_$(date +%Y%m%d-%H%M%S)"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_a1_terminal.sh"
