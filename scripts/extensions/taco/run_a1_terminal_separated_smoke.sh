#!/usr/bin/env bash
set -euo pipefail

# Four actor GPUs plus four standalone TP=1 rollout replicas.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TACO_A1_TRAINER_GPUS=4
export TACO_A1_STANDALONE_ROLLOUT=true
export TACO_A1_ROLLOUT_GPUS=4
export TACO_A1_ROLLOUT_NNODES=1
export TACO_A1_TENSOR_PARALLEL_SIZE=1
export TACO_A1_CHECKPOINT_ENGINE_BACKEND=nccl
export TACO_A1_CHECKPOINT_BUCKET_MB=128
export TACO_A1_TOTAL_TRAINING_STEPS=1
export TACO_A1_SMOKE=true
export TACO_A1_TRAIN_BATCH_SIZE=4
export TACO_A1_TRAIN_MAX_SAMPLES=4
export TACO_A1_SAVE_FREQ=1
export TACO_A1_MAX_PROMPT_LENGTH=2048
export TACO_A1_MAX_RESPONSE_LENGTH=1024
export TACO_A1_MAX_MODEL_LENGTH=4096
export TACO_A1_MAX_BATCHED_TOKENS=4096
export TACO_A1_MAX_NUM_SEQS=4
export TACO_A1_VLLM_MEMORY_UTILIZATION=0.50
export TACO_A1_VLLM_KV_CACHE_BYTES=0
export TACO_A1_ACTOR_PARAM_OFFLOAD=false
export TACO_A1_ACTOR_OPTIMIZER_OFFLOAD=false
export RUN_ID="qwen35-4b_taco_a1_terminal_separated_smoke1_actor4_rollout4_$(date +%Y%m%d-%H%M%S)"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../taco/run_a1_terminal.sh"
