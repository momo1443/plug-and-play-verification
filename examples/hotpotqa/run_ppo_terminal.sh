#!/usr/bin/env bash
set -euo pipefail

# Experiment A: PPO terminal-EM-only baseline.
# The certificate agent interface remains enabled so the matched verifier arm
# changes only the optimizer-visible process reward.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A9_CERT_MIX
export HOTPOTQA_OPTIMIZER=ppo
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-8}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_A9_PROCESS_REWARD_ENABLED=false
export HOTPOTQA_A9_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-1500}"
export HOTPOTQA_A9_FORMAT_GATE=0
export HOTPOTQA_A9_FORMAT_PENALTY=0
export HOTPOTQA_PPO_LAM="${HOTPOTQA_PPO_LAM:-1.0}"
export HOTPOTQA_PPO_MICRO_BATCH_SIZE="${HOTPOTQA_PPO_MICRO_BATCH_SIZE:-1}"
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.22}"
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false

export RUN_ID="${RUN_ID:-qwen35-4b_ppo_terminal_em_${HOTPOTQA_RUN_MODE}_8gpu_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
