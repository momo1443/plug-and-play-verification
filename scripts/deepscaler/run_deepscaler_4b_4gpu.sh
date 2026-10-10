#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DEEPSCALER_OUTPUT_DIR="${DEEPSCALER_OUTPUT_DIR:-$PROJECT_DIR/logs/deepscaler_4b_4gpu_500step_$(date +%Y%m%d-%H%M%S)}"
export DEEPSCALER_VLLM_GPU_MEMORY_UTILIZATION="${DEEPSCALER_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export DEEPSCALER_VLLM_MAX_NUM_SEQS="${DEEPSCALER_VLLM_MAX_NUM_SEQS:-32}"
export DEEPSCALER_RAY_NUM_CPUS="${DEEPSCALER_RAY_NUM_CPUS:-16}"
export DEEPSCALER_AGENT_NUM_WORKERS="${DEEPSCALER_AGENT_NUM_WORKERS:-8}"
export DEEPSCALER_SAVE_FREQ="${DEEPSCALER_SAVE_FREQ:-10}"

# Use the same 500-update algorithm and token budgets as the three-GPU profile.
exec bash "$PROJECT_DIR/scripts/deepscaler/run_deepscaler_4b_3gpu.sh" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ALLOW_RUNTIME_LORA_UPDATING="false"' "$@"
