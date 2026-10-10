#!/usr/bin/env bash
set -euo pipefail

# PNP: terminal_weight ~ U(0,1), process_weight = 1 - terminal_weight.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A9_CERT_MIX
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
export HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-${#_VISIBLE_GPUS[@]}}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_A9_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-50}"
export HOTPOTQA_A9_FORMAT_GATE=1
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false
export HOTPOTQA_SKIP_PREFLIGHT="${HOTPOTQA_SKIP_PREFLIGHT:-0}"
export HOTPOTQA_A9_FORMAT_PENALTY=0

# Match old certmix experiment VLLM config
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"

# Save every 50 steps, keep only 1 actor checkpoint
export HOTPOTQA_SAVE_FREQ="${HOTPOTQA_SAVE_FREQ:-50}"
export HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP="${HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP:-1}"

# Use local /tmp for Ray (short path to avoid AF_UNIX 107-byte limit)
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/pnp-hotpotqa-$$}"
mkdir -p "$RAY_TMPDIR"
export RAY_TMPDIR

export RUN_ID="${RUN_ID:-pnp_hotpotqa_$(basename "$HOTPOTQA_MODEL_PATH")_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/pnp_hotpotqa_train.sh" "$@"
