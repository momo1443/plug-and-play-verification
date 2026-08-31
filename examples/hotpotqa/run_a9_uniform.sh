#!/usr/bin/env bash
set -euo pipefail

# A9 uniform random weights: terminal_weight ~ U(0,1), process_weight = 1 - terminal_weight
# 使用 GPU 0,1,4,5,6,7（6 张卡）

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A9_CERT_MIX
export CUDA_VISIBLE_DEVICES="0,1,4,5,6,7"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_A9_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-100}"
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false
export HOTPOTQA_SKIP_PREFLIGHT="${HOTPOTQA_SKIP_PREFLIGHT:-0}"

# Disk-saving: save every 100 steps, keep only 1 actor checkpoint
export HOTPOTQA_SAVE_FREQ=100
export HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP=1

# Use local /tmp for Ray (short path to avoid AF_UNIX 107-byte limit)
export RAY_TMPDIR="/tmp/rwa9uniform$$"
mkdir -p "$RAY_TMPDIR"
export RAY_TMPDIR

export RUN_ID="${RUN_ID:-qwen35-4b_a9_uniform_emwarm100_main30k_n4_1500step_6gpu_vllm040_refkl001_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
