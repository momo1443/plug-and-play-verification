#!/usr/bin/env bash
set -euo pipefail

# Deprecated name retained for compatibility. It now resumes with the common
# prompt-group-shared uniform A9 contract rather than a fixed reward mixture.
# Uses GPU 0,1,4,5,6,7（6 张卡）

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A9_CERT_MIX
export CUDA_VISIBLE_DEVICES="0,1,4,5,6,7"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_A9_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-50}"
export HOTPOTQA_A9_FORMAT_GATE=1
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false
export HOTPOTQA_SKIP_PREFLIGHT="${HOTPOTQA_SKIP_PREFLIGHT:-0}"
export HOTPOTQA_A9_FORMAT_PENALTY=0

# Match old certmix experiment VLLM config
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION=0.25

# Save every 50 steps, keep only 1 actor checkpoint
export HOTPOTQA_SAVE_FREQ=50
export HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP=1

# Resume from uniform experiment checkpoint at step 150
export HOTPOTQA_RESUME_MODE=resume_path
export HOTPOTQA_RESUME_FROM_PATH="$WORKSPACE_DIR/logs/qwen35-4b_a9_uniform_emwarm100_main10k_n4_500step_6gpu_vllm025_refkl001_20260902-100414/checkpoints/global_step_150"

# Use local /tmp for Ray (short path to avoid AF_UNIX 107-byte limit)
export RAY_TMPDIR="/tmp/rwa9fixed$$"
mkdir -p "$RAY_TMPDIR"
export RAY_TMPDIR

export RUN_ID="${RUN_ID:-qwen35-4b_a9_uniform_resume_emwarm50_main10k_n4_500step_6gpu_vllm025_refkl001_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/../../hotpotqa/run_rlvr.sh"
