#!/usr/bin/env bash
set -euo pipefail

# A9 strict format: uniform U(0,1) reward + format_gate
# 每条 trajectory 随机采样 w~U(0,1)，reward = w * terminal_em + (1-w) * process
# 如果模型没产出合法 finish tool call，归零全部 reward（format_gate）
# 使用 GPU 0,1,4,5,6,7（6 张卡）

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A9_CERT_MIX
export CUDA_VISIBLE_DEVICES="0,1,4,5,6,7"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_A9_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-50}"
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false
export HOTPOTQA_SKIP_PREFLIGHT="${HOTPOTQA_SKIP_PREFLIGHT:-0}"

# 严格格式强制：finish tool call 不合规时归零全部 reward
export HOTPOTQA_A9_FORMAT_GATE=1
export HOTPOTQA_A9_FORMAT_PENALTY="${HOTPOTQA_A9_FORMAT_PENALTY:-0.1}"

# Match old certmix experiment VLLM config
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION=0.25

# Save every 50 steps, keep only 1 actor checkpoint
export HOTPOTQA_SAVE_FREQ=50
export HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP=1

# Use local /tmp for Ray (short path to avoid AF_UNIX 107-byte limit)
export RAY_TMPDIR="/tmp/rwa9strict$$"
mkdir -p "$RAY_TMPDIR"
export RAY_TMPDIR

export RUN_ID="${RUN_ID:-qwen35-4b_a9_strict_formatgate_emwarm50_main10k_n4_500step_6gpu_vllm025_refkl001_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/../../hotpotqa/run_rlvr.sh"
