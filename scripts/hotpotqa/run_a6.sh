#!/usr/bin/env bash
set -euo pipefail

# A6: frozen Qwen3.5-9B causal process judge without gold supporting facts.
# Updates 1-50 use terminal EM only; later updates use a prompt-group-shared
# w ~ Uniform(0,1) mixture of terminal EM and judge process credits.
# Eight A40 GPUs: seven actor GPUs and one dedicated judge GPU.
# Final-answer tokens remain in the policy loss under raw completion.
#
# Self-contained entrypoint; shared scientific/runtime defaults are owned by
# pnp_hotpotqa_train.sh and remain overridable through documented HOTPOTQA_* variables.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A6
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

export HOTPOTQA_JUDGE_GPU="${HOTPOTQA_JUDGE_GPU:-7}"
export HOTPOTQA_JUDGE_MODEL="${HOTPOTQA_JUDGE_MODEL:-$WORKSPACE_DIR/models/Qwen3.5-9B}"
export HOTPOTQA_JUDGE_PORT="${HOTPOTQA_JUDGE_PORT:-29500}"
export HOTPOTQA_JUDGE_MAX_MODEL_LEN="${HOTPOTQA_JUDGE_MAX_MODEL_LEN:-4096}"
export HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION="${HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION:-0.50}"
export HOTPOTQA_JUDGE_EXTERNAL=1
IFS=',' read -r -a ACTOR_GPUS <<<"$CUDA_VISIBLE_DEVICES"
export HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-${#ACTOR_GPUS[@]}}"
export HOTPOTQA_AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-$HOTPOTQA_NUM_GPUS}"
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.40}"

export RUN_ID="${RUN_ID:-qwen35-4b_a6_causal_judge_emwarm50_uniform_actor7_judge1_first10k_n4_500step_$(date +%Y%m%d-%H%M%S)}"
export HOTPOTQA_OUTPUT_DIR="${HOTPOTQA_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"

exec bash "$HERE/run_with_judge.sh"
