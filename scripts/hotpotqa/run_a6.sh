#!/usr/bin/env bash
set -euo pipefail

# A6 (A3-LLM-MATCHED): step-causal GRPO with 0.5 times the
# LLM-judge cumulative semantic-coverage increment plus 0.5 times terminal
# exact match. Uses Qwen3.5-9B as the judge model on GPU 1.
# Final-answer tokens remain in the policy loss under raw completion.
#
# Self-contained entrypoint; shared scientific/runtime defaults are owned by
# run_rlvr.sh and remain overridable through documented HOTPOTQA_* variables.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A6
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

export HOTPOTQA_JUDGE_GPU="${HOTPOTQA_JUDGE_GPU:-1}"
export HOTPOTQA_JUDGE_MODEL="${HOTPOTQA_JUDGE_MODEL:-$WORKSPACE_DIR/models/Qwen3.5-9B}"
export HOTPOTQA_JUDGE_PORT="${HOTPOTQA_JUDGE_PORT:-29500}"
export HOTPOTQA_JUDGE_MAX_MODEL_LEN="${HOTPOTQA_JUDGE_MAX_MODEL_LEN:-4096}"
export HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION="${HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION:-0.50}"
export HOTPOTQA_JUDGE_EXTERNAL=1
export HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-6}"
export HOTPOTQA_AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-6}"
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.40}"

export RUN_ID="${RUN_ID:-qwen35-4b_a6_llmjudge_matched_6gpu_first30k_n4_fusedtriton_vllm040_actor8192_noentropy_refkl001_$(date +%Y%m%d-%H%M%S)}"
export HOTPOTQA_OUTPUT_DIR="${HOTPOTQA_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"

exec bash "$HERE/run_with_judge.sh"
