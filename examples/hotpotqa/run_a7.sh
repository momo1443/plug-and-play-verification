#!/usr/bin/env bash
set -euo pipefail

# A7 (A3-WEAK-MATCHED): step-causal GRPO with 0.5 times the weak execution
# process reward (1/3 per verified model search with non-empty observation)
# plus 0.5 times terminal exact match. The weak verifier checks only that
# a model-generated search action produced a valid, observable environment
# transition — it does NOT verify query relevance, evidence correctness, or
# gold-fact coverage. All training parameters match A3 exactly except for the
# process verifier.
#
# Self-contained entrypoint; shared scientific/runtime defaults are owned by
# run_rlvr.sh and remain overridable through documented HOTPOTQA_* variables.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A7
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

# A7 uses 6 GPUs: same resource layout as A6 minus the judge GPU.
# Actor FSDP + rollout on 6 GPUs; no judge server needed.
export HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-6}"
export HOTPOTQA_AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-6}"
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.40}"

export RUN_ID="${RUN_ID:-qwen35-4b_a7_weakexec_matched_6gpu_first30k_n4_fusedtriton_vllm040_actor8192_noentropy_refkl001_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
