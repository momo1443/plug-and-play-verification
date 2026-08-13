#!/usr/bin/env bash
set -euo pipefail

# A3 (combined agentic RLVR): step-causal GRPO with 0.5 times the
# deterministic per-search new-evidence reward plus 0.5 times terminal exact
# match. Final-answer tokens remain in the policy loss under raw completion.
#
# Self-contained entrypoint; shared scientific/runtime defaults are owned by
# run_rlvr.sh and remain overridable through documented HOTPOTQA_* variables.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A3
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

export RUN_ID="${RUN_ID:-qwen35-4b_a3_${HOTPOTQA_RUN_MODE}_grpo_stepcausal_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
