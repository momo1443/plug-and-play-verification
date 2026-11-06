#!/usr/bin/env bash
set -euo pipefail

# A3 (gold-evidence upper bound): step-causal GRPO combining terminal exact
# match with the privileged per-search new-gold-evidence reward. Like the
# verifier and judge arms, it uses 50 outcome-only warmup updates, then one
# prompt-group-shared w ~ U(0, 1): reward = w * EM + (1-w) * gold_process.
# Final-answer tokens remain in the policy loss under raw completion.
#
# Self-contained entrypoint; shared scientific/runtime defaults are owned by
# pnp_hotpotqa_train.sh and remain overridable through documented HOTPOTQA_* variables.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A3
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

export RUN_ID="${RUN_ID:-qwen35-4b_a3_${HOTPOTQA_RUN_MODE}_grpo_stepcausal_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/pnp_hotpotqa_train.sh"
