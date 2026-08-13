#!/usr/bin/env bash
set -euo pipefail

# A1 (terminal-only RLVR): GRPO with step_causal credit assignment, reward only on
# the final answer (exact match), using the raw model completion at the final turn.
#
# Self-contained entrypoint: `bash run_a1.sh` launches A1 with Qwen3.5-4B on the
# all eight GPUs and the shared first-30k-of-90k contract. Knobs remain overridable, e.g.
#   HOTPOTQA_TOTAL_TRAINING_STEPS=800 bash run_a1.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_a1.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A1
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

export RUN_ID="${RUN_ID:-qwen35-4b_a1_${HOTPOTQA_RUN_MODE}_grpo_stepcausal_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
