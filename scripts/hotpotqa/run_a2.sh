#!/usr/bin/env bash
set -euo pipefail

# A2 (verifier-only agentic RLVR): GRPO with step_causal credit assignment,
# per-step deterministic evidence reward on search actions, final answer masked out
# of the policy loss. The final turn stays a raw model completion; validation still
# reports terminal exact match.
#
# Self-contained entrypoint: `bash run_a2.sh` launches A2 with Qwen3.5-4B on the
# all eight GPUs and the shared first-30k-of-90k contract. Knobs remain overridable, e.g.
#   HOTPOTQA_TOTAL_TRAINING_STEPS=800 bash run_a2.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_a2.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A2
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

export RUN_ID="${RUN_ID:-qwen35-4b_a2_${HOTPOTQA_RUN_MODE}_grpo_stepcausal_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
