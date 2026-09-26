#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/../../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

export HOTPOTQA_REWARD_ARM=A8_LR30
export HOTPOTQA_LR_REASON_STEP_FORMAT=dsl
export HOTPOTQA_LR_REWARD_MODE=lr30
export HOTPOTQA_LR_EM_WARMUP_STEPS=0
source "$HERE/common_v2.sh"
export RUN_ID="${RUN_ID:-qwen35-4b_a8v2_lr30_main30k_n4_1500step_6gpu_vllm025_mlen8192_mseq20_autokv_save50_$(date +%Y%m%d-%H%M%S)}"
export HOTPOTQA_OUTPUT_DIR="${HOTPOTQA_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"

exec bash "$PROJECT_DIR/scripts/hotpotqa/run_rlvr.sh"
