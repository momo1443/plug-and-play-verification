#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

export HOTPOTQA_LR_REASON_STEP_FORMAT="${HOTPOTQA_LR_REASON_STEP_FORMAT:-dsl}"
case "$HOTPOTQA_LR_REASON_STEP_FORMAT" in
    dsl) export HOTPOTQA_REWARD_ARM=A8_LR_BASE ;;
    claim_source) export HOTPOTQA_REWARD_ARM=A8_LR_BASE_CS ;;
    *) echo "HOTPOTQA_LR_REASON_STEP_FORMAT must be dsl or claim_source" >&2; exit 2 ;;
esac
export HOTPOTQA_LR_REWARD_MODE=terminal_only
export HOTPOTQA_LR_EM_WARMUP_STEPS=0
source "$HERE/common_v2.sh"
export RUN_ID="${RUN_ID:-qwen35-4b_a8v2_lr_base_${HOTPOTQA_LR_REASON_STEP_FORMAT}_main30k_n4_1500step_6gpu_vllm025_mlen8192_mseq20_autokv_save50_$(date +%Y%m%d-%H%M%S)}"
export HOTPOTQA_OUTPUT_DIR="${HOTPOTQA_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"

exec bash "$PROJECT_DIR/examples/hotpotqa/run_rlvr.sh"
