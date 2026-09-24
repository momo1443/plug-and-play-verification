#!/usr/bin/env bash
set -euo pipefail

# A9: certificate-grounded process reward. Every search/finish action after
# the bootstrap carries a certificate sidecar with {source_id, support_span,
# target/answer_span}. The verifier checks grounding and coupling only —
# it does NOT verify selection correctness.
#
# Main A9: prompt-group-shared uniform reward after a 50-step EM warmup.
#   Steps   1–50:  1.0 * EM + 0.0 * cert
#   Steps 51–1500: w * EM + (1-w) * cert, w ~ U(0,1) per prompt group

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A9_CERT_MIX
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_A9_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-50}"
export HOTPOTQA_A9_FORMAT_GATE=0
export HOTPOTQA_A9_FORMAT_PENALTY=0
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false
# A new A9 run must create a formal manifest before Ray or vLLM starts.
export HOTPOTQA_SKIP_PREFLIGHT="${HOTPOTQA_SKIP_PREFLIGHT:-0}"

export RUN_ID="${RUN_ID:-qwen35-4b_a9_certmix_emwarm50_main30k_n4_1500step_6gpu_vllm040_refkl001_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
