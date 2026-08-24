#!/usr/bin/env bash
set -euo pipefail

# A9: certificate-grounded process reward. Every search/finish action after
# the bootstrap carries a certificate sidecar with {source_id, support_span,
# target/answer_span}. The verifier checks grounding and coupling only —
# it does NOT verify selection correctness.
#
# Single arm: cert_mix (0.8 EM + 0.2 cert) with 100-step EM warmup.
#   Steps   1–100:  1.0 * EM + 0.0 * cert  (cold start / protocol tax control)
#   Steps 101–1500: 0.8 * EM + 0.2 * cert  (certificate process reward active)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A9_CERT_MIX
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_A9_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-100}"
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false
# A new A9 run must create a formal manifest before Ray or vLLM starts.
export HOTPOTQA_SKIP_PREFLIGHT="${HOTPOTQA_SKIP_PREFLIGHT:-0}"

export RUN_ID="${RUN_ID:-qwen35-4b_a9_certmix_emwarm100_main30k_n4_1500step_6gpu_vllm040_refkl001_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
