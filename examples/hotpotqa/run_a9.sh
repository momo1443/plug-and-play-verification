#!/usr/bin/env bash
set -euo pipefail

# A9: outcome-independent behavioral process reward. Every eligible search
# earns 0.5 only when both probes pass. Both-fail earns 0; partial or
# unscorable probes mask only the local process component. Final EM keeps 0.5.
# The model-facing HotpotQA prompt, tool schema, and final-answer protocol are
# identical to the existing raw-completion arms.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A9
if [[ "${HOTPOTQA_RUN_MODE:-main}" != "main" ]]; then
    echo "A9 has one formal experiment: HOTPOTQA_RUN_MODE must be main" >&2
    exit 2
fi
export HOTPOTQA_RUN_MODE=main
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3,4,5,6,7}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_A9_PROBE_SEED="${HOTPOTQA_A9_PROBE_SEED:-42}"
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false
# A new A9 run must create a formal manifest before Ray or vLLM starts.
export HOTPOTQA_SKIP_PREFLIGHT="${HOTPOTQA_SKIP_PREFLIGHT:-0}"

export RUN_ID="${RUN_ID:-qwen35-4b_a9_behavioral_main30k_n4_1500step_6gpu_vllm040_refkl001_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/run_rlvr.sh"
