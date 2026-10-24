#!/usr/bin/env bash
set -euo pipefail
# Compatibility wrapper around the shared standalone PNP evaluator.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if (( $# != 0 )); then
    echo "Configure this wrapper with environment variables, or use run_pnp_eval.sh hotpotqa" >&2
    exit 2
fi
[[ "${HOTPOTQA_VAL_N:-1}" == 1 && "${HOTPOTQA_VAL_TEMPERATURE:-0.6}" =~ ^0[.]60*$ &&
   "${HOTPOTQA_VAL_TOP_P:-1}" =~ ^1([.]0+)?$ && "${HOTPOTQA_VAL_TOP_K:--1}" == -1 &&
   "${HOTPOTQA_VAL_DO_SAMPLE:-true}" =~ ^([Tt][Rr][Uu][Ee]|1)$ ]] || {
    echo "PNP main evaluation requires n=1, temperature=0.6, top-p=1, top-k=-1 and do_sample=true" >&2
    exit 2
}
DATA_ROOT="${HOTPOTQA_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa}"
CORPUS_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa_corpus}"
args=(hotpotqa
    --model "${HOTPOTQA_MODEL_PATH:?HOTPOTQA_MODEL_PATH must identify a vLLM-loadable model export}"
    --dataset "${HOTPOTQA_VAL_PATH:-$DATA_ROOT/validation.parquet}"
    --dataset-version "${HOTPOTQA_DATASET_VERSION:-distractor}"
    --dataset-revision "${HOTPOTQA_DATASET_REVISION:?Set HOTPOTQA_DATASET_REVISION to the frozen dataset snapshot}"
    --split validation --gpus "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" --corpus-dir "$CORPUS_ROOT"
    --gpu-memory-utilization "${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.3}")
[[ -z "${HOTPOTQA_SOURCE_CHECKPOINT:-}" ]] || args+=(--checkpoint "$HOTPOTQA_SOURCE_CHECKPOINT")
[[ -z "${HOTPOTQA_EVIDENCE_SIDECAR:-}" ]] || args+=(--evidence-sidecar "$HOTPOTQA_EVIDENCE_SIDECAR")
[[ -z "${VALIDATION_DATA_DIR:-}" ]] || args+=(--output "$VALIDATION_DATA_DIR")
[[ -z "${HOTPOTQA_EMBEDDING_MODEL:-}" ]] || args+=(--embedding-model "$HOTPOTQA_EMBEDDING_MODEL")
[[ -z "${HOTPOTQA_VLLM_MAX_MODEL_LEN:-}" ]] || args+=(--context-limit "$HOTPOTQA_VLLM_MAX_MODEL_LEN")
if [[ "${HOTPOTQA_VAL_MAX_SAMPLES:--1}" != -1 ]]; then
    args+=(--max-problems "$HOTPOTQA_VAL_MAX_SAMPLES")
fi
exec bash "$PROJECT_DIR/run_pnp_eval.sh" "${args[@]}"
