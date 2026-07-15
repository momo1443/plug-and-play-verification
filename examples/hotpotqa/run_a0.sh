#!/usr/bin/env bash
set -euo pipefail
set -x

if (( $# != 0 )); then
    echo "Formal A0 does not accept trailing overrides; use frozen environment variables" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HOTPOTQA_FORMAL_A0=1
export HOTPOTQA_ENABLE_THINKING=false
export HOTPOTQA_FORCE_FIRST_SEARCH=false
export HOTPOTQA_REQUIRE_SENTENCE_EVIDENCE=true
export HOTPOTQA_DATA_ROOT="${HOTPOTQA_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa}"
export HOTPOTQA_CORPUS_DATA_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa_corpus}"
export HOTPOTQA_EVIDENCE_SIDECAR="${HOTPOTQA_EVIDENCE_SIDECAR:-$HOTPOTQA_CORPUS_DATA_ROOT/hotpotqa_evidence_v1.sqlite3}"
export HOTPOTQA_EMBEDDING_MODEL="${HOTPOTQA_EMBEDDING_MODEL:-$WORKSPACE_DIR/models/bge-large-en-v1.5}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
export HOTPOTQA_VAL_MAX_SAMPLES="${HOTPOTQA_VAL_MAX_SAMPLES:--1}"
export HOTPOTQA_NUM_GPUS=8
export HOTPOTQA_AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-8}"
export HOTPOTQA_STREAMING_RESUME="${HOTPOTQA_STREAMING_RESUME:-0}"
export RUN_ID="${RUN_ID:-qwen35-4b_formal_a0_agentflow_8a40_$(date +%Y%m%d-%H%M%S)}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$WORKSPACE_DIR/results/$RUN_ID}"

mkdir -p "$VALIDATION_DATA_DIR"
cd "$PROJECT_DIR"

"$PYTHON_BIN" -m recipes.hotpotqa.prepare_formal_a0_run \
    --project_dir "$PROJECT_DIR" \
    --validation_path "${HOTPOTQA_VAL_PATH:-$HOTPOTQA_DATA_ROOT/validation.parquet}" \
    --corpus_dir "$HOTPOTQA_CORPUS_DATA_ROOT" \
    --evidence_sidecar_path "$HOTPOTQA_EVIDENCE_SIDECAR" \
    --model_path "$HOTPOTQA_MODEL_PATH" \
    --embedding_model_path "$HOTPOTQA_EMBEDDING_MODEL" \
    --output_dir "$VALIDATION_DATA_DIR" \
    --artifact_lock_path "$HOTPOTQA_CORPUS_DATA_ROOT/formal_a0_artifact_lock.json" \
    --val_max_samples "$HOTPOTQA_VAL_MAX_SAMPLES" \
    --num_gpus 8 \
    --agent_workers "$HOTPOTQA_AGENT_WORKERS"

bash examples/hotpotqa/run_validation.sh

"$PYTHON_BIN" -m recipes.hotpotqa.validate_formal_a0_output \
    "$VALIDATION_DATA_DIR/0.jsonl" \
    --manifest "$VALIDATION_DATA_DIR/run_manifest.json" \
    --mark_complete
