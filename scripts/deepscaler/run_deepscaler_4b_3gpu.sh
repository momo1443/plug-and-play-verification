#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export DEEPSCALER_MODEL_PATH="${DEEPSCALER_MODEL_PATH:-$PROJECT_DIR/models/Qwen3.5-4B}"
export DEEPSCALER_TRAIN_BATCH_SIZE=20
export DEEPSCALER_ROLLOUT_N=4
export DEEPSCALER_TRAIN_MAX_SAMPLES=10000
export DEEPSCALER_TOTAL_TRAINING_STEPS=500
export DEEPSCALER_MAX_RESPONSE_LENGTH=4096
export DEEPSCALER_MAX_MODEL_LENGTH=8192
export DEEPSCALER_MAX_STEPS=5
export DEEPSCALER_FINAL_TURN_SUBMISSION="${DEEPSCALER_FINAL_TURN_SUBMISSION:-True}"
export DEEPSCALER_FILTER_OVERLONG_PROMPTS=True
# Periodic held-out evaluation is disabled; rollout verification still runs.
export DEEPSCALER_TEST_FREQ="${DEEPSCALER_TEST_FREQ:--1}"
export DEEPSCALER_SAVE_FREQ="${DEEPSCALER_SAVE_FREQ:-50}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/pnp-ray-$$}"
# LR has no warmup. The existing A9 EM-only reward phase stays at 50 updates.
export DEEPSCALER_EM_WARMUP_STEPS="${DEEPSCALER_EM_WARMUP_STEPS:-50}"
export DEEPSCALER_OUTPUT_DIR="${DEEPSCALER_OUTPUT_DIR:-$PROJECT_DIR/logs/deepscaler_4b_3gpu_500step_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$DEEPSCALER_OUTPUT_DIR"
cd "$PROJECT_DIR"

if [[ -z "${DEEPSCALER_TRAIN_PATH:-}" ]]; then
    TRAIN_SOURCE_PATH="$PROJECT_DIR/data/corpus/deepscaler/train.parquet"
    if [[ "${DEEPSCALER_CLEAN_DATA:-True}" == "True" ]]; then
        CLEAN_DATA_DIR="$DEEPSCALER_OUTPUT_DIR/_meta/data"
        "$PYTHON_BIN" -m recipes.deepscaler.clean_data --output-dir "$CLEAN_DATA_DIR"
        TRAIN_SOURCE_PATH="$CLEAN_DATA_DIR/train.parquet"
        export DEEPSCALER_VAL_PATH="${DEEPSCALER_VAL_PATH:-$CLEAN_DATA_DIR/validation.parquet}"
    fi
    export DEEPSCALER_TRAIN_PATH="$DEEPSCALER_OUTPUT_DIR/train_10000_prompt2048.parquet"
    "$PYTHON_BIN" -m recipes.deepscaler.prepare_training_subset \
        --input "$TRAIN_SOURCE_PATH" \
        --output "$DEEPSCALER_TRAIN_PATH" --model "$DEEPSCALER_MODEL_PATH"
fi

exec bash scripts/deepscaler/run_deepscaler_a9_uniform.sh \
    ray_kwargs.ray_init.num_cpus="${DEEPSCALER_RAY_NUM_CPUS:-12}" \
    data.dataloader_num_workers=4 data.filter_overlong_prompts_workers=4 \
    actor_rollout_ref.rollout.agent.num_workers="${DEEPSCALER_AGENT_NUM_WORKERS:-6}" "$@"
