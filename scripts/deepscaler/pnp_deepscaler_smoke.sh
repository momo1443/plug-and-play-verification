#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_DIR/scripts/common/pnp.sh"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DEEPSCALER_TRAIN_MAX_SAMPLES="${DEEPSCALER_TRAIN_MAX_SAMPLES:-4}"
export DEEPSCALER_TRAIN_BATCH_SIZE="${DEEPSCALER_TRAIN_BATCH_SIZE:-4}"
export DEEPSCALER_ROLLOUT_N="${DEEPSCALER_ROLLOUT_N:-2}"
export DEEPSCALER_TOTAL_TRAINING_STEPS="${DEEPSCALER_TOTAL_TRAINING_STEPS:-$PNP_DEFAULT_TRAINING_STEPS}"
# Exercise process rewards immediately; this is not the paper warmup schedule.
export DEEPSCALER_EM_WARMUP_STEPS="${DEEPSCALER_EM_WARMUP_STEPS:-0}"
export DEEPSCALER_VAL_MAX_SAMPLES="${DEEPSCALER_VAL_MAX_SAMPLES:-2}"
export DEEPSCALER_VAL_BATCH_SIZE="${DEEPSCALER_VAL_BATCH_SIZE:-2}"
export DEEPSCALER_MAX_RESPONSE_LENGTH="${DEEPSCALER_MAX_RESPONSE_LENGTH:-512}"
export DEEPSCALER_MAX_MODEL_LENGTH="${DEEPSCALER_MAX_MODEL_LENGTH:-4096}"
export DEEPSCALER_VLLM_MAX_NUM_SEQS="${DEEPSCALER_VLLM_MAX_NUM_SEQS:-4}"
export DEEPSCALER_OUTPUT_DIR="${DEEPSCALER_OUTPUT_DIR:-$PROJECT_DIR/logs/deepscaler_4b_verification_smoke_$(date +%Y%m%d-%H%M%S)}"
export DEEPSCALER_TEST_FREQ="${DEEPSCALER_TEST_FREQ:-1}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/pnp-math-smoke-$$}"
export HOTPOTQA_SAVE_FREQ="${HOTPOTQA_SAVE_FREQ:--1}"

cd "$PROJECT_DIR"
exec bash scripts/deepscaler/pnp_deepscaler.sh \
    ray_kwargs.ray_init.num_cpus=8 actor_rollout_ref.rollout.agent.num_workers=2 "$@"
