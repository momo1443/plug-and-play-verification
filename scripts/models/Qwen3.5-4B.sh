#!/usr/bin/env bash
set -euo pipefail

# Model profile for PNP on the Appendix C node with eight A40 GPUs.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_DIR/scripts/common/pnp.sh"
DATASET="${1:?Usage: Qwen3.5-4B.sh DATASET [HYDRA_OVERRIDES...] }"
shift
pnp_model_path "$DATASET" Qwen3.5-4B
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export AGENT_R1_MODEL_SCALE="${AGENT_R1_MODEL_SCALE:-4b}"

if [[ "$DATASET" == deepscaler ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    export DEEPSCALER_TRAIN_BATCH_SIZE="${DEEPSCALER_TRAIN_BATCH_SIZE:-20}"
    export DEEPSCALER_ROLLOUT_N="${DEEPSCALER_ROLLOUT_N:-4}"
    export DEEPSCALER_TRAIN_MAX_SAMPLES="${DEEPSCALER_TRAIN_MAX_SAMPLES:-10000}"
    export DEEPSCALER_TOTAL_TRAINING_STEPS="${DEEPSCALER_TOTAL_TRAINING_STEPS:-500}"
    export DEEPSCALER_MAX_RESPONSE_LENGTH="${DEEPSCALER_MAX_RESPONSE_LENGTH:-4096}"
    export DEEPSCALER_MAX_MODEL_LENGTH="${DEEPSCALER_MAX_MODEL_LENGTH:-8192}"
    export DEEPSCALER_MAX_STEPS="${DEEPSCALER_MAX_STEPS:-5}"
    export DEEPSCALER_FINAL_TURN_SUBMISSION="${DEEPSCALER_FINAL_TURN_SUBMISSION:-True}"
    export DEEPSCALER_FILTER_OVERLONG_PROMPTS=True
    export DEEPSCALER_TEST_FREQ="${DEEPSCALER_TEST_FREQ:--1}"
    export DEEPSCALER_SAVE_FREQ="${DEEPSCALER_SAVE_FREQ:-10}"
    export DEEPSCALER_EM_WARMUP_STEPS="${DEEPSCALER_EM_WARMUP_STEPS:-50}"
    export DEEPSCALER_VLLM_GPU_MEMORY_UTILIZATION="${DEEPSCALER_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
    export DEEPSCALER_VLLM_MAX_NUM_SEQS="${DEEPSCALER_VLLM_MAX_NUM_SEQS:-32}"
    export DEEPSCALER_OUTPUT_DIR="${DEEPSCALER_OUTPUT_DIR:-$PROJECT_DIR/logs/pnp_deepscaler_Qwen3.5-4B_$(date +%Y%m%d-%H%M%S)}"
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/pnp-ray-$$}"
    export RAY_enable_open_telemetry="${RAY_enable_open_telemetry:-0}"
    # Requires the installed checkpoint patch documented in the recipe.
    export VERL_FSDP_CHECKPOINT_OFFLOAD_TO_CPU="${VERL_FSDP_CHECKPOINT_OFFLOAD_TO_CPU:-0}"
    mkdir -p "$DEEPSCALER_OUTPUT_DIR"
    cd "$PROJECT_DIR"

    if [[ -z "${DEEPSCALER_TRAIN_PATH:-}" ]]; then
        TRAIN_SOURCE_PATH="$PROJECT_DIR/data/corpus/deepscaler/train.parquet"
        if [[ "${DEEPSCALER_CLEAN_DATA:-True}" == True ]]; then
            CLEAN_DATA_DIR="$DEEPSCALER_OUTPUT_DIR/_meta/data"
            "$PYTHON_BIN" -m recipes.deepscaler.clean_data --output-dir "$CLEAN_DATA_DIR"
            TRAIN_SOURCE_PATH="$CLEAN_DATA_DIR/train.parquet"
            export DEEPSCALER_VAL_PATH="${DEEPSCALER_VAL_PATH:-$CLEAN_DATA_DIR/validation.parquet}"
        fi
        export DEEPSCALER_TRAIN_PATH="$DEEPSCALER_OUTPUT_DIR/_meta/data/train_10000_prompt2048.parquet"
        "$PYTHON_BIN" -m recipes.deepscaler.prepare_training_subset \
            --input "$TRAIN_SOURCE_PATH" --output "$DEEPSCALER_TRAIN_PATH" --model "$DEEPSCALER_MODEL_PATH"
    fi
    IFS=',' read -r -a _VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
    exec bash "$PROJECT_DIR/scripts/deepscaler/pnp_deepscaler.sh" \
        ray_kwargs.ray_init.num_cpus="${DEEPSCALER_RAY_NUM_CPUS:-$(( ${#_VISIBLE_GPUS[@]} * 4 ))}" \
        data.dataloader_num_workers=4 data.filter_overlong_prompts_workers=4 \
        actor_rollout_ref.rollout.agent.num_workers="${DEEPSCALER_AGENT_NUM_WORKERS:-$(( ${#_VISIBLE_GPUS[@]} * 2 ))}" \
        '++ray_kwargs.ray_init._system_config.enable_open_telemetry=false' \
        '++ray_kwargs.ray_init.runtime_env.env_vars.RAY_enable_open_telemetry="0"' \
        '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ALLOW_RUNTIME_LORA_UPDATING="false"' "$@"
fi

pnp_launch_dataset "$DATASET" "$@"
