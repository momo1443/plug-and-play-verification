#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
    echo "Formal HotpotQA RLVR does not accept trailing Hydra overrides; use documented environment variables" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

ARM="${HOTPOTQA_REWARD_ARM:?HOTPOTQA_REWARD_ARM must identify a supported formal arm}"
case "$ARM" in
    A1|A2|A3|A6|A7|A9|A9_CERT_MIX|A8_LR30|A8_LR30_CS|A8_LR_BASE|A8_LR_BASE_CS) ;;
    *) echo "Unsupported HOTPOTQA_REWARD_ARM: $ARM" >&2; exit 2 ;;
esac
if [[ "$ARM" == "A8_LR30_CS" || "$ARM" == "A8_LR_BASE_CS" ]]; then
    export HOTPOTQA_LR_REASON_STEP_FORMAT="${HOTPOTQA_LR_REASON_STEP_FORMAT:-claim_source}"
    if [[ "$HOTPOTQA_LR_REASON_STEP_FORMAT" != "claim_source" ]]; then
        echo "A8_LR30_CS requires HOTPOTQA_LR_REASON_STEP_FORMAT=claim_source" >&2
        exit 2
    fi
elif [[ "$ARM" == "A8_LR30" || "$ARM" == "A8_LR_BASE" ]]; then
    export HOTPOTQA_LR_REASON_STEP_FORMAT="${HOTPOTQA_LR_REASON_STEP_FORMAT:-dsl}"
    if [[ "$HOTPOTQA_LR_REASON_STEP_FORMAT" != "dsl" ]]; then
        echo "A8_LR30 requires HOTPOTQA_LR_REASON_STEP_FORMAT=dsl; use A8_LR30_CS for claim_source" >&2
        exit 2
    fi
fi
if [[ "$ARM" == "A8_LR_BASE" || "$ARM" == "A8_LR_BASE_CS" ]]; then
    export HOTPOTQA_LR_REWARD_MODE="${HOTPOTQA_LR_REWARD_MODE:-terminal_only}"
    [[ "$HOTPOTQA_LR_REWARD_MODE" == "terminal_only" ]] || {
        echo "$ARM requires HOTPOTQA_LR_REWARD_MODE=terminal_only" >&2
        exit 2
    }
elif [[ "$ARM" == A8_LR30* ]]; then
    export HOTPOTQA_LR_REWARD_MODE="${HOTPOTQA_LR_REWARD_MODE:-lr30}"
    [[ "$HOTPOTQA_LR_REWARD_MODE" == "lr30" ]] || {
        echo "$ARM requires HOTPOTQA_LR_REWARD_MODE=lr30" >&2
        exit 2
    }
fi
if [[ "$ARM" == A9* ]]; then
    # A9 uses its own warmup env var; pipe it to the shared LR_EM_WARMUP_STEPS
    # so preflight/manifest records the correct value.
    LR_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-100}"
fi

if [[ "$ARM" == A8_LR* ]]; then
    AGENT_FLOW_CONFIG="$PROJECT_DIR/recipes/hotpotqa_lr/base.yaml"
    DEFAULT_AGENT_FLOW=hotpotqa_local_reasoning_agent
    REWARD_FUNCTION_PATH="$PROJECT_DIR/recipes/hotpotqa_lr/reward_fn.py"
elif [[ "$ARM" == A9* ]]; then
    AGENT_FLOW_CONFIG="$PROJECT_DIR/recipes/hotpotqa_a9/base.yaml"
    DEFAULT_AGENT_FLOW=hotpotqa_certificate_agent
    REWARD_FUNCTION_PATH="$PROJECT_DIR/recipes/hotpotqa_a9/reward_fn.py"
else
    AGENT_FLOW_CONFIG="$PROJECT_DIR/recipes/hotpotqa/base.yaml"
    DEFAULT_AGENT_FLOW=hotpotqa_agent
    REWARD_FUNCTION_PATH="$PROJECT_DIR/recipes/hotpotqa/reward_fn.py"
fi

RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
case "$RUN_MODE" in
    main)
        DEFAULT_TRAIN_MAX_SAMPLES=30000
        DEFAULT_TRAIN_BATCH_SIZE=20
        ;;
    pilot64)
        DEFAULT_TRAIN_MAX_SAMPLES=64
        DEFAULT_TRAIN_BATCH_SIZE=16
        ;;
    pilot2048)
        DEFAULT_TRAIN_MAX_SAMPLES=2048
        DEFAULT_TRAIN_BATCH_SIZE=16
        ;;
    *)
        echo "HOTPOTQA_RUN_MODE must be main, pilot64, or pilot2048, got: $RUN_MODE" >&2
        exit 2
        ;;
esac

# Every formal mode uses a frozen deterministic source prefix.
TRAIN_MAX_SAMPLES="${HOTPOTQA_TRAIN_MAX_SAMPLES:-$DEFAULT_TRAIN_MAX_SAMPLES}"
TRAIN_BATCH_SIZE="${HOTPOTQA_TRAIN_BATCH_SIZE:-$DEFAULT_TRAIN_BATCH_SIZE}"
ROLLOUT_N="${HOTPOTQA_ROLLOUT_N:-4}"
GRPO_MICRO_BATCH_SIZE="${HOTPOTQA_GRPO_MICRO_BATCH_SIZE:-2}"
if [[ -n "${HOTPOTQA_TOTAL_TRAINING_STEPS:-}" ]]; then
    TOTAL_TRAINING_STEPS="$HOTPOTQA_TOTAL_TRAINING_STEPS"
elif (( TRAIN_MAX_SAMPLES % TRAIN_BATCH_SIZE == 0 )); then
    TOTAL_TRAINING_STEPS="$((TRAIN_MAX_SAMPLES / TRAIN_BATCH_SIZE))"
else
    echo "$RUN_MODE requires HOTPOTQA_TOTAL_TRAINING_STEPS when train samples are not divisible by batch size" >&2
    exit 2
fi
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra"]'

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export HOTPOTQA_FORMAL_A0=0
export HOTPOTQA_FORMAL_EXPERIMENT=1
export HOTPOTQA_REWARD_ARM="$ARM"
export HOTPOTQA_ENABLE_THINKING=false
export HOTPOTQA_FORCE_FIRST_SEARCH=false
# Raw-final-only is enforced by the shared AgentFlow; no runtime switch exists.
export HOTPOTQA_REQUIRE_SENTENCE_EVIDENCE=true
export HOTPOTQA_EMBEDDING_PER_WORKER_GPU=0
export HOTPOTQA_EMBEDDING_DEVICE=cpu

export HOTPOTQA_DATA_ROOT="${HOTPOTQA_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa}"
export HOTPOTQA_CORPUS_DATA_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa_corpus}"
export HOTPOTQA_EVIDENCE_SIDECAR="${HOTPOTQA_EVIDENCE_SIDECAR:-$HOTPOTQA_CORPUS_DATA_ROOT/hotpotqa_evidence_v1.sqlite3}"
export HOTPOTQA_EMBEDDING_MODEL="${HOTPOTQA_EMBEDDING_MODEL:-$WORKSPACE_DIR/models/bge-large-en-v1.5}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

TRAIN_PATH="${HOTPOTQA_TRAIN_PATH:-$HOTPOTQA_DATA_ROOT/train.parquet}"
VAL_PATH="${HOTPOTQA_VAL_PATH:-$HOTPOTQA_DATA_ROOT/validation.parquet}"
VAL_MAX_SAMPLES=7405
VAL_BATCH_SIZE="${HOTPOTQA_VAL_BATCH_SIZE:-8}"
if [[ -n "${HOTPOTQA_NUM_GPUS:-}" ]]; then
    NUM_GPUS="$HOTPOTQA_NUM_GPUS"
else
    IFS=',' read -r -a _VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
    NUM_GPUS="${#_VISIBLE_GPUS[@]}"
fi
AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-$NUM_GPUS}"
GRPO_GAMMA=1.0
# Enable verl/GRPO's existing actor-loss reference KL uniformly for every
# trainable arm so A1/A2/A3 differ only in their task-reward contracts.
REFERENCE_KL_ENABLED=true
REFERENCE_KL_LOSS_COEF=0.001
REFERENCE_KL_LOSS_TYPE=low_var_kl
KL_IN_REWARD=false
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LENGTH="${HOTPOTQA_VLLM_MAX_MODEL_LEN:-12288}"
MAX_NUM_BATCHED_TOKENS="${HOTPOTQA_VLLM_MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LENGTH}"
MAX_NUM_SEQS="${HOTPOTQA_VLLM_MAX_NUM_SEQS:-$((TRAIN_BATCH_SIZE * ROLLOUT_N))}"
VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.40}"
VLLM_ENABLE_SLEEP_MODE="${HOTPOTQA_VLLM_ENABLE_SLEEP_MODE:-false}"
VLLM_FREE_CACHE_ENGINE="${HOTPOTQA_VLLM_FREE_CACHE_ENGINE:-false}"
VLLM_KV_CACHE_MEMORY_BYTES="${HOTPOTQA_VLLM_KV_CACHE_MEMORY_BYTES:-}"
LR_EM_WARMUP_STEPS="${LR_EM_WARMUP_STEPS:-${HOTPOTQA_LR_EM_WARMUP_STEPS:-0}}"
LR_REWARD_MODE="${HOTPOTQA_LR_REWARD_MODE:-lr30}"
DATA_SHUFFLE="${HOTPOTQA_DATA_SHUFFLE:-false}"
MODEL_ENABLE_GRADIENT_CHECKPOINTING="${HOTPOTQA_ENABLE_GRADIENT_CHECKPOINTING:-true}"
ACTOR_USE_DYNAMIC_BSZ="${HOTPOTQA_ACTOR_USE_DYNAMIC_BSZ:-true}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${HOTPOTQA_ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}"
ENTROPY_CHUNK_ROWS="${HOTPOTQA_ENTROPY_CHUNK_ROWS:-512}"
CALCULATE_ENTROPY="${HOTPOTQA_CALCULATE_ENTROPY:-false}"
USE_FUSED_KERNELS="${HOTPOTQA_USE_FUSED_KERNELS:-true}"
FUSED_KERNEL_BACKEND="${HOTPOTQA_FUSED_KERNEL_BACKEND:-triton}"
ACTOR_PARAM_OFFLOAD="${HOTPOTQA_ACTOR_PARAM_OFFLOAD:-false}"
ACTOR_OPTIMIZER_OFFLOAD="${HOTPOTQA_ACTOR_OPTIMIZER_OFFLOAD:-false}"
SAVE_FREQ="${HOTPOTQA_SAVE_FREQ:-100}"
MAX_ACTOR_CKPT_TO_KEEP="${HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP:-2}"
RESUME_MODE="${HOTPOTQA_RESUME_MODE:-disable}"
RESUME_FROM_PATH="${HOTPOTQA_RESUME_FROM_PATH:-}"
CALIBRATION_REPORT="${HOTPOTQA_LR_CALIBRATION_REPORT:-}"
ALLOW_UNCALIBRATED_LAUNCH="${HOTPOTQA_LR_ALLOW_UNCALIBRATED_LAUNCH:-0}"
EXPERIMENT_SEED="${HOTPOTQA_SEED:-42}"
if ! [[ "$EXPERIMENT_SEED" =~ ^[0-9]+$ ]]; then
    echo "HOTPOTQA_SEED must be a non-negative integer, got $EXPERIMENT_SEED" >&2
    exit 2
fi

for bool_name in \
    VLLM_ENABLE_SLEEP_MODE VLLM_FREE_CACHE_ENGINE DATA_SHUFFLE \
    MODEL_ENABLE_GRADIENT_CHECKPOINTING ACTOR_USE_DYNAMIC_BSZ CALCULATE_ENTROPY USE_FUSED_KERNELS \
    ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD; do
    case "${!bool_name}" in
        true|false) ;;
        *) echo "$bool_name must be true or false, got ${!bool_name}" >&2; exit 2 ;;
    esac
done
case "$FUSED_KERNEL_BACKEND" in
    triton|torch) ;;
    *) echo "FUSED_KERNEL_BACKEND must be triton or torch, got $FUSED_KERNEL_BACKEND" >&2; exit 2 ;;
esac
if ! [[ "$ENTROPY_CHUNK_ROWS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ENTROPY_CHUNK_ROWS must be a positive integer, got $ENTROPY_CHUNK_ROWS" >&2
    exit 2
fi
if ! [[ "$ACTOR_MAX_TOKEN_LEN_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACTOR_MAX_TOKEN_LEN_PER_GPU must be a positive integer, got $ACTOR_MAX_TOKEN_LEN_PER_GPU" >&2
    exit 2
fi
for int_name in MAX_MODEL_LENGTH MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS; do
    if ! [[ "${!int_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$int_name must be a positive integer, got ${!int_name}" >&2
        exit 2
    fi
done
if [[ -n "$VLLM_KV_CACHE_MEMORY_BYTES" ]] && ! [[ "$VLLM_KV_CACHE_MEMORY_BYTES" =~ ^[1-9][0-9]*$ ]]; then
    echo "HOTPOTQA_VLLM_KV_CACHE_MEMORY_BYTES must be a positive integer, got $VLLM_KV_CACHE_MEMORY_BYTES" >&2
    exit 2
fi
if ! [[ "$LR_EM_WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
    echo "HOTPOTQA_LR_EM_WARMUP_STEPS must be a non-negative integer, got $LR_EM_WARMUP_STEPS" >&2
    exit 2
fi
case "$ALLOW_UNCALIBRATED_LAUNCH" in
    0|1) ;;
    *) echo "HOTPOTQA_LR_ALLOW_UNCALIBRATED_LAUNCH must be 0 or 1" >&2; exit 2 ;;
esac
export HOTPOTQA_LR_EM_WARMUP_STEPS="$LR_EM_WARMUP_STEPS"
case "$RESUME_MODE" in
    disable|auto)
        if [[ -n "$RESUME_FROM_PATH" ]]; then
            echo "HOTPOTQA_RESUME_FROM_PATH is only valid with HOTPOTQA_RESUME_MODE=resume_path" >&2
            exit 2
        fi
        RESUME_FROM_CONFIG=null
        ;;
    resume_path)
        if [[ -z "$RESUME_FROM_PATH" ]]; then
            echo "HOTPOTQA_RESUME_FROM_PATH is required when HOTPOTQA_RESUME_MODE=resume_path" >&2
            exit 2
        fi
        if [[ ! -d "$RESUME_FROM_PATH" ]]; then
            echo "Resume checkpoint directory does not exist: $RESUME_FROM_PATH" >&2
            exit 2
        fi
        RESUME_FROM_PATH="$(cd "$RESUME_FROM_PATH" && pwd -P)"
        checkpoint_name="${RESUME_FROM_PATH##*/}"
        if [[ ! "$checkpoint_name" =~ ^global_step_([0-9]+)$ ]]; then
            echo "Resume checkpoint must end in global_step_<N>, got: $RESUME_FROM_PATH" >&2
            exit 2
        fi
        RESUME_GLOBAL_STEP="${BASH_REMATCH[1]}"
        if (( RESUME_GLOBAL_STEP >= TOTAL_TRAINING_STEPS )); then
            echo "Resume step $RESUME_GLOBAL_STEP must be below total training steps $TOTAL_TRAINING_STEPS" >&2
            exit 2
        fi
        if [[ ! -f "$RESUME_FROM_PATH/data.pt" || ! -d "$RESUME_FROM_PATH/actor" ]]; then
            echo "Resume checkpoint is missing data.pt or actor/: $RESUME_FROM_PATH" >&2
            exit 2
        fi
        model_shard="$(find "$RESUME_FROM_PATH/actor" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' -print -quit)"
        if [[ ! "${model_shard##*/}" =~ ^model_world_size_([0-9]+)_rank_[0-9]+\.pt$ ]]; then
            echo "Could not determine checkpoint world size from $RESUME_FROM_PATH/actor" >&2
            exit 2
        fi
        CHECKPOINT_WORLD_SIZE="${BASH_REMATCH[1]}"
        if (( CHECKPOINT_WORLD_SIZE != NUM_GPUS )); then
            echo "Checkpoint world size $CHECKPOINT_WORLD_SIZE does not match requested GPU count $NUM_GPUS" >&2
            exit 2
        fi
        for shard_kind in model optim extra_state; do
            shard_count="$(find "$RESUME_FROM_PATH/actor" -maxdepth 1 -type f \
                -name "${shard_kind}_world_size_${CHECKPOINT_WORLD_SIZE}_rank_*.pt" | wc -l)"
            if (( shard_count != CHECKPOINT_WORLD_SIZE )); then
                echo "Checkpoint requires $CHECKPOINT_WORLD_SIZE $shard_kind shards, found $shard_count" >&2
                exit 2
            fi
        done
        RESUME_FROM_CONFIG="$RESUME_FROM_PATH"
        ;;
    *)
        echo "HOTPOTQA_RESUME_MODE must be disable, auto, or resume_path; got $RESUME_MODE" >&2
        exit 2
        ;;
esac
export AGENT_R1_ENTROPY_CHUNK_ROWS="$ENTROPY_CHUNK_ROWS"

RUN_ID="${RUN_ID:-qwen35-4b_${ARM,,}_${RUN_MODE}_grpo_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${HOTPOTQA_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"
# Keep Ray's lexical path short for AF_UNIX sockets while storing its sessions
# on the project filesystem instead of the nearly-full root filesystem.
if [[ -z "${RAY_TMPDIR:-}" ]]; then
    RAY_STORAGE_ROOT="${HOTPOTQA_RAY_STORAGE_ROOT:-$WORKSPACE_DIR/.ray_tmp}"
    mkdir -p "$RAY_STORAGE_ROOT"
    RAY_STORAGE_ROOT="$(cd "$RAY_STORAGE_ROOT" && pwd -P)"
    RAY_TMP_LINK="${HOTPOTQA_RAY_TMP_LINK:-/tmp/ar1r-${UID}}"
    if [[ -L "$RAY_TMP_LINK" ]]; then
        if [[ "$(readlink "$RAY_TMP_LINK")" != "$RAY_STORAGE_ROOT" ]]; then
            echo "Ray temp link points to an unexpected target: $RAY_TMP_LINK -> $(readlink "$RAY_TMP_LINK")" >&2
            exit 2
        fi
    elif [[ -e "$RAY_TMP_LINK" ]]; then
        echo "Ray temp link path exists and is not a symlink: $RAY_TMP_LINK" >&2
        exit 2
    else
        ln -s "$RAY_STORAGE_ROOT" "$RAY_TMP_LINK"
    fi
    export RAY_TMPDIR="$RAY_TMP_LINK/${ARM,,}-$$"
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# Formal launches fail closed by default. Config-only rendering is the only path
# that intentionally avoids artifact and live-resource checks.
SKIP_PREFLIGHT="${HOTPOTQA_SKIP_PREFLIGHT:-0}"
case "$SKIP_PREFLIGHT" in
    0|1) ;;
    *) echo "HOTPOTQA_SKIP_PREFLIGHT must be 0 or 1, got: $SKIP_PREFLIGHT" >&2; exit 2 ;;
esac
if [[ "$ARM" == A8_LR* && "$SKIP_PREFLIGHT" == "1" ]]; then
    echo "A8-LR-v2 preflight cannot be skipped" >&2
    exit 2
fi
if [[ "${HOTPOTQA_HYDRA_CONFIG_ONLY:-0}" != "1" && "$SKIP_PREFLIGHT" != "1" ]]; then
    if [[ "$ARM" == A8_LR* ]]; then
        "$PYTHON_BIN" -m recipes.hotpotqa_lr.prepare_run \
        --project-dir "$PROJECT_DIR" \
        --arm "$ARM" \
        --reward-mode "$LR_REWARD_MODE" \
        --train-path "$TRAIN_PATH" \
        --validation-path "$VAL_PATH" \
        --corpus-dir "$HOTPOTQA_CORPUS_DATA_ROOT" \
        --evidence-sidecar-path "$HOTPOTQA_EVIDENCE_SIDECAR" \
        --model-path "$HOTPOTQA_MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --train-max-samples "$TRAIN_MAX_SAMPLES" \
        --val-max-samples "$VAL_MAX_SAMPLES" \
        --train-batch-size "$TRAIN_BATCH_SIZE" \
        --rollout-n "$ROLLOUT_N" \
        --total-training-steps "$TOTAL_TRAINING_STEPS" \
        --grpo-micro-batch-size "$GRPO_MICRO_BATCH_SIZE" \
        --data-shuffle "$DATA_SHUFFLE" \
        --actor-use-dynamic-bsz "$ACTOR_USE_DYNAMIC_BSZ" \
        --actor-max-token-len-per-gpu "$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
        --reference-kl-enabled "$REFERENCE_KL_ENABLED" \
        --reference-kl-loss-coef "$REFERENCE_KL_LOSS_COEF" \
        --reference-kl-loss-type "$REFERENCE_KL_LOSS_TYPE" \
        --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --vllm-max-model-len "$MAX_MODEL_LENGTH" \
        --vllm-max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --vllm-max-num-seqs "$MAX_NUM_SEQS" \
        --vllm-kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES" \
        --save-freq "$SAVE_FREQ" \
        --max-actor-ckpt-to-keep "$MAX_ACTOR_CKPT_TO_KEEP" \
        --num-gpus "$NUM_GPUS" \
        --agent-workers "$AGENT_WORKERS" \
        --em-warmup-steps "$LR_EM_WARMUP_STEPS" \
        --gamma "$GRPO_GAMMA" \
        --calibration-report "$CALIBRATION_REPORT" \
        --allow-uncalibrated-launch "$ALLOW_UNCALIBRATED_LAUNCH" \
        --seed "$EXPERIMENT_SEED"
    elif [[ "$ARM" == A9* ]]; then
        "$PYTHON_BIN" -m recipes.hotpotqa_a9.prepare_run \
        --project-dir "$PROJECT_DIR" \
        --arm "$ARM" \
        --train-path "$TRAIN_PATH" \
        --validation-path "$VAL_PATH" \
        --corpus-dir "$HOTPOTQA_CORPUS_DATA_ROOT" \
        --evidence-sidecar-path "$HOTPOTQA_EVIDENCE_SIDECAR" \
        --model-path "$HOTPOTQA_MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --train-max-samples "$TRAIN_MAX_SAMPLES" \
        --val-max-samples "$VAL_MAX_SAMPLES" \
        --train-batch-size "$TRAIN_BATCH_SIZE" \
        --rollout-n "$ROLLOUT_N" \
        --total-training-steps "$TOTAL_TRAINING_STEPS" \
        --grpo-micro-batch-size "$GRPO_MICRO_BATCH_SIZE" \
        --data-shuffle "$DATA_SHUFFLE" \
        --actor-use-dynamic-bsz "$ACTOR_USE_DYNAMIC_BSZ" \
        --actor-max-token-len-per-gpu "$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
        --reference-kl-enabled "$REFERENCE_KL_ENABLED" \
        --reference-kl-loss-coef "$REFERENCE_KL_LOSS_COEF" \
        --reference-kl-loss-type "$REFERENCE_KL_LOSS_TYPE" \
        --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --vllm-max-model-len "$MAX_MODEL_LENGTH" \
        --vllm-max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --vllm-max-num-seqs "$MAX_NUM_SEQS" \
        --vllm-kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES" \
        --save-freq "$SAVE_FREQ" \
        --max-actor-ckpt-to-keep "$MAX_ACTOR_CKPT_TO_KEEP" \
        --num-gpus "$NUM_GPUS" \
        --agent-workers "$AGENT_WORKERS" \
        --em-warmup-steps "$LR_EM_WARMUP_STEPS" \
        --gamma "$GRPO_GAMMA" \
        --allow-uncalibrated-launch "$ALLOW_UNCALIBRATED_LAUNCH" \
        --seed "$EXPERIMENT_SEED"
    else
        "$PYTHON_BIN" -m recipes.hotpotqa.prepare_formal_rlvr_run \
        --project_dir "$PROJECT_DIR" \
        --arm "$ARM" \
        --run_mode "$RUN_MODE" \
        --train_path "$TRAIN_PATH" \
        --validation_path "$VAL_PATH" \
        --corpus_dir "$HOTPOTQA_CORPUS_DATA_ROOT" \
        --evidence_sidecar_path "$HOTPOTQA_EVIDENCE_SIDECAR" \
        --model_path "$HOTPOTQA_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --train_max_samples "$TRAIN_MAX_SAMPLES" \
        --val_max_samples "$VAL_MAX_SAMPLES" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --rollout_n "$ROLLOUT_N" \
        --total_training_steps "$TOTAL_TRAINING_STEPS" \
        --grpo_micro_batch_size "$GRPO_MICRO_BATCH_SIZE" \
        --data_shuffle "$DATA_SHUFFLE" \
        --actor_use_dynamic_bsz "$ACTOR_USE_DYNAMIC_BSZ" \
        --actor_max_token_len_per_gpu "$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
        --entropy_chunk_rows "$ENTROPY_CHUNK_ROWS" \
        --calculate_entropy "$CALCULATE_ENTROPY" \
        --reference_kl_enabled "$REFERENCE_KL_ENABLED" \
        --reference_kl_loss_coef "$REFERENCE_KL_LOSS_COEF" \
        --reference_kl_loss_type "$REFERENCE_KL_LOSS_TYPE" \
        --kl_in_reward "$KL_IN_REWARD" \
        --use_fused_kernels "$USE_FUSED_KERNELS" \
        --fused_kernel_backend "$FUSED_KERNEL_BACKEND" \
        --actor_param_offload "$ACTOR_PARAM_OFFLOAD" \
        --actor_optimizer_offload "$ACTOR_OPTIMIZER_OFFLOAD" \
        --enable_gradient_checkpointing "$MODEL_ENABLE_GRADIENT_CHECKPOINTING" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --vllm_enable_sleep_mode "$VLLM_ENABLE_SLEEP_MODE" \
        --vllm_free_cache_engine "$VLLM_FREE_CACHE_ENGINE" \
        --save_freq "$SAVE_FREQ" \
        --max_actor_ckpt_to_keep "$MAX_ACTOR_CKPT_TO_KEEP" \
        --resume_mode "$RESUME_MODE" \
        --resume_from_path "$RESUME_FROM_CONFIG" \
        --num_gpus "$NUM_GPUS" \
        --agent_workers "$AGENT_WORKERS" \
        --gamma "$GRPO_GAMMA"
    fi
fi

if [[ "${HOTPOTQA_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

HYDRA_CONFIG_ARGS=()
if [[ "${HOTPOTQA_HYDRA_CONFIG_ONLY:-0}" == "1" ]]; then
    HYDRA_CONFIG_ARGS=(--cfg job)
fi
VLLM_ENGINE_EXTRA_ARGS=()
if [[ -n "$VLLM_KV_CACHE_MEMORY_BYTES" ]]; then
    VLLM_ENGINE_EXTRA_ARGS+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes="$VLLM_KV_CACHE_MEMORY_BYTES")
fi

# Upstream verl ActorConfig retains ppo_* field names below; they carry GRPO batch settings.
"$PYTHON_BIN" -m agent_r1.trainer.main_agent_grpo \
    "${HYDRA_CONFIG_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    ++algorithm.grpo.credit_assignment=step_causal \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.gamma="$GRPO_GAMMA" \
    algorithm.use_kl_in_reward="$KL_IN_REWARD" \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.shuffle="$DATA_SHUFFLE" \
    data.seed="$EXPERIMENT_SEED" \
    data.train_max_samples="$TRAIN_MAX_SAMPLES" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.val_max_samples="$VAL_MAX_SAMPLES" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$HOTPOTQA_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS" \
    actor_rollout_ref.model.fused_kernel_options.impl_backend="$FUSED_KERNEL_BACKEND" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing="$MODEL_ENABLE_GRADIENT_CHECKPOINTING" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$GRPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz="$ACTOR_USE_DYNAMIC_BSZ" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss="$REFERENCE_KL_ENABLED" \
    actor_rollout_ref.actor.kl_loss_coef="$REFERENCE_KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type="$REFERENCE_KL_LOSS_TYPE" \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.calculate_entropy="$CALCULATE_ENTROPY" \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.seed="$EXPERIMENT_SEED" \
    actor_rollout_ref.actor.checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS" \
    actor_rollout_ref.actor.checkpoint.load_contents="$CHECKPOINT_SAVE_CONTENTS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    +actor_rollout_ref.rollout.enable_sleep_mode="$VLLM_ENABLE_SLEEP_MODE" \
    actor_rollout_ref.rollout.free_cache_engine="$VLLM_FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.seed="$EXPERIMENT_SEED" \
    "${VLLM_ENGINE_EXTRA_ARGS[@]}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$AGENT_FLOW_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_flow="$DEFAULT_AGENT_FLOW" \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    critic.enable=False \
    reward_model.enable=False \
    +reward_model.launch_reward_fn_async=False \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    reward.custom_reward_function.name=compute_score \
    +trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console"]' \
    trainer.project_name=HotpotQA_AGENT_R1 \
    trainer.experiment_name="$RUN_ID" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="$RESUME_FROM_CONFIG" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.rollout_data_file="$OUTPUT_DIR/rollouts.jsonl" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" \
    trainer.log_val_generations=0 2>&1 | tee "$OUTPUT_DIR/train.log"
