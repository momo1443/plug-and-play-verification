#!/usr/bin/env bash
set -euo pipefail
set -x

# A9 certificate-grounded validation-only launcher.
# Uses the hotpotqa_certificate_agent flow and A9 reward function.

if (( $# != 0 )); then
    echo "This launcher takes no arguments; configure it with environment variables" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

[[ "${HOTPOTQA_VALIDATION_INTERFACE:-a9}" == "a9" ]] || {
    echo "A9 validation requires HOTPOTQA_VALIDATION_INTERFACE=a9" >&2
    exit 2
}

export HOTPOTQA_FORMAL_A0=0
export HOTPOTQA_FORMAL_EXPERIMENT=1
export HOTPOTQA_REWARD_ARM=A9_CERT_MIX
export HOTPOTQA_VALIDATION_INTERFACE=a9
export HOTPOTQA_ENABLE_THINKING=false
export HOTPOTQA_FORCE_FIRST_SEARCH=false
export HOTPOTQA_REQUIRE_SENTENCE_EVIDENCE=true
export HOTPOTQA_EMBEDDING_PER_WORKER_GPU=0
export HOTPOTQA_EMBEDDING_DEVICE=cpu
export HOTPOTQA_STREAMING_RESULTS=1
export HOTPOTQA_STREAMING_FSYNC=1
export HOTPOTQA_STREAMING_METRIC_KEYS=acc,terminal_em
export HOTPOTQA_A9_EM_WARMUP_STEPS="${HOTPOTQA_A9_EM_WARMUP_STEPS:-50}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HOTPOTQA_DATA_ROOT="${HOTPOTQA_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa}"
export HOTPOTQA_CORPUS_DATA_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa_corpus}"
export HOTPOTQA_EVIDENCE_SIDECAR="${HOTPOTQA_EVIDENCE_SIDECAR:-$HOTPOTQA_CORPUS_DATA_ROOT/hotpotqa_evidence_v1.sqlite3}"
export HOTPOTQA_EMBEDDING_MODEL="${HOTPOTQA_EMBEDDING_MODEL:-$WORKSPACE_DIR/models/bge-large-en-v1.5}"

MODEL_PATH="${HOTPOTQA_MODEL_PATH:?HOTPOTQA_MODEL_PATH is required}"
TRAIN_PATH="${HOTPOTQA_TRAIN_PATH:-$HOTPOTQA_DATA_ROOT/train.parquet}"
VAL_PATH="${HOTPOTQA_VAL_PATH:-$HOTPOTQA_DATA_ROOT/validation.parquet}"
AGENT_CONFIG="$PROJECT_DIR/recipes/hotpotqa_a9/base.yaml"
REWARD_PATH="$PROJECT_DIR/recipes/hotpotqa_a9/reward_fn.py"
SOURCE_CHECKPOINT="${HOTPOTQA_SOURCE_CHECKPOINT:-}"

VAL_MAX_SAMPLES="${HOTPOTQA_VAL_MAX_SAMPLES:--1}"
VAL_BATCH_SIZE="${HOTPOTQA_VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH_PER_STEP=1024
MAX_MODEL_LENGTH="${HOTPOTQA_VLLM_MAX_MODEL_LEN:-12288}"
MAX_NUM_BATCHED_TOKENS="${HOTPOTQA_VLLM_MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LENGTH}"
IFS=',' read -r -a CUDA_DEVICE_LIST <<<"$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${HOTPOTQA_NUM_GPUS:-${#CUDA_DEVICE_LIST[@]}}"
TRAIN_BATCH_SIZE="${HOTPOTQA_TRAIN_BATCH_SIZE:-$(( (8 + NUM_GPUS - 1) / NUM_GPUS * NUM_GPUS ))}"
AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-$NUM_GPUS}"
VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
VLLM_MAX_NUM_SEQS="${HOTPOTQA_VLLM_MAX_NUM_SEQS:-8}"
VLLM_ENABLE_SLEEP_MODE="${HOTPOTQA_VLLM_ENABLE_SLEEP_MODE:-false}"
VLLM_FREE_CACHE_ENGINE="${HOTPOTQA_VLLM_FREE_CACHE_ENGINE:-false}"
VAL_N="${HOTPOTQA_VAL_N:-1}"
VAL_DO_SAMPLE="${HOTPOTQA_VAL_DO_SAMPLE:-false}"
VAL_TEMPERATURE="${HOTPOTQA_VAL_TEMPERATURE:-0}"
VAL_TOP_P="${HOTPOTQA_VAL_TOP_P:-1}"
VAL_TOP_K="${HOTPOTQA_VAL_TOP_K:--1}"

if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || (( NUM_GPUS != ${#CUDA_DEVICE_LIST[@]} )); then
    echo "HOTPOTQA_NUM_GPUS must match CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi
declare -A SEEN_CUDA_DEVICES=()
for cuda_device in "${CUDA_DEVICE_LIST[@]}"; do
    if [[ ! "$cuda_device" =~ ^[0-9]+$ || -n "${SEEN_CUDA_DEVICES[$cuda_device]:-}" ]]; then
        echo "CUDA_VISIBLE_DEVICES entries must be unique non-negative integers" >&2
        exit 2
    fi
    SEEN_CUDA_DEVICES["$cuda_device"]=1
done
unset cuda_device SEEN_CUDA_DEVICES
if (( TRAIN_BATCH_SIZE <= 0 || TRAIN_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "HOTPOTQA_TRAIN_BATCH_SIZE must be positive and divisible by the GPU count" >&2
    exit 2
fi
if [[ "$VAL_N" != "1" || "${VAL_DO_SAMPLE,,}" != "false" || "$VAL_TEMPERATURE" != "0" ]]; then
    echo "A9 validation requires n=1, do_sample=false, temperature=0" >&2
    exit 2
fi
for bool_name in VLLM_ENABLE_SLEEP_MODE VLLM_FREE_CACHE_ENGINE; do
    case "${!bool_name}" in
        true|false) ;;
        *) echo "$bool_name must be true or false" >&2; exit 2 ;;
    esac
done
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config: $MODEL_PATH" >&2; exit 2; }
find "$MODEL_PATH" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q . || {
    echo "Missing model safetensors: $MODEL_PATH" >&2
    exit 2
}

RUN_ID="${RUN_ID:-qwen35-4b_a9-certmix_validation_${NUM_GPUS}gpu_$(date +%Y%m%d-%H%M%S)}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ar1-a9val-$$}"
mkdir -p "$VALIDATION_DATA_DIR"
cd "$PROJECT_DIR"

# Skip A9 preflight for validation-only runs
export HOTPOTQA_SKIP_PREFLIGHT=1

"$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.train_max_samples="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.val_max_samples="$VAL_MAX_SAMPLES" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH_PER_STEP" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    +actor_rollout_ref.rollout.enable_sleep_mode="$VLLM_ENABLE_SLEEP_MODE" \
    actor_rollout_ref.rollout.free_cache_engine="$VLLM_FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$VLLM_MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$AGENT_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_flow=hotpotqa_certificate_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K" \
    critic.enable=False \
    reward_model.enable=False \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    +trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console"]' \
    trainer.project_name=HotpotQA_AGENT_R1 \
    trainer.experiment_name="$RUN_ID" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.log_val_generations=4 2>&1 | tee "$VALIDATION_DATA_DIR/launcher.log"
