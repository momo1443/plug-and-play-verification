#!/usr/bin/env bash
set -euo pipefail

# Paper single-turn DeepScaleR GRPO with pluggable process verification.
# For the multi-turn 4096/2048 ToolEnv profile use run_deepscaler_tool_a9.sh.
# Paper allocation: 2048/4096 (4B), 2048/5120 (9B).
#
# Schedule (DEEPSCALER_EM_WARMUP_STEPS=50 by default):
#   Steps 1-50:    Strict terminal EM reward — cold-start stabilization
#   Steps 51-500:  w ~ U(0,1), reward = w * EM + (1-w) * process_reward
#
# Process reward: average verified-equation fractions over equation-bearing
# reasoning steps; the process reward is zero when no equations are extracted.
#
# 6-GPU configuration with Qwen3.5-4B:
#   - vllm_gpu_memory_utilization=0.25
#   - max_model_len=8192
#   - max_num_seqs=20
#   - grpo_micro_batch_size=1
#   - rollout_n=4
#   - train_batch_size=20
#   - total_training_steps=500 by default
#   - save_freq=50
#   - ref_kl=0.001 (low_var_kl)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

# Model
export HOTPOTQA_MODEL_PATH="${DEEPSCALER_MODEL_PATH:-${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}}"
source "$PROJECT_DIR/examples/common/model_training.sh"
agent_r1_model_overrides "$HOTPOTQA_MODEL_PATH"
agent_r1_optimizer_overrides "$HOTPOTQA_MODEL_PATH"
agent_r1_paper_profile deepscaler "$HOTPOTQA_MODEL_PATH"
export DEEPSCALER_REWARD_MODE="${DEEPSCALER_REWARD_MODE:-uniform_equation_process}"

# ── A9 Schedule Config ───────────────────────────────────────────
export DEEPSCALER_EM_WARMUP_STEPS="${DEEPSCALER_EM_WARMUP_STEPS:-50}"

# Data paths — DeepScaleR
TRAIN_PATH="$PROJECT_DIR/data/corpus/deepscaler/train.parquet"
VAL_PATH="$PROJECT_DIR/data/corpus/deepscaler/validation.parquet"

# Training config
TRAIN_MAX_SAMPLES="${DEEPSCALER_TRAIN_MAX_SAMPLES:-${HOTPOTQA_TRAIN_MAX_SAMPLES:-$PAPER_TRAIN_SAMPLES}}"
TRAIN_BATCH_SIZE="${HOTPOTQA_TRAIN_BATCH_SIZE:-20}"
ROLLOUT_N="${HOTPOTQA_ROLLOUT_N:-4}"

TOTAL_TRAINING_STEPS="${DEEPSCALER_TOTAL_TRAINING_STEPS:-${HOTPOTQA_TOTAL_TRAINING_STEPS:-$PAPER_TRAIN_STEPS}}"
if ! [[ "$TOTAL_TRAINING_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "HOTPOTQA_TOTAL_TRAINING_STEPS must be a positive integer" >&2
    exit 2
fi

VAL_MAX_SAMPLES=2015
VAL_BATCH_SIZE="${HOTPOTQA_VAL_BATCH_SIZE:-8}"

# GPU config
IFS=',' read -r -a _VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#_VISIBLE_GPUS[@]}"
GRPO_MICRO_BATCH_SIZE=1

# GRPO config
GRPO_GAMMA=1.0
REFERENCE_KL_ENABLED=true
REFERENCE_KL_LOSS_COEF=0.001
REFERENCE_KL_LOSS_TYPE=low_var_kl
KL_IN_REWARD=false

# Sequence length config — aligned with DeepMath experiment
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=$PAPER_RESPONSE_LENGTH
MAX_MODEL_LENGTH=8192
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=20
VLLM_GPU_MEMORY_UTILIZATION=0.25
VLLM_ENABLE_SLEEP_MODE=false
VLLM_FREE_CACHE_ENGINE=false

# Checkpoint config
SAVE_FREQ="${HOTPOTQA_SAVE_FREQ:-50}"
MAX_ACTOR_CKPT_TO_KEEP="${HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP:-2}"
EXPERIMENT_SEED="${HOTPOTQA_SEED:-42}"

# Resume config
RESUME_MODE="${HOTPOTQA_RESUME_MODE:-disable}"
RESUME_FROM_PATH="${HOTPOTQA_RESUME_FROM_PATH:-}"

# Resolve resume
case "$RESUME_MODE" in
    disable|auto)
        RESUME_FROM_CONFIG=null
        ;;
    resume_path)
        if [[ -z "$RESUME_FROM_PATH" ]]; then
            echo "HOTPOTQA_RESUME_FROM_PATH is required when HOTPOTQA_RESUME_MODE=resume_path" >&2
            exit 2
        fi
        RESUME_FROM_PATH="$(cd "$RESUME_FROM_PATH" && pwd -P)"
        RESUME_FROM_CONFIG="$RESUME_FROM_PATH"
        ;;
    *)
        echo "HOTPOTQA_RESUME_MODE must be disable, auto, or resume_path" >&2
        exit 2
        ;;
esac

# Ray tmp
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
    export RAY_TMPDIR="$RAY_TMP_LINK/deepscaler-a9-$$"
fi

# Run ID and output
RUN_ID="${RUN_ID:-${AGENT_R1_MODEL_NAME}_deepscaler_a9uniform_grpo_stepcausal_main30k_n4_300step_6gpu_vllm025_mlen8192_mseq20_mb1_refkl001_save50_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${HOTPOTQA_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

"$PYTHON_BIN" -m recipes.deepscaler.prepare_run \
    --project-dir "$PROJECT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$HOTPOTQA_MODEL_PATH" \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --resume-from-path "$RESUME_FROM_PATH" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --num-gpus "$NUM_GPUS" \
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm-max-model-len "$MAX_MODEL_LENGTH" \
    --vllm-max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --vllm-max-num-seqs "$MAX_NUM_SEQS" \
    --train-max-samples "$TRAIN_MAX_SAMPLES" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --rollout-n "$ROLLOUT_N" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-response-length "$MAX_RESPONSE_LENGTH" \
    --total-training-steps "$TOTAL_TRAINING_STEPS" \
    --em-warmup-steps "$DEEPSCALER_EM_WARMUP_STEPS" \
    --save-freq "$SAVE_FREQ" \
    --seed "$EXPERIMENT_SEED"

echo "=== DeepScaleR A9 Uniform GRPO ==="
echo "Model: $HOTPOTQA_MODEL_PATH"
echo "Train: $TRAIN_PATH ($TRAIN_MAX_SAMPLES samples)"
echo "Val:   $VAL_PATH ($VAL_MAX_SAMPLES samples)"
echo "GPUs:  $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
echo "Steps: $TOTAL_TRAINING_STEPS"
echo "Warmup: $DEEPSCALER_EM_WARMUP_STEPS (EM-only steps)"
echo "Runtime profile: deepscaler_paper; prompt=$MAX_PROMPT_LENGTH; completion=$MAX_RESPONSE_LENGTH"
echo "Output: $OUTPUT_DIR"
echo "=================================="

CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra"]'

"$PYTHON_BIN" -m "$AGENT_R1_TRAINER_MODULE" \
    "${AGENT_R1_ALGORITHM_ARGS[@]}" \
    algorithm.gamma="$GRPO_GAMMA" \
    algorithm.use_kl_in_reward="$KL_IN_REWARD" \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.shuffle=false \
    data.seed="$EXPERIMENT_SEED" \
    data.train_max_samples="$TRAIN_MAX_SAMPLES" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.val_max_samples="$VAL_MAX_SAMPLES" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HOTPOTQA_MODEL_PATH" \
    "${AGENT_R1_MODEL_OVERRIDES[@]}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$GRPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss="$REFERENCE_KL_ENABLED" \
    actor_rollout_ref.actor.kl_loss_coef="$REFERENCE_KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type="$REFERENCE_KL_LOSS_TYPE" \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.seed="$EXPERIMENT_SEED" \
    actor_rollout_ref.actor.checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS" \
    actor_rollout_ref.actor.checkpoint.load_contents="$CHECKPOINT_SAVE_CONTENTS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
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
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$PROJECT_DIR/recipes/deepscaler/paper.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_flow=deepscaler_paper_agent \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    "${AGENT_R1_CRITIC_ARGS[@]}" \
    reward_model.enable=False \
    +reward_model.launch_reward_fn_async=False \
    custom_reward_function.path="$PROJECT_DIR/recipes/deepscaler/reward_fn.py" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$PROJECT_DIR/recipes/deepscaler/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    +trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console"]' \
    trainer.project_name=DeepScaler_A9_AGENT_R1 \
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
