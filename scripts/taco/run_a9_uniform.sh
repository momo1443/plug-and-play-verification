#!/usr/bin/env bash
set -euo pipefail

# Main A9-Code pilot: terminal private-test reward for steps 1--50, then
# per-prompt-group uniform A9 mixing for steps 51--300.
# Five turns let the model write, test, revise, retest, and submit a final code
# artifact with a replayable execution certificate. This launcher only starts
# after model-safe TACO parquet files and their runner-only sidecar exist.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
# A1 and A9 intentionally share one frozen task/test split.
export TACO_A9_DATA_ROOT="${TACO_A9_DATA_ROOT:-$WORKSPACE_DIR/data/coding_benchmarks/taco_a1_prepared}"
export TACO_A9_SIDECAR="${TACO_A9_SIDECAR:-$TACO_A9_DATA_ROOT/taco_a9_sidecar.jsonl}"
export TACO_A9_SIDECAR_INDEX="${TACO_A9_SIDECAR_INDEX:-$TACO_A9_DATA_ROOT/taco_a9_sidecar.index.json}"
export TACO_A9_TIMEOUT_SECONDS="${TACO_A9_TIMEOUT_SECONDS:-2}"
export TACO_A9_MAX_CODE_PROMPT_CHARS="${TACO_A9_MAX_CODE_PROMPT_CHARS:-12000}"
export TACO_A9_TERMINAL_WARMUP_STEPS="${TACO_A9_TERMINAL_WARMUP_STEPS:-50}"
REWARD_MODE="${TACO_A9_REWARD_MODE:-uniform_certificate}"
case "$REWARD_MODE" in
    uniform_certificate) AGENT_FLOW=taco_a9_code_agent ;;
    llm_judge) AGENT_FLOW=taco_llm_judge_code_agent ;;
    *) echo "TACO_A9_REWARD_MODE must be uniform_certificate or llm_judge" >&2; exit 2 ;;
esac

TRAIN_PATH="$TACO_A9_DATA_ROOT/train.parquet"
VAL_PATH="$TACO_A9_DATA_ROOT/validation.parquet"
for required in "$TRAIN_PATH" "$VAL_PATH" "$TACO_A9_SIDECAR" "$TACO_A9_SIDECAR_INDEX"; do
    [[ -f "$required" ]] || { echo "Missing TACO A9 artifact: $required" >&2; exit 2; }
done
command -v bwrap >/dev/null || { echo "bubblewrap is required for TACO A9" >&2; exit 2; }

MODEL_PATH="${TACO_A9_MODEL_PATH:-${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}}"
source "$PROJECT_DIR/scripts/common/model_training.sh"
agent_r1_model_overrides "$MODEL_PATH"
agent_r1_optimizer_overrides "$MODEL_PATH"
agent_r1_paper_profile taco "$MODEL_PATH"
TRAIN_BATCH_SIZE="${TACO_A9_TRAIN_BATCH_SIZE:-20}"
ROLLOUT_N="${TACO_A9_ROLLOUT_N:-4}"
TOTAL_EPOCHS="${TACO_A9_TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TACO_A9_TOTAL_TRAINING_STEPS:-$PAPER_TRAIN_STEPS}"
SAVE_FREQ="${TACO_A9_SAVE_FREQ:-50}"
SEED="${TACO_A9_SEED:-42}"
TP_SIZE="${TACO_A9_TENSOR_PARALLEL_SIZE:-8}"
MAX_PROMPT_LENGTH="${TACO_A9_MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${TACO_A9_MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LENGTH="${TACO_A9_MAX_MODEL_LENGTH:-12288}"
MAX_BATCHED_TOKENS="${TACO_A9_MAX_BATCHED_TOKENS:-12288}"
MAX_NUM_SEQS="${TACO_A9_MAX_NUM_SEQS:-80}"
VLLM_MEMORY_UTILIZATION="${TACO_A9_VLLM_MEMORY_UTILIZATION:-0.90}"
VLLM_KV_CACHE_BYTES="${TACO_A9_VLLM_KV_CACHE_BYTES:-8589934592}"
ACTOR_PARAM_OFFLOAD="${TACO_A9_ACTOR_PARAM_OFFLOAD:-true}"
ACTOR_OPTIMIZER_OFFLOAD="${TACO_A9_ACTOR_OPTIMIZER_OFFLOAD:-true}"
ROLLOUT_GPUS="${TACO_A9_ROLLOUT_GPUS:-8}"
ROLLOUT_NNODES="${TACO_A9_ROLLOUT_NNODES:-0}"
CHECKPOINT_ENGINE_BACKEND="${TACO_A9_CHECKPOINT_ENGINE_BACKEND:-naive}"
CHECKPOINT_BUCKET_MB="${TACO_A9_CHECKPOINT_BUCKET_MB:-2048}"
TRAIN_ROWS="$($PYTHON_BIN - "$TRAIN_PATH" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)"
TRAIN_MAX_SAMPLES="${TACO_A9_TRAIN_MAX_SAMPLES:-$PAPER_TRAIN_SAMPLES}"
if (( TRAIN_MAX_SAMPLES > TRAIN_ROWS || TRAIN_MAX_SAMPLES % TRAIN_BATCH_SIZE != 0 )); then
    echo "TACO_A9_TRAIN_MAX_SAMPLES must not exceed $TRAIN_ROWS and must divide batch size $TRAIN_BATCH_SIZE" >&2
    exit 2
fi
if (( TOTAL_STEPS > TRAIN_MAX_SAMPLES / TRAIN_BATCH_SIZE * TOTAL_EPOCHS )); then
    echo "TACO_A9_TOTAL_TRAINING_STEPS exceeds the selected data/epoch budget" >&2
    exit 2
fi
IFS=',' read -r -a VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#VISIBLE_GPUS[@]}"
if [[ "$REWARD_MODE" == "uniform_certificate" ]]; then
    (( NUM_GPUS == 8 )) || { echo "TACO A9 main run requires eight visible GPUs" >&2; exit 2; }
    (( TP_SIZE == NUM_GPUS )) || { echo "TACO A9 tensor parallel size must equal visible GPU count" >&2; exit 2; }
else
    (( NUM_GPUS >= 2 )) || { echo "TACO Judge training requires at least two actor GPUs" >&2; exit 2; }
    (( TP_SIZE > 0 && NUM_GPUS % TP_SIZE == 0 )) || { echo "TACO tensor parallel size must divide actor GPU count" >&2; exit 2; }
fi
RUN_ID="${RUN_ID:-${AGENT_R1_MODEL_NAME}_taco_${REWARD_MODE}_5turn_n4_${TOTAL_STEPS}step_${NUM_GPUS}gpu_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${TACO_A9_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m recipes.taco_a9.prepare_run \
    --project-dir "$PROJECT_DIR" --output-dir "$OUTPUT_DIR" --model-path "$MODEL_PATH" \
    --train-path "$TRAIN_PATH" --validation-path "$VAL_PATH" --sidecar-path "$TACO_A9_SIDECAR" \
    --num-gpus "$NUM_GPUS" --train-max-samples "$TRAIN_MAX_SAMPLES" --train-batch-size "$TRAIN_BATCH_SIZE" --rollout-n "$ROLLOUT_N" \
    --total-training-steps "$TOTAL_STEPS" --total-epochs "$TOTAL_EPOCHS" --terminal-warmup-steps "$TACO_A9_TERMINAL_WARMUP_STEPS" --save-freq "$SAVE_FREQ" --seed "$SEED" \
    --reward-mode "$REWARD_MODE"

"$PYTHON_BIN" -m "$AGENT_R1_TRAINER_MODULE" \
    "${AGENT_R1_ALGORITHM_ARGS[@]}" \
    algorithm.gamma=1.0 algorithm.use_kl_in_reward=false \
    data.train_files="$TRAIN_PATH" data.val_files="$VAL_PATH" data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.shuffle=false data.seed="$SEED" data.train_max_samples="$TRAIN_MAX_SAMPLES" \
    data.val_batch_size=8 data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=False data.truncation=error data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.use_remove_padding=False \
    "${AGENT_R1_MODEL_OVERRIDES[@]}" \
    actor_rollout_ref.model.use_fused_kernels=true actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES" actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS" \
    actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_ENGINE_BACKEND" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$CHECKPOINT_BUCKET_MB" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_MEMORY_UTILIZATION" actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.enable_sleep_mode=false actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.free_cache_engine=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes="$VLLM_KV_CACHE_BYTES" \
    actor_rollout_ref.rollout.multi_turn.enable=true actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$PROJECT_DIR/recipes/taco_a9/base.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_flow="$AGENT_FLOW" actor_rollout_ref.rollout.agent.num_workers="$NUM_GPUS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 "${AGENT_R1_CRITIC_ARGS[@]}" reward_model.enable=false \
    +reward_model.launch_reward_fn_async=false \
    custom_reward_function.path="$PROJECT_DIR/recipes/taco_a9/reward_fn.py" custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$PROJECT_DIR/recipes/taco_a9/reward_fn.py" reward.custom_reward_function.name=compute_score \
    +trainer.use_legacy_worker_impl=disable trainer.logger='["console"]' trainer.project_name=TACO_A9_AGENT_R1 \
    trainer.experiment_name="$RUN_ID" trainer.n_gpus_per_node="$NUM_GPUS" trainer.nnodes=1 \
    trainer.val_before_train=false trainer.resume_mode=disable trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep=2 trainer.test_freq=-1 trainer.total_epochs="$TOTAL_EPOCHS" trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" trainer.rollout_data_file="$OUTPUT_DIR/rollouts.jsonl" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" trainer.log_val_generations=0 2>&1 | tee "$OUTPUT_DIR/train.log"
