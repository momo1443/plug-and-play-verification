#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

ARM="${VISION_R1_ARM:-A9}"
case "$ARM" in
    A1) AGENT_FLOW=vision_r1_a1_visual_agent ;;
    A9) AGENT_FLOW=vision_r1_a9_visual_agent ;;
    JUDGE) AGENT_FLOW=vision_r1_llm_judge_visual_agent ;;
    *) echo "VISION_R1_ARM must be A1, A9, or JUDGE" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

RAW_DATA_ROOT="${VISION_R1_RAW_DATA_ROOT:-$WORKSPACE_DIR/data/vision_r1_rl}"
PREPARED_ROOT="${VISION_R1_PREPARED_ROOT:-$PROJECT_DIR/data/vision_r1_rl}"
TRAIN_PATH="$PREPARED_ROOT/train.parquet"
VAL_PATH="$PREPARED_ROOT/test.parquet"
for split in train test; do
    raw_path="$RAW_DATA_ROOT/$split.parquet"
    prepared_path="$PREPARED_ROOT/$split.parquet"
    [[ -f "$raw_path" ]] || { echo "Missing official Vision-R1-rl file: $raw_path" >&2; exit 2; }
    if [[ ! -f "$prepared_path" ]]; then
        "$PYTHON_BIN" -m recipes.vision_r1.prepare_data \
            --input "$raw_path" --output "$prepared_path" --split "$split"
    fi
done

MODEL_PATH="${VISION_R1_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
source "$PROJECT_DIR/examples/common/model_training.sh"
agent_r1_model_overrides "$MODEL_PATH"
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config: $MODEL_PATH/config.json" >&2; exit 2; }
TRAIN_BATCH_SIZE="${VISION_R1_TRAIN_BATCH_SIZE:-8}"
ROLLOUT_N="${VISION_R1_ROLLOUT_N:-4}"
TOTAL_STEPS="${VISION_R1_TOTAL_TRAINING_STEPS:-300}"
TOTAL_EPOCHS="${VISION_R1_TOTAL_EPOCHS:-1}"
TRAIN_MAX_SAMPLES="${VISION_R1_TRAIN_MAX_SAMPLES:-$((TOTAL_STEPS * TRAIN_BATCH_SIZE))}"
WARMUP_STEPS="${VISION_R1_TERMINAL_WARMUP_STEPS:-50}"
export VISION_R1_TERMINAL_WARMUP_STEPS="$WARMUP_STEPS"
SAVE_FREQ="${VISION_R1_SAVE_FREQ:-50}"
SEED="${VISION_R1_SEED:-42}"
TP_SIZE="${VISION_R1_TENSOR_PARALLEL_SIZE:-8}"
MAX_PROMPT_LENGTH="${VISION_R1_MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${VISION_R1_MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LENGTH="${VISION_R1_MAX_MODEL_LENGTH:-12288}"
MAX_NUM_SEQS="${VISION_R1_MAX_NUM_SEQS:-32}"
VLLM_MEMORY_UTILIZATION="${VISION_R1_VLLM_MEMORY_UTILIZATION:-0.85}"

IFS=',' read -r -a VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#VISIBLE_GPUS[@]}"
if [[ "$ARM" == "JUDGE" ]]; then
    (( NUM_GPUS >= 2 )) || { echo "Vision-R1 Judge training requires at least two actor GPUs" >&2; exit 2; }
    (( TP_SIZE > 0 && NUM_GPUS % TP_SIZE == 0 )) || { echo "Tensor parallel size must divide actor GPU count" >&2; exit 2; }
else
    (( NUM_GPUS == 8 )) || { echo "Vision-R1 launcher expects eight visible GPUs" >&2; exit 2; }
    (( TP_SIZE == NUM_GPUS )) || { echo "Tensor parallel size must equal visible GPU count" >&2; exit 2; }
fi

RUN_ID="${RUN_ID:-${AGENT_R1_MODEL_NAME}_vision-r1_${ARM,,}_visual-certificate_n${ROLLOUT_N}_${TOTAL_STEPS}step_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${VISION_R1_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m agent_r1.trainer.main_agent_grpo \
    algorithm.adv_estimator=grpo ++algorithm.grpo.credit_assignment=step_causal \
    algorithm.norm_adv_by_std_in_grpo=True algorithm.gamma=1.0 algorithm.use_kl_in_reward=false \
    data.train_files="$TRAIN_PATH" data.val_files="$VAL_PATH" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" data.train_max_samples="$TRAIN_MAX_SAMPLES" \
    data.val_batch_size=8 data.val_max_samples=500 data.shuffle=false data.seed="$SEED" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=false data.truncation=error data.return_raw_chat=true \
    data.image_key=images data.prompt_key=prompt +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.trust_remote_code=true \
    "${AGENT_R1_MODEL_OVERRIDES[@]}" \
    actor_rollout_ref.model.use_remove_padding=false actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=3 \
    actor_rollout_ref.rollout.multi_turn.enable=true actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$PROJECT_DIR/recipes/vision_r1/base.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_flow="$AGENT_FLOW" \
    actor_rollout_ref.rollout.agent.num_workers="$NUM_GPUS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 critic.enable=false reward_model.enable=false \
    +reward_model.launch_reward_fn_async=false \
    custom_reward_function.path="$PROJECT_DIR/recipes/vision_r1/reward_fn.py" \
    custom_reward_function.name=compute_score reward.custom_reward_function.path="$PROJECT_DIR/recipes/vision_r1/reward_fn.py" \
    reward.custom_reward_function.name=compute_score +trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console"]' trainer.project_name=VISION_R1_AGENT_R1 trainer.experiment_name="$RUN_ID" \
    trainer.n_gpus_per_node="$NUM_GPUS" trainer.nnodes=1 trainer.val_before_train=false \
    trainer.resume_mode=disable trainer.save_freq="$SAVE_FREQ" trainer.test_freq=50 \
    trainer.total_epochs="$TOTAL_EPOCHS" trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" trainer.rollout_data_file="$OUTPUT_DIR/rollouts.jsonl" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" trainer.log_val_generations=0 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
