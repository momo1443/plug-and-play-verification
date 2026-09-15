#!/usr/bin/env bash
set -euo pipefail

# DeepScaleR multi-turn ToolEnv GRPO.  This is the DeepScaleR equivalent of
# Agent-R1's GSM8K + Tool recipe: each trajectory may check a candidate answer
# up to five times, receives only correct/incorrect feedback, then earns the
# strict terminal exact-match reward from its final boxed answer.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${DEEPSCALER_TOOL_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
SOURCE_DATA_DIR="${DEEPSCALER_SOURCE_DATA_DIR:-$PROJECT_DIR/data/corpus/deepscaler}"
TOOL_DATA_DIR="${DEEPSCALER_TOOL_DATA_DIR:-$PROJECT_DIR/data/corpus/deepscaler_tool}"
TRAIN_PATH="$TOOL_DATA_DIR/train.parquet"
VAL_PATH="$TOOL_DATA_DIR/validation.parquet"
REWARD_FN_PATH="$PROJECT_DIR/recipes/deepscaler/reward_fn_tool_terminal.py"
AGENT_CONFIG_PATH="$PROJECT_DIR/recipes/deepscaler/base.yaml"

TRAIN_BATCH_SIZE="${DEEPSCALER_TOOL_TRAIN_BATCH_SIZE:-20}"
ROLLOUT_N="${DEEPSCALER_TOOL_ROLLOUT_N:-4}"
TOTAL_STEPS="${DEEPSCALER_TOOL_TOTAL_STEPS:-300}"
TRAIN_MAX_SAMPLES="${DEEPSCALER_TOOL_TRAIN_MAX_SAMPLES:-$((TRAIN_BATCH_SIZE * TOTAL_STEPS))}"
SAVE_FREQ="${DEEPSCALER_TOOL_SAVE_FREQ:-50}"
SEED="${DEEPSCALER_TOOL_SEED:-42}"
MAX_STEPS="${DEEPSCALER_TOOL_MAX_STEPS:-5}"
MAX_PROMPT_LENGTH="${DEEPSCALER_TOOL_MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${DEEPSCALER_TOOL_MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LENGTH="${DEEPSCALER_TOOL_MAX_MODEL_LENGTH:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${DEEPSCALER_TOOL_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"

if ! [[ "$TOTAL_STEPS" =~ ^[1-9][0-9]*$ && "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPSCALER_TOOL_TOTAL_STEPS and DEEPSCALER_TOOL_MAX_STEPS must be positive integers" >&2
    exit 2
fi
if (( TRAIN_MAX_SAMPLES < TRAIN_BATCH_SIZE || TRAIN_MAX_SAMPLES % TRAIN_BATCH_SIZE != 0 )); then
    echo "DEEPSCALER_TOOL_TRAIN_MAX_SAMPLES must be at least one whole training batch" >&2
    exit 2
fi

IFS=',' read -r -a visible_gpus <<<"$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#visible_gpus[@]}"
if (( NUM_GPUS == 0 )); then
    echo "CUDA_VISIBLE_DEVICES must name at least one GPU" >&2
    exit 2
fi

if [[ ! -f "$TRAIN_PATH" || ! -f "$VAL_PATH" ]]; then
    "$PYTHON_BIN" -m recipes.deepscaler.prepare_tool_data \
        --source-dir "$SOURCE_DATA_DIR" \
        --output-dir "$TOOL_DATA_DIR"
fi

RUN_ID="${RUN_ID:-qwen35-4b_deepscaler_tool_grpo_terminal_em_n${ROLLOUT_N}_${TOTAL_STEPS}step_${NUM_GPUS}gpu_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${DEEPSCALER_TOOL_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

echo "=== DeepScaleR ToolEnv GRPO ==="
echo "Model: $MODEL_PATH"
echo "Train: $TRAIN_PATH ($TRAIN_MAX_SAMPLES samples)"
echo "Val:   $VAL_PATH"
echo "Agent turns: $MAX_STEPS; rollout n: $ROLLOUT_N"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"

"$PYTHON_BIN" -m agent_r1.trainer.main_agent_grpo \
    algorithm.adv_estimator=grpo ++algorithm.grpo.credit_assignment=step_causal \
    algorithm.norm_adv_by_std_in_grpo=True algorithm.gamma=1.0 algorithm.use_kl_in_reward=false \
    data.train_files="$TRAIN_PATH" data.val_files="$VAL_PATH" data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.shuffle=false data.seed="$SEED" data.train_max_samples="$TRAIN_MAX_SAMPLES" data.val_batch_size=8 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=false data.truncation=error data.return_raw_chat=true \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.use_fused_kernels=true actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    +actor_rollout_ref.rollout.enable_sleep_mode=false actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LENGTH" actor_rollout_ref.rollout.max_num_seqs="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.multi_turn.enable=true actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$AGENT_CONFIG_PATH" \
    actor_rollout_ref.rollout.agent.default_agent_flow=deepscaler_tool \
    actor_rollout_ref.rollout.agent.max_steps="$MAX_STEPS" actor_rollout_ref.rollout.agent.num_workers="$NUM_GPUS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    critic.enable=false reward_model.enable=false +reward_model.launch_reward_fn_async=false \
    custom_reward_function.path="$REWARD_FN_PATH" custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$REWARD_FN_PATH" reward.custom_reward_function.name=compute_score \
    +trainer.use_legacy_worker_impl=disable trainer.logger='["console"]' trainer.project_name=DeepScalerTool_AGENT_R1 \
    trainer.experiment_name="$RUN_ID" trainer.n_gpus_per_node="$NUM_GPUS" trainer.nnodes=1 \
    trainer.val_before_train=false trainer.resume_mode=disable trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep=2 trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" trainer.rollout_data_file="$OUTPUT_DIR/rollouts.jsonl" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" trainer.log_val_generations=0 2>&1 | tee "$OUTPUT_DIR/train.log"
