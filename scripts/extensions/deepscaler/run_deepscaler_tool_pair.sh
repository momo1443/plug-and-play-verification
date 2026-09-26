#!/usr/bin/env bash
set -euo pipefail

# Shared frozen runtime for the paired DeepScaleR ToolEnv A1/A9 experiment.
# The sole intentional training difference is the trajectory reward contract.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

ARM="${DEEPSCALER_TOOL_ARM:-}"
case "$ARM" in
    A1)
        AGENT_NAME=deepscaler_tool_a1
        PROJECT_NAME=DeepScalerTool_A1_AGENT_R1
        ;;
    A9)
        AGENT_NAME=deepscaler_tool_a9
        PROJECT_NAME=DeepScalerTool_A9_AGENT_R1
        ;;
    JUDGE)
        AGENT_NAME=deepscaler_tool_llm_judge
        PROJECT_NAME=DeepScalerTool_LLM_JUDGE_AGENT_R1
        AGENT_CONFIG_PATH="$PROJECT_DIR/recipes/llm_judge/deepscaler.yaml"
        ;;
    *)
        echo "DEEPSCALER_TOOL_ARM must be A1, A9, or JUDGE" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export DEEPSCALER_TOOL_EM_WARMUP_STEPS="${DEEPSCALER_TOOL_EM_WARMUP_STEPS:-50}"

MODEL_PATH="${DEEPSCALER_TOOL_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
source "$PROJECT_DIR/scripts/common/model_training.sh"
agent_r1_model_overrides "$MODEL_PATH"
SOURCE_DATA_DIR="${DEEPSCALER_SOURCE_DATA_DIR:-$PROJECT_DIR/data/corpus/deepscaler}"
TOOL_DATA_DIR="${DEEPSCALER_TOOL_PAIR_DATA_DIR:-$PROJECT_DIR/data/corpus/deepscaler_tool_pair}"
TRAIN_PATH="$TOOL_DATA_DIR/train.parquet"
VAL_PATH="$TOOL_DATA_DIR/validation.parquet"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-$PROJECT_DIR/recipes/deepscaler/tool_reward_arms.yaml}"

TRAIN_BATCH_SIZE="${DEEPSCALER_TOOL_TRAIN_BATCH_SIZE:-20}"
TRAIN_MAX_SAMPLES="${DEEPSCALER_TOOL_TRAIN_MAX_SAMPLES:-30000}"
ROLLOUT_N="${DEEPSCALER_TOOL_ROLLOUT_N:-4}"
TOTAL_STEPS="${DEEPSCALER_TOOL_TOTAL_STEPS:-1500}"
SAVE_FREQ="${DEEPSCALER_TOOL_SAVE_FREQ:-50}"
SEED="${DEEPSCALER_TOOL_SEED:-42}"
MAX_STEPS="${DEEPSCALER_TOOL_MAX_STEPS:-5}"
MAX_PROMPT_LENGTH="${DEEPSCALER_TOOL_MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${DEEPSCALER_TOOL_MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LENGTH="${DEEPSCALER_TOOL_MAX_MODEL_LENGTH:-8192}"
MAX_NUM_SEQS="${DEEPSCALER_TOOL_MAX_NUM_SEQS:-20}"
VLLM_GPU_MEMORY_UTILIZATION="${DEEPSCALER_TOOL_VLLM_GPU_MEMORY_UTILIZATION:-0.25}"

if ! [[ "$TRAIN_MAX_SAMPLES" =~ ^[1-9][0-9]*$ && "$TOTAL_STEPS" =~ ^[1-9][0-9]*$ && "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Sample, step, and max-step settings must be positive integers" >&2
    exit 2
fi
if (( TRAIN_MAX_SAMPLES != TRAIN_BATCH_SIZE * TOTAL_STEPS )); then
    echo "Paired runs require TRAIN_MAX_SAMPLES == TRAIN_BATCH_SIZE * TOTAL_STEPS" >&2
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
        --output-dir "$TOOL_DATA_DIR" \
        --omit-agent-name
fi

"$PYTHON_BIN" - "$TRAIN_PATH" "$VAL_PATH" <<'PY'
import sys
import pyarrow.parquet as pq
for path in sys.argv[1:]:
    if "agent_name" in pq.ParquetFile(path).schema.names:
        raise SystemExit(f"Paired ToolEnv parquet must not set agent_name: {path}")
PY

RUN_ID="${RUN_ID:-${AGENT_R1_MODEL_NAME}_deepscaler_tool_${ARM,,}_n${ROLLOUT_N}_${TOTAL_STEPS}step_${NUM_GPUS}gpu_seed${SEED}_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${DEEPSCALER_TOOL_OUTPUT_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"
mkdir -p "$(dirname "$OUTPUT_DIR")"
cd "$PROJECT_DIR"

"$PYTHON_BIN" -m recipes.deepscaler.prepare_tool_pair_run \
    --project-dir "$PROJECT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$RUN_ID" \
    --arm "$ARM" \
    --model-path "$MODEL_PATH" \
    --train-path "$TRAIN_PATH" \
    --validation-path "$VAL_PATH" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --train-max-samples "$TRAIN_MAX_SAMPLES" \
    --total-training-steps "$TOTAL_STEPS" \
    --rollout-n "$ROLLOUT_N" \
    --seed "$SEED" \
    --max-steps "$MAX_STEPS" \
    --em-warmup-steps "$DEEPSCALER_TOOL_EM_WARMUP_STEPS" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-response-length "$MAX_RESPONSE_LENGTH" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --num-gpus "$NUM_GPUS"

echo "=== DeepScaleR ToolEnv $ARM ==="
echo "Model: $MODEL_PATH"
echo "Train: $TRAIN_PATH ($TRAIN_MAX_SAMPLES samples)"
echo "Agent turns: $MAX_STEPS; rollout n: $ROLLOUT_N"
echo "Runtime profile: deepscaler_toolenv; prompt=$MAX_PROMPT_LENGTH; response_per_turn=$MAX_RESPONSE_LENGTH"
echo "A9 warmup: $DEEPSCALER_TOOL_EM_WARMUP_STEPS"
echo "Output: $OUTPUT_DIR"

"$PYTHON_BIN" -m agent_r1.trainer.main_agent_grpo \
    algorithm.adv_estimator=grpo ++algorithm.grpo.credit_assignment=step_causal \
    algorithm.norm_adv_by_std_in_grpo=True algorithm.gamma=1.0 algorithm.use_kl_in_reward=false \
    data.train_files="$TRAIN_PATH" data.val_files="$VAL_PATH" data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.shuffle=false data.seed="$SEED" data.train_max_samples="$TRAIN_MAX_SAMPLES" data.val_batch_size=8 \
    data.val_max_samples=2015 data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=false data.truncation=error data.return_raw_chat=true \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.use_remove_padding=false \
    "${AGENT_R1_MODEL_OVERRIDES[@]}" \
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
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LENGTH" actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.multi_turn.enable=true actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$AGENT_CONFIG_PATH" \
    actor_rollout_ref.rollout.agent.default_agent_flow="$AGENT_NAME" \
    actor_rollout_ref.rollout.agent.max_steps="$MAX_STEPS" actor_rollout_ref.rollout.agent.num_workers="$NUM_GPUS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    critic.enable=false reward_model.enable=false +reward_model.launch_reward_fn_async=false \
    +trainer.use_legacy_worker_impl=disable trainer.logger='["console"]' trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$RUN_ID" trainer.n_gpus_per_node="$NUM_GPUS" trainer.nnodes=1 \
    trainer.val_before_train=false trainer.resume_mode=disable trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep=2 trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" trainer.rollout_data_file="$OUTPUT_DIR/rollouts.jsonl" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" trainer.log_val_generations=0 2>&1 | tee "$OUTPUT_DIR/train.log"
