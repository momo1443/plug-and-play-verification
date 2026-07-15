#!/usr/bin/env bash
set -euo pipefail
set -x

# Internal formal A0 runner. A0 and A2 share HotpotQAAgentFlow; this entrypoint
# changes only trainer mode to validation-only and performs no optimizer update.

if (( $# != 0 )); then
    echo "run_validation.sh does not accept Hydra overrides in the formal path" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export HOTPOTQA_EMBEDDING_PER_WORKER_GPU=0
export HOTPOTQA_EMBEDDING_DEVICE=cpu
export HOTPOTQA_STREAMING_RESULTS=1
export HOTPOTQA_STREAMING_FSYNC=1

MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$PROJECT_DIR/../models/Qwen3.5-4B}"
TRAIN_PATH="${HOTPOTQA_TRAIN_PATH:-$HOTPOTQA_DATA_ROOT/train.parquet}"
VAL_PATH="${HOTPOTQA_VAL_PATH:-$HOTPOTQA_DATA_ROOT/validation.parquet}"
AGENT_CONFIG="$PROJECT_DIR/recipes/hotpotqa/base.yaml"
REWARD_PATH="$PROJECT_DIR/recipes/hotpotqa/reward_fn.py"

VAL_MAX_SAMPLES="${HOTPOTQA_VAL_MAX_SAMPLES:--1}"
VAL_BATCH_SIZE="${HOTPOTQA_VAL_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE=8
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH_PER_STEP=1024
MAX_MODEL_LENGTH=12288
ENABLE_THINKING="${HOTPOTQA_ENABLE_THINKING:-false}"
NUM_GPUS=8
AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
VLLM_MAX_NUM_SEQS="${HOTPOTQA_VLLM_MAX_NUM_SEQS:-8}"
RUN_ID="${RUN_ID:?RUN_ID must be set by run_a0.sh}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:?VALIDATION_DATA_DIR must be set by run_a0.sh}"

cd "$PROJECT_DIR"

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
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
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
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_seqs="$VLLM_MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$AGENT_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_flow=hotpotqa_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
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
    trainer.log_val_generations=4
