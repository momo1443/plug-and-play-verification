#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
    echo "Formal A1/A2 does not accept trailing Hydra overrides; use documented environment variables" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

ARM="${HOTPOTQA_REWARD_ARM:?HOTPOTQA_REWARD_ARM must be A1 or A2}"
case "$ARM" in
    A1|A2) ;;
    *) echo "HOTPOTQA_REWARD_ARM must be A1 or A2, got: $ARM" >&2; exit 2 ;;
esac

RUN_MODE="${HOTPOTQA_RUN_MODE:-smoke}"
case "$RUN_MODE" in
    smoke)
        TRAIN_MAX_SAMPLES="${HOTPOTQA_TRAIN_MAX_SAMPLES:-4}"
        TRAIN_BATCH_SIZE="${HOTPOTQA_TRAIN_BATCH_SIZE:-4}"
        ROLLOUT_N="${HOTPOTQA_ROLLOUT_N:-2}"
        TOTAL_TRAINING_STEPS="${HOTPOTQA_TOTAL_TRAINING_STEPS:-1}"
        CHECKPOINT_SAVE_CONTENTS='["model","extra"]'
        ;;
    main)
        TRAIN_MAX_SAMPLES=90447
        TRAIN_BATCH_SIZE="${HOTPOTQA_TRAIN_BATCH_SIZE:-8}"
        ROLLOUT_N="${HOTPOTQA_ROLLOUT_N:-4}"
        TOTAL_TRAINING_STEPS="${HOTPOTQA_TOTAL_TRAINING_STEPS:?main requires an explicit optimizer-step budget}"
        CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra"]'
        ;;
    *) echo "HOTPOTQA_RUN_MODE must be smoke or main, got: $RUN_MODE" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export HOTPOTQA_FORMAL_A0=0
export HOTPOTQA_FORMAL_EXPERIMENT=1
export HOTPOTQA_REWARD_ARM="$ARM"
export HOTPOTQA_ENABLE_THINKING=false
export HOTPOTQA_FORCE_FIRST_SEARCH=false
export HOTPOTQA_FORCE_FINAL_ANSWER=true
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
NUM_GPUS="${HOTPOTQA_NUM_GPUS:-4}"
AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-$NUM_GPUS}"
GRPO_GAMMA=1.0
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LENGTH=12288
MAX_NUM_SEQS="$((TRAIN_BATCH_SIZE * ROLLOUT_N))"
VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.40}"

RUN_ID="${RUN_ID:-qwen35-4b_${ARM,,}_${RUN_MODE}_grpo_$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${HOTPOTQA_OUTPUT_DIR:-$WORKSPACE_DIR/results/$RUN_ID}"
# Ray appends a long session/sockets suffix. Keep this prefix short enough for
# Linux's 107-byte AF_UNIX pathname limit even when RUN_ID is descriptive.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ar1-${ARM,,}-${RUN_MODE}-$$}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

if [[ "${HOTPOTQA_HYDRA_CONFIG_ONLY:-0}" != "1" ]]; then
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
        --num_gpus "$NUM_GPUS" \
        --agent_workers "$AGENT_WORKERS" \
        --gamma "$GRPO_GAMMA"
fi

if [[ "${HOTPOTQA_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

HYDRA_CONFIG_ARGS=()
if [[ "${HOTPOTQA_HYDRA_CONFIG_ONLY:-0}" == "1" ]]; then
    HYDRA_CONFIG_ARGS=(--cfg job)
fi

"$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo \
    "${HYDRA_CONFIG_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    ++algorithm.grpo.credit_assignment=step_causal \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.gamma="$GRPO_GAMMA" \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
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
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
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
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$PROJECT_DIR/recipes/hotpotqa/base.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_flow=hotpotqa_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    critic.enable=False \
    reward_model.enable=False \
    custom_reward_function.path="$PROJECT_DIR/recipes/hotpotqa/reward_fn.py" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$PROJECT_DIR/recipes/hotpotqa/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    +trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console"]' \
    trainer.project_name=HotpotQA_AGENT_R1 \
    trainer.experiment_name="$RUN_ID" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.resume_mode=disable \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" \
    trainer.log_val_generations=0 2>&1 | tee "$OUTPUT_DIR/train.log"
