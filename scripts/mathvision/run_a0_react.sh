#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="${WORKSPACE_DIR:-$(cd "$AGENT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA="$PROJECT_DIR/data/MathVision/data/test-00000-of-00001-3532b8d3f1b4047a.parquet"
PREPARED="$AGENT_DIR/data/mathvision/test.parquet"

if [[ ! -f "$PREPARED" ]]; then
  cd "$AGENT_DIR"
  "$PYTHON_BIN" recipes/mathvision/prepare_data.py --input "$DATA" --output "$PREPARED"
fi

run_one() {
  local name="$1" model="$2" devices="$3" out="$4"
  mkdir -p "$out"
  cd "$AGENT_DIR"
  local ray_tmp
  if [[ "$name" == *"4b"* ]]; then ray_tmp=/tmp/rmv4; else ray_tmp=/tmp/rmv9; fi
  CUDA_VISIBLE_DEVICES="$devices" RAY_TMPDIR="$ray_tmp" \
  "$PYTHON_BIN" -m agent_r1.trainer.main_agent_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files="$PREPARED" data.val_files="$PREPARED" \
    data.train_batch_size=8 data.val_batch_size=8 data.val_max_samples=-1 \
    data.max_prompt_length=8192 data.max_response_length=32768 \
    data.filter_overlong_prompts=false data.truncation=error data.return_raw_chat=true \
    data.image_key=images data.prompt_key=prompt \
    actor_rollout_ref.model.path="$model" actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=65536 actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.multi_turn.enable=true actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$AGENT_DIR/recipes/mathvision/base.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_flow=mathvision_react \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    critic.enable=false reward_model.enable=false \
    custom_reward_function.path="$AGENT_DIR/recipes/mathvision/reward_fn.py" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$AGENT_DIR/recipes/mathvision/reward_fn.py" \
    reward.custom_reward_function.name=compute_score \
    +trainer.use_legacy_worker_impl=disable \
    trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.val_before_train=true trainer.val_only=true \
    trainer.resume_mode=disable trainer.logger='["console"]' \
    trainer.project_name=MathVision_A0_ReAct trainer.experiment_name="$name" \
    trainer.validation_data_dir="$out" \
    2>&1 | tee "$out/launcher.log"
}

ALL_GPUS=0,1,2,3,4,5,6,7
run_one qwen35_4b_mathvision_a0_react "$PROJECT_DIR/models/Qwen3.5-4B" "$ALL_GPUS" "$PROJECT_DIR/logs/qwen35-4b_mathvision_a0_react"
run_one qwen35_9b_mathvision_a0_react "$PROJECT_DIR/models/Qwen3.5-9B" "$ALL_GPUS" "$PROJECT_DIR/logs/qwen35-9b_mathvision_a0_react"
