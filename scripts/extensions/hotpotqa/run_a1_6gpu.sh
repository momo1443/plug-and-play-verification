#!/usr/bin/env bash
set -euo pipefail

# A1 (terminal-only RLVR): GRPO with step_causal credit assignment.
# Reward = terminal exact match only (process_weight=0, terminal_weight=1).
# No FrozenProver, no Verifier — pure EM GRPO baseline.
#
# 6-GPU configuration aligned with A9 runtime parameters:
#   - vllm_gpu_memory_utilization=0.25
#   - max_model_len=8192
#   - max_num_seqs=20
#   - grpo_micro_batch_size=1
#   - rollout_n=4
#   - train_batch_size=20
#   - total_training_steps=1500
#   - save_freq=50
#   - ref_kl=0.001 (low_var_kl)
#
# Self-contained entrypoint: `bash run_a1_6gpu.sh` launches A1 on
# GPUs 0,1,3,4,5,6 with the shared first-30k-of-90k contract.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../../.." && pwd)"

export HOTPOTQA_REWARD_ARM=A1
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4,5,6}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"

# 6-GPU resource configuration (aligned with A9)
export HOTPOTQA_NUM_GPUS=6
export HOTPOTQA_AGENT_WORKERS=6
export HOTPOTQA_GRPO_MICRO_BATCH_SIZE=1
export HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION=0.25
export HOTPOTQA_VLLM_MAX_MODEL_LEN=8192
export HOTPOTQA_VLLM_MAX_NUM_SEQS=20
export HOTPOTQA_VLLM_ENABLE_SLEEP_MODE=false
export HOTPOTQA_VLLM_FREE_CACHE_ENGINE=false
export HOTPOTQA_SAVE_FREQ=50

export RUN_ID="${RUN_ID:-qwen35-4b_a1_${HOTPOTQA_RUN_MODE}_grpo_stepcausal_main30k_n4_1500step_6gpu_vllm025_mlen8192_mseq20_mb1_refkl001_save50_$(date +%Y%m%d-%H%M%S)}"

exec bash "$HERE/../../hotpotqa/run_rlvr.sh"
