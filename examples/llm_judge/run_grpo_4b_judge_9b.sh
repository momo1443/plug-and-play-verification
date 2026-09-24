#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: run_grpo_4b_judge_9b.sh {deepscaler|hotpotqa|taco|vision|all}" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
DATASET="$1"

if [[ "$DATASET" == "all" ]]; then
    for name in deepscaler hotpotqa taco vision; do
        "$0" "$name"
    done
    exit 0
fi

export WORKSPACE_DIR
export AGENT_R1_LLM_JUDGE_ENABLED=1
export AGENT_R1_JUDGE_MODEL="${AGENT_R1_JUDGE_MODEL:-$WORKSPACE_DIR/models/Qwen3.5-9B}"
export AGENT_R1_JUDGE_GPU="${AGENT_R1_JUDGE_GPU:-7}"
export CUDA_VISIBLE_DEVICES="${AGENT_R1_ACTOR_GPUS:-0,1,2,3,4,5,6}"
IFS=',' read -r -a ACTOR_GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
ACTOR_GPU_COUNT="${#ACTOR_GPU_IDS[@]}"
for gpu_id in "${ACTOR_GPU_IDS[@]}"; do
    if [[ "$gpu_id" == "$AGENT_R1_JUDGE_GPU" ]]; then
        echo "Actor GPUs and Judge GPU must be disjoint; GPU $gpu_id appears in both" >&2
        exit 2
    fi
done
export HOTPOTQA_JUDGE_MODEL="$AGENT_R1_JUDGE_MODEL"
export HOTPOTQA_JUDGE_GPU="$AGENT_R1_JUDGE_GPU"
export HOTPOTQA_JUDGE_PORT="${AGENT_R1_JUDGE_PORT:-29500}"
export HOTPOTQA_JUDGE_MAX_MODEL_LEN="${AGENT_R1_JUDGE_MAX_MODEL_LEN:-8192}"
export HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION="${AGENT_R1_JUDGE_GPU_MEMORY_UTILIZATION:-0.80}"
export HOTPOTQA_JUDGE_EXTERNAL=1

case "$DATASET" in
    deepscaler)
        export DEEPSCALER_TOOL_ARM=JUDGE
        export DEEPSCALER_TOOL_MODEL_PATH="${DEEPSCALER_TOOL_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
        export DEEPSCALER_TOOL_TRAIN_BATCH_SIZE="${DEEPSCALER_TOOL_TRAIN_BATCH_SIZE:-20}"
        export DEEPSCALER_TOOL_TRAIN_MAX_SAMPLES="${DEEPSCALER_TOOL_TRAIN_MAX_SAMPLES:-10000}"
        export DEEPSCALER_TOOL_TOTAL_STEPS="${DEEPSCALER_TOOL_TOTAL_STEPS:-500}"
        command=(bash "$PROJECT_DIR/examples/deepmath/run_deepscaler_tool_pair.sh")
        ;;
    hotpotqa)
        export HOTPOTQA_REWARD_ARM=A6
        export HOTPOTQA_RUN_MODE=main
        export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
        export HOTPOTQA_TRAIN_BATCH_SIZE="${HOTPOTQA_TRAIN_BATCH_SIZE:-20}"
        export HOTPOTQA_TRAIN_MAX_SAMPLES="${HOTPOTQA_TRAIN_MAX_SAMPLES:-10000}"
        export HOTPOTQA_TOTAL_TRAINING_STEPS="${HOTPOTQA_TOTAL_TRAINING_STEPS:-500}"
        export HOTPOTQA_NUM_GPUS="${HOTPOTQA_NUM_GPUS:-$ACTOR_GPU_COUNT}"
        export HOTPOTQA_AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-$ACTOR_GPU_COUNT}"
        command=(bash "$PROJECT_DIR/examples/hotpotqa/run_rlvr.sh")
        ;;
    taco)
        export TACO_A9_REWARD_MODE=llm_judge
        export TACO_A9_MODEL_PATH="${TACO_A9_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
        export TACO_A9_TRAIN_BATCH_SIZE="${TACO_A9_TRAIN_BATCH_SIZE:-20}"
        export TACO_A9_TRAIN_MAX_SAMPLES="${TACO_A9_TRAIN_MAX_SAMPLES:-10000}"
        export TACO_A9_TOTAL_TRAINING_STEPS="${TACO_A9_TOTAL_TRAINING_STEPS:-500}"
        export TACO_A9_TENSOR_PARALLEL_SIZE="${TACO_A9_TENSOR_PARALLEL_SIZE:-1}"
        export TACO_A9_ROLLOUT_GPUS="${TACO_A9_ROLLOUT_GPUS:-$ACTOR_GPU_COUNT}"
        export TACO_A9_VLLM_GPU_MEMORY_UTILIZATION="${TACO_A9_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
        export TACO_A9_VLLM_KV_CACHE_BYTES="${TACO_A9_VLLM_KV_CACHE_BYTES:-4294967296}"
        command=(bash "$PROJECT_DIR/examples/taco/run_a9_uniform.sh")
        ;;
    vision)
        export VISION_R1_ARM=JUDGE
        export VISION_R1_MODEL_PATH="${VISION_R1_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3.5-4B}"
        export VISION_R1_TRAIN_BATCH_SIZE="${VISION_R1_TRAIN_BATCH_SIZE:-8}"
        export VISION_R1_TRAIN_MAX_SAMPLES="${VISION_R1_TRAIN_MAX_SAMPLES:-10000}"
        export VISION_R1_TOTAL_TRAINING_STEPS="${VISION_R1_TOTAL_TRAINING_STEPS:-1250}"
        export VISION_R1_TENSOR_PARALLEL_SIZE="${VISION_R1_TENSOR_PARALLEL_SIZE:-1}"
        export VISION_R1_VLLM_GPU_MEMORY_UTILIZATION="${VISION_R1_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
        command=(bash "$PROJECT_DIR/examples/vision_r1/run_visual_agent.sh")
        ;;
    *)
        echo "Unknown dataset: $DATASET" >&2
        exit 2
        ;;
esac

if [[ "${AGENT_R1_JUDGE_CONFIG_ONLY:-0}" == "1" ]]; then
    printf 'dataset=%s\nactor_model=%s\njudge_model=%s\nactor_gpus=%s\njudge_gpu=%s\ncommand=' \
        "$DATASET" "$WORKSPACE_DIR/models/Qwen3.5-4B" "$AGENT_R1_JUDGE_MODEL" \
        "$CUDA_VISIBLE_DEVICES" "$AGENT_R1_JUDGE_GPU"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

exec bash "$HERE/run_with_frozen_judge.sh" "${command[@]}"
