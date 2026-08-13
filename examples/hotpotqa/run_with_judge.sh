#!/usr/bin/env bash
set -euo pipefail

# Shared lifecycle wrapper for Judge-backed HotpotQA arm (A6 only).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${HOTPOTQA_REWARD_ARM:?HOTPOTQA_REWARD_ARM is required}"
case "$ARM" in
    A6) ;;
    *) echo "run_with_judge.sh only supports A6, got: $ARM" >&2; exit 2 ;;
esac

: "${HOTPOTQA_JUDGE_GPU:?HOTPOTQA_JUDGE_GPU is required}"
: "${HOTPOTQA_JUDGE_MODEL:?HOTPOTQA_JUDGE_MODEL is required}"
: "${HOTPOTQA_JUDGE_PORT:?HOTPOTQA_JUDGE_PORT is required}"
: "${HOTPOTQA_JUDGE_MAX_MODEL_LEN:?HOTPOTQA_JUDGE_MAX_MODEL_LEN is required}"
: "${HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION:?HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION is required}"

export HOTPOTQA_JUDGE_EXTERNAL=1
export PYTHON_BIN="${HOTPOTQA_PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

# Config rendering and CPU preflight do not need a live Judge process.
if [[ "${HOTPOTQA_HYDRA_CONFIG_ONLY:-0}" == "1" || "${HOTPOTQA_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exec bash "$HERE/run_rlvr.sh"
fi

: "${HOTPOTQA_OUTPUT_DIR:?HOTPOTQA_OUTPUT_DIR is required}"
JUDGE_LOG="$HOTPOTQA_OUTPUT_DIR/judge_server.log"
JUDGE_PID_FILE="$HOTPOTQA_OUTPUT_DIR/judge_server.pid"
TRAIN_PID=""
JUDGE_PID=""

mkdir -p "$HOTPOTQA_OUTPUT_DIR"

terminate_process() {
    local pid="$1"
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return
        fi
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$TRAIN_PID" ]]; then
        terminate_process "$TRAIN_PID"
    fi
    if [[ -n "$JUDGE_PID" ]]; then
        terminate_process "$JUDGE_PID"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if "$PYTHON_BIN" -c \
    'import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' \
    "$HOTPOTQA_JUDGE_PORT"; then
    echo "Judge port $HOTPOTQA_JUDGE_PORT is already in use." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="$HOTPOTQA_JUDGE_GPU" "$PYTHON_BIN" \
    -m vllm.entrypoints.openai.api_server \
    --model "$HOTPOTQA_JUDGE_MODEL" \
    --port "$HOTPOTQA_JUDGE_PORT" \
    --gpu-memory-utilization "$HOTPOTQA_JUDGE_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$HOTPOTQA_JUDGE_MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --no-enable-log-requests \
    >"$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!
printf '%s\n' "$JUDGE_PID" >"$JUDGE_PID_FILE"

judge_healthy=false
for _ in {1..120}; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        wait "$JUDGE_PID" || true
        echo "Judge server exited before becoming healthy; see $JUDGE_LOG." >&2
        exit 1
    fi
    if "$PYTHON_BIN" -c \
        'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
        "http://127.0.0.1:$HOTPOTQA_JUDGE_PORT/v1/models" \
        >/dev/null 2>&1; then
        judge_healthy=true
        break
    fi
    sleep 1
done
if [[ "$judge_healthy" != true ]]; then
    echo "Judge server did not become healthy within 120 seconds; see $JUDGE_LOG." >&2
    exit 1
fi

bash "$HERE/run_rlvr.sh" &
TRAIN_PID=$!
wait "$TRAIN_PID"
