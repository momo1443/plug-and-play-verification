#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: run_with_frozen_judge.sh COMMAND [ARG ...]" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$HERE/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

export AGENT_R1_LLM_JUDGE_ENABLED=1
export AGENT_R1_JUDGE_MODEL="${AGENT_R1_JUDGE_MODEL:-$WORKSPACE_DIR/models/Qwen3.5-9B}"
export AGENT_R1_JUDGE_PORT="${AGENT_R1_JUDGE_PORT:-29500}"
export AGENT_R1_JUDGE_MAX_MODEL_LEN="${AGENT_R1_JUDGE_MAX_MODEL_LEN:-8192}"
export AGENT_R1_JUDGE_GPU_MEMORY_UTILIZATION="${AGENT_R1_JUDGE_GPU_MEMORY_UTILIZATION:-0.80}"
export AGENT_R1_JUDGE_REQUEST_TIMEOUT_S="${AGENT_R1_JUDGE_REQUEST_TIMEOUT_S:-120}"
export AGENT_R1_JUDGE_GPU="${AGENT_R1_JUDGE_GPU:-7}"
export AGENT_R1_JUDGE_EXTERNAL=1

if [[ -n "${AGENT_R1_JUDGE_API_KEY:-}" ]]; then
    exec "$@"
fi

[[ -f "$AGENT_R1_JUDGE_MODEL/config.json" ]] || {
    echo "Missing Judge model config: $AGENT_R1_JUDGE_MODEL/config.json" >&2
    exit 2
}

JUDGE_LOG_DIR="${AGENT_R1_JUDGE_LOG_DIR:-$WORKSPACE_DIR/logs/llm_judge_server}"
mkdir -p "$JUDGE_LOG_DIR"
JUDGE_LOG="$JUDGE_LOG_DIR/qwen35-9b-judge-$(date +%Y%m%d-%H%M%S).log"
JUDGE_PID=""
TRAIN_PID=""

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
    [[ -z "$TRAIN_PID" ]] || terminate_process "$TRAIN_PID"
    [[ -z "$JUDGE_PID" ]] || terminate_process "$JUDGE_PID"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if "$PYTHON_BIN" -c \
    'import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' \
    "$AGENT_R1_JUDGE_PORT"; then
    echo "Judge port $AGENT_R1_JUDGE_PORT is already in use" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="$AGENT_R1_JUDGE_GPU" "$PYTHON_BIN" \
    -m vllm.entrypoints.openai.api_server \
    --model "$AGENT_R1_JUDGE_MODEL" \
    --port "$AGENT_R1_JUDGE_PORT" \
    --gpu-memory-utilization "$AGENT_R1_JUDGE_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$AGENT_R1_JUDGE_MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --no-enable-log-requests \
    >"$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!

healthy=false
for _ in {1..180}; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        wait "$JUDGE_PID" || true
        echo "Judge exited before health check passed; see $JUDGE_LOG" >&2
        exit 1
    fi
    if "$PYTHON_BIN" -c \
        'import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
        "http://127.0.0.1:$AGENT_R1_JUDGE_PORT/v1/models" >/dev/null 2>&1; then
        healthy=true
        break
    fi
    sleep 1
done
[[ "$healthy" == true ]] || { echo "Judge health check timed out; see $JUDGE_LOG" >&2; exit 1; }

"$@" &
TRAIN_PID=$!
wait "$TRAIN_PID"
