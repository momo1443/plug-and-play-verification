#!/usr/bin/env bash
# AIME 2025 A0 baseline evaluation on GPU 2 and GPU 3
# Runs Qwen3.5-4B on GPU 2 and Qwen3.5-9B on GPU 3 in parallel
# v3: score one final answer independently of the reference
set -euo pipefail

AGENT_R1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="${WORKSPACE_DIR:-$(cd "$AGENT_R1_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== AIME 2025 A0 Baseline Evaluation (v3 final answer only) ==="
echo "GPU 2: Qwen3.5-4B"
echo "GPU 3: Qwen3.5-9B"
echo "================================================================"

# --- Ensure AIME 2025 data is prepared ---
AIME_DATA_DIR="${AIME2025_DATA_DIR:-$AGENT_R1_DIR/data/corpus/aime2025}"
export AIME2025_DATA_DIR="$AIME_DATA_DIR"
if [[ ! -f "$AIME_DATA_DIR/test.parquet" ]]; then
    echo "Missing prepared AIME data: $AIME_DATA_DIR/test.parquet. Set AIME2025_DATA_DIR to the prepared dataset directory." >&2
    exit 2
fi

mkdir -p "$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v3" "$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v3"
cd "$AGENT_R1_DIR/scripts/deepscaler"

# --- Launch Qwen3.5-4B on GPU 2 ---
CUDA_VISIBLE_DEVICES=2 \
MODEL_PATH="$PROJECT_DIR/models/Qwen3.5-4B" \
OUTPUT_DIR="$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v3" \
MAX_SAMPLES=-1 \
"$PYTHON_BIN" eval_aime2025.py 2>&1 | tee "$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v3/run.log" &

PID_4B=$!
echo "Qwen3.5-4B launched on GPU 2 (PID: $PID_4B)"

# --- Launch Qwen3.5-9B on GPU 3 ---
CUDA_VISIBLE_DEVICES=3 \
MODEL_PATH="$PROJECT_DIR/models/Qwen3.5-9B" \
OUTPUT_DIR="$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v3" \
MAX_SAMPLES=-1 \
"$PYTHON_BIN" eval_aime2025.py 2>&1 | tee "$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v3/run.log" &

PID_9B=$!
echo "Qwen3.5-9B launched on GPU 3 (PID: $PID_9B)"

echo ""
echo "Waiting for both evaluations to complete..."
echo "  PID $PID_4B: Qwen3.5-4B on GPU 2"
echo "  PID $PID_9B: Qwen3.5-9B on GPU 3"

# Wait for both
wait $PID_4B
EXIT_4B=$?
echo "Qwen3.5-4B finished (exit code: $EXIT_4B)"

wait $PID_9B
EXIT_9B=$?
echo "Qwen3.5-9B finished (exit code: $EXIT_9B)"

echo ""
echo "=== Final Results ==="
echo ""
echo "--- Qwen3.5-4B ---"
if [[ -f "$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v3/summary.json" ]]; then
    cat "$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v3/summary.json"
else
    echo "Summary not found (evaluation may have failed)"
fi

echo ""
echo "--- Qwen3.5-9B ---"
if [[ -f "$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v3/summary.json" ]]; then
    cat "$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v3/summary.json"
else
    echo "Summary not found (evaluation may have failed)"
fi
