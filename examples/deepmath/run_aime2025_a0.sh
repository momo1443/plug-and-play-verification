#!/usr/bin/env bash
# AIME 2025 A0 baseline evaluation on GPU 2 and GPU 3
# Runs Qwen3.5-4B on GPU 2 and Qwen3.5-9B on GPU 3 in parallel
# v2: robust answer extraction with multi-strategy matching
set -euo pipefail

PROJECT_DIR="/nas/deepresearch/zsb/corhort/project/agenticrl"
AGENT_R1_DIR="$PROJECT_DIR/Agent-R1"
PYTHON_BIN="/nas/deepresearch/conda/envs/agenticrl/bin/python"

echo "=== AIME 2025 A0 Baseline Evaluation (v2 robust extraction) ==="
echo "GPU 2: Qwen3.5-4B"
echo "GPU 3: Qwen3.5-9B"
echo "================================================================"

# --- Ensure AIME 2025 data is prepared ---
AIME_DATA_DIR="$AGENT_R1_DIR/data/corpus/aime2025"
if [[ ! -f "$AIME_DATA_DIR/test.parquet" ]]; then
    echo "Preparing AIME 2025 data..."
    cd "$AGENT_R1_DIR"
    "$PYTHON_BIN" recipes/deepmath/data_preprocess/process_aime2025.py \
        --output_dir "$AIME_DATA_DIR"
fi

cd "$AGENT_R1_DIR/examples/deepmath"

# --- Launch Qwen3.5-4B on GPU 2 ---
CUDA_VISIBLE_DEVICES=2 \
MODEL_PATH="$PROJECT_DIR/models/Qwen3.5-4B" \
OUTPUT_DIR="$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v2" \
MAX_SAMPLES=-1 \
"$PYTHON_BIN" eval_aime2025.py 2>&1 | tee "$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v2/run.log" &

PID_4B=$!
echo "Qwen3.5-4B launched on GPU 2 (PID: $PID_4B)"

# --- Launch Qwen3.5-9B on GPU 3 ---
CUDA_VISIBLE_DEVICES=3 \
MODEL_PATH="$PROJECT_DIR/models/Qwen3.5-9B" \
OUTPUT_DIR="$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v2" \
MAX_SAMPLES=-1 \
"$PYTHON_BIN" eval_aime2025.py 2>&1 | tee "$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v2/run.log" &

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
if [[ -f "$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v2/summary.json" ]]; then
    cat "$PROJECT_DIR/logs/qwen35-4b_a0_aime2025_v2/summary.json"
else
    echo "Summary not found (evaluation may have failed)"
fi

echo ""
echo "--- Qwen3.5-9B ---"
if [[ -f "$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v2/summary.json" ]]; then
    cat "$PROJECT_DIR/logs/qwen35-9b_a0_aime2025_v2/summary.json"
else
    echo "Summary not found (evaluation may have failed)"
fi
