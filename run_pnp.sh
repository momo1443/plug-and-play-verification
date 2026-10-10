#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash run_pnp.sh DATASET [--model MODEL] [--model-path PATH] [--gpus IDS] [-- HYDRA_OVERRIDES...]

Datasets: deepscaler, hotpotqa, taco, vision_r1
Models:   Qwen3.5-4B (default), Qwen3.5-9B

Examples:
  bash run_pnp.sh deepscaler --model Qwen3.5-4B --gpus 0,1,2,3
  bash run_pnp.sh hotpotqa --model Qwen3.5-9B --gpus 0,1,2,3,4,5,6,7
  bash run_pnp.sh deepscaler --model-path /path/to/Qwen3.5-4B -- trainer.total_training_steps=10

Uses Python from the active environment unless PYTHON_BIN is set.
EOF
}

if (( $# == 0 )); then usage; exit 2; fi
case "$1" in -h|--help) usage; exit 0 ;; esac
DATASET="$1"
shift
case "$DATASET" in
    deepscaler|hotpotqa|taco|vision_r1) ;;
    vision) DATASET=vision_r1 ;;
    *) echo "Unknown PNP dataset: $DATASET" >&2; usage >&2; exit 2 ;;
esac

MODEL=Qwen3.5-4B
while (( $# > 0 )); do
    case "$1" in
        --model|--model-path|--gpus)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || {
                echo "Missing value for $1" >&2; exit 2;
            }
            case "$1" in
                --model) MODEL="$2" ;;
                --model-path) export PNP_MODEL_PATH="$2" ;;
                --gpus)
                    [[ "$2" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
                        echo "GPU IDs must be a comma-separated list, e.g. 0,1,2,3" >&2; exit 2;
                    }
                    export CUDA_VISIBLE_DEVICES="$2"
                    ;;
            esac
            shift 2
            ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        -* ) echo "Unknown PNP option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
case "$MODEL" in
    Qwen3.5-4B|Qwen3.5-9B) ;;
    *) echo "Unknown PNP model profile: $MODEL" >&2; exit 2 ;;
esac
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
exec bash "$PROJECT_DIR/scripts/models/$MODEL.sh" "$DATASET" "$@"
