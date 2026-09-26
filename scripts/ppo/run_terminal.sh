#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# != 1 )); then
    echo "Usage: run_terminal.sh {deepscaler|hotpotqa|taco|vision}" >&2
    exit 2
fi
export AGENT_R1_OPTIMIZER=ppo
case "$1" in
    deepscaler) exec bash "$HERE/deepscaler/run_deepscaler_a1.sh" ;;
    hotpotqa) exec bash "$HERE/hotpotqa/run_ppo_terminal.sh" ;;
    taco) exec bash "$HERE/taco/run_a1_terminal.sh" ;;
    vision) exec bash "$HERE/vision_r1/run_a1_terminal.sh" ;;
    *) echo "Unknown paper domain: $1" >&2; exit 2 ;;
esac
