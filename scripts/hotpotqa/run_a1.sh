#!/usr/bin/env bash
set -euo pipefail

# Outcome-only GRPO on the shared HotpotQA certificate interface: identical prompts, tools,
# data, budgets and runtime settings as pnp_hotpotqa.sh; only the reward arm differs.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOTPOTQA_REWARD_ARM=A1
export HOTPOTQA_RUN_MODE="${HOTPOTQA_RUN_MODE:-main}"
exec bash "$HERE/pnp_hotpotqa.sh" "$@"
