#!/usr/bin/env bash
set -euo pipefail
export DEEPSCALER_REWARD_MODE=terminal_only
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_deepscaler_a9_uniform.sh" "$@"
