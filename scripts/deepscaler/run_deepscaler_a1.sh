#!/usr/bin/env bash
set -euo pipefail
export DEEPSCALER_REWARD_MODE=terminal_only
exec bash "$(dirname "${BASH_SOURCE[0]}")/pnp_deepscaler.sh" "$@"
