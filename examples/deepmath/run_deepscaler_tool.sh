#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible terminal-only ToolEnv entrypoint.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/run_deepscaler_tool_a1.sh" "$@"
