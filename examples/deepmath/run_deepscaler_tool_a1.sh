#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEEPSCALER_TOOL_ARM=A1
exec bash "$HERE/run_deepscaler_tool_pair.sh" "$@"
