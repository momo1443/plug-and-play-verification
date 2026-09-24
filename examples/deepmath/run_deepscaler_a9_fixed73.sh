#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias. DeepScaleR A9 uses the uniform reward schedule;
# there is no fixed 0.7/0.3 reward mode.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "run_deepscaler_a9_fixed73.sh is deprecated; launching A9 uniform." >&2
exec bash "$HERE/run_deepscaler_a9_uniform.sh"
