#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VISION_R1_ARM=A9
exec bash "$HERE/run_visual_agent.sh" "$@"
