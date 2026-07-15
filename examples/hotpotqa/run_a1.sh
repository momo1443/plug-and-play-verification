#!/usr/bin/env bash
set -euo pipefail

export HOTPOTQA_REWARD_ARM=A1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_rlvr.sh" "$@"
