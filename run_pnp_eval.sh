#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="$(command -v python)"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-$DEFAULT_PYTHON}" -m agent_r1.evaluation.pnp "$@"
