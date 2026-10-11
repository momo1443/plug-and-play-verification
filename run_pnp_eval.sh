#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="$PROJECT_DIR/../miniforge/envs/pnp/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-$DEFAULT_PYTHON}" -m agent_r1.evaluation.pnp "$@"
