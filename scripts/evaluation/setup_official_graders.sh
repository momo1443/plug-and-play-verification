#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRADER_ENV="${PNP_GRADER_ENV:-$PROJECT_DIR/.venv-official-graders}"
"${PYTHON_BIN:-python3}" -m venv "$GRADER_ENV"
"$GRADER_ENV/bin/python" -m pip install -r "$PROJECT_DIR/agent_r1/evaluation/requirements-official-graders.txt"
echo "Official grader Python: $GRADER_ENV/bin/python"
echo "Set PNP_GRADER_PYTHON to this path when using a custom environment location."
