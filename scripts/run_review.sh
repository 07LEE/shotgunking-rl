#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing .venv. Run: python -m venv .venv && .venv/bin/python -m pip install -e ." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" -m shotgun_king_rl.review.server --port 0 --open "$@"
