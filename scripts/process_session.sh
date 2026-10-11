#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
COLLECTION="$PROJECT_ROOT/data/collection"

usage() {
    echo "Usage: ./scripts/process_session.sh SESSION_ID [--crop LEFT TOP RIGHT BOTTOM] [--prepare-only]" >&2
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0
    exit 2
fi

SESSION_ID="$1"
shift

if [[ ! "$SESSION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Invalid session ID: use only letters, digits, '-' and '_'." >&2
    exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing .venv. Run: python -m venv .venv && .venv/bin/python -m pip install -e ." >&2
    exit 1
fi

mkdir -p "$COLLECTION/inbox"
INBOX_COUNT="$(find "$COLLECTION/inbox" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -printf '.' | wc -c)"
ORIGINALS="$COLLECTION/sessions/$SESSION_ID/originals"
ORIGINAL_COUNT=0
if [[ -d "$ORIGINALS" ]]; then
    ORIGINAL_COUNT="$(find "$ORIGINALS" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -printf '.' | wc -c)"
fi

if [[ "$INBOX_COUNT" -eq 0 && "$ORIGINAL_COUNT" -eq 0 ]]; then
    echo "No images found in data/collection/inbox or session $SESSION_ID." >&2
    exit 1
fi

SPLIT="$("$PYTHON" - "$COLLECTION" "$SESSION_ID" "$((INBOX_COUNT + ORIGINAL_COUNT))" <<'PY'
from pathlib import Path
import sys

from shotgun_king_rl.collection.dataset import choose_session_split

collection = Path(sys.argv[1])
session_id = sys.argv[2]
screen_count = int(sys.argv[3])
print(choose_session_split(collection, session_id, screen_count))
PY
)"
echo "Session: $SESSION_ID"
echo "Split: $SPLIT"
echo "Inbox images: $INBOX_COUNT"
echo "Existing session originals: $ORIGINAL_COUNT"

"$PROJECT_ROOT/scripts/run_review.sh" --session "$SESSION_ID" "$@"

echo
echo "Collection audit after session processing:"
"$PYTHON" "$PROJECT_ROOT/tools/preprocess/audit_collection.py"
