#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
COLLECTION="$PROJECT_ROOT/data/collection"

usage() {
    cat >&2 <<'EOF'
Usage: ./scripts/run_collect.sh [--session SESSION_ID] [--review] [MONITOR_OPTIONS...]

Capture recognition gaps while you play. Stop with Ctrl+C.
  --session ID   Session to create or resume (default: today's date and the next run number)
  --review       Open the review page after capture stops
Other options go to tools/monitor_collection.py (for example --save-all, --duration 1800).
EOF
}

SESSION_ID=""
REVIEW=0
MONITOR_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --session)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            SESSION_ID="$2"; shift 2 ;;
        --review) REVIEW=1; shift ;;
        *) MONITOR_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing .venv. Run: python3 -m venv .venv && .venv/bin/python -m pip install -e ." >&2
    exit 1
fi

if [[ -z "$SESSION_ID" ]]; then
    # Run numbers continue across days, matching existing session names.
    # Each annotation name starts with its session ID and "__".
    LAST_RUN="$(find "$COLLECTION/annotations" -maxdepth 1 -type f -name '*__*.json' -printf '%f\n' 2>/dev/null \
        | sed 's/__.*//' | sed -n 's/.*-run-\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1)"
    SESSION_ID="$(date +%F)-run-$(printf '%02d' $((10#${LAST_RUN:-0} + 1)))"
fi
if [[ ! "$SESSION_ID" =~ ^[A-Za-z0-9]+([-_][A-Za-z0-9]+)*$ ]]; then
    echo "Invalid session ID: join letters and digits with single '-' or '_'." >&2
    exit 2
fi

count_drafts() {
    local annotations="$COLLECTION/annotations"
    if [[ -d "$annotations" ]]; then
        find "$annotations" -maxdepth 1 -type f -name "${SESSION_ID}__*.json" -printf '.' | wc -c
    else
        echo 0
    fi
}

BEFORE="$(count_drafts)"
echo "Session: $SESSION_ID"
cd "$PROJECT_ROOT"
# The monitor handles Ctrl+C itself and exits normally, so the summary still runs.
"$PYTHON" "$PROJECT_ROOT/tools/monitor_collection.py" --session "$SESSION_ID" ${MONITOR_ARGS[@]+"${MONITOR_ARGS[@]}"}
AFTER="$(count_drafts)"

echo
echo "New drafts in $SESSION_ID: $((AFTER - BEFORE)) (total $AFTER)"
if [[ "$REVIEW" -eq 1 && "$AFTER" -gt 0 ]]; then
    # Without --session, review does not move inbox files into this session.
    exec "$SCRIPT_DIR/run_review.sh"
fi
if [[ "$AFTER" -gt 0 ]]; then
    echo "Review drafts with: ./scripts/run_review.sh"
fi
