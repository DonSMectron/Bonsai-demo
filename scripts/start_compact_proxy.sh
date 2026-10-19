#!/usr/bin/env bash
# Starts the /compact proxy in front of llama-server (see COMPACT.md).
# Open http://localhost:8081 instead of http://localhost:8080 for the chat UI.
# Extra arguments pass through to scripts/compact_proxy.py (e.g. --threshold 0.6).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

if [ -x "$DEMO_DIR/.venv/bin/python" ]; then
    PYTHON="$DEMO_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" "$SCRIPT_DIR/compact_proxy.py" "$@"
