#!/bin/bash
# Start the STT Watchdog in headless mode (replaces start_server.sh for production use).
# The watchdog manages speech_to_text.py: starts it, restarts on crash,
# and checks for GitHub updates daily at 1am.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
[ ! -f "$PYTHON_BIN" ] && PYTHON_BIN="python3"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Prevent double-start
if "$PYTHON_BIN" -c "
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
try:
    s.bind(('127.0.0.1', 57337))
    s.close()
except OSError:
    sys.exit(1)
" 2>/dev/null; then
    : # port free, proceed
else
    echo "[INFO] Watchdog is already running."
    exit 0
fi

nohup "$PYTHON_BIN" "$SCRIPT_DIR/watchdog.py" --headless \
    >> "$LOG_DIR/watchdog.log" 2>&1 &

echo "[OK] Watchdog started (PID $!)"
echo "     Logs: tail -f $LOG_DIR/watchdog.log"
echo "     STT:  tail -f $LOG_DIR/stt.log"
