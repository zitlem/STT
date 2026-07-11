#!/bin/bash
# Start the STT Watchdog in headless mode.
# Works with both the compiled binary and Python source installs.
# The watchdog manages STT: starts it, restarts on crash, and auto-updates daily at 1am.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/STT-Watchdog"

# Prevent double-start
if nc -z 127.0.0.1 57337 2>/dev/null || \
   python3 -c "import socket,sys; s=socket.socket(); s.bind(('127.0.0.1',57337)); s.close()" 2>/dev/null; then
    :  # port free — but nc returning 0 means port is IN USE, so invert:
    true
fi
if nc -z 127.0.0.1 57337 2>/dev/null; then
    echo "[INFO] Watchdog is already running."
    exit 0
fi

if [ -f "$BINARY" ]; then
    # ── Compiled binary ──────────────────────────────────────────────────────
    LOG_DIR="$HOME/.stt/logs"
    mkdir -p "$LOG_DIR"
    nohup "$BINARY" --headless >> "$LOG_DIR/watchdog.log" 2>&1 &
    echo "[OK] Watchdog started (PID $!) — binary mode"
    echo "     Logs: tail -f $LOG_DIR/watchdog.log"
    echo "     STT:  tail -f $LOG_DIR/stt.log"
else
    # ── Python source fallback ───────────────────────────────────────────────
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
    [ ! -f "$PYTHON_BIN" ] && PYTHON_BIN="python3"
    LOG_DIR="$SCRIPT_DIR/logs"
    mkdir -p "$LOG_DIR"
    nohup "$PYTHON_BIN" "$SCRIPT_DIR/stt/watchdog.py" --headless >> "$LOG_DIR/watchdog.log" 2>&1 &
    echo "[OK] Watchdog started (PID $!) — Python source mode"
    echo "     Logs: tail -f $LOG_DIR/watchdog.log"
    echo "     STT:  tail -f $LOG_DIR/stt.log"
fi
