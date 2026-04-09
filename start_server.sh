#!/bin/bash

# Speech-to-Text Start Script (Linux & macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Read port from config.json
PORT=$(python3 -c "import json; print(json.load(open('config.json')).get('web_server',{}).get('port',80))" 2>/dev/null || echo 80)

# Check if already running
if pgrep -f "speech_to_text\.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}[WARNING]${NC} Server is already running"
    echo "Use ./restart_server.sh to restart or ./stop_server.sh to stop"
    exit 1
fi

# Check if port is in use
if command -v fuser &> /dev/null && fuser "$PORT/tcp" 2>/dev/null; then
    echo -e "${RED}[ERROR]${NC} Port $PORT is already in use by another process"
    exit 1
elif command -v lsof &> /dev/null && lsof -i :"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
    echo -e "${RED}[ERROR]${NC} Port $PORT is already in use by another process"
    exit 1
fi

# Determine Python binary
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

OS=$(uname -s)

if [ "$OS" = "Linux" ]; then
    # Linux: try systemd first
    for service_name in stt-server stt; do
        if systemctl list-unit-files "${service_name}.service" 2>/dev/null | grep -q "$service_name"; then
            echo "Starting via systemd ($service_name)..."
            sudo systemctl start "$service_name"
            sleep 2
            if systemctl is-active --quiet "$service_name"; then
                echo -e "${GREEN}[OK]${NC} Server started (systemd: $service_name)"
                echo "View logs: sudo journalctl -u $service_name -f"
                exit 0
            fi
        fi
    done
fi

# macOS launchd check
if [ "$OS" = "Darwin" ]; then
    if launchctl list com.stt.server &> /dev/null; then
        echo "Starting via launchd..."
        launchctl start com.stt.server
        sleep 2
        echo -e "${GREEN}[OK]${NC} Server started (launchd: com.stt.server)"
        echo "View logs: tail -f $SCRIPT_DIR/server.log"
        exit 0
    fi
fi

# Fallback: start manually
echo "Starting server on port $PORT..."
if [ "$PORT" -le 1024 ] && [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}[WARNING]${NC} Port $PORT requires root. Running with sudo..."
    sudo nohup "$PYTHON_BIN" "$SCRIPT_DIR/speech_to_text.py" > "$SCRIPT_DIR/server.log" 2>&1 &
else
    nohup "$PYTHON_BIN" "$SCRIPT_DIR/speech_to_text.py" > "$SCRIPT_DIR/server.log" 2>&1 &
fi

sleep 3

if pgrep -f "speech_to_text\.py" > /dev/null; then
    echo -e "${GREEN}[OK]${NC} Server started on port $PORT"
    echo "View logs: tail -f $SCRIPT_DIR/server.log"
else
    echo -e "${RED}[ERROR]${NC} Server failed to start. Check server.log"
    exit 1
fi
