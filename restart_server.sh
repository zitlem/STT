#!/bin/bash

# Speech-to-Text Restart Script (Linux & macOS)
# Called from server settings page via /api/server/restart

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

OS=$(uname -s)

# Check if running as root (Linux needs it for port 80 and systemctl)
if [ "$OS" = "Linux" ] && [ "$EUID" -ne 0 ]; then
    echo "Not running as root. Please run with: sudo -E ./restart_server.sh"
    echo "The -E flag preserves environment variables (needed for Python packages)"
    exit 1
fi

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# Read port from config.json
PORT=$(python3 -c "import json; print(json.load(open('config.json')).get('web_server',{}).get('port',80))" 2>/dev/null || echo 80)

echo "Stopping server..."

# ─── Stop managed services ──────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    systemctl stop stt-server 2>/dev/null
    systemctl stop stt 2>/dev/null
elif [ "$OS" = "Darwin" ]; then
    launchctl stop com.stt.server 2>/dev/null
fi

# ─── Kill port holder ───────────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    fuser -k "$PORT/tcp" 2>/dev/null
elif [ "$OS" = "Darwin" ]; then
    lsof -ti :"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null
fi

# ─── Kill speech_to_text processes ───────────────────────────────────
pkill -TERM -f "speech_to_text\.py" 2>/dev/null
sleep 1
pkill -9 -f "speech_to_text\.py" 2>/dev/null

# ─── Kill orphaned ffmpeg processes ──────────────────────────────────
if [ "$OS" = "Linux" ]; then
    pkill -TERM -f "ffmpeg.*alsa.*pipe:1" 2>/dev/null
    sleep 1
    pkill -9 -f "ffmpeg.*alsa.*pipe:1" 2>/dev/null
elif [ "$OS" = "Darwin" ]; then
    pkill -TERM -f "ffmpeg.*avfoundation" 2>/dev/null
    sleep 1
    pkill -9 -f "ffmpeg.*avfoundation" 2>/dev/null
fi
sleep 1

# ─── Wait for clean shutdown ────────────────────────────────────────
RETRIES=0
while pgrep -f "speech_to_text\.py" > /dev/null 2>&1; do
    echo "Waiting for processes to stop..."
    pkill -9 -f "speech_to_text\.py" 2>/dev/null
    if [ "$OS" = "Linux" ]; then
        fuser -k "$PORT/tcp" 2>/dev/null
    elif [ "$OS" = "Darwin" ]; then
        lsof -ti :"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null
    fi
    sleep 1
    RETRIES=$((RETRIES + 1))
    if [ "$RETRIES" -ge 10 ]; then
        echo -e "${RED}[ERROR]${NC} Could not stop all processes after 10 attempts"
        break
    fi
done
echo "All server processes stopped"
sleep 2

# ─── Start server ───────────────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    # Try systemd first
    for service_name in stt-server stt; do
        if systemctl list-unit-files "${service_name}.service" 2>/dev/null | grep -q "$service_name"; then
            echo "Starting via systemd ($service_name)..."
            systemctl start "$service_name"
            sleep 3
            if systemctl is-active --quiet "$service_name" 2>/dev/null; then
                echo -e "${GREEN}[OK]${NC} Server started ($service_name)"
                exit 0
            fi
        fi
    done
elif [ "$OS" = "Darwin" ]; then
    # Try launchd first
    if launchctl list com.stt.server &> /dev/null; then
        echo "Starting via launchd..."
        launchctl start com.stt.server
        sleep 3
        echo -e "${GREEN}[OK]${NC} Server started (launchd)"
        exit 0
    fi
fi

# Fallback: start manually
echo "No managed service found, starting manually..."
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python3"
fi

nohup "$PYTHON_BIN" "$SCRIPT_DIR/speech_to_text.py" > "$SCRIPT_DIR/server.log" 2>&1 &

# ─── Verify ──────────────────────────────────────────────────────────
sleep 3
if pgrep -f "speech_to_text\.py" > /dev/null; then
    echo -e "${GREEN}[OK]${NC} Server started successfully"
else
    echo -e "${RED}[ERROR]${NC} Failed to start server. Check server.log or journalctl -u stt-server"
    exit 1
fi
