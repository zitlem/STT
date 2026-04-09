#!/bin/bash

# Speech-to-Text Stop Script (Linux & macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running as root (Linux only — macOS doesn't need it for basic kill)
OS=$(uname -s)
if [ "$OS" = "Linux" ] && [ "$EUID" -ne 0 ]; then
    echo "Not running as root. Please run with: sudo ./stop_server.sh"
    exit 1
fi

echo "Stopping server..."

# Read port from config.json
PORT=$(python3 -c "import json; print(json.load(open('config.json')).get('web_server',{}).get('port',80))" 2>/dev/null || echo 80)

# ─── Stop managed services ──────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    for service_name in stt-server stt; do
        if systemctl is-active --quiet "$service_name" 2>/dev/null; then
            echo "Stopping $service_name systemd service..."
            systemctl stop "$service_name"
        fi
    done
elif [ "$OS" = "Darwin" ]; then
    if launchctl list com.stt.server &> /dev/null; then
        echo "Stopping launchd service..."
        launchctl stop com.stt.server
    fi
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

# ─── Verify ──────────────────────────────────────────────────────────
sleep 1
if pgrep -f "speech_to_text\.py" > /dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} Some processes still running:"
    ps aux | grep speech_to_text | grep -v grep
else
    echo -e "${GREEN}[OK]${NC} Server stopped"
fi
