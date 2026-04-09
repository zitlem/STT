#!/bin/bash

# Speech-to-Text Installation Script
# Works on bare Ubuntu/Debian installations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$(pwd)"
SERVICE_NAME="stt-server"
PYTHON_BIN=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if running as root
check_not_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "This script should not be run as root for security reasons."
        print_error "Please run as a regular user with sudo access."
        exit 1
    fi
}

# Function to detect OS
detect_os() {
    local uname_s
    uname_s=$(uname -s)

    case $uname_s in
        Darwin)
            OS=macos
            ARCH=$(uname -m)
            print_success "Detected macOS ($ARCH)"
            ;;
        Linux)
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                OS=$ID
                VERSION_ID=$VERSION_ID
            else
                print_warning "Cannot read /etc/os-release. Assuming generic Linux."
                OS=linux
            fi
            case $OS in
                ubuntu|debian)
                    print_success "Detected $OS $VERSION_ID"
                    ;;
                *)
                    print_warning "Untested Linux distro: $OS. Script designed for Debian/Ubuntu."
                    read -p "Continue anyway? (y/N): " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        exit 1
                    fi
                    ;;
            esac
            ;;
        *)
            print_error "Unsupported OS: $uname_s. This script supports Linux (Debian/Ubuntu) and macOS."
            exit 1
            ;;
    esac
}

# Function to check and install system packages
install_system_deps() {
    print_status "Checking system packages..."

    if [ "$OS" = "macos" ]; then
        # macOS: use Homebrew
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew not found. Install it from https://brew.sh/ then re-run this script."
            exit 1
        fi
        print_status "Installing system packages via Homebrew..."
        brew install python3 ffmpeg portaudio
    else
        # Linux: use apt-get
        print_status "Updating package list..."
        sudo apt-get update

        PACKAGES=(
            "python3"
            "python3-dev"
            "build-essential"
            "git"
            "curl"
            "wget"
            "ffmpeg"
            "portaudio19-dev"
            "libasound2-dev"
            "alsa-utils"
            "psmisc"
        )

        print_status "Installing system packages: ${PACKAGES[*]}"
        sudo apt-get install -y "${PACKAGES[@]}"
    fi

    # Verify installations
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 installation failed"
        exit 1
    fi

    if ! command -v ffmpeg &> /dev/null; then
        print_error "FFmpeg installation failed"
        exit 1
    fi

    print_success "System packages installed successfully"
}

# Function to add user to audio group (Linux only)
setup_audio_permissions() {
    if [ "$OS" = "macos" ]; then
        print_status "Skipping audio group setup (not needed on macOS)"
        return
    fi

    print_status "Setting up audio permissions..."

    # Check if user is already in audio group
    if groups "$USER" | grep -q '\baudio\b'; then
        print_success "User already in audio group"
    else
        print_status "Adding user to audio group..."
        sudo usermod -a -G audio "$USER"
        print_warning "User added to audio group. You may need to log out and back in for changes to take effect."
    fi

    # Check if pulseaudio or pipewire is running
    if pgrep -x "pulseaudio" > /dev/null || pgrep -x "pipewire" > /dev/null; then
        print_success "Audio server detected"
    else
        print_warning "No audio server detected. For desktop audio, consider installing pulseaudio or pipewire."
    fi
}

# Function to install uv (fast Python package installer)
install_uv() {
    print_status "Installing uv package manager..."

    if command -v uv &> /dev/null; then
        print_success "uv already installed: $(uv --version)"
        return
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        print_success "uv installed: $(uv --version)"
    else
        print_error "uv installation failed"
        exit 1
    fi
}

# Function to create Python virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    uv venv "$INSTALL_DIR/.venv"
    export VIRTUAL_ENV="$INSTALL_DIR/.venv"
    PYTHON_BIN="$INSTALL_DIR/.venv/bin/python3"
    print_success "Virtual environment created at $INSTALL_DIR/.venv"
}

# Function to detect GPU and configure PyTorch
detect_gpu() {
    print_status "Detecting GPU..."

    if [ "$OS" = "macos" ]; then
        if [ "$(uname -m)" = "arm64" ]; then
            print_success "Apple Silicon detected — MPS (Metal) GPU acceleration available"
            return 0
        else
            print_warning "Intel Mac detected — no GPU acceleration (CPU only)"
            return 1
        fi
    else
        if command -v nvidia-smi &> /dev/null; then
            print_success "NVIDIA GPU detected:"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

            # Check CUDA version
            CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
            if [ -n "$CUDA_VERSION" ]; then
                print_status "CUDA Version: $CUDA_VERSION"
            fi

            return 0
        else
            print_warning "No NVIDIA GPU detected or drivers not installed"
            print_warning "Will install CPU-only PyTorch"
            return 1
        fi
    fi
}

# Function to install Python packages
install_python_deps() {
    print_status "Installing Python dependencies from requirements.txt..."

    if [ "$OS" = "macos" ]; then
        # macOS: standard PyPI PyTorch includes MPS support; no CUDA index needed
        detect_gpu  # Just for informational output
        print_status "Installing packages (MPS/CPU)..."
        uv pip install -r requirements.txt
    else
        # Linux: install CUDA PyTorch if NVIDIA GPU present, otherwise CPU
        if detect_gpu; then
            print_status "Installing GPU-enabled packages (CUDA)..."
            uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
        else
            print_status "Installing CPU-only packages..."
            uv pip install -r requirements.txt
        fi
    fi

    print_success "Python dependencies installed"
}

# Function to verify Python installation
verify_python_deps() {
    print_status "Verifying Python installation..."

    "$PYTHON_BIN" -c "
import sys
try:
    import torch
    print(f'PyTorch {torch.__version__} installed')
    if torch.cuda.is_available():
        print(f'CUDA available: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('MPS (Apple Silicon GPU) available')
    else:
        print('GPU not available (CPU mode)')

    import speech_recognition
    print('SpeechRecognition installed')

    import numpy
    print('NumPy installed')

    import pyaudio
    print('PyAudio installed')

    import flask
    print('Flask installed')

    print('\nAll critical packages verified!')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"

    print_success "Python verification passed"
}


# Function to list audio devices
list_audio_devices() {
    print_status "Listing available audio devices..."

    "$PYTHON_BIN" -c "
from audio_capture import list_audio_devices
devices = list_audio_devices()
print('\\nAvailable audio devices:')
print('-' * 60)
for i, device in enumerate(devices):
    print(f'{i}: {device[\"name\"]}')
    print(f'   Display: {device.get(\"display_name\", device[\"name\"])}')
    if device.get('is_default'):
        print('   [DEFAULT]')
    print()
"
}

# Function to test audio capture
test_audio_capture() {
    print_status "Would you like to test audio capture?"
    read -p "Test audio capture? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    print_status "Testing audio capture for 3 seconds..."
    "$PYTHON_BIN" << 'PYEOF'
import sys
sys.path.insert(0, '.')
from audio_capture import FFmpegAudioCapture
import time

try:
    capture = FFmpegAudioCapture(sample_rate=16000, chunk_duration=1.0, device_name=None)
    capture.ts_enabled = False  # Don't create backup for test
    
    print("Starting capture...")
    capture.start()
    time.sleep(3)
    capture.stop()
    print("\nAudio capture test completed successfully!")
except Exception as e:
    import platform
    print(f"\nAudio capture test failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check if microphone is connected and permitted")
    if platform.system() == "Darwin":
        print("2. macOS: Grant microphone access in System Settings > Privacy & Security > Microphone")
        print("3. List devices: ffmpeg -f avfoundation -list_devices true -i \"\"")
        print("4. Try device ':0' or ':1' in config.json default_microphone")
    else:
        print("2. Check if user is in 'audio' group (log out and back in if added)")
        print("3. Try: arecord -l  (to list recording devices)")
        print("4. Check ALSA: cat /proc/asound/cards")
PYEOF
}

# Function to create systemd service (Linux) or LaunchAgent (macOS)
setup_systemd_service() {
    if [ "$OS" = "macos" ]; then
        setup_launchd_agent
    else
        setup_systemd_service_linux
    fi
}

setup_launchd_agent() {
    local PLIST_LABEL="com.stt.server"
    local PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_LABEL.plist"
    local PYTHON_BIN="$INSTALL_DIR/.venv/bin/python3"

    print_status "Would you like to set up a LaunchAgent for auto-start on login?"
    read -p "Setup LaunchAgent? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    mkdir -p "$HOME/Library/LaunchAgents"

    cat > "$PLIST_PATH" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>$INSTALL_DIR/speech_to_text.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/server.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/server.log</string>
</dict>
</plist>
PLISTEOF

    launchctl load "$PLIST_PATH"

    print_success "LaunchAgent created and loaded"
    print_status "Service commands:"
    print_status "  launchctl start $PLIST_LABEL   - Start service"
    print_status "  launchctl stop $PLIST_LABEL    - Stop service"
    print_status "  launchctl unload $PLIST_PATH   - Remove service"
    print_status "  tail -f $INSTALL_DIR/server.log  - View logs"
}

setup_systemd_service_linux() {
    print_status "Would you like to set up a systemd service for auto-start?"
    read -p "Setup systemd service? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    print_status "Creating systemd service..."

    # Create service file
    sudo tee "/etc/systemd/system/$SERVICE_NAME.service" > /dev/null <<SYSEOF
[Unit]
Description=STT Speech-to-Text Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment="PYTHONUNBUFFERED=1"
Environment="HOME=$HOME"
ExecStart=$INSTALL_DIR/.venv/bin/python3 $INSTALL_DIR/speech_to_text.py
ExecStop=/bin/bash -c 'pkill -TERM -f "speech_to_text\\.py" 2>/dev/null; sleep 2; pkill -9 -f "speech_to_text\\.py" 2>/dev/null; pkill -9 -f "ffmpeg.*alsa.*pipe:1" 2>/dev/null'
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
SYSEOF

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"

    print_success "Systemd service created and enabled"
    print_status "Service commands:"
    print_status "  sudo systemctl start $SERVICE_NAME   - Start service"
    print_status "  sudo systemctl stop $SERVICE_NAME    - Stop service"
    print_status "  sudo systemctl status $SERVICE_NAME  - Check status"
    print_status "  sudo journalctl -u $SERVICE_NAME -f  - View logs"
}


# Function to display final instructions
show_final_instructions() {
    echo
    echo "========================================"
    echo "  Installation Complete!"
    echo "========================================"
    echo
    print_success "Speech-to-Text has been installed successfully"
    echo
    echo "Quick Start:"
    if [ "$OS" = "macos" ]; then
        echo "  1. Start the application: sudo $INSTALL_DIR/.venv/bin/python3 speech_to_text.py"
        echo "     (sudo needed for port 80; or change port to 8080 in config.json)"
        echo "  2. Grant microphone access if prompted by macOS"
        echo "  3. Open browser: http://localhost:80"
        echo "  4. Set microphone in config.json: default_microphone: ':0'"
        echo "     (run: ffmpeg -f avfoundation -list_devices true -i \"\" to find device index)"
    else
        echo "  1. Start the application: sudo ./restart_server.sh"
        echo "  2. Open browser: http://localhost:80"
    fi
    echo "  - Go to /model-manager to download models"
    echo "  - Go to /live-settings to configure audio"
    echo
    echo "Configuration:"
    echo "  - Main config: $INSTALL_DIR/config.json"
    echo
    echo "Useful commands:"
    echo

    if [ "$OS" != "macos" ]; then
        if ! groups "$USER" | grep -q '\baudio\b'; then
            echo -e "${YELLOW}IMPORTANT:${NC} You were added to the 'audio' group."
            echo -e "${YELLOW}Please log out and log back in for audio permissions to take effect.${NC}"
            echo
        fi

        if [ -f "/etc/systemd/system/$SERVICE_NAME.service" ]; then
            echo "Systemd service installed:"
            echo "  sudo systemctl start $SERVICE_NAME  - Start"
            echo "  sudo systemctl stop $SERVICE_NAME   - Stop"
            echo "  sudo systemctl status $SERVICE_NAME - Status"
            echo
        fi
    fi
}

# Main installation function
main() {
    echo "========================================"
    echo "  Speech-to-Text Installation Script"
    echo "========================================"
    echo

    # Pre-flight checks
    check_not_root
    detect_os

    # Installation steps
    install_system_deps
    install_uv
    create_venv
    setup_audio_permissions
    install_python_deps
    verify_python_deps

    # Optional setups
    list_audio_devices
    test_audio_capture
    setup_systemd_service

    # Final instructions
    show_final_instructions
}

# Handle script interruption
trap 'print_error "Installation interrupted"; exit 1' INT TERM

# Run main function
main "$@"
