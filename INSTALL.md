# Installation Guide

## Quick Install (Debian 12 / Ubuntu 22.04+)

### Automated Installation

Run the installation script:

```bash
chmod +x install.sh
./install.sh
```

The script will:
- Install all system dependencies
- Detect and configure NVIDIA GPU (if available)
- Create a Python virtual environment (optional)
- Install all Python packages
- Test GPU availability

---

## Manual Installation

### 1. System Dependencies

#### Debian 12 / Ubuntu 22.04+

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libasound2-dev \
    build-essential \
    git \
    ffmpeg \
    curl \
    wget
```

#### For NVIDIA GPU Support

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not installed:
sudo apt-get install -y nvidia-driver firmware-misc-nonfree

# Reboot after installation
sudo reboot
```

### 2. Python Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 3. Install Python Dependencies

#### macOS (Apple Silicon or Intel) / CPU-only Linux:
```bash
pip install -r requirements.txt
```

#### Linux with NVIDIA GPU (CUDA 12.1):
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Note: The install.sh script detects your platform and GPU automatically.

### 4. Verify Installation

```bash
# Linux/NVIDIA GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# macOS Apple Silicon
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## Running the Application

### Option 1: Watchdog — recommended for production

The Watchdog process manages STT with automatic crash recovery and auto-updates.
Run it headless (no GUI) so it keeps STT alive in the background.

**Binary install (downloaded release):**

```bash
# Linux / macOS
./STT-Watchdog --headless

# Windows
STT-Watchdog.exe --headless
```

The helper scripts handle logging automatically:

```bash
# Linux / macOS
./start_watchdog.sh

# Windows (cmd)
start_watchdog.bat

# Windows (PowerShell)
.\start_watchdog.ps1
```

Logs go to `~/.stt/logs/watchdog.log` (binary) or `logs/watchdog.log` (source).

**Source install:**

```bash
# Activate virtual environment first if you use one
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

python3 watchdog.py --headless
```

### Option 2: Direct start — for development / testing

```bash
# If using virtual environment, activate it first:
source venv/bin/activate

# Run the application
python3 speech_to_text.py
```

### Access the Web UI

Open your browser to:
- **Local:** http://localhost:80
- **Network:** http://your-server-ip:80

### First Time Setup

1. Go to `/model-manager` to download Whisper models
2. Go to `/server-settings` to configure paths and network
3. Go to `/live-settings` to configure audio device and language
4. Start transcribing on the home page!

---

## Service Setup (Auto-start on Boot)

### Linux — systemd

```bash
sudo cp stt-watchdog.service /etc/systemd/system/
```

Open the file and set `User=` to your username. For a **source install** the `ExecStart`
line is already uncommented; for a **binary install** swap it with the commented binary line.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stt-watchdog

# Useful commands
sudo systemctl status stt-watchdog
sudo journalctl -u stt-watchdog -f
```

### macOS — LaunchAgent

```bash
cp com.stt.watchdog.plist ~/Library/LaunchAgents/
```

Edit the file and replace every `INSTALL_DIR` placeholder with the full path to your
STT directory. For a **binary install** follow the comment in the plist to switch
`ProgramArguments` to the single binary path.

```bash
launchctl load ~/Library/LaunchAgents/com.stt.watchdog.plist

# Check it's running
launchctl list | grep stt

# View logs
tail -f ~/.stt/logs/watchdog.log
```

To stop and remove:

```bash
launchctl unload ~/Library/LaunchAgents/com.stt.watchdog.plist
```

### Windows — Task Scheduler

1. Open **Task Scheduler** → *Create Basic Task*
2. **Trigger:** At log on (or At startup for system-wide)
3. **Action:** Start a program
   - Binary install: `C:\path\to\STT-Watchdog.exe`, argument `--headless`
   - Source install: `pythonw.exe`, argument `"C:\path\to\watchdog.py" --headless`
4. **Settings:** Enable *Run whether user is logged on or not* for true headless operation

Alternatively, run `start_watchdog.bat` or `start_watchdog.ps1` from the Startup folder
(`shell:startup` in Run dialog) for a simpler per-user setup.

---

## System Requirements

### Minimum (CPU Only)
- **OS:** Debian 12, Ubuntu 22.04+, or similar Linux
- **CPU:** 4 cores
- **RAM:** 8 GB
- **Storage:** 10 GB free space
- **Python:** 3.9 - 3.13

### Recommended (with GPU)
- **OS:** Debian 12, Ubuntu 22.04+
- **CPU:** 8 cores
- **RAM:** 16 GB
- **GPU:** NVIDIA GPU with 4GB+ VRAM (RTX 2060 or better)
- **CUDA:** 12.1 compatible drivers
- **Storage:** 20 GB free space
- **Python:** 3.10 - 3.13

---

## Troubleshooting

### PyAudio Build Fails

```bash
# Install missing dependencies
sudo apt-get install -y python3-dev portaudio19-dev
```

### NVIDIA GPU Not Detected

```bash
# Check driver installation
nvidia-smi

# Reinstall NVIDIA drivers
sudo apt-get install --reinstall nvidia-driver

# Reboot
sudo reboot
```

### Port 80 Permission Denied

Either:
1. Run with sudo (not recommended for production)
2. Change port in `config.json` to 8080 or higher
3. Use port forwarding/reverse proxy

### ImportError or Missing Modules

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

---

## Configuration

### Main Config File
`config.json` is created automatically on first run by copying `config.default.json`.
It is gitignored — edit it freely without worrying about committing passwords or personal settings.

To start from scratch: delete `config.json` and restart the app.
To update the default template for others: edit `config.default.json` and commit it.

Key settings:
- Network host, port, and IP whitelist
- Audio device and capture backend
- Database paths and filename format
- Model selection and backend
- Audio backup paths and formats

### Web Interface
Most settings can be configured through the web UI:
- `/server-settings` - Server configuration
- `/live-settings` - Live transcription settings
- `/timezone-settings` - Timestamp configuration
- `/model-manager` - Download and manage AI models
- `/file-manager` - File management and backup settings

---

## Updating

```bash
# Pull latest changes
git pull

# Activate virtual environment if using one
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart the application
```

---

## Uninstallation

```bash
# Stop the application (Ctrl+C)

# Remove virtual environment
rm -rf venv

# Remove downloaded models (optional)
rm -rf models

# Remove recordings and databases (optional)
rm -rf _AUTOMATIC_BACKUP

# Uninstall system packages (optional, be careful!)
# sudo apt-get remove portaudio19-dev python3-dev
```

---

## Support

For issues, questions, or feature requests:
- Check the configuration in `/server-settings`
- Review logs in the terminal
- Check the `config.json` file for errors

---

## License

See LICENSE file for details.
