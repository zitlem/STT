# STT - Speech-To-Text

Real-time speech transcription platform with a modern web interface, powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). Designed for continuous transcription scenarios like church services, lectures, and meetings.

## Features

### Transcription
- **Real-time transcription** - Live microphone capture with sub-second latency via WebSocket (FFmpeg or PyAudio backends)
- **File transcription** - Upload and transcribe audio/video files in batch
- **Selectable engines** - Faster-Whisper (CTranslate2), OpenAI Whisper, and HuggingFace models (distil-whisper, wav2vec2)
- **Voice Activity Detection** - Silero VAD filters silence and noise before transcription
- **Speech/music/quiet detection** - PANNs-based three-way classification; detected music can be transcribed but auto-hidden, restorable from corrections
- **Microphone calibration** - Guided wizard measures ambient noise and suggests threshold settings
- **Accuracy tuning** - Loudness normalization, sentence-completion buffering, context prompting, and per-mode Whisper decoding parameters
- **Hallucination filter** - Removes Whisper phantom phrases ("thanks for watching", etc.)
- **Device auto-recovery** - Re-finds the microphone by card name after reboots or device-index changes

### Translation & speech
- **Translation** - Real-time translation to 200+ languages using Facebook NLLB-200
- **Remote translation offload** - Pair with another STT machine and offload translation to it, with reachability checks and configurable fallback
- **Custom dictionary & glossary** - Domain-specific term corrections, forced NLLB translations, synced to the paired remote machine
- **Text-to-Speech** - Edge-TTS (cloud) and Piper-TTS (local) with auto voice switching per language and speed control

### Display & output
- **Display profiles** - Named, recallable output layouts built in `/url-builder` and served at `/profile/<name>`
- **Layout modes** - Translated-only, side-by-side, or stacked, with drip-feed word-by-word reveal
- **Browser audio streaming** - Remote viewers can listen to the live room microphone and TTS audio in their browser

### Review & content control
- **Corrections workflow** - Review queue with confidence scores, low-confidence word flagging, and alternative translations
- **Staged output delay** - Hold segments for N seconds to approve or discard before they go live
- **Word highlighting** - Mark and emphasize specific words or phrases in transcriptions
- **Profanity filter** - Masks configured words with `****` in output

### Storage & files
- **Database storage** - SQLite per-session with SRT subtitle and HTML export, plus optional partial-snapshot recording
- **Audio backup** - WAV and MPEG-TS formats with power-fail-safe continuous backup
- **File manager** - Web-based browser for backups: rename, download, hide, bulk operations, type/day filters
- **Remote file delivery** - Automatic backup to SMB/NAS shares

### Operations
- **Model manager** - Browse, search, and download Whisper, NLLB, VAD, and PANNs models from Hugging Face, or upload local models
- **Security** - IP whitelist (CIDR), password authentication, session timeouts
- **Hardware acceleration** - NVIDIA CUDA and Apple Silicon (MPS) with automatic detection
- **Crash recovery & auto-update** - Watchdog process manager restarts STT on crashes; idle-gated updates with stable/beta channels
- **Server tools** - Uptime/version display, disk-space monitor, timezone settings, runtime language switching

## Quick Start

```bash
# Install dependencies
./install.sh

# Start the server
./start_server.sh        # Linux / macOS (start_server.bat on Windows)
```

Open http://localhost:8080 in your browser (port is configurable in `config/config.json`).

## First Time Setup

1. Go to `/model-manager` to download Whisper models
2. Go to `/live-settings` to select your microphone and language
3. Start transcribing on the home page

## Running Headless (No GUI)

The **Watchdog** manages STT with crash recovery and auto-updates. Run it headless to keep STT running in the background without a desktop.

### Binary install (downloaded release)

| Platform | Command |
|----------|---------|
| Linux / macOS | `./STT-Watchdog --headless` |
| Windows | `STT-Watchdog.exe --headless` |

Or use the provided scripts which handle logging automatically:

```bash
# Linux / macOS
./start_watchdog.sh

# Windows (cmd)
start_watchdog.bat

# Windows (PowerShell)
.\start_watchdog.ps1
```

### Source install

```bash
python3 stt/watchdog.py --headless
```

### Persistent service (auto-start on boot)

**Linux (systemd)**

```bash
sudo cp deploy/stt-watchdog.service /etc/systemd/system/
# Edit the file to set User= and adjust paths if needed
sudo systemctl daemon-reload
sudo systemctl enable --now stt-watchdog
sudo journalctl -u stt-watchdog -f   # view logs
```

**macOS (LaunchAgent)**

```bash
cp deploy/com.stt.watchdog.plist ~/Library/LaunchAgents/
# Edit INSTALL_DIR placeholders to your actual install path
launchctl load ~/Library/LaunchAgents/com.stt.watchdog.plist
```

**Windows (Task Scheduler)**

Run once at startup via Task Scheduler:
- Action: `STT-Watchdog.exe --headless` (or `pythonw stt/watchdog.py --headless` for source)
- Trigger: At log on / At startup
- Settings: Run whether user is logged on or not

## Web Interface Pages

| Page | Description |
|------|-------------|
| `/` | Live transcription with real-time updates |
| `/file` | File upload and batch transcription |
| `/translation` | Translation settings, language pairs, TTS voice selection |
| `/corrections` | Review and edit transcription segments |
| `/word-highlighting` | Manage highlighted phrases |
| `/url-builder` | Build and save display profiles (fonts, colors, layout URL parameters) |
| `/live-settings` | Audio device, language, VAD settings |
| `/server-settings` | Network, database, backup configuration |
| `/model-manager` | Download and manage AI models |
| `/file-manager` | File browser and SMB/NAS settings |
| `/profile/<name>` | Named display profile output layout |

## Documentation

See [INSTALL.md](INSTALL.md) for detailed installation instructions, system requirements, and troubleshooting.

## System Requirements

### Minimum (CPU Only)
- **CPU:** 4 cores | **RAM:** 8 GB | **Storage:** 10 GB
- **Python:** 3.9 - 3.13
- **OS:** Linux, Windows, or macOS

### Recommended (with GPU)
- **CPU:** 8 cores | **RAM:** 16 GB | **Storage:** 20 GB
- **GPU:** NVIDIA with 4GB+ VRAM (RTX 2060 or better)
- **CUDA:** 12.8 compatible drivers (NVIDIA driver R570+)

## Configuration

Edit `config/config.json` or use the web interface. Key settings include:

- Model selection (Whisper variant, backend)
- Audio device and capture backend (FFmpeg or PyAudio)
- Voice Activity Detection (Silero VAD) threshold
- Database paths and naming format
- Audio backup paths and formats
- Translation model and glossary
- Network host, port, and security

## Privacy & Telemetry

- **Error reporting (Sentry)** - Crash reports, logs, and performance traces are sent to Sentry to help improve STT. Disable via the toggle on `/server-settings` or set `sentry_enabled: false` in `config/config.json` — crash dumps are then kept locally only.

## Tech Stack

- **Backend:** Python 3.9+ with Flask and Flask-SocketIO
- **Speech Recognition:** Faster-Whisper (CTranslate2)
- **Translation:** Facebook NLLB-200 via Hugging Face Transformers
- **TTS:** Edge-TTS and Piper-TTS
- **Audio:** FFmpeg for capture and processing, Silero VAD
- **ML:** PyTorch with CUDA 12.8 support
- **Frontend:** Bootstrap, jQuery, Socket.IO
- **Database:** SQLite (per-session)

## Cross-Platform Support (Source Install)

Installation scripts are provided for all platforms:

| Platform | Install | Start | Stop | Restart |
|----------|---------|-------|------|---------|
| Linux/macOS | `install.sh` | `start_server.sh` | `stop_server.sh` | `restart_server.sh` |
| Windows | `install.bat` / `install.ps1` | `start_server.bat` | `stop_server.bat` | `restart_server.bat` |

On Linux, the installer can also set up a **systemd service** for auto-start on boot.
