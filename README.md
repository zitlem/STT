# STT - Speech-To-Text

Real-time speech transcription platform with a modern web interface, powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). Designed for continuous transcription scenarios like church services, lectures, and meetings.

## Features

- **Real-time transcription** - Live microphone capture with sub-second latency via WebSocket
- **File transcription** - Upload and transcribe audio/video files in batch
- **Translation** - Real-time translation to 200+ languages using Facebook NLLB-200
- **Text-to-Speech** - Edge-TTS (cloud) and Piper-TTS (local) with auto voice switching per language
- **Corrections workflow** - Edit transcriptions with a review queue and approval system
- **Word highlighting** - Mark and emphasize specific words or phrases in transcriptions
- **Custom glossary** - Domain-specific term mapping for improved accuracy
- **Database storage** - SQLite per-session with SRT subtitle and HTML export
- **Audio backup** - WAV and MPEG-TS formats with power-fail-safe continuous backup
- **Remote file delivery** - Automatic backup to SMB/NAS shares
- **Model manager** - Browse, search, and download Whisper models from Hugging Face
- **Security** - IP whitelist (CIDR), password authentication, session timeouts
- **GPU acceleration** - CUDA support with automatic detection

## Quick Start

```bash
# Install dependencies
./install.sh

# Start the server
python3 speech_to_text.py
```

Open http://localhost:80 in your browser.

## First Time Setup

1. Go to `/model-manager` to download Whisper models
2. Go to `/live-settings` to select your microphone and language
3. Start transcribing on the home page

## Web Interface Pages

| Page | Description |
|------|-------------|
| `/` | Live transcription with real-time updates |
| `/file` | File upload and batch transcription |
| `/translation` | Translation settings, language pairs, TTS voice selection |
| `/corrections` | Review and edit transcription segments |
| `/word-highlighting` | Manage highlighted phrases |
| `/live-settings` | Audio device, language, VAD settings |
| `/server-settings` | Network, database, backup configuration |
| `/model-manager` | Download and manage AI models |
| `/file-manager` | File browser and SMB/NAS settings |

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
- **CUDA:** 12.1 compatible drivers

## Configuration

Edit `config.json` or use the web interface. Key settings include:

- Model selection (Whisper variant, backend)
- Audio device and capture backend (FFmpeg or PyAudio)
- Voice Activity Detection (Silero VAD) threshold
- Database paths and naming format
- Audio backup paths and formats
- Translation model and glossary
- Network host, port, and security

## Tech Stack

- **Backend:** Python 3.9+ with Flask and Flask-SocketIO
- **Speech Recognition:** Faster-Whisper (CTranslate2)
- **Translation:** Facebook NLLB-200 via Hugging Face Transformers
- **TTS:** Edge-TTS and Piper-TTS
- **Audio:** FFmpeg for capture and processing, Silero VAD
- **ML:** PyTorch with CUDA 12.1 support
- **Frontend:** Bootstrap, jQuery, Socket.IO
- **Database:** SQLite (per-session)

## Cross-Platform Support

Installation scripts are provided for all platforms:

| Platform | Install | Start | Stop | Restart |
|----------|---------|-------|------|---------|
| Linux/macOS | `install.sh` | `start_server.sh` | `stop_server.sh` | `restart_server.sh` |
| Windows | `install.bat` / `install.ps1` | `start_server.bat` | `stop_server.bat` | `restart_server.bat` |

On Linux, the installer can also set up a **systemd service** for auto-start on boot.
