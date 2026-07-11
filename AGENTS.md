# STT — Agent Instructions

## Project overview

Real-time speech transcription platform powered by Faster-Whisper, with a web UI for live event operators (church services, lectures, meetings). Live microphone transcription streams over WebSocket; also supports batch file transcription, real-time translation (Facebook NLLB-200), text-to-speech (Edge-TTS cloud / Piper-TTS local), and PANNs-based music/speech detection.

Stack: Python 3.9+, Flask + Flask-SocketIO, SQLite (per-session databases), Jinja2 server-rendered templates. No frontend build step — `static/` contains vendored jQuery, socket.io, and Font Awesome.

## Commands

The project venv (`.venv/`, Python 3.9) is uv-managed and has no pip — use `uv pip install` or the venv's python directly.

```bash
.venv/bin/python3 speech_to_text.py      # Run the server (port 80) — ./start_server.sh / .bat wrappers exist
.venv/bin/python3 -m pytest              # Run tests (testpaths = tests/, config in pyproject.toml)
.venv/bin/python3 -m ruff check .        # Lint (line-length 200, target py39)
./install.sh                             # Install runtime deps (install.bat / install.ps1 on Windows)
uv pip install -r requirements-dev.txt   # Dev/test deps (pytest, ruff)
```

Ruff is configured with hard-error rules only (E9, F63, F7, F82, F811) and excludes `.venv`, `_AUTOMATIC_BACKUP`, `models`, `installer`.

## Architecture

The server is mostly a monolith — most changes land in `speech_to_text.py`.

| Path | Role |
|------|------|
| `speech_to_text.py` (~15,700 lines) | The entire server: Flask routes, SocketIO events, transcription pipeline, translation, TTS, SQLite storage, settings |
| `stt/audio_capture.py` | Microphone capture layer |
| `stt/watchdog.py` | Separate process manager: crash recovery, auto-update, headless mode (`--headless`) |
| `stt/file_mover.py` | SMB/NAS remote file delivery |
| `templates/` | Jinja2 pages: index, live-settings, model-manager, server-settings, translation, corrections, file-manager, word-highlighting, url-builder |
| `static/` | Vendored JS/CSS — no build step, no npm |
| `tests/` | Pytest suite: download state, path safety, staging, text utils, watchdog update |
| `packaging/` | Watchdog binary build tooling (build.py, make_icon.py, watchdog.spec) — NOT `build/`, which PyInstaller uses as its workdir |
| `deploy/` | OS service templates: stt-watchdog.service (systemd), com.stt.watchdog.plist (launchd) |

## Configuration

- `config/config.default.json` — defaults/schema. **New settings need a default entry here.**
- `config/config.json` — the user's live runtime settings (gitignored, as are all non-.default files in `config/`); the server writes to it. Don't clobber it with defaults.

## Conventions

- **Commits**: conventional-commit style with a scope, e.g. `fix(translation): …`, `feat(server): …`. **No co-author lines** in commit messages — Claude attribution goes in `git notes` instead (push with `refs/notes/commits`).
- **UI work**: follow the design tokens in `DESIGN.md` (colors, typography, spacing, component styles).
- **Do not touch** `_AUTOMATIC_BACKUP/`, `models/`, `panns_data/`, `logs/` — generated/runtime data.
