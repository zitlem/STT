@echo off
REM Start the STT Watchdog in headless mode (Windows).
REM Works with both the compiled binary and Python source installs.

setlocal

set SCRIPT_DIR=%~dp0

if exist "%SCRIPT_DIR%STT-Watchdog.exe" (
    REM ── Compiled binary ──────────────────────────────────────────────────
    start /min "STT Watchdog" "%SCRIPT_DIR%STT-Watchdog.exe" --headless
    echo [OK] Watchdog started (binary mode).
    echo      Logs: %USERPROFILE%\.stt\logs\watchdog.log
) else (
    REM ── Python source fallback ────────────────────────────────────────────
    set PYTHON_BIN=%SCRIPT_DIR%.venv\Scripts\python.exe
    if not exist "%PYTHON_BIN%" set PYTHON_BIN=python
    if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"
    start /min "STT Watchdog" "%PYTHON_BIN%" "%SCRIPT_DIR%stt\watchdog.py" --headless
    echo [OK] Watchdog started (Python source mode).
    echo      Logs: %SCRIPT_DIR%logs\watchdog.log
)
