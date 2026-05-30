@echo off
REM Start the STT Watchdog in headless mode (Windows).
REM The watchdog manages speech_to_text.py: starts it, restarts on crash,
REM and checks for GitHub updates daily at 1am.

setlocal

set SCRIPT_DIR=%~dp0
set PYTHON_BIN=%SCRIPT_DIR%.venv\Scripts\python.exe
if not exist "%PYTHON_BIN%" set PYTHON_BIN=python

if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

start /min "STT Watchdog" "%PYTHON_BIN%" "%SCRIPT_DIR%watchdog.py" --headless

echo [OK] Watchdog started.
echo      Logs: %SCRIPT_DIR%logs\watchdog.log
echo      STT:  %SCRIPT_DIR%logs\stt.log
