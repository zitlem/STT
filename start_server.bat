@echo off
REM Speech-to-Text Start Script (Windows)

cd /d "%~dp0"

REM Check if already running
tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>nul | findstr /I "speech_to_text" >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARNING] Server is already running.
    echo Use restart_server.bat to restart or stop_server.bat to stop.
    pause
    exit /b 1
)

REM Determine Python binary
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo Starting Speech-to-Text server...
start "" "%PYTHON_BIN%" speech_to_text.py
echo [OK] Server starting...
echo Open your browser to http://localhost (or configured port)
timeout /t 3 >nul
