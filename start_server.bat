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

set "PORT="
for /f "delims=" %%p in ('"%PYTHON_BIN%" -c "import json;print(json.load(open('config/config.json')).get('web_server',{}).get('port',8080))" 2^>nul') do set "PORT=%%p"
if not defined PORT set "PORT=8080"

echo Open your browser to http://localhost:%PORT%
timeout /t 3 >nul
