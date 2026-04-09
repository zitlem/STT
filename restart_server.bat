@echo off
setlocal EnableDelayedExpansion
REM Speech-to-Text Restart Script (Windows)
REM Called from server settings page via /api/server/restart

cd /d "%~dp0"

echo [RESTART] Stopping server...

REM ─── Kill python processes running speech_to_text.py ───────────────
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul') do (
    wmic process where "ProcessId=%%~a" get CommandLine /FORMAT:LIST 2>nul | findstr /I "speech_to_text" >nul 2>&1
    if !errorlevel! equ 0 (
        echo Killing PID %%~a...
        taskkill /F /PID %%~a >nul 2>&1
    )
)

REM Also kill by window title
taskkill /F /FI "WINDOWTITLE eq speech_to_text*" >nul 2>&1

REM ─── Kill orphaned ffmpeg processes ────────────────────────────────
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq ffmpeg.exe" /FO CSV /NH 2^>nul') do (
    wmic process where "ProcessId=%%~a" get CommandLine /FORMAT:LIST 2>nul | findstr /I "dshow\|wasapi\|pipe" >nul 2>&1
    if !errorlevel! equ 0 (
        taskkill /F /PID %%~a >nul 2>&1
    )
)

REM Wait for processes to die
timeout /t 2 /nobreak >nul

REM ─── Verify stopped ───────────────────────────────────────────────
set "RETRIES=0"
:wait_loop
set "STILL_RUNNING=0"
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul') do (
    wmic process where "ProcessId=%%~a" get CommandLine /FORMAT:LIST 2>nul | findstr /I "speech_to_text" >nul 2>&1
    if !errorlevel! equ 0 (
        set "STILL_RUNNING=1"
        taskkill /F /PID %%~a >nul 2>&1
    )
)
if "!STILL_RUNNING!"=="1" (
    set /a RETRIES+=1
    if !RETRIES! lss 10 (
        timeout /t 1 /nobreak >nul
        goto wait_loop
    ) else (
        echo [WARNING] Could not stop all processes after 10 attempts
    )
)

echo [RESTART] All server processes stopped.
timeout /t 2 /nobreak >nul

REM ─── Start server ─────────────────────────────────────────────────
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo [RESTART] Starting server...
start "" "!PYTHON_BIN!" speech_to_text.py

REM ─── Verify started ───────────────────────────────────────────────
timeout /t 3 /nobreak >nul
tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>nul | findstr /I "python" >nul 2>&1
if %errorlevel% equ 0 (
    echo [RESTART] Server started successfully.
) else (
    echo [RESTART] WARNING: Server may not have started. Check for errors.
)
