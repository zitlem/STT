@echo off
REM Speech-to-Text Stop Script (Windows)

cd /d "%~dp0"

echo Stopping Speech-to-Text server...

REM Kill python processes running speech_to_text.py
set "FOUND=0"
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul') do (
    set "PID=%%~a"
    wmic process where "ProcessId=%%~a" get CommandLine /FORMAT:LIST 2>nul | findstr /I "speech_to_text" >nul 2>&1
    if !errorlevel! equ 0 (
        taskkill /F /PID %%~a >nul 2>&1
        set "FOUND=1"
    )
)

REM Also kill by window title if started via start_server.bat
taskkill /F /FI "WINDOWTITLE eq speech_to_text*" >nul 2>&1

REM Kill orphaned ffmpeg processes related to audio capture
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq ffmpeg.exe" /FO CSV /NH 2^>nul') do (
    wmic process where "ProcessId=%%~a" get CommandLine /FORMAT:LIST 2>nul | findstr /I "dshow\|wasapi\|pipe" >nul 2>&1
    if !errorlevel! equ 0 (
        taskkill /F /PID %%~a >nul 2>&1
    )
)

echo [OK] Server stopped.
