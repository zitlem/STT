@echo off
REM Speech-to-Text Installation Script for Windows
REM Launches the PowerShell installer

echo ========================================
echo   Speech-to-Text Windows Installer
echo ========================================
echo.

REM Check if running as admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Not running as Administrator.
    echo [WARNING] Some features (port 80, ffmpeg install) may require admin rights.
    echo [WARNING] Right-click this file and select "Run as administrator" if needed.
    echo.
)

REM Launch the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"

if %errorLevel% neq 0 (
    echo.
    echo [ERROR] Installation failed. Check the output above for details.
    pause
    exit /b 1
)

pause
