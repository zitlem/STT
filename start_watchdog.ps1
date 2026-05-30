# Start the STT Watchdog in headless mode (PowerShell / Windows).
# The watchdog manages speech_to_text.py: starts it, restarts on crash,
# and checks for GitHub updates daily at 1am.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonBin = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonBin)) { $PythonBin = "python" }

$LogDir = Join-Path $ScriptDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$proc = Start-Process `
    -FilePath $PythonBin `
    -ArgumentList "`"$(Join-Path $ScriptDir 'watchdog.py')`" --headless" `
    -WorkingDirectory $ScriptDir `
    -WindowStyle Minimized `
    -PassThru

Write-Host "[OK] Watchdog started (PID $($proc.Id))"
Write-Host "     Logs: $(Join-Path $LogDir 'watchdog.log')"
Write-Host "     STT:  $(Join-Path $LogDir 'stt.log')"
