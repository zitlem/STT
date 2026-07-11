# Start the STT Watchdog in headless mode (PowerShell / Windows).
# Works with both the compiled binary and Python source installs.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Binary    = Join-Path $ScriptDir "STT-Watchdog.exe"

if (Test-Path $Binary) {
    # ── Compiled binary ──────────────────────────────────────────────────────
    $LogDir = Join-Path $env:USERPROFILE ".stt\logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $proc = Start-Process -FilePath $Binary -ArgumentList "--headless" `
        -WindowStyle Minimized -PassThru
    Write-Host "[OK] Watchdog started (PID $($proc.Id)) — binary mode"
    Write-Host "     Logs: $LogDir\watchdog.log"
} else {
    # ── Python source fallback ────────────────────────────────────────────────
    $PythonBin = Join-Path $ScriptDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $PythonBin)) { $PythonBin = "python" }
    $LogDir = Join-Path $ScriptDir "logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $proc = Start-Process -FilePath $PythonBin `
        -ArgumentList "`"$(Join-Path $ScriptDir 'stt/watchdog.py')`" --headless" `
        -WorkingDirectory $ScriptDir -WindowStyle Minimized -PassThru
    Write-Host "[OK] Watchdog started (PID $($proc.Id)) — Python source mode"
    Write-Host "     Logs: $LogDir\watchdog.log"
}
