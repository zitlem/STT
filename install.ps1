# Speech-to-Text Installation Script for Windows
# Installs Python, ffmpeg, creates venv, and installs all dependencies

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Configuration
$INSTALL_DIR = $PSScriptRoot
$SERVICE_NAME = "stt-server"
$PYTHON_BIN = ""
$VENV_DIR = Join-Path $INSTALL_DIR ".venv"

# ─── Output helpers ──────────────────────────────────────────────────
function Print-Status  { param($msg) Write-Host "[INFO] "    -ForegroundColor Blue   -NoNewline; Write-Host $msg }
function Print-Success { param($msg) Write-Host "[OK] "      -ForegroundColor Green  -NoNewline; Write-Host $msg }
function Print-Error   { param($msg) Write-Host "[ERROR] "   -ForegroundColor Red    -NoNewline; Write-Host $msg }
function Print-Warning { param($msg) Write-Host "[WARNING] " -ForegroundColor Yellow -NoNewline; Write-Host $msg }

# ─── Check admin ─────────────────────────────────────────────────────
function Test-Admin {
    $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ─── Refresh PATH from registry (picks up new installs without restart) ──
function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path    = "$machinePath;$userPath"
}

# ─── Check if a command exists ───────────────────────────────────────
function Test-Command {
    param($Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# ─── Install winget packages ────────────────────────────────────────
function Install-SystemDeps {
    Print-Status "Checking system dependencies..."

    # --- Python ---
    if (Test-Command "python") {
        $pyVer = & python --version 2>&1
        Print-Success "Python already installed: $pyVer"
    } else {
        Print-Status "Installing Python via winget..."
        if (Test-Command "winget") {
            winget install --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
            Refresh-Path
        } else {
            Print-Error "Python not found and winget is not available."
            Print-Error "Please install Python 3.10+ from https://www.python.org/downloads/"
            Print-Error "IMPORTANT: Check 'Add Python to PATH' during installation."
            exit 1
        }
    }

    # Verify Python
    if (-not (Test-Command "python")) {
        Refresh-Path
        if (-not (Test-Command "python")) {
            Print-Error "Python not found in PATH after installation."
            Print-Error "Please install Python manually and ensure 'Add Python to PATH' is checked."
            exit 1
        }
    }

    # --- Git ---
    if (Test-Command "git") {
        Print-Success "Git already installed"
    } else {
        Print-Status "Installing Git via winget..."
        if (Test-Command "winget") {
            winget install --id Git.Git --accept-source-agreements --accept-package-agreements
            Refresh-Path
        } else {
            Print-Warning "Git not found. Install from https://git-scm.com/download/win"
            Print-Warning "Git is required for installing the OpenAI Whisper package."
        }
    }

    # --- FFmpeg ---
    if (Test-Command "ffmpeg") {
        Print-Success "FFmpeg already installed"
    } else {
        Print-Status "Installing FFmpeg via winget..."
        if (Test-Command "winget") {
            winget install --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
            Refresh-Path
        }

        if (-not (Test-Command "ffmpeg")) {
            Print-Warning "FFmpeg not found in PATH after install attempt."
            Print-Warning "You may need to add FFmpeg to your PATH manually or restart your terminal."
            Print-Warning "Download from: https://www.gyan.dev/ffmpeg/builds/"
        }
    }

    Print-Success "System dependency check complete"
}

# ─── Install uv ──────────────────────────────────────────────────────
function Install-Uv {
    Print-Status "Installing uv package manager..."

    if (Test-Command "uv") {
        $uvVer = & uv --version 2>&1
        Print-Success "uv already installed: $uvVer"
        return
    }

    Print-Status "Downloading uv installer..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    Refresh-Path

    # Also check common install location
    $uvPath = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path $uvPath) {
        $env:Path = "$uvPath;$env:Path"
    }
    # Also check cargo bin (alternative install location)
    $cargoPath = Join-Path $env:USERPROFILE ".cargo\bin"
    if (Test-Path $cargoPath) {
        $env:Path = "$cargoPath;$env:Path"
    }

    if (Test-Command "uv") {
        $uvVer = & uv --version 2>&1
        Print-Success "uv installed: $uvVer"
    } else {
        Print-Error "uv installation failed. Falling back to pip."
    }
}

# ─── Create virtual environment ──────────────────────────────────────
function Create-Venv {
    Print-Status "Creating Python virtual environment..."

    if (Test-Command "uv") {
        & uv venv $VENV_DIR
    } else {
        & python -m venv $VENV_DIR
    }

    $script:PYTHON_BIN = Join-Path $VENV_DIR "Scripts\python.exe"

    if (-not (Test-Path $PYTHON_BIN)) {
        Print-Error "Failed to create virtual environment"
        exit 1
    }

    Print-Success "Virtual environment created at $VENV_DIR"
}

# ─── Detect GPU ──────────────────────────────────────────────────────
function Detect-Gpu {
    Print-Status "Detecting GPU..."

    if (Test-Command "nvidia-smi") {
        Print-Success "NVIDIA GPU detected:"
        & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null

        $cudaLine = & nvidia-smi 2>$null | Select-String "CUDA Version"
        if ($cudaLine) {
            $cudaVer = ($cudaLine -split '\s+' | Where-Object { $_ -match '^\d+\.\d+' } | Select-Object -Last 1)
            Print-Status "CUDA Version: $cudaVer"
        }
        return $true
    } else {
        Print-Warning "No NVIDIA GPU detected (nvidia-smi not found)"
        Print-Warning "Will install CPU-only PyTorch"
        return $false
    }
}

# ─── Install Python packages ────────────────────────────────────────
function Install-PythonDeps {
    Print-Status "Installing Python dependencies from requirements.txt..."

    $reqFile = Join-Path $INSTALL_DIR "requirements.txt"
    $hasGpu  = Detect-Gpu

    if (Test-Command "uv") {
        if ($hasGpu) {
            Print-Status "Installing GPU-enabled packages (CUDA)..."
            & uv pip install --python $PYTHON_BIN -r $reqFile --extra-index-url https://download.pytorch.org/whl/cu121
        } else {
            Print-Status "Installing CPU-only packages..."
            & uv pip install --python $PYTHON_BIN -r $reqFile
        }
    } else {
        # Fallback to pip
        & $PYTHON_BIN -m pip install --upgrade pip
        if ($hasGpu) {
            Print-Status "Installing GPU-enabled packages (CUDA)..."
            & $PYTHON_BIN -m pip install -r $reqFile --extra-index-url https://download.pytorch.org/whl/cu121
        } else {
            Print-Status "Installing CPU-only packages..."
            & $PYTHON_BIN -m pip install -r $reqFile
        }
    }

    if ($LASTEXITCODE -ne 0) {
        Print-Error "Some packages failed to install. Check the output above."
        Print-Warning "You can try installing manually: .venv\Scripts\pip install -r requirements.txt"
    } else {
        Print-Success "Python dependencies installed"
    }
}

# ─── Verify installation ────────────────────────────────────────────
function Verify-PythonDeps {
    Print-Status "Verifying Python installation..."

    $verifyScript = @"
import sys
errors = []
try:
    import torch
    print(f'  PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('  GPU not available (CPU mode)')
except ImportError as e:
    errors.append(str(e))

for mod in ['speech_recognition', 'numpy', 'flask', 'soundfile']:
    try:
        __import__(mod)
        print(f'  {mod} OK')
    except ImportError as e:
        errors.append(str(e))

try:
    import pyaudio
    print('  pyaudio OK')
except ImportError:
    print('  pyaudio MISSING - microphone input will not work')
    print('  Try: .venv\\Scripts\\pip install pyaudio')

if errors:
    print(f'\nMissing packages: {errors}')
    sys.exit(1)
else:
    print('\n  All critical packages verified!')
"@

    & $PYTHON_BIN -c $verifyScript

    if ($LASTEXITCODE -eq 0) {
        Print-Success "Python verification passed"
    } else {
        Print-Warning "Some packages could not be verified (see above)"
    }
}

# ─── Verify scripts exist ────────────────────────────────────────────
function Verify-Scripts {
    Print-Status "Checking server management scripts..."

    $scripts = @("start_server.bat", "stop_server.bat", "restart_server.bat")
    foreach ($s in $scripts) {
        $path = Join-Path $INSTALL_DIR $s
        if (Test-Path $path) {
            Print-Success "$s found"
        } else {
            Print-Warning "$s not found — download it from the repository"
        }
    }
}

# ─── Optional: Task Scheduler auto-start ─────────────────────────────
function Setup-TaskScheduler {
    Write-Host ""
    $reply = Read-Host "Would you like to set up auto-start on login via Task Scheduler? (y/N)"
    if ($reply -notmatch '^[Yy]$') { return }

    if (-not (Test-Admin)) {
        Print-Warning "Task Scheduler setup requires Administrator privileges."
        Print-Warning "Please re-run this script as Administrator to set up auto-start."
        return
    }

    $taskAction  = New-ScheduledTaskAction `
        -Execute (Join-Path $VENV_DIR "Scripts\python.exe") `
        -Argument "speech_to_text.py" `
        -WorkingDirectory $INSTALL_DIR

    $taskTrigger = New-ScheduledTaskTrigger -AtLogOn
    $taskSettings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask `
        -TaskName $SERVICE_NAME `
        -Action $taskAction `
        -Trigger $taskTrigger `
        -Settings $taskSettings `
        -Description "Speech-to-Text Server" `
        -RunLevel Highest `
        -Force

    Print-Success "Task Scheduler entry created: $SERVICE_NAME"
    Print-Status "The server will auto-start when you log in."
    Print-Status "Manage it in Task Scheduler (taskschd.msc) or with:"
    Print-Status "  schtasks /Run /TN $SERVICE_NAME    - Start now"
    Print-Status "  schtasks /End /TN $SERVICE_NAME    - Stop"
    Print-Status "  schtasks /Delete /TN $SERVICE_NAME - Remove"
}

# ─── Final instructions ─────────────────────────────────────────────
function Show-FinalInstructions {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  Installation Complete!"
    Write-Host "========================================"
    Write-Host ""
    Print-Success "Speech-to-Text has been installed successfully"
    Write-Host ""
    Write-Host "Quick Start:"
    Write-Host "  1. Run: start_server.bat"
    Write-Host "  2. Open browser: http://localhost:80"
    Write-Host "  3. Go to /model-manager to download models"
    Write-Host "  4. Go to /live-settings to configure audio"
    Write-Host ""
    Write-Host "Configuration:"
    Write-Host "  - Main config: $INSTALL_DIR\config.json"
    Write-Host ""
    Write-Host "Manual start (from this directory):"
    Write-Host "  .venv\Scripts\python.exe speech_to_text.py"
    Write-Host ""

    if (-not (Test-Command "ffmpeg")) {
        Print-Warning "FFmpeg was not found in PATH. You may need to:"
        Print-Warning "  1. Restart your terminal/PC after installation"
        Print-Warning "  2. Or install manually from https://www.gyan.dev/ffmpeg/builds/"
        Write-Host ""
    }
}

# ─── Main ────────────────────────────────────────────────────────────
function Main {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  Speech-to-Text Windows Installer"
    Write-Host "========================================"
    Write-Host ""

    Push-Location $INSTALL_DIR
    try {
        Install-SystemDeps
        Install-Uv
        Create-Venv
        Install-PythonDeps
        Verify-PythonDeps
        Verify-Scripts
        Setup-TaskScheduler
        Show-FinalInstructions
    } finally {
        Pop-Location
    }
}

# Run
Main
