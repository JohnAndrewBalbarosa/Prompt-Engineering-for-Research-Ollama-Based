param(
    [string]$ConfigPath = "config/experiment.json",
    [string]$ModelName = "llama3.1:8b",
    [switch]$SkipRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Test-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-PythonCommand {
    if (Test-CommandExists "py") {
        return @("py", "-3")
    }
    if (Test-CommandExists "python") {
        return @("python")
    }
    throw "Python was not found. Install Python 3.10+ and re-run this script."
}

function Get-VenvPythonPath {
    $candidates = @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\bin\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "Virtual environment python executable not found after setup."
}

function Ensure-Venv {
    if (-not (Test-Path (Join-Path $RepoRoot ".venv"))) {
        Write-Host "[setup] Creating virtual environment..."
        $pythonCmd = Get-PythonCommand
        if ($pythonCmd.Length -gt 1) {
            & $pythonCmd[0] $pythonCmd[1..($pythonCmd.Length - 1)] -m venv .venv
        }
        else {
            & $pythonCmd[0] -m venv .venv
        }
    }
}

function Ensure-Dependencies {
    $venvPython = Get-VenvPythonPath
    & $venvPython -m pip install --upgrade pip | Out-Null

    $requirementsPath = Join-Path $RepoRoot "requirements.txt"
    if (Test-Path $requirementsPath) {
        $effectiveLines = Get-Content $requirementsPath | Where-Object {
            $line = $_.Trim()
            $line -and -not $line.StartsWith("#")
        }
        if ($effectiveLines.Count -gt 0) {
            Write-Host "[setup] Installing python dependencies..."
            & $venvPython -m pip install -r $requirementsPath
        }
    }
}

function Ensure-EnvFile {
    $envPath = Join-Path $RepoRoot ".env"
    $examplePath = Join-Path $RepoRoot ".env.example"

    if (-not (Test-Path $envPath) -and (Test-Path $examplePath)) {
        Copy-Item $examplePath $envPath
        Write-Host "[setup] Created .env from .env.example"
    }
}

function Ensure-OllamaInstalled {
    if (Test-CommandExists "ollama") {
        return
    }

    Write-Host "[setup] Ollama not found. Attempting installation via winget..."
    if (-not (Test-CommandExists "winget")) {
        throw "winget is not available. Install Ollama manually from https://ollama.com/download/windows and re-run this script."
    }

    winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements

    if (-not (Test-CommandExists "ollama")) {
        throw "Ollama installation completed but command is still unavailable. Restart terminal and re-run this script."
    }
}

function Ensure-OllamaRunning {
    try {
        ollama list | Out-Null
        return
    }
    catch {
        Write-Host "[setup] Starting Ollama server..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden | Out-Null
        Start-Sleep -Seconds 3
        ollama list | Out-Null
    }
}

function Pull-Model {
    param([Parameter(Mandatory = $true)][string]$Name)
    Write-Host "[setup] Ensuring model '$Name' is available locally..."
    ollama pull $Name
}

Write-Host "[setup] Repository root: $RepoRoot"
Ensure-Venv
Ensure-Dependencies
Ensure-EnvFile
Ensure-OllamaInstalled
Ensure-OllamaRunning
Pull-Model -Name $ModelName

$env:EXPERIMENT_CONFIG_PATH = $ConfigPath
if (-not $env:OLLAMA_BASE_URL) {
    $env:OLLAMA_BASE_URL = "http://localhost:11434"
}

if ($SkipRun) {
    $venvPythonForMessage = Get-VenvPythonPath
    Write-Host "[done] Setup completed. Run manually with:"
    Write-Host "       $venvPythonForMessage -m src.main --config $ConfigPath"
    exit 0
}

$venvPython = Get-VenvPythonPath
Write-Host "[run] Starting experiment with config '$ConfigPath'..."
& $venvPython -m src.main --config $ConfigPath
