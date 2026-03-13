param(
    [string]$ConfigPath = "config/experiment.json",
    [string]$ModelName = "",
    [string[]]$ModelNames = @(),
    [string[]]$MainArgs = @(),
    [int]$NumTests = 100,
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
        $effectiveLines = @(Get-Content $requirementsPath | Where-Object {
            $line = $_.Trim()
            $line -and -not $line.StartsWith("#")
        })
        if (@($effectiveLines).Count -gt 0) {
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

function Get-OllamaExecutable {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Ensure-OllamaInstalled {
    $ollamaExe = Get-OllamaExecutable
    if ($ollamaExe) {
        return
    }

    throw "Ollama executable was not found. Install it manually, then re-run this script. Recommended command: irm https://ollama.com/install.ps1 | iex"
}

function Ensure-OllamaRunning {
    $ollamaExe = Get-OllamaExecutable
    if (-not $ollamaExe) {
        throw "Ollama executable was not found. Install Ollama from https://ollama.com/download/windows."
    }

    try {
        & $ollamaExe list | Out-Null
        return
    }
    catch {
        if (-not $env:OLLAMA_NUM_GPU) {
            # Prefer GPU offload by default; users can override via env var.
            $env:OLLAMA_NUM_GPU = "999"
        }
        Write-Host "[setup] Starting Ollama server..."
        Write-Host "[setup] OLLAMA_NUM_GPU=$($env:OLLAMA_NUM_GPU)"
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden | Out-Null
        Start-Sleep -Seconds 3
        & $ollamaExe list | Out-Null
    }
}

function Get-ModelsFromConfig {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Config file not found: $Path"
    }

    $raw = Get-Content -Path $Path -Raw
    $config = $raw | ConvertFrom-Json
    $models = @()

    if ($null -ne $config.models) {
        foreach ($model in $config.models) {
            if ($null -eq $model.provider -or $null -eq $model.model_name) {
                continue
            }
            if ($model.provider.ToString().Trim().ToLowerInvariant() -eq "ollama") {
                $candidate = $model.model_name.ToString().Trim()
                if ($candidate) {
                    $models += $candidate
                }
            }
        }
    }

    if ($null -ne $config.judge -and $config.judge.enabled -eq $true) {
        $judgeProvider = ""
        if ($null -ne $config.judge.provider) {
            $judgeProvider = $config.judge.provider.ToString().Trim().ToLowerInvariant()
        }
        if ($judgeProvider -eq "ollama" -and $null -ne $config.judge.model) {
            $judgeModel = $config.judge.model.ToString().Trim()
            if ($judgeModel) {
                $models += $judgeModel
            }
        }
    }

    return $models
}

function Resolve-TargetModels {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$Single,
        [string[]]$Multiple
    )

    $resolved = @()
    $resolved += Get-ModelsFromConfig -Path $Path

    if ($Single) {
        $resolved += $Single.Trim()
    }

    if ($null -ne $Multiple) {
        foreach ($name in $Multiple) {
            if ($name) {
                $trimmed = $name.Trim()
                if ($trimmed) {
                    $resolved += $trimmed
                }
            }
        }
    }

    $distinct = @{}
    foreach ($name in $resolved) {
        if ($name) {
            $distinct[$name] = $true
        }
    }

    return @($distinct.Keys | Sort-Object)
}

function Pull-Model {
    param([Parameter(Mandatory = $true)][string]$Name)
    $ollamaExe = Get-OllamaExecutable
    if (-not $ollamaExe) {
        throw "Ollama executable was not found. Cannot pull model '$Name'."
    }
    Write-Host "[setup] Ensuring model '$Name' is available locally..."
    & $ollamaExe pull $Name
}

function Wait-BeforeExit {
    Write-Host ""
    Read-Host "Press Enter to exit"
}

function Ensure-DefaultNumTestsArg {
    param(
        [AllowEmptyCollection()][string[]]$Args = @(),
        [Parameter(Mandatory = $true)][int]$DefaultNumTests
    )

    foreach ($arg in $Args) {
        if ($arg -match "^--(sample-size|num-tests)(=.+)?$") {
            return $Args
        }
    }

    return @("--num-tests", $DefaultNumTests.ToString()) + $Args
}

$exitCode = 0

try {
    Write-Host "[setup] Repository root: $RepoRoot"
    Ensure-Venv
    Ensure-Dependencies
    Ensure-EnvFile
    Ensure-OllamaInstalled
    Ensure-OllamaRunning

    $targetModels = @(Resolve-TargetModels -Path $ConfigPath -Single $ModelName -Multiple $ModelNames)
    if (@($targetModels).Count -eq 0) {
        throw "No Ollama models resolved. Add models to '$ConfigPath' or pass -ModelName / -ModelNames."
    }

    Write-Host "[setup] Models to ensure: $($targetModels -join ', ')"
    foreach ($targetModel in $targetModels) {
        Pull-Model -Name $targetModel
    }

    $env:EXPERIMENT_CONFIG_PATH = $ConfigPath
    if (-not $env:OLLAMA_BASE_URL) {
        $env:OLLAMA_BASE_URL = "http://localhost:11434"
    }

    if ($SkipRun) {
        $venvPythonForMessage = Get-VenvPythonPath
        $effectiveMainArgs = @(Ensure-DefaultNumTestsArg -Args $MainArgs -DefaultNumTests $NumTests)
        Write-Host "[done] Setup completed. Run manually with:"
        if (@($effectiveMainArgs).Count -gt 0) {
            Write-Host "       $venvPythonForMessage -m src.main --config $ConfigPath $($effectiveMainArgs -join ' ')"
        }
        else {
            Write-Host "       $venvPythonForMessage -m src.main --config $ConfigPath"
        }
    }
    else {
        $venvPython = Get-VenvPythonPath
        $effectiveMainArgs = @(Ensure-DefaultNumTestsArg -Args $MainArgs -DefaultNumTests $NumTests)
        Write-Host "[run] Starting experiment with config '$ConfigPath'..."
        & $venvPython -m src.main --config $ConfigPath @effectiveMainArgs
    }
}
catch {
    $exitCode = 1
    Write-Host ""
    Write-Host "[error] Setup/run failed."
    Write-Host "[error] $($_.Exception.Message)"
    if ($_.InvocationInfo -and $_.InvocationInfo.ScriptLineNumber) {
        Write-Host "[error] At line $($_.InvocationInfo.ScriptLineNumber)"
    }
}
finally {
    Wait-BeforeExit
}

exit $exitCode
