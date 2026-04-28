param(
    [string]$PythonExe = "python",
    [string]$VenvDir = "venv",
    [switch]$SkipOllamaPull,
    [switch]$SkipSpacyModel
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

Write-Host "Project root: $ProjectRoot"

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment: $VenvDir"
    & $PythonExe -m venv $VenvDir
}

$VenvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python not found at $VenvPython"
}

Write-Host "Upgrading pip"
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing requirements"
& $VenvPython -m pip install -r requirements.txt

if (-not $SkipSpacyModel) {
    Write-Host "Installing spaCy model: en_core_web_sm"
    & $VenvPython -m spacy download en_core_web_sm
}

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example"
    Copy-Item ".env.example" ".env"
}

if (Test-Path ".env") {
    $envText = Get-Content ".env" -Raw
    $envText = [regex]::Replace($envText, "(?m)^LLM_PROVIDER=.*(\r?\n)?", "")
    $envText = [regex]::Replace($envText, "(?m)^VECTOR_STORE=.*(\r?\n)?", "")

    if ($envText -match "(?m)^DEPLOYMENT_MODE=") {
        $envText = [regex]::Replace($envText, "(?m)^DEPLOYMENT_MODE=.*$", "DEPLOYMENT_MODE=local")
    } else {
        $envText += "`r`nDEPLOYMENT_MODE=local`r`n"
    }

    if ($envText -match "(?m)^ENABLE_VERIFICATION=") {
        $envText = [regex]::Replace($envText, "(?m)^ENABLE_VERIFICATION=.*$", "ENABLE_VERIFICATION=true")
    } else {
        $envText += "`r`nENABLE_VERIFICATION=true`r`n"
    }

    Set-Content ".env" $envText -NoNewline
}

@("data", "data\raw", "data\processed", "data\index", "data\eval") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
    }
}

if (-not $SkipOllamaPull) {
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaCmd) {
        Write-Host "Pulling Ollama model: llama3.2:3b"
        & ollama pull llama3.2:3b
    } else {
        Write-Warning "ollama not found. Install Ollama and run: ollama pull llama3.2:3b"
    }
}

Write-Host ""
Write-Host "Local setup complete."
Write-Host "Verification is enabled by default for the full claim analysis pipeline."
Write-Host "Run: .\scripts\run_local.ps1"
