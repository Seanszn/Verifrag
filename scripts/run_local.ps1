param(
    [string]$VenvDir = "venv"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment not found at $VenvDir. Run .\scripts\setup_local.ps1 first."
}

$env:DEPLOYMENT_MODE = "local"

if (-not (Test-Path ".env")) {
    Write-Warning ".env not found. Run .\scripts\setup_local.ps1 first to create a local default .env."
}

Write-Host "Starting Streamlit in local mode..."
Write-Host "DEPLOYMENT_MODE=$env:DEPLOYMENT_MODE LLM_MODEL=$env:LLM_MODEL OLLAMA_HOST=$env:OLLAMA_HOST"

& $VenvPython -m streamlit run src/app.py
