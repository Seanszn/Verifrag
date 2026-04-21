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
if (-not $env:API_HOST) { $env:API_HOST = "127.0.0.1" }
if (-not $env:API_PORT) { $env:API_PORT = "8000" }
if (-not $env:ENABLE_VERIFICATION) { $env:ENABLE_VERIFICATION = "true" }
if (-not $env:APP_LOG_LEVEL) { $env:APP_LOG_LEVEL = "INFO" }

if (-not (Test-Path ".env")) {
    Write-Warning ".env not found. Run .\scripts\setup_local.ps1 first to create a local default .env."
}

Write-Host "Starting FastAPI in local mode..."
Write-Host "DEPLOYMENT_MODE=$env:DEPLOYMENT_MODE API_HOST=$env:API_HOST API_PORT=$env:API_PORT ENABLE_VERIFICATION=$env:ENABLE_VERIFICATION APP_LOG_LEVEL=$env:APP_LOG_LEVEL"

& $VenvPython -m uvicorn src.api.main:app --host $env:API_HOST --port $env:API_PORT --log-level $env:APP_LOG_LEVEL.ToLower()
