param(
    [string]$VenvDir = "venv"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $FallbackPython = Resolve-Path (Join-Path $ProjectRoot "..\$VenvDir\Scripts\python.exe") -ErrorAction SilentlyContinue
    if ($FallbackPython) {
        $VenvPython = $FallbackPython.Path
    } else {
        throw "Virtual environment not found at $VenvDir or ..\$VenvDir. Run .\scripts\setup_local.ps1 first."
    }
}

$env:DEPLOYMENT_MODE = "local"

Write-Host "Starting FastAPI backend..."
Write-Host "API at http://127.0.0.1:8000"

& $VenvPython -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
