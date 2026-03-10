param(
    [string]$VenvDir = "venv",
    [string]$OutputDir = "",
    [string]$LogDir = "logs",
    [int]$Limit = 0,
    [switch]$VerboseDownload
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment not found at $VenvDir. Run .\scripts\setup_local.ps1 first."
}

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $LogDir "weekly-corpus-update-$timestamp.log"

$argsList = @("scripts/download_corpus.py", "--update")
if ($OutputDir -and $OutputDir.Trim()) {
    $argsList += @("--output-dir", $OutputDir)
}
if ($Limit -gt 0) {
    $argsList += @("--limit", "$Limit")
}
if ($VerboseDownload) {
    $argsList += "--verbose"
}

$cmdPreview = "$VenvPython " + ($argsList -join " ")
Write-Host "Starting weekly corpus update..."
Write-Host "Project root: $ProjectRoot"
Write-Host "Log: $logPath"
Write-Host "Command: $cmdPreview"

"[$(Get-Date -Format s)] START weekly corpus update" | Out-File -FilePath $logPath -Encoding utf8
"[$(Get-Date -Format s)] Command: $cmdPreview" | Out-File -FilePath $logPath -Append -Encoding utf8

& $VenvPython @argsList *>&1 | Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE

"[$(Get-Date -Format s)] EXIT code=$exitCode" | Out-File -FilePath $logPath -Append -Encoding utf8

if ($exitCode -ne 0) {
    throw "Weekly corpus update failed with exit code $exitCode. See log: $logPath"
}

Write-Host "Weekly corpus update completed successfully."
Write-Host "Log saved to: $logPath"
