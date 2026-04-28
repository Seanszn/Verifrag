param(
    [string]$VenvDir = "venv",
    [string]$NliModel = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    [string]$NliDevice = "cpu",
    [switch]$DownloadNliModel,
    [switch]$OfflineModels,
    [switch]$SkipVerification
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
if (-not $env:VERIFICATION_VERIFIER_MODE) { $env:VERIFICATION_VERIFIER_MODE = "live" }
if (-not $env:VERIFICATION_FALLBACK_TO_HEURISTIC) { $env:VERIFICATION_FALLBACK_TO_HEURISTIC = "true" }
if (-not $env:APP_LOG_LEVEL) { $env:APP_LOG_LEVEL = "INFO" }

$env:NLI_MODEL = $NliModel
$env:NLI_DEVICE = $NliDevice

if ($SkipVerification) {
    $env:SKIP_NLI_VERIFICATION = "true"
}

if ($OfflineModels) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:HF_LOCAL_FILES_ONLY = "1"
} else {
    if (-not $env:HF_HUB_OFFLINE) { $env:HF_HUB_OFFLINE = "0" }
    if (-not $env:TRANSFORMERS_OFFLINE) { $env:TRANSFORMERS_OFFLINE = "0" }
    if (-not $env:HF_LOCAL_FILES_ONLY) { $env:HF_LOCAL_FILES_ONLY = "0" }
}

if (-not (Test-Path ".env")) {
    Write-Warning ".env not found. Run .\scripts\setup_local.ps1 first to create a local default .env."
}

if ($DownloadNliModel) {
    Write-Host "Downloading NLI model: $env:NLI_MODEL"
    & $VenvPython -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; import os; model=os.environ['NLI_MODEL']; AutoTokenizer.from_pretrained(model); AutoModelForSequenceClassification.from_pretrained(model); print('NLI model cache ready:', model)"
}

if ($OfflineModels) {
    Write-Host "Checking if NLI model is cached locally: $env:NLI_MODEL"
    
    # Fast check using huggingface_hub CLI instead of importing heavy libraries
    $modelName = $env:NLI_MODEL
    $cacheDir = & $VenvPython -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)" 2>$null
    
    if (-not $cacheDir) {
        # Fallback: common cache locations
        $possibleCacheDirs = @(
            "$env:USERPROFILE\.cache\huggingface\hub",
            "$env:LOCALAPPDATA\huggingface\hub"
        )
        foreach ($dir in $possibleCacheDirs) {
            if (Test-Path $dir) {
                $cacheDir = $dir
                break
            }
        }
    }
    
    # Convert model name to cache folder format (replace / with --)
    $modelCacheFolder = $modelName -replace "/", "--"
    $modelCached = $false
    
    if ($cacheDir -and (Test-Path $cacheDir)) {
        # Look for model snapshots in cache
        $modelFolders = Get-ChildItem -Path $cacheDir -Filter "models--$modelCacheFolder" -Directory -ErrorAction SilentlyContinue
        if ($modelFolders) {
            foreach ($folder in $modelFolders) {
                $snapshotsDir = Join-Path $folder.FullName "snapshots"
                if (Test-Path $snapshotsDir) {
                    $snapshots = Get-ChildItem -Path $snapshotsDir -Directory -ErrorAction SilentlyContinue
                    if ($snapshots) {
                        $modelCached = $true
                        break
                    }
                }
            }
        }
    }
    
    if (-not $modelCached) {
        Write-Host "Model not found in local cache. Downloading now..." -ForegroundColor Yellow
        # Temporarily disable offline mode for download
        $env:HF_HUB_OFFLINE = "0"
        $env:TRANSFORMERS_OFFLINE = "0"
        $env:HF_LOCAL_FILES_ONLY = "0"
        
        & $VenvPython -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; import os; model=os.environ['NLI_MODEL']; AutoTokenizer.from_pretrained(model); AutoModelForSequenceClassification.from_pretrained(model); print('NLI model downloaded:', model)"
        
        Write-Host "Model downloaded successfully. Re-enabling offline mode." -ForegroundColor Green
        # Re-enable offline mode
        $env:HF_HUB_OFFLINE = "1"
        $env:TRANSFORMERS_OFFLINE = "1"
        $env:HF_LOCAL_FILES_ONLY = "1"
    } else {
        Write-Host "Model found in local cache." -ForegroundColor Green
    }
    
    # Skip verification if explicitly disabled (useful for faster startup)
    if ($env:SKIP_NLI_VERIFICATION -eq "true") {
        Write-Host "Skipping NLI model verification (SKIP_NLI_VERIFICATION=true). Will verify on first use." -ForegroundColor Yellow
    } else {
        Write-Host "Verifying offline NLI model cache: $env:NLI_MODEL" -NoNewline
        Write-Host " (This may take 30-60 seconds on first load...)" -ForegroundColor DarkGray
        
        # Use a Python script with progress output
        $verifyScript = @"
import sys
import os
model = os.environ['NLI_MODEL']

sys.stdout.write("  Loading tokenizer...")
sys.stdout.flush()
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
sys.stdout.write(" Done!\n")

sys.stdout.write("  Loading model (this is the slow part)...")
sys.stdout.flush()
model_obj = AutoModelForSequenceClassification.from_pretrained(model, local_files_only=True)
sys.stdout.write(" Done!\n")

print("Offline NLI model cache ready:", model)
"@
        
        & $VenvPython -c $verifyScript
    }
}

Write-Host "Starting FastAPI in local mode..."
Write-Host "DEPLOYMENT_MODE=$env:DEPLOYMENT_MODE API_HOST=$env:API_HOST API_PORT=$env:API_PORT ENABLE_VERIFICATION=$env:ENABLE_VERIFICATION VERIFICATION_VERIFIER_MODE=$env:VERIFICATION_VERIFIER_MODE NLI_MODEL=$env:NLI_MODEL NLI_DEVICE=$env:NLI_DEVICE APP_LOG_LEVEL=$env:APP_LOG_LEVEL"

& $VenvPython -m uvicorn src.api.main:app --host $env:API_HOST --port $env:API_PORT --log-level $env:APP_LOG_LEVEL.ToLower()
