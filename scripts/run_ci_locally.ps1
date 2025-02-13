# Ensure errors stop the script
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to repo root (script is in scripts/)
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==> Using Python:" -ForegroundColor Cyan
python --version

Write-Host "==> Upgrading pip and installing dependencies..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install flake8

# Mirror CI environment
if (-not $env:OPENAI_API_KEY) {
    # Set a dummy key so anything requiring it won't crash locally
    $env:OPENAI_API_KEY = "dummy_local_key"
}
$env:PYTHONPATH = "src"

Write-Host "==> Running flake8 (strict)..." -ForegroundColor Cyan
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

Write-Host "==> Running flake8 (stats-only)..." -ForegroundColor Cyan
python -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

Write-Host "==> Running pytest..." -ForegroundColor Cyan
python -m pytest
