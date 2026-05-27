# Full pre-commit / pre-tag check pipeline. Bails on the first failure.
#
# Usage:
#     .\scripts\check.ps1
#     .\scripts\check.ps1 -SkipDocs       # skip mkdocs build (which doesn't exist until v0.7)
#     .\scripts\check.ps1 -SkipTests      # skip pytest

[CmdletBinding()]
param(
    [switch]$SkipDocs,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$startTime = Get-Date

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "FAIL: $Label (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Step 'ruff check --fix' { uv run ruff check . --fix }
Invoke-Step 'ruff format'      { uv run ruff format . }
Invoke-Step 'mypy strict'      { uv run mypy src }

if (-not $SkipTests) {
    Invoke-Step 'pytest' { uv run pytest }
}

# mkdocs doesn't land until v0.7 - skip by default until then.
# Use .\scripts\check.ps1 (no -SkipDocs flag) once mkdocs.yml exists.
if (-not $SkipDocs -and (Test-Path .\mkdocs.yml)) {
    Invoke-Step 'mkdocs build --strict' { uv run mkdocs build --strict }
}

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "==> All checks passed in $([int]$elapsed.TotalSeconds)s." -ForegroundColor Green
