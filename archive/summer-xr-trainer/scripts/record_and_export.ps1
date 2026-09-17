param(
    [int]$Duration = 10,
    [double]$Hz = 5
)

$ErrorActionPreference = "Stop"

# Move to repo root regardless of where the script is launched from.
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# Activate venv if it isn't already active in this shell.
if (-not $env:VIRTUAL_ENV) {
    & ".\.venv\Scripts\Activate.ps1"
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$jsonl = "recordings\realglove_$stamp.jsonl"
$csv   = "recordings\realglove_$stamp.csv"

Write-Host "`n=== 1/3  Recording $Duration s live at $Hz Hz (3D viewer auto-closes when done) ===" -ForegroundColor Cyan
python scripts/run_osc.py --record $jsonl --duration $Duration --hz $Hz
if ($LASTEXITCODE -ne 0) { throw "recording failed" }

Write-Host "`n=== 2/3  Playing back (close the 3D window to continue) ===" -ForegroundColor Cyan
python scripts/playback.py $jsonl
if ($LASTEXITCODE -ne 0) { throw "playback failed" }

Write-Host "`n=== 3/3  Exporting to CSV ===" -ForegroundColor Cyan
python scripts/export.py $jsonl --output $csv
if ($LASTEXITCODE -ne 0) { throw "export failed" }

Write-Host "`nAll done." -ForegroundColor Green
Write-Host "  recording: $jsonl"
Write-Host "  csv:       $csv"
