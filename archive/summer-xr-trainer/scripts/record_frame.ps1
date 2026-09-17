param(
    [Parameter(Mandatory = $true)][string]$Frame,   # e.g. frame_128166
    [int]$Takes = 1,
    [int]$Duration = 5,
    [int]$Prep = 5,
    [switch]$BothHands,   # keep BOTH hands in the txt (default: only the prof's hand)
    [switch]$Mock
)

$ErrorActionPreference = "Stop"

# Repo root, regardless of where the script is launched from.
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
if (-not $env:VIRTUAL_ENV) { & ".\.venv\Scripts\Activate.ps1" }

# Pose folders live in "..\xr trainer poses\<frame>", one per professor image.
# Folders may carry progress suffixes (_DONE, _NA) or legacy hand suffixes.
$posesRoot = Join-Path (Split-Path -Parent (Get-Location)) "xr trainer poses"
$Frame = $Frame -replace "_(left|right|DONE|NA)$", ""   # accept suffixed ids too
$match = Get-ChildItem $posesRoot -Directory | Where-Object {
    $_.Name -eq $Frame -or $_.Name -match ('^' + [regex]::Escape($Frame) + '_(DONE|NA|left|right)$')
} | Select-Object -First 1
if (-not $match) { throw "no pose folder matching '$Frame' in $posesRoot" }
$folder = $match.FullName

# Team decision (2026-07): all poses are recorded with the LEFT hand,
# regardless of which hand the professor's reference frame used — so the
# export keeps whatever was recorded, no per-hand filtering.
$hand = $null

if ($BothHands) { $hand = $null }

$rec = Join-Path $folder "glove_recording"
$mockArg = @(); if ($Mock) { $mockArg = @("--mock") }
$handArg = @(); if ($hand) { $handArg = @("--hand", $hand) }
$handNote = if ($hand) { "using your $($hand.ToUpper()) hand" } elseif ($BothHands) { "keeping BOTH hands" } else { "any hand" }

Write-Host "`n=== 1/2  Recording '$Frame' ($Takes take(s) x $Duration s) - match $Frame.png $handNote ===" -ForegroundColor Cyan
python scripts/record_poses.py --poses $Frame --takes $Takes --duration $Duration --prep $Prep --out-dir $rec @mockArg
if ($LASTEXITCODE -ne 0) { throw "recording failed" }

Write-Host "`n=== 2/2  Exporting keypoints ($handNote) ===" -ForegroundColor Cyan
python scripts/export_prof_format.py $rec --out-dir $rec @handArg
if ($LASTEXITCODE -ne 0) { throw "export failed" }

# A recording must always yield a txt: if the professor's hand wasn't in the
# stream, export whatever WAS recorded and warn instead of leaving nothing.
$outFile = Join-Path $rec "$($Frame)_keypoints.txt"
if (-not (Test-Path $outFile) -and $hand) {
    Write-Host "`nWARNING: no $($hand.ToUpper())-hand frames in this recording, but the" -ForegroundColor Yellow
    Write-Host "professor's $Frame uses the $($hand.ToUpper()) hand. Writing the txt from the" -ForegroundColor Yellow
    Write-Host "recorded hand anyway - re-record with the $($hand.ToUpper()) glove streaming for the real one." -ForegroundColor Yellow
    python scripts/export_prof_format.py $rec --out-dir $rec
    if ($LASTEXITCODE -ne 0) { throw "export failed" }
}
if (-not (Test-Path $outFile)) { throw "no frames recorded at all" }
Write-Host "`nDone: $outFile" -ForegroundColor Green
