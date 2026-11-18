$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo '.venv\Scripts\python.exe'
Set-Location $repo
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$run = Join-Path $repo "recordings\glove_camera_pilot\$stamp"
if (Test-Path -LiteralPath $run) { throw "Run already exists: $run. Start again in a new second." }
New-Item -ItemType Directory -Path $run | Out-Null

Write-Host 'Turn on and connect the Reality Gloves. Open XR Train.'
Write-Host 'Enable OSC to 127.0.0.1:9002. Close other OSC viewers and recorders.'
Write-Host 'Connect the Ultraleap directly by USB and check its Control Panel.'
Write-Host 'Wear and calibrate the LEFT glove in XR Train. Keep the right hand away.'
Read-Host 'Press Enter when you are ready for the device checks' | Out-Null

& $py scripts\leap\check_setup.py --seconds 3
if ($LASTEXITCODE -ne 0) { throw 'Camera check failed. Follow the printed fix and start again.' }
& $py scripts\glove_camera_pilot_check.py probe --seconds 5 --out "$run\osc_probe"
if ($LASTEXITCODE -ne 0) { throw 'No OSC stream. Check XR Train and its OSC settings, then start again.' }
$probe = Get-Content -Raw -LiteralPath "$run\osc_probe\osc_probe.json" | ConvertFrom-Json
if (-not $probe.addresses.'/v1/animation/kinematic/all') {
    throw 'OSC messages arrived, but no kinematic skeleton. Enable the kinematic stream in XR Train.'
}
if (-not ($probe.device_labels -match '\(L\)')) {
    throw 'No left glove label arrived. Check that the left Reality Glove is connected in XR Train.'
}

Write-Host 'Follow the recorder prompts: OPEN PALM, FIST, then PINCH.'
Write-Host 'Before each take, show an open left hand, palm toward the lenses, 18-28 cm above them.'
Read-Host 'Press Enter to start the three-pose recording' | Out-Null
& $py scripts\record_simultaneous.py --camera leap --hand left --poses open_palm,fist,pinch --takes 1 --duration 5 --prep 5 --retries 1 --acquire-timeout 30 --out-dir "$run\capture" --note 'Glove-camera pilot; Reality Gloves through XR Train; device agreement only'
if ($LASTEXITCODE -ne 0) { throw "Capture stopped. Keep all files and review: $run" }
& $py scripts\glove_camera_pilot_check.py audit --session "$run\capture" --hand left --out "$run\comparison"
if ($LASTEXITCODE -ne 0) { throw "Some poses were missing or unusable. Keep all files and review: $run" }
Write-Host "Pilot files and comparison saved in $run"
Write-Host 'The report measures agreement between the two solved skeletons. Anatomical accuracy remains unverified.'
