<#
.SYNOPSIS
    Build and install the Ultraleap Python bindings into this repo's venv.

.DESCRIPTION
    The `leap` package cannot be pip-installed: it is built on `leapc_cffi`,
    a cffi module that has to be compiled against the LeapSDK on this machine
    (the pre-compiled ones Ultraleap ships cover Python 3.8 only, and this
    venv is 3.14). So this script does what the bindings' README does, in
    order, and stops with a readable message the moment a step fails:

      1. clone github.com/ultraleap/leapc-python-bindings into TEMP
      2. pip install its requirements, plus `build`
      3. python -m build leapc-cffi     (the compile step; needs LeapSDK and
                                         the MSVC C++ toolset)
      4. pip install the built sdist/wheel
      5. pip install -e leapc-python-api
      6. run check_setup.py

    Prerequisites: Ultraleap Hyperion 6.2.0 installed (for the LeapSDK), git,
    and Visual Studio Build Tools with the C++ toolset. Set
    LEAPSDK_INSTALL_LOCATION first if the SDK is not at
    C:\Program Files\Ultraleap\LeapSDK.

.PARAMETER Source
    Where to clone/find the bindings. Default: $env:TEMP\leapc-python-bindings.

.PARAMETER Python
    The interpreter to install into. Default: this repo's .venv.

.PARAMETER Fresh
    Delete an existing clone and start over.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\leap\setup_bindings.ps1

.EXAMPLE
    $env:LEAPSDK_INSTALL_LOCATION = "D:\LeapSDK"
    powershell -ExecutionPolicy Bypass -File scripts\leap\setup_bindings.ps1 -Fresh
#>
[CmdletBinding()]
param(
    [string] $Source = (Join-Path $env:TEMP "leapc-python-bindings"),
    [string] $Python,
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/ultraleap/leapc-python-bindings"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Step([string] $Message) {
    Write-Host ""
    Write-Host "== $Message" -ForegroundColor Cyan
}

function Stop-With([string] $Message, [string] $Fix) {
    Write-Host ""
    Write-Host "FAILED: $Message" -ForegroundColor Red
    Write-Host "  fix: $Fix" -ForegroundColor Yellow
    exit 1
}

# --- the interpreter --------------------------------------------------------
if (-not $Python) {
    $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) {
    Stop-With "no Python at $Python" `
        "create the venv first:  python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -e ."
}
$version = & $Python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "Installing the Ultraleap bindings into: $Python  (Python $version)"

# --- the SDK ----------------------------------------------------------------
Step "LeapSDK"
$sdk = $env:LEAPSDK_INSTALL_LOCATION
if (-not $sdk) { $sdk = "C:\Program Files\Ultraleap\LeapSDK" }
if (-not (Test-Path $sdk)) {
    Stop-With "no LeapSDK at $sdk" `
        "install Ultraleap Hyperion 6.2.0 for Windows from https://www.ultraleap.com/downloads/sir170/, or set LEAPSDK_INSTALL_LOCATION to the SDK folder and re-run"
}
foreach ($needed in @("include\LeapC.h", "lib\x64\LeapC.dll", "lib\x64\LeapC.lib")) {
    if (-not (Test-Path (Join-Path $sdk $needed))) {
        Stop-With "$sdk is missing $needed" `
            "reinstall Hyperion - the cffi build cannot compile without the header and the 64-bit library"
    }
}
$env:LEAPSDK_INSTALL_LOCATION = $sdk
Write-Host "  using $sdk"

# --- git clone --------------------------------------------------------------
Step "bindings source"
if ($Fresh -and (Test-Path $Source)) {
    Write-Host "  -Fresh: removing $Source"
    Remove-Item -Recurse -Force $Source
}
if (Test-Path $Source) {
    Write-Host "  reusing the existing clone at $Source"
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Stop-With "git is not on PATH" `
            "install Git for Windows, or download the repo as a zip from $RepoUrl and pass -Source <folder>"
    }
    Write-Host "  cloning $RepoUrl -> $Source"
    & git clone --depth 1 $RepoUrl $Source
    if ($LASTEXITCODE -ne 0) {
        # This network has had certificate-revocation failures before (the
        # same reason downloads here need curl --ssl-no-revoke).
        Write-Host "  clone failed; retrying with revocation checking off" -ForegroundColor Yellow
        & git -c http.sslBackend=schannel -c http.schannelCheckRevoke=false clone --depth 1 $RepoUrl $Source
        if ($LASTEXITCODE -ne 0) {
            Stop-With "could not clone $RepoUrl" `
                "check the network, or download the zip by hand and pass -Source <folder>"
        }
    }
}

$cffiDir = Join-Path $Source "leapc-cffi"
$apiDir = Join-Path $Source "leapc-python-api"
foreach ($d in @($cffiDir, $apiDir)) {
    if (-not (Test-Path $d)) {
        Stop-With "$d is missing" `
            "the clone at $Source does not look like leapc-python-bindings; re-run with -Fresh"
    }
}

# --- requirements -----------------------------------------------------------
# The bindings' requirements.txt lists opencv-python, which their example
# viewer uses and we do not. Installing it here would put a second cv2 in this
# venv beside mediapipe's opencv-contrib-python, and two packages providing
# cv2 shadow each other - the exact thing pyproject.toml warns about. So the
# opencv lines are dropped and everything else is installed as published.
Step "pip install -r requirements.txt build   (minus opencv: see pyproject.toml)"
$req = Join-Path $Source "requirements.txt"
if (Test-Path $req) {
    $wanted = Get-Content $req | Where-Object { $_.Trim() -and $_ -notmatch 'opencv' }
    $dropped = Get-Content $req | Where-Object { $_ -match 'opencv' }
    foreach ($d in $dropped) {
        Write-Host "  skipping $($d.Trim()) - this repo gets cv2 from mediapipe's opencv-contrib-python" -ForegroundColor Yellow
    }
    if ($wanted) {
        $filtered = Join-Path $env:TEMP "leap-requirements-no-opencv.txt"
        $wanted | Set-Content -Path $filtered -Encoding ascii
        & $Python -m pip install -r $filtered
        if ($LASTEXITCODE -ne 0) {
            Stop-With "pip could not install $filtered" "read pip's error above; a network failure here is usually the certificate issue (see README)"
        }
    }
} else {
    Write-Host "  no requirements.txt in this revision; skipping"
}
& $Python -m pip install build
if ($LASTEXITCODE -ne 0) { Stop-With "pip could not install 'build'" "read pip's error above" }

# --- compile leapc_cffi -----------------------------------------------------
Step "python -m build leapc-cffi   (compiles against $sdk)"
Push-Location $Source
try {
    & $Python -m build $cffiDir
    if ($LASTEXITCODE -ne 0) {
        Stop-With "the leapc_cffi build failed" `
            "this is the compile step: it needs Visual Studio Build Tools with the C++ toolset, and a LeapC.h matching the DLL. If it fails only on Python $version, create a 3.11 venv (winget install Python.Python.3.11) and re-run with -Python <that venv>\Scripts\python.exe (see docs/ultraleap_ir170_plan.md section 4)"
    }
} finally {
    Pop-Location
}

$artifact = Get-ChildItem (Join-Path $cffiDir "dist") -Include *.whl, *.tar.gz -Recurse |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $artifact) {
    Stop-With "the build produced nothing in $cffiDir\dist" "re-run with -Fresh and read the build output"
}

Step "pip install $($artifact.Name)"
& $Python -m pip install $artifact.FullName
if ($LASTEXITCODE -ne 0) { Stop-With "pip could not install $($artifact.Name)" "read pip's error above" }

Step "pip install -e leapc-python-api"
& $Python -m pip install -e $apiDir
if ($LASTEXITCODE -ne 0) { Stop-With "pip could not install $apiDir" "read pip's error above" }

# --- verify -----------------------------------------------------------------
Step "import leap"
& $Python -c "import leap; print('  leap imported from', leap.__file__)"
if ($LASTEXITCODE -ne 0) {
    Stop-With "the bindings installed but 'import leap' fails" `
        "usually a LeapC.dll that cannot be found: confirm $sdk\lib\x64\LeapC.dll exists and re-run this script with -Fresh"
}

Step "scripts\leap\check_setup.py"
& $Python (Join-Path $RepoRoot "scripts\leap\check_setup.py")
$checkExit = $LASTEXITCODE

Write-Host ""
if ($checkExit -eq 0) {
    Write-Host "Done - the bindings are installed and the camera is tracking." -ForegroundColor Green
} else {
    Write-Host "The bindings are installed, but check_setup.py still reports problems above." -ForegroundColor Yellow
    Write-Host "Those are device/service issues, not build issues; fix them top to bottom." -ForegroundColor Yellow
}
Write-Host "Archive the installer and $sdk into reference\ultraleap\ today (plan section 1)." -ForegroundColor Yellow
exit $checkExit
