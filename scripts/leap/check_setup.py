"""Phase 0 checker: is this machine ready to record with the Stereo IR 170?

Runs the whole chain from the SDK on disk to hands arriving in this process,
prints PASS or FAIL for each link with the fix for every failure, and exits
0 only if everything passed (2 otherwise, so a script can gate on it).

  1  LeapSDK folder          C:\\Program Files\\Ultraleap\\LeapSDK, or wherever
                             LEAPSDK_INSTALL_LOCATION points
  2  LeapC.h                 include\\LeapC.h — the cffi build needs it
  3  LeapC.dll / LeapC.lib   lib\\x64\\ — the bindings link against these
  4  tracking service        the Ultraleap service, running
  5  import leap             the Python bindings, in THIS interpreter
  6  live tracking           a device, hands, and the tracking framerate,
                             over a 3 s connection

Until the hardware and Hyperion are installed this is expected to fail from
line 1 and exit 2; that is the point of running it first. To check that the
rest of the pipeline works meanwhile:

  python scripts/leap/check_setup.py --mock

which skips the environment (there is nothing to check) and exercises the
mock stream, conversion and joint layout instead.

Usage:
  python scripts/leap/check_setup.py
  python scripts/leap/check_setup.py --mock
  python scripts/leap/check_setup.py --seconds 10
"""
import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_SDK = Path(r"C:\Program Files\Ultraleap\LeapSDK")
DOWNLOAD_PAGE = "https://www.ultraleap.com/downloads/sir170/"
SETUP_SCRIPT = r"powershell -ExecutionPolicy Bypass -File scripts\leap\setup_bindings.ps1"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""

    @property
    def counts(self) -> bool:
        return self.status != SKIP


def sdk_root() -> Path:
    """Where the LeapSDK should be: the env override wins, as the build does."""
    env = os.environ.get("LEAPSDK_INSTALL_LOCATION")
    return Path(env) if env else DEFAULT_SDK


def check_sdk(root: Path) -> List[Check]:
    source = ("LEAPSDK_INSTALL_LOCATION" if os.environ.get("LEAPSDK_INSTALL_LOCATION")
              else "default path")
    if not root.is_dir():
        install = (f"install Ultraleap Hyperion 6.2.0 for Windows from "
                   f"{DOWNLOAD_PAGE} (it installs the SDK here), or set "
                   "LEAPSDK_INSTALL_LOCATION to an existing LeapSDK folder")
        return [
            Check("LeapSDK folder", FAIL,
                  f"{root} ({source}) does not exist", install),
            Check("LeapC.h", FAIL, "no SDK folder to look in", "same fix as above"),
            Check("LeapC.dll / LeapC.lib", FAIL, "no SDK folder to look in",
                  "same fix as above"),
        ]

    checks = [Check("LeapSDK folder", PASS, f"{root} ({source})")]

    header = root / "include" / "LeapC.h"
    override = os.environ.get("LEAPC_HEADER_OVERRIDE")
    if override:
        header = Path(override)
    checks.append(
        Check("LeapC.h", PASS, str(header)) if header.is_file() else
        Check("LeapC.h", FAIL, f"{header} not found",
              "the SDK is installed but incomplete — reinstall Hyperion, or "
              "point LEAPC_HEADER_OVERRIDE at the header")
    )

    lib_dir = root / "lib" / "x64"
    missing = [n for n in ("LeapC.dll", "LeapC.lib") if not (lib_dir / n).is_file()]
    checks.append(
        Check("LeapC.dll / LeapC.lib", PASS, str(lib_dir)) if not missing else
        Check("LeapC.dll / LeapC.lib", FAIL,
              f"missing in {lib_dir}: {', '.join(missing)}",
              "reinstall Hyperion (the 64-bit runtime is part of it), or point "
              "LEAPC_LIB_OVERRIDE at the folder holding LeapC.lib")
    )
    return checks


def _powershell(command: str, timeout: float = 20.0) -> Optional[str]:
    """Run one PowerShell command, or return None if that is not possible."""
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def check_service() -> Check:
    """Find the Ultraleap tracking service whatever this version calls it.

    Hyperion's display name is "Ultraleap Tracking Service"; Gemini 5.x
    shipped the key `LeapService`. Rather than hard-code either, match on
    both and report what was actually found.
    """
    if platform.system() != "Windows":
        return Check("tracking service", FAIL,
                     f"not Windows ({platform.system()})",
                     "the tracking service is Windows-only; this repo's Leap "
                     "work assumes Windows 11")

    out = _powershell(
        "Get-Service | Where-Object { $_.DisplayName -like '*Ultraleap*' -or "
        "$_.Name -eq 'LeapService' -or $_.DisplayName -like '*Leap*' } | "
        "ForEach-Object { \"$($_.Name)|$($_.DisplayName)|$($_.Status)\" }"
    )
    fix = (f"install Ultraleap Hyperion 6.2.0 from {DOWNLOAD_PAGE}; if it is "
           "already installed, open the Ultraleap Control Panel (it starts the "
           "service), or run  Start-Service -DisplayName 'Ultraleap Tracking "
           "Service'  from an elevated PowerShell")
    if out is None:
        return Check("tracking service", FAIL, "could not run PowerShell", fix)

    found = [line.strip().split("|") for line in out.splitlines() if "|" in line]
    if not found:
        return Check("tracking service", FAIL,
                     "no service whose name or display name mentions Ultraleap "
                     "or Leap", fix)

    # Prefer a running one, and the Ultraleap-branded one over a bare "Leap".
    def rank(entry):
        name, display, status = entry
        return (status.lower() != "running", "ultraleap" not in display.lower())

    name, display, status = sorted(found, key=rank)[0]
    detail = f"{display} [{name}] is {status}"
    if status.lower() != "running":
        return Check("tracking service", FAIL, detail,
                     f"start it:  Start-Service -Name {name}")
    return Check("tracking service", PASS, detail)


def check_bindings() -> Check:
    try:
        import leap
    except ImportError as e:
        return Check("import leap", FAIL, str(e),
                     f"build and install the bindings:  {SETUP_SCRIPT}")
    except Exception as e:
        # A built leapc_cffi that cannot find LeapC.dll fails here, not above.
        return Check("import leap", FAIL, f"{type(e).__name__}: {e}",
                     "the bindings are installed but cannot load LeapC.dll — "
                     f"check the SDK lines above, then re-run {SETUP_SCRIPT}")
    where = getattr(leap, "__file__", "?")
    return Check("import leap", PASS, f"{where} (python {sys.executable})")


def check_live(seconds: float) -> Check:
    """Connect for `seconds` and report device, hands and tracking rate."""
    from leap_hand.stream import LeapStream, LeapUnavailable

    try:
        stream = LeapStream(mode="desktop", device_timeout=max(seconds, 3.0))
        stream.start()
    except LeapUnavailable as e:
        first, _, rest = str(e).partition("\n")
        return Check("live tracking", FAIL, first,
                     rest.strip() or "see the message above")
    except Exception as e:      # pragma: no cover - hardware path
        return Check("live tracking", FAIL, f"{type(e).__name__}: {e}",
                     "unexpected SDK error — re-run with the Control Panel open "
                     "and check that no other app is holding the camera")

    try:
        sides = set()
        hands = 0
        t_end = time.time() + seconds
        while time.time() < t_end:
            for side, _lh in stream.drain(64):
                sides.add(side)
                hands += 1
            time.sleep(0.01)
        rate = stream.framerate
        detail = (f"device {stream.device_serial}, {stream.frames} events, "
                  f"{hands} hands ({', '.join(sorted(sides)) or 'none'}), "
                  f"tracking {rate:.1f} Hz")
    finally:
        stream.stop()

    if not stream.frames:
        return Check("live tracking", FAIL, detail,
                     "the device is connected but sent no tracking events — "
                     "check the Control Panel visualiser and the tracking mode")
    if not hands:
        return Check("live tracking", FAIL, detail,
                     "tracking runs but saw no hand — hold a hand 20 to 50 cm "
                     "above the module, lenses up, away from sunlight and other "
                     "IR sources")
    return Check("live tracking", PASS, detail)


def check_mock(seconds: float) -> List[Check]:
    """The same pipeline, driven by MockLeapStream: no SDK, no camera."""
    from leap_hand.mock import MockLeapStream
    from leap_hand.to_openxr import to_hand_frame
    from xr_hand.joints import JOINT_NAMES
    from xr_hand.kinematics import forward_kinematics

    stream = MockLeapStream()
    stream.start()
    try:
        sides = set()
        hands = 0
        worst = 0.0
        t_end = time.time() + seconds
        while time.time() < t_end:
            for side, lh in stream.drain(64):
                sides.add(side)
                hands += 1
                frame = to_hand_frame(lh)
                if [j.name for j in frame.joints] != JOINT_NAMES:
                    return [Check("mock pipeline", FAIL, "joint order is wrong",
                                  "this is a bug in leap_hand, not in setup")]
                for got, want in zip(forward_kinematics(frame), lh.abs26):
                    worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
            time.sleep(0.005)
    finally:
        stream.stop()

    if not hands:
        return [Check("mock pipeline", FAIL, "the mock produced no hands",
                      "this is a bug in leap_hand, not in setup")]
    return [
        Check("mock pipeline", PASS,
              f"{hands} hands ({', '.join(sorted(sides))}), 26 joints each"),
        Check("mock round trip", PASS,
              f"absolute -> relative -> forward kinematics within "
              f"{worst * 1000:.2g} mm"),
    ]


def report(checks: List[Check]) -> int:
    width = max(len(c.name) for c in checks)
    print()
    for c in checks:
        print(f"{c.status}  {c.name:<{width}}  {c.detail}")
        if c.status == FAIL and c.fix:
            print(f"      {' ' * width}  fix: {c.fix}")
    counted = [c for c in checks if c.counts]
    failed = [c for c in counted if c.status == FAIL]
    print()
    if failed:
        print(f"{len(counted) - len(failed)}/{len(counted)} checks passed — "
              f"not ready. Fix the FAIL lines above, top to bottom.")
        return 2
    print(f"{len(counted)}/{len(counted)} checks passed.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check that this machine can record with the Stereo IR 170.")
    p.add_argument("--mock", action="store_true",
                   help="skip the environment; exercise the mock pipeline instead")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="how long to watch for hands (default: 3)")
    args = p.parse_args()

    print("=" * 72)
    print("Ultraleap Stereo IR 170 — setup check" + ("  [mock]" if args.mock else ""))
    print("=" * 72)

    if args.mock:
        checks = [Check("environment", SKIP, "--mock: SDK, service and device "
                                             "are not needed")]
        checks += check_mock(args.seconds)
        sys.exit(report(checks))

    checks = check_sdk(sdk_root())
    checks.append(check_service())
    checks.append(check_bindings())
    if checks[-1].status == PASS:
        checks.append(check_live(args.seconds))
    else:
        checks.append(Check(
            "live tracking", FAIL, "skipped: the bindings did not import",
            "fix the import leap line first; everything below depends on it"))
    sys.exit(report(checks))


if __name__ == "__main__":
    main()
