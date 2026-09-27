"""Record the professor's Set B (finger flexion) and Set C (sequences), hands-free.

Gloves on, camera as the reference, one hand per run. Every movement is cued:
a beep (higher to bend or flex, lower to straighten or extend, as
`scripts/leap/finger_sweep.py` cues them), the cue's words on a window and a
console line, and the same cue written to the take's events file at the
instant of the beep. The glove is recorded at its full incoming rate and the
camera alongside it, each line carrying the session, item and take. What
gets written, and where, is fixed by `docs/protocol_formats.md`; the
protocol itself (which fingers, how long, how many) is data in
`protocols/finger_flexion.json` and `protocols/sequences.json`, loaded and
checked by `cam_hand.recording_protocol`.

Why it is built this way (plan: `docs/grasp_and_flexion_protocol_plan.md`):

  * the cue window shows ONLY the take name, the cue words and a countdown.
    No skeleton and no camera image, with or without the camera. One person
    cues, performs and judges, and a live skeleton invites shaping the
    movement until glove and camera agree; the operator follows the beep.
  * every take is checked before the next one starts (the quick check in
    `recording_protocol`), because a bad take is cheap to redo while the
    gloves are on and expensive to discover afterwards. A rejected attempt
    is MOVED to `rejected/` with its reason, never deleted: the two sensors
    under evaluation are the ones judging, so every exclusion has to stay
    countable.
  * each finger is measured against its own open and fist for THIS session
    (the warm-up), not a number from another day. A glove that barely moves
    between the two refuses the session instead of producing fractions of
    nothing.
  * Set C runs all seven sequences once per round in a shuffled order, with
    the seed saved, so the three takes of a sequence are independent
    repetitions rather than practice.
  * `session.json` is rewritten after every take, so a crash or a Ctrl+C
    keeps everything done so far, with the reason for every rejection.

Pieces reused rather than rewritten: the glove recorder with the packet
arrival time (`StampedFrameRecorder`) and the beep from
`scripts/record_simultaneous.py`; the acquire gate, HUD, async beeper, crop
box and still bookkeeping from `leap_hand.protocol`; `LeapRecorder`; the
camera viewer (`scripts/leap/camera_view.py`) run with no window, only to
cut the one hand-cropped IR still per take.

Rehearse with no hardware at all (a mock glove that follows the cues, a mock
camera, a quarter of the real durations):

  python scripts/record_protocol.py --set finger_flexion --hand left \\
      --items index_fast --mock-glove --mock-leap --no-view --no-open \\
      --calibrated-at 2026-09-28T14:00:00 --time-scale 0.25

`--mock-glove-ignore-cues` keeps the mock hand open through every take, so
the reject path (and `rejected/`) can be rehearsed too. A run with any mock
sensor writes under `recordings/protocol_mock/` unless `--out-dir` says
otherwise, and its session.json says `"mock": true`, which the packager
refuses: a rehearsal cannot be handed in as data.

Real session (XR Trainer streaming to 127.0.0.1:9002, camera plugged in):

  python scripts/record_protocol.py --set finger_flexion --hand left
  python scripts/record_protocol.py --set sequences --hand left

During the few seconds after each take, `r` redoes it (the operator's own
verdict, which does not use up a retry) and `q` stops the session, on the
cue window or in the console. Exit code 0 when every planned take has an
accepted attempt, 1 when some take has none or the session was stopped, 2
when the session was refused before any take (protocol file, glove, camera,
acquire gate or warm-up).
"""
import argparse
import bisect
import datetime
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cam_hand import recording_protocol as rp  # noqa: E402
from cam_hand.features import flexion_features  # noqa: E402
from leap_hand.protocol import (  # noqa: E402
    DEFAULT_BAND,
    NO_TRACKED_HAND,
    AsyncBeeper,
    CameraView,
    Hud,
    acquire_failures,
    band_text,
    coverage,
    hand_crop_box,
    hud_line,
    move_still,
    parse_band,
    read_hand,
    skipped_still_path,
    skipped_still_text,
    still_status,
    stream_health,
)
from leap_hand.recorder import LeapRecorder  # noqa: E402
from leap_hand.stream import LeapUnavailable, open_stream  # noqa: E402
from xr_hand.joints import HEADER_LEN, JOINT_NAMES, VALUES_PER_JOINT  # noqa: E402
from xr_hand.keypoints21 import MP21_TO_OPENXR_IDX, frame_to_keypoints21  # noqa: E402
from xr_hand.parser import parse_hand_message  # noqa: E402
from xr_hand.receiver import OSCHandReceiver, QueueItem  # noqa: E402
from xr_hand.validator import StreamMonitor, validate_raw_message  # noqa: E402

# --- timings (seconds at --time-scale 1) -------------------------------------
PREP_S = 3.0              # the item's words on screen before its first cue
PAUSE_S = 3.0             # after a take: r = redo, q = stop
WARMUP_COUNTDOWN_S = 3.0  # between the acquire gate and the open palm
WARMUP_OPEN_S = 3.0       # contract section 4: open palm 3 s ...
WARMUP_FIST_S = 3.0       # ... then a full fist 3 s
# The first second of each warm-up window is the movement into the pose (and
# the glove's lag, up to 0.47 s on the right hand), so the medians are taken
# over the rest of the window. The windows in warmup.json are those.
WARMUP_SETTLE_S = 1.0
STILL_WAIT_S = 2.0        # how long to wait for the viewer's still after a take
GLOVE_WAIT_PACKETS = 10
# Below this share of a take with the operator's hand tracked, the console
# says so after the take (the coached recorder's completeness bar).
CAMERA_COVERAGE_NOTE = 0.90

# Beeps (Hz, ms). Cue pitches come from `recording_protocol.cue_pitch`.
BEEP_MS = 150
PREP_BEEP = (500, 80)
END_BEEP = (500, 250)
WARMUP_OPEN_BEEP = (1000, 200)
WARMUP_FIST_BEEP = (1200, 200)

REJECTED = "rejected"
# Where sessions go when --out-dir is not given. A rehearsal with a mock
# sensor goes to its own tree (contract, section 7), so it can never sit
# beside a real session and be packaged as data by mistake.
OUT_DIR = Path("recordings") / "protocol"
MOCK_OUT_DIR = Path("recordings") / "protocol_mock"
SET_CHOICES = (rp.FLEXION, rp.SEQUENCES)
CAMERA_CHOICES = ("leap", "none")
EXIT_OK, EXIT_INCOMPLETE, EXIT_REFUSED = 0, 1, 2


# --- the coached recorder, imported rather than copied -----------------------
_COACHED = None


def coached():
    """`scripts/record_simultaneous.py` as a module (it is a script, not a
    package), loaded once. Its glove recorder, beep and thresholds are the
    ones every earlier session used, so they are imported from there."""
    global _COACHED
    if _COACHED is None:
        path = HERE / "record_simultaneous.py"
        spec = importlib.util.spec_from_file_location(
            "record_simultaneous_for_protocol", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _COACHED = module
    return _COACHED


class KeysWriter:
    """A recorder's file handle that adds fixed keys to every line.

    The contract adds `session`, `item` and `take` to every frame line. The
    recorders already write `take`; the other two are injected where the
    line meets the file, the same trick as `_CaptureTimeWriter` in the
    coached recorder, so neither recorder is changed and every existing
    reader (which ignores keys it does not know) reads these files as is.
    """

    def __init__(self, fh, keys: dict):
        self._fh = fh
        self._keys = dict(keys)

    def write(self, text: str) -> int:
        if text.strip():
            d = json.loads(text)
            d.update(self._keys)
            text = json.dumps(d) + "\n"
        return self._fh.write(text)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def stamp_keys(recorder, keys: dict) -> None:
    """Make an already-started recorder write `keys` on every line."""
    recorder._file = KeysWriter(recorder._file, keys)


# --- the mock glove that follows the cues ------------------------------------
_PHALANX = ("_PROXIMAL", "_INTERMEDIATE", "_DISTAL", "_TIP")
_PREFIX = {"thumb": "THUMB_", "index": "INDEX_", "middle": "MIDDLE_",
           "ring": "RING_", "pinky": "LITTLE_"}
# Where each finger's joint quaternions sit in the 187-value wire payload.
_QUAT_AT = {f: [HEADER_LEN + j * VALUES_PER_JOINT + 3
                for j, name in enumerate(JOINT_NAMES)
                if name.startswith(p) and name.endswith(_PHALANX)]
            for f, p in _PREFIX.items()}


class CueFollowingGlove:
    """Both gloves at ~60 Hz behind the receiver's drain() API, and the
    operator's hand DOES what it is told.

    The stock mock curls every finger on one slow sine, which no quick check
    could ever accept, so a rehearsal would only ever rehearse the reject
    path. This one keeps a timeline of what the hand was asked to do (every
    cue, and the warm-up) and bends each finger accordingly: a slow ramp over
    a bend or straighten phase, a quick move and a hold for a sequence step,
    all behind the glove's measured lag for that hand. The other hand rests
    open on the table.

    `follow_cues=False` is `--mock-glove-ignore-cues`: the hand still does
    the warm-up (or the session would be refused before any take) and then
    stays open through every take, so the reject path gets rehearsed.

    Built on `xr_hand.mock.MockHandGenerator` for the wire layout and packet
    counter; only the finger joints' rotations are replaced.
    """

    HZ = 60.0
    MAX_ANGLE = 1.1                       # radians per finger joint, full bend
    LAG_S = {"left": 0.10, "right": 0.47}  # glove lag per hand, measured
    MOVE_S = 0.8                          # seconds to move between two steps
    NOISE = 0.01                          # bend noise, fraction of full

    def __init__(self, hand: str, follow_cues: bool = True,
                 time_scale: float = 1.0, seed: int = 7):
        from xr_hand.mock import MockHandGenerator
        self.hand = hand
        self.follow_cues = bool(follow_cues)
        self.lag = self.LAG_S[hand] * float(time_scale)
        self.move = self.MOVE_S * float(time_scale)
        self.gens = {side: MockHandGenerator(hand=side)
                     for side in ("left", "right")}
        self._t0s: List[float] = []
        self._segments: List[dict] = []
        self._rng = random.Random(seed)
        self._last = time.time()

    def start(self) -> None:
        self._last = time.time()

    def stop(self) -> None:
        pass

    # --- what the hand is doing ----------------------------------------
    def follow(self, flexed, duration: float, phase: Optional[str] = None,
               t: Optional[float] = None, warmup: bool = False) -> None:
        """From `t` on, do this: `flexed` fingers bent, for `duration`."""
        t = time.time() if t is None else float(t)
        if not self.follow_cues and not warmup:
            flexed, phase = (), None
        start = ({f: 0.0 for f in rp.FINGERS} if not self._segments
                 else self._bend_in(self._segments[-1],
                                    t - self._segments[-1]["t0"]))
        self._segments.append({"t0": t, "flexed": frozenset(flexed),
                               "dur": max(float(duration), 1e-6),
                               "phase": phase, "start": start})
        self._t0s.append(t)

    def _bend_in(self, seg: dict, elapsed: float) -> Dict[str, float]:
        out = {}
        dur, e = seg["dur"], max(0.0, elapsed)
        for f in rp.FINGERS:
            p = seg["start"][f]
            if seg["phase"] == rp.BEND and f in seg["flexed"]:
                v = p + (1.0 - p) * min(1.0, e / dur)
            elif seg["phase"] == rp.STRAIGHTEN and f not in seg["flexed"]:
                v = p * (1.0 - min(1.0, e / dur))
            else:
                target = 1.0 if f in seg["flexed"] else 0.0
                move = min(self.move, 0.5 * dur)
                a = 1.0 if move <= 0 else min(1.0, e / move)
                v = p + (target - p) * a * a * (3.0 - 2.0 * a)
            out[f] = v
        return out

    def bend_at(self, t: float) -> Dict[str, float]:
        """Each finger's bend (0 open, 1 fully bent) at wall time `t`."""
        t -= self.lag
        k = bisect.bisect_right(self._t0s, t) - 1
        if k < 0:
            return {f: 0.0 for f in rp.FINGERS}
        seg = self._segments[k]
        return self._bend_in(seg, t - seg["t0"])

    # --- the stream ------------------------------------------------------
    def _shape(self, raw: list, bend: Dict[str, float]) -> None:
        for f, slots in _QUAT_AT.items():
            b = max(0.0, min(1.05, bend[f] + self._rng.gauss(0.0, self.NOISE)))
            h = -0.5 * b * self.MAX_ANGLE
            q = (math.sin(h), 0.0, 0.0, math.cos(h))
            for i in slots:
                raw[i:i + 4] = q

    def drain(self, max_items: int = 64):
        now = time.time()
        n = min(int((now - self._last) * self.HZ), max_items // 2)
        if n <= 0:
            return []
        t0 = self._last
        self._last += n / self.HZ
        rest = {f: 0.0 for f in rp.FINGERS}
        out = []
        for k in range(n):
            t = t0 + k / self.HZ
            for side, gen in self.gens.items():
                raw = gen.next_frame()
                self._shape(raw, self.bend_at(t) if side == self.hand else rest)
                out.append(QueueItem(side, raw, recv_time=t))
        return out


# --- the cue window ----------------------------------------------------------
class CueWindow:
    """What the operator looks at: the take name, the cue words, a countdown.

    Nothing else, in either camera mode, on purpose: see the module
    docstring. A plain OpenCV canvas, kept on top, redrawn at most 30 times
    a second. `enabled=False` (`--no-view`) makes every call a no-op.
    """

    NAME = "Protocol cues"
    W, H = 1000, 560
    EVERY = 1.0 / 30.0

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._open = False
        self._at = 0.0

    @staticmethod
    def _wrap(text: str, scale: float, thick: int, width: int) -> List[str]:
        words, lines, line = str(text).split(), [], ""
        for w in words:
            trial = (line + " " + w).strip()
            size = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                   thick)[0][0]
            if size > width and line:
                lines.append(line)
                line = w
            else:
                line = trial
        if line:
            lines.append(line)
        return lines or [""]

    def show(self, title: str, words: str, seconds_left: Optional[float],
             sub: str = "", force: bool = False) -> str:
        """Redraw; returns the key pressed on the window ('' for none)."""
        if not self.enabled:
            return ""
        now = time.time()
        if not force and now - self._at < self.EVERY:
            return ""
        self._at = now
        if not self._open:
            cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.NAME, self.W, self.H)
            try:
                cv2.setWindowProperty(self.NAME, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
            self._open = True
        img = np.zeros((self.H, self.W, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, title or "", (24, 44), font, 0.8, (170, 170, 170), 2,
                    cv2.LINE_AA)
        lines = self._wrap(words or "", 1.8, 4, self.W - 80)
        if len(lines) > 2:
            lines = self._wrap(words or "", 1.0, 2, self.W - 60)[:5]
            scale, thick, step = 1.0, 2, 44
        else:
            scale, thick, step = 1.8, 4, 74
        y = int(self.H * 0.42) - (len(lines) - 1) * step // 2
        for line in lines:
            size = cv2.getTextSize(line, font, scale, thick)[0][0]
            cv2.putText(img, line, ((self.W - size) // 2, y), font, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
            y += step
        for line in self._wrap(sub or "", 0.7, 2, self.W - 60)[:2]:
            size = cv2.getTextSize(line, font, 0.7, 2)[0][0]
            cv2.putText(img, line, ((self.W - size) // 2, y + 10), font, 0.7,
                        (170, 170, 170), 2, cv2.LINE_AA)
            y += 34
        if seconds_left is not None:
            left = max(0.0, float(seconds_left))
            text = f"{left:.1f}" if left < 1.0 else f"{math.ceil(left):d}"
            size = cv2.getTextSize(text, font, 2.2, 5)[0][0]
            cv2.putText(img, text, ((self.W - size) // 2, self.H - 40), font,
                        2.2, (0, 220, 255), 5, cv2.LINE_AA)
        cv2.imshow(self.NAME, img)
        k = cv2.waitKey(1) & 0xFF
        return chr(k).lower() if 32 <= k < 127 else ""

    def close(self) -> None:
        if self._open:
            try:
                cv2.destroyWindow(self.NAME)
            except Exception:
                pass
            self._open = False


def console_key() -> str:
    """A key pressed in the console, or ''. Only when a person is at it."""
    try:
        if not sys.stdin.isatty():
            return ""
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch().lower()
    except Exception:
        return ""
    return ""


def flush_console_keys() -> None:
    for _ in range(64):
        if not console_key():
            return


# --- the one still per take --------------------------------------------------
class HiddenCameraView(CameraView):
    """`scripts/leap/camera_view.py` with `--no-window`: stills, no picture.

    The viewer already knows how to cut a still to the tracked hand (and to
    leave a note when there is no hand), which is the still the contract
    asks for. What it must not do here is show the IR image and the skeleton
    while the operator performs, so it runs with no window at all.
    `CameraView.start` builds its command inline, which is why this one is
    repeated with the one flag added.
    """

    def start(self):
        if not self.enabled or self._proc is not None:
            return self
        import atexit
        import tempfile
        self._status = (Path(tempfile.gettempdir())
                        / f"protocol_view_status_{os.getpid()}.txt")
        self.caption("starting")
        cmd = [sys.executable, str(self.SCRIPT), "--hand", self.hand,
               "--status-file", str(self._status),
               "--parent-pid", str(os.getpid()), "--no-window"]
        if self.band:
            cmd += ["--band", f"{self.band[0]:g},{self.band[1]:g}"]
        try:
            self._proc = subprocess.Popen(cmd)
        except OSError:
            self.enabled = False
            return self
        atexit.register(self.close)
        return self


class NoStills:
    """No camera, no still."""

    def request(self, path, caption: str, hand) -> None:
        pass

    def ready(self, path) -> bool:
        return True

    def close(self) -> None:
        pass


class ViewerStills(NoStills):
    """Stills from the real camera, through the hidden viewer."""

    def __init__(self, hand: str, band):
        self.view = HiddenCameraView(hand=hand, band=band).start()

    def request(self, path, caption: str, hand) -> None:
        self.view.caption(caption)
        self.view.snapshot(path)

    def ready(self, path) -> bool:
        # A viewer that could not start will never write anything; the
        # take then records the still as missing for an unknown reason.
        return (not self.view.enabled or Path(path).is_file()
                or skipped_still_path(path).is_file())

    def close(self) -> None:
        self.view.close()


class MockStills(NoStills):
    """A still cut from the synthetic IR image around the mock hand.

    Obviously synthetic on sight (a gradient with MOCK written on it), and
    that is the point: it exercises the crop, the file name and the
    rejected/ move without ever looking like evidence.
    """

    SIZE = 384
    FOCAL_PX = 200.0

    def __init__(self):
        from leap_hand.images import MockImageSampler
        self.sampler = MockImageSampler()

    def _pixels(self, hand) -> list:
        out = []
        for x, y, z in (getattr(hand, "abs26", None) or []):
            if y > 0.05:
                out.append((self.SIZE / 2 + self.FOCAL_PX * x / y,
                            self.SIZE / 2 + self.FOCAL_PX * z / y))
        return out

    def request(self, path, caption: str, hand) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        box = hand_crop_box(self._pixels(hand), self.SIZE) if hand else None
        if box is None:
            skipped_still_path(path).write_text(
                skipped_still_text(NO_TRACKED_HAND, time.time()),
                encoding="utf-8")
            return
        x0, y0, x1, y1 = box
        crop = cv2.cvtColor(self.sampler.latest().left[y0:y1, x0:x1].copy(),
                            cv2.COLOR_GRAY2BGR)
        for text, y in (("MOCK", 28), (caption, 54)):
            cv2.putText(crop, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(path), crop)

    def ready(self, path) -> bool:
        return True


# --- small helpers -----------------------------------------------------------
def iso_now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def tool_commit() -> Optional[str]:
    """The repo's short commit, read-only; None when git cannot say."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short",
                              "HEAD"], capture_output=True, text=True,
                             timeout=5)
    except Exception:
        return None
    text = (out.stdout or "").strip()
    return text if out.returncode == 0 and text else None


def display_path(path: Path) -> str:
    """A path relative to the repo when it is inside it, forward slashes."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data) -> None:
    """Write through a temporary file, so a crash never leaves half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for _ in range(20):                    # OneDrive can hold a file briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


class NotReady(Exception):
    """The session cannot start (glove, camera or warm-up). Exit code 2."""


class StopSession(Exception):
    """The operator stopped the session."""


# --- the session -------------------------------------------------------------
class ProtocolSession:
    """One hand, one protocol file, one folder. See the module docstring."""

    def __init__(self, args, protocol: rp.Protocol, items: List[str],
                 plan: List[List[str]], seed, glove, leap, stills,
                 window: CueWindow, beep):
        self.args = args
        self.hand = args.hand
        self.protocol = protocol
        self.items = items
        self.plan = plan
        self.seed = seed
        self.glove = glove
        self.leap = leap
        self.stills = stills
        self.window = window
        self.beeper = AsyncBeeper(beep)
        self.scale = float(args.time_scale)
        self.band = args.band
        self.coached = coached()
        self.monitors = {"left": StreamMonitor("left"),
                         "right": StreamMonitor("right")}
        self.hud = Hud(self._write)
        self.glove_rec = None
        self.cam_rec = None
        self.collect: Optional[dict] = None
        self.cam_reading = None
        self.cam_at = 0.0
        self.other_at = 0.0
        self.last_hand = None
        self.cam_times: List[float] = []
        self.glove_total = 0
        self.glove_other = 0
        self.glove_sides: set = set()
        self._rate_at, self._rate_mark, self._glove_hz = time.time(), 0, 0.0
        self.display = ("", "", None, "")
        self._snap = None
        self._warned: set = set()
        self.warmup: Optional[dict] = None
        self.dir: Optional[Path] = None
        self.takes: List[dict] = []
        self.accepted: Dict[str, int] = {i: 0 for i in items}
        self.failed: List[str] = []
        self.stopped: Optional[str] = None
        self.meta: dict = {}

    # --- plumbing ---------------------------------------------------------
    @staticmethod
    def _write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def say(self, *lines: str) -> None:
        self.hud.close()
        for line in lines:
            print(line, flush=True)

    def warn(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            if len(self._warned) <= 8:
                self.say(f"      ! {msg}")

    def beep(self, freq: int, ms: int) -> None:
        self.beeper.beep(int(freq), int(ms))

    def seconds(self, s: float) -> float:
        return float(s) * self.scale

    def mock_follow(self, flexed, duration, phase=None, t=None,
                    warmup=False) -> None:
        """Tell the mock sensors what the hand is being asked to do."""
        if hasattr(self.glove, "follow"):
            self.glove.follow(flexed, duration, phase, t=t, warmup=warmup)
        if self.args.mock_leap and hasattr(self.leap, "set_pose"):
            self.leap.set_pose("fist" if len(set(flexed)) == len(rp.FINGERS)
                               else "open_palm")

    def show(self, title: str, words: str, deadline: Optional[float] = None,
             sub: str = "") -> None:
        self.display = (title, words, deadline, sub)

    # --- one pass over both sensors ----------------------------------------
    def tick(self, record: bool = True) -> str:
        """Drain both sensors once, record what belongs in the take, redraw.
        Returns the key pressed on the cue window, if any."""
        now = time.time()
        for item in self.glove.drain(256):
            hand, raw = item
            result = validate_raw_message(raw)
            if not result.is_valid:
                self.warn(f"[{hand} glove] invalid packet: "
                          + "; ".join(result.errors))
                continue
            frame = parse_hand_message(raw, hand_side_hint=hand)
            for w in self.monitors[hand].update(frame.packet_counter):
                self.warn(f"[{hand} glove] {w}")
            self.glove_sides.add(frame.hand_side)
            if frame.hand_side != self.hand:
                self.glove_other += 1
                continue
            self.glove_total += 1
            t = float(getattr(item, "recv_time", 0.0) or now)
            if record and self.glove_rec is not None:
                self.glove_rec.record(frame, capture_time=t)
            if self.collect is not None:
                self.collect["glove"].append(
                    (t, flexion_features(frame_to_keypoints21(frame))))
        if self.leap is not None:
            for _side, lh in self.leap.drain(256):
                if lh.hand_side != self.hand:
                    self.other_at = now
                else:
                    self.cam_reading, self.cam_at = read_hand(lh), now
                if lh.visible_time_us < self.coached.MIN_VISIBLE_TIME_US:
                    continue                  # still settling; not data yet
                if record and self.cam_rec is not None:
                    self.cam_rec.record(lh)
                if lh.hand_side != self.hand:
                    continue
                self.last_hand = lh
                t = float(lh.capture_time if lh.capture_time is not None
                          else now)
                if record and self.cam_rec is not None:
                    self.cam_times.append(t)
                if self.collect is not None and lh.abs26:
                    self.collect["camera"].append(
                        (t, flexion_features([lh.abs26[i]
                                              for i in MP21_TO_OPENXR_IDX])))
        if now - self._rate_at >= 1.0:
            self._glove_hz = ((self.glove_total - self._rate_mark)
                              / (now - self._rate_at))
            self._rate_at, self._rate_mark = now, self.glove_total
        title, words, deadline, sub = self.display
        key = self.window.show(title, words,
                               None if deadline is None else deadline - now,
                               sub)
        time.sleep(0.004)
        return key

    def hold_until(self, deadline: float, record: bool = True) -> None:
        """Keep draining until `deadline`, asking for the still on time."""
        while True:
            now = time.time()
            if self._snap is not None and now >= self._snap[0]:
                _at, path, caption = self._snap
                self._snap = None
                self.stills.request(path, caption, self.last_hand)
            if now >= deadline:
                return
            self.tick(record)

    def reset_monitors(self) -> None:
        """Forget the last packet counter of each glove, after a stretch in
        which this process, not the glove, stopped reading the queue."""
        self.monitors = {"left": StreamMonitor("left"),
                         "right": StreamMonitor("right")}

    def discard_backlog(self) -> None:
        """Empty both queues before a take's files open, as the coached
        recorder does: what is queued is the past, not the take."""
        for _ in range(8):
            before = self.glove_total
            n = len(self.leap.drain(256)) if self.leap is not None else 0
            self.tick(record=False)
            if self.glove_total == before and n == 0:
                return

    # --- readiness ----------------------------------------------------------
    def wait_for_glove(self, timeout: float) -> None:
        source = "mock" if self.args.mock_glove else \
            f"OSC {self.args.host}:{self.args.port}"
        self.say(f"Waiting for the {self.hand.upper()} glove ({source})...")
        self.show("SETUP", f"waiting for the {self.hand} glove")
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.tick(record=False)
            if self.glove_total >= GLOVE_WAIT_PACKETS:
                self.say(f"  OK, glove streaming: {self.hand}")
                return
        if self.glove_sides and self.hand not in self.glove_sides:
            raise NotReady(
                f"The glove is streaming {', '.join(sorted(self.glove_sides))}"
                f", not {self.hand}. Put the glove on the {self.hand} hand, or "
                f"run with --hand "
                f"{'right' if self.hand == 'left' else 'left'}.")
        raise NotReady(
            f"Only {self.glove_total} glove packets from the {self.hand} hand "
            f"in {timeout:g} s. Check XR Trainer is streaming "
            "(python scripts/glove/run_osc.py --dump --no-viz).")

    def acquire(self, timeout: float) -> None:
        """The coached recorder's acquire gate, once, before the warm-up."""
        self.say("",
                 f"ACQUIRE: hold the {self.hand.upper()} hand OPEN over the "
                 f"camera, palm toward the lens, {band_text(self.band)} above "
                 "the module.",
                 "         Rest the other hand on the table.")
        t0 = time.time()
        while True:
            self.tick(record=False)
            now = time.time()
            fresh = (self.cam_reading
                     if now - self.cam_at < self.coached.HUD_STALE_S else None)
            bad = acquire_failures(fresh, self.hand, self.band)
            need = "need: " + ", ".join(bad) if bad else "hold still"
            self.show("SETUP", "OPEN PALM over the module", None, need)
            self.hud.show(hud_line(
                "ACQUIRE", timeout - (now - t0), fresh, self.hand, self.band,
                self._glove_hz,
                saw_other_hand=now - self.other_at < self.coached.HUD_STALE_S,
                extra=need if bad else ""), now)
            if not bad:
                self.say(f"  OK, camera has the {self.hand} hand "
                         f"({fresh.height_cm:.0f} cm, view "
                         f"{fresh.view_angle_deg:.0f} deg)")
                return
            if now - t0 > timeout:
                self.hud.close()
                raise NotReady(
                    f"no acquirable {self.hand} hand in {timeout:g} s (still "
                    f"missing: {', '.join(bad)}). Open palm, "
                    f"{band_text(self.band)} above the module, palm toward "
                    "the lens, other hand away.")

    # --- the folder -----------------------------------------------------------
    def make_folder(self, root: Path) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path(root) / self.protocol.name
        folder = base / f"{stamp}_{self.hand}"
        n = 1
        while folder.exists():                 # two sessions in one second
            n += 1
            folder = base / f"{stamp}_{self.hand}_{n}"
        for sub in ("glove", "events") + (("leap", "stills")
                                          if self.leap is not None else ()):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        self.dir = folder
        return folder

    def write_session(self, ended: Optional[str] = None) -> None:
        if self.dir is None:
            return
        data = dict(self.meta)
        data["ended"] = ended
        data["stopped"] = self.stopped
        data["takes"] = self.takes
        write_json(self.dir / "session.json", data)

    # --- the warm-up --------------------------------------------------------
    def run_warmup(self) -> dict:
        """Open palm then full fist, cued; each finger's endpoints for today."""
        all5 = rp.FINGERS
        self.say("", "WARM-UP: an open palm, then a full fist, on the beeps.")
        start = time.time()
        countdown = self.seconds(WARMUP_COUNTDOWN_S)
        self.mock_follow((), countdown, warmup=True)
        self.show("WARM-UP", "next: OPEN PALM, fingers straight",
                  start + countdown)
        self.hold_until(start + countdown, record=False)

        self.collect = {"glove": [], "camera": []}
        t_open = time.time()
        self.beep(*WARMUP_OPEN_BEEP)
        open_s = self.seconds(WARMUP_OPEN_S)
        self.mock_follow((), open_s, t=t_open, warmup=True)
        self.show("WARM-UP", "OPEN PALM, fingers straight", t_open + open_s)
        self.say(f"      open palm ({open_s:g} s)")
        self.hold_until(t_open + open_s, record=False)

        t_fist = time.time()
        self.beep(*WARMUP_FIST_BEEP)
        fist_s = self.seconds(WARMUP_FIST_S)
        self.mock_follow(all5, fist_s, t=t_fist, warmup=True)
        self.show("WARM-UP", "FULL FIST", t_fist + fist_s)
        self.say(f"      full fist ({fist_s:g} s)")
        self.hold_until(t_fist + fist_s, record=False)
        t_end = time.time()
        self.beep(*END_BEEP)
        self.mock_follow((), self.seconds(PREP_S), t=t_end, warmup=True)
        collected, self.collect = self.collect, None

        settle = self.seconds(WARMUP_SETTLE_S)
        record = rp.warmup_record(
            self.hand, (t_open + settle, t_fist), (t_fist + settle, t_end),
            collected["glove"],
            collected["camera"] if self.leap is not None else None,
            settle_s=settle)
        write_json(self.dir / "warmup.json", record)
        for sensor in ("glove", "camera"):
            ends = record[sensor]
            if ends is None:
                continue
            self.say(f"      {sensor:<6} open  " + "  ".join(
                f"{f} {ends['open'][f]:.3f}" for f in rp.FINGERS))
            self.say(f"      {sensor:<6} fist  " + "  ".join(
                f"{f} {ends['fist'][f]:.3f}" for f in rp.FINGERS))
            self.say(f"      {sensor:<6} span  " + "  ".join(
                f"{f} {ends['span'][f]:.3f}" for f in rp.FINGERS))
        self.warmup = record
        return record

    # --- one attempt at one take ----------------------------------------------
    def _item_hint(self, item: dict) -> str:
        if self.protocol.name == rp.FLEXION:
            words = rp.FINGER_WORDS[item["finger"]]
            return (f"{item['cycles']} cycles of the {words}: bend "
                    f"{item['bend_s']:g} s, hold {item['hold_s']:g} s, "
                    f"straighten {item['straighten_s']:g} s, rest "
                    f"{item['rest_s']:g} s. Other fingers straight.")
        return " > ".join(s["label"] for s in item["steps"])

    def _unique_name(self, item_id: str, take: int) -> str:
        t = time.time()
        while True:
            name = rp.take_name(item_id, self.hand, take,
                                time.strftime("%Y%m%d_%H%M%S",
                                              time.localtime(t)))
            clash = [self.dir / "glove" / f"{name}.jsonl",
                     self.dir / REJECTED / "glove" / f"{name}.jsonl"]
            if not any(p.exists() for p in clash):
                return name
            t += 1.0

    def _paths(self, name: str, root: Optional[Path] = None) -> dict:
        root = self.dir if root is None else root
        out = {"glove": root / "glove" / f"{name}.jsonl",
               "events": root / "events" / f"{name}.events.jsonl"}
        if self.leap is not None:
            out["leap"] = root / "leap" / f"{name}.jsonl"
            out["still"] = root / "stills" / f"{name}.png"
        return out

    def _rel(self, path: Path) -> str:
        return Path(path).relative_to(self.dir).as_posix()

    def run_take(self, item_id: str, take: int, attempt: int, round_no: int
                 ) -> dict:
        item = self.protocol.item(item_id)
        cues = rp.build_schedule(self.protocol, item_id, self.scale)
        label = item["label"]
        hint = self._item_hint(item)
        self._warned = set()

        # --- preparation: the item's words, hand open -----------------
        self.say("", f"--- round {round_no}/{len(self.plan)}: {label}, take "
                     f"{take} (attempt {attempt}) ---", f"    {hint}")
        prep = self.seconds(PREP_S)
        t_prep = time.time()
        self.beep(*PREP_BEEP)
        self.mock_follow((), prep, t=t_prep)
        self.show(f"NEXT: {item_id} take {take}", label, t_prep + prep, hint)
        self.hold_until(t_prep + prep, record=False)

        # --- the take --------------------------------------------------
        name = self._unique_name(item_id, take)
        paths = self._paths(name)
        keys = {"session": self.dir.name, "item": item_id}
        self.discard_backlog()
        self.glove_rec = self.coached.StampedFrameRecorder(hz=None, take=take)
        self.glove_rec.start(paths["glove"])
        stamp_keys(self.glove_rec, keys)
        if self.leap is not None:
            self.cam_rec = LeapRecorder(hz=None, take=take)
            self.cam_rec.start(paths["leap"])
            stamp_keys(self.cam_rec, keys)
        self.cam_times = []
        events = rp.EventLog(paths["events"])
        offsets = rp.cue_offsets(cues)
        total = rp.schedule_seconds(cues)
        k_still = rp.still_cue(cues)
        t0 = time.time()
        events.write("take_start", t=t0, item=item_id, take=take,
                     attempt=attempt, name=name)
        if "still" in paths:
            self._snap = (t0 + offsets[k_still] + cues[k_still].hold_s / 2.0,
                          paths["still"], f"{name}  {cues[k_still].label}")
        interrupted = False
        previous = None
        try:
            for k, cue in enumerate(cues):
                self.hold_until(t0 + offsets[k])
                t = time.time()
                self.beep(rp.cue_pitch(cue, previous),
                          max(40, min(BEEP_MS, int(cue.hold_s * 600))))
                events.write("cue", t=t, **cue.event_fields())
                self.mock_follow(cue.flexed, cue.hold_s, cue.phase, t=t)
                # `step` as the events file and the reject reasons count it
                # (from 0), so "step 3" means the same cue everywhere.
                where = (f"cycle {cue.cycle}/{item['cycles']}"
                         if cue.cycle is not None else f"step {cue.step}")
                self.show(name, cue.label.upper(),
                          t0 + offsets[k] + cue.hold_s, where)
                self.say(f"      [{k + 1:>2}/{len(cues)}] {where:<11} "
                         f"{cue.label}  ({cue.hold_s:g} s)")
                previous = cue
            self.hold_until(t0 + total)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            t1 = time.time()
            events.write("take_end", t=t1, item=item_id, take=take)
            events.close()
            self.glove_rec.stop()
            self.glove_rec = None
            if self.cam_rec is not None:
                cam_frames = self.cam_rec.count
                self.cam_rec.stop()
                self.cam_rec = None
            else:
                cam_frames = None
            self._snap = None
            self.beep(*END_BEEP)
            self.mock_follow((), self.seconds(PAUSE_S), t=t1)

        # --- the still, then the quick check --------------------------
        still_name = still_missing = None
        curls: list = []
        check = rp.CheckResult(False, "interrupted before the check")
        try:
            if "still" in paths:
                wait_end = time.time() + (0.0 if interrupted
                                          else STILL_WAIT_S)
                while not self.stills.ready(paths["still"]) and \
                        time.time() < wait_end:
                    self.tick(record=False)
                still_name, still_missing = still_status(paths["still"])
            curls = rp.read_curls(paths["glove"], self.hand)
            check = rp.check_take(self.protocol, item_id, curls,
                                  rp.read_events(paths["events"]),
                                  self.warmup["glove"], self.scale)
        except KeyboardInterrupt:
            # Ctrl+C while the take was being checked: it gets a verdict
            # like any other interrupted take, so no file is left behind
            # without an entry in session.json.
            interrupted = True
        # The queues were not drained while the take was read back; a
        # counter gap that opens now is ours, not the glove's.
        self.reset_monitors()
        health = stream_health([t for t, _c in curls], t0, t1)
        if interrupted:
            accepted, reason, by = (False, "interrupted by the operator "
                                    "(Ctrl+C) during the take", "operator")
        else:
            accepted, reason, by = check.accepted, check.reason, "auto"
        decided_at = time.time()
        rp.append_event(paths["events"], "decision", t=decided_at,
                        accepted=accepted, reason=reason, by=by)
        entry = {
            "item": item_id, "take": take, "name": name,
            "accepted": accepted, "reason": reason,
            "decided_at": round(decided_at, 6), "decided_by": by,
            "files": {"glove": self._rel(paths["glove"]),
                      "leap": (self._rel(paths["leap"])
                               if "leap" in paths else None),
                      "events": self._rel(paths["events"]),
                      "still": (self._rel(paths["still"])
                                if still_name else None)},
            "round": round_no, "attempt": attempt,
            "t_start": round(t0, 6), "t_end": round(t1, 6),
            "glove_frames": len(curls),
            "glove_rate_hz": health.get("rate_hz"),
            "glove_max_gap_ms": health.get("max_gap_ms"),
            "camera_frames": cam_frames,
            "camera_coverage": (round(coverage(self.cam_times, t0, t1), 4)
                                if self.leap is not None else None),
            "still_missing": (still_missing if "still" in paths
                              and not still_name else None),
            "check": check.as_dict(),
        }
        if not accepted:
            self.reject(entry, by)
        self.takes.append(entry)
        self.write_session()

        verdict = "ACCEPTED" if accepted else f"REJECTED: {reason}"
        self.say(f"      {verdict}")
        for note in check.notes:
            self.say(f"      note: {note}")
        gap = health.get("max_gap_ms") or 0
        if gap > self.coached.GLOVE_GAP_WARN_S * 1000:
            self.say(f"      ! the glove stream stopped for {gap:.0f} ms "
                     f"during this take ({health.get('rate_hz')} Hz overall)")
        cover = entry["camera_coverage"]
        if cover is not None and cover < CAMERA_COVERAGE_NOTE:
            # Not a reason to reject (the camera is the reference, not the
            # subject), but the operator can fix the hand's position now.
            self.say(f"      note: the camera tracked the {self.hand} hand "
                     f"for {cover * 100:.0f} % of the take; keep it "
                     f"{band_text(self.band)} above the module, palm to lens")
        if interrupted:
            raise KeyboardInterrupt
        return entry

    def reject(self, entry: dict, by: str) -> None:
        """Move one attempt's files under rejected/, with the reason. Never
        delete: an exclusion nobody can look at is one nobody can check."""
        name = entry["name"]
        here = self._paths(name)
        there = self._paths(name, self.dir / REJECTED)
        files = {"glove": None, "leap": None, "events": None, "still": None}
        for key, src in here.items():
            dst = there[key]
            if key == "still":
                move_still(src, dst)
                if dst.is_file():
                    files[key] = self._rel(dst)
                continue
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
                files[key] = self._rel(dst)
        entry["files"] = files
        note = self.dir / REJECTED / f"{name}.reason.txt"
        note.parent.mkdir(parents=True, exist_ok=True)
        when = datetime.datetime.fromtimestamp(entry["decided_at"])
        note.write_text(
            f"{entry['reason']}\n"
            f"decided by: {by}\n"
            f"decided at: {when.isoformat(timespec='seconds')} "
            f"({entry['decided_at']:.3f})\n"
            f"item: {entry['item']}  take: {entry['take']}  attempt: "
            f"{entry['attempt']}\n", encoding="utf-8")

    def operator_redo(self, entry: dict) -> None:
        """`r` in the pause: the operator throws out a take the check kept."""
        decided_at = time.time()
        reason = "redo asked by the operator"
        events = self.dir / entry["files"]["events"]
        rp.append_event(events, "decision", t=decided_at, accepted=False,
                        reason=reason, by="operator")
        entry.update({"accepted": False, "reason": reason,
                      "decided_at": round(decided_at, 6),
                      "decided_by": "operator"})
        self.reject(entry, "operator")
        self.write_session()
        self.say(f"      REDO: {entry['name']} moved to {REJECTED}/")

    def pause(self, entry: dict) -> str:
        """The few seconds after a take: 'r' redo, 'q' stop, or nothing."""
        flush_console_keys()
        end = time.time() + self.seconds(PAUSE_S)
        words = ("ACCEPTED" if entry["accepted"]
                 else f"REJECTED: {entry['reason']}")
        self.show(entry["name"], words, end,
                  "press r to redo this take, q to stop the session")
        while time.time() < end:
            key = self.tick(record=False) or console_key()
            if key in ("r", "q"):
                return key
        return ""

    def record_take(self, item_id: str, round_no: int) -> None:
        """One planned take: attempts until one is accepted, or the retries
        run out. The operator's own redo does not use up a retry."""
        take = self.accepted[item_id] + 1
        auto_rejects = 0
        attempt = 0
        while True:
            attempt += 1
            entry = self.run_take(item_id, take, attempt, round_no)
            key = self.pause(entry)
            if key == "r" and entry["accepted"]:
                self.operator_redo(entry)
                continue
            if key == "q":
                if entry["accepted"]:
                    self.accepted[item_id] += 1
                raise StopSession("stopped by the operator (q)")
            if entry["accepted"]:
                self.accepted[item_id] += 1
                return
            auto_rejects += 1
            if auto_rejects > self.args.retries:
                self.say(f"      FAILED: {item_id} take {take} after "
                         f"{attempt} attempt(s)")
                self.failed.append(f"{item_id} take {take}")
                return
            self.say(f"      retrying ({auto_rejects}/{self.args.retries})")

    def run_rounds(self) -> None:
        for r, order in enumerate(self.plan, 1):
            for item_id in order:
                self.record_take(item_id, r)

    # --- the end --------------------------------------------------------------
    def summary(self) -> None:
        self.say("", "=" * 72,
                 f"Session {self.dir.name if self.dir else '(no folder)'}: "
                 f"{sum(1 for t in self.takes if t['accepted'])} accepted, "
                 f"{sum(1 for t in self.takes if not t['accepted'])} rejected")
        if self.takes:
            self.say(f"  {'item':<22} {'take':>4}  {'accepted':<8}  reason")
            for t in self.takes:
                self.say(f"  {t['item']:<22} {t['take']:>4}  "
                         f"{'yes' if t['accepted'] else 'no':<8}  "
                         f"{t['reason']}")
        if self.failed:
            self.say("", "  takes with no accepted attempt: "
                     + ", ".join(self.failed))
        if self.stopped:
            self.say("", f"  stopped: {self.stopped}")
        if self.dir is not None:
            self.say("", f"  folder: {self.dir.resolve()}")

    def close(self) -> None:
        self.hud.close()
        self.beeper.stop()
        self.window.close()
        for closer in (getattr(self.stills, "close", None),
                       getattr(self.glove, "stop", None),
                       getattr(self.leap, "stop", None)):
            if closer is None:
                continue
            try:
                closer()
            except Exception:
                pass


# --- command line ------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Cue-driven glove + camera recorder for the professor's "
                    "finger-flexion (Set B) and sequence (Set C) protocols.")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--set", choices=SET_CHOICES,
                       help="record protocols/<set>.json")
    which.add_argument("--protocol", type=Path,
                       help="record this protocol file instead")
    p.add_argument("--hand", required=True, choices=("left", "right"),
                   help="the operator's hand for this run (the other rests "
                        "on the table)")
    p.add_argument("--camera", default="leap", choices=CAMERA_CHOICES,
                   help="leap (default): the Ultraleap as the reference; "
                        "none: glove only")
    p.add_argument("--out-dir", type=Path, default=None,
                   help=f"where the session folder goes (default: {OUT_DIR}, "
                        f"or {MOCK_OUT_DIR} with a mock sensor)")
    p.add_argument("--items", default=None,
                   help="comma-separated item ids to record (default: all)")
    p.add_argument("--takes", type=int, default=None,
                   help="takes per item (default: the protocol's "
                        "takes_per_item)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the shuffled round order (default: a new "
                        "one, saved in session.json)")
    p.add_argument("--retries", type=int, default=2,
                   help="automatic retries after a rejected attempt "
                        "(default 2)")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="multiply every cue duration (rehearsals and tests; "
                        "a real session uses 1.0)")
    p.add_argument("--calibrated-at", default=None, metavar="ISO",
                   help="when both gloves were calibrated in XR Trainer, e.g. "
                        "2026-09-28T14:00:00 (skips the Enter prompt)")
    p.add_argument("--operator", default="N Kim")
    p.add_argument("--band", default=None, metavar="LOW,HIGH",
                   help=f"palm height band in cm for the acquire gate "
                        f"(default {DEFAULT_BAND[0]:g},{DEFAULT_BAND[1]:g})")
    p.add_argument("--acquire-timeout", type=float, default=60.0)
    p.add_argument("--glove-timeout", type=float, default=120.0)
    p.add_argument("--mode", default="desktop",
                   choices=("desktop", "hmd", "screentop"),
                   help="Ultraleap tracking mode")
    p.add_argument("--mock-glove", action="store_true",
                   help="synthetic glove that follows the cues")
    p.add_argument("--mock-glove-ignore-cues", action="store_true",
                   help="synthetic glove that does the warm-up and then "
                        "stays open (rehearses the reject path)")
    p.add_argument("--mock-leap", action="store_true",
                   help="synthetic Ultraleap stream")
    p.add_argument("--no-view", action="store_true",
                   help="no cue window (the console still shows every cue)")
    p.add_argument("--no-open", action="store_true",
                   help="do not open the folder in Explorer at the end")
    p.add_argument("--no-beep", action="store_true",
                   help="silent cues (tests)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9002)
    p.add_argument("--address", default="/v1/animation/kinematic/all")
    args = p.parse_args(argv)

    if args.mock_glove_ignore_cues:
        args.mock_glove = True
    args.mock = bool(args.mock_glove or args.mock_leap)
    if args.out_dir is None:
        args.out_dir = MOCK_OUT_DIR if args.mock else OUT_DIR
    if args.mock_leap and args.camera != "leap":
        p.error("--mock-leap only means anything with --camera leap")
    if args.retries < 0:
        p.error("--retries cannot be negative")
    if args.takes is not None and args.takes < 1:
        p.error("--takes must be at least 1")
    if not args.time_scale > 0:
        p.error("--time-scale must be above 0")
    try:
        args.band = parse_band(args.band) if args.band else DEFAULT_BAND
    except ValueError as e:
        p.error(str(e))
    if args.calibrated_at is not None:
        try:
            args.calibrated_at = datetime.datetime.fromisoformat(
                args.calibrated_at).isoformat(timespec="seconds")
        except ValueError:
            p.error(f"--calibrated-at wants an ISO time such as "
                    f"2026-09-28T14:00:00, got {args.calibrated_at!r}")
    return args


def ask_calibration(args) -> Optional[str]:
    """The one keyboard step: Enter once both gloves are calibrated."""
    if args.calibrated_at is not None:
        print(f"XR Trainer calibration time given: {args.calibrated_at}")
        return args.calibrated_at
    print("Calibrate BOTH gloves in XR Trainer now. Press Enter here when "
          "that is done.")
    try:
        input()
    except EOFError:
        print("  (no console to press Enter in; the calibration time is not "
              "recorded)")
        return None
    return iso_now()


def run(args) -> int:
    path = args.protocol if args.protocol is not None else \
        ROOT / "protocols" / f"{args.set}.json"
    try:
        protocol = rp.load_protocol(path)
        if protocol.name not in SET_CHOICES:
            raise rp.ProtocolError(
                f"{path}: Set A ({protocol.name}) is recorded by "
                "scripts/leap/record_poses.py --protocol, not by this recorder")
        items = rp.select_items(protocol, (args.items or "").split(",")
                                if args.items else None)
    except rp.ProtocolError as e:
        print(f"\nProtocol file problem: {e}\n")
        return EXIT_REFUSED
    coached()        # a few seconds of imports, before the Enter, not after
    takes = args.takes if args.takes is not None else protocol.takes_per_item
    shuffle = protocol.shuffle_rounds
    seed = args.seed
    if shuffle and seed is None:
        seed = random.SystemRandom().randrange(1, 2 ** 31)
    plan = rp.rounds(items, takes, seed=seed, shuffle=shuffle)

    camera = args.camera
    setup = ("gloves on + camera, one hand" if camera == "leap"
             else "gloves on, no camera, one hand")
    n_takes = sum(len(r) for r in plan)
    per_take = {i: rp.schedule_seconds(rp.build_schedule(protocol, i,
                                                         args.time_scale))
                for i in items}
    eta = sum(per_take[i] + (PREP_S + PAUSE_S) * args.time_scale
              for r in plan for i in r)
    print("=" * 72)
    print(f"PROTOCOL {protocol.name} v{protocol.version}: {n_takes} takes in "
          f"{len(plan)} round(s), about {eta / 60:.1f} min of recording")
    print(f"  Setup: {setup}. The {args.hand.upper()} hand works; the other "
          "hand rests on the table.")
    print(f"  items: {', '.join(items)}")
    if shuffle:
        print(f"  rounds shuffled, seed {seed}")
    glove_text = ("MOCK (ignores the cues)" if args.mock_glove_ignore_cues
                  else "MOCK (follows the cues)" if args.mock_glove
                  else f"{args.host}:{args.port}")
    cam_text = ("none" if camera == "none" else
                "MOCK leap" if args.mock_leap else "Ultraleap SIR 170")
    print(f"  glove: {glove_text}   camera: {cam_text}")
    if args.mock:
        print(f"  MOCK session: written under {args.out_dir} and marked "
              "\"mock\": true, so it is never packaged as data")
    if args.time_scale != 1.0:
        print(f"  time scale {args.time_scale:g}: every duration is scaled "
              "(rehearsal, not a real session)")
    print("  Follow the beeps: high = bend or flex, low = straighten or "
          "extend. The window shows only the words.")
    print("=" * 72)

    calibrated_at = ask_calibration(args)
    started = iso_now()

    if args.mock_glove:
        glove = CueFollowingGlove(args.hand,
                                  follow_cues=not args.mock_glove_ignore_cues,
                                  time_scale=args.time_scale)
    else:
        glove = OSCHandReceiver(host=args.host, port=args.port,
                                kinematic_addr=args.address)
    leap = None
    stills = NoStills()
    try:
        glove.start()
    except OSError as e:
        print(f"\nCannot listen for the glove on {args.host}:{args.port} "
              f"({e}). Is another recorder running?\n")
        return EXIT_REFUSED
    if camera == "leap":
        try:
            leap = open_stream(mock=args.mock_leap, mode=args.mode,
                               pose="open_palm" if args.mock_leap else None)
        except LeapUnavailable as e:
            glove.stop()
            print(f"\nNo live tracking: {e}\n")
            return EXIT_REFUSED
        stills = MockStills() if args.mock_leap else \
            ViewerStills(args.hand, args.band)

    beep = (lambda f, ms: None) if args.no_beep else coached().beep
    session = ProtocolSession(args, protocol, items, plan, seed, glove, leap,
                              stills, CueWindow(enabled=not args.no_view),
                              beep)
    session.meta = {
        "set": protocol.name,
        "protocol_file": display_path(protocol.path),
        "protocol_name": protocol.name,
        "protocol_version": protocol.version,
        "protocol_sha256": protocol.sha256,
        "hand": args.hand,
        "operator": args.operator,
        "camera": camera,
        "glove": True,
        "started": started,
        "ended": None,
        "xr_trainer_calibrated_at": calibrated_at,
        "seed": seed if shuffle else None,
        "rounds": plan,
        "tool_commit": tool_commit(),
        "takes": [],
        "session": None,
        "items": items,
        "takes_per_item": takes,
        "retries": args.retries,
        "time_scale": args.time_scale,
        # The packager refuses a session with "mock": true (contract 7);
        # which sensor was synthetic, and how, is in "mock_flags".
        "mock": args.mock,
        "mock_flags": {"glove": bool(args.mock_glove),
                       "glove_follows_cues": bool(
                           args.mock_glove and not args.mock_glove_ignore_cues),
                       "leap": bool(args.mock_leap)},
        "warmup": "warmup.json",
        "stopped": None,
    }
    code = EXIT_OK
    try:
        session.wait_for_glove(args.glove_timeout)
        if leap is not None:
            session.acquire(args.acquire_timeout)
        folder = session.make_folder(args.out_dir)
        session.meta["session"] = folder.name
        session.write_session()
        warm = session.run_warmup()
        if warm["refused"]:
            session.stopped = f"warm-up refused: {warm['refused']}"
            raise NotReady(f"warm-up: {warm['refused']}")
        session.write_session()
        session.run_rounds()
        if session.failed:
            code = EXIT_INCOMPLETE
    except NotReady as e:
        session.say("", f"REFUSED: {e}")
        if session.stopped is None:
            session.stopped = f"refused: {e}"
        code = EXIT_REFUSED
    except StopSession as e:
        session.stopped = str(e)
        code = EXIT_INCOMPLETE
    except KeyboardInterrupt:
        session.stopped = "interrupted by the operator (Ctrl+C)"
        code = EXIT_INCOMPLETE
    finally:
        session.close()
        session.write_session(ended=iso_now())
        session.summary()
    if session.dir is not None and not args.no_open and \
            hasattr(os, "startfile"):
        try:
            os.startfile(str(session.dir.resolve()))
        except OSError:
            pass
    return code


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
