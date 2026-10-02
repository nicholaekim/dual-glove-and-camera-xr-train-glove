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

  * the cue window shows the take name, the cue words, a countdown and,
    along its bottom third, five bars: each finger's GLOVE reading as a
    fraction of its range from today's warm-up, the cued finger drawn wide,
    green when it is where the cue wants it and amber when not. On
    2026-10-01 three Set B sessions ended on the thumb because nobody could
    tell whether the glove had registered a bend at all. The bars show the
    glove against its own warm-up, never against the camera, so the two
    sensors under comparison are still not put side by side for the
    operator to make them agree; the operator follows the beep.
    `--no-bars` keeps the plan's cue-only screen.
  * every take is checked before the next one starts (the quick check in
    `recording_protocol`), because a bad take is cheap to redo while the
    gloves are on and expensive to discover afterwards. A rejected attempt
    is MOVED to `rejected/` with its reason, never deleted: the two sensors
    under evaluation are the ones judging, so every exclusion has to stay
    countable.
  * each finger is measured against its own range for THIS session (the
    warm-up: open palm, full fist, then each finger bent on its own), not a
    number from another day. A glove that barely moves between open and
    fist, or a finger that barely bends on its own, refuses the warm-up
    instead of producing fractions of nothing; it is repeated up to
    WARMUP_TRIES times before the session is refused. The glove's thumb is
    measured by its joint angles in degrees (`recording_protocol.
    glove_bends`), because its curl barely moves when it bends.
  * a session cut short (Ctrl+C, `q`, a refused warm-up) carries on in the
    same folder with `--resume <folder>` or `--resume latest`: the saved
    round order is reused, items that already have their accepted takes are
    skipped, and a new warm-up (`warmup_<HHMMSS>.json`) is recorded, because
    the gloves came off and their readings with them.
  * Set C runs all seven sequences once per round in a shuffled order, with
    the seed saved, so the three takes of a sequence are independent
    repetitions rather than practice.
  * `session.json` is rewritten after every take, so a crash or a Ctrl+C
    keeps everything done so far, with the reason for every rejection.

Pieces reused rather than rewritten: the glove recorder with the packet
arrival time (`StampedFrameRecorder`) and the beep from
`scripts/record_simultaneous.py`; the acquire gate, HUD, async beeper, crop
box and still bookkeeping from `leap_hand.protocol`; `LeapRecorder`; the
camera viewer (`scripts/leap/camera_view.py`), which shows the IR image
and the skeleton so the operator can see the hand (`--hide-camera` runs it
with no window, only to cut the one hand-cropped IR still per take).

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
verdict, which does not use up a retry), `s` skips the rest of that item in
this run and `q` stops the session, on the cue window or in the console.
During a take, `s` ends it at once: the attempt is rejected ("skipped by
the operator (s) during the take") and the item is skipped. Carry on later:

  python scripts/record_protocol.py --set finger_flexion --hand left \\
      --resume latest

Exit code 0 when every planned take has an accepted attempt, 1 when some
take has none, an item was skipped with no accepted take or the session was
stopped, 2 when the session was refused before any take (protocol file,
glove, camera, acquire gate, warm-up, or a folder that cannot be resumed).
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
from typing import Dict, List, Optional, Tuple

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
from xr_hand.keypoints21 import MP21_TO_OPENXR_IDX  # noqa: E402
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
# Then each finger on its own, thumb to little finger: bent alone and held
# for WARMUP_SINGLE_S (its median over the last WARMUP_SETTLE_S of that is
# its `single`), then straightened for WARMUP_SINGLE_REST_S. The fist ends
# with one straighten too, so no finger starts its bend from the fist.
WARMUP_SINGLE_S = 3.0
WARMUP_SINGLE_REST_S = 1.5
# A refused warm-up is done again, whole, this many times in all before the
# session is refused: the usual cause is a bend that stopped short, which the
# refusal names, and the gloves are on and calibrated by then.
WARMUP_TRIES = 3
# The live bars only draw a glove frame this recent; older means the glove
# stopped, and an old bar would say the finger is somewhere it is not.
BARS_FRESH_S = 0.5
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
WARMUP_BEND_BEEP = (1200, 200)        # high: bend this finger on its own
WARMUP_STRAIGHTEN_BEEP = (800, 200)   # low: straighten it

REJECTED = "rejected"
WARMUP_FILE = "warmup.json"
SKIP_KEY = "s"
SKIPPED_DURING_TAKE = "skipped by the operator (s) during the take"
SKIPPED_TEXT = "skipped by the operator"
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
    """What the operator looks at: the take name, the cue words, a countdown,
    and the glove's five bars (module docstring).

    A plain OpenCV canvas, kept on top, redrawn at most 30 times a second.
    `render` draws one frame and touches no window, so it can be tested and
    looked at without a screen; `show` puts it on the window.
    `enabled=False` (`--no-view`) makes `show` a no-op.

    The bars sit in the bottom third, thumb to little finger, each filled
    from the bottom (0 = the warm-up open palm) to the top (1 = the end of
    the finger's range), the fill clipped to [0, 1] and the number printed
    as it is. Ticks on both sides of every bar mark STRAIGHT_BELOW and
    FLEXED_ABOVE. The cued finger(s) are wide and bright: green when the
    reading is where the phase wants it (at or above FLEXED_ABOVE to bend,
    flex or hold; at or below STRAIGHT_BELOW to straighten, extend or rest),
    amber when it is not, white when there is no phase to judge by (the
    pause). The others are narrow and grey.
    """

    NAME = "Protocol cues"
    W, H = 1000, 560
    EVERY = 1.0 / 30.0
    # Colours (BGR) and sizes of the bars.
    GREEN = (0, 200, 0)
    AMBER = (0, 170, 255)
    NEUTRAL = (235, 235, 235)
    GREY = (120, 120, 120)
    TRACK = (70, 70, 70)
    TICK = (200, 200, 200)
    CUED_W, OTHER_W = 70, 26
    LABEL_H = 28                       # finger names under the bars
    BENT_PHASES = ("bend", "flex", "hold")
    OPEN_PHASES = ("straighten", "extend", "rest")
    BAR_LABELS = {"thumb": "thumb", "index": "index", "middle": "middle",
                  "ring": "ring", "pinky": "little"}

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

    @classmethod
    def bar_colour(cls, fraction: Optional[float],
                   phase: Optional[str]) -> Tuple[int, int, int]:
        """A cued bar's colour: green when the reading is where `phase`
        wants it, amber when not, neutral with no phase to judge by."""
        if phase in cls.BENT_PHASES:
            ok = fraction is not None and fraction >= rp.FLEXED_ABOVE
        elif phase in cls.OPEN_PHASES:
            ok = fraction is not None and fraction <= rp.STRAIGHT_BELOW
        else:
            return cls.NEUTRAL
        return cls.GREEN if ok else cls.AMBER

    @classmethod
    def _draw_bars(cls, img: np.ndarray, bars: dict) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fractions = bars.get("fractions") or {}
        cued = set(bars.get("cued") or ())
        phase = bars.get("phase")
        top = cls.H - cls.H // 3                  # the bottom third
        y_top = top + 8
        y_bot = cls.H - cls.LABEL_H
        height = y_bot - y_top
        slot = cls.W // len(rp.FINGERS)
        for k, f in enumerate(rp.FINGERS):
            cx = k * slot + slot // 2
            is_cued = f in cued
            half = (cls.CUED_W if is_cued else cls.OTHER_W) // 2
            x0, x1 = cx - half, cx + half
            v = fractions.get(f)
            colour = cls.bar_colour(v, phase) if is_cued else cls.GREY
            cv2.rectangle(img, (x0, y_top), (x1, y_bot), cls.TRACK, 1)
            if v is not None:
                fill = min(1.0, max(0.0, float(v)))
                y_fill = y_bot - int(round(fill * height))
                if y_fill < y_bot:
                    cv2.rectangle(img, (x0, y_fill), (x1, y_bot), colour, -1)
            for mark in (rp.STRAIGHT_BELOW, rp.FLEXED_ABOVE):
                y = y_bot - int(round(mark * height))
                cv2.line(img, (x0 - 9, y), (x0 - 2, y), cls.TICK, 1)
                cv2.line(img, (x1 + 2, y), (x1 + 9, y), cls.TICK, 1)
            text = "-" if v is None else f"{float(v):.2f}"
            shade = (255, 255, 255) if is_cued else (150, 150, 150)
            cv2.putText(img, text, (x1 + 12, y_top + 18), font, 0.55, shade,
                        1, cv2.LINE_AA)
            label = cls.BAR_LABELS.get(f, f)
            size = cv2.getTextSize(label, font, 0.6, 1)[0][0]
            cv2.putText(img, label, (cx - size // 2, cls.H - 8), font, 0.6,
                        shade, 1, cv2.LINE_AA)

    @classmethod
    def render(cls, title: str, words: str, seconds_left: Optional[float],
               sub: str = "", bars: Optional[dict] = None) -> np.ndarray:
        """One frame of the window as a BGR image; no window is touched.

        `bars` is {"fractions": {finger: float or None}, "cued": [finger,
        ...], "phase": "bend" | "hold" | "straighten" | "rest" | "flex" |
        "extend" | None} (`ProtocolSession.bars_now`), or None for the
        cue-only screen. With bars the words move up and the countdown goes
        to the top right, so the bottom third holds the bars alone.
        """
        img = np.zeros((cls.H, cls.W, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, title or "", (24, 44), font, 0.8, (170, 170, 170), 2,
                    cv2.LINE_AA)
        lines = cls._wrap(words or "", 1.8, 4, cls.W - 80)
        if len(lines) > 2:
            lines = cls._wrap(words or "", 1.0, 2, cls.W - 60)[:5]
            scale, thick, step = 1.0, 2, 44
        else:
            scale, thick, step = 1.8, 4, 74
        middle = 0.30 if bars is not None else 0.42
        y = int(cls.H * middle) - (len(lines) - 1) * step // 2
        if bars is not None:
            y = max(y, 100)
        for line in lines:
            size = cv2.getTextSize(line, font, scale, thick)[0][0]
            cv2.putText(img, line, ((cls.W - size) // 2, y), font, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
            y += step
        floor = cls.H - cls.H // 3 - 8 if bars is not None else cls.H
        for line in cls._wrap(sub or "", 0.7, 2, cls.W - 60)[:2]:
            if bars is not None and y + 10 > floor:
                break
            size = cv2.getTextSize(line, font, 0.7, 2)[0][0]
            cv2.putText(img, line, ((cls.W - size) // 2, y + 10), font, 0.7,
                        (170, 170, 170), 2, cv2.LINE_AA)
            y += 34
        if seconds_left is not None:
            left = max(0.0, float(seconds_left))
            text = f"{left:.1f}" if left < 1.0 else f"{math.ceil(left):d}"
            if bars is None:
                size = cv2.getTextSize(text, font, 2.2, 5)[0][0]
                cv2.putText(img, text, ((cls.W - size) // 2, cls.H - 40),
                            font, 2.2, (0, 220, 255), 5, cv2.LINE_AA)
            else:
                size = cv2.getTextSize(text, font, 1.6, 4)[0][0]
                cv2.putText(img, text, (cls.W - size - 24, 60), font, 1.6,
                            (0, 220, 255), 4, cv2.LINE_AA)
        if bars is not None:
            cls._draw_bars(img, bars)
        return img

    def show(self, title: str, words: str, seconds_left: Optional[float],
             sub: str = "", force: bool = False,
             bars: Optional[dict] = None) -> str:
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
        img = self.render(title, words, seconds_left, sub, bars)
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


def console_has_key() -> bool:
    """Is a key waiting in the console? Only when a person is at it.

    Asked before reading a key DURING a take, so a key is read there only
    when one was pressed; the pause after the take reads `console_key` as
    it always has (and flushes what was typed during the take first)."""
    try:
        if not sys.stdin.isatty():
            return False
        import msvcrt
        return bool(msvcrt.kbhit())
    except Exception:
        return False


# --- the one still per take --------------------------------------------------
class HiddenCameraView(CameraView):
    """`scripts/leap/camera_view.py` with `--no-window`: stills, no picture.

    The viewer already knows how to cut a still to the tracked hand (and to
    leave a note when there is no hand), which is the still the contract
    asks for. This one shows nothing while the operator performs (the
    plan's cue-only screen, `--hide-camera`). It is no longer the default:
    on 2026-10-01 the operator could not place the hand or see how far the
    thumb bent without the picture, and two Set B sessions ended on the
    first take. `CameraView.start` builds its command inline, which is why
    this one is repeated with the one flag added.
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
    """Stills from the real camera, through the viewer: its window (the IR
    image, the skeleton, height against the band, the tracking line) by
    default, or hidden with `window=False` (`--hide-camera`)."""

    def __init__(self, hand: str, band, window: bool = True):
        cls = CameraView if window else HiddenCameraView
        self.view = cls(hand=hand, band=band).start()

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


class SkipTake(Exception):
    """`s` during a take: end it now, reject it, skip the item."""


class ResumeRefused(Exception):
    """`--resume` names a folder this run cannot carry on. Exit code 2."""


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
        self.warmup_name = WARMUP_FILE
        self.dir: Optional[Path] = None
        self.takes: List[dict] = []
        self.accepted: Dict[str, int] = {i: 0 for i in items}
        # Every round records each item once, so a plan's length is its
        # takes per item; `run` sets it from the protocol or the session.
        self.takes_per_item = len(plan)
        self.failed: List[str] = []
        self.skipped_items: set = set()       # `s`: this run only
        self.skips: List[dict] = []
        self.stopped: Optional[str] = None
        self.meta: dict = {}
        # The live bars: the newest glove frame of the operator's hand as
        # (time, five readings), the warm-up block the fractions are taken
        # against, and (cued fingers, phase) while bars are wanted.
        self.show_bars = not getattr(args, "no_bars", False)
        self.last_bends: Optional[Tuple[float, List[float]]] = None
        self.bar_ends: Optional[dict] = None
        self.bar_cue: Optional[Tuple[List[str], Optional[str]]] = None

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

    def bars_now(self, now: Optional[float] = None) -> Optional[dict]:
        """The cue window's bars from the newest glove frame, or None.

        None outside the stretches that show bars (a take, the warm-up's
        single-finger part, the pause after a take), before the warm-up has
        endpoints, and when the newest frame is older than BARS_FRESH_S. A
        cued finger is put on its own range (`recording_protocol.
        range_ends`), the others on the fist, as the quick check reads them.
        """
        if self.bar_cue is None or not self.bar_ends or \
                self.last_bends is None:
            return None
        now = time.time() if now is None else float(now)
        t, bends = self.last_bends
        if now - t > BARS_FRESH_S:
            return None
        cued, phase = self.bar_cue
        fractions: Dict[str, Optional[float]] = {}
        for i, f in enumerate(rp.FINGERS):
            try:
                o, e = rp.range_ends(self.bar_ends, f, cued=f in cued)
            except (KeyError, TypeError, ValueError):
                fractions[f] = None
                continue
            fractions[f] = rp.fraction(o, e, bends[i])
        return {"fractions": fractions, "cued": list(cued), "phase": phase}

    # --- one pass over both sensors ----------------------------------------
    def tick(self, record: bool = True) -> str:
        """Drain both sensors once, record what belongs in the take, redraw.
        Returns the key pressed on the cue window, if any."""
        now = time.time()
        newest = None
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
                bends = rp.glove_bends(frame)
                self.collect["glove"].append((t, bends))
                self.last_bends, newest = (t, bends), None
            else:
                newest = (t, frame)
        if newest is not None:
            # Outside the warm-up only the newest frame of a drain is
            # measured: the bars draw one frame, and a thumb's joint table
            # per frame is the costly part of a tick.
            self.last_bends = (newest[0], rp.glove_bends(newest[1]))
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
        bars = (self.bars_now(now) if self.show_bars and self.window.enabled
                else None)
        key = self.window.show(title, words,
                               None if deadline is None else deadline - now,
                               sub, bars=bars)
        time.sleep(0.004)
        return key

    def hold_until(self, deadline: float, record: bool = True,
                   keys: bool = False) -> None:
        """Keep draining until `deadline`, asking for the still on time.

        `keys` (during a take): `s` on the cue window or in the console ends
        the take at once (SkipTake). The console is read only when a key is
        waiting, so the pause after the take still gets the keys meant for
        it."""
        while True:
            now = time.time()
            if self._snap is not None and now >= self._snap[0]:
                _at, path, caption = self._snap
                self._snap = None
                self.stills.request(path, caption, self.last_hand)
            if now >= deadline:
                return
            key = self.tick(record)
            if keys:
                if not key and console_has_key():
                    key = console_key()
                if key == SKIP_KEY:
                    raise SkipTake()

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
        return self.use_folder(folder)

    def use_folder(self, folder: Path) -> Path:
        """Record into `folder` (a new one, or a resumed session's own),
        with the subfolders this run writes."""
        folder = Path(folder)
        for sub in ("glove", "events") + (("leap", "stills")
                                          if self.leap is not None else ()):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        self.dir = folder
        return folder

    def resumed_warmup_name(self) -> str:
        """`warmup_<HHMMSS>.json` for a resumed run's warm-up: the first
        run's warmup.json is never rewritten, because the takes recorded
        before were judged against it."""
        stamp = time.strftime("%H%M%S")
        name, n = f"warmup_{stamp}.json", 1
        while (self.dir / name).exists() or \
                name in (self.meta.get("warmups") or []):
            n += 1
            name = f"warmup_{stamp}_{n}.json"
        return name

    def write_session(self, ended: Optional[str] = None) -> None:
        if self.dir is None:
            return
        data = dict(self.meta)
        data["ended"] = ended
        data["stopped"] = self.stopped
        data["takes"] = self.takes
        write_json(self.dir / "session.json", data)

    # --- the warm-up --------------------------------------------------------
    def warm_up(self, name: str = WARMUP_FILE) -> dict:
        """The warm-up, done again whole while it is refused, up to
        WARMUP_TRIES times; the last record is returned (refused or not)."""
        refusals: List[str] = []
        record: dict = {}
        for k in range(1, WARMUP_TRIES + 1):
            if k > 1:
                self.say("", f"warm-up again ({k} of {WARMUP_TRIES})")
            record = self.run_warmup(name, try_no=k, earlier=refusals)
            warmups = self.meta.setdefault("warmups", [])
            if name not in warmups:
                warmups.append(name)
            if not record["refused"]:
                return record
            refusals.append(record["refused"])
            self.say(f"      warm-up refused: {record['refused']}")
        return record

    def _warmup_straighten(self, finger: Optional[str], seconds: float,
                           words: str) -> None:
        t = time.time()
        self.beep(*WARMUP_STRAIGHTEN_BEEP)
        self.mock_follow((), seconds, t=t, warmup=True)
        self.bar_cue = ([finger] if finger else [], rp.STRAIGHTEN)
        self.show("WARM-UP", words, t + seconds)
        self.hold_until(t + seconds, record=False)

    def _num(self, sensor: str, finger: str, value) -> str:
        if value is None:
            return "-"
        if sensor == "glove" and finger == rp.THUMB:
            return f"{float(value):.1f}"            # degrees
        return f"{float(value):.3f}"

    def run_warmup(self, name: str = WARMUP_FILE, try_no: int = 1,
                   earlier=()) -> dict:
        """Open palm, full fist, then each finger on its own, cued; each
        finger's range for today, written to `name` in the session folder.

        The refusal rule and the camera's endpoints stay on the open palm
        and the fist; each finger's single bend is the median over the last
        WARMUP_SETTLE_S of its bend window (`recording_protocol.
        warmup_record`). The bars run through the single-finger part,
        against the open palm and fist just measured.
        """
        all5 = rp.FINGERS
        self.say("", "WARM-UP: an open palm, a full fist, then each finger "
                     "on its own, on the beeps.")
        self.bar_cue = None
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
        t_fist_end = time.time()
        settle = self.seconds(WARMUP_SETTLE_S)
        w_open = (t_open + settle, t_fist)
        w_fist = (t_fist + settle, t_fist_end)
        self.bar_ends = rp.window_endpoints(self.collect["glove"], w_open,
                                            w_fist)

        bend_s = self.seconds(WARMUP_SINGLE_S)
        rest_s = self.seconds(WARMUP_SINGLE_REST_S)
        self._warmup_straighten(None, rest_s, "OPEN the hand, fingers "
                                              "straight")
        t_single: Dict[str, Tuple[float, float]] = {}
        for f in all5:
            words = rp.FINGER_WORDS[f]
            t_bend = time.time()
            self.beep(*WARMUP_BEND_BEEP)
            self.mock_follow((f,), bend_s, t=t_bend, warmup=True)
            self.bar_cue = ([f], rp.BEND)
            self.show("WARM-UP", f"bend the {words.upper()} only",
                      t_bend + bend_s, "the other fingers stay straight; "
                      "hold it bent until the low beep")
            self.say(f"      {words} on its own ({bend_s:g} s)")
            self.hold_until(t_bend + bend_s, record=False)
            t_up = time.time()
            t_single[f] = (t_up - settle, t_up)
            self._warmup_straighten(f, rest_s, "straighten")
        self.bar_cue = None
        t_end = time.time()
        self.beep(*END_BEEP)
        self.mock_follow((), self.seconds(PREP_S), t=t_end, warmup=True)
        collected, self.collect = self.collect, None

        record = rp.warmup_record(
            self.hand, w_open, w_fist, collected["glove"],
            collected["camera"] if self.leap is not None else None,
            settle_s=settle, t_single=t_single, units=rp.GLOVE_UNITS)
        # Which try of WARMUP_TRIES this is, and why the ones before it
        # were refused: a repeated warm-up is not a hidden one.
        record["try"] = int(try_no)
        record["earlier_refusals"] = list(earlier)
        write_json(self.dir / name, record)
        self.say(f"      {name}:")
        for sensor in ("glove", "camera"):
            ends = record[sensor]
            if ends is None:
                continue
            for key in ("open", "fist", "span", "single", "single_span"):
                values = ends.get(key)
                if not isinstance(values, dict):
                    continue
                self.say(f"      {sensor:<6} {key:<11} " + "  ".join(
                    f"{f} {self._num(sensor, f, values.get(f))}"
                    for f in rp.FINGERS))
        self.say("      (glove thumb in degrees, larger = more bent; every "
                 "other number is a curl, smaller = more bent)")
        self.warmup = record
        self.bar_ends = record["glove"]
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

    def _item_cued(self, item: dict) -> List[str]:
        """The fingers the bars draw wide outside a cue: Set B's finger."""
        return [item["finger"]] if self.protocol.name == rp.FLEXION else []

    def _cue_bars(self, cue: rp.Cue, previous: Optional[rp.Cue],
                  item: dict) -> Tuple[List[str], Optional[str]]:
        """(cued fingers, phase) for the bars during one cue.

        Set B: the item's finger, in the cue's phase. Set C: the step's
        flexed fingers; the phase by `cue_pitch`'s rule (a step that flexes
        a finger is "flex", one that only straightens is "extend", one that
        changes nothing is "hold", and its flexed fingers stay bent).
        """
        if cue.phase is not None:
            return [item["finger"]], cue.phase
        pitch = rp.cue_pitch(cue, previous)
        phase = {rp.PITCH_FLEX: "flex",
                 rp.PITCH_EXTEND: "extend"}.get(pitch, rp.HOLD)
        return list(cue.flexed), phase

    def run_take(self, item_id: str, take: int, attempt: int, round_no: int
                 ) -> dict:
        item = self.protocol.item(item_id)
        cues = rp.build_schedule(self.protocol, item_id, self.scale)
        label = item["label"]
        hint = self._item_hint(item)
        self._warned = set()
        self.bar_cue = (self._item_cued(item), None)

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
        interrupted = skipped = False
        previous = None
        try:
            for k, cue in enumerate(cues):
                self.hold_until(t0 + offsets[k], keys=True)
                t = time.time()
                self.beep(rp.cue_pitch(cue, previous),
                          max(40, min(BEEP_MS, int(cue.hold_s * 600))))
                events.write("cue", t=t, **cue.event_fields())
                self.mock_follow(cue.flexed, cue.hold_s, cue.phase, t=t)
                self.bar_cue = self._cue_bars(cue, previous, item)
                # `step` as the events file and the reject reasons count it
                # (from 0), so "step 3" means the same cue everywhere.
                where = (f"cycle {cue.cycle}/{item['cycles']}"
                         if cue.cycle is not None else f"step {cue.step}")
                self.show(name, cue.label.upper(),
                          t0 + offsets[k] + cue.hold_s, where)
                self.say(f"      [{k + 1:>2}/{len(cues)}] {where:<11} "
                         f"{cue.label}  ({cue.hold_s:g} s)")
                previous = cue
            self.hold_until(t0 + total, keys=True)
        except KeyboardInterrupt:
            interrupted = True
        except SkipTake:
            skipped = True
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
                wait_end = time.time() + (0.0 if interrupted or skipped
                                          else STILL_WAIT_S)
                while not self.stills.ready(paths["still"]) and \
                        time.time() < wait_end:
                    self.tick(record=False)
                still_name, still_missing = still_status(paths["still"])
            curls = rp.read_bends(paths["glove"], self.hand)
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
        elif skipped:
            accepted, reason, by = False, SKIPPED_DURING_TAKE, "operator"
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
            # The warm-up file this take was judged against (contract
            # section 3); a resumed run has its own.
            "warmup": self.warmup_name,
            "check": check.as_dict(),
        }
        if not accepted:
            self.reject(entry, by)
        self.takes.append(entry)
        self.write_session()

        verdict = "ACCEPTED" if accepted else f"REJECTED: {reason}"
        self.say(f"      {verdict}")
        if not accepted and not interrupted and not skipped and check.hint:
            # What to change before the retry that starts in a few seconds,
            # in one line: the reason says what the glove measured, this
            # says what the hand should do about it.
            self.say(f"      what to do: {check.hint}")
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
        """The few seconds after a take: 'r' redo, 's' skip the rest of the
        item, 'q' stop, or nothing. The bars stay up, so the operator can
        see the glove answer before the next take."""
        flush_console_keys()
        end = time.time() + self.seconds(PAUSE_S)
        words = ("ACCEPTED" if entry["accepted"]
                 else f"REJECTED: {entry['reason']}")
        self.bar_cue = (self._item_cued(self.protocol.item(entry["item"])),
                        None)
        self.show(entry["name"], words, end,
                  "press r to redo this take, s to skip the rest of this "
                  "item, q to stop the session")
        try:
            while time.time() < end:
                key = self.tick(record=False) or console_key()
                if key in ("r", "q", SKIP_KEY):
                    return key
            return ""
        finally:
            self.bar_cue = None

    def skip_item(self, item_id: str, take: int) -> None:
        """`s`: no more attempts at `item_id` in this run. Written to
        session.json under "skipped"; the skip belongs to this run, not to
        the data, so a --resume records the item again."""
        self.skipped_items.add(item_id)
        rec = {"item": item_id, "at": iso_now(), "take": int(take)}
        self.meta.setdefault("skipped", []).append(rec)
        self.skips.append(rec)
        self.write_session()
        self.say(f"      SKIPPED: no more {item_id} in this run "
                 f"({SKIPPED_TEXT})")

    def skipped_without_take(self) -> List[str]:
        """Items skipped in this run that have no accepted take: they make
        the run incomplete, as a failed take does."""
        return sorted(i for i in self.skipped_items
                      if self.accepted.get(i, 0) == 0)

    def record_take(self, item_id: str, round_no: int) -> None:
        """One planned take: attempts until one is accepted, or the retries
        run out. The operator's own redo does not use up a retry; `s` ends
        the item for this run, during the take or in the pause after it."""
        take = self.accepted[item_id] + 1
        auto_rejects = 0
        attempt = 0
        while True:
            attempt += 1
            entry = self.run_take(item_id, take, attempt, round_no)
            if entry["decided_by"] == "operator" and \
                    entry["reason"] == SKIPPED_DURING_TAKE:
                self.skip_item(item_id, take)
                return
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
            if key == SKIP_KEY:
                self.skip_item(item_id, take)
                return
            if entry["accepted"]:
                return
            auto_rejects += 1
            if auto_rejects > self.args.retries:
                self.say(f"      FAILED: {item_id} take {take} after "
                         f"{attempt} attempt(s)")
                self.failed.append(f"{item_id} take {take}")
                return
            self.say(f"      retrying ({auto_rejects}/{self.args.retries})")

    def run_rounds(self) -> None:
        """Every planned take, round by round. An item skipped in this run
        is passed over, and so is one that already has its takes (a resumed
        session): take numbers go on from the accepted count."""
        for r, order in enumerate(self.plan, 1):
            for item_id in order:
                if item_id in self.skipped_items:
                    continue
                if self.accepted[item_id] >= self.takes_per_item:
                    continue
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
            for s in self.skips:
                self.say(f"  {s['item']:<22} {s['take']:>4}  {'-':<8}  "
                         f"{SKIPPED_TEXT} (no more attempts in this run)")
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
    p.add_argument("--time-scale", type=float, default=None,
                   help="multiply every cue duration (rehearsals and tests; "
                        "a real session uses 1.0, the default; --resume "
                        "keeps the session's own)")
    p.add_argument("--resume", default=None, metavar="FOLDER|latest",
                   help="carry on a session that was cut short, in its own "
                        "folder: its items, takes and round order are "
                        "reused and items that have their takes are "
                        "skipped; the calibration prompt, the acquire gate "
                        "and a new warm-up (warmup_<HHMMSS>.json) are done "
                        "again. latest = the newest folder of this set and "
                        "hand under --out-dir")
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
    p.add_argument("--no-bars", action="store_true",
                   help="no finger bars on the cue window, only the words "
                        "and the countdown; the bars (the default) show "
                        "each finger's glove reading against today's "
                        "warm-up")
    p.add_argument("--hide-camera", action="store_true",
                   help="run the camera with no window (stills only); the "
                        "default shows the IR image and the skeleton so you "
                        "can see the hand")
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
    if args.resume is not None:
        clash = [flag for flag, value in (("--items", args.items),
                                          ("--takes", args.takes),
                                          ("--seed", args.seed))
                 if value is not None]
        if clash:
            p.error("--resume carries on a session that already fixes its "
                    "items, takes and round order; drop "
                    + " and ".join(clash))
    args.time_scale_given = args.time_scale is not None
    if args.time_scale is None:
        args.time_scale = 1.0
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


def latest_session(out_dir: Path, set_name: str, hand: str) -> Optional[Path]:
    """The newest folder under `<out_dir>/<set>/` named `..._<hand>` with a
    session.json, or None. Folder names start with the session's stamp, so
    the newest is the last by name."""
    base = Path(out_dir) / set_name
    if not base.is_dir():
        return None
    found = [p for p in base.iterdir()
             if p.is_dir() and p.name.endswith(f"_{hand}")
             and (p / "session.json").is_file()]
    return max(found, key=lambda p: p.name) if found else None


def _any_mock(value) -> bool:
    """session.json's "mock": a boolean, or (Sets B and C before their fix)
    a dict of flags, any of which counts."""
    if isinstance(value, dict):
        return any(bool(v) for v in value.values())
    return bool(value)


def resume_folder(args, protocol: rp.Protocol) -> Tuple[Path, dict]:
    """The folder `--resume` names and its session.json.

    Raises ResumeRefused, naming every reason, when this run cannot carry
    the session on: another set, hand or protocol file (its sha256), a
    mock session resumed for real or the other way round (`mock` only, not
    which mock: a rehearsal can change its mock glove), another camera or
    time scale, no saved round order, or a session whose warm-up predates
    the thumb in degrees (its takes and new ones would measure the thumb two
    ways; the checker could not put them on one scale).
    """
    if str(args.resume).strip().lower() == "latest":
        folder = latest_session(args.out_dir, protocol.name, args.hand)
        if folder is None:
            raise ResumeRefused(
                f"no {args.hand}-hand {protocol.name} session (a folder "
                f"ending in _{args.hand} with a session.json) under "
                f"{Path(args.out_dir) / protocol.name}")
    else:
        folder = Path(args.resume)
    try:
        meta = json.loads((folder / "session.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    if not isinstance(meta, dict):
        raise ResumeRefused(f"no readable session.json in {folder}")
    why = []
    if meta.get("set") != protocol.name:
        why.append(f"it is a {meta.get('set')} session and this run records "
                   f"{protocol.name}")
    if meta.get("hand") != args.hand:
        why.append(f"it is the {meta.get('hand')} hand's session and this "
                   f"run is --hand {args.hand}")
    if meta.get("protocol_sha256") != protocol.sha256:
        why.append(f"{display_path(protocol.path)} is not the protocol file "
                   "the session was recorded under (sha256 differs)")
    if _any_mock(meta.get("mock")) != bool(args.mock):
        why.append(f"it is a {'mock' if _any_mock(meta.get('mock')) else 'real'}"
                   f" session and this run is "
                   f"{'mock' if args.mock else 'real'}")
    if meta.get("camera", "leap") != args.camera:
        why.append(f"it ran with --camera {meta.get('camera')}")
    scale = float(meta.get("time_scale") or 1.0)
    if args.time_scale_given and scale != float(args.time_scale):
        why.append(f"it ran at --time-scale {scale:g}")
    if not isinstance(meta.get("rounds"), list) or not meta["rounds"]:
        why.append("its session.json has no round order to carry on")
    first = folder / WARMUP_FILE
    if first.is_file():
        try:
            warm = json.loads(first.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            warm = None
        if isinstance(warm, dict) and not warm.get("units"):
            why.append("its warm-up was recorded before the glove thumb was "
                       "measured in degrees (warmup.json has no units), so "
                       "its takes and new ones would measure the thumb two "
                       "ways; record a new session instead")
    if why:
        raise ResumeRefused(f"{folder}: " + "; ".join(why))
    return folder, meta


def planned_left(plan: List[List[str]], done: Dict[str, int],
                 takes: int) -> List[str]:
    """The planned takes still to record, in order, when every one of them
    is accepted: an item is passed over once it has `takes` accepted."""
    counts = dict(done)
    out = []
    for order in plan:
        for item_id in order:
            if counts.get(item_id, 0) < takes:
                counts[item_id] = counts.get(item_id, 0) + 1
                out.append(item_id)
    return out


def run(args) -> int:
    path = args.protocol if args.protocol is not None else \
        ROOT / "protocols" / f"{args.set}.json"
    resumed = None
    try:
        protocol = rp.load_protocol(path)
        if protocol.name not in SET_CHOICES:
            raise rp.ProtocolError(
                f"{path}: Set A ({protocol.name}) is recorded by "
                "scripts/leap/record_poses.py --protocol, not by this recorder")
        if args.resume is not None:
            try:
                resumed = resume_folder(args, protocol)
            except ResumeRefused as e:
                print(f"\nCannot resume: {e}\n")
                return EXIT_REFUSED
            items = rp.select_items(protocol, resumed[1].get("items") or [])
        else:
            items = rp.select_items(protocol, (args.items or "").split(",")
                                    if args.items else None)
    except rp.ProtocolError as e:
        print(f"\nProtocol file problem: {e}\n")
        return EXIT_REFUSED
    coached()        # a few seconds of imports, before the Enter, not after
    shuffle = protocol.shuffle_rounds
    done: Dict[str, int] = {i: 0 for i in items}
    if resumed is not None:
        folder, old = resumed
        if not args.time_scale_given:
            args.time_scale = float(old.get("time_scale") or 1.0)
        takes = int(old.get("takes_per_item") or protocol.takes_per_item)
        seed = old.get("seed")
        plan = [list(r) for r in old["rounds"]]
        for t in old.get("takes") or []:
            if t.get("accepted") and t.get("item") in done:
                done[t["item"]] += 1
    else:
        takes = args.takes if args.takes is not None else \
            protocol.takes_per_item
        seed = args.seed
        if shuffle and seed is None:
            seed = random.SystemRandom().randrange(1, 2 ** 31)
        plan = rp.rounds(items, takes, seed=seed, shuffle=shuffle)
    left = planned_left(plan, done, takes)

    camera = args.camera
    setup = ("gloves on + camera, one hand" if camera == "leap"
             else "gloves on, no camera, one hand")
    n_takes = sum(len(r) for r in plan)
    per_take = {i: rp.schedule_seconds(rp.build_schedule(protocol, i,
                                                         args.time_scale))
                for i in items}
    eta = sum(per_take[i] + (PREP_S + PAUSE_S) * args.time_scale
              for i in left)
    print("=" * 72)
    if resumed is not None:
        print(f"RESUMING {resumed[0]}: {n_takes - len(left)} of {n_takes} "
              f"planned takes accepted, {len(left)} to record, about "
              f"{eta / 60:.1f} min")
    else:
        print(f"PROTOCOL {protocol.name} v{protocol.version}: {n_takes} takes "
              f"in {len(plan)} round(s), about {eta / 60:.1f} min of "
              "recording")
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
          "extend. The window shows the words" + (
              "." if args.no_bars else
              " and each finger's glove reading (green = where the cue "
              "wants it)."))
    print("  After a take: r = redo, s = skip the rest of this item, q = "
          "stop. During a take: s = skip it now.")
    print("=" * 72)
    if resumed is not None and not left:
        print(f"Nothing left to record in {resumed[0]}: every item has its "
              f"{takes} accepted take(s).")
        return EXIT_OK

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
            ViewerStills(args.hand, args.band, window=not args.hide_camera)

    beep = (lambda f, ms: None) if args.no_beep else coached().beep
    session = ProtocolSession(args, protocol, items, plan, seed, glove, leap,
                              stills, CueWindow(enabled=not args.no_view),
                              beep)
    session.takes_per_item = takes
    if resumed is not None:
        folder, old = resumed
        session.meta = dict(old)
        session.takes = list(old.get("takes") or [])
        session.accepted = dict(done)
        # The first run's calibration stays under its old key; every run's
        # is listed under "calibrated", in order.
        cal = old.get("calibrated")
        session.meta["calibrated"] = (
            list(cal) if isinstance(cal, list)
            else [old.get("xr_trainer_calibrated_at")]) + [calibrated_at]
        session.meta["resumed"] = list(old.get("resumed") or []) + [started]
        session.meta["warmups"] = list(
            old.get("warmups")
            or ([WARMUP_FILE] if (folder / WARMUP_FILE).is_file() else []))
        session.meta["skipped"] = list(old.get("skipped") or [])
        session.meta.update(ended=None, stopped=None)
    else:
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
            "calibrated": [calibrated_at],
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
                               args.mock_glove
                               and not args.mock_glove_ignore_cues),
                           "leap": bool(args.mock_leap)},
            "warmup": WARMUP_FILE,
            # Every warm-up file of the session, in order; a take entry's
            # "warmup" names the one it was judged against.
            "warmups": [],
            "resumed": [],
            "skipped": [],
            "stopped": None,
        }
    code = EXIT_OK
    try:
        session.wait_for_glove(args.glove_timeout)
        if leap is not None:
            session.acquire(args.acquire_timeout)
        if resumed is not None:
            session.use_folder(resumed[0])
            session.warmup_name = session.resumed_warmup_name()
        else:
            folder = session.make_folder(args.out_dir)
            session.meta["session"] = folder.name
            session.warmup_name = WARMUP_FILE
        session.write_session()
        warm = session.warm_up(session.warmup_name)
        if warm["refused"]:
            session.stopped = f"warm-up refused: {warm['refused']}"
            raise NotReady(f"warm-up: {warm['refused']}")
        session.write_session()
        session.run_rounds()
        if session.failed or session.skipped_without_take():
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
    if session.dir is not None and code != EXIT_OK:
        session.say(f"  carry on later in the same folder: --resume "
                    f"\"{session.dir.resolve()}\" (or --resume latest)")
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
