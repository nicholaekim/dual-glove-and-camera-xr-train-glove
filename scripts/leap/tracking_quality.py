"""The 60 second tracking test: why does the camera lose the hand?

Hardware setup: camera only, bare hand. The Stereo IR 170 flat on the desk,
lenses up, desktop mode; no glove.

Why this exists. The operator's words were "the hand tracking for the camera
is not good and it does not register sometimes". The tracking service is
healthy (its log shows 90 Hz, no dropped frames, about 10 ms of latency, the
config unchanged since install) and earlier bare-hand static poses tracked
97 to 100 percent of their frames. So the losses happen in the moment: the
hand too high, too low or out at the edge of the field, the palm turned
away, a closed hand seen from below, a hand moving fast, sunlight or another
infrared source, smudged lenses, a device warning. Once the moment has
passed nothing says which. This script coaches the operator through one
minute of six slow moves that visit those on purpose, records every
tracking frame (empty ones included) with the IR image brightness and the
device's own status flags beside it, and then explains each loss:

  results\\diagnostics\\tracking_quality_<stamp>.csv   one row per tracking frame
  results\\diagnostics\\tracking_quality_<stamp>.txt   the report printed at the end

The report gives the tracked percentage, the frame rate, every loss (gone
longer than 100 ms, or a new hand id) with where the hand was and why, the
causes ranked by how many losses each explains, and the fix to try first.
The last line printed starts with `VERDICT:`. The classifier is
`leap_hand.tracking_quality`, the same one the camera window's "tracking:"
line and the grasp recorder's reject reasons use.

How it coaches. The operator's rule: one action at a time, one short
sentence, no alternatives, the hardware setup named. So the minute is six
moves of equal length (`MOVES`), each a phrase of at most six words. Every
move change is a beep and the phrase spoken by the Windows voice, because
the operator's eyes are on the window and their hand is over the module.
The camera window, drawn by this script from its own stream (no second
client), shows the phrase in large letters at the top, a bar counting the
move down under it, the IR image with the fitted skeleton, and one line of
tracking status at the bottom. The console prints one setup line, one line
per move as it starts, and the report: nothing that repeats.

Usage:
  .venv\\Scripts\\python.exe scripts\\leap\\tracking_quality.py --hand left
  .venv\\Scripts\\python.exe scripts\\leap\\tracking_quality.py --seconds 60 --hand left --out results\\diagnostics
  .venv\\Scripts\\python.exe scripts\\leap\\tracking_quality.py --mock --seconds 5 --no-view --no-voice --no-beep
  .venv\\Scripts\\python.exe scripts\\leap\\tracking_quality.py --from results\\diagnostics\\tracking_quality_<stamp>.csv

`--no-view` keeps the window shut, `--no-voice` the voice quiet and
`--no-beep` the beeps off; the console prints the same lines either way.
A machine without the Windows voice runs the test silently, with no error.

`--mock` needs no hardware and opens no window: a scripted mock hand is lost
on purpose in every way the report can explain, so the whole report is
exercised; the beeps and the voice still play unless turned off. `--from`
re-reads a saved CSV (or a recorded take's JSONL) and prints the report
again, for after the fact.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Callable, List, Optional, Sequence, Tuple

from leap_hand.live import beep as live_beep
from leap_hand.protocol import image_banner
from leap_hand.stream import LeapStream, LeapUnavailable
from leap_hand.tracking_quality import (
    LOSS_GAP_S,
    LiveLossTracker,
    MockScenario,
    SERIES_COLUMNS,
    Sample,
    cadence_gap_s,
    cause_label,
    decode_device_status,
    format_report,
    image_stats,
    rows_to_samples,
    samples_from_series_rows,
    series_rows,
    state_from_leaphand,
    summarise,
)

DEFAULT_OUT = Path("results") / "diagnostics"
GET_READY_S = 3.0
IMAGE_EVERY_S = 0.1          # IR brightness is measured ten times a second
IMAGE_FRESH_S = 0.25         # older than this is not attached to a frame

# The coached minute: six moves, each an equal share of `--seconds` (10 s of
# the default 60). Each is something that has cost tracking before, done
# slowly so the moment it fails is visible and recorded. One short phrase
# each, at most six words: it is printed, spoken and shown in the window
# exactly as written here. The first one starts at 30 cm above the module
# with the palm to the lenses; the get-ready phrase puts the hand there.
MOVES = (
    "Hand open, hold still",
    "Slowly up, then back down",
    "Slowly left, then right",
    "Turn the palm away, then back",
    "Slow fist, then open",
    "Three grasp shapes, slowly",
)
GET_READY = "Get ready: hand 30 cm up"

# The cues. A higher, longer beep starts the run; the rest mark a new move;
# a low one ends it. `live.beep` plays each on its own thread.
BEEP_FIRST = (1000, 250)
BEEP_MOVE = (880, 150)
BEEP_END = (500, 300)

# The Windows voice: ONE PowerShell process for the whole run, started before
# the first move so its second of start-up is over by then, reading one
# phrase per line from its stdin and speaking it. A process per phrase would
# say every move a second late. `$ErrorActionPreference = 'Stop'` makes a
# machine without System.Speech end the process at once; the next write to
# it then fails and `Voice` goes quiet, never raising.
VOICE_LOOP = (
    "$ErrorActionPreference = 'Stop'; "
    "Add-Type -AssemblyName System.Speech; "
    "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "while ($null -ne ($line = [Console]::In.ReadLine())) { "
    "if ($line.Trim()) { $voice.Speak($line) } }"
)
VOICE_COMMAND = ("powershell", "-NoProfile", "-NonInteractive", "-Command", VOICE_LOOP)

# The camera window. The IR image is 384 px square, drawn at twice that;
# the phrase and its bar sit in a band above the picture and the status line
# in a band below it, so no text covers the hand.
WINDOW_NAME = "Tracking test"
VIEW_SIZE = 768
VIEW_SCALE = 2                    # 384 px sensor image -> 768 px
VIEW_TOP = 128                    # the phrase and the countdown bar
VIEW_BOTTOM = 52                  # the status line
VIEW_EVERY_S = 1.0 / 30.0         # redraw rate; the frames are buffered anyway
HANDS_FRESH_S = 0.25              # older tracking than this draws no hand
BASELINE_HALF_MM = 32.0           # as scripts/leap/camera_view.py: the left lens
LEFT_CAMERA = 1                   # eLeapPerspectiveType_stereo_left
SIDE_COLOURS = {"left": (255, 220, 0), "right": (60, 60, 255)}      # BGR, as the viewer


@dataclass
class FrameRecord:
    """One tracking frame as it came off the camera, before it is reduced."""

    t: float                                 # seconds, the camera's own clock
    frame_id: Optional[int]
    framerate: Optional[float]
    hands: list = field(default_factory=list)          # LeapHand
    image_mean: Optional[float] = None
    image_saturated: Optional[float] = None
    status: Optional[tuple] = None


def to_sample(rec: FrameRecord) -> Sample:
    return Sample(t=rec.t, hands=tuple(state_from_leaphand(lh) for lh in rec.hands),
                  framerate=rec.framerate, image_mean=rec.image_mean,
                  image_saturated=rec.image_saturated, status=rec.status,
                  frame_id=rec.frame_id)


# --- the real camera ------------------------------------------------------------
class QualityStream(LeapStream):
    """`LeapStream` plus what a loss report needs and a plain stream drops.

    Every tracking frame, the empty ones included (LeapStream only queues
    hands, and a loss is exactly a run of frames with no hand); the device's
    own status flags, from the device event at connect and every status
    change after it; dropped-frame events; and the IR image brightness. All
    of it arrives on the SDK's thread, so it is only stored here, never
    printed or drawn.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._records: deque = deque(maxlen=90 * 600)
        self.status: Optional[tuple] = None      # None until a device event says
        self.status_raw: Optional[int] = None
        self.status_events = 0
        self.status_history: List[tuple] = []
        self.dropped_frame_events = 0
        self.images = 0
        self.image_errors = 0
        self._image = None                       # (mean, saturated, wall time)
        # The picture itself, for the camera window: kept only when a window
        # asks for it, at the window's own rate. (array, wall time); one
        # tuple, so the main thread never reads half an update.
        self.keep_frames = False
        self.frame = None

    def _note_status(self, event) -> None:
        try:
            raw = int(event.c_data.status)
        except Exception:                        # pragma: no cover - hardware path
            return
        self.status_raw = raw & 0xFFFFFFFF
        self.status = decode_device_status(raw)
        self.status_events += 1
        self.status_history.append((time.time(), self.status_raw, self.status))

    def _on_image(self, event) -> None:
        now = time.time()
        stats_due = self._image is None or now - self._image[2] >= IMAGE_EVERY_S
        frame_due = self.keep_frames and (self.frame is None
                                          or now - self.frame[1] >= VIEW_EVERY_S)
        if not (stats_due or frame_due):
            return
        try:
            from leap_hand.images import image_to_numpy
            picture = image_to_numpy(event.image[0])
            if stats_due:
                mean, sat = image_stats(picture)
        except Exception:                        # pragma: no cover - hardware path
            self.image_errors += 1
            return
        if frame_due:
            self.frame = (picture, now)
        if stats_due:
            self.images += 1
            self._image = (mean, sat, now)

    def _build_listener(self):
        base = super()._build_listener()
        stream = self

        class _QualityListener(type(base)):
            def on_device_event(self, event):
                stream._note_status(event)
                super().on_device_event(event)

            def on_device_status_change_event(self, event):
                stream._note_status(event)

            def on_device_failure_event(self, event):
                stream._note_status(event)

            def on_dropped_frame_event(self, event):
                stream.dropped_frame_events += 1

            def on_image_event(self, event):
                stream._on_image(event)

        return _QualityListener()

    def _on_tracking(self, event) -> None:
        super()._on_tracking(event)            # converts and queues the hands
        hands = []
        while True:
            try:
                hands.append(self.queue.get_nowait()[1])
            except Empty:
                break
        fid = getattr(event, "tracking_frame_id", None)
        hands = [lh for lh in hands if fid is None or lh.frame_id == fid]
        img = self._image
        if img is not None and time.time() - img[2] > IMAGE_FRESH_S:
            img = None
        rate = float(getattr(event, "framerate", 0.0) or 0.0)
        self._records.append(FrameRecord(
            t=float(event.timestamp) / 1e6, frame_id=fid, framerate=rate or None,
            hands=hands, image_mean=None if img is None else img[0],
            image_saturated=None if img is None else img[1], status=self.status))

    def _discard_pending(self) -> None:
        super()._discard_pending()
        self._records.clear()

    def records(self) -> List[FrameRecord]:
        out = []
        while True:
            try:
                out.append(self._records.popleft())
            except IndexError:
                return out

    def enable_images(self) -> str:
        """Ask for the IR images; returns a note when the answer is not yes.

        The answer is not trusted either way: whether images are there is
        decided at the end by whether any arrived (see
        `leap_hand.protocol.image_banner` for why the reply alone misled).
        """
        try:
            flags = self.connection.set_policy_flags(
                flags_to_set=[self._leap.enums.PolicyFlag.Images])
        except Exception as e:                   # pragma: no cover - hardware path
            return f"the image policy request failed ({type(e).__name__}: {e})"
        names = {getattr(f, "name", str(f)) for f in (flags or [])}
        return "" if "Images" in names else (
            f"the service did not confirm the image policy (flags: "
            f"{', '.join(sorted(names)) or 'none'})")


# --- the mock camera ------------------------------------------------------------
class MockSource:
    """`MockLeapStream` acting out `MockScenario`, one record per frame.

    The mock emits nothing for a dropped frame, the way LeapC emits no hand,
    so the frames in between are filled in as empty ones: every frame index
    is accounted for, as the real stream's empty tracking events are.
    """

    def __init__(self, seconds: float, hand: Optional[str]):
        from leap_hand.mock import MockLeapStream
        self.scenario = MockScenario(seconds)
        self.hz = self.scenario.hz
        self.mock = MockLeapStream(hz=self.hz, dropout_every=0, reacquire_every=0,
                                   sides=(hand or "left",), script=self.scenario.script)
        self.device_serial = self.mock.device_serial
        self._clock: Optional[float] = None

    def start(self) -> None:
        self._clock = time.time()

    def stop(self) -> None:
        self._clock = None

    def records(self) -> List[FrameRecord]:
        """The frames due by now, paced against the wall clock."""
        now = time.time()
        n = int((now - self._clock) * self.hz)
        if n <= 0:
            return []
        self._clock += n / self.hz
        return self.frames(n)

    def frames(self, n: int) -> List[FrameRecord]:
        """The next `n` frames, ignoring the clock (tests drive this)."""
        first = self.mock.frames
        by_frame: dict = {}
        for _side, lh in self.mock.generate(n):
            by_frame.setdefault(lh.frame_id, []).append(lh)
        out = []
        for k in range(first, first + n):
            t = k / self.hz
            env = self.scenario.environment(t)
            hands = by_frame.get(k, [])
            out.append(FrameRecord(
                t=t, frame_id=k, framerate=env["framerate"], hands=hands,
                image_mean=env["image_mean_hand"] if hands else env["image_mean"],
                image_saturated=env["image_saturated"], status=env["status"]))
        return out


# --- the moves and the cues -----------------------------------------------------------
def move_length(seconds: float) -> float:
    """Seconds per move: `--seconds` split evenly across the six."""
    return float(seconds) / len(MOVES)


def scaled_steps(seconds: float) -> List[Tuple[float, str]]:
    """(start in seconds, phrase) of each move in a run of `seconds`."""
    span = move_length(seconds)
    return [(k * span, phrase) for k, phrase in enumerate(MOVES)]


def move_index(elapsed: float, seconds: float) -> int:
    """The move under way `elapsed` seconds into the run, 0 to 5."""
    return max(0, min(len(MOVES) - 1, int(elapsed // move_length(seconds))))


def setup_line(hand: Optional[str]) -> str:
    """The one line the console prints before the run: the hardware setup."""
    who = f"bare {hand} hand" if hand else "bare hand"
    return f"Camera only, {who}. Module flat on the desk, lenses up."


def move_line(k: int) -> str:
    """The one line the console prints as move `k` (from 0) starts."""
    return f"Move {k + 1} of {len(MOVES)}: {MOVES[k]}"


class Voice:
    """Speaks each phrase with the Windows voice and never blocks the loop.

    `command` is a process that reads phrases from its stdin, one per line,
    and speaks them (`VOICE_COMMAND`); None is a voice that is off. Writing a
    line to it returns at once. Every failure (no PowerShell, no
    System.Speech, the process gone) turns the voice off without a word,
    because the test runs the same without it.
    """

    def __init__(self, command: Optional[Sequence[str]] = VOICE_COMMAND):
        self._proc = None
        if not command:
            return
        try:
            # Unbuffered: a failed write leaves nothing behind for Python to
            # retry, and complain about on stderr, when the pipe is collected.
            self._proc = subprocess.Popen(
                list(command), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            self._proc = None

    @property
    def on(self) -> bool:
        return self._proc is not None

    def say(self, text: str) -> None:
        proc = self._proc
        if proc is None:
            return
        line = " ".join(str(text).split()).encode("ascii", "replace") + b"\n"
        try:
            proc.stdin.write(line)
        except Exception:
            self.close()

    def close(self) -> None:
        """No more phrases. One still being spoken finishes on its own; this
        never waits for it."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass


# --- the camera window ------------------------------------------------------------------
def leapc_projector(connection) -> Optional[Callable]:
    """Tracking-space millimetres -> pixel in the unmirrored window image.

    LeapC's own lens model, the same mapping as `to_px` in
    `scripts/leap/camera_view.py` (checked there against saved IR stills).
    None when LeapC is not there to ask; the window then draws no skeleton.
    """
    try:
        from leapc_cffi import ffi, libleapc
        ptr = connection.get_connection_ptr()
    except Exception:
        return None

    def to_px(p):
        x, y, z = p
        if y <= 5.0:
            return None
        v = ffi.new("LEAP_VECTOR*")
        v.x = -(x - BASELINE_HALF_MM) / y
        v.y = z / y
        v.z = 1.0
        try:
            r = libleapc.LeapRectilinearToPixel(ptr, LEFT_CAMERA, v[0])
        except Exception:                        # pragma: no cover - hardware path
            return None
        if not (math.isfinite(r.x) and math.isfinite(r.y)):
            return None
        px, py = r.x * VIEW_SCALE, r.y * VIEW_SCALE
        if abs(px) > 1e4 or abs(py) > 1e4:
            return None
        return int(px), int(py)

    return to_px


def status_line(live: LiveLossTracker, hand_now: bool) -> str:
    """The window's one line of tracking status."""
    pct = 100.0 * (live.tracked_fraction or 0.0)
    last = "none" if live.last is None else cause_label(live.last.cause, live.last.status)
    seen = "hand seen" if hand_now else "NO HAND"
    return f"{seen}   tracked {pct:.0f} %   lost {live.total_losses}   last: {last}"


_FONT = 0                         # cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_AMBER = (0, 200, 255)
_GREY = (130, 130, 130)
_GREEN = (80, 255, 80)
_RED = (60, 60, 255)
_PHRASE_SCALE = []                # computed once: one size for every phrase


def _put(cv2, frame, text, org, scale, colour, thick):
    """Text with a black outline, readable on the IR image and on black."""
    cv2.putText(frame, text, org, _FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, scale, colour, thick, cv2.LINE_AA)


def _fit(cv2, text, width, largest, thick):
    w = cv2.getTextSize(text, _FONT, 1.0, thick)[0][0] or 1
    return min(largest, width / w)


def phrase_scale(cv2) -> float:
    """The largest text size at which every phrase fits across the window."""
    if not _PHRASE_SCALE:
        _PHRASE_SCALE.append(min(_fit(cv2, p, VIEW_SIZE - 48, 1.6, 3)
                                 for p in MOVES + (GET_READY,)))
    return _PHRASE_SCALE[0]


def compose_frame(picture, hands, phrase: str, fraction_left: float, status: str,
                  to_px: Optional[Callable] = None, banner: str = "",
                  hand_now: bool = False):
    """The window's picture, top to bottom: the phrase in large letters, the
    bar counting the move down, the IR image (mirrored, so the left hand is
    on the left) with the fitted skeleton and a green frame while a hand is
    seen, red while none is, and the status line.

    Pure: no window and no camera, so the tests can draw it.
    """
    import cv2
    import numpy as np
    from xr_hand.joints import BONES

    size = VIEW_SIZE
    if picture is None:
        image = np.zeros((size, size, 3), np.uint8)
    else:
        grey = cv2.resize(picture, (size, size), interpolation=cv2.INTER_LINEAR)
        grey = cv2.convertScaleAbs(grey, alpha=2.2, beta=8)   # the raw IR is dark
        image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    if to_px is not None:
        for lh in hands:
            colour = SIDE_COLOURS.get(lh.hand_side, _WHITE)
            pts = [to_px((p[0] * 1000.0, p[1] * 1000.0, p[2] * 1000.0)) for p in lh.abs26]
            for a, b in BONES:
                if pts[a] and pts[b]:
                    cv2.line(image, pts[a], pts[b], colour, 2, cv2.LINE_AA)
            for p in pts:
                if p:
                    cv2.circle(image, p, 3, _WHITE, -1, cv2.LINE_AA)
    image = cv2.flip(image, 1)
    cv2.rectangle(image, (0, 0), (size - 1, size - 1), _GREEN if hand_now else _RED, 6)
    if banner:
        _put(cv2, image, banner, (16, size - 20), _fit(cv2, banner, size - 32, 0.55, 1),
             _AMBER, 1)

    frame = np.zeros((VIEW_TOP + size + VIEW_BOTTOM, size, 3), np.uint8)
    frame[VIEW_TOP:VIEW_TOP + size] = image
    scale = phrase_scale(cv2)
    (w, h), _base = cv2.getTextSize(phrase, _FONT, scale, 3)
    _put(cv2, frame, phrase, (max(12, (size - w) // 2), 20 + h), scale, _WHITE, 3)
    x0, x1, y0, y1 = 24, size - 24, 86, 110
    fill = x0 + int(round((x1 - x0) * min(1.0, max(0.0, float(fraction_left)))))
    if fill > x0:
        cv2.rectangle(frame, (x0, y0), (fill, y1), _AMBER, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), _GREY, 1)
    _put(cv2, frame, status, (14, VIEW_TOP + size + 34),
         _fit(cv2, status, size - 28, 0.75, 2), _WHITE if hand_now else _AMBER, 2)
    return frame


class CoachWindow:
    """The camera window, drawn in this process from the test's own stream.

    Not a second client of the tracking service: the IR picture and the
    hands come from the same connection that records the run, so what the
    window shows is exactly what the report is about.
    """

    def __init__(self, stream: QualityStream):
        import cv2
        self._cv2 = cv2
        self.stream = stream
        stream.keep_frames = True
        self.to_px = leapc_projector(stream.connection)
        self.started = time.time()
        self._drawn_at = 0.0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, VIEW_SIZE, VIEW_TOP + VIEW_SIZE + VIEW_BOTTOM)
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    def show(self, phrase: str, fraction_left: float, status: str, hands,
             hand_now: bool, now: float) -> bool:
        """Redraw, at most 30 times a second. False once the operator has
        closed the window (q, Esc or its close box)."""
        if now - self._drawn_at < VIEW_EVERY_S:
            return True
        self._drawn_at = now
        cv2 = self._cv2
        frame = self.stream.frame
        stats = self.stream._image
        seen = [t for t in (None if frame is None else frame[1],
                            None if stats is None else stats[2]) if t is not None]
        banner = image_banner(now, max(seen) if seen else None, self.started)
        cv2.imshow(WINDOW_NAME, compose_frame(
            None if frame is None else frame[0], hands, phrase, fraction_left, status,
            self.to_px, banner, hand_now))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(WINDOW_NAME)
            self._cv2.waitKey(1)
        except Exception:
            pass


# --- the run ----------------------------------------------------------------------
def out_paths(out_dir: Path, stamp: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"tracking_quality_{stamp}"
    k = 2
    while base.with_suffix(".csv").exists() or base.with_suffix(".txt").exists():
        base = out_dir / f"tracking_quality_{stamp}_{k}"
        k += 1
    return base.with_suffix(".csv"), base.with_suffix(".txt")


def write_csv(path: Path, samples, prefer, t0) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SERIES_COLUMNS))
        writer.writeheader()
        writer.writerows(series_rows(samples, prefer, t0))


def run(args) -> int:
    mock = bool(args.mock)
    hand = args.hand
    print(setup_line(hand), flush=True)
    # Started now, so the voice's second of start-up is spent while the
    # camera connects and the operator gets ready.
    voice = Voice(None if args.no_voice else VOICE_COMMAND)
    beep_on = not args.no_beep

    notes: List[str] = []
    if mock:
        source = MockSource(args.seconds, hand)
        source.start()
        notes.append("Mock run: a scripted hand, IR brightness and device status, no "
                     "camera. Nothing here is evidence about the real camera.")
    else:
        source = QualityStream(mode=args.mode)
        try:
            source.start()
        except LeapUnavailable as e:
            voice.close()
            print(f"\nNo live tracking: {e}\n")
            return 2
        policy = source.enable_images()
        if policy:
            notes.append(f"Image policy: {policy}.")

    window = None
    if not mock and not args.no_view:
        try:
            window = CoachWindow(source)
        except Exception as e:                   # pragma: no cover - display path
            notes.append(f"The camera window could not open ({type(e).__name__}: {e}).")
    live = LiveLossTracker(prefer=hand)
    series: List[Sample] = []
    span = move_length(args.seconds)
    shown = ([], 0.0)            # the newest frame's hands, and when it was drained
    stopped_early = None
    stopped_how = "with Ctrl+C"
    end_beep_until = 0.0
    t_run0 = None

    def drain(keep: bool) -> None:
        """Take the frames due; `keep` adds them to the run's series."""
        nonlocal shown
        recs = source.records()
        if keep:
            for rec in recs:
                sample = to_sample(rec)
                series.append(sample)
                live.add(sample)
        if recs:
            shown = (recs[-1].hands, time.time())

    def draw(phrase: str, fraction_left: float) -> bool:
        """Redraw the window; False once the operator has closed it."""
        if window is None:
            return True
        now = time.time()
        hands = shown[0] if now - shown[1] < HANDS_FRESH_S else []
        return window.show(phrase, fraction_left, status_line(live, bool(hands)),
                           hands, bool(hands), now)

    def cue(k: int) -> None:
        """A beep and the phrase spoken, as move `k` starts."""
        if beep_on:
            live_beep(*(BEEP_FIRST if k == 0 else BEEP_MOVE))
        voice.say(MOVES[k])

    try:
        closed = False
        if not mock:
            t_ready = time.time() + GET_READY_S
            while time.time() < t_ready:
                drain(keep=False)                # the hand in the window, not the run
                if not draw(GET_READY, (t_ready - time.time()) / GET_READY_S):
                    closed = True
                    break
                time.sleep(0.01)
        if closed:
            stopped_early, stopped_how = 0.0, "from the camera window"
        else:
            t_run0 = time.time()
            k_shown = -1
            while True:
                elapsed = time.time() - t_run0
                if elapsed >= args.seconds:
                    break
                k = move_index(elapsed, args.seconds)
                if k != k_shown:
                    # One line per move, even if a slow pass stepped over one.
                    for j in range(k_shown + 1, k + 1):
                        print(move_line(j), flush=True)
                    k_shown = k
                    cue(k)
                drain(keep=True)
                if not draw(MOVES[k], ((k + 1) * span - elapsed) / span):
                    stopped_early = time.time() - t_run0
                    stopped_how = "from the camera window"
                    break
                time.sleep(0.005)
    except KeyboardInterrupt:
        stopped_early = time.time() - (t_run0 or time.time())
    finally:
        if t_run0 is not None:                   # the frames of the last pass
            try:
                drain(keep=True)
            except Exception:
                pass
        source.stop()
        if window is not None:
            window.close()
        if beep_on:
            live_beep(*BEEP_END)
            end_beep_until = time.time() + BEEP_END[1] / 1000.0
        voice.close()

    if stopped_early is not None:
        notes.append(f"Stopped early {stopped_how} after {stopped_early:.1f} s.")
    if not mock:
        notes.extend(real_notes(source))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path, txt_path = out_paths(Path(args.out), stamp)
    t0 = series[0].t if series else 0.0
    report = summarise(series, prefer=hand, t_start=t0,
                       t_end=series[-1].t if series else None)
    who = f"bare {hand} hand" if hand else "bare hand"
    setup = (f"camera only, {who}; module flat on the desk, lenses up, desktop mode"
             + ("; MOCK camera" if mock
                else f"; device {source.device_serial or 'serial not read'}"))
    lines = format_report(report, title=f"Tracking quality test, {stamp}",
                          setup=setup, notes=notes)
    write_csv(csv_path, series, hand, t0)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPer-frame series: {csv_path}")
    print(f"Report:           {txt_path}\n")
    for line in lines:
        print(line)
    sys.stdout.flush()
    # The end beep plays on a daemon thread: let it finish before the exit.
    time.sleep(max(0.0, end_beep_until - time.time()))
    return 0


def real_notes(stream: QualityStream) -> List[str]:
    """What the camera could and could not tell this run, stated, not guessed."""
    notes = []
    if stream.images:
        notes.append(f"IR images: {stream.images} brightness readings "
                     f"({1 / IMAGE_EVERY_S:.0f} per second, left lens).")
    else:
        notes.append("IR images: none arrived, so image brightness is not available and "
                     "'bright background' could not be judged. Turn on 'Allow Images' "
                     "in the Ultraleap Control Panel and run it again.")
    if stream.status_events:
        raw = "-" if stream.status_raw is None else f"0x{stream.status_raw:08X}"
        notes.append(f"Device status: {stream.status_events} device event(s) from the "
                     f"service, last raw value {raw}, decoded here from the raw value "
                     "(the bindings' own decoding misreads plain streaming as bad "
                     "calibration and bad transport). LeapC has no low frame rate or "
                     "low USB bandwidth flag; the frame rate above is measured instead.")
    else:
        notes.append("Device status: the service sent no device event to this program, "
                     "so the smudged, robust (infrared interference) and low resource "
                     "flags could not be read.")
    notes.append(f"Dropped-frame events from the service: {stream.dropped_frame_events}.")
    return notes


def recompute(args) -> int:
    """`--from FILE`: the report again, from a saved CSV or a take's JSONL."""
    path = Path(args.from_file)
    if not path.is_file():
        print(f"no such file: {path}")
        return 2
    notes = [f"Recomputed from {path}."]
    gap_s = LOSS_GAP_S
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8", newline="") as fh:
            samples = samples_from_series_rows(csv.DictReader(fh))
    else:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        samples = rows_to_samples(rows)
        notes.append("A recording holds only frames with a hand in them: no image "
                     "brightness and no device status, so those causes are not judged.")
        gap_s = cadence_gap_s(samples, args.hand)
        if gap_s > LOSS_GAP_S:
            notes.append(f"This file was saved at a reduced rate, so only a hole longer "
                         f"than {gap_s * 1000:.0f} ms counts as a loss.")
    report = summarise(samples, prefer=args.hand, gap_s=gap_s)
    for line in format_report(report, title=f"Tracking quality, {path.name}",
                              notes=notes):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="The 60 second tracking test: camera only, bare hand. Explains "
                    "every time the camera loses the hand.")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="length of the coached run, split evenly across the six "
                        "moves (default: 60, so 10 s a move)")
    p.add_argument("--hand", choices=("left", "right"), default=None,
                   help="the operator's hand; preferred when the tracker sees two "
                        "(its label is not trusted otherwise)")
    p.add_argument("--mock", action="store_true",
                   help="no hardware: a scripted mock hand, lost on purpose")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window (the mock never opens it)")
    p.add_argument("--no-voice", action="store_true",
                   help="do not speak each move with the Windows voice")
    p.add_argument("--no-beep", action="store_true",
                   help="no beeps")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help=f"folder for the CSV and the report (default: {DEFAULT_OUT})")
    p.add_argument("--mode", default="desktop", choices=("desktop", "hmd", "screentop"))
    p.add_argument("--from", dest="from_file", default=None,
                   help="re-read a saved tracking_quality CSV or a take's JSONL and "
                        "print the report, no camera")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.seconds <= 0:
        print("--seconds must be above 0")
        return 2
    if args.from_file:
        return recompute(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
