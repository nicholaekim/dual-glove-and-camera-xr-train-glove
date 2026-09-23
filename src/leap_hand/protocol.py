"""What the operator is asked to do, as data: acquire, schedule, band, HUD.

Two sessions need the same coaching — `scripts/record_simultaneous.py`
(Path A takes) and `scripts/leap/gate.py` (the glove-vs-bare gate) — and both
had it written as prose in a docstring, which is why two sessions in a row
produced numbers nobody could read. This module makes the protocol a value.

Why it exists, measured on real takes (2026-09-17, left gloved hand, one hand
at a time, `recordings/sync/`):

  * `open_palm` tracked 3/3 takes on one continuous hand id. Every other pose
    failed: fist 34 %/0 %/66 %, index_point 0/8 %/tracked-but-labelled-RIGHT,
    thumbs_up 0/0/0, pinch 0 as left, peace 1/3.
  * The tracked hand id changed on almost every take and was usually acquired
    LATE. That is the finding: the tracker will follow an open hand into a
    closed pose, but it cannot acquire a gloved hand that is already closed —
    and when it does re-acquire from a closed pose it sometimes guesses the
    wrong chirality, i.e. a mirrored skeleton labelled as the other hand.
  * The hand sat at 131-154 mm, near the bottom of the working range. The
    takes that worked earlier sat at 160-210 mm.
  * The operator's idle other hand, 20 cm off to the side, was picked up and
    recorded.

So the protocol becomes four values:

  acquire    the hand must be OPEN, settled, high enough, facing the lens and
             roughly on the module's axis before a pose is ever asked for.
             `acquire_failures` is that gate, and it is the same one in both
             scripts.
  schedule   which pose is being held, second by second, identically in every
             condition. Each recorded frame carries the plan and its own
             offset into it, so `bare` and `glove` are compared fist against
             fist instead of take against take.
  band       how high above the module the hand is supposed to be. Out of band
             is never blocked — the run continues and the frames are recorded
             — but it is flagged live and counted in the report.
  hud        one line, rewritten in place, that says all of it while the
             operator's hands are busy and their eyes are on the camera.

Everything here is plain values and formatting with no I/O, so the live
sessions can run it and `leap_hand.stats` can re-apply it to a recording
months later.
"""
import math
import re
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Callable, List, Optional, Sequence, Tuple

from cam_hand.features import palm_normal
# Imported, never reimplemented: the viewing angle the fusion gates already
# judge a camera frame on is the one the operator is coached against, or the
# take would pass live and be thrown away later.
from cam_hand.fusion import viewing_angle_deg
from xr_hand.keypoints21 import MP21_TO_OPENXR_IDX

# The four poses of the default schedule, in order. `pinch` and `fist` are the
# ones the glove is expected to make hardest for the camera (the fabric closes
# over itself and the fingertips disappear into the palm), which is exactly
# why they are in the schedule rather than left to whatever the operator
# happened to do.
DEFAULT_POSES: Tuple[str, ...] = ("open_palm", "fist", "pinch", "spread")
DEFAULT_SECONDS = 20.0

# Centimetres above the module. The plan's working volume is 20-50 cm, but the
# takes that actually tracked sat at 16-21 cm and the ones that failed sat at
# 13-15 cm, so the band the operator is held to is the narrow middle of what
# has been observed to work rather than the whole documented volume.
DEFAULT_BAND: Tuple[float, float] = (18.0, 28.0)
# A condition named `<anything>_<N>cm` is a distance run: it says its own band.
NAMED_BAND_TOLERANCE = 5.0
_NAMED_BAND_RE = re.compile(r"_(\d+(?:\.\d+)?)cm$")

IN_BAND = "OK"
TOO_LOW = "TOO LOW"
TOO_HIGH = "TOO HIGH"

# --- the acquire gate --------------------------------------------------------
# A hand counts as acquired only when all four hold at once. The thresholds are
# the plan's and the fusion gates', not new numbers.
#
#   visible_time  500 ms, not the 300 ms the recorders settle on. 300 ms is
#                 "this is data"; this is "the tracker has a stable hold on it
#                 and will follow it into a pose".
#   view angle    40 degrees, the angle beyond which `cam_hand.fusion` stops
#                 trusting a camera-owned DOF. Coaching to a looser number
#                 than the analysis uses just moves the failure downstream.
#   centring      lateral offset below 60 % of the height, i.e. the palm is
#                 nearer the module's axis than 31 degrees off it. This is
#                 what keeps the idle other hand, 20 cm off to the side, from
#                 being the one that gets acquired.
ACQUIRE_VISIBLE_TIME_US = 500_000
VIEW_ANGLE_MAX_DEG = 40.0
LATERAL_FRACTION_MAX = 0.60

# A take is `complete` when the expected hand, on ONE hand id, covers this
# much of it. Below that the pose was recorded but the skeleton behind it was
# not one continuous track, which is the failure this whole module is about.
COMPLETE_COVERAGE = 0.90
# A hole this long or longer is a loss rather than a dropped frame. At 90 Hz a
# tracking interval is 11 ms, so 250 ms is ~22 missed frames in a row.
LOSS_GAP_S = 0.25

DEFAULT_SETTLE = 1.5
DEFAULT_RETRIES = 3

# Why a hand is not acquired yet, shortest first — these are HUD strings.
NOT_TRACKED = "no hand"
WRONG_HAND = "WRONG HAND"
TOO_YOUNG = "hold still"
NOT_FACING = "turn palm to lens"
OFF_AXIS = "centre over module"

# What the operator is told at the ACQUIRE prompt, per pose. Only poses whose
# FIRST attempt is what fails need one: the tracker follows an open hand into
# any pose, so the hint is about how to present the pose once it is asked for,
# not about how to make the shape.
PRESENTATION_HINTS = {
    "thumbs_up": ("tilt the whole forearm 30-45 degrees so the camera still "
                  "sees some palm; keep the finger shape"),
    "pinch": "pinch with the palm facing the lens",
    "fist": "close slowly",
}


@dataclass(frozen=True)
class PoseWindow:
    """One block of the schedule: hold `pose` from `start` to `end` seconds."""

    index: int
    pose: str
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Schedule:
    """A timed sequence of poses, and the text it was written as.

    `text` round-trips: it is what `--schedule` accepts, what is written into
    every recorded frame as `pose_plan`, and what the report prints. A
    recording is therefore self-describing — `--recompute` needs nothing but
    the file to know which second was which pose.
    """

    windows: Tuple[PoseWindow, ...]
    text: str

    @property
    def total(self) -> float:
        return self.windows[-1].end if self.windows else 0.0

    @property
    def poses(self) -> List[str]:
        """The distinct pose names, in the order they are first held."""
        out: List[str] = []
        for w in self.windows:
            if w.pose not in out:
                out.append(w.pose)
        return out

    def windows_for(self, pose: str) -> List[PoseWindow]:
        return [w for w in self.windows if w.pose == pose]

    def seconds_of(self, pose: str) -> float:
        """Total scheduled seconds of one pose, repeats added together."""
        return sum(w.seconds for w in self.windows_for(pose))

    def at(self, t: float) -> Optional[PoseWindow]:
        """The window holding second `t`, or None past the end."""
        for w in self.windows:
            if w.start <= t < w.end:
                return w
        return None

    def pose_at(self, t: float) -> str:
        """The pose being held at second `t`, or "" past the end."""
        w = self.at(t)
        return w.pose if w else ""

    def boundaries(self) -> List[float]:
        """The instants a new pose starts, the first one excluded."""
        return [w.start for w in self.windows[1:]]


def parse_schedule(text: str) -> Schedule:
    """`"open_palm:5,fist:5"` -> a `Schedule`. Raises ValueError with the fix.

    Durations are seconds and may be fractional. The total is the sum, which
    is what makes `--seconds` redundant once a schedule is given.
    """
    windows: List[PoseWindow] = []
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError(
            "empty schedule. Write it as pose:seconds pairs, for example "
            f"--schedule \"{','.join(f'{p}:5' for p in DEFAULT_POSES)}\"")
    t = 0.0
    for i, part in enumerate(parts):
        name, sep, secs = part.partition(":")
        name = name.strip()
        if not sep or not name:
            raise ValueError(
                f"schedule entry {part!r} is not pose:seconds — write it as "
                "open_palm:5")
        try:
            seconds = float(secs)
        except ValueError:
            raise ValueError(
                f"schedule entry {part!r} has a non-numeric duration "
                f"({secs!r} is not seconds)") from None
        if seconds <= 0:
            raise ValueError(f"schedule entry {part!r} lasts {seconds:g} s; "
                             "every pose needs a positive duration")
        windows.append(PoseWindow(index=i, pose=name, start=t, end=t + seconds))
        t += seconds
    return Schedule(windows=tuple(windows), text=format_schedule(windows))


def format_schedule(windows: Sequence[PoseWindow]) -> str:
    """The canonical `pose:seconds,...` text of a list of windows."""
    return ",".join(f"{w.pose}:{w.seconds:g}" for w in windows)


def default_schedule(seconds: float = DEFAULT_SECONDS) -> Schedule:
    """The four default poses, sharing `seconds` equally.

    At the default 20 s that is the plan's schedule exactly — open_palm 0-5,
    fist 5-10, pinch 10-15, spread 15-20 — and a shorter `--seconds` scales it
    rather than dropping poses, so a 4 s smoke test still visits all four.
    """
    each = float(seconds) / len(DEFAULT_POSES)
    return parse_schedule(",".join(f"{p}:{each:g}" for p in DEFAULT_POSES))


def parse_band(text: str) -> Tuple[float, float]:
    """`"18,28"` -> (18.0, 28.0), centimetres. Raises ValueError with the fix."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--band wants LOW,HIGH in centimetres, got {text!r}")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"--band wants two numbers in centimetres, got "
                         f"{text!r}") from None
    if low >= high:
        raise ValueError(f"--band low must be below high, got {low:g},{high:g}")
    return low, high


def band_for_condition(condition: str,
                       override: Optional[Tuple[float, float]] = None
                       ) -> Tuple[float, float]:
    """The height band this condition is held to, in centimetres.

    `--band` wins if given; then a name ending `_<N>cm` (`glove_20cm`,
    `bare_50cm`), which means N plus or minus 5 cm and is the whole point of
    running a distance sweep; otherwise `DEFAULT_BAND`.
    """
    if override is not None:
        return override
    m = _NAMED_BAND_RE.search(condition or "")
    if m:
        centre = float(m.group(1))
        return (centre - NAMED_BAND_TOLERANCE, centre + NAMED_BAND_TOLERANCE)
    return DEFAULT_BAND


def baseline_condition(condition: str, available: Sequence[str],
                       reference: str = "bare") -> Optional[str]:
    """The bare condition this gloved one is compared against, or None.

    A distance run is compared against the bare run at the SAME distance where
    there is one (`glove_20cm` -> `bare_20cm`), because detection falls off
    with height and comparing 20 cm against 35 cm measures the height, not the
    glove. Otherwise the plain reference.
    """
    m = _NAMED_BAND_RE.search(condition or "")
    if m:
        named = f"{reference}_{m.group(1)}cm"
        if named in available:
            return named
    return reference if reference in available else None


def band_flag(height_cm: Optional[float],
              band: Tuple[float, float]) -> str:
    """`OK` / `TOO LOW` / `TOO HIGH` for one palm height."""
    if height_cm is None:
        return ""
    low, high = band
    if height_cm < low:
        return TOO_LOW
    if height_cm > high:
        return TOO_HIGH
    return IN_BAND


def band_text(band: Tuple[float, float]) -> str:
    return f"{band[0]:g}-{band[1]:g} cm"


# --- measuring one hand ------------------------------------------------------
def palm_height_cm(palm_abs: Sequence[float]) -> float:
    """Height above the module, centimetres. Leap desktop space puts +y up."""
    return float(palm_abs[1]) * 100.0


def lateral_offset_cm(palm_abs: Sequence[float]) -> float:
    """Distance from the module's vertical axis, centimetres."""
    return math.hypot(float(palm_abs[0]), float(palm_abs[2])) * 100.0


def palm_normal_abs(abs26: Sequence[Sequence[float]], hand_side: str
                    ) -> List[float]:
    """The palm normal in ABSOLUTE leap space, oriented alike on both hands.

    `viewing_angle_deg` needs the normal in the same space as the palm
    position, because the module is the origin of that space and so the palm's
    position IS the direction it is being seen from. `features.palm_normal`
    already flips the normal for a left hand, which is what lets one threshold
    cover both.
    """
    return palm_normal([abs26[i] for i in MP21_TO_OPENXR_IDX], hand_side)


def hand_view_angle_deg(lh) -> Optional[float]:
    """Viewing angle of one `LeapHand`, degrees, or None without geometry."""
    if not getattr(lh, "abs26", None) or not getattr(lh, "palm_pos", None):
        return None
    return viewing_angle_deg(lh.palm_pos,
                             palm_normal_abs(lh.abs26, lh.hand_side))


def row_view_angle_deg(row: dict) -> Optional[float]:
    """The same angle, from a recorded JSONL line. None without geometry.

    The live session and `--recompute` have to agree to the degree, so they
    read the same two keys through the same function rather than each
    rebuilding the palm basis their own way.
    """
    palm, abs26 = row.get("palm_abs"), row.get("abs26")
    if not palm or not abs26:
        return None
    return viewing_angle_deg(
        palm, palm_normal_abs(abs26, row.get("hand_side", "right")))


def row_height_cm(row: dict) -> Optional[float]:
    palm = row.get("palm_abs")
    return None if not palm else palm_height_cm(palm)


@dataclass
class HandReading:
    """The four numbers the coaching turns on, for one tracked hand."""

    hand_side: str
    hand_id: int
    visible_time_us: int
    height_cm: float
    lateral_cm: float
    view_angle_deg: Optional[float]

    @property
    def centred_fraction(self) -> float:
        """Lateral offset as a fraction of height. Above 1 is beside, not over."""
        if self.height_cm <= 0:
            return float("inf")
        return self.lateral_cm / self.height_cm


def read_hand(lh) -> HandReading:
    """One `LeapHand` -> the numbers the acquire gate and the HUD both read."""
    return HandReading(
        hand_side=lh.hand_side,
        hand_id=lh.hand_id,
        visible_time_us=int(lh.visible_time_us),
        height_cm=palm_height_cm(lh.palm_pos),
        lateral_cm=lateral_offset_cm(lh.palm_pos),
        view_angle_deg=hand_view_angle_deg(lh),
    )


def acquire_failures(reading: Optional[HandReading],
                     expected_hand: str,
                     band: Tuple[float, float]) -> List[str]:
    """Why this hand is not ready to be given a pose. Empty = acquired.

    The order is the order the operator can fix them in: be there, be the
    right hand, hold still, get the height right, turn the palm, centre it.
    """
    if reading is None:
        return [NOT_TRACKED]
    if reading.hand_side != expected_hand:
        return [WRONG_HAND]
    bad: List[str] = []
    if reading.visible_time_us < ACQUIRE_VISIBLE_TIME_US:
        bad.append(TOO_YOUNG)
    flag = band_flag(reading.height_cm, band)
    if flag != IN_BAND:
        bad.append(flag)
    angle = reading.view_angle_deg
    if angle is None or angle >= VIEW_ANGLE_MAX_DEG:
        bad.append(NOT_FACING)
    if reading.centred_fraction >= LATERAL_FRACTION_MAX:
        bad.append(OFF_AXIS)
    return bad


def acquire_prompt(pose: str, expected_hand: str) -> List[str]:
    """The lines printed at the start of an attempt, hint included.

    OPEN PALM first, always: the measured failure is that the tracker cannot
    acquire a gloved hand that is already in a closed pose, so every pose is
    reached by transition from an open one.
    """
    lines = [f"ACQUIRE: hold the {expected_hand.upper()} hand OPEN PALM over "
             "the camera, palm down, fingers spread."]
    hint = PRESENTATION_HINTS.get(pose)
    if hint:
        lines.append(f"Then, for {pose.replace('_', ' ')}: {hint}")
    return lines


# --- coverage ----------------------------------------------------------------
def coverage(times: Sequence[float], t0: float, t1: float,
             gap_s: float = LOSS_GAP_S) -> float:
    """Fraction of [t0, t1] the hand was actually there, 0..1.

    Not "frames recorded / frames expected": that needs a denominator nobody
    agrees on (see `stats.choose_rate`) and it scores a throttled file at 6 %.
    This measures the hole instead — every stretch of the take longer than
    `gap_s` with no frame in it, the head and the tail included — which is the
    thing being asked about, and is the same number whatever the save rate is.
    """
    span = float(t1) - float(t0)
    if span <= 0:
        return 0.0
    marks = sorted(float(t) for t in times if t0 <= t <= t1)
    if not marks:
        return 0.0
    lost = 0.0
    for a, b in zip([t0] + marks, marks + [t1]):
        hole = b - a
        if hole > gap_s:
            lost += hole
    return max(0.0, min(1.0, (span - lost) / span))


def longest_gap(times: Sequence[float], t0: Optional[float] = None,
                t1: Optional[float] = None) -> float:
    """The longest stretch with no frame in it, seconds. The 'longest loss'."""
    marks = sorted(float(t) for t in times)
    if not marks:
        return float(t1) - float(t0) if (t0 is not None and t1 is not None) else 0.0
    edges = ([float(t0)] if t0 is not None else []) + marks + (
        [float(t1)] if t1 is not None else [])
    return max((b - a for a, b in zip(edges, edges[1:])), default=0.0)


def median(values: Sequence[float]) -> Optional[float]:
    """Plain median, None for an empty sequence. No numpy for three numbers."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


# --- the live line -----------------------------------------------------------
HUD_EVERY = 0.25                    # ~4 refreshes a second


def hud_line(phase: str, seconds_left: Optional[float],
             reading: Optional[HandReading], expected_hand: str,
             band: Tuple[float, float], glove_hz: Optional[float] = None,
             saw_other_hand: bool = False, extra: str = "") -> str:
    """The one line the operator reads while their hands are over the camera.

    Everything on it is actionable in under a second: which phase, how long
    is left, is the right hand seen, is it at the right height, is the palm
    turned the right way, is the glove still talking. `WRONG HAND` is called
    out separately from `no hand` because the fix is completely different —
    one is "put your hand there", the other is "the tracker has decided your
    left hand is a right hand, open it and start again".

    `glove_hz` is None where there is no glove in the session (the gate runs
    the camera alone), and the field is left off rather than printed as zero.
    """
    left = "  --" if seconds_left is None else f"{max(0.0, seconds_left):4.1f}s"
    if reading is not None and reading.hand_side == expected_hand:
        tracked = f"{expected_hand} YES"
        height = f"{reading.height_cm:5.1f} cm {band_flag(reading.height_cm, band):<8}"
        angle = ("  --" if reading.view_angle_deg is None
                 else f"{reading.view_angle_deg:3.0f}")
        angle = f"view {angle} deg"
    else:
        tracked = WRONG_HAND if saw_other_hand else f"{expected_hand} no "
        height = f"{'   -- cm':<9} {'':<8}"
        angle = "view  -- deg"
    line = f"{phase:<8} {left}  {tracked:<12} {height} {angle}"
    if glove_hz is not None:
        line += f"  glove {glove_hz:5.1f}/s"
    if extra:
        line += f"  {extra}"
    return line


# --- the window's caption ----------------------------------------------------
# The operator's eyes are on the camera window, not on the terminal. Until
# 2026-09-17 the window's caption was `f"{phase}  {extra}{secs}"` and `extra`
# is empty during SETTLE and REC, so the window said "SETTLE   1s" and never
# once named the pose — "NOW: FIST" was printed to a terminal nobody was
# looking at. Nine of that session's 36 takes hold the wrong pose, on both
# sensors at once. So the caption is a function, with tests, and every phase
# of it says which pose is wanted.
#
# The window is 768 px wide and the caption is drawn at font scale 0.75, which
# is about 60 characters. Everything here is built to fit that.
WRONG_POSE = "WRONG POSE"


def pose_label(pose: str, stay: bool = True) -> str:
    """`fist` -> `FIST`; `open_palm` -> `OPEN PALM (stay)`.

    In the coached recorder the hand is ALREADY open when the pose is called
    — every take is acquired open palm — so "NOW: OPEN PALM" reads as an
    instruction to change something, and `(stay)` is the whole difference
    between an operator who moves and one who does not. `stay=False` for a
    caller whose hand is not coming from an open palm: the gate runs a timed
    schedule where `open_palm` can follow a fist and really does mean open.
    """
    if not pose:
        return ""
    text = str(pose).replace("_", " ").upper()
    return f"{text} (stay)" if (stay and str(pose) == "open_palm") else text


def view_caption(phase: str, pose: str, extra: str = "",
                 seconds_left: Optional[float] = None,
                 take: Optional[int] = None,
                 takes: Optional[int] = None,
                 stay: bool = True) -> str:
    """The one line on the camera window. Names the pose in EVERY phase.

        ACQUIRE   OPEN PALM first   next: FIST (take 2/3)
        SETTLE    NOW: FIST   1s
        REC       HOLD: FIST   REC 4s

    ACQUIRE leads with what the hand has to do NOW (open) and then what is
    coming, because a hand that is already closed cannot be acquired at all —
    that is the measured failure the whole protocol is built around. SETTLE
    and REC lead with the pose, because by then it is the only thing left to
    get wrong.
    """
    ph = str(phase or "").upper()
    secs = ("" if seconds_left is None
            else f"{max(0.0, float(seconds_left)):.0f}s")
    label = pose_label(pose, stay)
    extra = (extra or "").strip()

    if ph == WRONG_POSE:
        return f"{WRONG_POSE}: {extra}" if extra else WRONG_POSE
    if ph == "ACQUIRE":
        parts = ["OPEN PALM first"]
        if label:
            nxt = f"next: {label}"
            if take and takes:
                nxt += f" (take {take}/{takes})"
            parts.append(nxt)
        if extra:
            parts.append(extra)
        if secs:
            parts.append(secs)
        return "   ".join(parts)
    if ph == "SETTLE":
        parts = [f"NOW: {label}" if label else "NOW"]
        if secs:
            parts.append(secs)
        if extra:
            parts.append(extra)
        return "   ".join(parts)
    if ph == "REC":
        parts = [f"HOLD: {label}" if label else "HOLD"]
        parts.append(f"REC {secs}" if secs else "REC")
        if extra:
            parts.append(extra)
        return "   ".join(parts)
    parts = [ph] if ph else []
    if label:
        parts.append(label)
    if extra:
        parts.append(extra)
    if secs:
        parts.append(secs)
    return "   ".join(parts)


# --- the status file the window is driven through ----------------------------
# One tiny text file, rewritten only when something changes. Line 1 is the
# caption; the rest are `key=value` and may be in any order, so a new key can
# be added without the viewer and the recorder having to agree on line
# numbers. `__quit__` as the caption closes the window.
QUIT = "__quit__"


def status_text(caption: str, band=None, snap=None) -> str:
    """The status file's contents, as the viewer parses them back."""
    body = (caption or "").strip()
    if band:
        body += f"\nband={band[0]:g},{band[1]:g}"
    if snap:
        body += f"\nsnap={snap}"
    return body


def parse_status(text: str):
    """`status_text` backwards: (caption, band or None, snap or None).

    Unknown keys and malformed values are ignored rather than raising: the
    viewer is a child process watching a file that the parent may be halfway
    through rewriting, and a viewer that dies on a torn read is worse than
    one that shows the previous caption for another 30 ms.
    """
    lines = (text or "").splitlines()
    caption = lines[0].strip() if lines else ""
    band = None
    snap = None
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("band="):
            try:
                low, high = (float(v) for v in line[5:].split(","))
            except ValueError:
                continue
            if low < high:
                band = (low, high)
        elif line.startswith("snap="):
            snap = line[5:].strip() or None
    return caption, band, snap


# --- the viewer's "no picture" banner ----------------------------------------
# How long the IR image may be missing before the window says so. A few
# seconds: long enough that the first frame after connect, or one dropped
# image event, does not raise an alarm, short enough that a viewer showing
# black says why while the operator is still looking at it.
IMAGE_STALE_S = 3.0
IMAGES_OFF = ("camera images are off: enable 'Allow Images' in the Ultraleap "
              "Control Panel")


def image_banner(now: float, last_image_time: Optional[float],
                 started: float, stale_s: float = IMAGE_STALE_S) -> str:
    """The warning the camera window shows about the IR image, or "".

    Until 2026-09-20 the window decided this from the ANSWER to
    `set_policy_flags(Images)` alone, and it was wrong in the one direction
    that matters: every still in `recordings/sync_day2/stills/` carries
    "camera images are off" printed over a perfectly good IR picture of the
    hand. The policy call is a request whose reply the service may report in
    a form the viewer did not recognise; whether the picture is THERE is a
    fact about the stream, so it is read off the stream.

    Three states, because they need three different things from whoever is
    reading the window:

      ""                image events are arriving; say nothing
      IMAGES_OFF        none has EVER arrived and the grace period is over —
                        the policy really is off, and the fix is in the
                        Control Panel
      "stopped ...s ago" images were arriving and stopped, which is a
                        different fault (the service, the USB, another app
                        taking the device) and must not be reported as a
                        setting the operator has already set.

    `started` is when the viewer connected, so the grace period is measured
    from there and a window that has only just opened does not accuse the
    Control Panel of anything.
    """
    if last_image_time is not None and now - last_image_time <= stale_s:
        return ""
    if last_image_time is None:
        return IMAGES_OFF if now - started > stale_s else ""
    return f"camera images stopped {now - last_image_time:.0f} s ago"


# --- the per-take still --------------------------------------------------------
def hand_crop_box(points: Sequence[Sequence[float]], size: int,
                  margin: float = 0.35, min_side: int = 192
                  ) -> Optional[Tuple[int, int, int, int]]:
    """The part of the camera frame to keep as a take's still, or None.

    The window is a square `size` pixels on a side, and the IR camera looks up
    past the hand at whoever is holding it, so a still of the whole frame has
    the operator's face in it. Those stills had to be cropped by hand before
    anyone else could see them. This box is what the viewer crops to instead.
    It reduces what the still shows to the hand and a margin around it; a
    hand held near the face can still bring part of the face into the box,
    which is one reason the stills stay out of git.

    `points` are the (x, y) pixels of the tracked hand in the frame the still
    is cut from. Points outside the frame are ignored (that part of the hand
    is not in the picture). The box around the rest grows on every side by
    `margin` times its larger side, so the fingertips are not at the edge,
    and is made at least `min_side` wide and tall around its centre, so a
    small or distant hand still gives a still that can be read. A box that
    runs off the frame is slid back inside it before it is cut to the frame,
    so a hand near the edge keeps a full-size still whenever the frame is big
    enough for one.

    Returns (x0, y0, x1, y1) with x1 and y1 exclusive, ready for
    `image[y0:y1, x0:x1]`, or None when no point is inside the frame: with no
    hand to crop to, the caller writes no still rather than the whole frame.
    """
    inside = [(float(p[0]), float(p[1])) for p in points
              if 0 <= p[0] < size and 0 <= p[1] < size]
    if not inside or size <= 0:
        return None
    xs = [p[0] for p in inside]
    ys = [p[1] for p in inside]
    lo = [min(xs), min(ys)]
    hi = [max(xs) + 1.0, max(ys) + 1.0]          # a pixel covers [x, x + 1)
    pad = margin * max(hi[0] - lo[0], hi[1] - lo[1])
    box = []
    for a in range(2):
        a0, a1 = lo[a] - pad, hi[a] + pad
        if a1 - a0 < min_side:
            mid = (a0 + a1) / 2.0
            a0, a1 = mid - min_side / 2.0, mid + min_side / 2.0
        a0, a1 = math.floor(a0), math.ceil(a1)
        if a0 < 0:                               # slide back inside...
            a0, a1 = 0, a1 - a0
        if a1 > size:
            a0, a1 = a0 - (a1 - size), size
        box.append((max(0, a0), min(size, a1)))  # ...and cut what still sticks out
    (x0, x1), (y0, y1) = box
    return int(x0), int(y0), int(x1), int(y1)


# Why a take has no still. Written into its meta.json as
# `still_missing_reason`, so a still the viewer skipped on purpose can be told
# apart from one that is missing because something failed.
NO_TRACKED_HAND = "no_tracked_hand"   # the viewer saw no hand, and wrote a note
STILL_UNKNOWN = "unknown"             # no still and no note: the viewer crashed,
                                      # was never started (--no-view, mocks), or
                                      # was never asked before the take ended
SKIPPED_SUFFIX = ".skipped.txt"


def skipped_still_path(still_path) -> Path:
    """Where the viewer leaves its note for a still it chose not to write:
    `<stem>.skipped.txt`, beside where the JPEG would have been."""
    p = Path(still_path)
    return p.with_name(p.stem + SKIPPED_SUFFIX)


def skipped_still_text(reason: str, when: float) -> str:
    """The note's contents: the reason, then when the still was asked for."""
    local = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when))
    return f"still_missing_reason={reason}\ntime={when:.3f}  ({local})\n"


def still_status(still_path) -> Tuple[Optional[str], Optional[str]]:
    """(the still's file name, None) when the JPEG is there, otherwise
    (None, why it is not).

    The why is the reason in the viewer's note when there is one (normally
    NO_TRACKED_HAND), and STILL_UNKNOWN when there is no note either, or the
    note cannot be read: nothing on disk says the still was skipped on
    purpose, so it is treated as missing for a reason nobody recorded.
    """
    p = Path(still_path)
    if p.is_file():
        return p.name, None
    note = skipped_still_path(p)
    try:
        text = note.read_text(encoding="utf-8")
    except OSError:
        return None, STILL_UNKNOWN
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "still_missing_reason" and value.strip():
            return None, value.strip()
    return None, STILL_UNKNOWN


def move_still(src, dst) -> None:
    """Move a take's still, and the viewer's note if it left one instead, to
    the name `dst`. The note follows the same renaming rule as the JPEG so
    `still_status(dst)` still finds it. Nothing to move, nothing created."""
    src, dst = Path(src), Path(dst)
    for a, b in ((src, dst), (skipped_still_path(src), skipped_still_path(dst))):
        if a.is_file():
            b.parent.mkdir(parents=True, exist_ok=True)
            a.replace(b)


# --- stream health -----------------------------------------------------------
def stream_health(times: Sequence[float], t0: Optional[float] = None,
                  t1: Optional[float] = None,
                  gap_s: float = 0.1) -> dict:
    """Rate, longest hole and number of holes for one sensor over one take.

    Recorded because a take can be complete by every other measure and still
    be missing seconds of one sensor: the glove was throttled to 5 Hz by
    default, which is 24 frames in a five-second take, and 24 frames cannot
    tell a steady stream from one that stopped for three seconds. The rate is
    measured between the first and last frame — the cadence the frames
    actually arrived at — while the holes are measured against the take's own
    window when one is given, so a stream that starts late is not flattered
    by its own first frame.
    """
    marks = sorted(float(t) for t in times if t is not None)
    out = {"frames": len(marks), "rate_hz": None, "max_gap_ms": None,
           "gaps_over_ms": round(gap_s * 1000.0), "gaps_over": 0}
    if not marks:
        if t0 is not None and t1 is not None and t1 > t0:
            out["max_gap_ms"] = round((t1 - t0) * 1000.0, 1)
            out["gaps_over"] = 1
        return out
    span = marks[-1] - marks[0]
    if len(marks) > 1 and span > 0:
        out["rate_hz"] = round((len(marks) - 1) / span, 2)
    edges = ([float(t0)] if t0 is not None else []) + marks + (
        [float(t1)] if t1 is not None else [])
    holes = [b - a for a, b in zip(edges, edges[1:])]
    out["max_gap_ms"] = round(max(holes, default=0.0) * 1000.0, 1)
    out["gaps_over"] = sum(1 for h in holes if h > gap_s)
    return out


class Hud:
    """Rewrites one terminal line in place, at most `every` seconds apart."""

    def __init__(self, write: Callable[[str], None], every: float = HUD_EVERY):
        self._write = write
        self.every = float(every)
        self._at = 0.0
        self._width = 0
        self.open = False

    def show(self, line: str, now: float, force: bool = False) -> bool:
        """Print `line` over the previous one. Returns whether it printed."""
        if not force and now - self._at < self.every:
            return False
        self._at = now
        pad = max(0, self._width - len(line))
        self._width = len(line)
        self._write("\r" + line + " " * pad)
        self.open = True
        return True

    def close(self) -> None:
        """End the line, so ordinary printing starts on a fresh one."""
        if self.open:
            self._write("\n")
            self.open = False
            self._width = 0


class AsyncBeeper:
    """Beeps on a worker thread, so a blocking beep never stalls the loop.

    `winsound.Beep` blocks for its whole duration. In the old protocol that
    was handled by beeping BEFORE the files were open and throwing the backlog
    away, which works for one beep at the head of a take — but the coached
    protocol beeps in the middle of a live capture clock, at the moment the
    pose is called for, and a quarter second of stalled frame loop there is a
    quarter second of hand nobody recorded. So the beep goes to a thread and
    the loop keeps draining.

    Beeps are serialised through one queue: two overlapping beeps on Windows
    are one beep and one dropped call, and the second cue is the one that
    tells the operator to move.
    """

    def __init__(self, beep: Callable[[int, int], None]):
        self._beep = beep
        self._queue: "Queue" = Queue()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._beep(*item)
            except Exception:                    # pragma: no cover - audio path
                pass

    def beep(self, freq: int = 880, ms: int = 180) -> None:
        """Ask for a beep and return immediately."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="beeper")
            self._thread.start()
        self._queue.put((freq, ms))

    def stop(self, timeout: float = 1.0) -> None:
        """Let the queued beeps finish, then retire the thread."""
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None


class CameraView:
    """The live camera window (`scripts/leap/camera_view.py`) as a child process.

    The operator cannot position a hand they cannot see: the first coached
    sessions failed on hands held 13 cm above the lens and on a camera that
    had silently locked onto the wrong hand. The window shows the IR image,
    the fitted skeleton, height against the band and palm facing, plus a
    caption this process controls through a tiny status file. It is a second,
    read-only LeapC client, so it never touches the recording connection.

    Disabled (`enabled=False`, the mocks) it is a no-op with the same API.
    """

    SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "leap" / "camera_view.py"

    def __init__(self, hand=None, band=None, enabled=True):
        self.hand = hand if hand in ("left", "right") else "both"
        self.band = band
        self.enabled = bool(enabled) and self.SCRIPT.exists()
        self._proc = None
        self._status = None
        self._last = None
        # The caption and the snapshot request share one file, so each is
        # remembered here and the file is rewritten from both. Otherwise
        # asking for a still would blank the caption for a frame, at the exact
        # moment the still is taken.
        self._caption = ""
        self._snap = None

    def start(self):
        if not self.enabled or self._proc is not None:
            return self
        import atexit
        import os
        import subprocess
        import sys
        import tempfile
        self._status = Path(tempfile.gettempdir()) / f"leap_view_status_{os.getpid()}.txt"
        self.caption("starting")
        cmd = [sys.executable, str(self.SCRIPT), "--hand", self.hand,
               "--status-file", str(self._status), "--parent-pid", str(os.getpid())]
        if self.band:
            cmd += ["--band", f"{self.band[0]:g},{self.band[1]:g}"]
        try:
            self._proc = subprocess.Popen(cmd)
        except OSError:
            self.enabled = False
            return self
        atexit.register(self.close)
        return self

    def caption(self, text, band=None):
        """Set the window's caption (and optionally its height band). Cheap to
        call every HUD refresh: the file is only rewritten when it changes."""
        self._caption = (text or "").strip()
        self._write_status(band)

    def snapshot(self, path):
        """Ask the window to save its next composed frame to `path`, once.

        The viewer crops the still to the tracked hand, which reduces what it
        shows to the hand and a margin around it; a hand held near the face
        can still include part of the face, so the file stays out of git with
        the rest of `recordings/`. With no hand tracked at that moment it
        writes no JPEG, only a `<stem>.skipped.txt` note (see `still_status`).

        The take's own evidence: the IR image, the fitted skeleton and the
        caption that was on screen, as one JPEG, so a verdict about which
        pose was held can be checked against a picture of the hand instead of
        against two files of numbers. The viewer writes it and forgets it;
        asking for the same path twice does nothing.

        A no-op with no window (the mocks, `--no-view`), like every other
        method here.
        """
        if not self.enabled or self._status is None or not path:
            return None
        self._snap = str(Path(path))
        try:
            Path(self._snap).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._write_status(self.band)
        return self._snap

    def _write_status(self, band=None):
        if not self.enabled or self._status is None:
            return
        body = status_text(self._caption, band, self._snap)
        if body == self._last:
            return
        self._last = body
        try:
            self._status.write_text(body, encoding="utf-8")
        except OSError:
            pass

    def close(self):
        if self._status is not None:
            try:
                self._status.write_text(QUIT, encoding="utf-8")
            except OSError:
                pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.terminate()
            self._proc = None
        if self._status is not None:
            try:
                self._status.unlink()
            except OSError:
                pass
            self._status = None
