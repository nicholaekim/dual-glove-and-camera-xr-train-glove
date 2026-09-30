"""Why the camera loses the hand, one loss at a time.

The complaint this module answers: "the hand tracking for the camera is not
good and it does not register sometimes". The tracking service is not the
suspect. Its own log shows 90 Hz, no dropped frames and about 10 ms of
latency; the earlier bare-hand static poses tracked 97 to 100 percent of
their frames and the day-2 gloved takes 100 percent. So the losses happen in
the moment, because of where the hand is, how it is turned, what shape it
makes, how fast it moves, what the room is doing to the infrared image, or
what the device itself reports. None of that is written down anywhere once
the moment has passed. This module turns a stream of camera samples into
that evidence: every loss, what the hand and the room were doing just before
it, the likely cause, and the fix.

It is pure logic, no I/O and no SDK, so the same code runs in three places
and they cannot drift apart:

  * `scripts/leap/tracking_quality.py`, the 60 second test (`summarise`,
    `format_report`, `verdict`);
  * `scripts/leap/camera_view.py`, the live window's "tracking:" line
    (`LiveLossTracker`, fed one tracking frame at a time);
  * `scripts/leap/record_poses.py --protocol`, which names the losses when a
    take is rejected (`take_losses`, `take_reason`, `Loss.to_dict`).

What counts as a loss. A hand that was tracked is gone for MORE than 100 ms,
or comes back (or carries on) under a different hand id. 100 ms is nine
frames at 90 Hz: a dropped frame or two is noise, a tenth of a second is a
hand the operator would see blink out of the window. The gap is measured
between the last frame with the hand and the first frame with it back, so a
100 ms hole is not a loss and a 150 ms one is. A new hand id is a loss even
with no gap at all, because the tracker only issues one when it has given up
on the old hand and found a new one (and the recorder's acquisition gate
already rejects a take for it).

Which hand. The tracker's left/right label is its opinion, not a fact (it
called the operator's left hand "right" in 20 of 21 poses on 2026-09-23). So
the live tools follow ONE hand: the id they are already following while it
is still there, else the preferred label, else the hand nearest the module's
axis. `strict=True` follows one label only, which is what the recorder needs
because its tracked fraction is counted for one label.

The causes, and why each threshold is what it is:

  device status          the device reports a warning flag (smudged lenses,
                         infrared interference, low resource, paused, or a
                         failure code) in the 2 s before or during the loss.
  frame rate low         LeapC's own framerate under 80 Hz in the 2 s before.
  bright background      with no hand in view, the IR image is bright (mean
                         above 50 of 255; an empty desk measured 5.8 on
                         2026-09-30) or more than 1 percent of it saturated:
                         sunlight or another infrared source.
  too high               palm above 45 cm. The plan's working volume ends at
                         50 cm; 45 leaves a margin.
  too low                palm below 12 cm. Below that the fingers leave the
                         stereo overlap (the 99 mm take of 2026-09-17).
  off centre             the palm more than 60 degrees off the module's axis.
                         The field is nominally 170 degrees wide, but
                         tracking is only usable well inside it, so 60
                         degrees is treated as the edge.
  closed hand from below grab strength above 0.8 with the palm turned more
                         than 40 degrees from the lens (the angle beyond
                         which `cam_hand.fusion` stops trusting camera
                         fingers): curled fingers hidden behind each other.
  palm turned away       the palm more than 60 degrees from facing the lens.
  moving fast            palm speed above 0.5 m/s over the 150 ms before.
  unexplained            none of the above.

A loss can have several causes; the first one in the order above (device and
room first) is its primary cause. The summary counts every cause of every
loss, so the fix it names first is the one that would have prevented the
most losses.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .protocol import VIEW_ANGLE_MAX_DEG, hand_view_angle_deg, row_view_angle_deg

# --- thresholds ----------------------------------------------------------------
LOSS_GAP_S = 0.100          # longer than this with no hand is a loss
_EPS = 1e-6                 # so a 100 ms hole on a float clock is not a loss
TOO_HIGH_CM = 45.0
TOO_LOW_CM = 12.0
FIELD_OF_VIEW_DEG = 170.0   # the module's nominal field, edge to edge
EDGE_DEG = 60.0             # off axis beyond this is the edge of the usable field
PALM_AWAY_DEG = 60.0
FACING_AWAY_DEG = VIEW_ANGLE_MAX_DEG      # 40: "palm facing away" for a closed hand
CLOSED_GRAB = 0.8
FAST_M_S = 0.5
LOW_FRAMERATE_HZ = 80.0
LEADUP_S = 2.0              # "the seconds before" a loss
SPEED_WINDOW_S = 0.15       # palm speed is measured over this much before
BRIGHT_MEAN = 50.0          # of 255; an empty desk measured 5.8
BRIGHT_SATURATED = 0.01     # fraction of pixels at SATURATED_LEVEL or above
SATURATED_LEVEL = 250
TARGET_CM = (25.0, 35.0)    # where the hand is asked to be
LIVE_WINDOW_S = 60.0        # "lost n times this minute"

# --- causes --------------------------------------------------------------------
DEVICE = "device status"
LOW_FPS = "frame rate low"
BRIGHT = "bright background"
TOO_HIGH = "too high"
TOO_LOW = "too low"
OFF_CENTRE = "off centre"
CLOSED_BELOW = "closed hand from below"
PALM_AWAY = "palm turned away"
FAST = "moving fast"
UNEXPLAINED = "unexplained"

# Primary-cause order: the device and the room first (they explain a loss
# whatever the hand did), then where the hand was, then its shape and turn,
# then its speed.
CAUSES: Tuple[str, ...] = (DEVICE, LOW_FPS, BRIGHT, TOO_HIGH, TOO_LOW, OFF_CENTRE,
                           CLOSED_BELOW, PALM_AWAY, FAST, UNEXPLAINED)

FIXES: Dict[str, str] = {
    TOO_HIGH: "keep the palm 25 to 35 cm above the module",
    TOO_LOW: "raise the palm to 25 to 35 cm above the module",
    OFF_CENTRE: "keep the hand over the middle of the module, not out at the side",
    CLOSED_BELOW: ("open the hand over the module first, then close it slowly "
                   "with the palm toward the lenses"),
    PALM_AWAY: "turn the palm back toward the lenses; tilt it less than about 45 degrees",
    FAST: "move slowly, well under half a metre per second",
    LOW_FPS: ("plug the camera straight into the laptop (no hub), close other "
              "programs that use the camera, keep the laptop on its charger"),
    BRIGHT: ("keep sunlight and other infrared sources (lamps, heaters, other "
             "depth cameras) off the module, and nothing shiny under the hand"),
    DEVICE: "see the device status line",
    UNEXPLAINED: ("none of the measured causes fits: wipe the lenses, check for "
                  "sunlight or another infrared source, and watch the camera "
                  "window at the moment it loses the hand"),
}

# --- device status ---------------------------------------------------------------
# eLeapDeviceStatus from LeapC.h. The low five bits are flags; the failures
# are whole values (0xE801000x) that share bits with the flags. The bindings'
# own decoding (`leap.device.DeviceStatusInfo.flags`) ANDs every enum value
# against the status, so a plain "streaming" (0x1) comes back as streaming,
# bad calibration AND bad transport. The raw value is decoded here instead.
_STATUS_BITS = (("streaming", 0x01), ("paused", 0x02), ("robust", 0x04),
                ("smudged", 0x08), ("low_resource", 0x10))
_FAILURE_BASE = 0xE8010000
_FAILURES = {0xE8010000: "unknown_failure", 0xE8010001: "bad_calibration",
             0xE8010002: "bad_firmware", 0xE8010003: "bad_transport",
             0xE8010004: "bad_control"}
_REPLUG = ("unplug the camera, wait 5 seconds, plug it straight into the laptop "
           "(no hub), then run scripts\\leap\\check_setup.py")
# flag -> (plain name, fix), most serious first: the fix of the first flag
# present is the one given.
STATUS_TEXT: Dict[str, Tuple[str, str]] = {
    "bad_transport": ("faulty USB connection", _REPLUG),
    "bad_control": ("USB control failed", _REPLUG),
    "bad_firmware": ("bad firmware", _REPLUG),
    "bad_calibration": ("bad calibration record", _REPLUG),
    "unknown_failure": ("device failure", _REPLUG),
    "low_resource": ("low resource mode",
                     "plug the camera straight into the laptop (no hub) and close "
                     "other programs that use the camera"),
    "robust": ("infrared interference, robust mode",
               "take the module out of sunlight and away from other infrared sources"),
    "smudged": ("lenses smudged", "wipe the two lenses with a soft cloth"),
    "paused": ("streaming paused", "another program paused the camera; close it"),
}


def decode_device_status(raw) -> Tuple[str, ...]:
    """LeapC's device status value -> flag names, e.g. ('streaming', 'smudged')."""
    value = int(raw) & 0xFFFFFFFF
    if value & 0xFFFF0000 == _FAILURE_BASE:
        return (_FAILURES.get(value, "unknown_failure"),)
    return tuple(name for name, bit in _STATUS_BITS if value & bit)


def status_warnings(flags: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """The flags worth reporting: everything except plain 'streaming'."""
    return tuple(f for f in (flags or ()) if f != "streaming")


def status_text(flags: Iterable[str]) -> str:
    """('smudged', 'robust') -> 'infrared interference, robust mode, lenses smudged'."""
    flags = set(flags)
    names = [STATUS_TEXT[f][0] for f in STATUS_TEXT if f in flags]
    return ", ".join(names) or "no warning flags"


def fix_for(cause: str, status: Iterable[str] = ()) -> str:
    """The fix for one cause; a device status names the flag's own fix."""
    if cause == DEVICE:
        flags = set(status)
        for flag, (_name, fix) in STATUS_TEXT.items():
            if flag in flags:
                return fix
    return FIXES.get(cause, FIXES[UNEXPLAINED])


def cause_label(cause: str, status: Iterable[str] = ()) -> str:
    """The cause as the operator reads it: 'device status (lenses smudged)'."""
    if cause == DEVICE:
        warn = status_warnings(status)
        if warn:
            return f"{DEVICE} ({status_text(warn)})"
    return cause


# --- one camera sample ------------------------------------------------------------
@dataclass(frozen=True)
class HandState:
    """One tracked hand in one frame, reduced to what the causes turn on.

    `palm` is LeapC desktop space in metres (+y up from the module).
    """

    side: str
    hand_id: int
    palm: Tuple[float, float, float]
    view_deg: Optional[float] = None       # 0 = palm square to the lens
    grab: Optional[float] = None
    pinch: Optional[float] = None
    visible_s: Optional[float] = None

    @property
    def height_cm(self) -> float:
        return float(self.palm[1]) * 100.0

    @property
    def offset_cm(self) -> float:
        """Distance from the module's vertical axis."""
        return math.hypot(float(self.palm[0]), float(self.palm[2])) * 100.0

    @property
    def off_axis_deg(self) -> float:
        """Angle between the module's axis and the ray to the palm."""
        return math.degrees(math.atan2(self.offset_cm, self.height_cm))


@dataclass
class Sample:
    """One tracking frame: every hand in it, and what the room was doing.

    `t` is seconds on one clock (LeapC's for a live run, the recording's for
    a file). `status` is None when the device status is unknown and a tuple
    of flag names (possibly empty) when it is known. `image_mean` and
    `image_saturated` are None when no IR image was available.
    """

    t: float
    hands: Tuple[HandState, ...] = ()
    framerate: Optional[float] = None
    image_mean: Optional[float] = None
    image_saturated: Optional[float] = None
    status: Optional[Tuple[str, ...]] = None
    frame_id: Optional[int] = None

    def side(self, side: str) -> Optional[HandState]:
        return next((h for h in self.hands if h.side == side), None)


def state_from_leaphand(lh) -> HandState:
    """A `leap_hand.types.LeapHand` -> HandState, with the recorder's view angle."""
    return HandState(
        side=str(lh.hand_side), hand_id=int(lh.hand_id),
        palm=tuple(float(v) for v in lh.palm_pos),
        view_deg=hand_view_angle_deg(lh),
        grab=None if lh.grab_strength is None else float(lh.grab_strength),
        pinch=None if lh.pinch_strength is None else float(lh.pinch_strength),
        visible_s=None if lh.visible_time_us is None else lh.visible_time_us / 1e6)


def image_stats(image) -> Tuple[float, float]:
    """(mean brightness 0..255, fraction saturated) of an 8-bit IR image.

    Every fourth pixel in each direction: the answer is the same to a
    fraction of a grey level and it costs a sixteenth, which is what lets
    the camera window run it without lagging.
    """
    a = image[::4, ::4]
    return float(a.mean()), float((a >= SATURATED_LEVEL).mean())


# --- following one hand -----------------------------------------------------------
def pick(hands: Sequence[HandState], prefer: Optional[str] = None,
         strict: bool = False, last_id: Optional[int] = None) -> Optional[HandState]:
    """The hand being followed in this frame, or None.

    The id already being followed while it is still there, else the
    preferred label, else the hand nearest the module's axis. `strict` drops
    every other label first.
    """
    if strict and prefer:
        hands = [h for h in hands if h.side == prefer]
    if not hands:
        return None
    if last_id is not None:
        for h in hands:
            if h.hand_id == last_id:
                return h
    if prefer:
        for h in hands:
            if h.side == prefer:
                return h
    return min(hands, key=lambda h: (h.offset_cm, h.side, h.hand_id))


# --- one loss ---------------------------------------------------------------------
@dataclass
class Loss:
    """One loss: when, how long, what the hand was doing, and why."""

    start: float                    # the last frame with the hand
    end: Optional[float]            # the first frame with it back; None = not back
    duration: float
    old_id: int
    back_id: Optional[int]
    before: HandState               # the last known state
    speed: Optional[float] = None   # palm speed just before, m/s
    min_framerate: Optional[float] = None
    image_mean: Optional[float] = None       # median over the gap
    image_saturated: Optional[float] = None
    status: Tuple[str, ...] = ()             # warning flags before or during
    causes: List[str] = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        return self.end is not None

    @property
    def new_id(self) -> bool:
        return self.back_id is not None and self.back_id != self.old_id

    @property
    def gap(self) -> bool:
        """True when the hand was actually gone (not only a new id)."""
        return self.duration > LOSS_GAP_S + _EPS or not self.recovered

    @property
    def cause(self) -> str:
        return self.causes[0] if self.causes else UNEXPLAINED

    @property
    def fix(self) -> str:
        return fix_for(self.cause, self.status)

    @property
    def kind(self) -> str:
        if not self.recovered:
            return "gone, not back by the end"
        if self.gap and self.new_id:
            return "gone, back with a new hand id"
        if self.gap:
            return "gone, back with the same id"
        return "new hand id with no gap"

    def where(self) -> str:
        """Where the hand was and what it was doing, in one plain line."""
        b = self.before
        parts = [f"hand {b.height_cm:.0f} cm up",
                 f"{b.offset_cm:.0f} cm off the axis ({b.off_axis_deg:.0f} degrees)"]
        if b.view_deg is not None:
            parts.append(f"palm {b.view_deg:.0f} degrees from the lens")
        if b.grab is not None:
            parts.append(f"grab {b.grab:.2f}")
        if self.speed is not None:
            parts.append(f"moving {self.speed:.2f} m/s")
        return ", ".join(parts)

    def to_dict(self, t0: float = 0.0) -> dict:
        """The loss as JSON: one entry of `gate.losses` in a take's meta."""
        b = self.before

        def r(v, nd=3):
            return None if v is None else round(float(v), nd)

        return {
            "start_s": r(self.start - t0), "duration_s": r(self.duration),
            "kind": self.kind, "recovered": self.recovered,
            "hand_id": self.old_id, "back_id": self.back_id,
            "hand_label": b.side,
            "height_cm": r(b.height_cm, 1), "offset_cm": r(b.offset_cm, 1),
            "off_axis_deg": r(b.off_axis_deg, 1), "view_angle_deg": r(b.view_deg, 1),
            "grab_strength": r(b.grab), "pinch_strength": r(b.pinch),
            "visible_s": r(b.visible_s), "speed_m_s": r(self.speed),
            "min_framerate_hz": r(self.min_framerate, 1),
            "image_mean": r(self.image_mean, 1),
            "image_saturated": r(self.image_saturated, 4),
            "device_status": list(self.status),
            "cause": self.cause, "causes": list(self.causes), "fix": self.fix,
        }


def causes_for(before: HandState, speed: Optional[float] = None,
               min_framerate: Optional[float] = None,
               image_mean: Optional[float] = None,
               image_saturated: Optional[float] = None,
               status: Iterable[str] = ()) -> List[str]:
    """Every cause that fits one loss, primary first; never empty."""
    found: List[str] = []
    if status_warnings(status):
        found.append(DEVICE)
    if min_framerate is not None and min_framerate < LOW_FRAMERATE_HZ:
        found.append(LOW_FPS)
    if ((image_mean is not None and image_mean > BRIGHT_MEAN)
            or (image_saturated is not None and image_saturated > BRIGHT_SATURATED)):
        found.append(BRIGHT)
    if before.height_cm > TOO_HIGH_CM:
        found.append(TOO_HIGH)
    elif before.height_cm < TOO_LOW_CM:
        found.append(TOO_LOW)
    if before.off_axis_deg > EDGE_DEG:
        found.append(OFF_CENTRE)
    facing = before.view_deg
    if (before.grab is not None and before.grab > CLOSED_GRAB
            and facing is not None and facing > FACING_AWAY_DEG):
        found.append(CLOSED_BELOW)
    if facing is not None and facing > PALM_AWAY_DEG:
        found.append(PALM_AWAY)
    if speed is not None and speed > FAST_M_S:
        found.append(FAST)
    return found or [UNEXPLAINED]


def _median(values: Sequence[float]) -> Optional[float]:
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _speed(track: Sequence[Tuple[float, HandState]]) -> Optional[float]:
    """Palm speed over `track` (time order, ending on the last known frame)."""
    if len(track) < 2:
        return None
    (t0, a), (t1, b) = track[0], track[-1]
    if t1 - t0 <= 0:
        return None
    return math.dist(a.palm, b.palm) / (t1 - t0)


def build_loss(before_t: float, before: HandState, end_t: Optional[float],
               back: Optional[HandState], lead: Sequence[Sample],
               gap: Sequence[Sample], track: Sequence[Tuple[float, HandState]],
               now: Optional[float] = None) -> Loss:
    """One Loss from the frames around it, classified.

    `lead` are the frames of the `LEADUP_S` before the last frame with the
    hand, `gap` the frames with it missing, `track` the followed hand's own
    frames of the `SPEED_WINDOW_S` before. `now` closes a loss that is still
    open (live) or never recovered (the end of a run).
    """
    stop = end_t if end_t is not None else (now if now is not None else before_t)
    rates = [s.framerate for s in lead if s.framerate and s.framerate > 0]
    flags: List[str] = []
    for s in list(lead) + list(gap):
        for f in status_warnings(s.status):
            if f not in flags:
                flags.append(f)
    means = [s.image_mean for s in gap if s.image_mean is not None]
    sats = [s.image_saturated for s in gap if s.image_saturated is not None]
    loss = Loss(start=before_t, end=end_t, duration=max(0.0, stop - before_t),
                old_id=before.hand_id, back_id=None if back is None else back.hand_id,
                before=before, speed=_speed(track),
                min_framerate=min(rates) if rates else None,
                image_mean=_median(means), image_saturated=_median(sats),
                status=tuple(flags))
    loss.causes = causes_for(before, loss.speed, loss.min_framerate,
                             loss.image_mean, loss.image_saturated, loss.status)
    return loss


def find_losses(samples: Sequence[Sample], prefer: Optional[str] = None,
                strict: bool = False, t_end: Optional[float] = None,
                gap_s: float = LOSS_GAP_S) -> List[Loss]:
    """Every loss of the followed hand in a series, classified, in time order.

    `t_end` is when the series ended, on the same clock: a hand gone from
    its last frame until then for longer than `gap_s` is a loss that was
    never recovered. Nothing before the first frame with the hand is a loss:
    a hand that was never tracked was not lost.
    """
    samples = sorted(samples, key=lambda s: s.t)
    track: List[Tuple[int, HandState]] = []
    last_id = None
    for k, s in enumerate(samples):
        h = pick(s.hands, prefer, strict, last_id)
        if h is not None:
            track.append((k, h))
            last_id = h.hand_id

    def context(pos: int):
        ka, a = track[pos]
        ta = samples[ka].t
        lo = ka
        while lo > 0 and samples[lo - 1].t >= ta - LEADUP_S:
            lo -= 1
        j = pos
        while j > 0 and samples[track[j - 1][0]].t >= ta - SPEED_WINDOW_S:
            j -= 1
        points = [(samples[k].t, h) for k, h in track[j:pos + 1]]
        return ta, a, samples[lo:ka + 1], points

    losses: List[Loss] = []
    for pos in range(len(track) - 1):
        ka, a = track[pos]
        kb, b = track[pos + 1]
        dt = samples[kb].t - samples[ka].t
        if dt > gap_s + _EPS or a.hand_id != b.hand_id:
            ta, _a, lead, points = context(pos)
            losses.append(build_loss(ta, a, samples[kb].t, b, lead,
                                     samples[ka + 1:kb], points))
    if track and t_end is not None:
        ka, a = track[-1]
        if t_end - samples[ka].t > gap_s + _EPS:
            ta, _a, lead, points = context(len(track) - 1)
            losses.append(build_loss(ta, a, None, None, lead, samples[ka + 1:],
                                     points, now=t_end))
    return losses


def rank_causes(losses: Sequence[Loss]) -> List[Tuple[str, int, float]]:
    """(cause, losses it fits, seconds lost), most losses first.

    Every cause of every loss is counted, so the first row is the single
    fix that would have prevented the most losses. Ties go to the order of
    `CAUSES` (device and room first).
    """
    count: Dict[str, int] = {}
    seconds: Dict[str, float] = {}
    for loss in losses:
        for c in loss.causes:
            count[c] = count.get(c, 0) + 1
            seconds[c] = seconds.get(c, 0.0) + loss.duration
    order = {c: i for i, c in enumerate(CAUSES)}
    return sorted(((c, count[c], seconds[c]) for c in count),
                  key=lambda row: (-row[1], order.get(row[0], len(CAUSES))))


def _status_of(losses: Sequence[Loss], cause: str) -> Tuple[str, ...]:
    flags: List[str] = []
    for loss in losses:
        if cause in loss.causes:
            for f in loss.status:
                if f not in flags:
                    flags.append(f)
    return tuple(flags)


# --- a whole run ------------------------------------------------------------------
@dataclass
class Report:
    """Everything the 60 second test prints and writes."""

    seconds: float
    frames: int
    tracked_frames: int
    first_seen: Optional[float]             # seconds from the start
    frames_after_first: int
    tracked_after_first: int
    event_rate_hz: Optional[float]
    framerate_median: Optional[float]
    framerate_min: Optional[float]
    losses: List[Loss]
    labels: Dict[str, int]
    image: Optional[dict]                   # None: no IR image in the series
    status_seen: Optional[Tuple[str, ...]]  # None: device status never known
    t0: float = 0.0
    gap_s: float = LOSS_GAP_S               # the hole that counted as a loss

    @property
    def tracked_fraction(self) -> float:
        return self.tracked_frames / self.frames if self.frames else 0.0

    @property
    def tracked_fraction_after_first(self) -> Optional[float]:
        if not self.frames_after_first:
            return None
        return self.tracked_after_first / self.frames_after_first

    @property
    def ranked(self) -> List[Tuple[str, int, float]]:
        return rank_causes(self.losses)


def summarise(samples: Sequence[Sample], prefer: Optional[str] = None,
              strict: bool = False, t_start: Optional[float] = None,
              t_end: Optional[float] = None, gap_s: float = LOSS_GAP_S) -> Report:
    """The whole report for one series. `t_start`/`t_end` default to its ends."""
    samples = sorted(samples, key=lambda s: s.t)
    if not samples:
        return Report(0.0, 0, 0, None, 0, 0, None, None, None, [], {}, None, None,
                      t0=t_start or 0.0, gap_s=gap_s)
    t0 = samples[0].t if t_start is None else t_start
    t1 = samples[-1].t if t_end is None else t_end
    tracked = 0
    first = None
    after = after_tracked = 0
    labels: Dict[str, int] = {}
    last_id = None
    with_hand: List[float] = []
    no_hand: List[float] = []
    sat_no_hand: List[float] = []
    status_known = False
    status_seen: List[str] = []
    for s in samples:
        h = pick(s.hands, prefer, strict, last_id)
        if h is not None:
            tracked += 1
            last_id = h.hand_id
            labels[h.side] = labels.get(h.side, 0) + 1
            if first is None:
                first = s.t
        if first is not None:
            after += 1
            after_tracked += h is not None
        if s.image_mean is not None:
            (with_hand if s.hands else no_hand).append(s.image_mean)
            if not s.hands and s.image_saturated is not None:
                sat_no_hand.append(s.image_saturated)
        if s.status is not None:
            status_known = True
            for f in s.status:
                if f not in status_seen:
                    status_seen.append(f)
    span = samples[-1].t - samples[0].t
    rates = [s.framerate for s in samples if s.framerate and s.framerate > 0]
    image = None
    if with_hand or no_hand:
        image = {"frames": len(with_hand) + len(no_hand),
                 "mean_no_hand": _median(no_hand), "mean_with_hand": _median(with_hand),
                 "saturated_no_hand": _median(sat_no_hand)}
    return Report(
        seconds=max(0.0, t1 - t0), frames=len(samples), tracked_frames=tracked,
        first_seen=None if first is None else first - t0,
        frames_after_first=after, tracked_after_first=after_tracked,
        event_rate_hz=(len(samples) - 1) / span if span > 0 else None,
        framerate_median=_median(rates), framerate_min=min(rates) if rates else None,
        losses=find_losses(samples, prefer, strict, t_end=t1, gap_s=gap_s),
        labels=labels, image=image,
        status_seen=tuple(status_seen) if status_known else None, t0=t0, gap_s=gap_s)


def cadence_gap_s(samples: Sequence[Sample], prefer: Optional[str] = None,
                  strict: bool = False) -> float:
    """The shortest hole a series can call a loss: 100 ms, or more for a slow file.

    A take saved at 5 frames a second (the plain pose recorder's default)
    has a 200 ms hole between EVERY pair of lines, and calling each one a
    loss would be nonsense. When the followed hand's typical interval is over
    half of `LOSS_GAP_S`, the threshold becomes 2.5 of those intervals.
    """
    times = [s.t for s in samples if pick(s.hands, prefer, strict) is not None]
    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    if not gaps:
        return LOSS_GAP_S
    typical = gaps[len(gaps) // 2]
    return max(LOSS_GAP_S, 2.5 * typical) if typical > LOSS_GAP_S / 2 else LOSS_GAP_S


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 100:.1f} %"


def _times(n: int) -> str:
    return f"{n} time" if n == 1 else f"{n} times"


def verdict(report: Report) -> str:
    """The one line a script ends on. Always starts with 'VERDICT:'."""
    if not report.tracked_frames:
        return ("VERDICT: no hand was tracked, so there is nothing to judge: hold "
                "the hand 25 to 35 cm over the middle of the module and run it again")
    if not report.losses:
        return (f"VERDICT: no losses (tracked {_pct(report.tracked_fraction_after_first)} "
                "of the frames after the hand was first seen)")
    cause, count, _secs = report.ranked[0]
    status = _status_of(report.losses, cause)
    return (f"VERDICT: {cause_label(cause, status)} ({count} of "
            f"{len(report.losses)} losses): {fix_for(cause, status)}")


def format_report(report: Report, title: str = "Tracking quality",
                  setup: str = "", notes: Sequence[str] = ()) -> List[str]:
    """The report as lines, ending on the VERDICT line.

    `notes` are what the caller knows about the run that the series cannot
    say (mock, image access refused, stopped early); they are printed as
    given, never guessed at here.
    """
    r = report
    lines = [title]
    if setup:
        lines.append(f"Setup: {setup}")
    lines.append(f"Run: {r.seconds:.1f} s, {r.frames} tracking frames"
                 + (f", {r.event_rate_hz:.1f} frames per second" if r.event_rate_hz else ""))
    if r.framerate_median is not None:
        lines.append(f"Frame rate reported by the service: median {r.framerate_median:.1f} Hz, "
                     f"lowest {r.framerate_min:.1f} Hz")
    if r.first_seen is None:
        lines.append("Tracked: no hand was tracked at all")
    else:
        lines.append(f"Tracked: {_pct(r.tracked_fraction)} of the frames; "
                     f"{_pct(r.tracked_fraction_after_first)} after the hand was "
                     f"first seen at {r.first_seen:.1f} s")
    if r.labels:
        total = sum(r.labels.values())
        lines.append("Tracker's label for the hand: " + ", ".join(
            f"{side} {n / total * 100:.0f} %" for side, n in sorted(r.labels.items())))
    for note in notes:
        lines.append(note)
    if r.image is None:
        lines.append("Image brightness: not measured in this run (see the notes above)")
    else:
        im = r.image

        def num(v, scale=1.0, nd=1):
            return "-" if v is None else f"{v * scale:.{nd}f}"

        lines.append(f"Image brightness: {num(im['mean_no_hand'])} of 255 with no hand "
                     f"in view, {num(im['mean_with_hand'])} with a hand; "
                     f"{num(im['saturated_no_hand'], 100.0, 2)} % saturated with no hand "
                     f"(bright means above {BRIGHT_MEAN:.0f}, or above "
                     f"{BRIGHT_SATURATED * 100:.0f} % saturated)")
    if r.status_seen is None:
        lines.append("Device status: not reported to this program (see the notes above)")
    else:
        lines.append(f"Device status: {status_text(status_warnings(r.status_seen))}"
                     f" (flags seen: {', '.join(r.status_seen) or 'none'})")
    lines.append("")
    lines.append(f"Losses (gone longer than {r.gap_s * 1000:.0f} ms, or a new hand id): "
                 f"{len(r.losses)}")
    for k, loss in enumerate(r.losses, 1):
        lines.append(f"  {k:>2}. at {loss.start - r.t0:5.1f} s, {loss.kind}, "
                     f"{loss.duration:.2f} s")
        lines.append(f"      where: {loss.where()}")
        lines.append("      why:   " + ", ".join(cause_label(c, loss.status)
                                                for c in loss.causes))
    if r.losses:
        lines.append("")
        lines.append("Causes, most losses first (a loss can have more than one):")
        for cause, count, secs in r.ranked:
            status = _status_of(r.losses, cause)
            lines.append(f"  {cause_label(cause, status):<24} {count:>3} "
                         f"{'loss  ' if count == 1 else 'losses'} {secs:5.1f} s   "
                         f"fix: {fix_for(cause, status)}")
        cause, count, _ = r.ranked[0]
        status = _status_of(r.losses, cause)
        lines.append("")
        lines.append(f"Fix to try first: {fix_for(cause, status)} "
                     f"({cause_label(cause, status)}, {count} of {len(r.losses)} losses)")
    lines.append("")
    lines.append(verdict(r))
    return lines


# --- the live line ----------------------------------------------------------------
class LiveLossTracker:
    """The same classifier, fed one tracking frame at a time.

    Cheap on purpose: the camera window feeds it from the SDK's callback at
    90 Hz, so a frame costs a pick and a few deque operations, and the
    lead-up is only looked at on the rare frame that starts a loss. A loss
    is counted the moment the hand has been gone for `gap_s` (the operator
    should see it while it is happening), classified from what came before
    it, and not counted again when the hand comes back.
    """

    def __init__(self, prefer: Optional[str] = None,
                 window_s: float = LIVE_WINDOW_S, gap_s: float = LOSS_GAP_S):
        self.prefer = prefer if prefer in ("left", "right") else None
        self.window_s = float(window_s)
        self.gap_s = float(gap_s)
        self._recent: deque = deque()          # Samples, the last LEADUP_S + 1 s
        self._track: deque = deque()           # (t, HandState) of the followed hand
        self._losses: deque = deque()          # Loss, the last window_s
        self._last_t: Optional[float] = None
        self._last_h: Optional[HandState] = None
        self._open = False
        self.last: Optional[Loss] = None
        self.frames = 0
        self.tracked = 0
        self.total_losses = 0

    def add(self, sample: Sample) -> Optional[Loss]:
        """Feed one frame; returns the loss it started, if it started one."""
        t = sample.t
        self._recent.append(sample)
        while self._recent and self._recent[0].t < t - (LEADUP_S + 1.0):
            self._recent.popleft()
        self.frames += 1
        last_id = None if self._last_h is None else self._last_h.hand_id
        h = pick(sample.hands, self.prefer, False, last_id)
        new = None
        if h is None:
            if (self._last_h is not None and not self._open
                    and t - self._last_t > self.gap_s + _EPS):
                new = self._loss(None, None, t)
                self._open = True
        else:
            self.tracked += 1
            if self._last_h is not None:
                if self._open:
                    self._open = False          # counted when it opened
                elif t - self._last_t > self.gap_s + _EPS or h.hand_id != last_id:
                    new = self._loss(t, h, t)
            self._track.append((t, h))
            while self._track and self._track[0][0] < t - SPEED_WINDOW_S - 0.05:
                self._track.popleft()
            self._last_t, self._last_h = t, h
        if new is not None:
            self._losses.append(new)
            self.last = new
            self.total_losses += 1
        while self._losses and self._losses[0].start < t - self.window_s:
            self._losses.popleft()
        return new

    def _loss(self, end_t: Optional[float], back: Optional[HandState],
              now: float) -> Loss:
        ta = self._last_t
        lead = [s for s in self._recent if ta - LEADUP_S <= s.t <= ta]
        gap = [s for s in self._recent if ta < s.t < now or (end_t is None and s.t == now)]
        track = [(t, h) for t, h in self._track if t >= ta - SPEED_WINDOW_S]
        return build_loss(ta, self._last_h, end_t, back, lead, gap, track, now=now)

    @property
    def losing(self) -> bool:
        """True while a counted loss is still open (the hand is gone)."""
        return self._open

    def lost_in_window(self) -> int:
        return len(self._losses)

    @property
    def tracked_fraction(self) -> Optional[float]:
        return self.tracked / self.frames if self.frames else None

    def fps(self) -> Optional[float]:
        """Frames per second over the last second of the series clock."""
        if len(self._recent) < 2:
            return None
        newest = self._recent[-1].t
        n = 0
        oldest = newest
        for s in reversed(self._recent):
            if s.t < newest - 1.0:
                break
            n += 1
            oldest = s.t
        span = newest - oldest
        return (n - 1) / span if span > 0 else None

    def line(self) -> str:
        """`tracking: 90 Hz, lost 2 times this minute, last: too high`."""
        fps = self.fps()
        rate = "--" if fps is None else f"{fps:.0f}"
        last = "none" if self.last is None else cause_label(self.last.cause, self.last.status)
        return (f"tracking: {rate} Hz, lost {_times(self.lost_in_window())} this "
                f"minute, last: {last}")


# --- recorded takes (record_poses --protocol) -------------------------------------
def row_clock(rows: Sequence[dict]) -> str:
    """The best clock a take's lines share: capture, then LeapC, then write."""
    for key in ("capture_time", "timestamp"):
        if rows and all(r.get(key) is not None for r in rows):
            return key
    return "wall_time"


def rows_to_samples(rows: Sequence[dict], clock: Optional[str] = None) -> List[Sample]:
    """A recording's JSONL lines -> one Sample per tracking frame.

    Lines of one frame (both hands) share a `frame_id`. A recording only has
    lines for frames with a hand in them, so the series has no empty frames,
    no image and no device status: gaps are still gaps, and the causes that
    need the room are simply not judged.
    """
    clock = clock or row_clock(rows)
    frames: Dict[object, List[dict]] = {}
    for r in rows:
        if r.get("status", 1) == 0 or not r.get("palm_abs"):
            continue
        key = r.get("frame_id")
        if key is None:
            key = ("t", round(float(r[clock]), 6))
        frames.setdefault(key, []).append(r)
    out: List[Sample] = []
    for group in frames.values():
        hands = []
        for r in group:
            vis = r.get("visible_time_us")
            hands.append(HandState(
                side=str(r.get("hand_side")), hand_id=int(r.get("hand_id") or 0),
                palm=tuple(float(v) for v in r["palm_abs"]),
                view_deg=row_view_angle_deg(r),
                grab=None if r.get("grab_strength") is None else float(r["grab_strength"]),
                pinch=None if r.get("pinch_strength") is None else float(r["pinch_strength"]),
                visible_s=None if vis is None else float(vis) / 1e6))
        rates = [float(r["framerate"]) for r in group if r.get("framerate")]
        out.append(Sample(t=min(float(r[clock]) for r in group), hands=tuple(hands),
                          framerate=max(rates) if rates else None,
                          frame_id=group[0].get("frame_id")))
    out.sort(key=lambda s: s.t)
    return out


@dataclass
class TakeLosses:
    """The losses of one recorded take, on the take's own clock."""

    losses: List[Loss]
    t0: float                      # take start on the loss clock
    head_s: Optional[float]        # seconds before the hand was first tracked
    clock: str
    gap_s: float = LOSS_GAP_S      # see cadence_gap_s

    def to_list(self) -> List[dict]:
        return [loss.to_dict(self.t0) for loss in self.losses]


def take_losses(rows: Sequence[dict], hand: Optional[str],
                t_start: Optional[float] = None,
                t_stop: Optional[float] = None) -> TakeLosses:
    """Losses of one tracker label in a recorded take.

    `t_start`/`t_stop` are the take's window on the WALL clock (the
    recorder's). The lines are compared on their capture clock, so the
    window is moved onto it by the median difference between each line's
    write time and its capture time: the writer's delay, a few milliseconds
    live, and whatever offset a mock's synthetic clock carries.
    """
    rows = [r for r in rows if hand is None or str(r.get("hand_side")) == hand]
    if not rows:
        return TakeLosses([], t_start or 0.0, None, "wall_time")
    clock = row_clock(rows)
    offset = 0.0
    if clock != "wall_time" and all(r.get("wall_time") is not None for r in rows):
        offset = _median([float(r["wall_time"]) - float(r[clock]) for r in rows]) or 0.0
    samples = rows_to_samples(rows, clock)
    if not samples:
        return TakeLosses([], t_start or 0.0, None, clock)
    t0 = samples[0].t if t_start is None else float(t_start) - offset
    t1 = None if t_stop is None else float(t_stop) - offset
    gap_s = cadence_gap_s(samples, hand, hand is not None)
    losses = find_losses(samples, prefer=hand, strict=hand is not None, t_end=t1,
                         gap_s=gap_s)
    return TakeLosses(losses, t0, samples[0].t - t0, clock, gap_s)


def loss_sentence(losses: Sequence[Loss]) -> str:
    """How many losses, and the biggest one: where the hand was and why.

    `lost 3 times, longest 1.4 s with the hand at 49 cm (too high: keep the
    palm 25 to 35 cm above the module)`.
    """
    if not losses:
        return "no loss inside the take"
    n = len(losses)
    worst = max(losses, key=lambda loss: loss.duration)
    why = f"({cause_label(worst.cause, worst.status)}: {worst.fix})"
    at = f"with the hand at {worst.before.height_cm:.0f} cm"
    if not any(loss.gap for loss in losses):
        return f"the tracker gave the hand a new id {_times(n)} {at} {why}"
    size = f"longest {worst.duration:.1f} s" if n > 1 else f"for {worst.duration:.1f} s"
    if not worst.recovered:
        size += " (not back by the end)"
    return f"lost {_times(n)}, {size} {at} {why}"


def take_reason(gate_reason: str, tracked_fraction: float, min_tracked: float,
                interval_reacquisitions: Sequence[Tuple[float, int, int]],
                has_summary_frame: bool, losses: Sequence[Loss],
                head_s: Optional[float] = None) -> str:
    """The acquisition gate's reason with the losses named, or "" if it passed.

    Built from the same facts `static_interval.summarise_take` decided on,
    in the same order, so the text never disagrees with the decision.
    """
    if not gate_reason:
        return ""
    if tracked_fraction <= 0 and not losses:
        return gate_reason                      # no hand at all: nothing to name
    text = loss_sentence(losses)
    if head_s is not None and head_s > 2.5 * LOSS_GAP_S:
        text += f"; first tracked {head_s:.1f} s after the start"
    if tracked_fraction < min_tracked:
        return (f"tracked {tracked_fraction * 100:.0f} percent: {text}; the gate "
                f"needs {min_tracked * 100:.0f} percent")
    if interval_reacquisitions:
        _t, old, new = interval_reacquisitions[0]
        return f"re-acquired inside the static interval (id {old} -> {new}): {text}"
    if not has_summary_frame:
        return f"no tracked frame inside the static interval: {text}"
    return f"{gate_reason}: {text}"


# --- the scripted mock ------------------------------------------------------------
class MockScenario:
    """What `--mock` acts out: one slot per cause, each ending in a dropout.

    The slots follow the coached minute (hold, up, out, palm away, fist,
    fast, down, then the room and the device) so the report has every kind
    of loss to explain. A slot is at least `MIN_SLOT_S` long, so a short run
    acts out the first few slots only; "too high" comes twice so the ranking
    has a clear winner. The hold slot has a 60 ms blip that must NOT count.

    `script(i)` drives `MockLeapStream` (position, turn, curl, dropouts, id);
    `environment(t)` is the room (frame rate, IR brightness, device status)
    the mock source attaches to every frame.
    """

    SLOTS = ("hold", "too_high", "off_centre", "too_high", "palm_away", "closed",
             "fast", "too_low", "low_fps", "bright", "device", "id_swap")
    MIN_SLOT_S = 0.6
    NEUTRAL_MM = (0.0, 300.0, 40.0)            # wrist: palm 30 cm up, on the axis

    def __init__(self, seconds: float, hz: float = 90.0):
        self.seconds = float(seconds)
        self.hz = float(hz)
        n = max(1, min(len(self.SLOTS), int(self.seconds // self.MIN_SLOT_S)))
        self.slots = self.SLOTS[:n]
        self.slot_s = self.seconds / n
        self.drop_s = min(1.0, max(0.15, 0.25 * self.slot_s))
        # The hand comes back with a new id after the first "too high" drop
        # (a real re-acquisition) and changes id with no gap halfway through
        # the id_swap slot.
        self._id_times: List[float] = []
        if "too_high" in self.slots:
            k = self.slots.index("too_high")
            self._id_times.append(k * self.slot_s + 0.6 * self.slot_s + self.drop_s)
        if "id_swap" in self.slots:
            k = self.slots.index("id_swap")
            self._id_times.append(k * self.slot_s + 0.5 * self.slot_s)

    def _slot(self, t: float) -> Tuple[str, float]:
        k = min(len(self.slots) - 1, max(0, int(t // self.slot_s)))
        return self.slots[k], t - k * self.slot_s

    def at(self, t: float) -> dict:
        """The hand at time t (seconds from the start), as `MockLeapStream` takes it."""
        name, u = self._slot(t)
        span = self.slot_s
        x, y, z = self.NEUTRAL_MM
        roll = 0.0
        curl = 0.0
        drop_at = 0.6 * span
        back_at = drop_at + self.drop_s
        # 0 .. 1 .. 0 over the slot: out by 0.45 of it, held to the drop, back after
        if u < 0.45 * span:
            w = u / (0.45 * span)
            w = w * w * (3.0 - 2.0 * w)
        elif u < back_at:
            w = 1.0
        else:
            w = 1.0 - min(1.0, (u - back_at) / max(1e-9, span - back_at))
        drop = drop_at <= u < back_at
        if name == "hold":
            drop = 0.5 * span <= u < 0.5 * span + 0.06      # a blip, not a loss
        elif name == "too_high":
            y += 190.0 * w                                  # up to 49 cm
        elif name == "too_low":
            y -= 200.0 * w                                  # down to 10 cm
        elif name == "off_centre":
            x += 560.0 * w                                  # 62 degrees off the axis
            roll = -30.0 * w                                # palm still toward the lens
        elif name == "palm_away":
            roll = 80.0 * w
        elif name == "closed":
            roll, curl = 50.0 * w, w
        elif name == "fast":
            # still, then 12 cm in the last 0.12 s before the drop: 1 m/s
            if u < back_at:
                x += 120.0 * min(1.0, max(0.0, (u - (drop_at - 0.12)) / 0.12))
        elif name == "id_swap":
            drop = False
        out = {"origin_mm": (x, y, z), "roll_deg": roll, "curl": curl,
               "id_offset": sum(1 for s in self._id_times if t >= s),
               "framerate": self.environment(t)["framerate"]}
        if drop:
            out["drop"] = True
        return out

    def script(self, i: int) -> dict:
        return self.at(i / self.hz)

    def environment(self, t: float) -> dict:
        """The room at time t: frame rate, IR brightness, device status."""
        name, _u = self._slot(t)
        return {
            "framerate": 60.0 if name == "low_fps" else 90.0,
            "image_mean": 120.0 if name == "bright" else 6.0,        # no hand in view
            "image_mean_hand": 120.0 if name == "bright" else 22.0,  # with a hand
            "image_saturated": 0.05 if name == "bright" else 0.0,
            "status": ("streaming", "smudged") if name == "device" else ("streaming",),
        }


# --- the per-frame file -----------------------------------------------------------
_SIDE_FIELDS = ("id", "x_mm", "y_mm", "z_mm", "height_cm", "offset_cm", "off_axis_deg",
                "view_deg", "grab", "pinch", "visible_s")
SERIES_COLUMNS: Tuple[str, ...] = (
    ("t_s", "frame_id", "framerate_hz", "hands", "tracked", "followed_label", "followed_id")
    + tuple(f"{side}_{f}" for side in ("left", "right") for f in _SIDE_FIELDS)
    + ("image_mean", "image_saturated", "device_status"))


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def series_rows(samples: Sequence[Sample], prefer: Optional[str] = None,
                t0: Optional[float] = None) -> List[dict]:
    """The series as CSV rows (`SERIES_COLUMNS`), one per frame."""
    samples = sorted(samples, key=lambda s: s.t)
    if not samples:
        return []
    t0 = samples[0].t if t0 is None else t0
    rows = []
    last_id = None
    for s in samples:
        h = pick(s.hands, prefer, False, last_id)
        if h is not None:
            last_id = h.hand_id
        row = {"t_s": _fmt(s.t - t0, 4), "frame_id": _fmt(s.frame_id),
               "framerate_hz": _fmt(s.framerate, 2), "hands": str(len(s.hands)),
               "tracked": "1" if h is not None else "0",
               "followed_label": "" if h is None else h.side,
               "followed_id": "" if h is None else str(h.hand_id),
               "image_mean": _fmt(s.image_mean, 2),
               "image_saturated": _fmt(s.image_saturated, 5),
               "device_status": "" if s.status is None else ("|".join(s.status) or "none")}
        for side in ("left", "right"):
            hs = s.side(side)
            vals = {} if hs is None else {
                "id": str(hs.hand_id), "x_mm": _fmt(hs.palm[0] * 1000, 1),
                "y_mm": _fmt(hs.palm[1] * 1000, 1), "z_mm": _fmt(hs.palm[2] * 1000, 1),
                "height_cm": _fmt(hs.height_cm, 2), "offset_cm": _fmt(hs.offset_cm, 2),
                "off_axis_deg": _fmt(hs.off_axis_deg, 1), "view_deg": _fmt(hs.view_deg, 1),
                "grab": _fmt(hs.grab), "pinch": _fmt(hs.pinch),
                "visible_s": _fmt(hs.visible_s)}
            for f in _SIDE_FIELDS:
                row[f"{side}_{f}"] = vals.get(f, "")
        rows.append(row)
    return rows


def samples_from_series_rows(rows: Iterable[dict]) -> List[Sample]:
    """`series_rows` backwards, to re-analyse a saved CSV after the fact."""
    def num(v):
        return None if v in (None, "") else float(v)

    out = []
    for row in rows:
        hands = []
        for side in ("left", "right"):
            if row.get(f"{side}_id") in (None, ""):
                continue
            hands.append(HandState(
                side=side, hand_id=int(row[f"{side}_id"]),
                palm=(num(row[f"{side}_x_mm"]) / 1000, num(row[f"{side}_y_mm"]) / 1000,
                      num(row[f"{side}_z_mm"]) / 1000),
                view_deg=num(row.get(f"{side}_view_deg")), grab=num(row.get(f"{side}_grab")),
                pinch=num(row.get(f"{side}_pinch")),
                visible_s=num(row.get(f"{side}_visible_s"))))
        status = row.get("device_status", "")
        out.append(Sample(
            t=float(row["t_s"]), hands=tuple(hands), framerate=num(row.get("framerate_hz")),
            image_mean=num(row.get("image_mean")),
            image_saturated=num(row.get("image_saturated")),
            status=None if status in (None, "") else (
                () if status == "none" else tuple(status.split("|"))),
            frame_id=None if row.get("frame_id") in (None, "") else int(row["frame_id"])))
    return out
