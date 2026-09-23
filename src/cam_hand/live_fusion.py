"""Live glove + camera fusion: the offline pipeline, one glove frame at a time.

`scripts/fuse_poses.py` fuses a session after it has been recorded. This
module does the same fusion while the operator is wearing the glove, and it
does it with the SAME pieces: `RailOverrideTracker`, `DriftAnchor`,
`learn_rails`, `learn_flexion_scale`, `curl_gates_from_rails`, the template
fit and `fuse_skeletons` are all imported from where the offline report gets
them, and `LiveFusion.step` runs them in the order `fuse_all` does. A frame
fused here and the same frame fused from a recording go through one code
path, so a number the report prints is a number this module would produce.

What is different live, and why

  Pairing       offline, `pair_by_time` sees the whole take and picks each
                glove frame's nearest camera frame. Live there is no future
                to look into, so the camera's frames go into a `CameraBuffer`
                (three seconds per hand) and each glove frame, as it arrives,
                is paired with the buffered camera frame nearest to its own
                stamp minus that hand's glove lag. The glove trails the
                camera (about 0.10 s on the left glove and 0.47 s on the
                right, on this laptop), so the camera frame it describes has
                always arrived already: the lag costs no waiting.

  hand_id       offline, `flag_hand_id_stability` looks back over the take.
                Live, `CameraBuffer.add` keeps each hand's last id and when
                it changed, and answers the same question at the moment the
                frame arrives: stable once `hand_id_settle_s` has passed
                since the last change.

  Learning      the offline report learns the rails, the flexion endpoints
                and the operator's bone lengths from the whole session. Live
                they come from a coached WARM-UP (`Warmup`), one hand at a
                time: an ACQUIRE gate that waits until that hand's glove is
                streaming and the camera has held that hand, still and at
                the right height, for half a second; then an open palm held
                flat to the camera; then a fist. The open palm gives the
                glove's rails, the camera's open reference and the frames a
                bone-length measurement is taken over; the fist gives both
                sensors' flexed ends. Nothing is learned after that, so the
                fusion does not wander as the session goes on; the drift
                anchor (experimental, off unless asked for) is the one part
                that keeps learning, which is its job. A warm-up is an
                initial calibration, not an equivalent of learning over a
                session: it is a far smaller sample, taken once.

`replay_take` is the check that the per-frame path really is `fuse_all`'s:
it runs a recorded take through `LiveFusion.step`, handed what `fuse_all`
learned from the session (`learn_from_session`), and `scripts/fuse_live.py
--replay` compares the result with `fuse_all` frame by frame.

The sensors are wrapped (`GloveSource`, `CameraSource`) so that the real
hardware and the mocks produce the same per-frame dicts, and the outputs
(`JsonlSink`, `OscSink`, `hud_line`) read one `FusedFrame` per glove frame.
`scripts/fuse_live.py` is the command that puts them together.
"""
import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Callable, Deque, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np

from cam_hand.features import flexion_features, palm_normal
from cam_hand.fusion import (
    DEFAULT_GATES,
    DEFAULT_RAIL,
    FINGER_NAMES,
    GATED_DOFS,
    MIN_OPEN_REF_FRAMES,
    SENSOR_CAMERA,
    SENSOR_GLOVE,
    SPREAD_FINGERS,
    DriftAnchor,
    DriftAnchorParams,
    FlexionScale,
    GateParams,
    RailOverrideParams,
    RailOverrideTracker,
    curl_gates_from_rails,
    fuse_skeletons,
    WALL_CLOCK,
    learn_flexion_scale,
    learn_rails,
    pairing_clock,
)
from cam_hand.template_fit import (
    HandMeasurement,
    fit_refusal,
    fit_template,
    measure_hand,
)
from leap_hand.protocol import (
    ACQUIRE_VISIBLE_TIME_US,
    DEFAULT_BAND,
    NOT_FACING,
    OFF_AXIS,
    TOO_HIGH,
    TOO_LOW,
    TOO_YOUNG,
    HandReading,
    acquire_failures,
    read_hand,
)
from xr_hand.keypoints21 import MP21_TO_OPENXR_IDX, frame_to_keypoints21

# How long the camera buffer remembers each hand. The largest glove lag
# measured on this laptop is under half a second, so three seconds holds
# every camera frame a glove frame could want with room to spare, and is
# still only ~270 frames per hand at 90 Hz.
KEEP_S = 3.0

# A hand the camera has not reported for this long is STALE on the HUD. The
# same value `leap_hand.live` and `scripts/record_simultaneous.py` use.
STALE_S = 0.30
# Seconds a hand's camera must have been stale before the drift anchor
# forgets what it learned (see `LiveFusion._forget_after_camera_loss`).
ANCHOR_RESET_S = 1.0

# The Leap reports no per-frame confidence (LeapC's `confidence` is a
# constant 1.0), so a frame it reported is scored 1.0, exactly as the offline
# loader scores it; `MIN_SCORE` is `fuse_poses.py`'s default `--min-score`.
CAM_SCORE = 1.0
MIN_SCORE = 0.5

# --- --fit-template -----------------------------------------------------
FIT_AUTO = "auto"
FIT_NONE = "none"

# --- the warm-up ----------------------------------------------------------
# Each hand's warm-up, in order: the ACQUIRE gate, a short countdown, then the
# two phases the learning reads. STAGE_NOT_ACQUIRED is only ever announced,
# when a hand's gate timed out.
STAGE_ACQUIRE = "ACQUIRE"
STAGE_COUNTDOWN = "COUNTDOWN"
STAGE_NOT_ACQUIRED = "NOT ACQUIRED"
PHASE_OPEN = "OPEN PALM flat to the camera"
PHASE_FIST = "FIST"
WARMUP_OPEN_S = 4.0
WARMUP_FIST_S = 4.0
# The ACQUIRE gate: how long it waits for a hand, and how many glove packets
# of that hand the last second must hold (the glove sends 60; ten says the
# stream is really there, not one stray packet).
ACQUIRE_TIMEOUT_S = 60.0
ACQUIRE_MIN_PACKETS = 10
# Seconds between "acquired" and the OPEN PALM cue: time to hear the beep,
# read the line and hold still before anything is learned.
COUNTDOWN_S = 2.0
# The mock camera can be told which pose to hold (`MockLeapStream.set_pose`),
# so a rehearsal with no hardware shows an open palm while it is acquired and
# during the open phase, and then a fist.
MOCK_POSE = {STAGE_ACQUIRE: "open_palm", STAGE_COUNTDOWN: "open_palm",
             PHASE_OPEN: "open_palm", PHASE_FIST: "fist"}
# How many frames of a hand the camera must have tracked during the OPEN
# phase before that hand may be fused: the floor `learn_flexion_scale` uses
# for a median open reference, about a fifth of a second of Leap tracking.
MIN_OPEN_CAMERA_FRAMES = MIN_OPEN_REF_FRAMES


# --- the camera side ------------------------------------------------------

class CameraBuffer:
    """The last few seconds of camera rows per hand, searchable by time.

    Each row is a dict with at least `t` (capture time, wall clock seconds),
    `hand_side`, `pts` (21 x 3 wrist-centred metres), the capture facts the
    gates read (`visible_time_us`, `palm_abs`, `palm_normal_abs`) and
    `hand_id`. `add` stamps `hand_id_stable` on it: False for
    `gates.hand_id_settle_s` after the hand's id last changed, the live form
    of `fusion.flag_hand_id_stability`, which answers the same question
    looking back over a whole take.

    `abs26`, the 26 raw camera-space joints, is what a bone length is read
    off and nothing else needs it. It is kept on a row only while
    `keep_abs26` is set, which the warm-up does; afterwards it is dropped as
    each row arrives, so the buffer holds 21 points a frame and not 47.
    """

    def __init__(self, keep_s: float = KEEP_S,
                 gates: Optional[GateParams] = None,
                 keep_abs26: bool = False):
        self.keep_s = float(keep_s)
        self.gates = gates or DEFAULT_GATES
        self.keep_abs26 = bool(keep_abs26)
        self._rows: Dict[str, Deque[Tuple[float, dict]]] = defaultdict(deque)
        self._last_id: Dict[str, object] = {}
        self._changed_at: Dict[str, float] = {}

    def add(self, row: Mapping) -> dict:
        """Store one camera row and return the stored copy.

        The copy carries `hand_id_stable`, which is why the warm-up keeps the
        returned row rather than the one it passed in: a bone-length
        measurement gates on it too.
        """
        hand = str(row["hand_side"]).strip().lower()
        t = float(row["t"])
        hid = row.get("hand_id")
        last = self._last_id.get(hand)
        if last is not None and hid != last:
            self._changed_at[hand] = t
        self._last_id[hand] = hid
        changed = self._changed_at.get(hand)
        stored = dict(row)
        stored["hand_side"] = hand
        stored["hand_id_stable"] = (changed is None
                                    or (t - changed)
                                    >= self.gates.hand_id_settle_s)
        if not self.keep_abs26:
            stored["abs26"] = None
        rows = self._rows[hand]
        rows.append((t, stored))
        newest = rows[-1][0]
        while rows and rows[0][0] < newest - self.keep_s:
            rows.popleft()
        return stored

    def nearest(self, hand: str, t: float, max_dt: float) -> Optional[dict]:
        """The row of `hand` nearest to `t`, or None if none is within `max_dt`.

        Scans from the newest row back and stops once the rows are older
        than `t - max_dt`, so a lookup costs a few dozen comparisons at most.
        On a tie the LATER row wins, as it does in `fusion.pair_by_time`.
        """
        rows = self._rows.get(str(hand).strip().lower())
        if not rows:
            return None
        t = float(t)
        best, best_dt = None, float(max_dt)
        for stamp, row in reversed(rows):
            dt = abs(stamp - t)
            if dt < best_dt or (best is None and dt <= best_dt):
                best, best_dt = row, dt
            if stamp < t - max_dt:
                break
        return best

    def latest_t(self, hand: str) -> Optional[float]:
        """Capture time of the newest row of `hand`, or None."""
        rows = self._rows.get(str(hand).strip().lower())
        return rows[-1][0] if rows else None

    def __len__(self) -> int:
        return sum(len(r) for r in self._rows.values())


def camera_row(lh, now: Optional[float] = None) -> Optional[dict]:
    """One `LeapHand` -> the row dict `CameraBuffer` and the fusion read.

    The 21 points are built exactly as the offline loader builds them
    (`fuse_poses.load_leap_cam`): the Leap hand converted to the glove's
    `HandFrame` and run through the same forward kinematics, so both
    skeletons arrive wrist-centred, in metres, in MediaPipe-21 order. The palm
    normal is taken in ABSOLUTE leap space from `abs26`, because the module
    is the origin of that space and the palm's position is therefore also the
    direction it is seen from. Returns None for a hand with no joints.
    """
    from leap_hand.to_openxr import to_hand_frame

    abs26 = getattr(lh, "abs26", None)
    if not abs26:
        return None
    frame = to_hand_frame(lh)
    palm = getattr(lh, "palm_pos", None)
    t = getattr(lh, "capture_time", None)
    if t is None:
        t = time.time() if now is None else now
    return {
        "t": float(t),
        "hand_side": str(lh.hand_side).lower(),
        "pts": frame_to_keypoints21(frame),
        "palm_abs": list(palm) if palm else None,
        "palm_normal_abs": palm_normal([abs26[i] for i in MP21_TO_OPENXR_IDX],
                                       lh.hand_side),
        "visible_time_us": getattr(lh, "visible_time_us", None),
        "hand_id": getattr(lh, "hand_id", None),
        "abs26": [list(p) for p in abs26],
    }


class CameraSource:
    """The Ultraleap (or its mock) as a stream of camera row dicts.

    Rows of a hand that is not being fused are dropped here, before the
    forward kinematics, so a second hand in view costs nothing; only the
    moment it was seen is kept, so the ACQUIRE gate can say "RIGHT hand
    seen, need LEFT". For each fused hand the newest `HandReading` (height,
    time tracked, palm facing) is kept too, for the same gate.
    """

    def __init__(self, hands: Sequence[str], mock: bool = False,
                 mode: str = "desktop"):
        self.hands = tuple(hands)
        self.mock = bool(mock)
        self.mode = mode
        self.stream = None
        self.hands_seen = 0
        self._readings: Dict[str, Tuple[HandReading, float]] = {}
        self._seen_at: Dict[str, float] = {}

    def start(self) -> "CameraSource":
        """Open the stream. Raises `leap_hand.stream.LeapUnavailable` with the
        next command to type when there is no device or no bindings."""
        from leap_hand.stream import open_stream
        self.stream = open_stream(mock=self.mock, mode=self.mode)
        return self

    def coach(self, pose: Optional[str]) -> None:
        """Tell the MOCK camera which pose to hold (None to cycle again).

        A real camera sees whatever the operator does, so this is a no-op
        for it.
        """
        setter = getattr(self.stream, "set_pose", None)
        if self.mock and setter is not None:
            setter(pose)

    def drain(self, max_items: int = 256) -> List[dict]:
        out: List[dict] = []
        if self.stream is None:
            return out
        now = time.time()
        for _side, lh in self.stream.drain(max_items):
            self.hands_seen += 1
            side = str(getattr(lh, "hand_side", "")).lower()
            self._seen_at[side] = now
            if side not in self.hands:
                continue
            row = camera_row(lh)
            if row is not None:
                out.append(row)
                if getattr(lh, "palm_pos", None):
                    self._readings[side] = (read_hand(lh), now)
        return out

    def reading(self, hand: str,
                now: Optional[float] = None) -> Optional[HandReading]:
        """The newest reading of `hand`, or None once it is `STALE_S` old."""
        got = self._readings.get(hand)
        now = time.time() if now is None else now
        if got is None or now - got[1] > STALE_S:
            return None
        return got[0]

    def seen_recently(self, hand: str, now: Optional[float] = None) -> bool:
        """Did the camera report `hand` (fused or not) within `STALE_S`?"""
        at = self._seen_at.get(hand)
        now = time.time() if now is None else now
        return at is not None and now - at <= STALE_S

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass


# --- the glove side -------------------------------------------------------

class GloveSource:
    """The glove over OSC (or one mock glove per hand) as frame dicts.

    Each dict is `{t, hand_side, frame, pts}`: `t` is the packet's ARRIVAL
    time on the wall clock (`QueueItem.recv_time`), the stamp the offline
    pairing uses; `frame` is the parsed `HandFrame`, kept because a template
    fit rescales the frame and not the points; `pts` is the raw template's
    21 points. A malformed packet is skipped, not raised: it is not data.
    """

    RATE_WINDOW_S = 1.0

    def __init__(self, hands: Sequence[str], mock: bool = False,
                 port: int = 9002):
        self.hands = tuple(hands)
        self.mock = bool(mock)
        self.port = int(port)
        self._receivers: list = []
        self._stamps: Dict[str, Deque[float]] = defaultdict(deque)
        self.packets = 0
        self.other_hand = 0

    def start(self) -> "GloveSource":
        """Start listening. The real receiver raises OSError if the port is
        already taken (another recorder still running, usually)."""
        if self.mock:
            from leap_hand.live import MockGlove
            self._receivers = [MockGlove(hand) for hand in self.hands]
        else:
            from xr_hand.receiver import OSCHandReceiver
            self._receivers = [OSCHandReceiver(port=self.port)]
        for r in self._receivers:
            r.start()
        return self

    def drain(self, max_items: int = 256) -> List[dict]:
        from xr_hand.parser import parse_hand_message

        out: List[dict] = []
        for receiver in self._receivers:
            for item in receiver.drain(max_items):
                hand, raw = item
                try:
                    frame = parse_hand_message(raw, hand_side_hint=hand)
                except Exception:
                    continue
                self.packets += 1
                side = str(frame.hand_side).lower()
                # Every side is stamped, fused or not, so the ACQUIRE gate
                # can tell "no glove at all" from "only the other glove".
                t = float(getattr(item, "recv_time", 0.0) or time.time())
                stamps = self._stamps[side]
                stamps.append(t)
                while stamps and stamps[0] < t - self.RATE_WINDOW_S:
                    stamps.popleft()
                if side not in self.hands:
                    self.other_hand += 1
                    continue
                out.append({"t": t, "hand_side": side, "frame": frame,
                            "pts": frame_to_keypoints21(frame)})
        out.sort(key=lambda g: g["t"])
        return out

    def rate_hz(self, hand: str) -> Optional[float]:
        """Packets per second of `hand` over the last second, or None."""
        stamps = self._stamps.get(hand)
        if not stamps or len(stamps) < 2 or stamps[-1] <= stamps[0]:
            return None
        return (len(stamps) - 1) / (stamps[-1] - stamps[0])

    def recent_packets(self, hand: str, now: Optional[float] = None) -> int:
        """Packets of `hand` that arrived in the last `RATE_WINDOW_S`."""
        now = time.time() if now is None else now
        return sum(1 for t in self._stamps.get(hand, ())
                   if t >= now - self.RATE_WINDOW_S)

    def coach(self, hand: str, stage: Optional[str]) -> None:
        """Start the MOCK glove of `hand` open when its OPEN PALM phase begins.

        The mock glove opens and closes on a slow sine from the moment it
        starts, whatever the warm-up is asking for. An operator opens the
        hand on the cue, so the rehearsal restarts that hand's sine at its
        open end; its fist then comes about 2 s into the warm-up. A real
        glove is whatever the operator does, so this is a no-op for it.
        """
        if not self.mock or stage != PHASE_OPEN:
            return
        for r in self._receivers:
            if getattr(r, "hand", None) == hand:
                r.gen.t = 0.0

    def stop(self) -> None:
        for r in self._receivers:
            try:
                r.stop()
            except Exception:
                pass


# --- the ACQUIRE gate -------------------------------------------------------

def band_words(band: Tuple[float, float] = DEFAULT_BAND) -> str:
    """`(18, 28)` -> `18 to 28 cm`."""
    return f"{band[0]:g} to {band[1]:g} cm"


def _other(hand: str) -> str:
    return "right" if hand == "left" else "left"


@dataclass(frozen=True)
class AcquireStatus:
    """Where one hand stands at the ACQUIRE gate, in the operator's words.

    `glove_text` and `camera_text` each say "ok" or what is missing and what
    to do about it; `caption_text` is the short form for the camera window,
    which is about 60 characters wide.
    """

    hand: str
    glove_ok: bool
    glove_text: str
    camera_ok: bool
    camera_text: str
    caption_text: str

    @property
    def ok(self) -> bool:
        return self.glove_ok and self.camera_ok

    def line(self, seconds_left: Optional[float] = None) -> str:
        """The HUD line: `ACQUIRE LEFT  glove ok  camera: no LEFT hand ...`."""
        text = (f"ACQUIRE {self.hand.upper()}  {self.glove_text}  "
                f"{self.camera_text}")
        if seconds_left is not None:
            text += f"  ({max(0.0, seconds_left):.0f} s left)"
        return text

    def refusal(self, timeout: float) -> str:
        """Why the hand is refused when the gate times out."""
        missing = [t for ok, t in ((self.glove_ok, self.glove_text),
                                   (self.camera_ok, self.camera_text))
                   if not ok]
        return (f"the {self.hand.upper()} hand was not acquired in "
                f"{timeout:g} s: " + "; ".join(missing))


def acquire_status(hand: str, glove_packets: int, other_glove_packets: int,
                   reading: Optional[HandReading], other_hand_seen: bool,
                   band: Tuple[float, float] = DEFAULT_BAND,
                   min_packets: int = ACQUIRE_MIN_PACKETS) -> AcquireStatus:
    """Is `hand` ready for its warm-up, and if not, what is missing?

    The glove is ready when the last second held at least `min_packets` of
    this hand's packets. The camera is ready when `acquire_failures`, the
    recorder's own gate, finds nothing wrong with the hand's newest reading
    (`reading`, None when there is no fresh one): tracked for at least half
    a second, inside the height band, palm toward the lens and roughly over
    the module. `other_glove_packets` and `other_hand_seen` are the other
    hand's, so a message can say "only RIGHT" rather than "nothing".
    """
    name, other = hand.upper(), _other(hand).upper()
    words = band_words(band)
    short = f"{band[0]:g}-{band[1]:g} cm"
    if glove_packets >= min_packets:
        glove_ok, glove_text = True, "glove ok"
    elif other_glove_packets >= min_packets:
        glove_ok, glove_text = False, (
            f"glove: no {name} packets, only {other} (is the {name} glove "
            "on and connected in XR Trainer?)")
    else:
        glove_ok, glove_text = False, (
            f"glove: {glove_packets} {name} packets in the last second, need "
            f"{min_packets} (is XR Trainer streaming?)")

    failures = acquire_failures(reading, hand, band)
    if not failures:
        camera_ok, camera_text, caption = True, "camera ok", ""
    elif reading is None or reading.hand_side != hand:
        camera_ok = False
        if other_hand_seen:
            camera_text = f"camera: {other} hand seen, need {name}"
            caption = f"{other} hand seen, need {name}"
        else:
            camera_text = (f"camera: no {name} hand (raise it to {words} "
                           "above the module)")
            caption = f"no {name} hand, raise it to {short}"
    else:
        camera_ok = False
        fixes, short_fixes = [], []
        for why in failures:
            if why == TOO_YOUNG:
                fixes.append(
                    f"hold it still ({reading.visible_time_us / 1e6:.1f} of "
                    f"{ACQUIRE_VISIBLE_TIME_US / 1e6:.1f} s tracked)")
                short_fixes.append("hold still")
            elif why == TOO_LOW:
                fixes.append(f"raise it to {words} (now "
                             f"{reading.height_cm:.0f} cm)")
                short_fixes.append(f"raise to {short}")
            elif why == TOO_HIGH:
                fixes.append(f"lower it to {words} (now "
                             f"{reading.height_cm:.0f} cm)")
                short_fixes.append(f"lower to {short}")
            elif why == NOT_FACING:
                fixes.append("turn the palm to the lens")
                short_fixes.append("palm to lens")
            elif why == OFF_AXIS:
                fixes.append("centre it over the module")
                short_fixes.append("centre it")
            else:
                fixes.append(why)
                short_fixes.append(why)
        camera_text = f"camera: {name} hand seen, " + ", ".join(fixes)
        caption = ", ".join(short_fixes)
    if camera_ok and not glove_ok:
        caption = f"no {name} glove packets"
    return AcquireStatus(hand=hand, glove_ok=glove_ok, glove_text=glove_text,
                         camera_ok=camera_ok, camera_text=camera_text,
                         caption_text=caption)


def countdown_text(hand: str, seconds_left: float) -> str:
    """`LEFT hand acquired: open palm in 2`, then `... in 1`."""
    return (f"{hand.upper()} hand acquired: open palm in "
            f"{max(1, math.ceil(seconds_left))}")


# --- the warm-up ----------------------------------------------------------

@dataclass
class WarmupResult:
    """What the warm-up taught, in the shapes `fuse_all` holds them in.

    rails        the rail override's rails, keyed (hand, finger); empty when
                 the override is off (`rail_params` None), as in `fuse_all`
    gate_rails   the rails the gates and the drift anchor use, learned with
                 `DEFAULT_RAIL` when the override is off: switching the
                 override off decides who may TAKE a curl, not whether the
                 gates know where a straight finger reads
    rail_tol     the tolerance those rails were learned with
    scale        the flexion-fraction endpoints, per hand, finger and sensor
    curl_gates   per hand, the spread gate's thresholds carried across a
                 template fit; empty with no fit
    measurements the bone-length measurement in force per hand (fitted hands)
    fit_refusals per hand, why a fit was asked for and not made
    refused      per hand, why the hand cannot be fused at all, as one line
                 ("LEFT hand refused: ..."); that hand is not fused, and the
                 program stops with exit code 2 when every hand is refused
    causes       per refused hand, the same reasons one by one, each naming
                 what the operator can do about it
    not_acquired the hands whose ACQUIRE gate timed out (no warm-up at all)
    """

    hands: Tuple[str, ...]
    rails: Dict[Tuple[str, str], float] = field(default_factory=dict)
    gate_rails: Dict[Tuple[str, str], float] = field(default_factory=dict)
    rail_tol: float = DEFAULT_RAIL.tol
    scale: FlexionScale = field(default_factory=lambda: FlexionScale(hands={}))
    curl_gates: Dict[str, Optional[List[float]]] = field(default_factory=dict)
    measurements: Dict[str, HandMeasurement] = field(default_factory=dict)
    fit_refusals: Dict[str, str] = field(default_factory=dict)
    refused: Dict[str, str] = field(default_factory=dict)
    causes: Dict[str, List[str]] = field(default_factory=dict)
    not_acquired: Tuple[str, ...] = ()
    fit_asked: bool = False
    # per hand, the fingers the rail override is enabled on; empty when off
    override_enabled: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    glove_frames: Dict[str, int] = field(default_factory=dict)
    camera_frames: Dict[str, int] = field(default_factory=dict)
    camera_open_frames: Dict[str, int] = field(default_factory=dict)

    def index_spans(self, hand: str
                    ) -> Tuple[Optional[float], Optional[float]]:
        """(glove span, camera span) of the index finger's endpoints."""
        hs = self.scale.for_hand(hand)
        if hs is None:
            return None, None
        out = []
        for sensor in (SENSOR_GLOVE, SENSOR_CAMERA):
            ends = hs.endpoints(sensor, "index")
            out.append(None if ends is None else float(ends.span))
        return out[0], out[1]

    def lines(self) -> List[str]:
        """The learned numbers, printed before the fusion starts.

        A gate on a normalised quantity is only checkable if the
        normalisation is printed beside it, which is why the endpoints are
        here and not only in the end-of-run summary.
        """
        out = ["Warm-up learned:"]
        for hand in self.hands:
            if hand in self.not_acquired:
                out.append(f"  {hand}: not acquired, so no warm-up")
                continue
            out.append(f"  {hand}: {self.glove_frames.get(hand, 0)} glove "
                       f"frames, {self.camera_frames.get(hand, 0)} camera "
                       f"frames ({self.camera_open_frames.get(hand, 0)} "
                       "during the open palm)")
            if hand in self.measurements:
                m = self.measurements[hand]
                out.append(f"    template fit: fitted on {m.n_frames} open-"
                           f"palm camera frames ({m.n_segments} segments)")
            elif hand in self.fit_refusals:
                out.append("    template fit: REFUSED, fused on the raw "
                           f"template ({self.fit_refusals[hand]})")
            elif self.fit_asked:
                out.append("    template fit: not for this hand (the "
                           "measurement names the other one)")
            else:
                out.append("    template fit: off")
            rails = "  ".join(
                f"{f} {self.gate_rails[(hand, f)]:.3f}"
                if (hand, f) in self.gate_rails else f"{f} --"
                for f in FINGER_NAMES)
            out.append(f"    glove rails: {rails}")
            enabled = self.override_enabled.get(hand)
            if enabled is not None:
                no_rail = [f for f in enabled if (hand, f) not in self.rails]
                out.append(
                    "    rail override enabled on: "
                    + (", ".join(enabled) or "(no finger)")
                    + (f"; no rail learned for {', '.join(no_rail)}, so it "
                       "can never act there" if no_rail else ""))
            gates = self.curl_gates.get(hand)
            if gates:
                out.append("    curl gates after the fit: " + "  ".join(
                    f"{f} {v:.3f}" for f, v in zip(FINGER_NAMES, gates)))
            hs = self.scale.for_hand(hand)
            if hs is None:
                out.append("    endpoints: none (no frames of this hand)")
                continue
            out.append("    endpoints (open / flexed / span):")
            for finger in FINGER_NAMES:
                for sensor in (SENSOR_GLOVE, SENSOR_CAMERA):
                    ends = hs.endpoints(sensor, finger)
                    text = "--" if ends is None else ends.described()
                    out.append(f"      {finger:<6} {sensor:<6} {text}")
                if finger in hs.dropped:
                    out.append(f"      {finger:<6} not normalised: "
                               f"{hs.dropped[finger]}")
        for hand in self.hands:
            if hand not in self.refused:
                continue
            out.append(f"  {hand.upper()} hand refused:")
            for cause in self.causes.get(hand) or [self.refused[hand]]:
                out.append(f"    - {cause}")
        return out


class Warmup:
    """The coached warm-up, one hand at a time, both sensors watching.

    For each hand in turn: the ACQUIRE gate (`STAGE_ACQUIRE`, up to
    `acquire_s` seconds), a countdown (`STAGE_COUNTDOWN`, `countdown_s`),
    then `PHASE_OPEN` for `open_s` seconds and `PHASE_FIST` for `fist_s`.
    `run` drives them (the caller supplies the sensor polling, the gate's
    check and the cues, so this class holds no hardware); `add_glove` and
    `add_camera` collect what the phases produce, and `learn` turns it into
    a `WarmupResult`.

    Each hand learns from its own phases only. While one hand is warming up
    (`active`), frames of the other hand are dropped, and nothing is
    collected while a hand is being acquired or counted down: that is the
    operator getting into position, not a pose.

    Within a hand's phases, collection is label-free wherever it can be,
    like the offline learning: every glove and camera frame of both phases
    goes into the rails and the endpoints, whichever phase it arrived in,
    because the glove's lag puts the first half second of each phase in the
    previous pose anyway. The one use of the phase is choosing the camera
    rows a bone-length measurement may be taken over, and `measure_hand`
    re-checks every one of those for an open hand itself.
    """

    def __init__(self, hands: Sequence[str], open_s: float = WARMUP_OPEN_S,
                 fist_s: float = WARMUP_FIST_S,
                 acquire_s: float = ACQUIRE_TIMEOUT_S,
                 countdown_s: float = COUNTDOWN_S,
                 band: Tuple[float, float] = DEFAULT_BAND):
        self.hands = tuple(hands)
        self.open_s = float(open_s)
        self.fist_s = float(fist_s)
        self.acquire_s = float(acquire_s)
        self.countdown_s = float(countdown_s)
        self.band = band
        self.phase: Optional[str] = None
        # the hand whose warm-up is running; None collects every hand
        self.active: Optional[str] = None
        # per hand whose ACQUIRE gate timed out, what was still missing
        self.not_acquired: Dict[str, str] = {}
        # (hand, raw template curls, HandFrame or None)
        self.glove: List[Tuple[str, List[float], object]] = []
        self.camera: List[Tuple[str, List[float]]] = []
        self.open_rows: List[dict] = []
        self.camera_open = Counter()

    def phases(self) -> Tuple[Tuple[str, float], ...]:
        return ((PHASE_OPEN, self.open_s), (PHASE_FIST, self.fist_s))

    def begin(self, phase: Optional[str]) -> None:
        self.phase = phase

    def _collects(self, side: str) -> bool:
        """Is a frame of `side` part of the warm-up right now?"""
        return (side in self.hands
                and self.phase not in (STAGE_ACQUIRE, STAGE_COUNTDOWN)
                and (self.active is None or side == self.active))

    def add_glove(self, hand: str, curls: Sequence[float],
                  frame=None) -> None:
        """One glove frame: its raw-template curls and, for a template fit,
        the parsed frame the fit rescales."""
        side = str(hand).strip().lower()
        if self._collects(side):
            self.glove.append((side, [float(c) for c in curls], frame))

    def add_camera(self, row: Mapping,
                   curls: Optional[Sequence[float]] = None) -> None:
        """One camera row (as returned by `CameraBuffer.add`)."""
        side = str(row["hand_side"]).strip().lower()
        if not self._collects(side):
            return
        if curls is None:
            curls = flexion_features(np.asarray(row["pts"], float))
        self.camera.append((side, [float(c) for c in curls]))
        if self.phase == PHASE_OPEN:
            self.camera_open[side] += 1
            if row.get("abs26") is not None:
                self.open_rows.append(dict(row))

    def run(self, poll: Callable[["Warmup"], None],
            cue: Callable[[str, str, float], None],
            show: Optional[Callable[[str, str, float,
                                     Optional[AcquireStatus]], None]] = None,
            acquire: Optional[Callable[[str, float], AcquireStatus]] = None,
            clock: Callable[[], float] = time.time,
            sleep: Callable[[float], None] = time.sleep) -> None:
        """Every hand's warm-up, one hand after the other.

        `poll(self)` drains the sensors into this object.
        `cue(stage, hand, seconds)` announces a stage (beep, window caption,
        console line): `STAGE_ACQUIRE` with the timeout, `STAGE_COUNTDOWN`
        once the hand is acquired, `STAGE_NOT_ACQUIRED` when the gate timed
        out, and each phase with its length. `show(stage, hand, seconds_left,
        status)` refreshes the HUD; `status` is the gate's `AcquireStatus`
        while acquiring and None otherwise. `acquire(hand, now)` is the
        gate's check; without one the phases start at once.

        A hand the gate times out on is recorded in `not_acquired` (and
        refused by `learn`), and the next hand's warm-up starts.
        """
        for hand in self.hands:
            self.active = hand
            if acquire is not None and not self._acquire(
                    hand, poll, cue, show, acquire, clock, sleep):
                continue
            for phase, seconds in self.phases():
                self._hold(phase, hand, seconds, poll, cue, show, clock,
                           sleep)
        self.begin(None)
        self.active = None

    def _hold(self, stage, hand, seconds, poll, cue, show, clock,
              sleep) -> None:
        """Announce `stage` and keep the sensors drained for `seconds`."""
        self.begin(stage)
        cue(stage, hand, seconds)
        end = clock() + seconds
        while True:
            now = clock()
            poll(self)
            if show is not None:
                show(stage, hand, end - now, None)
            if now >= end:
                break
            sleep(0.002)

    def _acquire(self, hand, poll, cue, show, acquire, clock, sleep) -> bool:
        """The ACQUIRE gate, then the countdown. False if it timed out."""
        self.begin(STAGE_ACQUIRE)
        cue(STAGE_ACQUIRE, hand, self.acquire_s)
        end = clock() + self.acquire_s
        while True:
            now = clock()
            poll(self)
            status = acquire(hand, now)
            if status.ok:
                break
            if show is not None:
                show(STAGE_ACQUIRE, hand, end - now, status)
            if now >= end:
                self.not_acquired[hand] = status.refusal(self.acquire_s)
                cue(STAGE_NOT_ACQUIRED, hand, self.acquire_s)
                return False
            sleep(0.002)
        self._hold(STAGE_COUNTDOWN, hand, self.countdown_s, poll, cue, show,
                   clock, sleep)
        return True

    def learn(self, gates: Optional[GateParams] = None,
              rail_params: Optional[RailOverrideParams] = None,
              fit=FIT_NONE,
              min_open_camera: int = MIN_OPEN_CAMERA_FRAMES,
              min_fit_frames: Optional[int] = None) -> WarmupResult:
        """Learn what `fuse_all` learns, from the warm-up's frames.

        `fit` is `FIT_NONE`, `FIT_AUTO` (measure each hand's bones on the
        open-phase camera rows) or a loaded `HandMeasurement` (applied to the
        hand it names, or to every hand when it names none, as
        `fuse_poses.fit_measurements` does). A measurement `fit_refusal`
        refuses leaves that hand on the raw template, with the reason.

        Then everything `fuse_all` learns, by the same code (`_learn_core`).

        A hand is REFUSED when its ACQUIRE gate timed out; when the glove
        sent nothing or taught no index rail (it never held the finger
        straight and still); when the camera tracked it for fewer than
        `min_open_camera` frames of the open phase (either way there is no
        open endpoint to normalise against, and fusing it would gate on
        numbers that were never measured); or when the fist did not close it
        far enough: an index span under `gates.min_glove_span` on the glove
        or `gates.min_cam_span` on the camera, the same floors
        `learn_flexion_scale` refuses to normalise a finger below. Endpoints
        that close together turn every flexion fraction, and so every gate
        built on one, into noise. Every cause found is reported, each with
        what the operator can do about it (`WarmupResult.causes`).
        """
        gates = gates or DEFAULT_GATES
        result = WarmupResult(hands=self.hands)
        result.fit_asked = fit is not None and fit != FIT_NONE

        # --- the template fit, per hand ------------------------------
        if fit == FIT_AUTO:
            for hand in self.hands:
                if hand in self.not_acquired:
                    continue                # no open palm to measure on
                m = measure_hand(self.open_rows, hand=hand, gates=gates,
                                 source="live warm-up")
                why = (fit_refusal(m) if min_fit_frames is None
                       else fit_refusal(m, min_fit_frames))
                if why:
                    result.fit_refusals[hand] = why
                else:
                    result.measurements[hand] = m
        elif isinstance(fit, HandMeasurement):
            named = [fit.hand] if fit.hand else list(self.hands)
            why = (fit_refusal(fit) if min_fit_frames is None
                   else fit_refusal(fit, min_fit_frames))
            for hand in named:
                if hand not in self.hands:
                    continue
                if why:
                    result.fit_refusals[hand] = why
                else:
                    result.measurements[hand] = fit

        _learn_core(result, self.glove, self.camera, gates, rail_params)

        # --- counts, and the refusal -----------------------------------
        result.not_acquired = tuple(h for h in self.hands
                                    if h in self.not_acquired)
        for hand in self.hands:
            result.glove_frames[hand] = sum(1 for h, *_ in self.glove
                                            if h == hand)
            result.camera_frames[hand] = sum(1 for h, _c in self.camera
                                             if h == hand)
            result.camera_open_frames[hand] = int(self.camera_open[hand])
            if hand in self.not_acquired:
                causes = [self.not_acquired[hand]]
            else:
                causes = self._causes(result, hand, gates, min_open_camera)
            if causes:
                result.causes[hand] = causes
                result.refused[hand] = (f"{hand.upper()} hand refused: "
                                        + "; ".join(causes))
        return result

    def _causes(self, result: WarmupResult, hand: str, gates: GateParams,
                min_open_camera: int) -> List[str]:
        """Why a hand that went through its phases cannot be fused, if it
        cannot, each cause with what the operator can do about it.

        The spans are read on the index, the finger every gate and the rail
        override lean on hardest, and are the same spans
        `learn_flexion_scale` refuses to normalise a finger on.
        """
        name = hand.upper()
        words = band_words(self.band)
        glove_n = result.glove_frames[hand]
        cam_open = result.camera_open_frames[hand]
        glove_span, cam_span = result.index_spans(hand)
        out: List[str] = []
        if glove_n == 0:
            out.append(f"the {name} glove sent nothing during its warm-up: "
                       "check XR Trainer is streaming that glove")
        elif glove_span is None:
            out.append(f"the {name} glove never read the index finger "
                       f"straight and still (no rail in {glove_n} frames): "
                       "keep the fingers flat and still for the whole OPEN "
                       "PALM phase")
        elif glove_span < gates.min_glove_span:
            out.append(f"the {name} glove did not register the fist (index "
                       f"span {glove_span:.3f}, need "
                       f"{gates.min_glove_span:.2f}): close a full fist "
                       "during the FIST phase, and check the glove is "
                       "calibrated in XR Trainer")
        if cam_open == 0:
            out.append(f"the camera never saw the {name} hand during the "
                       f"open palm: hold it {words} above the module, palm "
                       "to the lens")
        elif cam_open < min_open_camera:
            out.append(f"the camera saw the {name} hand on only {cam_open} "
                       f"frames of the open palm (need {min_open_camera}): "
                       f"hold it {words} above the module, palm to the lens, "
                       "for the whole phase")
        elif cam_span is None or cam_span < gates.min_cam_span:
            span = "--" if cam_span is None else f"{cam_span:.3f}"
            out.append(f"the camera did not see the {name} fist close (index "
                       f"span {span}, need {gates.min_cam_span:.2f}): close "
                       "a full fist during the FIST phase, palm still to the "
                       "lens")
        return out


def _learn_core(result: WarmupResult,
                glove: Sequence[Tuple[str, Sequence[float], object]],
                camera: Sequence[Tuple[str, Sequence[float]]],
                gates: GateParams,
                rail_params: Optional[RailOverrideParams]) -> None:
    """`fuse_all`'s learning, on whatever frames it is handed.

    `glove` is (hand, raw-template curls, HandFrame or None) per frame and
    `camera` is (hand, curls) per frame, both in the order `fuse_all` would
    read them: `learn_rails` breaks a tie between two equally common values
    by which it saw first, so the order is part of the answer.
    `result.measurements` must already hold the fitted hands.

    The rails are learned on the curls of the hand actually being FUSED (the
    fitted one when fitted), because the override compares a frame's curl
    against the rail and a rail learned on the other hand would never match;
    the endpoints are learned on the same curls; and with a fit,
    `curl_gates_from_rails` carries the spread gate's constant across it
    using rails learned on both the template and the fitted hand.
    """
    def fused_curls(hand, curls, frame):
        m = result.measurements.get(hand)
        if m is None or frame is None:
            return curls
        return flexion_features(np.asarray(
            frame_to_keypoints21(fit_template(frame, m)), float))

    glove = list(glove)
    glove_curls = [(h, fused_curls(h, c, f)) for h, c, f in glove]
    cam_curls = list(camera)
    rail_for_gates = rail_params or DEFAULT_RAIL
    result.rail_tol = rail_for_gates.tol
    result.gate_rails = learn_rails(glove_curls, rail_for_gates)
    if rail_params is not None:
        result.rails = result.gate_rails
        result.override_enabled = {h: rail_params.fingers_for(h)
                                   for h in result.hands}
    result.scale = learn_flexion_scale(glove_curls, cam_curls,
                                       result.gate_rails, gates=gates,
                                       rail_params=rail_for_gates)
    if result.measurements:
        template_rails = learn_rails(
            ((h, c) for h, c, _f in glove), rail_for_gates)
        fitted_rails = learn_rails(glove_curls, rail_for_gates)
        result.curl_gates = {
            hand: curl_gates_from_rails(gates, template_rails,
                                        fitted_rails, hand)
            for hand in {h for h, _f in template_rails}}


def learn_from_session(glove_rows: Sequence[Mapping],
                       cam_rows: Sequence[Mapping],
                       gates: Optional[GateParams] = None,
                       rail_params: Optional[RailOverrideParams] = None,
                       measurements: Optional[Mapping[str, HandMeasurement]]
                       = None) -> WarmupResult:
    """What `fuse_all` learns from a recorded session, as a WarmupResult.

    For `replay_take`: a recorded session is fused offline with rails,
    endpoints and curl gates learned over the WHOLE session, and a replay
    that learned anything else would be comparing two different fusions.
    The rows are the offline loaders' dicts, every take's in `loaded` order;
    `measurements` are the fitted hands, as `fuse_poses.fit_measurements`
    returned them. Nothing is refused here: the session is what it is.
    """
    gates = gates or DEFAULT_GATES
    glove = [(g["hand_side"], flexion_features(np.asarray(g["pts"], float)),
              g.get("frame")) for g in glove_rows]
    camera = [(c["hand_side"], flexion_features(np.asarray(c["pts"], float)))
              for c in cam_rows]
    hands = tuple(sorted({h for h, _c, _f in glove}))
    result = WarmupResult(hands=hands)
    result.measurements = dict(measurements or {})
    result.fit_asked = bool(result.measurements)
    _learn_core(result, glove, camera, gates, rail_params)
    return result


# --- one fused frame --------------------------------------------------------

@dataclass
class FusedFrame:
    """One glove frame, fused. What every output writes.

    t_glove      the glove packet's arrival time (wall clock seconds)
    t_cam        the capture time of the camera frame it was paired with, or
                 None when no camera frame was within `max_dt`
    pts          21 x 3, wrist-centred metres, MediaPipe-21 order
    dof_source   per gated DOF, which sensor supplied it this frame
    rail_active  fingers whose curl the rail override gave to the camera
    corrections  per finger, the curl change the drift anchor made
    glove_in     the glove points this step fused (the fitted hand when a
                 template fit is in force, before any drift-anchor change)
    cam_in       the paired camera frame's points, or None when unpaired

    `glove_in` and `cam_in` are the step's two inputs, kept so a viewer
    (`scripts/demo.py`) can draw the glove, camera and fused hands of one
    step side by side without pairing again. They are not written by
    `to_json` and play no part in comparing frames.
    """

    t_glove: float
    t_cam: Optional[float]
    hand: str
    pts: List[List[float]]
    dof_source: Dict[str, str]
    camera_used: bool
    rail_active: Tuple[str, ...] = ()
    corrections: Dict[str, float] = field(default_factory=dict)
    glove_in: Optional[np.ndarray] = field(default=None, repr=False,
                                           compare=False)
    cam_in: Optional[np.ndarray] = field(default=None, repr=False,
                                         compare=False)

    def to_json(self) -> dict:
        return {
            "t_glove": self.t_glove,
            "t_cam": self.t_cam,
            "hand": self.hand,
            "pts": [[round(float(v), 6) for v in p] for p in self.pts],
            "dof_source": dict(self.dof_source),
            "camera_used": bool(self.camera_used),
            "rail_active": list(self.rail_active),
            "corrections": {k: round(float(v), 6)
                            for k, v in self.corrections.items()},
        }


class LiveFusion:
    """The per-frame half of `fuse_all`, one glove frame at a time.

    Holds everything a frame needs: the gates, the rail parameters, the
    unreliable fingers per hand, one `RailOverrideTracker` per hand (its
    hysteresis counts consecutive frames of that hand), one `DriftAnchor`
    for the run, each hand's glove lag, and what the warm-up learned.

    `step` pairs, then runs `fuse_all`'s sequence exactly:

      1. the rail tracker advances on the UNCORRECTED glove curls (the
         fitted hand's, when fitted, before the anchor has moved anything);
      2. on a frame with a camera partner the anchor LEARNS from those same
         uncorrected curls;
      3. the anchor APPLIES its offset, on every frame, paired or not;
      4. `fuse_skeletons` fuses the corrected points, with `gate_curls` set
         to the uncorrected curls whenever the anchor moved a finger, so
         every gate still decides on what the glove measured.
    """

    def __init__(self, learned: WarmupResult, buffer: CameraBuffer,
                 gates: Optional[GateParams] = None,
                 rail_params: Optional[RailOverrideParams] = None,
                 unreliable: Optional[Mapping[str, Sequence[str]]] = None,
                 anchor: Union[None, DriftAnchorParams,
                               DriftAnchor] = None,
                 lag: Optional[Mapping[str, float]] = None,
                 max_dt: float = 0.05,
                 min_score: float = MIN_SCORE,
                 anchor_reset_s: Optional[float] = ANCHOR_RESET_S):
        """`anchor` is None (off), `DriftAnchorParams` (a new anchor), or a
        `DriftAnchor` to carry on with, which is how `replay_take` keeps one
        anchor across the takes of a session as `fuse_all` does.

        `anchor_reset_s`: once a hand's camera has been STALE (no frame for
        `STALE_S`) for longer than this, the anchor forgets what it learned
        (`DriftAnchor.reset`); None never does. See `step`.
        """
        self.learned = learned
        self.hands = tuple(learned.hands)
        self.buffer = buffer
        self.gates = gates or DEFAULT_GATES
        self.rail_params = rail_params
        self.unreliable = {str(k).lower(): tuple(v)
                           for k, v in (unreliable or {}).items()}
        if isinstance(anchor, DriftAnchor):
            self.anchor, self.anchor_params = anchor, anchor.params
        else:
            self.anchor_params = anchor
            self.anchor = DriftAnchor(anchor) if anchor is not None else None
        self.anchor_reset_s = (None if anchor_reset_s is None
                               else float(anchor_reset_s))
        # per hand: has the anchor been reset for the camera gap in progress
        self._reset_for_gap: Dict[str, bool] = {}
        self.anchor_resets: Counter = Counter()
        self.lag = {h: float((lag or {}).get(h, 0.0)) for h in self.hands}
        self.max_dt = float(max_dt)
        self.min_score = float(min_score)
        self.scale = learned.scale
        self.curl_gates = dict(learned.curl_gates)
        self.measurements = dict(learned.measurements)
        self.trackers = ({h: RailOverrideTracker(learned.rails, rail_params,
                                                 self.gates)
                          for h in self.hands}
                         if rail_params is not None else {})
        # session counters, per hand
        self.frames: Counter = Counter()
        self.paired: Counter = Counter()
        self.camera_used: Counter = Counter()
        self.corrected: Counter = Counter()
        self.dof_used: Dict[str, Counter] = defaultdict(Counter)
        self.dof_total: Dict[str, Counter] = defaultdict(Counter)
        self.reasons: Dict[str, Counter] = defaultdict(Counter)

    def glove_points(self, g: Mapping) -> np.ndarray:
        """The glove points to FUSE: the fitted hand, or the raw template."""
        m = self.measurements.get(g["hand_side"])
        if m is not None and g.get("frame") is not None:
            return np.asarray(frame_to_keypoints21(fit_template(g["frame"], m)),
                              float)
        return np.asarray(g["pts"], float)

    def _forget_after_camera_loss(self, hand: str, now: float) -> None:
        """Reset the drift anchor once per camera gap longer than
        `STALE_S + anchor_reset_s`.

        The glove's error depends on the pose, and a hand the camera lost for
        more than a second can come back in any pose: an offset learned
        before the gap says nothing about the hand after it, and `hold_s`
        would otherwise keep applying it for two minutes. `DriftAnchor.reset`
        forgets every hand at once (it has no per-hand form), so with both
        hands fused, losing one hand also restarts the other's learning;
        that costs `min_frames` of trusted frames, not a wrong correction.
        """
        if self.anchor is None or self.anchor_reset_s is None:
            return
        latest = self.buffer.latest_t(hand)
        gap = float("inf") if latest is None else now - latest
        if gap <= STALE_S:
            self._reset_for_gap[hand] = False
        elif (gap > STALE_S + self.anchor_reset_s
              and not self._reset_for_gap.get(hand, False)):
            self.anchor.reset()
            self._reset_for_gap[hand] = True
            self.anchor_resets[hand] += 1

    def step(self, g: Mapping) -> Optional[FusedFrame]:
        """Fuse one glove frame dict (`GloveSource.drain`). None for a hand
        this run does not fuse."""
        hand = str(g["hand_side"]).lower()
        if hand not in self.hands:
            return None
        t_glove = float(g["t"])
        # The glove's stamp is when the SOLVED hand arrived; the hand it
        # describes was in that shape `lag` seconds earlier. The anchor keeps
        # time on this same shifted clock, as it does offline.
        t = t_glove - self.lag.get(hand, 0.0)
        self._forget_after_camera_loss(hand, t_glove)
        c = self.buffer.nearest(hand, t, self.max_dt)
        G = self.glove_points(g)
        glove_in = G
        C = None
        curls_g = flexion_features(G)
        hand_scale = self.scale.for_hand(hand)
        tracker = self.trackers.get(hand)
        drift = self.anchor
        moved: Dict[str, float] = {}
        self.frames[hand] += 1
        if c is None:
            rail = tracker.update(hand, curls_g) if tracker is not None else None
            if drift is not None:
                # No camera, but the offset learned on earlier frames still
                # describes this glove: the anchor applies on every frame.
                G, moved = drift.apply(hand, t, G, curls_g, hand_scale)
            fused, info = fuse_skeletons(G, None, min_score=self.min_score,
                                         with_scale=False, gates=self.gates,
                                         rail=rail)
        else:
            self.paired[hand] += 1
            C = np.asarray(c["pts"], float)
            meta = {"visible_time_us": c.get("visible_time_us"),
                    "hand_id_stable": c.get("hand_id_stable"),
                    "palm_abs": c.get("palm_abs"),
                    "palm_normal_abs": c.get("palm_normal_abs")}
            if meta["palm_abs"] is None:
                meta = None
            curls_c = flexion_features(C)
            rail = (tracker.update(hand, curls_g, curls_c, meta)
                    if tracker is not None else None)
            if drift is not None:
                # Learn from the UNCORRECTED glove, then correct it.
                drift.learn(hand, t, curls_g, curls_c, meta, hand_scale,
                            self.learned.gate_rails, self.learned.rail_tol,
                            disputed=(rail.disputed if rail is not None
                                      else ()),
                            gates=self.gates)
                G, moved = drift.apply(hand, t, G, curls_g, hand_scale)
            fused, info = fuse_skeletons(
                G, c["pts"], cam_score=c.get("score", CAM_SCORE),
                min_score=self.min_score, thumb_from_camera=True,
                with_scale=False, cam_meta=meta, gates=self.gates, rail=rail,
                unreliable_fingers=self.unreliable.get(hand, ()),
                curl_gates=self.curl_gates.get(hand),
                scale=hand_scale,
                gate_curls=curls_g if moved else None)
            for dof in GATED_DOFS:
                self.dof_total[hand][dof] += 1
                if info["dof_source"][dof].startswith("camera"):
                    self.dof_used[hand][dof] += 1
            for why in info["rejected"].values():
                self.reasons[hand][why] += 1
        if info["camera_used"]:
            self.camera_used[hand] += 1
        if moved:
            self.corrected[hand] += 1
        return FusedFrame(
            t_glove=t_glove,
            t_cam=None if c is None else float(c["t"]),
            hand=hand,
            pts=np.asarray(fused, float).tolist(),
            dof_source=dict(info["dof_source"]),
            camera_used=bool(info["camera_used"]),
            rail_active=tuple(rail.active) if rail is not None else (),
            corrections={k: float(v) for k, v in moved.items()},
            glove_in=glove_in,
            cam_in=C)

    def summary_lines(self) -> List[str]:
        """The end-of-run report: per hand, how much was paired, how often
        the camera supplied each gated DOF, why it was refused, and what the
        drift anchor learned and moved."""
        out = ["Live fusion summary:"]
        for hand in self.hands:
            n = self.frames[hand]
            p = self.paired[hand]
            if not n:
                out.append(f"  {hand}: no glove frames fused")
                continue
            out.append(
                f"  {hand}: {n} glove frames fused, {p} paired with a camera "
                f"frame ({100.0 * p / n:.1f} %), camera used on "
                f"{self.camera_used[hand]} ({100.0 * self.camera_used[hand] / n:.1f} %)"
                f", lag {self.lag.get(hand, 0.0):.3f} s")
            if p:
                out.append("    camera use per gated DOF (paired frames):")
                for dof in GATED_DOFS:
                    used = self.dof_used[hand][dof]
                    total = self.dof_total[hand][dof]
                    out.append(f"      {dof:<14} {used:6d} / {total:<6d} "
                               f"{100.0 * used / total if total else 0.0:5.1f} %")
            if self.reasons[hand]:
                out.append("    why the camera was refused (DOF-frames):")
                for why, count in self.reasons[hand].most_common():
                    out.append(f"      {count:7d}  {why}")
        if self.anchor is None:
            out.append("  drift anchor: off")
        else:
            if self.anchor_resets:
                out.append("  drift anchor reset after a camera gap: "
                           + ", ".join(f"{h} {n} time(s)" for h, n
                                       in sorted(self.anchor_resets.items())))
            rows = self.anchor.summary()
            if not rows:
                out.append("  drift anchor: on, learned nothing (no trusted "
                           "camera frame with the glove off its rail)")
            for r in rows:
                off = r["median_offset"]
                off_curl = r["median_offset_curl"]
                out.append(
                    f"  drift anchor {r['hand']} {r['finger']}: learned from "
                    f"{r['learned']} frames, in force on {r['in_force']}, "
                    "median offset "
                    + ("--" if off is None else f"{off:+.3f}")
                    + (f" ({off_curl:+.3f} curl)" if off_curl is not None
                       else "")
                    + f", moved {r['corrected']} frames, mean |change| "
                    f"{r['mean_abs']:.3f}, max {r['max_abs']:.3f}")
        return out


# --- replaying a recorded take through the live path ------------------------

def row_stamp(row: Mapping, clock: str) -> float:
    """A row's time on `clock`, wall_time if it lacks that one: the stamp
    `fusion.pair_by_time` pairs on."""
    v = row.get(clock)
    return float(v if v is not None else row[WALL_CLOCK])


def replay_take(glove_rows: Sequence[Mapping], cam_rows: Sequence[Mapping],
                learned: WarmupResult,
                gates: Optional[GateParams] = None,
                rail_params: Optional[RailOverrideParams] = None,
                unreliable: Optional[Mapping[str, Sequence[str]]] = None,
                anchor: Union[None, DriftAnchorParams,
                              DriftAnchor] = None,
                lag: Optional[Mapping[str, float]] = None,
                max_dt: float = 0.05,
                min_score: float = MIN_SCORE,
                clock: Optional[str] = None,
                with_rows: bool = False) -> list:
    """One recorded take through `LiveFusion.step`, in time order.

    `glove_rows` and `cam_rows` are the dicts `fuse_poses.load_glove` and
    `fuse_poses.load_leap_cam` produce. `learned` is what the live path is
    handed instead of a warm-up; for a replay that is to agree with
    `fuse_all` it must be what `fuse_all` learned from the same session
    (`learn_from_session`). `anchor` is None, `DriftAnchorParams` or a
    `DriftAnchor` carried over from the previous take. Returns one
    `FusedFrame` per glove row, in the order `fuse_all` fuses them, or with
    `with_rows` one (glove row, FusedFrame) pair, so a comparison can match
    frames by the row they came from rather than by position.

    The time order is the recording's, on the take's pairing clock
    (`fusion.pairing_clock`, as offline): glove rows in stamp order, and
    camera rows entering the buffer in capture order. A camera row enters
    once the replay reaches the latest instant a glove frame could pair
    with it, `glove stamp - lag + max_dt`. That is the one inherent
    difference from live running, stated precisely: live, a camera frame
    is in the buffer only once it has ARRIVED, so with a lag of at least
    `max_dt` plus the camera's delivery delay (the Leap's frame age, a few
    ms) every candidate has arrived before the glove frame that wants it
    and live pairing is this pairing; with a smaller lag a live glove frame
    can be fused before its nearest camera frame arrives, and pairs with an
    older one or none. Offline, `pair_by_time` looks at the whole take and
    has no such limit.

    The anchor is not reset on a camera gap here (`anchor_reset_s` None):
    `fuse_all` has no such rule, and this is a comparison with `fuse_all`.
    Nothing is written back into the rows: the buffer and the step work on
    copies.
    """
    gates = gates or DEFAULT_GATES
    lag = {str(k).lower(): float(v) for k, v in (lag or {}).items()}
    clock = clock or pairing_clock(glove_rows, cam_rows)
    buffer = CameraBuffer(gates=gates)
    fusion = LiveFusion(learned, buffer, gates=gates, rail_params=rail_params,
                        unreliable=unreliable, anchor=anchor, lag=lag,
                        max_dt=max_dt, min_score=min_score,
                        anchor_reset_s=None)
    cams = sorted(((row_stamp(c, clock), c) for c in cam_rows),
                  key=lambda sc: sc[0])
    gloves = sorted(((row_stamp(g, clock), g) for g in glove_rows),
                    key=lambda sg: sg[0])
    out: List[FusedFrame] = []
    i = 0
    for t_g, g in gloves:
        horizon = t_g - lag.get(str(g["hand_side"]).lower(), 0.0) + max_dt
        while i < len(cams) and cams[i][0] <= horizon:
            buffer.add(dict(cams[i][1], t=cams[i][0]))
            i += 1
        frame = fusion.step(dict(g, t=t_g))
        if frame is not None:
            out.append((g, frame) if with_rows else frame)
    return out


# --- outputs ---------------------------------------------------------------

class JsonlSink:
    """One JSON line per `FusedFrame`, flushed per line, so a session stopped
    with Ctrl-C (or a crash) still leaves every frame fused before it."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self.written = 0

    def write(self, frame: FusedFrame) -> None:
        self._f.write(json.dumps(frame.to_json()) + "\n")
        self._f.flush()
        self.written += 1

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


class OscSink:
    """Each fused frame as two OSC messages, for whatever renders the hand.

      /fused/<hand>/keypoints21   63 floats: the 21 points x, y, z in turn,
                                  wrist-centred metres, MediaPipe-21 order
      /fused/<hand>/sources       one string "dof=source" per gated DOF, in
                                  `GATED_DOFS` order

    UDP, so a receiver that is not running costs nothing and loses nothing
    but the frames it was not there for.
    """

    def __init__(self, host: str, port: int):
        from pythonosc.udp_client import SimpleUDPClient
        self.host = host
        self.port = int(port)
        self.client = SimpleUDPClient(host, self.port)
        self.sent = 0

    def write(self, frame: FusedFrame) -> None:
        flat = [float(v) for p in frame.pts for v in p]
        try:
            self.client.send_message(f"/fused/{frame.hand}/keypoints21", flat)
            self.client.send_message(
                f"/fused/{frame.hand}/sources",
                [f"{dof}={frame.dof_source.get(dof, '')}"
                 for dof in GATED_DOFS])
            self.sent += 1
        except OSError:
            # A receiver on another machine that went away can make Windows
            # report the ICMP "port unreachable" on the next send. The next
            # frame is worth more than a traceback.
            pass

    def close(self) -> None:
        pass


# --- the HUD ------------------------------------------------------------------

def _short(source: str) -> str:
    if source.startswith("camera (rail"):
        return "rail"
    return "cam" if source.startswith("camera") else "glove"


def hud_segment(hand: str, frame: Optional[FusedFrame],
                glove_hz: Optional[float], camera_fresh: bool) -> str:
    """One hand's part of the live HUD line.

    `R 1.97 1.95 1.80 1.60 1.44  spread cam 3/4  thumb cam  override index
    anchor index-0.04  glove 60/s  camera fresh`: the fused curls (thumb to
    pinky), how many of the four spreads the camera supplied this frame, who
    supplied the thumb, which fingers the rail override holds, what the
    drift anchor moved, and whether both sensors are still talking.
    """
    tag = hand[:1].upper()
    hz = "--" if glove_hz is None else f"{glove_hz:.0f}"
    cam = "camera fresh" if camera_fresh else "camera STALE"
    if frame is None:
        return f"{tag} (no fused frame yet)  glove {hz}/s  {cam}"
    curls = " ".join(f"{v:.2f}" for v in flexion_features(
        np.asarray(frame.pts, float)))
    spreads = sum(1 for f in SPREAD_FINGERS
                  if frame.dof_source.get(f"spread {f}", "").startswith("camera"))
    thumb = _short(frame.dof_source.get("thumb", "glove"))
    override = ",".join(frame.rail_active) or "-"
    anchor = (" ".join(f"{f}{v:+.2f}" for f, v in frame.corrections.items())
              or "-")
    return (f"{tag} {curls}  spread cam {spreads}/4  thumb {thumb}  "
            f"override {override}  anchor {anchor}  glove {hz}/s  {cam}")


def hud_line(segments: Sequence[str]) -> str:
    """Both hands' segments on the one line `protocol.Hud` rewrites."""
    return "  |  ".join(segments)


def camera_fresh(buffer: CameraBuffer, hand: str,
                 now: Optional[float] = None) -> bool:
    """Has the camera reported `hand` within the last `STALE_S` seconds?"""
    latest = buffer.latest_t(hand)
    now = time.time() if now is None else now
    return latest is not None and now - latest <= STALE_S
