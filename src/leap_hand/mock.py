"""Synthetic Leap hands — the whole pipeline with no camera and no SDK.

`MockLeapStream` has the same `start()` / `stop()` / `drain()` API as
`LeapStream`, and emits the same `(hand_side, LeapHand)` items, so every
script here runs with `--mock` on a machine that has never seen an Ultraleap
device. That is not only a convenience: it is how this backend was written
and tested at all, since the camera and the SDK arrive later.

What it generates, both hands at ~90 Hz, in LeapC desktop camera space
(millimetres, +x along the baseline, +y up, +z toward the user), a palm-down
hand hovering about 250 mm above the module:

  open_palm          fingers extended, maximum spread
  fist               fingers curled into the palm
  spread             a spread sweep, fingers together to fully splayed
  thumb_opposition   the thumb swinging across the palm and back

`pose=None` cycles through all four. Two artefacts are injected on purpose so
the analysis tooling has something to measure: a short dropout where no hand
is reported at all (20 frames in every 300, about 7% of frames), and a
hand-id change every `reacquire_every` frames — 5 s by default, which is what
a real re-acquisition looks like: new id, `visible_time` back to zero.
`scripts/leap/stats.py` counts both.

Dropouts that mean something. A periodic dropout at a fixed hover height
tells the analysis nothing about WHY the camera lost the hand, and
`leap_hand.tracking_quality` exists to answer exactly that. So a `script`
can drive the generator frame by frame: a callable `script(i)` that returns
None (nothing changes) or a dict with any of

  drop          True: no hand this frame, a dropout WHERE the script puts it
  origin_mm     (x, y, z) wrist position in camera millimetres
  roll_deg      turn about the forearm axis; 90 is the palm edge-on
  curl          0 open .. 1 fist (grab_strength follows it)
  framerate     the tracking rate the event reports
  id_offset     added to the hand id: a re-acquisition, with or without a gap
  per_side      {"left": {...}, "right": {...}}: keys for one hand only, on
                top of the rest (`drop` is not one of them: a dropout is
                the whole frame's)

and `sides` limits the generator to one hand, the way the protocol is run.
With no script the frames are exactly what they were before.

A take that is acted out. `CoachedActor` is a script that follows the coached
grasp take of `scripts/leap/record_poses.py --protocol` (OPEN HAND, MAKE THE
GRASP, HOLD STILL): the recorder calls `act(phase)` on the stream as it moves
from phase to phase, and the actor shows an open palm, closes it, and holds
the grasp. Asked to, it loses the hand while it closes, the way the camera
lost the operator's hand on 2026-10-01: the forearm turns until the palm is
edge-on to the lens, then the hand is gone and stays gone until the next
OPEN HAND, where it comes back under a new hand id.

Geometry is a plausible cartoon of a hand, not a calibrated one: segment
lengths are typical adult values and the joint angles come from two
parameters (curl, spread). It exists to exercise conversion, recording,
export and the viewers, never to stand in for measured data.
"""
import math
import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from xr_hand.kinematics import mat3_to_quat, quat_canonical

from .to_openxr import from_leap_mm
from .types import LeapHand

POSES = ("open_palm", "fist", "spread", "thumb_opposition")

# Hand skeleton in millimetres, right hand, in the hand's own frame:
#   +x across the palm toward the little finger, +y dorsal (out of the back of
#   the hand), +z from the fingertips back toward the wrist — LeapC's palm
#   basis. Fingers therefore extend along -z. The left hand is the same
#   numbers with every x negated (a proper mirror, built by negating
#   parameters rather than reflecting a matrix, so rotations stay right-handed).
#
# (metacarpal base, knuckle, [proximal, intermediate, distal] lengths, spread)
_FINGERS = {
    "index":  dict(base=(-18.0, 0.0, -6.0),  knuckle=(-22.0, 0.0, -70.0),
                   lengths=(40.0, 24.0, 20.0), spread=0.20),
    "middle": dict(base=(-5.0, 0.0, -6.0),   knuckle=(-6.0, 0.0, -73.0),
                   lengths=(45.0, 28.0, 22.0), spread=0.05),
    "ring":   dict(base=(9.0, 0.0, -6.0),    knuckle=(12.0, 0.0, -69.0),
                   lengths=(40.0, 26.0, 20.0), spread=-0.11),
    "little": dict(base=(21.0, 0.0, -6.0),   knuckle=(27.0, 0.0, -62.0),
                   lengths=(30.0, 20.0, 18.0), spread=-0.24),
}
_FINGER_ORDER = ("index", "middle", "ring", "little")

_THUMB_CMC = (-30.0, -6.0, -16.0)      # bones[1].prev_joint
_THUMB_LENGTHS = (40.0, 32.0, 25.0)    # metacarpal->proximal->distal->tip
_PALM_CENTRE = (-2.0, 0.0, -42.0)
_WRIST = (0.0, 0.0, 0.0)

# Where each wrist hovers. +x is the user's right in desktop mode (y up, z
# toward the user), so the right hand sits at +x; the thumb still comes out on
# the -x side of that hand, which is what a palm-down right hand looks like.
_HAND_X = {"right": 55.0, "left": -55.0}
_HAND_Y = 250.0                            # hover height, mm
_HAND_Z = 40.0                             # slightly toward the user


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_FORWARD = np.array([0.0, 0.0, -1.0])   # a bone points along its local -z


class MockLeapStream:
    """Both hands of synthetic Leap tracking, behind the LeapStream API."""

    def __init__(
        self,
        hz: float = 90.0,
        pose: Optional[str] = None,
        seed: int = 7,
        noise_mm: float = 0.35,
        dropout_every: int = 300,
        dropout_frames: int = 20,
        reacquire_every: int = 450,
        cycle_seconds: float = 4.0,
        sides: Tuple[str, ...] = ("left", "right"),
        script=None,
    ):
        self.hz = float(hz)
        self.pose = pose
        self.noise_mm = float(noise_mm)
        self.dropout_every = int(dropout_every)
        self.dropout_frames = int(dropout_frames)
        self.reacquire_every = int(reacquire_every)
        self.cycle_frames = max(1, int(cycle_seconds * hz))
        self.sides = tuple(sides)
        self.script = script             # see the module docstring
        self._rng = random.Random(seed)
        self._i = 0                      # frames generated so far
        self._buffer: deque = deque()
        self._clock: Optional[float] = None
        self._t0_us = 0
        # Last frame's quaternions per hand, so the sequence this generator
        # emits never flips sign between frames (see quat_canonical).
        self._prev_quats: dict = {}
        self.frames = 0                  # events emitted (dropouts included)
        self.hands_emitted = 0
        self.device_serial = "MOCK-SIR170"

    # --- lifecycle (mirrors LeapStream) --------------------------------
    def start(self) -> None:
        import time
        self._clock = time.time()
        self._t0_us = int(self._clock * 1e6)

    def stop(self) -> None:
        self._clock = None

    def set_pose(self, pose: Optional[str]) -> None:
        """Hold one pose (or None to keep cycling). Used by record_poses."""
        self.pose = pose

    def act(self, phase: Optional[str]) -> None:
        """Tell a phase-aware `script` (`CoachedActor`) where the recorder is.

        The phase starts at the frame that belongs to this wall-clock
        instant, not at the next frame generated: frames are generated
        lazily at the next `drain`, and the ones covering the time before
        this call must not be acted out as the new phase. Nothing happens
        without such a script.
        """
        if self.script is None or not hasattr(self.script, "phase"):
            return
        import time
        i = self._i
        if self._clock is not None:
            i += max(0, int((time.time() - self._clock) * self.hz))
        self.script.phase(phase, i)

    # --- generation -----------------------------------------------------
    def drain(self, max_items: int = 16) -> List[Tuple[str, LeapHand]]:
        """Pending (hand_side, LeapHand) items, paced against the wall clock."""
        import time
        if self._clock is None:
            self.start()
        now = time.time()
        n = int((now - self._clock) * self.hz)
        if n > 0:
            self._clock += n / self.hz   # keep the remainder, so pacing holds
            self._buffer.extend(self.generate(n))
        out: List[Tuple[str, LeapHand]] = []
        while self._buffer and len(out) < max_items:
            out.append(self._buffer.popleft())
        return out

    def generate(self, n_frames: int) -> List[Tuple[str, LeapHand]]:
        """`n_frames` tracking events, ignoring the wall clock.

        The deterministic entry point: tests drive this directly so a
        dropout and a re-acquisition land in known places instead of
        depending on how long the machine took to get here.
        """
        out: List[Tuple[str, LeapHand]] = []
        for _ in range(n_frames):
            i = self._i
            self._i += 1
            self.frames += 1
            # Dropout sits at the END of each period, so a short run still
            # starts with tracked frames.
            if (self.dropout_every
                    and (i % self.dropout_every) >= self.dropout_every - self.dropout_frames):
                continue  # hand not reported at all — a real tracking dropout
            extra = self.script(i) if self.script is not None else None
            if extra and extra.get("drop"):
                continue  # a scripted dropout: the script chose where it happens
            for side in self.sides:
                out.append((side, self._hand(side, i, self._side_extra(extra, side))))
                self.hands_emitted += 1
        return out

    @staticmethod
    def _side_extra(extra: Optional[dict], side: str) -> Optional[dict]:
        """One hand's script keys: `extra` with its `per_side[side]` on top."""
        if not extra or "per_side" not in extra:
            return extra
        own = (extra.get("per_side") or {}).get(side) or {}
        return {**{k: v for k, v in extra.items() if k != "per_side"}, **own}

    # --- one hand -------------------------------------------------------
    def _pose_params(self, i: int) -> Tuple[str, float, float, float]:
        """(pose name, curl, spread, thumb opposition) at frame i."""
        pose = self.pose
        if pose is None:
            pose = POSES[(i // self.cycle_frames) % len(POSES)]
        phase = 2.0 * math.pi * (i % self.cycle_frames) / self.cycle_frames
        sweep = 0.5 * (1.0 - math.cos(phase))       # 0 -> 1 -> 0
        if pose == "fist":
            return pose, 1.0, 0.0, 0.85
        if pose == "spread":
            return pose, 0.05, sweep, 0.25
        if pose == "thumb_opposition":
            return pose, 0.15, 0.30, sweep
        return "open_palm", 0.0, 1.0, 0.0           # open_palm and anything else

    def _hand(self, side: str, i: int, extra: Optional[dict] = None) -> LeapHand:
        _, curl, spread, opposition = self._pose_params(i)
        extra = extra or {}
        if "curl" in extra:
            curl = max(0.0, min(1.0, float(extra["curl"])))
            spread = 1.0 - curl
            opposition = 0.85 * curl
        mirror = -1.0 if side == "left" else 1.0
        origin = (np.array([float(v) for v in extra["origin_mm"]])
                  if "origin_mm" in extra
                  else np.array([_HAND_X[side], _HAND_Y, _HAND_Z]))
        # A turn about the forearm axis (+z, wrist to elbow), applied to the
        # whole hand about the wrist. None, not the identity, when there is
        # none, so an unscripted frame is computed exactly as it always was.
        roll = math.radians(float(extra.get("roll_deg", 0.0)))
        turn = _rot_z(roll) if roll else None

        def orient(rot: np.ndarray) -> np.ndarray:
            return rot if turn is None else turn @ rot

        def place(local) -> np.ndarray:
            v = np.array([local[0] * mirror, local[1], local[2]])
            if turn is not None:
                v = turn @ v
            return origin + v

        abs26: List[np.ndarray] = [None] * 26       # type: ignore[list-item]
        quat26: List[Tuple[float, float, float, float]] = [None] * 26  # type: ignore

        # PALM / WRIST. Palm down, fingers away from the user: the hand frame
        # coincides with the camera frame, so both are the identity rotation
        # (mirrored hands included — the mirror lives in the x offsets).
        hand_rot = orient(np.eye(3))
        abs26[0] = place(_PALM_CENTRE)
        quat26[0] = mat3_to_quat(hand_rot)
        abs26[1] = place(_WRIST)
        quat26[1] = mat3_to_quat(hand_rot)

        # Thumb: yaw swings it across the palm, pitch drops it toward the palm.
        yaw = (0.75 - 0.55 * opposition) * mirror
        pitch = -0.15 - 0.85 * opposition - 0.55 * curl
        p = place(_THUMB_CMC)
        for k, length in enumerate(_THUMB_LENGTHS):
            rot = orient(_rot_y(yaw) @ _rot_x(pitch - 0.35 * k * (opposition + curl)))
            abs26[2 + k] = p
            quat26[2 + k] = mat3_to_quat(rot)
            p = p + rot @ _FORWARD * length
        abs26[5] = p                                  # THUMB_TIP
        quat26[5] = quat26[4]                         # tip reuses the distal bone

        # Fingers.
        for f, name in enumerate(_FINGER_ORDER):
            spec = _FINGERS[name]
            j = 6 + f * 5
            abs26[j] = place(spec["base"])            # METACARPAL
            knuckle = place(spec["knuckle"])
            metacarpal_rot = orient(_rot_y(spec["spread"] * spread * mirror))
            quat26[j] = mat3_to_quat(metacarpal_rot)

            p = knuckle
            angle = 0.0
            for k, length in enumerate(spec["lengths"]):
                angle -= curl * (0.95 if k == 0 else 1.05)
                rot = orient(_rot_y(spec["spread"] * spread * mirror) @ _rot_x(angle))
                abs26[j + 1 + k] = p
                quat26[j + 1 + k] = mat3_to_quat(rot)
                p = p + rot @ _FORWARD * length
            abs26[j + 4] = p                          # TIP
            quat26[j + 4] = quat26[j + 3]

        sigma = self.noise_mm
        positions = [
            from_leap_mm([v[0] + self._rng.gauss(0.0, sigma),
                          v[1] + self._rng.gauss(0.0, sigma),
                          v[2] + self._rng.gauss(0.0, sigma)])
            for v in abs26
        ]

        # Keep the sign continuous with the previous frame of this hand. Real
        # LeapC rotations are whatever the tracker emits; this is a generator,
        # and a generator that flips sign mid-sweep would hand anyone
        # differencing quaternions a fake discontinuity to chase.
        previous = self._prev_quats.get(side)
        quat26 = [
            quat_canonical(q, previous[k] if previous else None)
            for k, q in enumerate(quat26)
        ]
        self._prev_quats[side] = quat26

        generation = i // self.reacquire_every if self.reacquire_every else 0
        since_reacquire = i - generation * self.reacquire_every if self.reacquire_every else i
        base_id = 1001 if side == "right" else 2001
        base_id += int(extra.get("id_offset", 0))

        # This generator's LeapC clock IS the wall clock: `_t0_us` is
        # `time.time()` at start(), and frame i sits one interval after it. So
        # `capture_time` is that same instant in seconds, and the mock behaves
        # like the device in the way that matters here — drain a backlog in
        # one pass and those hands carry capture times spread across the
        # period they were "captured" while sharing one `wall_time`. Anything
        # that pairs on the writer's clock fails against this mock for the
        # same reason it would fail against the camera.
        timestamp_us = self._t0_us + int(round(i * 1e6 / self.hz))

        return LeapHand(
            hand_side=side,
            hand_id=base_id + generation,
            timestamp_us=timestamp_us,
            frame_id=i,
            framerate=(float(extra["framerate"]) if "framerate" in extra
                       else self.hz + self._rng.gauss(0.0, 0.4)),
            visible_time_us=int(since_reacquire * 1e6 / self.hz),
            pinch_strength=round(min(1.0, opposition * 0.9 + curl * 0.1), 3),
            grab_strength=round(curl, 3),
            palm_pos=positions[0],
            palm_quat=list(quat26[0]),
            abs26=positions,
            quat26=[list(q) for q in quat26],
            frame_age_us=None,   # no LeapC clock behind a mock
            capture_time=timestamp_us / 1e6,
        )


class CoachedActor:
    """A `MockLeapStream` script that acts out the coached grasp take.

    `scripts/leap/record_poses.py --protocol` calls `MockLeapStream.act`
    with the phase it is in, and the hand does what the operator is asked:

      "open"   an open palm square to the lens where the stream puts it, 25 cm
               up. Absent for the first `absent_open_s` of every OPEN HAND,
               which makes the recorder wait for it (or time out, when that
               is longer than the recorder's timeout).
      "form"   the fingers close to `grasp_curl` over `close_s`, the palm
               kept toward the lens. On a try that is to be lost the forearm
               also rolls the palm edge-on to the lens (`lost_roll_deg`)
               over `lose_at_s`, as the hand turns when a grasp is copied
               with the fingertips down, and from then on no hand is
               reported until the next phase.
      "hold"   the grasp held, palm toward the lens. `hold_dropout` (0 to 1)
               drops that share of every `period` frames, counted from the
               start of the hold, so a rehearsal of the gate's rejection
               still gets through OPEN HAND and MAKE THE GRASP.
      anything else ("idle", None)  nothing scripted: the stream's own pose.

    The first `lose_forming` tries at MAKE THE GRASP of every attempt are
    lost; an attempt ends at "hold" or "idle". A hand that comes back after
    a loss carries a new hand id, as LeapC gives a re-acquired hand.

    `operator` is the hand doing all that. The stream shows both hands, and
    `other_hand` says what the other one does: "same" (the default, and
    what this actor always did) copies the operator's hand 11 cm beside
    it; "open" holds it as an open palm 25 cm to the side at the same
    height, never closing and never turned, the way the operator held the
    idle right hand in view on 2026-10-01 to help the tracker tell left from
    right. A scripted dropout still drops the whole frame, both hands.

    Each phase is kept with the frame it started at, so a frame generated
    after a phase change but belonging to the time before it is still
    acted out as the phase it belongs to.
    """

    OTHER_HAND_MODES = ("same", "open")

    def __init__(self, hz: float = 90.0, lose_forming: int = 0,
                 grasp_curl: float = 0.6, close_s: float = 1.0,
                 lose_at_s: float = 0.6, lost_roll_deg: float = 85.0,
                 absent_open_s: float = 0.0, hold_dropout: float = 0.0,
                 period: int = 90, operator: str = "left",
                 other_hand: str = "same"):
        if operator not in ("left", "right"):
            raise ValueError(f"operator must be left or right, not {operator!r}")
        if other_hand not in self.OTHER_HAND_MODES:
            raise ValueError(f"other_hand must be one of {self.OTHER_HAND_MODES}, "
                             f"not {other_hand!r}")
        self.hz = float(hz)
        self.operator = operator
        self.other_hand = other_hand
        self.lose_forming = max(0, int(lose_forming))
        self.grasp_curl = float(grasp_curl)
        self.close_s = float(close_s)
        self.lose_at_s = float(lose_at_s)
        self.lost_roll_deg = float(lost_roll_deg)
        self.absent_open_s = float(absent_open_s)
        self.period = max(2, int(period))
        self.hold_drop_frames = (
            0 if hold_dropout <= 0
            else max(1, min(self.period - 1, round(self.period * float(hold_dropout)))))
        self._segments: List[Tuple[int, Optional[str], bool, int]] = []
        self._tries = 0                  # MAKE THE GRASP tries in this attempt
        self._id_offset = 0
        self.lost = 0                    # tries acted out as lost, all attempts
        self.phases: List[Optional[str]] = []   # every phase asked for, in order

    def phase(self, name: Optional[str], i: int) -> None:
        """The recorder moved to phase `name` at frame `i`."""
        losing = False
        if name == "form":
            self._tries += 1
            losing = self._tries <= self.lose_forming
            self.lost += int(losing)
        elif name == "open":
            if self._segments and self._segments[-1][2]:
                self._id_offset += 1     # the hand comes back as a new hand
        else:
            self._tries = 0              # "hold" or "idle": the attempt is over
        self.phases.append(name)
        self._segments.append((int(i), name, losing, self._id_offset))
        del self._segments[:-8]

    def _segment(self, i: int):
        for seg in reversed(self._segments):
            if seg[0] <= i:
                return seg
        return None

    def __call__(self, i: int) -> Optional[dict]:
        extra = self._act(i)
        if self.other_hand == "same" or (extra and extra.get("drop")):
            return extra
        other = "right" if self.operator == "left" else "left"
        sign = 1.0 if other == "right" else -1.0
        out = dict(extra or {})
        out["per_side"] = {other: {"curl": 0.0, "roll_deg": 0.0,
                                   "origin_mm": [sign * 250.0, 250.0, -100.0]}}
        return out

    def _act(self, i: int) -> Optional[dict]:
        """The operator's hand at frame i (both hands' with other_hand "same")."""
        seg = self._segment(i)
        if seg is None:
            return None
        start, name, losing, offset = seg
        t = (i - start) / self.hz
        if name == "open":
            if t < self.absent_open_s:
                return {"drop": True}
            return {"curl": 0.0, "id_offset": offset}
        if name == "form":
            w = min(1.0, t / self.close_s) if self.close_s > 0 else 1.0
            curl = self.grasp_curl * w * w * (3.0 - 2.0 * w)
            if not losing:
                return {"curl": curl, "id_offset": offset}
            if t >= self.lose_at_s:
                return {"drop": True}
            turn = t / self.lose_at_s if self.lose_at_s > 0 else 1.0
            return {"curl": curl, "roll_deg": self.lost_roll_deg * turn,
                    "id_offset": offset}
        if name == "hold":
            k = i - start
            if (self.hold_drop_frames
                    and k % self.period >= self.period - self.hold_drop_frames):
                return {"drop": True}
            return {"curl": self.grasp_curl, "id_offset": offset}
        return {"id_offset": offset} if offset else None
