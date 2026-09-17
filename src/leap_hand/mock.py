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
    ):
        self.hz = float(hz)
        self.pose = pose
        self.noise_mm = float(noise_mm)
        self.dropout_every = int(dropout_every)
        self.dropout_frames = int(dropout_frames)
        self.reacquire_every = int(reacquire_every)
        self.cycle_frames = max(1, int(cycle_seconds * hz))
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
            for side in ("left", "right"):
                out.append((side, self._hand(side, i)))
                self.hands_emitted += 1
        return out

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

    def _hand(self, side: str, i: int) -> LeapHand:
        _, curl, spread, opposition = self._pose_params(i)
        mirror = -1.0 if side == "left" else 1.0
        origin = np.array([_HAND_X[side], _HAND_Y, _HAND_Z])

        def place(local) -> np.ndarray:
            v = np.array([local[0] * mirror, local[1], local[2]])
            return origin + v

        abs26: List[np.ndarray] = [None] * 26       # type: ignore[list-item]
        quat26: List[Tuple[float, float, float, float]] = [None] * 26  # type: ignore

        # PALM / WRIST. Palm down, fingers away from the user: the hand frame
        # coincides with the camera frame, so both are the identity rotation
        # (mirrored hands included — the mirror lives in the x offsets).
        hand_rot = np.eye(3)
        abs26[0] = place(_PALM_CENTRE)
        quat26[0] = mat3_to_quat(hand_rot)
        abs26[1] = place(_WRIST)
        quat26[1] = mat3_to_quat(hand_rot)

        # Thumb: yaw swings it across the palm, pitch drops it toward the palm.
        yaw = (0.75 - 0.55 * opposition) * mirror
        pitch = -0.15 - 0.85 * opposition - 0.55 * curl
        p = np.array([_THUMB_CMC[0] * mirror, _THUMB_CMC[1], _THUMB_CMC[2]]) + origin
        for k, length in enumerate(_THUMB_LENGTHS):
            rot = _rot_y(yaw) @ _rot_x(pitch - 0.35 * k * (opposition + curl))
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
            metacarpal_rot = _rot_y(spec["spread"] * spread * mirror)
            quat26[j] = mat3_to_quat(metacarpal_rot)

            p = knuckle
            angle = 0.0
            for k, length in enumerate(spec["lengths"]):
                angle -= curl * (0.95 if k == 0 else 1.05)
                rot = _rot_y(spec["spread"] * spread * mirror) @ _rot_x(angle)
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

        return LeapHand(
            hand_side=side,
            hand_id=base_id + generation,
            timestamp_us=self._t0_us + int(round(i * 1e6 / self.hz)),
            frame_id=i,
            framerate=self.hz + self._rng.gauss(0.0, 0.4),
            visible_time_us=int(since_reacquire * 1e6 / self.hz),
            pinch_strength=round(min(1.0, opposition * 0.9 + curl * 0.1), 3),
            grab_strength=round(curl, 3),
            palm_pos=positions[0],
            palm_quat=list(quat26[0]),
            abs26=positions,
            quat26=[list(q) for q in quat26],
            frame_age_us=None,   # no LeapC clock behind a mock
        )
