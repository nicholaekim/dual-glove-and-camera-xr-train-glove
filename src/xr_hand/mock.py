"""Mock hand-frame generator.

Produces the same 187-value list the OSC receiver will deliver, in the SAME
canonical wire layout the real glove uses:
  - per-joint translation is parent-relative (local space)
  - per-joint quaternion is XYZW

The rest of the pipeline (validate -> parse -> FK -> viz) is exercised
exactly like production.

Fault toggles let you inject the failure modes you care about:
  - stuck_fingertip:        one tip stops moving (frozen local offset)
  - freeze_counter:         packet counter never increments
  - zero_joint:             one joint becomes all zeros
  - wrong_length:           payload truncated by one value
  - random_missing_prob:    each value has p chance of being NaN
"""
import math
import random
from typing import List, Optional

from .joints import HEADER_LEN, JOINT_NAMES


def _local_rest_positions() -> dict:
    """Per-joint translation relative to parent at rest pose (right hand).

    Conventions for the mock (independent of whatever the real glove uses):
      +X = pinky direction across the palm
      +Y = fingertip direction
      +Z = palm-up
    Left hand mirrors finger spread by negating the X component on metacarpals.
    """
    return {
        "WRIST":              (0.0,    0.0,   0.0),
        "PALM":               (0.0,    0.04,  0.0),
        # Metacarpals: offset from wrist with finger spread
        "THUMB_METACARPAL":   (-0.04,  0.03,  0.0),
        "INDEX_METACARPAL":   (-0.025, 0.07,  0.0),
        "MIDDLE_METACARPAL":  (0.0,    0.075, 0.0),
        "RING_METACARPAL":    (0.025,  0.07,  0.0),
        "LITTLE_METACARPAL":  (0.045,  0.065, 0.0),
        # Phalanges: extend +Y from parent (bone length only)
        "THUMB_PROXIMAL":     (0.0, 0.04,  0.0),
        "THUMB_DISTAL":       (0.0, 0.03,  0.0),
        "THUMB_TIP":          (0.0, 0.025, 0.0),
        "INDEX_PROXIMAL":     (0.0, 0.04,  0.0),
        "INDEX_INTERMEDIATE": (0.0, 0.025, 0.0),
        "INDEX_DISTAL":       (0.0, 0.02,  0.0),
        "INDEX_TIP":          (0.0, 0.015, 0.0),
        "MIDDLE_PROXIMAL":     (0.0, 0.045, 0.0),
        "MIDDLE_INTERMEDIATE": (0.0, 0.03,  0.0),
        "MIDDLE_DISTAL":       (0.0, 0.022, 0.0),
        "MIDDLE_TIP":          (0.0, 0.018, 0.0),
        "RING_PROXIMAL":     (0.0, 0.04,  0.0),
        "RING_INTERMEDIATE": (0.0, 0.028, 0.0),
        "RING_DISTAL":       (0.0, 0.02,  0.0),
        "RING_TIP":          (0.0, 0.015, 0.0),
        "LITTLE_PROXIMAL":     (0.0, 0.03,  0.0),
        "LITTLE_INTERMEDIATE": (0.0, 0.022, 0.0),
        "LITTLE_DISTAL":       (0.0, 0.018, 0.0),
        "LITTLE_TIP":          (0.0, 0.015, 0.0),
    }


_REST = _local_rest_positions()
_PHALANX_SUFFIXES = ("_PROXIMAL", "_INTERMEDIATE", "_DISTAL", "_TIP")
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)  # XYZW


def _rotx_quat(angle: float) -> tuple:
    """XYZW quaternion for rotation around +X by `angle` radians."""
    h = angle * 0.5
    return (math.sin(h), 0.0, 0.0, math.cos(h))


class MockHandGenerator:
    def __init__(self, hand: str = "right", start_counter: int = 0):
        assert hand in ("left", "right")
        self.hand = hand
        self.counter = start_counter
        self.frame_id = 0
        self.t = 0.0
        self.dt = 1.0 / 60.0
        # Place each hand's wrist along X so left/right don't overlap.
        self.origin_offset = (0.12 if hand == "right" else -0.12, 0.0, 0.0)

        # fault toggles (default: clean stream)
        self.stuck_fingertip: Optional[str] = None
        self.freeze_counter: bool = False
        self.zero_joint: Optional[str] = None
        self.wrong_length: bool = False
        self.random_missing_prob: float = 0.0
        self._frozen_local: dict = {}

    def next_frame(self) -> List[float]:
        self.t += self.dt
        if not self.freeze_counter:
            self.counter += 1
        self.frame_id += 1

        # 0..1 curl driven by a slow sine; 0 = open hand, 1 = closed fist
        curl = 0.5 * (1 - math.cos(self.t * 1.5))
        # Negative angle so fingers curl toward -Z (toward palm) given our +Y bone axis
        curl_quat = _rotx_quat(-curl * math.pi / 4)

        # Header matches the real XR Trainer kinematic stream layout:
        # [counter, frame_id, status, str-placeholder, str-placeholder]
        values: List[float] = [
            float(self.counter),
            float(self.frame_id),
            1.0,
            0.0,
            0.0,
        ]

        for name in JOINT_NAMES:
            local = _REST[name]

            # Decide rotation: phalanges curl, everyone else is identity.
            is_phalanx = any(name.endswith(s) for s in _PHALANX_SUFFIXES)
            qx, qy, qz, qw = curl_quat if is_phalanx else _IDENTITY_QUAT

            # Decide position
            if name == "WRIST":
                # Apply hand offset at the root only; FK propagates it.
                x, y, z = self.origin_offset
            elif name.endswith("_METACARPAL"):
                # Mirror finger spread for left hand
                px, py, pz = local
                x = -px if self.hand == "left" else px
                y, z = py, pz
            else:
                x, y, z = local

            if self.stuck_fingertip == name:
                if name not in self._frozen_local:
                    self._frozen_local[name] = (x, y, z)
                x, y, z = self._frozen_local[name]

            if self.zero_joint == name:
                values.extend([0.0] * 7)
            else:
                # Wire order: [x, y, z, qx, qy, qz, qw]
                values.extend([x, y, z, qx, qy, qz, qw])

        if self.wrong_length:
            values.pop()

        if self.random_missing_prob > 0.0:
            for i in range(HEADER_LEN, len(values)):
                if random.random() < self.random_missing_prob:
                    values[i] = float("nan")

        return values
