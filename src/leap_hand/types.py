"""The one snapshot type the Leap pipeline passes around.

`LeapHand` is one hand in one tracking event, already mapped onto the 26
OpenXR joints this repo uses everywhere (`xr_hand.joints.JOINT_NAMES` order)
and already converted to the glove's unit.

Units: **metres**. LeapC reports millimetres; the conversion happens once, in
`to_openxr.from_leap_mm`, at the moment the API object is read. Everything
downstream of that — this dataclass, the mock, the recorder, the JSONL files
— is in metres, the same as glove `HandFrame` positions (see
`scripts/glove/export_prof_format.py`, which multiplies glove world
positions by 1000 to reach the professor's millimetres).

Frame: absolute LeapC desktop-mode camera space, right-handed, origin at the
module, +x along the camera baseline, +y up, +z toward the user. Unlike the
glove — which reports nothing outside the wrist — these coordinates are real
positions in the room, which is the whole point of adding the camera.

`confidence` is deliberately absent: LeapC documents it as "not currently
used (always 1.0)", so storing it would invite gating on a constant.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class LeapHand:
    """One tracked hand from one LeapC tracking event, in OpenXR joint order."""

    hand_side: str                  # 'left' or 'right' (from hand.type)
    hand_id: int                    # LeapC hand id; changes on re-acquisition
    timestamp_us: int               # event.timestamp, LeapC clock microseconds
    frame_id: int                   # event.tracking_frame_id
    framerate: float                # event.framerate, measured tracking Hz
    visible_time_us: int            # hand.visible_time, resets on re-acquisition
    pinch_strength: float           # 0..1
    grab_strength: float            # 0..1
    palm_pos: List[float] = field(default_factory=list)   # [x, y, z] metres
    palm_quat: List[float] = field(default_factory=list)  # [x, y, z, w]
    abs26: List[List[float]] = field(default_factory=list)   # 26 x [x, y, z] m
    quat26: List[List[float]] = field(default_factory=list)  # 26 x [x, y, z, w]
    # leap.get_now() - event.timestamp at the moment the callback ran: how old
    # the tracking data already was when we received it. Frame age at receipt,
    # not end-to-end latency (it excludes exposure, and our own processing
    # after this point). None when the clock was unavailable (mock, replay).
    frame_age_us: float | None = None

    def tips(self) -> List[List[float]]:
        """The five fingertip positions, thumb to little, in metres."""
        return [self.abs26[i] for i in (5, 10, 15, 20, 25)]
