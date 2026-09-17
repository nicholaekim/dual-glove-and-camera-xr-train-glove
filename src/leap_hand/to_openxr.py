"""LeapC hands -> the 26 OpenXR joints -> a glove-convention `HandFrame`.

Two steps, kept apart on purpose:

  `leap_hand_from_api(hand, event)`  reads the LeapC objects and lays the
      bone endpoints out in `JOINT_NAMES` order, still absolute in camera
      space. This is the only place that touches the SDK's object model, and
      the only place millimetres become metres.

  `to_hand_frame(lh)`  turns those absolute poses into the parent-relative
      translations + XYZW quaternions a `HandFrame` holds, via
      `xr_hand.kinematics.absolute_to_relative`. Feed the result to
      `forward_kinematics` and the original camera positions come back, so
      playback, keypoints21 and the exporters work on camera data unchanged.

The 26 quaternions are **Leap bone orientations mapped onto the OpenXR joint
order**, not OpenXR joint orientations: LeapC's convention is that a bone's
local -z runs from `prev_joint` to `next_joint` with +y dorsal, and the palm
basis is {normal x direction, -normal, -direction}. Positions are exact and
convention-free; orientation semantics stay Leap's. Everything this repo
exports (21 keypoints, the professor's txt format, the classifier features)
uses positions only, so the distinction costs nothing downstream — but do
not read these quaternions as if an OpenXR runtime had produced them.

Leap's thumb `bones[0]` is a zero-length metacarpal, so the thumb chain
starts at `bones[1]`; that is why the thumb row of the table below is offset
by one relative to the other fingers.
"""
from typing import List, Optional, Sequence, Tuple

from xr_hand.joints import JOINT_NAMES, HandFrame
from xr_hand.kinematics import absolute_to_relative

from .types import LeapHand

# --- units ------------------------------------------------------------------
# LeapC reports millimetres; glove HandFrames are in metres (see
# scripts/glove/export_prof_format.py, which scales glove world positions by
# 1000 to reach the professor's mm). This constant and the function below are
# the ONLY unit conversion in the Leap pipeline — LeapHand, the mock, the
# recorder and the JSONL files are all metres.
LEAP_UNITS = "mm"
FRAME_UNITS = "m"
MM_TO_FRAME_UNITS = 0.001


def from_leap_mm(v) -> List[float]:
    """A LeapC Vector (millimetres) -> [x, y, z] in the glove's unit (metres).

    Accepts anything with .x/.y/.z (the SDK's Vector) or any 3-sequence, so
    the mock and the tests can build positions without the SDK.
    """
    try:
        x, y, z = v.x, v.y, v.z
    except AttributeError:
        x, y, z = v[0], v[1], v[2]
    return [float(x) * MM_TO_FRAME_UNITS,
            float(y) * MM_TO_FRAME_UNITS,
            float(z) * MM_TO_FRAME_UNITS]


def quat_xyzw(q) -> List[float]:
    """A LeapC Quaternion -> [x, y, z, w]. Also accepts a 4-sequence."""
    try:
        x, y, z, w = q.x, q.y, q.z, q.w
    except AttributeError:
        x, y, z, w = q[0], q[1], q[2], q[3]
    return [float(x), float(y), float(z), float(w)]


# --- hand side --------------------------------------------------------------
def is_left(hand) -> bool:
    """True when this is a left hand.

    Compares against `leap.HandType.Left` when the bindings are importable,
    which is the correct test. Falls back to the string form for replayed,
    mocked or duck-typed hands on a machine without the SDK.
    """
    hand_type = getattr(hand, "type", None)
    try:
        import leap  # noqa: PLC0415 — optional dependency, imported lazily
        return hand_type == leap.HandType.Left
    except Exception:
        return str(hand_type).endswith("Left")


def hand_side(hand) -> str:
    """'left' or 'right', the strings the rest of this repo uses."""
    return "left" if is_left(hand) else "right"


# --- the joint table --------------------------------------------------------
# One row per OpenXR joint, in JOINT_NAMES order: (digit, bone, endpoint).
# digit/bone are indices into hand.digits[..].bones[..]; endpoint is which end
# of that bone the joint sits at. "palm" and "arm" are the two special rows.
# Orientation always comes from the bone that STARTS at the joint, which is
# why a TIP row reuses its DISTAL bone's rotation.
JointSource = Tuple[Optional[int], Optional[int], str]

JOINT_SOURCES: List[JointSource] = [
    (None, None, "palm"),   # 0  PALM                palm.position / .orientation
    (None, None, "arm"),    # 1  WRIST               arm.next_joint / arm.rotation
    (0, 1, "prev"),         # 2  THUMB_METACARPAL    thumb bones[0] is zero-length:
    (0, 2, "prev"),         # 3  THUMB_PROXIMAL      the chain starts at bones[1]
    (0, 3, "prev"),         # 4  THUMB_DISTAL
    (0, 3, "next"),         # 5  THUMB_TIP
    (1, 0, "prev"),         # 6  INDEX_METACARPAL
    (1, 1, "prev"),         # 7  INDEX_PROXIMAL
    (1, 2, "prev"),         # 8  INDEX_INTERMEDIATE
    (1, 3, "prev"),         # 9  INDEX_DISTAL
    (1, 3, "next"),         # 10 INDEX_TIP
    (2, 0, "prev"),         # 11 MIDDLE_METACARPAL
    (2, 1, "prev"),         # 12 MIDDLE_PROXIMAL
    (2, 2, "prev"),         # 13 MIDDLE_INTERMEDIATE
    (2, 3, "prev"),         # 14 MIDDLE_DISTAL
    (2, 3, "next"),         # 15 MIDDLE_TIP
    (3, 0, "prev"),         # 16 RING_METACARPAL
    (3, 1, "prev"),         # 17 RING_PROXIMAL
    (3, 2, "prev"),         # 18 RING_INTERMEDIATE
    (3, 3, "prev"),         # 19 RING_DISTAL
    (3, 3, "next"),         # 20 RING_TIP
    (4, 0, "prev"),         # 21 LITTLE_METACARPAL
    (4, 1, "prev"),         # 22 LITTLE_PROXIMAL
    (4, 2, "prev"),         # 23 LITTLE_INTERMEDIATE
    (4, 3, "prev"),         # 24 LITTLE_DISTAL
    (4, 3, "next"),         # 25 LITTLE_TIP
]
assert len(JOINT_SOURCES) == len(JOINT_NAMES) == 26


def joint_pose_from_api(hand, source: JointSource):
    """One table row -> that joint's (position, rotation) LeapC objects."""
    digit, bone, endpoint = source
    if endpoint == "palm":
        return hand.palm.position, hand.palm.orientation
    if endpoint == "arm":
        arm = hand.arm
        return arm.next_joint, arm.rotation
    b = hand.digits[digit].bones[bone]
    pos = b.next_joint if endpoint == "next" else b.prev_joint
    return pos, b.rotation


def leap_hand_from_api(hand, event, frame_age_us: Optional[float] = None) -> LeapHand:
    """A LeapC `hand` from a tracking `event` -> a `LeapHand` in metres.

    Args:
        hand: one entry of `event.hands`.
        event: the tracking event it came from (for timestamp, frame id, rate).
        frame_age_us: `leap.get_now() - event.timestamp`, sampled by the
            caller at the top of the callback. Frame age at receipt, not
            end-to-end latency.

    `hand.confidence` is not read: LeapC documents it as a constant 1.0.
    """
    abs26: List[List[float]] = []
    quat26: List[List[float]] = []
    for source in JOINT_SOURCES:
        pos, rot = joint_pose_from_api(hand, source)
        abs26.append(from_leap_mm(pos))
        quat26.append(quat_xyzw(rot))

    return LeapHand(
        hand_side=hand_side(hand),
        hand_id=int(hand.id),
        timestamp_us=int(event.timestamp),
        frame_id=int(event.tracking_frame_id),
        framerate=float(getattr(event, "framerate", 0.0) or 0.0),
        visible_time_us=int(hand.visible_time),
        pinch_strength=float(hand.pinch_strength),
        grab_strength=float(hand.grab_strength),
        palm_pos=from_leap_mm(hand.palm.position),
        palm_quat=quat_xyzw(hand.palm.orientation),
        abs26=abs26,
        quat26=quat26,
        frame_age_us=None if frame_age_us is None else float(frame_age_us),
    )


def to_hand_frame(lh: LeapHand, names: Sequence[str] = JOINT_NAMES) -> HandFrame:
    """A `LeapHand` -> the glove's `HandFrame` (parent-relative, metres).

    `status=1` marks a good frame, matching the mock glove stream; a hand
    that is not tracked simply produces no frame at all, the same way the
    glove pipeline treats a dropped packet. `packet_counter` carries the
    tracking frame id so the existing per-frame filenames and the playback
    viewer's frame numbers stay meaningful.
    """
    joints = absolute_to_relative(lh.abs26, lh.quat26, names=names)
    return HandFrame(
        timestamp=lh.timestamp_us / 1e6,
        packet_counter=lh.frame_id,
        hand_side=lh.hand_side,
        frame_id=lh.frame_id,
        status=1,
        joints=joints,
    )
