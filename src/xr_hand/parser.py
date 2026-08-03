"""Parse a raw 187-value OSC payload into a HandFrame.

Header layout (confirmed against XR Trainer real-glove stream):
    [0] tick counter (int)       -> packet_counter (used for frozen-counter detection)
    [1] frame id (int)
    [2] status / hand code (int)
    [3] device-label string      -> replaced with 0.0 in receiver (hand_side comes from hint)
    [4] device-family string     -> replaced with 0.0 in receiver
Followed by 26 joints x [x, y, z, qw, qx, qy, qz].
"""
from typing import Optional, Sequence

from .joints import (
    EXPECTED_VALUE_COUNT,
    HEADER_LEN,
    JOINT_COUNT,
    JOINT_NAMES,
    VALUES_PER_JOINT,
    HandFrame,
    Joint,
)


class PacketParseError(ValueError):
    pass


def parse_hand_message(
    values: Sequence[float], hand_side_hint: Optional[str] = None
) -> HandFrame:
    if len(values) != EXPECTED_VALUE_COUNT:
        raise PacketParseError(
            f"expected {EXPECTED_VALUE_COUNT} values, got {len(values)}"
        )

    packet_counter = int(values[0])
    timestamp = float(values[0])  # tick counter doubles as time-like
    frame_id = int(values[1])
    status = int(values[2])
    # Positions [3], [4] are string placeholders (replaced with 0.0 in receiver).
    # hand_side must come from the caller (receiver extracted it from device label).
    if hand_side_hint is None:
        raise PacketParseError(
            "hand_side_hint required: real-glove stream encodes hand in a string "
            "header field that the receiver normalises away"
        )
    hand_side = hand_side_hint

    joints = []
    for j in range(JOINT_COUNT):
        base = HEADER_LEN + j * VALUES_PER_JOINT
        # Wire order is XYZW (confirmed against XR Trainer kinematic stream):
        # cols [3,4,5,6] = qx, qy, qz, qw with magnitude == 1.000.
        x, y, z, qx, qy, qz, qw = values[base : base + VALUES_PER_JOINT]
        joints.append(
            Joint(
                name=JOINT_NAMES[j],
                x=float(x), y=float(y), z=float(z),
                qw=float(qw), qx=float(qx), qy=float(qy), qz=float(qz),
            )
        )

    return HandFrame(
        timestamp=timestamp,
        packet_counter=packet_counter,
        hand_side=hand_side,
        frame_id=frame_id,
        status=status,
        joints=joints,
    )
