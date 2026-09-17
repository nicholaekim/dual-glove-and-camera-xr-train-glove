"""Joint definitions, bone connectivity, and shared data structures.

26 joints follow the OpenXR XR_HAND_JOINT layout, which is the convention
StretchSense and most XR hand-tracking SDKs map to.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

JOINT_NAMES: List[str] = [
    "PALM",                # 0
    "WRIST",               # 1
    "THUMB_METACARPAL",    # 2
    "THUMB_PROXIMAL",      # 3
    "THUMB_DISTAL",        # 4
    "THUMB_TIP",           # 5
    "INDEX_METACARPAL",    # 6
    "INDEX_PROXIMAL",      # 7
    "INDEX_INTERMEDIATE",  # 8
    "INDEX_DISTAL",        # 9
    "INDEX_TIP",           # 10
    "MIDDLE_METACARPAL",   # 11
    "MIDDLE_PROXIMAL",     # 12
    "MIDDLE_INTERMEDIATE", # 13
    "MIDDLE_DISTAL",       # 14
    "MIDDLE_TIP",          # 15
    "RING_METACARPAL",     # 16
    "RING_PROXIMAL",       # 17
    "RING_INTERMEDIATE",   # 18
    "RING_DISTAL",         # 19
    "RING_TIP",            # 20
    "LITTLE_METACARPAL",   # 21
    "LITTLE_PROXIMAL",     # 22
    "LITTLE_INTERMEDIATE", # 23
    "LITTLE_DISTAL",       # 24
    "LITTLE_TIP",          # 25
]
assert len(JOINT_NAMES) == 26

JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

# (parent_idx, child_idx) pairs used for drawing skeleton lines.
BONES: List[Tuple[int, int]] = [
    (1, 0),                                     # WRIST -> PALM
    (1, 2), (2, 3), (3, 4), (4, 5),             # thumb chain
    (1, 6), (6, 7), (7, 8), (8, 9), (9, 10),    # index
    (1, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # middle
    (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),  # ring
    (1, 21), (21, 22), (22, 23), (23, 24), (24, 25),  # little
]

HEADER_LEN = 5
JOINT_COUNT = 26
VALUES_PER_JOINT = 7
EXPECTED_VALUE_COUNT = HEADER_LEN + JOINT_COUNT * VALUES_PER_JOINT  # 187


@dataclass
class Joint:
    name: str
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass
class HandFrame:
    timestamp: float
    packet_counter: int
    hand_side: str           # 'left' or 'right'
    frame_id: int
    status: int
    joints: List[Joint] = field(default_factory=list)
