"""Map the 26 OpenXR glove joints onto the 21-keypoint hand layout.

The 21-point layout is the MediaPipe / Ultralytics hand-keypoints standard
(WRIST, then 4 joints per finger), i.e. the format of the image datasets
this project compares against. The correspondence is anatomical:

  MP name             = OpenXR joint at the same anatomical landmark
  WRIST               = WRIST
  <finger>_MCP        = <finger>_PROXIMAL   (knuckle)
  <finger>_PIP        = <finger>_INTERMEDIATE
  <finger>_DIP        = <finger>_DISTAL
  <finger>_TIP        = <finger>_TIP
  THUMB_CMC/MCP/IP    = THUMB_METACARPAL/PROXIMAL/DISTAL (thumb has no PIP)

Dropped from the glove's 26: PALM and the four non-thumb metacarpals
(INDEX/MIDDLE/RING/LITTLE_METACARPAL) — the 21-point layout has no
equivalent for them.
"""
from typing import List, Tuple

from .joints import JOINT_INDEX, HandFrame
from .kinematics import forward_kinematics

# (21-keypoint name, OpenXR joint name) in the standard MP/Ultralytics order.
MP21_TO_OPENXR: List[Tuple[str, str]] = [
    ("WRIST",             "WRIST"),               # 0
    ("THUMB_CMC",         "THUMB_METACARPAL"),    # 1
    ("THUMB_MCP",         "THUMB_PROXIMAL"),      # 2
    ("THUMB_IP",          "THUMB_DISTAL"),        # 3
    ("THUMB_TIP",         "THUMB_TIP"),           # 4
    ("INDEX_FINGER_MCP",  "INDEX_PROXIMAL"),      # 5
    ("INDEX_FINGER_PIP",  "INDEX_INTERMEDIATE"),  # 6
    ("INDEX_FINGER_DIP",  "INDEX_DISTAL"),        # 7
    ("INDEX_FINGER_TIP",  "INDEX_TIP"),           # 8
    ("MIDDLE_FINGER_MCP", "MIDDLE_PROXIMAL"),     # 9
    ("MIDDLE_FINGER_PIP", "MIDDLE_INTERMEDIATE"), # 10
    ("MIDDLE_FINGER_DIP", "MIDDLE_DISTAL"),       # 11
    ("MIDDLE_FINGER_TIP", "MIDDLE_TIP"),          # 12
    ("RING_FINGER_MCP",   "RING_PROXIMAL"),       # 13
    ("RING_FINGER_PIP",   "RING_INTERMEDIATE"),   # 14
    ("RING_FINGER_DIP",   "RING_DISTAL"),         # 15
    ("RING_FINGER_TIP",   "RING_TIP"),            # 16
    ("PINKY_MCP",         "LITTLE_PROXIMAL"),     # 17
    ("PINKY_PIP",         "LITTLE_INTERMEDIATE"), # 18
    ("PINKY_DIP",         "LITTLE_DISTAL"),       # 19
    ("PINKY_TIP",         "LITTLE_TIP"),          # 20
]

MP21_NAMES: List[str] = [mp for mp, _ in MP21_TO_OPENXR]

# 21-keypoint index -> index into the 26-joint OpenXR arrays.
MP21_TO_OPENXR_IDX: List[int] = [JOINT_INDEX[xr] for _, xr in MP21_TO_OPENXR]


def frame_to_keypoints21(frame: HandFrame, wrist_centered: bool = True) -> list:
    """A HandFrame -> 21 (x, y, z) world positions in MP21_NAMES order.

    Positions are metres. With wrist_centered=True (default) the wrist is
    the origin, so hands and takes are directly comparable — the same
    convention as comparing against image-derived 3D keypoints.
    """
    world = forward_kinematics(frame)
    pts = [world[i] for i in MP21_TO_OPENXR_IDX]
    if wrist_centered:
        wx, wy, wz = pts[0]
        pts = [(x - wx, y - wy, z - wz) for x, y, z in pts]
    return pts
