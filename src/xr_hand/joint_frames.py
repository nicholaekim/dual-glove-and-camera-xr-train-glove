"""Every joint's own x y z frame, and the angles the professor's paper names.

Cobos et al. 2009 ("Human Hand Descriptions and Gesture Recognition for
Object Manipulation", Figure 1 and Table 1) model the hand with a local frame
at every joint and 24 named angles: per finger (I, M, R, L) the
carpometacarpal CMC, the metacarpophalangeal flexion MCP_fe and abduction
MCP_aa, PIP and DIP; for the thumb (T) the trapeziometacarpal flexion TMC_fe
and abduction TMC_aa, MCP_fe and IP. This module reads one recorded frame,
glove or camera, and gives back the same thing in numbers: for each of the
26 OpenXR joints its position, its orientation, its three unit axes, the
length of the bone to its parent and the rotation from the parent's frame to
its own split into flexion, abduction and twist. `dof24` then picks the
paper's 24 out of those 26 rows. Pure: no files, no SDK, no window.

THE WRIST FRAME
  Every position and orientation in a row is expressed in the WRIST frame:
  the wrist joint at the origin, the axes those of the wrist joint's own
  orientation, millimetres. The glove has no position in the room and the
  camera has nothing else, so this is the one frame in which the two
  sensors' numbers can sit side by side.

  One difference the numbers carry and the reader must know. On the glove
  the WRIST joint is the root of a hand that never moves (its WRIST and PALM
  have the same, constant orientation). On the camera the WRIST quaternion is
  LeapC's `arm.rotation`, the FOREARM bone (see `leap_hand.to_openxr`), so
  when the wrist is bent the whole camera hand is turned by that bend in the
  wrist frame, and the four metacarpal rows (the paper's CMC) include it.
  `reference="palm"` replaces the WRIST joint's orientation with the PALM
  joint's before anything else is computed: on the glove that changes
  nothing at all, on the camera it gives a hand-fixed frame. At the summary
  frames of the 20 fist and open-palm takes of day 2 the camera's forearm
  and palm differ by a median of 27 degrees (3 to 39), and the metacarpal
  rows of fist_left take 1 read CMC -17 to -19 against the forearm and +6 to
  +9 against the palm.
  The default stays what was asked for (the wrist joint); `wrist_palm_deg`
  gives the size of the difference for any frame, and the other reference
  is one argument away.

THE AXES, FROM THE TWO SOURCES' OWN CONVENTIONS
  Both sources already agree, which is why there is one rule and no case
  per source:

    LeapC  a bone's local -z runs from prev_joint to next_joint (so +z points
           back toward the wrist), +y is dorsal (the back of the hand), and
           x = y cross z because a quaternion is a proper rotation
           (`leap_hand.to_openxr`; the mock builds its bones the same way,
           `leap_hand.mock._FORWARD`).
    glove  every child sits along -z of its parent (INDEX_INTERMEDIATE is
           (0, 0, -0.0369) m from INDEX_PROXIMAL on every frame), and every
           finger joint turns about its local x only; on
           `recordings/sync_day2/glove/fist_left_take1*` the index PIP is
           -90.0 degrees about +x and the fingertip ends up along -y, so the
           palm is -y and dorsal is +y, as on the camera.

  So in both: x is the flexion axis, y (dorsal) the abduction axis, z the
  bone axis, and a finger curling toward the palm is a NEGATIVE rotation
  about x. `flex_deg` is therefore minus that rotation: positive = toward the
  palm, on both sources, both hands, every joint including the thumb.

  x = y cross z has a consequence: on a right hand +x points to the little
  finger, on a left hand to the thumb (both sources; the glove's right hand
  is its left hand with x negated). Flexion is about x and does not care,
  but a rotation about y or z has the opposite anatomical meaning on the two
  hands. So abduction and twist are signed by the frame line's own
  `hand_side` (for the camera the tracker's label, which is the chirality
  its skeleton was fitted as, even when it is wrong about whose hand it is):

    abd_deg    positive = toward the thumb side (radial), both hands
    twist_deg  positive = the back of the bone turns toward the little
               finger side (the pad turns toward the thumb), both hands

WHICH ORDER, AND WHY
  The rotation from parent to child is split as R = Rx(a) Ry(b) Rz(c)
  (intrinsic x, then the new y, then the new z), flex = -a. Two reasons:

    * The middle angle is the one that locks: here that is abduction, which
      no finger joint comes near 90 degrees on. A flexion-in-the-middle order
      (y, x, z) locks at 90 degrees of flexion, which every fist reaches, and
      its abduction and twist then swing wildly from frame to frame.
    * In this order flexion and abduction depend on the child's bone
      direction alone (the third column of R): abduction is the angle of the
      bone out of the parent's sagittal y-z plane, flexion the angle of its
      projection inside that plane. Twist is what is left about the bone.
      LeapC builds its finger joints as pure swings (quaternion z component
      zero), for which this twist is not 0 but about abduction times
      tan(flexion / 2): 2 degrees at 45 of flexion and 5 of abduction, 5 at
      90. A twist of that size on a camera row is the order, not the finger.

  For the thumb's first row (THUMB_METACARPAL relative to WRIST) the three
  angles include the thumb's resting roll: TMC_fe and TMC_aa are where the
  thumb metacarpal points (palmward and radially), its twist is its roll
  about its own axis, which is where opposition shows. None of them is zero
  at rest. The glove's thumb CMC is flexion only and its metacarpal splay a
  fixed 5/10/15 degree template (the glove's own model), so on the glove the
  CMC and MCP_aa values are constants; the camera measures them.

  xr_hand.mock is a cartoon with its own axes (bones along +y, dorsal +z);
  only its flexion, about x, reads the same here.

MEASURED ON DAY 2 (recordings/sync_day2, summary frame = medoid of the
whole take, index PIP flexion in degrees, wrist reference)

    glove   fist_left take1-5        89.8 to 90.0
    glove   open_palm_left take1-5   4.1 on every take
    camera  fist_left take1-5        77.3 to 80.3 (glove minus camera
                                     9.7 to 12.7)
    camera  open_palm_left take1-5   3.6 to 5.3 (within 1.2 of the glove)
    right hand, both sensors         fist 52 to 85, open palm 4 to 12

  Both hands, both sensors, positive toward the palm, with no case for
  either. The camera's index MCP_aa reads +0.4 to +9.5 (toward the thumb)
  in the ten open palms and -2.0 to -7.5 in the five left fists, as a
  spread and a closed hand should. The scratchpad script `check_joint_frames.py` re-measures
  the fist and open-palm numbers and fails when a sign flips.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .joints import JOINT_INDEX, JOINT_NAMES, HandFrame, Joint
from .kinematics import (PARENT, forward_kinematics_full, mat3_to_quat,
                         quat_canonical, quat_to_mat3)

SOURCES = ("glove", "camera")
REFERENCES = ("wrist", "palm")

WRIST = JOINT_INDEX["WRIST"]
PALM = JOINT_INDEX["PALM"]

# The row a table holds for each joint, in this order.
ROW_KEYS = ["joint", "parent", "x_mm", "y_mm", "z_mm", "qx", "qy", "qz", "qw",
            "axis_x", "axis_y", "axis_z", "bone_len_mm",
            "flex_deg", "abd_deg", "twist_deg"]

# One CSV column per number: the three axes are written out component by
# component, `axis_x_y` being the y component (in the wrist frame) of the
# joint's own x axis.
CSV_COLUMNS = (["joint", "parent", "x_mm", "y_mm", "z_mm",
                "qx", "qy", "qz", "qw"]
               + [f"axis_{a}_{c}" for a in "xyz" for c in "xyz"]
               + ["bone_len_mm", "flex_deg", "abd_deg", "twist_deg"])

# The paper's letters for the five digits, and the OpenXR prefix of each.
FINGERS = [("T", "THUMB"), ("I", "INDEX"), ("M", "MIDDLE"), ("R", "RING"),
           ("L", "LITTLE")]

# The paper's 24 angles (Table 1 order), each read off one row:
# name -> (joint, which angle of that row).
DOF24_SOURCE: Dict[str, Tuple[str, str]] = {
    "T_TMC_fe": ("THUMB_METACARPAL", "flex_deg"),
    "T_TMC_aa": ("THUMB_METACARPAL", "abd_deg"),
    "T_MCP_fe": ("THUMB_PROXIMAL", "flex_deg"),
    "T_IP": ("THUMB_DISTAL", "flex_deg"),
}
for _letter, _finger in FINGERS[1:]:
    DOF24_SOURCE.update({
        f"{_letter}_CMC": (f"{_finger}_METACARPAL", "flex_deg"),
        f"{_letter}_MCP_fe": (f"{_finger}_PROXIMAL", "flex_deg"),
        f"{_letter}_MCP_aa": (f"{_finger}_PROXIMAL", "abd_deg"),
        f"{_letter}_PIP": (f"{_finger}_INTERMEDIATE", "flex_deg"),
        f"{_letter}_DIP": (f"{_finger}_DISTAL", "flex_deg"),
    })
DOF24_NAMES: List[str] = list(DOF24_SOURCE)
assert len(DOF24_NAMES) == 24

# The paper's name for the joint each row's angles describe, for labels.
PAPER_JOINT: Dict[str, str] = {
    "THUMB_METACARPAL": "TMC", "THUMB_PROXIMAL": "MCP", "THUMB_DISTAL": "IP"}
for _letter, _finger in FINGERS[1:]:
    PAPER_JOINT.update({f"{_finger}_METACARPAL": "CMC",
                        f"{_finger}_PROXIMAL": "MCP",
                        f"{_finger}_INTERMEDIATE": "PIP",
                        f"{_finger}_DISTAL": "DIP"})


def finger_joints(finger: str) -> List[int]:
    """Row indices of one digit ('THUMB', 'INDEX', ... or the paper's letter)."""
    prefix = dict(FINGERS).get(finger, finger).upper()
    return [i for i, n in enumerate(JOINT_NAMES) if n.startswith(prefix + "_")]


# --- rotations ----------------------------------------------------------------
def xyz_angles(r: np.ndarray) -> Tuple[float, float, float]:
    """(a, b, c) in radians with r = Rx(a) @ Ry(b) @ Rz(c).

    b is the angle that locks, at +-90 degrees; there c is set to 0 and the
    whole remaining rotation goes into a, so a lone rotation still comes back
    whole.
    """
    r = np.asarray(r, dtype=float)
    sb = float(np.clip(r[0, 2], -1.0, 1.0))
    b = float(np.arcsin(sb))
    if abs(sb) < 1.0 - 1e-9:
        a = float(np.arctan2(-r[1, 2], r[2, 2]))
        c = float(np.arctan2(-r[0, 1], r[0, 0]))
    else:
        a = float(np.arctan2(r[2, 1], r[1, 1]))
        c = 0.0
    return a, b, c


def rot_x(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hand_sign(hand: str) -> float:
    """+1 for a right hand, -1 for a left: the mirror in abduction and twist."""
    if hand == "right":
        return 1.0
    if hand == "left":
        return -1.0
    raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")


def joint_angles(r_rel: np.ndarray, hand: str) -> Tuple[float, float, float]:
    """(flex, abd, twist) in degrees for a parent-to-child rotation.

    See the module docstring: flex > 0 toward the palm, abd > 0 toward the
    thumb, twist > 0 back of the bone toward the little finger, both hands.
    """
    a, b, c = xyz_angles(r_rel)
    s = hand_sign(hand)
    flex = -np.degrees(a)
    abd = s * np.degrees(b)
    twist = -s * np.degrees(c)
    return float(flex) + 0.0, float(abd) + 0.0, float(twist) + 0.0


def rotation_from_angles(flex: float, abd: float, twist: float,
                         hand: str) -> np.ndarray:
    """The parent-to-child rotation `joint_angles` reads back as these angles."""
    s = hand_sign(hand)
    return rot_x(-flex) @ rot_y(s * abd) @ rot_z(-s * twist)


# --- one frame line -> the hand in some world frame ------------------------------
def _has(line: dict, key: str) -> bool:
    """The line carries a non-empty `key` (a list or an array alike)."""
    value = line.get(key)
    return value is not None and len(value) > 0


def detect_source(line: dict) -> str:
    """'camera' for a line the Leap recorder wrote, 'glove' otherwise."""
    if (_has(line, "abs26") or _has(line, "quat26")
            or str(line.get("source", "")).lower() in ("leap", "camera")):
        return "camera"
    return "glove"


def _unit_scale(line: dict) -> float:
    """Metres per unit of the line's positions (every recording is metres)."""
    units = str(line.get("units", "m")).lower()
    if units == "mm":
        return 0.001
    if units in ("m", ""):
        return 1.0
    raise ValueError(f"unknown units {units!r} (expected 'm' or 'mm')")


def _joints_in_order(joints: Sequence[dict]) -> List[Joint]:
    by_name = {}
    for j in joints:
        by_name[str(j["name"])] = j
    missing = [n for n in JOINT_NAMES if n not in by_name]
    if missing:
        raise ValueError(f"frame line is missing joints: {', '.join(missing)}")
    return [Joint(name=n, x=float(by_name[n]["x"]), y=float(by_name[n]["y"]),
                  z=float(by_name[n]["z"]), qw=float(by_name[n]["qw"]),
                  qx=float(by_name[n]["qx"]), qy=float(by_name[n]["qy"]),
                  qz=float(by_name[n]["qz"]))
            for n in JOINT_NAMES]


def world_pose(frame_line: dict, source: str
               ) -> Tuple[np.ndarray, List[np.ndarray], str]:
    """(26 x 3 positions in metres, 26 rotation matrices, hand side).

    Two line shapes are read:

      absolute  `abs26` (positions) and `quat26` (XYZW rotations) of every
                joint in one world frame, which is what a live `LeapHand`
                holds (`line_from_leap_hand`).
      relative  `joints`: the glove convention every JSONL file uses,
                parent-relative translations and quaternions from the WRIST
                root, chained by `forward_kinematics_full`. Camera files are
                written this way too (their WRIST carries the camera-space
                pose), so both sensors' files take this path.

    `source` is checked ('glove' or 'camera') but does not change the maths:
    the two sensors share one axis convention (module docstring).
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    hand = str(frame_line.get("hand_side", ""))
    scale = _unit_scale(frame_line)
    if _has(frame_line, "quat26") and _has(frame_line, "abs26"):
        pos = np.asarray(frame_line["abs26"], dtype=float) * scale
        quats = frame_line["quat26"]
        if pos.shape != (26, 3) or len(quats) != 26:
            raise ValueError("abs26 / quat26 must hold 26 joints each")
        rots = [quat_to_mat3(*[float(v) for v in q]) for q in quats]
        return pos, rots, hand
    if not _has(frame_line, "joints"):
        raise ValueError("frame line has neither joints nor abs26 + quat26")
    frame = HandFrame(timestamp=0.0, packet_counter=0, hand_side=hand,
                      frame_id=0, status=1,
                      joints=_joints_in_order(frame_line["joints"]))
    positions, rotations = forward_kinematics_full(frame)
    pos = np.asarray(positions, dtype=float) * scale
    return pos, [np.asarray(r, dtype=float) for r in rotations], hand


# --- the table ---------------------------------------------------------------------
def table_from_world(positions_m, rotations, hand: str,
                     reference: str = "wrist") -> List[dict]:
    """The 26 rows from world poses (any world frame, metres).

    `reference` 'wrist' uses the WRIST joint's orientation as the frame's
    axes; 'palm' first gives the WRIST joint the PALM joint's orientation
    (module docstring: on the camera the WRIST quaternion is the forearm).
    """
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, "
                         f"got {reference!r}")
    hand_sign(hand)                                   # validates the side
    pos = np.asarray(positions_m, dtype=float)
    rots = [np.asarray(r, dtype=float) for r in rotations]
    if pos.shape != (26, 3) or len(rots) != 26:
        raise ValueError("need 26 positions and 26 rotations")
    if reference == "palm":
        rots = list(rots)
        rots[WRIST] = rots[PALM]
    origin = pos[WRIST]
    ref_t = rots[WRIST].T

    rows: List[dict] = []
    for i, name in enumerate(JOINT_NAMES):
        local_rot = ref_t @ rots[i]
        local_pos = ref_t @ (pos[i] - origin) * 1000.0
        q = quat_canonical(mat3_to_quat(local_rot))
        par = PARENT.get(i)
        if par is None:
            flex = abd = twist = 0.0
            bone = 0.0
            parent = ""
        else:
            flex, abd, twist = joint_angles(rots[par].T @ rots[i], hand)
            bone = float(np.linalg.norm(pos[i] - pos[par]) * 1000.0)
            parent = JOINT_NAMES[par]
        rows.append({
            "joint": name, "parent": parent,
            "x_mm": float(local_pos[0]) + 0.0,
            "y_mm": float(local_pos[1]) + 0.0,
            "z_mm": float(local_pos[2]) + 0.0,
            "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
            "axis_x": [float(v) for v in local_rot[:, 0]],
            "axis_y": [float(v) for v in local_rot[:, 1]],
            "axis_z": [float(v) for v in local_rot[:, 2]],
            "bone_len_mm": bone,
            "flex_deg": flex, "abd_deg": abd, "twist_deg": twist,
        })
    return rows


def joint_table(frame_line: dict, source: str, reference: str = "wrist",
                hand: Optional[str] = None) -> List[dict]:
    """One JSONL line (a dict) -> 26 rows, one per joint in JOINT_NAMES order.

    Each row: `joint`, `parent` ("" for the WRIST root), `x_mm y_mm z_mm`
    (position in the wrist frame), `qx qy qz qw` (orientation in the wrist
    frame, w >= 0), `axis_x axis_y axis_z` (the joint's unit axes as
    3-vectors in the wrist frame), `bone_len_mm` (distance to the parent) and
    `flex_deg abd_deg twist_deg` (parent frame to this frame, module
    docstring). `hand` overrides the line's `hand_side` for the signs.
    """
    pos, rots, side = world_pose(frame_line, source)
    return table_from_world(pos, rots, hand or side, reference)


def dof24(rows: Sequence[dict]) -> Dict[str, float]:
    """The paper's 24 named angles (degrees), read off a table's rows.

    Per finger (I, M, R, L): CMC = flexion of the metacarpal against the
    wrist frame (the paper's palm arc), MCP_fe and MCP_aa = flexion and
    abduction of the proximal phalanx, PIP, DIP. Thumb (T): TMC_fe and TMC_aa
    of the thumb metacarpal against the wrist frame, MCP_fe, IP.
    """
    by_name = {r["joint"]: r for r in rows}
    return {name: float(by_name[joint][key])
            for name, (joint, key) in DOF24_SOURCE.items()}


def wrist_palm_deg(frame_line: dict, source: str) -> float:
    """Angle in degrees between the WRIST and PALM joints' orientations.

    0 on the glove, whose palm never moves against its wrist. On the camera
    it is the forearm-to-palm angle, i.e. how far apart the two references
    of `table_from_world` are on this frame.
    """
    _pos, rots, _hand = world_pose(frame_line, source)
    rel = rots[WRIST].T @ rots[PALM]
    c = (float(np.trace(rel)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --- conversions for callers --------------------------------------------------------
def line_from_leap_hand(lh) -> dict:
    """A live `LeapHand` -> an absolute frame line (`abs26` + `quat26`, metres)."""
    return {"hand_side": lh.hand_side, "units": "m",
            "abs26": [list(map(float, p)) for p in lh.abs26],
            "quat26": [list(map(float, q)) for q in lh.quat26]}


def line_from_hand_frame(frame: HandFrame) -> dict:
    """A `HandFrame` (glove convention) -> the dict a JSONL line holds."""
    return {"hand_side": frame.hand_side,
            "joints": [{"name": j.name, "x": j.x, "y": j.y, "z": j.z,
                        "qw": j.qw, "qx": j.qx, "qy": j.qy, "qz": j.qz}
                       for j in frame.joints]}


def fmt_num(v: float, digits: int) -> str:
    """`v` to `digits` decimals, never '-0.00' (a rounded zero has no sign)."""
    text = f"{v:.{digits}f}"
    return text[1:] if text.startswith("-") and not text.strip("-0.") else text


def csv_rows(rows: Sequence[dict]) -> List[List[str]]:
    """The rows as CSV cells in `CSV_COLUMNS` order, rounded for reading.

    Millimetres and degrees to 0.01, quaternions and axis components to 1e-6.
    """
    out = []
    for r in rows:
        cells = [r["joint"], r["parent"]]
        cells += [fmt_num(r[k], 2) for k in ("x_mm", "y_mm", "z_mm")]
        cells += [fmt_num(r[k], 6) for k in ("qx", "qy", "qz", "qw")]
        for a in ("axis_x", "axis_y", "axis_z"):
            cells += [fmt_num(v, 6) for v in r[a]]
        cells += [fmt_num(r[k], 2) for k in ("bone_len_mm", "flex_deg",
                                           "abd_deg", "twist_deg")]
        out.append(cells)
    return out


def to_world(rows: Sequence[dict], origin_m, rotation) -> Tuple[np.ndarray,
                                                                 np.ndarray]:
    """Rows -> (26 x 3 positions in metres, 26 x 3 x 3 axes) placed in a world.

    The inverse of the wrist-frame step: the frame's origin goes to
    `origin_m` and its axes to the columns of `rotation`. Used to draw one
    sensor's hand at the other's wrist, e.g. the glove on the camera image.
    """
    rot = np.asarray(rotation, dtype=float)
    o = np.asarray(origin_m, dtype=float)
    pos = np.array([o + rot @ (np.array([r["x_mm"], r["y_mm"], r["z_mm"]])
                               / 1000.0) for r in rows])
    axes = np.array([rot @ np.column_stack([r["axis_x"], r["axis_y"],
                                            r["axis_z"]]) for r in rows])
    return pos, axes
