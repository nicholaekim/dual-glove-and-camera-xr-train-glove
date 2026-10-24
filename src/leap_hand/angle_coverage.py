"""Which angle of which joint the camera gives, under which name, and how.

One entry per joint and movement (flexion/extension and abduction/adduction
of every joint of the hand, plus wrist and twist): its output column(s), its
reference frame, zero, sign, method and one status out of STATUS. The table
is the one docs/angle_coverage_plan.md sets out; docs/angle_coverage.md
prints it from `coverage_markdown()`, so the two cannot drift apart.

STATUS
  calculated      read off the tracker's bone orientations by a stated
                  decomposition: the 18 Paper 3 coordinates the camera gives,
                  plus the paper-order MCP pairs.
  estimated       calculated against a calibration the dataset itself
                  supplies, with stated assumptions; kept in separate
                  `_est_<calibration>` columns, never merged with the
                  validated ones (the thumb TMC pair, leap_hand.tmc_estimate).
  unavailable     a coordinate the tracker does not measure: the cell is
                  empty, never 0 and never a Paper 3 prediction; the entry's
                  `requirement` says what recording or calibration would give
                  it.
  outside_paper3  a movement the tracker reports but Paper 3's 24-DoF model
                  does not have (thumb MCP and IP abduction, finger PIP and DIP
                  sideways components, twist, wrist): exported under a
                  `_tracker` name so it is never mistaken for a Paper 3
                  coordinate.

CONVENTIONS (xr_hand.joint_frames, both hands)
  Flexion positive toward the palm; abduction positive toward the thumb side
  (radial); twist positive with the back of the bone turning toward the
  little finger. `xyz` = R = Rx(a) Ry(b) Rz(c), flexion first
  (joint_frames.joint_angles); `paper_order` = abduction first, then flexion
  about the new x (joint_frames.paper_order_angles, the paper's D-H order).
  A parent bone frame is the parent joint's LeapC frame (x hinge, y dorsal,
  z along the bone toward the wrist); the WRIST joint's frame is LeapC's
  forearm, the PALM joint's frame the rigid hand.

`all_angles(pos, rot, hand_side)` gives every calculated and outside_paper3
column for one pose and None for the six unavailable Paper 3 coordinates:
one joint_frames.table_from_world call (the `_tracker` columns) and
paper3.frame_angles (the 24 and the 8 paper-order MCP values), so the Paper 3
numbers here are exactly the ones paper3.compare is run on. The two
`_est_openhand` columns are not in it: they need a calibration.
Pure: no files except `write_coverage_csv`, no window.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_NAMES

from . import paper3 as p3

STATUS = ("calculated", "estimated", "unavailable", "outside_paper3")
STATUS_LABELS = {"calculated": "calculated", "estimated": "estimated",
                 "unavailable": "unavailable", "outside_paper3": "outside Paper 3"}


@dataclass(frozen=True)
class Entry:
    """One joint x movement of the coverage table (module docstring)."""
    joint: str
    movement: str
    paper3_name: Optional[str]          # the validated Paper 3 name, if any
    status: str
    output_columns: Tuple[str, ...]     # empty for an unavailable entry
    reference_frame: str
    zero_position: str
    sign_convention: str
    method: str
    note: str
    requirement: Optional[str]          # what would give it (unavailable)
    dataset_column: Optional[str]       # where the dataset already has it


# --- wording (docs/angle_coverage_plan.md, "The table") ----------------------------
SIGN_FE = "flexion positive toward the palm (both hands)"
SIGN_AA = "abduction positive toward the thumb side, radial (both hands)"
SIGN_TWIST = ("positive with the back of the bone turning toward the little "
              "finger (both hands)")
ZERO_PHALANX = "bone in line with the parent"
ZERO_METACARPAL = "metacarpal along the reference frame's -z"
ZERO_OPEN_HAND = "the session's open-hand pose"
METHOD_XYZ = "xyz: R = Rx(a) Ry(b) Rz(c), flexion first (joint_frames.joint_angles)"
METHOD_MCP = ("xyz (flexion first, joint_frames.joint_angles); paper_order "
              "(abduction first then flexion, the paper's D-H order, "
              "joint_frames.paper_order_angles)")
METHOD_ESTIMATE = ("paper_order on R_rest^T (R_palm^T R_metacarpal); see "
                   "leap_hand.tmc_estimate")
METHOD_NONE = "none: not measured"
REF_ESTIMATE = "palm frame via the session's open-hand rest frame"

THUMB_TMC_REQUIREMENT = (
    "a calibration recording with the thumb in an anatomically defined "
    "neutral (ISB-style TMC frame from the trapezium and first metacarpal "
    "landmarks, Cheze et al. 2008 as cited by Paper 3) or a goniometer "
    "reading at a recorded reference pose, so the estimate's zero can be tied "
    "to the paper's; plus an independent sensor (the paper used a Cyberglove) "
    "to validate the mapping")
FINGER_CMC_REQUIREMENT = (
    "a sensor that measures each metacarpal against the carpus. LeapC fits a "
    "rigid palm; the StretchSense glove has no palm-arc sensor either. "
    "Options: optical markers on the four metacarpal heads and the wrist with "
    "a marker-based fit, or a tracker whose hand model has palm-arc joints. "
    "Without one, no recording of this setup can give it")
CMC_AA_REQUIREMENT = (
    "the same sensor as the CMC flexion (each metacarpal against the carpus); "
    "Paper 3's model has no CMC abduction, so even then it would be a "
    "tracker-only value outside the 24")
ESTIMATE_CALIBRATION = (
    "the session's open-hand frames: transitions with a movement_start_time_s, "
    "frames before movement_start_time_s - 0.1 s (leap_hand.tmc_estimate)")
TMC_NOTE = (
    "not measured as the paper defines it: the wrist-frame XYZ pair of the "
    "thumb metacarpal is about the hand's axes (the thumb's own flexion axis is "
    "about 81 deg from the palm's x), not the paper's TMC pair, and cannot be "
    "converted because the paper's thumb frame is a drawing (Figure 1, p.3); "
    "the dataset's cell is blank")
ESTIMATE_NOTE = (
    "NOT a validated Paper 3 coordinate: relative to the recorded open hand, "
    "it differs from the paper's zero by an unknown constant per coordinate "
    "(Paper 3 gives its zero only as a drawing, Figure 1 p.3); the Table 6 "
    "relation T_TMC_fe ~ 11/10 T_MCP_fe stays unavailable")
CMC_NOTE = (
    "the wrist-frame metacarpal flexion is the forearm-to-palm bend, not the "
    "palm arc; against the palm it is a constant of the tracker's rigid hand")
ONE_DOF = "the paper's {joint} has 1 DoF ({table}, p.5)"
TWIST_JOINTS: Tuple[str, ...] = tuple(n for n in JOINT_NAMES
                                      if n not in ("WRIST", "PALM"))
_DIGITS = (("T", "THUMB", "thumb"), ("I", "INDEX", "index"),
           ("M", "MIDDLE", "middle"), ("R", "RING", "ring"),
           ("L", "LITTLE", "little"))
DIGIT_NAMES = {letter: upper for letter, upper, _ in _DIGITS}

# `_tracker` column -> (row of joint_frames.table_from_world, which angle)
TRACKER_SOURCE: Dict[str, Tuple[str, str]] = {}


def _phalanx(joint: str, parent: str, letter: str, paper_joint: str,
             fe_name: str, aa_owner: str, table: str) -> List[Entry]:
    """A 1-DoF joint in the paper: flexion calculated, abduction outside."""
    ref = f"parent bone frame ({parent})"
    aa_col = f"{letter}_{paper_joint}_aa_tracker"
    TRACKER_SOURCE[aa_col] = (aa_owner, "abd_deg")
    return [
        Entry(joint, "flexion/extension", fe_name, "calculated", (fe_name,), ref,
              ZERO_PHALANX, SIGN_FE, METHOD_XYZ, "", None, fe_name),
        Entry(joint, "abduction/adduction", None, "outside_paper3", (aa_col,), ref,
              ZERO_PHALANX, SIGN_AA, "xyz second angle (joint_frames.joint_angles "
              "abd_deg)", ONE_DOF.format(joint=joint, table=table), None,
              f"{aa_owner}.abd_deg"),
    ]


def _thumb() -> List[Entry]:
    out = []
    for k, name in (("fe", "T_TMC_fe"), ("aa", "T_TMC_aa")):
        movement = "flexion/extension" if k == "fe" else "abduction/adduction"
        sign = SIGN_FE if k == "fe" else SIGN_AA
        out += [
            Entry("thumb TMC", movement, name, "unavailable", (),
                  "Paper 3's thumb frame: TMC flexion axis parallel to the MCP "
                  "and IP flexion axes (Table 3, p.5)",
                  "Paper 3's thumb zero, given only as a drawing (Figure 1, p.3)",
                  sign, METHOD_NONE, TMC_NOTE, THUMB_TMC_REQUIREMENT, name),
            Entry("thumb TMC", movement, None, "estimated",
                  (f"{name}_est_openhand",), REF_ESTIMATE, ZERO_OPEN_HAND, sign,
                  METHOD_ESTIMATE, ESTIMATE_NOTE, ESTIMATE_CALIBRATION, None),
        ]
    out += _phalanx("thumb MCP", "THUMB_METACARPAL", "T", "MCP", "T_MCP_fe",
                    "THUMB_PROXIMAL", "Table 3")
    out += _phalanx("thumb IP", "THUMB_PROXIMAL", "T", "IP", "T_IP",
                    "THUMB_DISTAL", "Table 3")
    return out


def _finger(letter: str, upper: str, lower: str) -> List[Entry]:
    out = [
        Entry(f"{lower} CMC", "flexion/extension (palm arc)", f"{letter}_CMC",
              "unavailable", (), "the carpus (Paper 3's palm arc); the tracker's "
              "palm is one rigid frame", ZERO_METACARPAL, SIGN_FE, METHOD_NONE,
              CMC_NOTE, FINGER_CMC_REQUIREMENT, f"{letter}_CMC"),
        Entry(f"{lower} CMC", "abduction/adduction", None, "unavailable", (),
              "the carpus; the tracker's palm is one rigid frame",
              ZERO_METACARPAL, SIGN_AA, METHOD_NONE,
              "outside Paper 3's model and not measured: rigid palm",
              CMC_AA_REQUIREMENT, None),
    ]
    ref = f"parent bone frame ({upper}_METACARPAL)"
    for k, movement, sign in (("fe", "flexion/extension", SIGN_FE),
                              ("aa", "abduction/adduction", SIGN_AA)):
        name = f"{letter}_MCP_{k}"
        out.append(Entry(
            f"{lower} MCP", movement, name, "calculated",
            (name, f"{name}_paper_order"), ref, ZERO_PHALANX, sign, METHOD_MCP,
            "two readings of one bone direction: sin(fe_paper) = sin(fe) cos(aa), "
            "tan(aa_paper) = tan(aa) / cos(fe); paper3.compare takes the "
            "paper-order pair as its primary value", None,
            f"{name}; {name}_paper_order"))
    out += _phalanx(f"{lower} PIP", f"{upper}_PROXIMAL", letter, "PIP",
                    f"{letter}_PIP", f"{upper}_INTERMEDIATE", "Table 2")
    out += _phalanx(f"{lower} DIP", f"{upper}_INTERMEDIATE", letter, "DIP",
                    f"{letter}_DIP", f"{upper}_DISTAL", "Table 2")
    return out


def _wrist_and_twist() -> List[Entry]:
    TRACKER_SOURCE["WRIST_fe_tracker"] = ("PALM", "flex_deg")
    TRACKER_SOURCE["WRIST_dev_tracker"] = ("PALM", "abd_deg")
    for joint in TWIST_JOINTS:
        TRACKER_SOURCE[f"{joint}_twist_tracker"] = (joint, "twist_deg")
    ref = "WRIST frame (LeapC's forearm) to the PALM frame"
    zero = "PALM axes aligned with the WRIST (forearm) axes"
    note = ("not one of Paper 3's 24; the forearm is LeapC's estimate "
            "(arm.rotation), not a tracked bone")
    return [
        Entry("wrist", "flexion/extension", None, "outside_paper3",
              ("WRIST_fe_tracker",), ref, zero,
              "flexion positive toward the palm side of the forearm (both hands)",
              "xyz, from the PALM row of joint_frames.table_from_world (flex_deg)",
              note, None, "PALM.flex_deg"),
        Entry("wrist", "radial/ulnar deviation", None, "outside_paper3",
              ("WRIST_dev_tracker",), ref, zero,
              "positive toward the thumb side, radial deviation (both hands)",
              "xyz, from the PALM row of joint_frames.table_from_world (abd_deg)",
              note, None, "PALM.abd_deg"),
        Entry("every bone (the 24 joints other than WRIST and PALM)",
              "twist about the bone axis", None, "outside_paper3",
              tuple(f"{j}_twist_tracker" for j in TWIST_JOINTS),
              "parent bone frame", "no turn about the bone axis against the "
              "parent (third xyz angle 0)", SIGN_TWIST,
              "xyz third angle (joint_frames.joint_angles twist_deg)",
              "Paper 3's bones have no rotation about their own axis; on "
              "LeapC's pure-swing finger joints this twist is -abd tan(flex/2), "
              "the order's, not the finger's (joint_frames, WHICH ORDER)",
              None, "<JOINT>.twist_deg"),
    ]


COVERAGE: List[Entry] = (_thumb()
                         + [e for letter, upper, lower in _DIGITS[1:]
                            for e in _finger(letter, upper, lower)]
                         + _wrist_and_twist())

# Every column all_angles returns, in table order (the six unavailable Paper 3
# names included, always None); the estimate's two; the six.
ALL_ANGLE_COLUMNS: List[str] = []
for _e in COVERAGE:
    if _e.status in ("calculated", "outside_paper3"):
        ALL_ANGLE_COLUMNS += list(_e.output_columns)
    elif _e.status == "unavailable" and _e.paper3_name in p3.UNAVAILABLE:
        ALL_ANGLE_COLUMNS.append(_e.paper3_name)
ESTIMATE_COLUMNS: Tuple[str, ...] = tuple(
    c for e in COVERAGE if e.status == "estimated" for c in e.output_columns)
UNAVAILABLE_COLUMNS: Tuple[str, ...] = tuple(p3.UNAVAILABLE)

_STATUS_OF: Dict[str, str] = {c: "unavailable" for c in UNAVAILABLE_COLUMNS}
for _e in COVERAGE:
    _STATUS_OF.update({c: _e.status for c in _e.output_columns})

assert len(COVERAGE) == 43
assert len(ALL_ANGLE_COLUMNS) == len(set(ALL_ANGLE_COLUMNS)) == 68
assert set(p3.ANGLE_NAMES) <= set(ALL_ANGLE_COLUMNS)
assert set(ALL_ANGLE_COLUMNS) == set(p3.ANGLE_NAMES) | set(TRACKER_SOURCE)
assert ESTIMATE_COLUMNS == ("T_TMC_fe_est_openhand", "T_TMC_aa_est_openhand")
assert all(e.status in STATUS for e in COVERAGE)
assert all(e.requirement for e in COVERAGE if e.status == "unavailable")


def status_of(column: str) -> str:
    """The status (one of STATUS) of any output column of the table, or of
    one of the six unavailable Paper 3 names."""
    try:
        return _STATUS_OF[column]
    except KeyError:
        raise KeyError(f"{column!r} is not a column of the coverage table") from None


# --- the angles of one pose -----------------------------------------------------------
def all_angles(pos, rot, hand_side: str) -> Dict[str, Optional[float]]:
    """ALL_ANGLE_COLUMNS -> degrees for one pose (pos (26, 3) m, rot (26, 3, 3),
    any world frame); None for the six UNAVAILABLE_COLUMNS, never 0.

    The Paper 3 names and the paper-order MCP pairs are paper3.frame_angles;
    the `_tracker` columns are rows of one joint_frames.table_from_world
    (wrist reference): abd_deg of the thumb proximal and distal and of every
    finger's intermediate and distal, the PALM row's flex_deg and abd_deg
    (wrist), and the twist_deg of the 24 joints other than WRIST and PALM.
    """
    out = dict(p3.frame_angles(pos, rot, hand_side))      # checks the shapes
    rows = {r["joint"]: r for r in jf.table_from_world(
        np.asarray(pos, dtype=float), np.asarray(rot, dtype=float), hand_side,
        "wrist")}
    for column, (joint, key) in TRACKER_SOURCE.items():
        out[column] = float(rows[joint][key])
    return {c: out[c] for c in ALL_ANGLE_COLUMNS}


# --- the table as CSV and Markdown -----------------------------------------------------
COVERAGE_CSV_COLUMNS = ["joint", "movement", "paper3_name", "status",
                        "output_columns", "reference_frame", "zero_position",
                        "sign_convention", "method", "note", "requirement",
                        "dataset_column"]


def coverage_rows() -> List[dict]:
    """One dict per entry, COVERAGE_CSV_COLUMNS as keys, output columns joined
    by '; ', None as ''."""
    rows = []
    for e in COVERAGE:
        row = {k: getattr(e, k) for k in COVERAGE_CSV_COLUMNS}
        row["output_columns"] = "; ".join(e.output_columns)
        rows.append({k: "" if v is None else v for k, v in row.items()})
    return rows


def write_coverage_csv(path) -> Path:
    """coverage_rows() as a CSV (utf-8, header first); returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COVERAGE_CSV_COLUMNS)
        w.writeheader()
        w.writerows(coverage_rows())
    return path


def coverage_markdown() -> str:
    """coverage_rows() as a Markdown table, the one docs/angle_coverage.md
    holds (columns in backticks, '|' escaped)."""
    head = ["Joint", "Movement", "Paper 3 name", "Status", "Output column(s)",
            "Reference frame", "Zero", "Sign", "Method", "Note", "Requirement",
            "Dataset column"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in coverage_rows():
        cells = []
        for k in COVERAGE_CSV_COLUMNS:
            v = row[k].replace("|", "\\|")
            if k in ("paper3_name", "dataset_column") and v:
                v = f"`{v}`"
            elif k == "output_columns" and v:
                v = ", ".join(f"`{c}`" for c in v.split("; "))
            elif k == "status":
                v = STATUS_LABELS[v]
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
