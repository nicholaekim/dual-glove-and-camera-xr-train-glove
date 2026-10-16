"""Which joint each row of a joint-frame table is, and which of the paper's
angles is read from it: the key a reader needs next to the numbers.

A joint-frame table (`xr_hand.joint_frames`) has 26 rows in OpenXR order
(`xr_hand.joints.JOINT_NAMES`). A row's flex_deg, abd_deg and twist_deg are
the rotation from its PARENT row's frame to its own, and a joint's frame is
the frame of the bone that STARTS at that joint (`leap_hand.to_openxr`,
`JOINT_SOURCES`), so a row carries the name of a bone, not of a joint:
INDEX_PROXIMAL is the index MCP, the joint at which the proximal phalanx
starts. `NAMING_RULE` says this in two sentences; `key_rows` spells it out
row by row:

  number                  1..26, the row's place in the table
  row                     the OpenXR name, as in the tables
  anatomical_joint        the joint the row's angles describe
  bone_before, bone_after the two bones whose relative turn the angles are
                          (bone_before is the parent row's bone, read off
                          `xr_hand.kinematics.PARENT`)
  flex_deg_means,         what a positive value of each column means on
  abd_deg_means           that row
  paper_angle_flex,       the name of Cobos et al. 2009 (Table 1) read from
  paper_angle_abd         that column, built from `joint_frames.DOF24_SOURCE`
  given                   "yes" / "no: not measured by this tracker" (the six
                          of `joint_frames.CAMERA_NOT_GIVEN`, why in
                          `NOT_GIVEN_WHY`) / "not one of the 24"
  landmark_in_text_files  the label of the joint's point in the 21-landmark
                          keypoint text files (`scripts/glove/export_prof_
                          format.py` LANDMARKS writes them, `cam_hand.
                          prof_format` reads them); the four finger CMC are
                          not in those files

SIGNS. flex_deg > 0 turns the child bone toward its parent's -y, the palm
side; abd_deg > 0 on a left hand toward its parent's +x (`joint_frames`,
`rotation_from_angles`). For the hand's bones +x is the thumb side (on the
69 joint-frame tables of the session 20261001_101621_left the palm's +x
points from the little finger's MCP to the index's, dot product 0.95 to
0.98). The thumb's bones are turned about their length: on the same 69
tables the +x of THUMB_METACARPAL and THUMB_PROXIMAL points palmward (dot
product with the palm's -y 0.70 to 0.91 and 0.69 to 0.94), their +y (the
nail) toward the thumb side (dot product with the palm's +x 0.65 to 0.95
and 0.69 to 0.99). So abd_deg on the thumb's MCP and IP rows is a lean
toward the palm's face, not toward the thumb side; on the THUMB_METACARPAL
row, whose parent is the forearm, it is toward the thumb side like the
fingers'. (Measured with a scratch script on 2026-10-06, the session's
joint_frames/*.csv, axis columns.)

`write_key_csv` and `write_key_pdf` write the key for a hand-in folder;
`readme_lines` gives it as README lines of at most `WIDTH` characters. The
PDF's hand is a drawing (joint positions set by hand, `HAND_MM`), not a
measured hand.
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .joint_frames import CAMERA_NOT_GIVEN, DOF24_SOURCE, FINGERS, PAPER_JOINT
from .joints import BONES, JOINT_NAMES
from .kinematics import PARENT

KEY_COLUMNS = ["number", "row", "anatomical_joint", "bone_before",
               "bone_after", "flex_deg_means", "abd_deg_means",
               "paper_angle_flex", "paper_angle_abd", "given",
               "landmark_in_text_files"]

WIDTH = 96                       # README page width, characters
GIVEN_YES = "yes"
GIVEN_NO = "no: not measured by this tracker"
NOT_OF_24 = "not one of the 24"
NOT_IN_TEXT = "not in the text files"
FOREARM = "forearm (the WRIST joint's frame)"
PDF_TITLE = "Joint key of the joint-frame tables"

NAMING_RULE = (
    "A row is named after the bone that starts at its joint, and its angles "
    "are the turn from the bone before it (the parent row's) to that bone. "
    "So on the index, middle, ring and little finger *_METACARPAL = CMC, "
    "*_PROXIMAL = MCP, *_INTERMEDIATE = PIP, *_DISTAL = DIP; on the thumb "
    "THUMB_METACARPAL = TMC, THUMB_PROXIMAL = MCP, THUMB_DISTAL = IP; "
    "PALM = the wrist (the palm against the forearm); WRIST = the root (the "
    "forearm's frame, no angles); *_TIP = the fingertip, not a joint "
    "(angles 0).")

# Why the six of CAMERA_NOT_GIVEN are not given, by the paper's joint
# (joint_frames, "WHERE THIS IS NOT THE PAPER'S MODEL").
NOT_GIVEN_WHY: Dict[str, str] = {
    "CMC": ("the paper's CMC is the arch of the palm; this tracker's palm "
            "is rigid, so the value (the metacarpal seen from the forearm) "
            "is the bend of the wrist, not the arch."),
    "TMC": ("the paper's TMC pair is about the thumb's own axes, from a "
            "thumb frame it gives only as a drawing; the tracker's pair is "
            "about the hand's axes, not the thumb's, and cannot be "
            "converted."),
}

ABBREVIATION = {
    "CMC": "carpometacarpal",
    "TMC": "trapeziometacarpal, the thumb's CMC",
    "MCP": "metacarpophalangeal",
    "PIP": "proximal interphalangeal",
    "DIP": "distal interphalangeal",
    "IP": "interphalangeal",
}

DIGIT = {"THUMB": "thumb", "INDEX": "index", "MIDDLE": "middle",
         "RING": "ring", "LITTLE": "little"}

# The keypoint text files' labels: "Wrist: (x, y, z)", then "0 (Palm): ..."
# and four points per digit, "1 (THUMB)" ... "20 (Pinky)"
# (scripts/glove/export_prof_format.py LANDMARKS).
WRIST_LABEL = "Wrist"
TEXT_LABEL = {"PALM": "Palm", "THUMB": "THUMB", "INDEX": "Index",
              "MIDDLE": "Middle", "RING": "Ring", "LITTLE": "Pinky"}


def _digit(name: str) -> str:
    return name.split("_", 1)[0]


def _kind(name: str) -> str:
    return name.split("_", 1)[1] if "_" in name else name


def _in_text_files(name: str) -> bool:
    """The 21 landmarks are every joint but the WRIST and the four finger
    metacarpal bases, in JOINT_NAMES order."""
    return name != "WRIST" and not (_kind(name) == "METACARPAL"
                                    and _digit(name) != "THUMB")


def landmarks() -> Dict[str, str]:
    """Row name -> its label in the keypoint text files."""
    out: Dict[str, str] = {}
    k = 0
    for name in JOINT_NAMES:
        if name == "WRIST":
            out[name] = WRIST_LABEL
        elif _in_text_files(name):
            out[name] = f"{k} ({TEXT_LABEL[_digit(name)]})"
            k += 1
        else:
            out[name] = NOT_IN_TEXT
    assert k == 21
    return out


def _bone(name: str) -> str:
    """The bone whose frame the row `name` carries."""
    if name == "WRIST":
        return "forearm"
    if name == "PALM":
        return "palm"
    digit, kind = DIGIT[_digit(name)], _kind(name)
    if kind == "METACARPAL":
        return f"{digit} metacarpal"
    if kind == "TIP":
        return f"{digit} distal phalanx"
    return f"{digit} {kind.lower()} phalanx"


def paper_names() -> Dict[str, Dict[str, str]]:
    """Row -> {"flex_deg": name, "abd_deg": name} from DOF24_SOURCE."""
    out: Dict[str, Dict[str, str]] = {n: {} for n in JOINT_NAMES}
    for paper, (row, column) in DOF24_SOURCE.items():
        if column not in ("flex_deg", "abd_deg"):
            raise ValueError(f"{paper}: column {column!r} has no key column")
        if column in out[row]:
            raise ValueError(f"{row} {column} holds two paper angles")
        out[row][column] = paper
    return out


def _meanings(name: str) -> Tuple[str, str, str, str]:
    """(anatomical joint, short joint, what flex_deg means, what abd_deg
    means) for the row `name`."""
    if name == "WRIST":
        none = "none: the root has no parent, its angles are 0"
        return ("root: the forearm at the wrist (no parent, no angles)",
                "root", none, none)
    if name == "PALM":
        return ("wrist (the palm against the forearm)", "wrist",
                "bending of the palm toward the palm side of the forearm "
                "(wrist flexion)",
                "tilt of the palm toward the thumb side (radial deviation "
                "of the wrist)")
    digit, kind = DIGIT[_digit(name)], _kind(name)
    if kind == "TIP":
        zero = ("0 by construction: the tip repeats the frame of the "
                f"{digit} distal phalanx")
        return (f"{digit} fingertip (not a joint; angles 0 by construction)",
                f"{digit} tip", zero, zero)
    abbr = PAPER_JOINT[name]
    anatomical = f"{digit} {abbr} ({ABBREVIATION[abbr]})"
    short = f"{digit} {abbr}"
    if kind == "METACARPAL":
        note = (" (includes the bend of the wrist)" if digit == "thumb" else
                " (on this tracker it changes with the bend of the wrist "
                "and nothing else)")
        flex = (f"where the {digit} metacarpal points toward the palm side, "
                "seen from the forearm" + note)
        abd = (f"where the {digit} metacarpal points toward the thumb side, "
               "seen from the forearm" + note)
        return anatomical, short, flex, abd
    if digit == "thumb":
        phalanx = f"the thumb's {kind.lower()} phalanx"
        return (anatomical, short,
                f"bending of {phalanx} toward the palm, about the thumb's "
                "own flexion axis",
                f"sideways lean of {phalanx}, about the thumb's nail axis, "
                "toward the palm's face")
    if kind == "PROXIMAL":
        return (anatomical, short,
                "bending of the proximal phalanx toward the palm",
                "spread of the finger toward the thumb side")
    phalanx = f"the {kind.lower()} phalanx"
    return (anatomical, short,
            f"bending of {phalanx} toward the palm",
            f"sideways lean of {phalanx} toward the thumb side (the paper's "
            f"{abbr} is a hinge)")


def short_joint(name: str) -> str:
    """The joint of the row `name` in a few characters, for tables."""
    return _meanings(name)[1]


def key_rows() -> List[dict]:
    """The 26 rows of the key, in JOINT_NAMES order, keys KEY_COLUMNS."""
    papers = paper_names()
    marks = landmarks()
    rows: List[dict] = []
    for i, name in enumerate(JOINT_NAMES):
        anatomical, _short, flex, abd = _meanings(name)
        par = PARENT.get(i)
        if par is None:
            before, after = ("none (the root)",
                             "forearm (the WRIST joint's orientation is the "
                             "forearm bone's)")
        else:
            parent = JOINT_NAMES[par]
            before = FOREARM if parent == "WRIST" else _bone(parent)
            after = _bone(name)
            if name == "PALM":
                after = "palm (the tracker's palm is one rigid piece)"
            elif _kind(name) == "TIP":
                after = ("none: the tip repeats the frame of the "
                         f"{_bone(name)}")
        p_flex = papers[name].get("flex_deg", "")
        p_abd = papers[name].get("abd_deg", "")
        named = [p for p in (p_flex, p_abd) if p]
        not_given = [p for p in named if p in CAMERA_NOT_GIVEN]
        if not_given and len(not_given) != len(named):
            raise ValueError(f"{name}: given and not given on one row")
        given = (GIVEN_NO if not_given else GIVEN_YES if named
                 else NOT_OF_24)
        rows.append({
            "number": i + 1, "row": name, "anatomical_joint": anatomical,
            "bone_before": before, "bone_after": after,
            "flex_deg_means": flex, "abd_deg_means": abd,
            "paper_angle_flex": p_flex, "paper_angle_abd": p_abd,
            "given": given, "landmark_in_text_files": marks[name],
        })
    return rows


def _paper_cell(row: dict) -> str:
    flex, abd = row["paper_angle_flex"], row["paper_angle_abd"]
    if not flex and not abd:
        return "-"
    return f"{flex or '-'} / {abd or '-'}"


def _given_cell(row: dict) -> str:
    return {GIVEN_YES: "yes", GIVEN_NO: "no"}.get(row["given"], "-")


def _wrap(text: str, indent: str = "  ") -> List[str]:
    return textwrap.wrap(text, width=WIDTH, initial_indent=indent,
                         subsequent_indent=indent, break_on_hyphens=False,
                         break_long_words=False)


LEGEND_JOINT = (
    "joint: wrist = the palm against the forearm; root = the forearm's own "
    "frame, no parent and no angles; tip = the fingertip, not a joint "
    "(angles 0 by construction). " + "; ".join(
        f"{k} {v}" for k, v in ABBREVIATION.items()) + ".")
LEGEND_GIVEN = (
    "paper (flex / abd): the angle of Cobos et al. 2009 (Table 1) read from "
    "the row's flex_deg / abd_deg, a dash where none is. given: yes = "
    "given under the paper's name; no = not measured by this tracker and "
    "not given; a dash = not one of the paper's 24 (nor is twist_deg, on "
    "any row). Not given, CMC: " + NOT_GIVEN_WHY["CMC"] + " TMC: "
    + NOT_GIVEN_WHY["TMC"])
LEGEND_SIGNS = (
    "Signs: flex_deg > 0 = bent toward the palm; abd_deg > 0 = toward the "
    "thumb side, except on the thumb's MCP and IP rows (THUMB_PROXIMAL, "
    "THUMB_DISTAL), where it is toward the palm's face.")
LEGEND_TEXT = (
    "landmark: the label of the joint's point in the keypoint text files "
    "(the line \"Wrist: (x, y, z)\" and \"0 (Palm)\" to \"20 (Pinky)\"); the "
    "four finger CMC are not in the text files.")


def landmark_summary() -> str:
    """Which rows the keypoint text files' points are, in one sentence."""
    groups: List[Tuple[str, List[int], List[int]]] = []
    for r in key_rows():
        mark = r["landmark_in_text_files"]
        if mark == NOT_IN_TEXT:
            continue
        if mark == WRIST_LABEL:
            groups.append((WRIST_LABEL, [], [r["number"]]))
            continue
        idx, label = mark.split(" ", 1)
        if groups and groups[-1][0] == label:
            groups[-1][1].append(int(idx))
            groups[-1][2].append(r["number"])
        else:
            groups.append((label, [int(idx)], [r["number"]]))
    parts = []
    for label, idx, nums in groups:
        if not idx:
            parts.append(f"{label} = row {nums[0]}")
        elif len(idx) == 1:
            parts.append(f"{idx[0]} {label} = row {nums[0]}")
        else:
            parts.append(f"{idx[0]} to {idx[-1]} {label} = rows {nums[0]} "
                         f"to {nums[-1]}")
    missing = [str(r["number"]) for r in key_rows()
               if r["landmark_in_text_files"] == NOT_IN_TEXT]
    return ("Keypoint text files: " + "; ".join(parts) + ". Rows "
            + ", ".join(missing[:-1]) + " and " + missing[-1]
            + " (the four finger CMC) are not in the text files.")


def readme_lines() -> List[str]:
    """The key as README lines (each at most WIDTH characters): the naming
    rule, the 26-row table, and what its columns' short values mean."""
    rows = key_rows()
    w_row = max(len(r["row"]) for r in rows)
    w_joint = max(len("joint"), max(len(short_joint(r["row"]))
                                    for r in rows))
    head_paper = "paper (flex / abd)"
    w_paper = max(len(head_paper), max(len(_paper_cell(r)) for r in rows))
    w_given = len("given")

    def line(nr: str, row: str, joint: str, paper: str, given: str,
             mark: str) -> str:
        return (f"  {nr:>2}  {row:<{w_row}}  {joint:<{w_joint}}  "
                f"{paper:<{w_paper}}  {given:<{w_given}}  {mark}").rstrip()

    lines = _wrap(NAMING_RULE) + [""]
    lines.append(line("nr", "row", "joint", head_paper, "given",
                      "landmark in the text files"))
    for r in rows:
        lines.append(line(str(r["number"]), r["row"], short_joint(r["row"]),
                          _paper_cell(r), _given_cell(r),
                          r["landmark_in_text_files"]))
    lines.append("")
    for text in (LEGEND_JOINT, LEGEND_GIVEN, LEGEND_SIGNS, LEGEND_TEXT):
        lines += _wrap(text)
    too_wide = [x for x in lines if len(x) > WIDTH]
    if too_wide:
        raise ValueError(f"README line wider than {WIDTH}: {too_wide[0]!r}")
    return lines


def write_key_csv(dest) -> None:
    """KEY_COLUMNS and the 26 rows, UTF-8; the same bytes on every call."""
    with open(Path(dest), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        w.writeheader()
        for r in key_rows():
            w.writerow(r)


# --- the drawing ----------------------------------------------------------------
# A flat LEFT hand seen from the palm (thumb on the left of the drawing, as
# a left palm facing the viewer shows it), fingers up, millimetres. Set by
# hand to show which point is which, not measured. The same handedness as
# the palm side view of the joint-frame pages (`scripts/joint_frames_view.py`
# `_view_points`: right = -x, up = -z, and a left hand's +x points to its
# thumb), checked in tests/test_joint_key.py.
HAND_MM: Dict[str, Tuple[float, float]] = {
    "WRIST": (0.0, 0.0), "PALM": (1.0, 58.0),
    "THUMB_METACARPAL": (-22.0, 20.0), "THUMB_PROXIMAL": (-45.0, 52.0),
    "THUMB_DISTAL": (-57.0, 78.0), "THUMB_TIP": (-64.0, 98.0),
    "INDEX_METACARPAL": (-17.0, 30.0), "INDEX_PROXIMAL": (-27.0, 96.0),
    "INDEX_INTERMEDIATE": (-31.0, 134.0), "INDEX_DISTAL": (-33.0, 158.0),
    "INDEX_TIP": (-34.0, 178.0),
    "MIDDLE_METACARPAL": (-5.0, 32.0), "MIDDLE_PROXIMAL": (-8.0, 100.0),
    "MIDDLE_INTERMEDIATE": (-9.0, 143.0), "MIDDLE_DISTAL": (-10.0, 170.0),
    "MIDDLE_TIP": (-10.0, 192.0),
    "RING_METACARPAL": (7.0, 31.0), "RING_PROXIMAL": (11.0, 96.0),
    "RING_INTERMEDIATE": (14.0, 135.0), "RING_DISTAL": (16.0, 161.0),
    "RING_TIP": (17.0, 181.0),
    "LITTLE_METACARPAL": (17.0, 27.0), "LITTLE_PROXIMAL": (29.0, 86.0),
    "LITTLE_INTERMEDIATE": (35.0, 116.0), "LITTLE_DISTAL": (39.0, 136.0),
    "LITTLE_TIP": (41.0, 153.0),
}
FOREARM_MM = [(-22.0, 0.0), (22.0, 0.0), (24.0, -42.0), (-24.0, -42.0)]
PALM_OUTLINE_MM = [(-22.0, 0.0), (-28.0, 14.0), (-38.0, 32.0), (-47.0, 47.0),
                   (-44.0, 57.0), (-34.0, 67.0), (-28.5, 82.0),
                   (-28.5, 98.0), (-25.0, 103.0), (-8.0, 106.0),
                   (12.0, 105.0), (27.0, 95.0), (29.6, 89.0), (30.6, 80.0),
                   (30.0, 30.0), (22.0, 0.0)]
# Which side of its dot a joint's number plate sits on: thumb and index to
# the left, middle to the left (between index and middle), ring and little
# to the right, the palm above its dot, the wrist below.
PLATE_SIDE: Dict[str, str] = {"PALM": "U", "WRIST": "D"}
for _f, _side in (("THUMB", "L"), ("INDEX", "L"), ("MIDDLE", "L"),
                  ("RING", "R"), ("LITTLE", "R")):
    for _n in JOINT_NAMES:
        if _n.startswith(_f + "_"):
            PLATE_SIDE[_n] = _side

DOT_R = 2.2                  # pt
PLATE_FONT = ("Helvetica-Bold", 6.5)
PLATE_H = 8.0
PLATE_GAP = 3.5              # pt between a dot's edge and its plate
LETTER_FONT = ("Helvetica-Bold", 9.0)


def _rect_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """Two (x0, y0, x1, y1) rectangles share area."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _segment_hits_rect(p: Sequence[float], q: Sequence[float],
                       r: Sequence[float]) -> bool:
    """The segment p-q passes through the inside of the rectangle r
    (Liang-Barsky clipping)."""
    t0, t1 = 0.0, 1.0
    dx, dy = q[0] - p[0], q[1] - p[1]
    for den, num in ((-dx, p[0] - r[0]), (dx, r[2] - p[0]),
                     (-dy, p[1] - r[1]), (dy, r[3] - p[1])):
        if den == 0.0:
            if num < 0.0:
                return False
            continue
        t = num / den
        if den < 0.0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return t1 - t0 > 1e-9


def hand_layout(x0: float, y0: float, scale: float) -> dict:
    """The drawing in page points: the hand's origin (the WRIST) at
    (x0, y0), `scale` points per millimetre. Returns {"dots": {row: (x, y)},
    "plates": {row: (x0, y0, x1, y1)}, "letters": {letter: (x, y, w)},
    "bones": [((x, y), (x, y), heavy)], "forearm": [...], "palm": [...]}."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    def at(p: Tuple[float, float]) -> Tuple[float, float]:
        return (x0 + scale * p[0], y0 + scale * p[1])

    dots = {n: at(HAND_MM[n]) for n in JOINT_NAMES}
    plates: Dict[str, Tuple[float, float, float, float]] = {}
    for i, n in enumerate(JOINT_NAMES):
        w = stringWidth(str(i + 1), *PLATE_FONT) + 3.0
        x, y = dots[n]
        side = PLATE_SIDE[n]
        if side == "L":
            cx, cy = x - DOT_R - PLATE_GAP - w / 2, y
        elif side == "R":
            cx, cy = x + DOT_R + PLATE_GAP + w / 2, y
        elif side == "U":
            cx, cy = x, y + DOT_R + PLATE_GAP + PLATE_H / 2
        else:
            cx, cy = x, y - DOT_R - PLATE_GAP - PLATE_H / 2
        plates[n] = (cx - w / 2, cy - PLATE_H / 2, cx + w / 2,
                     cy + PLATE_H / 2)
    letters = {}
    for letter, finger in FINGERS:
        x, y = dots[f"{finger}_TIP"]
        letters[letter] = (x, y + DOT_R + 4.0,
                           stringWidth(letter, *LETTER_FONT))
    bones = []
    for par, child in BONES:
        heavy = (JOINT_NAMES[par] != "WRIST")
        bones.append((dots[JOINT_NAMES[par]], dots[JOINT_NAMES[child]],
                      heavy))
    return {"dots": dots, "plates": plates, "letters": letters,
            "bones": bones, "forearm": [at(p) for p in FOREARM_MM],
            "palm": [at(p) for p in PALM_OUTLINE_MM]}


def layout_clashes(layout: dict) -> List[str]:
    """Everything that would make a number hard to read: two plates sharing
    area, a plate over a dot, a digit letter, a bone line or the outline of
    the palm or the forearm."""
    out = []
    plates, dots = layout["plates"], layout["dots"]
    names = list(plates)
    outlines = []
    for poly in (layout["palm"], layout["forearm"]):
        outlines += [(poly[k], poly[(k + 1) % len(poly)])
                     for k in range(len(poly))]
    for a in names:
        for p, q in outlines:
            if _segment_hits_rect(p, q, plates[a]):
                out.append(f"plate {a} lies on an outline")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _rect_overlap(plates[a], plates[b]):
                out.append(f"plate {a} overlaps plate {b}")
        for n, (x, y) in dots.items():
            if _rect_overlap(plates[a], (x - DOT_R, y - DOT_R, x + DOT_R,
                                         y + DOT_R)):
                out.append(f"plate {a} covers the dot of {n}")
        for letter, (x, y, w) in layout["letters"].items():
            if _rect_overlap(plates[a], (x - w / 2, y, x + w / 2, y + 7.0)):
                out.append(f"plate {a} overlaps the letter {letter}")
        for p, q, _heavy in layout["bones"]:
            if _segment_hits_rect(p, q, plates[a]):
                out.append(f"plate {a} lies on a bone line")
    return out


def _wrap_pt(text: str, font: str, size: float, width: float) -> List[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    lines: List[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}" if cur else word
        if cur and stringWidth(trial, font, size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def write_key_pdf(dest, footer: str) -> None:
    """One A4 page: a flat left hand with its 26 joints numbered as the rows
    of the table, and the table beside it. The same bytes on every call
    (reportlab's invariant mode, a fixed title, no dates)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    page_w, page_h = A4
    margin = 28.0
    rows = key_rows()
    c = canvas.Canvas(str(Path(dest)), pagesize=A4, invariant=1)
    c.setTitle(PDF_TITLE)
    c.setAuthor("")
    c.setSubject("")

    # title and the naming rule across the page
    y = page_h - margin - 12
    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Which joint each row of a joint-frame table "
                            "is, and which of the paper's angles it gives")
    y -= 14
    c.setFont("Helvetica", 8)
    for line in _wrap_pt(NAMING_RULE, "Helvetica", 8, page_w - 2 * margin):
        c.drawString(margin, y, line)
        y -= 10
    top = y - 8

    # the table's columns, measured
    size, lead = 7.0, 10.6
    cols = [("nr", lambda r: str(r["number"])),
            ("row", lambda r: r["row"]),
            ("joint", lambda r: short_joint(r["row"])),
            ("paper angle (flex / abd)", _paper_cell),
            ("given", _given_cell)]
    widths = []
    for head, cell in cols:
        widths.append(max([stringWidth(head, "Helvetica-Bold", size)]
                          + [stringWidth(cell(r), "Helvetica", size)
                             for r in rows]))
    gap = 7.0
    table_w = sum(widths) + gap * (len(widths) - 1)
    tx = page_w - margin - table_w
    left_w = tx - margin - 16.0

    # --- left: the hand
    c.setFont("Helvetica-Bold", 8)
    c.drawString(margin, top, "A LEFT hand seen from the palm (thumb on "
                              "the left), fingers up")
    c.setFont("Helvetica", 7)
    sub = _wrap_pt("As you see your own left palm, and as the palm side view "
                   "of the joint-frame pages draws a left hand. A drawing to "
                   "show which point is which: positions set by hand, not "
                   "measured.", "Helvetica", 7, left_w)
    for k, line in enumerate(sub):
        c.drawString(margin, top - 9 * (k + 1), line)
    head = 12.0 + 9.0 * len(sub)                 # heading and subtitle
    xs = [p[0] for p in HAND_MM.values()]
    ys = [p[1] for p in list(HAND_MM.values()) + FOREARM_MM]
    pad = 22.0                                   # room for the plates
    scale = min((left_w - 2 * pad) / (max(xs) - min(xs)),
                (top - head - 150.0) / (max(ys) - min(ys) + 8.0))
    x0 = margin + pad - scale * min(xs)
    y0 = top - head - 12.0 - scale * max(ys)
    lay = hand_layout(x0, y0, scale)
    clashes = layout_clashes(lay)
    if clashes:
        raise ValueError("joint key drawing: " + "; ".join(clashes))

    c.setFillGray(0.90)
    c.setStrokeGray(0.75)
    c.setLineWidth(0.6)
    path = c.beginPath()
    path.moveTo(*lay["forearm"][0])
    for p in lay["forearm"][1:]:
        path.lineTo(*p)
    path.close()
    c.drawPath(path, stroke=1, fill=1)
    fx = sum(p[0] for p in lay["forearm"]) / 4
    fy = sum(p[1] for p in lay["forearm"]) / 4
    c.setFillGray(0.35)
    c.setFont("Helvetica", 7)
    c.drawCentredString(fx, fy - 6, "forearm")
    c.setFillGray(0.95)
    c.setStrokeGray(0.8)
    path = c.beginPath()
    path.moveTo(*lay["palm"][0])
    for p in lay["palm"][1:]:
        path.lineTo(*p)
    path.close()
    c.drawPath(path, stroke=1, fill=1)

    for p, q, heavy in lay["bones"]:
        if heavy:
            c.setStrokeGray(0.15)
            c.setLineWidth(1.6)
            c.setDash()
        else:
            c.setStrokeGray(0.55)
            c.setLineWidth(0.7)
            c.setDash(2, 1.5)
        c.line(p[0], p[1], q[0], q[1])
    c.setDash()
    for n, (x, y) in lay["dots"].items():
        not_in_text = not _in_text_files(n) and n != "WRIST"
        c.setFillColorRGB(*((0.85, 0.35, 0.0) if not_in_text
                            else (0.0, 0.0, 0.0)))
        c.circle(x, y, DOT_R, stroke=0, fill=1)
    c.setLineWidth(0.4)
    for i, n in enumerate(JOINT_NAMES):
        r = lay["plates"][n]
        c.setFillGray(1.0)
        c.setStrokeGray(0.45)
        c.rect(r[0], r[1], r[2] - r[0], r[3] - r[1], stroke=1, fill=1)
        c.setFillGray(0.0)
        c.setFont(*PLATE_FONT)
        c.drawCentredString((r[0] + r[2]) / 2, r[1] + 2.0, str(i + 1))
    c.setFont(*LETTER_FONT)
    for letter, (x, y, _w) in lay["letters"].items():
        c.drawCentredString(x, y, letter)

    ly = y0 - scale * 42.0 - 16.0
    legend = [
        ("Helvetica-Bold", "numbers = row of the joint-frame table "
                           "(1 = PALM ... 26 = LITTLE_TIP)"),
        ("Helvetica", "T I M R L = thumb, index, middle, ring, little "
                      "finger."),
        ("Helvetica", "Solid lines: the bones. A row's angles are the turn "
                      "from the bone before its dot (at a dashed line: the "
                      "forearm) to the bone that starts at it. Dashed: from "
                      "the WRIST joint (the root, the forearm's frame) to "
                      "the rows whose parent it is."),
        ("Helvetica", "Orange dots: the four finger CMC, not in the "
                      "keypoint text files."),
        ("Helvetica", landmark_summary()),
    ]
    for font, text in legend:
        for line in _wrap_pt(text, font, 7, left_w):
            c.setFont(font, 7)
            c.setFillGray(0.0)
            c.drawString(margin, ly, line)
            ly -= 9

    # --- right: the table
    ty = top
    c.setFont("Helvetica-Bold", size)
    x = tx
    for (head, _cell), w in zip(cols, widths):
        c.drawString(x, ty, head)
        x += w + gap
    ty -= 3
    c.setStrokeGray(0.3)
    c.setLineWidth(0.5)
    c.line(tx, ty, tx + table_w, ty)
    ty -= lead - 2.5
    for k, r in enumerate(rows):
        if k % 2 == 1:
            c.setFillGray(0.93)
            c.rect(tx - 2, ty - 2.6, table_w + 4, lead, stroke=0, fill=1)
        c.setFillGray(0.0)
        c.setFont("Helvetica", size)
        x = tx
        for j, ((_head, cell), w) in enumerate(zip(cols, widths)):
            text = cell(r)
            if j == 0:
                c.drawRightString(x + w, ty, text)
            else:
                c.drawString(x, ty, text)
            x += w + gap
        if r["row"] in ("WRIST", "THUMB_TIP", "INDEX_TIP", "MIDDLE_TIP",
                        "RING_TIP"):
            c.setStrokeGray(0.6)
            c.setLineWidth(0.4)
            c.line(tx, ty - 2.6, tx + table_w, ty - 2.6)
        ty -= lead
    c.setStrokeGray(0.3)
    c.setLineWidth(0.5)
    c.line(tx, ty + lead - 2.6, tx + table_w, ty + lead - 2.6)
    ty -= 6
    for text in (LEGEND_JOINT, LEGEND_GIVEN, LEGEND_SIGNS, LEGEND_TEXT,
                 "Every row's meaning in words, its two bones and its "
                 "landmark are in joint_key.csv."):
        for line in _wrap_pt(text, "Helvetica", 6.8, table_w):
            c.setFont("Helvetica", 6.8)
            c.setFillGray(0.0)
            c.drawString(tx, ty, line)
            ty -= 8.4
        ty -= 3

    # footer
    c.setFont("Helvetica", 7)
    c.setFillGray(0.3)
    fy = margin - 10
    for line in reversed(_wrap_pt(footer, "Helvetica", 7,
                                  page_w - 2 * margin)):
        c.drawString(margin, fy, line)
        fy += 9
    c.showPage()
    c.save()
