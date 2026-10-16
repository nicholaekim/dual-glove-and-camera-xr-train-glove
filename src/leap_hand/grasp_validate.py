"""Check a built grasp hand-in folder against its own files.

`scripts/package_professor_set.py` writes the folder from the session;
this reads the finished folder back, as a stranger would, and measures
everything again from the files in it: it never looks at the builder's
memory, and it needs no session folder (`--session` adds two checks against
session.json). It answers the list the hand-in is judged by:

  the grasp types and how many recordings each has, against the target;
  every recording's duration, recounted from the timestamps in its JSONL;
  the other hand and the hand's identity, recounted from the lines;
  every file a row of the summary names: there, well-formed, and the
    labelled hand's (keypoints = the representative line's joints, the
    joint-frame table = that line's table, the all-frames file = the
    labelled hand's lines of the grasp's takes and nothing else);
  the representative frame: the frame nearest the mean hand of the
    least-moving window (what the recorder calls the medoid), found again
    with `leap_hand.static_interval`'s own functions;
  angles: present for every recording, finite, the 18 of the paper's 24
    that this tracker gives equal to the cells of the joint-frame table,
    and the six it does not give left empty;
  the README's own formulas: every joint-frame table (positions, axes,
    angles), every column of the angle table, the stillness and the pose
    speed of every static interval, the shape of every whole recording and
    the hand's size, computed a second time by the few functions under
    "the README's formulas" below, which share no code with the library
    that made them;
  transitions: a row only where a recording of the change exists, a
    duration only on a row the rule passes, and the rule run again on the
    recording giving the row's numbers; no row that says MOCK (a take of
    a synthetic hand) unless the check is told allow_mock, and none
    either way without the README's MOCK DATA line at its top;
  the README: the map of what was asked for at its top, its twenty
    sections, and its headline counts; section 1's counts of what the
    pictures show add up to the recordings, a picture not yet read by eye
    counted on a line of its own;
  the pictures: every image a summary row names is there, a PNG and not
    all black, grasp_images.pdf has a page for every grasp type with one,
    and nothing else in the folder is an image no row names;
  the joint key: joint_key.csv has the 26 rows of a joint-frame table in
    order, and each of the paper's angles this tracker gives sits in the
    row and column the angle table takes it from; joint_key.pdf is one
    page; the README carries the key's naming rule and every row.

Three statuses are reported, because they are three different questions:

  STATIC-POSE TASK             one usable labelled static configuration for
                               every grasp type (the task as it was set)
  FULL MULTI-TAKE COLLECTION   the takes-per-grasp target (our
                               replication target)
  TRANSITION TIMING            PARTIAL with a valid timing of a recorded
                               change of pose, MISSING without one; never
                               more, because no number of them was asked for

and one verdict on the folder itself, FOLDER CHECK: PASSED or FAILED. The
exit code follows the verdict only: an incomplete collection is a state of
the work, not a fault in the folder.
"""
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES

from . import grasp_handin as gh
from . import transition as tr
from .grasp_audit import BY_LABEL, frame_key
from .static_interval import medoid_in_window, static_interval

TIME_TOL_S = 0.0015          # a summary time against its recount (3 decimals)
POINT_TOL_MM = 0.011         # a text coordinate against the line it came from
ANGLE_TOL_DEG = 0.011
CELL_TOL = 0.006             # a cell rounded to 0.01 against the README's formula
CHAIN_TOL_MM = 0.002         # the chained joints against abs26 (the same hand twice)
# an axis component rounded to 1e-6, or the matrix of a quaternion rounded
# to 1e-6 (each of its entries is a sum of products of two such numbers)
AXIS_TOL = 5e-6
POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), "
                      r"(-?\d+(?:\.\d+)?)\)")
HEAD_RE = re.compile(r"^Frame (\d+) \| Hand ID: (\w+) \| Time: (\S+)$")
# The 22 points of a text block, as joints of the 26.
BLOCK_JOINTS = (["WRIST", "PALM", "THUMB_METACARPAL", "THUMB_PROXIMAL",
                 "THUMB_DISTAL", "THUMB_TIP"]
                + [f"{f}_{s}" for f in ("INDEX", "MIDDLE", "RING", "LITTLE")
                   for s in ("PROXIMAL", "INTERMEDIATE", "DISTAL", "TIP")])
SECTION_TITLES = [
    "PURPOSE", "DATASET AT A GLANCE", "GRASP TYPES AND PAPER SOURCES",
    "FOLDER / FILE STRUCTURE", "HOW EACH RECORDING WAS CAPTURED",
    "RECORDING DURATION", "STATIC-POSE / REPRESENTATIVE-FRAME SELECTION",
    "RAW LEAP / OPENXR JOINT DATA", "21-POINT KEYPOINT FORMAT",
    "XYZ COORDINATE CONVENTIONS", "JOINT LOCAL FRAMES", "JOINT ANGLES",
    "MAPPING TO THE PAPER'S 24 ANGLES", "PER-POSE DATASETS",
    "QUALITY CONTROL", "OTHER-HAND DETECTION", "POSE TRANSITION TIMING",
    "CURRENT DATASET COMPLETENESS", "RECORDINGS STILL REQUIRED",
    "REPRODUCING / REBUILDING THE EXPORT"]


@dataclass
class Report:
    """What `validate` found. `errors` fail the folder; `notes` do not."""

    folder: Path
    facts: Dict[str, object] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def error(self, text: str) -> None:
        self.errors.append(text)


# --- reading the folder -------------------------------------------------------
def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_jsonl(path: Path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for text in f:
            if text.strip():
                out.append(json.loads(text))
    return out


def read_blocks(path: Path) -> List[Tuple[int, str, np.ndarray]]:
    """[(frame number, Hand ID, 22 x 3 points in mm)] of a text file."""
    out = []
    text = path.read_text(encoding="utf-8")
    for block in text.strip("\n").split("\n\n"):
        lines = [x for x in block.splitlines() if x.strip()]
        if not lines:
            continue
        head = HEAD_RE.match(lines[0])
        if head is None:
            raise ValueError(f"not a frame header: {lines[0][:60]!r}")
        pts = []
        for x in lines[1:]:
            m = POINT_RE.search(x)
            if m is None:
                raise ValueError(f"not a coordinate line: {x[:60]!r}")
            pts.append([float(v) for v in m.groups()])
        out.append((int(head.group(1)), head.group(2), np.array(pts)))
    return out


def num(cell: Optional[str]) -> Optional[float]:
    """A CSV cell as a number; None for an empty or non-numeric cell."""
    if cell is None or cell == "":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def hand_lines(lines: Sequence[dict], row: Dict[str, str]
               ) -> List[Tuple[int, dict]]:
    """(index, line) of the labelled hand in a recording, from what the
    summary row says: the lines of the tracker ids in `hand_ids`, and, for
    a take whose hand was chosen by the tracker's label, of that label."""
    ids = {int(x) for x in (row.get("hand_ids") or "").split()}
    label = (row.get("tracker_label_on_representative_frame")
             or row.get("hand"))
    by_label = row.get("hand_selection") == BY_LABEL or not ids
    return [(i, d) for i, d in enumerate(lines)
            if (not ids or (d.get("hand_id") is not None
                            and int(d["hand_id"]) in ids))
            and (not by_label or d.get("hand_side") == label)]


def block_points(line: dict) -> np.ndarray:
    """The 22 points a text block holds for one line, in camera mm."""
    pos = np.asarray(line["abs26"], dtype=float) * 1000.0
    return np.array([pos[JOINT_INDEX[j]] for j in BLOCK_JOINTS])


def finite(values) -> bool:
    try:
        return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))
    except (TypeError, ValueError):
        return False


# --- the README's formulas, written out again ---------------------------------
# Sections 7, 8, 10 and 12 of the README say how a joint-frame table, the
# stillness of a static interval and the hand's size follow from the lines
# of a recording. The functions below are that text turned into code a
# second time, on purpose without `xr_hand.joint_frames`,
# `leap_hand.grasp_audit` or `cam_hand.align`: a number that agrees with
# them agrees with what the README tells a reader to compute, whatever the
# library does.
def readme_parent(names: Sequence[str], name: str) -> Optional[str]:
    """Section 8: the joint before it on its own digit; the WRIST for the
    PALM and the five METACARPAL joints; none for the WRIST."""
    if name == "WRIST":
        return None
    if name == "PALM" or name.endswith("_METACARPAL"):
        return "WRIST"
    return names[names.index(name) - 1]


def readme_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Section 12, step 1: a unit quaternion (qx, qy, qz, qw) as a matrix."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def readme_angles(r_rel: np.ndarray, s: float) -> Tuple[float, float, float]:
    """Section 12, steps 3 and 4: (flex, abd, twist) in degrees."""
    r13 = min(1.0, max(-1.0, float(r_rel[0, 2])))
    b = math.asin(r13)
    if abs(r13) >= 1 - 1e-9:
        a, c = math.atan2(r_rel[2, 1], r_rel[1, 1]), 0.0
    else:
        a = math.atan2(-r_rel[1, 2], r_rel[2, 2])
        c = math.atan2(-r_rel[0, 1], r_rel[0, 0])
    return -math.degrees(a), s * math.degrees(b), -s * math.degrees(c)


def readme_chain(line: dict) -> Tuple[List[str], Dict[str, np.ndarray],
                                      Dict[str, np.ndarray],
                                      Dict[str, np.ndarray]]:
    """Section 10 C: (joint names in the line's order, position in the
    camera frame in metres, rotation in the camera frame, rotation relative
    to the parent) of every joint, chained from the WRIST: position =
    parent position + parent rotation x local position, rotation = parent
    rotation x local rotation."""
    joints = line["joints"]
    names = [j["name"] for j in joints]
    by = {j["name"]: j for j in joints}
    rel = {n: readme_rotation(j["qx"], j["qy"], j["qz"], j["qw"])
           for n, j in by.items()}
    rot: Dict[str, np.ndarray] = {}
    pos: Dict[str, np.ndarray] = {}

    def place(name: str) -> None:
        if name in rot:
            return
        local = np.array([by[name]["x"], by[name]["y"], by[name]["z"]],
                         dtype=float)
        parent = readme_parent(names, name)
        if parent is None:
            rot[name], pos[name] = rel[name], local
        else:
            place(parent)
            rot[name] = rot[parent] @ rel[name]
            pos[name] = pos[parent] + rot[parent] @ local
    for n in names:
        place(n)
    return names, pos, rot, rel


def chain_gap_mm(line: dict) -> float:
    """The largest distance along an axis, in mm, between the chained
    joints of a line and its `abs26`: the same hand stored twice."""
    names, pos, _rot, _rel = readme_chain(line)
    cam = np.asarray(line["abs26"], dtype=float)
    return float(max(np.abs(pos[n] - cam[k]).max()
                     for k, n in enumerate(names))) * 1000.0


def readme_table(line: dict) -> Dict[str, Dict[str, object]]:
    """joint -> what the README says its row holds, for one line of a
    recording: `xyz` (mm, wrist frame), `axes` (3 x 3, columns = the
    joint's +x +y +z in the wrist frame), `bone` (mm) and `angles`. The
    positions are the chained ones, which is what the tables are made of."""
    names, pos, rot, rel = readme_chain(line)
    s = 1.0 if line.get("hand_side") == "right" else -1.0
    r_wrist, p_wrist = rot["WRIST"], pos["WRIST"]
    out = {}
    for n in names:
        parent = readme_parent(names, n)
        out[n] = {
            "parent": parent or "",
            "xyz": r_wrist.T @ (pos[n] - p_wrist) * 1000.0,   # section 10 B
            "axes": r_wrist.T @ rot[n],
            "bone": (0.0 if parent is None else
                     float(np.linalg.norm(pos[n] - pos[parent])) * 1000.0),
            "angles": ((0.0, 0.0, 0.0) if parent is None
                       else readme_angles(rel[n], s))}
    return out


def table_gap(line: dict, table: Sequence[Dict[str, str]]
              ) -> Tuple[float, float, str]:
    """(largest difference of a 0.01-rounded cell, largest of an axis or
    quaternion component, first thing that is simply wrong or "") between
    a joint-frame CSV and `readme_table` of the line it claims to be."""
    want = readme_table(line)
    cell = axis = 0.0
    for r in table:
        w = want.get(r["joint"])
        if w is None:
            return cell, axis, f"{r['joint']} is not a joint of the line"
        if r["parent"] != w["parent"]:
            return cell, axis, f"{r['joint']}: parent {r['parent']!r}"
        xyz = np.array([float(r[c]) for c in ("x_mm", "y_mm", "z_mm")])
        got = (float(r["flex_deg"]), float(r["abd_deg"]),
               float(r["twist_deg"]))
        cell = max(cell, float(np.abs(xyz - w["xyz"]).max()),
                   abs(float(r["bone_len_mm"]) - w["bone"]),
                   *(abs(a - b) for a, b in zip(got, w["angles"])))
        axes = np.array([[float(r[f"axis_{a}_{c}"]) for c in "xyz"]
                         for a in "xyz"]).T
        quat = readme_rotation(*(float(r[c])
                                 for c in ("qx", "qy", "qz", "qw")))
        axis = max(axis, float(np.abs(axes - w["axes"]).max()),
                   float(np.abs(quat - axes).max()))
        if float(r["qw"]) < 0:
            return cell, axis, f"{r['joint']}: qw is negative"
    return cell, axis, ""


def rigid_fit(src: np.ndarray, dst: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    """The rotation and translation (least squares, no scaling, no
    mirroring) that best lay the points `src` on `dst`."""
    a, b = src.mean(axis=0), dst.mean(axis=0)
    u, _d, vt = np.linalg.svd((dst - b).T @ (src - a))
    fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        fix[2, 2] = -1.0
    r = u @ fix @ vt
    return r, b - r @ a


def readme_stillness(window: Sequence[dict], rep_line: dict
                     ) -> Tuple[float, float, float]:
    """Section 7, HOW STILL IT WAS, for the lines of a static interval in
    time order: (mean joint speed in mm/s, median shape RMS in mm, largest
    shape RMS in mm)."""
    pos = np.array([d["abs26"] for d in window], dtype=float)
    t = np.array([float(d["wall_time"]) for d in window])
    path = np.linalg.norm(pos[1:] - pos[:-1], axis=2).mean(axis=1).sum()
    speed = float(path) / float(t[-1] - t[0]) * 1000.0
    ref = np.asarray(rep_line["abs26"], dtype=float)
    rms = []
    for q in pos:
        r, move = rigid_fit(q, ref)
        rms.append(math.sqrt(float((((r @ q.T).T + move - ref) ** 2)
                                   .sum(axis=1).mean())) * 1000.0)
    return speed, float(np.median(rms)), float(np.max(rms))


PALM_NAMES = ("WRIST", "PALM", "THUMB_METACARPAL", "INDEX_METACARPAL",
              "MIDDLE_METACARPAL", "RING_METACARPAL", "LITTLE_METACARPAL",
              "INDEX_PROXIMAL", "MIDDLE_PROXIMAL", "RING_PROXIMAL",
              "LITTLE_PROXIMAL")


def readme_pose_speed(window: Sequence[dict], step_s: float = 0.1
                      ) -> np.ndarray:
    """Section 17, the pose speed of every line of `window` (in time
    order): the earliest and the latest line within half a step of it, the
    later hand laid on the earlier by the rigid fit of the palm's eleven
    joints, the mean over the 26 joints of the distance left, over the
    time between the two (mm/s; the line's capture_time)."""
    names = [j["name"] for j in window[0]["joints"]]
    palm = [names.index(n) for n in PALM_NAMES]
    t = np.array([float(d["wall_time"] if d.get("capture_time") is None
                        else d["capture_time"]) for d in window])
    pos = np.array([d["abs26"] for d in window], dtype=float)
    out = np.full(len(t), np.nan)
    for k in range(len(t)):
        a = int(np.searchsorted(t, t[k] - step_s / 2.0, side="left"))
        b = int(np.searchsorted(t, t[k] + step_s / 2.0, side="right")) - 1
        if b <= a or t[b] <= t[a]:
            continue
        r, move = rigid_fit(pos[b][palm], pos[a][palm])
        left = (r @ pos[b].T).T + move - pos[a]
        out[k] = np.linalg.norm(left, axis=1).mean() / (t[b] - t[a]) * 1000.0
    return out


def readme_whole_take(lines: Sequence[dict], rep_line: dict
                      ) -> Tuple[float, float, float, float]:
    """Section 14, a whole recording against its representative frame:
    (largest shape distance in mm, percent of the lines within 5 mm, mean
    over the first 0.2 s, mean over the last 0.2 s), the shape distance
    being the RMS over the 26 joints after a rigid fit of all of them."""
    ref = np.asarray(rep_line["abs26"], dtype=float)
    t = np.array([float(d["wall_time"]) for d in lines])
    d = []
    for line in lines:
        q = np.asarray(line["abs26"], dtype=float)
        r, move = rigid_fit(q, ref)
        d.append(math.sqrt(float((((r @ q.T).T + move - ref) ** 2)
                                 .sum(axis=1).mean())) * 1000.0)
    d = np.array(d)
    return (float(d.max()), 100.0 * float((d <= 5.0).mean()),
            float(d[t <= t[0] + 0.2].mean()),
            float(d[t >= t[-1] - 0.2].mean()))


def readme_angle_row(line: dict) -> Dict[str, float]:
    """Section 13, every filled column of the angle table for one line of
    a recording: the 18 given angles, forearm_to_palm_deg, the twelve
    *_metacarpal_* columns and the eight *_paper_order columns."""
    names, _pos, rot, rel = readme_chain(line)
    s = 1.0 if line.get("hand_side") == "right" else -1.0
    out: Dict[str, float] = {}
    for letter, finger in (("T", "THUMB"), ("I", "INDEX"), ("M", "MIDDLE"),
                           ("R", "RING"), ("L", "LITTLE")):
        mc = f"{finger}_METACARPAL"
        wrist = readme_angles(rel[mc], s)                   # parent WRIST
        palm = readme_angles(rot["PALM"].T @ rot[mc], s)    # from the PALM
        out[f"{letter}_metacarpal_flex_wrist_ref"] = wrist[0]
        out[f"{letter}_metacarpal_flex_palm_ref"] = palm[0]
        prox = readme_angles(rel[f"{finger}_PROXIMAL"], s)
        dist = readme_angles(rel[f"{finger}_DISTAL"], s)
        if letter == "T":
            out["T_metacarpal_abd_wrist_ref"] = wrist[1]
            out["T_metacarpal_abd_palm_ref"] = palm[1]
            out["T_MCP_fe"], out["T_IP"] = prox[0], dist[0]
            continue
        inter = readme_angles(rel[f"{finger}_INTERMEDIATE"], s)
        out[f"{letter}_MCP_fe"], out[f"{letter}_MCP_aa"] = prox[0], prox[1]
        out[f"{letter}_PIP"], out[f"{letter}_DIP"] = inter[0], dist[0]
        # the paper's order (item 1): from the bone's direction alone
        zx, zy, zz = (float(v) for v in rel[f"{finger}_PROXIMAL"][:, 2])
        k = 1.0 if zz >= 0.0 else -1.0
        out[f"{letter}_MCP_fe_paper_order"] = math.degrees(
            math.atan2(zy, k * math.hypot(zx, zz)))
        out[f"{letter}_MCP_aa_paper_order"] = s * math.degrees(
            math.atan2(k * zx, k * zz))
    c = (float(np.trace(rot["WRIST"].T @ rot["PALM"])) - 1.0) / 2.0
    out["forearm_to_palm_deg"] = math.degrees(math.acos(min(1.0, max(-1.0,
                                                                     c))))
    return out


def readme_hand_size(lines: Sequence[dict]) -> Tuple[float, float]:
    """Section 10, SCALE: (median, largest minus smallest) of the index
    metacarpal's length in mm over a hand's lines."""
    names = [j["name"] for j in lines[0]["joints"]]
    a, b = names.index("INDEX_METACARPAL"), names.index("INDEX_PROXIMAL")
    lengths = np.array([np.linalg.norm(np.subtract(d["abs26"][b],
                                                   d["abs26"][a]))
                        for d in lines]) * 1000.0
    return float(np.median(lengths)), float(lengths.max() - lengths.min())


# --- the checks -----------------------------------------------------------------
def _check_take(rep: Report, folder: Path, row: Dict[str, str],
                seen: Dict[str, object]) -> None:
    """Everything about one summary row; `seen` collects what the per-grasp
    checks need."""
    name = f"{row['grasp_name']} take {row['take']}"
    raw = folder / row["raw_jsonl_file"]
    if not raw.is_file():
        rep.error(f"{name}: the recording {row['raw_jsonl_file']} is missing")
        seen["missing_raw"] += 1
        return
    try:
        lines = read_jsonl(raw)
    except ValueError as e:
        rep.error(f"{name}: {row['raw_jsonl_file']} does not parse ({e})")
        return
    mine = hand_lines(lines, row)
    if not mine:
        rep.error(f"{name}: no line of the labelled hand in the recording")
        return

    # the label, on every line
    wrong = [d for d in lines if (d.get("item") or d.get("pose"))
             != row["grasp_name"]]
    if wrong:
        rep.error(f"{name}: {len(wrong)} line(s) of the recording carry "
                  f"another label ({wrong[0].get('item')!r})")
        seen["label_errors"] += 1
    expect = f"{gh.GRASPS_DIR}/{row['grasp_name']}_{row['hand']}_take" \
             f"{row['take']}.jsonl"
    if row["raw_jsonl_file"] != expect:
        rep.error(f"{name}: recording file is {row['raw_jsonl_file']}, the "
                  f"label wants {expect}")
        seen["label_errors"] += 1

    # duration and frames, from the timestamps and the lines
    walls = [float(d["wall_time"]) for _i, d in mine]
    duration = max(walls) - min(walls)
    seen["durations"].append(duration)
    got = num(row.get("recording_duration_s"))
    if got is None or not math.isfinite(got) or got <= 0:
        rep.error(f"{name}: recording_duration_s is missing or not a "
                  f"positive number ({row.get('recording_duration_s')!r})")
        seen["bad_duration"] += 1
    elif abs(got - duration) > TIME_TOL_S:
        rep.error(f"{name}: recording_duration_s {got} but the timestamps "
                  f"give {duration:.3f}")
        seen["bad_duration"] += 1
    frames = {frame_key(i, d) for i, d in enumerate(lines)}
    for column, value in (("jsonl_row_count", len(lines)),
                          ("unique_frame_count", len(frames)),
                          ("hand_frame_count", len(mine))):
        if num(row.get(column)) != float(value):
            rep.error(f"{name}: {column} is {row.get(column)!r}, the file "
                      f"has {value}")

    # the other hand, and the hand's identity
    own = {i for i, _d in mine}
    other_frames = {frame_key(i, d) for i, d in enumerate(lines)
                    if i not in own}
    detected = bool(other_frames)
    if (row.get("other_hand_detected") == "true") != detected:
        rep.error(f"{name}: other_hand_detected is "
                  f"{row.get('other_hand_detected')!r}, the recording "
                  f"{'has' if detected else 'has no'} second hand")
    fraction = len(other_frames) / len(frames)
    if abs((num(row.get("other_hand_frame_fraction")) or 0.0)
           - fraction) > 6e-5:
        rep.error(f"{name}: other_hand_frame_fraction is "
                  f"{row.get('other_hand_frame_fraction')!r}, the recording "
                  f"gives {fraction:.4f}")
    seen["other_hand"] += int(detected)
    labels = sorted({str(d.get("hand_side")) for _i, d in mine})
    changes = sum(1 for (_a, a), (_b, b) in zip(mine, mine[1:])
                  if a.get("hand_id") != b.get("hand_id"))
    flagged = bool(row.get("identity_flag"))
    if (changes or len(labels) > 1) and not flagged:
        rep.error(f"{name}: the hand's id changed {changes} time(s) and it "
                  f"carries the label(s) {labels}, but identity_flag is empty")
    seen["identity"] += int(flagged)
    if row.get("notes"):
        seen["noted"].append(f"{name}: {row['notes']}")

    # XYZ in the raw lines
    bad_xyz = sum(1 for _i, d in mine
                  if np.asarray(d.get("abs26", []), dtype=float).shape
                  != (26, 3) or not finite(d["abs26"]))
    if bad_xyz:
        rep.error(f"{name}: {bad_xyz} line(s) of the labelled hand without "
                  "26 finite XYZ joint positions")
        seen["bad_xyz"] += 1

    # the representative frame
    index = num(row.get("representative_frame_index"))
    rep_line = None
    if index is None or not 0 <= int(index) < len(lines):
        rep.error(f"{name}: representative_frame_index "
                  f"{row.get('representative_frame_index')!r} is not a line "
                  "of the recording")
    else:
        rep_line = lines[int(index)]
        if int(index) not in own:
            rep.error(f"{name}: the representative frame is not a line of "
                      "the labelled hand")
        start = num(row.get("static_interval_start_s"))
        end = num(row.get("static_interval_end_s"))
        at = float(rep_line["wall_time"]) - min(walls)
        if start is not None and end is not None and not (
                start - TIME_TOL_S <= at <= end + TIME_TOL_S):
            rep.error(f"{name}: the representative frame ({at:.3f} s) is "
                      f"outside the static interval ({start} to {end} s)")
        # the rule itself: the least-moving window, and its medoid
        measured = [(i, d) for i, d in mine
                    if d.get("hand_side") == row.get("hand")]
        length = num(row.get("static_interval_duration_s"))
        if measured and length:
            again = static_interval(lines, length, picked=measured)
            found = (None if again is None else
                     medoid_in_window(lines, again[0], again[1],
                                      picked=measured))
            same = (found is not None and found[1] == int(index)
                    and start is not None
                    and abs((again[0] - min(walls)) - start) <= TIME_TOL_S)
            seen["rule_same"] += int(same)
            seen["rule_checked"] += 1
            if not same:
                seen["rule_differs"].append(name)

    # how still, and how big: the README's formulas on the file's own lines
    usable_lines = [d for _i, d in mine
                    if np.asarray(d.get("abs26", []), dtype=float).shape
                    == (26, 3)]
    a, b = num(row.get("static_interval_start")), num(
        row.get("static_interval_end"))
    if rep_line is not None and a is not None and b is not None:
        window = [d for d in usable_lines
                  if d.get("hand_side") == row.get("hand")
                  and a - 1e-6 <= float(d["wall_time"]) <= b + 1e-6]
        if len(window) >= 3:
            speed, rms, rms_max = readme_stillness(window, rep_line)
            size, size_range = readme_hand_size(usable_lines)
            pose = float(np.nanmedian(readme_pose_speed(window)))
            labelled = [d for d in usable_lines
                        if d.get("hand_side") == row.get("hand")]
            far, within, at_start, at_end = readme_whole_take(labelled,
                                                              rep_line)
            same = True
            for column, value in (
                    ("static_interval_first_frame_id",
                     window[0].get("frame_id")),
                    ("static_interval_last_frame_id",
                     window[-1].get("frame_id"))):
                if value is not None and num(row.get(column)) != float(value):
                    rep.error(f"{name}: {column} is {row.get(column)!r}, "
                              f"the static interval's frame is {value}")
                    same = False
            for column, value, tol in (
                    ("static_interval_mean_speed_mm_s", speed, 0.06),
                    ("static_interval_pose_speed_mm_s", pose, 0.06),
                    ("static_interval_shape_rms_mm", rms, CELL_TOL),
                    ("static_interval_shape_rms_max_mm", rms_max, CELL_TOL),
                    ("take_shape_rms_max_mm", far, CELL_TOL),
                    ("frames_within_5mm_percent", within, 0.06),
                    ("shape_rms_start_mm", at_start, CELL_TOL),
                    ("shape_rms_end_mm", at_end, CELL_TOL),
                    ("index_metacarpal_mm", size, CELL_TOL),
                    ("index_metacarpal_range_mm", size_range, CELL_TOL)):
                got = num(row.get(column))
                if got is None or abs(got - value) > tol:
                    rep.error(f"{name}: {column} is {row.get(column)!r}, "
                              f"the README's formula gives {value:.2f}")
                    same = False
            seen["stillness_same"] += int(same)
            seen["stillness_checked"] += 1
            seen["speeds"].append(speed)

    # the keypoints file
    kp = row.get("representative_keypoints_file")
    if not kp or not (folder / kp).is_file():
        rep.error(f"{name}: no representative keypoints file")
        seen["missing_keypoints"] += 1
    elif rep_line is not None:
        try:
            blocks = read_blocks(folder / kp)
        except ValueError as e:
            rep.error(f"{name}: {kp} is malformed ({e})")
            seen["bad_xyz"] += 1
            blocks = []
        if blocks:
            fid, _side, pts = blocks[0]
            if len(blocks) != 1 or pts.shape != (22, 3) or not finite(pts):
                rep.error(f"{name}: {kp} should hold one frame of 22 "
                          f"points, has {len(blocks)} block(s) of shape "
                          f"{pts.shape}")
                seen["bad_xyz"] += 1
            elif (fid != int(rep_line.get("frame_id", fid))
                  or float(np.abs(pts - block_points(rep_line)).max())
                  > POINT_TOL_MM):
                rep.error(f"{name}: {kp} is not the representative frame "
                          "of the recording")

    # the joint-frame table and its angles
    jf_file = row.get("joint_frames_file")
    if not jf_file or not (folder / jf_file).is_file():
        rep.error(f"{name}: no joint-frame CSV")
        seen["missing_joint_frames"] += 1
    else:
        header, table = read_csv(folder / jf_file)
        if header != jf.CSV_COLUMNS or len(table) != 26:
            rep.error(f"{name}: {jf_file} should have 26 rows of "
                      f"{len(jf.CSV_COLUMNS)} columns")
            seen["bad_xyz"] += 1
        else:
            cells = [[r[c] for c in jf.CSV_COLUMNS[2:]] for r in table]
            if not finite(cells):
                rep.error(f"{name}: {jf_file} holds an empty, NaN or "
                          "infinite number")
                seen["nan_angles"] += 1
            elif rep_line is not None:
                want = jf.csv_rows(jf.joint_table(rep_line, "camera"))
                if [[r[c] for c in jf.CSV_COLUMNS] for r in table] != want:
                    rep.error(f"{name}: {jf_file} is not the table of the "
                              "representative frame")
                # and by the README's formulas, with no library code
                cell, axis, wrong = table_gap(rep_line, table)
                seen["formula_checked"] += 1
                if wrong or cell > CELL_TOL or axis > AXIS_TOL:
                    rep.error(f"{name}: {jf_file} is not what the README's "
                              "formulas give for the representative frame ("
                              + (wrong or f"a cell differs by {cell:.4f}, "
                                 f"an axis component by {axis:.1e}") + ")")
                else:
                    seen["formula_same"] += 1
                # the same hand stored twice in the line: the chain and abs26
                if np.asarray(rep_line.get("abs26", []),
                              dtype=float).shape == (26, 3):
                    gap = chain_gap_mm(rep_line)
                    seen["chain_gap"] = max(seen["chain_gap"], gap)
                    if gap > CHAIN_TOL_MM:
                        rep.error(f"{name}: the chained joints of the "
                                  "representative frame are "
                                  f"{gap:.4f} mm from its abs26")
            seen["joint_tables"][(row["grasp_name"], row["take"])] = {
                r["joint"]: r for r in table}
    # what the per-grasp checks need of this recording, and no more: a take
    # is tens of megabytes as parsed lines
    key = (row["grasp_name"], row["take"])
    usable = [d for _i, d in mine
              if np.asarray(d.get("abs26", []), dtype=float).shape == (26, 3)]
    seen["frames"][key] = (
        [int(d.get("frame_id", -1)) for d in usable],
        np.array([block_points(d) for d in usable]).reshape(-1, 22, 3))
    seen["rep_line"][key] = rep_line


def _check_per_grasp(rep: Report, folder: Path, summary: List[Dict[str, str]],
                     audit: List[Dict[str, str]],
                     seen: Dict[str, object]) -> None:
    by_grasp: Dict[str, List[Dict[str, str]]] = {}
    for row in summary:
        by_grasp.setdefault(row["grasp_name"], []).append(row)
    known = {a["grasp_name"] for a in audit}
    for g in by_grasp:
        if g not in known:
            rep.error(f"{g}: recordings carry this label but it is not a "
                      "grasp type of the audit table")
            seen["label_errors"] += 1
    ready = 0
    for a in audit:
        g = a["grasp_name"]
        rows = sorted(by_grasp.get(g, []), key=lambda r: int(r["take"]))
        if not a.get("source_paper") or not a.get("source_figure"):
            rep.error(f"{g}: no paper or figure in the audit table")
            seen["no_source"] += 1
        if num(a.get("accepted_recordings")) != float(len(rows)):
            rep.error(f"{g}: the audit table says "
                      f"{a.get('accepted_recordings')} recording(s), the "
                      f"summary has {len(rows)}")
        clean = sum(1 for r in rows if not r.get("notes"))
        if num(a.get("recordings_without_a_note")) != float(clean):
            rep.error(f"{g}: recordings_without_a_note is "
                      f"{a.get('recordings_without_a_note')!r}, the summary "
                      f"has {clean} recording(s) with no note")
        if rows and not clean:
            seen["all_noted"].append(g)
        split = [num(a.get(c)) for c in ("attempts_not_kept",
                                         "open_hand_timeouts",
                                         "rejected_recordings")]
        if None in split or split[0] != split[1] + split[2]:
            rep.error(f"{g}: attempts_not_kept is not open_hand_timeouts "
                      "plus rejected_recordings")
        for r in rows:
            if (r.get("source_paper"), r.get("source_figure")) != (
                    a.get("source_paper"), a.get("source_figure")):
                rep.error(f"{g} take {r['take']}: paper or figure differs "
                          "from the audit table")
                seen["no_source"] += 1
        usable = [r for r in rows
                  if (r["grasp_name"], r["take"]) in seen["joint_tables"]
                  and r.get("representative_keypoints_file")
                  and (folder / r["representative_keypoints_file"]).is_file()]
        status = gh.READY if usable else gh.INCOMPLETE
        ready += int(status == gh.READY)
        if a.get("static_pose_status") != status:
            rep.error(f"{g}: static_pose_status is "
                      f"{a.get('static_pose_status')!r}, the files give "
                      f"{status}")
        if not rows:
            continue
        # the frame-by-frame dataset: the labelled hand of these takes only
        path = folder / gh.GRASPS_DIR / f"{g}_{rows[0]['hand']}_all_frames.txt"
        if not path.is_file():
            rep.error(f"{g}: no all-frames dataset")
            seen["missing_all_frames"] += 1
            continue
        try:
            blocks = read_blocks(path)
        except ValueError as e:
            rep.error(f"{g}: {path.name} is malformed ({e})")
            seen["bad_xyz"] += 1
            continue
        got = [seen["frames"][(g, r["take"])] for r in rows
               if (g, r["take"]) in seen["frames"]]
        want_ids = [fid for ids, _pts in got for fid in ids]
        if len(blocks) != len(want_ids):
            rep.error(f"{g}: {path.name} has {len(blocks)} frame(s), the "
                      f"labelled hand of its takes has {len(want_ids)}")
            continue
        if ([b[0] for b in blocks] != want_ids
                or any(b[2].shape != (22, 3) for b in blocks)):
            rep.error(f"{g}: {path.name} is not the labelled hand's frames "
                      "of its takes, in take order")
            continue
        have = np.array([b[2] for b in blocks]).reshape(-1, 22, 3)
        want = np.concatenate([pts for _ids, pts in got]) if got else have
        worst = float(np.abs(have - want).max()) if len(have) else 0.0
        if worst > POINT_TOL_MM or not finite(have):
            rep.error(f"{g}: {path.name} differs from the recordings by up "
                      f"to {worst:.3f} mm")
            seen["bad_xyz"] += 1
        if num(a.get("all_frames_frame_count")) != float(len(blocks)):
            rep.error(f"{g}: all_frames_frame_count is "
                      f"{a.get('all_frames_frame_count')!r}, the file has "
                      f"{len(blocks)}")
    seen["ready"] = ready


def _check_angles(rep: Report, folder: Path, summary: List[Dict[str, str]],
                  audit: List[Dict[str, str]],
                  seen: Dict[str, object]) -> None:
    path = folder / gh.GRASPS_DIR / gh.ANGLES_FILE
    tables = seen["joint_tables"]
    if not path.is_file():
        if tables:
            rep.error(f"{gh.ANGLES_FILE} is missing")
        seen["missing_angles"] = len(summary)
        return
    header, rows = read_csv(path)
    if header != gh.ANGLE_COLUMNS:
        rep.error(f"{gh.ANGLES_FILE}: unexpected columns")
        return
    have = {(r["grasp_name"], r["take"]): r for r in rows}
    missing = [k for k in tables if k not in have]
    seen["missing_angles"] = len(missing) + sum(
        1 for r in summary if (r["grasp_name"], r["take"]) not in tables)
    for k in missing:
        rep.error(f"{k[0]} take {k[1]}: no row in {gh.ANGLES_FILE}")
    filled = [c for c in gh.ANGLE_COLUMNS[5:] if c not in gh.NOT_GIVEN]
    for key, r in have.items():
        name = f"{key[0]} take {key[1]}"
        given_away = [n for n in gh.NOT_GIVEN if r.get(n)]
        if given_away:
            rep.error(f"{name}: {', '.join(given_away)} filled in, which "
                      "this tracker does not give (README section 13)")
        if not finite([r.get(n) for n in filled]):
            rep.error(f"{name}: an empty, NaN or infinite angle in "
                      f"{gh.ANGLES_FILE}")
            seen["nan_angles"] += 1
            continue
        table = tables.get(key)
        if table is None:
            rep.error(f"{name}: a row in {gh.ANGLES_FILE} but no "
                      "joint-frame table")
            continue
        # a given angle, and a metacarpal seen from the wrist, is a cell of
        # the take's joint-frame table
        cells = {n: jf.DOF24_SOURCE[n] for n in gh.GIVEN_NAMES}
        cells.update({c: jf.DOF24_SOURCE[k]
                      for c, k in gh.METACARPAL_COLUMNS.items()
                      if k in jf.DOF24_SOURCE})
        for column, (joint, cell) in cells.items():
            if abs(float(r[column]) - float(table[joint][cell])) > 1e-9:
                rep.error(f"{name}: {column} is {r[column]} in "
                          f"{gh.ANGLES_FILE} and {table[joint][cell]} in "
                          "the joint-frame table")
                break
        # every filled column, by the README's formulas on the recording
        line = seen["rep_line"].get(key)
        if line is None:
            continue
        want = readme_angle_row(line)
        seen["angles_checked"] += 1
        worst = max(abs(float(r[c]) - want[c]) for c in filled)
        if set(want) != set(filled) or worst > CELL_TOL:
            rep.error(f"{name}: {gh.ANGLES_FILE} is not what the README's "
                      f"formulas give (a cell differs by {worst:.4f})")
        else:
            seen["angles_same"] += 1
    # the audit's agreement between takes, from this table
    by_grasp: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_grasp.setdefault(r["grasp_name"], []).append(r)
    for a in audit:
        got = by_grasp.get(a["grasp_name"], [])
        if len(got) < 2:
            if a.get("between_take_max_spread_deg"):
                rep.error(f"{a['grasp_name']}: a spread between takes with "
                          "fewer than two takes")
            continue
        spreads = {n: max(float(r[n]) for r in got) - min(float(r[n])
                                                          for r in got)
                   for n in gh.GIVEN_NAMES}
        worst = max(spreads.values())
        said = num(a.get("between_take_max_spread_deg"))
        named = a.get("between_take_max_spread_angle")
        if (said is None or abs(said - worst) > 0.06
                or named not in spreads
                or abs(spreads[named] - worst) > 0.011):
            rep.error(f"{a['grasp_name']}: between_take_max_spread is "
                      f"{a.get('between_take_max_spread_deg')!r} "
                      f"({named}), the angle table gives {worst:.1f}")
        seen["spreads"].append(worst)


def _check_transitions(rep: Report, folder: Path,
                       summary: List[Dict[str, str]],
                       seen: Dict[str, object]) -> None:
    path = folder / gh.TRANSITIONS_DIR / gh.TRANSITIONS_FILE
    if not path.is_file():
        rep.error(f"{gh.TRANSITIONS_DIR}/{gh.TRANSITIONS_FILE} is missing")
        return
    header, rows = read_csv(path)
    if header != gh.TRANSITION_COLUMNS:
        rep.error(f"{gh.TRANSITIONS_FILE}: unexpected columns")
        return
    captures = {r["transition_capture_file"] for r in summary
                if r.get("transition_capture_file")}
    listed = {r["recording_file"] for r in rows}
    for name in sorted(captures - listed):
        rep.error(f"{name}: a recorded change of pose with no row in "
                  f"{gh.TRANSITIONS_FILE}")
    # a coached take's own change belongs to a static take, which names
    # it; a take of the transition recorder is a recording of its own
    from_open = {r["recording_file"] for r in rows
                 if r.get("static_take_file")}
    for name in sorted(from_open - captures):
        rep.error(f"{name}: a row in {gh.TRANSITIONS_FILE} for a take "
                  "whose summary row names no recording of the change")
    seen["transition_files"] = listed
    valid = 0
    for r in rows:
        name = f"{r['from_grasp']} to {r['to_grasp']} take {r['take']}"
        rec = folder / r["recording_file"]
        if not rec.is_file():
            rep.error(f"{name}: the recording {r['recording_file']} is "
                      "missing")
            continue
        if r["validity"] not in (tr.VALID, tr.QUESTIONABLE, tr.INVALID):
            rep.error(f"{name}: validity {r['validity']!r}")
        start, settled = num(r["movement_start_time_s"]), num(
            r["settled_time_s"])
        duration = num(r["transition_duration_s"])
        timed = r["validity"] in (tr.VALID, tr.QUESTIONABLE)
        if duration is not None and not timed:
            rep.error(f"{name}: a transition_duration_s on a row marked "
                      f"{r['validity']}")
        if timed and (duration is None or start is None or settled is None
                      or abs(duration - (settled - start)) > TIME_TOL_S):
            rep.error(f"{name}: transition_duration_s {duration} is not "
                      f"settled {settled} minus start {start}")
        notes = r.get("notes") or ""
        if MOCK_RE.search(notes):
            seen["mock_transitions"].append(name)
        if r["validity"] == tr.INVALID and not mock_free(notes):
            rep.error(f"{name}: invalid, with no reason given")
        valid += int(r["validity"] == tr.VALID)
        seen["questionable_transitions"] += int(
            r["validity"] == tr.QUESTIONABLE)
        # the rule, run again on the recording
        zero, cue = num(r["first_frame_capture_time"]), num(
            r["cue_wall_time"])
        until = num(r["analysed_until_s"])
        ids = {int(x) for x in (r.get("hand_ids") or "").split()}
        if zero is None or cue is None or not ids:
            continue
        lines = [d for d in read_jsonl(rec)
                 if d.get("hand_id") is not None and int(d["hand_id"]) in ids
                 and d.get("hand_side") == r["hand"]]
        again = tr.transition_timing(
            lines, cue, None if until is None else zero + until)
        if (again.movement_start_method != r["movement_start_method"]
                or again.settled_method != r["settled_method"]):
            rep.notes.append(f"{name}: timed with other thresholds than "
                             "this check's defaults; not run again")
            continue
        for column, value in (
                ("movement_start_time_s", again.movement_start_time_s),
                ("settled_time_s", again.settled_time_s),
                ("transition_duration_s",
                 again.transition_duration_s if again.validity in (
                     tr.VALID, tr.QUESTIONABLE) else None)):
            got = num(r[column])
            if (got is None) != (value is None) or (
                    got is not None and abs(got - value) > 0.012):
                rep.error(f"{name}: {column} is {r[column]!r}, the "
                          f"recording gives {value}")
        if again.validity != r["validity"]:
            rep.error(f"{name}: validity is {r['validity']!r}, the "
                      f"recording gives {again.validity!r}")
    seen["transitions"] = len(rows)
    seen["valid_transitions"] = valid
    seen["between_grasps"] = sum(
        1 for r in rows
        if gh.OPEN_HAND not in (r["from_grasp"], r["to_grasp"]))
    # a static take is never timed
    for r in summary:
        for column in r:
            if column.startswith("transition_") and column != \
                    "transition_capture_file" and r[column]:
                rep.error(f"{r['grasp_name']} take {r['take']}: the summary "
                          f"carries a transition value in {column}")


# A row of the transition table that says MOCK is a take of a synthetic hand
# (`grasp_handin.MOCK_NOTE` heads its notes): a rehearsal, never a hand-in.
MOCK_RE = re.compile(r"\bMOCK\b")
MOCK_BANNER = "MOCK DATA"


def mock_free(notes: str) -> str:
    """The notes without the mock marker at their head: the reason a row
    gives, which the marker alone is not."""
    if notes.startswith(gh.MOCK_NOTE):
        notes = notes[len(gh.MOCK_NOTE):].lstrip(": ")
    return notes


def _check_mock(rep: Report, folder: Path, seen: Dict[str, object],
                allow_mock: bool) -> None:
    """Rows of the transition table that say MOCK fail the folder unless
    `allow_mock` (a rehearsal folder, built with --allow-mock); either way
    the README must say MOCK DATA at its top, above the operator line."""
    mocked: List[str] = seen["mock_transitions"]
    if not mocked:
        return
    listed = ", ".join(mocked)
    if allow_mock:
        rep.notes.append(
            f"{_count(len(mocked), 'row')} of {gh.TRANSITIONS_FILE} "
            f"{'says' if len(mocked) == 1 else 'say'} MOCK (synthetic hand), "
            f"allowed by --allow-mock: {listed}. This folder is a "
            "rehearsal, not a hand-in.")
    else:
        rep.error(
            f"{_count(len(mocked), 'row')} of {gh.TRANSITIONS_FILE} "
            f"{'says' if len(mocked) == 1 else 'say'} MOCK, a take of a "
            f"synthetic hand recorded with --mock: {listed}. A folder with "
            "mock data is not a hand-in (a rehearsal is checked with "
            "--allow-mock)")
    path = folder / "README.txt"
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        top = next((k for k, x in enumerate(lines)
                    if x.startswith("Operator:")), len(lines))
        if not any(MOCK_BANNER in x for x in lines[:top]):
            rep.error(f"README.txt does not say {MOCK_BANNER} at its top, "
                      f"yet {gh.TRANSITIONS_FILE} has mock rows: {listed}")


def _count(n: int, one: str) -> str:
    return f"{n} {one}" if n == 1 else f"{n} {one}s"


def readme_section(lines: Sequence[str], k: int) -> str:
    """Section `k` of the README, its lines joined by single spaces (a
    sentence wrapped over two lines reads as one)."""
    head = f"{k}. {SECTION_TITLES[k - 1]}"
    end = (f"{k + 1}. {SECTION_TITLES[k]}" if k < len(SECTION_TITLES)
           else None)
    try:
        a = lines.index(head)
    except ValueError:
        return ""
    b = lines.index(end, a) if end in lines[a:] else len(lines)
    return " ".join(" ".join(lines[a:b]).split())


def _check_readme(rep: Report, folder: Path, n_takes: int,
                  n_types: int, n_images: int = 0,
                  keyed: bool = False,
                  summary: Sequence[Dict[str, str]] = ()) -> None:
    path = folder / "README.txt"
    if not path.is_file():
        rep.error("README.txt is missing")
        return
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    at = -1
    for k, title in enumerate(SECTION_TITLES, 1):
        head = f"{k}. {title}"
        try:
            at = lines.index(head, at + 1)
        except ValueError:
            rep.error(f"README.txt: section \"{head}\" is missing or out of "
                      "order")
            return
    # the map of what was asked for, before the list of contents
    top = lines[:lines.index("CONTENTS")] if "CONTENTS" in lines else []
    unplaced = [a for a in gh.ASKED_FOR if f"  {a}" not in top]
    if gh.ASKED_TITLE not in top or unplaced:
        rep.error("README.txt does not say at its top where to find: "
                  + ", ".join(unplaced or gh.ASKED_FOR))
    # the line under the title on pictures: what the folder holds
    head = " ".join(" ".join(top[:top.index(gh.ASKED_TITLE)]
                             if gh.ASKED_TITLE in top else top).split())
    if n_images:
        if ("Images: one infrared picture per recording" not in head
                or "No other photographs are included." not in head
                or "No photographs are included." in head):
            rep.error("README.txt: the line under its title does not say "
                      "that the folder holds the infrared pictures")
    elif "No photographs are included." not in head:
        rep.error("README.txt: the line under its title does not say that "
                  "no photograph is included")
    # what the pictures replaced: the claims the data does not bear out
    if "No object was held" in readme_section(lines, 1):
        rep.error("README.txt section 1 says \"No object was held\", which "
                  "the pictures do not bear out")
    if "held open" in readme_section(lines, 16):
        rep.error("README.txt section 16 says the other hand was \"held "
                  "open\", which is not measured")
    if keyed:
        from xr_hand import joint_key as jk
        frames = readme_section(lines, 11)
        rule = " ".join(jk.NAMING_RULE.split())
        unnamed = [n for n in JOINT_NAMES
                   if re.search(rf"\b{re.escape(n)}\b", frames) is None]
        if rule not in frames or unnamed:
            rep.error("README.txt section 11 does not carry the joint key: "
                      + ("its naming rule" if rule not in frames else "")
                      + (("; " if rule not in frames else "") + "the rows "
                         + ", ".join(unnamed) if unnamed else ""))
    m = re.search(r"^  Accepted recordings \(takes\)\s+(\d+);", text, re.M)
    if m is None or int(m.group(1)) != n_takes:
        rep.error("README.txt: the accepted-recordings count is not the "
                  f"summary's {n_takes}")
    m = re.search(r"^  Grasp types in the protocol\s+(\d+);", text, re.M)
    if m is None or int(m.group(1)) != n_types:
        rep.error("README.txt: the grasp-type count is not the audit "
                  f"table's {n_types}")
    if re.search(r"\d+\s+frames\s*=\s*\d+(\.\d+)?\s*s(ec|\b)", text):
        rep.error("README.txt equates a number of frames with seconds")
    unexplained = [c for c in gh.SUMMARY_COLUMNS
                   if re.search(rf"^    {re.escape(c)}: ", text, re.M) is None]
    if unexplained:
        rep.error("README.txt does not explain the summary column(s) "
                  + ", ".join(unexplained))
    for status in ("STATIC-POSE TASK: ", "FULL MULTI-TAKE DATA COLLECTION: ",
                   "TRANSITION TIMING: "):
        if status not in text:
            rep.error(f"README.txt does not give its \"{status.strip()}\" "
                      "status")
    _check_objects(rep, lines, summary)


# Section 1's lines on what was seen in the hands, read by eye on the
# recorder's uncut stills (`grasp_handin.objects_seen`), read in the section
# joined into one line.
OBJECTS_HEAD = "What was seen in the hands, read by eye on "
OBJECTS_COUNTS = (
    ("in or against the recorded hand", re.compile(
        r"(\d+) recordings? with something in or against the recorded "
        r"hand[:.]")),
    ("in the other hand", re.compile(
        r"(\d+) with something in the other hand[:.]")),
    ("cannot tell", re.compile(
        r"(\d+) where the picture does not let one tell[:.]")),
    ("nothing seen", re.compile(
        r"the other (\d+) with a picture: nothing seen in the hand\.")))
UNREAD_RE = re.compile(r"(\d+) with a picture not yet read by eye: "
                       r"([^.]*)\.")
NO_PICTURE_RE = re.compile(r"(\d+) without a picture of the hand: ")
UNREAD_NOTE = "the infrared picture has not been read by eye"


def _check_objects(rep: Report, lines: Sequence[str],
                   summary: Sequence[Dict[str, str]]) -> None:
    """Section 1's counts of what the pictures show add up to the summary's
    rows: in or against the hand + in the other hand + cannot tell +
    nothing seen + not yet read (0 without its line) + without a picture
    (0 without its line); and every row whose notes say its picture has not
    been read by eye is named on the not-yet-read line, which names no
    other."""
    s1 = readme_section(lines, 1)
    m = UNREAD_RE.search(s1)
    listed = ([x.strip() for x in m.group(2).split(",") if x.strip()]
              if m else [])
    if m is not None and int(m.group(1)) != len(listed):
        rep.error(f"README.txt section 1: {m.group(1)} pictures not yet "
                  f"read by eye, and {len(listed)} named")
    noted = {f"{r['grasp_name']}_{r['hand']}_take{r['take']}" for r in summary
             if UNREAD_NOTE in r.get("notes", "")}
    uncounted = sorted(noted - set(listed))
    if uncounted:
        rep.error("README.txt section 1 does not count among the pictures "
                  "not yet read by eye: " + ", ".join(uncounted))
    unnoted = sorted(set(listed) - noted)
    if unnoted:
        rep.error("README.txt section 1 names as not yet read by eye a "
                  "recording whose notes do not say so: "
                  + ", ".join(unnoted))
    if OBJECTS_HEAD not in s1:
        return
    counts = {}
    for name, pattern in OBJECTS_COUNTS:
        found = pattern.search(s1)
        if found is None:
            rep.error(f"README.txt section 1 does not give its count "
                      f"\"{name}\"")
            return
        counts[name] = int(found.group(1))
    bare = NO_PICTURE_RE.search(s1)
    counts["not yet read"] = int(m.group(1)) if m else 0
    counts["without a picture"] = int(bare.group(1)) if bare else 0
    if sum(counts.values()) != len(summary):
        rep.error("README.txt section 1: "
                  + " + ".join(f"{n} {k}" for k, n in counts.items())
                  + f" = {sum(counts.values())}, the summary has "
                  f"{len(summary)} recordings")


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff",
                  ".webp", ".mp4")
IMAGE_HEADS = (b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM")


def looks_like_image(path: Path) -> bool:
    """A picture by its name or by its first bytes."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return head.startswith(IMAGE_HEADS) or head[4:8] == b"ftyp"


def pdf_pages(rep: Report, path: Path, name: str) -> Optional[int]:
    """The pages of a PDF of the folder, or None (an error when it does not
    open, a note when pypdf is not there to count them)."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)
    except ImportError:
        rep.notes.append(f"{name}: pages not counted (pypdf is not "
                         "installed)")
    except Exception as e:                        # a damaged file
        rep.error(f"{name} does not open ({e})")
    return None


def _check_pictures(rep: Report, folder: Path,
                    summary: List[Dict[str, str]],
                    audit: List[Dict[str, str]],
                    seen: Dict[str, object]) -> None:
    """Every picture a summary row names: there, a PNG that opens, not all
    black; the audit's count of them per grasp; and grasp_images.pdf with
    one page per grasp type that has one."""
    from PIL import Image

    imaged = [r for r in summary if r.get("image_file")]
    seen["missing_images"] = len(summary) - len(imaged)
    for a in audit:
        n = sum(1 for r in imaged if r["grasp_name"] == a["grasp_name"])
        if num(a.get("recordings_with_an_image")) != n:
            rep.error(f"{a['grasp_name']}: recordings_with_an_image is "
                      f"{a.get('recordings_with_an_image')!r}, the summary "
                      f"names {n} picture(s)")
    for r in imaged:
        name = f"{r['grasp_name']} take {r['take']}"
        path = folder / r["image_file"]
        want = gh.rel(gh.GRASPS_DIR, gh.IMAGES_DIR, "")
        if not r["image_file"].startswith(want) or not \
                r["image_file"].endswith(".png"):
            rep.error(f"{name}: image_file {r['image_file']} is not a .png "
                      f"under {want}")
            continue
        if not path.is_file() or path.stat().st_size == 0:
            rep.error(f"{name}: its picture {r['image_file']} is missing or "
                      "empty")
            continue
        try:
            with Image.open(path) as im:
                kind = im.format
                im.load()
                grey = np.asarray(im.convert("L"))
        except Exception as e:                    # not an image at all
            rep.error(f"{name}: {r['image_file']} does not open as a "
                      f"picture ({type(e).__name__})")
            continue
        if kind != "PNG":
            rep.error(f"{name}: {r['image_file']} is a {kind}, not a PNG")
        elif not grey.size or int(grey.max()) == 0:
            rep.error(f"{name}: {r['image_file']} is all black")
        else:
            seen["images"] += 1
    pdf = folder / gh.GRASPS_DIR / gh.IMAGES_DIR / gh.IMAGES_PDF
    types = len({r["grasp_name"] for r in imaged})
    if not imaged:
        return
    if not pdf.is_file():
        rep.error(f"{gh.IMAGES_PDF} is missing")
        return
    pages = pdf_pages(rep, pdf, gh.IMAGES_PDF)
    seen["image_pdf_pages"] = pages
    if pages is not None and pages != types:
        rep.error(f"{gh.IMAGES_PDF} has {pages} page(s) for {types} grasp "
                  "type(s) with a picture")


def _check_key(rep: Report, folder: Path, seen: Dict[str, object]) -> bool:
    """The joint key beside the joint-frame tables, when there are tables:
    26 rows in the tables' order, each given angle of the paper in the row
    and column the angle table takes it from, and its one-page PDF. True
    when there are tables, so the README must carry the key."""
    if not seen["joint_tables"]:
        return False
    from xr_hand import joint_key as jk

    base = folder / gh.GRASPS_DIR / gh.JOINT_FRAMES_DIR
    path = base / gh.KEY_CSV
    if not path.is_file():
        rep.error(f"{gh.KEY_CSV} is missing")
    else:
        header, rows = read_csv(path)
        seen["key_rows"] = len(rows)
        if header != list(jk.KEY_COLUMNS):
            rep.error(f"{gh.KEY_CSV}: unexpected columns")
        elif [r["row"] for r in rows] != list(JOINT_NAMES):
            rep.error(f"{gh.KEY_CSV}: its rows are not the {len(JOINT_NAMES)} "
                      "joints of a joint-frame table in their order")
        else:
            by = {r["row"]: r for r in rows}
            column = {"flex_deg": "paper_angle_flex",
                      "abd_deg": "paper_angle_abd"}
            for name in gh.GIVEN_NAMES:
                joint, cell = jf.DOF24_SOURCE[name]
                held = re.split(r"[\s,;/]+", by[joint].get(
                    column.get(cell, ""), "") or "")
                if name not in held:
                    rep.error(f"{gh.KEY_CSV}: {name} is not in the "
                              f"{column.get(cell, cell)} of row {joint}, "
                              "where the angle table takes it from")
    pdf = base / gh.KEY_PDF
    if not pdf.is_file():
        rep.error(f"{gh.KEY_PDF} is missing")
    else:
        pages = pdf_pages(rep, pdf, gh.KEY_PDF)
        if pages is not None and pages != 1:
            rep.error(f"{gh.KEY_PDF} has {pages} pages, not one")
    return True


def _check_files(rep: Report, folder: Path, summary: List[Dict[str, str]],
                 audit: List[Dict[str, str]], seen: Dict[str, object]) -> None:
    """Nothing in the two dataset folders that no table accounts for."""
    known = {gh.rel(gh.GRASPS_DIR, n) for n in (gh.SUMMARY_FILE,
                                                gh.AUDIT_FILE, gh.ANGLES_FILE)}
    known.add(gh.rel(gh.TRANSITIONS_DIR, gh.TRANSITIONS_FILE))
    known.add(gh.rel(gh.GRASPS_DIR, gh.JOINT_FRAMES_DIR, "joint_frames.pdf"))
    if seen["joint_tables"]:
        known |= {gh.rel(gh.GRASPS_DIR, gh.JOINT_FRAMES_DIR, n)
                  for n in (gh.KEY_CSV, gh.KEY_PDF)}
    if any(r.get("image_file") for r in summary):
        known.add(gh.rel(gh.GRASPS_DIR, gh.IMAGES_DIR, gh.IMAGES_PDF))
    for r in summary:
        for column in ("raw_jsonl_file", "representative_keypoints_file",
                       "all_frames_file", "joint_frames_file", "image_file",
                       "transition_capture_file"):
            if r.get(column):
                known.add(r[column])
    known |= set(seen.get("transition_files") or ())
    extra = []
    for sub in (gh.GRASPS_DIR, gh.TRANSITIONS_DIR):
        base = folder / sub
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.relative_to(folder).as_posix() not in known:
                    extra.append(p.relative_to(folder).as_posix())
    for name in extra:
        rep.error(f"{name}: in the folder, in no table")
    # anywhere in the folder: an image no summary row names
    pictures = {r["image_file"] for r in summary if r.get("image_file")}
    for p in sorted(folder.rglob("*")):
        name = p.relative_to(folder).as_posix()
        if p.is_file() and name not in pictures and looks_like_image(p) \
                and name not in extra:
            rep.error(f"{name}: an image in the folder that no summary row "
                      "names")
    pdf = folder / gh.GRASPS_DIR / gh.JOINT_FRAMES_DIR / "joint_frames.pdf"
    tables = len(seen["joint_tables"])
    if tables and not pdf.is_file():
        rep.error("joint_frames.pdf is missing")
    elif pdf.is_file():
        pages = pdf_pages(rep, pdf, "joint_frames.pdf")
        if pages is not None:
            seen["pdf_pages"] = pages
            if pages != tables:
                rep.error(f"joint_frames.pdf has {pages} page(s) for "
                          f"{tables} joint-frame table(s)")


def validate(folder: Path, session: Optional[Path] = None,
             allow_mock: bool = False) -> Report:
    """Check a built grasp hand-in folder; `session` adds session.json's
    own count of accepted takes and its protocol's grasp list. A row of
    the transition table that says MOCK (a take of a mock transition
    session, built with --allow-mock) fails the folder unless
    `allow_mock`, which only turns that error into a note."""
    folder = gh.long_path(folder)     # a deep folder's files are past MAX_PATH
    rep = Report(folder=folder)
    grasps = folder / gh.GRASPS_DIR
    for name in (gh.SUMMARY_FILE, gh.AUDIT_FILE):
        if not (grasps / name).is_file():
            rep.error(f"{gh.GRASPS_DIR}/{name} is missing")
    if rep.errors:
        return rep
    s_head, summary = read_csv(grasps / gh.SUMMARY_FILE)
    a_head, audit = read_csv(grasps / gh.AUDIT_FILE)
    if s_head != gh.SUMMARY_COLUMNS:
        rep.error(f"{gh.SUMMARY_FILE}: unexpected columns")
    if a_head != gh.AUDIT_COLUMNS:
        rep.error(f"{gh.AUDIT_FILE}: unexpected columns")
    if rep.errors:
        return rep

    seen: Dict[str, object] = {
        "durations": [], "noted": [], "rule_differs": [], "frames": {},
        "all_noted": [],
        "rep_line": {}, "joint_tables": {}, "missing_raw": 0,
        "bad_duration": 0,
        "other_hand": 0, "identity": 0, "missing_keypoints": 0,
        "missing_all_frames": 0, "missing_joint_frames": 0,
        "missing_angles": 0, "nan_angles": 0, "bad_xyz": 0, "no_source": 0,
        "label_errors": 0, "rule_same": 0, "rule_checked": 0, "ready": 0,
        "formula_same": 0, "formula_checked": 0, "stillness_same": 0,
        "stillness_checked": 0, "speeds": [], "chain_gap": 0.0,
        "angles_same": 0, "angles_checked": 0, "spreads": [],
        "questionable_transitions": 0, "mock_transitions": [],
        "transitions": 0, "valid_transitions": 0, "between_grasps": 0,
        "pdf_pages": None, "images": 0, "missing_images": 0,
        "image_pdf_pages": None, "key_rows": None}
    keys = [(r["grasp_name"], r["take"]) for r in summary]
    if len(set(keys)) != len(keys):
        rep.error(f"{gh.SUMMARY_FILE}: two rows for the same grasp and take")
    for row in summary:
        if not row.get("source_paper") or not row.get("source_figure"):
            rep.error(f"{row['grasp_name']} take {row['take']}: no paper or "
                      "figure in the summary")
            seen["no_source"] += 1
        if row.get("status") != "accepted":
            rep.error(f"{row['grasp_name']} take {row['take']}: status "
                      f"{row.get('status')!r} in a folder of accepted takes")
        _check_take(rep, folder, row, seen)
    _check_per_grasp(rep, folder, summary, audit, seen)
    _check_angles(rep, folder, summary, audit, seen)
    _check_transitions(rep, folder, summary, seen)
    _check_pictures(rep, folder, summary, audit, seen)
    keyed = _check_key(rep, folder, seen)
    _check_readme(rep, folder, len(summary), len(audit),
                  sum(1 for r in summary if r.get("image_file")), keyed,
                  summary)
    _check_mock(rep, folder, seen, allow_mock)
    _check_files(rep, folder, summary, audit, seen)
    if seen["rule_differs"]:
        rep.error("the representative frame is not the frame nearest the "
                  "mean hand of the least-moving window in: "
                  + ", ".join(seen["rule_differs"]))

    target = sum(int(num(a.get("target_takes")) or 0) for a in audit)
    per = {a["grasp_name"]: int(num(a.get("accepted_recordings")) or 0)
           for a in audit}
    missing = {a["grasp_name"]: int(num(a.get("missing_takes")) or 0)
               for a in audit}
    if session is not None:
        meta = json.loads((Path(session) / "session.json").read_text(
            encoding="utf-8"))
        accepted = sum(1 for t in meta.get("takes", []) if t.get("accepted"))
        if accepted != len(summary):
            rep.error(f"session.json has {accepted} accepted take(s), the "
                      f"summary {len(summary)}")
        seen["session_accepted"] = accepted
    durations = sorted(seen["durations"])
    rep.facts = {
        "grasp_types": len(audit), "recordings": len(summary),
        "target": target, "takes_per_grasp": per,
        "missing_takes": sum(missing.values()),
        "missing_by_grasp": {g: n for g, n in missing.items() if n},
        "ready": seen["ready"],
        "duration": ((durations[0], float(np.median(durations)),
                      durations[-1]) if durations else None),
        **{k: seen[k] for k in (
            "bad_duration", "other_hand", "identity", "missing_keypoints",
            "missing_all_frames", "missing_joint_frames", "missing_angles",
            "nan_angles", "bad_xyz", "no_source", "label_errors",
            "rule_same", "rule_checked", "formula_same", "formula_checked",
            "stillness_same", "stillness_checked", "transitions",
            "valid_transitions", "between_grasps", "pdf_pages", "noted",
            "chain_gap", "all_noted", "angles_same", "angles_checked",
            "questionable_transitions", "spreads", "images",
            "missing_images", "image_pdf_pages", "key_rows")},
        "static_speed": ((min(seen["speeds"]),
                          float(np.median(seen["speeds"])),
                          max(seen["speeds"])) if seen["speeds"] else None),
        "static_complete": bool(audit) and seen["ready"] == len(audit),
        "multi_complete": bool(audit) and sum(missing.values()) == 0,
        "mock_transitions": len(seen["mock_transitions"])}
    return rep


def render(rep: Report, checked: str = "") -> str:
    """The report as text: the counts, the two statuses, the verdict."""
    f = rep.facts
    L = ["Validation of the grasp hand-in folder", "=" * 78]
    if checked:
        L.append(f"Checked: {checked}")
    L.append(f"Folder: {rep.folder.name}")
    L.append("")
    if not f:
        L += ["The folder could not be read:"] + [f"  {e}" for e in rep.errors]
        L += ["", "FOLDER CHECK: FAILED"]
        return "\n".join(L) + "\n"
    per: Dict[int, int] = {}
    for n in f["takes_per_grasp"].values():
        per[n] = per.get(n, 0) + 1
    d = f["duration"]
    rows = [
        ("canonical grasp types", f["grasp_types"]),
        ("accepted recordings", f["recordings"]),
        ("expected recording target (our replication target)",
         f["target"]),
        ("takes per grasp", ", ".join(f"{k} take(s): {per[k]} grasp(s)"
                                      for k in sorted(per, reverse=True))),
        ("missing takes (against the replication target)",
         f["missing_takes"]),
        ("recordings with invalid or missing duration", f["bad_duration"]),
        ("recordings with another hand detected", f["other_hand"]),
        ("recordings with possible identity ambiguity", f["identity"]),
        ("missing _keypoints.txt files", f["missing_keypoints"]),
        ("missing all-frame datasets", f["missing_all_frames"]),
        ("missing joint-frame CSVs", f["missing_joint_frames"]),
        ("missing angle values", f["missing_angles"]),
        ("NaN or infinite angle values", f["nan_angles"]),
        ("malformed XYZ coordinates", f["bad_xyz"]),
        ("missing source-paper labels", f["no_source"]),
        ("recordings with a wrong label", f["label_errors"]),
        ("grasp types whose every recording carries a note",
         "none" if not f["all_noted"] else
         f"{len(f['all_noted'])}: " + ", ".join(f["all_noted"])),
        ("duration, minimum / median / maximum",
         "not measured" if d is None
         else f"{d[0]:.3f} / {d[1]:.3f} / {d[2]:.3f} s"),
        ("representative frame = nearest the mean hand of the stillest "
         "window", f"{f['rule_same']} of {f['rule_checked']} recordings"),
        ("joint-frame table = the README's formulas, again",
         f"{f['formula_same']} of {f['formula_checked']} recordings"),
        ("stillness, shape and hand size = the README's formulas, again",
         f"{f['stillness_same']} of {f['stillness_checked']} recordings"),
        ("angle table = the README's formulas, again",
         f"{f['angles_same']} of {f['angles_checked']} recordings"),
        ("of the paper's 24 angles, not given by this tracker",
         f"{len(gh.NOT_GIVEN)} per recording: " + ", ".join(gh.NOT_GIVEN)),
        ("between-take spread of an angle, minimum / median / maximum",
         "not measured" if not f["spreads"] else
         "{:.1f} / {:.1f} / {:.1f} degrees".format(
             min(f["spreads"]), float(np.median(f["spreads"])),
             max(f["spreads"]))),
        ("chained joints against abs26, largest difference",
         f"{f['chain_gap']:.4f} mm"),
        ("static interval speed, minimum / median / maximum",
         "not measured" if f["static_speed"] is None else
         "{:.1f} / {:.1f} / {:.1f} mm/s".format(*f["static_speed"])),
        ("joint_frames.pdf pages",
         "not counted" if f["pdf_pages"] is None else f["pdf_pages"]),
        ("pictures cut to the hand: PNGs that open and are not black",
         f["images"]),
        ("recordings without a picture", f["missing_images"]),
        ("grasp_images.pdf pages",
         "none" if f["image_pdf_pages"] is None else f["image_pdf_pages"]),
        ("joint key rows",
         "none" if f["key_rows"] is None else f["key_rows"]),
        ("recordings of a change of pose", f"{f['transitions']}: "
         f"{f['valid_transitions']} valid, "
         f"{f['questionable_transitions']} questionable"
         + (f"; {f['mock_transitions']} of them MOCK (synthetic hand)"
            if f.get("mock_transitions") else "")),
    ]
    w = max(len(k) for k, _v in rows)
    L += [f"  {k:<{w}}  {v}" for k, v in rows]
    if f["missing_by_grasp"]:
        L += ["", "Takes still to record for the replication target, by "
              "grasp:"]
        L += [f"  {g}: {n}" for g, n in f["missing_by_grasp"].items()]
    if f["noted"]:
        L += ["", "Accepted recordings that carry a note (kept as recorded):"]
        L += [f"  {x}" for x in f["noted"]]
    if rep.notes:
        L += ["", "Notes:"] + [f"  {x}" for x in rep.notes]
    L.append("")
    L.append("STATIC-POSE TASK: "
             + (gh.COMPLETE if f["static_complete"] else gh.INCOMPLETE)
             + f" ({f['ready']} of {f['grasp_types']} grasp types have a "
             "recording with its keypoints, frame-by-frame file, joint "
             "frames and angles)")
    L.append("FULL MULTI-TAKE DATA COLLECTION: "
             + (gh.COMPLETE if f["multi_complete"] else gh.INCOMPLETE)
             + f" ({f['recordings']} of {f['target']} recordings against "
             f"our replication target; {f['missing_takes']} still to "
             "record)")
    L.append("TRANSITION TIMING: "
             + ("PARTIAL" if f["valid_transitions"] else "MISSING")
             + f" ({f['valid_transitions']} valid and "
             f"{f['questionable_transitions']} questionable of "
             f"{f['transitions']} recording(s) of a change of pose; "
             + (f"{f['between_grasps']} of them between two grasp types"
                if f["between_grasps"] else
                "none between two grasp types") + ")")
    L.append("")
    if rep.errors:
        L += [f"Errors ({len(rep.errors)}):"] + [f"  {e}" for e in rep.errors]
        L.append("")
    L.append("FOLDER CHECK: " + ("PASSED" if rep.passed else
                                 f"FAILED ({len(rep.errors)} error(s))"))
    return "\n".join(L) + "\n"
