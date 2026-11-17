"""Replay a recorded hand as the tracker recorded it, on the camera's clock.

The numerical core of the replay viewer and of the batch fidelity run
(`docs/replay_plan.md`, section "leap_hand.replay_core"). Pure: it reads
files, never writes one, and draws nothing, so every number the viewer or
the batch report shows can be reproduced from a Python prompt.

What "as recorded" means, and how each part is checked:

  pose      `reconstruct` chains the 26 parent-relative joints the recorder
            wrote through the repository's own matrix chain
            (`xr_hand.kinematics.forward_kinematics_full`). The recorder also
            wrote `abs26`, the 26 world positions, so the two can be compared
            joint by joint (`frame_fidelity`). That comparison is the replay
            fidelity number; it cannot beat abs26's rounding (1e-6 m).
  chain     `reconstruct_quaternion_chain` redoes the chain with quaternion
            arithmetic alone. A cross-check of two code paths over the same
            data, not an independent measurement.
  bones     each bone's reconstructed orientation (its LeapC -z axis) against
            the direction between its two recorded positions. Orientation and
            position were written separately, so this one is independent.
  time      `timestamp_us` (the camera's acquisition clock) unless it is
            defective on the selected frames; then `capture_time`, then the
            writer's `wall_time`, then frame index at the nominal rate. Every
            fallback is named in `Timeline.reason` and as a `clock_defect`
            event. Frames are never reordered, and never interpolated across
            a hand-id change, a time gap or an invalid-tracking line (a
            status-0 line of the selected hand between two selected frames).

None of this is physical accuracy against the person's hand: that needs a
reference measurement this dataset does not have.
"""
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from xr_hand.joints import JOINT_INDEX, JOINT_NAMES, HandFrame, Joint
from xr_hand.kinematics import (
    PARENT,
    forward_kinematics_full,
    mat3_to_quat,
    quat_canonical,
    quat_conjugate,
    quat_mul,
    quat_normalize,
    quat_to_mat3,
)

CLOCK_PREFERENCE = ("timestamp_us", "capture_time", "wall_time")
GAP_THRESHOLD_S = 0.05          # about 4.5 periods at 90 Hz
DEFAULT_RATE_HZ = 90.0          # LeapC's nominal rate when a line carries none
SIDES = ("left", "right")
REFERENCES = ("camera", "wrist", "palm")
MODES = ("hold", "interpolate")
WRIST = JOINT_INDEX["WRIST"]
PALM = JOINT_INDEX["PALM"]

# The 19 bones whose direction the recording gives twice: the bone starting at
# joint i ends at joint i + 1 (thumb metacarpal..distal, each finger's
# metacarpal..distal). Named by the joint they start at, which in LeapC's
# convention carries that bone's rotation (`leap_hand.to_openxr`).
BONES19 = [(i, i + 1) for i, n in enumerate(JOINT_NAMES)
           if n.endswith(("_METACARPAL", "_PROXIMAL", "_INTERMEDIATE", "_DISTAL"))]
BONE_NAMES = [JOINT_NAMES[i] for i, _ in BONES19]
assert len(BONES19) == 19 and all(PARENT[c] == p for p, c in BONES19)


def _depth(i: int) -> int:
    d = 0
    while i in PARENT:
        i, d = PARENT[i], d + 1
    return d


_CHAIN_ORDER = sorted(range(len(JOINT_NAMES)), key=_depth)  # parents first

# Declared before any data was looked at (docs/replay_plan.md). `gate` marks
# the three that decide `within_tolerance`; the others are reported beside it.
TOLERANCES: Dict[str, dict] = {
    "position_m": {
        "value": 1e-6, "gate": True,
        "quantity": "FK position vs recorded abs26, per joint (Euclidean)",
        "why": ("abs26 is rounded to 1e-6 m per coordinate "
                "(leap_hand.recorder._round); worst case sqrt(3)/2 um = 0.87 um; "
                "float64 chain error about 1e-15 m")},
    "chain_deg": {
        "value": 1e-6, "gate": True,
        "quantity": "matrix chain vs quaternion chain, world orientation per joint",
        "why": "same inputs, float64, at most five compositions: expected 1e-12"},
    "chain_m": {
        "value": 1e-9, "gate": True,
        "quantity": "matrix chain vs quaternion chain, world position per joint",
        "why": "same inputs, float64, at most five compositions: expected 1e-12"},
    "bone_direction_deg_expectation": {
        "value": 0.05, "gate": False,
        "quantity": "bone axis (-z) vs recorded bone vector, 19 bones (reported)",
        "why": ("LeapC defines a bone's local -z as prev_joint to next_joint; "
                "abs26 rounding over the shortest recorded bone (7.36 mm, INDEX_DISTAL "
                "in p1_lateral take 2) is at most "
                "0.0135 degrees; the plan assumed 14 mm / 0.004 degrees before measuring")},
    "clock_interval_ms": {
        "value": 0.5, "gate": False,
        "quantity": "capture_time vs timestamp_us intervals between consecutive "
                    "frames (reported)",
        "why": "capture_time rounded to 1e-6 s, frame_age_us to 0.1 us, two clocks"},
    "quat_norm": {
        "value": 1e-9, "gate": False,
        "quantity": "norm of each relative quaternion, deviation from unity (reported)",
        "why": "the relative quaternions were normalised when written"},
}


# --- data -----------------------------------------------------------------------
@dataclass
class SourceFrame:
    """One selected line, as read. Arrays are in JOINT_NAMES order."""
    index: int                          # position in the selected sequence
    line_no: int                        # 1-based line number in the file
    frame_id: int
    hand_id: Optional[int]
    hand_side: str
    status: Optional[int]
    wall_time: Optional[float]          # writer clock (None = missing)
    capture_time: Optional[float]
    timestamp_us: Optional[int]
    rel_pos: np.ndarray                 # (26, 3) m, parent-relative
    rel_quat: np.ndarray                # (26, 4) XYZW, as recorded
    abs_recorded: Optional[np.ndarray]  # (26, 3) m, the recorder's abs26
    raw: dict = field(default_factory=dict)   # the other scalar fields


@dataclass
class ClockReport:
    field: str
    present: int
    missing: int
    repeated: int
    nonmonotonic: int
    valid: bool                         # present everywhere, strictly increasing
    note: str
    first_defect: Optional[int] = None  # index of the first missing/repeat/decrease


@dataclass
class Timeline:
    source: str                         # a CLOCK_PREFERENCE field or "frame_index_nominal"
    fallback: bool                      # True whenever source is not timestamp_us
    reason: str
    t: np.ndarray                       # seconds from the first selected frame
    reports: Dict[str, ClockReport]
    nominal_period_s: float
    cross_check: dict


@dataclass
class Event:
    """Something the viewer marks on the timeline.

    `index` is the selected frame the marker is drawn at: for a boundary the
    first frame after it, for an excluded line the next selected frame (the
    last one when nothing follows). An `invalid_status` event carries the
    excluded line's own frame_id in both frame_id fields.
    """
    kind: str          # gap | id_change | dropped_frames | invalid_status | clock_defect
    index: int
    frame_id_before: Optional[int]
    frame_id_after: Optional[int]
    dt_s: Optional[float]
    detail: str


@dataclass
class Segment:
    start: int                          # inclusive selected-frame indices
    end: int
    hand_id: Optional[int]


@dataclass
class Recording:
    path: Path
    kind: str                           # static | transition | file
    name: str
    take: Optional[int]
    hand_side: str
    hand_ids: Optional[List[int]]
    provenance: str
    frames: List[SourceFrame]
    lines_total: int
    lines_excluded: Dict[str, int]      # other_hand, other_id, invalid_status
    timeline: Timeline
    segments: List[Segment]
    events: List[Event]
    gap_threshold_s: float
    metadata: dict


@dataclass
class Reconstruction:
    pos: np.ndarray                     # (26, 3) m, world
    rot: np.ndarray                     # (26, 3, 3), columns = the joint's axes
    quat: np.ndarray                    # (26, 4) XYZW, w >= 0


@dataclass
class DisplayedPose:
    """What the viewer draws at one media time, and where it came from.

    recorded/held: i0 == i1 is the frame shown, alpha 0. interpolated: i0 and
    i1 bracket t, alpha is i1's weight. unavailable: everything None.
    `frame` is the frame at i0.
    """
    pos: Optional[np.ndarray]
    rot: Optional[np.ndarray]
    quat: Optional[np.ndarray]
    source: str                         # recorded | interpolated | held | unavailable
    i0: Optional[int]
    i1: Optional[int]
    alpha: Optional[float]
    frame: Optional[SourceFrame]
    note: str


# --- reading ----------------------------------------------------------------------
def _num(v, kind=float):
    """A finite number, or None: absent, null, non-numeric and NaN are all missing."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = kind(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def _read_lines(path) -> List[Tuple[int, dict]]:
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for no, text in enumerate(f, start=1):
            if not text.strip():
                continue
            try:
                d = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {no}: not JSON ({e.msg})") from None
            if not isinstance(d, dict):
                raise ValueError(f"{path} line {no}: not a JSON object")
            out.append((no, d))
    return out


def read_jsonl(path) -> List[dict]:
    """Every non-blank line of a JSONL file. A malformed line is an error:
    skipping it would drop a frame without anyone being told."""
    return [d for _, d in _read_lines(path)]


_BROKEN_OUT = {"joints", "abs26", "wall_time", "capture_time", "timestamp_us",
               "frame_id", "hand_id", "hand_side", "status"}


def _source_frame(d: dict, index: int, line_no: int) -> SourceFrame:
    """One line -> SourceFrame. Joints are matched by name, never by position."""
    by_name = {str(j.get("name")): j for j in d.get("joints") or []}
    missing = [n for n in JOINT_NAMES if n not in by_name]
    if missing:
        raise ValueError(f"line {line_no}: joints missing: {', '.join(missing)}")
    frame_id = _num(d.get("frame_id"), int)
    if frame_id is None:
        raise ValueError(f"line {line_no}: no frame_id")
    abs_rec = d.get("abs26")
    if abs_rec is not None:
        abs_rec = np.asarray(abs_rec, dtype=float)
        if abs_rec.shape != (26, 3):
            raise ValueError(f"line {line_no}: abs26 has shape {abs_rec.shape}")
    return SourceFrame(
        index=index, line_no=line_no, frame_id=frame_id,
        hand_id=_num(d.get("hand_id"), int), hand_side=str(d.get("hand_side")),
        status=_num(d.get("status"), int), wall_time=_num(d.get("wall_time")),
        capture_time=_num(d.get("capture_time")),
        timestamp_us=_num(d.get("timestamp_us"), int),
        rel_pos=np.array([[float(by_name[n][k]) for k in ("x", "y", "z")]
                          for n in JOINT_NAMES]),
        rel_quat=np.array([[float(by_name[n][k]) for k in ("qx", "qy", "qz", "qw")]
                           for n in JOINT_NAMES]),
        abs_recorded=abs_rec,
        raw={k: v for k, v in d.items()
             if k not in _BROKEN_OUT and not isinstance(v, (list, dict))},
    )


def load_recording(path, hand_side: str, hand_ids: Optional[Sequence[int]] = None, *,
                   provenance: str = "", gap_threshold_s: float = GAP_THRESHOLD_S,
                   kind: str = "file", name: Optional[str] = None,
                   take: Optional[int] = None,
                   metadata: Optional[dict] = None) -> Recording:
    """Read one JSONL file and select one hand: the Recording the viewer plays.

    Selected: `hand_side == hand_side`, `hand_id in hand_ids` (any id when
    None) and `status != 0`. Every other line is counted in `lines_excluded`;
    a status-0 line of the selected hand is also an `invalid_status` event,
    and when it sits between two selected frames it ends a segment there
    (`find_segments`): playback never interpolates across it. The file order
    is kept: a clock that disagrees with it is rejected by `build_timeline`,
    the frames are not sorted to suit it.
    """
    if hand_side not in SIDES:
        raise ValueError(f"hand_side must be one of {SIDES}, got {hand_side!r}")
    if isinstance(hand_ids, str):      # the metadata tables write "573 574"
        hand_ids = hand_ids.split()
    ids = None if hand_ids is None else [int(h) for h in hand_ids]
    path = Path(path)
    lines = _read_lines(path)
    frames: List[SourceFrame] = []
    events: List[Event] = []
    excluded = {"other_hand": 0, "other_id": 0, "invalid_status": 0}
    invalid_before: List[int] = []     # the selected frame each status-0 line precedes
    for line_no, d in lines:
        if d.get("hand_side") != hand_side:
            excluded["other_hand"] += 1
        elif ids is not None and _num(d.get("hand_id"), int) not in ids:
            excluded["other_id"] += 1
        elif _num(d.get("status"), int) == 0:
            excluded["invalid_status"] += 1
            fid = _num(d.get("frame_id"), int)
            events.append(Event(
                "invalid_status", len(frames), fid, fid, None,
                f"line {line_no}: status 0 (frame_id {fid}, hand_id "
                f"{d.get('hand_id')}), excluded from the pose sequence"))
            invalid_before.append(len(frames))
        else:
            frames.append(_source_frame(d, len(frames), line_no))
    if not frames:
        raise ValueError(f"{path}: no {hand_side} hand line with hand_ids {ids} "
                         f"and status != 0 ({len(lines)} lines, excluded {excluded})")
    for e in events:
        if e.index == 0:
            e.detail += "; before the first selected frame"
        elif e.index > len(frames) - 1:
            e.index = len(frames) - 1
            e.detail += "; after the last selected frame"

    timeline = build_timeline(frames)
    if timeline.fallback:
        i = timeline.reports[CLOCK_PREFERENCE[0]].first_defect or 0
        events.append(Event("clock_defect", i, frames[i - 1].frame_id if i else None,
                            frames[i].frame_id, None, timeline.reason))
    segments, boundary_events = find_segments(frames, timeline.t, gap_threshold_s,
                                              invalid_before=invalid_before)
    events = sorted(events + boundary_events, key=lambda e: e.index)
    return Recording(
        path=path, kind=kind, name=name or path.stem, take=take,
        hand_side=hand_side, hand_ids=ids, provenance=provenance, frames=frames,
        lines_total=len(lines), lines_excluded=excluded, timeline=timeline,
        segments=segments, events=events, gap_threshold_s=gap_threshold_s,
        metadata=dict(metadata or {}))


# --- the dataset's metadata tables --------------------------------------------------
def _table(dataset_root, name: str) -> List[dict]:
    # utf-8-sig: the tables were written for Excel and start with a BOM
    with open(Path(dataset_root) / "metadata" / name, encoding="utf-8-sig",
              newline="") as f:
        return list(csv.DictReader(f))


def list_static(dataset_root) -> List[dict]:
    """The rows of metadata/grasps_summary.csv (one per static take)."""
    return _table(dataset_root, "grasps_summary.csv")


def list_transitions(dataset_root) -> List[dict]:
    """The rows of metadata/pose_transitions.csv (one per transition capture)."""
    return _table(dataset_root, "pose_transitions.csv")


def _find(dataset_root, table: str, key: str, file_col: str, kind: str,
          value: str, take, gap_threshold_s: float) -> Recording:
    rows = [r for r in _table(dataset_root, table)
            if r[key] == value and int(r["take"]) == int(take)]
    if len(rows) != 1:
        raise LookupError(f"metadata/{table}: {len(rows)} rows with {key}={value} "
                          f"take={take}, expected exactly one")
    row = rows[0]
    provenance = (f"metadata/{table}: {key}={value} take={int(take)} "
                  f"hand={row['hand']} hand_ids={row['hand_ids']}")
    return load_recording(Path(dataset_root) / row[file_col], row["hand"],
                          [int(h) for h in row["hand_ids"].split()],
                          provenance=provenance, gap_threshold_s=gap_threshold_s,
                          kind=kind, name=value, take=int(take), metadata=row)


def find_static(dataset_root, grasp: str, take, *,
                gap_threshold_s: float = GAP_THRESHOLD_S) -> Recording:
    """A static take, its hand and hand ids taken from grasps_summary.csv."""
    return _find(dataset_root, "grasps_summary.csv", "grasp_name",
                 "raw_jsonl_file", "static", grasp, take, gap_threshold_s)


def find_transition(dataset_root, to_grasp: str, take, *,
                    gap_threshold_s: float = GAP_THRESHOLD_S) -> Recording:
    """A transition capture, its hand and ids taken from pose_transitions.csv."""
    return _find(dataset_root, "pose_transitions.csv", "to_grasp",
                 "recording_file", "transition", to_grasp, take, gap_threshold_s)


def hold_mask(rec: Recording) -> Optional[np.ndarray]:
    """Frames inside the static interval the metadata row declares, or None
    when the row has none (transitions, loose files)."""
    a = rec.metadata.get("static_interval_first_frame_id")
    b = rec.metadata.get("static_interval_last_frame_id")
    if a in (None, "") or b in (None, ""):
        return None
    a, b = int(float(a)), int(float(b))
    return np.array([a <= f.frame_id <= b for f in rec.frames], dtype=bool)


# --- time ---------------------------------------------------------------------------
def evaluate_clock(values: Sequence, field: str = "") -> ClockReport:
    """How usable one clock is on a sequence of frames.

    Repeats and decreases are counted between consecutive present values, so
    a missing value neither hides one nor creates one. Valid means present on
    every frame and strictly increasing: the only clock playback may use as is.
    """
    present = missing = repeated = decreasing = 0
    first_defect, prev = None, None
    for i, v in enumerate(values):
        x = _num(v)
        bad = x is None
        if bad:
            missing += 1
        else:
            present += 1
            if prev is not None and x == prev:
                repeated, bad = repeated + 1, True
            elif prev is not None and x < prev:
                decreasing, bad = decreasing + 1, True
            prev = x
        if bad and first_defect is None:
            first_defect = i
    valid = present > 0 and missing == repeated == decreasing == 0
    note = (f"present on all {present} frames, strictly increasing" if valid else
            f"missing {missing}, repeated {repeated}, nonmonotonic {decreasing} "
            f"of {present + missing} frames")
    return ClockReport(field, present, missing, repeated, decreasing, valid, note,
                       first_defect)


def _span(values: Sequence[Optional[float]]) -> Optional[float]:
    p = [v for v in values if v is not None]
    return float(p[-1] - p[0]) if len(p) >= 2 else None


def _cross_check(frames: Sequence[SourceFrame]) -> dict:
    """The three clocks against each other; reported, never used for playback."""
    ts = [f.timestamp_us for f in frames]
    cap = [f.capture_time for f in frames]
    wall = [f.wall_time for f in frames]
    diffs = [abs((cap[i + 1] - cap[i]) - (ts[i + 1] - ts[i]) / 1e6) * 1e3
             for i in range(len(frames) - 1)
             if None not in (ts[i], ts[i + 1], cap[i], cap[i + 1])]
    delays = [(w - c) * 1e3 for w, c in zip(wall, cap)
              if w is not None and c is not None]
    camera = _span(ts)
    return {
        "capture_vs_timestamp_interval_max_ms": max(diffs) if diffs else None,
        "capture_vs_timestamp_pairs": len(diffs),
        "within_clock_interval_tolerance": (
            max(diffs) <= TOLERANCES["clock_interval_ms"]["value"] if diffs else None),
        "writer_delay_max_ms": max(delays) if delays else None,
        "writer_delay_median_ms": float(np.median(delays)) if delays else None,
        "writer_span_s": _span(wall),
        "camera_span_s": None if camera is None else camera / 1e6,
        "capture_span_s": _span(cap),
    }


def build_timeline(frames: Sequence[SourceFrame],
                   framerate_hint: Optional[float] = None) -> Timeline:
    """Seconds from the first selected frame, on the best clock that is valid.

    timestamp_us first: the camera's own acquisition clock in integer
    microseconds. capture_time is the same instant on the wall clock, rounded
    to 1e-6 s. wall_time is when the writer got to the line, and it stalls
    (hundreds of ms in this dataset). With none valid, t = index x the
    nominal period (1 / framerate_hint, else the median `framerate` field,
    else 1/90 s).
    """
    reports = {f: evaluate_clock([getattr(fr, f) for fr in frames], field=f)
               for f in CLOCK_PREFERENCE}
    rates = [r for r in (_num(fr.raw.get("framerate")) for fr in frames)
             if r is not None and r > 0]
    rate = (float(framerate_hint) if framerate_hint else
            float(np.median(rates)) if rates else DEFAULT_RATE_HZ)
    period = 1.0 / rate
    source, words = "frame_index_nominal", []
    for f in CLOCK_PREFERENCE:
        if reports[f].valid:
            source = f
            break
        words.append(f"{f} rejected ({reports[f].note})")
    if source == "frame_index_nominal":
        t = np.arange(len(frames)) * period
        words.append(f"no valid clock: t = frame index x {period:.6f} s "
                     f"(1/{rate:g} Hz)")
    elif source == "timestamp_us":     # integer difference first: exact
        t = np.array([fr.timestamp_us - frames[0].timestamp_us for fr in frames],
                     dtype=float) / 1e6
        words.append(f"timestamp_us used: {reports[source].note}")
    else:
        v = np.array([getattr(fr, source) for fr in frames], dtype=float)
        t = v - v[0]
        words.append(f"{source} used: {reports[source].note}")
    return Timeline(source=source, fallback=source != CLOCK_PREFERENCE[0],
                    reason="; ".join(words), t=t, reports=reports,
                    nominal_period_s=period, cross_check=_cross_check(frames))


def find_segments(frames: Sequence[SourceFrame], t: np.ndarray,
                  gap_threshold_s: float = GAP_THRESHOLD_S, *,
                  invalid_before: Sequence[int] = ()
                  ) -> Tuple[List[Segment], List[Event]]:
    """Split where playback must not interpolate: a hand-id change (the
    tracker lost the hand; the new id may not even be the same hand), a
    camera-clock gap over the threshold, or an invalid-tracking line between
    the two frames (`invalid_before` holds the index of the selected frame
    each status-0 line of the selected hand precedes; `load_recording` fills
    it). The tracker reported no valid hand there, so the two neighbours are
    not one continuous motion however short the interval and whatever the id
    and the clock say. A boundary that is an id change and a gap gives both
    events; the frame ids skipped there are counted in their details, not as
    a third event. An invalid-tracking cut adds no event of its own: the
    `invalid_status` event at that index already names the line. A frame-id
    skip with the id, the clock and the tracking continuous is a
    `dropped_frames` event inside the segment."""
    segments: List[Segment] = []
    events: List[Event] = []
    if not frames:
        return segments, events
    invalid = {int(i) for i in invalid_before}
    start = 0
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        dt = float(t[i + 1] - t[i])
        skipped = b.frame_id - a.frame_id - 1
        ids = (f"{skipped} frame ids skipped" if skipped >= 0 else
               f"frame_id went from {a.frame_id} back to {b.frame_id}")
        cut = []
        if b.hand_id != a.hand_id:
            cut.append(Event("id_change", i + 1, a.frame_id, b.frame_id, dt,
                             f"hand_id {a.hand_id} -> {b.hand_id} (tracker lost and "
                             f"re-acquired the hand); dt {dt:.4f} s; {ids}"))
        if dt > gap_threshold_s:
            cut.append(Event("gap", i + 1, a.frame_id, b.frame_id, dt,
                             f"dt {dt:.4f} s > {gap_threshold_s:g} s; {ids}"))
        if cut or (i + 1) in invalid:
            segments.append(Segment(start, i, a.hand_id))
            events.extend(cut)
            start = i + 1
        elif skipped != 0:
            events.append(Event("dropped_frames", i + 1, a.frame_id, b.frame_id, dt,
                                f"{ids}; same hand_id, dt {dt:.4f} s: "
                                f"still interpolated"))
    segments.append(Segment(start, len(frames) - 1, frames[-1].hand_id))
    return segments, events


# --- rotations ----------------------------------------------------------------------
def quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """v rotated by XYZW q through the sandwich q (v, 0) q*: no matrix."""
    q = quat_normalize(q)
    x, y, z, _ = quat_mul(quat_mul(q, (float(v[0]), float(v[1]), float(v[2]), 0.0)),
                          quat_conjugate(q))
    return np.array([x, y, z])


def _unit_q(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _scalar(a: np.ndarray):
    return float(a) if np.ndim(a) == 0 else a


def angular_distance_deg(qa, qb):
    """Rotation angle between XYZW quaternions (..., 4), sign-blind.

    4 atan2(|a - b|, |a + b|) after aligning signs, not 2 acos(|a.b|): acos
    cannot resolve angles below about 1e-6 degrees in float64, which is the
    size of the chain cross-check's tolerance.
    """
    a, b = _unit_q(qa), _unit_q(qb)
    b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0.0, -b, b)
    return _scalar(4.0 * np.degrees(np.arctan2(np.linalg.norm(a - b, axis=-1),
                                               np.linalg.norm(a + b, axis=-1))))


def rotation_angle_deg(Ra, Rb):
    """Rotation angle of Ra^T Rb for (..., 3, 3) matrices, via atan2 of its
    skew part and trace (precise near zero, where acos of the trace is not)."""
    r = np.swapaxes(np.asarray(Ra, dtype=float), -1, -2) @ np.asarray(Rb, dtype=float)
    skew = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0],
                     r[..., 1, 0] - r[..., 0, 1]], axis=-1)
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    return _scalar(np.degrees(np.arctan2(np.linalg.norm(skew, axis=-1), trace - 1.0)))


def slerp(q0, q1, alpha: float) -> np.ndarray:
    """Constant-rate interpolation along the shorter arc; (..., 4) XYZW in and
    out, result with w >= 0."""
    a, b = _unit_q(q0), _unit_q(q1)
    b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0.0, -b, b)
    phi = 2.0 * np.arctan2(np.linalg.norm(a - b, axis=-1, keepdims=True),
                           np.linalg.norm(a + b, axis=-1, keepdims=True))
    s = np.sin(phi)
    small = s < 1e-12                     # (nearly) the same rotation: lerp
    s = np.where(small, 1.0, s)
    w0 = np.where(small, 1.0 - alpha, np.sin((1.0 - alpha) * phi) / s)
    w1 = np.where(small, alpha, np.sin(alpha * phi) / s)
    out = _unit_q(w0 * a + w1 * b)
    return np.where(out[..., 3:4] < 0.0, -out, out)


# --- reconstruction -------------------------------------------------------------------
def _hand_frame(frame: SourceFrame) -> HandFrame:
    joints = [Joint(name=n, x=p[0], y=p[1], z=p[2], qw=q[3], qx=q[0], qy=q[1], qz=q[2])
              for n, p, q in zip(JOINT_NAMES, frame.rel_pos.tolist(),
                                 frame.rel_quat.tolist())]
    return HandFrame(timestamp=0.0, packet_counter=frame.frame_id,
                     hand_side=frame.hand_side, frame_id=frame.frame_id,
                     status=1 if frame.status is None else frame.status, joints=joints)


def reconstruct(frame: SourceFrame) -> Reconstruction:
    """World pose of one frame through the repository's matrix chain
    (R_child = R_parent R_local, p_child = p_parent + R_parent p_local): the
    same code every other tool in the repository draws hands with."""
    positions, rotations = forward_kinematics_full(_hand_frame(frame))
    rot = np.array([np.asarray(r, dtype=float) for r in rotations])
    quat = np.array([quat_canonical(mat3_to_quat(r)) for r in rot])
    return Reconstruction(pos=np.array(positions, dtype=float), rot=rot, quat=quat)


def reconstruct_quaternion_chain(frame: SourceFrame) -> Tuple[np.ndarray, np.ndarray]:
    """The same chain with quaternion arithmetic only: world_q = world_q[parent]
    * rel_q, world_p = world_p[parent] + world_q[parent] rel_p world_q[parent]*.

    A cross-check of two code paths over the same data (it catches a bug in
    either path or in the conversions between them), not an independent
    measurement of the hand. Returns (pos (26, 3), quat (26, 4) w >= 0).
    """
    wq: List[Optional[tuple]] = [None] * len(JOINT_NAMES)
    wp: List[Optional[np.ndarray]] = [None] * len(JOINT_NAMES)
    for i in _CHAIN_ORDER:
        rq = quat_normalize(frame.rel_quat[i])  # quat_to_mat3 normalises too
        par = PARENT.get(i)
        if par is None:
            wq[i], wp[i] = rq, np.array(frame.rel_pos[i], dtype=float)
        else:
            wq[i] = quat_mul(wq[par], rq)
            wp[i] = wp[par] + quat_rotate(wq[par], frame.rel_pos[i])
    return np.array(wp), np.array([quat_canonical(q) for q in wq])


def _stats(values) -> Dict[str, Optional[float]]:
    v = [x for x in values if x is not None]
    return {"max": max(v) if v else None, "mean": float(np.mean(v)) if v else None}


def frame_fidelity(frame: SourceFrame) -> dict:
    """Every replay-fidelity number for one frame (plain floats and lists).

    position_residual_m   reconstructed vs recorded abs26 per joint (None
                          without abs26)
    chain_deg, chain_m    matrix chain vs quaternion chain (cross-check)
    bone_direction_deg    per bone: reconstructed -z axis vs the recorded
                          bone vector abs26[i+1] - abs26[i] (independent)
    bone_length_mm        |abs26[i+1] - abs26[i]| per bone
    quat_norm             of the 26 relative quaternions as recorded
    rotation_det, rotation_orthogonality   of the 26 reconstructed rotations
    """
    rec = reconstruct(frame)
    qpos, qquat = reconstruct_quaternion_chain(frame)
    norms = np.linalg.norm(frame.rel_quat, axis=1)
    dets = np.linalg.det(rec.rot)
    ortho = np.abs(np.swapaxes(rec.rot, 1, 2) @ rec.rot - np.eye(3)).max()
    out = {
        "index": frame.index, "frame_id": frame.frame_id,
        "position_residual_m": {"per_joint": None, "max": None, "mean": None},
        "chain_deg": {"max": float(np.max(angular_distance_deg(rec.quat, qquat)))},
        "chain_m": {"max": float(np.linalg.norm(rec.pos - qpos, axis=1).max())},
        "bone_direction_deg": {"per_bone": None, "max": None, "mean": None},
        "bone_length_mm": None,
        "quat_norm": {"min": float(norms.min()), "max": float(norms.max())},
        "rotation_det": {"min": float(dets.min()), "max": float(dets.max())},
        "rotation_orthogonality": {"max": float(ortho)},
    }
    rec_abs = frame.abs_recorded
    if rec_abs is None:
        return out
    res = [float(x) for x in np.linalg.norm(rec.pos - rec_abs, axis=1)]
    out["position_residual_m"] = {"per_joint": res, **_stats(res)}
    start = [i for i, _ in BONES19]
    vec = rec_abs[[c for _, c in BONES19]] - rec_abs[start]
    axis = -rec.rot[start][:, :, 2]
    length = np.linalg.norm(vec, axis=1)
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(axis, vec), axis=1),
                                  np.einsum("ij,ij->i", axis, vec)))
    per_bone = {n: (float(a) if ln > 0.0 else None)   # no direction for length 0
                for n, a, ln in zip(BONE_NAMES, angle, length)}
    out["bone_direction_deg"] = {"per_bone": per_bone, **_stats(per_bone.values())}
    out["bone_length_mm"] = {n: float(ln * 1000.0) for n, ln in zip(BONE_NAMES, length)}
    return out


def _worst(per: List[dict], key: str) -> Tuple[Optional[float], Optional[dict]]:
    scored = [(p[key]["max"], p) for p in per if p[key]["max"] is not None]
    return max(scored, key=lambda s: s[0]) if scored else (None, None)


def recording_fidelity(rec: Recording) -> dict:
    """`frame_fidelity` over every selected frame, plus the timeline, the
    segments, the events and the tolerance verdict (JSON-able).

    Means are over every joint (or bone) of every frame. `within_tolerance`
    needs the position residual, the chain angle and the chain position at or
    under their declared values; a recording without abs26 cannot be
    declared within tolerance, because its positions were never checked.
    """
    per = [frame_fidelity(f) for f in rec.frames]
    measured = [p for p in per if p["position_residual_m"]["per_joint"] is not None]

    p_max, p_frame = _worst(per, "position_residual_m")
    position = {**_stats([v for p in measured for v in p["position_residual_m"]["per_joint"]]),
                "frames_measured": len(measured),
                "worst_frame_id": p_frame["frame_id"] if p_frame else None,
                "worst_joint": (JOINT_NAMES[int(np.argmax(p_frame["position_residual_m"]
                                                          ["per_joint"]))]
                                if p_frame else None)}
    c_deg, c_deg_frame = _worst(per, "chain_deg")
    c_m, c_m_frame = _worst(per, "chain_m")
    b_max, b_frame = _worst(per, "bone_direction_deg")
    expectation = TOLERANCES["bone_direction_deg_expectation"]["value"]
    per_bone_max = {n: _stats([p["bone_direction_deg"]["per_bone"][n]
                               for p in measured])["max"] for n in BONE_NAMES}
    bones = {
        **_stats([v for p in measured
                  for v in p["bone_direction_deg"]["per_bone"].values()]),
        "worst_frame_id": b_frame["frame_id"] if b_frame else None,
        "worst_bone": (max((n for n in BONE_NAMES if per_bone_max[n] is not None),
                           key=per_bone_max.get) if b_frame else None),
        "per_bone_max": per_bone_max,
        "expectation": expectation,
        "within_expectation": None if b_max is None else b_max <= expectation,
    }
    lengths = {}
    for n in BONE_NAMES:
        v = np.array([p["bone_length_mm"][n] for p in measured])
        lengths[n] = ({"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1
                       else 0.0, "min": float(v.min()), "max": float(v.max())}
                      if len(v) else None)
    q_min = min(p["quat_norm"]["min"] for p in per)
    q_max = max(p["quat_norm"]["max"] for p in per)
    q_dev = max(abs(q_max - 1.0), abs(1.0 - q_min))

    tl = rec.timeline
    dts = np.diff(tl.t)
    ids = [f.frame_id for f in rec.frames]
    detail = {}
    for key, value in (("position_m", p_max), ("chain_deg", c_deg), ("chain_m", c_m)):
        limit = TOLERANCES[key]["value"]
        detail[key] = {"measured": value, "limit": limit,
                       "ok": value is not None and value <= limit}
    return {
        "recording": {"path": str(rec.path), "kind": rec.kind, "name": rec.name,
                      "take": rec.take, "hand_side": rec.hand_side,
                      "hand_ids": rec.hand_ids, "provenance": rec.provenance,
                      "gap_threshold_s": rec.gap_threshold_s},
        "counts": {"frames_selected": len(rec.frames), "lines_total": rec.lines_total,
                   "lines_excluded": dict(rec.lines_excluded),
                   "lines_excluded_total": sum(rec.lines_excluded.values())},
        "frame_id": {"first": ids[0], "last": ids[-1],
                     "missing_in_range": max(ids) - min(ids) + 1 - len(set(ids))},
        "position_residual_m": position,
        "chain_deg": {"max": c_deg, "worst_frame_id": c_deg_frame["frame_id"]},
        "chain_m": {"max": c_m, "worst_frame_id": c_m_frame["frame_id"]},
        "bone_direction_deg": bones,
        "bone_length_mm": lengths,
        "quat_norm": {"min": q_min, "max": q_max, "max_deviation": q_dev,
                      "within_tolerance": q_dev <= TOLERANCES["quat_norm"]["value"]},
        "rotation_det": {"min": min(p["rotation_det"]["min"] for p in per),
                         "max": max(p["rotation_det"]["max"] for p in per)},
        "rotation_orthogonality": {"max": max(p["rotation_orthogonality"]["max"]
                                              for p in per)},
        "timeline": {"source": tl.source, "fallback": tl.fallback, "reason": tl.reason,
                     "duration_s": float(tl.t[-1] - tl.t[0]),
                     "nominal_period_s": tl.nominal_period_s,
                     "max_dt_s": float(dts.max()) if len(dts) else None,
                     "median_dt_s": float(np.median(dts)) if len(dts) else None,
                     "cross_check": dict(tl.cross_check),
                     "reports": {k: asdict(r) for k, r in tl.reports.items()}},
        "segments": [{**asdict(s), "frames": s.end - s.start + 1,
                      "frame_id_first": ids[s.start], "frame_id_last": ids[s.end],
                      "t_start_s": float(tl.t[s.start]), "t_end_s": float(tl.t[s.end])}
                     for s in rec.segments],
        "events": [asdict(e) for e in rec.events],
        "tolerances": {k: dict(v) for k, v in TOLERANCES.items()},
        "within_tolerance": all(d["ok"] for d in detail.values()),
        "within_tolerance_detail": detail,
    }


# --- what to show at a media time -----------------------------------------------------
def _shown(frame: SourceFrame, source: str, i: int, note: str) -> DisplayedPose:
    r = reconstruct(frame)
    return DisplayedPose(r.pos, r.rot, r.quat, source, i, i, 0.0, frame, note)


def pose_at(rec: Recording, t: float, mode: str = "hold") -> DisplayedPose:
    """The pose at media time t (seconds on the recording's timeline).

    hold: the latest source frame at or before t ("recorded" when t is exactly
    a source time, else "held"). interpolate: lerp of world positions and
    slerp of world quaternions between the two bracketing frames, but only
    inside one segment; across a boundary (id change, gap, invalid tracking)
    the earlier frame is held and the note says why. Before the first frame
    nothing is shown ("unavailable"); after the last, the last is held.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not math.isfinite(t):
        raise ValueError(f"media time must be finite, got {t!r}")
    times = rec.timeline.t
    n = len(rec.frames)
    if n == 0 or t < times[0]:
        return DisplayedPose(None, None, None, "unavailable", None, None, None, None,
                             "before the first frame" if n else "no frames")
    i0 = int(np.searchsorted(times, t, side="right")) - 1
    frame = rec.frames[i0]
    if t == times[i0]:
        return _shown(frame, "recorded", i0, "")
    if i0 == n - 1:
        return _shown(frame, "held", i0, "after the last frame: holding it")
    if mode == "hold":
        return _shown(frame, "held", i0, "")
    nxt = rec.frames[i0 + 1]
    if any(s.end == i0 for s in rec.segments):
        here = [e for e in rec.events if e.index == i0 + 1]
        kinds = [e.kind for e in here if e.kind in ("id_change", "gap")]
        # A cut with no id change and no gap is an invalid-tracking cut. An
        # invalid_status event at the last frame may instead be a line after
        # it (load_recording clamps it there), so it is named only when the
        # cut cannot be anything else or the index is not the last frame.
        if any(e.kind == "invalid_status" for e in here) and (not kinds or i0 + 1 < n - 1):
            kinds.append("invalid_status")
        return _shown(frame, "held", i0,
                      f"discontinuity ({', '.join(kinds) or 'segment boundary'}) "
                      f"between frame_id {frame.frame_id} and {nxt.frame_id}: "
                      f"holding the earlier frame, not interpolated")
    a, b = reconstruct(frame), reconstruct(nxt)
    alpha = float((t - times[i0]) / (times[i0 + 1] - times[i0]))
    quat = slerp(a.quat, b.quat, alpha)
    skipped = nxt.frame_id - frame.frame_id - 1
    return DisplayedPose(
        pos=(1.0 - alpha) * a.pos + alpha * b.pos,
        rot=np.array([quat_to_mat3(*q) for q in quat]), quat=quat,
        source="interpolated", i0=i0, i1=i0 + 1, alpha=alpha, frame=frame,
        note=f"interpolated across {skipped} dropped frame ids" if skipped > 0 else "")


def to_reference(pos, rot, reference: str = "camera") -> Tuple[np.ndarray, np.ndarray]:
    """Express a pose in another frame, by the rule `xr_hand.joint_frames.
    table_from_world` uses: origin at the WRIST joint, axes of the WRIST
    ('wrist') or of the PALM ('palm'); local p = R^T (p - o), local R = R^T R_i.

    On the camera the WRIST rotation is LeapC's forearm (`arm.rotation`), so
    'wrist' shows the hand against the forearm's axes; 'palm' uses the palm's.
    'camera' returns the inputs: the recorded motion itself.
    """
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}")
    pos, rot = np.asarray(pos, dtype=float), np.asarray(rot, dtype=float)
    if reference == "camera":
        return pos, rot
    axes = rot[WRIST] if reference == "wrist" else rot[PALM]
    return (pos - pos[WRIST]) @ axes, axes.T @ rot


# --- playback timing ------------------------------------------------------------------
def _check_speed(speed: float) -> float:
    speed = float(speed)
    if not (math.isfinite(speed) and speed > 0.0):
        raise ValueError(f"speed must be a positive number, got {speed!r}")
    return speed


class PlaybackClock:
    """Media time for time-driven playback: wall time x speed while playing.

    Pure: every method takes `now`; when it is omitted the `now` callable
    given at construction is read. Media time is continuous through speed
    changes, frozen while paused, and jumps only on `seek` or `start(...,
    media_t)`. The viewer shows `pose_at(rec, clock.media_time(now))`, so a
    late tick shows a later frame instead of stretching time.
    """

    def __init__(self, speed: float = 1.0, now=time.perf_counter):
        self._clock = now
        self.speed = _check_speed(speed)
        self.playing = False
        self._media0 = 0.0     # media time at the anchor
        self._wall0 = 0.0      # wall time at the anchor (meaningful while playing)

    def _now(self, now) -> float:
        return float(self._clock() if now is None else now)

    def media_time(self, now=None) -> float:
        if not self.playing:
            return self._media0
        return self._media0 + (self._now(now) - self._wall0) * self.speed

    def start(self, now=None, media_t=None) -> None:
        n = self._now(now)
        if media_t is not None:
            self._media0 = float(media_t)
        elif self.playing:
            self._media0 += (n - self._wall0) * self.speed
        self._wall0, self.playing = n, True

    def pause(self, now=None) -> None:
        if self.playing:
            self._media0 = self.media_time(now)
            self.playing = False

    def set_speed(self, speed: float, now=None) -> None:
        speed = _check_speed(speed)
        if self.playing:
            n = self._now(now)
            self._media0 += (n - self._wall0) * self.speed
            self._wall0 = n
        self.speed = speed

    def seek(self, media_t: float, now=None) -> None:
        self._media0 = float(media_t)
        if self.playing:
            self._wall0 = self._now(now)


class SchedulingLog:
    """Per tick: the media time the viewer meant to show vs the media time the
    wall clock had reached when it showed it. A property of the viewer, not
    of the data. `frame_index` (optional) counts source frames never shown."""

    def __init__(self):
        self.intended: List[float] = []
        self.actual: List[float] = []
        self.frame_index: List[Optional[int]] = []

    def record(self, intended_media_t: float, actual_media_t: float,
               frame_index: Optional[int] = None) -> None:
        self.intended.append(float(intended_media_t))
        self.actual.append(float(actual_media_t))
        self.frame_index.append(frame_index)

    def stats(self) -> dict:
        """n, then mean / p95 / max of |actual - intended| in ms, and
        frames_skipped: source frames between consecutive displayed ones."""
        shown = [i for i in self.frame_index if i is not None]
        skipped = sum(max(0, b - a - 1) for a, b in zip(shown, shown[1:]))
        if not self.intended:
            return {"n": 0, "mean_ms": None, "p95_ms": None, "max_ms": None,
                    "frames_skipped": skipped}
        err = np.abs(np.array(self.actual) - np.array(self.intended)) * 1e3
        return {"n": len(err), "mean_ms": float(err.mean()),
                "p95_ms": float(np.percentile(err, 95)), "max_ms": float(err.max()),
                "frames_skipped": int(skipped)}
