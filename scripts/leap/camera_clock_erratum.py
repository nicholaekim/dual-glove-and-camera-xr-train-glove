"""Camera-clock erratum for the timing columns of `grasp set 2026-10-08 organized`.

A derived table, not a dataset change. The dataset's metadata gives its
durations on the writer clock: `wall_time`, stamped when the recorder's
writer reached a line, which stalls (up to 0.86 s in this dataset,
`docs/replay_fidelity.md` section 5). Every JSONL line also carries the
camera's own acquisition clock, `timestamp_us`, and the same instant on
the wall clock, `capture_time`. This script recomputes, on the camera clock,
what the metadata gives on the writer clock, and writes the two side by
side with their difference and the largest writer stall inside each
interval. The dataset is opened read-only; the script refuses an output
folder inside it.

Static takes (metadata/grasps_summary.csv, one row per take)
  recording duration     `recording_duration_s` (last minus first wall_time
                         of the hand's lines) against the camera-clock span
                         of the selected frames. The metadata's own
                         `camera_clock_duration_s` is copied beside it.
  static interval        the frames with `static_interval_first_frame_id`
                         <= frame_id <= `static_interval_last_frame_id`:
                         `static_interval_duration_s` (declared, end minus
                         start of the interval the recorder chose, writer
                         clock) against the camera-clock span of those
                         frames; the writer-clock span of the same frames
                         is given too, so the convention (the declared end
                         is an instant, not a frame) and the clock can be
                         told apart. Start and end as seconds from the first
                         selected frame: `static_interval_start_s` /
                         `_end_s` (from `first_wall_time`) against the
                         first and last interval frame's camera time.
Transitions (metadata/pose_transitions.csv, one row per capture)
  Measured on the real data before this was written: the metadata's
  `movement_start_time_s` and `settled_time_s` are seconds from
  `first_frame_capture_time` on the CAPTURE clock (`leap_hand.transition.
  frame_time`), not the writer clock: each lands on a selected frame's
  capture_time within the table's 1 ms rounding, while on wall_time the
  nearest frame is a different one. So each instant is mapped to the
  nearest selected frame by `capture_time`, and that frame's camera time
  is the camera-clock value; the residual on both clocks is written so a
  reader can check the choice. The cue (`cue_time_s`) is not a frame: it is
  carried onto the camera timeline by the first frame captured at or after
  it, keeping its distance to that frame. `transition_duration_s` is then
  settled minus movement start on the camera clock, beside the writer-clock
  span of the same frames and the largest writer stall between them.

Clocks: `timestamp_us` first, falling back exactly as
`leap_hand.replay_core.build_timeline` does (capture_time, wall_time, frame
index x nominal period); `clock_source` and `clock_fallback` say which was
used. Hand selection is `replay_core.find_static` / `find_transition`: the
hand side and tracker ids the metadata row names, status != 0. Seconds are
rounded to 1e-6 (the clocks' resolution), milliseconds to 1e-3.

  .venv\\Scripts\\python.exe scripts\\leap\\camera_clock_erratum.py --dataset "<root>" --out results\\replay_fidelity\\camera_clock_erratum
  .venv\\Scripts\\python.exe scripts\\leap\\camera_clock_erratum.py --dataset "<root>" --out <folder> --static p3_circular_precision_thumb_2_fingers 4

Writes static_takes.csv, transitions.csv, README.md and summary.md. Prints
`LOAD FAILED: <kind> <name> take <n>: <why>` per recording it cannot read,
then one line: `ERRATUM OK: static=N transitions=M
static_interval_over_one_period=K` (exit 0) or `ERRATUM FAIL: ...
load_failures=F` (exit 1). Exit 2: bad arguments or an output folder inside
the dataset. Whether the dataset is unchanged is checked separately:
`scripts\\leap\\replay_verify.py --check-dataset-unchanged <baseline.json> "<root>"`.
"""
import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from leap_hand import replay_core as rc

DEFAULT_OUT = Path("results") / "replay_fidelity" / "camera_clock_erratum"

STATIC_COLUMNS = [
    "grasp_name", "take", "hand", "hand_ids", "raw_jsonl_file", "frames_selected",
    "clock_source", "clock_fallback", "frame_period_s",
    "first_frame_wall_minus_first_wall_time_ms",
    "recording_duration_metadata_s", "recording_duration_camera_s",
    "recording_duration_difference_s", "metadata_camera_clock_duration_s",
    "recording_largest_writer_stall_s",
    "static_interval_first_frame_id", "static_interval_last_frame_id",
    "static_interval_frames",
    "static_interval_start_metadata_s", "static_interval_start_camera_s",
    "static_interval_start_difference_s",
    "static_interval_end_metadata_s", "static_interval_end_camera_s",
    "static_interval_end_difference_s",
    "static_interval_duration_metadata_s", "static_interval_writer_span_s",
    "static_interval_camera_s", "static_interval_difference_s",
    "static_interval_camera_minus_writer_span_s", "static_interval_over_one_period",
    "static_interval_largest_writer_stall_s", "stall_frame_id_before",
    "stall_frame_id_after", "stall_camera_ms",
]
TRANSITION_COLUMNS = [
    "to_grasp", "take", "hand", "hand_ids", "recording_file", "validity",
    "frames_selected", "clock_source", "clock_fallback", "frame_period_s",
    "first_frame_capture_minus_first_frame_capture_time_ms",
    "recording_writer_span_s", "recording_camera_s", "recording_largest_writer_stall_s",
    "cue_time_metadata_s", "cue_frame_id", "cue_time_camera_s", "cue_time_difference_s",
    "movement_start_metadata_s", "movement_start_frame_id",
    "movement_start_capture_residual_ms", "movement_start_wall_residual_ms",
    "movement_start_camera_s", "movement_start_difference_s",
    "settled_metadata_s", "settled_frame_id", "settled_capture_residual_ms",
    "settled_wall_residual_ms", "settled_camera_s", "settled_difference_s",
    "transition_duration_metadata_s", "transition_duration_writer_span_s",
    "transition_duration_camera_s", "transition_duration_difference_s",
    "transition_largest_writer_stall_s", "stall_frame_id_before", "stall_frame_id_after",
    "stall_camera_ms",
]


# --- small helpers ---------------------------------------------------------------------
def _opt(v):
    """A float, or None for a blank metadata cell."""
    return None if v in (None, "") else float(v)


def _r6(x):
    return None if x is None else round(float(x), 6)


def _r3(x):
    return None if x is None else round(float(x), 3)


def _sub(a, b):
    return None if a is None or b is None else a - b


def _cell(v) -> str:
    """CSV text: blank for None, lowercase booleans, floats in full (repr)."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        return repr(float(v)) if math.isfinite(v) else ""
    return str(v)


def _write_csv(path: Path, columns, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows:
            w.writerow([_cell(row.get(c)) for c in columns])


def _path_key(path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path)))).casefold()


def _is_under(path, root) -> bool:
    p, r = _path_key(path), _path_key(root)
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


# --- clocks over a run of frames ------------------------------------------------------------
def writer_span(frames):
    """max minus min wall_time, as the metadata's recording_duration_s is cut."""
    w = [f.wall_time for f in frames if f.wall_time is not None]
    return float(max(w) - min(w)) if len(w) >= 2 else None


def largest_writer_stall(rec, frames):
    """The largest wall_time step between consecutive frames of `frames`
    (the first of equal maxima), with the camera time across the same pair."""
    pairs = [(b.wall_time - a.wall_time, a, b) for a, b in zip(frames, frames[1:])
             if a.wall_time is not None and b.wall_time is not None]
    if not pairs:
        return {}
    dj, a, b = max(pairs, key=lambda p: p[0])
    t = rec.timeline.t
    return {"stall_s": _r6(dj), "before": a.frame_id, "after": b.frame_id,
            "camera_ms": _r3((t[b.index] - t[a.index]) * 1e3)}


def _stall_cells(prefix, stall):
    return {f"{prefix}_largest_writer_stall_s": stall.get("stall_s"),
            "stall_frame_id_before": stall.get("before"),
            "stall_frame_id_after": stall.get("after"),
            "stall_camera_ms": stall.get("camera_ms")}


def _common(rec):
    return {"take": rec.take, "hand": rec.hand_side, "hand_ids": rec.metadata.get("hand_ids"),
            "frames_selected": len(rec.frames), "clock_source": rec.timeline.source,
            "clock_fallback": rec.timeline.fallback,
            "frame_period_s": _r6(rec.timeline.nominal_period_s)}


# --- static takes -----------------------------------------------------------------------------
def static_row(rec: rc.Recording) -> dict:
    m, fr, t = rec.metadata, rec.frames, rec.timeline.t
    first_wall = _opt(m.get("first_wall_time"))
    camera_total = float(t[-1] - t[0])
    meta_total = _opt(m.get("recording_duration_s"))
    row = {"grasp_name": rec.name, "raw_jsonl_file": m.get("raw_jsonl_file"), **_common(rec),
           "first_frame_wall_minus_first_wall_time_ms": _r3(
               None if first_wall is None or fr[0].wall_time is None
               else (fr[0].wall_time - first_wall) * 1e3),
           "recording_duration_metadata_s": meta_total,
           "recording_duration_camera_s": _r6(camera_total),
           "recording_duration_difference_s": _r6(_sub(camera_total, meta_total)),
           "metadata_camera_clock_duration_s": _opt(m.get("camera_clock_duration_s")),
           "recording_largest_writer_stall_s": largest_writer_stall(rec, fr).get("stall_s")}
    mask = rc.hold_mask(rec)
    if mask is None or not mask.any():
        return row
    idx = np.flatnonzero(mask)
    i0, i1 = int(idx[0]), int(idx[-1])
    win = [fr[i] for i in idx]
    camera = float(t[i1] - t[i0])
    writer = writer_span(win)
    declared = _opt(m.get("static_interval_duration_s"))
    start_meta, end_meta = _opt(m.get("static_interval_start_s")), _opt(m.get("static_interval_end_s"))
    diff = _sub(camera, declared)
    row.update({
        "static_interval_first_frame_id": int(float(m["static_interval_first_frame_id"])),
        "static_interval_last_frame_id": int(float(m["static_interval_last_frame_id"])),
        "static_interval_frames": len(idx),
        "static_interval_start_metadata_s": start_meta,
        "static_interval_start_camera_s": _r6(t[i0]),
        "static_interval_start_difference_s": _r6(_sub(float(t[i0]), start_meta)),
        "static_interval_end_metadata_s": end_meta,
        "static_interval_end_camera_s": _r6(t[i1]),
        "static_interval_end_difference_s": _r6(_sub(float(t[i1]), end_meta)),
        "static_interval_duration_metadata_s": declared,
        "static_interval_writer_span_s": _r6(writer),
        "static_interval_camera_s": _r6(camera),
        "static_interval_difference_s": _r6(diff),
        "static_interval_camera_minus_writer_span_s": _r6(_sub(camera, writer)),
        "static_interval_over_one_period": (
            None if diff is None else bool(abs(diff) > rec.timeline.nominal_period_s)),
        **_stall_cells("static_interval", largest_writer_stall(rec, win)),
    })
    return row


# --- transitions --------------------------------------------------------------------------------
def _nearest(values, instant):
    """Index of the frame whose clock value is nearest `instant`, and the
    signed residual in ms; (None, None) when no frame carries the clock."""
    best = None
    for i, v in enumerate(values):
        if v is not None and (best is None or abs(v - instant) < abs(values[best] - instant)):
            best = i
    return (None, None) if best is None else (best, (values[best] - instant) * 1e3)


def place_instant(rec: rc.Recording, instant: float) -> dict:
    """A capture-clock instant (absolute wall-clock seconds) on the camera
    timeline: the nearest selected frame by capture_time, that frame's camera
    time, and the residual on both clocks."""
    k, cap_res = _nearest([f.capture_time for f in rec.frames], instant)
    _, wall_res = _nearest([f.wall_time for f in rec.frames], instant)
    if k is None:
        return {}
    return {"index": k, "frame_id": rec.frames[k].frame_id,
            "camera_s": float(rec.timeline.t[k]),
            "capture_residual_ms": cap_res, "wall_residual_ms": wall_res}


def place_cue(rec: rc.Recording, cue: float) -> dict:
    """The cue is not a frame: it is carried onto the camera timeline by the
    first selected frame captured at or after it (the last frame when none
    is), keeping its distance to that frame's capture_time."""
    with_cap = [f for f in rec.frames if f.capture_time is not None]
    if not with_cap:
        return {}
    f = next((f for f in with_cap if f.capture_time >= cue), with_cap[-1])
    return {"frame_id": f.frame_id,
            "camera_s": float(rec.timeline.t[f.index]) - (f.capture_time - cue)}


def transition_row(rec: rc.Recording) -> dict:
    m, fr = rec.metadata, rec.frames
    ffc = _opt(m.get("first_frame_capture_time"))
    row = {"to_grasp": rec.name, "recording_file": m.get("recording_file"),
           "validity": m.get("validity"), **_common(rec),
           "first_frame_capture_minus_first_frame_capture_time_ms": _r3(
               None if ffc is None or fr[0].capture_time is None
               else (fr[0].capture_time - ffc) * 1e3),
           "recording_writer_span_s": _r6(writer_span(fr)),
           "recording_camera_s": _r6(rec.timeline.t[-1] - rec.timeline.t[0]),
           "recording_largest_writer_stall_s": largest_writer_stall(rec, fr).get("stall_s")}
    cue_meta = _opt(m.get("cue_time_s"))
    row["cue_time_metadata_s"] = cue_meta
    # the unrounded instant when the table has it; cue_time_s is rounded to 1 ms
    cue_abs = _opt(m.get("cue_wall_time"))
    if cue_abs is None and ffc is not None and cue_meta is not None:
        cue_abs = ffc + cue_meta
    if cue_abs is not None and cue_meta is not None:
        placed = place_cue(rec, cue_abs)
        row.update({"cue_frame_id": placed.get("frame_id"),
                    "cue_time_camera_s": _r6(placed.get("camera_s")),
                    "cue_time_difference_s": _r6(_sub(placed.get("camera_s"), cue_meta))})
    placed = {}
    for key, col in (("movement_start", "movement_start_time_s"), ("settled", "settled_time_s")):
        meta = _opt(m.get(col))
        row[f"{key}_metadata_s"] = meta
        if ffc is None or meta is None:
            continue
        p = place_instant(rec, ffc + meta)
        placed[key] = p
        row.update({f"{key}_frame_id": p.get("frame_id"),
                    f"{key}_capture_residual_ms": _r3(p.get("capture_residual_ms")),
                    f"{key}_wall_residual_ms": _r3(p.get("wall_residual_ms")),
                    f"{key}_camera_s": _r6(p.get("camera_s")),
                    f"{key}_difference_s": _r6(_sub(p.get("camera_s"), meta))})
    duration_meta = _opt(m.get("transition_duration_s"))
    row["transition_duration_metadata_s"] = duration_meta
    a, b = placed.get("movement_start", {}), placed.get("settled", {})
    if a and b:
        lo, hi = sorted((a["index"], b["index"]))
        between = fr[lo:hi + 1]
        camera = b["camera_s"] - a["camera_s"]
        row.update({"transition_duration_writer_span_s": _r6(writer_span(between)),
                    "transition_duration_camera_s": _r6(camera),
                    "transition_duration_difference_s": _r6(_sub(camera, duration_meta)),
                    **_stall_cells("transition", largest_writer_stall(rec, between))})
    return row


# --- the summary and the README --------------------------------------------------------------
def _dist(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not v.size:
        return None
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def _dist_rows(title, items, unit_digits):
    lines = [f"| {title} | n | min | median | p95 | max |", "|---|---:|---:|---:|---:|---:|"]
    for label, values in items:
        d = _dist(values)
        if d is None:
            lines.append(f"| {label} | 0 | | | | |")
            continue
        def f(x):                      # + 0.0 turns a rounded -0.0 into 0.0
            return f"{round(x, unit_digits) + 0.0:.{unit_digits}f}"
        lines.append(f"| {label} | {d['n']} | {f(d['min'])} | {f(d['median'])} "
                     f"| {f(d['p95'])} | {f(d['max'])} |")
    return lines


def _col(rows, key):
    return [r.get(key) for r in rows]


def summary_text(root, static_rows, transition_rows, failures, verdict) -> str:
    L = ["# Camera-clock erratum: summary", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/leap/camera_clock_erratum.py` "
         f"over `{root}`. Method, clocks and columns: `README.md` beside this file.", "",
         f"```", verdict, "```", ""]
    if failures:
        L += ["## Recordings that could not be read", ""] + [f"- {f}" for f in failures] + [""]

    L += [f"## Static takes ({len(static_rows)} rows: `static_takes.csv`)", "",
          "Every difference is the camera-clock value minus the metadata value, in "
          "seconds; the writer span is the wall_time span of the same frames.", ""]
    L += _dist_rows("Static takes, seconds", [
        ("recording duration: camera - metadata", _col(static_rows, "recording_duration_difference_s")),
        ("static interval duration: camera - metadata (declared)",
         _col(static_rows, "static_interval_difference_s")),
        ("static interval duration: camera - writer span of the same frames",
         _col(static_rows, "static_interval_camera_minus_writer_span_s")),
        ("static interval start: camera - metadata", _col(static_rows, "static_interval_start_difference_s")),
        ("static interval end: camera - metadata", _col(static_rows, "static_interval_end_difference_s")),
        ("largest writer stall inside the interval",
         _col(static_rows, "static_interval_largest_writer_stall_s")),
        ("largest writer stall in the whole take", _col(static_rows, "recording_largest_writer_stall_s")),
    ], 6)
    L += ["", "Consistency of the two origins (milliseconds):", ""]
    L += _dist_rows("Static takes, ms", [
        ("first selected frame's wall_time - first_wall_time",
         _col(static_rows, "first_frame_wall_minus_first_wall_time_ms")),
        ("metadata camera_clock_duration_s - recomputed camera duration",
         [None if r.get("metadata_camera_clock_duration_s") is None
          or r.get("recording_duration_camera_s") is None
          else (r["metadata_camera_clock_duration_s"] - r["recording_duration_camera_s"]) * 1e3
          for r in static_rows]),
    ], 3)
    periods = sorted({r["frame_period_s"] for r in static_rows if r.get("frame_period_s")})
    span = (f"{periods[0]:.6f} s" if len(periods) == 1 else
            f"{periods[0]:.6f} to {periods[-1]:.6f} s, 1 / the take's median framerate"
            if periods else "?")
    flagged = sorted((r for r in static_rows if r.get("static_interval_over_one_period")),
                     key=lambda r: -abs(r["static_interval_difference_s"]))
    L += ["", f"### Takes whose static interval differs by more than one frame period "
          f"(|camera - metadata| > {span}): {len(flagged)} of {len(static_rows)}", ""]
    for r in flagged:
        L.append(f"- {r['grasp_name']} take {r['take']}: metadata {r['static_interval_duration_metadata_s']:.3f} s, "
                 f"writer span {r['static_interval_writer_span_s']:.6f} s, camera "
                 f"{r['static_interval_camera_s']:.6f} s, difference {r['static_interval_difference_s']:+.6f} s, "
                 f"largest writer stall {r['static_interval_largest_writer_stall_s']:.6f} s between frames "
                 f"{r['stall_frame_id_before']} and {r['stall_frame_id_after']}")
    if not flagged:
        L.append("- none")
    clock_only = [r for r in static_rows
                  if r.get("static_interval_camera_minus_writer_span_s") is not None
                  and abs(r["static_interval_camera_minus_writer_span_s"]) > r["frame_period_s"]]
    L += ["", f"The declared end is an instant up to one period after the last interval frame, "
          f"so a difference near one period can be the convention alone. Across all "
          f"{len(static_rows)} takes, the camera span differs from the writer span of the same "
          f"frames by more than one period (the clock alone) in {len(clock_only)}: "
          + (", ".join(f"{r['grasp_name']} take {r['take']} ({r['static_interval_camera_minus_writer_span_s']:+.6f} s)"
                       for r in clock_only) or "none") + ".", ""]

    timed = [r for r in transition_rows if r.get("transition_duration_camera_s") is not None]
    L += [f"## Transitions ({len(transition_rows)} rows: `transitions.csv`; {len(timed)} with a "
          f"movement start and a settle, the others blank in the metadata)", ""]
    L += _dist_rows("Transitions, seconds", [
        ("cue: camera - metadata", _col(transition_rows, "cue_time_difference_s")),
        ("movement start: camera - metadata", _col(transition_rows, "movement_start_difference_s")),
        ("settled: camera - metadata", _col(transition_rows, "settled_difference_s")),
        ("transition duration: camera - metadata", _col(transition_rows, "transition_duration_difference_s")),
        ("transition duration: camera - writer span of the same frames",
         [None if r.get("transition_duration_camera_s") is None
          or r.get("transition_duration_writer_span_s") is None
          else r["transition_duration_camera_s"] - r["transition_duration_writer_span_s"]
          for r in transition_rows]),
        ("largest writer stall between movement start and settle",
         _col(transition_rows, "transition_largest_writer_stall_s")),
        ("largest writer stall in the whole capture", _col(transition_rows, "recording_largest_writer_stall_s")),
        ("recording: camera span", _col(transition_rows, "recording_camera_s")),
        ("recording: writer span", _col(transition_rows, "recording_writer_span_s")),
    ], 6)
    L += ["", "The writer span of a whole capture is shorter than its camera span, and its "
          "largest stall large, because the frames from before the cue were held in memory "
          "and written at the cue (`scripts/leap/record_transitions.py`); the stall between "
          "movement start and settle is the writer's ordinary stalling."]
    cap_res = _col(transition_rows, "movement_start_capture_residual_ms") + _col(
        transition_rows, "settled_capture_residual_ms")
    wall_res = _col(transition_rows, "movement_start_wall_residual_ms") + _col(
        transition_rows, "settled_wall_residual_ms")
    L += ["", "Which clock the metadata's instants were cut on (milliseconds, nearest "
          "selected frame minus the instant):", ""]
    L += _dist_rows("Transitions, ms", [
        ("|capture_time residual|, movement start and settle", [abs(x) for x in cap_res if x is not None]),
        ("|wall_time residual|, movement start and settle", [abs(x) for x in wall_res if x is not None]),
        ("first selected frame's capture_time - first_frame_capture_time",
         _col(transition_rows, "first_frame_capture_minus_first_frame_capture_time_ms")),
    ], 3)
    dc, dw = _dist([abs(x) for x in cap_res if x is not None]), _dist([abs(x) for x in wall_res if x is not None])
    if dc and dw:
        L += ["", f"The metadata rounds its times to 1 ms, so an instant cut on a clock lies "
              f"within 0.5 ms of a frame on that clock. By `capture_time` every movement start "
              f"and settle is within {dc['max']:.3f} ms of a selected frame; by `wall_time` the "
              f"nearest frame is up to {dw['max']:.3f} ms away (median {dw['median']:.3f} ms). "
              f"The transition instants were cut on the capture clock, and were mapped by it."]
    return "\n".join(L) + "\n"


README = """# Camera-clock erratum for the timing columns of `grasp set 2026-10-08 organized`

Generated {date} by `scripts/leap/camera_clock_erratum.py` (repository
`non glove xr trainer`) over `{root}`.

**What this is.** Derived tables beside the dataset, not a change to it. The
dataset was opened read-only and nothing in it was written, renamed or
moved; the script refuses an output folder inside the dataset. Whether the
dataset is byte-for-byte unchanged is checked separately against the
forensic SHA-256 baseline:

```
.venv\\Scripts\\python.exe scripts\\leap\\replay_verify.py --check-dataset-unchanged <baseline.json> "{root}"
```

The metadata tables themselves are left as they are. Where this erratum
and `metadata/grasps_summary.csv` or `metadata/pose_transitions.csv`
disagree, the metadata is the writer-clock number and the erratum is the
camera-clock number; both are kept.

**Reproduce.** From the repository root, with the dataset at the path above;
the four files are rewritten in full on every run:

```
.venv\\Scripts\\python.exe scripts\\leap\\camera_clock_erratum.py --dataset "{root}" --out <folder>
```

**How to apply.** Keep the dataset's tables as they are and read this
erratum beside them, joined on `grasp_name` + `take` (static takes) or
`to_grasp` + `take` (transitions):

1. For a static take's hold duration use `static_interval_camera_s` from
   `static_takes.csv` in place of `static_interval_duration_s` in
   `metadata/grasps_summary.csv`.
2. For the hold's start and end as seconds into the take use
   `static_interval_start_camera_s` and `static_interval_end_camera_s` in
   place of `static_interval_start_s` and `static_interval_end_s`.
3. For a take's length use `recording_duration_camera_s` in place of
   `recording_duration_s`.
4. The frames of the hold (`static_interval_first_frame_id` to
   `static_interval_last_frame_id`), the representative frame, the joint
   angles, the keypoints and the images are the same frames as before and
   need no correction: only the durations and instants change.
5. The transitions' `movement_start_time_s`, `settled_time_s` and
   `transition_duration_s` already count on the capture clock and stand;
   `transitions.csv` confirms them (the `*_difference_s` columns) and adds
   the frame id each instant falls on.

`static_interval_over_one_period` marks the takes where the correction is
larger than one frame period; `summary.md` lists them, largest first.

**The three clocks on every JSONL line.**

- `timestamp_us`: the camera's own acquisition clock (LeapC device clock),
  integer microseconds. The camera-clock values here are on it.
- `capture_time`: the same instant placed on the wall clock, rounded to
  1e-6 s. Its intervals between consecutive frames agree with those of
  `timestamp_us` to within 0.33 ms everywhere in this dataset
  (`docs/replay_fidelity.md`, section 5).
- `wall_time`: when the recorder's writer reached the line. It stalls: up
  to 0.86 s in this dataset. The metadata's durations for the static takes
  were cut on it.

Time is taken as `leap_hand.replay_core.build_timeline` takes it:
`timestamp_us` when it is present on every selected frame and strictly
increasing, else `capture_time`, else `wall_time`, else frame index x the
nominal period. `clock_source` and `clock_fallback` in each row say what
was used. Every camera-clock value is seconds from the first selected
frame. Selected frames are the metadata row's hand side and tracker ids
with `status != 0` (`replay_core.find_static` / `find_transition`), so a
line of the other hand or of a stray tracker id is never counted.

**Static takes (`static_takes.csv`, one row per take).**

- Recording duration: `recording_duration_metadata_s` is the metadata's
  `recording_duration_s` (last minus first `wall_time` of the hand's
  lines). `recording_duration_camera_s` is the camera-clock span of the
  selected frames. `metadata_camera_clock_duration_s` copies the
  metadata's own `camera_clock_duration_s` for comparison.
- Static interval: the selected frames with `static_interval_first_frame_id`
  <= `frame_id` <= `static_interval_last_frame_id` (`static_interval_frames`
  of them). `static_interval_duration_metadata_s` is the declared
  `static_interval_duration_s`: the recorder's interval end minus start on
  the writer clock, where the start is the first interval frame's
  `wall_time` and the end is an instant, not a frame. Three spans are
  given so the convention and the clock can be told apart:
  `static_interval_writer_span_s` (last minus first `wall_time` of the
  interval frames), `static_interval_camera_s` (the same frames on the
  camera clock), and the two differences `static_interval_difference_s`
  (camera minus declared) and `static_interval_camera_minus_writer_span_s`.
  `static_interval_over_one_period` is true when |camera minus declared|
  exceeds one nominal frame period (`frame_period_s`: 1 / the median
  `framerate` of the take's lines, about 1/90 s).
- Start and end: `static_interval_start_metadata_s` / `_end_metadata_s` are
  the metadata's `static_interval_start_s` / `_end_s`, seconds from
  `first_wall_time` on the writer clock. The camera values are the first
  and last interval frame's camera time from the first selected frame.
  `first_frame_wall_minus_first_wall_time_ms` checks that the two origins
  are the same frame (0 when they are).
- `static_interval_largest_writer_stall_s`: the largest `wall_time` step
  between consecutive interval frames, between `stall_frame_id_before` and
  `stall_frame_id_after`; `stall_camera_ms` is the camera time across the
  same pair. `recording_largest_writer_stall_s` is the same over the whole
  take.

**Transitions (`transitions.csv`, one row per capture).**

- Which clock. Before this was written it was measured on the real data
  that `movement_start_time_s` and `settled_time_s` are seconds from
  `first_frame_capture_time` on the **capture** clock (they are cut by
  `leap_hand.transition.frame_time`, which reads `capture_time`), not on the
  writer clock: each lands on a selected frame's `capture_time` within the
  table's 1 ms rounding, while on `wall_time` the nearest frame is another
  one. `summary.md` gives both residual distributions for this run. So each
  instant is mapped to the **nearest selected frame by `capture_time`**,
  and that frame's camera time is the camera-clock value
  (`movement_start_camera_s`, `settled_camera_s`, with the frame id and the
  signed residuals `*_capture_residual_ms` and `*_wall_residual_ms`).
- The cue is not a frame. Its instant is the table's unrounded
  `cue_wall_time` (the beep's wall-clock time; `cue_time_metadata_s` is
  the table's `cue_time_s`, that instant minus `first_frame_capture_time`
  rounded to 1 ms). It is carried onto the camera timeline by the first
  selected frame captured at or after it (`cue_frame_id`), keeping its
  distance to that frame's `capture_time`, so `cue_time_difference_s` is
  the metadata's rounding (at most 0.5 ms) plus any clock drift before
  that frame.
- `transition_duration_camera_s` is settled minus movement start on the
  camera clock; `transition_duration_metadata_s` is the metadata's
  `transition_duration_s`; `transition_duration_writer_span_s` is the
  `wall_time` span of the frames from the movement-start frame to the
  settled frame, and `transition_largest_writer_stall_s` the largest
  `wall_time` step between them.
- `recording_writer_span_s` and `recording_camera_s` are the whole
  capture's spans on the two clocks (the metadata has no duration column
  for a capture). A capture whose metadata has no movement start or settle
  (an invalid timing) has those cells blank here too.

**Rounding.** Seconds are rounded to 1e-6 (the clocks' resolution),
milliseconds to 1e-3. Differences are camera minus metadata.

**Files.** `static_takes.csv`, `transitions.csv`, this `README.md`, and
`summary.md` (the run's verdict line, the distribution of every difference,
the takes whose static interval differs by more than one frame period,
and the clock evidence for the transitions).
"""


# --- the run --------------------------------------------------------------------------------------
def selection(args, root):
    keys = []
    if not args.static and not args.transition:
        keys += [("static", r["grasp_name"], int(r["take"])) for r in rc.list_static(root)]
        keys += [("transition", r["to_grasp"], int(r["take"]))
                 for r in rc.list_transitions(root)]
    keys += [("static", g, int(t)) for g, t in args.static or []]
    keys += [("transition", g, int(t)) for g, t in args.transition or []]
    return list(dict.fromkeys(keys))


def run(args) -> int:
    root, out = Path(args.dataset), Path(args.out)
    if _is_under(out, root):
        print(f"refusing to write inside the dataset: {out} is inside {root}")
        return 2
    static_rows, transition_rows, failures = [], [], []
    for kind, name, take in selection(args, root):
        try:
            if kind == "static":
                static_rows.append(static_row(rc.find_static(root, name, take)))
            else:
                transition_rows.append(transition_row(rc.find_transition(root, name, take)))
        except (OSError, ValueError, LookupError, KeyError) as e:
            failures.append(f"LOAD FAILED: {kind} {name} take {take}: {e}")
            print(failures[-1])
    flagged = sum(1 for r in static_rows if r.get("static_interval_over_one_period"))
    verdict = (f"ERRATUM {'FAIL' if failures else 'OK'}: static={len(static_rows)} "
               f"transitions={len(transition_rows)} static_interval_over_one_period={flagged}"
               + (f" load_failures={len(failures)}" if failures else ""))
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "static_takes.csv", STATIC_COLUMNS, static_rows)
    _write_csv(out / "transitions.csv", TRANSITION_COLUMNS, transition_rows)
    (out / "README.md").write_text(README.format(date=f"{datetime.now():%Y-%m-%d}", root=root),
                                   encoding="utf-8")
    (out / "summary.md").write_text(
        summary_text(root, static_rows, transition_rows, failures, verdict), encoding="utf-8")
    print(f"wrote {out}")
    print(verdict)
    return 1 if failures else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Recompute the metadata's writer-clock timings on the camera clock "
                    "(a derived erratum; the dataset is not modified).")
    p.add_argument("--dataset", type=Path, required=True,
                   help="the organized dataset root (read only)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output folder (default {DEFAULT_OUT})")
    p.add_argument("--static", nargs=2, action="append", metavar=("GRASP", "TAKE"),
                   help="one static take (repeatable); with --transition absent too, all")
    p.add_argument("--transition", nargs=2, action="append", metavar=("TO_GRASP", "TAKE"),
                   help="one transition capture (repeatable)")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
