r"""Batch replay fidelity over the organized grasp dataset, the four named
scenarios, and the two checks the gate ledger runs (GATES_replay.md R6, R7, R9).

Every recording is read and measured with the replay viewer's own code
(leap_hand.replay_core), so what this reports is what the viewer shows:

  fidelity  recording_fidelity per recording: the forward kinematics of the
            recorded parent-relative chain against the recorded abs26 (gated,
            1e-6 m); the matrix chain against the quaternion chain (gated,
            1e-6 deg and 1e-9 m; two code paths over the same data, a
            cross-check, not an independent measurement); each bone's
            reconstructed axis against its recorded bone vector (independent:
            orientation and position were written separately; reported); the
            three clocks, the segments and the events.
  Paper 3   leap_hand.paper3 on the same reconstructed poses: the residuals of
            Cobos et al. 2009's relationships per recording and pooled per
            class over frames, and the eq. 9-11 fingertip error. A research
            comparison: the paper gives no threshold, so there is no verdict.

Each frame is reconstructed twice: inside recording_fidelity, and once here
for the Paper 3 angles. The dataset is only read; an --out inside it is
refused.

Run (combinable): --all, --static GRASP TAKE, --transition TO_GRASP TAKE
    write into --out: fidelity.json, fidelity_per_recording.csv, events.csv,
    paper3_residuals.csv and .json, paper3_pooled_by_class.csv,
    paper3_fingertip_error.csv, summary.md, and end with one line
      FIDELITY OK: static=<n> transitions=<m> frames=<f> within_tolerance=<k>/<total>
    or FIDELITY FAIL: ... and exit 1 when a recording is outside the declared
    tolerances or cannot be loaded.
  --scenarios   also writes OUT/scenarios/<name>.json for the four SCENARIOS.
Checks (alone):
  --check-scenarios DIR                     SCENARIOS OK | SCENARIOS FAIL: <which>
  --check-dataset-unchanged BASELINE ROOT   DATASET UNCHANGED | DATASET CHANGED

From the repository root:
  .venv\Scripts\python.exe scripts\leap\replay_verify.py --dataset "<root>" --out results\replay_fidelity --all --scenarios
  .venv\Scripts\python.exe scripts\leap\replay_verify.py --check-scenarios results\replay_fidelity
  .venv\Scripts\python.exe scripts\leap\replay_verify.py --check-dataset-unchanged <baseline.json> "<root>"
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from leap_hand import paper3 as p3
from leap_hand import replay_core as rc

DEFAULT_OUT = Path("results") / "replay_fidelity"
KINDS = ("static", "transition")
EVENT_KINDS = ("gap", "id_change", "dropped_frames", "invalid_status", "clock_defect")

# name -> (kind, grasp, take): the four recordings GATES_replay.md R7 names
SCENARIOS = {
    "circular": ("static", "p3_circular_power", 1),
    "prismatic": ("static", "p3_prismatic_precision", 1),
    "timestamp_anomaly": ("static", "p3_circular_precision_thumb_2_fingers", 4),
    "identity_change": ("static", "p2_s03_power_grasp_finger_side", 1),
}
STALL_FRAME_IDS = (151749, 151855)          # the writer-stall window, docs/replay_plan.md
# The known facts --check-scenarios holds the reports to: (value, tolerance).
WRITER_SPAN_S = (1.999, 0.01)
CAMERA_SPAN_S = (1.178, 0.01)
ID_CHANGE_DT_S = (0.455, 0.005)
ID_CHANGE_HAND_IDS = (573, 574)
ID_CHANGE_FRAME_IDS = (88967, 89008)
NOT_GIVEN = ("T_TMC_fe", "T_TMC_aa", "I_CMC", "M_CMC", "R_CMC", "L_CMC")

STAT_KEYS = ("n", "mean", "mae", "rmse", "p05", "p50", "p95", "max_abs", "slope",
             "intercept", "r", "fraction_satisfied", "max_excess")
STAT_GROUPS = (("all", "stats_all"), ("hold", "stats_hold"),
               ("all_main", "stats_all_main"), ("hold_main", "stats_hold_main"))
STAT_COLUMNS = [f"{g}_{k}" for g, _ in STAT_GROUPS for k in STAT_KEYS]
RELATION_COLUMNS = ["relation", "source", "page", "lhs", "coeff", "rhs", "status",
                    "reason", "mcp_order", "duplicate_of"]
RESIDUAL_COLUMNS = ["kind", "name", "take", "class"] + RELATION_COLUMNS + STAT_COLUMNS
POOLED_COLUMNS = (["kind", "class", "recordings", "frames", "hold_frames"]
                  + RELATION_COLUMNS + STAT_COLUMNS)
RECORDING_COLUMNS = [
    "kind", "name", "take", "hand_side", "hand_ids", "frames_selected", "lines_total",
    "excluded_other_hand", "excluded_other_id", "excluded_invalid_status",
    "frame_id_first", "frame_id_last", "frame_ids_missing", "clock_source",
    "clock_fallback", "duration_s", "max_dt_s", "capture_vs_timestamp_ms_max",
    "writer_delay_ms_max", "writer_span_s", "camera_span_s", "segments",
    *[f"n_{k}" for k in EVENT_KINDS], "pos_residual_um_max", "pos_residual_um_mean",
    "chain_deg_max", "bone_dir_deg_max", "bone_dir_deg_mean", "bone_len_sd_mm_max",
    "quat_norm_dev_max", "within_tolerance", "paper3_class",
    # beside the list above: the third gated number, where the worst values sit
    "chain_m_max", "bone_len_sd_bone", "bone_dir_worst_bone", "hold_frames", "file"]
EVENT_COLUMNS = ["kind", "recording_kind", "name", "take", "index", "frame_id_before",
                 "frame_id_after", "dt_s", "detail"]
TIP_STATS = ("n", "mean_pct", "p50_pct", "p95_pct", "max_pct")
TIP_COLUMNS = (["kind", "name", "take", "class", "status"]
               + [f"{f}_{s}" for f in p3.FINGERS for s in TIP_STATS + ("mean_pct_hold",)]
               + ["gesture_error_pct", "gesture_error_pct_hold", "paper_table8_pct",
                  "thumb", "bone_lengths_mm", "note"])
TIP_NOTE = ("eq. 9 divides by |rp| as printed, so a finger whose rebuilt tip curls "
            "back toward its base can pass 100 %; CMC = 0 in both tips; u_i left "
            "out; thumb unavailable, so eq. 11 averages four fingers")
QUIRKS = [
    "Eq. 8 (p.14, PIP ~ 3/2 DIP) repeats Table 6's PIP/DIP relations for a circular "
    "grasp: both rows are reported and the eq. 8 row carries duplicate_of; count "
    "one of them.",
    "Eq. 9 (p.16) divides by |rp| as printed; with the wrist offset u_i unknown the "
    "tips are relative to each finger's base, so where the rebuilt finger curls "
    "back toward its base |rp| is small and the per-frame error passes 100 %.",
    "Table 5 (p.12): the second column is hyperextension, not abduction; display "
    "only, no bound test.",
    "Finger MCP relations use the paper-order MCP pair (abduction first, Table 2 "
    "p.5); the main XYZ-order reading is in the *_main columns.",
    "T_TMC_fe, T_TMC_aa and the four CMC are not given by the camera: their "
    "relations are 'unavailable', with no numbers.",
]


# --- small helpers ----------------------------------------------------------------------
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _label(kind, name, take) -> str:
    return f"{kind} {name} take {take}"


def _clean(x):
    """JSON-safe: numpy scalars and arrays to Python, NaN and inf to None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def _write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(doc), f, indent=1, allow_nan=False)


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


def _scale(v, k):
    return None if v is None else v * k


def _clock(frames, name) -> np.ndarray:
    return np.array([np.nan if getattr(f, name) is None else float(getattr(f, name))
                     for f in frames])


def _dist(v) -> dict:
    v = np.asarray(v, dtype=float)
    if not v.size:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {"n": int(v.size), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def _pct_stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0, **dict.fromkeys(TIP_STATS[1:])}
    return {"n": int(v.size), "mean_pct": float(v.mean()),
            "p50_pct": float(np.percentile(v, 50)),
            "p95_pct": float(np.percentile(v, 95)), "max_pct": float(v.max())}


def _gesture(fingers: dict):
    """Eq. 11: the mean of the finger means (fingers without a value left out)."""
    means = [s["mean_pct"] for s in fingers.values() if s["mean_pct"] is not None]
    return float(np.mean(means)) if means else None


def _path_key(path) -> str:
    """Absolute, normalised and case-folded: how two paths are compared."""
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path)))).casefold()


def _is_under(path, root) -> bool:
    p, r = _path_key(path), _path_key(root)
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


# --- one recording --------------------------------------------------------------------------
def load(root, key, gap_threshold_s) -> rc.Recording:
    kind, name, take = key
    find = rc.find_static if kind == "static" else rc.find_transition
    return find(root, name, take, gap_threshold_s=gap_threshold_s)


def duplicates(rows) -> dict:
    """relation id -> the earlier applied relation it repeats (same lhs, rhs,
    coefficient and kind): eq. 8 against Table 6's PIP/DIP in a circular grasp.
    Both are reported; a reader counts one."""
    seen, out = {}, {}
    for row in rows:
        if row["status"] != "applied":
            continue
        key = (row["lhs"], row["rhs"], row["coeff"], row["kind"])
        if key in seen:
            out[row["id"]] = seen[key]
        else:
            seen[key] = row["id"]
    return out


def analyse(rec: rc.Recording) -> dict:
    """Everything the outputs need from one recording (no Recording kept)."""
    fid = rc.recording_fidelity(rec)                        # reconstructs each frame
    poses = [(r.pos, r.rot) for r in map(rc.reconstruct, rec.frames)]   # once more
    angles = p3.series_angles(poses, rec.hand_side)
    cls, cls_reason = p3.grasp_class(rec.name)
    hold = rc.hold_mask(rec)
    comparison = p3.compare(angles, cls, hold)
    dup = duplicates(comparison["relations"])
    each = [p3.finger_bone_lengths_mm(pos) for pos, _ in poses]
    lengths = {f: tuple(float(np.median([e[f][k] for e in each])) for k in range(4))
               for f in p3.FINGERS}                         # recorded, median over frames
    tip = p3.fingertip_reconstruction_error(angles, lengths, cls)
    per_frame_pct = tip.pop("per_frame_pct", None)

    dts = np.diff(rec.timeline.t)
    gap = None
    if len(dts):
        i = int(np.argmax(dts))
        gap = {"dt_s": float(dts[i]), "frame_id_before": rec.frames[i].frame_id,
               "frame_id_after": rec.frames[i + 1].frame_id}
    wall, cap = _clock(rec.frames, "wall_time"), _clock(rec.frames, "capture_time")
    ts = _clock(rec.frames, "timestamp_us")
    delay = (wall - cap) * 1e3
    pairs = np.abs(np.diff(cap) - np.diff(ts) / 1e6) * 1e3
    return {
        "kind": rec.kind, "name": rec.name, "take": rec.take,
        "label": _label(rec.kind, rec.name, rec.take), "hand_side": rec.hand_side,
        "hand_ids": rec.hand_ids,
        "file": (rec.metadata.get("raw_jsonl_file") or rec.metadata.get("recording_file")
                 or str(rec.path)),
        "fidelity": fid, "class": cls, "class_reason": cls_reason,
        "summary": comparison["summary"],
        "relations": [{**row, "duplicate_of": dup.get(row["id"])}
                      for row in comparison["relations"]],
        "fingertip": tip, "per_frame_pct": per_frame_pct, "lengths_mm": lengths,
        "angles": angles, "hold": hold, "longest_gap": gap,
        "writer_delay_ms": delay[np.isfinite(delay)],
        "clock_pair_ms": pairs[np.isfinite(pairs)],
    }


def recording_row(a) -> dict:
    f = a["fidelity"]
    cc = f["timeline"]["cross_check"]
    ex = f["counts"]["lines_excluded"]
    kinds = Counter(e["kind"] for e in f["events"])
    sds = {n: v["sd"] for n, v in f["bone_length_mm"].items() if v}
    sd_bone = max(sds, key=sds.get) if sds else None
    pos, bone = f["position_residual_m"], f["bone_direction_deg"]
    return {
        "kind": a["kind"], "name": a["name"], "take": a["take"],
        "hand_side": a["hand_side"], "hand_ids": " ".join(map(str, a["hand_ids"] or [])),
        "frames_selected": f["counts"]["frames_selected"],
        "lines_total": f["counts"]["lines_total"],
        "excluded_other_hand": ex["other_hand"], "excluded_other_id": ex["other_id"],
        "excluded_invalid_status": ex["invalid_status"],
        "frame_id_first": f["frame_id"]["first"], "frame_id_last": f["frame_id"]["last"],
        "frame_ids_missing": f["frame_id"]["missing_in_range"],
        "clock_source": f["timeline"]["source"], "clock_fallback": f["timeline"]["fallback"],
        "duration_s": f["timeline"]["duration_s"], "max_dt_s": f["timeline"]["max_dt_s"],
        "capture_vs_timestamp_ms_max": cc["capture_vs_timestamp_interval_max_ms"],
        "writer_delay_ms_max": cc["writer_delay_max_ms"],
        "writer_span_s": cc["writer_span_s"], "camera_span_s": cc["camera_span_s"],
        "segments": len(f["segments"]), **{f"n_{k}": kinds.get(k, 0) for k in EVENT_KINDS},
        "pos_residual_um_max": _scale(pos["max"], 1e6),
        "pos_residual_um_mean": _scale(pos["mean"], 1e6),
        "chain_deg_max": f["chain_deg"]["max"], "bone_dir_deg_max": bone["max"],
        "bone_dir_deg_mean": bone["mean"], "bone_len_sd_mm_max": sds.get(sd_bone),
        "quat_norm_dev_max": f["quat_norm"]["max_deviation"],
        "within_tolerance": f["within_tolerance"], "paper3_class": a["class"],
        "chain_m_max": f["chain_m"]["max"], "bone_len_sd_bone": sd_bone,
        "bone_dir_worst_bone": bone["worst_bone"],
        "hold_frames": None if a["hold"] is None else int(a["hold"].sum()),
        "file": a["file"],
    }


def event_rows(a) -> list:
    return [{"kind": e["kind"], "recording_kind": a["kind"], "name": a["name"],
             "take": a["take"], "index": e["index"],
             "frame_id_before": e["frame_id_before"], "frame_id_after": e["frame_id_after"],
             "dt_s": e["dt_s"], "detail": e["detail"]} for e in a["fidelity"]["events"]]


def relation_row(meta: dict, row: dict) -> dict:
    out = {**meta, "relation": row["id"], "source": row["source"], "page": row["page"],
           "lhs": row["lhs"], "coeff": row["coeff"], "rhs": row["rhs"],
           "status": row["status"], "reason": row["reason"],
           "mcp_order": row["mcp_order"], "duplicate_of": row["duplicate_of"]}
    for group, key in STAT_GROUPS:
        stats = row.get(key) or {}
        out.update({f"{group}_{k}": stats.get(k) for k in STAT_KEYS})
    return out


def fingertip_row(a) -> dict:
    tip = a["fingertip"]
    row = {"kind": a["kind"], "name": a["name"], "take": a["take"], "class": a["class"],
           "status": tip["status"], "thumb": tip.get("thumb"),
           "paper_table8_pct": tip.get("paper_table8_pct"),
           "gesture_error_pct": tip.get("gesture_error_pct"),
           "note": TIP_NOTE if tip["status"] == "applied" else tip.get("reason"),
           "bone_lengths_mm": "; ".join(
               f"{f} " + "/".join(f"{x:.2f}" for x in a["lengths_mm"][f])
               for f in p3.FINGERS)}
    hold = {}
    if a["per_frame_pct"] is not None and a["hold"] is not None:
        hold = {f: _pct_stats(a["per_frame_pct"][f][a["hold"]]) for f in p3.FINGERS}
        row["gesture_error_pct_hold"] = _gesture(hold)
    for f in p3.FINGERS:
        stats = (tip.get("fingers") or {}).get(f) or {}
        row.update({f"{f}_{s}": stats.get(s) for s in TIP_STATS})
        row[f"{f}_mean_pct_hold"] = hold[f]["mean_pct"] if hold else None
    return row


# --- the whole selection ----------------------------------------------------------------------
def pooled_by_class(results) -> list:
    """p3.compare over ONE series per (recording kind, class): every recording's
    frames concatenated, so each frame counts once; a recording mean is never
    averaged. Static takes give all frames and hold frames (their metadata
    static intervals); transitions all frames."""
    out = []
    for kind in KINDS:
        for cls in p3.CLASSES:
            group = [a for a in results if a["kind"] == kind and a["class"] == cls]
            if not group:
                continue
            angles = {n: np.concatenate([a["angles"][n] for a in group])
                      for n in p3.ANGLE_NAMES}
            hold = None
            if kind == "static":
                hold = np.concatenate([
                    np.zeros(len(a["angles"]["I_DIP"]), bool) if a["hold"] is None
                    else a["hold"] for a in group])
            comparison = p3.compare(angles, cls, hold)
            dup = duplicates(comparison["relations"])
            tip = None
            if group[0]["per_frame_pct"] is not None:       # circular and prismatic
                per = {f: np.concatenate([a["per_frame_pct"][f] for a in group])
                       for f in p3.FINGERS}
                tip = {"paper_table8_pct": p3.TABLE8_9DOF_PCT[cls]}
                for label, mask in (("all", None), ("hold", hold)):
                    if label == "hold" and mask is None:
                        tip[label] = None
                        continue
                    fingers = {f: _pct_stats(v if mask is None else v[mask])
                               for f, v in per.items()}
                    tip[label] = {"fingers": fingers, "gesture_error_pct": _gesture(fingers)}
            out.append({
                "kind": kind, "class": cls, "recordings": len(group),
                "frames": int(len(angles["I_DIP"])),
                "hold_frames": None if hold is None else int(hold.sum()),
                "summary": comparison["summary"],
                "relations": [{**row, "duplicate_of": dup.get(row["id"])}
                              for row in comparison["relations"]],
                "fingertip_error": tip})
    return out


def _worst(results, value, where=None, tolerance=None):
    scored = [(value(a), a) for a in results]
    scored = [(v, a) for v, a in scored if v is not None]
    if not scored:
        return None
    v, a = max(scored, key=lambda s: s[0])
    return {"value": v, "tolerance": tolerance, "recording": a["label"],
            "kind": a["kind"], "name": a["name"], "take": a["take"],
            **(where(a) if where else {})}


def summarise(results, failures, n_selected):
    fd = lambda a: a["fidelity"]
    cc = lambda a: fd(a)["timeline"]["cross_check"]
    tol = {k: v["value"] for k, v in rc.TOLERANCES.items()}

    def kind_totals(group):
        excluded = Counter()
        for a in group:
            excluded.update(fd(a)["counts"]["lines_excluded"])
        return {"recordings": len(group),
                "frames_selected": sum(fd(a)["counts"]["frames_selected"] for a in group),
                "hold_frames": sum(int(a["hold"].sum()) for a in group
                                   if a["hold"] is not None),
                "lines_total": sum(fd(a)["counts"]["lines_total"] for a in group),
                "lines_excluded": {k: excluded.get(k, 0)
                                   for k in ("other_hand", "other_id", "invalid_status")},
                "within_tolerance": sum(bool(fd(a)["within_tolerance"]) for a in group),
                "events_by_kind": dict(sorted(Counter(
                    e["kind"] for a in group for e in fd(a)["events"]).items()))}

    every = kind_totals(results)
    totals = {
        "recordings_selected": n_selected,
        "static": sum(a["kind"] == "static" for a in results),
        "transitions": sum(a["kind"] == "transition" for a in results),
        "frames_selected": every["frames_selected"], "hold_frames": every["hold_frames"],
        "lines_total": every["lines_total"], "lines_excluded": every["lines_excluded"],
        "within_tolerance": every["within_tolerance"],
        "outside_tolerance": [a["label"] for a in results
                              if not fd(a)["within_tolerance"]],
        "load_failures": failures,
        "events_by_kind": every["events_by_kind"],
        "recordings_with_events": [
            {"recording": a["label"], "events": dict(Counter(e["kind"]
                                                             for e in fd(a)["events"]))}
            for a in results if fd(a)["events"]],
        "clock_sources": dict(Counter(fd(a)["timeline"]["source"] for a in results)),
        "clock_fallbacks": [a["label"] for a in results if fd(a)["timeline"]["fallback"]],
        "by_kind": {k: kind_totals([a for a in results if a["kind"] == k]) for k in KINDS},
    }
    sd_of = lambda a: {n: v["sd"] for n, v in fd(a)["bone_length_mm"].items() if v}
    worst = {
        "position_residual_um": _worst(
            results, lambda a: _scale(fd(a)["position_residual_m"]["max"], 1e6),
            lambda a: {"frame_id": fd(a)["position_residual_m"]["worst_frame_id"],
                       "joint": fd(a)["position_residual_m"]["worst_joint"]},
            tol["position_m"] * 1e6),
        "chain_deg": _worst(results, lambda a: fd(a)["chain_deg"]["max"],
                            lambda a: {"frame_id": fd(a)["chain_deg"]["worst_frame_id"]},
                            tol["chain_deg"]),
        "chain_m": _worst(results, lambda a: fd(a)["chain_m"]["max"],
                          lambda a: {"frame_id": fd(a)["chain_m"]["worst_frame_id"]},
                          tol["chain_m"]),
        "bone_direction_deg": _worst(
            results, lambda a: fd(a)["bone_direction_deg"]["max"],
            lambda a: {"frame_id": fd(a)["bone_direction_deg"]["worst_frame_id"],
                       "bone": fd(a)["bone_direction_deg"]["worst_bone"]},
            tol["bone_direction_deg_expectation"]),
        "capture_vs_timestamp_interval_ms": _worst(
            results, lambda a: cc(a)["capture_vs_timestamp_interval_max_ms"],
            tolerance=tol["clock_interval_ms"]),
        "writer_delay_ms": _worst(results, lambda a: cc(a)["writer_delay_max_ms"]),
        "longest_gap_s": _worst(
            results, lambda a: a["longest_gap"] and a["longest_gap"]["dt_s"],
            lambda a: {"frame_id_before": a["longest_gap"]["frame_id_before"],
                       "frame_id_after": a["longest_gap"]["frame_id_after"]}),
        "quat_norm_deviation": _worst(results,
                                      lambda a: fd(a)["quat_norm"]["max_deviation"],
                                      tolerance=tol["quat_norm"]),
        "bone_length_sd_mm": _worst(
            results, lambda a: max(sd_of(a).values(), default=None),
            lambda a: {"bone": max(sd_of(a), key=sd_of(a).get)}),
    }
    shortest = None
    for a in results:
        for bone, v in fd(a)["bone_length_mm"].items():
            if v and (shortest is None or v["min"] < shortest["value_mm"]):
                shortest = {"value_mm": v["min"], "bone": bone, "recording": a["label"]}
    if shortest:   # abs26 rounding (sqrt(3)/2 um per end) over the shortest bone
        shortest["rounding_bound_deg"] = math.degrees(math.atan(
            math.sqrt(3) * 1e-6 / (shortest["value_mm"] * 1e-3)))
    means = {
        "position_residual_um": _scale(_frame_weighted(results, "position_residual_m"), 1e6),
        "bone_direction_deg": _frame_weighted(results, "bone_direction_deg")}
    clocks = {}
    for k in KINDS + ("all",):
        group = [a for a in results if k in ("all", a["kind"])]
        clocks[k] = {
            "writer_delay_ms": _dist(np.concatenate(
                [a["writer_delay_ms"] for a in group]) if group else []),
            "capture_vs_timestamp_interval_ms": _dist(np.concatenate(
                [a["clock_pair_ms"] for a in group]) if group else [])}
    return totals, worst, shortest, means, clocks


def _frame_weighted(results, key):
    """Mean over every joint (or bone) of every measured frame of the selection:
    each recording's mean weighted by its measured frames."""
    num = den = 0.0
    for a in results:
        f = a["fidelity"]
        m, n = f[key]["mean"], f["position_residual_m"]["frames_measured"]
        if m is not None and n:
            num, den = num + m * n, den + n
    return num / den if den else None


# --- the scenario reports -----------------------------------------------------------------------
def _window(rec, lo, hi) -> dict:
    """Clock spans over the selected frames with lo <= frame_id <= hi."""
    win = [f for f in rec.frames if lo <= f.frame_id <= hi]

    def span(values):
        v = [x for x in values if x is not None]
        return float(max(v) - min(v)) if len(v) >= 2 else None

    camera = span([f.timestamp_us for f in win])
    out = {"frame_id_first": lo, "frame_id_last": hi, "frames": len(win),
           "writer_span_s": span([f.wall_time for f in win]),
           "camera_span_s": None if camera is None else camera / 1e6,
           "capture_span_s": span([f.capture_time for f in win]),
           "clock_source": rec.timeline.source, "clock_fallback": rec.timeline.fallback,
           "clock_reason": rec.timeline.reason, "largest_writer_jump": None}
    jumps = [(b.wall_time - a.wall_time, i) for i, (a, b) in enumerate(zip(win, win[1:]))
             if a.wall_time is not None and b.wall_time is not None]
    if jumps:
        dj, i = max(jumps)
        a, b = win[i], win[i + 1]
        out["largest_writer_jump"] = {
            "writer_s": dj, "frame_id_before": a.frame_id, "frame_id_after": b.frame_id,
            "camera_ms": (None if None in (a.timestamp_us, b.timestamp_us)
                          else (b.timestamp_us - a.timestamp_us) / 1e3),
            "capture_ms": (None if None in (a.capture_time, b.capture_time)
                           else (b.capture_time - a.capture_time) * 1e3)}
    return out


def _midpoint(rec):
    """pose_at, interpolate mode, halfway across the first segment boundary."""
    if len(rec.segments) < 2:
        return None
    s1, s2 = rec.segments[0], rec.segments[1]
    t = rec.timeline.t
    t_mid = float((t[s1.end] + t[s2.start]) / 2.0)
    shown = rc.pose_at(rec, t_mid, "interpolate")
    return {"mode": "interpolate", "t_s": t_mid,
            "between_frame_ids": [rec.frames[s1.end].frame_id,
                                  rec.frames[s2.start].frame_id],
            "source": shown.source, "note": shown.note, "i0": shown.i0, "i1": shown.i1,
            "alpha": shown.alpha,
            "frame_id": None if shown.frame is None else shown.frame.frame_id}


def scenario_report(name, rec, a, root) -> dict:
    report = {
        "scenario": name, "generated": _now(), "dataset": str(root),
        "recording": {"kind": rec.kind, "name": rec.name, "take": rec.take,
                      "hand_side": rec.hand_side, "hand_ids": rec.hand_ids,
                      "provenance": rec.provenance, "file": a["file"]},
        "fidelity": a["fidelity"],
        "paper3": {"class": a["class"], "class_reason": a["class_reason"],
                   "summary": a["summary"], "relations": a["relations"],
                   "note": p3.COMPARE_NOTE},
    }
    if name in ("circular", "prismatic"):
        status = {n: "available" if np.isfinite(a["angles"][n]).any() else "unavailable"
                  for n in p3.ANGLE_NAMES}               # measured on the series
        report["coordinate_status"] = status
        report["unavailable_coordinates"] = [n for n in p3.ANGLE_NAMES
                                             if status[n] == "unavailable"]
        report["fingertip_error"] = a["fingertip"]
    if name == "timestamp_anomaly":
        report["window"] = _window(rec, *STALL_FRAME_IDS)
    if name == "identity_change":
        report["events"] = a["fidelity"]["events"]
        report["segments"] = a["fidelity"]["segments"]
        report["midpoint_pose"] = _midpoint(rec)
    return report


def _near(value, target, tol) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and abs(value - target) <= tol)


def _check_anomaly(s) -> list:
    w, p = s["window"], []
    if not _near(w.get("writer_span_s"), *WRITER_SPAN_S):
        p.append(f"writer span {w.get('writer_span_s')} s not within "
                 f"{WRITER_SPAN_S[1]} s of {WRITER_SPAN_S[0]} s")
    if not _near(w.get("camera_span_s"), *CAMERA_SPAN_S):
        p.append(f"camera span {w.get('camera_span_s')} s not within "
                 f"{CAMERA_SPAN_S[1]} s of {CAMERA_SPAN_S[0]} s")
    if w.get("clock_source") != "timestamp_us":
        p.append(f"clock source {w.get('clock_source')!r}, expected 'timestamp_us'")
    if w.get("clock_fallback") is not False:
        p.append(f"clock fallback {w.get('clock_fallback')!r}, expected false")
    return p


def _check_identity(s) -> list:
    p = []
    changes = [e for e in s["events"] if e.get("kind") == "id_change"]
    if len(changes) != 1:
        p.append(f"{len(changes)} id_change events, expected 1")
    else:
        e = changes[0]
        if (e.get("frame_id_before"), e.get("frame_id_after")) != ID_CHANGE_FRAME_IDS:
            p.append(f"id change between frame ids {e.get('frame_id_before')} and "
                     f"{e.get('frame_id_after')}, expected {ID_CHANGE_FRAME_IDS}")
        if not _near(e.get("dt_s"), *ID_CHANGE_DT_S):
            p.append(f"camera gap {e.get('dt_s')} s not within {ID_CHANGE_DT_S[1]} s "
                     f"of {ID_CHANGE_DT_S[0]} s")
    segs = s["segments"]
    if len(segs) != 2:
        p.append(f"{len(segs)} segments, expected 2")
    elif (segs[0].get("hand_id"), segs[1].get("hand_id")) != ID_CHANGE_HAND_IDS:
        p.append(f"hand ids {segs[0].get('hand_id')} -> {segs[1].get('hand_id')}, "
                 f"expected {ID_CHANGE_HAND_IDS}")
    m = s.get("midpoint_pose") or {}
    if m.get("source") != "held":
        p.append(f"midpoint pose source {m.get('source')!r}, expected 'held' "
                 "(no interpolation across the boundary)")
    elif m.get("frame_id") != ID_CHANGE_FRAME_IDS[0]:
        p.append(f"midpoint pose holds frame {m.get('frame_id')}, expected "
                 f"{ID_CHANGE_FRAME_IDS[0]}")
    return p


def _check_grasp(s) -> list:
    p = []
    if not any(r.get("status") == "applied" and ((r.get("stats_all") or {}).get("n") or 0) > 0
               for r in s["paper3"]["relations"]):
        p.append("no applied relation with stats_all n > 0")
    status = s.get("coordinate_status") or {}
    wrong = [n for n in NOT_GIVEN if status.get(n) != "unavailable"]
    if wrong:
        p.append(f"not marked unavailable: {', '.join(wrong)}")
    return p


_CHECKS = {"circular": _check_grasp, "prismatic": _check_grasp,
           "timestamp_anomaly": _check_anomaly, "identity_change": _check_identity}


def check_scenarios(directory) -> list:
    """The four reports against the known facts: one failure string per
    scenario that does not hold them ('<name>: <what>'); empty when all do.
    `directory` is the --out folder (its scenarios/ is read) or scenarios/."""
    d = Path(directory)
    if (d / "scenarios").is_dir():
        d = d / "scenarios"
    failures = []
    for name, (_, grasp, take) in SCENARIOS.items():
        path = d / f"{name}.json"
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            failures.append(f"{name}: cannot read {path} ({type(e).__name__})")
            continue
        try:
            rec = s.get("recording") or {}
            problems = ([] if (rec.get("name"), rec.get("take")) == (grasp, take) else
                        [f"report is for {rec.get('name')} take {rec.get('take')}, "
                         f"expected {grasp} take {take}"])
            problems += _CHECKS[name](s)
        except (KeyError, TypeError, AttributeError, IndexError) as e:
            problems = [f"malformed report ({type(e).__name__}: {e})"]
        if problems:
            failures.append(f"{name}: {'; '.join(problems)}")
    return failures


# --- the dataset-unchanged check ------------------------------------------------------------------
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_dataset_unchanged(baseline_json, dataset_root):
    """(ok, lines): every baseline entry under dataset_root rehashed. An empty
    intersection is a failure: nothing compared is nothing proved."""
    try:
        base = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, [f"DATASET CHECK FAILED: cannot read {baseline_json} ({e})"]
    if not isinstance(base, dict):
        return False, [f"DATASET CHECK FAILED: {baseline_json} is not a JSON object "
                       "of path -> sha256"]
    prefix = _path_key(dataset_root).rstrip("\\/") + os.sep
    entries = sorted((p, str(h).lower()) for p, h in base.items()
                     if _path_key(p).startswith(prefix))
    if not entries:
        return False, [f"DATASET CHECK FAILED: no baseline entry lies under "
                       f"{dataset_root} ({len(base)} entries in {baseline_json}); "
                       "nothing was compared"]
    bad = []
    for path, digest in entries:
        try:
            if _sha256(path) != digest:
                bad.append(f"differs: {path}")
        except OSError as e:
            bad.append(f"missing: {path} ({type(e).__name__})")
    if bad:
        return False, ([f"DATASET CHANGED: {len(bad)} files differ/missing (of "
                        f"{len(entries)} baseline files under the root)"]
                       + [f"  {b}" for b in bad[:10]]
                       + ([f"  ... and {len(bad) - 10} more"] if len(bad) > 10 else []))
    jsonl = sum(p.lower().endswith(".jsonl") for p, _ in entries)
    types = ", ".join(sorted({os.path.splitext(p)[1].lower() or "(none)"
                              for p, _ in entries}))
    return True, [f"DATASET UNCHANGED ({len(entries)} files, {jsonl} JSONL)",
                  f"  the baseline's files under this root are of type {types}; "
                  "files of other types are not in it and were not compared"]


# --- outputs ---------------------------------------------------------------------------------------
def _g(v, fmt=".4g") -> str:
    return "n/a" if v is None else format(v, fmt)


def summary_markdown(doc, line) -> str:
    t, w, bk = doc["totals"], doc["worst"], doc["totals"]["by_kind"]
    rows = [("recordings", "recordings"), ("frames selected", "frames_selected"),
            ("hold frames (static interval)", "hold_frames"),
            ("lines in the files", "lines_total")]
    out = ["# Replay fidelity run", "",
           f"- generated {doc['generated']}, wall time {doc['wall_time_s']:.1f} s",
           f"- dataset `{doc['dataset']}`",
           f"- gap threshold {doc['gap_threshold_s']} s", "",
           "```", line, "```", "", "## Totals", "",
           "| | static | transition | all |", "|---|---:|---:|---:|"]
    for label, key in rows:
        out.append(f"| {label} | {bk['static'][key]} | {bk['transition'][key]} | "
                   f"{bk['static'][key] + bk['transition'][key]} |")
    for reason in ("other_hand", "other_id", "invalid_status"):
        s, tr = bk["static"]["lines_excluded"][reason], bk["transition"]["lines_excluded"][reason]
        out.append(f"| lines excluded: {reason} | {s} | {tr} | {s + tr} |")
    out.append(f"| within tolerance | {bk['static']['within_tolerance']} | "
               f"{bk['transition']['within_tolerance']} | {t['within_tolerance']} |")
    if t["load_failures"]:
        out += ["", "Load failures:"] + [f"- {f['recording']}: {f['error']}"
                                         for f in t["load_failures"]]
    out += ["", "## Worst values", "",
            "| quantity | worst | declared | recording | where |", "|---|---:|---:|---|---|"]
    for key, label in (("position_residual_um", "FK vs abs26 (um)"),
                       ("chain_deg", "matrix vs quaternion chain (deg)"),
                       ("chain_m", "matrix vs quaternion chain (m)"),
                       ("bone_direction_deg", "bone axis vs recorded bone (deg)"),
                       ("capture_vs_timestamp_interval_ms",
                        "capture_time vs timestamp_us interval (ms)"),
                       ("writer_delay_ms", "writer delay wall - capture (ms)"),
                       ("longest_gap_s", "longest camera-clock step (s)"),
                       ("quat_norm_deviation", "relative quaternion norm - 1"),
                       ("bone_length_sd_mm", "bone length SD in a recording (mm)")):
        e = w.get(key)
        if not e:
            continue
        where = ", ".join(f"{k} {e[k]}" for k in ("frame_id", "joint", "bone",
                                                  "frame_id_before", "frame_id_after")
                          if e.get(k) is not None)
        out.append(f"| {label} | {_g(e['value'])} | {_g(e['tolerance'])} | "
                   f"{e['recording']} | {where} |")
    sb = doc.get("shortest_bone")
    if sb:
        out += ["", f"Shortest recorded bone {sb['value_mm']:.2f} mm ({sb['bone']}, "
                    f"{sb['recording']}): abs26 rounding alone allows up to "
                    f"{sb['rounding_bound_deg']:.4f} deg of bone direction there."]
    out += ["", "## Clocks", "",
            f"Sources: {', '.join(f'{k} {v}' for k, v in t['clock_sources'].items())}; "
            f"fallbacks: {', '.join(t['clock_fallbacks']) or 'none'}.", "",
            "| | writer delay median / p95 / max (ms) | capture vs timestamp "
            "interval median / p95 / max (ms) |", "|---|---|---|"]
    for k, c in doc["clocks"].items():
        wd, cv = c["writer_delay_ms"], c["capture_vs_timestamp_interval_ms"]
        out.append(f"| {k} | {_g(wd['median'])} / {_g(wd['p95'])} / {_g(wd['max'])} | "
                   f"{_g(cv['median'])} / {_g(cv['p95'])} / {_g(cv['max'])} |")
    out += ["", "## Events", "", "| kind | static | transition |", "|---|---:|---:|"]
    for k in EVENT_KINDS:
        out.append(f"| {k} | {bk['static']['events_by_kind'].get(k, 0)} | "
                   f"{bk['transition']['events_by_kind'].get(k, 0)} |")
    out += ["", "Recordings with events:", ""]
    out += ([f"- {r['recording']}: " + ", ".join(f"{k} {n}" for k, n in r["events"].items())
             for r in t["recordings_with_events"]] or ["- none"])
    out += ["", "## Paper 3", "",
            "A research comparison, not a fidelity check: the paper gives no "
            "threshold. Per recording in paper3_residuals.csv, pooled over frames "
            "per class in paper3_pooled_by_class.csv, the eq. 9-11 fingertip error "
            "in paper3_fingertip_error.csv. See docs/replay_fidelity.md.", ""]
    return "\n".join(out)


def write_outputs(out, root, args, results, failures, n_selected, wall_s, line):
    totals, worst, shortest, means, clocks = summarise(results, failures, n_selected)
    pooled = pooled_by_class(results)
    doc = {
        "generated": _now(), "dataset": str(root), "gap_threshold_s": args.gap_threshold,
        "selection": {"all": args.all, "static": args.static or [],
                      "transition": args.transition or []},
        "summary_line": line, "wall_time_s": wall_s,
        "tolerances": rc.TOLERANCES, "totals": totals, "worst": worst,
        "shortest_bone": shortest, "frame_weighted_means": means, "clocks": clocks,
        "per_recording": [{"kind": a["kind"], "name": a["name"], "take": a["take"],
                           **a["fidelity"],
                           "paper3": {"class": a["class"], "class_reason": a["class_reason"],
                                      "summary": a["summary"],
                                      "gesture_error_pct": a["fingertip"].get(
                                          "gesture_error_pct")}}
                          for a in results],
    }
    _write_json(out / "fidelity.json", doc)
    _write_csv(out / "fidelity_per_recording.csv", RECORDING_COLUMNS,
               [recording_row(a) for a in results])
    _write_csv(out / "events.csv", EVENT_COLUMNS,
               [r for a in results for r in event_rows(a)])
    _write_csv(out / "paper3_residuals.csv", RESIDUAL_COLUMNS, [
        relation_row({"kind": a["kind"], "name": a["name"], "take": a["take"],
                      "class": a["class"]}, row)
        for a in results for row in a["relations"]])
    _write_csv(out / "paper3_pooled_by_class.csv", POOLED_COLUMNS, [
        relation_row({k: g[k] for k in ("kind", "class", "recordings", "frames",
                                        "hold_frames")}, row)
        for g in pooled for row in g["relations"]])
    _write_csv(out / "paper3_fingertip_error.csv", TIP_COLUMNS,
               [fingertip_row(a) for a in results])
    _write_json(out / "paper3_residuals.json", {
        "generated": doc["generated"], "dataset": str(root),
        "note": p3.COMPARE_NOTE, "fingertip_note": p3.ERROR_NOTE, "quirks": QUIRKS,
        "per_recording": [{"kind": a["kind"], "name": a["name"], "take": a["take"],
                           "class": a["class"], "class_reason": a["class_reason"],
                           "summary": a["summary"], "relations": a["relations"],
                           "fingertip_error": a["fingertip"]} for a in results],
        "pooled_by_class": pooled})
    (out / "summary.md").write_text(summary_markdown(doc, line), encoding="utf-8")


# --- run ------------------------------------------------------------------------------------------
def selection(args, root) -> list:
    keys = []
    if args.all:
        keys += [("static", r["grasp_name"], int(r["take"])) for r in rc.list_static(root)]
        keys += [("transition", r["to_grasp"], int(r["take"]))
                 for r in rc.list_transitions(root)]
    keys += [("static", g, int(t)) for g, t in args.static or []]
    keys += [("transition", g, int(t)) for g, t in args.transition or []]
    return list(dict.fromkeys(keys))                         # in order, each once


def run(args) -> int:
    root, out = Path(args.dataset), Path(args.out)
    if _is_under(out, root):
        print(f"refusing to write inside the dataset: {out} is inside {root}")
        return 2
    keys = selection(args, root)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    wanted = {key: name for name, key in SCENARIOS.items()} if args.scenarios else {}
    results, failures, written, scenario_failures = [], [], [], []

    def write_scenario(name, rec, a):
        _write_json(out / "scenarios" / f"{name}.json", scenario_report(name, rec, a, root))
        written.append(name)

    for i, key in enumerate(keys, 1):
        try:
            rec = load(root, key, args.gap_threshold)
            a = analyse(rec)
        except Exception as e:   # reported and counted, never skipped silently
            failures.append({"recording": _label(*key), "error": f"{type(e).__name__}: {e}"})
            print(f"LOAD FAILED: {_label(*key)}: {type(e).__name__}: {e}")
            continue
        results.append(a)
        if key in wanted:
            write_scenario(wanted[key], rec, a)
        if i % 20 == 0 or i == len(keys):
            print(f"[{i:>3}/{len(keys)}] {time.perf_counter() - started:6.1f} s  "
                  f"{_label(*key)}", flush=True)
    for name, key in SCENARIOS.items():
        if args.scenarios and name not in written:
            try:
                rec = load(root, key, args.gap_threshold)
                write_scenario(name, rec, analyse(rec))
            except Exception as e:
                scenario_failures.append(name)
                print(f"SCENARIO FAILED: {name} ({_label(*key)}): {type(e).__name__}: {e}")
    if args.scenarios:
        print(f"SCENARIOS WRITTEN: {len(written)} of {len(SCENARIOS)} in "
              f"{out / 'scenarios'}")
    ok = not scenario_failures
    if keys:
        static = sum(a["kind"] == "static" for a in results)
        frames = sum(a["fidelity"]["counts"]["frames_selected"] for a in results)
        within = sum(bool(a["fidelity"]["within_tolerance"]) for a in results)
        good = within == len(keys) and not failures
        line = (f"FIDELITY {'OK' if good else 'FAIL'}: static={static} "
                f"transitions={len(results) - static} frames={frames} "
                f"within_tolerance={within}/{len(keys)}"
                + ("" if good else f" load_failures={len(failures)}"))
        wall_s = time.perf_counter() - started
        write_outputs(out, root, args, results, failures, len(keys), wall_s, line)
        for a in results:
            if not a["fidelity"]["within_tolerance"]:
                detail = a["fidelity"]["within_tolerance_detail"]
                print(f"OUTSIDE TOLERANCE: {a['label']}: " + ", ".join(
                    f"{k} {v['measured']} > {v['limit']}" for k, v in detail.items()
                    if not v["ok"]))
        print(f"wrote {out}; wall time {time.perf_counter() - started:.1f} s")
        print(line)
        ok = ok and good
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Replay fidelity over the organized grasp dataset, the four "
                    "named scenarios, and the scenario and dataset checks.")
    p.add_argument("--dataset", type=Path, help="the organized dataset root (read only)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output folder (default {DEFAULT_OUT})")
    p.add_argument("--gap-threshold", type=float, default=rc.GAP_THRESHOLD_S,
                   help="camera-clock step that splits segments, seconds (default "
                        f"{rc.GAP_THRESHOLD_S})")
    p.add_argument("--all", action="store_true",
                   help="every static take and every transition in the metadata tables")
    p.add_argument("--static", nargs=2, action="append", metavar=("GRASP", "TAKE"),
                   help="one static take (repeatable)")
    p.add_argument("--transition", nargs=2, action="append", metavar=("TO_GRASP", "TAKE"),
                   help="one transition capture (repeatable)")
    p.add_argument("--scenarios", action="store_true",
                   help="also write OUT/scenarios/<name>.json for the four scenarios")
    p.add_argument("--check-scenarios", type=Path, metavar="DIR",
                   help="check the four scenario reports in DIR against the known facts")
    p.add_argument("--check-dataset-unchanged", nargs=2,
                   metavar=("BASELINE_JSON", "DATASET_ROOT"),
                   help="rehash every baseline file under DATASET_ROOT")
    args = p.parse_args(argv)
    runs = args.all or args.static or args.transition or args.scenarios
    checks = [c for c in (args.check_scenarios, args.check_dataset_unchanged) if c]
    if len(checks) > 1 or (checks and runs):
        p.error("--check-scenarios and --check-dataset-unchanged each run alone")
    if args.check_scenarios:
        failures = check_scenarios(args.check_scenarios)
        if not failures:
            print("SCENARIOS OK")
            return 0
        print("SCENARIOS FAIL: " + ", ".join(f.split(":")[0] for f in failures))
        for f in failures:
            print(f"  {f}")
        return 1
    if args.check_dataset_unchanged:
        ok, lines = check_dataset_unchanged(*args.check_dataset_unchanged)
        print("\n".join(lines))
        return 0 if ok else 1
    if not runs:
        p.error("nothing to do: give --all, --static, --transition, --scenarios "
                "or one of the checks")
    if args.dataset is None:
        p.error("--dataset is required for a run")
    if not args.gap_threshold > 0:
        p.error("--gap-threshold must be positive")
    for _, take in (args.static or []) + (args.transition or []):
        if not take.lstrip("-").isdigit():
            p.error(f"TAKE must be an integer, got {take!r}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
