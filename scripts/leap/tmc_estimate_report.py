r"""The open-hand thumb TMC estimate over the organized grasp dataset.

ESTIMATES, NOT VALIDATED PAPER 3 COORDINATES (`leap_hand.tmc_estimate`).
Calibrates the rest frame on the transitions' open-hand frames, then reads
every static take and every transition with the replay viewer's own code
(`leap_hand.replay_core`) and writes into --out:

  calibration.json             the Calibration: R_rest, the per-transition
                               table, the agreement, the skipped rows
  assumption_checks.json       A1 to A5 measured or stated, and the
                               metadata-vs-timeline time origin check
  estimates_per_recording.csv  per recording: representative frame (static
                               takes), hold-frame and all-frame mean/min/max,
                               residual max, metacarpal motion, provenance
  estimates_per_frame.csv      one row per selected frame
  estimates_by_class.csv       static takes per Paper 3 class: min, median
                               and max over the takes of the hold-frame means

and ends with one line
  TMC ESTIMATE REPORT OK: transitions_used=<n> recordings=<m>
or TMC ESTIMATE REPORT FAIL: <why> (exit 1). Both counts are measured here:
n is the number of transitions the calibration used, checked against this
script's own count of the table's rows with a movement start; m is the
number of recordings read and estimated. The dataset is only read; an --out
inside it is refused (exit 2).

From the repository root:
  .venv\Scripts\python.exe scripts\leap\tmc_estimate_report.py --dataset "<root>" --out results\replay_fidelity\tmc_estimate
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from leap_hand import paper3 as p3
from leap_hand import replay_core as rc
from leap_hand import tmc_estimate as te
from xr_hand.joint_frames import fmt_num

DEFAULT_OUT = Path("results") / "replay_fidelity" / "tmc_estimate"
FE, AA, TW, RES = te.KEYS
STATS = ("mean", "min", "max")
FRAME_COLUMNS = ["kind", "name", "take", "frame_index", "frame_id", "t_s", *te.KEYS]
RECORDING_COLUMNS = (
    ["kind", "name", "take", "class", "n_frames", "rep_frame_id",
     f"{FE}_rep", f"{AA}_rep", f"{TW}_rep", "hold_frames"]
    + [f"{k}_hold_{s}" for k in (FE, AA) for s in STATS]
    + [f"{k}_all_{s}" for k in (FE, AA, TW) for s in STATS]
    + [f"{RES}_max", "metacarpal_spread_median_deg", "metacarpal_spread_max_deg",
       "metacarpal_mean_to_rest_deg", "metacarpal_max_frame_to_rest_deg",
       "calibration"])
CLASS_COLUMNS = (["class", "static_takes"]
                 + [f"{k}_hold_mean_{s}" for k in (FE, AA)
                    for s in ("min", "median", "max")])

A1_TEXT = (
    "The paper's thumb chain (Table 3, p.5) has the TMC flexion axis parallel "
    "to the MCP and IP flexion axes (joints 2 to 4, alpha = 0); the estimate "
    "takes the tracker's metacarpal x axis as that axis. Measured on every "
    "selected frame: the angle between the x (hinge) axes of the thumb "
    "metacarpal and proximal frames, and of the proximal and distal frames. "
    "metacarpal_palm_deg is the metacarpal's x against the palm's x: the "
    "reason the wrist-frame T_TMC pair is not the paper's.")
A2_TEXT = (
    "The open-hand rest orientation is stable across the session: the angle "
    "of each transition's open-hand mean (R_palm^T R_metacarpal) to the "
    "session mean R_rest. Its median, p90 and max are the estimate's floor "
    "uncertainty. spread_*: the frames of one window about that window's mean.")
A3_TEXT = (
    "The open hand is the zero. Paper 3 gives its zero only as a drawing "
    "(Figure 1, p.3) and Table 5's ranges fix no zero, so the estimate is "
    "relative to the recorded open hand and differs from the paper's T_TMC_fe "
    "and T_TMC_aa by an unknown constant per coordinate. It is never used as "
    "a validated Paper 3 coordinate: it is written only under *_est_openhand "
    "names, and paper3.compare keeps the Table 6 relation T_TMC_fe ~ 11/10 "
    "T_MCP_fe 'unavailable'. Not measurable with this dataset.")
A4_TEXT = (
    "The tracker's thumb metacarpal is a tracked bone, not a rigid part of its "
    "palm: per recording, the spread of R_palm^T R_metacarpal about its own "
    "chordal mean (median and max over the frames), the angle of that mean to "
    "R_rest, and the largest angle of a single frame to R_rest.")
A5_TEXT = (
    "The palm frame stands for the carpus. The trapezium is not tracked and "
    "LeapC fits a rigid palm, so the TMC is measured against the PALM joint's "
    "frame. How far that frame is from an anatomical trapezium frame is not "
    "measured by this dataset; it is part of A3's unknown constant (and of a "
    "possible scale or cross-talk error a validation would have to measure).")
ORIGIN_TEXT = (
    "rec.timeline.t counts from the first selected frame on the camera clock "
    "(timestamp_us); movement_start_time_s counts from first_frame_capture_time "
    "on the capture clock. first_frame_ms = the first selected frame's "
    "capture_time minus first_frame_capture_time; window_max_ms = the largest "
    "|t - (capture_time - first_frame_capture_time)| over the open-hand window "
    "(origin and clocks together); compared with one frame period.")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _blank(v) -> bool:
    return v is None or str(v).strip() == ""


def _cell(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return fmt_num(float(v), digits)
    return str(v)


def _clean(obj):
    """JSON-safe: numpy scalars to floats, non-finite floats to None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(_clean(data), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, columns, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _dist(values, where=None) -> dict:
    """median, p95, max of an array, and where the max is."""
    v = np.asarray(values, dtype=float)
    out = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
           "max": float(np.max(v))}
    if where is not None:
        out["worst"] = where[int(np.argmax(v))]
    return out


# --- one recording ----------------------------------------------------------------------
def analyse(rec, cal: te.Calibration):
    """(estimates KEYS -> array, A1 axis angles AXIS_KEYS -> array, A4 motion)
    over every selected frame, each frame reconstructed once."""
    n = len(rec.frames)
    est = {k: np.empty(n) for k in te.KEYS}
    axes = {k: np.empty(n) for k in te.AXIS_KEYS}
    pm = np.empty((n, 3, 3))
    for i, frame in enumerate(rec.frames):
        r = rc.reconstruct(frame)
        e = te.estimate(r.pos, r.rot, rec.hand_side, cal)
        a = te.axis_parallelism(r.rot)
        for k in te.KEYS:
            est[k][i] = e[k]
        for k in te.AXIS_KEYS:
            axes[k][i] = a[k]
        pm[i] = te.palm_to_metacarpal(r.rot)
    return est, axes, te.rotation_spread(pm, cal)


def recording_row(rec, est, motion, cal: te.Calibration):
    """One estimates_per_recording.csv row (cells as text), plus the
    hold-frame means (None for a transition or a take without a hold)."""
    row = {c: "" for c in RECORDING_COLUMNS}
    row.update(kind=rec.kind, name=rec.name, take=rec.take,
               **{"class": p3.grasp_class(rec.name)[0]},
               n_frames=len(rec.frames), calibration=cal.provenance)
    for k in (FE, AA, TW):
        for s, f in (("mean", np.mean), ("min", np.min), ("max", np.max)):
            row[f"{k}_all_{s}"] = _cell(float(f(est[k])))
    row[f"{RES}_max"] = _cell(float(np.max(est[RES])))
    row["metacarpal_spread_median_deg"] = _cell(motion["spread_median_deg"])
    row["metacarpal_spread_max_deg"] = _cell(motion["spread_max_deg"])
    row["metacarpal_mean_to_rest_deg"] = _cell(motion["mean_to_rest_deg"])
    row["metacarpal_max_frame_to_rest_deg"] = _cell(motion["max_frame_to_rest_deg"])
    hold_means, note = None, None
    if rec.kind == "static":
        rid = rec.metadata.get("representative_frame_id")
        idx = ([] if _blank(rid) else
               [i for i, f in enumerate(rec.frames) if f.frame_id == int(float(rid))])
        if idx:
            row["rep_frame_id"] = rec.frames[idx[0]].frame_id
            for k in (FE, AA, TW):
                row[f"{k}_rep"] = _cell(float(est[k][idx[0]]))
        else:
            note = (f"representative frame_id {rid!r} is not among the selected "
                    f"frames of {rec.name} take {rec.take}")
        mask = rc.hold_mask(rec)
        if mask is not None and mask.any():
            row["hold_frames"] = int(mask.sum())
            hold_means = {}
            for k in (FE, AA):
                v = est[k][mask]
                hold_means[k] = float(np.mean(v))
                for s, f in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                    row[f"{k}_hold_{s}"] = _cell(float(f(v)))
    return row, hold_means, note


def class_rows(static_means):
    """[(class, {FE: mean, AA: mean})] -> estimates_by_class.csv rows."""
    rows = []
    for cls in p3.CLASSES:
        group = [m for c, m in static_means if c == cls]
        if not group:
            continue
        row = {"class": cls, "static_takes": len(group)}
        for k in (FE, AA):
            v = np.array([m[k] for m in group])
            row[f"{k}_hold_mean_min"] = _cell(float(v.min()), 2)
            row[f"{k}_hold_mean_median"] = _cell(float(np.median(v)), 2)
            row[f"{k}_hold_mean_max"] = _cell(float(v.max()), 2)
        rows.append(row)
    return rows


# --- the run ----------------------------------------------------------------------------
def run(args) -> int:
    root, out = Path(args.dataset), Path(args.out)
    if _is_under(out, root):
        print(f"refusing to write inside the dataset: {out} is inside {root}")
        return 2
    started = time.perf_counter()
    try:
        cal = te.calibrate(root, min_frames=args.min_frames)
    except Exception as e:      # reported, never hidden
        print(f"TMC ESTIMATE REPORT FAIL: calibration failed: {type(e).__name__}: {e}")
        return 1
    statics, transitions = rc.list_static(root), rc.list_transitions(root)
    with_start = sum(not _blank(r.get("movement_start_time_s")) for r in transitions)
    blank = len(transitions) - with_start
    short = sum("window" in s["reason"] for s in cal.skipped)
    print(f"calibration: {cal.n_transitions} transitions used, {cal.n_frames} frames; "
          f"table: {len(transitions)} transitions, {with_start} with a movement start, "
          f"{blank} blank, {short} with a window under {args.min_frames} frames; "
          f"agreement median {cal.agreement['median_deg']:.2f}, p90 "
          f"{cal.agreement['p90_deg']:.2f}, max {cal.agreement['max_deg']:.2f} deg "
          f"({cal.agreement['worst']['name']} take {cal.agreement['worst']['take']})",
          flush=True)

    out.mkdir(parents=True, exist_ok=True)
    keys = ([("static", r["grasp_name"], int(r["take"])) for r in statics]
            + [("transition", r["to_grasp"], int(r["take"])) for r in transitions])
    rec_rows, motions, static_means, failures, notes, origins = [], [], [], [], [], []
    axes_all = {k: [] for k in te.AXIS_KEYS}
    where_all = []
    frames_written = frames_selected = 0
    with open(out / "estimates_per_frame.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_COLUMNS)
        for i, (kind, name, take) in enumerate(keys, 1):
            try:
                rec = (rc.find_static(root, name, take) if kind == "static"
                       else rc.find_transition(root, name, take))
                est, axes, motion = analyse(rec, cal)
            except Exception as e:   # reported and counted, never skipped silently
                failures.append(f"{kind} {name} take {take}: {type(e).__name__}: {e}")
                print(f"LOAD FAILED: {failures[-1]}")
                continue
            frames_selected += len(rec.frames)
            row, hold_means, note = recording_row(rec, est, motion, cal)
            rec_rows.append(row)
            if note:
                notes.append(note)
                print(f"NOTE: {note}")
            if hold_means is not None:
                static_means.append((row["class"], hold_means))
            motions.append({"kind": kind, "name": name, "take": take, **motion})
            for k in te.AXIS_KEYS:
                axes_all[k].append(axes[k])
            where_all += [{"kind": kind, "name": name, "take": take,
                           "frame_id": fr.frame_id} for fr in rec.frames]
            if kind == "transition" and not _blank(rec.metadata.get("movement_start_time_s")):
                o = te.timeline_origin_ms(rec)
                if o is not None:
                    origins.append({"name": name, "take": take, **o,
                                    "nominal_period_ms": rec.timeline.nominal_period_s * 1e3})
            for j, fr in enumerate(rec.frames):
                w.writerow([kind, name, take, j, fr.frame_id,
                            _cell(float(rec.timeline.t[j]), 6)]
                           + [_cell(float(est[k][j])) for k in te.KEYS])
                frames_written += 1
            if i % 20 == 0 or i == len(keys):
                print(f"[{i:>3}/{len(keys)}] {time.perf_counter() - started:6.1f} s  "
                      f"{kind} {name} take {take}", flush=True)

    _write_csv(out / "estimates_per_recording.csv", RECORDING_COLUMNS, rec_rows)
    by_class = class_rows(static_means)
    _write_csv(out / "estimates_by_class.csv", CLASS_COLUMNS, by_class)
    _write_json(out / "calibration.json", cal.to_json())

    axes_cat = {k: np.concatenate(v) if v else np.array([]) for k, v in axes_all.items()}
    n_axes = len(where_all)
    spread_max = [m["spread_max_deg"] for m in motions]
    to_rest = [m["mean_to_rest_deg"] for m in motions]
    frame_to_rest = [m["max_frame_to_rest_deg"] for m in motions]
    labels = [{"kind": m["kind"], "name": m["name"], "take": m["take"]} for m in motions]
    origin_first = [abs(o["first_frame_ms"]) for o in origins]
    origin_window = [o["window_max_ms"] for o in origins if o["window_max_ms"] is not None]
    period_ms = min((o["nominal_period_ms"] for o in origins), default=None)
    spreads = [u["spread_median_deg"] for u in cal.per_transition]
    widest = max(cal.per_transition, key=lambda u: u["spread_max_deg"])
    checks = {
        "note": ("ESTIMATES relative to the recorded open hand, not validated Paper 3 "
                 "coordinates (leap_hand.tmc_estimate)"),
        "calibration": cal.provenance,
        "A1_parallel_hinge_axes": {
            "statement": A1_TEXT, "frames": n_axes, "recordings": len(motions),
            **({k: _dist(axes_cat[k], where_all) for k in te.AXIS_KEYS}
               if n_axes else {})},
        "A2_rest_stable": {
            "statement": A2_TEXT, "transitions": cal.n_transitions,
            "frames": cal.n_frames, "agreement": cal.agreement,
            "spread_median_deg_range": [min(spreads), max(spreads)],
            "spread_max_deg": widest["spread_max_deg"],
            "spread_max_where": {"name": widest["name"], "take": widest["take"]},
            "per_transition": cal.per_transition, "skipped": cal.skipped},
        "A3_open_hand_is_zero": {"statement": A3_TEXT},
        "A4_metacarpal_tracked": {
            "statement": A4_TEXT,
            "spread_max_deg": ({"median": float(np.median(spread_max)),
                                "max": float(np.max(spread_max)),
                                "worst": labels[int(np.argmax(spread_max))]}
                               if motions else None),
            "mean_to_rest_deg": ({"median": float(np.median(to_rest)),
                                  "max": float(np.max(to_rest)),
                                  "worst": labels[int(np.argmax(to_rest))]}
                                 if motions else None),
            "max_frame_to_rest_deg": ({"median": float(np.median(frame_to_rest)),
                                       "max": float(np.max(frame_to_rest)),
                                       "worst": labels[int(np.argmax(frame_to_rest))]}
                                      if motions else None),
            "per_recording": motions},
        "A5_palm_is_carpus": {"statement": A5_TEXT},
        "timeline_origin": {
            "statement": ORIGIN_TEXT, "transitions_checked": len(origins),
            "first_frame_ms_max_abs": max(origin_first) if origin_first else None,
            "window_max_ms": max(origin_window) if origin_window else None,
            "frame_period_ms": period_ms,
            "within_one_frame_period": (bool(origin_first and period_ms is not None
                                             and max(origin_first + origin_window)
                                             < period_ms)),
            "per_transition": origins},
        "representative_frame_notes": notes,
        "run": {"dataset": str(root), "min_frames": args.min_frames,
                "generated": datetime.now().isoformat(timespec="seconds"),
                "wall_time_s": time.perf_counter() - started},
    }
    _write_json(out / "assumption_checks.json", checks)

    print("A1 hinge axes (deg, median / p95 / max): " + "; ".join(
        f"{k} {checks['A1_parallel_hinge_axes'][k]['median']:.2f} / "
        f"{checks['A1_parallel_hinge_axes'][k]['p95']:.2f} / "
        f"{checks['A1_parallel_hinge_axes'][k]['max']:.2f}"
        for k in te.AXIS_KEYS if k in checks["A1_parallel_hinge_axes"]))
    if motions:
        a4 = checks["A4_metacarpal_tracked"]
        print(f"A4 metacarpal within-take spread max: median "
              f"{a4['spread_max_deg']['median']:.1f}, max {a4['spread_max_deg']['max']:.1f} "
              f"deg; take mean to rest: median {a4['mean_to_rest_deg']['median']:.1f}, "
              f"max {a4['mean_to_rest_deg']['max']:.1f} deg")
    for r in by_class:
        print(f"class {r['class']} ({r['static_takes']} static takes), hold-frame means "
              f"min / median / max: fe {r[f'{FE}_hold_mean_min']} / "
              f"{r[f'{FE}_hold_mean_median']} / {r[f'{FE}_hold_mean_max']}, aa "
              f"{r[f'{AA}_hold_mean_min']} / {r[f'{AA}_hold_mean_median']} / "
              f"{r[f'{AA}_hold_mean_max']} deg")

    problems = list(failures)
    if cal.n_transitions + len(cal.skipped) != len(transitions):
        problems.append(f"calibration accounted for {cal.n_transitions} + "
                        f"{len(cal.skipped)} of {len(transitions)} transition rows")
    if cal.n_transitions + short != with_start:
        problems.append(f"{with_start} rows with a movement start, but "
                        f"{cal.n_transitions} used + {short} short windows")
    if len(rec_rows) != len(keys):
        problems.append(f"{len(rec_rows)} of {len(keys)} recordings estimated")
    if frames_written != frames_selected:
        problems.append(f"{frames_written} frame rows for {frames_selected} frames")
    print(f"wrote {out}; {frames_written} frames; wall time "
          f"{time.perf_counter() - started:.1f} s")
    if problems:
        print("TMC ESTIMATE REPORT FAIL: " + "; ".join(problems))
        return 1
    print(f"TMC ESTIMATE REPORT OK: transitions_used={cal.n_transitions} "
          f"recordings={len(rec_rows)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Open-hand thumb TMC estimate (not a validated Paper 3 "
                    "coordinate) over the organized grasp dataset.")
    p.add_argument("--dataset", type=Path, required=True,
                   help="the organized dataset root (read only)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output folder (default {DEFAULT_OUT})")
    p.add_argument("--min-frames", type=int, default=te.MIN_FRAMES,
                   help="shortest open-hand window a transition needs to be used "
                        f"(default {te.MIN_FRAMES})")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
