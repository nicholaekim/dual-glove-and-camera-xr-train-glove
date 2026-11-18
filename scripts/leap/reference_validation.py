"""Import, check and compare independent reference measurements of the thumb
TMC pair and the finger CMC (leap_hand.reference_validation).

Setup: no hardware (reads files). The capture itself is
scripts/leap/record_validation.py (camera only, bare hand, with the reference
device recording at the same time); the procedure is
docs/validation_capture_protocol.md, the format docs/reference_validation.md.

Subcommands (every one exits 0 on success, 2 on a refusal, which it prints
with the fix):
  check DIR             read a reference capture folder and check it
                        -> REFERENCE OK: ... | REFERENCE REFUSED: ...
  taps --session DIR    the thumb-index taps detected in the session's sync
                        takes, to pair with the reference device's
                        -> SYNC TAPS OK: ... | SYNC TAPS CHECK FAILED: ... (exit 1)
  compare --reference DIR --tracker-session DIR --calibration-trials 1 --out DIR
          [--tmc-calibration PATH [--tmc-calibration-hand left|right]]
                        the comparison: report.json, report.md,
                        matched_samples.csv and mapping.json in --out
                        -> COMPARISON DONE: ... | COMPARISON REFUSED: ...
                        (the DONE line ends with "sync: <the report's sync
                        note>" when fewer than three sync events left their
                        pairing unchecked)
  observability --dataset ROOT --out DIR
                        over every recording of the dataset: how far each
                        metacarpal moves against the tracker's palm
                        -> OBSERVABILITY: ... then VERDICT: ...
  example --out DIR     the end-to-end example on SYNTHETIC data (a tracker
                        session, a reference capture, the comparison)
                        -> EXAMPLE DONE: data_origin=synthetic ...

Usage:
  .venv\\Scripts\\python.exe scripts\\leap\\reference_validation.py check recordings\\validation\\reference\\P01_left
  .venv\\Scripts\\python.exe scripts\\leap\\reference_validation.py taps --session recordings\\protocol\\validation\\20261101_101500_left
  .venv\\Scripts\\python.exe scripts\\leap\\reference_validation.py compare --reference <ref folder> --tracker-session <session folder> --calibration-trials 1 --out results\\reference_validation\\<session>
  .venv\\Scripts\\python.exe scripts\\leap\\reference_validation.py observability --dataset "C:\\Users\\nkim2\\OneDrive\\Desktop\\xr trainer\\grasp set 2026-10-08 organized" --out results\\reference_validation\\observability
  .venv\\Scripts\\python.exe scripts\\leap\\reference_validation.py example --out results\\reference_validation\\example
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from leap_hand import reference_validation as rv
from leap_hand import replay_core as rc
from leap_hand import tmc_estimate as te


def _f(v, fmt=".2f") -> str:
    return "n/a" if v is None else format(v, fmt)


def cmd_check(args) -> int:
    ref = rv.load_reference(args.folder)
    cols = ", ".join(f"{c.column} ({c.joint} {c.movement}{', sign flipped' if c.sign < 0 else ''})"
                     for c in ref.coordinates)
    print(f"REFERENCE OK: participant={ref.participant} hand={ref.hand} "
          f"session={ref.session} device={ref.device['kind']} "
          f"data_origin={ref.data_origin} role={ref.role} status={ref.status} "
          f"samples={len(ref.samples)} valid={sum(s.valid for s in ref.samples)} "
          f"sync={ref.sync['method']}")
    print(f"  coordinates: {cols}")
    print(f"  movements: {', '.join(f'{k} ({v or 'hold'})' for k, v in ref.movements.items())}")
    print(f"  status: {rv.STATUS_TEXT[ref.status]}")
    return 0


def cmd_taps(args) -> int:
    trk = rv.load_tracker_session(args.session)
    bad = []
    lines = []
    for t in trk.takes:
        if not t.item.startswith("sync"):
            continue
        taps = rv.tap_times(t.rec)
        lines.append(f"  {t.file}: {len(taps)} taps" + "".join(
            f"\n    tap {k}: timestamp_us {x['timestamp_us']} frame_id {x['frame_id']} "
            f"{x['distance_mm']:.1f} mm" for k, x in enumerate(taps, 1)))
        if len(taps) != rv.SYNC_TAPS:
            bad.append(f"{t.file} has {len(taps)} taps, the protocol makes {rv.SYNC_TAPS}")
    if not lines:
        bad.append("the session has no sync take")
    head = (f"SYNC TAPS OK: {len(lines)} sync takes, {rv.SYNC_TAPS} taps each" if not bad
            else "SYNC TAPS CHECK FAILED: " + "; ".join(bad))
    print(head)
    print("\n".join(lines))
    return 0 if not bad else 1


def cmd_compare(args) -> int:
    ref = rv.load_reference(args.reference)
    trk = rv.load_tracker_session(args.tracker_session)
    cal = None
    if args.tmc_calibration is not None:
        doc = json.loads(Path(args.tmc_calibration).read_text(encoding="utf-8"))
        cal = te.Calibration.from_json(doc, hand=args.tmc_calibration_hand)
    trials = [int(x) for x in str(args.calibration_trials).split(",") if x.strip()]
    report = rv.compare(ref, trk, trials, tmc_calibration=cal,
                        tmc_calibration_path=args.tmc_calibration)
    rv.write_report(report, args.out)
    sync = report["sync"]     # events but no residual: the report's note says what was not checked
    unchecked = sync.get("events") and sync.get("residual_max_s") is None
    print(summary_line("COMPARISON DONE", report)
          + (f" sync: {sync['note']}" if unchecked else ""))
    print(f"  {rv.STATUS_TEXT[report['status']]}")
    print(f"  report: {Path(args.out) / 'report.md'}")
    return 0


def summary_line(head: str, report: dict) -> str:
    tmc = report["joints"].get("thumb_tmc") or {}
    mapped = ((tmc.get("heldout") or {}).get("mapped") or {}).get("all") or {}
    mp = tmc.get("mapping") or {}
    cmc = {j: v["verdict"] for j, v in report["joints"].items() if j != "thumb_tmc"}
    est = {k: v["all"] for k, v in (tmc.get("heldout") or {}).items() if k.startswith("estimate_")}
    return (f"{head}: data_origin={report['reference']['data_origin']} "
            f"status={report['status']} "
            f"calibration_samples={len(report['split']['calibration_samples'])} "
            f"heldout_samples={len(report['split']['heldout_samples'])} "
            f"excluded={report['samples']['excluded']} "
            f"mapping={report.get('mapping_id') or ('refused' if 'refused' in mp else 'none')} "
            f"mapped_fe_mae={_f((mapped.get('fe') or {}).get('mae_deg'))} "
            f"mapped_aa_mae={_f((mapped.get('aa') or {}).get('mae_deg'))}"
            + "".join(f" {k}_fe_mae={_f(v['fe'].get('mae_deg'))} "
                      f"{k}_aa_mae={_f(v['aa'].get('mae_deg'))}" for k, v in est.items())
            + "".join(f" {j}={v}" for j, v in cmc.items()))


OBS_COLUMNS = ["kind", "name", "take", "hand_side", "frames", "bone", "max_from_mean_deg",
               "rms_axis1_deg", "rms_axis2_deg"]


def cmd_observability(args) -> int:
    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ([("static", r["grasp_name"], int(r["take"])) for r in rc.list_static(root)]
            + [("transition", r["to_grasp"], int(r["take"])) for r in rc.list_transitions(root)])
    rows, frames, started = [], 0, time.perf_counter()
    for kind, name, take in dict.fromkeys(keys):
        find = rc.find_static if kind == "static" else rc.find_transition
        rec = find(root, name, take)
        frames += len(rec.frames)
        for bone, s in rv.observability(rec).items():
            rows.append({"kind": kind, "name": name, "take": take, "hand_side": rec.hand_side,
                         "frames": len(rec.frames), "bone": bone,
                         "max_from_mean_deg": s["max_from_mean_deg"],
                         "rms_axis1_deg": s["rms_axis1_deg"],
                         "rms_axis2_deg": s["rms_axis2_deg"]})
    with open(out / "observability.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OBS_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    per_bone = {}
    for b in rv.OBSERVED_BONES:
        v = np.array([r["max_from_mean_deg"] for r in rows if r["bone"] == b])
        per_bone[b] = {"recordings": int(v.size), "max_deg": float(v.max()),
                       "median_deg": float(np.median(v)), "min_deg": float(v.min())}
    fingers = [b for b in rv.OBSERVED_BONES if b != "THUMB_METACARPAL"]
    finger_unobs = all(per_bone[b]["max_deg"] < rv.UNOBSERVABLE_DEG for b in fingers)
    thumb_obs = per_bone["THUMB_METACARPAL"]["max_deg"] >= rv.UNOBSERVABLE_DEG
    verdict = {"finger_cmc": "unobservable" if finger_unobs else "moves",
               "thumb_tmc": "observable" if thumb_obs else "does_not_move"}
    doc = {"dataset": str(root), "recordings": len(keys), "frames": frames,
           "measure": ("per recording, for each metacarpal, the largest angle of any "
                       "selected frame's orientation against the PALM joint from the "
                       "recording's chordal mean (leap_hand.reference_validation."
                       "observability)"),
           "unobservable_below_deg": rv.UNOBSERVABLE_DEG,
           "threshold_kind": "chosen for this software (a 'does it move at all' test), "
                             "not an accuracy threshold",
           "per_bone": per_bone, "verdict": verdict,
           "seconds": round(time.perf_counter() - started, 1)}
    (out / "observability.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"OBSERVABILITY: recordings={len(keys)} frames={frames} "
          + " ".join(f"{b.split('_')[0].lower()}_max={per_bone[b]['max_deg']:.6f}deg"
                     for b in rv.OBSERVED_BONES)
          + f" thumb_median={per_bone['THUMB_METACARPAL']['median_deg']:.2f}deg")
    print(f"VERDICT: finger_cmc={verdict['finger_cmc']} thumb_tmc={verdict['thumb_tmc']}")
    print(f"  files: {out / 'observability.csv'}, {out / 'observability.json'}")
    return 0


def cmd_example(args) -> int:
    started = time.perf_counter()
    res = rv.write_example(args.out)
    r = res["report"]
    print(summary_line("EXAMPLE DONE", r))
    print(f"  SYNTHETIC: {rv.STATUS_TEXT[r['status']]}")
    print(f"  folder: {args.out} ({time.perf_counter() - started:.1f} s)")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Independent reference measurements: "
                                "check, sync taps, compare, observability, example.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="read and check a reference capture folder")
    c.add_argument("folder", type=Path)
    t = sub.add_parser("taps", help="thumb-index taps in a tracker session's sync takes")
    t.add_argument("--session", type=Path, required=True)
    m = sub.add_parser("compare", help="compare a reference capture with its tracker session")
    m.add_argument("--reference", type=Path, required=True)
    m.add_argument("--tracker-session", type=Path, required=True)
    m.add_argument("--calibration-trials", required=True,
                   help="comma-separated trial numbers the mapping is fitted on; every "
                        "other trial is held out for the evaluation")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--tmc-calibration", type=Path,
                   help="a tracker calibration.json: its unvalidated estimate is "
                        "evaluated beside the mapped values")
    m.add_argument("--tmc-calibration-hand", choices=rc.SIDES,
                   help="the hand of a calibration file that records none")
    o = sub.add_parser("observability", help="metacarpal motion against the palm, dataset-wide")
    o.add_argument("--dataset", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("example", help="the end-to-end example on synthetic data")
    e.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    run = {"check": cmd_check, "taps": cmd_taps, "compare": cmd_compare,
           "observability": cmd_observability, "example": cmd_example}[args.cmd]
    head = {"check": "REFERENCE REFUSED", "compare": "COMPARISON REFUSED",
            "taps": "SYNC TAPS REFUSED", "observability": "OBSERVABILITY REFUSED",
            "example": "EXAMPLE REFUSED"}[args.cmd]
    try:
        return run(args)
    except (rv.ReferenceError, ValueError, LookupError, OSError) as e:
        print(f"{head}: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
