r"""Measure the real viewer's rendering rate: repeated launches of
scripts/leap/replay_view.py at 1x on one recording, each on the real TkAgg
window for --duration seconds, read back from the viewer's own timing logs.

Setup: no hardware (replays a recording); needs the display (a real window
opens and closes for every trial). Run it on an idle machine: no test suite
or batch run at the same time, or the numbers measure the competition.

What is measured (leap_hand.replay_viewer.TIMING_DOC): for the first play run
of each trial, the timer ticks, the rendered updates (blits that reached the
screen), the distinct source frames shown, the source frames skipped, the
wall duration from the start of play to the last tick, and the scheduling
error (lateness mean / p95 / max). The gated number is the distinct frames
shown per second: a redraw of the same pose is a rendered update but not a
new frame on screen. The media clock is wall time and is never slowed to
make the rate: skipped frames are reported, not hidden.

Prints one line per trial and ends with
  PLAYBACK OK: trials=<n> min_frames_shown_per_s=<x> min_rendered_per_s=<y> threshold=<t>
or PLAYBACK FAIL: ... (exit 1) when a trial's first run shows fewer than
--min-frames-per-s distinct frames per second, or has no play run. Writes
OUT/trial<k>.json (the viewer's timing logs) and OUT/summary.json.

  .venv\Scripts\python.exe scripts\leap\replay_playback_check.py --dataset "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer\xr trainer\grasp set 2026-10-08 organized" --grasp p3_circular_power --take 1 --trials 3 --duration 6 --min-frames-per-s 30 --out results\replay_fidelity\codex_corrections_20261024\timing\after
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIEW = REPO / "scripts" / "leap" / "replay_view.py"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rendering rate of the real replay window over repeated 1x "
                    "playback trials (no hardware; a window opens per trial).")
    p.add_argument("--dataset", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--grasp", help="a static take (grasp_name in grasps_summary.csv)")
    src.add_argument("--transition", metavar="TO_GRASP")
    p.add_argument("--take", type=int, required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--duration", type=float, default=6.0,
                   help="seconds each window plays before it closes (default 6)")
    p.add_argument("--min-frames-per-s", type=float, default=30.0,
                   help="distinct source frames shown per second each trial's first "
                        "play run must reach (default 30)")
    p.add_argument("--out", type=Path, required=True, help="folder for the logs")
    return p


def trial_row(k: int, log: dict) -> dict:
    runs = log.get("runs_detail") or []
    run = runs[0] if runs else None
    return {
        "trial": k, "ticks": log.get("n"), "lateness_mean_ms": log.get("mean_ms"),
        "lateness_p95_ms": log.get("p95_ms"), "lateness_max_ms": log.get("max_ms"),
        "backend": log.get("backend"), "timer_interval_ms": log.get("timer_interval_ms"),
        "first_run": run,
    }


def fmt(v, spec=".1f") -> str:
    return "n/a" if v is None else format(v, spec)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials < 1 or args.duration <= 0:
        print("PLAYBACK FAIL: --trials must be at least 1 and --duration positive")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    what = (["--grasp", args.grasp] if args.grasp else ["--transition", args.transition])
    rows, failures = [], []
    for k in range(1, args.trials + 1):
        log_path = out / f"trial{k}.json"
        cmd = [sys.executable, "-B", str(VIEW), "--dataset", str(args.dataset), *what,
               "--take", str(args.take), "--speed", "1", "--duration", str(args.duration),
               "--timing-log", str(log_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        if proc.returncode != 0 or not log_path.is_file():
            failures.append(f"trial {k}: replay_view.py exit {proc.returncode}: "
                            f"{proc.stderr.strip()[-400:]}")
            rows.append({"trial": k, "error": proc.returncode})
            print(f"trial {k}: FAILED to run (exit {proc.returncode})")
            continue
        log = json.loads(log_path.read_text(encoding="utf-8"))
        row = trial_row(k, log)
        rows.append(row)
        run = row["first_run"]
        if run is None:
            failures.append(f"trial {k}: no play run in the log")
            print(f"trial {k}: no play run")
            continue
        print(f"trial {k}: {log['backend']} timer {log['timer_interval_ms']} ms; first run: "
              f"{run['ticks']} ticks, {run['rendered_updates']} rendered updates, "
              f"{run['frames_shown']} frames shown, {run['frames_skipped']} skipped, "
              f"{run['media_end_s'] - run['media_start_s']:.3f} s of media in "
              f"{run['wall_duration_s']:.3f} s wall; {fmt(run['rendered_per_s'])} "
              f"updates/s, {fmt(run['frames_shown_per_s'])} frames/s; lateness mean "
              f"{fmt(log['mean_ms'])} p95 {fmt(log['p95_ms'])} max {fmt(log['max_ms'])} ms")
        if run["frames_shown_per_s"] is None or run["frames_shown_per_s"] < args.min_frames_per_s:
            failures.append(f"trial {k}: {fmt(run['frames_shown_per_s'])} frames shown per "
                            f"second < {args.min_frames_per_s:g}")
    good = [r["first_run"] for r in rows if r.get("first_run")]
    min_shown = min((r["frames_shown_per_s"] for r in good if r["frames_shown_per_s"]),
                    default=None)
    min_rendered = min((r["rendered_per_s"] for r in good if r["rendered_per_s"]),
                       default=None)
    verdict = ("PLAYBACK OK" if not failures and len(good) == args.trials else "PLAYBACK FAIL")
    line = (f"{verdict}: trials={args.trials} min_frames_shown_per_s={fmt(min_shown)} "
            f"min_rendered_per_s={fmt(min_rendered)} threshold={args.min_frames_per_s:g}")
    (out / "summary.json").write_text(json.dumps({
        "written": datetime.now().isoformat(timespec="seconds"),
        "machine": platform.node(), "python": sys.version.split()[0],
        "recording": {"dataset": str(args.dataset), "grasp": args.grasp,
                      "transition": args.transition, "take": args.take},
        "duration_s": args.duration, "threshold_frames_shown_per_s": args.min_frames_per_s,
        "trials": rows, "failures": failures, "verdict": line,
        "note": "first play run of each trial; the gated number is distinct source frames "
                "shown per second of wall time; the media clock is wall time x speed and "
                "is not slowed; skipped frames are reported (TIMING_DOC in the logs)",
    }, indent=2), encoding="utf-8")
    for f in failures:
        print(f"  {f}")
    print(line)
    return 0 if verdict == "PLAYBACK OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
