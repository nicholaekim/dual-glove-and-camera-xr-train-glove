"""Replay one recorded hand frame by frame, on the camera's own clock.

Setup: no hardware (replays a recording).

Opens `leap_hand.replay_viewer.ReplayViewer` on one recording of the grasp
set (or any camera JSONL file): the reconstructed skeleton in camera space or
relative to the wrist or palm, the recorded abs26 positions as an overlay,
per-joint axes, a timeline with the gaps, identity changes and dropped frames,
a HUD with the clock, the raw stamps and the fidelity numbers of the shown
frame, and a separate Paper 3 panel. Hold mode by default: only source
frames are shown, never an interpolated pose unless `i` switches it on.

Controls (keys and buttons): space play/pause   +/- speed (0.1 to 4)
left/right step one frame   p/n previous/next event   r camera/wrist/palm
i hold/interpolate   o overlay   a joint axes   s screenshot   q quit.
The slider scrubs by source frame.

On exit the timing log (--timing-log, default
results/replay_fidelity/timing_<name>_take<take>.json) records the lateness of
every playback tick and the frames skipped (leap_hand.replay_viewer.TIMING_DOC).

Usage:
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --dataset "C:\\Users\\nkim2\\OneDrive\\Desktop\\xr trainer\\grasp set 2026-10-08 organized" --grasp p3_circular_power --take 1
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --dataset "<root>" --transition p3_circular_precision_thumb_2_fingers --take 4
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --file recordings\\some_take.jsonl --hand left --hand-ids "573 574"
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --dataset "<root>" --grasp p2_s03_power_grasp_finger_side --take 1 --mode interpolate --reference wrist --overlay --axes
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --dataset "<root>" --grasp p3_circular_power --take 1 --duration 5
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --dataset "<root>" --grasp p3_circular_power --take 1 --headless --screenshot out.png
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --dataset "<root>" --scenario-screenshots results\\replay_fidelity\\screens

--scenario-screenshots replays four static takes headless (Agg) and writes a
PNG per state: camera view, wrist view, overlay on and axes on at the
metadata's representative frame; the HUD at the writer stall
(p3_circular_precision_thumb_2_fingers take 4, frame_id 151853); interpolation
at the identity change (p2_s03_power_grasp_finger_side take 1, between
frame_id 88967 and 89008) and its timeline; then a 2 s headless playback whose
timing log is written beside them. It ends with "SCREENSHOTS WRITTEN <n> files
to <dir>" (n PNG files).
"""
import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from leap_hand import replay_core as rc

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "replay_fidelity"
SCENARIOS = (("p3_circular_power", 1), ("p3_prismatic_precision", 1),
             ("p3_circular_precision_thumb_2_fingers", 4),
             ("p2_s03_power_grasp_finger_side", 1))
WRITER_STALL = ("p3_circular_precision_thumb_2_fingers", 4, 151853)
ID_CHANGE = ("p2_s03_power_grasp_finger_side", 1, 88967, 89008)
HEADLESS_LOOP_S = 2.0
HEADLESS_LABEL = "headless Agg draw loop"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay one recorded hand on the camera's clock "
                    "(no hardware: replays a recording).")
    p.add_argument("--dataset", type=Path,
                   help="the grasp set root (holds metadata/ and grasps/)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--grasp", help="a static take: grasp_name in "
                                     "metadata/grasps_summary.csv (with --take)")
    src.add_argument("--transition", metavar="TO_GRASP",
                     help="a transition capture: to_grasp in "
                          "metadata/pose_transitions.csv (with --take)")
    src.add_argument("--file", type=Path, help="any camera JSONL file (with --hand)")
    p.add_argument("--take", type=int)
    p.add_argument("--hand", choices=rc.SIDES, help="the hand to replay (--file)")
    p.add_argument("--hand-ids", help='tracker ids of that hand, e.g. "573 574" '
                                      "(--file; default: any id)")
    p.add_argument("--gap-threshold", type=float, default=rc.GAP_THRESHOLD_S,
                   help=f"camera-clock gap that splits segments, s "
                        f"(default {rc.GAP_THRESHOLD_S:g})")
    p.add_argument("--mode", choices=rc.MODES, default="hold",
                   help="hold: source frames only (default); interpolate: "
                        "between frames of one segment")
    p.add_argument("--reference", choices=rc.REFERENCES, default="camera")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--start-frame", type=int, default=0,
                   help="source frame index to open at (default 0)")
    p.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False,
                   help="recorded abs26 markers and residual segments")
    p.add_argument("--axes", action="store_true", help="1 cm joint axes")
    p.add_argument("--duration", type=float, default=0.0,
                   help="play, then close after this many seconds (0: stay open)")
    p.add_argument("--timing-log", type=Path,
                   help="where the timing log goes on exit (default "
                        "results/replay_fidelity/timing_<name>_take<take>.json)")
    p.add_argument("--headless", action="store_true",
                   help="no window (Agg): write --screenshot and exit")
    p.add_argument("--screenshot", type=Path, help="PNG to write (with --headless)")
    p.add_argument("--scenario-screenshots", type=Path, metavar="DIR",
                   help="headless: the four scenario recordings, one PNG per "
                        "state, and a timing log")
    return p


def check_args(args, parser) -> None:
    source = args.grasp or args.transition or args.file
    if args.scenario_screenshots is not None:
        if args.dataset is None:
            parser.error("--scenario-screenshots needs --dataset")
        if source or args.screenshot:
            parser.error("--scenario-screenshots replays its own four recordings: "
                         "leave out --grasp/--transition/--file/--screenshot")
        return
    if not source:
        parser.error("give one of --grasp, --transition or --file "
                     "(or --scenario-screenshots)")
    if (args.grasp or args.transition) and (args.dataset is None or args.take is None):
        parser.error("--grasp and --transition need --dataset and --take")
    if not args.file and (args.hand or args.hand_ids):
        parser.error("--hand and --hand-ids go with --file; for a take the "
                     "metadata row gives them")
    if args.file and args.hand is None:
        parser.error("--file needs --hand left|right")
    if args.headless != (args.screenshot is not None):
        parser.error("--headless and --screenshot go together")
    if not args.speed > 0:
        parser.error("--speed must be positive")
    if not args.duration >= 0:
        parser.error("--duration must be 0 or more seconds")


def load(args) -> rc.Recording:
    if args.grasp:
        return rc.find_static(args.dataset, args.grasp, args.take,
                              gap_threshold_s=args.gap_threshold)
    if args.transition:
        return rc.find_transition(args.dataset, args.transition, args.take,
                                  gap_threshold_s=args.gap_threshold)
    ids = args.hand_ids or "any"
    return rc.load_recording(args.file, args.hand, args.hand_ids,
                             provenance=f"--file {args.file} --hand {args.hand} "
                                        f"--hand-ids {ids} (no metadata row)",
                             gap_threshold_s=args.gap_threshold, kind="file")


def default_timing_path(rec: rc.Recording) -> Path:
    take = f"_take{rec.take}" if rec.take is not None else ""
    return RESULTS / f"timing_{rec.name}{take}.json"


def index_of(rec: rc.Recording, frame_id: int) -> int:
    """The selected-frame index of a frame_id (exactly one, or an error)."""
    hits = [f.index for f in rec.frames if f.frame_id == int(frame_id)]
    if len(hits) != 1:
        raise LookupError(f"{rec.name} take {rec.take}: {len(hits)} selected "
                          f"frames with frame_id {frame_id}")
    return hits[0]


def representative_index(rec: rc.Recording) -> int:
    """The metadata's representative frame as a selected-frame index. The
    table's representative_frame_index counts lines of the file (0-based), so
    it is found by its frame_id and the line number is checked against it."""
    i = index_of(rec, rec.metadata["representative_frame_id"])
    line = int(rec.metadata["representative_frame_index"])
    if rec.frames[i].line_no - 1 != line:
        print(f"  note: representative frame_id {rec.frames[i].frame_id} is on "
              f"line {rec.frames[i].line_no}, the table says line index {line}")
    return i


def headless_play(viewer, seconds: float, start_frame: int = 0) -> dict:
    """Play at speed 1 for `seconds` of wall time, the real perf_counter
    driving tick(). On Agg a tick draws synchronously; the loop then sleeps
    one timer interval, as Tk's timer re-arms after its callback returns."""
    viewer.set_speed(1.0)
    viewer.seek_frame(start_frame)
    viewer.toggle_play()
    end = time.perf_counter() + seconds
    while viewer.playing and time.perf_counter() < end:
        viewer.tick()
        time.sleep(viewer.interval_s)
    if viewer.playing:
        viewer.toggle_play()
    return viewer.timing_summary()


def scenario_screenshots(root: Path, out_dir: Path, gap_threshold: float) -> int:
    """The four scenario takes, one PNG per state, plus the timing log.
    Returns the number of PNG files written."""
    from leap_hand.replay_viewer import ReplayViewer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def shot(viewer, stem, state, timeline=False):
        path = out_dir / f"{stem}_{state}.png"
        (viewer.screenshot_timeline if timeline else viewer.screenshot)(path)
        written.append(path)
        print(f"  {path.name}")

    for name, take in SCENARIOS:
        rec = rc.find_static(root, name, take, gap_threshold_s=gap_threshold)
        stem = f"{name}_take{take}"
        rep = representative_index(rec)
        print(f"{stem}: {len(rec.frames)} frames, clock {rec.timeline.source}, "
              f"{len(rec.events)} events, representative frame {rep} "
              f"(frame_id {rec.frames[rep].frame_id})")
        viewer = ReplayViewer(rec, dpi=100, screenshot_dir=out_dir)
        viewer.seek_frame(rep)
        shot(viewer, stem, "camera")
        viewer.set_reference("wrist")
        shot(viewer, stem, "wrist")
        viewer.set_reference("camera")
        viewer.toggle_overlay()
        shot(viewer, stem, "overlay")
        viewer.toggle_overlay()
        viewer.toggle_axes()
        shot(viewer, stem, "axes")
        viewer.toggle_axes()
        if (name, take) == WRITER_STALL[:2]:
            viewer.seek_frame(index_of(rec, WRITER_STALL[2]))
            shot(viewer, stem, "writer_stall")
        if (name, take) == ID_CHANGE[:2]:
            a, b = index_of(rec, ID_CHANGE[2]), index_of(rec, ID_CHANGE[3])
            if b != a + 1:
                raise LookupError(f"{stem}: frame_id {ID_CHANGE[2]} and "
                                  f"{ID_CHANGE[3]} are not consecutive frames")
            viewer.set_mode("interpolate")
            viewer.seek_time((float(rec.timeline.t[a]) + float(rec.timeline.t[b])) / 2)
            shown = viewer.displayed
            print(f"  interpolation at the gap: {shown.source} ({shown.note})")
            shot(viewer, stem, "interpolate_gap")
            shot(viewer, stem, "timeline", timeline=True)
        viewer.close()

    name, take = SCENARIOS[0]
    rec = rc.find_static(root, name, take, gap_threshold_s=gap_threshold)
    viewer = ReplayViewer(rec, dpi=100, timing_label=HEADLESS_LABEL)
    s = headless_play(viewer, HEADLESS_LOOP_S)
    log = viewer.write_timing_log(out_dir / "timing_headless_agg.json")
    viewer.close()
    print(f"timing log ({HEADLESS_LABEL}, {name} take {take}, "
          f"{HEADLESS_LOOP_S:g} s at speed 1): {timing_line(s)} -> {log}")
    return len(written)


def timing_line(s: dict) -> str:
    if not s["n"]:
        return "no ticks (never played)"
    return (f"{s['n']} ticks, lateness mean {s['mean_ms']:.1f} ms, p95 "
            f"{s['p95_ms']:.1f} ms, max {s['max_ms']:.1f} ms, frames skipped "
            f"{s['frames_skipped']}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    check_args(args, parser)
    if args.headless or args.scenario_screenshots is not None:
        import matplotlib
        matplotlib.use("Agg")            # before the viewer imports pyplot

    if args.scenario_screenshots is not None:
        n = scenario_screenshots(args.dataset, args.scenario_screenshots,
                                 args.gap_threshold)
        print(f"SCREENSHOTS WRITTEN {n} files to {args.scenario_screenshots}")
        return 0

    from leap_hand.replay_viewer import ReplayViewer
    try:
        rec = load(args)
    except (LookupError, ValueError, OSError) as e:
        print(f"cannot load the recording: {e}")
        return 1
    viewer = ReplayViewer(rec, mode=args.mode, reference=args.reference,
                          speed=args.speed, start_frame=args.start_frame,
                          overlay=args.overlay, axes=args.axes,
                          screenshot_dir=RESULTS / "screens")
    if args.headless:
        path = viewer.screenshot(args.screenshot)
        print(f"screenshot: {path}")
        if args.timing_log:
            print(f"timing log: {viewer.write_timing_log(args.timing_log)}")
        viewer.close()
        return 0

    tl = rec.timeline
    print(f"Replay: {rec.name} take {rec.take} ({rec.kind}), {rec.hand_side} hand, "
          f"{len(rec.frames)} frames, {len(rec.segments)} segments, "
          f"{len(rec.events)} events")
    print(f"  clock: {tl.source}{' (FALLBACK)' if tl.fallback else ''}; "
          f"{tl.t[-1]:.3f} s")
    print(f"  provenance: {rec.provenance}")
    print("  keys: space play/pause, +/- speed, left/right step, p/n event, "
          "r reference, i interpolate, o overlay, a axes, s screenshot, q quit")
    viewer.show(autoplay=args.duration > 0, duration=args.duration)
    path = viewer.write_timing_log(args.timing_log or default_timing_path(rec))
    print(f"timing log: {path} ({timing_line(viewer.timing_summary())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
