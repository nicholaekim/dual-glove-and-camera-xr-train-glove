"""Replay one recorded hand frame by frame, on the camera's own clock.

Setup: no hardware (replays a recording).

Opens `leap_hand.replay_viewer.ReplayViewer` on one recording of the grasp
set (or any camera JSONL file): the reconstructed skeleton in camera space or
relative to the wrist or palm, the recorded abs26 positions as an overlay,
per-joint axes, a timeline with the gaps, identity changes and dropped frames,
a HUD with the clock, the raw stamps and the fidelity numbers of the shown
frame, an angle panel, and a separate Paper 3 panel. Hold mode by default:
only source frames are shown, never an interpolated pose unless `i` switches
it on.

The angle panel lists every angle of the shown pose that the coverage table
names (leap_hand.angle_coverage), each with its column and status, with the
finger MCP pairs labelled by order. Below it, apart, the "estimated" block:
the thumb TMC pair relative to the calibration's reference, named by the
calibration's kind (T_TMC_fe_est_openhand / T_TMC_aa_est_openhand for the
session's recorded open hand, T_TMC_fe_est_neutral / T_TMC_aa_est_neutral for
a tmc_neutral calibration take; leap_hand.tmc_estimate,
leap_hand.tmc_calibration_take), its twist and residual, and the
calibration's provenance; the block's title says which kind. These are
tracker-relative estimates, not validated Paper 3 coordinates
(docs/tmc_estimate.md). The calibration is --tmc-calibration PATH (a
calibration.json of either writer: scripts/leap/tmc_estimate_report.py, or
record_poses.py --calibration tmc_neutral; a file without a "kind" is read
by its provenance's first words or refused with the fix spelled out); with
--dataset and no path, results/replay_fidelity/tmc_estimate/calibration.json
when it exists, else one computed from the dataset's transitions (the time it
took is printed). --no-tmc leaves the block "not computed"; so does --file
without --tmc-calibration. A calibration of another session or another hand
than the recording's is not applied (the block says why), and neither is one
whose hand is not known: a file written before the hand field existed is
applied only with --tmc-calibration-hand left|right (refused when the file
records another hand) or when --dataset's metadata/pose_transitions.csv
names its transitions and their hand (the console says which); never is the
hand taken from the recording or from a file name.

Above the estimate, the two "mapped (...)" rows: with --tmc-mapping PATH (a
mapping.json from scripts/leap/reference_validation.py compare, i.e. a
mapping fitted and held-out-tested against an independent reference device;
leap_hand.reference_validation), T_TMC_fe_mapped / T_TMC_aa_mapped of the
shown pose, but only when the recording is of the mapping's participant, hand
and tracker session (the dataset's metadata, or a loose file's .meta.json
sidecar from record_validation.py) and inside its tested range; else the
rows say why it is empty. The first row's tag says what the reference was:
reference-evaluated, SYNTHETIC (not measured), or a cross-check only. Without
--tmc-mapping there are no such rows.

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
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --dataset "<root>" --grasp p3_circular_power --take 1 --no-tmc
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --file recordings\\some_take.jsonl --hand left --tmc-calibration results\\replay_fidelity\\tmc_estimate\\calibration.json
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --dataset "<root>" --scenario-screenshots results\\replay_fidelity\\screens
  .venv\\Scripts\\python.exe scripts\\leap\\replay_view.py --file <validation take>.jsonl --hand left --tmc-mapping <comparison folder>\\mapping.json

--scenario-screenshots replays four static takes headless (Agg) and writes a
PNG per state: camera view, wrist view, overlay on and axes on at the
metadata's representative frame; the HUD at the writer stall
(p3_circular_precision_thumb_2_fingers take 4, frame_id 151853); interpolation
at the identity change (p2_s03_power_grasp_finger_side take 1, between
frame_id 88967 and 89008) and its timeline; then a 2 s headless playback whose
timing log is written beside them, every one with the same estimated block
as the window. It ends with "SCREENSHOTS WRITTEN <n> files to <dir>" (n PNG
files).
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from leap_hand import angle_coverage as ac
from leap_hand import reference_validation as rv
from leap_hand import replay_core as rc
from leap_hand import tmc_estimate as te

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "replay_fidelity"
DEFAULT_TMC = RESULTS / "tmc_estimate" / "calibration.json"
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
    tmc = p.add_mutually_exclusive_group()
    tmc.add_argument("--tmc-calibration", type=Path, metavar="PATH",
                     help="calibration.json for the estimated thumb TMC block: of "
                          "tmc_estimate_report.py (open hand, *_est_openhand) or of "
                          "record_poses.py --calibration tmc_neutral (neutral posture "
                          "take, *_est_neutral) (default with --dataset: "
                          "results/replay_fidelity/tmc_estimate/calibration.json "
                          "if it exists, else computed from the dataset)")
    tmc.add_argument("--no-tmc", action="store_true",
                     help="no thumb TMC estimate (the block says not computed)")
    p.add_argument("--tmc-mapping", type=Path, metavar="PATH",
                   help="a mapping.json of scripts/leap/reference_validation.py compare: "
                        "the reference-mapped thumb TMC pair, shown only inside the "
                        "mapping's participant, hand, session and tested range")
    p.add_argument("--tmc-calibration-hand", choices=rc.SIDES, metavar="HAND",
                   help="the hand (left|right) of a calibration file written before the "
                        "hand field existed; refused when the file records another hand. "
                        "Without it such a file is applied only when --dataset's "
                        "metadata/pose_transitions.csv names its transitions and their "
                        "hand, else the estimate is not applied")
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


def load_calibration(path, hand=None) -> te.Calibration:
    """A Calibration from a calibration.json (`Calibration.to_json`); `hand`
    declares the hand of a file that records none (refused when it
    contradicts a recorded hand)."""
    return te.Calibration.from_json(json.loads(Path(path).read_text(encoding="utf-8")),
                                    hand=hand)


def resolve_hand(cal: te.Calibration, declared, dataset):
    """(the calibration with its hand known where it can be, one phrase saying
    how). The file's own hand first; else the user's declaration (already
    applied by load_calibration); else, with a dataset, the hand of the
    transitions the calibration names in metadata/pose_transitions.csv
    (tmc_estimate.hand_from_transitions); else unknown, and the estimate is
    not applied (tmc_estimate.mismatch says so in the block)."""
    if cal.hand is not None:
        how = ("declared with --tmc-calibration-hand" if declared else "recorded in the file")
        return cal, f"hand {cal.hand} ({how})"
    why = "no --dataset to recover it from"
    if dataset is not None:
        hand, why = te.hand_from_transitions(cal, dataset)
        if hand is not None:
            return replace(cal, hand=hand), f"hand {hand} ({why})"
    return cal, (f"hand not recorded in the file ({why}): the estimate is not applied; "
                 f"declare it with --tmc-calibration-hand left|right or regenerate the file")


def tmc_calibration(args):
    """(Calibration or None, one line saying where it came from, how its hand
    is known, or why there is none) for the estimated block (module
    docstring). A --tmc-calibration that cannot be read, or whose recorded
    hand contradicts --tmc-calibration-hand, raises."""
    if args.no_tmc:
        return None, "TMC estimate: off (--no-tmc)"
    declared = args.tmc_calibration_hand
    if args.tmc_calibration is not None:
        cal, how = resolve_hand(load_calibration(args.tmc_calibration, declared), declared,
                                args.dataset)
        return cal, f"TMC calibration: {args.tmc_calibration}; {how}"
    if args.dataset is None:
        return None, "TMC estimate: not computed (no --dataset, no --tmc-calibration)"
    if DEFAULT_TMC.is_file():
        try:
            cal, how = resolve_hand(load_calibration(DEFAULT_TMC, declared), declared,
                                    args.dataset)
            return cal, f"TMC calibration: {DEFAULT_TMC}; {how}"
        except (OSError, ValueError, KeyError, TypeError) as e:   # reported; replay opens
            return None, (f"TMC estimate: not computed (cannot read {DEFAULT_TMC}: "
                          f"{type(e).__name__}: {e})")
    started = time.perf_counter()
    try:
        cal = te.calibrate(args.dataset)
    except (ValueError, LookupError, OSError) as e:   # reported; the replay still opens
        return None, (f"TMC estimate: not computed (calibration failed: "
                      f"{type(e).__name__}: {e})")
    return cal, (f"TMC calibration: computed from the dataset's transitions in "
                 f"{time.perf_counter() - started:.1f} s (no {DEFAULT_TMC})")


def provenance_text(cal: te.Calibration) -> str:
    """One panel row (80 characters on this session): the block's title
    already says the calibration kind and that it is not a validated Paper 3
    coordinate; the full provenance (frames, session) is printed when the
    calibration loads. The 804 px window holds about 57 rows: the worst case
    (interpolated pose, an estimate and a mapping) fills them."""
    a = cal.agreement
    what, unit = (("open hand", "transitions") if cal.kind == "openhand"
                  else ("neutral posture take", "takes"))
    return (f"{what}, {cal.n_transitions} {unit}; agreement median "
            f"{a['median_deg']:.1f}/p90 {a['p90_deg']:.1f}/max {a['max_deg']:.1f} deg")


def tmc_provider(cal: te.Calibration, rec: rc.Recording):
    """The viewer's tmc_provider: tmc_estimate.estimate on the displayed pose
    (its outputs named by the calibration's kind), with provenance_text and
    the kind. A calibration of another hand, or of another session than the
    recording's (both named), is not applied: that kind's values None, the
    reason as the provenance (tmc_estimate.mismatch)."""
    reason = te.mismatch(cal, rec.hand_side, rec.metadata.get("session"))
    columns = ac.ESTIMATE_COLUMNS_BY_KIND[cal.kind]
    if reason:
        return lambda pos, rot, hand_side: {**dict.fromkeys(columns), "kind": cal.kind,
                                            "provenance": f"{reason}: not applied"}
    text = provenance_text(cal)

    def provider(pos, rot, hand_side):
        return {**te.estimate(pos, rot, hand_side, cal), "provenance": text}
    return provider


def mapping_provider(mapping: rv.TmcMapping, rec: rc.Recording):
    """(the viewer's mapping_provider, one console line). Outside the mapping's
    scope (another hand, session or participant, or a recording that names
    none) every value is None and the reason is the note
    (reference_validation.scope_mismatch); inside it, apply_mapping on the
    shown pose, which leaves values outside the tested range empty too."""
    scope = rv.recording_scope(rec)
    reason = rv.scope_mismatch(mapping, rec.hand_side, scope["session"],
                               scope["participant"])
    base = {"mapping_status": mapping.status, "scope": mapping.scope_text(),
            "scope_row": mapping.scope_row()}
    if reason:
        empty = {**dict.fromkeys(ac.MAPPED_COLUMNS), "status": "not_applied",
                 "note": reason, **base}
        return (lambda pos, rot, hand_side: empty), f"not applied to this recording: {reason}"

    def provider(pos, rot, hand_side):
        return {**rv.apply_mapping(mapping, rot), **base}
    return provider, "applied to this recording (inside its tested range only)"


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


def scenario_screenshots(root: Path, out_dir: Path, gap_threshold: float,
                         cal=None) -> int:
    """The four scenario takes, one PNG per state, plus the timing log, the
    estimated block from `cal` (None: not computed). Returns the number of
    PNG files written."""
    from leap_hand.replay_viewer import ReplayViewer

    provider = (lambda rec: None) if cal is None else (lambda rec: tmc_provider(cal, rec))
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
        viewer = ReplayViewer(rec, dpi=100, screenshot_dir=out_dir,
                              tmc_provider=provider(rec))
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
    viewer = ReplayViewer(rec, dpi=100, timing_label=HEADLESS_LABEL,
                          tmc_provider=provider(rec))
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

    rec = None
    if args.scenario_screenshots is None:
        try:
            rec = load(args)
        except (LookupError, ValueError, OSError) as e:
            print(f"cannot load the recording: {e}")
            return 1
    try:
        cal, cal_line = tmc_calibration(args)
    except (OSError, ValueError, KeyError, TypeError) as e:   # the --tmc-calibration file
        print(f"cannot read the TMC calibration {args.tmc_calibration}: "
              f"{type(e).__name__}: {e}")
        return 1
    print(cal_line)
    mapping = None
    if args.tmc_mapping is not None:
        try:
            mapping = rv.load_mapping(args.tmc_mapping)
        except rv.ReferenceError as e:
            print(f"cannot read the TMC mapping: {e}")
            return 1
        print(f"TMC mapping: {args.tmc_mapping}; {mapping.id}; "
              f"{rv.STATUS_TEXT[mapping.status]}")
        print(f"  scope: {mapping.scope_text()}")
    if cal is not None:
        print(f"  kind: {cal.kind} ({te.KIND_LABELS[cal.kind]}); hand "
              f"{cal.hand or 'not recorded in the file'}; outputs "
              f"{', '.join(ac.ESTIMATE_COLUMNS_BY_KIND[cal.kind])}")
        print(f"  {cal.provenance}")

    if args.scenario_screenshots is not None:
        n = scenario_screenshots(args.dataset, args.scenario_screenshots,
                                 args.gap_threshold, cal)
        print(f"SCREENSHOTS WRITTEN {n} files to {args.scenario_screenshots}")
        return 0

    from leap_hand.replay_viewer import ReplayViewer
    mprov = None
    if mapping is not None:
        mprov, how = mapping_provider(mapping, rec)
        print(f"  {how}")
    viewer = ReplayViewer(rec, mode=args.mode, reference=args.reference,
                          speed=args.speed, start_frame=args.start_frame,
                          overlay=args.overlay, axes=args.axes,
                          screenshot_dir=RESULTS / "screens",
                          tmc_provider=None if cal is None else tmc_provider(cal, rec),
                          mapping_provider=mprov)
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
