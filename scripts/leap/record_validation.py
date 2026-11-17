"""Record the tracker side of a reference validation capture.

Setup: camera only, bare hand (the one --hand names), WITH THE REFERENCE
DEVICE RECORDING THE SAME HAND AT THE SAME TIME (docs/validation_capture_protocol.md
says how to set both up). --mock: no hardware, synthetic hands, a rehearsal
of the files only.

One session folder, recordings/protocol/validation/<YYYYMMDD_HHMMSS>_<hand>/
(recordings/protocol_mock/validation/ with --mock): the start taps, then
--trials rounds of every trial item of leap_hand.reference_validation
(TRIAL_ITEMS: the neutral hold, an isolated TMC flexion sweep, an isolated
TMC abduction sweep, three static holds, the palm arc), then the end taps.
Each take is one JSONL in the camera line format, every frame kept, with a
`<take>.meta.json` sidecar (participant, hand, session, item, instruction,
the reference device as given, the commanded start and end on the wall
clock), recorded by scripts/leap/record_poses.py's own Session, so a take
here is the same file a grasp take is. At the end, validation_session.json
lists every take with its sha256 (leap_hand.reference_validation.MANIFEST_FILE)
and the folder is read back through the comparison's own loader.

Ends with
  VALIDATION CAPTURE DONE: participant=<id> hand=<hand> session=<name> takes=<ok>/<n> mock=<yes|no> folder=<path>
(exit 0), or VALIDATION CAPTURE INCOMPLETE: ... (exit 1) when a take failed;
then the next commands (the sync-tap check and the comparison).

Usage:
  .venv\\Scripts\\python.exe scripts\\leap\\record_validation.py --participant P01 --hand left --reference-device "<make model serial, software version>" --operator "<name>"
  .venv\\Scripts\\python.exe scripts\\leap\\record_validation.py --participant REHEARSAL --hand left --reference-device "none (mock rehearsal)" --mock --trials 1 --duration 1 --prep 0
"""
import argparse
import importlib.util
import sys
from pathlib import Path

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from leap_hand import reference_validation as rv
from leap_hand.stream import LeapUnavailable

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("record_poses_for_validation",
                                               HERE / "record_poses.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

OUT = Path("recordings") / "protocol" / "validation"
MOCK_OUT = Path("recordings") / "protocol_mock" / "validation"
PREP_S = 5.0
KIND = "reference_validation_take"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record the tracker side of a reference "
                                "validation capture (docs/validation_capture_protocol.md).")
    p.add_argument("--participant", required=True,
                   help="the participant's id, as the reference capture will name it")
    p.add_argument("--hand", required=True, choices=("left", "right"))
    p.add_argument("--reference-device", required=True, metavar="TEXT",
                   help="the reference device recording at the same time (make, model, "
                        "serial, software and version), kept in every sidecar")
    p.add_argument("--operator", default=None)
    p.add_argument("--trials", type=int, default=rv.TRIALS,
                   help=f"rounds of the trial items (default {rv.TRIALS}; a fit by trial "
                        "needs at least 2: one to calibrate, one held out)")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds per take for every item (a rehearsal); default: each "
                        "item's own duration")
    p.add_argument("--prep", type=float, default=PREP_S,
                   help=f"countdown before each take (default {PREP_S:g} s)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--mock", action="store_true", help="synthetic hands, no camera")
    p.add_argument("--mode", default="desktop", choices=("desktop", "hmd", "screentop"))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    root = args.out_dir if args.out_dir is not None else (MOCK_OUT if args.mock else OUT)
    folder = rp.make_session_dir(root, args.hand)
    seq = rv.capture_sequence(args.trials)
    total = sum((args.duration or i.duration_s) + args.prep for i, _ in seq)
    print("=" * 62)
    print(f"Validation capture ({rv.PROTOCOL}): participant {args.participant}, "
          f"{args.hand} hand, {len(seq)} takes (~{total / 60:.1f} min)")
    print(f"  reference device: {args.reference_device}")
    print(f"  output: {folder}")
    print("  The reference device must be recording now, and keep recording to the "
          "last take.")
    print("=" * 62 + "\n")
    try:
        source = rp.open_stream(mock=args.mock, mode=args.mode)
    except LeapUnavailable as e:
        raise SystemExit(f"\nNo live tracking: {e}\n")
    serial = getattr(source, "device_serial", None)
    device = f"Ultraleap tracker, serial {serial or 'not reported by the stream'}" + (
        " (mock)" if args.mock else "")
    session = rp.Session(source, hz=None, out_dir=folder, calibration={})
    fixed = {"kind": KIND, "protocol": rv.PROTOCOL, "participant": args.participant,
             "hand": args.hand, "session": folder.name, "operator": args.operator,
             "reference_device": args.reference_device, "tracker_device": device,
             "mock": bool(args.mock), "recorder_rules": rp.RECORDER_RULES,
             "tool_commit": rp.tool_commit()}
    try:
        if not args.mock:
            session.wait_for_stream()
        for i, (item, trial) in enumerate(seq, 1):
            n_takes = 1 if item in rv.SYNC_ITEMS else args.trials
            print(f"[{i}/{len(seq)}] {item.id}, trial {trial}: {item.instruction}")
            session.calibration = {**fixed, "item": item.id, "joints": list(item.joints),
                                   "isolated": item.isolated, "instruction": item.instruction}
            session.run_take(item.id, trial, n_takes, i, len(seq),
                             args.duration or item.duration_s, args.prep)
    except KeyboardInterrupt:
        print("\nInterrupted - keeping the takes recorded so far.")
    finally:
        source.stop()
        session.print_summary()
    ok = [r for r in session.results if r["ok"]]
    takes = [{"file": r["file"], "item": r["pose"], "take": r["take"],
              "frames": r["frames"], "meta": r.get("meta")} for r in ok]
    missing = [f"{r['pose']} take {r['take']}" for r in session.results if not r["ok"]]
    missing += [f"{item.id} take {trial}" for item, trial in seq[len(session.results):]]
    rv.write_manifest(folder, participant=args.participant, hand=args.hand,
                      session=folder.name, tracker_device=device, takes=takes,
                      extra={"protocol": rv.PROTOCOL, "trials": args.trials,
                             "operator": args.operator,
                             "reference_device": args.reference_device,
                             "mock": bool(args.mock), "tool_commit": rp.tool_commit(),
                             "recorder_rules": rp.RECORDER_RULES,
                             "recorded": rp.iso_now(), "missing": missing})
    trk = rv.load_tracker_session(folder)       # the comparison's own reader, hashes checked
    head = "VALIDATION CAPTURE DONE" if not missing else "VALIDATION CAPTURE INCOMPLETE"
    print(f"{head}: participant={args.participant} hand={args.hand} session={folder.name} "
          f"takes={len(trk.takes)}/{len(seq)} mock={'yes' if args.mock else 'no'} "
          f"folder={folder}")
    if missing:
        print(f"  missing: {', '.join(missing)}")
    print("  next: check the sync taps, then convert the reference export "
          "(docs/reference_validation.md) and compare:")
    print(f"    .venv\\Scripts\\python.exe scripts\\leap\\reference_validation.py taps "
          f"--session {folder}")
    print(f"    .venv\\Scripts\\python.exe scripts\\leap\\reference_validation.py compare "
          f"--reference <reference folder> --tracker-session {folder} "
          f"--calibration-trials 1 --out results\\reference_validation\\{folder.name}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
