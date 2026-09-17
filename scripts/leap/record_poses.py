"""Guided pose-recording session with the Ultraleap camera - hands-free.

The camera twin of `scripts/glove/record_poses.py`: same pose list, same beep
protocol, same file naming, same labels. Announces the pose, counts down with
beeps, records a few seconds, moves on. Nothing to type once it starts, which
is the point - your hands are in front of the camera, and on Path A they are
also inside the glove.

Takes land in `recordings/leap/poses/` as labeled JSONL, named like
`fist_both_take2_20260916_154212.jsonl` - the glove's scheme, so the two
datasets line up folder for folder and every existing tool reads both:

  python scripts/glove/playback.py recordings/leap/poses/<file>.jsonl
  python scripts/glove/export_keypoints21.py recordings/leap/poses
  python scripts/glove/export_prof_format.py recordings/leap/poses

With --raw, each take also writes `<same name>.lmt`, LeapC's own recording of
the raw tracking stream. It is the only artefact that can be re-processed if
the joint mapping or the units later turn out to be wrong, which is worth the
disk space on the first real session.

Protocol (plan section 6): module flat on the table, lenses up, hand 20 to 50
cm above it, palm roughly facing the camera for the spread and thumb poses,
no sunlight and no other IR sources.

  python scripts/leap/record_poses.py                        # 6 poses x 3 takes x 5 s
  python scripts/leap/record_poses.py --poses pinch,fist --takes 2 --duration 4
  python scripts/leap/record_poses.py --hz 0                 # keep every frame (~90/s)
  python scripts/leap/record_poses.py --raw                  # also write .lmt
  python scripts/leap/record_poses.py --mock --takes 1 --duration 2 --prep 1

Ctrl+C at any point keeps the takes recorded so far.
"""
import argparse
import time
from contextlib import nullcontext
from pathlib import Path

from leap_hand.recorder import LeapRecorder
from leap_hand.stream import LeapUnavailable, open_stream
from xr_hand.recorder import finalize_pose_name, hand_tag, pose_filename, slugify

# The same list the glove records, so the two datasets are comparable pose for
# pose. `three` is in POSE_HINTS but not the default set, exactly as in the
# glove script.
DEFAULT_POSES = ["open_palm", "fist", "index_point", "thumbs_up", "peace", "pinch"]

POSE_HINTS = {
    "open_palm": "all five fingers extended and spread",
    "fist": "all fingers curled into a tight fist",
    "index_point": "index finger extended, all others curled",
    "thumbs_up": "thumb extended up, all four fingers curled",
    "peace": "index + middle extended in a V, others curled",
    "pinch": "thumb and index fingertips touching, others relaxed",
    "three": "index + middle + ring extended, little and thumb curled",
}

STREAM_WAIT_TIMEOUT = 120.0   # s to wait for the first hands
STREAM_WAIT_HANDS = 10        # hands seen before the first take starts
MIN_VISIBLE_TIME_US = 300_000  # plan section 6: a hand counts after 0.3 s


def beep(freq: int = 880, ms: int = 180) -> None:
    """Audible cue; falls back to the terminal bell off Windows."""
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


class Session:
    def __init__(self, source, hz, out_dir: Path, raw: bool = False):
        self.source = source
        self.hz = hz
        self.out_dir = out_dir
        self.raw = raw
        self.results = []
        self._ids = {}            # hand_side -> last hand_id, for re-acquisitions
        self._reacquired = 0
        self._skipped_young = 0

    # --- stream plumbing ------------------------------------------------
    def _consume(self, recorder=None) -> None:
        """Drain pending hands: count everything, record if asked.

        Runs during the countdowns too (recorder=None) so the queue stays
        fresh and a stale hand from the previous pose never leaks into a take.
        """
        for _side, lh in self.source.drain(64):
            previous = self._ids.get(lh.hand_side)
            if previous is not None and previous != lh.hand_id:
                self._reacquired += 1
            self._ids[lh.hand_side] = lh.hand_id
            # Gate on presence and settling time, never on confidence: LeapC
            # documents confidence as a constant 1.0 (plan section 6).
            if lh.visible_time_us < MIN_VISIBLE_TIME_US:
                self._skipped_young += 1
                continue
            if recorder is not None:
                recorder.record(lh)

    def wait_for_stream(self) -> None:
        print(f"Waiting for hands (need {STREAM_WAIT_HANDS}, timeout "
              f"{int(STREAM_WAIT_TIMEOUT)} s)...")
        print("  Hold a hand 20 to 50 cm above the module, lenses up.")
        t0 = time.time()
        seen = 0
        sides = set()
        while time.time() - t0 < STREAM_WAIT_TIMEOUT:
            for side, _lh in self.source.drain(64):
                seen += 1
                sides.add(side)
            if seen >= STREAM_WAIT_HANDS:
                print(f"  OK - tracking: {', '.join(sorted(sides))}\n")
                return
            time.sleep(0.05)
        raise SystemExit(
            "No hands seen. Is the camera plugged into a direct USB port and "
            "listed in the Ultraleap Control Panel?\n"
            "  check with: python scripts/leap/check_setup.py"
        )

    # --- protocol -------------------------------------------------------
    def run_take(self, pose: str, take: int, n_takes: int,
                 pose_idx: int, n_poses: int, duration: float, prep: float) -> None:
        title = pose.replace("_", " ").upper()
        print(f"--- Pose {pose_idx}/{n_poses}: {title}  (take {take}/{n_takes}) ---")
        hint = POSE_HINTS.get(pose)
        if hint:
            print(f"    Hold: {hint}")
        # The mock can act out the pose, so a dry run looks like a real one.
        if hasattr(self.source, "set_pose"):
            self.source.set_pose(pose if pose in ("open_palm", "fist") else None)

        for s in range(int(round(prep)), 0, -1):
            if s <= 3:
                print(f"      {s}...")
                beep(660, 120)
            t_end = time.time() + 1.0
            while time.time() < t_end:
                self._consume()
                time.sleep(0.02)

        recorder = LeapRecorder(hz=self.hz, pose=pose, take=take)
        path = self.out_dir / pose_filename(pose, take)
        raw_path = path.with_suffix(".lmt")
        before = self._reacquired

        with self._raw_capture(raw_path):
            recorder.start(path)
            beep(1000, 250)
            print(f"      REC {duration:g} s - hold it ", end="", flush=True)
            try:
                t_end = time.time() + duration
                next_dot = time.time() + 0.5
                while time.time() < t_end:
                    self._consume(recorder)
                    if time.time() >= next_dot:
                        print(".", end="", flush=True)
                        next_dot += 0.5
                    time.sleep(0.005)
            finally:
                # Runs on Ctrl+C too: close the file, then either finalize the
                # take (partial data is still labeled data) or drop it if empty.
                print(flush=True)
                recorder.stop()
                beep(500, 300)
                entry = {"pose": pose, "take": take, "frames": recorder.count,
                         "hands": hand_tag(recorder.hands_seen),
                         "ok": recorder.count > 0,
                         "reacquired": self._reacquired - before}
                if recorder.count == 0:
                    path.unlink(missing_ok=True)
                    raw_path.unlink(missing_ok=True)
                    print("      FAILED: no frames captured (did tracking stop?)\n")
                else:
                    final = finalize_pose_name(path, recorder.hands_seen)
                    entry["file"] = final.name
                    note = (f", {entry['reacquired']} re-acquisition(s)"
                            if entry["reacquired"] else "")
                    print(f"      saved {recorder.count} frames "
                          f"({entry['hands']}{note}) -> {final.name}\n")
                self.results.append(entry)

    def _raw_capture(self, path: Path):
        """LeapC's own .lmt recorder for this take, or nothing."""
        if not self.raw:
            return nullcontext()
        from leap_hand.replay import RawRecording
        return RawRecording(self.source, path)

    def print_summary(self) -> None:
        if not self.results:
            print("\nNothing recorded.")
            return
        ok = [r for r in self.results if r["ok"]]
        print("=" * 62)
        print(f"Session summary: {len(ok)}/{len(self.results)} takes captured")
        by_pose = {}
        for r in self.results:
            by_pose.setdefault(r["pose"], []).append(r)
        for pose, takes in by_pose.items():
            parts = []
            for r in takes:
                if r["ok"]:
                    note = f"take{r['take']}: {r['frames']}f/{r['hands']}"
                    if r["reacquired"]:
                        note += f" ({r['reacquired']} reacq)"
                else:
                    note = f"take{r['take']}: FAILED"
                parts.append(note)
            print(f"  {pose:<12} " + "   ".join(parts))
        if self._skipped_young:
            print(f"\n  {self._skipped_young} hands skipped: tracked for less "
                  f"than {MIN_VISIBLE_TIME_US / 1000:.0f} ms (settling)")
        if ok:
            print(f"\nFiles in {self.out_dir}")
            print(f"  stats:     python scripts/leap/stats.py {self.out_dir}")
            print("  playback:  python scripts/glove/playback.py <file>")
            print(f"  21 points: python scripts/glove/export_keypoints21.py {self.out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Guided, hands-free pose recording with the Ultraleap camera.")
    p.add_argument("--poses", default=",".join(DEFAULT_POSES),
                   help=f"comma-separated pose names (default: {','.join(DEFAULT_POSES)})")
    p.add_argument("--takes", type=int, default=3, help="repetitions per pose (default: 3)")
    p.add_argument("--duration", type=float, default=5.0,
                   help="seconds recorded per take (default: 5)")
    p.add_argument("--prep", type=float, default=5.0,
                   help="seconds to get into the pose before each take (default: 5)")
    p.add_argument("--hz", type=float, default=5.0,
                   help="frames saved per second (default: 5; 0 = keep every frame)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("recordings") / "leap" / "poses")
    p.add_argument("--mock", action="store_true",
                   help="dry-run with synthetic hands; no camera needed")
    p.add_argument("--mode", default="desktop",
                   choices=("desktop", "hmd", "screentop"))
    p.add_argument("--raw", action="store_true",
                   help="also write LeapC's own .lmt recording beside each take")
    args = p.parse_args()

    poses = [slugify(x) for x in args.poses.split(",") if slugify(x)]
    if not poses:
        raise SystemExit("no poses given")
    if args.raw and args.mock:
        raise SystemExit("--raw needs a live camera: there is no LeapC stream "
                         "behind --mock")

    total = len(poses) * args.takes
    eta = total * (args.prep + args.duration)
    print("=" * 62)
    print(f"Leap pose session: {len(poses)} poses x {args.takes} takes "
          f"x {args.duration:g} s  (~{eta / 60:.1f} min)")
    print(f"  poses: {', '.join(poses)}")
    print(f"  rate:  {'every frame' if not args.hz else f'{args.hz:g} frames/s'}"
          f"   output: {args.out_dir}")
    print("  Once started it runs itself; beeps mark record start/stop.")
    print("=" * 62 + "\n")

    try:
        source = open_stream(mock=args.mock, mode=args.mode)
    except LeapUnavailable as e:
        raise SystemExit(f"\nNo live tracking: {e}\n")
    if args.mock:
        print("Mock mode: synthetic hands (no camera needed).\n")

    session = Session(source, hz=args.hz or None, out_dir=args.out_dir, raw=args.raw)
    try:
        if not args.mock:
            session.wait_for_stream()
        for i, pose in enumerate(poses, 1):
            for take in range(1, args.takes + 1):
                session.run_take(pose, take, args.takes, i, len(poses),
                                 args.duration, args.prep)
    except KeyboardInterrupt:
        print("\nInterrupted - keeping the takes recorded so far.")
    finally:
        source.stop()
        session.print_summary()


if __name__ == "__main__":
    main()
