"""Guided pose-recording session for the StretchSense glove — hands-free.

Walks through a list of finger poses, recording each one several times:
announces the pose, counts down with beeps, records a few seconds of
validated frames, then moves on. No keyboard needed once it starts, so it
works while wearing the glove. Each take lands in recordings/poses/ as its
own labeled JSONL (every frame carries "pose" and "take" fields), named
like fist_left_take2_20260701_154212.jsonl.

Requires XR Trainer streaming on 127.0.0.1:9002 (same as record.py); the
script waits until it sees valid glove packets before the first take, so
start it first and put the glove on calmly. Or dry-run without hardware:

  python scripts/record_poses.py --mock --takes 1 --duration 2 --prep 2

Real sessions:
  python scripts/record_poses.py                          # 6 poses x 3 takes x 5 s
  python scripts/record_poses.py --poses fist,peace --takes 2 --duration 4
  python scripts/record_poses.py --hz 0                   # keep every frame (~60/s)

Ctrl+C at any point keeps the takes recorded so far.
"""
import argparse
import time
from pathlib import Path

from xr_hand.mock import MockHandGenerator
from xr_hand.parser import parse_hand_message
from xr_hand.receiver import OSCHandReceiver
from xr_hand.recorder import (
    FrameRecorder,
    finalize_pose_name,
    hand_tag,
    pose_filename,
    slugify,
)
from xr_hand.validator import StreamMonitor, validate_raw_message

# The candidate finger-pose set for the glove feasibility test. All but
# `pinch` differ strongly in finger flexion; pinch probes the glove's limit
# (contact vs near-contact looks similar to stretch sensors).
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

STREAM_WAIT_TIMEOUT = 120.0  # s to wait for the first valid glove packets
STREAM_WAIT_PACKETS = 10     # valid packets required before starting
MAX_WARNINGS_PER_TAKE = 8    # distinct warnings printed before suppressing


def beep(freq: int = 880, ms: int = 180) -> None:
    """Audible cue; falls back to the terminal bell off Windows."""
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


class MockSource:
    """Both mock hands at ~60 Hz behind the same drain() API as the receiver."""

    def __init__(self):
        self.gens = {
            "left": MockHandGenerator(hand="left"),
            "right": MockHandGenerator(hand="right"),
        }
        self._last = time.time()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def drain(self, max_items: int = 64) -> list:
        now = time.time()
        n = int((now - self._last) * 60.0)
        if n <= 0:
            return []
        n = min(n, max_items // 2)
        self._last += n / 60.0  # keep the fractional remainder for pacing
        out = []
        for _ in range(n):
            for hand, gen in self.gens.items():
                out.append((hand, gen.next_frame()))
        return out


class Session:
    def __init__(self, source, hz, out_dir: Path):
        self.source = source
        self.hz = hz
        self.out_dir = out_dir
        self.monitors = {"left": StreamMonitor("left"), "right": StreamMonitor("right")}
        self.results = []          # one dict per take
        self._warnings = {}        # message -> count, reset per take

    # --- stream plumbing ------------------------------------------------
    def _consume(self, recorder=None) -> None:
        """Drain pending packets: validate + monitor everything, record if asked.

        Runs during countdowns too (with recorder=None) so the queue stays
        fresh and counter-freeze detection spans the whole session.
        """
        for hand, raw in self.source.drain(64):
            result = validate_raw_message(raw)
            if not result.is_valid:
                self._warn(f"[{hand}] INVALID packet: " + "; ".join(result.errors))
                continue
            frame = parse_hand_message(raw, hand_side_hint=hand)
            for w in self.monitors[hand].update(frame.packet_counter):
                self._warn(f"[{hand}] {w}")
            for w in result.warnings:
                self._warn(f"[{hand}] {w}")
            if recorder is not None:
                recorder.record(frame)

    def _warn(self, msg: str) -> None:
        if msg not in self._warnings:
            self._warnings[msg] = 0
            if len(self._warnings) <= MAX_WARNINGS_PER_TAKE:
                print(f"      ! {msg}")
            elif len(self._warnings) == MAX_WARNINGS_PER_TAKE + 1:
                print("      ! (more warnings suppressed; see session summary)")
        self._warnings[msg] += 1

    def wait_for_stream(self) -> None:
        print(f"Waiting for glove stream (need {STREAM_WAIT_PACKETS} valid packets, "
              f"timeout {int(STREAM_WAIT_TIMEOUT)} s)...")
        print("  XR Trainer must be running, glove connected and calibrated.")
        t0 = time.time()
        good = 0
        hands = set()
        while time.time() - t0 < STREAM_WAIT_TIMEOUT:
            for hand, raw in self.source.drain(64):
                if validate_raw_message(raw).is_valid:
                    good += 1
                    hands.add(hand)
            if good >= STREAM_WAIT_PACKETS:
                print(f"  OK - receiving valid frames from: {', '.join(sorted(hands))}\n")
                return
            time.sleep(0.05)
        raise SystemExit(
            "No valid glove data received. Is XR Trainer running and streaming "
            "to this host/port? (try: python scripts/run_osc.py --dump --no-viz)"
        )

    # --- protocol ---------------------------------------------------------
    def run_take(self, pose: str, take: int, n_takes: int,
                 pose_idx: int, n_poses: int, duration: float, prep: float) -> None:
        self._warnings = {}
        title = pose.replace("_", " ").upper()
        print(f"--- Pose {pose_idx}/{n_poses}: {title}  (take {take}/{n_takes}) ---")
        hint = POSE_HINTS.get(pose)
        if hint:
            print(f"    Hold: {hint}")

        # Prep countdown — keep draining so stale frames never leak into the take.
        for s in range(int(round(prep)), 0, -1):
            if s <= 3:
                print(f"      {s}...")
                beep(660, 120)
            t_end = time.time() + 1.0
            while time.time() < t_end:
                self._consume()
                time.sleep(0.02)

        recorder = FrameRecorder(hz=self.hz, pose=pose, take=take)
        path = self.out_dir / pose_filename(pose, take)
        recorder.start(path)
        beep(1000, 250)
        print(f"      REC {duration:g} s - hold it ", end="", flush=True)
        try:
            t_end = time.time() + duration
            next_dot = time.time() + 0.5
            while time.time() < t_end:
                self._consume(recorder)
                if time.time() >= next_dot:
                    print(".", end="", flush=True)  # heartbeat: still recording
                    next_dot += 0.5
                time.sleep(0.01)
        finally:
            # Runs on Ctrl+C too: close the file and either finalize the
            # take (partial data is still labeled data) or drop it if empty.
            print(flush=True)
            recorder.stop()
            beep(500, 300)
            entry = {"pose": pose, "take": take, "frames": recorder.count,
                     "hands": hand_tag(recorder.hands_seen),
                     "ok": recorder.count > 0,
                     "warnings": sum(self._warnings.values())}
            if recorder.count == 0:
                path.unlink(missing_ok=True)
                print("      FAILED: no frames captured (did the stream stop?)\n")
            else:
                final = finalize_pose_name(path, recorder.hands_seen)
                entry["file"] = final.name
                print(f"      saved {recorder.count} frames ({entry['hands']}) "
                      f"-> {final.name}\n")
            self.results.append(entry)

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
                    if r["warnings"]:
                        note += f" ({r['warnings']} warn)"
                else:
                    note = f"take{r['take']}: FAILED"
                parts.append(note)
            print(f"  {pose:<12} " + "   ".join(parts))
        if ok:
            print(f"\nFiles in {self.out_dir}")
            print("  playback:  python scripts/playback.py <file>")
            print("  to csv:    python scripts/export.py <file>")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Guided, hands-free pose recording with the glove.")
    p.add_argument("--poses", default=",".join(DEFAULT_POSES),
                   help=f"comma-separated pose names (default: {','.join(DEFAULT_POSES)})")
    p.add_argument("--takes", type=int, default=3, help="repetitions per pose (default: 3)")
    p.add_argument("--duration", type=float, default=5.0,
                   help="seconds recorded per take (default: 5)")
    p.add_argument("--prep", type=float, default=5.0,
                   help="seconds to get into the pose before each take (default: 5)")
    p.add_argument("--hz", type=float, default=5.0,
                   help="frames saved per second (default: 5; 0 = keep every frame)")
    p.add_argument("--out-dir", type=Path, default=Path("recordings") / "poses")
    p.add_argument("--mock", action="store_true", help="dry-run with synthetic hands")
    p.add_argument("--port", type=int, default=9002)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--address", default="/v1/animation/kinematic/all")
    args = p.parse_args()

    poses = [slugify(x) for x in args.poses.split(",") if slugify(x)]
    if not poses:
        raise SystemExit("no poses given")

    total = len(poses) * args.takes
    eta = total * (args.prep + args.duration)
    print("=" * 62)
    print(f"Pose recording session: {len(poses)} poses x {args.takes} takes "
          f"x {args.duration:g} s  (~{eta / 60:.1f} min)")
    print(f"  poses: {', '.join(poses)}")
    print(f"  rate:  {'every frame' if not args.hz else f'{args.hz:g} frames/s'}"
          f"   output: {args.out_dir}")
    print("  Once started it runs itself; beeps mark record start/stop.")
    print("=" * 62 + "\n")

    if args.mock:
        source = MockSource()
        print("Mock mode: streaming synthetic hands (no glove needed).\n")
    else:
        source = OSCHandReceiver(host=args.host, port=args.port,
                                 kinematic_addr=args.address)
    source.start()

    session = Session(source, hz=args.hz or None, out_dir=args.out_dir)
    try:
        if not args.mock:
            session.wait_for_stream()
        for i, pose in enumerate(poses, 1):
            for take in range(1, args.takes + 1):
                session.run_take(pose, take, args.takes, i, len(poses),
                                 args.duration, args.prep)
    except KeyboardInterrupt:
        print("\nInterrupted — keeping the takes recorded so far.")
    finally:
        source.stop()
        session.print_summary()


if __name__ == "__main__":
    main()
