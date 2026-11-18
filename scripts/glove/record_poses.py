"""Guided pose-recording session for the StretchSense glove — hands-free.

Glove only: this recorder opens no camera.

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

A pose list and a picture of the glove's application (13 Nov 2026):

  --pose-list FILE.json   take the poses from a pose list (`items` with `id`,
                          `label`, `shape`) and show each pose's label and
                          shape before its takes
  --start-at ID           begin at that pose of the list
  --hand left|right       keep only that glove's frames
  --first-take auto       number a pose's takes after the ones already in
                          the output folder
  --app-screenshot "XR Train"
                          save a picture of that application's window for
                          every take, at the midpoint of the recording, to
                          xr_train/<take>.png (`xr_hand.app_screenshot`)
  --note TEXT             free text stored with every take

With any of these, every take also gets a `<take>.meta.json` beside it: pose,
label, shape, hand, frames, hold_start and hold_end (time.time() seconds, the
recording as commanded), when it actually stopped, the note, and the
`app_screenshot*` keys (where the picture is, when and how it was captured,
or why there is none). The recorder refuses to start if the window cannot be
captured; a capture that fails later never fails the take, and a take with no
frames keeps neither file nor picture. Without these flags the recorder
writes exactly what it always wrote.

Ctrl+C at any point keeps the takes recorded so far.
"""
import argparse
import json
import re
import tempfile
import threading
import time
from pathlib import Path

from xr_hand.app_screenshot import (capture_window, empty_record, meta_keys,
                                    window_titles)
from xr_hand.mock import MockHandGenerator
from xr_hand.parser import parse_hand_message
from xr_hand.receiver import OSCHandReceiver
from xr_hand.recorder import (
    FrameRecorder,
    finalize_pose_name,
    hand_tag,
    load_pose_list,
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

# `--app-screenshot`: where the application window's pictures go, how long a
# take waits for its picture once the recording is over, and why there is none.
APP_SHOTS = "xr_train"
APP_SHOT_WAIT_S = 5.0
APP_NOT_REACHED = "the take ended before the midpoint of its recording"
APP_TIMEOUT = f"the capture did not finish within {APP_SHOT_WAIT_S:g} s of the recording"
APP_VANISHED = "the capture reported a file that is not there"
UNKNOWN = "unknown"
SENSORS = "glove only, no camera"


def beep(freq: int = 880, ms: int = 180) -> None:
    """Audible cue; falls back to the terminal bell off Windows."""
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


def next_take(out_dir: Path, pose: str) -> int:
    """1 + the highest take number of `pose` among the takes in `out_dir`
    (`--first-take auto`)."""
    name = re.compile(rf"^{re.escape(pose)}_(?:left|right|both|nohand)_take(\d+)"
                      r"_\d{8}_\d{6}\.jsonl$")
    numbers = [int(m.group(1)) for p in Path(out_dir).glob(f"{pose}_*_take*.jsonl")
               if (m := name.match(p.name))]
    return max(numbers, default=0) + 1


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
    def __init__(self, source, hz, out_dir: Path, hand=None, app_window=None,
                 app_capture=None, pose_text=None, note=None, mock=False):
        self.source = source
        self.hz = hz
        self.out_dir = out_dir
        self.monitors = {"left": StreamMonitor("left"), "right": StreamMonitor("right")}
        self.results = []          # one dict per take
        self._warnings = {}        # message -> count, reset per take
        # `--hand`: the one glove whose frames are kept, None for both.
        self.hand = hand
        # `--app-screenshot`: the title of an application window to save a
        # picture of at the midpoint of every take, None for none. The capture
        # function is passed in so a test can stand in for the window.
        self.app_window = app_window or None
        self.app_capture = app_capture
        self._app_record: dict = {}
        # `--pose-list`: {pose id: (label, shape)}, shown before the takes.
        self.pose_text = dict(pose_text or {})
        self.note = note
        self.mock = bool(mock)
        # A sidecar per take only when one of the newer flags asks for
        # something a take file cannot hold.
        self.sidecar = bool(self.app_window or self.pose_text or self.hand
                            or self.note is not None)

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
            if recorder is not None and (self.hand is None
                                         or frame.hand_side == self.hand):
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
                if self.hand and self.hand not in hands:
                    print(f"  ! no frames from the {self.hand} glove yet: only "
                          f"{', '.join(sorted(hands))} is streaming\n")
                return
            time.sleep(0.05)
        raise SystemExit(
            "No valid glove data received. Is XR Trainer running and streaming "
            "to this host/port? (try: python scripts/run_osc.py --dump --no-viz)"
        )

    # --- the application window's picture ----------------------------------
    def _app_shot_start(self, path: Path) -> threading.Thread:
        """Save the application window to `path` on its own thread: drawing a
        window can take tens of milliseconds, and the recording loop must not
        stop for it."""
        self._app_record = {}
        capture = self.app_capture or capture_window

        def work():
            try:
                self._app_record = capture(self.app_window, path)
            except Exception as e:       # a picture never fails a take
                self._app_record = empty_record(
                    self.app_window, f"capture error: {type(e).__name__}: {e}")

        thread = threading.Thread(target=work, name="app-screenshot", daemon=True)
        thread.start()
        return thread

    def _app_shot_finish(self, thread, path) -> dict:
        """The record of the take's picture once its thread is done; {} for a
        session that takes none."""
        if path is None:
            return {}
        if thread is None:               # the recording ended before its midpoint
            return empty_record(self.app_window, APP_NOT_REACHED)
        thread.join(APP_SHOT_WAIT_S)
        if thread.is_alive():
            return empty_record(self.app_window, APP_TIMEOUT)
        record = dict(self._app_record) or empty_record(self.app_window, UNKNOWN)
        if record.get("file") and not path.is_file():
            record.update(file=None, missing_reason=APP_VANISHED)
        return record

    def _write_meta(self, final: Path, pose: str, take: int, recorder,
                    t0: float, duration: float, stopped: float, shot: dict) -> None:
        """`<take>.meta.json`: what a take file cannot say about itself."""
        label, shape = self.pose_text.get(pose, ("", ""))
        meta = {
            "file": final.name,
            "pose": pose,
            "take": take,
            "label": label or None,
            "shape": shape or None,
            "sensors": SENSORS,
            # the glove asked for, and the gloves whose frames are in the file
            "hand": self.hand,
            "hands_recorded": hand_tag(recorder.hands_seen),
            "frames": recorder.count,
            "frames_per_second_kept": self.hz,      # None = every frame
            # The recording as commanded, time.time() seconds: from the moment
            # it started to that plus the duration; then when it really
            # stopped (earlier only when the session was interrupted).
            "hold_start": round(t0, 6),
            "hold_end": round(t0 + duration, 6),
            "recording_stopped": round(stopped, 6),
            "mock": self.mock,
            "note": self.note,
            # `--app-screenshot`: no keys at all for a session that takes none
            **(meta_keys(shot, self.app_window, APP_SHOTS, UNKNOWN)
               if self.app_window else {}),
        }
        final.with_name(final.stem + ".meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # --- protocol ---------------------------------------------------------
    def run_take(self, pose: str, take: int, n_takes: int,
                 pose_idx: int, n_poses: int, duration: float, prep: float) -> None:
        self._warnings = {}
        title = pose.replace("_", " ").upper()
        print(f"--- Pose {pose_idx}/{n_poses}: {title}  (take {take}/{n_takes}) ---")
        if pose in self.pose_text:
            label, shape = self.pose_text[pose]
            print(f"    {label}: {shape}" if shape else f"    {label}")
        else:
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
        app_png = (self.out_dir / APP_SHOTS / f"{path.stem}.png"
                   if self.app_window else None)
        shot_thread = None
        recorder.start(path)
        beep(1000, 250)
        print(f"      REC {duration:g} s - hold it ", end="", flush=True)
        t0 = time.time()
        try:
            t_end = t0 + duration
            midpoint = t0 + duration / 2.0
            next_dot = time.time() + 0.5
            while time.time() < t_end:
                self._consume(recorder)
                now = time.time()
                if app_png is not None and shot_thread is None and now >= midpoint:
                    shot_thread = self._app_shot_start(app_png)
                if now >= next_dot:
                    print(".", end="", flush=True)  # heartbeat: still recording
                    next_dot += 0.5
                time.sleep(0.01)
        finally:
            # Runs on Ctrl+C too: close the file and either finalize the
            # take (partial data is still labeled data) or drop it if empty.
            print(flush=True)
            recorder.stop()
            stopped = time.time()
            beep(500, 300)
            shot = self._app_shot_finish(shot_thread, app_png)
            entry = {"pose": pose, "take": take, "frames": recorder.count,
                     "hands": hand_tag(recorder.hands_seen),
                     "ok": recorder.count > 0,
                     "warnings": sum(self._warnings.values())}
            if recorder.count == 0:
                path.unlink(missing_ok=True)
                if app_png is not None:      # no take, so nothing to show beside
                    app_png.unlink(missing_ok=True)
                print("      FAILED: no frames captured (did the stream stop?)\n")
            else:
                final = finalize_pose_name(path, recorder.hands_seen)
                entry["file"] = final.name
                if shot.get("file"):
                    # keep the picture's name matched to the take's
                    renamed = app_png.with_name(f"{final.stem}.png")
                    try:
                        app_png.replace(renamed)
                        shot["file"] = renamed.name
                    except OSError as e:
                        shot.update(file=None, missing_reason=(
                            f"the picture could not be renamed to its take: {e}"))
                entry["app_shot"] = shot
                if self.sidecar:
                    self._write_meta(final, pose, take, recorder, t0, duration,
                                     stopped, shot)
                print(f"      saved {recorder.count} frames ({entry['hands']}) "
                      f"-> {final.name}")
                if self.app_window and shot.get("file"):
                    print(f"      {self.app_window} picture -> {APP_SHOTS}\\{shot['file']}"
                          + ("  (BLACK: not a picture of the hand)"
                             if shot.get("black") else ""))
                elif self.app_window:
                    print(f"      ! no {self.app_window} picture: "
                          f"{shot.get('missing_reason') or UNKNOWN}")
                print()
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
        if self.app_window:
            shots = [r.get("app_shot") or {} for r in ok]
            have = sum(1 for a in shots if a.get("file"))
            black = sum(1 for a in shots if a.get("file") and a.get("black"))
            print(f"\n  APP SCREENSHOTS: {have} of {len(ok)} takes have one "
                  f"(the {self.app_window!r} window, in {self.out_dir / APP_SHOTS})"
                  + (f"; {black} of them BLACK, not a picture of the hand"
                     if black else ""))
            for r in ok:
                shot = r.get("app_shot") or {}
                if not shot.get("file"):
                    print(f"    no screenshot for {r['file']}: "
                          f"{shot.get('missing_reason') or UNKNOWN}")
        if ok:
            print(f"\nFiles in {self.out_dir}")
            print("  playback:  python scripts/playback.py <file>")
            print("  to csv:    python scripts/export.py <file>")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Guided, hands-free pose recording with the glove (no camera).")
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
    p.add_argument("--pose-list", default=None, metavar="FILE.json",
                   help="take the poses from a pose list (items with id, label, "
                        "shape) instead of --poses, and show each pose's label "
                        "and shape")
    p.add_argument("--start-at", default=None, metavar="POSE_ID",
                   help="with --pose-list: begin at this pose of the list")
    p.add_argument("--hand", default=None, choices=("left", "right"),
                   help="keep only this glove's frames (default: both gloves)")
    p.add_argument("--first-take", default="1", metavar="N|auto",
                   help="number of the first take of each pose (default: 1); "
                        "'auto' continues after the takes already in --out-dir")
    p.add_argument("--app-screenshot", default=None, metavar="WINDOW_TITLE",
                   help="also save a picture of the application window with "
                        "exactly this title (e.g. \"XR Train\") for every take, "
                        "at the midpoint of its recording, to xr_train/<take>.png. "
                        "The recorder refuses to start if the window cannot be "
                        "captured")
    p.add_argument("--note", default=None,
                   help="free text stored as 'note' in every take's meta.json")
    args = p.parse_args()

    pose_text: dict = {}
    if args.pose_list:
        try:
            poses, pose_text = load_pose_list(args.pose_list)
        except ValueError as e:
            raise SystemExit(f"--pose-list: {e}")
        if args.start_at:
            if args.start_at not in poses:
                raise SystemExit(f"--start-at {args.start_at}: not in "
                                 f"{args.pose_list} ({poses[0]} ... {poses[-1]})")
            poses = poses[poses.index(args.start_at):]
    elif args.start_at:
        raise SystemExit("--start-at needs --pose-list: it names a pose of the list")
    else:
        poses = [slugify(x) for x in args.poses.split(",") if slugify(x)]
    if not poses:
        raise SystemExit("no poses given")
    auto_take = str(args.first_take).strip().lower() == "auto"
    first_take = 1
    if not auto_take:
        try:
            first_take = int(args.first_take)
        except ValueError:
            raise SystemExit(f"--first-take must be a number or 'auto', not {args.first_take!r}")
        if first_take < 1:
            raise SystemExit("--first-take must be 1 or more")
    if args.takes < 1:
        raise SystemExit("--takes must be 1 or more")

    probe: dict = {}
    if args.app_screenshot:
        # One trial capture before anything is recorded: a session whose every
        # take says "window minimised" is found out now, not afterwards.
        with tempfile.TemporaryDirectory() as tmp:
            probe = capture_window(args.app_screenshot, Path(tmp) / "probe.png")
        if not probe.get("file"):
            raise SystemExit(
                f"--app-screenshot: the window titled {args.app_screenshot!r} "
                f"cannot be captured ({probe.get('missing_reason')}). Start the "
                "application and leave its window open, not minimised. Open "
                f"windows: {', '.join(window_titles()) or 'none'}")
        if probe.get("black"):
            raise SystemExit(
                f"--app-screenshot: the {args.app_screenshot!r} window captures "
                "as a black picture. Bring it to the front, uncovered, and "
                "start again")

    total = len(poses) * args.takes
    eta = total * (args.prep + args.duration)
    print("=" * 62)
    print(f"Pose recording session: {len(poses)} poses x {args.takes} takes "
          f"x {args.duration:g} s  (~{eta / 60:.1f} min)")
    print(f"  sensors: {SENSORS}"
          + (f"   hand: {args.hand}" if args.hand else ""))
    print(f"  poses: {', '.join(poses)}")
    print(f"  rate:  {'every frame' if not args.hz else f'{args.hz:g} frames/s'}"
          f"   output: {args.out_dir}")
    if args.app_screenshot:
        print(f"  app screenshots: the {args.app_screenshot!r} window "
              f"({probe.get('width')}x{probe.get('height')}, {probe.get('method')}), "
              "one per take at the midpoint of its recording")
    print("  Once started it runs itself; beeps mark record start/stop.")
    print("=" * 62 + "\n")

    if args.mock:
        source = MockSource()
        print("Mock mode: streaming synthetic hands (no glove needed).\n")
    else:
        source = OSCHandReceiver(host=args.host, port=args.port,
                                 kinematic_addr=args.address)
    source.start()

    session = Session(source, hz=args.hz or None, out_dir=args.out_dir,
                      hand=args.hand, app_window=args.app_screenshot,
                      pose_text=pose_text, note=args.note, mock=args.mock)
    try:
        if not args.mock:
            session.wait_for_stream()
        for i, pose in enumerate(poses, 1):
            first = next_take(args.out_dir, pose) if auto_take else first_take
            last = first + args.takes - 1
            for take in range(first, last + 1):
                session.run_take(pose, take, last, i, len(poses),
                                 args.duration, args.prep)
    except KeyboardInterrupt:
        print("\nInterrupted — keeping the takes recorded so far.")
    finally:
        source.stop()
        session.print_summary()


if __name__ == "__main__":
    main()
