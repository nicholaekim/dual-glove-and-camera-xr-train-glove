"""Replicate one of the professor's reference frames with the IR camera.

The camera half of the July `record_frame.ps1` flow (plan section 3). You are
given a frame number, the script shows you the reference image to copy, counts
you in, records a few seconds of your hand, and writes one keypoint file in
the professor's own format:

    recordings/leap/prof_frames/<frame>_keypoints.txt     the deliverable
    recordings/leap/prof_frames/<frame>/<take>.jsonl      what it came from

Why a medoid and not an average. The file holds ONE block per hand, chosen
from the seconds recorded: the medoid — the actual recorded frame closest to
the mean of that take. `scripts/glove/export_keypoints21.py` picks its summary
skeleton the same way, and for the same reason: averaging joint positions over
a take that wobbles shortens every bone, so the "average" hand is a hand
nobody has. A medoid is a real frame, with real bone lengths, and the take it
came from is kept beside it.

Why this matters for Phase 3. Of the professor's 102 frames the glove
replicated 81 and could not do 21 at all (`reference/frames/_folder_status.csv`
lists them NA): they need spread, thumb opposition or wrist angles the glove
has no sensor for. Those 21 are exactly what the camera is here to cover, and
this is the script that covers them one frame at a time.

  python scripts/leap/record_frame.py 128166
  python scripts/leap/record_frame.py frame_128166 --seconds 5
  python scripts/leap/record_frame.py 128166 --hand left
  python scripts/leap/record_frame.py 128166 --mock --prep 0    # pipeline test

Protocol (plan section 6): module flat on the table, lenses up, hand 20 to 50
cm above it, palm roughly toward the camera. Ctrl+C before the beep costs you
nothing; after it, whatever was recorded is still written.

One difference from the glove's files, and it is a feature: a Leap frame
carries the hand's real position in camera space, so the `Wrist:` line is a
measured position rather than the glove's `(0, 0, 0)`. The 0..20 landmark
block is unchanged, and `compare_to_tracker.py` aligns rigidly before scoring.
"""
import argparse
import importlib.util
import re
import time
from pathlib import Path

from leap_hand.recorder import LeapRecorder
from leap_hand.stream import LeapUnavailable, open_stream
from xr_hand.keypoints21 import frame_to_keypoints21
from xr_hand.recorder import FrameRecorder, hand_tag, pose_filename

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("recordings") / "leap" / "prof_frames"
DEFAULT_REFERENCE = Path("reference") / "frames"
MIN_VISIBLE_TIME_US = 300_000     # plan section 6: a hand counts after 0.3 s
MAX_DRAIN_ROUNDS = 8              # bound on the post-beep flush
STREAM_WAIT_TIMEOUT = 60.0
STREAM_WAIT_HANDS = 10

# frame_128166, frame_128166_DONE, frame_128166_NA, 128166 — all the same frame.
_FRAME_RE = re.compile(r"(?:frame_)?(\d+)(?:_(?:DONE|NA|left|right))?$",
                       re.IGNORECASE)


def beep(freq: int = 880, ms: int = 180) -> None:
    """Audible cue; falls back to the terminal bell off Windows."""
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


def frame_name(raw: str) -> str:
    """'128166', 'frame_128166_DONE' -> 'frame_128166'."""
    m = _FRAME_RE.match(str(raw).strip())
    if not m:
        raise SystemExit(
            f"not a frame number: {raw!r}. Give the number (128166) or the "
            "folder name (frame_128166).")
    return f"frame_{m.group(1)}"


def find_reference(name: str, root: Path):
    """The professor's image for this frame, or None if it is not here.

    His dataset is one folder per frame holding `<frame>.png` and `<frame>.txt`
    (progress suffixes `_DONE` / `_NA` were folded into `_folder_status.csv`
    when it was imported). A missing image is not an error: the frame number
    alone is enough to record against if you have the sheet in front of you.
    """
    if not root.is_dir():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        direct = root / name / f"{name}{ext}"
        if direct.is_file():
            return direct
    matches = sorted(root.rglob(f"{name}.png"))
    return matches[0] if matches else None


def glove_exporter():
    """`scripts/glove/export_prof_format.py`, imported by path.

    The professor's block layout — the 0..20 landmark table, the `Wrist:`
    line, the millimetres — is defined there, and it has to stay defined in
    exactly one place or the camera's files and the glove's files will drift
    apart while both claim to be his format. `scripts/glove/` is not a
    package, so this is how a script in `scripts/leap/` reaches it. Its
    `main()` is behind `if __name__ == "__main__"`, so importing runs nothing.
    """
    path = REPO / "scripts" / "glove" / "export_prof_format.py"
    spec = importlib.util.spec_from_file_location("glove_prof_format", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def medoid_index(frames) -> int:
    """The frame closest to the mean of the take, over 21 keypoints.

    Exactly what `scripts/glove/export_keypoints21.py` does for its summary
    row: flatten each frame to wrist-centred coordinates, take the mean of
    the take, and return the REAL frame with the smallest squared distance to
    it. Never the mean itself — that hand has shortened bones.
    """
    rows = [[c for pt in frame_to_keypoints21(f) for c in pt] for f in frames]
    n = len(rows)
    mean = [sum(r[i] for r in rows) / n for i in range(len(rows[0]))]
    best, best_d = 0, None
    for i, r in enumerate(rows):
        d = sum((a - b) ** 2 for a, b in zip(r, mean))
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def record_take(source, path: Path, name: str, seconds: float,
                prep: float) -> LeapRecorder:
    """Count down, record `seconds` of hands, return the recorder."""
    recorder = LeapRecorder(pose=name, take=1)      # every frame: 3 s is small
    skipped = 0

    def consume(rec=None) -> int:
        nonlocal skipped
        seen = 0
        for _side, lh in source.drain(64):
            seen += 1
            # Presence and settling time, never confidence: LeapC documents
            # confidence as a constant 1.0 (plan section 6).
            if lh.visible_time_us < MIN_VISIBLE_TIME_US:
                skipped += 1
                continue
            if rec is not None:
                rec.record(lh)
        return seen

    def discard_backlog() -> int:
        """Drop what queued while the start beep blocked.

        `winsound.Beep` blocks for its whole duration while LeapC's polling
        thread keeps filling the queue, so hands drained straight after a
        250 ms beep are up to 250 ms old and get written with the current
        time. Here they are also hands from BEFORE the go signal, while the
        operator is still moving into the pose — and the medoid is computed
        over whatever ends up in the file.
        """
        dropped = 0
        for _ in range(MAX_DRAIN_ROUNDS):
            n = consume()
            dropped += n
            if not n:
                break
        return dropped

    for s in range(int(round(prep)), 0, -1):
        print(f"      {s}...")
        beep(660, 120)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            consume()
            time.sleep(0.02)

    # Beep, drop what queued behind the beep, and only then open the file.
    beep(1000, 250)
    discard_backlog()
    recorder.start(path)
    print(f"      REC {seconds:g} s - hold it ", end="", flush=True)
    try:
        t_end = time.time() + seconds
        next_dot = time.time() + 0.5
        while time.time() < t_end:
            consume(recorder)
            if time.time() >= next_dot:
                print(".", end="", flush=True)
                next_dot += 0.5
            time.sleep(0.005)
    finally:
        # Runs on Ctrl+C too: close the file and keep whatever was captured.
        print(flush=True)
        recorder.stop()
        beep(500, 300)
    recorder.skipped_young = skipped
    return recorder


def wait_for_stream(source) -> None:
    print(f"Waiting for hands (need {STREAM_WAIT_HANDS}, timeout "
          f"{int(STREAM_WAIT_TIMEOUT)} s)...")
    print("  Hold your hand 20 to 50 cm above the module, lenses up.")
    t0 = time.time()
    seen = 0
    while time.time() - t0 < STREAM_WAIT_TIMEOUT:
        seen += len(source.drain(64))
        if seen >= STREAM_WAIT_HANDS:
            print("  OK - tracking\n")
            return
        time.sleep(0.05)
    raise SystemExit(
        "No hands seen. Is the camera plugged into a direct USB port and "
        "listed in the Ultraleap Control Panel?\n"
        "  check with: python scripts/leap/check_setup.py")


def write_prof_file(take_path: Path, out_file: Path, hand_filter=None) -> dict:
    """Medoid block per hand from one take -> the professor-format file."""
    exporter = glove_exporter()
    by_hand: dict = {}
    for frame, wall in FrameRecorder.load(take_path):
        if hand_filter and frame.hand_side != hand_filter:
            continue
        by_hand.setdefault(frame.hand_side, []).append((frame, wall))

    blocks, chosen = [], {}
    for side in sorted(by_hand):
        pairs = by_hand[side]
        i = medoid_index([f for f, _w in pairs])
        frame, wall = pairs[i]
        blocks.append(exporter.frame_block(frame, wall))
        chosen[side] = (i, len(pairs))
    if not blocks:
        return {}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(
        description="Record one professor reference frame with the Ultraleap "
                    "camera and write it in his keypoint format.")
    p.add_argument("frame", help="frame number (128166) or folder name "
                                 "(frame_128166)")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="seconds recorded (default: 3)")
    p.add_argument("--prep", type=float, default=5.0,
                   help="countdown seconds before recording (default: 5)")
    p.add_argument("--hand", choices=("left", "right"), default=None,
                   help="keep only this hand (default: every hand recorded)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"output root (default: {DEFAULT_OUT})")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                   help=f"the professor's frames (default: {DEFAULT_REFERENCE})")
    p.add_argument("--mock", action="store_true",
                   help="synthetic hands; no camera needed (pipeline test)")
    p.add_argument("--mode", default="desktop",
                   choices=("desktop", "hmd", "screentop"))
    args = p.parse_args()

    name = frame_name(args.frame)
    image = find_reference(name, args.reference)

    print("=" * 62)
    print(f"Professor frame {name}" + ("  [mock]" if args.mock else ""))
    if image is not None:
        print(f"  reference image: {image}")
        print("  Open it and copy the pose with your hand.")
    else:
        print(f"  no reference image under {args.reference} — frame {name}")
        print("  Copy the pose from the sheet or the folder you have.")
    print(f"  recording {args.seconds:g} s, then the medoid frame is written")
    print("=" * 62 + "\n")

    try:
        source = open_stream(mock=args.mock, mode=args.mode)
    except LeapUnavailable as e:
        raise SystemExit(f"\nNo live tracking: {e}\n")

    take_dir = args.out_dir / name
    take_path = take_dir / pose_filename(name, 1)
    try:
        if not args.mock:
            wait_for_stream(source)
        recorder = record_take(source, take_path, name, args.seconds, args.prep)
    finally:
        source.stop()

    if recorder.count == 0:
        take_path.unlink(missing_ok=True)
        raise SystemExit("No frames captured — the tracker reported no hand. "
                         "Nothing written.")
    print(f"      saved {recorder.count} frames "
          f"({hand_tag(recorder.hands_seen)}) -> {take_path.name}")
    if getattr(recorder, "skipped_young", 0):
        print(f"      ({recorder.skipped_young} hands skipped: tracked for "
              f"less than {MIN_VISIBLE_TIME_US / 1000:.0f} ms — settling)")

    out_file = args.out_dir / f"{name}_keypoints.txt"
    chosen = write_prof_file(take_path, out_file, hand_filter=args.hand)
    if not chosen:
        raise SystemExit(
            f"No {args.hand} frames in this take, so no keypoint file was "
            "written. Re-record, or drop --hand to keep the hand you used.")

    print()
    for side, (i, n) in sorted(chosen.items()):
        print(f"  {side:<5} medoid frame {i + 1} of {n}")
    print(f"\nwrote {out_file}")
    print(f"  take kept in {take_dir}")
    print(f"  compare against the tracker: python scripts/compare_to_tracker.py")


if __name__ == "__main__":
    main()
