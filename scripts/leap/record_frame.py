"""Replicate one of the professor's reference frames with the IR camera.

The camera half of the July `record_frame.ps1` flow (plan section 3). You are
given a frame number, the script shows you the reference image to copy, counts
you in, records a few seconds of your hand, and writes one keypoint file in
the professor's own format:

    recordings/leap/prof_frames/<frame>_keypoints.txt     the deliverable
    recordings/leap/prof_frames/<frame>/<take>.jsonl      what it came from

Why a medoid and not an average. The file holds ONE block, chosen from the
seconds recorded: the medoid — the actual recorded frame closest to the mean
of that take. `scripts/glove/export_keypoints21.py` picks its summary
skeleton the same way, and for the same reason: averaging joint positions over
a take that wobbles shortens every bone, so the "average" hand is a hand
nobody has. A medoid is a real frame, with real bone lengths, and the take it
came from is kept beside it.

Candidates are compared after a RIGID alignment onto the take's mean (see
`medoid_index`), because a hand held still for three seconds still turns
slowly and a 5-degree drift moves a fingertip 8.7 mm — far above the tracker's
0.20 mm jitter. Without that, "most typical" quietly means "held at the
average angle" rather than "in the right pose". The frame written out is the
original, untouched one: the alignment decides the winner and nothing else.

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

Which hand. The left/right label on a frame is the tracker's opinion, not a
fact, and for these poses it is unreliable on both sides: the professor's
tracker calls the same physical arm "right" in frame 153624 and "left" in
156023, and on 2026-09-23 the Ultraleap labelled the operator's left hand
"right" in every take of the second run. So nothing is filtered by label by
default. His format holds one hand per frame, so when a take has frames under
both labels the file keeps the label with the most frames, which is the hand
that was actually held over the camera whatever it was called, and the
script prints what it kept and what it dropped. `--hand left|right` still
forces one label, for a take where two hands really were in view.

One difference from the glove's files, and it is a feature: a Leap frame
carries the hand's real position in camera space, so the `Wrist:` line is a
measured position rather than the glove's `(0, 0, 0)`. The 0..20 landmark
block is unchanged, and `compare_to_tracker.py` aligns rigidly before scoring.
"""
import os
import argparse
import importlib.util
import re
import time
from pathlib import Path

import numpy as np

from cam_hand.align import align_points
from cam_hand.fusion import PALM_IDX
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
    """Which frame of the take is the most typical POSE. Returns its index.

    Same idea as `scripts/glove/export_keypoints21.py`: take the mean of the
    take and pick the REAL frame nearest it, never the mean itself — averaged
    joint positions have shortened bones, so the "average" hand is a hand
    nobody has.

    What differs, and why. Wrist-centring alone removes translation but not
    ORIENTATION. A hand held still for three seconds still rotates slowly,
    and at 100 mm from the wrist a 5-degree drift moves a fingertip 8.7 mm —
    an order of magnitude above the 0.20 mm jitter the gate measured. Judged
    on raw wrist-centred coordinates, "closest to the mean" therefore means
    "held at the average ANGLE", and a frame in a plainly wrong pose taken
    mid-sweep can win over a correct one recorded early.

    So each candidate is first aligned RIGIDLY — rotation and translation,
    scale 1 — onto the take's mean using the palm landmarks, which are the
    near-rigid part of a hand. What is left after that alignment is the only
    thing this is supposed to be ranking: how the FINGERS are posed.

    The alignment is used for scoring only. The caller exports the original,
    untouched frame, because the deliverable is a measurement of the hand in
    camera space, not a re-oriented copy of it.
    """
    pts = [np.asarray(frame_to_keypoints21(f), dtype=float) for f in frames]
    if len(pts) == 1:
        return 0

    # The mean is only a reference to align against, so a plain elementwise
    # mean is fine here: nothing is exported from it.
    mean = np.mean(np.stack(pts), axis=0)
    best, best_d = 0, None
    for i, p in enumerate(pts):
        aligned, _rmse, _err, _s = align_points(p, mean, with_scale=False,
                                                subset=PALM_IDX)
        d = float(((aligned - mean) ** 2).sum())
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
    """The take's medoid frame -> the professor-format file. Returns what it did.

    One block, because his format holds one hand per frame. Without a filter
    the block comes from the label with the most frames; a tie goes to the
    label that sorts first, so a rerun picks the same one. With `hand_filter`
    only that label is considered. The label decides nothing else: see the
    module docstring for why it cannot be trusted.

    Returns {"side", "index", "counts"}: the label kept (None when nothing
    was written), the medoid's index among that label's frames, and how many
    frames the take held under each label.
    """
    exporter = glove_exporter()
    by_hand: dict = {}
    for frame, wall in FrameRecorder.load(take_path):
        by_hand.setdefault(frame.hand_side, []).append((frame, wall))
    counts = {side: len(by_hand[side]) for side in sorted(by_hand)}

    if hand_filter:
        side = hand_filter if hand_filter in by_hand else None
    else:
        side = max(sorted(by_hand), key=lambda s: len(by_hand[s]), default=None)
    if side is None:
        return {"side": None, "index": None, "counts": counts}

    pairs = by_hand[side]
    i = medoid_index([f for f, _w in pairs])
    frame, wall = pairs[i]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(exporter.frame_block(frame, wall) + "\n", encoding="utf-8")
    return {"side": side, "index": i, "counts": counts}


def kept_summary(result: dict) -> str:
    """'kept 155 frames labelled left, dropped 68 labelled right'."""
    counts, side = result["counts"], result["side"]
    parts = []
    if side is not None:
        parts.append(f"kept {counts[side]} frames labelled {side}")
    dropped = [f"{n} labelled {s}" for s, n in counts.items() if s != side]
    if dropped:
        parts.append("dropped " + ", ".join(dropped))
    return ", ".join(parts) if parts else "no frames in the take"


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
                   help="keep only frames with this label (default: the label "
                        "with the most frames, since labels are unreliable)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"output root (default: {DEFAULT_OUT})")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                   help=f"the professor's frames (default: {DEFAULT_REFERENCE})")
    p.add_argument("--no-open", action="store_true",
                   help="do not open the reference photo on screen; only print its path")
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
        if args.no_open:
            print("  Open it and copy the pose with your hand.")
        else:
            # Put the professor's photo on screen so the operator is not
            # hunting for a file while the countdown runs. Windows opens it
            # in the default viewer; elsewhere the path is all we can offer.
            try:
                os.startfile(str(image))
                print("  The photo is opening on screen: copy the pose with your hand.")
            except (AttributeError, OSError):
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
    result = write_prof_file(take_path, out_file, hand_filter=args.hand)
    print(f"\n  {kept_summary(result)}")
    if result["side"] is None:
        raise SystemExit(
            f"No frames labelled {args.hand} in this take, so no keypoint file "
            "was written. The label is the tracker's guess: drop --hand to keep "
            "the hand you actually used.")
    side = result["side"]
    print(f"  medoid frame {result['index'] + 1} of {result['counts'][side]}")
    print(f"\nwrote {out_file}")
    print(f"  take kept in {take_dir}")
    print(f"  compare against the tracker: python scripts/compare_to_tracker.py")


if __name__ == "__main__":
    main()
