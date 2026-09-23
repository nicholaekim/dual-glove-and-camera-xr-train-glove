"""Record the professor's reference frames with the camera, one after another.

The July glove work reproduced 81 of the professor's 102 frames and marked 21
as not reproducible (folders ending in `_NA`). This runs `record_frame.py` for
each of those, in order, so the operator only has to copy the photo the window
shows and hold still: one command for the whole batch instead of 21.

Usage:
  python scripts/leap/record_prof_frames.py --reference "..\\xr trainer\\xr trainer poses"
  python scripts/leap/record_prof_frames.py --reference ... --only 153624,156023
  python scripts/leap/record_prof_frames.py --reference ... --all      # every frame, DONE ones too
  python scripts/leap/record_prof_frames.py --reference ... --rewrite  # no camera, redo the files

Which hand. No hand filter is passed to `record_frame.py` by default, so each
file keeps the hand that was actually recorded (the label with the most
frames in the take). The left/right label is the tracker's opinion and it is
unreliable for these poses on both sides: the professor's tracker calls the
same physical arm "right" in frame 153624 and "left" in 156023, and the
Ultraleap labelled the operator's left hand "right" in every take of the
second run on 2026-09-23, so filtering on "left" wrote nothing for 9 frames.
The label in his file (`Frame 153624 | Hand ID 1499 (right)`) is shown for
reference only. `--hand left|right` forces a filter; `--hand any` is the
default spelled out.

Frames whose keypoint file (`<out>/frame_<id>_keypoints.txt`) already exists
are skipped unless `--redo` is given, so an interrupted batch can simply be
run again.

`--rewrite` records nothing: for each listed frame it takes the newest take in
`<out>/frame_<id>/` and writes the keypoint file again from it. Use it after
a change to how the file is chosen, instead of asking the operator to hold
21 poses again.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD_FRAME = HERE / "record_frame.py"
_FRAME_DIR = re.compile(r"^frame_(\d+)(?:_(DONE|NA))?$")
_HAND_LINE = re.compile(r"\((left|right)\)", re.I)
_STAMP = re.compile(r"_(\d{8}_\d{6})")


def list_frames(reference: Path, include_done: bool = False) -> list[tuple[str, Path]]:
    """(frame id, folder) for every `frame_<id>_NA` folder, or every frame with --all."""
    out = []
    for d in sorted(reference.iterdir()):
        m = _FRAME_DIR.match(d.name)
        if not d.is_dir() or not m:
            continue
        status = m.group(2)
        if status == "NA" or (include_done and status in (None, "DONE")):
            out.append((m.group(1), d))
    return out


def hand_in_reference(folder: Path, frame_id: str) -> str | None:
    """The hand the professor's file names for this frame, or None."""
    txt = folder / f"frame_{frame_id}.txt"
    if not txt.exists():
        return None
    first = txt.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    m = _HAND_LINE.search(first[0]) if first else None
    return m.group(1).lower() if m else None


def keypoint_file(out_dir: Path, frame_id: str) -> Path:
    """Where `record_frame.py` writes the frame: next to its take folder."""
    return out_dir / f"frame_{frame_id}_keypoints.txt"


def already_recorded(out_dir: Path, frame_id: str) -> bool:
    return keypoint_file(out_dir, frame_id).is_file()


def newest_take(out_dir: Path, frame_id: str) -> Path | None:
    """The most recent take .jsonl for this frame, by the stamp in its name."""
    takes = list((out_dir / f"frame_{frame_id}").glob("*.jsonl"))
    if not takes:
        return None

    def key(p: Path):
        m = _STAMP.search(p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)
    return max(takes, key=key)


def record_frame_module():
    """`record_frame.py`, imported by path (scripts/leap is not a package)."""
    spec = importlib.util.spec_from_file_location("record_frame", RECORD_FRAME)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewrite(frames, out_dir: Path, hand: str | None, dry_run: bool = False) -> int:
    """Regenerate each frame's keypoint file from its newest take."""
    rf = record_frame_module()
    failed = []
    for fid, _folder in frames:
        take = newest_take(out_dir, fid)
        if take is None:
            print(f"{fid}  no take in {out_dir / f'frame_{fid}'}")
            failed.append(fid)
            continue
        if dry_run:
            print(f"{fid}  {take.name}  (dry run)")
            continue
        result = rf.write_prof_file(take, keypoint_file(out_dir, fid), hand_filter=hand)
        counts = ", ".join(f"{s} {n}" for s, n in result["counts"].items()) or "none"
        kept = result["side"] or "nothing written"
        print(f"{fid}  {take.name}  kept {kept}  (frames: {counts})")
        if result["side"] is None:
            failed.append(fid)
    print(f"\n{len(frames) - len(failed)} keypoint file(s) written, {len(failed)} not"
          + (f" ({', '.join(failed)})" if failed else ""))
    return 1 if failed else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reference", type=Path, required=True,
                   help="folder holding the professor's frame_<id>[_NA|_DONE] folders")
    p.add_argument("--out-dir", type=Path, default=Path("recordings") / "leap" / "prof_frames")
    p.add_argument("--hand", choices=("any", "left", "right"), default="any",
                   help="keep only frames with this label (default: any, the file "
                        "keeps the label with the most frames; labels are unreliable)")
    p.add_argument("--only", default=None, help="comma-separated frame ids to record")
    p.add_argument("--all", action="store_true", help="every frame, not only the 21 _NA ones")
    p.add_argument("--redo", action="store_true", help="record frames that already have output")
    p.add_argument("--rewrite", action="store_true",
                   help="record nothing; rewrite each keypoint file from its newest take")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--prep", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true", help="list what would run and stop")
    args = p.parse_args(argv)
    hand = None if args.hand == "any" else args.hand

    frames = list_frames(args.reference, include_done=args.all)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        frames = [f for f in frames if f[0] in wanted]
    if not frames:
        print(f"no frames found under {args.reference}")
        return 2

    if args.rewrite:
        return rewrite(frames, args.out_dir, hand, dry_run=args.dry_run)

    todo = [(fid, folder) for fid, folder in frames if args.redo or not already_recorded(args.out_dir, fid)]
    print(f"{len(frames)} frame(s) listed, {len(todo)} to record, {len(frames) - len(todo)} already done")
    failed = []
    for n, (fid, folder) in enumerate(todo, 1):
        cmd = [sys.executable, str(RECORD_FRAME), fid, "--reference", str(args.reference),
               "--out-dir", str(args.out_dir),
               "--seconds", str(args.seconds), "--prep", str(args.prep)]
        if hand:
            cmd += ["--hand", hand]
        his = hand_in_reference(folder, fid) or "no"
        print(f"\n[{n}/{len(todo)}] frame {fid}, keeping {hand or 'any'} hand"
              f" (his file says {his})" + ("  (dry run)" if args.dry_run else ""))
        if args.dry_run:
            print("   ", " ".join(cmd))
            continue
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            print("\nstopped; run the same command again to continue where you left off")
            return 130
        if rc != 0:
            failed.append(fid)
            print(f"    frame {fid} did not complete (exit {rc}); continuing")
    print(f"\ndone: {len(todo) - len(failed)} recorded, {len(failed)} failed"
          + (f" ({', '.join(failed)}); re-run with --only {','.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
