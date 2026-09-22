"""Record the professor's reference frames with the camera, one after another.

The July glove work reproduced 81 of the professor's 102 frames and marked 21
as not reproducible (folders ending in `_NA`). This runs `record_frame.py` for
each of those, in order, so the operator only has to copy the photo the window
shows and hold still: one command for the whole batch instead of 21.

Usage:
  python scripts/leap/record_prof_frames.py --reference "..\\xr trainer\\xr trainer poses"
  python scripts/leap/record_prof_frames.py --reference ... --only 153624,156023
  python scripts/leap/record_prof_frames.py --reference ... --all      # every frame, DONE ones too

Each frame is recorded with the hand named in the professor's own file (the
first line reads `Frame 153624 | Hand ID 1499 (right)`), unless `--hand`
forces one. Frames that already have an output file are skipped unless
`--redo` is given, so an interrupted batch can simply be run again.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD_FRAME = HERE / "record_frame.py"
_FRAME_DIR = re.compile(r"^frame_(\d+)(?:_(DONE|NA))?$")
_HAND_LINE = re.compile(r"\((left|right)\)", re.I)


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


def already_recorded(out_dir: Path, frame_id: str) -> bool:
    folder = out_dir / f"frame_{frame_id}"
    return folder.is_dir() and any(folder.glob("*.txt"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reference", type=Path, required=True,
                   help="folder holding the professor's frame_<id>[_NA|_DONE] folders")
    p.add_argument("--out-dir", type=Path, default=Path("recordings") / "leap" / "prof_frames")
    p.add_argument("--hand", choices=("left", "right"), default=None,
                   help="force one hand (default: the hand named in each frame's file)")
    p.add_argument("--only", default=None, help="comma-separated frame ids to record")
    p.add_argument("--all", action="store_true", help="every frame, not only the 21 _NA ones")
    p.add_argument("--redo", action="store_true", help="record frames that already have output")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--prep", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true", help="list what would run and stop")
    args = p.parse_args(argv)

    frames = list_frames(args.reference, include_done=args.all)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        frames = [f for f in frames if f[0] in wanted]
    if not frames:
        print(f"no frames found under {args.reference}")
        return 2

    todo = [(fid, folder) for fid, folder in frames if args.redo or not already_recorded(args.out_dir, fid)]
    print(f"{len(frames)} frame(s) listed, {len(todo)} to record, {len(frames) - len(todo)} already done")
    failed = []
    for n, (fid, folder) in enumerate(todo, 1):
        hand = args.hand or hand_in_reference(folder, fid) or "left"
        cmd = [sys.executable, str(RECORD_FRAME), fid, "--reference", str(args.reference),
               "--out-dir", str(args.out_dir), "--hand", hand,
               "--seconds", str(args.seconds), "--prep", str(args.prep)]
        print(f"\n[{n}/{len(todo)}] frame {fid}, {hand} hand" + ("  (dry run)" if args.dry_run else ""))
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
