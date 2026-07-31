"""Export camera recordings in the professor's tracker keypoint text format.

Each frame becomes a block:

    Frame <n> | Hand ID: <left|right> | Time: <ISO local time, ms>
    Wrist: (0.00, 0.00, 0.00)
    0 (Palm): (x, y, z)
    1 (THUMB): (x, y, z)
    ... through ...
    20 (Pinky): (x, y, z)

Millimetres, wrist at the origin. Two differences from his tracker files,
both stated here so they are never mistaken for data:

  * The wrist is the origin. A single webcam gives no absolute position in
    room space, so every coordinate is relative to the wrist — the same
    convention the glove exporter uses.
  * MediaPipe has no palm landmark, so landmark 0 (Palm) is synthesized as
    the midpoint of the wrist and the middle-finger knuckle.

By default one file per pose (all frames as consecutive blocks); --per-frame
writes one file per frame named like his dataset.

  python scripts/export_prof_format_cam.py
  python scripts/export_prof_format_cam.py recordings/poses_cam --per-frame
"""
import argparse
from collections import defaultdict
from pathlib import Path

from cam_hand.export21 import iter_cam_files, iter_records, prof_block


def main() -> None:
    p = argparse.ArgumentParser(
        description="Camera recordings -> tracker-format keypoint text files.")
    p.add_argument("input", type=Path, nargs="?",
                   default=Path("recordings") / "poses_cam")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output folder (default: <input>/keypoints_txt)")
    p.add_argument("--per-frame", action="store_true",
                   help="one file per frame instead of one per pose")
    p.add_argument("--hand", choices=("left", "right"), default=None,
                   help="keep only frames from this hand")
    args = p.parse_args()

    files = iter_cam_files(args.input)
    if not files:
        raise SystemExit(f"no .jsonl recordings in {args.input}")
    base = args.input if args.input.is_dir() else args.input.parent
    out_dir = args.out_dir or (base / "keypoints_txt")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pose = defaultdict(list)
    n_frames = 0
    for d in iter_records(files):
        if args.hand and d["hand_side"] != args.hand:
            continue
        n_frames += 1
        block = prof_block(d, n_frames)
        if args.per_frame:
            (out_dir / f"frame_{n_frames}_{d['hand_side']}.txt").write_text(
                block + "\n", encoding="utf-8")
        else:
            by_pose[d.get("pose", "") or "unlabeled"].append(block)

    if not n_frames:
        raise SystemExit(f"no camera frames found in {args.input}")
    if args.per_frame:
        print(f"wrote {n_frames} per-frame files to {out_dir}")
        return

    for pose, blocks in sorted(by_pose.items()):
        out = out_dir / f"{pose}_keypoints.txt"
        out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        print(f"wrote {out.name}  ({len(blocks)} frames)")
    print(f"\n{n_frames} frames across {len(by_pose)} poses -> {out_dir}")


if __name__ == "__main__":
    main()
