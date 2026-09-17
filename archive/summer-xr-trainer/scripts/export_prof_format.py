"""Export glove pose recordings as keypoint text files matching the format of
the professor's tracker dataset.

Each frame is written as a block:

    Frame <counter> | Hand ID: <left|right> | Time: <ISO local time, ms>
    Wrist: (x, y, z)
    0 (Palm): (x, y, z)
    1 (THUMB): (x, y, z)
    ... through ...
    20 (Pinky): (x, y, z)

21 landmarks per hand: the palm, then four joints for each finger
(thumb, index, middle, ring, pinky), in that fixed order. Coordinates are in
millimetres. The wrist is the origin and is always (0.00, 0.00, 0.00): unlike
the tracker data, the gloves do not sense the hand's absolute position in
space, so every position is relative to the wrist rather than to an external
camera. The wrist line makes that convention explicit in each block.

By default one text file is written per pose (all that pose's frames, both
hands, as consecutive blocks). Use --per-frame to instead write one file per
frame, named like the tracker dataset (frame_<counter>_<hand>.txt).

Usage:
  python scripts/export_prof_format.py                      # recordings/poses
  python scripts/export_prof_format.py recordings/poses --out-dir somewhere
  python scripts/export_prof_format.py --per-frame
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

from xr_hand.joints import JOINT_INDEX
from xr_hand.kinematics import forward_kinematics
from xr_hand.recorder import FrameRecorder

# (index, finger label as it appears in the tracker files, OpenXR joint name).
# Landmark 0 is the palm; each finger then contributes four joints from the
# knuckle out to the tip. This mirrors the professor's 0..20 layout exactly.
LANDMARKS = [
    (0,  "Palm",   "PALM"),
    (1,  "THUMB",  "THUMB_METACARPAL"),
    (2,  "THUMB",  "THUMB_PROXIMAL"),
    (3,  "THUMB",  "THUMB_DISTAL"),
    (4,  "THUMB",  "THUMB_TIP"),
    (5,  "Index",  "INDEX_PROXIMAL"),
    (6,  "Index",  "INDEX_INTERMEDIATE"),
    (7,  "Index",  "INDEX_DISTAL"),
    (8,  "Index",  "INDEX_TIP"),
    (9,  "Middle", "MIDDLE_PROXIMAL"),
    (10, "Middle", "MIDDLE_INTERMEDIATE"),
    (11, "Middle", "MIDDLE_DISTAL"),
    (12, "Middle", "MIDDLE_TIP"),
    (13, "Ring",   "RING_PROXIMAL"),
    (14, "Ring",   "RING_INTERMEDIATE"),
    (15, "Ring",   "RING_DISTAL"),
    (16, "Ring",   "RING_TIP"),
    (17, "Pinky",  "LITTLE_PROXIMAL"),
    (18, "Pinky",  "LITTLE_INTERMEDIATE"),
    (19, "Pinky",  "LITTLE_DISTAL"),
    (20, "Pinky",  "LITTLE_TIP"),
]

MM = 1000.0  # glove world positions are in metres; the tracker files use mm


def frame_block(frame, wall_time: float) -> str:
    """One frame -> the professor-format text block (mm, wrist at origin)."""
    world = forward_kinematics(frame)
    stamp = (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall_time))
             + f".{int(wall_time * 1000) % 1000:03d}")
    lines = [f"Frame {frame.packet_counter} | Hand ID: {frame.hand_side} | Time: {stamp}"]
    wx, wy, wz = world[JOINT_INDEX["WRIST"]]
    lines.append(f"Wrist: ({wx * MM:.2f}, {wy * MM:.2f}, {wz * MM:.2f})")
    for idx, label, joint in LANDMARKS:
        x, y, z = world[JOINT_INDEX[joint]]
        lines.append(f"{idx} ({label}): ({x * MM:.2f}, {y * MM:.2f}, {z * MM:.2f})")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Export glove poses in the tracker keypoint text format.")
    p.add_argument("input", type=Path, nargs="?", default=Path("recordings") / "poses",
                   help=".jsonl file or folder of recordings (default: recordings/poses)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output folder (default: <input>/keypoints_txt)")
    p.add_argument("--per-frame", action="store_true",
                   help="write one file per frame instead of one per pose")
    p.add_argument("--hand", choices=("left", "right"), default=None,
                   help="keep only frames from this hand (e.g. when a pose "
                        "must match a specific hand in the reference dataset)")
    args = p.parse_args()

    files = sorted(args.input.rglob("*.jsonl")) if args.input.is_dir() else [args.input]
    if not files:
        raise SystemExit(f"no .jsonl recordings in {args.input}")
    out_dir = args.out_dir or ((args.input if args.input.is_dir() else args.input.parent) / "keypoints_txt")
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    by_pose = defaultdict(list)   # pose -> [block, ...]
    n_frames = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as fin:
            labels = [json.loads(l) for l in fin if l.strip()]
        for d, (frame, wall) in zip(labels, FrameRecorder.load(path)):
            if args.hand and frame.hand_side != args.hand:
                continue
            pose = d.get("pose", "") or "unlabeled"
            block = frame_block(frame, wall)
            n_frames += 1
            if args.per_frame:
                name = f"frame_{frame.packet_counter}_{frame.hand_side}.txt"
                (out_dir / name).write_text(block + "\n", encoding="utf-8")
            else:
                by_pose[pose].append(block)

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
