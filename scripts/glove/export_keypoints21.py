"""Export pose recordings as 21-keypoint hand data (MediaPipe/Ultralytics layout).

Converts glove recordings (26 OpenXR joints, local transforms) into the
21-keypoint format used by the image hand-pose datasets: world XYZ per
keypoint, wrist at the origin, metres. Writes two CSVs:

  keypoints21_frames.csv    one row per frame per hand (the full data)
  keypoints21_summary.csv   one row per pose x hand: the medoid frame —
                            the actual recorded frame closest to the pose
                            average. A real frame (not a mean of positions)
                            so bone lengths stay exactly rigid; averaging
                            positions over a wobbly take shrinks segments.
  <pose>_keypoints21.csv    the same frame rows split into one CSV per pose.

Usage:
  python scripts/export_keypoints21.py recordings/poses
  python scripts/export_keypoints21.py recordings/poses --out-dir recordings/poses
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from xr_hand.keypoints21 import MP21_NAMES, frame_to_keypoints21
from xr_hand.recorder import FrameRecorder

COORD_COLS = [f"{name}_{axis}" for name in MP21_NAMES for axis in ("x", "y", "z")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path,
                   help=".jsonl recording or a folder of them")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write the CSVs (default: the input folder)")
    args = p.parse_args()

    # rglob: recordings may be organized into per-pose subfolders
    files = sorted(args.input.rglob("*.jsonl")) if args.input.is_dir() else [args.input]
    if not files:
        raise SystemExit(f"no .jsonl recordings in {args.input}")
    out_dir = args.out_dir or (args.input if args.input.is_dir() else args.input.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_csv = out_dir / "keypoints21_frames.csv"
    summary_csv = out_dir / "keypoints21_summary.csv"

    # all_coords[(pose, hand)] -> list of per-frame coord vectors
    all_coords = defaultdict(list)
    # pose_rows[pose] -> written frame rows, for the per-pose CSVs
    pose_rows = defaultdict(list)
    n_rows = 0

    with open(frames_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pose", "take", "hand", "wall_time"] + COORD_COLS)
        for path in files:
            # pose/take live in each JSONL line; read alongside the frames
            with open(path, "r", encoding="utf-8") as fin:
                raw_lines = [json.loads(l) for l in fin if l.strip()]
            for d, (frame, wall) in zip(raw_lines, FrameRecorder.load(path)):
                pose = d.get("pose", "")
                take = d.get("take", "")
                pts = frame_to_keypoints21(frame)
                coords = [c for pt in pts for c in pt]
                row = ([pose, take, frame.hand_side, wall]
                       + [f"{c:.6f}" for c in coords])
                writer.writerow(row)
                all_coords[(pose, frame.hand_side)].append(coords)
                pose_rows[pose].append(row)
                n_rows += 1

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pose", "hand", "n_frames"] + COORD_COLS)
        for (pose, hand), rows in sorted(all_coords.items()):
            n = len(rows)
            mean = [sum(r[i] for r in rows) / n for i in range(len(COORD_COLS))]
            medoid = min(rows, key=lambda r: sum((a - b) ** 2 for a, b in zip(r, mean)))
            writer.writerow([pose, hand, n] + [f"{c:.6f}" for c in medoid])

    n_pose_csvs = 0
    for pose, rows in sorted(pose_rows.items()):
        if not pose:
            continue  # unlabeled recordings only appear in the combined CSV
        with open(out_dir / f"{pose}_keypoints21.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pose", "take", "hand", "wall_time"] + COORD_COLS)
            writer.writerows(rows)
        n_pose_csvs += 1

    print(f"wrote {frames_csv}  ({n_rows} rows from {len(files)} recordings)")
    print(f"wrote {summary_csv}  ({len(all_coords)} pose x hand skeletons)")
    print(f"wrote {n_pose_csvs} per-pose CSVs (<pose>_keypoints21.csv)")
    print("Units: metres, wrist-centred (WRIST = 0,0,0), "
          "21-keypoint MediaPipe/Ultralytics order.")


if __name__ == "__main__":
    main()
