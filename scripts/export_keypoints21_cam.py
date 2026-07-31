"""Export camera recordings as 21-keypoint CSVs — same layout as the glove's
export_keypoints21.py, so glove and camera files can be diffed directly.

  keypoints21_frames.csv    one row per frame per hand
  keypoints21_summary.csv   one row per pose x hand: the medoid frame (the
                            real recorded frame closest to the pose average).
                            A real frame, not a coordinate-wise mean, so bone
                            lengths stay rigid — same reasoning as the glove
                            exporter.
  <pose>_keypoints21.csv    the frame rows split per pose.

Units: metres, wrist-centred (WRIST = 0,0,0), MediaPipe 21-keypoint order.
Camera world coordinates are model-estimated for an average-sized hand, so
treat them as approximate metric — see README.

Usage:
  python scripts/export_keypoints21_cam.py                       # recordings/poses_cam
  python scripts/export_keypoints21_cam.py <folder-or-file> --out-dir <dir>
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from cam_hand.export21 import iter_cam_files, iter_records, wrist_centered
from cam_hand.landmarks import MP21_NAMES

COORD_COLS = [f"{name}_{axis}" for name in MP21_NAMES for axis in ("x", "y", "z")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, nargs="?",
                   default=Path("recordings") / "poses_cam",
                   help="camera .jsonl or folder of them")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    files = iter_cam_files(args.input)
    if not files:
        raise SystemExit(f"no .jsonl recordings in {args.input}")
    out_dir = args.out_dir or (args.input if args.input.is_dir() else args.input.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_csv = out_dir / "keypoints21_frames.csv"
    summary_csv = out_dir / "keypoints21_summary.csv"

    all_coords = defaultdict(list)   # (pose, hand) -> [coord vectors]
    pose_rows = defaultdict(list)
    n_rows = 0

    with open(frames_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pose", "take", "hand", "wall_time", "score"] + COORD_COLS)
        for d in iter_records(files):
            pose = d.get("pose", "")
            take = d.get("take", "")
            coords = [c for pt in wrist_centered(d) for c in pt]
            row = ([pose, take, d["hand_side"], d["wall_time"], d.get("score", "")]
                   + [f"{c:.6f}" for c in coords])
            writer.writerow(row)
            all_coords[(pose, d["hand_side"])].append(coords)
            pose_rows[pose].append(row)
            n_rows += 1

    if not n_rows:
        raise SystemExit(f"no camera frames found in {args.input}")

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
            continue
        with open(out_dir / f"{pose}_keypoints21.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pose", "take", "hand", "wall_time", "score"] + COORD_COLS)
            writer.writerows(rows)
        n_pose_csvs += 1

    print(f"wrote {frames_csv}  ({n_rows} rows from {len(files)} recordings)")
    print(f"wrote {summary_csv}  ({len(all_coords)} pose x hand skeletons)")
    print(f"wrote {n_pose_csvs} per-pose CSVs (<pose>_keypoints21.csv)")
    print("Units: metres, wrist-centred (WRIST = 0,0,0), "
          "21-keypoint MediaPipe order (camera world coords are approximate).")


if __name__ == "__main__":
    main()
