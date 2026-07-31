"""Fuse simultaneous glove + camera takes and score all three against each other.

Reads the paired recordings written by scripts/record_simultaneous.py, matches
frames by wall-clock, fuses each pair (glove curl + camera spread — see
cam_hand/fusion.py), and then runs the SAME leave-one-out nearest-centroid
test on three datasets:

    glove only     what the current pipeline can do
    camera only    what a webcam alone can do
    fused          curl from the glove, spread and thumb from the camera

That three-row table is the point: it says whether fusion actually buys
accuracy, per pose, instead of asserting that it should. Watch pinch in
particular — it is the pose the glove misses because thumb opposition is
invisible to stretch sensors.

Also reports the plumbing that has to be right for any of it to mean anything:
how many frames found a partner within the time window, and how often the
camera was confident enough to contribute.

Usage:
  python scripts/fuse_poses.py                          # recordings/sync
  python scripts/fuse_poses.py recordings/sync --write  # also REPORT.txt
  python scripts/fuse_poses.py --export-csv fused.csv   # fused 21-kp CSV
  python scripts/fuse_poses.py --max-dt 0.1             # looser time matching
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cam_hand.export21 import wrist_centered
from cam_hand.features import (
    ALL_COLS,
    ALL_NAMES,
    FLEXION_COLS,
    all_features,
    loo_nearest_centroid,
    mean_vector,
)
from cam_hand.fusion import fuse_skeletons, pair_by_time
from cam_hand.landmarks import MP21_NAMES
from cam_hand.recorder import CamRecorder

from xr_hand.keypoints21 import frame_to_keypoints21
from xr_hand.recorder import FrameRecorder

COORD_COLS = [f"{n}_{a}" for n in MP21_NAMES for a in ("x", "y", "z")]


def load_glove(path: Path):
    """Glove JSONL -> dicts with wall_time, hand_side, pose, take, pts (21x3 m)."""
    with open(path, "r", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    out = []
    for d, (frame, wall) in zip(labels, FrameRecorder.load(path)):
        out.append({
            "wall_time": wall,
            "hand_side": frame.hand_side,
            "pose": d.get("pose", ""),
            "take": d.get("take", ""),
            "pts": frame_to_keypoints21(frame),
        })
    return out


def load_cam(path: Path):
    out = []
    for d in CamRecorder.load(path):
        if "world" not in d:
            continue
        out.append({
            "wall_time": d["wall_time"],
            "hand_side": d["hand_side"],
            "pose": d.get("pose", ""),
            "take": d.get("take", ""),
            "score": d.get("score", 1.0),
            "pts": wrist_centered(d),
        })
    return out


def loo_table(samples, cols, label, lines):
    """samples: [(pose, hand, file, features)] -> print a scored row."""
    if not samples:
        lines.append(f"  {label:<14} (no samples)")
        return None
    ok, n, wrong = loo_nearest_centroid([(s[0], s[3]) for s in samples], cols)
    lines.append(f"  {label:<14} {ok:>3}/{n} correct ({100.0 * ok / n:3.0f}%)")
    for true_lab, got, i in wrong:
        _p, hand, fname, _f = samples[i]
        lines.append(f"       miss: {true_lab:<12} ({hand}, {fname}) -> {got}")
    return {p for p, *_ in (samples[i] for _, _, i in wrong)}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fuse simultaneous glove+camera takes and compare all three.")
    p.add_argument("input", type=Path, nargs="?", default=Path("recordings") / "sync",
                   help="folder holding glove/ and cam/ subfolders")
    p.add_argument("--max-dt", type=float, default=0.05,
                   help="max seconds between paired glove/camera frames")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="camera confidence below which the glove is kept as-is")
    p.add_argument("--no-thumb-camera", action="store_true",
                   help="do not take the thumb direction from the camera")
    p.add_argument("--export-csv", type=Path, default=None,
                   help="write the fused frames as a 21-keypoint CSV")
    p.add_argument("--write", action="store_true", help="write <input>/REPORT.txt")
    args = p.parse_args()

    glove_dir, cam_dir = args.input / "glove", args.input / "cam"
    if not glove_dir.is_dir() or not cam_dir.is_dir():
        raise SystemExit(
            f"expected {glove_dir} and {cam_dir}\n"
            "Record them with: python scripts/record_simultaneous.py")

    takes = sorted(glove_dir.glob("*.jsonl"))
    if not takes:
        raise SystemExit(f"no glove recordings in {glove_dir}")

    glove_samples, cam_samples, fused_samples = [], [], []
    fused_rows = []
    n_pairs = n_matched = n_cam_used = 0
    skipped = []

    for gpath in takes:
        cpath = cam_dir / gpath.name
        if not cpath.is_file():
            skipped.append((gpath.name, "no matching camera file"))
            continue
        glove, cam = load_glove(gpath), load_cam(cpath)
        if not glove or not cam:
            skipped.append((gpath.name, "one side is empty"))
            continue

        per_hand_g = defaultdict(list)
        per_hand_c = defaultdict(list)
        per_hand_f = defaultdict(list)
        pose = glove[0]["pose"]

        for g, c in pair_by_time(glove, cam, max_dt=args.max_dt):
            n_pairs += 1
            hand = g["hand_side"]
            G = np.asarray(g["pts"], dtype=float)
            per_hand_g[hand].append(all_features(G, hand_side=hand))
            if c is None:
                fused, info = fuse_skeletons(G, None, min_score=args.min_score)
            else:
                n_matched += 1
                per_hand_c[hand].append(
                    all_features(np.asarray(c["pts"], float), hand_side=hand))
                fused, info = fuse_skeletons(
                    G, c["pts"], cam_score=c.get("score", 1.0),
                    min_score=args.min_score,
                    thumb_from_camera=not args.no_thumb_camera)
            if info["camera_used"]:
                n_cam_used += 1
            per_hand_f[hand].append(all_features(fused, hand_side=hand))
            if args.export_csv is not None:
                fused_rows.append([pose, g["take"], hand, g["wall_time"],
                                   int(info["camera_used"])]
                                  + [f"{v:.6f}" for v in np.asarray(fused).ravel()])

        for hand, feats in per_hand_g.items():
            glove_samples.append((pose, hand, gpath.name, mean_vector(feats)))
        for hand, feats in per_hand_c.items():
            cam_samples.append((pose, hand, gpath.name, mean_vector(feats)))
        for hand, feats in per_hand_f.items():
            fused_samples.append((pose, hand, gpath.name, mean_vector(feats)))

    if not glove_samples:
        raise SystemExit("no usable paired takes found")

    lines = []
    lines.append("=" * 66)
    lines.append(f"Sensor fusion report — {len(takes)} takes from {args.input}")
    lines.append("")
    lines.append("Pairing (glove frames matched to a camera frame by wall clock)")
    lines.append(f"  glove frames            {n_pairs}")
    lines.append(f"  matched within {args.max_dt * 1000:.0f} ms   {n_matched} "
                 f"({100.0 * n_matched / max(n_pairs, 1):.0f}%)")
    lines.append(f"  camera actually used    {n_cam_used} "
                 f"({100.0 * n_cam_used / max(n_pairs, 1):.0f}%)  "
                 f"[score >= {args.min_score}]")
    if skipped:
        lines.append(f"  skipped takes           {len(skipped)}")
        for name, why in skipped:
            lines.append(f"    {name}: {why}")
    lines.append("")
    lines.append("Leave-one-out nearest-centroid, same test for all three")
    loo_table(glove_samples, FLEXION_COLS, "glove only", lines)
    loo_table(cam_samples, ALL_COLS, "camera only", lines)
    loo_table(fused_samples, ALL_COLS, "fused", lines)
    lines.append("")
    lines.append("  glove only uses the 5 flexion features (all it can measure);")
    lines.append("  camera only and fused use flexion + spread.")
    lines.append(f"  feature order: {', '.join(ALL_NAMES)}")

    report = "\n".join(lines)
    print(report)

    if args.export_csv is not None:
        args.export_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pose", "take", "hand", "wall_time", "camera_used"] + COORD_COLS)
            w.writerows(fused_rows)
        print(f"\nwrote {args.export_csv}  ({len(fused_rows)} fused frames)")

    if args.write:
        out = args.input / "REPORT.txt"
        out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
