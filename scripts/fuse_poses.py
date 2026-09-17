"""Fuse simultaneous glove + camera takes and score all three against each other.

Reads the paired recordings written by scripts/record_simultaneous.py, matches
frames by wall-clock, fuses each pair (glove curl + camera spread — see
cam_hand/fusion.py), and then runs the SAME leave-one-out nearest-centroid
test on three datasets:

    glove only     what the current pipeline can do
    camera only    what the camera alone can do
    fused          curl from the glove, spread and thumb from the camera

That three-row table is the point: it says whether fusion actually buys
accuracy, per pose, instead of asserting that it should. Watch pinch in
particular — it is the pose the glove misses because thumb opposition is
invisible to stretch sensors.

Two kinds of camera take, told apart by the file itself and never by a flag:

  recordings/sync/cam/     MediaPipe. A `world` key per line: 21 normalised
                           landmarks, a hand shape with no size. Aligned onto
                           the glove with rotation + scale (Umeyama).
  recordings/sync/leap/    Ultraleap. `source: "leap"`, the glove's own
                           26-joint schema plus camera extras, read back with
                           FrameRecorder.load + frame_to_keypoints21 — real
                           metres. Aligned RIGIDLY, no scale, because both
                           sides are already metric (plan section 3).

A session folder may hold both; each take is read the way its own first line
says, and the report names which camera every take came from.

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

LEAP = "leap"
MEDIAPIPE = "mediapipe"
# Where record_simultaneous puts each backend's camera takes. Both are read.
CAM_DIRS = ("cam", LEAP)


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


def camera_source(path: Path) -> str:
    """Which camera wrote this take: 'leap' or 'mediapipe'.

    Read off the first line's `source` key, not off the folder name, so a
    file that was moved, renamed or handed over still says what it is.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return (LEAP if json.loads(line).get("source") == LEAP
                        else MEDIAPIPE)
    return MEDIAPIPE


def load_leap_cam(path: Path):
    """Leap JSONL -> the same row dicts load_cam returns, in metres.

    `FrameRecorder.load` reads these files unchanged — that is the whole
    point of the Leap recorder's schema — and `frame_to_keypoints21` runs the
    same forward kinematics the glove side does, so both skeletons arrive in
    the 21-keypoint layout, wrist-centred, in real metres.

    There is no confidence to carry: LeapC reports `confidence` as a constant
    1.0, so gating on it would be gating on nothing. A line in the file is a
    frame the tracker actually reported, and `score` is 1.0 to say so.
    """
    with open(path, "r", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    out = []
    for d, (frame, wall) in zip(labels, FrameRecorder.load(path)):
        out.append({
            "wall_time": wall,
            "hand_side": frame.hand_side,
            "pose": d.get("pose", ""),
            "take": d.get("take", ""),
            "score": 1.0,
            "pts": frame_to_keypoints21(frame),
        })
    return out


def load_cam(path: Path):
    """One camera take, whichever camera wrote it -> (rows, source)."""
    source = camera_source(path)
    if source == LEAP:
        return load_leap_cam(path), source
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
    return out, source


def find_camera_take(input_dir: Path, name: str):
    """The camera file paired with a glove take, in cam/ or in leap/."""
    for folder in CAM_DIRS:
        candidate = input_dir / folder / name
        if candidate.is_file():
            return candidate
    return None


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
                   help="folder holding glove/ plus cam/ and/or leap/")
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

    glove_dir = args.input / "glove"
    cam_dirs = [args.input / d for d in CAM_DIRS if (args.input / d).is_dir()]
    if not glove_dir.is_dir() or not cam_dirs:
        raise SystemExit(
            f"expected {glove_dir} and one of "
            + " or ".join(str(args.input / d) for d in CAM_DIRS) + "\n"
            "Record them with: python scripts/record_simultaneous.py\n"
            "                  python scripts/record_simultaneous.py --camera leap")

    takes = sorted(glove_dir.glob("*.jsonl"))
    if not takes:
        raise SystemExit(f"no glove recordings in {glove_dir}")

    glove_samples, cam_samples, fused_samples = [], [], []
    fused_rows = []
    n_pairs = n_matched = n_cam_used = 0
    skipped = []
    by_source = defaultdict(int)          # camera -> takes read from it

    for gpath in takes:
        cpath = find_camera_take(args.input, gpath.name)
        if cpath is None:
            skipped.append((gpath.name, "no matching camera file"))
            continue
        glove = load_glove(gpath)
        cam, source = load_cam(cpath)
        if not glove or not cam:
            skipped.append((gpath.name, "one side is empty"))
            continue
        by_source[source] += 1
        # Metric camera, rigid fit; normalised camera, fit the scale too.
        with_scale = source != LEAP

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
                fused, info = fuse_skeletons(G, None, min_score=args.min_score,
                                             with_scale=with_scale)
            else:
                n_matched += 1
                per_hand_c[hand].append(
                    all_features(np.asarray(c["pts"], float), hand_side=hand))
                fused, info = fuse_skeletons(
                    G, c["pts"], cam_score=c.get("score", 1.0),
                    min_score=args.min_score,
                    thumb_from_camera=not args.no_thumb_camera,
                    with_scale=with_scale)
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
    lines.append("Camera")
    for source, n in sorted(by_source.items()):
        if source == LEAP:
            what = ("Ultraleap Stereo IR 170 — metric 3D joints in metres, "
                    "aligned rigidly (no scale)")
        else:
            what = ("MediaPipe webcam — normalised landmarks, aligned with "
                    "rotation + scale")
        lines.append(f"  {n} take(s)   {what}")
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
