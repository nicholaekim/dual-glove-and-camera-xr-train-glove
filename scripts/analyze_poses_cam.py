"""Pose-separability report for the CAMERA dataset (REPORT.txt).

Same structure as the glove pipeline's analyze_poses.py — one feature vector
per take x hand, an extension table, then leave-one-out nearest-centroid
classification — with one addition that is the whole point of the camera:

it scores three feature sets separately.

  flexion only   the 5 wrist-to-fingertip distances. This is what the glove
                 measures, so this row is the camera's score on the glove's
                 own terms.
  spread only    adjacent fingertip gaps + thumb opposition — the degrees of
                 freedom the glove cannot sense at all.
  flexion+spread everything the camera sees.

If "flexion only" struggles on pinch and "flexion+spread" fixes it, that is
the complementary-sensing argument measured rather than asserted.

Usage:
  python scripts/analyze_poses_cam.py
  python scripts/analyze_poses_cam.py recordings/poses_cam --write
  python scripts/analyze_poses_cam.py --raw          # metres, not palm-normalized
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path

from cam_hand.export21 import iter_cam_files, wrist_centered
from cam_hand.features import (
    ALL_COLS,
    ALL_NAMES,
    FLEXION_COLS,
    FLEXION_NAMES,
    SPREAD_COLS,
    SPREAD_NAMES,
    all_features,
    loo_nearest_centroid,
    mean_vector,
)
from cam_hand.recorder import CamRecorder

CORE_POSES = {"open_palm", "fist", "index_point", "thumbs_up", "peace"}


def take_samples(path: Path, normalize: bool):
    """One recording -> {hand: (pose, mean feature vector over its frames)}."""
    per_hand = defaultdict(list)
    poses = {}
    for d in CamRecorder.load(path):
        if "world" not in d:
            continue
        hand = d["hand_side"]
        per_hand[hand].append(all_features(wrist_centered(d), normalize, hand))
        poses[hand] = d.get("pose", "")
    return {h: (poses[h], mean_vector(v)) for h, v in per_hand.items() if v}


def score_block(samples, cols, title, lines) -> None:
    ok, n, wrong = loo_nearest_centroid([(s[0], s[3]) for s in samples], cols)
    lines.append(f"  {title:<16} {ok:>3}/{n} correct ({100.0 * ok / n:.0f}%)")
    for true_lab, got, i in wrong:
        _pose, hand, fname, _f = samples[i]
        lines.append(f"      miss: {true_lab:<12} ({hand}, {fname}) -> {got}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pose separability report for camera recordings.")
    p.add_argument("input", type=Path, nargs="?",
                   default=Path("recordings") / "poses_cam")
    p.add_argument("--write", action="store_true",
                   help="also write <input>/REPORT.txt")
    p.add_argument("--raw", action="store_true",
                   help="raw metres instead of palm-length-normalized features")
    args = p.parse_args()

    files = iter_cam_files(args.input)
    if not files:
        raise SystemExit(f"no .jsonl recordings in {args.input}")
    normalize = not args.raw

    # samples: (pose, hand, filename, features)
    samples = []
    for path in files:
        for hand, (pose, feats) in sorted(take_samples(path, normalize).items()):
            samples.append((pose, hand, path.name, feats))
    if not samples:
        raise SystemExit(f"no camera frames found in {args.input}")

    unit = "palm lengths" if normalize else "cm"
    mult = 1.0 if normalize else 100.0
    lines = [f"Camera pose report — {len(samples)} samples (take x hand) "
             f"from {len(files)} recordings",
             f"Features in {unit}"
             + ("  (normalized by wrist->middle-knuckle distance)" if normalize else ""),
             ""]

    by_pose = defaultdict(list)
    for pose, _hand, _file, feats in samples:
        by_pose[pose].append(feats)

    W = 16   # column width: wide enough for "thumb-out-of-pl" plus a space
    for group, names, cols in (("FLEXION (glove-comparable)", FLEXION_NAMES, FLEXION_COLS),
                               ("SPREAD (camera-only)", SPREAD_NAMES, SPREAD_COLS)):
        lines.append(group)
        lines.append(f"{'pose':<12} {'n':>3} | "
                     + " ".join(f"{n[:W - 1]:>{W}}" for n in names))
        lines.append("-" * (18 + (W + 1) * len(names)))
        for pose in sorted(by_pose):
            rows = by_pose[pose]
            n = len(rows)
            cells = []
            for k in cols:
                vals = [r[k] * mult for r in rows]
                mean = sum(vals) / n
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
                cells.append(f"{mean:>9.2f}+/-{std:<5.2f}")
            lines.append(f"{pose:<12} {n:>3} | " + " ".join(cells))
        lines.append("")

    lines.append("Leave-one-out nearest-centroid classification")
    score_block(samples, FLEXION_COLS, "flexion only", lines)
    score_block(samples, SPREAD_COLS, "spread only", lines)
    score_block(samples, ALL_COLS, "flexion+spread", lines)

    core = [s for s in samples if s[0] in CORE_POSES]
    if core and len(core) < len(samples):
        lines.append("")
        lines.append(f"Core 5 poses only ({', '.join(sorted(CORE_POSES))})")
        score_block(core, FLEXION_COLS, "flexion only", lines)
        score_block(core, ALL_COLS, "flexion+spread", lines)

    lines.append("")
    lines.append("Note: all three rows are scored on CAMERA data. 'flexion only'")
    lines.append("is the camera's own flexion features, not the glove's score —")
    lines.append("abduction shifts a fingertip slightly even at constant curl, so")
    lines.append("the camera's flexion numbers still carry a little spread")
    lines.append("information that a stretch sensor would not have. For the real")
    lines.append("glove-vs-camera-vs-fused comparison use scripts/fuse_poses.py on")
    lines.append("a simultaneous recording.")
    lines.append("")
    lines.append(f"Feature order: {', '.join(ALL_NAMES)}")

    report = "\n".join(lines)
    print(report)
    if args.write:
        base = args.input if args.input.is_dir() else args.input.parent
        out = base / "REPORT.txt"
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
