"""Pose-separability analysis of the recorded glove dataset (REPORT.txt).

Reads the labeled pose recordings, reduces every take to a 5-number feature
vector — the wrist-to-fingertip distance ("fingertip extension") of each
finger, averaged over the take's frames — then:

  1. prints the per-pose fingertip-extension table (mean +/- spread, cm),
     which shows *why* poses separate: each pose is a distinct pattern of
     which fingers are extended vs curled, exactly what the stretch
     sensors measure;
  2. runs leave-one-out nearest-centroid classification: each take is
     held out, pose centroids are rebuilt from the remaining takes, and
     the held-out take is assigned to the nearest centroid (Euclidean,
     5-D). No training, no hyperparameters — if this simple geometric
     test scores high, the poses are genuinely separable in glove space.

Usage:
  python scripts/analyze_poses.py                     # analyze recordings/poses
  python scripts/analyze_poses.py recordings/poses --write   # also update REPORT.txt
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path

from xr_hand.keypoints21 import MP21_NAMES, frame_to_keypoints21
from xr_hand.recorder import FrameRecorder

FINGERS = ["thumb", "index", "middle", "ring", "little"]
TIP_IDX = [MP21_NAMES.index(n) for n in
           ("THUMB_TIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_TIP", "PINKY_TIP")]

# The 5 poses the gloves sense reliably (pinch/three probe the limits).
CORE_POSES = {"open_palm", "fist", "index_point", "thumbs_up", "peace"}


def take_features(path: Path) -> dict:
    """One recording -> {hand: (pose, [5 mean fingertip extensions, metres])}."""
    sums = defaultdict(lambda: [0.0] * len(TIP_IDX))
    counts = defaultdict(int)
    poses = {}
    import json
    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    for d, (frame, _wall) in zip(raw, FrameRecorder.load(path)):
        pts = frame_to_keypoints21(frame)  # wrist-centred -> |TIP| = extension
        hand = frame.hand_side
        for k, ti in enumerate(TIP_IDX):
            x, y, z = pts[ti]
            sums[hand][k] += math.sqrt(x * x + y * y + z * z)
        counts[hand] += 1
        poses[hand] = d.get("pose", "")
    return {h: (poses[h], [s / counts[h] for s in sums[h]]) for h in sums}


def main() -> None:
    p = argparse.ArgumentParser(description="Pose separability report for glove recordings.")
    p.add_argument("input", type=Path, nargs="?", default=Path("recordings") / "poses",
                   help=".jsonl file or folder of recordings (default: recordings/poses)")
    p.add_argument("--write", action="store_true",
                   help="also write the report to <input>/REPORT.txt")
    args = p.parse_args()

    files = sorted(args.input.rglob("*.jsonl")) if args.input.is_dir() else [args.input]
    if not files:
        raise SystemExit(f"no .jsonl recordings in {args.input}")

    # samples: (pose, hand, filename, [5 extensions, metres]) — one per take x hand
    samples = []
    for path in files:
        for hand, (pose, feats) in sorted(take_features(path).items()):
            samples.append((pose, hand, path.name, feats))

    lines = []

    # --- fingertip-extension table --------------------------------------
    by_pose = defaultdict(list)
    for pose, _hand, _file, feats in samples:
        by_pose[pose].append(feats)
    header = (f"{'pose':<12} {'n':>3} | "
              + "".join(f"{f:>12}" for f in FINGERS)
              + "   (mean +/- spread, cm)")
    lines.append(header)
    lines.append("-" * 76)
    for pose in sorted(by_pose):
        rows = by_pose[pose]
        n = len(rows)
        cells = []
        for k in range(len(FINGERS)):
            vals = [r[k] * 100.0 for r in rows]  # metres -> cm
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
            cells.append(f"{mean:>7.1f}+/-{std:<4.1f}")
        lines.append(f"{pose:<12} {n:>3} | " + " ".join(cells))

    # --- leave-one-out nearest-centroid ----------------------------------
    def loo_accuracy(subset):
        wrong = []
        for i, (pose, hand, fname, feats) in enumerate(subset):
            centroids = defaultdict(lambda: [0.0] * len(FINGERS))
            counts = defaultdict(int)
            for j, (p2, _h2, _f2, f2) in enumerate(subset):
                if j == i:
                    continue  # hold this take out
                for k in range(len(FINGERS)):
                    centroids[p2][k] += f2[k]
                counts[p2] += 1
            best, best_d = None, float("inf")
            for p2, c in centroids.items():
                c = [v / counts[p2] for v in c]
                d = sum((a - b) ** 2 for a, b in zip(feats, c))
                if d < best_d:
                    best, best_d = p2, d
            if best != pose:
                wrong.append((pose, hand, fname, best))
        return wrong

    wrong = loo_accuracy(samples)
    n = len(samples)
    ok = n - len(wrong)
    lines.append("")
    lines.append(f"Leave-one-out nearest-centroid classification: "
                 f"{ok}/{n} samples correct ({100.0 * ok / n:.0f}%)")
    if wrong:
        lines.append("Misclassified:")
        for pose, hand, fname, got in wrong:
            lines.append(f"  {pose:<12} ({hand}, {fname}) classified as {got}")

    core = [s for s in samples if s[0] in CORE_POSES]
    core_wrong = loo_accuracy(core)
    lines.append(f"Core 5 poses only ({', '.join(sorted(CORE_POSES))}): "
                 f"{len(core) - len(core_wrong)}/{len(core)} correct")

    report = "\n".join(lines)
    print(report)
    if args.write:
        out = (args.input if args.input.is_dir() else args.input.parent) / "REPORT.txt"
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
