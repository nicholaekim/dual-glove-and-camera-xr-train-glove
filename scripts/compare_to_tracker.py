"""Score the camera pipeline against the professor's tracker, frame by frame.

His dataset gives both the photo and his tracker's 21 keypoints for the same
instant, which makes it a ready-made reference: run MediaPipe on the photo,
align the two skeletons, and measure how far apart they are.

What is measured
  detection rate   fraction of his frames where MediaPipe finds a hand at all
  shape error      after a rotation+scale alignment (Umeyama) the two point
                   sets are in one frame; RMS distance per landmark is then a
                   fair shape comparison. Scale is solved rather than assumed
                   because a single camera cannot recover absolute size — the
                   solved factor is reported, and its spread across frames is
                   itself a result.
  chirality        every frame is also fitted with the camera skeleton
                   mirrored. Rotation alone cannot turn a left hand into a
                   right one, so if the mirrored fit is clearly better the two
                   devices reconstructed opposite hands — a real disagreement,
                   independent of what either one *labelled* the hand. Both
                   are reported: label agreement and shape agreement.
  per-landmark     which landmarks disagree most (fingertips and the thumb
                   are where occlusion and depth guessing hurt).

Alignment is solved on the near-rigid palm landmarks by default (--fit palm:
wrist, knuckles) and the error reported over all 21, so finger differences
are measured rather than absorbed into the fit. --fit all solves on every
landmark, which flatters both sides equally.

Usage:
  python scripts/compare_to_tracker.py "..\\xr trainer\\xr trainer poses"
  python scripts/compare_to_tracker.py <folder> --csv out.csv --fit all
  python scripts/compare_to_tracker.py <folder> --annotate out_dir   # save overlays
"""
import argparse
import csv
import statistics
from pathlib import Path

import numpy as np

from cam_hand.align import align_points
from cam_hand.landmarks import DEFAULT_MODEL, HandTracker
from cam_hand.prof_format import find_pairs, load_file

# His landmark 0 is the palm; MediaPipe has none, so it is synthesized as the
# wrist / middle-knuckle midpoint. Indices 1..20 line up one-to-one.
PALM_FIT_IDX = [0, 1, 5, 9, 13, 17]   # palm + finger bases: near-rigid
LANDMARK_LABELS = (
    ["Palm"] + [f"THUMB{i}" for i in range(1, 5)]
    + [f"Index{i}" for i in range(1, 5)] + [f"Middle{i}" for i in range(1, 5)]
    + [f"Ring{i}" for i in range(1, 5)] + [f"Pinky{i}" for i in range(1, 5)]
)


def mp_points_mm(hand) -> np.ndarray:
    """CamHand -> 21 points in his layout (palm first), millimetres."""
    w = np.asarray(hand.world, dtype=float) * 1000.0
    palm = (w[0] + w[9]) / 2.0
    return np.vstack([palm, w[1:]])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare MediaPipe camera keypoints to the tracker dataset.")
    p.add_argument("dataset", type=Path,
                   help="folder of frame_<n>/ subfolders with .png + .txt")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--fit", choices=("palm", "all"), default="palm",
                   help="landmarks used to solve the alignment (default: palm)")
    p.add_argument("--no-scale", action="store_true",
                   help="rotation only; do not solve a scale factor")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="skip detections below this handedness confidence")
    p.add_argument("--csv", type=Path, default=None, help="write per-frame results")
    p.add_argument("--annotate", type=Path, default=None,
                   help="folder for annotated overlay images (slow)")
    p.add_argument("--limit", type=int, default=0, help="only the first N frames")
    args = p.parse_args()

    pairs = find_pairs(args.dataset)
    if not pairs:
        raise SystemExit(f"no image+txt pairs under {args.dataset}")
    if args.limit:
        pairs = pairs[: args.limit]

    tracker = HandTracker(model_path=args.model, running_mode="image", num_hands=2)
    fit_idx = PALM_FIT_IDX if args.fit == "palm" else list(range(21))

    rows = []
    per_landmark = [[] for _ in range(21)]
    misses = []
    hand_mismatch = 0

    import cv2
    for img_path, txt_path in pairs:
        ref_frames = load_file(txt_path)
        if not ref_frames:
            continue
        ref = ref_frames[0]
        img = cv2.imread(str(img_path))
        if img is None:
            misses.append((img_path.name, "unreadable image"))
            continue
        hands = sorted([h for h in tracker.detect(img) if h.score >= args.min_score],
                       key=lambda h: -h.score)
        if not hands:
            misses.append((img_path.name, "no hand detected"))
            continue
        # prefer the hand whose side matches his label; else the most confident
        same = [h for h in hands if h.hand_side == ref.hand]
        hand = same[0] if same else hands[0]
        if not same:
            hand_mismatch += 1

        src = mp_points_mm(hand)
        dst = np.asarray(ref.points, dtype=float)
        _aligned, rmse, err, scale = align_points(
            src, dst, with_scale=not args.no_scale, subset=fit_idx)

        # Mirrored refit: rotation cannot undo a reflection, so a clearly
        # better mirrored fit means the two devices disagree on chirality.
        mirrored = src.copy()
        mirrored[:, 0] *= -1.0
        _m, rmse_mir, _e, _s = align_points(
            mirrored, dst, with_scale=not args.no_scale, subset=fit_idx)
        chirality_ok = rmse <= rmse_mir

        for i, e in enumerate(err):
            per_landmark[i].append(float(e))
        rows.append({
            "frame": ref.frame,
            "file": img_path.name,
            "ref_hand": ref.hand,
            "mp_hand": hand.hand_side,
            "score": round(hand.score, 3),
            "rmse_mm": round(rmse, 2),
            "rmse_mirrored_mm": round(rmse_mir, 2),
            "chirality_ok": int(chirality_ok),
            "max_err_mm": round(float(err.max()), 2),
            "scale": round(scale, 4),
        })

        if args.annotate:
            from cam_hand.draw import draw_hand
            args.annotate.mkdir(parents=True, exist_ok=True)
            draw_hand(img, hand)
            cv2.putText(img, f"rmse {rmse:.1f} mm", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"rmse {rmse:.1f} mm", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 1, cv2.LINE_AA)
            cv2.imwrite(str(args.annotate / img_path.name), img)

    tracker.close()

    n_total = len(pairs)
    n_ok = len(rows)
    print("=" * 68)
    print(f"Camera vs tracker — {n_total} reference frames")
    print(f"  hand detected in      {n_ok}/{n_total} ({100.0 * n_ok / n_total:.0f}%)")
    if not rows:
        raise SystemExit("no frames could be compared")
    n_label_ok = n_ok - hand_mismatch
    print(f"  left/right label      agrees on {n_label_ok}/{n_ok} "
          f"({100.0 * n_label_ok / n_ok:.0f}%)")
    n_chir = sum(r["chirality_ok"] for r in rows)
    print(f"  shape chirality       agrees on {n_chir}/{n_ok} "
          f"({100.0 * n_chir / n_ok:.0f}%)  "
          f"[mirrored refit is worse = same hand reconstructed]")

    rmses = [r["rmse_mm"] for r in rows]
    scales = [r["scale"] for r in rows]
    print(f"  alignment             {args.fit} landmarks, "
          f"{'rotation only' if args.no_scale else 'rotation + scale'}")
    print(f"  shape RMSE (mm)       median {statistics.median(rmses):.1f}   "
          f"mean {statistics.mean(rmses):.1f}   "
          f"min {min(rmses):.1f}   max {max(rmses):.1f}")
    agree = [r["rmse_mm"] for r in rows if r["chirality_ok"]]
    if agree and len(agree) != len(rows):
        print(f"  ... chirality-agreeing frames only: median "
              f"{statistics.median(agree):.1f} mm over {len(agree)} frames")
    if not args.no_scale:
        print(f"  solved scale factor   median {statistics.median(scales):.3f}   "
              f"spread {min(scales):.3f}-{max(scales):.3f}")
        print("    (camera size is model-estimated; a consistent factor means "
              "consistent shape,\n     a varying one means the depth guess moves "
              "frame to frame)")

    print("\n  per-landmark error (mm, median over frames)")
    order = sorted(range(21), key=lambda i: -statistics.median(per_landmark[i] or [0]))
    for i in order:
        vals = per_landmark[i]
        if not vals:
            continue
        bar = "#" * int(min(40, statistics.median(vals) / 2))
        print(f"    {LANDMARK_LABELS[i]:<9} {statistics.median(vals):6.1f}  {bar}")

    if misses:
        print(f"\n  {len(misses)} frame(s) with no comparable detection:")
        for name, why in misses[:10]:
            print(f"    {name}: {why}")
        if len(misses) > 10:
            print(f"    ... and {len(misses) - 10} more")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    if args.annotate:
        print(f"annotated overlays -> {args.annotate}")


if __name__ == "__main__":
    main()
