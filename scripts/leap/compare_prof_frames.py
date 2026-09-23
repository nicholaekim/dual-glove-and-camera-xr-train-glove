"""Compare our recorded keypoint files with the professor's tracker, frame by frame.

`record_frame.py` writes each replicated frame as
`<out>/frame_<id>_keypoints.txt`, in the professor's own 21-point format; his
dataset holds the same frame as `frame_<id>[_NA|_DONE]/frame_<id>.txt`. Both
are read with `cam_hand.prof_format.load_file`, and for every frame that has
both, the two skeletons are aligned and compared. No camera and no MediaPipe:
this is file against file.

What is measured
  shape       our skeleton is aligned onto his with rotation, translation and
              scale (Umeyama), solved on the palm point and the four knuckles
              (landmarks 0, 5, 9, 13, 17), the near-rigid part of a hand. The
              RMS is then taken over all 21 landmarks, so finger differences
              are measured rather than absorbed into the fit. It is in his
              units (see scale).
  chirality   every frame is fitted a second time with our skeleton mirrored
              (x negated), as `scripts/compare_to_tracker.py` does. Rotation
              cannot turn a left hand into a right one, so when the mirrored
              fit is clearly better (its RMS under 0.8 of the direct one) the
              two skeletons are opposite hands, whatever either tracker called
              them. The labels are shown beside it, but they are the trackers'
              opinions and unreliable for these poses: his tracker calls the
              same physical arm "right" in frame 153624 and "left" in 156023.
              One limit: the five fit points lie close to one plane, and the
              mirror image of points in a plane is also a rotation of them. So
              only what sticks out of the palm plane (finger flexion, thumb
              opposition) tells the two fits apart, and for a flat open hand
              the ratio sits near 1 and says nothing.
  scale       his units per our millimetre, from the better of the two fits.
              His files are nominally millimetres and ours are Ultraleap
              millimetres, so the factor mixes his units with the difference
              in size between his subject's hand and our operator's. It is a
              result, not a nuisance: a steady factor across frames means both
              trackers see a hand of consistent size.
  worst       the three landmarks furthest apart after the better fit.

Usage:
  python scripts/leap/compare_prof_frames.py recordings/leap/prof_frames --reference "..\\xr trainer\\xr trainer poses"
  python scripts/leap/compare_prof_frames.py recordings/leap/prof_frames --csv results/prof_frames.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import numpy as np

from cam_hand.align import align_points
from cam_hand.prof_format import load_file

DEFAULT_OURS = Path("recordings") / "leap" / "prof_frames"
DEFAULT_REFERENCE = Path("reference") / "frames"
PALM_FIT_IDX = [0, 5, 9, 13, 17]      # palm point + the four knuckles: near-rigid
CLEARLY_MIRRORED = 0.8                # mirrored RMS / direct RMS below this
FAR_ABOVE_MEDIAN = 2.0                # better-fit RMS over this many medians
LANDMARK_LABELS = (
    ["Palm"] + [f"THUMB{i}" for i in range(1, 5)]
    + [f"Index{i}" for i in range(1, 5)] + [f"Middle{i}" for i in range(1, 5)]
    + [f"Ring{i}" for i in range(1, 5)] + [f"Pinky{i}" for i in range(1, 5)]
)
_OURS = re.compile(r"^frame_(\d+)_keypoints\.txt$")
_THEIRS_DIR = re.compile(r"^frame_(\d+)(?:_(?:DONE|NA))?$")


def our_files(folder: Path) -> dict[str, Path]:
    """frame id -> our keypoint file."""
    out = {}
    for p in sorted(folder.glob("frame_*_keypoints.txt")):
        m = _OURS.match(p.name)
        if m:
            out[m.group(1)] = p
    return out


def reference_files(reference: Path) -> dict[str, Path]:
    """frame id -> his keypoint file, in `frame_<id>[_NA|_DONE]/` folders or flat."""
    out = {}
    for d in sorted(reference.iterdir()):
        m = _THEIRS_DIR.match(d.name)
        if m and d.is_dir() and (d / f"frame_{m.group(1)}.txt").is_file():
            out[m.group(1)] = d / f"frame_{m.group(1)}.txt"
    for p in sorted(reference.glob("frame_*.txt")):
        m = re.match(r"^frame_(\d+)\.txt$", p.name)
        if m:
            out.setdefault(m.group(1), p)
    return out


def compare(ours, his) -> dict:
    """Direct and mirrored palm fits of ours onto his; the numbers per frame."""
    src = np.asarray(ours, dtype=float)
    dst = np.asarray(his, dtype=float)
    _a, rms, err, scale = align_points(src, dst, with_scale=True, subset=PALM_FIT_IDX)
    mirrored = src.copy()
    mirrored[:, 0] *= -1.0
    _m, rms_mir, err_mir, scale_mir = align_points(
        mirrored, dst, with_scale=True, subset=PALM_FIT_IDX)
    ratio = rms_mir / rms if rms > 0 else float("inf")
    if rms <= rms_mir:
        fit, best_err, best_scale = "same", err, scale
    else:
        fit = "MIRROR" if ratio < CLEARLY_MIRRORED else "mirror?"
        best_err, best_scale = err_mir, scale_mir
    worst = sorted(range(len(best_err)), key=lambda i: -best_err[i])[:3]
    return {
        "rms": float(rms),
        "rms_mirrored": float(rms_mir),
        "ratio": float(ratio),
        "fit": fit,
        "rms_best": float(min(rms, rms_mir)),
        "scale": float(best_scale),
        "worst": [(LANDMARK_LABELS[i], float(best_err[i])) for i in worst],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ours", type=Path, nargs="?", default=DEFAULT_OURS,
                   help=f"folder of frame_<id>_keypoints.txt files (default: {DEFAULT_OURS})")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                   help=f"the professor's frames (default: {DEFAULT_REFERENCE})")
    p.add_argument("--only", default=None, help="comma-separated frame ids")
    p.add_argument("--csv", type=Path, default=None, help="write the per-frame table")
    args = p.parse_args(argv)

    ours = our_files(args.ours)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        ours = {k: v for k, v in ours.items() if k in wanted}
    if not ours:
        print(f"no frame_<id>_keypoints.txt files under {args.ours}")
        return 2
    theirs = reference_files(args.reference)

    rows, missing = [], []
    for fid, path in ours.items():
        if fid not in theirs:
            missing.append((fid, "no reference file"))
            continue
        mine, his = load_file(path), load_file(theirs[fid])
        if not mine or not his:
            missing.append((fid, "unreadable block" if his else "unreadable reference"))
            continue
        if len(mine) > 1:
            print(f"  note: {path.name} holds {len(mine)} blocks; the first is used")
        r = compare(mine[0].points, his[0].points)
        rows.append({"frame": fid, "his_label": his[0].hand, "our_label": mine[0].hand, **r})

    if not rows:
        print("no frames could be compared")
        for fid, why in missing:
            print(f"  {fid}: {why}")
        return 2

    print(f"{'frame':<7} {'his':<5} {'ours':<5} {'rms':>6} {'mirror':>7} {'ratio':>6} "
          f"{'fit':<7} {'scale':>6}  worst three (his units)")
    for r in rows:
        worst = ", ".join(f"{name} {e:.1f}" for name, e in r["worst"])
        print(f"{r['frame']:<7} {r['his_label']:<5} {r['our_label']:<5} {r['rms']:6.1f} "
              f"{r['rms_mirrored']:7.1f} {r['ratio']:6.2f} {r['fit']:<7} {r['scale']:6.3f}  {worst}")

    n = len(rows)
    med = statistics.median(r["rms"] for r in rows)
    med_best = statistics.median(r["rms_best"] for r in rows)
    mirrored = [r["frame"] for r in rows if r["fit"] == "MIRROR"]
    labels = sum(r["his_label"] == r["our_label"] for r in rows)
    scales = [r["scale"] for r in rows]
    far = [r["frame"] for r in rows if r["rms_best"] > FAR_ABOVE_MEDIAN * med_best]
    print(f"\nframes compared         {n}" + (f"  ({len(missing)} skipped)" if missing else ""))
    print(f"median RMS              {med:.1f} direct fit, {med_best:.1f} better of the two fits")
    print(f"mirrored fit clearly    {len(mirrored)}/{n} ({100.0 * len(mirrored) / n:.0f}%)"
          f" better (ratio under {CLEARLY_MIRRORED})" + (f": {', '.join(mirrored)}" if mirrored else ""))
    print(f"labels agree            {labels}/{n} (labels are unreliable; shape decides)")
    print(f"scale, his per our mm   median {statistics.median(scales):.3f}, "
          f"spread {min(scales):.3f} to {max(scales):.3f}")
    print(f"RMS over {FAR_ABOVE_MEDIAN:g}x median     " + (", ".join(far) if far else "none"))
    for fid, why in missing:
        print(f"  skipped {fid}: {why}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "his_label", "our_label", "rms", "rms_mirrored", "ratio",
                        "fit", "scale", "worst1", "worst2", "worst3"])
            for r in rows:
                w.writerow([r["frame"], r["his_label"], r["our_label"], f"{r['rms']:.2f}",
                            f"{r['rms_mirrored']:.2f}", f"{r['ratio']:.3f}", r["fit"],
                            f"{r['scale']:.4f}"]
                           + [f"{name} {e:.1f}" for name, e in r["worst"]])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
