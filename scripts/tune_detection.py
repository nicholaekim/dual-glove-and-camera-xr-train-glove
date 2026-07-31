"""Find settings that let the camera see a hand it is currently missing.

Point the camera at the hand that is NOT being detected — typically the gloved
one — hold still, and this sweeps the settings that could plausibly matter,
measuring the detection rate of each instead of guessing:

  detection threshold   0.50 (default) down to 0.05. MediaPipe may be finding
                        the hand with low confidence and discarding it; if so,
                        lowering the threshold recovers it outright.
  image boost           brighten + local contrast. Included for completeness,
                        but measured on a real hand photo, MediaPipe still
                        detects at 0.97 after darkening to gamma 0.25 — so
                        exposure is rarely the cause and this rarely helps.

Read the result honestly. If every row is 0%, no setting will fix it: the
model was trained on bare skin and does not recognise the glove as a hand.
That is a finding about the sensor, not a bug, and the fallback is a detector
fine-tuned on gloved hands (see ../xr trainer/vision-experiment).

  python scripts/tune_detection.py
  python scripts/tune_detection.py --frames 60      # longer, steadier sample
  python scripts/tune_detection.py --compare        # also measure a bare hand
"""
import argparse
import csv
import time

from pathlib import Path

import cv2

from cam_hand.capture import open_camera, read_frame
from cam_hand.draw import draw_banner, draw_hud
from cam_hand.landmarks import DEFAULT_MODEL, HandTracker

# (label, gamma, clahe)
BOOSTS = [("none", 1.0, 0.0), ("boost", 0.55, 2.5)]
THRESHOLDS = [0.5, 0.3, 0.15, 0.05]


def countdown(cap, seconds: float, message: str, sub: str) -> None:
    end = time.time() + seconds
    while time.time() < end:
        frame, _ = read_frame(cap)
        if frame is None:
            break
        frame = cv2.flip(frame, 1)
        draw_banner(frame, message, sub=f"{sub}  ({int(end - time.time()) + 1})")
        cv2.imshow("cam_hand detection tuner", frame)
        cv2.waitKey(1)


def measure(cap, model, frames: int, min_det: float, gamma: float,
            clahe: float, label: str):
    """Run N frames through one configuration; return (rate, mean score)."""
    tracker = HandTracker(model_path=model, running_mode="video",
                          min_detection_confidence=min_det,
                          min_tracking_confidence=min_det,
                          gamma=gamma, clahe=clahe)
    hits, scores = 0, []
    try:
        for i in range(frames):
            frame, ts = read_frame(cap)
            if frame is None:
                break
            hands = tracker.detect(frame, ts)
            if hands:
                hits += 1
                scores.append(max(h.score for h in hands))
            frame = cv2.flip(frame, 1)
            draw_banner(frame, label, sub=f"{i + 1}/{frames}   hits {hits}")
            draw_hud(frame, ["measuring - hold still"])
            cv2.imshow("cam_hand detection tuner", frame)
            cv2.waitKey(1)
    finally:
        tracker.close()
    rate = 100.0 * hits / max(frames, 1)
    return rate, (sum(scores) / len(scores) if scores else 0.0)


def sweep(cap, model, frames: int, title: str, collected: list):
    print(f"\n=== {title} ===")
    print(f"{'boost':<8}{'min_det':>9}{'detected':>11}{'mean score':>12}")
    rows = []
    for bname, gamma, clahe in BOOSTS:
        for thr in THRESHOLDS:
            label = f"{bname} / thr {thr:g}"
            rate, score = measure(cap, model, frames, thr, gamma, clahe, label)
            rows.append((rate, score, bname, thr))
            collected.append({"condition": title, "boost": bname,
                              "min_det": f"{thr:g}", "detected_pct": f"{rate:.0f}",
                              "mean_score": f"{score:.3f}", "frames": frames})
            print(f"{bname:<8}{thr:>9.2f}{rate:>10.0f}%{score:>12.2f}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure which settings detect a hand the camera is missing.")
    p.add_argument("--frames", type=int, default=40,
                   help="frames measured per configuration (default 40)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--prep", type=float, default=5.0,
                   help="seconds to get into position before each sweep")
    p.add_argument("--out", type=Path,
                   default=Path("results") / "detection_tuning.csv",
                   help="where to save the measured table")
    p.add_argument("--compare", action="store_true",
                   help="also sweep a bare hand, as a control")
    args = p.parse_args()

    n = len(BOOSTS) * len(THRESHOLDS)
    print("=" * 62)
    print(f"Detection tuner: {n} configurations x {args.frames} frames each")
    print("  Hold the hand steady, fully inside the frame, palm to the camera.")
    print("=" * 62)

    cap = open_camera(args.camera, args.width, args.height)
    collected: list = []
    try:
        countdown(cap, args.prep, "GLOVED HAND", "hold it still, fully in frame")
        rows = sweep(cap, args.model, args.frames, "gloved hand", collected)

        best = max(rows, key=lambda r: (r[0], r[1]))
        print()
        if best[0] == 0:
            print("Nothing detected in ANY configuration.")
            print("  The glove is not recognisable to this model — no setting")
            print("  fixes that. Options, in order of effort:")
            print("   1. retry with the whole hand smaller in frame and lit")
            print("      from the front (see README, known limits)")
            print("   2. record the camera on the BARE hand and the glove on")
            print("      the other, accepting they are different hands")
            print("   3. fine-tune a pose detector on gloved-hand images")
            print("      (../xr trainer/vision-experiment already does this)")
        else:
            print(f"Best: boost={best[2]}, min_det={best[3]:g} "
                  f"-> {best[0]:.0f}% of frames, mean score {best[1]:.2f}")
            gamma, clahe = next((g, c) for nme, g, c in BOOSTS if nme == best[2])
            flags = f"--min-det {best[3]:g}"
            if gamma != 1.0:
                flags += f" --gamma {gamma:g} --clahe {clahe:g}"
            print(f"  Use it:  python scripts/record_simultaneous.py {flags}")
            if best[0] < 80:
                print("  Under 80% is shaky for recording — expect gaps where")
                print("  the take pairs poorly. Improve lighting and framing.")

        if args.compare:
            countdown(cap, args.prep, "BARE HAND", "same position, glove off")
            bare = sweep(cap, args.model, args.frames, "bare hand (control)",
                         collected)
            b = max(bare, key=lambda r: (r[0], r[1]))
            print(f"\nControl: bare hand best {b[0]:.0f}% vs gloved {best[0]:.0f}%")
            print("  A large gap confirms the glove itself is the problem,")
            print("  not the lighting or the camera.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Always write what was measured, even after an interrupt — the whole
        # point is to have a record instead of a table that scrolls away.
        if collected:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(collected[0].keys()))
                w.writeheader()
                w.writerows(collected)
            print(f"\nwrote {args.out}  ({len(collected)} configurations)")


if __name__ == "__main__":
    main()
