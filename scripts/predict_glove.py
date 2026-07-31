"""Run the fine-tuned gloved-hand model on images or the live webcam.

This is the test that matters. The model is trained on synthetically gloved
hands, so it is only proven against synthetic gloves until it is pointed at
the real one. Put the glove on, run with --source cam, and watch whether the
skeleton tracks.

Run with the OLD project's interpreter, which has ultralytics:

  & "..\\xr trainer\\summer-xr-trainer\\.venv\\Scripts\\python.exe" scripts/predict_glove.py --source cam
  & "...python.exe" scripts/predict_glove.py --source some/folder --save

Deliberately depends on ultralytics only, not on this project's cam_hand
package, so it runs in that other venv without any path wiring.

Reading the result: box confidence tells you whether it found the hand at all;
whether the keypoints sit on the right knuckles is a separate question and has
to be judged by eye until a hand-labelled set of real gloved frames exists.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = HERE / "runs" / "glove_pose_v1" / "weights" / "best.pt"


def epochs_finished(run_dir: Path) -> int:
    """Rows in results.csv — one per completed epoch, header excluded."""
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return 0
    try:
        lines = [l for l in csv_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return max(0, len(lines) - 1)
    except OSError:
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Predict gloved-hand keypoints.")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--source", default="cam",
                   help='"cam" for the webcam, or an image / folder path')
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25,
                   help="box confidence floor")
    p.add_argument("--save", action="store_true",
                   help="write annotated images (folder/image sources)")
    p.add_argument("--out", type=Path, default=HERE / "runs" / "predict")
    p.add_argument("--duration", type=float, default=0.0,
                   help="webcam: auto-close after N seconds")
    args = p.parse_args()

    if not args.weights.is_file():
        # best.pt is only written when a run finishes, but last.pt is rewritten
        # after every epoch — so a run still in progress can already be tested.
        latest = args.weights.with_name("last.pt")
        if latest.is_file():
            done = epochs_finished(args.weights.parent.parent)
            print(f"{args.weights.name} does not exist yet — training has not "
                  "finished.")
            print(f"Using {latest.name} instead"
                  + (f", after {done} epoch(s)" if done else "")
                  + ". Results improve as training continues; re-run this")
            print("later to see it get better.\n")
            args.weights = latest
        else:
            raise SystemExit(
                f"weights not found: {args.weights}\n"
                "Nothing has finished an epoch yet. Either training is still on "
                "epoch 1, or it was never started:\n"
                "  python scripts/train_glove_model.py --smoke   (2 quick epochs)")

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    if args.source != "cam":
        results = model.predict(source=str(args.source), conf=args.conf,
                                device="cpu", save=args.save,
                                project=str(args.out.parent),
                                name=args.out.name, exist_ok=True, verbose=False)
        found = sum(1 for r in results if r.boxes is not None and len(r.boxes))
        print(f"{found}/{len(results)} images with a detection "
              f"(conf >= {args.conf})")
        for r in results[:10]:
            n = 0 if r.boxes is None else len(r.boxes)
            best = f"{float(r.boxes.conf.max()):.2f}" if n else "-"
            print(f"  {Path(r.path).name:<32} {n} detection(s)  best conf {best}")
        if args.save:
            print(f"annotated images -> {args.out}")
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    import time
    t0 = time.time()
    frames = hits = 0
    print("q or Esc to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            res = model.predict(frame, conf=args.conf, device="cpu",
                                verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                hits += 1
            annotated = cv2.flip(res.plot(), 1)
            rate = 100.0 * hits / max(frames, 1)
            cv2.putText(annotated, f"detected {rate:.0f}% of {frames} frames",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(annotated, f"detected {rate:.0f}% of {frames} frames",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 1,
                        cv2.LINE_AA)
            cv2.imshow("gloved-hand model", annotated)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            if args.duration and time.time() - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n{hits}/{frames} frames detected "
              f"({100.0 * hits / max(frames, 1):.0f}%)")
        print("Compare against MediaPipe's 0% on the gloved hand "
              "(results/detection_tuning.csv).")


if __name__ == "__main__":
    main()
