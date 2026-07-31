"""Fine-tune a YOLO pose model to find keypoints on a gloved hand.

Run this with the OLD project's interpreter, which already has ultralytics and
the pretrained weights; this new project's venv deliberately does not carry a
second multi-gigabyte torch install:

  & "..\\xr trainer\\summer-xr-trainer\\.venv\\Scripts\\python.exe" scripts/train_glove_model.py --epochs 50

The dataset comes from scripts/build_glove_dataset.py — bare hands from the
Ultralytics hand-keypoints set, repainted to look gloved, keeping their
ground-truth 21-keypoint labels.

Training is CPU-only on this machine (torch reports cuda False), so a real run
is hours, not minutes. Start with --smoke to prove the loop completes in a
couple of minutes before committing to one.

The starting weights were pretrained on 17 human-body keypoints, so the pose
head is reinitialised for 21 hand keypoints and keypoint accuracy begins at
zero by construction. Box detection transfers immediately; keypoints need
epochs. A first run whose pose mAP is ~0 is undertrained, not broken.
"""
import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OLD_VISION = HERE.parent / "xr trainer" / "vision-experiment"
DEFAULT_WEIGHTS = OLD_VISION / "yolo11n-pose.pt"
DEFAULT_DATA = HERE / "datasets" / "glove_synth" / "glove_synth.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune YOLO pose on gloved hands.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0,
                   help="dataloader workers; 0 avoids Windows spawn overhead")
    p.add_argument("--device", default="cpu")
    p.add_argument("--project", type=Path, default=HERE / "runs")
    p.add_argument("--name", default="glove_pose")
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs at 320px — just prove the loop runs")
    args = p.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"dataset config not found: {args.data}\n"
                         "Build one first: python scripts/build_glove_dataset.py")
    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights}")

    # Same certificate workaround the earlier vision experiment needed: this
    # network fails revocation checks, so ultralytics' own downloads (fonts,
    # any missing weights) go through the curl config that sets --ssl-no-revoke.
    os.environ.setdefault("CURL_HOME", str(OLD_VISION))

    epochs = 2 if args.smoke else args.epochs
    imgsz = 320 if args.smoke else args.imgsz
    name = f"{args.name}_smoke" if args.smoke else args.name

    from ultralytics import YOLO

    print(f"data    {args.data}")
    print(f"weights {args.weights}")
    print(f"{epochs} epochs at {imgsz}px, batch {args.batch}, device {args.device}")

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=epochs,
        imgsz=imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    best = args.project / name / "weights" / "best.pt"
    print(f"\nweights: {best}")
    print("Predict on real gloved frames with:")
    print(f'  python scripts/predict_glove.py --weights "{best}" <image-or-folder>')


if __name__ == "__main__":
    main()
