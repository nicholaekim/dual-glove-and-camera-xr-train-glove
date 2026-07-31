"""Build a gloved-hand training set from labelled bare-hand images.

Training a detector to see the gloved hand needs gloved images with 21
keypoints marked on them. Marking those by hand is thousands of clicks per
hundred images. This avoids that entirely: the Ultralytics hand-keypoints
dataset ships 26,768 bare-hand images that already carry ground-truth
21-keypoint labels, and repainting a hand to look gloved
(cam_hand.synth_glove) does not move a single landmark — so the existing
labels stay exactly correct.

Validated before use: MediaPipe detects the bare source images at 100% and
only 10% after repainting, which is the same collapse the real glove causes.
The synthetic glove is therefore simulating the right problem rather than
producing images that merely look dark.

Reads image+label pairs straight out of the zip; the full 369 MB is never
extracted. Output is a standard YOLO pose dataset ready for training.

  python scripts/build_glove_dataset.py --n 2000
  python scripts/build_glove_dataset.py --n 500 --bare-frac 0.2   # mix in bare hands
  python scripts/build_glove_dataset.py --n 40 --preview          # eyeball it first
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from cam_hand.synth_glove import apply_glove

HERE = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = (HERE.parent / "xr trainer" / "vision-experiment" / "datasets"
               / "hand-keypoints.zip")
N_KPT = 21
MIN_VISIBLE = 17          # instances with fewer usable landmarks are skipped

# Left/right mirror pairing for horizontal-flip augmentation, in the standard
# 21-point order. The layout is symmetric per finger, so a flip maps each index
# to itself; Ultralytics still requires the list to be present.
FLIP_IDX = list(range(N_KPT))


def parse_label(text: str):
    """YOLO pose label -> list of (class, bbox, kpts[21,3]) in normalised units."""
    out = []
    for line in text.strip().splitlines():
        v = line.split()
        if len(v) != 5 + N_KPT * 3:
            continue
        cls = int(float(v[0]))
        bbox = [float(x) for x in v[1:5]]
        kpts = np.array([float(x) for x in v[5:]], dtype=float).reshape(N_KPT, 3)
        out.append((cls, bbox, kpts))
    return out


def usable(kpts: np.ndarray) -> bool:
    """Enough visible landmarks, all inside the image, to build a hand mask."""
    vis = kpts[:, 2] > 0
    if vis.sum() < MIN_VISIBLE:
        return False
    xy = kpts[vis][:, :2]
    return bool(((xy >= -0.02) & (xy <= 1.02)).all())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Repaint labelled bare hands as gloved hands for training.")
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--n", type=int, default=1000, help="image pairs to use")
    p.add_argument("--out", type=Path, default=HERE / "datasets" / "glove_synth")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--bare-frac", type=float, default=0.0,
                   help="fraction left un-repainted, to keep bare hands working")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true",
                   help="also write a side-by-side sheet of the first few")
    args = p.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"dataset zip not found: {args.zip}")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    with zipfile.ZipFile(args.zip) as zf:
        names = [m.replace("\\", "/") for m in zf.namelist()]
        labels = {n for n in names if n.startswith("labels/") and n.endswith(".txt")}
        images = [n for n in names
                  if n.startswith("images/") and n.lower().endswith((".jpg", ".png"))]

        def label_for(img_name: str) -> str:
            return ("labels/" + img_name.split("images/", 1)[1]).rsplit(".", 1)[0] + ".txt"

        pairs = [(i, label_for(i)) for i in images if label_for(i) in labels]
        if not pairs:
            raise SystemExit("no image/label pairs found in the zip")
        random.shuffle(pairs)

        if args.out.exists():
            shutil.rmtree(args.out)
        for split in ("train", "val"):
            (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

        kept = skipped = gloved = bare = 0
        previews = []
        for img_name, lbl_name in pairs:
            if kept >= args.n:
                break
            buf = np.frombuffer(zf.read(img_name), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
            instances = parse_label(zf.read(lbl_name).decode("utf-8", "replace"))
            good = [k for k in instances if usable(k[2])]
            if not good:
                skipped += 1
                continue

            h, w = img.shape[:2]
            leave_bare = rng.random() < args.bare_frac
            out_img = img
            if not leave_bare:
                for _cls, _bbox, kpts in good:
                    px = np.stack([kpts[:, 0] * w, kpts[:, 1] * h], axis=1)
                    out_img = apply_glove(out_img, px, rng=rng)
                gloved += 1
            else:
                bare += 1

            split = "val" if kept < args.n * args.val_frac else "train"
            stem = Path(img_name).stem
            cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), out_img)
            # labels are copied verbatim: repainting moves no landmark
            (args.out / "labels" / split / f"{stem}.txt").write_bytes(
                zf.read(lbl_name))
            if args.preview and len(previews) < 4:
                previews.append(np.hstack([img, out_img]))
            kept += 1

    yaml_path = args.out / "glove_synth.yaml"
    yaml_path.write_text(
        f"path: {args.out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{N_KPT}, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        "names:\n"
        "  0: hand\n", encoding="utf-8")

    n_val = min(int(args.n * args.val_frac), kept)
    print(f"wrote {kept} images to {args.out}")
    print(f"  gloved {gloved} | left bare {bare} | skipped {skipped} unusable")
    print(f"  train {kept - n_val} | val {n_val}")
    print(f"  dataset config: {yaml_path}")
    if previews:
        sheet = args.out / "preview.jpg"
        cv2.imwrite(str(sheet), np.vstack(
            [cv2.resize(p, (960, int(960 * p.shape[0] / p.shape[1]))) for p in previews]))
        print(f"  preview (bare | gloved): {sheet}")
    print("\nNext: train on it (the old project's venv has ultralytics)")
    print(f'  yolo pose train model=yolo11n-pose.pt data="{yaml_path}" epochs=50 imgsz=640')


if __name__ == "__main__":
    main()
