"""Click 21 keypoints on real frames to build a hand-labelled training set.

Why this exists: two rounds of synthetic gloving failed on the real glove.
Same webcam, same room, same hand — the model detects a synthetically gloved
hand 20/20 and the real glove 0/80. Simulation is not converging, so the
remaining route is real labels.

Controls
  left click    place the next landmark (its name is shown top-left)
  u / backspace undo the last point
  r             restart this image
  s             skip this image (hand not visible, too blurred, etc.)
  n / enter     save and go to the next image (only once 21 points are set)
  q / esc       quit; everything saved so far is kept

Landmark order is the MediaPipe standard, same as everything else here:
wrist, then thumb -> pinky, each finger base to tip. The on-screen prompt
names each one, and already-placed points stay drawn with the skeleton so a
mistake is visible immediately.

Output is a YOLO pose dataset that trains directly:

  python scripts/label_frames.py --source captures/gloved --out datasets/real_gloved
  & "..\\xr trainer\\summer-xr-trainer\\.venv\\Scripts\\python.exe" scripts/train_glove_model.py \\
      --data datasets/real_gloved/real_gloved.yaml --weights runs/glove_pose_v2/weights/best.pt \\
      --epochs 40 --imgsz 512
"""
import argparse
import shutil
from pathlib import Path

import cv2

from cam_hand.draw import CONNECTIONS, draw_text
from cam_hand.landmarks import MP21_NAMES

N_KPT = 21
WIN = "label - click each landmark"
DONE_COLOR = (0, 220, 255)
NEXT_COLOR = (80, 255, 80)


def draw_state(img, pts, idx: int, name: str, i: int, total: int, saved: int):
    canvas = img.copy()
    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(canvas, pts[a], pts[b], (200, 120, 0), 2, cv2.LINE_AA)
    for k, p in enumerate(pts):
        cv2.circle(canvas, p, 4, DONE_COLOR, -1, cv2.LINE_AA)
        draw_text(canvas, str(k), (p[0] + 5, p[1] - 5), 0.4, DONE_COLOR)
    draw_text(canvas, f"image {i + 1}/{total}   labelled {saved}",
              (10, 24), 0.6, (255, 255, 255))
    if idx < N_KPT:
        draw_text(canvas, f"[{idx}/21] click: {name}", (10, 50), 0.7, NEXT_COLOR)
    else:
        draw_text(canvas, "21/21 done - press N for next image", (10, 50),
                  0.7, DONE_COLOR)
    draw_text(canvas, "u undo   r restart   s skip   n next   q quit",
              (10, canvas.shape[0] - 12), 0.5, (200, 200, 200))
    return canvas


def yolo_line(pts, w: int, h: int) -> str:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad_x, pad_y = 0.04 * w, 0.04 * h
    x0, x1 = max(min(xs) - pad_x, 0), min(max(xs) + pad_x, w)
    y0, y1 = max(min(ys) - pad_y, 0), min(max(ys) + pad_y, h)
    vals = ["0",
            f"{((x0 + x1) / 2) / w:.6f}", f"{((y0 + y1) / 2) / h:.6f}",
            f"{(x1 - x0) / w:.6f}", f"{(y1 - y0) / h:.6f}"]
    for x, y in pts:
        vals += [f"{x / w:.6f}", f"{y / h:.6f}", "2"]
    return " ".join(vals)


def main() -> None:
    p = argparse.ArgumentParser(description="Hand-label 21 keypoints per frame.")
    p.add_argument("--source", type=Path, default=Path("captures") / "gloved")
    p.add_argument("--out", type=Path, default=Path("datasets") / "real_gloved")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=0, help="stop after N images")
    p.add_argument("--scale", type=float, default=1.5,
                   help="display zoom; clicks are mapped back to full size")
    args = p.parse_args()

    frames = sorted(q for q in args.source.rglob("*")
                    if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not frames:
        raise SystemExit(f"no images in {args.source}")
    if args.limit:
        frames = frames[: args.limit]

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    state = {"pts": []}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["pts"]) < N_KPT:
            state["pts"].append((int(x / args.scale), int(y / args.scale)))

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    saved = 0
    quit_now = False
    for i, path in enumerate(frames):
        if quit_now:
            break
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        state["pts"] = []

        while True:
            idx = len(state["pts"])
            name = MP21_NAMES[idx] if idx < N_KPT else ""
            shown = draw_state(img, state["pts"], idx, name, i, len(frames), saved)
            shown = cv2.resize(shown, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WIN, shown)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                quit_now = True
                break
            if key in (ord("u"), 8) and state["pts"]:
                state["pts"].pop()
            elif key == ord("r"):
                state["pts"] = []
            elif key == ord("s"):
                break
            elif key in (ord("n"), 13) and idx == N_KPT:
                split = "val" if saved % max(2, int(1 / max(args.val_frac, 1e-6))) == 0 else "train"
                stem = path.stem
                shutil.copy(path, args.out / "images" / split / f"{stem}.jpg")
                (args.out / "labels" / split / f"{stem}.txt").write_text(
                    yolo_line(state["pts"], w, h) + "\n", encoding="utf-8")
                saved += 1
                break

    cv2.destroyAllWindows()

    yaml_path = args.out / "real_gloved.yaml"
    yaml_path.write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{N_KPT}, 3]\n"
        f"flip_idx: {list(range(N_KPT))}\n"
        "names:\n"
        "  0: hand\n", encoding="utf-8")

    n_train = len(list((args.out / "images" / "train").glob("*.jpg")))
    n_val = len(list((args.out / "images" / "val").glob("*.jpg")))
    print(f"labelled {saved} frames  (train {n_train}, val {n_val}) -> {args.out}")
    if n_train and n_val:
        print("\nFine-tune the v2 model on them:")
        print('  & "..\\xr trainer\\summer-xr-trainer\\.venv\\Scripts\\python.exe" '
              f'scripts/train_glove_model.py --data {yaml_path} '
              '--weights runs/glove_pose_v2/weights/best.pt --epochs 40 --imgsz 512 '
              '--name glove_pose_real')
    elif saved:
        print("Need at least one image in each of train and val — label a few more.")


if __name__ == "__main__":
    main()
