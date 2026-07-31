"""Save plain webcam frames to a folder — no detection, no labels.

Needed the moment a model trained on synthetic gloves fails on the real one.
Everything after that point depends on having real gloved images on disk: to
compare against the synthetic ones and see what the repaint gets wrong, to
hand-label a validation set, and to mix real data into training.

Deliberately dumb. It only grabs frames, so it works no matter what any
detector thinks of them.

  python scripts/capture_frames.py --n 40 --out captures/gloved
  python scripts/capture_frames.py --n 40 --out captures/bare --prep 5

Move the hand slowly through several angles and distances while it runs; a set
of forty near-identical frames is worth about as much as one.
"""
import argparse
import time
from pathlib import Path

import cv2

from cam_hand.capture import open_camera, read_frame
from cam_hand.draw import draw_banner, draw_hud


def main() -> None:
    p = argparse.ArgumentParser(description="Save raw webcam frames to a folder.")
    p.add_argument("--n", type=int, default=40, help="frames to save")
    p.add_argument("--out", type=Path, default=Path("captures") / "gloved")
    p.add_argument("--every", type=float, default=0.4,
                   help="seconds between saved frames (default 0.4)")
    p.add_argument("--prep", type=float, default=4.0,
                   help="seconds before capture starts")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--label", default="",
                   help="prefix for the filenames, e.g. gloved")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prefix = args.label or args.out.name
    cap = open_camera(args.camera, args.width, args.height)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    saved = 0
    try:
        end = time.time() + args.prep
        while time.time() < end:
            frame, _ = read_frame(cap)
            if frame is None:
                raise SystemExit("camera stopped delivering frames")
            shown = cv2.flip(frame, 1)
            draw_banner(shown, "GET READY",
                        sub=f"hand in frame  ({int(end - time.time()) + 1})")
            cv2.imshow("capture", shown)
            cv2.waitKey(1)

        next_save = time.time()
        while saved < args.n:
            frame, _ = read_frame(cap)
            if frame is None:
                break
            now = time.time()
            if now >= next_save:
                # saved unmirrored: this is data, and every other script in the
                # project treats the raw camera orientation as the truth
                path = args.out / f"{prefix}_{stamp}_{saved:03d}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                next_save = now + args.every
            shown = cv2.flip(frame, 1)
            draw_banner(shown, f"{saved}/{args.n}",
                        sub="move slowly: angles, distances, poses", rec=True)
            draw_hud(shown, ["q = stop early"])
            cv2.imshow("capture", shown)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"saved {saved} frames to {args.out}")
    if saved:
        print("Check what a detector makes of them:")
        print('  & "..\\xr trainer\\summer-xr-trainer\\.venv\\Scripts\\python.exe" '
              f'scripts/predict_glove.py --source {args.out} --conf 0.05 --save')


if __name__ == "__main__":
    main()
