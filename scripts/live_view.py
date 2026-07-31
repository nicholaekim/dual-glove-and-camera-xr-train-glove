"""Live webcam hand tracking — the camera counterpart of run_osc.py.

Opens the webcam, runs MediaPipe hand landmarks on every frame, and shows the
21-point skeleton glued to your hand (cyan = left, red = right, same colours
as the glove viewer). The preview is mirrored like a selfie so it feels
natural; recorded/printed data is never mirrored.

Sanity check for handedness: raise your right hand — the label should say
"right". If the labels read backwards, run with --swap-hands (and say so,
because it means the model's convention changed; see cam_hand/landmarks.py).

Usage:
  python scripts/live_view.py                # q or Esc quits
  python scripts/live_view.py --duration 10  # auto-close after 10 s
  python scripts/live_view.py --camera 1     # second webcam
  python scripts/live_view.py --no-window    # headless: console stats only
"""
import argparse
import time

import cv2

from cam_hand.capture import open_camera, read_frame
from cam_hand.draw import draw_hand, draw_hud, label_hands
from cam_hand.landmarks import DEFAULT_MODEL, HandTracker


def main() -> None:
    p = argparse.ArgumentParser(description="Live webcam hand-keypoint viewer.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--min-det", type=float, default=0.5)
    p.add_argument("--duration", type=float, default=0.0,
                   help="auto-close after N seconds (0 = run until q)")
    p.add_argument("--no-mirror", action="store_true",
                   help="show the raw camera view instead of the selfie view")
    p.add_argument("--swap-hands", action="store_true",
                   help="swap left/right labels (only if they read backwards)")
    p.add_argument("--no-window", action="store_true",
                   help="headless: no preview window, just console stats")
    args = p.parse_args()

    cap = open_camera(args.camera, args.width, args.height)
    tracker = HandTracker(model_path=args.model, num_hands=args.max_hands,
                          running_mode="video",
                          swap_handedness=args.swap_hands,
                          min_detection_confidence=args.min_det)
    mirror = not args.no_mirror

    t0 = time.time()
    frames = 0
    fps = 0.0
    last_report = t0
    detected_frames = 0
    try:
        while True:
            frame, ts_ms = read_frame(cap)
            if frame is None:
                print("camera stopped delivering frames")
                break
            hands = tracker.detect(frame, ts_ms)
            frames += 1
            if hands:
                detected_frames += 1

            now = time.time()
            fps = frames / max(now - t0, 1e-6)

            if not args.no_window:
                for h in hands:
                    draw_hand(frame, h)
                if mirror:
                    frame = cv2.flip(frame, 1)
                label_hands(frame, hands, mirrored=mirror)
                draw_hud(frame, [f"{fps:5.1f} fps   hands: {len(hands)}",
                                 "q / Esc to quit"])
                cv2.imshow("cam_hand live view", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if now - last_report >= 1.0:
                sides = ",".join(h.hand_side for h in hands) or "-"
                print(f"  {fps:5.1f} fps | hands now: {sides} | "
                      f"frames with a hand: {detected_frames}/{frames}")
                last_report = now
            if args.duration and now - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        dt = time.time() - t0
        print(f"\n{frames} frames in {dt:.1f} s ({frames / max(dt, 1e-6):.1f} fps), "
              f"hand visible in {detected_frames} ({100.0 * detected_frames / max(frames, 1):.0f}%)")


if __name__ == "__main__":
    main()
