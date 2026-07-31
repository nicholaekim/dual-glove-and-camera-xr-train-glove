"""Guided pose-recording session with the WEBCAM — the camera twin of the
glove pipeline's record_poses.py, same protocol, same defaults, same labels.

Walks the same pose list: announces each pose on the preview window, counts
down with beeps, records a few seconds of detected hand frames, moves on.
Each take is its own labeled JSONL in recordings/poses_cam/ (every frame
carries "pose" and "take"), named like fist_right_take2_20260731_101500.jsonl
— the same naming scheme as the glove takes, so the two datasets line up
folder-for-folder.

  python scripts/record_poses_cam.py                          # 6 poses x 3 takes x 5 s
  python scripts/record_poses_cam.py --poses fist,peace --takes 2 --duration 4
  python scripts/record_poses_cam.py --hz 0                   # keep every frame (~30/s)

Hold the pose facing the camera; keep the whole hand in frame. Ctrl+C (or q
in the window) keeps the takes recorded so far.
"""
import argparse
import time
from pathlib import Path

import cv2

from cam_hand.capture import open_camera, read_frame
from cam_hand.draw import draw_banner, draw_hand, draw_hud, label_hands
from cam_hand.landmarks import DEFAULT_MODEL, HandTracker
from cam_hand.recorder import (
    CamRecorder,
    finalize_pose_name,
    hand_tag,
    pose_filename,
    slugify,
)

DEFAULT_POSES = ["open_palm", "fist", "index_point", "thumbs_up", "peace", "pinch"]

POSE_HINTS = {
    "open_palm": "all five fingers extended and spread",
    "fist": "all fingers curled into a tight fist",
    "index_point": "index finger extended, all others curled",
    "thumbs_up": "thumb extended up, all four fingers curled",
    "peace": "index + middle extended in a V, others curled",
    "pinch": "thumb and index fingertips touching, others relaxed",
    "three": "index + middle + ring extended, little and thumb curled",
}

WAIT_TIMEOUT = 120.0   # s to wait for a hand before the first take
WAIT_FRAMES = 10       # frames containing a hand required before starting


def beep(freq: int = 880, ms: int = 180) -> None:
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


class QuitSession(Exception):
    """Raised when the user presses q in the preview window."""


class CamSession:
    def __init__(self, cap, tracker, hz, out_dir: Path, mirror: bool = True,
                 show: bool = True):
        self.cap = cap
        self.tracker = tracker
        self.hz = hz
        self.out_dir = out_dir
        self.mirror = mirror
        self.show = show
        self.results = []

    # --- camera plumbing -------------------------------------------------
    def tick(self, recorder=None, banner="", sub="", rec=False):
        """Grab one frame, detect, optionally record, optionally preview."""
        frame, ts_ms = read_frame(self.cap)
        if frame is None:
            raise RuntimeError("camera stopped delivering frames")
        hands = self.tracker.detect(frame, ts_ms)
        if recorder is not None:
            size = (frame.shape[1], frame.shape[0])
            for h in hands:
                recorder.record(h, ts_ms, size)
        if self.show:
            for h in hands:
                draw_hand(frame, h)
            if self.mirror:
                frame = cv2.flip(frame, 1)
            label_hands(frame, hands, mirrored=self.mirror)
            if banner:
                draw_banner(frame, banner, sub=sub, rec=rec)
            draw_hud(frame, ["q = stop session"])
            cv2.imshow("cam_hand pose recorder", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise QuitSession
        return hands

    def wait_for_hand(self) -> None:
        print(f"Waiting for a hand in view (need {WAIT_FRAMES} frames, "
              f"timeout {int(WAIT_TIMEOUT)} s)...")
        t0 = time.time()
        good = 0
        while time.time() - t0 < WAIT_TIMEOUT:
            hands = self.tick(banner="SHOW YOUR HAND",
                              sub="hold it inside the frame")
            if hands:
                good += 1
                if good >= WAIT_FRAMES:
                    sides = ", ".join(sorted({h.hand_side for h in hands}))
                    print(f"  OK - tracking ({sides})\n")
                    return
        raise SystemExit("No hand detected. Check lighting and camera index "
                         "(try: python scripts/live_view.py)")

    # --- protocol --------------------------------------------------------
    def run_take(self, pose: str, take: int, n_takes: int,
                 pose_idx: int, n_poses: int, duration: float, prep: float) -> None:
        title = pose.replace("_", " ").upper()
        print(f"--- Pose {pose_idx}/{n_poses}: {title}  (take {take}/{n_takes}) ---")
        hint = POSE_HINTS.get(pose, "")
        if hint:
            print(f"    Hold: {hint}")

        t_end = time.time() + prep
        while time.time() < t_end:
            s = int(t_end - time.time()) + 1
            if s <= 3 and abs((t_end - time.time()) % 1.0) < 0.05:
                beep(660, 80)
            self.tick(banner=f"NEXT: {title}", sub=f"{hint}   ({s})")

        recorder = CamRecorder(hz=self.hz, pose=pose, take=take)
        path = self.out_dir / pose_filename(pose, take)
        recorder.start(path)
        beep(1000, 250)
        print(f"      REC {duration:g} s - hold it ", end="", flush=True)
        try:
            t_end = time.time() + duration
            next_dot = time.time() + 0.5
            while time.time() < t_end:
                self.tick(recorder, banner=title, rec=True)
                if time.time() >= next_dot:
                    print(".", end="", flush=True)
                    next_dot += 0.5
        finally:
            print(flush=True)
            recorder.stop()
            beep(500, 300)
            entry = {"pose": pose, "take": take, "frames": recorder.count,
                     "hands": hand_tag(recorder.hands_seen),
                     "ok": recorder.count > 0}
            if recorder.count == 0:
                path.unlink(missing_ok=True)
                print("      FAILED: no hand frames captured (hand out of view?)\n")
            else:
                final = finalize_pose_name(path, recorder.hands_seen)
                entry["file"] = final.name
                print(f"      saved {recorder.count} frames ({entry['hands']}) "
                      f"-> {final.name}\n")
            self.results.append(entry)

    def print_summary(self) -> None:
        if not self.results:
            print("\nNothing recorded.")
            return
        ok = [r for r in self.results if r["ok"]]
        print("=" * 62)
        print(f"Session summary: {len(ok)}/{len(self.results)} takes captured")
        by_pose = {}
        for r in self.results:
            by_pose.setdefault(r["pose"], []).append(r)
        for pose, takes in by_pose.items():
            parts = []
            for r in takes:
                parts.append(f"take{r['take']}: {r['frames']}f/{r['hands']}"
                             if r["ok"] else f"take{r['take']}: FAILED")
            print(f"  {pose:<12} " + "   ".join(parts))
        if ok:
            print(f"\nFiles in {self.out_dir}")
            print("  to 21-kp CSV:  python scripts/export_keypoints21_cam.py")
            print("  to prof txt:   python scripts/export_prof_format_cam.py")
            print("  separability:  python scripts/analyze_poses_cam.py")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Guided pose recording with the webcam.")
    p.add_argument("--poses", default=",".join(DEFAULT_POSES))
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--prep", type=float, default=5.0)
    p.add_argument("--hz", type=float, default=5.0,
                   help="frames saved per second per hand (default 5; 0 = all)")
    p.add_argument("--out-dir", type=Path, default=Path("recordings") / "poses_cam")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-preview", action="store_true",
                   help="no preview window (console + beeps only)")
    args = p.parse_args()

    poses = [slugify(x) for x in args.poses.split(",") if slugify(x)]
    if not poses:
        raise SystemExit("no poses given")

    total = len(poses) * args.takes
    eta = total * (args.prep + args.duration)
    print("=" * 62)
    print(f"Camera pose session: {len(poses)} poses x {args.takes} takes "
          f"x {args.duration:g} s  (~{eta / 60:.1f} min)")
    print(f"  poses: {', '.join(poses)}")
    print(f"  rate:  {'every frame' if not args.hz else f'{args.hz:g} frames/s'}"
          f"   output: {args.out_dir}")
    print("  Face the camera, keep the whole hand in frame; beeps mark takes.")
    print("=" * 62 + "\n")

    cap = open_camera(args.camera, args.width, args.height)
    tracker = HandTracker(model_path=args.model, running_mode="video")
    session = CamSession(cap, tracker, hz=args.hz or None, out_dir=args.out_dir,
                         mirror=not args.no_mirror, show=not args.no_preview)
    try:
        session.wait_for_hand()
        for i, pose in enumerate(poses, 1):
            for take in range(1, args.takes + 1):
                session.run_take(pose, take, args.takes, i, len(poses),
                                 args.duration, args.prep)
    except (KeyboardInterrupt, QuitSession):
        print("\nInterrupted — keeping the takes recorded so far.")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        session.print_summary()


if __name__ == "__main__":
    main()
