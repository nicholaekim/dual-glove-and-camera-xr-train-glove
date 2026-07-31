"""Record the glove and the camera AT THE SAME TIME, one guided session.

This is the data-collection step every fusion result depends on: the same
physical hand, the same instant, seen by both sensors. Each take writes two
files with the same name into

    recordings/sync/glove/<pose>_<hand>_take<N>_<stamp>.jsonl
    recordings/sync/cam/<pose>_<hand>_take<N>_<stamp>.jsonl

Both recorders stamp time.time() at write, so the two streams share one wall
clock and scripts/fuse_poses.py can pair frames afterwards (the glove runs at
~60 Hz, the camera at ~30 Hz; they are matched by nearest timestamp).

The protocol is the glove pipeline's: announce the pose, count down with
beeps, record, move on — no keyboard while wearing the glove.

Needs XR Trainer streaming to 127.0.0.1:9002 plus a webcam. Rehearse the whole
thing with no hardware at all:

  python scripts/record_simultaneous.py --mock-glove --takes 1 --duration 3 --prep 2

Usage:
  python scripts/record_simultaneous.py                       # 6 poses x 3 takes
  python scripts/record_simultaneous.py --poses pinch,fist --takes 2
"""
import argparse
import time
from pathlib import Path

import cv2

from cam_hand.capture import open_camera, read_frame
from cam_hand.draw import draw_banner, draw_hand, draw_hud, label_hands
from cam_hand.landmarks import DEFAULT_MODEL, HandTracker
from cam_hand.recorder import CamRecorder, pose_filename, slugify
from cam_hand.recorder import finalize_pose_name as cam_finalize
from cam_hand.recorder import hand_tag as cam_hand_tag

# The glove side comes from the existing pipeline (see README for the .pth
# that puts summer-xr-trainer/src on the path).
from xr_hand.parser import parse_hand_message
from xr_hand.receiver import OSCHandReceiver
from xr_hand.recorder import FrameRecorder
from xr_hand.validator import StreamMonitor, validate_raw_message

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


def beep(freq: int = 880, ms: int = 180) -> None:
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


class QuitSession(Exception):
    pass


class MockGloveSource:
    """Synthetic glove stream at ~60 Hz behind the receiver's drain() API."""

    def __init__(self):
        from xr_hand.mock import MockHandGenerator
        self.gens = {"left": MockHandGenerator(hand="left"),
                     "right": MockHandGenerator(hand="right")}
        self._last = time.time()

    def start(self):
        pass

    def stop(self):
        pass

    def drain(self, max_items: int = 64):
        now = time.time()
        n = min(int((now - self._last) * 60.0), max_items // 2)
        if n <= 0:
            return []
        self._last += n / 60.0
        return [(hand, gen.next_frame())
                for _ in range(n) for hand, gen in self.gens.items()]


class SyncSession:
    def __init__(self, cap, tracker, glove_source, hz, out_dir: Path,
                 mirror: bool = True, show: bool = True):
        self.cap = cap
        self.tracker = tracker
        self.glove = glove_source
        self.hz = hz
        self.glove_dir = out_dir / "glove"
        self.cam_dir = out_dir / "cam"
        self.mirror = mirror
        self.show = show
        self.monitors = {"left": StreamMonitor("left"), "right": StreamMonitor("right")}
        self.results = []
        self._warned = set()

    def _pump_glove(self, recorder=None) -> int:
        """Drain and optionally record glove packets. Returns frames seen."""
        seen = 0
        for hand, raw in self.glove.drain(64):
            result = validate_raw_message(raw)
            if not result.is_valid:
                self._warn(f"[{hand}] invalid packet: " + "; ".join(result.errors))
                continue
            frame = parse_hand_message(raw, hand_side_hint=hand)
            for w in self.monitors[hand].update(frame.packet_counter):
                self._warn(f"[{hand}] {w}")
            seen += 1
            if recorder is not None:
                recorder.record(frame)
        return seen

    def _warn(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            if len(self._warned) <= 8:
                print(f"      ! {msg}")

    def tick(self, cam_rec=None, glove_rec=None, banner="", sub="", rec=False):
        """One pass over both sensors, plus the preview window."""
        self._pump_glove(glove_rec)
        frame, ts_ms = read_frame(self.cap)
        if frame is None:
            raise RuntimeError("camera stopped delivering frames")
        hands = self.tracker.detect(frame, ts_ms)
        if cam_rec is not None:
            size = (frame.shape[1], frame.shape[0])
            for h in hands:
                cam_rec.record(h, ts_ms, size)
        if self.show:
            for h in hands:
                draw_hand(frame, h)
            if self.mirror:
                frame = cv2.flip(frame, 1)
            label_hands(frame, hands, mirrored=self.mirror)
            if banner:
                draw_banner(frame, banner, sub=sub, rec=rec)
            draw_hud(frame, ["glove + camera", "q = stop session"])
            cv2.imshow("cam_hand simultaneous recorder", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                raise QuitSession
        return hands

    def wait_for_both(self, timeout: float = 120.0) -> None:
        print("Waiting for BOTH sensors (glove packets + a hand in view)...")
        t0 = time.time()
        glove_ok = cam_ok = 0
        while time.time() - t0 < timeout:
            glove_ok += self._pump_glove()
            hands = self.tick(banner="SHOW YOUR HAND",
                              sub="glove on, hand inside the frame")
            if hands:
                cam_ok += 1
            if glove_ok >= 10 and cam_ok >= 10:
                print(f"  OK - glove packets and camera tracking both live\n")
                return
        raise SystemExit(
            f"Only got {glove_ok} glove packets and {cam_ok} camera detections. "
            "Check XR Trainer is streaming (scripts/run_osc.py --dump --no-viz in "
            "the glove project) and the webcam (scripts/live_view.py).")

    def run_take(self, pose: str, take: int, n_takes: int, pose_idx: int,
                 n_poses: int, duration: float, prep: float) -> None:
        self._warned = set()
        title = pose.replace("_", " ").upper()
        hint = POSE_HINTS.get(pose, "")
        print(f"--- Pose {pose_idx}/{n_poses}: {title}  (take {take}/{n_takes}) ---")
        if hint:
            print(f"    Hold: {hint}")

        t_end = time.time() + prep
        while time.time() < t_end:
            s = int(t_end - time.time()) + 1
            self.tick(banner=f"NEXT: {title}", sub=f"{hint}   ({s})")

        name = pose_filename(pose, take)      # one name, two files
        cam_rec = CamRecorder(hz=self.hz, pose=pose, take=take)
        glove_rec = FrameRecorder(hz=self.hz, pose=pose, take=take)
        cam_rec.start(self.cam_dir / name)
        glove_rec.start(self.glove_dir / name)
        beep(1000, 250)
        print(f"      REC {duration:g} s - hold it ", end="", flush=True)
        try:
            t_end = time.time() + duration
            next_dot = time.time() + 0.5
            while time.time() < t_end:
                self.tick(cam_rec, glove_rec, banner=title, rec=True)
                if time.time() >= next_dot:
                    print(".", end="", flush=True)
                    next_dot += 0.5
        finally:
            print(flush=True)
            cam_rec.stop()
            glove_rec.stop()
            beep(500, 300)
            entry = {"pose": pose, "take": take,
                     "cam_frames": cam_rec.count, "glove_frames": glove_rec.count,
                     "ok": cam_rec.count > 0 and glove_rec.count > 0}
            hands = cam_rec.hands_seen | glove_rec.hands_seen
            if cam_rec.count == 0 and glove_rec.count == 0:
                (self.cam_dir / name).unlink(missing_ok=True)
                (self.glove_dir / name).unlink(missing_ok=True)
                print("      FAILED: neither sensor captured anything\n")
            else:
                # rename both with the same hand tag so the pair keeps one name
                tag = cam_hand_tag(hands)
                for rec, folder in ((cam_rec, self.cam_dir), (glove_rec, self.glove_dir)):
                    if rec.count:
                        cam_finalize(folder / name, hands)
                    else:
                        # This sensor saw nothing. Drop its empty file rather
                        # than leave a zero-byte orphan under the un-renamed
                        # name: pairs are matched by filename, so an orphan can
                        # never pair and only clutters the folder.
                        (folder / name).unlink(missing_ok=True)
                entry["file"] = name.replace("_take", f"_{tag}_take", 1)
                entry["hands"] = tag
                print(f"      saved  glove {glove_rec.count} frames | "
                      f"camera {cam_rec.count} frames  ({tag})")
                if not entry["ok"]:
                    which = "camera" if cam_rec.count == 0 else "glove"
                    print(f"      WARNING: {which} captured nothing — "
                          "this take cannot be fused")
                print()
            self.results.append(entry)

    def print_summary(self) -> None:
        if not self.results:
            print("\nNothing recorded.")
            return
        ok = [r for r in self.results if r["ok"]]
        print("=" * 62)
        print(f"Session summary: {len(ok)}/{len(self.results)} takes have BOTH sensors")
        for r in self.results:
            status = (f"glove {r['glove_frames']:>4}f | cam {r['cam_frames']:>4}f"
                      if r["ok"] else "INCOMPLETE")
            print(f"  {r['pose']:<12} take{r['take']}  {status}")
        if ok:
            print(f"\n  glove files: {self.glove_dir}")
            print(f"  cam files:   {self.cam_dir}")
            print("  fuse + compare:  python scripts/fuse_poses.py")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Guided pose session recording glove and camera together.")
    p.add_argument("--poses", default=",".join(DEFAULT_POSES))
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--prep", type=float, default=5.0)
    p.add_argument("--hz", type=float, default=5.0,
                   help="frames saved per second per hand, both sensors (0 = all)")
    p.add_argument("--out-dir", type=Path, default=Path("recordings") / "sync")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--min-det", type=float, default=0.5,
                   help="detection confidence floor; lower it when a "
                        "gloved hand is missed (see tune_detection.py)")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="brighten before detection (<1 = brighter)")
    p.add_argument("--clahe", type=float, default=0.0,
                   help="local contrast boost before detection")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9002)
    p.add_argument("--address", default="/v1/animation/kinematic/all")
    p.add_argument("--mock-glove", action="store_true",
                   help="synthetic glove stream (rehearse without hardware)")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()

    poses = [slugify(x) for x in args.poses.split(",") if slugify(x)]
    if not poses:
        raise SystemExit("no poses given")

    eta = len(poses) * args.takes * (args.prep + args.duration)
    print("=" * 62)
    print(f"SIMULTANEOUS session: {len(poses)} poses x {args.takes} takes "
          f"x {args.duration:g} s  (~{eta / 60:.1f} min)")
    print(f"  poses: {', '.join(poses)}")
    print(f"  glove: {'MOCK' if args.mock_glove else f'{args.host}:{args.port}'}"
          f"   camera: index {args.camera}   output: {args.out_dir}")
    print("  Wear the glove AND keep the hand in the camera frame.")
    print("=" * 62 + "\n")

    cap = open_camera(args.camera, args.width, args.height)
    tracker = HandTracker(model_path=args.model, running_mode="video",
                          min_detection_confidence=args.min_det,
                          min_tracking_confidence=args.min_det,
                          gamma=args.gamma, clahe=args.clahe)
    glove = (MockGloveSource() if args.mock_glove
             else OSCHandReceiver(host=args.host, port=args.port,
                                  kinematic_addr=args.address))
    glove.start()

    session = SyncSession(cap, tracker, glove, hz=args.hz or None,
                          out_dir=args.out_dir, mirror=not args.no_mirror,
                          show=not args.no_preview)
    try:
        session.wait_for_both()
        for i, pose in enumerate(poses, 1):
            for take in range(1, args.takes + 1):
                session.run_take(pose, take, args.takes, i, len(poses),
                                 args.duration, args.prep)
    except (KeyboardInterrupt, QuitSession):
        print("\nInterrupted — keeping the takes recorded so far.")
    finally:
        glove.stop()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        session.print_summary()


if __name__ == "__main__":
    main()
