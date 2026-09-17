"""Record the glove and a camera AT THE SAME TIME, one guided session.

This is the data-collection step every fusion result depends on: the same
physical hand, the same instant, seen by both sensors. Each take writes two
files with the same name into

    recordings/sync/glove/<pose>_<hand>_take<N>_<stamp>.jsonl
    recordings/sync/cam/<pose>_<hand>_take<N>_<stamp>.jsonl     (--camera 0)
    recordings/sync/leap/<pose>_<hand>_take<N>_<stamp>.jsonl    (--camera leap)

Both recorders stamp time.time() at write, so the two streams share one wall
clock and scripts/fuse_poses.py can pair frames afterwards (the glove runs at
~60 Hz, a webcam at ~30 Hz, the Ultraleap at ~90 Hz; they are matched by
nearest timestamp).

Two camera backends, one protocol:

  --camera 0      a webcam through MediaPipe. Normalised landmarks, no
                  absolute scale, and it did not see the black glove at all
                  in August — this is the path that motivated the IR camera.
  --camera leap   the Ultraleap Stereo IR 170. Metric 3D joints in metres,
                  and the Phase 2 gate says it tracks the gloved hand
                  (98.6% of frames, 0 re-acquisitions), which is what makes
                  simultaneous capture — Path A — possible at all.

The leap backend opens no OpenCV window: there is nothing photographic to
look at and your hands are over the module, so it prints a one-line HUD
(hands seen, tracking framerate) once a second instead.

The protocol is the glove pipeline's either way: announce the pose, count
down with beeps, record, move on — no keyboard while wearing the glove.

Needs XR Trainer streaming to 127.0.0.1:9002 plus the camera. Rehearse the
whole thing with no hardware at all:

  python scripts/record_simultaneous.py --mock-glove --takes 1 --duration 3 --prep 2
  python scripts/record_simultaneous.py --camera leap --mock-glove --mock-leap \
      --poses fist --takes 1 --duration 3 --prep 1

Usage:
  python scripts/record_simultaneous.py                       # 6 poses x 3 takes
  python scripts/record_simultaneous.py --poses pinch,fist --takes 2
  python scripts/record_simultaneous.py --camera leap         # Path A
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from cam_hand.capture import open_camera, read_frame
from cam_hand.draw import draw_banner, draw_hand, draw_hud, label_hands
from cam_hand.landmarks import DEFAULT_MODEL, HandTracker
from cam_hand.recorder import CamRecorder, pose_filename, slugify
from cam_hand.recorder import finalize_pose_name as cam_finalize
from cam_hand.recorder import hand_tag as cam_hand_tag

# The Ultraleap backend (--camera leap). Importing these is free: leap_hand
# only touches the `leap` bindings inside LeapStream.start().
from leap_hand.recorder import LeapRecorder
from leap_hand.stream import LeapUnavailable, open_stream

# The glove side comes from the xr_hand package in this repo.
from xr_hand.parser import parse_hand_message
from xr_hand.receiver import OSCHandReceiver, QueueItem
from xr_hand.recorder import FrameRecorder
from xr_hand.validator import StreamMonitor, validate_raw_message

DEFAULT_POSES = ["open_palm", "fist", "index_point", "thumbs_up", "peace", "pinch"]
LEAP = "leap"
MIN_VISIBLE_TIME_US = 300_000     # plan section 6: a hand counts after 0.3 s
# Bounds on the post-beep flush (see SyncSession.discard_backlog). A 250 ms
# beep leaves at most ~45 Leap hands and ~30 glove packets behind it, so a
# handful of drain(64) rounds always clears it; the bound is only there so a
# sensor that never stops delivering cannot pin us here.
MAX_DRAIN_ROUNDS = 8
CAM_BUFFER_FRAMES = 2             # webcam frames to grab and drop
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


class _CaptureTimeWriter:
    """The recorder's file handle, with `capture_time` added to each line.

    `FrameRecorder.record` builds its dict and writes it in one call, and
    `FrameRecorder` itself is not changed here: other callers' files must
    keep exactly the keys they have. So the stamp is added where the line
    meets the file, rather than by reimplementing `record()` along with its
    per-hand rate throttle — one key injected, no logic duplicated.

    `FrameRecorder.load` ignores keys it does not know, so playback, the
    exporters and keypoints21 read these files unchanged.
    """

    def __init__(self, fh, owner):
        self._fh = fh
        self._owner = owner

    def write(self, text: str) -> int:
        capture = self._owner.capture_time
        if capture is not None and text.strip():
            d = json.loads(text)
            d["capture_time"] = round(float(capture), 6)
            text = json.dumps(d) + "\n"
        return self._fh.write(text)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class StampedFrameRecorder(FrameRecorder):
    """A glove recorder that also records WHEN THE PACKET ARRIVED.

    `wall_time` is the moment the line was written, which on a busy loop can
    be tens of milliseconds after the packet landed and is near-identical
    across a drained burst. `capture_time` is `QueueItem.recv_time`, taken on
    the OSC server thread as the packet arrived — the glove's closest
    equivalent to the camera's capture instant, and the clock `fuse_poses.py`
    pairs on when both sides have it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_time = None

    def start(self, path):
        super().start(path)
        self._file = _CaptureTimeWriter(self._file, self)

    def record(self, frame, capture_time=None) -> None:
        self.capture_time = capture_time
        super().record(frame)


def hands_text(sides) -> str:
    """{'left'} -> 'left'; set() -> 'nothing'."""
    return ", ".join(sorted(sides)) if sides else "nothing"


def describe_mismatch(cam_rec, glove_rec) -> str:
    """Why a take with frames on both sides still cannot be fused.

    Named rather than counted, because the two failures need different
    fixes: a silent sensor is a cable or a service, two different hands is
    the operator wearing the glove on one hand and holding the other up.
    """
    if cam_rec.count == 0:
        return "the camera captured nothing"
    if glove_rec.count == 0:
        return "the glove captured nothing"
    return (f"the camera saw {hands_text(cam_rec.hands_seen)} but the glove "
            f"streamed {hands_text(glove_rec.hands_seen)} — no hand in common, "
            "so no frame can pair")


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
        # `recv_time` is reconstructed from the paced clock rather than set to
        # `now`, so a backlog drained in one pass carries the arrival times it
        # would have had — the same property the real receiver gives, and the
        # reason pairing on the writer's clock fails against this mock too.
        t0 = self._last
        self._last += n / 60.0
        return [QueueItem(hand, gen.next_frame(), recv_time=t0 + k / 60.0)
                for k in range(n) for hand, gen in self.gens.items()]


class SyncSession:
    """The webcam + glove session. The protocol lives here; see LeapSyncSession.

    Everything specific to the camera behind it is in four places, and a
    backend overrides those and nothing else: `cam_dir`, `make_cam_recorder`,
    `tick` and `wait_for_both`. `dots` is off for a backend that prints its
    own HUD, so the two do not fight over the same line.
    """

    dots = True

    def __init__(self, cap, tracker, glove_source, hz, out_dir: Path,
                 mirror: bool = True, show: bool = True):
        self.cap = cap
        self.tracker = tracker
        self.glove = glove_source
        self.hz = hz
        self.out_dir = out_dir
        self.glove_dir = out_dir / "glove"
        self.cam_dir = out_dir / "cam"
        self.mirror = mirror
        self.show = show
        self.monitors = {"left": StreamMonitor("left"), "right": StreamMonitor("right")}
        self.results = []
        self._warned = set()
        # Which hand sides each sensor has actually delivered. Readiness is
        # about the INTERSECTION, not the totals: a camera watching the left
        # hand while the glove streams the right one has plenty of both and
        # can never produce a pair.
        self.glove_sides: set = set()
        self.cam_sides: set = set()

    def make_cam_recorder(self, pose: str, take: int):
        """The recorder for this session's camera. One per take."""
        return CamRecorder(hz=self.hz, pose=pose, take=take)

    def discard_backlog(self) -> int:
        """Empty both sensors' backlogs. Returns how many frames were dropped.

        Called after the start beep and before the recorders open their
        files: see the comment in `run_take` for why that ordering is the
        whole point.
        """
        dropped = self._discard_camera_backlog()
        # The glove queue reports emptiness honestly, so drain it until it
        # says so rather than guessing a number of rounds. `_pump_glove(None)`
        # and not a raw queue drain: the StreamMonitor has to see these packet
        # counters go by, or it reports the gap we just made as a dropout.
        for _ in range(MAX_DRAIN_ROUNDS):
            n = self._pump_glove(None)
            dropped += n
            if not n:
                break
        return dropped

    def _discard_camera_backlog(self) -> int:
        """Flush the webcam's buffer without decoding or tracking anything.

        `grab()` pulls a frame off the driver's queue without decoding it,
        which is what is wanted here: the next `read_frame` should return
        what the camera sees now, not what it saw during the beep. A fixed
        small count, because `grab()` blocks for the next frame once the
        buffer is empty and so cannot tell us when to stop.
        """
        if self.cap is None:
            return 0
        return sum(1 for _ in range(CAM_BUFFER_FRAMES) if self.cap.grab())

    def _pump_glove(self, recorder=None) -> int:
        """Drain and optionally record glove packets. Returns frames seen."""
        seen = 0
        for item in self.glove.drain(64):
            hand, raw = item
            result = validate_raw_message(raw)
            if not result.is_valid:
                self._warn(f"[{hand}] invalid packet: " + "; ".join(result.errors))
                continue
            frame = parse_hand_message(raw, hand_side_hint=hand)
            for w in self.monitors[hand].update(frame.packet_counter):
                self._warn(f"[{hand}] {w}")
            seen += 1
            self.glove_sides.add(frame.hand_side)
            if recorder is not None:
                # recv_time: when the OSC packet landed, stamped on the server
                # thread. A drained burst shares a write time but not this.
                recorder.record(frame, capture_time=getattr(item, "recv_time",
                                                            None) or None)
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
        self.cam_sides.update(h.hand_side for h in hands)
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
        print("Waiting for BOTH sensors on the SAME hand "
              "(glove packets + that hand in view)...")
        t0 = time.time()
        glove_ok = cam_ok = 0
        self.glove_sides, self.cam_sides = set(), set()
        while time.time() - t0 < timeout:
            glove_ok += self._pump_glove()
            hands = self.tick(banner="SHOW YOUR HAND",
                              sub="glove on, hand inside the frame")
            if hands:
                cam_ok += 1
            if (glove_ok >= 10 and cam_ok >= 10
                    and (self.glove_sides & self.cam_sides)):
                shared = hands_text(self.glove_sides & self.cam_sides)
                print(f"  OK - glove and camera both on: {shared}\n")
                return
        raise SystemExit(self.not_ready_message(glove_ok, cam_ok))

    def not_ready_message(self, glove_ok: int, cam_ok: int) -> str:
        """Why the session will not start, naming the actual disagreement."""
        if glove_ok and cam_ok and not (self.glove_sides & self.cam_sides):
            return (f"The glove is streaming {hands_text(self.glove_sides)} "
                    f"and the camera is seeing {hands_text(self.cam_sides)} — "
                    "no hand in common, so nothing could ever pair. Put the "
                    "glove on the hand you are showing the camera (or show "
                    "the camera the gloved hand).")
        return (f"Only got {glove_ok} glove packets and {cam_ok} camera "
                "detections. Check XR Trainer is streaming "
                "(scripts/glove/run_osc.py --dump --no-viz) and the webcam "
                "(scripts/live_view.py).")

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
        cam_rec = self.make_cam_recorder(pose, take)
        glove_rec = StampedFrameRecorder(hz=self.hz, pose=pose, take=take)
        # Order matters, and it is not the obvious one. `winsound.Beep`
        # BLOCKS for its whole duration while the OSC thread and the LeapC
        # polling thread keep filling their queues behind it, so whatever is
        # drained straight after the beep is up to a quarter of a second old
        # — and both recorders stamp a frame with the time of the WRITE.
        # Recording through that would put 250 ms of backdated frames at the
        # head of every take. So: beep, throw the backlog away, and only then
        # open the files, immediately before the timed loop.
        beep(1000, 250)
        dropped = self.discard_backlog()
        cam_rec.start(self.cam_dir / name)
        glove_rec.start(self.glove_dir / name)
        if dropped:
            self._warn(f"dropped {dropped} frame(s) queued during the beep")
        print(f"      REC {duration:g} s - hold it ",
              end="" if self.dots else "\n", flush=True)
        try:
            t_end = time.time() + duration
            next_dot = time.time() + 0.5
            while time.time() < t_end:
                self.tick(cam_rec, glove_rec, banner=title, rec=True)
                if self.dots and time.time() >= next_dot:
                    print(".", end="", flush=True)
                    next_dot += 0.5
        finally:
            if self.dots:
                print(flush=True)
            cam_rec.stop()
            glove_rec.stop()
            beep(500, 300)
            # A take is usable only if ONE HAND was seen by BOTH sensors.
            # Counting frames per sensor is not enough: a camera that saw the
            # left hand and a glove that streamed the right one both report
            # plenty of frames, and `pair_by_time` never crosses hands, so the
            # take fuses into nothing. Say which hands each side had.
            shared = cam_rec.hands_seen & glove_rec.hands_seen
            entry = {"pose": pose, "take": take,
                     "cam_frames": cam_rec.count, "glove_frames": glove_rec.count,
                     "cam_hands": sorted(cam_rec.hands_seen),
                     "glove_hands": sorted(glove_rec.hands_seen),
                     "shared_hands": sorted(shared),
                     "ok": bool(shared)}
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
                    print(f"      WARNING: {describe_mismatch(cam_rec, glove_rec)}"
                          " — this take cannot be fused")
                print()
            self.results.append(entry)

    def print_summary(self) -> None:
        if not self.results:
            print("\nNothing recorded.")
            return
        ok = [r for r in self.results if r["ok"]]
        print("=" * 62)
        print(f"Session summary: {len(ok)}/{len(self.results)} takes have BOTH "
              "sensors on one hand")
        for r in self.results:
            if r["ok"]:
                status = (f"glove {r['glove_frames']:>4}f | "
                          f"cam {r['cam_frames']:>4}f  "
                          f"({hands_text(r.get('shared_hands', []))})")
            elif r["glove_frames"] and r["cam_frames"]:
                status = (f"NO SHARED HAND: glove "
                          f"{hands_text(r.get('glove_hands', []))} vs cam "
                          f"{hands_text(r.get('cam_hands', []))}")
            else:
                status = "INCOMPLETE"
            print(f"  {r['pose']:<12} take{r['take']}  {status}")
        if ok:
            print(f"\n  glove files: {self.glove_dir}")
            print(f"  cam files:   {self.cam_dir}")
            print(f"  fuse + compare:  python scripts/fuse_poses.py "
                  f"{self.out_dir}")


class LeapSyncSession(SyncSession):
    """The same protocol, with the Ultraleap Stereo IR 170 as the camera.

    Everything the MediaPipe session does around the two files — one name,
    one clock, one beep protocol, one summary — is inherited unchanged. What
    differs is what a tick is:

      * hands come from `LeapStream.drain()` (or the mock) rather than from a
        decoded video frame, and go straight into a `LeapRecorder`, so the
        camera file is the ordinary Leap JSONL every other tool already
        reads. `fuse_poses.py` recognises it by `source: "leap"`;
      * there is no window. An 850 nm brightness image is not something to
        check a pose against, and on Path A both hands are over the module
        anyway, so the feedback is a one-line HUD printed once a second;
      * a hand is skipped until it has been tracked for MIN_VISIBLE_TIME_US,
        the same settling gate the gate runner and record_poses use. LeapC's
        `confidence` is a constant 1.0 and is never consulted.
      * the camera is NOT throttled by default (`leap_hz=None`). Both
        recorders schedule the next sample from the write that just happened,
        so two recorders throttled to the same rate drift apart by the
        difference in how long each waits for its next frame — measured on a
        mock 3 s take at 5 Hz both sides, that drift reached 48 ms and cost
        two thirds of the pairs at `fuse_poses --max-dt 0.05`. Keeping every
        camera frame removes the problem instead of tuning around it: at
        90 Hz every glove frame has a partner within ~6 ms, and a 5 s take is
        a couple of megabytes. `--leap-hz` throttles it anyway if disk ever
        matters more than pairing.
    """

    dots = False                      # the HUD owns the line instead
    HUD_EVERY = 1.0

    def __init__(self, leap_source, glove_source, hz, out_dir: Path,
                 leap_hz=None):
        super().__init__(cap=None, tracker=None, glove_source=glove_source,
                         hz=hz, out_dir=out_dir, mirror=False, show=False)
        self.leap = leap_source
        self.leap_hz = leap_hz
        self.cam_dir = out_dir / "leap"
        self.skipped_young = 0
        self.glove_total = 0            # cumulative, for wait_for_both
        self.hand_total = 0
        # Start the clock now, so the first HUD line reports a real second
        # rather than the zeros of a session that has not begun.
        self._hud_at = time.time()
        self._hud_mark = (0, 0)         # the totals at the last HUD line
        self._sides: set = set()
        self._framerate = 0.0

    def make_cam_recorder(self, pose: str, take: int):
        return LeapRecorder(hz=self.leap_hz, pose=pose, take=take)

    def _discard_camera_backlog(self) -> int:
        """Empty the LeapC queue. At 90 Hz a blocking beep fills a lot of it."""
        dropped = 0
        for _ in range(MAX_DRAIN_ROUNDS):
            n = len(self.leap.drain(64))
            dropped += n
            if not n:
                break
        return dropped

    def tick(self, cam_rec=None, glove_rec=None, banner="", sub="", rec=False):
        """Drain both sensors once and record what each gave. No window."""
        self.glove_total += self._pump_glove(glove_rec)
        hands = []
        for _side, lh in self.leap.drain(64):
            if lh.framerate:
                self._framerate = float(lh.framerate)
            self._sides.add(lh.hand_side)
            self.hand_total += 1
            if lh.visible_time_us < MIN_VISIBLE_TIME_US:
                self.skipped_young += 1      # still settling; not data yet
                continue
            self.cam_sides.add(lh.hand_side)
            hands.append(lh)
            if cam_rec is not None:
                cam_rec.record(lh)
        self._print_hud(rec)
        # Nothing in this loop blocks (there is no video frame to wait on), so
        # yield the CPU rather than spin on an empty queue.
        time.sleep(0.005)
        return hands

    def _print_hud(self, rec: bool) -> None:
        """One line a second: which hands, how fast, how much of each sensor."""
        now = time.time()
        if now - self._hud_at < self.HUD_EVERY:
            return
        glove_0, hands_0 = self._hud_mark
        seen = ",".join(sorted(self._sides)) or "none"
        print(f"      {'REC' if rec else '   '} [leap] hands {seen:<11}"
              f"{self._framerate:5.1f} Hz   "
              f"{self.hand_total - hands_0:>4} hands/s, "
              f"{self.glove_total - glove_0:>4} glove/s", flush=True)
        self._hud_at = now
        self._hud_mark = (self.glove_total, self.hand_total)
        self._sides = set()

    def wait_for_both(self, timeout: float = 120.0) -> None:
        print("Waiting for BOTH sensors on the SAME hand "
              "(glove packets + that hand tracked)...")
        print("  Glove on, hand 20 to 50 cm above the module, lenses up.")
        t0 = time.time()
        self.glove_sides, self.cam_sides = set(), set()
        while time.time() - t0 < timeout:
            self.tick()
            if (self.glove_total >= 10 and self.hand_total >= 10
                    and (self.glove_sides & self.cam_sides)):
                shared = hands_text(self.glove_sides & self.cam_sides)
                print(f"  OK - glove and Ultraleap both on: {shared}\n")
                return
        raise SystemExit(self.not_ready_message(self.glove_total,
                                                self.hand_total))

    def not_ready_message(self, glove_ok: int, cam_ok: int) -> str:
        if glove_ok and cam_ok and not (self.glove_sides & self.cam_sides):
            return (f"The glove is streaming {hands_text(self.glove_sides)} "
                    f"and the camera is tracking {hands_text(self.cam_sides)} "
                    "— no hand in common, so nothing could ever pair. Hold "
                    "the GLOVED hand over the module.")
        return (f"Only got {glove_ok} glove packets and {cam_ok} tracked "
                "hands. Check XR Trainer is streaming "
                "(scripts/glove/run_osc.py --dump --no-viz) and the camera "
                "(python scripts/leap/check_setup.py).")

    def print_summary(self) -> None:
        super().print_summary()
        if self.skipped_young:
            print(f"\n  {self.skipped_young} hands skipped: tracked for less "
                  f"than {MIN_VISIBLE_TIME_US / 1000:.0f} ms (settling)")


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
    p.add_argument("--camera", default="0",
                   help="webcam index (0, 1, ...) for the MediaPipe backend, "
                        "or 'leap' for the Ultraleap Stereo IR 170 (Path A)")
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
    p.add_argument("--mock-leap", action="store_true",
                   help="synthetic Ultraleap stream; only with --camera leap")
    p.add_argument("--leap-hz", type=float, default=0.0,
                   help="camera frames saved per second with --camera leap "
                        "(default: 0 = keep every frame, which is what keeps "
                        "a glove frame's partner within a few ms)")
    p.add_argument("--mode", default="desktop",
                   choices=("desktop", "hmd", "screentop"),
                   help="Ultraleap tracking mode (--camera leap)")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()

    poses = [slugify(x) for x in args.poses.split(",") if slugify(x)]
    if not poses:
        raise SystemExit("no poses given")

    backend = str(args.camera).strip().lower()
    if backend != LEAP:
        try:
            camera_index = int(args.camera)
        except ValueError:
            raise SystemExit(
                f"--camera takes a webcam index or 'leap', not {args.camera!r}")
    if args.mock_leap and backend != LEAP:
        raise SystemExit("--mock-leap only means anything with --camera leap")

    eta = len(poses) * args.takes * (args.prep + args.duration)
    camera_text = (("MOCK leap" if args.mock_leap else "Ultraleap SIR 170")
                   if backend == LEAP else f"webcam index {camera_index}")
    print("=" * 62)
    print(f"SIMULTANEOUS session: {len(poses)} poses x {args.takes} takes "
          f"x {args.duration:g} s  (~{eta / 60:.1f} min)")
    print(f"  poses: {', '.join(poses)}")
    print(f"  glove: {'MOCK' if args.mock_glove else f'{args.host}:{args.port}'}"
          f"   camera: {camera_text}   output: {args.out_dir}")
    print("  Wear the glove AND keep the hand "
          + ("20 to 50 cm above the module." if backend == LEAP
             else "in the camera frame."))
    print("=" * 62 + "\n")

    glove = (MockGloveSource() if args.mock_glove
             else OSCHandReceiver(host=args.host, port=args.port,
                                  kinematic_addr=args.address))
    glove.start()

    cap = tracker = leap = None
    try:
        if backend == LEAP:
            try:
                leap = open_stream(mock=args.mock_leap, mode=args.mode)
            except LeapUnavailable as e:
                raise SystemExit(f"\nNo live tracking: {e}\n")
            session = LeapSyncSession(leap, glove, hz=args.hz or None,
                                      out_dir=args.out_dir,
                                      leap_hz=args.leap_hz or None)
        else:
            cap = open_camera(camera_index, args.width, args.height)
            tracker = HandTracker(model_path=args.model, running_mode="video",
                                  min_detection_confidence=args.min_det,
                                  min_tracking_confidence=args.min_det,
                                  gamma=args.gamma, clahe=args.clahe)
            session = SyncSession(cap, tracker, glove, hz=args.hz or None,
                                  out_dir=args.out_dir,
                                  mirror=not args.no_mirror,
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
            session.print_summary()
    finally:
        glove.stop()
        if leap is not None:
            leap.stop()
        if cap is not None:
            cap.release()
        if tracker is not None:
            tracker.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
