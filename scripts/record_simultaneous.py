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

With `--camera leap` it is the COACHED protocol, one hand at a time, because
the plain one does not work on a gloved hand. Measured 2026-09-17 on the left
gloved hand: `open_palm` tracked 3/3 takes on one hand id and every other pose
failed — fist 34 %/0 %/66 %, thumbs_up 0/0/0, pinch tracked only as the wrong
hand. The tracker follows an OPEN hand into a pose but cannot acquire a gloved
hand that is already closed, and when it re-acquires from a closed pose it
sometimes returns a mirrored skeleton labelled as the other hand. So each take
is: acquire the expected hand open and steady, beep, call the pose, let the
tracker follow it through the transition, and record only if the same hand id
survived. See `leap_hand.protocol` for the numbers and why they are those.

And then, because a surviving hand id says nothing about what the hand was
DOING: the take is refused if both sensors say the hand was not in the pose
that was asked for (`leap_hand.pose_check`). That check exists because the
session of 2026-09-17 produced 9 takes out of 36 holding the wrong pose, with
both sensors agreeing — the window never named the pose, so the operator was
working from memory. A refused attempt is MOVED to `<out-dir>/rejected/`,
never deleted: the two sensors being evaluated are the ones vetoing the take,
so the exclusions have to stay countable. `--no-pose-check` turns it off, and
it is off for mock sensors, whose hand shape does not follow the pose called.

Needs XR Trainer streaming to 127.0.0.1:9002 plus the camera. Rehearse the
whole thing with no hardware at all:

  python scripts/record_simultaneous.py --mock-glove --takes 1 --duration 3 --prep 2
  python scripts/record_simultaneous.py --camera leap --hand left \
      --mock-glove --mock-leap --poses fist --takes 1 --duration 3

Usage:
  python scripts/record_simultaneous.py                       # 6 poses x 3 takes
  python scripts/record_simultaneous.py --poses pinch,fist --takes 2
  python scripts/record_simultaneous.py --camera leap --hand left   # Path A
  python scripts/record_simultaneous.py --camera leap --hand both   # uncoached
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2

from cam_hand.capture import open_camera, read_frame
from cam_hand.draw import draw_banner, draw_hand, draw_hud, label_hands
from cam_hand.landmarks import DEFAULT_MODEL, HandTracker
from cam_hand.recorder import CamRecorder, pose_filename, slugify
from cam_hand.recorder import finalize_pose_name as cam_finalize
from cam_hand.recorder import hand_tag as cam_hand_tag

# The Ultraleap backend (--camera leap). Importing these is free: leap_hand
# only touches the `leap` bindings inside LeapStream.start().
from leap_hand.protocol import (
    COMPLETE_COVERAGE,
    DEFAULT_BAND,
    DEFAULT_RETRIES,
    DEFAULT_SETTLE,
    WRONG_POSE,
    AsyncBeeper,
    Hud,
    acquire_failures,
    acquire_prompt,
    band_text,
    coverage,
    hud_line,
    median,
    parse_band,
    pose_label,
    read_hand,
    stream_health,
    view_caption,
)
from leap_hand.pose_check import (
    DEFAULT_PARAMS,
    MISMATCH,
    UNCHECKED,
    PoseCheck,
    check_pose,
    read_take,
    short_summary,
)
from leap_hand.recorder import LeapRecorder
from leap_hand.stream import LeapUnavailable, open_stream

# The glove side comes from the xr_hand package in this repo.
from xr_hand.parser import parse_hand_message
from xr_hand.receiver import OSCHandReceiver, QueueItem
from xr_hand.recorder import FrameRecorder
from xr_hand.validator import StreamMonitor, validate_raw_message

DEFAULT_POSES = ["open_palm", "fist", "index_point", "thumbs_up", "peace", "pinch"]
LEAP = "leap"
BOTH = "both"
MIN_VISIBLE_TIME_US = 300_000     # plan section 6: a hand counts after 0.3 s
# A tracked hand goes stale on the HUD this long after its last frame, so the
# line says "no hand" while the hand is actually gone rather than freezing on
# the last good reading. Well under the loss threshold below.
HUD_STALE_S = 0.30
# No frame from the acquired hand for this long IS a loss, not a dropped
# frame: at 90 Hz it is 45 missed tracking intervals in a row.
LOST_S = 0.50
# Bounds on the post-beep flush (see SyncSession.discard_backlog). A 250 ms
# beep leaves at most ~45 Leap hands and ~30 glove packets behind it, so a
# handful of drain(64) rounds always clears it; the bound is only there so a
# sensor that never stops delivering cannot pin us here.
MAX_DRAIN_ROUNDS = 8
CAM_BUFFER_FRAMES = 2             # webcam frames to grab and drop
# How long "WRONG POSE: saw X, want Y" stays on the window before the retry
# starts. Long enough to read while your hands are still over the module.
WRONG_POSE_SECONDS = 2.0
# A glove hole longer than this is called out after the take. 250 ms is 15
# frames of a 60 Hz stream in a row: a dropout, not jitter.
GLOVE_GAP_WARN_S = 0.25
# Where a rejected attempt goes. NOT deleted: the two sensors under
# evaluation are the ones judging the take, so every exclusion has to stay on
# disk where it can be counted, looked at and argued with.
REJECTED = "rejected"
STILLS = "stills"
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
            if recorder is not None and self.accept_glove(frame):
                # recv_time: when the OSC packet landed, stamped on the server
                # thread. A drained burst shares a write time but not this.
                recorder.record(frame, capture_time=getattr(item, "recv_time",
                                                            None) or None)
        return seen

    def accept_glove(self, frame) -> bool:
        """Does this glove frame belong in the take? Everything does, here.

        XR Trainer streams both gloves whatever the session is doing, so a
        one-hand session overrides this — see `CoachedLeapSession`.
        """
        return True

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

    def finish(self) -> None:
        """Leave the terminal usable. Overridden where there is a live HUD."""

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
        seen = ",".join(sorted(self._sides))
        # The stream counts every tracking event, hand or no hand, so the HUD
        # can tell "camera running, nothing in view" from "camera dead". A
        # bare "0.0 Hz" read as a dead camera and cost a whole first session.
        cam_frames = int(getattr(self.leap, "frames", 0) or 0)
        cam_new = cam_frames - getattr(self, "_cam_frames_mark", 0)
        self._cam_frames_mark = cam_frames
        rate = float(getattr(self.leap, "framerate", 0.0) or self._framerate)
        if seen:
            cam_text = f"tracking {seen}"
        elif cam_new > 0 or rate > 0:
            cam_text = "running, NO HAND IN VIEW"
        else:
            cam_text = "NO FRAMES - check the camera"
        print(f"      {'REC' if rec else '   '} camera: {cam_text:<28}"
              f"{rate:5.1f} Hz {self.hand_total - hands_0:>4} hands/s"
              f"  |  glove: {self.glove_total - glove_0:>4} packets/s",
              flush=True)
        self._hud_at = now
        self._hud_mark = (self.glove_total, self.hand_total)
        self._sides = set()

    def wait_for_both(self, timeout: float = 120.0) -> None:
        print("Waiting for BOTH sensors on the SAME hand "
              "(glove packets + that hand tracked)...")
        print("  Glove on, hand 20 to 50 cm above the module, lenses up.")
        print("  >>> HOLD THE GLOVED HAND OVER THE CAMERA NOW, palm down. <<<")
        print("  The session starts by itself the moment the camera tracks it.")
        t0 = time.time()
        nagged = t0
        self.glove_sides, self.cam_sides = set(), set()
        while time.time() - t0 < timeout:
            self.tick()
            if not self.cam_sides and time.time() - nagged > 6.0:
                nagged = time.time()
                beep(440, 120)
                print("  ... still no hand tracked: 20 to 30 cm above the "
                      "lenses, palm facing down, fingers open.", flush=True)
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


@dataclass
class Attempt:
    """One try at one take: what it produced, or why it was thrown away."""

    complete: bool = False
    why: str = ""
    hand_id: Optional[int] = None
    coverage: float = 0.0
    cam_frames: int = 0
    glove_frames: int = 0
    median_height_cm: Optional[float] = None
    median_view_deg: Optional[float] = None
    rejected_chirality: int = 0
    second_hand: int = 0
    file: str = ""
    # Was the hand actually in the pose that was asked for? See
    # `leap_hand.pose_check`; None until the take has survived everything else.
    check: Optional[PoseCheck] = None
    # How the glove stream behaved over the take: rate, longest hole, holes.
    glove_health: Optional[dict] = None
    # The one still the camera window saved, and where the attempt's files
    # ended up if it was rejected. Both relative to the session folder.
    still: str = ""
    rejected_to: str = ""


@dataclass
class TakeResult:
    """Every attempt at one take, and the one that stuck."""

    pose: str
    take: int
    attempts: int = 0
    complete: bool = False
    why: str = ""
    final: Optional[Attempt] = None
    tries: List[Attempt] = field(default_factory=list)


class Quit(Exception):
    """The operator asked to stop (Ctrl+C), mid-acquire."""


from leap_hand.protocol import CameraView  # noqa: E402  (the live camera window)


class CoachedLeapSession(LeapSyncSession):
    """One hand at a time: acquire it OPEN, call the pose, verify the track.

    The uncoached session records whatever the tracker hands it for
    `--duration` seconds. On a gloved hand that produced takes in which the
    pose was never tracked at all, or was tracked as the other hand — a
    mirrored skeleton — and nobody knew until the files were analysed. This
    session refuses to produce those files. Per take:

      ACQUIRE   the expected hand, OPEN, held for half a second, inside the
                height band, palm toward the lens and roughly over the module.
                Nothing is recorded and no pose is asked for until all of that
                holds at once, because the measured failure is that the
                tracker cannot pick up a gloved hand that is already closed.
      SETTLE    a beep, "NOW: <pose>", and `--settle` seconds for the hand to
                change shape while the tracker follows it. The hand id is
                pinned at acquire; if it changes here, the tracker let go and
                found the hand again, which is exactly the re-acquisition that
                guesses the wrong chirality.
      REC       `--duration` seconds, and only if that same id is still on the
                hand. A take counts as complete when the expected hand, on
                that one id, covers >= 90 % of it.

    Anything less is discarded — both files, so a failed attempt leaves no
    half-take on disk to be analysed later by mistake — and retried from
    ACQUIRE up to `--retries` times before the take is marked failed.

    Chirality is enforced on every frame, not checked afterwards: a camera
    hand of the other side is counted and dropped, never written. That is the
    one failure the old files could not be rescued from, because a mirrored
    left hand labelled `right` fuses against the right glove and produces a
    plausible, wrong result.
    """

    dots = False

    def __init__(self, leap_source, glove_source, hz, out_dir: Path,
                 leap_hz=None, hand: str = "left", band=DEFAULT_BAND,
                 settle: float = DEFAULT_SETTLE, retries: int = DEFAULT_RETRIES,
                 acquire_timeout: float = 60.0, pose_check: bool = True,
                 pose_check_params=DEFAULT_PARAMS):
        super().__init__(leap_source, glove_source, hz=hz, out_dir=out_dir,
                         leap_hz=leap_hz)
        self.hand = hand
        self.band = band
        self.settle = float(settle)
        self.retries = int(retries)
        self.acquire_timeout = float(acquire_timeout)
        self.pose_check = bool(pose_check)
        # Named and passed in rather than reached for, exactly like
        # `cam_hand.fusion.GateParams`: a threshold nobody can substitute is a
        # threshold nobody can test against a different hand.
        self.pose_check_params = pose_check_params
        self.rejected_dir = out_dir / REJECTED
        self.stills_dir = out_dir / STILLS
        # Every attempt the pose check threw out, as (pose, what was seen).
        # Printed per pose at the end: the sensors are the arbiter here, so
        # how often they vetoed a take is part of the session's result.
        self.pose_rejects: List[tuple] = []

        # Counted for the whole session and per attempt (see `_mark`).
        self.rejected_chirality = 0      # camera hands of the OTHER side
        self.second_hand = 0             # expected side, but not our hand id
        self.glove_other_hand = 0        # glove packets from the other hand
        self.takes: List[TakeResult] = []

        self.beeper = AsyncBeeper(beep)
        self.hud = Hud(self._write)
        self.view = None                 # CameraView, set by main() for a real camera
        self._phase = "start"
        self._deadline: Optional[float] = None
        self._hud_extra = ""
        # What the window is currently asking for. Held on the session because
        # the HUD refreshes four times a second from `tick`, nowhere near the
        # code that knows which take is running — which is exactly why the
        # window used to say "SETTLE   1s" and never name the pose at all.
        self._pose = ""
        self._take: Optional[int] = None
        self._takes: Optional[int] = None
        # The latest reading of the expected hand, and when it arrived. Held
        # with its timestamp rather than cleared, so the HUD can tell "the
        # hand is gone" from "no frame has been drained this millisecond".
        self._reading = None
        self._reading_at = 0.0
        self._other_at = 0.0
        self._pinned_id: Optional[int] = None
        self._pinned_at = 0.0
        self._changed_to: Optional[int] = None
        self._heights: List[float] = []
        self._angles: List[float] = []
        self._times: List[float] = []
        self._glove_hz = 0.0
        self._rate_at = time.time()
        self._rate_mark = 0

    # --- plumbing --------------------------------------------------------
    @staticmethod
    def _write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def accept_glove(self, frame) -> bool:
        """Only the hand this session is about. See the class docstring.

        XR Trainer streams both gloves regardless, and a take whose camera
        file holds one hand and whose glove file holds two is tagged `both`
        and half of it can never pair.
        """
        if frame.hand_side == self.hand:
            return True
        self.glove_other_hand += 1
        return False

    def tick(self, cam_rec=None, glove_rec=None, banner="", sub="", rec=False):
        """Drain both sensors once. The chirality gate lives here.

        This is the only place a camera hand can reach a recorder, which is
        why the wrong-hand and second-hand checks are here and not in a
        post-pass: a frame that is never accepted here is never written, and
        `--hand left` therefore cannot produce a right-handed line.
        """
        self.glove_total += self._pump_glove(glove_rec)
        hands = []
        now = time.time()
        for _side, lh in self.leap.drain(64):
            if lh.framerate:
                self._framerate = float(lh.framerate)
            self._sides.add(lh.hand_side)
            self.hand_total += 1
            if lh.hand_side != self.hand:
                self.rejected_chirality += 1
                self._other_at = now
                continue
            if self._pinned_id is not None and lh.hand_id != self._pinned_id:
                # Right side, wrong hand: either a second hand in the field or
                # the tracker re-acquiring ours under a new id. Which one it is
                # comes out of `_lost_reason`, which knows whether the pinned
                # id is still delivering.
                self.second_hand += 1
                self._changed_to = lh.hand_id
                continue
            reading = read_hand(lh)
            self._reading, self._reading_at = reading, now
            if lh.visible_time_us < MIN_VISIBLE_TIME_US:
                self.skipped_young += 1
                continue
            self.cam_sides.add(lh.hand_side)
            hands.append(lh)
            if self._pinned_id is not None:
                self._pinned_at = now
            if cam_rec is not None:
                cam_rec.record(lh)
                self._heights.append(reading.height_cm)
                if reading.view_angle_deg is not None:
                    self._angles.append(reading.view_angle_deg)
                self._times.append(float(lh.capture_time)
                                   if lh.capture_time is not None else now)
        self._show_hud(now)
        # Nothing in this loop blocks — the beeps are on their own thread — so
        # yield the CPU rather than spin on an empty queue.
        time.sleep(0.005)
        return hands

    def _print_hud(self, rec: bool) -> None:
        """The once-a-second HUD of the uncoached session. Replaced here."""

    def _show_hud(self, now: float) -> None:
        fresh = (self._reading
                 if now - self._reading_at < HUD_STALE_S else None)
        if now - self._rate_at >= 1.0:
            self._glove_hz = ((self.glove_total - self._rate_mark)
                              / (now - self._rate_at))
            self._rate_at, self._rate_mark = now, self.glove_total
        left = None if self._deadline is None else self._deadline - now
        if self.view is not None:
            self.view.caption(
                view_caption(self._phase, self._pose, self._hud_extra, left,
                             self._take, self._takes),
                band=self.band)
        # The terminal line names the pose too, in the phases where the window
        # used to be the only thing that could have.
        extra = self._hud_extra
        if not extra and self._phase in ("SETTLE", "REC") and self._pose:
            extra = f"HOLD: {pose_label(self._pose)}"
        self.hud.show(hud_line(self._phase, left, fresh, self.hand, self.band,
                               self._glove_hz,
                               saw_other_hand=now - self._other_at < HUD_STALE_S,
                               extra=extra), now)

    def _say(self, *lines: str) -> None:
        """Print above the HUD line, leaving the HUD to reopen underneath."""
        self.hud.close()
        for line in lines:
            print(line, flush=True)

    # --- readiness -------------------------------------------------------
    def wait_for_both(self, timeout: float = 120.0) -> None:
        """Wait for the GLOVE only; the camera is the ACQUIRE phase's job.

        The uncoached session will not start until both sensors have seen the
        same hand. Here that check would be in the wrong place twice over: the
        camera side is re-checked, harder, at the start of every single take,
        and blocking on it up front means an operator with no hand over the
        module gets a timeout instead of the ACQUIRE prompt telling them what
        to do about it.
        """
        print(f"Waiting for the glove on the {self.hand.upper()} hand "
              f"({self.host_text()})...")
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.glove_total += self._pump_glove()
            time.sleep(0.01)
            if self.glove_total >= 10 and self.hand in self.glove_sides:
                print(f"  OK - glove streaming: {self.hand}\n")
                return
        raise SystemExit(self.not_ready_message(self.glove_total,
                                                self.hand_total))

    def host_text(self) -> str:
        return "mock" if isinstance(self.glove, MockGloveSource) else "OSC"

    def not_ready_message(self, glove_ok: int, cam_ok: int) -> str:
        if glove_ok and self.hand not in self.glove_sides:
            return (f"The glove is streaming {hands_text(self.glove_sides)}, "
                    f"not {self.hand}. Put the glove on the {self.hand} hand, "
                    f"or re-run with --hand "
                    f"{'right' if self.hand == 'left' else 'left'}.")
        return (f"Only got {glove_ok} glove packets. Check XR Trainer is "
                "streaming (scripts/glove/run_osc.py --dump --no-viz).")

    # --- one attempt -----------------------------------------------------
    def _mark(self) -> tuple:
        """The two per-frame rejection counters, for differencing."""
        return (self.rejected_chirality, self.second_hand)

    def _lost_reason(self, now: float) -> str:
        """Why the acquired hand is no longer the acquired hand, or ""."""
        if self._changed_to is not None and now - self._pinned_at > LOST_S:
            return (f"the tracker let go and re-acquired the hand "
                    f"(id {self._pinned_id} -> {self._changed_to})")
        if now - self._pinned_at > LOST_S:
            return (f"the {self.hand} hand was lost for "
                    f"{now - self._pinned_at:.1f} s")
        return ""

    def _acquire(self, pose: str, attempt: int) -> None:
        """Block until the expected hand is open, steady and well placed."""
        self._pinned_id, self._changed_to = None, None
        self._reading, self._reading_at = None, 0.0
        self._phase, self._deadline = "ACQUIRE", None
        self._say(*[f"      {line}" for line in acquire_prompt(pose, self.hand)],
                  f"      Height band {band_text(self.band)}; "
                  f"attempt {attempt}/{self.retries + 1}.")
        t0 = time.time()
        while True:
            self.tick()
            now = time.time()
            fresh = (self._reading
                     if now - self._reading_at < HUD_STALE_S else None)
            bad = acquire_failures(fresh, self.hand, self.band)
            self._hud_extra = "" if not bad else "need: " + ", ".join(bad)
            if not bad:
                self._pinned_id = fresh.hand_id
                self._pinned_at = now
                self._changed_to = None
                self._hud_extra = ""
                return
            if now - t0 > self.acquire_timeout:
                raise TimeoutError(
                    f"no acquirable {self.hand} hand in "
                    f"{self.acquire_timeout:g} s ({self._hud_extra})")

    def _hold(self, phase: str, seconds: float, cam_rec=None,
              glove_rec=None, snap=None) -> str:
        """Tick for `seconds`, watching the pinned hand. "" = it survived.

        `snap` is a path for the camera window to save one composed frame to,
        taken at the MIDPOINT of the phase: the hand is settled by then and
        the take is not yet over, so the still shows the pose that the two
        files claim to contain. It is the only evidence in the session that
        does not come from the same two sensors that judge it.
        """
        self._phase = phase
        t_end = time.time() + seconds
        self._deadline = t_end
        midpoint = t_end - seconds / 2.0
        while time.time() < t_end:
            self.tick(cam_rec, glove_rec)
            now = time.time()
            if snap and now >= midpoint:
                if self.view is not None:
                    self.view.snapshot(snap)
                snap = None
            reason = self._lost_reason(now)
            if reason:
                return reason
        return ""

    def _attempt(self, pose: str, take: int, duration: float,
                 attempt: int) -> Attempt:
        self._acquire(pose, attempt)
        before = self._mark()

        # The beep goes to a thread: this one is the capture clock now, and a
        # quarter second of blocked frame loop in the middle of a take is a
        # quarter second of hand that nobody recorded.
        self.beeper.beep(1000, 200)
        self._say(f"      NOW: {pose.replace('_', ' ').upper()}  "
                  f"({self.settle:g} s to change shape, "
                  f"hand id {self._pinned_id})")
        lost = self._hold("SETTLE", self.settle)
        if lost:
            return Attempt(why=f"{lost} during the settle",
                           hand_id=self._pinned_id,
                           rejected_chirality=self.rejected_chirality - before[0],
                           second_hand=self.second_hand - before[1])

        name = pose_filename(pose, take)          # one name, two files
        stem = name[:-len(".jsonl")]
        still = self.stills_dir / f"{stem}.jpg"
        cam_rec = self.make_cam_recorder(pose, take)
        glove_rec = StampedFrameRecorder(hz=self.hz, pose=pose, take=take)
        self._heights, self._angles, self._times = [], [], []
        self.discard_backlog()
        cam_rec.start(self.cam_dir / name)
        glove_rec.start(self.glove_dir / name)
        self.beeper.beep(1400, 120)
        t0 = time.time()
        try:
            lost = self._hold("REC", duration, cam_rec, glove_rec, snap=still)
        except BaseException:
            # Ctrl+C included. An interrupted take is an unverified take, and
            # the whole point of this session is that unverified takes do not
            # reach the disk. Deleted rather than moved to `rejected/`: the
            # operator stopped mid-recording, so there is no verdict to audit.
            cam_rec.stop()
            glove_rec.stop()
            self._deadline = None
            (self.cam_dir / name).unlink(missing_ok=True)
            (self.glove_dir / name).unlink(missing_ok=True)
            still.unlink(missing_ok=True)
            raise
        cam_rec.stop()
        glove_rec.stop()
        self.beeper.beep(500, 200)
        self._deadline = None
        t1 = time.time()

        got = Attempt(
            hand_id=self._pinned_id,
            coverage=coverage(self._times, t0, min(t1, t0 + duration)),
            cam_frames=cam_rec.count, glove_frames=glove_rec.count,
            median_height_cm=median(self._heights),
            median_view_deg=median(self._angles),
            rejected_chirality=self.rejected_chirality - before[0],
            second_hand=self.second_hand - before[1],
            still=still.name if still.is_file() else "",
        )
        # Read the glove file back rather than counting frames: `glove_frames`
        # cannot tell a steady stream from one that stopped for three seconds
        # in the middle, and a take whose glove went quiet is a take whose
        # fused numbers are an interpolation nobody asked for.
        glove_read = read_take(self.glove_dir / name, self.hand)
        got.glove_health = stream_health(glove_read.times, t0,
                                         min(t1, t0 + duration))
        if lost:
            got.why = f"{lost} during the take"
        elif got.coverage < COMPLETE_COVERAGE:
            got.why = (f"the {self.hand} hand covered only "
                       f"{got.coverage * 100:.0f} % of the take "
                       f"(need {COMPLETE_COVERAGE * 100:.0f} %)")
        else:
            got.complete = True

        # Only now, on a take that survived the tracking checks, is it worth
        # asking the expensive question: was the hand in the right SHAPE?
        if got.complete:
            got.check = self._check_shape(pose, name, glove_read)
            if got.check.verdict == MISMATCH:
                got.complete = False
                got.why = got.check.description
                self.pose_rejects.append((pose, got.check.seen))
                self._announce_wrong_pose(got.check)

        if not got.complete:
            got.rejected_to = self._reject(name, stem, attempt, got)
            return got

        final = cam_finalize(self.cam_dir / name, {self.hand})
        cam_finalize(self.glove_dir / name, {self.hand})
        got.file = final.name
        got.still = self._rename_still(still, final.stem)
        if (got.glove_health.get("max_gap_ms") or 0) > GLOVE_GAP_WARN_S * 1000:
            self._say(f"      ! the glove stream stopped for "
                      f"{got.glove_health['max_gap_ms']:.0f} ms during this "
                      f"take ({got.glove_health['rate_hz']} Hz overall)")
        return got

    # --- the pose check --------------------------------------------------
    def _check_shape(self, pose: str, name: str, glove_read) -> PoseCheck:
        """Is the hand in `pose`? Read both files back and ask `pose_check`.

        The recorders keep nothing in memory — one line is built, written and
        forgotten — so the take is read back off the disk. That is also the
        honest thing to check: it verifies the bytes that were actually
        written, not a parallel copy of them.
        """
        if not self.pose_check:
            return PoseCheck(
                pose=pose, verdict=UNCHECKED,
                description="the pose check is off for this session")
        return check_pose(pose, glove_read,
                          read_take(self.cam_dir / name, self.hand),
                          self.pose_check_params)

    def _announce_wrong_pose(self, check: PoseCheck) -> None:
        """Say it on the window and in the terminal, long enough to read."""
        self._say(f"      WRONG POSE: {short_summary(check)}",
                  f"      {check.description}")
        self._phase, self._hud_extra = WRONG_POSE, short_summary(check)
        self._deadline = None
        t_end = time.time() + WRONG_POSE_SECONDS
        while time.time() < t_end:
            self.tick()
        self._hud_extra = ""

    # --- what happens to an attempt that did not stand --------------------
    def _reject(self, name: str, stem: str, attempt: int, got: Attempt) -> str:
        """Move both files of a failed attempt under `rejected/`. Never delete.

        The old behaviour was to unlink them, which was right while the only
        thing that could fail a take was the tracker losing the hand. It is
        not right now that the two sensors being evaluated also decide which
        takes survive: an exclusion nobody can look at is an exclusion nobody
        can check. So the files move, keep their name plus which attempt they
        were, and get a `meta.json` saying `accepted: false` and why.
        """
        target = f"{stem}_attempt{attempt}"
        moved = []
        for folder in (self.cam_dir, self.glove_dir):
            src = folder / name
            if not src.is_file():
                continue
            dst = self.rejected_dir / folder.name / f"{target}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
            moved.append(dst)
        still = self.stills_dir / f"{stem}.jpg"
        still_name = ""
        if still.is_file():
            dst = self.rejected_dir / STILLS / f"{target}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            still.replace(dst)
            still_name = dst.name
        if not moved:
            return ""
        meta = self.rejected_dir / self.cam_dir.name / f"{target}.meta.json"
        meta.write_text(json.dumps(
            self._meta_dict(got, accepted=False, file=f"{target}.jsonl",
                            still=still_name, attempts=attempt),
            indent=2) + "\n", encoding="utf-8")
        return str(Path(REJECTED) / self.cam_dir.name / f"{target}.jsonl")

    @staticmethod
    def _rename_still(still: Path, final_stem: str) -> str:
        """Keep the still's name matched to the take's, as the pair is."""
        if not still.is_file():
            return ""
        final = still.with_name(f"{final_stem}.jpg")
        still.replace(final)
        return final.name

    # --- one take --------------------------------------------------------
    def run_take(self, pose: str, take: int, n_takes: int, pose_idx: int,
                 n_poses: int, duration: float, prep: float) -> None:
        self._warned = set()
        title = pose.replace("_", " ").upper()
        self._pose, self._take, self._takes = pose, take, n_takes
        self._say("", f"--- Pose {pose_idx}/{n_poses}: {title}  "
                      f"(take {take}/{n_takes}, {self.hand} hand) ---")
        result = TakeResult(pose=pose, take=take)
        try:
            for attempt in range(1, self.retries + 2):
                result.attempts = attempt
                got = self._attempt(pose, take, duration, attempt)
                result.tries.append(got)
                if got.complete:
                    result.complete, result.final = True, got
                    self._say(f"      OK  {got.file}  coverage "
                              f"{got.coverage * 100:.0f} %, hand id "
                              f"{got.hand_id}, {got.cam_frames} camera / "
                              f"{got.glove_frames} glove frames")
                    break
                self._say(f"      DISCARDED: {got.why}")
                if attempt <= self.retries:
                    self._say(f"      retrying ({attempt}/{self.retries})")
            if not result.complete:
                result.why = result.tries[-1].why if result.tries else "no attempt"
                result.final = result.tries[-1] if result.tries else None
                self._say(f"      FAILED after {result.attempts} attempt(s): "
                          f"{result.why}")
        except TimeoutError as e:
            result.why = str(e)
            self._say(f"      FAILED: {e}")
        except (KeyboardInterrupt, QuitSession):
            # The take still goes in the summary, with the reason it has —
            # a blank "last failure" reads as a bug in the recorder.
            result.why = "interrupted before the take finished"
            raise
        finally:
            self.hud.close()
            self._phase, self._deadline, self._hud_extra = "idle", None, ""
            self._pinned_id = None
            self.takes.append(result)
            self._write_meta(result)
            self._pose = ""

    def _meta_dict(self, got: Attempt, accepted: bool, file: str,
                   still: str, attempts: int) -> dict:
        """Everything known about one attempt, accepted or not.

        One builder for both, so a rejected attempt is described in exactly
        the same terms as a kept one and the two can be counted together.
        `accepted` is the only field that says which it is.
        """
        health = got.glove_health or {}
        return {
            "accepted": bool(accepted),
            "hand": self.hand,
            "pose": self._pose,
            "take": self._take,
            "hand_id": got.hand_id,
            "coverage": round(got.coverage, 4),
            "attempts": attempts,
            "why": got.why,
            "median_height_cm": (None if got.median_height_cm is None
                                 else round(got.median_height_cm, 1)),
            "median_view_angle_deg": (None if got.median_view_deg is None
                                      else round(got.median_view_deg, 1)),
            "rejected_chirality": got.rejected_chirality,
            "second_hand_frames": got.second_hand,
            "band_cm": list(self.band),
            "settle_s": self.settle,
            "cam_frames": got.cam_frames,
            "glove_frames": got.glove_frames,
            # The glove stream's own health over this take: the rate it
            # actually delivered at, its longest hole, and how many holes.
            "glove_rate_hz": health.get("rate_hz"),
            "glove_max_gap_ms": health.get("max_gap_ms"),
            "glove_gaps_over_100ms": health.get("gaps_over"),
            "pose_check": (got.check.as_dict() if got.check is not None
                           else None),
            # Relative to the session folder, and it never leaves it: a still
            # of the IR image has the operator in it.
            "still": (str((Path(STILLS) if accepted
                           else Path(REJECTED) / STILLS) / still)
                      if still else None),
            "file": file,
        }

    def _write_meta(self, result: TakeResult) -> None:
        """`<take>.meta.json`: how the take was got, beside the take itself.

        Only for a take that produced a file — a rejected attempt gets its own
        meta.json under `rejected/` at the moment it is rejected, which is
        where the ones that did not stand are counted from.
        """
        got = result.final
        if not (result.complete and got and got.file):
            return
        path = self.cam_dir / (got.file[:-len(".jsonl")] + ".meta.json")
        path.write_text(json.dumps(
            self._meta_dict(got, accepted=True, file=got.file,
                            still=got.still, attempts=result.attempts),
            indent=2) + "\n", encoding="utf-8")

    # --- the end ---------------------------------------------------------
    def finish(self) -> None:
        """End the HUD line and let the queued beeps drain. Ctrl+C safe."""
        self.hud.close()
        self.beeper.stop()

    def redo_command(self, failed: List[str], takes: int) -> str:
        return ("python scripts/record_simultaneous.py --camera leap "
                f"--hand {self.hand} --poses {','.join(failed)} "
                f"--takes {takes}")

    def print_summary(self) -> None:
        self.hud.close()
        if not self.takes:
            print("\nNothing recorded.")
            return
        by_pose: dict = {}
        for r in self.takes:
            by_pose.setdefault(r.pose, []).append(r)
        done = sum(1 for r in self.takes if r.complete)
        print("=" * 62)
        print(f"Session summary ({self.hand} hand): {done}/{len(self.takes)} "
              "takes complete")
        for pose, rows in by_pose.items():
            ok = [r for r in rows if r.complete]
            note = ""
            if len(ok) < len(rows):
                last = next(r.why for r in reversed(rows) if not r.complete)
                note = f"   last failure: {last}"
            attempts = sum(r.attempts for r in rows)
            print(f"  {pose:<12} {len(ok)}/{len(rows)} complete "
                  f"({attempts} attempt(s)){note}")
        if self.pose_rejects:
            by_pose: dict = {}
            for pose, seen in self.pose_rejects:
                by_pose.setdefault(pose, []).append(seen)
            print(f"\n  attempts rejected by the pose check: "
                  f"{len(self.pose_rejects)}  (kept under "
                  f"{self.rejected_dir.name}\\, never deleted)")
            for pose, seens in by_pose.items():
                shapes = ", ".join(sorted(set(seens)))
                print(f"    {pose:<12} {len(seens):>2}   the hand was "
                      f"{shapes.lower()}")
        elif self.pose_check:
            print("\n  attempts rejected by the pose check: 0")
        print(f"\n  camera hands dropped as the wrong chirality: "
              f"{self.rejected_chirality}")
        print(f"  camera hands dropped as a second hand in view:  "
              f"{self.second_hand}")
        if self.glove_other_hand:
            print(f"  glove packets dropped from the other hand:     "
                  f"{self.glove_other_hand}")
        if self.skipped_young:
            print(f"  camera hands skipped while settling (<"
                  f"{MIN_VISIBLE_TIME_US / 1000:.0f} ms): {self.skipped_young}")
        if done:
            print(f"\n  glove files: {self.glove_dir}")
            print(f"  cam files:   {self.cam_dir}   (+ <take>.meta.json)")
            if self.stills_dir.is_dir():
                print(f"  stills:      {self.stills_dir}   (one per take, "
                      "what the window saw mid-record)")
            print(f"  fuse + compare:  python scripts/fuse_poses.py "
                  f"{self.out_dir}")
            print(f"  re-check labels: python scripts/check_take_labels.py "
                  f"{self.out_dir}")
        if self.rejected_dir.is_dir():
            print(f"\n  rejected attempts kept in: {self.rejected_dir}"
                  "   (fuse_poses and check_take_labels ignore it)")
        failed = [p for p, rows in by_pose.items()
                  if any(not r.complete for r in rows)]
        if failed:
            takes = max(sum(1 for r in by_pose[p] if not r.complete)
                        for p in failed)
            print("\n  Redo ONLY the poses that failed:")
            print(f"    {self.redo_command(failed, takes)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Guided pose session recording glove and camera together.")
    p.add_argument("--poses", default=",".join(DEFAULT_POSES))
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--prep", type=float, default=5.0)
    p.add_argument("--hz", type=float, default=None,
                   help="glove frames saved per second per hand (0 = all). "
                        "Default: every frame with --camera leap, 5 with the "
                        "webcam backend")
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
    p.add_argument("--hand", default=None, choices=("left", "right", BOTH),
                   help="which hand this session is about. REQUIRED with "
                        "--camera leap: 'left' or 'right' runs the coached "
                        "one-hand protocol (acquire open, then pose) and "
                        "records only that hand; 'both' is the old uncoached "
                        "behaviour. Ignored by the webcam backend.")
    p.add_argument("--settle", type=float, default=DEFAULT_SETTLE,
                   help="seconds to change from open palm into the pose while "
                        f"the tracker follows (default: {DEFAULT_SETTLE:g})")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help="retries per take after a lost or re-acquired hand, "
                        f"so 1 + this many attempts (default: {DEFAULT_RETRIES})")
    p.add_argument("--band", default=None,
                   help="palm height band LOW,HIGH in centimetres for the "
                        f"coached HUD (default: {DEFAULT_BAND[0]:g},"
                        f"{DEFAULT_BAND[1]:g})")
    p.add_argument("--acquire-timeout", type=float, default=60.0,
                   help="seconds to wait for an acquirable hand before the "
                        "take is marked failed (default: 60)")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window (--camera leap; it "
                        "is never opened for --mock-leap)")
    p.add_argument("--no-pose-check", action="store_true",
                   help="record the take even when the hand is not in the "
                        "pose that was asked for (the check is off for mock "
                        "sensors either way — see below)")
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
    if backend == LEAP and args.hand is None:
        raise SystemExit(
            "--hand is required with --camera leap. The tracker cannot pick "
            "up a gloved hand that is already in a pose, and when it "
            "re-acquires one it sometimes returns the mirror image labelled "
            "as the other hand, so a session has to say which hand it is "
            "about:\n"
            "  --hand left     coached, one hand, nothing else recorded\n"
            "  --hand right    the same for the right hand\n"
            "  --hand both     the old uncoached behaviour, both hands kept")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if args.settle < 0:
        raise SystemExit("--settle cannot be negative")
    try:
        band = parse_band(args.band) if args.band else DEFAULT_BAND
    except ValueError as e:
        raise SystemExit(str(e))
    coached = backend == LEAP and args.hand in ("left", "right")

    # The glove is kept at FULL RATE on the leap backend. It used to be
    # throttled to 5 Hz like the webcam session, which is 24 frames in a
    # five-second take — too few to tell a steady stream from one that
    # stopped for three seconds in the middle, which is a dropout we have
    # since measured. The camera is the reason the throttle existed (a 90 Hz
    # camera and a 5 Hz glove pair badly), and `--leap-hz` handles that side.
    if args.hz is None:
        glove_hz = None if backend == LEAP else 5.0
    else:
        glove_hz = args.hz or None

    # A mock hand's shape does not follow the pose being called: the leap mock
    # cycles four cartoon poses on its own 4 s timer and the glove mock's curl
    # is a sine wave. Checking a rehearsal against the pose it was asked for
    # would fail every take for a reason that has nothing to do with the
    # operator, so the check is off whenever either sensor is a mock. That is
    # honest about what a mock is; it is not a loophole, because a mock
    # session produces no data anyone analyses.
    mocked = args.mock_leap or args.mock_glove
    pose_check_on = coached and not args.no_pose_check and not mocked

    eta = len(poses) * args.takes * (args.prep + args.duration)
    camera_text = (("MOCK leap" if args.mock_leap else "Ultraleap SIR 170")
                   if backend == LEAP else f"webcam index {camera_index}")
    print("=" * 62)
    print(f"SIMULTANEOUS session: {len(poses)} poses x {args.takes} takes "
          f"x {args.duration:g} s  (~{eta / 60:.1f} min)")
    print(f"  poses: {', '.join(poses)}")
    print(f"  glove: {'MOCK' if args.mock_glove else f'{args.host}:{args.port}'}"
          f"   camera: {camera_text}   output: {args.out_dir}")
    if coached:
        print(f"  COACHED, {args.hand.upper()} HAND ONLY: each take is "
              "acquired with an open palm, then")
        print(f"  the pose is called and the same hand id has to survive "
              f"{args.settle:g} s of transition.")
        print(f"  Height band {band_text(band)}; up to {args.retries} "
              "retries per take. ONE HAND OVER THE")
        print("  MODULE — keep the other one out of the field.")
        if pose_check_on:
            print("  The take is REFUSED and retried if both sensors say the "
                  "hand was not in the pose;")
            print("  refused attempts are kept under "
                  f"{args.out_dir / REJECTED}, never deleted.")
        elif mocked:
            print("  Pose check OFF: a mock hand's shape does not follow the "
                  "pose being called.")
        else:
            print("  Pose check OFF (--no-pose-check): the hand's shape is "
                  "not verified.")
    else:
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
            if coached:
                session = CoachedLeapSession(
                    leap, glove, hz=glove_hz, out_dir=args.out_dir,
                    leap_hz=args.leap_hz or None, hand=args.hand, band=band,
                    settle=args.settle, retries=args.retries,
                    acquire_timeout=args.acquire_timeout,
                    pose_check=pose_check_on)
                session.view = CameraView(
                    hand=args.hand, band=band,
                    enabled=not (args.no_view or args.mock_leap)).start()
            else:
                session = LeapSyncSession(leap, glove, hz=glove_hz,
                                          out_dir=args.out_dir,
                                          leap_hz=args.leap_hz or None)
        else:
            cap = open_camera(camera_index, args.width, args.height)
            tracker = HandTracker(model_path=args.model, running_mode="video",
                                  min_detection_confidence=args.min_det,
                                  min_tracking_confidence=args.min_det,
                                  gamma=args.gamma, clahe=args.clahe)
            session = SyncSession(cap, tracker, glove, hz=glove_hz,
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
            session.finish()
            print("\nInterrupted — keeping the takes recorded so far.")
        finally:
            session.finish()
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
