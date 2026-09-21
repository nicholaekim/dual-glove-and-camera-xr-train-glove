"""The rig the two glove diagnostics run on: both sensors, one hand, one loop.

`scripts/leap/hold_test.py` and `scripts/leap/finger_sweep.py` need exactly
the same things before they can measure anything:

  * the glove streaming over OSC, drained on the caller's schedule, with the
    packet counter watched (`diagnostics.StreamLog`);
  * the Ultraleap tracking the SAME hand, drained the same way;
  * both sensors' curls computed with `cam_hand.features`, so the numbers in
    a diagnostic and the numbers in `scripts/fuse_poses.py`'s per-DOF table
    are the same quantity;
  * the live camera window with a caption this process controls
    (`protocol.CameraView`), because the operator's hands are over the module
    and their eyes are not on the terminal;
  * and the ACQUIRE gate — the expected hand, OPEN, settled, at the right
    height, palm toward the lens, over the module — before anything is asked
    of it. `protocol.acquire_failures` is that gate, and it is the same one
    `scripts/record_simultaneous.py` uses, so a hand good enough to record is
    a hand good enough to measure.

Writing that twice is how two tools drift apart, so it is written here once.
What stays in each script is its own protocol: which pose, which finger, how
long, and what to do with the numbers.

WHY THE CURLS COME FROM `abs26`
  The camera's curl is `flexion_features` of the 21 keypoints picked out of
  `LeapHand.abs26` by `MP21_TO_OPENXR_IDX` — the same indices
  `fuse_poses.load_leap_cam` uses for the palm normal. Curl is a RATIO
  (fingertip-to-wrist over palm length), so absolute camera-space
  coordinates give exactly the number the wrist-centred recording would, one
  conversion earlier and one allocation cheaper at 90 Hz.
"""
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from cam_hand.features import flexion_features, thumb_index_gap
from xr_hand.keypoints21 import MP21_TO_OPENXR_IDX, frame_to_keypoints21
from xr_hand.parser import parse_hand_message

from .diagnostics import StreamLog
from .protocol import (
    DEFAULT_BAND,
    HandReading,
    acquire_failures,
    band_text,
    hud_line,
    read_hand,
)

# A tracked hand goes stale on the HUD this long after its last frame, so the
# line says "no hand" while the hand really is gone. Same value as
# scripts/record_simultaneous.py.
STALE_S = 0.30


def beep(freq: int = 880, ms: int = 160) -> None:
    """A cue the operator can hear with their eyes on the camera window.

    On a worker thread: `winsound.Beep` blocks for its whole duration, and a
    quarter second of stalled loop in the middle of a timed measurement is a
    quarter second of hand nobody sampled.
    """
    def _beep():
        try:
            import winsound
            winsound.Beep(int(freq), int(ms))
        except Exception:
            print("\a", end="", flush=True)

    threading.Thread(target=_beep, daemon=True).start()


@dataclass(frozen=True)
class GloveSample:
    """One glove packet of the expected hand, reduced to what is measured."""

    t: float                                  # OSC arrival time, wall clock
    curls: Tuple[float, ...]
    counter: int


@dataclass(frozen=True)
class CameraSample:
    """One tracked hand of the expected side, reduced the same way."""

    t: float                                  # capture time, wall clock
    curls: Tuple[float, ...]
    gap: float                                # thumb tip to index tip
    reading: HandReading                       # height, facing, id, age


class MockGlove:
    """A synthetic 60 Hz glove behind the receiver's `drain()` API.

    Enough of `xr_hand.receiver.OSCHandReceiver` for a rehearsal with no
    hardware: the same `(hand_side, raw_values)` items, the same `recv_time`
    attribute, and a packet counter that really does increment, so the stream
    log has something to check.
    """

    def __init__(self, hand: str = "right"):
        from xr_hand.mock import MockHandGenerator
        from xr_hand.receiver import QueueItem
        self._item = QueueItem
        self.gen = MockHandGenerator(hand=hand)
        self.hand = hand
        self._last = time.time()

    def start(self) -> None:
        self._last = time.time()

    def stop(self) -> None:
        pass

    def drain(self, max_items: int = 64):
        now = time.time()
        n = min(int((now - self._last) * 60.0), max_items)
        if n <= 0:
            return []
        t0 = self._last
        self._last += n / 60.0
        return [self._item(self.hand, self.gen.next_frame(),
                           recv_time=t0 + k / 60.0) for k in range(n)]


class DiagnosticRig:
    """Both sensors on one hand, plus the window, plus the ACQUIRE gate."""

    def __init__(self, hand: str, band: Tuple[float, float] = DEFAULT_BAND,
                 view: bool = True, mock_glove: bool = False,
                 mock_leap: bool = False, mode: str = "desktop"):
        self.hand = hand
        self.band = band
        self.mock_glove = bool(mock_glove)
        self.mock_leap = bool(mock_leap)
        self.mode = mode
        self.glove = None
        self.leap = None
        self.view = None
        self._want_view = bool(view)
        self.stream = StreamLog()

        # newest reading of each sensor, and when it arrived
        self.glove_curls: Optional[Tuple[float, ...]] = None
        self.glove_at = 0.0
        self.cam: Optional[CameraSample] = None
        self.cam_at = 0.0
        self.other_hand_at = 0.0
        self.glove_packets = 0
        self.glove_other = 0
        self.cam_hands = 0

    # --- lifecycle ------------------------------------------------------
    def start(self) -> "DiagnosticRig":
        from .protocol import CameraView
        from .stream import open_stream

        if self.mock_glove:
            self.glove = MockGlove(self.hand)
        else:
            from xr_hand.receiver import OSCHandReceiver
            self.glove = OSCHandReceiver()
        self.glove.start()
        # `open_stream` raises LeapUnavailable with the next command to type;
        # the caller prints it and exits rather than letting a traceback out
        # of the SDK.
        self.leap = open_stream(mock=self.mock_leap, mode=self.mode)
        # CameraView passes --parent-pid itself, so the window closes with
        # this process even if it is killed; nothing here ever signals it.
        self.view = CameraView(hand=self.hand, band=self.band,
                               enabled=self._want_view).start()
        return self

    def stop(self) -> None:
        for closer in (getattr(self.glove, "stop", None),
                       getattr(self.leap, "stop", None),
                       getattr(self.view, "close", None)):
            if closer is None:
                continue
            try:
                closer()
            except Exception:
                pass

    def caption(self, text: str) -> None:
        if self.view is not None:
            self.view.caption(text, band=self.band)

    # --- one pass over both sensors --------------------------------------
    def poll(self) -> Tuple[List[GloveSample], List[CameraSample]]:
        """Drain both sensors once. Returns this pass's samples, per sensor.

        Nothing here blocks — there is no video frame to wait on — so the
        caller is expected to sleep a few milliseconds between passes.
        """
        return self._poll_glove(), self._poll_camera()

    def _poll_glove(self) -> List[GloveSample]:
        out: List[GloveSample] = []
        for item in self.glove.drain(256):
            hand, raw = item
            try:
                frame = parse_hand_message(raw, hand_side_hint=hand)
            except Exception:
                continue                      # a malformed packet is not data
            self.glove_packets += 1
            if frame.hand_side != self.hand:
                self.glove_other += 1
                continue
            t = float(getattr(item, "recv_time", 0.0) or time.time())
            # The stream log gets EVERY packet of this hand, recorded or not:
            # a drift measured over a stream that stopped for two seconds is
            # an artefact of the stream, and the counter says whose fault it
            # was.
            self.stream.add(t, frame.packet_counter)
            curls = tuple(flexion_features(frame_to_keypoints21(frame)))
            self.glove_curls, self.glove_at = curls, t
            out.append(GloveSample(t=t, curls=curls,
                                   counter=int(frame.packet_counter)))
        return out

    def _poll_camera(self) -> List[CameraSample]:
        out: List[CameraSample] = []
        for _side, lh in self.leap.drain(256):
            self.cam_hands += 1
            if lh.hand_side != self.hand:
                self.other_hand_at = time.time()
                continue
            if not getattr(lh, "abs26", None):
                continue
            pts = [lh.abs26[i] for i in MP21_TO_OPENXR_IDX]
            t = float(lh.capture_time or time.time())
            sample = CameraSample(t=t, curls=tuple(flexion_features(pts)),
                                  gap=float(thumb_index_gap(pts)),
                                  reading=read_hand(lh))
            self.cam, self.cam_at = sample, time.time()
            out.append(sample)
        return out

    # --- what the HUD and the gate read ----------------------------------
    def fresh_camera(self, now: Optional[float] = None
                     ) -> Optional[CameraSample]:
        """The latest camera sample, or None if it is already stale."""
        now = time.time() if now is None else now
        if self.cam is None or now - self.cam_at > STALE_S:
            return None
        return self.cam

    def glove_hz(self, now: Optional[float] = None) -> Optional[float]:
        rate = self.stream.summary().get("rate_hz")
        return None if rate is None else float(rate)

    def hud(self, phase: str, seconds_left: Optional[float],
            extra: str = "") -> str:
        now = time.time()
        fresh = self.fresh_camera(now)
        return hud_line(phase, seconds_left,
                        None if fresh is None else fresh.reading,
                        self.hand, self.band, self.glove_hz(now),
                        saw_other_hand=now - self.other_hand_at < STALE_S,
                        extra=extra)

    # --- the ACQUIRE gate -------------------------------------------------
    def acquire(self, timeout: float, show: Callable[[str], None],
                caption: Callable[[List[str]], str],
                poll: Optional[Callable[[], None]] = None) -> CameraSample:
        """Block until the expected hand is OPEN, settled and well placed.

        The same four conditions `scripts/record_simultaneous.py` holds every
        take to, for the same measured reason: the tracker follows an open
        hand into any pose but cannot acquire a gloved hand that is already
        closed, and when it re-acquires from a closed pose it sometimes
        returns a mirrored skeleton labelled as the other hand. Measuring a
        hand the camera never properly had is the failure both of these tools
        exist to stop.

        `show` is given the HUD line; `caption` is given the list of
        outstanding failures and returns the window's caption. Raises
        TimeoutError with what was still missing.
        """
        t0 = time.time()
        missing: List[str] = ["no hand"]
        while True:
            self.poll()
            if poll is not None:
                poll()
            now = time.time()
            fresh = self.fresh_camera(now)
            missing = acquire_failures(None if fresh is None else fresh.reading,
                                       self.hand, self.band)
            if not missing and self.glove_curls is not None:
                return fresh
            if not missing:
                missing = ["no glove packets"]
            show(self.hud("ACQUIRE", None,
                          "need: " + ", ".join(missing)))
            self.caption(caption(missing))
            if now - t0 > timeout:
                raise TimeoutError(
                    f"no acquirable {self.hand} hand in {timeout:g} s "
                    f"(still missing: {', '.join(missing)}). "
                    f"Open palm, {band_text(self.band)} above the module, "
                    "palm toward the lens, other hand away.")
            time.sleep(0.004)


def median_curls(samples: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """Per-finger median of a list of curl vectors, or None if there are none."""
    rows = [list(s) for s in samples]
    if not rows:
        return None
    width = min(len(r) for r in rows)
    out = []
    for i in range(width):
        column = sorted(float(r[i]) for r in rows)
        n = len(column)
        out.append(column[n // 2] if n % 2
                   else 0.5 * (column[n // 2 - 1] + column[n // 2]))
    return out


def curl_text(curls: Optional[Sequence[float]]) -> str:
    """`thumb 1.43  index 1.97 ...`, or `(none)`."""
    from .diagnostics import FINGERS
    if curls is None:
        return "(none)"
    return "  ".join(f"{n} {v:.3f}"
                     for n, v in zip(FINGERS, curls)
                     if not math.isnan(v))
