"""Write Leap recordings as JSONL that the glove's own loader can read.

Every line is exactly what `xr_hand.recorder._frame_to_dict` writes for a
glove frame — `wall_time`, `timestamp`, `packet_counter`, `hand_side`,
`frame_id`, `status`, `joints` — plus the camera-only extras listed below.
`FrameRecorder.load` ignores keys it does not know, so `playback.py`,
`export_keypoints21.py` and `export_prof_format.py` read these files with no
changes at all. That is the whole point: the camera is a new sensor, not a
new format.

Two clocks, and the difference matters:

  wall_time     `time.time()` as the line is written, exactly as the glove
                and camera recorders stamp it. It is the only clock the three
                sensors share, and it is what `fuse_poses.py` pairs takes on.
  timestamp     `event.timestamp` in seconds - the **LeapC clock**, whose
                epoch is arbitrary. Right for intervals inside one recording
                (free of the jitter our writer adds, so `stats.py` prefers
                it), meaningless against wall time or another sensor. The
                glove's `timestamp` is likewise its own tick counter, so this
                matches the convention rather than inventing one.

The extras, none of which the glove can produce:

  source            "leap"
  timestamp_us      event.timestamp unrounded, LeapC's integer microseconds
  space             "leap_desktop" — absolute LeapC desktop-mode camera space
  units             "m" — same unit as glove HandFrames (LeapC's millimetres
                    are converted once, in to_openxr.from_leap_mm)
  hand_id           LeapC hand id; a change means the tracker lost the hand
                    and re-acquired it
  visible_time_us   how long this hand has been tracked; resets with the id
  framerate         event.framerate, the measured tracking rate
  pinch_strength    0..1
  grab_strength     0..1
  frame_age_us      leap.get_now() - event.timestamp at receipt: frame age,
                    not end-to-end latency. null for mock and replayed data
  palm_abs          palm position in camera space, metres
  abs26             all 26 joint positions in camera space, metres — the real
                    geometry, before it was folded into parent-relative form

`abs26` is redundant with `joints` (forward kinematics reproduces it), and
that is deliberate: it is the raw measurement, it survives any later change
to the kinematics, and `stats.py` reads fingertip jitter straight out of it.

Same API as `xr_hand.recorder.FrameRecorder` and `cam_hand.recorder.
CamRecorder` — start / record / stop, per-hand rate throttling, pose and
take labels, one flushed line per frame — so guided-session code can drive
any of the three interchangeably.
"""
import json
import time
from pathlib import Path
from typing import Optional

from xr_hand.recorder import _frame_to_dict

from .to_openxr import FRAME_UNITS, to_hand_frame
from .types import LeapHand

SOURCE = "leap"
SPACE = "leap_desktop"


def _round(v, nd: int = 6):
    """Round positions to micrometres — well past what the camera resolves."""
    return [round(float(c), nd) for c in v]


class LeapRecorder:
    def __init__(
        self,
        hz: Optional[float] = None,
        pose: Optional[str] = None,
        take: Optional[int] = None,
    ):
        """hz: save at most this many frames/sec per hand. None = keep all.

        pose/take: optional gesture label + repetition number stamped into
        every recorded frame, exactly as the glove recorder does.
        """
        self._file = None
        self.path: Optional[Path] = None
        self.count = 0
        self.pose = pose
        self.take = take
        self.hands_seen: set = set()
        self._interval = 1.0 / hz if hz else None
        self._next_sample: dict[str, float] = {}

    def start(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self.count = 0
        self.hands_seen = set()
        self._next_sample = {}

    def record(self, lh: LeapHand) -> None:
        """Convert one tracked hand and append it as a line of JSONL."""
        if self._file is None:
            raise RuntimeError("call start() before record()")
        now = time.time()
        if self._interval is not None:
            # Throttle each hand on its own schedule so one hand cannot
            # starve the other (same rule as the glove and camera recorders).
            if now < self._next_sample.get(lh.hand_side, 0.0):
                return
            self._next_sample[lh.hand_side] = now + self._interval

        frame = to_hand_frame(lh)
        # wall_time is time.time() at the write, the same stamp the glove and
        # camera recorders use; frame.timestamp is the LeapC clock.
        d = _frame_to_dict(frame, wall_time=now)
        if self.pose is not None:
            d["pose"] = self.pose
        if self.take is not None:
            d["take"] = self.take
        d.update({
            "source": SOURCE,
            "space": SPACE,
            "units": FRAME_UNITS,
            "timestamp_us": lh.timestamp_us,
            "hand_id": lh.hand_id,
            "visible_time_us": lh.visible_time_us,
            "framerate": round(lh.framerate, 3),
            "pinch_strength": lh.pinch_strength,
            "grab_strength": lh.grab_strength,
            "frame_age_us": (None if lh.frame_age_us is None
                             else round(lh.frame_age_us, 1)),
            "palm_abs": _round(lh.palm_pos),
            "abs26": [_round(p) for p in lh.abs26],
        })
        self._file.write(json.dumps(d) + "\n")
        self._file.flush()
        self.count += 1
        self.hands_seen.add(lh.hand_side)

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
