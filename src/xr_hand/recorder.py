"""Save and load validated hand frames as JSONL.

One JSON object per line — easy to inspect in a text editor, grep for
specific joints, or send to a professor / StretchSense support.

Format of each line:
{
  "wall_time": 1748123456.789,    # real clock time (for replay timing)
  "timestamp": 4.233,             # OSC packet timestamp
  "packet_counter": 254,
  "hand_side": "right",
  "frame_id": 254,
  "status": 0,
  "pose": "fist",                 # optional: gesture label (pose recordings)
  "take": 1,                      # optional: repetition number
  "joints": [
    {"name": "PALM", "x": 0.0, "y": 0.04, "z": 0.0,
     "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
    ...
  ]
}

`pose` / `take` are only present when the recorder was given them (e.g.
`record.py --pose fist` or `record_poses.py`). Loaders ignore unknown keys,
so labeled and unlabeled recordings stay interchangeable.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Generator, Optional

from .joints import HandFrame, Joint

log = logging.getLogger(__name__)


def _frame_to_dict(frame: HandFrame, wall_time: float) -> dict:
    return {
        "wall_time": wall_time,
        "timestamp": frame.timestamp,
        "packet_counter": frame.packet_counter,
        "hand_side": frame.hand_side,
        "frame_id": frame.frame_id,
        "status": frame.status,
        "joints": [
            {
                "name": j.name,
                "x": j.x, "y": j.y, "z": j.z,
                "qw": j.qw, "qx": j.qx, "qy": j.qy, "qz": j.qz,
            }
            for j in frame.joints
        ],
    }


def _dict_to_frame(d: dict) -> tuple[HandFrame, float]:
    joints = [
        Joint(
            name=j["name"],
            x=j["x"], y=j["y"], z=j["z"],
            qw=j["qw"], qx=j["qx"], qy=j["qy"], qz=j["qz"],
        )
        for j in d["joints"]
    ]
    frame = HandFrame(
        timestamp=d["timestamp"],
        packet_counter=d["packet_counter"],
        hand_side=d["hand_side"],
        frame_id=d["frame_id"],
        status=d["status"],
        joints=joints,
    )
    return frame, d["wall_time"]


def hand_tag(hands: set) -> str:
    """Filename tag for the set of hand sides that made it into a recording."""
    if hands == {"left"}:
        return "left"
    if hands == {"right"}:
        return "right"
    if hands == {"left", "right"}:
        return "both"
    return "nohand"


def slugify(name: str) -> str:
    """'Open Palm!' -> 'open_palm' (filename-safe pose names)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pose_filename(pose: str, take: int) -> str:
    """fist, 2 -> 'fist_take2_20260701_154212.jsonl' (hand added on finalize)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{slugify(pose)}_take{take}_{stamp}.jsonl"


def finalize_pose_name(path: Path, hands: set) -> Path:
    """Insert the recorded hand side into a pose filename after the fact.

    fist_take1_<stamp>.jsonl -> fist_left_take1_<stamp>.jsonl. The hand isn't
    known until frames arrive, so the file is renamed once recording stops.
    Returns the final path (the original one if the rename fails, e.g. when
    OneDrive briefly locks the file).
    """
    tag = hand_tag(hands)
    target = path.with_name(path.name.replace("_take", f"_{tag}_take", 1))
    try:
        path.rename(target)
        return target
    except OSError as e:
        log.warning("could not rename %s -> %s (%s); keeping original name",
                    path.name, target.name, e)
        return path


class FrameRecorder:
    def __init__(
        self,
        hz: Optional[float] = None,
        pose: Optional[str] = None,
        take: Optional[int] = None,
    ):
        """hz: downsample to this many frames/sec (e.g. 5.0). None = keep all.

        pose/take: optional gesture label + repetition number stamped into
        every recorded frame.
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

    def record(self, frame: HandFrame) -> None:
        if self._file is None:
            raise RuntimeError("call start() before record()")
        now = time.time()
        if self._interval is not None:
            # throttle each hand independently so both keep the full target rate
            if now < self._next_sample.get(frame.hand_side, 0.0):
                return  # too soon — drop this frame to hold the target rate
            # schedule next sample from this write; avoids a burst after a gap
            self._next_sample[frame.hand_side] = now + self._interval
        d = _frame_to_dict(frame, wall_time=now)
        if self.pose is not None:
            d["pose"] = self.pose
        if self.take is not None:
            d["take"] = self.take
        self._file.write(json.dumps(d) + "\n")
        self._file.flush()
        self.count += 1
        self.hands_seen.add(frame.hand_side)

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def load(path: str | Path) -> Generator[tuple[HandFrame, float], None, None]:
        """Yield (HandFrame, wall_time) for each recorded frame."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield _dict_to_frame(json.loads(line))
