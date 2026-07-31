"""Save and load camera hand frames as JSONL (schema "cam21.v1").

Mirrors the glove pipeline's recorder (xr_hand.recorder.FrameRecorder) —
same start/record/stop API, same per-hand rate throttling, same pose/take
labels, same crash-safe line-per-frame flushing — so guided-session code can
drive a glove recorder and a camera recorder interchangeably.

Format of each line:
{
  "wall_time": 1785345678.123,   # time.time() at write — aligns with glove files
  "ts_ms": 123456,               # monotonic capture stamp fed to MediaPipe
  "hand_side": "right",
  "score": 0.98,                 # MediaPipe handedness confidence
  "frame_w": 640, "frame_h": 480,
  "pose": "fist", "take": 1,     # only when labels were given
  "img":   [[x_px, y_px, z_rel] x21],   # image-space landmarks
  "world": [[x_m, y_m, z_m] x21]        # metric landmarks, hand-centred, metres
}
"""
import json
import re
import time
from pathlib import Path
from typing import Generator, Optional

from .landmarks import CamHand


def slugify(name: str) -> str:
    """'Open Palm!' -> 'open_palm' (filename-safe pose names)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pose_filename(pose: str, take: int) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{slugify(pose)}_take{take}_{stamp}.jsonl"


def hand_tag(hands: set) -> str:
    if hands == {"left"}:
        return "left"
    if hands == {"right"}:
        return "right"
    if hands == {"left", "right"}:
        return "both"
    return "nohand"


def finalize_pose_name(path: Path, hands: set) -> Path:
    """fist_take1_<stamp>.jsonl -> fist_left_take1_<stamp>.jsonl (post-hoc)."""
    target = path.with_name(path.name.replace("_take", f"_{hand_tag(hands)}_take", 1))
    try:
        path.rename(target)
        return target
    except OSError:
        return path


class CamRecorder:
    def __init__(self, hz: Optional[float] = None, pose: Optional[str] = None,
                 take: Optional[int] = None):
        """hz: save at most this many frames/sec per hand. None = keep all."""
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

    def record(self, hand: CamHand, ts_ms: int, frame_size: tuple) -> None:
        if self._file is None:
            raise RuntimeError("call start() before record()")
        now = time.time()
        if self._interval is not None:
            if now < self._next_sample.get(hand.hand_side, 0.0):
                return
            self._next_sample[hand.hand_side] = now + self._interval
        d = {
            "wall_time": now,
            "ts_ms": int(ts_ms),
            "hand_side": hand.hand_side,
            "score": round(hand.score, 4),
            "frame_w": int(frame_size[0]),
            "frame_h": int(frame_size[1]),
        }
        if self.pose is not None:
            d["pose"] = self.pose
        if self.take is not None:
            d["take"] = self.take
        d["img"] = [[round(x, 2), round(y, 2), round(z, 5)] for x, y, z in hand.img]
        d["world"] = [[round(x, 6), round(y, 6), round(z, 6)] for x, y, z in hand.world]
        self._file.write(json.dumps(d) + "\n")
        self._file.flush()
        self.count += 1
        self.hands_seen.add(hand.hand_side)

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def load(path: str | Path) -> Generator[dict, None, None]:
        """Yield the raw dict of each recorded frame."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
