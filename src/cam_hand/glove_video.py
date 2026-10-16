"""A take's glove data drawn as a video: the hand the glove streamed, moving.

Sets B and C are recorded with the glove alone, so nothing filmed the hand.
What a take does hold is every joint of the glove's hand model at 60 Hz, and
that can be drawn: `render_take` writes one MP4 per take, in real time, with

  * the hand seen at its palm and from its side (the side view is where
    flexion shows), every finger in its own colour and the cued finger drawn
    heavier,
  * the cue in force at that instant in words (the cue whose beep was at or
    before the frame, `angle_series.frame_rows`' rule), the take's clock,
  * each digit's three flexion angles in degrees (`angle_series.angles_of`),
    the numbers the per-frame tables of the hand-in hold.

IT IS A DRAWING OF THE DATA, NOT FOOTAGE. Every picture says so along its
bottom edge. The glove senses finger flexion only: the hand is drawn at the
glove's own origin with its template bone lengths, it never turns or moves,
and the fingers never spread. A finger that looks wrong here is the data
looking wrong, which is what the video is for.

The video's clock is the take's: picture k shows the newest glove frame at
or before take_start + k / fps, so playing it takes as long as the take did
and a stalled glove shows as a hand that stops (drawn grey once its newest
frame is older than STALE_S).
"""
import bisect
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cam_hand import angle_series as asr
from cam_hand.demo_view import BG_BGR, DIM_BGR, PANEL_BGR, TEXT_BGR, put
from cam_hand.fusion import FINGER_CHAINS, palm_basis
from xr_hand.keypoints21 import frame_to_keypoints21

WIDTH, HEIGHT = 1280, 720
FPS = 30.0
STALE_S = 0.25               # a glove frame older than this is drawn grey
MM_PER_PX = 0.46
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_WORDS = {"thumb": "thumb", "index": "index", "middle": "middle",
                "ring": "ring", "pinky": "little"}
# One colour per finger (BGR), told apart in the side view where they overlap.
FINGER_BGR = {"thumb": (60, 170, 255), "index": (90, 220, 90),
              "middle": (240, 200, 60), "ring": (235, 120, 200),
              "pinky": (90, 110, 245)}
STALE_BGR = (110, 110, 110)
# The three flexions of each digit and the words drawn over them.
DIGIT_ANGLES = {"thumb": asr.DIGIT_FLEXIONS["thumb"],
                "index": asr.DIGIT_FLEXIONS["index"],
                "middle": asr.DIGIT_FLEXIONS["middle"],
                "ring": asr.DIGIT_FLEXIONS["ring"],
                "pinky": asr.DIGIT_FLEXIONS["little"]}
JOINT_WORDS = {"thumb": ("TMC", "MCP", "IP")}
BAR_FULL_DEG = 240.0         # summed flexion that fills a digit's bar
CAPTION = ("Drawn from the glove's joint data (StretchSense through XR "
           "Trainer). Not camera footage: the glove senses finger flexion "
           "only.")
PALM_PANEL = (40, 110, 600, 480)       # x, y, w, h
SIDE_PANEL = (640, 110, 600, 480)
H264_CRF = 23


# --- the take, read once ---------------------------------------------------------

class TakeFrames:
    """A take's glove frames of one hand in time order: when each arrived,
    its 21 points in the palm's own axes (millimetres) and its 24 angles."""

    def __init__(self, times: Sequence[float], local_mm: Sequence[np.ndarray],
                 angles: Sequence[Dict[str, float]], hand: str,
                 events: Sequence[dict]):
        self.times = list(times)
        self.local_mm = list(local_mm)
        self.angles = list(angles)
        self.hand = hand
        self.cues = asr.cues_of(events)
        self.cue_times = [float(c["t"]) for c in self.cues]
        start = asr._event_time(events, "take_start")
        end = asr._event_time(events, "take_end")
        self.t0 = start if start is not None else (
            self.times[0] if self.times else 0.0)
        self.t1 = end if end is not None else (
            self.times[-1] if self.times else self.t0)
        self.side_sign = side_sign(self.local_mm)

    @property
    def seconds(self) -> float:
        return max(0.0, self.t1 - self.t0)

    def index_at(self, t: float) -> Optional[int]:
        """The newest frame at or before `t` (None before the first)."""
        k = bisect.bisect_right(self.times, t) - 1
        return k if k >= 0 else None

    def cue_at(self, t: float) -> Optional[dict]:
        """The cue in force at `t`: the last one whose beep was at or
        before it; none before the first cue or from take_end on."""
        if t >= self.t1:
            return None
        k = bisect.bisect_right(self.cue_times, t) - 1
        return self.cues[k] if k >= 0 else None

    def take_fingers(self) -> List[str]:
        """Every finger a cue of the take flexes, thumb to little."""
        named = {f for c in self.cues for f in (c.get("flexed") or [])}
        return [f for f in FINGERS if f in named]


def local_points_mm(frame) -> np.ndarray:
    """21 x 3 millimetres in the hand's own palm axes, wrist at 0: columns
    up (wrist to middle knuckle), across (towards the little finger) and
    the palm's normal."""
    pts = np.asarray(frame_to_keypoints21(frame), dtype=float)
    pts = pts - pts[0]
    return (pts @ palm_basis(pts)) * 1000.0


def side_sign(local_mm: Sequence[np.ndarray]) -> float:
    """+1 or -1: which way along the palm's normal the fingers curl, read
    from the frame whose fingertips are furthest off the palm's plane, so
    the side view always has the palm side on the right."""
    tips = [FINGER_CHAINS[f][-1] for f in FINGERS[1:]]
    best, sign = 0.0, 1.0
    for pts in local_mm:
        off = float(np.mean(pts[tips, 2]))
        if abs(off) > best:
            best, sign = abs(off), (1.0 if off > 0 else -1.0)
    return sign


def load_take(glove_path, events: Sequence[dict], hand: str) -> TakeFrames:
    """A take's glove file -> `TakeFrames` of `hand`."""
    from cam_hand import protocol_check as pc
    frames, _bad = pc.read_frames(Path(glove_path))
    mine = sorted(((float(d["wall_time"]), d, fr) for d, fr in frames
                   if d.get("hand_side") == hand), key=lambda x: x[0])
    return TakeFrames([t for t, _d, _f in mine],
                      [local_points_mm(fr) for _t, _d, fr in mine],
                      [asr.angles_of(d, hand) for _t, d, _f in mine],
                      hand, events)


# --- drawing -----------------------------------------------------------------------

def _pixels(xy_mm: np.ndarray, panel) -> np.ndarray:
    """Millimetres (x right, y up, wrist at 0) -> pixels in a panel whose
    wrist sits at the middle of its bottom edge, a little above it."""
    x, y, w, h = panel
    px = x + w / 2.0 + xy_mm[:, 0] / MM_PER_PX
    py = y + h - 40 - xy_mm[:, 1] / MM_PER_PX
    return np.rint(np.column_stack([px, py])).astype(int)


def palm_xy(local_mm: np.ndarray, hand: str) -> np.ndarray:
    """The hand seen looking at its palm (`demo_view.palm_view_mm`'s
    convention: a right hand is mirrored, a left hand is not)."""
    across = local_mm[:, 1] if str(hand).lower().startswith("l") \
        else -local_mm[:, 1]
    return np.column_stack([across, local_mm[:, 0]])


def side_xy(local_mm: np.ndarray, sign: float) -> np.ndarray:
    """The hand seen from its side, the palm side to the right."""
    return np.column_stack([sign * local_mm[:, 2], local_mm[:, 0]])


def draw_hand(img, px: np.ndarray, heavy: Sequence[str], stale: bool) -> None:
    """Each finger's chain from the wrist in its colour; the `heavy`
    fingers thicker and drawn last, so they stay on top in the side view."""
    order = [f for f in FINGERS if f not in heavy] + [
        f for f in FINGERS if f in heavy]
    knuckles = [FINGER_CHAINS[f][0] for f in FINGERS[1:]]
    for a, b in zip(knuckles, knuckles[1:]):
        cv2.line(img, tuple(px[a]), tuple(px[b]), DIM_BGR, 2, cv2.LINE_AA)
    for f in order:
        chain = [0] + list(FINGER_CHAINS[f])
        colour = STALE_BGR if stale else FINGER_BGR[f]
        thick = 6 if f in heavy else 3
        for a, b in zip(chain, chain[1:]):
            cv2.line(img, tuple(px[a]), tuple(px[b]), colour, thick,
                     cv2.LINE_AA)
        for i in chain[1:]:
            r = (6 if f in heavy else 4) + (1 if i == chain[-1] else 0)
            cv2.circle(img, tuple(px[i]), r + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(px[i]), r, colour, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(px[0]), 6, (240, 240, 240), -1, cv2.LINE_AA)


def cue_words(take: TakeFrames, cue: Optional[dict]) -> str:
    """The cue as the picture's headline."""
    if cue is None:
        return ""
    label = str(cue.get("label", ""))
    if cue.get("phase"):
        cycles = max((c.get("cycle") or 0) for c in take.cues)
        return f"cycle {cue.get('cycle')}/{cycles}:  {label}"
    return f"step {int(cue.get('step', 0)) + 1}/{len(take.cues)}:  {label}"


def heavy_fingers(take: TakeFrames, cue: Optional[dict]) -> List[str]:
    """The fingers drawn heavier: Set B's one finger for the whole take,
    Set C's fingers of the cue in force."""
    if any(c.get("phase") for c in take.cues):
        return take.take_fingers()
    return list((cue or {}).get("flexed") or [])


def _angle_strip(img, angles: Optional[Dict[str, float]],
                 heavy: Sequence[str]) -> None:
    """Along the bottom: per digit its three flexions in degrees and a bar
    of their sum."""
    y0, col_w = 612, (WIDTH - 80) // 5
    for k, f in enumerate(FINGERS):
        x = 40 + k * col_w
        colour = FINGER_BGR[f]
        put(img, FINGER_WORDS[f], (x, y0), 0.6, colour,
            2 if f in heavy else 1)
        if angles is None:
            continue
        names = JOINT_WORDS.get(f, ("MCP", "PIP", "DIP"))
        values = [angles[a] for a in DIGIT_ANGLES[f]]
        put(img, "  ".join(f"{n} {v:3.0f}" for n, v in zip(names, values)),
            (x, y0 + 24), 0.5, TEXT_BGR)
        full = col_w - 30
        cv2.rectangle(img, (x, y0 + 34), (x + full, y0 + 42), PANEL_BGR, -1)
        part = int(full * min(1.0, max(0.0, sum(values) / BAR_FULL_DEG)))
        if part > 0:
            cv2.rectangle(img, (x, y0 + 34), (x + part, y0 + 42), colour, -1)


def render_frame(take: TakeFrames, t: float, title: str) -> np.ndarray:
    """The picture for time `t` of the take (seconds, the take's clock)."""
    img = np.full((HEIGHT, WIDTH, 3), BG_BGR, np.uint8)
    for panel, words in ((PALM_PANEL, "palm view"),
                         (SIDE_PANEL, "side view (palm side to the right)")):
        x, y, w, h = panel
        cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BGR, -1)
        put(img, words, (x + 12, y + 24), 0.55, DIM_BGR)
    cue = take.cue_at(t)
    heavy = heavy_fingers(take, cue)
    k = take.index_at(t)
    put(img, title, (40, 34), 0.6, DIM_BGR)
    clock = f"{max(0.0, t - take.t0):6.2f} s of {take.seconds:.1f} s"
    put(img, clock, (WIDTH - 40 - 250, 34), 0.6, DIM_BGR)
    words = cue_words(take, cue)
    put(img, words or "(no cue)", (40, 84), 1.05,
        TEXT_BGR if words else DIM_BGR, 2)
    if k is None:
        put(img, "no glove frame yet", (PALM_PANEL[0] + 12, 300), 0.7,
            DIM_BGR)
    else:
        stale = (t - take.times[k]) > STALE_S
        local = take.local_mm[k]
        draw_hand(img, _pixels(palm_xy(local, take.hand), PALM_PANEL),
                  heavy, stale)
        draw_hand(img, _pixels(side_xy(local, take.side_sign), SIDE_PANEL),
                  heavy, stale)
        if stale:
            put(img, f"glove silent for {t - take.times[k]:.1f} s",
                (SIDE_PANEL[0] + 12, SIDE_PANEL[1] + 52), 0.6, (70, 70, 235))
    _angle_strip(img, None if k is None else take.angles[k], heavy)
    frac = 0.0 if take.seconds <= 0 else min(1.0, max(
        0.0, (t - take.t0) / take.seconds))
    cv2.rectangle(img, (40, 668), (WIDTH - 40, 672), PANEL_BGR, -1)
    cv2.rectangle(img, (40, 668), (40 + int((WIDTH - 80) * frac), 672),
                  DIM_BGR, -1)
    put(img, CAPTION, (40, 702), 0.5, DIM_BGR)
    return img


# --- the file ----------------------------------------------------------------------

def frame_count(seconds: float, fps: float) -> int:
    """Pictures in a video of `seconds` at `fps`: one at time zero and one
    every 1/fps up to the end."""
    return int(math.floor(seconds * fps + 1e-9)) + 1


def reencode_h264(path: Path, crf: int = H264_CRF) -> str:
    """Re-encode `path` to H.264 with ffmpeg, in place, when ffmpeg is on
    PATH; say in words what the file is. OpenCV's own file stays when
    there is no ffmpeg or it fails."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "mp4v (ffmpeg is not on PATH, so not re-encoded)"
    tmp = path.with_name(path.stem + ".h264" + path.suffix)
    try:
        done = subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(path),
             "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", str(tmp)],
            capture_output=True, text=True, timeout=900)
        ok = done.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0
        why = (done.stderr.strip().splitlines()
               or [f"exit code {done.returncode}"])[-1]
        if ok:
            os.replace(tmp, path)
    except (OSError, subprocess.SubprocessError) as e:
        ok, why = False, str(e)
    if not ok:
        try:
            tmp.unlink()
        except OSError:
            pass
        return f"mp4v (the ffmpeg re-encode failed: {why})"
    return f"H.264 (CRF {crf}, re-encoded by ffmpeg)"


def render_take(dest, glove_path, events: Sequence[dict], hand: str,
                title: str, fps: float = FPS, h264: bool = True) -> dict:
    """Write the take's video to `dest`. Returns what was written: path,
    pictures, fps, seconds, glove frames and the codec in words."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    take = load_take(glove_path, events, hand)
    if not take.times:
        raise ValueError(f"no {hand} glove frame in {glove_path}")
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise OSError(f"OpenCV cannot write an mp4v video to {dest}")
    n = frame_count(take.seconds, fps)
    try:
        for k in range(n):
            writer.write(render_frame(take, take.t0 + k / fps, title))
    finally:
        writer.release()
    codec = reencode_h264(dest) if h264 else "mp4v"
    return {"path": dest, "pictures": n, "fps": float(fps),
            "seconds": take.seconds, "glove_frames": len(take.times),
            "codec": codec}
