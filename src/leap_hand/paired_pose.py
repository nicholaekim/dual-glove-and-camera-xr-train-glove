"""One paired glove + camera take, each sensor's frame nearest the photo's
moment, in ONE frame.

What the professor asked for (her request), per pose: a photo of the hand,
the Leap hand model and the glove hand model, "all showing the same moment",
and the xyz of the same 21 hand points from both sensors in the same units,
origin and axis directions, so that where the two systems disagree can be
read off.

The moment is the PHOTO's: the IR still the camera window saves at the
midpoint of the REC hold (`take_moment`). Each sensor contributes the frame
of its own file nearest that moment on its own arrival clock
(`frame_nearest`), not each sensor's own most typical frame, which can sit
seconds apart. That is an arrival-time matched, held-pose comparison:
physical simultaneity of the two selected frames is not verified.

The common frame (`to_palm_frame`, `axis_definition`): millimetres, the
wrist as origin, and the palm axes of `cam_hand.fusion.palm_basis` built on
each sensor's OWN hand from directions between three of its landmarks. No
fit, no least squares, no scaling: a fit would absorb part of the
disagreement it is supposed to show. Coordinates only, no joint angles.
What it yields is a wrist-centred, independently palm-aligned shape
disagreement: the hand's orientation is removed on each sensor separately,
not measured between them.

Pure functions; numpy plus this repo's own readers. Nothing here draws.
"""
import json
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from cam_hand.fusion import palm_basis
from leap_hand.transition import POSE_SETTLE_SPEED_MM_S, SPEED_STEP_S
from xr_hand.keypoints21 import MP21_NAMES, frame_to_keypoints21
from xr_hand.recorder import FrameRecorder

MM = 1000.0
WRIST = 0
LABEL = "wrist-centred, independently palm-aligned shape disagreement"
TIMING_LABEL = "arrival-time matched, held-pose comparison"
SIMULTANEITY = ("Physical simultaneity of the two selected frames is not verified; they are "
                "the frames nearest the photo's time on each sensor's own arrival clock, "
                "taken during a held pose.")
# The four landmarks the palm frame is built on: wrist, middle, pinky, index MCP.
W, M, P, I = 0, 9, 17, 5
MIN_PALM_ANGLE_DEG = 10.0       # x against (P - I): below this the normal is ill-defined
ORTHONORMAL_TOL = 1e-9
HANDEDNESS_MARGIN_MM = 5.0      # |mean z of points 1-20| below this: no verdict
SAME = "same"
OPPOSITE = "opposite (suspected mirrored tracker fit)"
INDETERMINATE = "indeterminate (margin under 5 mm)"
# "Still": the same 10 mm/s `leap_hand.transition` calls a settled pose.
STILL_MM_S = POSE_SETTLE_SPEED_MM_S
STILL_WINDOW_S = 0.25           # frames within this of the selected frame
# The five chains of the 21-point (MediaPipe) order, knuckle to tip.
CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
          "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
# The 20 bones: wrist to each chain's first point, then along the chain.
BONES: List[Tuple[int, int]] = [pair for chain in CHAINS.values()
                                for pair in zip((WRIST,) + chain[:-1], chain)]

STILL_CAPTURE = "still capture time"
STILL_MTIME = "still file write time (no capture stamp in this take)"
CAPTURE_CLOCK = "capture_time"
WALL_CLOCK = "wall_time"


class Moment(NamedTuple):
    t: float                    # time.time() seconds
    source: str                 # STILL_CAPTURE or STILL_MTIME
    still: Optional[Path]       # the still's file, None when it is not on disk


class TakeFrame(NamedTuple):
    frame: object               # xr_hand.joints.HandFrame
    wall_time: float            # when the line was written (writer clock)
    capture_time: Optional[float]   # when the sensor had the frame, same clock
    line: dict                  # the JSONL line, every key the recorder wrote


class Nearest(NamedTuple):
    frame: object
    dt: float                   # this frame's time minus the moment, seconds
    clock: str                  # CAPTURE_CLOCK, or WALL_CLOCK when a frame lacks it
    index: int                  # into the list it was picked from
    time: float                 # this frame's time on `clock`


# --- the take on disk ------------------------------------------------------------
def find_meta(session, take_stem: str) -> Optional[Path]:
    """`leap/<take>.meta.json`, else `rejected/leap/<take>.meta.json`."""
    session = Path(session)
    for folder in (session / "leap", session / "rejected" / "leap"):
        path = folder / f"{take_stem}.meta.json"
        if path.is_file():
            return path
    return None


def still_file(session, meta: dict) -> Optional[Path]:
    """The meta's `still` (relative to the session root, written with
    Windows separators by the recorder) as a path, or None without one."""
    rel = meta.get("still")
    return Path(session) / str(rel).replace("\\", "/") if rel else None


def take_moment(session, take_stem: str, meta: Optional[dict] = None) -> Optional[Moment]:
    """The photo's moment, or None when the take has no way to say it.

    Priority: the meta's `still_capture_time` (the viewer's record of when
    the still's IR image was captured), else the still file's modification
    time (when the JPEG finished writing; sessions made before the record
    existed), else None and the caller refuses the take."""
    if meta is None:
        path = find_meta(session, take_stem)
        if path is None:
            return None
        meta = json.loads(path.read_text(encoding="utf-8"))
    still = still_file(session, meta)
    on_disk = still if still is not None and still.is_file() else None
    stamp = meta.get("still_capture_time")
    if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
        return Moment(float(stamp), STILL_CAPTURE, on_disk)
    if on_disk is not None:
        return Moment(on_disk.stat().st_mtime, STILL_MTIME, on_disk)
    return None


def load_frames(path, hand: Optional[str] = None) -> List[TakeFrame]:
    """Every frame of a recorder JSONL (glove or Leap), with both stamps.
    `hand` keeps only lines whose `hand_side` is that hand."""
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(s) for s in f if s.strip()]
    out = []
    for line, (frame, wall) in zip(lines, FrameRecorder.load(path)):
        if hand is None or line.get("hand_side") == hand:
            cap = line.get(CAPTURE_CLOCK)
            out.append(TakeFrame(frame, float(wall), None if cap is None else float(cap), line))
    return out


def frame_nearest(frames: Sequence[TakeFrame], t: float,
                  max_dt_s: float = 0.05) -> Optional[Nearest]:
    """The frame nearest `t`, or None when none is within `max_dt_s`.

    On `capture_time` when every frame has it; otherwise on `wall_time` for
    all of them, named in `clock` (one clock for the whole list, never a
    mixture of the two, as `cam_hand.fusion.pairing_clock` does)."""
    if not frames:
        return None
    clock = (CAPTURE_CLOCK if all(f.capture_time is not None for f in frames)
             else WALL_CLOCK)
    times = np.array([f.capture_time if clock == CAPTURE_CLOCK else f.wall_time
                      for f in frames], float)
    i = int(np.argmin(np.abs(times - t)))
    dt = float(times[i] - t)
    if abs(dt) > max_dt_s:
        return None
    return Nearest(frames[i].frame, dt, clock, i, float(times[i]))


# --- the common frame ------------------------------------------------------------
def points_mm(frame) -> np.ndarray:
    """21 x 3 in millimetres, wrist at the origin, MediaPipe order
    (`xr_hand.keypoints21`), the same function for either sensor."""
    return np.asarray(frame_to_keypoints21(frame), float) * MM


class DegeneratePalm(ValueError):
    """The palm frame cannot be built on this hand; str() is the reason."""


def palm_angle_deg(pts) -> float:
    """The acute angle between x = M - W and P - I, degrees (0 to 90)."""
    p = np.asarray(pts, float)
    x, across = p[M] - p[W], p[P] - p[I]
    cos = abs(float(np.dot(x, across))) / (np.linalg.norm(x) * np.linalg.norm(across))
    return float(np.degrees(np.arccos(min(1.0, cos))))


def check_orthonormal(basis, sensor: str = "hand") -> None:
    """Raise unless |B.T B - I| < ORTHONORMAL_TOL in every entry."""
    dev = float(np.abs(basis.T @ basis - np.eye(3)).max())
    if not dev < ORTHONORMAL_TOL:
        raise AssertionError(f"palm basis of the {sensor} is not orthonormal "
                             f"(|B.T B - I| = {dev:.2e})")


def to_palm_frame(pts, sensor: str = "hand") -> Tuple[np.ndarray, np.ndarray]:
    """(the 21 points in this hand's palm frame, the 3x3 basis).

    Wrist-centred first, then expressed on the columns of
    `cam_hand.fusion.palm_basis` (x, y, n), so the result does not depend
    on where the hand was or how it was turned. Raises DegeneratePalm when
    any of W, M, P, I is not finite or the angle between M - W and P - I is
    under MIN_PALM_ANGLE_DEG; the basis is checked orthonormal."""
    p = np.asarray(pts, float)
    bad = [MP21_NAMES[k] for k in (W, M, P, I) if not np.all(np.isfinite(p[k]))]
    if bad:
        raise DegeneratePalm(f"degenerate palm frame ({sensor}, angle undefined: "
                             f"{', '.join(bad)} missing or not finite)")
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = palm_angle_deg(p)
    if not angle >= MIN_PALM_ANGLE_DEG:            # NaN too (a zero-length direction)
        raise DegeneratePalm(f"degenerate palm frame ({sensor}, angle {angle:.1f} deg)")
    p = p - p[WRIST]
    basis = palm_basis(p)
    check_orthonormal(basis, sensor)
    return p @ basis, basis


def axis_definition() -> str:
    """The recipe of the common frame, in words, for the README."""
    return (
        "Units are millimetres. W is the wrist (point 0), M the middle-finger "
        "knuckle (9, MIDDLE_FINGER_MCP), P the pinky knuckle (17, PINKY_MCP), I the "
        "index knuckle (5, INDEX_FINGER_MCP), each of ONE sensor. "
        "x = (M - W) / |M - W|; "
        "z = x × (P - I) / |x × (P - I)| (the palm normal); "
        "y = z × x (in the palm plane, from the index side toward the pinky side). "
        "A point p's local coordinates are the dot products of (p - W) with x, y "
        "and z. This is exactly what cam_hand.fusion.palm_frame / palm_basis "
        "compute (x = unit(M - W), n = unit(x × (P - I)), y = unit(n × x), "
        "basis columns x, y, n; the points are multiplied by that basis after the "
        "wrist is subtracted). The axes are orthonormal and right-handed "
        "(checked: every entry of |BᵀB - I| under 1e-9, else the take stops). "
        "A take is refused as \"degenerate palm frame (<sensor>, angle N deg)\" when "
        f"the angle between x and (P - I) is under {MIN_PALM_ANGLE_DEG:g} degrees, or "
        "when any of W, M, P, I is missing or not finite. Nothing is fitted, nothing "
        "is scaled. By construction both hands have the wrist at the origin and the "
        "middle knuckle on the +x axis (y = z = 0), and the index and pinky knuckles "
        "at the same z as each other; how far the middle knuckle is from the wrist "
        "is not forced to agree.")


# --- the comparison --------------------------------------------------------------
def compare_points(glove_local, cam_local) -> Tuple[List[dict], dict]:
    """(21 rows, summary) for two hands already in their palm frames.

    Each row: index, name, glove gx gy gz, camera cx cy cz, distance_mm.
    The summary (median_mm, max_mm, max_point, n_points) is over points
    1..20 only: WRIST is the origin of both frames, so its distance is zero
    by construction and would only dilute the median."""
    g = np.asarray(glove_local, float)
    c = np.asarray(cam_local, float)
    d = np.linalg.norm(g - c, axis=1)
    rows = [{"index": i, "name": MP21_NAMES[i],
             "gx": float(g[i, 0]), "gy": float(g[i, 1]), "gz": float(g[i, 2]),
             "cx": float(c[i, 0]), "cy": float(c[i, 1]), "cz": float(c[i, 2]),
             "distance_mm": float(d[i])} for i in range(len(MP21_NAMES))]
    rest = d[1:]
    k = 1 + int(np.argmax(rest))
    return rows, {"median_mm": float(np.median(rest)), "max_mm": float(d[k]),
                  "max_point": MP21_NAMES[k], "n_points": int(rest.size)}


def bone_lengths(pts) -> List[Tuple[str, float]]:
    """The 20 bones' lengths, (`A-B` by MediaPipe names, mm), wrist-to-
    knuckle first then along each finger, thumb to pinky."""
    p = np.asarray(pts, float)
    return [(f"{MP21_NAMES[a]}-{MP21_NAMES[b]}", float(np.linalg.norm(p[b] - p[a])))
            for a, b in BONES]


def palm_side_mm(local) -> float:
    """Mean palm-normal (z) coordinate of points 1..20, in the palm frame.

    Which side of the palm plane the hand's points lie on: fingers flex
    toward the palm and the thumb sits in front of it. Through the
    palm_basis recipe the palm faces -z on a left hand and +z on a right
    one, and a mirror image flips the sign of every z, so the sign is the
    hand's chirality as each sensor's model has it."""
    return float(np.mean(np.asarray(local, float)[1:, 2]))


def same_handedness(glove_local, cam_local) -> bool:
    """True when both hands lie on the same side of their palm planes (the
    bare sign test; `handedness` is the reported diagnostic)."""
    return palm_side_mm(glove_local) * palm_side_mm(cam_local) > 0


def handedness(glove_local, cam_local) -> str:
    """SAME, OPPOSITE or INDETERMINATE: a diagnostic, never a classification.

    INDETERMINATE when either hand's margin (|palm_side_mm|) is under
    HANDEDNESS_MARGIN_MM: a flat hand lies in its palm plane and its sign
    says nothing. OPPOSITE suggests the tracker's fit is mirrored against the
    glove; nothing is reflected or corrected. The thumb's in-plane side
    cannot tell this: y runs from the index to the pinky knuckle on either
    hand, which puts the thumb on -y for a left and a right hand alike."""
    g, c = palm_side_mm(glove_local), palm_side_mm(cam_local)
    if min(abs(g), abs(c)) < HANDEDNESS_MARGIN_MM:
        return INDETERMINATE
    return SAME if g * c > 0 else OPPOSITE


# --- stillness -------------------------------------------------------------------
def window_speed_mm_s(t, pts, t0: float, half_s: float = STILL_WINDOW_S,
                      step_s: float = SPEED_STEP_S) -> float:
    """Mean 21-point speed, mm/s, over the frames within `half_s` of `t0`.

    `pts` is N x 21 x 3 in mm; each frame is wrist-centred first (the hand's
    translation removed, its orientation and shape changes kept). Each
    frame's speed is on the `leap_hand.transition` base: the mean point
    displacement between the earliest and the latest frame within half a
    `step_s` of it, over their time apart. NaN when no frame has such a pair."""
    t = np.asarray(t, float)
    p = np.asarray(pts, float)
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order] - p[order][:, WRIST:WRIST + 1, :]
    half = step_s / 2.0
    speeds = []
    for k in np.flatnonzero(np.abs(t - t0) <= half_s):
        a = int(np.searchsorted(t, t[k] - half, side="left"))
        b = int(np.searchsorted(t, t[k] + half, side="right")) - 1
        if b <= a or t[b] - t[a] <= 0:
            continue
        d = np.linalg.norm(p[b] - p[a], axis=1)
        if np.all(np.isfinite(d)):
            speeds.append(float(d.mean()) / float(t[b] - t[a]))
    return float(np.mean(speeds)) if speeds else float("nan")


def stillness(glove_mm_s: float, cam_mm_s: float) -> str:
    """"still", "moving (glove)", "moving (camera)", "moving (both)", or
    "speed not measured (...)" when a sensor has no speed."""
    missing = [n for n, v in (("glove", glove_mm_s), ("camera", cam_mm_s))
               if not np.isfinite(v)]
    if missing:
        return f"speed not measured ({', '.join(missing)})"
    moving = [n for n, v in (("glove", glove_mm_s), ("camera", cam_mm_s)) if v > STILL_MM_S]
    if not moving:
        return "still"
    return f"moving ({'both' if len(moving) == 2 else moving[0]})"
