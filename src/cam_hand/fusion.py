"""Fuse a glove skeleton and a camera skeleton into one hand.

The two sensors are not averaged. Averaging a measurement with a guess makes
both worse; instead each degree of freedom is taken from the sensor that
actually measures it:

  flexion (curl)        GLOVE — stretch sensors measure it directly, and they
                        keep working when the fingers hide behind the palm
  abduction (spread)    CAMERA — the glove has no sensor for side-to-side
  thumb opposition      CAMERA — rotation at the thumb base, the motion that
                        makes pinch look like an open hand to the glove
  absolute pose         CAMERA — the glove reports nothing outside the wrist

How a fused finger is built
  1. Scale + rotate the camera skeleton into the glove's frame, solving on
     the near-rigid palm landmarks (wrist + the four knuckles) — align.py.
  2. Build a palm frame from the glove: the palm normal plus two in-plane
     axes. In that frame, a finger's direction splits into an out-of-plane
     component (curl — the glove's) and an in-plane azimuth (spread — the
     camera's).
  3. Rotate each glove finger chain rigidly about its knuckle so its azimuth
     matches the camera's while its curl is untouched.

Because step 3 is a rotation about the knuckle, every bone keeps exactly the
glove's length: the fused hand can never shrink, which a per-joint blend of
two point clouds would do (the same reason the exporters use a medoid frame
rather than a mean).

The thumb is handled differently: its whole direction is taken from the
camera, not just the azimuth, because opposition IS out-of-plane rotation and
the glove cannot see it.

Confidence gating: below `min_score`, or when the camera never saw the hand,
the glove skeleton is returned untouched. Fusion therefore degrades to
glove-only rather than to garbage.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .align import umeyama
from .features import INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST

# Finger chains as (knuckle, ..., tip) indices in the 21-keypoint layout.
FINGER_CHAINS: Dict[str, List[int]] = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# Near-rigid landmarks used to solve the camera -> glove transform.
PALM_IDX = [WRIST, 5, 9, 13, 17]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def palm_frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(normal, in-plane x, in-plane y) for a 21-point hand.

    x runs wrist -> middle knuckle (down the palm); the normal comes from the
    index/pinky knuckle spread, so it is the axis fingers abduct around.
    """
    wrist = pts[WRIST]
    x = _unit(pts[MIDDLE_MCP] - wrist)
    across = pts[PINKY_MCP] - pts[INDEX_MCP]
    n = _unit(np.cross(x, across))
    y = _unit(np.cross(n, x))
    return n, x, y


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate pi about any axis orthogonal to a
        axis = _unit(np.cross(a, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 1e-9:
            axis = _unit(np.cross(a, np.array([0.0, 1.0, 0.0])))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def camera_into_glove_frame(glove: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Scale + rotate the camera hand onto the glove hand via the palm."""
    R, s, t = umeyama(cam[PALM_IDX], glove[PALM_IDX], with_scale=True)
    return (s * (R @ cam.T)).T + t


def fuse_skeletons(
    glove_pts: Sequence[Sequence[float]],
    cam_pts: Optional[Sequence[Sequence[float]]],
    cam_score: float = 1.0,
    min_score: float = 0.5,
    thumb_from_camera: bool = True,
    fingers: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, dict]:
    """Fuse one glove frame with one camera frame -> 21 points + info.

    Both inputs are 21 x 3 wrist-centred metres. Returns the fused points
    (wrist-centred) and a dict describing what was actually used, so callers
    can report how often the camera contributed.
    """
    G = np.asarray(glove_pts, dtype=float)
    info = {"camera_used": False, "reason": "", "fingers_adjusted": []}

    if cam_pts is None:
        info["reason"] = "no camera frame"
        return G, info
    if cam_score < min_score:
        info["reason"] = f"camera score {cam_score:.2f} < {min_score:.2f}"
        return G, info

    C = camera_into_glove_frame(G, np.asarray(cam_pts, dtype=float))
    n, _x, _y = palm_frame(G)
    fused = G.copy()
    names = list(FINGER_CHAINS) if fingers is None else list(fingers)

    for finger in names:
        chain = FINGER_CHAINS[finger]
        knuckle = G[chain[0]]
        d_g = _unit(G[chain[-1]] - knuckle)
        d_c = _unit(C[chain[-1]] - C[chain[0]])
        if np.linalg.norm(d_g) < 1e-9 or np.linalg.norm(d_c) < 1e-9:
            continue

        if finger == "thumb" and thumb_from_camera:
            # Opposition is out-of-plane rotation at the base: take the
            # camera's whole direction, keep the glove's bone lengths.
            target = d_c
        else:
            # Keep the glove's out-of-plane component (curl), adopt the
            # camera's in-plane component (spread).
            out_g = float(np.dot(d_g, n))
            in_c = d_c - float(np.dot(d_c, n)) * n
            if np.linalg.norm(in_c) < 1e-9:
                continue
            in_len = float(np.sqrt(max(0.0, 1.0 - out_g * out_g)))
            target = _unit(out_g * n + in_len * _unit(in_c))

        R = rotation_between(d_g, target)
        fused[chain] = (R @ (G[chain] - knuckle).T).T + knuckle
        info["fingers_adjusted"].append(finger)

    info["camera_used"] = bool(info["fingers_adjusted"])
    fused = fused - fused[WRIST]
    return fused, info


# --- time alignment ----------------------------------------------------

def pair_by_time(glove: Sequence[dict], cam: Sequence[dict],
                 max_dt: float = 0.05) -> List[Tuple[dict, Optional[dict]]]:
    """Match camera frames to glove frames by wall-clock, per hand.

    Both recorders stamp time.time() at write, so the streams share a clock
    even though the glove runs at ~60 Hz and the camera at ~30 Hz. Each glove
    frame takes the nearest camera frame of the SAME hand within max_dt
    seconds; frames with no partner pair with None and stay glove-only.
    """
    by_hand: Dict[str, List[dict]] = {}
    for c in cam:
        by_hand.setdefault(c["hand_side"], []).append(c)
    for v in by_hand.values():
        v.sort(key=lambda d: d["wall_time"])

    out: List[Tuple[dict, Optional[dict]]] = []
    for g in sorted(glove, key=lambda d: d["wall_time"]):
        candidates = by_hand.get(g["hand_side"], [])
        best, best_dt = None, max_dt
        # linear scan is fine: takes are seconds long, not hours
        for c in candidates:
            dt = abs(c["wall_time"] - g["wall_time"])
            if dt <= best_dt:
                best, best_dt = c, dt
        out.append((g, best))
    return out
