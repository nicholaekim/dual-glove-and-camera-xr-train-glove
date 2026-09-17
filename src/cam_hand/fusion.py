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
  1. Rotate the camera skeleton into the glove's frame, solving on the
     near-rigid palm landmarks (wrist + the four knuckles) — align.py.
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

Step 1 (`with_scale`) depends on which camera took the frame, and the plan
(section 3, "Coordinate policy") is explicit about it:

  MediaPipe      with_scale=True. Umeyama over the five palm landmarks,
                 rotation + uniform scale. Its world landmarks are a
                 normalised hand — a shape, not a size — so the fit has to
                 solve for scale or the palms will not sit on each other at
                 all, and there is no real size to protect.
  Ultraleap      with_scale=False. NOT a rigid least-squares fit either:
                 `palm_frame_transfer` builds an orthonormal palm basis on
                 each hand out of unit direction vectors and rotates by
                 `B_glove @ B_camera.T`, then puts the camera wrist on the
                 glove wrist. Scale 1, no least squares.

                 Fitting five palm points minimises a squared distance, so
                 when the palms are not the same SIZE the fit trades rotation
                 against that mismatch and tilts the hand a few degrees to
                 split the difference. The glove reports the XR Trainer
                 TEMPLATE hand, whose palm is not the operator's, so the
                 mismatch is always present — and a few degrees of tilt lands
                 directly in the spread the camera is there to supply. A
                 basis of unit vectors cannot make that trade.

`info["alignment"]` records which was used, and on the metric path
`info["kabsch_rmse_mm"]` reports the residual of a rigid 5-point palm fit:
how far the two palms are from being one object. It is a diagnostic — the
number Phase 5 would act on — and is never compared against a threshold,
because a large value means the template is the wrong size, not that this
frame is bad.

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


def palm_basis(pts: np.ndarray) -> np.ndarray:
    """Orthonormal 3x3 whose columns are the palm axes of this hand.

    Built from `palm_frame`, so every column is a unit vector derived from a
    DIRECTION between landmarks. Palm size therefore cannot enter it: a hand
    and a 10% larger copy of the same hand have the same basis.

    Both hands go through the same recipe, so a left hand's basis is the
    mirror of a right one's and the two cancel when one is expressed in the
    other — which is what keeps `R` below a proper rotation rather than a
    reflection, without needing a determinant correction.
    """
    n, x, y = palm_frame(pts)
    return np.column_stack([x, y, n])


def palm_frame_transfer(glove: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Rotate the camera hand into the glove's palm frame; wrist on wrist.

    The metric path. `R = B_glove @ B_camera.T` takes the camera's palm axes
    onto the glove's, then the camera wrist is placed on the glove wrist.
    Rotation and translation only — no scale, no least squares.

    Why not Kabsch here. Fitting five palm landmarks minimises a squared
    distance, so when the two palms are not the same SHAPE the fit trades
    rotation against that mismatch and returns a hand tilted to split the
    difference. The XR Trainer glove reports a template hand whose palm is
    not the operator's, so the mismatch is always present, and the tilt lands
    straight in the spread the camera is supposed to be supplying.

    Measured on the synthetic hands in `tests/test_fusion.py`, holding
    orientation fixed and changing only palm proportions (20% wider, 20%
    longer, wider-and-shorter, 15% narrower): this transfer tilts by exactly
    0 degrees in every case, because a basis of unit vectors gives a
    dimension nowhere to enter. The 5-point rigid fit tilts by 0.35 to 0.70
    degrees, in a fixed direction per mismatch — small, but systematic and
    pure artefact. A hand that is evenly BIGGER fools neither (0 degrees for
    both); it is a change of PROPORTIONS that does it, which is precisely
    what a template hand worn on a real hand is.

    The size disagreement is still worth knowing about, so it is measured
    and reported (`info["kabsch_rmse_mm"]`) rather than silently absorbed.
    """
    R = palm_basis(glove) @ palm_basis(cam).T
    return (R @ (cam - cam[WRIST]).T).T + glove[WRIST]


def palm_fit_rmse_mm(glove: np.ndarray, cam: np.ndarray) -> float:
    """Residual of a RIGID 5-point palm fit, in millimetres. Diagnostic only.

    How far the two palms are from being the same rigid object: mostly the
    difference between the glove's template hand and the real one. Reported
    so it can be watched (and is what Phase 5 would measure), never compared
    against a threshold and never used to reject a frame — a big number here
    means the template is the wrong size, not that the tracking is bad.
    """
    R, _s, t = umeyama(cam[PALM_IDX], glove[PALM_IDX], with_scale=False)
    fitted = (R @ cam[PALM_IDX].T).T + t
    err = np.linalg.norm(fitted - glove[PALM_IDX], axis=1)
    return float(np.sqrt((err ** 2).mean()) * 1000.0)


def camera_into_glove_frame(glove: np.ndarray, cam: np.ndarray,
                            with_scale: bool = True) -> np.ndarray:
    """Put the camera hand in the glove's frame, using the palm.

    with_scale=True is the MediaPipe fit: Umeyama over the five palm
    landmarks, rotation + uniform scale, because a normalised hand has no
    size of its own to preserve. with_scale=False is the metric path and goes
    through `palm_frame_transfer`. See the module docstring.
    """
    if not with_scale:
        return palm_frame_transfer(glove, cam)
    R, s, t = umeyama(cam[PALM_IDX], glove[PALM_IDX], with_scale=True)
    return (s * (R @ cam.T)).T + t


def fuse_skeletons(
    glove_pts: Sequence[Sequence[float]],
    cam_pts: Optional[Sequence[Sequence[float]]],
    cam_score: float = 1.0,
    min_score: float = 0.5,
    thumb_from_camera: bool = True,
    fingers: Optional[Sequence[str]] = None,
    with_scale: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Fuse one glove frame with one camera frame -> 21 points + info.

    Both inputs are 21 x 3 wrist-centred metres. `with_scale` picks the
    alignment: True for a normalised (MediaPipe) camera, False for a metric
    one (Ultraleap). Returns the fused points (wrist-centred) and a dict
    describing what was actually used, so callers can report how often the
    camera contributed and under which fit.
    """
    G = np.asarray(glove_pts, dtype=float)
    info = {"camera_used": False, "reason": "", "fingers_adjusted": [],
            "alignment": "similarity" if with_scale else "palm_frame",
            "kabsch_rmse_mm": None}

    if cam_pts is None:
        info["reason"] = "no camera frame"
        return G, info
    if cam_score < min_score:
        info["reason"] = f"camera score {cam_score:.2f} < {min_score:.2f}"
        return G, info

    C_in = np.asarray(cam_pts, dtype=float)
    if not with_scale:
        # Diagnostic, not a gate: how far the two palms are from being one
        # rigid object. Only meaningful when both sides are metric.
        info["kabsch_rmse_mm"] = palm_fit_rmse_mm(G, C_in)
    C = camera_into_glove_frame(G, C_in, with_scale=with_scale)
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

CAPTURE_CLOCK = "capture_time"
WALL_CLOCK = "wall_time"
AUTO = "auto"


def pairing_clock(*streams: Sequence[dict]) -> str:
    """Which clock these streams can be paired on: capture_time or wall_time.

    `wall_time` is a WRITER timestamp — the moment a line was written. Both
    recorders drain a queue and write the burst it held, so frames the
    sensors captured hundreds of milliseconds apart can carry wall_times a
    millisecond apart. Pairing on it matches on write order rather than on
    when the hand was in the pose, and at 90 Hz against 5 Hz that is most of
    the error budget.

    `capture_time` is when the sensor actually had the frame: for the camera,
    `time.time()` in the tracking callback minus the frame's age; for the
    glove, the arrival stamp taken on the OSC server thread. It is the right
    clock, and it is used only when EVERY frame of EVERY stream carries it —
    a mixture would silently compare two different clocks, which is worse
    than using the blunt one consistently. Recordings made before this
    existed have no `capture_time` at all, so `wall_time` stays the fallback
    and those files still fuse.
    """
    for rows in streams:
        if not rows or any(r.get(CAPTURE_CLOCK) is None for r in rows):
            return WALL_CLOCK
    return CAPTURE_CLOCK


def _stamp(d: dict, clock: str) -> float:
    """One frame's time on `clock`, falling back to wall_time if it lacks it.

    The fallback cannot fire under `pairing_clock`, which only chooses a
    clock every frame has; it is there for a caller that forces one.
    """
    v = d.get(clock)
    return float(v if v is not None else d[WALL_CLOCK])


def pair_by_time(glove: Sequence[dict], cam: Sequence[dict],
                 max_dt: float = 0.05,
                 clock: str = AUTO) -> List[Tuple[dict, Optional[dict]]]:
    """Match camera frames to glove frames by a shared clock, per hand.

    Each glove frame takes the nearest camera frame of the SAME hand within
    max_dt seconds; frames with no partner pair with None and stay
    glove-only. `clock` is "auto" (ask `pairing_clock`) or a key to force —
    see `pairing_clock` for why the choice matters.
    """
    if clock == AUTO:
        clock = pairing_clock(glove, cam)

    by_hand: Dict[str, List[dict]] = {}
    for c in cam:
        by_hand.setdefault(c["hand_side"], []).append(c)
    for v in by_hand.values():
        v.sort(key=lambda d: _stamp(d, clock))

    out: List[Tuple[dict, Optional[dict]]] = []
    for g in sorted(glove, key=lambda d: _stamp(d, clock)):
        t_g = _stamp(g, clock)
        candidates = by_hand.get(g["hand_side"], [])
        best, best_dt = None, max_dt
        # linear scan is fine: takes are seconds long, not hours
        for c in candidates:
            dt = abs(_stamp(c, clock) - t_g)
            if dt <= best_dt:
                best, best_dt = c, dt
        out.append((g, best))
    return out
