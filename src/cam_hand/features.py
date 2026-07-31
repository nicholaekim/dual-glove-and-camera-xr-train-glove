"""Pose features over 21 wrist-centred keypoints, and a parameter-free
classifier to test how well they separate poses.

Two feature groups, deliberately split by which sensor can measure them:

  FLEXION (5)   wrist->fingertip distance per finger — "extension profile".
                Exactly what stretch sensors measure, so glove and camera
                recordings produce comparable numbers. This is the glove
                pipeline's feature set (scripts/analyze_poses.py).

  SPREAD (6)    adjacent fingertip gaps, thumb-to-pinky-base distance, and
                the thumb tip's offset from the palm plane. These encode
                abduction and thumb opposition — the degrees of freedom the
                glove is blind to and the camera sees directly.

Both are normalized by palm length (wrist -> middle knuckle) by default,
which removes hand-size and camera-scale differences and makes glove and
camera features live on the same axes. Set normalize=False for raw metres.

Chirality matters for the signed features and is handled explicitly. A left
hand is the mirror of a right one, so the palm normal built from the knuckles
points out of the back of one hand and out of the palm of the other: the same
physical thumb opposition would then get opposite signs on the two hands, and
a classifier trained on both would see one gesture as two. Every function
that uses the normal therefore takes `hand_side` and flips it for the left
hand, putting both hands in one consistent space.

The classifier is leave-one-out nearest centroid: hold out one sample,
rebuild each pose's centroid from the rest, assign the held-out sample to the
nearest centroid. Zero parameters, nothing to tune — so a high score means
the data separates, not that a model was fitted well.
"""
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .landmarks import MP21_NAMES

WRIST = 0
TIP_IDX = [MP21_NAMES.index(n) for n in
           ("THUMB_TIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_TIP", "PINKY_TIP")]
INDEX_MCP = MP21_NAMES.index("INDEX_FINGER_MCP")
MIDDLE_MCP = MP21_NAMES.index("MIDDLE_FINGER_MCP")
PINKY_MCP = MP21_NAMES.index("PINKY_MCP")

FLEXION_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
SPREAD_NAMES = ["thumb-index", "index-middle", "middle-ring", "ring-pinky",
                "thumb-pinkyMCP", "thumb-out-of-palm"]
ALL_NAMES = FLEXION_NAMES + SPREAD_NAMES


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def palm_length(pts: Sequence[Sequence[float]]) -> float:
    """Wrist -> middle knuckle: a stable per-hand scale reference."""
    return _norm(_sub(pts[MIDDLE_MCP], pts[WRIST]))


def flexion_features(pts, normalize: bool = True) -> List[float]:
    """5 wrist-to-fingertip distances (the glove-comparable profile)."""
    scale = palm_length(pts) if normalize else 1.0
    scale = scale if scale > 1e-9 else 1.0
    return [_norm(_sub(pts[t], pts[WRIST])) / scale for t in TIP_IDX]


def palm_normal(pts, hand_side: str = "right") -> List[float]:
    """Palm normal, oriented the same way on both hands (see module docstring)."""
    n = _cross(_sub(pts[INDEX_MCP], pts[WRIST]), _sub(pts[PINKY_MCP], pts[WRIST]))
    if str(hand_side).lower().startswith("l"):
        n = [-n[0], -n[1], -n[2]]
    return n


def spread_features(pts, normalize: bool = True,
                    hand_side: str = "right") -> List[float]:
    """6 camera-only features: finger abduction + thumb opposition."""
    scale = palm_length(pts) if normalize else 1.0
    scale = scale if scale > 1e-9 else 1.0
    tips = [pts[t] for t in TIP_IDX]
    gaps = [_norm(_sub(tips[i + 1], tips[i])) / scale for i in range(4)]
    opposition = _norm(_sub(tips[0], pts[PINKY_MCP])) / scale

    # Signed distance of the thumb tip from the palm plane (wrist, index MCP,
    # pinky MCP). Opposition lifts the thumb out of that plane; pure flexion
    # keeps it in. This is the motion that makes pinch invisible to the glove.
    n = palm_normal(pts, hand_side)
    nn = _norm(n)
    out_of_plane = (_dot(_sub(tips[0], pts[WRIST]), n) / nn / scale) if nn > 1e-9 else 0.0
    return gaps + [opposition, out_of_plane]


def all_features(pts, normalize: bool = True,
                 hand_side: str = "right") -> List[float]:
    return (flexion_features(pts, normalize)
            + spread_features(pts, normalize, hand_side))


def mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


def loo_nearest_centroid(samples: Sequence[Tuple[str, Sequence[float]]],
                         cols: Sequence[int] = None):
    """Leave-one-out nearest-centroid over (label, features) samples.

    cols selects a feature subset (e.g. flexion only). Returns
    (n_correct, n_total, [(true, predicted, index), ...]) for the misses.
    """
    def pick(v):
        return [v[i] for i in cols] if cols is not None else list(v)

    wrong = []
    for i, (label, feats) in enumerate(samples):
        sums: Dict[str, List[float]] = defaultdict(lambda: None)
        counts: Dict[str, int] = defaultdict(int)
        for j, (lab2, f2) in enumerate(samples):
            if j == i:
                continue
            v = pick(f2)
            if sums[lab2] is None:
                sums[lab2] = list(v)
            else:
                for k in range(len(v)):
                    sums[lab2][k] += v[k]
            counts[lab2] += 1
        target = pick(feats)
        best, best_d = None, float("inf")
        for lab2, s in sums.items():
            c = [v / counts[lab2] for v in s]
            d = sum((a - b) ** 2 for a, b in zip(target, c))
            if d < best_d:
                best, best_d = lab2, d
        if best != label:
            wrong.append((label, best, i))
    return len(samples) - len(wrong), len(samples), wrong


FLEXION_COLS = list(range(len(FLEXION_NAMES)))
SPREAD_COLS = list(range(len(FLEXION_NAMES), len(ALL_NAMES)))
ALL_COLS = list(range(len(ALL_NAMES)))
