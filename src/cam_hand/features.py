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
                         cols: Sequence[int] = None,
                         standardize: bool = False):
    """Leave-one-out nearest-centroid over (label, features) samples.

    cols selects a feature subset (e.g. flexion only). Returns
    (n_correct, n_total, [(true, predicted, index), ...]) for the misses.

    standardize z-scores each feature before measuring distance. It matters
    whenever features live on different scales: raw Euclidean distance weights
    a dimension by its magnitude, so large-valued features (wrist-to-fingertip
    distances, ~1-2 palm lengths) drown out smaller ones (thumb offset out of
    the palm plane, ~0.1-0.6) even when the small ones separate the classes
    better. Mean and spread are computed from the TRAINING samples only, so
    the held-out sample never influences its own scaling.
    """
    def pick(v):
        return [v[i] for i in cols] if cols is not None else list(v)

    wrong = []
    for i, (label, feats) in enumerate(samples):
        if standardize:
            train = [pick(f) for j, (_l, f) in enumerate(samples) if j != i]
            n_f = len(train[0])
            mu = [sum(r[k] for r in train) / len(train) for k in range(n_f)]
            sd = []
            for k in range(n_f):
                var = sum((r[k] - mu[k]) ** 2 for r in train) / len(train)
                sd.append(math.sqrt(var) or 1.0)

            def pick_scaled(v, _mu=mu, _sd=sd, _pick=pick):
                return [(a - m) / s for a, m, s in zip(_pick(v), _mu, _sd)]
        else:
            pick_scaled = pick
        sums: Dict[str, List[float]] = defaultdict(lambda: None)
        counts: Dict[str, int] = defaultdict(int)
        for j, (lab2, f2) in enumerate(samples):
            if j == i:
                continue
            v = pick_scaled(f2)
            if sums[lab2] is None:
                sums[lab2] = list(v)
            else:
                for k in range(len(v)):
                    sums[lab2][k] += v[k]
            counts[lab2] += 1
        target = pick_scaled(feats)
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


# --- proximal-bone geometry -------------------------------------------
# Everything below is ADDITIVE: no existing feature, column index or
# classifier result changes. It exists because a fingertip is the wrong place
# to read abduction from.
#
# A finger's in-plane direction (its azimuth in the palm plane) is what the
# camera is there to supply. Reading it from knuckle -> TIP mixes it with
# curl: once the PIP and DIP are bent, the tip swings far off the bone's own
# line and a few degrees of flexion error becomes tens of degrees of apparent
# abduction. The PROXIMAL bone — knuckle (MCP) to the next joint (PIP) — moves
# only with the joint that actually abducts, so its azimuth is the abduction
# and nothing else. The thumb has no PIP, so its proximal bone is CMC -> MCP,
# the same "first bone out of the chain" rule.

PROXIMAL_BONE: Dict[str, Tuple[int, int]] = {
    "thumb":  (1, 2),      # THUMB_CMC -> THUMB_MCP
    "index":  (5, 6),
    "middle": (9, 10),
    "ring":   (13, 14),
    "pinky":  (17, 18),
}

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# Adjacent pairs whose in-plane angle is the spread a camera can see.
ADJACENT_PAIRS = [("thumb", "index"), ("index", "middle"),
                  ("middle", "ring"), ("ring", "pinky")]
ADJACENT_SPREAD_NAMES = [f"{a}-{b}" for a, b in ADJACENT_PAIRS]


def _unit3(v):
    n = _norm(v)
    return [v[0] / n, v[1] / n, v[2] / n] if n > 1e-12 else list(v)


def palm_axes(pts, hand_side: str = "right"):
    """Orthonormal (normal, in-plane x, in-plane y) for measuring azimuths.

    x runs wrist -> middle knuckle. The normal is `palm_normal`, so it is
    already flipped for a left hand; y is then flipped back for the left hand
    too, which makes an azimuth measured here mean the SAME physical abduction
    on both hands instead of changing sign with chirality.
    """
    wrist = pts[WRIST]
    x = _unit3(_sub(pts[MIDDLE_MCP], wrist))
    n = _unit3(palm_normal(pts, hand_side))
    # re-orthogonalise: the knuckle triangle is not exactly perpendicular to x
    n = _unit3([n[i] - _dot(n, x) * x[i] for i in range(3)])
    y = _cross(n, x)
    if str(hand_side).lower().startswith("l"):
        y = [-y[0], -y[1], -y[2]]
    return n, x, y


def proximal_azimuth_deg(pts, finger: str, hand_side: str = "right") -> float:
    """In-plane angle of a finger's PROXIMAL bone, degrees, 0 = down the palm.

    Positive is toward the thumb side on both hands (see `palm_axes`).
    """
    a, b = PROXIMAL_BONE[finger]
    d = _sub(pts[b], pts[a])
    _n, x, y = palm_axes(pts, hand_side)
    return math.degrees(math.atan2(_dot(d, y), _dot(d, x)))


def adjacent_spreads_deg(pts, hand_side: str = "right") -> List[float]:
    """The 4 adjacent in-plane finger gaps in degrees, unsigned.

    Unsigned because it is the OPENING between two fingers, which is the same
    physical quantity whichever hand it is on.
    """
    az = {f: proximal_azimuth_deg(pts, f, hand_side) for f in FINGER_ORDER}
    out = []
    for a, b in ADJACENT_PAIRS:
        d = (az[a] - az[b] + 180.0) % 360.0 - 180.0
        out.append(abs(d))
    return out


def thumb_index_gap(pts, normalize: bool = True) -> float:
    """Thumb tip to index tip over palm length — the pinch measurement."""
    scale = palm_length(pts) if normalize else 1.0
    scale = scale if scale > 1e-9 else 1.0
    return _norm(_sub(pts[TIP_IDX[1]], pts[TIP_IDX[0]])) / scale


# Rows of the per-DOF report: (label, kind, index-within-kind).
DOF_ROWS = ([(f"curl {n}", "curl", i) for i, n in enumerate(FLEXION_NAMES)]
            + [(f"spread {n}", "spread_deg", i)
               for i, n in enumerate(ADJACENT_SPREAD_NAMES)]
            + [("thumb-index gap", "ti_gap", 0)])


def dof_values(pts, hand_side: str = "right") -> List[float]:
    """The per-DOF report row values for one hand, in DOF_ROWS order."""
    curls = flexion_features(pts)
    spreads = adjacent_spreads_deg(pts, hand_side)
    gap = thumb_index_gap(pts)
    picked = {"curl": curls, "spread_deg": spreads, "ti_gap": [gap]}
    return [picked[kind][i] for _label, kind, i in DOF_ROWS]


def median(values: Sequence[float]) -> float:
    v = sorted(values)
    n = len(v)
    if not n:
        return float("nan")
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def loo_take_nearest_centroid(samples: Sequence[Tuple[str, str, Sequence[float]]],
                              cols: Sequence[int] = None):
    """Leave-one-TAKE-out nearest centroid over (label, take, features).

    The difference from `loo_nearest_centroid` is what "held out" means. There,
    one SAMPLE is held out, so the other hand of the very same take — the same
    five seconds, the same hand pose, the same tracking state — is still in the
    training set and the test mostly measures whether the two hands of one take
    look alike. Here every sample sharing the held-out sample's take is removed,
    so nothing recorded at that moment can vote on it.

    Consequence worth stating: with a single take per pose, holding out that
    take removes the pose's only training samples, there is no centroid for the
    true label to be nearest to, and every held-out sample is necessarily wrong.
    That is not a fusion result — it is a statement about the recording, and it
    stays 0% until a second take of each pose exists.

    Returns (n_correct, n_total, [(true, predicted, index), ...]).
    """
    def pick(v):
        return [v[i] for i in cols] if cols is not None else list(v)

    wrong = []
    for i, (label, take, feats) in enumerate(samples):
        sums: Dict[str, List[float]] = {}
        counts: Dict[str, int] = defaultdict(int)
        for j, (lab2, take2, f2) in enumerate(samples):
            if take2 == take:
                continue                      # the whole held-out take, not one row
            v = pick(f2)
            if lab2 not in sums:
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
