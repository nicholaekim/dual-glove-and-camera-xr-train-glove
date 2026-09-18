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

WHEN THE CAMERA IS WRONG
------------------------
The split above says WHICH sensor owns a degree of freedom. It does not say
whether the camera actually measured it on this frame, and the first real
simultaneous session showed that the difference matters:

  pinch       the glove is numerically identical to its open palm (curl
              1.97 on the index, thumb-index gap 0.99 — the template hand
              has no sensor for opposition). The camera sees index curl
              1.27 and the gap closing to 0.35. The camera is right.
  thumbs_up   the glove has the four fingers curled (0.66). The camera,
              looking at an edge-on hand from below, reports them nearly
              straight (1.69) with grab_strength 0.04. The glove is right
              and the camera is confidently wrong.

Taking the camera whenever a camera frame exists therefore imports the second
case along with the first, and the fused classifier scored BELOW glove-only.
So each camera-owned DOF is now gated, per frame, on evidence that does not
come from the tracker's own opinion of itself:

  frame level   the hand has been visible for at least `min_visible_time_us`,
                its `hand_id` has not changed in the last `hand_id_settle_s`,
                and the palm is inside the module's central field.
  spread        per finger: only when the GLOVE says that finger is not
                strongly curled (a curled finger has no abduction to see and
                its proximal bone points at the camera) and the palm is
                turned toward the module.
  thumb         only when the palm is turned toward the module AND the camera
                agrees with the glove about the other four fingers. A camera
                that has the four fingers wrong has the hand's orientation
                wrong, and the thumb is the DOF that orientation error moves
                the most. This is what rejects thumbs_up and admits pinch.

`grab_strength`, `pinch_strength` and `confidence` are deliberately NOT used.
The first two are model outputs — the same model that produced the joints, so
they cannot corroborate it — and LeapC reports `confidence` as a constant 1.0.

Every threshold below is a named parameter on `GateParams` with an empirical
starting value read off that one session, not a calibrated constant, and every
one of them is printed in the report so a later session can move it.

No frame is ever dropped. A frame that fails every gate is the glove skeleton,
unchanged, which is exactly what the pipeline produced before the camera
existed.

WHEN THE GLOVE IS WRONG: THE RAIL-DISAGREEMENT OVERRIDE
-------------------------------------------------------
The split above gives the glove the finger curls outright, and on
`recordings/sync_day1` that is wrong in one specific, reproducible way.

Whenever a finger is straight the glove does not report a measurement, it
reports a CONSTANT: index 1.97 on the left hand and 1.98 on the right, middle
2.07/2.08, ring 1.97, pinky 1.71, thumb 1.43 — identical to two decimals on
every open-palm frame of all 59 takes. That constant is the finger's RAIL: the
top of the stretch sensor's range, where the fabric has stopped stretching and
the number has stopped meaning anything.

In all ten pinch takes the glove index sits exactly on its rail while the
camera watches the index fold down to meet the thumb (camera index curl
1.16-1.39 against 1.72-1.81 for a genuinely open palm, thumb-index tip gap
0.11-0.28 palm lengths). The fused pinch therefore had a perfectly straight
index — the one joint the gesture is named after.

This is NOT a dead zone. An isolated slow index bend leaves the rail as soon
as the camera sees any flexion, so the glove does measure that finger. It
fails only at the top of its range, and only there.

So one narrow override, per finger, per frame, with three conditions that must
hold together:

  on the rail    the glove's curl is within `tol` (0.005 — the value is
                 bit-exact, so this is a float-equality test, not a band) of
                 the rail LEARNED for that hand and finger.
  camera trusted the same frame gates the spread and thumb already use —
                 visible time, no recent hand-id change, central field, and the
                 viewing angle inside `view_gate_deg`.
  camera flexed  the camera's curl for that finger is below its open reference
                 minus `margin`.

...and then a hysteresis, because a single frame is not evidence: the override
arms only after `enter_frames` consecutive qualifying frames and disarms only
after `exit_frames` consecutive non-qualifying ones.

WHY THE RAIL IS LEARNED AND THE CAMERA'S OPEN REFERENCE IS NOT
  The rail is a SENSOR ARTEFACT. There is no physical reason index should
  saturate at 1.97 and pinky at 1.71, or that the two hands differ in the
  third decimal; those numbers are facts about this glove, and the only way to
  know them is to look. `learn_rails` finds them as the most frequent curl
  value per hand and finger — a saturating sensor puts a huge bit-exact spike
  where a moving one spreads out — and refuses to believe one unless it also
  sits at the top of the observed range, so a session that never shows a
  finger straight teaches no rail and the override simply never arms.

  The camera's open reference is a GEOMETRIC FACT, so it is a constant
  (`cam_open_curl`). Curl here is fingertip-to-wrist over palm length: a
  dimensionless ratio, so a straight finger reads about the same number on any
  hand and any metric camera. Measured on sync_day1's open-palm frames the two
  hands differ by 0.03 (index) to 0.05 (pinky) — an order of magnitude under
  the 0.25 margin.

  It must NOT be learned from the frames being fused, and that is the whole
  point. Without pose labels the only label-free way to call a frame "open" is
  to ask the glove — and the glove saying "open" is exactly the claim under
  suspicion. A reference learned that way would let a session of nothing but
  pinches teach the detector that a pinched index is what open looks like, and
  the override would never fire on the one case it exists for. A constant
  cannot be poisoned by the data it is judging.

WHAT THE OVERRIDE DOES
  The finger is rebuilt from the glove's knuckle with the camera's three bone
  DIRECTIONS and the glove's three bone LENGTHS (`transfer_finger_flexion`).
  It is the spread transfer's idea carried one step further: there a single
  rotation about the knuckle adopted the camera's azimuth, here each bone in
  turn adopts the camera's direction, and in both cases the template's bones
  keep exactly their lengths.

  A consequence worth stating plainly, because it shows up in every report:
  the fused curl does NOT land on the camera's number. It lands about 11%
  above it, because the curl metric is a LENGTH ratio and the glove's template
  index is about 11% longer per palm length than the operator's (rail 1.97
  against the camera's 1.75 open). Measured over the ten pinch takes the fused
  index is 1.29-1.56 against the camera's 1.16-1.39 — and the two agree to
  within 0.003 once each is expressed as a fraction of its own sensor's open
  value, which is the comparison that is actually free of the template. The
  joint ANGLES are the camera's exactly; only the bones they are hung on are
  the glove's, and making the number match outright would mean rescaling the
  template, which is the one thing this module never does.

`fingers` defaults to ("index",) — the finger the session actually shows the
failure on. The rest are implemented and off.
"""
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .align import umeyama
from .features import (
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    PROXIMAL_BONE,
    WRIST,
    flexion_features,
    median,
)

# Finger chains as (knuckle, ..., tip) indices in the 21-keypoint layout.
FINGER_CHAINS: Dict[str, List[int]] = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# Same order as features.FLEXION_NAMES, so a curl index is a finger name.
FINGER_NAMES = list(FINGER_CHAINS)

# Near-rigid landmarks used to solve the camera -> glove transform.
PALM_IDX = [WRIST, 5, 9, 13, 17]

# The camera-owned degrees of freedom, in report order. Four finger azimuths
# plus the thumb's whole direction; the four finger CURLS are never the
# camera's, so they are not listed and cannot be gated.
SPREAD_FINGERS = ("index", "middle", "ring", "pinky")
CAMERA_DOFS = tuple(f"spread {f}" for f in SPREAD_FINGERS) + ("thumb",)

# The curls the rail-disagreement override can take from the camera. The thumb
# is not among them: its whole DIRECTION is already the camera's when the thumb
# gate passes, so there is no separate curl left for an override to win.
RAIL_FINGERS = ("index", "middle", "ring", "pinky")
RAIL_DOFS = tuple(f"curl {f}" for f in RAIL_FINGERS)

# Every DOF the report accounts for: owned by the camera by design, or takeable
# from it by the override.
GATED_DOFS = CAMERA_DOFS + RAIL_DOFS

# dof_source values.
SRC_GLOVE = "glove"
SRC_CAMERA = "camera"
SRC_RAIL = "camera (rail override)"


@dataclass(frozen=True)
class GateParams:
    """Thresholds for the per-frame, per-DOF camera gates.

    EMPIRICAL STARTING POINTS, all read off the first real simultaneous
    session (6 poses, 1 take, both hands) — not calibrated constants. They are
    named, passed through, and printed in every report so the next session can
    argue with them.

    curl_gate            normalised curl (tip-to-wrist over palm length) above
                         which the glove says a finger is extended enough for
                         its abduction to be worth reading. 1.2 is the midpoint
                         between the glove's fist values (0.63-0.8) and its
                         open-palm values (1.71-2.07).
    view_gate_deg        angle between the camera's palm normal and the ray
                         from the palm to the module. Small = the palm faces
                         the sensor. Open palm and pinch measure 25-45 deg in
                         that session; the edge-on thumbs_up hand measures
                         70-78, so 50 separates them.
    curl_agree_tol       median absolute curl disagreement across index..little
                         below which the camera may also supply the thumb.
                         pinch measures 0.24-0.28, thumbs_up 0.88.
    min_visible_time_us  how long LeapC must have held this hand. 300 ms is
                         about 27 frames at 90 Hz: past the re-acquisition
                         transient, still well inside a 5 s take.
    hand_id_settle_s     after a hand_id change the tracker has re-identified
                         the hand and its handedness may still flip; ignore the
                         camera for this long.
    field_half_angle_deg the palm must sit within this angle of the module's
                         vertical axis — lateral offset less than height.
    """
    curl_gate: float = 1.2
    view_gate_deg: float = 50.0
    curl_agree_tol: float = 0.35
    min_visible_time_us: int = 300_000
    hand_id_settle_s: float = 0.25
    field_half_angle_deg: float = 45.0

    def described(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_GATES = GateParams()

# Rejection reasons. Fixed strings so the report can count them.
R_NO_FRAME = "no camera frame"
R_SCORE = "camera score below min_score"
R_VISIBLE = "hand not visible long enough"
R_HAND_ID = "hand id changed recently"
R_FIELD = "palm outside the module's central field"
R_NO_GEOMETRY = "no palm geometry to measure the view from"
R_VIEW = "palm turned away from the camera"
R_CURLED = "glove says the finger is curled"
R_DISAGREE = "camera disagrees with the glove about the fingers"

# Rail-override reasons: why a finger's curl stayed the glove's.
R_RAIL_NONE = "no rail learned for this finger"
R_RAIL_OFF = "glove is off its rail, so it is measuring"
R_RAIL_EXTENDED = "camera does not see the finger flexed"
R_RAIL_ARMING = "rail disagreement not sustained yet"


@dataclass(frozen=True)
class RailOverrideParams:
    """Thresholds for the rail-disagreement override. See the module docstring.

    tol              how close the glove's curl must be to the learned rail to
                     count as ON it. The glove's open-palm value is bit-exact
                     to three decimals, so 0.005 is a float-equality test with
                     room for the last digit's jitter, not a tolerance band.
    margin           how far BELOW its open reference the camera's curl must
                     sit before it counts as seeing real flexion. 0.25 puts
                     the index threshold at 1.50: clear of every genuinely
                     straight index in sync_day1 (lowest 1.74) and clear above
                     every pinch take's median (highest 1.39).
    enter_frames     consecutive qualifying frames before the override arms.
                     10 at the glove's 60 Hz is about 170 ms — long enough
                     that a single mistracked frame cannot straighten or bend
                     a finger, short enough to be inside a 5 s take many times
                     over.
    exit_frames      consecutive non-qualifying frames before it disarms.
                     Deliberately shorter than enter_frames: the override is
                     the exception, so it should be easier to leave than to
                     enter.
    fingers          which fingers it is enabled on. Default ("index",): the
                     one finger sync_day1 actually shows the failure on.
    cam_open_curl    the camera's curl for a STRAIGHT finger, per finger in
                     FINGER_NAMES order. A constant on purpose — see the
                     module docstring. Read off sync_day1's open-palm frames,
                     taking the lower of the two hands so the threshold errs
                     toward not firing.
    min_rail_share   a learned rail must account for at least this share of a
                     finger's frames, so a value that merely happens to be
                     most common is not mistaken for a saturation spike.
    rail_max_gap     a learned rail must also sit within this of the LARGEST
                     curl seen for that finger. A rail is the top of the
                     sensor's range; a mode well below the maximum is a pose
                     that was simply held a lot, and teaches no rail.
    """
    tol: float = 0.005
    margin: float = 0.25
    enter_frames: int = 10
    exit_frames: int = 5
    fingers: Tuple[str, ...] = ("index",)
    cam_open_curl: Tuple[float, ...] = (1.30, 1.75, 1.82, 1.69, 1.44)
    min_rail_share: float = 0.05
    rail_max_gap: float = 0.02

    def open_curl(self, finger: str) -> float:
        return self.cam_open_curl[FINGER_NAMES.index(finger)]

    def described(self) -> Dict[str, str]:
        d = asdict(self)
        d["fingers"] = ", ".join(self.fingers) if self.fingers else "(none)"
        d["cam_open_curl"] = "  ".join(
            f"{n} {v:.2f}" for n, v in zip(FINGER_NAMES, self.cam_open_curl))
        return d


DEFAULT_RAIL = RailOverrideParams()


def learn_rails(samples: Iterable[Tuple[str, Sequence[float]]],
                params: Optional[RailOverrideParams] = None
                ) -> Dict[Tuple[str, str], float]:
    """Find each (hand, finger)'s rail from the frames about to be fused.

    `samples` is (hand_side, curls) per frame, `curls` in FINGER_NAMES order —
    i.e. `features.flexion_features` of the GLOVE skeleton.

    A saturating sensor writes the same bits every time it is against the stop,
    so the rail is a spike in the histogram: on sync_day1 the mode accounts for
    27-69% of a finger's frames while every other value is under 13%. Moving
    fingers spread out and cannot compete with that.

    Two sanity conditions, both of which exist to make a session with no open
    hand in it teach nothing rather than teach nonsense:

      share   the mode must cover `min_rail_share` of the finger's frames
      top     it must sit within `rail_max_gap` of the finger's LARGEST curl,
              because a rail is by definition the straightest reading there is

    A finger failing either gets no entry, and the override can never arm for
    it. The value returned is the median of the frames within `tol` of the
    mode, not the rounded mode itself, so the third decimal is not an artefact
    of the histogram's bin.
    """
    params = params or DEFAULT_RAIL
    hist: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    seen: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for hand, curls in samples:
        for finger, c in zip(FINGER_NAMES, curls):
            c = float(c)
            hist[(hand, finger)][round(c, 3)] += 1
            seen[(hand, finger)].append(c)

    rails: Dict[Tuple[str, str], float] = {}
    for key, counter in hist.items():
        candidate = counter.most_common(1)[0][0]
        values = seen[key]
        near = [v for v in values if abs(v - candidate) <= params.tol]
        if len(near) < params.min_rail_share * len(values):
            continue
        if max(values) - candidate > params.rail_max_gap:
            continue
        rails[key] = median(near)
    return rails


@dataclass(frozen=True)
class RailDecision:
    """One frame's verdict: which fingers the override owns, and why not.

    Produced by `RailOverrideTracker`, consumed by `fuse_skeletons`. Keeping it
    a value means the hysteresis — the only state in this module — lives in one
    object the caller holds, and `fuse_skeletons` stays a pure function of its
    arguments.
    """
    active: Tuple[str, ...] = ()
    rejected: Mapping[str, str] = field(default_factory=dict)


NO_RAIL_OVERRIDE = RailDecision()


class RailOverrideTracker:
    """Per-hand, per-finger hysteresis over the rail-disagreement test.

    One tracker per fusion run, NOT per take: it holds the learned rails and
    the run-length counters. `update` is called once per glove frame, in time
    order, and returns that frame's `RailDecision`.

    Counters are keyed by (hand, finger) because the two hands interleave in
    one file and are separate pieces of evidence.
    """

    def __init__(self, rails: Optional[Mapping[Tuple[str, str], float]] = None,
                 params: Optional[RailOverrideParams] = None,
                 gates: Optional[GateParams] = None):
        self.params = params or DEFAULT_RAIL
        self.gates = gates or DEFAULT_GATES
        self.rails = dict(rails or {})
        self._run: Dict[Tuple[str, str], int] = defaultdict(int)
        self._idle: Dict[Tuple[str, str], int] = defaultdict(int)
        self._active: set = set()

    def qualifies(self, hand: str, finger: str, glove_curl: float,
                  cam_curl: Optional[float],
                  frame_reasons: Sequence[str] = (),
                  view_deg: Optional[float] = None) -> Optional[str]:
        """None if this FRAME qualifies for the override, else why it does not.

        `frame_reasons` is `frame_trust`'s verdict, empty when the frame passed.
        Its first entry is reported verbatim rather than re-derived, so the
        override's rejections land in the same buckets the spread's and thumb's
        do and the report's table stays one table.

        Order matters only for which reason gets reported, and it is chosen so
        the report names the state of the HAND before the state of the capture:
        a fist rejects because the glove is off its rail (it is measuring, and
        should be believed), not because of anything about the camera.
        """
        rail = self.rails.get((hand, finger))
        if rail is None:
            return R_RAIL_NONE
        if abs(float(glove_curl) - rail) > self.params.tol:
            return R_RAIL_OFF
        if cam_curl is None:
            return R_NO_FRAME
        if frame_reasons:
            return frame_reasons[0]
        if view_deg is None:
            return R_NO_GEOMETRY
        if view_deg >= self.gates.view_gate_deg:
            return R_VIEW
        if float(cam_curl) >= self.params.open_curl(finger) - self.params.margin:
            return R_RAIL_EXTENDED
        return None

    def update(self, hand: str, glove_curls: Sequence[float],
               cam_curls: Optional[Sequence[float]] = None,
               cam_meta: Optional[dict] = None) -> RailDecision:
        """Advance the hysteresis one frame and return this frame's decision.

        `glove_curls` and `cam_curls` are `features.flexion_features` of the
        two skeletons; `cam_meta` is what `frame_trust` reads. A frame with no
        camera (either argument None) is a non-qualifying frame — it decays the
        run, it does not reset the whole state — which is what keeps a dropped
        camera frame mid-pinch from flickering the override off and on.
        """
        frame_reasons: Sequence[str] = (R_NO_FRAME,)
        view_deg = None
        if cam_meta is not None:
            _ok, frame_reasons, metrics = frame_trust(cam_meta, self.gates)
            view_deg = metrics["view_angle_deg"]

        active: List[str] = []
        rejected: Dict[str, str] = {}
        for finger in self.params.fingers:
            if finger not in RAIL_FINGERS:
                continue
            key = (hand, finger)
            i = FINGER_NAMES.index(finger)
            why = self.qualifies(
                hand, finger, glove_curls[i],
                None if cam_curls is None else cam_curls[i],
                frame_reasons, view_deg)
            if why is None:
                self._run[key] += 1
                self._idle[key] = 0
                if self._run[key] >= self.params.enter_frames:
                    self._active.add(key)
            else:
                self._idle[key] += 1
                self._run[key] = 0
                if self._idle[key] >= self.params.exit_frames:
                    self._active.discard(key)
            if key in self._active:
                active.append(finger)
            else:
                rejected[f"curl {finger}"] = why if why is not None else R_RAIL_ARMING
        return RailDecision(tuple(active), rejected)


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


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-handed rotation of `angle` radians about a unit `axis`."""
    a = _unit(np.asarray(axis, float))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def proximal_direction(pts: np.ndarray, finger: str) -> np.ndarray:
    """Unit direction of a finger's proximal bone (knuckle -> next joint).

    Abduction happens at the knuckle, so this bone carries the azimuth and
    nothing else. Knuckle -> TIP would carry the azimuth plus whatever the PIP
    and DIP are doing, which is the curl the glove already measures better.
    """
    a, b = PROXIMAL_BONE[finger]
    return _unit(np.asarray(pts, float)[b] - np.asarray(pts, float)[a])


def azimuth_in_frame(d: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """In-plane angle of `d` in the palm frame, radians. 0 = down the palm."""
    return math.atan2(float(np.dot(d, y)), float(np.dot(d, x)))


def transfer_finger_flexion(glove: np.ndarray, cam_aligned: np.ndarray,
                            finger: str) -> Optional[np.ndarray]:
    """Rebuild one finger: the camera's bone DIRECTIONS on the glove's LENGTHS.

    `cam_aligned` is the camera hand already rotated into the glove's palm
    frame (`camera_into_glove_frame`), so a direction read off it is directly
    comparable with the glove's. The chain is walked from the knuckle outwards:
    the knuckle itself never moves — it belongs to the palm, which the camera
    is not being asked about — and each following joint is placed one GLOVE
    bone length along the corresponding CAMERA bone's direction.

    That is the spread transfer taken one step further. There a single rotation
    about the knuckle gave the whole chain the camera's azimuth and kept the
    glove's curl; here each of the three bones takes the camera's direction
    outright, so the MCP, PIP and DIP angles all become the camera's. Both are
    built out of unit directions and glove lengths, so both preserve every bone
    length exactly — the invariant the rest of this module rests on.

    Returns 4 points (knuckle, PIP, DIP, tip) for `FINGER_CHAINS[finger]`, or
    None if a camera bone has no length to take a direction from.
    """
    chain = FINGER_CHAINS[finger]
    G = np.asarray(glove, float)
    C = np.asarray(cam_aligned, float)
    out = [G[chain[0]]]
    for k in range(len(chain) - 1):
        d = C[chain[k + 1]] - C[chain[k]]
        if float(np.linalg.norm(d)) < 1e-12:
            return None
        length = float(np.linalg.norm(G[chain[k + 1]] - G[chain[k]]))
        out.append(out[-1] + length * _unit(d))
    return np.asarray(out)


# --- frame-level trust, measured on the camera's own geometry ----------

def viewing_angle_deg(palm_abs: Sequence[float],
                      palm_normal_abs: Sequence[float]) -> float:
    """Angle between the palm normal and the ray palm -> module, degrees.

    The Leap module is the ORIGIN of leap space, so the ray from the palm
    centre to the camera is simply -palm. Zero means the palm is square to the
    sensor; 90 means the hand is edge-on and the tracker is inferring the
    fingers it cannot see rather than measuring them.

    Both hands are handled the same way because `palm_normal_abs` is expected
    to come from `features.palm_normal`, which already flips the normal for a
    left hand so that "out of the palm" means the same thing on both.
    """
    p = np.asarray(palm_abs, float)
    n = _unit(np.asarray(palm_normal_abs, float))
    ray = _unit(-p)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(n, ray))))))


def palm_field_angle_deg(palm_abs: Sequence[float]) -> float:
    """Angle of the palm off the module's vertical axis, degrees.

    Leap desktop space puts +y up out of the module, so this is
    atan2(lateral offset, height): below 45 degrees the palm is nearer the
    axis than it is high, which is the central part of the field where both
    cameras see the whole hand. A palm at or below the module's own height
    comes out >= 90 and is rejected.
    """
    p = np.asarray(palm_abs, float)
    return math.degrees(math.atan2(float(math.hypot(p[0], p[2])), float(p[1])))


def frame_trust(cam_meta: dict,
                gates: Optional[GateParams] = None) -> Tuple[bool, List[str], dict]:
    """Do this camera frame's own numbers say it is worth reading?

    Returns (ok, reasons, metrics). None of the three checks uses a model
    output: how long the hand has been held, whether it was just re-acquired,
    and where it sits in the field are all facts about the capture.
    """
    gates = gates or DEFAULT_GATES
    reasons: List[str] = []

    visible = cam_meta.get("visible_time_us")
    if visible is None or float(visible) < gates.min_visible_time_us:
        reasons.append(R_VISIBLE)
    if cam_meta.get("hand_id_stable") is False:
        reasons.append(R_HAND_ID)

    palm = cam_meta.get("palm_abs")
    normal = cam_meta.get("palm_normal_abs")
    metrics = {"view_angle_deg": None, "field_angle_deg": None,
               "visible_time_us": visible}
    if palm is None:
        reasons.append(R_NO_GEOMETRY)
    else:
        metrics["field_angle_deg"] = palm_field_angle_deg(palm)
        if metrics["field_angle_deg"] >= gates.field_half_angle_deg:
            reasons.append(R_FIELD)
        if normal is None:
            reasons.append(R_NO_GEOMETRY)
        else:
            metrics["view_angle_deg"] = viewing_angle_deg(palm, normal)
    return (not reasons), reasons, metrics


def flag_hand_id_stability(rows: Sequence[dict], gates: Optional[GateParams] = None,
                           clock: str = "capture_time") -> None:
    """Set `hand_id_stable` on each camera row, in place.

    A `hand_id` change means LeapC dropped the hand and picked it up again.
    For a short while afterwards the new track is still settling — handedness
    can flip, and the joints it reports are a fresh fit rather than a tracked
    one. Each row is stable only if no change happened within
    `hand_id_settle_s` before it. Rows are grouped by hand and ordered in time
    first, because two hands interleave in one file.
    """
    gates = gates or DEFAULT_GATES
    by_hand: Dict[str, List[dict]] = {}
    for r in rows:
        by_hand.setdefault(r.get("hand_side"), []).append(r)
    for group in by_hand.values():
        group = sorted(group, key=lambda d: _stamp(d, clock)
                       if d.get(clock) is not None else d[WALL_CLOCK])
        last_id, changed_at = None, None
        for r in group:
            t = _stamp(r, clock) if r.get(clock) is not None else r[WALL_CLOCK]
            hid = r.get("hand_id")
            if last_id is not None and hid != last_id:
                changed_at = t
            last_id = hid
            r["hand_id_stable"] = (changed_at is None
                                   or (t - changed_at) >= gates.hand_id_settle_s)


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


def _blank_info(with_scale: bool) -> dict:
    return {"camera_used": False, "reason": "", "fingers_adjusted": [],
            "alignment": "similarity" if with_scale else "palm_frame",
            "kabsch_rmse_mm": None,
            # bookkeeping: which DOFs the camera supplied, and why not
            "gated": False,
            "dof_source": {dof: SRC_GLOVE for dof in GATED_DOFS},
            "rejected": {},
            "frame_reasons": [],
            "rail_override": [],
            "view_angle_deg": None, "field_angle_deg": None,
            "curl_disagreement": None}


def _reject_all(info: dict, reason: str, rail_dofs: Sequence[str] = ()) -> dict:
    """Blame `reason` for every DOF the camera could have supplied.

    `rail_dofs` is passed separately because a curl DOF is only the camera's to
    lose when the override is enabled on that finger: a finger nobody asked for
    has not been rejected, it was never in the running, and counting it would
    bury the real reasons under one disabled-by-default line per frame.
    """
    for dof in tuple(CAMERA_DOFS) + tuple(rail_dofs):
        info["rejected"][dof] = reason
    return info


def fuse_skeletons(
    glove_pts: Sequence[Sequence[float]],
    cam_pts: Optional[Sequence[Sequence[float]]],
    cam_score: float = 1.0,
    min_score: float = 0.5,
    thumb_from_camera: bool = True,
    fingers: Optional[Sequence[str]] = None,
    with_scale: bool = True,
    cam_meta: Optional[dict] = None,
    gates: Optional[GateParams] = None,
    rail: Optional[RailDecision] = None,
) -> Tuple[np.ndarray, dict]:
    """Fuse one glove frame with one camera frame -> 21 points + info.

    Both inputs are 21 x 3 wrist-centred metres. `with_scale` picks the
    alignment: True for a normalised (MediaPipe) camera, False for a metric
    one (Ultraleap).

    `cam_meta` is what turns the gates on, and its absence is not a failure:
    it carries the camera's own capture facts for this frame —
    `visible_time_us`, `hand_id_stable`, `palm_abs`, `palm_normal_abs` — which
    only a metric camera reporting absolute joints has. A MediaPipe frame has
    none of them, passes `cam_meta=None`, and is fused exactly as before, with
    every camera-owned DOF taken. Gating a sensor on evidence it does not
    produce would mean rejecting all of it.

    `rail` is this frame's rail-disagreement verdict, from
    `RailOverrideTracker.update`. It is the ONLY way a finger's curl can come
    from the camera, and leaving it None (the default) reproduces the fusion
    exactly as it was before the override existed. The hysteresis behind it
    needs memory of earlier frames, which is why the decision is made outside
    and handed in: this function stays a pure function of its arguments.

    Returns the fused points and an info dict: `dof_source` says where each
    camera-owned DOF actually came from, `rejected` says why the glove kept
    the ones it kept, `rail_override` lists the fingers the override owned,
    and `frame_reasons` lists any frame-level failure. A frame that fails
    everything returns the glove skeleton unchanged — it is never dropped.
    """
    G = np.asarray(glove_pts, dtype=float)
    gates = gates or DEFAULT_GATES
    rail = rail if rail is not None else NO_RAIL_OVERRIDE
    info = _blank_info(with_scale)
    # Only fingers the override was actually asked about can be "rejected" by
    # it: the ones it named, plus any it is currently winning.
    rail_dofs = tuple(dict.fromkeys(tuple(rail.rejected)
                                    + tuple(f"curl {f}" for f in rail.active)))

    if cam_pts is None:
        info["reason"] = R_NO_FRAME
        _reject_all(info, R_NO_FRAME, rail_dofs)
        return G, info
    if cam_score < min_score:
        info["reason"] = f"camera score {cam_score:.2f} < {min_score:.2f}"
        _reject_all(info, R_SCORE, rail_dofs)
        return G, info

    C_in = np.asarray(cam_pts, dtype=float)
    if not with_scale:
        # Diagnostic, not a gate: how far the two palms are from being one
        # rigid object. Only meaningful when both sides are metric.
        info["kabsch_rmse_mm"] = palm_fit_rmse_mm(G, C_in)

    # --- what this frame is allowed to contribute ----------------------
    view_deg = None
    if cam_meta is not None:
        info["gated"] = True
        ok, frame_reasons, metrics = frame_trust(cam_meta, gates)
        info["frame_reasons"] = frame_reasons
        info["view_angle_deg"] = metrics["view_angle_deg"]
        info["field_angle_deg"] = metrics["field_angle_deg"]
        view_deg = metrics["view_angle_deg"]
        if not ok:
            info["reason"] = "; ".join(frame_reasons)
            _reject_all(info, frame_reasons[0], rail_dofs)
            return G, info              # glove skeleton, unchanged

    curl_g = flexion_features(G)
    curl_c = flexion_features(C_in)      # a ratio, so alignment cannot change it
    # index..little only: the thumb is the DOF being decided, so it cannot vote
    disagreement = median([abs(a - b) for a, b in
                           zip(curl_g[1:], curl_c[1:])])
    info["curl_disagreement"] = disagreement

    def spread_verdict(finger: str) -> Optional[str]:
        """None if the camera may set this finger's azimuth, else the reason."""
        if cam_meta is None:
            return None
        if curl_g[FINGER_NAMES.index(finger)] <= gates.curl_gate:
            return R_CURLED
        if view_deg is None:
            return R_NO_GEOMETRY
        if view_deg >= gates.view_gate_deg:
            return R_VIEW
        return None

    def thumb_verdict() -> Optional[str]:
        if cam_meta is None:
            return None
        if view_deg is None:
            return R_NO_GEOMETRY
        if view_deg >= gates.view_gate_deg:
            return R_VIEW
        if disagreement >= gates.curl_agree_tol:
            return R_DISAGREE
        return None

    C = camera_into_glove_frame(G, C_in, with_scale=with_scale)
    n, x, y = palm_frame(G)
    fused = G.copy()
    names = list(FINGER_CHAINS) if fingers is None else list(fingers)

    # Why the override kept its hands off the fingers it did. The ones it WON
    # are recorded below, after the transfer has actually succeeded.
    for dof, why in rail.rejected.items():
        info["rejected"][dof] = why

    for finger in names:
        chain = FINGER_CHAINS[finger]
        knuckle = G[chain[0]]
        is_thumb = finger == "thumb" and thumb_from_camera
        dof = "thumb" if is_thumb else f"spread {finger}"

        if finger in rail.active:
            # The glove is on this finger's rail and a trusted camera has seen
            # it flexed for long enough. Rebuild the whole chain from the
            # camera's bone directions: that carries the azimuth too, so the
            # spread rotation below MUST NOT also run on this finger or the
            # same correction would be applied twice.
            rebuilt = transfer_finger_flexion(G, C, finger)
            if rebuilt is not None:
                fused[chain] = rebuilt
                info["fingers_adjusted"].append(finger)
                info["rail_override"].append(finger)
                info["dof_source"][f"curl {finger}"] = SRC_RAIL
                # The proximal bone's direction came from the camera as well,
                # so the spread is the camera's — by the ordinary route, which
                # is what the spread row should keep reporting.
                if dof in info["dof_source"]:
                    info["dof_source"][dof] = SRC_CAMERA
                continue
            info["rejected"][f"curl {finger}"] = R_NO_GEOMETRY

        why = thumb_verdict() if is_thumb else spread_verdict(finger)
        if why is not None:
            if dof in info["dof_source"]:
                info["rejected"][dof] = why
            continue

        if is_thumb:
            # Opposition is out-of-plane rotation at the base: take the
            # camera's whole direction, keep the glove's bone lengths.
            d_g = _unit(G[chain[-1]] - knuckle)
            d_c = _unit(C[chain[-1]] - C[chain[0]])
            if np.linalg.norm(d_g) < 1e-9 or np.linalg.norm(d_c) < 1e-9:
                continue
            R = rotation_between(d_g, d_c)
        else:
            # Adopt the camera's azimuth, keep the glove's curl. Measured on
            # the PROXIMAL bone on BOTH sides: a curled tip swings far off its
            # own bone and would import flexion error as fake abduction.
            d_g = proximal_direction(G, finger)
            d_c = proximal_direction(C, finger)
            if np.linalg.norm(d_g) < 1e-9 or np.linalg.norm(d_c) < 1e-9:
                continue
            delta = azimuth_in_frame(d_c, x, y) - azimuth_in_frame(d_g, x, y)
            # A rotation about the palm normal changes only the azimuth, so
            # every bone's angle out of the palm plane — the glove's curl —
            # survives exactly, not approximately.
            R = rotation_about(n, delta)

        fused[chain] = (R @ (G[chain] - knuckle).T).T + knuckle
        info["fingers_adjusted"].append(finger)
        if dof in info["dof_source"]:
            info["dof_source"][dof] = SRC_CAMERA

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
