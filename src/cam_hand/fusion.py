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
                agrees with the glove about the other fingers, measured as a
                FLEXION FRACTION on each sensor's own endpoints — see "THE
                TWO SENSORS DO NOT SHARE A SCALE" below. A camera that
                has the fingers wrong has the hand's orientation wrong, and
                the thumb is the DOF that orientation error moves the most.
                This is what rejects thumbs_up and admits pinch.

                "the other fingers" is not all four. A finger only votes if
                its glove curl is a MEASUREMENT: not pinned on its rail WHILE
                the camera reads it flexed, not already overridden by the
                rail-disagreement rule, and not named in
                `unreliable_fingers`. A railed finger the camera also calls
                extended votes normally — the two sensors agree about it, and
                that is evidence. Below `min_usable_fingers` survivors the
                glove casts no veto and the camera's own geometry decides —
                see `fuse_skeletons`.

`grab_strength`, `pinch_strength` and `confidence` are deliberately NOT used.
The first two are model outputs — the same model that produced the joints, so
they cannot corroborate it — and LeapC reports `confidence` as a constant 1.0.

Every threshold below is a named parameter on `GateParams` with an empirical
starting value read off that one session, not a calibrated constant, and every
one of them is printed in the report so a later session can move it.

No frame is ever dropped. A frame that fails every gate is the glove skeleton,
unchanged, which is exactly what the pipeline produced before the camera
existed.

THE TWO SENSORS DO NOT SHARE A SCALE, SO THE THUMB VOTE IS NORMALISED
---------------------------------------------------------------------
The thumb gate asks whether the two sensors agree about index..little, and
the first version of it thresholded the median |glove curl - camera curl| at
a raw 0.35. Curl is tip-to-wrist over palm length, which is dimensionless —
but the two sensors do not put a hand on the same part of that axis. The
glove's straight index reads 1.97 on the TEMPLATE hand and its fist reads
0.63; the camera's straight index reads 1.75 on the operator's and its fist
0.55. A fixed distance between those two numbers means a different amount of
DISAGREEMENT depending on which glove hand the value came off, and
`template_fit` changed exactly that: rescaling the template's bones moved
every glove curl by a few per cent, the raw gap shrank with them, and the
thumb's camera-use jumped from 77 % to 94 % on sync_day1. Some of that was
the fit measuring a real hand; some of it was a threshold quietly getting
looser. A number that moves when the units move cannot tell the two apart.

So each finger's curl is first put on ITS OWN SENSOR's, HAND's and FINGER's
endpoints, as a flexion fraction:

    frac = (open - curl) / (open - flexed)          clipped to [-0.2, 1.2]

0 is that sensor's straight finger, 1 is that sensor's most flexed, and the
median |frac_glove - frac_camera| is what `agree_tol_frac` thresholds. Both
ends are LEARNED from the session, per hand, per finger, per sensor, with no
pose labels anywhere:

  glove open    the finger's learned rail (`learn_rails`) — the glove's own
                bit-exact full-extension constant, which is already measured
                for the override and is by construction on the hand being
                fused, template or fitted.
  camera open   the median camera curl over the session's open-palm-LIKE
                frames — every finger reading above `cam_open_curl - margin`,
                the same label-free test `template_fit` measures a hand on —
                falling back to the 98th percentile of that finger's camera
                curl when the session holds too few of them.
  flexed        the 2nd percentile of that finger's curl over the session, on
                each sensor separately. A percentile and not the minimum, for
                the same reason `diagnostics.camera_range` uses one: a single
                mistracked frame is not the bottom of a sensor's range.

A finger whose range on either sensor is too small to normalise — below
`min_glove_span` or `min_cam_span` — is not put on a fraction at all and is
DROPPED from the vote for that run, and the report says which and why. A
session in which a finger never straightened teaches no endpoints, and
dividing by that would turn measurement noise into a full-scale
disagreement.

What this buys is comparability: the camera's endpoints are identical in a
fitted and an unfitted run (fitting does not touch the camera) and the
glove's move by exactly the factor its curls moved by, so the fraction is
very nearly invariant and the two runs' camera-use can be compared. The raw
tolerance survives only as the fallback for a caller with no learned
endpoints to hand (`curl_agree_tol`), and `scripts/fuse_poses.py` always has
them.

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

It may also be written PER HAND (`{"right": ("index", "ring"), "left":
("index",)}`), because the evidence for enabling a finger is a sweep of that
finger on that hand and the two gloves do not fail alike: on sync_day1 it is
the RIGHT glove's ring and pinky that read partly extended through poses the
left glove reports correctly. A plain sequence still means both hands, so
nothing written before the mapping existed changes. `fingers_for(hand)` is the
only reader.

WHEN THE TWO SENSORS ARE DESCRIBING DIFFERENT INSTANTS
------------------------------------------------------
Everything above assumes a paired glove frame and camera frame are two views
of one hand at one moment. `pair_by_time` matches them on the capture clock,
which is the best clock either sensor offers — and it is still not the instant
the hand was in that shape on the glove side: the glove's solved hand TRAILS
the camera, measured on this laptop at roughly 100 ms on the left hand and
450-485 ms on the right (cross-correlated index curl over close/open
transitions).

For a HELD pose it makes no difference, and that is worth stating as a
measurement rather than an argument: on day 1's 59 static takes, applying the
right hand's 460 ms moves no per-take median by more than 0.01. For a MOVING
hand it pairs a glove frame with a camera frame from a different instant, and
every gate above then compares two different hands.

So `pair_by_time` takes `glove_lag` seconds and shifts the glove's PAIRING
stamp back by it — a glove frame stamped t describes the hand at t - lag.
Nothing is resampled and no recorded stamp is rewritten; the shift lives
inside the match. `estimate_glove_lag` measures it per hand with
`leap_hand.diagnostics.estimate_lag`, and refuses to answer at all on a clip
whose camera index curl barely moved, because a cross-correlation of two flat
lines has a maximum and it means nothing.

WHEN THE GLOVE'S CURL IS WRONG IN A WAY THE CAMERA CAN SEE
----------------------------------------------------------
Everything above takes the glove's curl as it comes. Two corrections use the
camera to fix it, both off by default, both changing only what is FUSED: the
gates go on deciding from the glove's uncorrected curls (`fuse_skeletons`'s
`gate_curls`), so a finger that was corrected is not thereby refused the
camera's spread. Both bend a finger with `bend_finger_to_curl`, which shares
the bend between the three joints and keeps every bone its length.

`DriftAnchor` (EXPERIMENTAL) was built for creep: a held glove finger drifts
by 0.05 to 0.26 curl per minute. On frames the camera can be trusted it
records the residual between the two sensors' flexion FRACTIONS and applies
its recent median to the glove on every frame. Measured on sync_day1 and
sync_day2, though, the glove's error over these takes is dominated by POSE:
per-pose medians of that residual span up to a whole flexion range while it
moves by a few hundredths within a take. An offset learned in one pose is
wrong in the next, so the anchor is reset at every take boundary and must be
reset whenever the hand is lost.

`GloveRecalibration` models that pose dependence directly: per hand, each
finger's camera fraction as a ridge regression on the glove's fractions,
either the finger's own (`own`) or all five (`cross`, which can learn that a
glove channel also answers to its neighbour). It is fitted on trusted
camera frames, and evaluated so that no take is corrected by a model that
saw it.
"""
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from typing import (Deque, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

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
                         open-palm values (1.71-2.07) — ON THE TEMPLATE HAND.
                         Curl is a LENGTH ratio, so rescaling the template's
                         bones to a real hand (`template_fit`) moves every one
                         of those values, and a constant tuned on the template
                         then sits in a different place between them. It is
                         carried across a fit by `curl_gates_from_rails`,
                         which re-expresses it as the same fraction of each
                         hand and finger's own learned rail; with no fit the
                         rails are the template's and the number is exactly
                         this one. See `fuse_skeletons`'s `curl_gates`.
    view_gate_deg        angle between the camera's palm normal and the ray
                         from the palm to the module. Small = the palm faces
                         the sensor. Open palm and pinch measure 25-45 deg in
                         that session; the edge-on thumbs_up hand measures
                         70-78, so 50 separates them.
    agree_tol_frac       median absolute FLEXION-FRACTION disagreement across
                         index..little below which the camera may also supply
                         the thumb. This is the gate that actually runs
                         whenever learned endpoints are available; see "THE
                         TWO SENSORS DO NOT SHARE A SCALE" in the module
                         docstring for why a raw curl difference could not be
                         compared across a template fit. 0.20 is the value
                         that reproduces the raw gate's verdict on the
                         UNFITTED sync_day1 — thumb camera-use 77.4 % raw,
                         77.3 % normalised — chosen so that the change of
                         units is not also a change of strictness. On the
                         FITTED sync_day1 the same tolerance gives 77.5 %,
                         against the raw gate's 93.9 %: that 16-point jump
                         was the gate getting looser, not the camera getting
                         better.
    min_glove_span       a finger whose glove curl range (rail minus its 2nd
                         percentile) is below this cannot be put on a
                         fraction and is dropped from the thumb vote. 0.4 on
                         the glove's template scale: sync_day1's smallest
                         real span is the pinky's 1.71 - 0.66 = 1.05, and a
                         finger that never left its rail measures ~0.
    min_cam_span         the same for the camera, 0.3 — lower because the
                         camera's straight finger reads lower than the
                         template's and its fist about the same, so every
                         camera span is the smaller of the two.
    curl_agree_tol       the RAW median absolute curl disagreement, used only
                         when no learned endpoints were passed. Kept because
                         a caller with one frame and no session behind it has
                         nothing to normalise with; pinch measures 0.24-0.28
                         on it, thumbs_up 0.88.
    min_visible_time_us  how long LeapC must have held this hand. 300 ms is
                         about 27 frames at 90 Hz: past the re-acquisition
                         transient, still well inside a 5 s take.
    hand_id_settle_s     after a hand_id change the tracker has re-identified
                         the hand and its handedness may still flip; ignore the
                         camera for this long.
    field_half_angle_deg the palm must sit within this angle of the module's
                         vertical axis — lateral offset less than height.
    min_usable_fingers   how many fingers must still be worth comparing before
                         the glove is allowed to veto the camera's thumb. See
                         `fuse_skeletons`: a finger on its rail, a finger the
                         rail override has taken, and a finger the operator has
                         declared unreliable are all excluded from the vote,
                         and with fewer than this many left the glove has no
                         opinion worth acting on and does not cast one.
    """
    curl_gate: float = 1.2
    view_gate_deg: float = 50.0
    agree_tol_frac: float = 0.20
    min_glove_span: float = 0.4
    min_cam_span: float = 0.3
    curl_agree_tol: float = 0.35
    min_visible_time_us: int = 300_000
    hand_id_settle_s: float = 0.25
    field_half_angle_deg: float = 45.0
    min_usable_fingers: int = 2

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

# Why a finger could not be put on a flexion fraction, and so does not vote.
N_NO_RAIL = "no glove rail learned, so there is no open endpoint"
N_GLOVE_SPAN = "the glove's curl range is too small to normalise"
N_CAM_SPAN = "the camera's curl range is too small to normalise"
N_NO_FRAMES = "no {sensor} frames of this hand to learn endpoints from"
R_NOT_NORMALISED = "no finger could be put on a flexion fraction"

# Rail-override reasons: why a finger's curl stayed the glove's.
R_RAIL_NONE = "no rail learned for this finger"
R_RAIL_OFF = "glove is off its rail, so it is measuring"
R_RAIL_EXTENDED = "camera does not see the finger flexed"
R_RAIL_ARMING = "rail disagreement not sustained yet"

# Which fingers the rail override is enabled on: one list for both hands, or
# one list per hand. See `RailOverrideParams.fingers`.
FingerSpec = Union[Sequence[str], Mapping[str, Sequence[str]]]


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

                     PER HAND, because a glove fails per hand and the
                     evidence for enabling a finger is a sweep of THAT
                     finger on THAT hand. Two forms are accepted:

                       ("index",)                    both hands
                       {"right": ("index", "ring"),
                        "left":  ("index",)}         named per hand

                     A plain sequence keeps meaning "both hands", so every
                     caller and recording made before the mapping existed
                     behaves exactly as it did. Read it through
                     `fingers_for(hand)`, never directly: that is the one
                     place the two forms are told apart.
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
    fingers: FingerSpec = ("index",)
    cam_open_curl: Tuple[float, ...] = (1.30, 1.75, 1.82, 1.69, 1.44)
    min_rail_share: float = 0.05
    rail_max_gap: float = 0.02

    def open_curl(self, finger: str) -> float:
        return self.cam_open_curl[FINGER_NAMES.index(finger)]

    def fingers_for(self, hand: Optional[str] = None) -> Tuple[str, ...]:
        """The fingers the override may act on for `hand`.

        `hand` None asks the question the report's header asks — "which
        fingers does this configuration touch at all" — and answers it with
        the union over every hand, in FINGER_NAMES order so two profiles that
        name the same fingers in a different order read the same.

        An unknown hand gets () from a mapping, not the mapping's first
        entry: a hand nobody wrote a line for has not been enabled.
        """
        spec = self.fingers
        if isinstance(spec, Mapping):
            if hand is None:
                named = {f for fs in spec.values() for f in fs}
            else:
                named = set(spec.get(str(hand).strip().lower(), ()))
        else:
            named = set(spec)
        return tuple(f for f in FINGER_NAMES if f in named)

    @property
    def per_hand(self) -> bool:
        """Is `fingers` written per hand, rather than once for both?"""
        return isinstance(self.fingers, Mapping)

    def described(self) -> Dict[str, str]:
        d = asdict(self)
        if self.per_hand:
            d["fingers"] = "; ".join(
                f"{hand}: {', '.join(self.fingers_for(hand)) or '(none)'}"
                for hand in sorted(self.fingers)) or "(none)"
        else:
            d["fingers"] = (", ".join(self.fingers_for()) + " (both hands)"
                            if self.fingers_for() else "(none)")
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


def curl_gates_from_rails(gates: Optional[GateParams],
                          template_rails: Mapping[Tuple[str, str], float],
                          fitted_rails: Mapping[Tuple[str, str], float],
                          hand: str) -> Optional[List[float]]:
    """`curl_gate` for one hand, per finger, carried across a template fit.

    `curl_gate` decides whether the GLOVE says a finger is extended enough for
    its abduction to be worth reading, and it is a constant tuned on the
    template hand's curls: 1.2, the midpoint between its fists (0.63-0.8) and
    its open palms (1.71-2.07). Curl is a length RATIO, so fitting the
    template's bones to a real hand multiplies every one of those numbers by
    that finger's scale — measured on sync_day1, the index's open value goes
    1.974 -> 1.707 — and 1.2 is then no longer the midpoint but a bar the
    partly-extended fingers fall under. Left alone it cost 13 points of
    camera-use on the ring finger's spread.

    The fix is not a new constant. A constant retuned on one operator's fitted
    hand would be wrong for the next operator by exactly the amount the fit
    exists to remove. So the threshold is expressed as the same FRACTION of
    that hand and finger's own rail — the glove's own open-palm reading for
    it, which `learn_rails` measures from the session — and evaluated against
    the rails the fusion is actually comparing:

        gate = curl_gate * (fitted rail / template rail)

    Two properties make this safe. With no fit the two rails are the same
    number and the gate is exactly `curl_gate`, so nothing that ran before
    behaves differently. With a fit, the threshold moves by the SAME factor
    the finger's open value moved by, so the decision is preserved rather
    than shifted: measured over both sessions the spread camera-use comes
    back within half a point on index, middle and pinky and 3.6 to 6.2 points
    BETTER on the ring, against the 13-point loss the bare constant caused.
    It is not bit-exact, and cannot be — a bent finger's tip-to-wrist
    distance does not scale by quite the same factor as a straight one's,
    because the segments are rescaled by different amounts each.

    A finger with no rail on this hand (never seen straight, so `learn_rails`
    refuses to guess) keeps the constant: there is no open value to take a
    fraction of. Returns None when there is nothing to carry, which the caller
    passes straight through as "use the constant".
    """
    gates = gates or DEFAULT_GATES
    if not template_rails or not fitted_rails:
        return None
    out = []
    moved = False
    for finger in FINGER_NAMES:
        was = template_rails.get((hand, finger))
        now = fitted_rails.get((hand, finger))
        if was is None or now is None or was <= 1e-9:
            out.append(gates.curl_gate)
            continue
        out.append(gates.curl_gate * float(now) / float(was))
        moved = moved or abs(now - was) > 1e-12
    return out if moved else None


# --- one scale both sensors can be compared on -------------------------

SENSOR_GLOVE = "glove"
SENSOR_CAMERA = "camera"
SENSORS = (SENSOR_GLOVE, SENSOR_CAMERA)

# A fraction is clipped rather than left to run away: a curl a little past
# either learned endpoint is a finger at the end of its range, not a finger
# three times more flexed than the session ever saw. The band is deliberately
# wider than [0, 1] so that "slightly past the rail" stays distinguishable
# from "exactly on it".
FRAC_MIN = -0.2
FRAC_MAX = 1.2

# The flexed endpoint is a percentile and not the minimum, for the reason
# `diagnostics.camera_range` uses one: a single mistracked frame is not the
# bottom of a sensor's range. The camera's open fallback is its mirror.
FLEXED_PERCENTILE = 2.0
OPEN_PERCENTILE = 98.0
# Below this many open-palm-like frames in a session the camera's open
# reference is that percentile instead of a median over them. 20 frames is
# about a fifth of a second of Leap tracking — the same floor
# `template_fit.MIN_OPEN_FRAMES` uses for the same question.
MIN_OPEN_REF_FRAMES = 20


def _percentile(values: Sequence[float], pct: float) -> float:
    return float(np.percentile(np.asarray(list(values), float), pct))


@dataclass(frozen=True)
class Endpoints:
    """One sensor's straight and most-flexed readings for one finger.

    `open` and `flexed` are curls — tip-to-wrist over palm length — on THIS
    sensor's own hand, so the pair is a ruler with that sensor's units baked
    in. `fraction` is what makes two such rulers comparable.
    """

    open: float = 0.0
    flexed: float = 0.0
    n: int = 0
    how: str = ""

    @property
    def span(self) -> float:
        return float(self.open) - float(self.flexed)

    def fraction(self, curl: float) -> float:
        """Where `curl` sits between the two ends: 0 straight, 1 fully flexed."""
        span = self.span
        if span <= 1e-9:
            return 0.0
        frac = (float(self.open) - float(curl)) / span
        return max(FRAC_MIN, min(FRAC_MAX, frac))

    def described(self) -> str:
        return (f"open {self.open:5.3f}  flexed {self.flexed:5.3f}  span "
                f"{self.span:5.3f}  {self.how}")


@dataclass(frozen=True)
class HandScale:
    """One hand's endpoints on both sensors — what `fuse_skeletons` is given.

    Per hand rather than per session because `fuse_skeletons` is a pure
    function of one frame and deliberately does not know which hand it is
    looking at: the caller already selects the per-hand `unreliable_fingers`
    the same way.

    `dropped` names the fingers the guard refused to normalise and why. A
    dropped finger is not a finger the vote is neutral about — it is one the
    vote must not count, because the fraction it would contribute is a
    division by a range that was never measured.
    """

    hand: str = ""
    ends: Mapping[Tuple[str, str], Endpoints] = field(default_factory=dict)
    dropped: Mapping[str, str] = field(default_factory=dict)

    def endpoints(self, sensor: str, finger: str) -> Optional[Endpoints]:
        return self.ends.get((sensor, finger))

    def normalisable(self, finger: str) -> bool:
        return (finger not in self.dropped
                and all((s, finger) in self.ends for s in SENSORS))

    def fraction(self, sensor: str, finger: str,
                 curl: float) -> Optional[float]:
        got = self.ends.get((sensor, finger))
        return None if got is None else got.fraction(curl)

    def disagreement(self, fingers: Sequence[str],
                     curl_glove: Sequence[float],
                     curl_cam: Sequence[float]) -> Optional[float]:
        """Median |frac_glove - frac_camera| over `fingers`, or None.

        Every finger passed must be normalisable; the caller has already
        dropped the ones that are not, and silently skipping them here would
        let a vote be held over a set nobody reported.
        """
        gaps = []
        for finger in fingers:
            i = FINGER_NAMES.index(finger)
            g = self.fraction(SENSOR_GLOVE, finger, curl_glove[i])
            c = self.fraction(SENSOR_CAMERA, finger, curl_cam[i])
            if g is None or c is None:
                continue
            gaps.append(abs(g - c))
        return median(gaps) if gaps else None


@dataclass(frozen=True)
class FlexionScale:
    """Every hand's endpoints, learned once per fusion run.

    Held as a value so a run can be handed one and the report can print it;
    `for_hand` is the only reader `fuse_skeletons` needs.
    """

    hands: Mapping[str, HandScale] = field(default_factory=dict)

    def for_hand(self, hand: Optional[str]) -> Optional[HandScale]:
        return self.hands.get(str(hand).strip().lower())

    def __bool__(self) -> bool:
        return bool(self.hands)


def learn_flexion_scale(
        glove_samples: Iterable[Tuple[str, Sequence[float]]],
        cam_samples: Iterable[Tuple[str, Sequence[float]]],
        rails: Mapping[Tuple[str, str], float],
        gates: Optional[GateParams] = None,
        rail_params: Optional[RailOverrideParams] = None,
        min_open_frames: int = MIN_OPEN_REF_FRAMES) -> FlexionScale:
    """Learn each hand and finger's endpoints on both sensors.

    `glove_samples` and `cam_samples` are (hand_side, curls) per frame, curls
    in FINGER_NAMES order — `features.flexion_features` of whichever skeleton
    is about to be fused, which on the glove side means the FITTED hand when
    a fit is in force. The two streams do not have to be paired or even the
    same length: an endpoint is a fact about a sensor over a session, not
    about a frame, and pairing would only shrink the sample.

    LABEL-FREE, and that is the whole design. Nothing here asks the glove
    whether a frame is open — the glove saying "open" is the claim the rail
    override exists because it cannot be trusted — and nothing asks for a
    pose label, so a session recorded without one is measured the same way.

      glove open     the learned rail. It is already the glove's own
                     full-extension constant for this hand and finger, and it
                     is measured on the same hand the fusion compares against
                     it. A finger with no rail is dropped: `learn_rails`
                     refuses to guess one for a finger the session never
                     showed straight, and so does this.
      camera open    the median over the session's open-palm-LIKE frames —
                     every one of index..little reading above
                     `cam_open_curl - margin`, the camera's own geometric
                     test — or the 98th percentile of that finger's curl when
                     there are fewer than `min_open_frames` of them.
      flexed         the 2nd percentile of that finger's curl on that sensor.

    A finger whose span comes out under `min_glove_span` / `min_cam_span` is
    recorded in `dropped` with the reason, not silently normalised by a
    number that is mostly noise.
    """
    gates = gates or DEFAULT_GATES
    rail_params = rail_params or DEFAULT_RAIL

    g_curls: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for hand, curls in glove_samples:
        side = str(hand).strip().lower()
        for finger, c in zip(FINGER_NAMES, curls):
            g_curls[side][finger].append(float(c))

    c_rows: Dict[str, List[List[float]]] = defaultdict(list)
    for hand, curls in cam_samples:
        c_rows[str(hand).strip().lower()].append([float(c) for c in curls])

    hands: Dict[str, HandScale] = {}
    for side in sorted(set(g_curls) | set(c_rows)):
        ends: Dict[Tuple[str, str], Endpoints] = {}
        dropped: Dict[str, str] = {}
        rows = c_rows.get(side, [])
        # The camera's own open-hand frames, by the same test `template_fit`
        # picks a measurement's frames with.
        open_rows = [r for r in rows
                     if all(r[FINGER_NAMES.index(f)]
                            >= rail_params.open_curl(f) - rail_params.margin
                            for f in RAIL_FINGERS)]
        from_open = len(open_rows) >= min_open_frames

        for finger in FINGER_NAMES:
            i = FINGER_NAMES.index(finger)
            glove_vals = g_curls.get(side, {}).get(finger, [])
            rail = rails.get((side, finger))
            if not glove_vals:
                dropped[finger] = N_NO_FRAMES.format(sensor=SENSOR_GLOVE)
            elif rail is None:
                dropped[finger] = N_NO_RAIL
            else:
                got = Endpoints(
                    open=float(rail),
                    flexed=_percentile(glove_vals, FLEXED_PERCENTILE),
                    n=len(glove_vals),
                    how=f"rail / p{FLEXED_PERCENTILE:.0f} of "
                        f"{len(glove_vals)} frames")
                ends[(SENSOR_GLOVE, finger)] = got
                if got.span < gates.min_glove_span:
                    dropped[finger] = (f"{N_GLOVE_SPAN} "
                                       f"({got.span:.2f} < "
                                       f"{gates.min_glove_span:.2f})")

            cam_vals = [r[i] for r in rows]
            if not cam_vals:
                dropped.setdefault(finger,
                                   N_NO_FRAMES.format(sensor=SENSOR_CAMERA))
                continue
            if from_open:
                open_c = median([r[i] for r in open_rows])
                how = f"median of {len(open_rows)} open-palm-like frames"
            else:
                open_c = _percentile(cam_vals, OPEN_PERCENTILE)
                how = (f"p{OPEN_PERCENTILE:.0f} ({len(open_rows)} "
                       f"open-palm-like frames, need {min_open_frames})")
            got = Endpoints(
                open=float(open_c),
                flexed=_percentile(cam_vals, FLEXED_PERCENTILE),
                n=len(cam_vals), how=how)
            ends[(SENSOR_CAMERA, finger)] = got
            if got.span < gates.min_cam_span:
                dropped.setdefault(finger, f"{N_CAM_SPAN} ({got.span:.2f} < "
                                           f"{gates.min_cam_span:.2f})")
        hands[side] = HandScale(hand=side, ends=ends, dropped=dropped)
    return FlexionScale(hands=hands)


@dataclass(frozen=True)
class RailDecision:
    """One frame's verdict: which fingers the override owns, and why not.

    Produced by `RailOverrideTracker`, consumed by `fuse_skeletons`. Keeping it
    a value means the hysteresis — the only state in this module — lives in one
    object the caller holds, and `fuse_skeletons` stays a pure function of its
    arguments.

    `disputed` is the wider fact `active` is drawn from: every finger sitting
    on its learned rail this frame WHILE the camera reads that finger as
    flexed — whether or not the override is enabled on it, and without the
    consecutive-frame run that arming needs. The thumb gate reads it.

    Being on the rail is not by itself a reason to distrust a finger, and an
    earlier version of this that excluded every railed finger from the thumb
    vote was wrong about that. In peace, open palm and index point the
    extended fingers sit on their rails and the camera agrees they are
    extended — that agreement is the best evidence the vote has. Discarding it
    left the vote to the curled fingers alone and refused a correct camera
    thumb on almost every right-hand peace frame. A rail is suspect only when
    the camera contradicts it, which is the same disagreement the override
    itself is built on.
    """
    active: Tuple[str, ...] = ()
    rejected: Mapping[str, str] = field(default_factory=dict)
    disputed: Tuple[str, ...] = ()


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

        # Every finger the two sensors CONTRADICT each other about — on its
        # rail while the camera reads it flexed — computed for all four, not
        # only the enabled ones, because the thumb gate needs it even when
        # nothing is being overridden. No run length here: this is the
        # instantaneous disagreement, and it is arming that needs patience.
        #
        # A railed finger the camera ALSO calls extended is not on this list.
        # The two sensors agree about it, which is exactly the evidence the
        # thumb vote wants.
        disputed = tuple(
            f for f in RAIL_FINGERS
            if (hand, f) in self.rails
            and abs(float(glove_curls[FINGER_NAMES.index(f)])
                    - self.rails[(hand, f)]) <= self.params.tol
            and cam_curls is not None
            and float(cam_curls[FINGER_NAMES.index(f)])
            < self.params.open_curl(f) - self.params.margin)

        active: List[str] = []
        rejected: Dict[str, str] = {}
        # Per hand: the override is enabled on the fingers whose failure has
        # actually been measured on THIS hand, which is not in general the
        # same list as the other hand's.
        for finger in self.params.fingers_for(hand):
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
        return RailDecision(tuple(active), rejected, disputed)


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
            "thumb_vote_fingers": [],
            "thumb_vote_dropped": {},
            "view_angle_deg": None, "field_angle_deg": None,
            "curl_disagreement": None, "flex_disagreement": None}


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
    unreliable_fingers: Sequence[str] = (),
    curl_gates: Optional[Sequence[float]] = None,
    scale: Optional[HandScale] = None,
    gate_curls: Optional[Sequence[float]] = None,
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

    `unreliable_fingers` names fingers of THIS hand whose glove curl the
    operator does not trust, excluding them from the thumb gate's vote. It is
    per hand because gloves fail per hand: on sync_day1 the right glove's ring
    and pinky read partly extended through thumbs_up and peace while the left
    glove's do not.

    `curl_gates` replaces `gates.curl_gate` per finger, in FINGER_NAMES order.
    It exists for one caller — a glove hand whose bones have been rescaled to
    the operator's by `template_fit`, which moves every curl the gate reads —
    and `curl_gates_from_rails` is what builds it. None, the default, uses the
    one constant for every finger, which is what every caller did before.

    `scale` is THIS hand's learned flexion endpoints (`learn_flexion_scale`,
    then `FlexionScale.for_hand`). With it the thumb vote is held on flexion
    FRACTIONS against `gates.agree_tol_frac`, which is the comparison that
    survives a template fit; fingers it could not normalise drop out of the
    vote and are listed in `info["thumb_vote_dropped"]`. Without it the vote
    falls back to raw curls against `gates.curl_agree_tol` — a caller with
    one frame and no session behind it has no endpoints to normalise with.

    `gate_curls` are the glove curls every curl-based DECISION in here reads,
    in FINGER_NAMES order: the spread gate's "is this finger curled" and the
    thumb vote's glove fractions and disagreement. None, the default, reads
    them off `glove_pts`, which is what every caller did before. It exists
    for the drift anchor (`DriftAnchor`), which bends the finger that is
    FUSED and must not change what the gates know about the glove: a gate
    that read the bent finger would refuse the camera's spread on exactly
    the frames the anchor had just fixed (measured on sync_day1 and
    sync_day2, the spread camera-use rates fell by 3 to 15 points with the
    anchor on). So the anchor's caller passes the UNCORRECTED curls here and
    the corrected points as `glove_pts`: the gates decide on what the glove
    measured, and the fused hand carries the anchor's correction.

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

    # What the gates below know about the glove: its own curls, or the ones
    # the caller says it measured (see `gate_curls`). Every curl-based
    # decision reads `curl_g` and nothing else.
    curl_g = (flexion_features(G) if gate_curls is None
              else [float(v) for v in gate_curls])
    curl_c = flexion_features(C_in)      # a ratio, so alignment cannot change it

    # Who is allowed to vote on whether the camera has this hand right.
    #
    # index..little only: the thumb is the DOF being decided, so it cannot vote
    # for itself. And of those four, a finger only counts if its GLOVE curl is
    # a measurement:
    #
    #   disputed      the glove is on this finger's rail — reporting a
    #                 constant — AND the camera reads it flexed. In
    #                 sync_day1's pinch the railed index "disagrees" with the
    #                 camera by 0.70 purely because the glove stopped
    #                 measuring, and one such number is enough to drag the
    #                 median over the tolerance and veto a thumb the camera
    #                 had right.
    #
    #                 Being on the rail is NOT enough on its own, and an
    #                 earlier version of this rule that excluded every railed
    #                 finger was wrong. In peace, open palm and index point
    #                 the extended fingers are on their rails and the camera
    #                 agrees they are extended: that agreement is the best
    #                 evidence this vote has, and discarding it left the
    #                 verdict to the curled fingers alone and refused a
    #                 correct camera thumb on almost every right-hand peace
    #                 frame. A rail is suspect only when the camera
    #                 contradicts it.
    #   overridden    the rail override has already ruled the glove wrong
    #                 about this finger. Letting it vote would be asking the
    #                 loser of one argument to judge the next. Not implied by
    #                 `disputed`: hysteresis keeps a finger overridden for
    #                 `exit_frames` after the disagreement stops.
    #   unreliable    the operator has said so. On sync_day1 the RIGHT glove
    #                 reports ring and pinky partly extended through thumbs_up
    #                 and peace (pinky 1.55-1.68 where the camera says
    #                 0.78-0.90, consistently, take after take), which is a
    #                 fault in that glove's fingers rather than evidence about
    #                 the camera.
    #
    #   unnormalisable with `scale` in hand, a finger whose curl range on
    #                 either sensor was too small to learn endpoints from.
    #                 See `learn_flexion_scale`.
    #
    # What the survivors are compared ON is a flexion FRACTION on each
    # sensor's own endpoints, not a raw curl difference: the glove's curls
    # live on the template hand (or on the fitted one) and the camera's on the
    # operator's, so a fixed raw tolerance means different things in a fitted
    # and an unfitted run. See the module docstring.
    #
    # Below `min_usable_fingers` the glove has no opinion worth acting on, so
    # it casts no veto at all and the camera's own geometry — visibility, id
    # continuity, central field, viewing angle — is left to decide. That is a
    # deliberate choice to fail toward the sensor that still has evidence.
    excluded = set(rail.disputed) | set(rail.active) | set(unreliable_fingers)
    usable = [f for f in SPREAD_FINGERS if f not in excluded]
    # ...and a fourth exclusion when the vote is held on fractions: a finger
    # whose range on either sensor was too small to learn endpoints from
    # cannot be put on one, and a fraction over an unmeasured range is noise
    # divided by noise. Recorded, not silently skipped.
    if scale is not None:
        dropped = {f: scale.dropped.get(f, R_NOT_NORMALISED)
                   for f in usable if not scale.normalisable(f)}
        info["thumb_vote_dropped"] = dropped
        usable = [f for f in usable if f not in dropped]
    info["thumb_vote_fingers"] = list(usable)
    if usable:
        disagreement = median([abs(curl_g[FINGER_NAMES.index(f)]
                                   - curl_c[FINGER_NAMES.index(f)])
                               for f in usable])
    else:
        disagreement = None
    info["curl_disagreement"] = disagreement
    # The number the gate actually reads when endpoints are available. Kept
    # beside the raw one rather than replacing it, so a report can print both
    # and a reader can see the change of units for what it is.
    flex_disagreement = (None if scale is None or not usable
                         else scale.disagreement(usable, curl_g, curl_c))
    info["flex_disagreement"] = flex_disagreement

    def curl_gate_for(finger: str) -> float:
        i = FINGER_NAMES.index(finger)
        if curl_gates is None or i >= len(curl_gates):
            return gates.curl_gate
        return float(curl_gates[i])

    def spread_verdict(finger: str) -> Optional[str]:
        """None if the camera may set this finger's azimuth, else the reason."""
        if cam_meta is None:
            return None
        if curl_g[FINGER_NAMES.index(finger)] <= curl_gate_for(finger):
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
        # too few fingers still measuring to hold a vote: no veto
        if len(usable) < gates.min_usable_fingers:
            return None
        if flex_disagreement is not None:
            if flex_disagreement >= gates.agree_tol_frac:
                return R_DISAGREE
            return None
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


# --- bending a glove finger to a curl ----------------------------------

# How one bend is shared between the finger's three joints: the whole distal
# chain turns about the knuckle by half of it, the part beyond the PIP by a
# further 30 %, the part beyond the DIP by the last 20 %. A real finger
# flexes at all three, and putting the whole bend at the knuckle (as this did
# at first) left a folded finger with a proximal bone pointing nearly
# straight out of the palm: the spread transfer then read that bone's
# azimuth off a direction with almost no in-plane component, and on
# sync_day1's pinch_right_take1 it produced 44 and 59 degree spreads.
BEND_SHARES = (0.5, 0.3, 0.2)

# The range of the ONE angle parameter, the bend summed over the three
# joints, in degrees. Positive folds the finger toward the palm. +240 puts
# 120 at the knuckle, 72 at the PIP and 48 at the DIP, enough to carry an
# open finger into a fist; -60 opens a finger the glove reads too FLEXED past
# its current line by 30 / 18 / 12. The search prefers the smallest bend that
# reaches the target, so neither end is visited unless the target needs it.
BEND_MIN_DEG = -60.0
BEND_MAX_DEG = 240.0
# The one-degree grid the search starts from; it contains t = 0 exactly.
_BEND_GRID = np.radians(np.arange(BEND_MIN_DEG, BEND_MAX_DEG + 0.5, 1.0))


def _palmar_sign(pts: np.ndarray, normal: np.ndarray) -> float:
    """+1 if the palm faces along `normal`, -1 if it faces the other way.

    `palm_frame`'s normal is built from the knuckles, so it points out of the
    palm on one hand and out of the back of the other (see
    `features.palm_normal`). The skeleton says which: fingers fold toward the
    palm, so the four fingertips sit on the palm side of the knuckles. The
    glove's template does this even on an open palm (every fingertip 0.04 to
    0.12 of its reach toward the palm on sync_day1, on both hands, the sign
    flipping with the hand), so the sum is a reliable vote without being told
    the hand. A perfectly flat hand has no vote and gets +1, the right hand's
    answer.
    """
    lean = 0.0
    for finger in RAIL_FINGERS:
        chain = FINGER_CHAINS[finger]
        lean += float(np.dot(pts[chain[-1]] - pts[chain[0]], normal))
    return -1.0 if lean < -1e-12 else 1.0


def _turn(vec: np.ndarray, axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """`vec` rotated about unit `axis` by each of `angles` (radians), (m, 3)."""
    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    return (vec[None, :] * c + np.cross(axis, vec)[None, :] * s
            + axis[None, :] * float(axis @ vec) * (1.0 - c))


def bend_finger_to_curl(pts21: Sequence[Sequence[float]], finger: str,
                        target_curl: float) -> np.ndarray:
    """Fold one finger until its curl reads `target_curl`.

    The bend is shared between the three joints (`BEND_SHARES`): the PIP,
    DIP and tip turn RIGIDLY about the knuckle by half of one angle t, the
    DIP and tip then about the PIP by a further 0.3 t, and the tip about the
    DIP by the last 0.2 t, all around the palm frame's across-the-palm axis
    and signed so a positive t folds toward the palm and so toward the wrist.
    Every step is a rotation about a joint, so every bone keeps its length,
    the invariant the rest of this module rests on. Nothing else moves: the
    knuckle belongs to the palm, the palm length is wrist to middle knuckle,
    and so every other finger's curl is untouched.

    Because the three rotations share one axis they compose by adding
    angles, so the bone from knuckle to PIP has turned by 0.5 t, the next
    by 0.8 t and the last by t: the tip is a closed-form function of t and
    the search is a bounded one over t alone. t is sampled every degree over
    [BEND_MIN_DEG, BEND_MAX_DEG]; of the crossings of the target the one
    nearest t = 0 is taken, since the glove's own pose is the best guess at
    the rest of the finger, and it is refined on a hundredth-of-a-degree grid
    and interpolated, which lands within 1e-6 of the target. A target no t
    in the range reaches gets the reachable curl nearest it.

    Pure: `pts21` is not modified and a new array is returned.
    """
    P = np.array(pts21, dtype=float)
    chain = FINGER_CHAINS[finger]
    wrist = P[WRIST]
    palm = float(np.linalg.norm(P[MIDDLE_MCP] - wrist))
    if palm <= 1e-12:
        return P
    n, x, _y = palm_frame(P)
    palmar = _palmar_sign(P, n) * n
    # A positive rotation about cross(x, palmar) carries the direction down
    # the palm (x) toward the palm side, which is the fold a flexing finger
    # makes.
    axis = _unit(np.cross(x, palmar))
    knuckle = P[chain[0]].copy()
    bones = np.array([P[chain[k + 1]] - P[chain[k]] for k in range(3)])
    turns = np.cumsum(BEND_SHARES)            # 0.5, 0.8, 1.0 of t per bone
    target = float(target_curl)
    # Rodrigues, split once: each bone is its part along the axis (which no
    # rotation about it moves), plus cos and sin of its own angle times two
    # fixed vectors. The tip for many t is then two matrix products.
    along = bones @ axis
    fixed = knuckle - wrist + along.sum() * axis
    cos_part = bones - along[:, None] * axis[None, :]
    sin_part = np.cross(axis[None, :], bones)

    def curls(t: np.ndarray) -> np.ndarray:
        ang = t[:, None] * turns[None, :]
        rel = fixed + np.cos(ang) @ cos_part + np.sin(ang) @ sin_part
        return np.sqrt(np.einsum("ij,ij->i", rel, rel)) / palm

    lo, hi = math.radians(BEND_MIN_DEG), math.radians(BEND_MAX_DEG)
    coarse = _BEND_GRID
    err = curls(coarse) - target
    brackets = np.nonzero((err[:-1] == 0.0) | (err[:-1] * err[1:] < 0.0))[0]
    if len(brackets):
        near = np.minimum(np.abs(coarse[brackets]), np.abs(coarse[brackets + 1]))
        k = int(brackets[int(np.argmin(near))])
        fine = np.linspace(coarse[k], coarse[k + 1], 101)
        e = curls(fine) - target
        j = int(np.nonzero((e[:-1] == 0.0) | (e[:-1] * e[1:] <= 0.0))[0][0])
        span = e[j] - e[j + 1]
        angle = float(fine[j] + (fine[j + 1] - fine[j])
                      * (e[j] / span if span != 0.0 else 0.0))
    else:
        # Nearest reachable: the smallest error, and of equal errors the
        # smallest bend (lexsort's LAST key is its primary one).
        k = int(np.lexsort((np.abs(coarse), np.abs(err)))[0])
        fine = np.linspace(max(lo, coarse[k] - math.radians(1.0)),
                           min(hi, coarse[k] + math.radians(1.0)), 201)
        e = np.abs(curls(fine) - target)
        angle = (float(coarse[k]) if abs(err[k]) <= float(e.min())
                 else float(fine[int(np.argmin(e))]))
    if angle == 0.0:
        return P
    point = knuckle
    for k, (bone, share) in enumerate(zip(bones, turns)):
        point = point + _turn(bone, axis, np.array([share * angle]))[0]
        P[chain[k + 1]] = point
    return P


# --- the camera anchors the glove's slow drift (experimental) ----------

def camera_trusted(cam_meta: Optional[dict],
                   gates: Optional[GateParams] = None) -> bool:
    """May this camera frame's curls be used as a REFERENCE for the glove?

    `frame_trust` passes AND the palm faces the module within the view gate.
    An edge-on hand is the camera inferring fingers it cannot see (the
    thumbs_up failure), and a confident wrong curl there is not a reference.
    A frame with no capture facts (MediaPipe) cannot be judged and is not
    trusted. Shared by `DriftAnchor` and `GloveRecalibration`, so the two
    learn from exactly the same frames.
    """
    if cam_meta is None:
        return False
    gates = gates or DEFAULT_GATES
    ok, _reasons, metrics = frame_trust(cam_meta, gates)
    view = metrics["view_angle_deg"]
    return bool(ok and view is not None and view < gates.view_gate_deg)


@dataclass(frozen=True)
class DriftAnchorParams:
    """Thresholds for the camera drift anchor. EXPERIMENTAL, off by default.

    Measured on sync_day1 and sync_day2, POSE DEPENDENCE dominates the
    glove's error over the timescale of these takes: the median residual
    per pose spans 0.4 to 1.05 flexion fractions on the right hand (right
    middle -0.19 in fist, +0.86 in index_point on day 2), while the change
    within a take, first second against last, is 0.02 to 0.10. Creep may
    coexist with it, but an offset learned in one pose is wrong in the next:
    carried across takes it folded the extended fingers of
    peace_right_take2 into a pinch. So `fuse_all` resets the anchor at every
    take boundary, and a live caller must reset it whenever the hand is lost
    (`DriftAnchor.reset`). `GloveRecalibration` models the pose dependence
    itself.

    window_s    the offset is the median residual over the trusted frames
                within this many seconds before the NEWEST trusted frame.
                5 s still holds up to 300 frames at 60 Hz, so a few
                mistracked ones cannot move a median, and is short against
                a creep of 0.05 to 0.26 curl per minute: a median over a
                5 s ramp trails the ramp by 2.5 s of creep, 0.002 to 0.011.
                Chosen by a sweep on sync_day1 and sync_day2 (profile plus
                anchor, fused leave-one-take-out): 5 s gave 58/59 and
                54/60, 10 s 57/59 and 53/60, 20 s 57/59 and 52/60. One
                take either way on each day, so this is the better of three
                close settings, not a measured optimum.
    min_frames  a window with fewer trusted residuals than this teaches no
                new offset. 30 frames is half a second of glove at 60 Hz.
    hold_s      an offset is applied for at most this long after the last
                trusted frame that taught it. 120 s is two minutes of creep
                at the measured rate, after which the offset describes a
                glove that no longer exists. With a reset at every take
                boundary it only matters inside a long take.
    deadband    in FRACTION units. The applied offset is
                sign(o) * max(0, |o| - deadband): continuous, so a median
                wandering across the band's edge does not switch a correction
                on and off, and zero for the few hundredths two honest
                sensors disagree by anyway.
    """
    window_s: float = 5.0
    min_frames: int = 30
    hold_s: float = 120.0
    deadband: float = 0.03

    def described(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_ANCHOR = DriftAnchorParams()

# How many of the most recent in-force offsets per (hand, finger) the report's
# median is taken over. Everything else the anchor reports is a running count,
# sum or maximum, so this bound is what keeps a multi-hour live run at
# constant memory. 100,000 is about 28 minutes of glove at 60 Hz: every frame
# of a recorded session, so a session report's median is exact.
ANCHOR_MEDIAN_KEEP = 100_000


class DriftAnchor:
    """Learn the glove's slow error from the camera and take it back out.

    EXPERIMENTAL (see `DriftAnchorParams`): the glove's error is mostly
    pose-dependent, so an offset is only valid while the pose it was learned
    in lasts. One object per fusion run for its report counters, but its
    windows and offsets are cleared by `reset`, which `fuse_all` calls at
    every take boundary and a LIVE caller must call whenever the tracker
    loses the hand (the next hand it sees may be in any pose). Keyed by
    (hand, finger) for the four fingers in RAIL_FINGERS. The thumb is left
    alone: when the thumb gate passes its whole direction is already the
    camera's, and when it fails the glove is being kept on purpose.

    `learn` is called on frames that have a camera partner, `apply` on every
    frame, both in time order. The residual is taken on flexion FRACTIONS,
    each sensor on its own learned endpoints (`HandScale`), because the raw
    curls are on two different rulers: the template index reads 1.97 open
    where the camera reads 1.75, and that gap is bone length, not drift.

    It keeps counters for the report: per key, the frames it learned from,
    the learned offset in force on each frame it was consulted, the frames
    it actually moved and by how much in curl units.
    """

    def __init__(self, params: Optional[DriftAnchorParams] = None):
        self.params = params or DEFAULT_ANCHOR
        self._window: Dict[Tuple[str, str], Deque[Tuple[float, float]]] = (
            defaultdict(deque))
        # (median residual, stamp of the newest trusted frame behind it):
        # the offset as last established from a window of at least
        # `min_frames`, kept while the window refills after being pruned.
        self._held: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._newest: Dict[Tuple[str, str], float] = {}
        # Report counters, all constant-size per key: running counts, sums
        # and maxima, plus the most recent ANCHOR_MEDIAN_KEEP offsets for
        # the median.
        self.learned: Dict[Tuple[str, str], int] = defaultdict(int)
        self.in_force: Dict[Tuple[str, str], int] = defaultdict(int)
        self.offset_sum: Dict[Tuple[str, str], float] = defaultdict(float)
        self.offset_sq: Dict[Tuple[str, str], float] = defaultdict(float)
        self._recent: Dict[Tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=ANCHOR_MEDIAN_KEEP))
        self.corrected: Dict[Tuple[str, str], int] = defaultdict(int)
        self.sum_abs: Dict[Tuple[str, str], float] = defaultdict(float)
        self.max_abs: Dict[Tuple[str, str], float] = defaultdict(float)
        self.spans: Dict[Tuple[str, str], float] = {}

    def reset(self, hand: Optional[str] = None) -> None:
        """Forget windows and offsets; keep the report's counters.

        `hand` None forgets both hands, which is what `fuse_all` does at
        every take boundary. A live caller loses ONE hand at a time and
        passes it, so the other hand keeps what it has learned. Either way
        the reason is the same: the glove's error depends on the pose, and
        nothing learned before the break says which pose the hand is in
        after it.
        """
        for store in (self._window, self._held, self._newest):
            if hand is None:
                store.clear()
            else:
                for key in [k for k in store if k[0] == hand]:
                    del store[key]

    def learn(self, hand: str, t: float, glove_curls: Sequence[float],
              cam_curls: Optional[Sequence[float]],
              cam_meta: Optional[dict],
              scale: Optional[HandScale],
              rails: Mapping[Tuple[str, str], float],
              rail_tol: float,
              disputed: Sequence[str] = (),
              gates: Optional[GateParams] = None) -> Tuple[str, ...]:
        """Record this frame's residuals, if the frame can teach any.

        `glove_curls` must be the UNCORRECTED glove: the residual is the
        glove's own error, and measuring it on a hand this anchor has already
        moved would teach the anchor about itself.

        A finger teaches only when every one of these holds:

          the camera frame is trusted   `frame_trust` passes AND the palm
                                        faces the module within the view
                                        gate. An edge-on hand is the camera
                                        inferring fingers it cannot see (the
                                        thumbs_up failure), and a confident
                                        wrong curl there is not a reference.
          the finger is normalisable    both sensors have endpoints for it.
          it is not disputed            glove on its rail while the camera
                                        sees it flexed: that is the rail
                                        override's case, not drift.
          the glove is OFF its rail     a rail reading is the sensor against
                                        its stop, the same bits whatever the
                                        finger does, so it carries no
                                        information about slope or creep. A
                                        finger with no learned rail counts
                                        as off it.

        Returns the fingers that taught, for tests and diagnostics.
        """
        if scale is None or cam_curls is None:
            return ()
        if not camera_trusted(cam_meta, gates):
            return ()
        taught = []
        t = float(t)
        for finger in RAIL_FINGERS:
            if finger in disputed or not scale.normalisable(finger):
                continue
            i = FINGER_NAMES.index(finger)
            g = float(glove_curls[i])
            rail = rails.get((hand, finger))
            if rail is not None and abs(g - float(rail)) <= rail_tol:
                continue
            gf = scale.fraction(SENSOR_GLOVE, finger, g)
            cf = scale.fraction(SENSOR_CAMERA, finger, float(cam_curls[i]))
            if gf is None or cf is None:
                continue
            key = (hand, finger)
            win = self._window[key]
            win.append((t, cf - gf))
            newest = max(t, self._newest.get(key, t))
            self._newest[key] = newest
            while win and win[0][0] < newest - self.params.window_s:
                win.popleft()
            self.learned[key] += 1
            if len(win) >= self.params.min_frames:
                self._held[key] = (median([r for _s, r in win]), newest)
            taught.append(finger)
        return tuple(taught)

    def _deadbanded(self, raw: float) -> float:
        return math.copysign(max(0.0, abs(raw) - self.params.deadband), raw)

    def learned_offset(self, hand: str, finger: str,
                       t: float) -> Optional[float]:
        """The offset in force at `t` BEFORE the deadband, or None."""
        held = self._held.get((hand, finger))
        if held is None:
            return None
        value, last = held
        if float(t) - last > self.params.hold_s:
            return None
        return value

    def offset(self, hand: str, finger: str, t: float) -> Optional[float]:
        """The offset to apply at `t`, in fraction units, or None.

        None before a window of `min_frames` trusted residuals has ever been
        seen, and once the newest trusted frame behind the offset is more than
        `hold_s` older than `t`. Otherwise the median residual, shrunk by the
        deadband.
        """
        raw = self.learned_offset(hand, finger, t)
        return None if raw is None else self._deadbanded(raw)

    def apply(self, hand: str, t: float, glove_pts: Sequence[Sequence[float]],
              glove_curls: Sequence[float],
              scale: Optional[HandScale]
              ) -> Tuple[np.ndarray, Dict[str, float]]:
        """The glove hand with its learned drift taken out, and by how much.

        For each finger with an offset: the target is the glove's own
        fraction plus the offset, kept within [0, 1] (straight to fully
        flexed), and put back into GLOVE curl on the glove's endpoints; the
        finger is then folded to it by `bend_finger_to_curl`. The clip never
        reverses a correction: a glove already past an endpoint is left where
        it is rather than pulled back against the offset's sign.

        Returns the new points and the curl change actually made per finger
        (signed, curl units). Fingers with no offset, or whose offset is
        inside the deadband, are not in the dict and are not moved.
        """
        pts = np.asarray(glove_pts, dtype=float)
        corrections: Dict[str, float] = {}
        if scale is None:
            return pts, corrections
        out = None
        for finger in RAIL_FINGERS:
            key = (hand, finger)
            raw = self.learned_offset(hand, finger, t)
            ends = scale.endpoints(SENSOR_GLOVE, finger)
            if raw is None or ends is None or ends.span <= 1e-9:
                continue
            self.in_force[key] += 1
            self.offset_sum[key] += raw
            self.offset_sq[key] += raw * raw
            self._recent[key].append(raw)
            self.spans[key] = ends.span
            o = self._deadbanded(raw)
            if o == 0.0:
                continue
            i = FINGER_NAMES.index(finger)
            curl = float(glove_curls[i])
            # Unclipped, unlike `Endpoints.fraction`: the correction is a
            # difference of two fractions, and clipping only one of them
            # would move a finger that is past an endpoint for no reason.
            frac = (float(ends.open) - curl) / ends.span
            target = min(max(frac + o, min(0.0, frac)), max(1.0, frac))
            if target == frac:
                continue
            base = pts if out is None else out
            out = bend_finger_to_curl(base, finger, ends.open
                                      - target * ends.span)
            tip = FINGER_CHAINS[finger][-1]
            palm = float(np.linalg.norm(out[MIDDLE_MCP] - out[WRIST]))
            change = float(np.linalg.norm(out[tip] - out[WRIST])) / palm - curl
            corrections[finger] = change
            if abs(change) > 1e-12:
                self.corrected[key] += 1
                self.sum_abs[key] += abs(change)
                self.max_abs[key] = max(self.max_abs[key], abs(change))
        return (pts if out is None else out), corrections

    def summary(self) -> List[dict]:
        """One row per (hand, finger) the anchor learned from or moved.

        `median_offset` is over the most recent ANCHOR_MEDIAN_KEEP frames the
        offset was in force (all of them, for any recorded session);
        `mean_offset` and `sd_offset` are over every one, from running sums.
        """
        rows = []
        keys = set(self.learned) | set(self.in_force)
        for key in sorted(keys, key=lambda k: (k[0], FINGER_NAMES.index(k[1]))):
            hand, finger = key
            n_used = self.in_force.get(key, 0)
            recent = self._recent.get(key)
            span = self.spans.get(key)
            off = median(list(recent)) if recent else None
            mean_off = (self.offset_sum[key] / n_used) if n_used else None
            sd_off = (math.sqrt(max(0.0, self.offset_sq[key] / n_used
                                    - mean_off * mean_off))
                      if n_used else None)
            n_corr = self.corrected.get(key, 0)
            rows.append({
                "hand": hand, "finger": finger,
                "learned": self.learned.get(key, 0),
                "in_force": n_used,
                "median_offset": off,
                "mean_offset": mean_off,
                "sd_offset": sd_off,
                "median_offset_curl": (off * span if off is not None
                                       and span is not None else None),
                "corrected": n_corr,
                "mean_abs": (self.sum_abs.get(key, 0.0) / n_corr
                             if n_corr else 0.0),
                "max_abs": self.max_abs.get(key, 0.0),
            })
        return rows


# --- camera-referenced glove recalibration -----------------------------

RECAL_OWN = "own"
RECAL_CROSS = "cross"
RECAL_VARIANTS = (RECAL_OWN, RECAL_CROSS)


@dataclass(frozen=True)
class RecalibrationParams:
    """How `GloveRecalibration` fits. See the module docstring.

    variant       `own`: finger i's camera fraction from finger i's glove
                  fraction alone, a per-finger gain and offset. `cross`: from
                  all five glove fractions, thumb included, so a channel that
                  also answers to its neighbour (a sensor that stretches when
                  the finger beside it bends) can be untangled.
    ridge_lambda  the ridge penalty on STANDARDISED inputs, on the mean
                  squared error: minimise mean((y - a - Z b)^2) + lambda |b|^2
                  with Z each input minus its mean over its standard
                  deviation, the intercept unpenalised. 0.01 is small beside
                  a standardised input's unit variance, so it only matters
                  where inputs are nearly collinear (a finger that barely
                  moved), which is what it is for.
    min_frames    a hand with fewer trusted frames than this to fit on gets
                  no model and no correction, and the report says why. 50 is
                  under a second of glove: a floor against fitting six
                  numbers to a handful of frames, not a sufficiency claim.
    """
    variant: str = RECAL_CROSS
    ridge_lambda: float = 1e-2
    min_frames: int = 50

    def described(self) -> Dict[str, object]:
        return asdict(self)


DEFAULT_RECAL = RecalibrationParams()


@dataclass
class RecalModel:
    """One hand's fitted recalibration, or why there is none.

    `coef[finger]` is (bias, {input finger: coefficient}) in FRACTION units,
    already un-standardised: the finger's camera fraction is predicted as
    bias + sum(coefficient * glove fraction of the input finger).
    """
    hand: str
    variant: str
    n_frames: int = 0
    refused: str = ""
    inputs: Tuple[str, ...] = ()
    coef: Dict[str, Tuple[float, Dict[str, float]]] = field(
        default_factory=dict)

    def predict(self, finger: str,
                glove_fracs: Mapping[str, float]) -> Optional[float]:
        got = self.coef.get(finger)
        if got is None:
            return None
        bias, weights = got
        return float(bias + sum(w * float(glove_fracs[f])
                                for f, w in weights.items()))


def _glove_fraction(ends: Optional[Endpoints], curl: float) -> float:
    """Unclipped glove fraction, NaN without usable endpoints."""
    if ends is None or ends.span <= 1e-9:
        return float("nan")
    return (float(ends.open) - float(curl)) / ends.span


class GloveRecalibration:
    """Camera-referenced recalibration of the glove's finger curls.

    The glove's error on these sessions is mostly a function of the POSE,
    not of time (`DriftAnchorParams`), which is what a calibration is for:
    per hand, for each finger i of RAIL_FINGERS,

        camera_fraction_i = a_i + sum over j of b_ij * glove_fraction_j

    with the inputs the glove's own fractions on its learned endpoints,
    UNCLIPPED (a reading past an endpoint is information), and the target
    the camera's fraction on its endpoints. Ridge regression in closed form
    (`RecalibrationParams`); `own` keeps only j = i.

    Fitted on TRUSTED frames only (`camera_trusted`, the anchor's rule).

    THE RAIL RULE. A finger's reading on its own learned rail (within
    `rail_tol` of it) means anything from straight to flexed: the sensor is
    against its stop and reports the same bits whatever the finger does. It
    has no slope information, so
      - a finger's OWN on-rail frames never teach that finger's model (on
        sync_day1's pinch the railed index taught its model a +0.24 bias,
        which then bent every open palm's index by a quarter of its range);
      - a finger on its rail on a frame is not corrected on that frame: a
        rail dispute belongs to the rail override, not to a calibration.
    The OTHER fingers' readings stay valid INPUTS whether or not they are on
    their rails: a saturated neighbour is still information about the hand's
    state. A finger with no learned rail is always off it, as everywhere
    else in this module.

    `observe` collects rows tagged with a group (a take index); `model(hand,
    exclude=k)` fits on every group but k and caches it, which is how the
    evaluation stays leave-one-take-out. `apply` corrects one frame with a
    given held-out model and keeps the report's counters.

    CROSS-SESSION. Given `fitted_on`, another session's
    `GloveRecalibration`, every model this object hands out is THAT
    session's, fitted on all of its trusted frames, and there is no
    leave-one-take-out: nothing in this session was in the fit. Everything
    else stays this session's own: the rows it scores, the endpoints the
    fractions are taken on (the `scale` passed to `apply`) and the rails the
    rail rule checks. That is how a live warm-up would deploy it: the
    coefficients carry over, the endpoints and rails are re-learned.

    It is a BATCH fit over a recorded session, and it holds its training
    rows (ten numbers per trusted frame) to refit the held-out models. A
    live caller would fit once from a warm-up and keep only `RecalModel`.
    """

    def __init__(self, params: Optional[RecalibrationParams] = None,
                 gates: Optional[GateParams] = None,
                 rails: Optional[Mapping[Tuple[str, str], float]] = None,
                 rail_tol: float = DEFAULT_RAIL.tol,
                 fitted_on: Optional["GloveRecalibration"] = None,
                 source: str = ""):
        self.params = params or DEFAULT_RECAL
        if self.params.variant not in RECAL_VARIANTS:
            raise ValueError(f"variant {self.params.variant!r}: choose from "
                             f"{', '.join(RECAL_VARIANTS)}")
        if fitted_on is not None and fitted_on.params != self.params:
            raise ValueError("a cross-session model must be fitted with the "
                             "same RecalibrationParams it is applied with")
        # The session whose trusted frames the models are fitted on, when it
        # is not this one, and how the report names it.
        self.fitted_on = fitted_on
        self.source = str(source)
        self.gates = gates or DEFAULT_GATES
        self.rails = dict(rails or {})
        self.rail_tol = float(rail_tol)
        # hand -> list of (group, glove fracs [5], camera fracs [4],
        # on-its-own-rail flags [4])
        self._rows: Dict[str, List[tuple]] = defaultdict(list)
        self._arrays: Dict[str, tuple] = {}
        # (hand, finger) -> trusted frames left out of that finger's fit
        # because the glove sat on its rail
        self.railed: Dict[Tuple[str, str], int] = defaultdict(int)
        self._models: Dict[Tuple[str, object], RecalModel] = {}
        self.corrected: Dict[Tuple[str, str], int] = defaultdict(int)
        self.sum_abs: Dict[Tuple[str, str], float] = defaultdict(float)
        self.max_abs: Dict[Tuple[str, str], float] = defaultdict(float)
        # (hand, pose, finger) -> ([before], [after]) held-out residuals
        self.residuals: Dict[Tuple[str, str, str],
                             Tuple[List[float], List[float]]] = {}

    def on_rail(self, hand: str, finger: str, glove_curl: float) -> bool:
        """Is this glove reading on the finger's own learned rail?"""
        rail = self.rails.get((str(hand), finger))
        return rail is not None and abs(float(glove_curl) - rail) <= self.rail_tol

    def observe(self, group: object, hand: str,
                glove_curls: Sequence[float],
                cam_curls: Optional[Sequence[float]],
                cam_meta: Optional[dict],
                scale: Optional[HandScale]) -> bool:
        """Record one paired frame if it is trusted. `glove_curls` UNCORRECTED.

        A finger on its own rail keeps its place as an input to the other
        fingers' models and is flagged, so it teaches its own model nothing
        (the rail rule, above)."""
        if scale is None or cam_curls is None:
            return False
        if not camera_trusted(cam_meta, self.gates):
            return False
        hand = str(hand)
        g = []
        for finger in FINGER_NAMES:
            ends = scale.endpoints(SENSOR_GLOVE, finger)
            # An input whose glove range is too small to normalise is noise
            # over noise, the same guard the thumb vote applies.
            if ends is not None and ends.span < self.gates.min_glove_span:
                ends = None
            g.append(_glove_fraction(ends,
                                     glove_curls[FINGER_NAMES.index(finger)]))
        c = []
        for finger in RAIL_FINGERS:
            got = (scale.fraction(SENSOR_CAMERA, finger,
                                  cam_curls[FINGER_NAMES.index(finger)])
                   if scale.normalisable(finger) else None)
            c.append(float("nan") if got is None else float(got))
        railed = [self.on_rail(hand, finger,
                               glove_curls[FINGER_NAMES.index(finger)])
                  for finger in RAIL_FINGERS]
        for finger, flag in zip(RAIL_FINGERS, railed):
            if flag:
                self.railed[(hand, finger)] += 1
        self._rows[hand].append((group, g, c, railed))
        self._arrays.pop(hand, None)
        self._models = {k: v for k, v in self._models.items() if k[0] != hand}
        return True

    def n_frames(self, hand: str) -> int:
        return len(self._rows.get(str(hand), ()))

    def hands(self) -> List[str]:
        return sorted(self._rows)

    def _arrays_for(self, hand: str):
        if hand not in self._arrays:
            rows = self._rows.get(hand, [])
            groups = [r[0] for r in rows]
            G = np.array([r[1] for r in rows], dtype=float).reshape(-1, 5)
            C = np.array([r[2] for r in rows], dtype=float).reshape(-1, 4)
            R = np.array([r[3] for r in rows], dtype=bool).reshape(-1, 4)
            self._arrays[hand] = (groups, G, C, R)
        return self._arrays[hand]

    @property
    def cross_session(self) -> bool:
        return self.fitted_on is not None

    def model(self, hand: str, exclude: object = None) -> RecalModel:
        """The model fitted on every trusted frame of `hand` outside group
        `exclude` (None: all of them). Cached. Cross-session, the other
        session's model fitted on all of its frames, whatever `exclude`."""
        hand = str(hand)
        if self.fitted_on is not None:
            return self.fitted_on.model(hand)
        key = (hand, exclude)
        if key in self._models:
            return self._models[key]
        groups, G, C, R = self._arrays_for(hand)
        keep = np.array([exclude is None or g != exclude for g in groups],
                        dtype=bool)
        n = int(keep.sum())
        out = RecalModel(hand=hand, variant=self.params.variant, n_frames=n)
        if n < self.params.min_frames:
            out.refused = (f"{n} trusted frames of the {hand} hand to fit on, "
                           f"need {self.params.min_frames}")
            self._models[key] = out
            return out
        Gk, Ck, Rk = G[keep], C[keep], R[keep]
        usable = [j for j in range(5) if np.all(np.isfinite(Gk[:, j]))]
        out.inputs = tuple(FINGER_NAMES[j] for j in usable)
        lam = float(self.params.ridge_lambda)
        for i, finger in enumerate(RAIL_FINGERS):
            own = FINGER_NAMES.index(finger)
            y = Ck[:, i]
            # the rail rule: this finger's own railed readings do not teach
            rows = np.isfinite(y) & ~Rk[:, i]
            if own not in usable or rows.sum() < self.params.min_frames:
                continue
            cols = [own] if self.params.variant == RECAL_OWN else usable
            X = Gk[rows][:, cols]
            yy = y[rows]
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[sd < 1e-9] = 1.0
            Z = (X - mu) / sd
            m = len(yy)
            beta = np.linalg.solve(Z.T @ Z / m + lam * np.eye(len(cols)),
                                   Z.T @ (yy - yy.mean()) / m)
            b = beta / sd
            bias = float(yy.mean() - float(b @ mu))
            out.coef[finger] = (bias, {FINGER_NAMES[j]: float(w)
                                       for j, w in zip(cols, b)})
        if not out.coef:
            out.refused = f"no finger of the {hand} hand could be fitted"
        self._models[key] = out
        return out

    def evaluate(self, pose_of: Mapping[object, str]) -> None:
        """Held-out residuals, per (hand, pose, finger), before and after.

        For every group k, the rows of k are predicted by the model fitted
        WITHOUT k (cross-session: by the other session's model, which never
        saw any of them). Before: |camera fraction - glove fraction|, the
        glove's clipped as `HandScale.fraction` clips the camera's. After:
        |camera fraction - prediction clipped to [0, 1]|, which is the
        target the finger is bent to. A refused model corrects nothing, and
        a finger on its own rail is not corrected (the rail rule), so for
        both the after is the before.
        """
        self.residuals = {}
        for hand in self.hands():
            groups, G, C, R = self._arrays_for(hand)
            for k in sorted(set(groups), key=str):
                mdl = self.model(hand, exclude=k)
                pose = pose_of.get(k, "?")
                for r in (j for j, g in enumerate(groups) if g == k):
                    fr = dict(zip(FINGER_NAMES, G[r]))
                    for i, finger in enumerate(RAIL_FINGERS):
                        cam = C[r, i]
                        glove = fr[finger]
                        if not (np.isfinite(cam) and np.isfinite(glove)):
                            continue
                        before = abs(cam - min(FRAC_MAX, max(FRAC_MIN, glove)))
                        pred = (None if mdl.refused or R[r, i]
                                else mdl.predict(finger, fr))
                        after = (before if pred is None
                                 else abs(cam - min(1.0, max(0.0, pred))))
                        slot = self.residuals.setdefault(
                            (hand, pose, finger), ([], []))
                        slot[0].append(float(before))
                        slot[1].append(float(after))

    def apply(self, hand: str, exclude: object,
              glove_pts: Sequence[Sequence[float]],
              glove_curls: Sequence[float],
              scale: Optional[HandScale],
              skip: Sequence[str] = ()) -> Tuple[np.ndarray, Dict[str, float]]:
        """Correct one glove frame with the model fitted WITHOUT `exclude`.

        Each finger with a fitted row is bent so its glove curl reads the
        predicted camera fraction, clipped to [0, 1], on the GLOVE's
        endpoints. A finger in `skip` (the ones the rail override owns this
        frame) is never touched, and neither is one sitting on its own rail
        (the rail rule). Returns the points and the curl change per finger
        moved.
        """
        hand = str(hand)
        pts = np.asarray(glove_pts, dtype=float)
        mdl = self.model(hand, exclude=exclude)
        if scale is None or mdl.refused:
            return pts, {}
        fr = {}
        for finger in FINGER_NAMES:
            ends = scale.endpoints(SENSOR_GLOVE, finger)
            fr[finger] = _glove_fraction(
                ends, glove_curls[FINGER_NAMES.index(finger)])
        out = None
        moved: Dict[str, float] = {}
        for finger in RAIL_FINGERS:
            if finger in skip or finger not in mdl.coef:
                continue
            if self.on_rail(hand, finger,
                            glove_curls[FINGER_NAMES.index(finger)]):
                continue
            _bias, weights = mdl.coef[finger]
            if not all(np.isfinite(fr[f]) for f in weights):
                continue
            ends = scale.endpoints(SENSOR_GLOVE, finger)
            pred = min(1.0, max(0.0, mdl.predict(finger, fr)))
            curl = float(glove_curls[FINGER_NAMES.index(finger)])
            target = float(ends.open) - pred * ends.span
            if abs(target - curl) <= 1e-12:
                continue
            out = bend_finger_to_curl(pts if out is None else out, finger,
                                      target)
            tip = FINGER_CHAINS[finger][-1]
            palm = float(np.linalg.norm(out[MIDDLE_MCP] - out[WRIST]))
            change = float(np.linalg.norm(out[tip] - out[WRIST])) / palm - curl
            moved[finger] = change
            key = (hand, finger)
            if abs(change) > 1e-12:
                self.corrected[key] += 1
                self.sum_abs[key] += abs(change)
                self.max_abs[key] = max(self.max_abs[key], abs(change))
        return (pts if out is None else out), moved


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


def _lag_of(glove_lag, hand: Optional[str]) -> float:
    """`glove_lag` for this hand: one number, or one per hand side."""
    if isinstance(glove_lag, Mapping):
        return float(glove_lag.get(str(hand).strip().lower(), 0.0))
    return float(glove_lag or 0.0)


def pair_by_time(glove: Sequence[dict], cam: Sequence[dict],
                 max_dt: float = 0.05,
                 clock: str = AUTO,
                 glove_lag: Union[float, Mapping[str, float]] = 0.0
                 ) -> List[Tuple[dict, Optional[dict]]]:
    """Match camera frames to glove frames by a shared clock, per hand.

    Each glove frame takes the nearest camera frame of the SAME hand within
    max_dt seconds; frames with no partner pair with None and stay
    glove-only. `clock` is "auto" (ask `pairing_clock`) or a key to force —
    see `pairing_clock` for why the choice matters.

    `glove_lag` is SECONDS the glove's solved hand TRAILS the camera, and it
    shifts the glove's pairing stamp back by that much: a glove frame stamped
    t describes the hand at t - lag, so t - lag is the instant a camera frame
    has to be near. Nothing is resampled and no stamp is rewritten — the
    shift exists only inside the match, and the rows handed back are the rows
    that were passed in.

    It is measured, not assumed: `estimate_glove_lag`. On this laptop the
    glove's solved hand trails the camera by roughly 100 ms on the left hand
    and 450-485 ms on the right. For a HELD pose that does not matter — both
    sensors are describing a hand that is not moving, and day 1's static takes
    move no per-take median by more than 0.01 whether the shift is applied or
    not. For a MOVING hand it is the difference between pairing two views of
    one instant and pairing two different instants.

    One number is one lag for every hand in these rows; a mapping
    (`{"left": 0.10, "right": 0.46}`) is per hand side, because the two
    gloves are separate garments on separate stretch sensors and do not
    answer at the same speed. `max_dt` is unchanged and still measured on the
    SHIFTED stamps, so a lag larger than the real one loses pairs rather than
    quietly matching further apart.
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
        # The glove's stamp is when the SOLVED hand arrived; the hand it
        # describes was in that shape `lag` seconds earlier.
        t_g = _stamp(g, clock) - _lag_of(glove_lag, g.get("hand_side"))
        candidates = by_hand.get(g["hand_side"], [])
        best, best_dt = None, max_dt
        # linear scan is fine: takes are seconds long, not hours
        for c in candidates:
            dt = abs(_stamp(c, clock) - t_g)
            if dt <= best_dt:
                best, best_dt = c, dt
        out.append((g, best))
    return out


# --- how far the glove trails the camera -------------------------------

# The camera's index curl has to move at least this much over a clip before a
# lag measured on it means anything. A cross-correlation of two flat lines has
# a maximum and it is noise; on a held pose that is exactly what both traces
# are. 0.1 of a palm length is well under the ~0.9 an open-to-fist transition
# covers and well over the 0.01-0.03 a held pose wanders by.
#
# It is required of the standard deviation AND of the median absolute
# deviation, and the second is the one that does the work. Measured on
# sync_day1: `open_palm_right_take3` is flat at 1.77 for five seconds with one
# 0.4 s tracking dip to 1.05 in the middle, and that single dip gives it a
# std of 0.230 — over any threshold a real transition would need — while its
# MAD is 0.005. A robust spread is the difference between "this clip is a
# transition" and "this clip is a held pose with one mistracked patch in it",
# and the same lesson is already why `diagnostics.camera_range` measures
# between percentiles instead of min to max. For a ramp the two agree to
# within 15 % (std = span/sqrt(12), MAD = span/4), so one threshold serves.
MIN_LAG_CAM_STD = 0.1
# ...and the shift has to EXPLAIN both traces. The correlation is the measure
# of that, so it is the thing to put a floor under: on sync_day1's
# `thumbs_up_right_take1` the camera steps from 0.89 to 1.7 halfway through
# and stays (the known edge-on thumbs_up failure) while the glove holds 0.7,
# which is a sustained camera move by any spread measure and correlates at
# -0.43. A planted lag on two views of one hand correlates above 0.95.
MIN_LAG_CORRELATION = 0.8

# The two signals a lag can be read off, in the order they are tried. Both are
# `features.flexion_features` of the two skeletons, so they are the same
# quantity the reports print.
LAG_SIGNAL_INDEX = "index curl"
LAG_SIGNAL_MEDIAN = "median index..pinky curl"

NOT_MEASURABLE = "not measurable"


@dataclass(frozen=True)
class GloveLagEstimate:
    """How far one hand's glove trails the camera over one clip.

    `seconds` is positive when the GLOVE trails, which is the direction
    `pair_by_time`'s `glove_lag` expects. `trusted` is the only field a caller
    should branch on: an untrusted estimate carries whatever the
    cross-correlation happened to return and `why` says why it is not to be
    believed, because a number with no reason beside it invites being used.
    """

    seconds: float = 0.0
    correlation: float = 0.0
    trusted: bool = False
    cam_std: float = 0.0
    cam_spread: float = 0.0
    n_glove: int = 0
    n_cam: int = 0
    signal: str = ""
    why: str = ""

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0

    def described(self) -> str:
        if not self.trusted:
            return self.why or NOT_MEASURABLE
        return (f"{self.ms:+.0f} ms  (r {self.correlation:.2f}, "
                f"camera index curl std {self.cam_std:.2f} / spread "
                f"{self.cam_spread:.2f}, {self.signal})")


def _curl_series(rows: Sequence[dict], clock: str
                 ) -> Tuple[List[float], List[List[float]]]:
    """(times, curls-per-frame) for one hand's rows, in time order."""
    ordered = sorted(rows, key=lambda d: _stamp(d, clock))
    times = [_stamp(d, clock) for d in ordered]
    curls = [list(flexion_features(np.asarray(d["pts"], float)))
             for d in ordered]
    return times, curls


def estimate_glove_lag(glove_rows: Sequence[dict], cam_rows: Sequence[dict],
                       clock: str = AUTO,
                       min_cam_std: float = MIN_LAG_CAM_STD,
                       min_correlation: float = MIN_LAG_CORRELATION,
                       hand: Optional[str] = None) -> GloveLagEstimate:
    """Cross-correlate the two curl traces: how far does the glove trail?

    `glove_rows` and `cam_rows` are the row dicts the fusion loaders produce
    (`pts`, `hand_side`, and the clocks) for ONE hand — pass `hand` to have
    them filtered here instead. The arithmetic is
    `leap_hand.diagnostics.estimate_lag`, unchanged and not reimplemented:
    both traces are resampled onto one grid, z-scored over each candidate
    overlap and matched on SHAPE, so the glove's different range and its
    offset cannot influence the answer. That function is also what
    `scripts/leap/finger_sweep.py` reports a per-finger lag with, and two
    numbers that will be compared have to come from one implementation.

    Imported inside the call on purpose: `leap_hand.protocol` imports this
    module, so a module-level import here would be a cycle.

    TRUST IS DECIDED FIRST ON THE CAMERA
      A held pose gives two flat traces whose best shift is noise, so the
      first test is whether the HAND MOVED, measured on the camera's index
      curl — the sensor that is not the one under suspicion. Both its standard
      deviation and its median absolute deviation over the clip must exceed
      `min_cam_std`; see that constant for why the robust one is the one that
      matters. Then the shift has to EXPLAIN the two traces
      (`min_correlation`), and its peak has to lie strictly inside the search
      window — a best lag sitting on `diagnostics.LAG_MIN` is a wall, not a
      maximum. An estimate failing any of these comes back untrusted with
      "not measurable" on it and the caller applies nothing.

    TWO SIGNALS, THE BETTER FIT WINS
      The index curl alone is the finger a close/open transition moves most.
      The median over index..pinky is four stretch sensors instead of one, so
      a single mistracked finger cannot carry it. Both are tried and the
      higher correlation is kept, with `signal` saying which — the correlation
      IS the measure of how well a shift explains the two traces, so choosing
      on it is choosing the better-explained alignment rather than the more
      convenient number.
    """
    from leap_hand.diagnostics import (LAG_MAX, LAG_MIN,  # cycle: see docstring
                                       estimate_lag)

    if hand is not None:
        side = str(hand).strip().lower()
        glove_rows = [r for r in glove_rows
                      if str(r.get("hand_side", "")).lower() == side]
        cam_rows = [r for r in cam_rows
                    if str(r.get("hand_side", "")).lower() == side]
    blank = GloveLagEstimate(n_glove=len(glove_rows), n_cam=len(cam_rows))
    if len(glove_rows) < 2 or len(cam_rows) < 2:
        return replace(blank, why=f"{NOT_MEASURABLE}: too few frames "
                                  f"({len(glove_rows)} glove, {len(cam_rows)} "
                                  "camera)")
    if clock == AUTO:
        clock = pairing_clock(glove_rows, cam_rows)

    gt, g_curls = _curl_series(glove_rows, clock)
    ct, c_curls = _curl_series(cam_rows, clock)
    i_index = FINGER_NAMES.index("index")
    cam_index = np.asarray([row[i_index] for row in c_curls], float)
    cam_std = float(cam_index.std())
    # The robust twin of that std: half the clip is within this of the middle.
    cam_spread = float(np.median(np.abs(cam_index - np.median(cam_index))))
    blank = replace(blank, cam_std=cam_std, cam_spread=cam_spread,
                    n_glove=len(gt), n_cam=len(ct))
    if cam_std <= min_cam_std or cam_spread <= min_cam_std:
        return replace(blank, why=(
            f"{NOT_MEASURABLE}: the camera's index curl moved by "
            f"{cam_std:.3f} (spread {cam_spread:.3f}) over this clip, need "
            f"both over {min_cam_std:.2f} — a held pose has no transition to "
            "align on"))

    four = [FINGER_NAMES.index(f) for f in RAIL_FINGERS]

    def series(rows, which):
        if which == LAG_SIGNAL_INDEX:
            return [row[i_index] for row in rows]
        return [median([row[i] for i in four]) for row in rows]

    best = None
    for which in (LAG_SIGNAL_INDEX, LAG_SIGNAL_MEDIAN):
        got = estimate_lag(gt, series(g_curls, which),
                           ct, series(c_curls, which))
        if got is None:
            continue
        if best is None or got.correlation > best[0].correlation:
            best = (got, which)
    if best is None:
        return replace(blank, why=(
            f"{NOT_MEASURABLE}: neither curl trace moved enough for a shift "
            "to be searched over — the glove is pinned on its rail, or the "
            "two takes overlap for too little time"))
    got, which = best
    found = replace(blank, seconds=float(got.seconds),
                    correlation=float(got.correlation), signal=which)
    if got.correlation < min_correlation:
        return replace(found, why=(
            f"{NOT_MEASURABLE}: the best shift ({found.ms:+.0f} ms) explains "
            f"the two traces at r {got.correlation:.2f}, under "
            f"{min_correlation:.2f} — they are not two views of one moving "
            "hand"))
    if not (LAG_MIN < got.seconds < LAG_MAX):
        return replace(found, why=(
            f"{NOT_MEASURABLE}: the best shift is {found.ms:+.0f} ms, at the "
            f"edge of the {LAG_MIN * 1000:+.0f}..{LAG_MAX * 1000:+.0f} ms "
            "search window — that is a wall, not a peak"))
    return replace(found, trusted=True)
