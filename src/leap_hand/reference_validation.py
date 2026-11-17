"""Independent reference measurements for the six Paper 3 coordinates the
tracker does not give, and the comparison that evaluates the tracker against
them.

WHAT THIS IS FOR
  Six of Paper 3's 24 coordinates are unavailable from the camera
  (angle_coverage.UNAVAILABLE_COLUMNS): the thumb TMC pair (T_TMC_fe,
  T_TMC_aa) and the four finger CMC (I_CMC .. L_CMC). Software cannot make
  them up. This module is the path real measurements take: a documented file
  format for an INDEPENDENT reference device (FORMAT, `load_reference`), the
  tracker session recorded at the same time (`load_tracker_session`, written
  by scripts/leap/record_validation.py), time alignment (`build_clock`),
  sample matching that never crosses a tracking discontinuity
  (`match_samples`), a mapping fitted on calibration trials only
  (`fit_mapping`), the held-out comparison (`compare`), and the scope a
  result is valid in (`TmcMapping`, `scope_mismatch`, `apply_mapping`).
  The specification of each coordinate is docs/measurement_specification.md;
  the format and the comparison are docs/reference_validation.md; the live
  procedure is docs/validation_capture_protocol.md.

WHAT IT IS NOT
  * Not a measurement by itself. Without a reference capture recorded by a
    device that is not the tracker, nothing here produces a validated value.
    A synthetic capture (data_origin "synthetic", the `example` generator
    below) exercises the code and is labelled so everywhere: its report and
    mapping carry status "synthetic_demonstration", never "evaluated".
  * A second calculation from the tracker's own skeleton is a software
    cross-check (data_origin and role "software_cross_check"), not a
    reference; its results carry status "cross_check".
  * No accuracy threshold is applied. No sourced or agreed angular accuracy
    threshold exists for these coordinates; Paper 3's coupling ratios
    (Table 6, eq. 3-8) are model assumptions and its fingertip errors (Table
    8: 7.67 % / 5.90 % for 9 DoF) are position errors of a reconstruction,
    neither is an angular-accuracy threshold. The report gives the numbers
    and says which conditions below are identifiability conditions chosen
    for this software (IDENTIFIABILITY_NOTE), not accuracy limits.

CANONICAL CONVENTION (every angle compared is converted to it on import)
  For a joint, the proximal segment's frame at the declared zero posture:
    e3  along the distal bone (metacarpal) toward its head,
    e2  the palmar normal (pointing out of the palm),
    e1  = e2 x e3: toward the ulnar side on a right hand, the radial side on
        a left hand (both frames right-handed, so one proper rotation maps
        the tracker's palm frame onto either).
  Thumb TMC (Paper 3 Table 1, its D-H chain Table 3; anatomical TMC motion):
    flexion_extension     about e2, in the plane of the palm; POSITIVE =
                          the metacarpal head toward the ulnar side, across
                          the palm (flexion); radial = extension.
    abduction_adduction   about e1, out of the plane of the palm; POSITIVE =
                          the head palmar (palmar abduction); dorsal = adduction.
    order                 "flexion_then_abduction": R = Ry(s fe) Rx(-aa)
                          (flexion about the fixed proximal axis, then about
                          the moved one: the ISB joint coordinate system's
                          order, Wu et al. 2005, 4.4.1); "abduction_then_flexion":
                          R = Rx(-aa) Ry(s fe) (Paper 3's D-H order, Table 3).
                          s = +1 right hand, -1 left. Direction u = R e3.
  Finger CMC (Paper 3 Table 2 joint 1, the palm arc): flexion_extension of
    the metacarpal against the carpus about e1; POSITIVE = the head palmar
    (the palm arching). One angle; this tracker reports no CMC motion, so
    its reading (the metacarpal's flexion against the palm,
    joint_frames.joint_angles) is compared as it is, never mapped.
  A reference that declares the opposite positive direction has its sign
  flipped on import; one that declares neither word, or no rotation order
  for the thumb, is refused (ambiguous).

THE MAPPING (thumb TMC only, `fit_mapping`)
  The tracker gives the thumb metacarpal against its rigid palm, R_pm =
  R_palm^T R_metacarpal (tmc_estimate.palm_to_metacarpal). The palm stands
  for the carpus that holds the trapezium (assumption A5 of tmc_estimate),
  and LeapC's metacarpal axis is a model axis, not a palpated one. So the
  reference direction is modelled as d = A R_pm b: A a constant rotation from
  the tracker's palm frame to the reference's proximal (trapezium) frame, b
  the metacarpal's anatomical long axis in the tracker's metacarpal frame
  (start (0, 0, -1): LeapC's -z runs base to head). Both are fitted on the
  calibration trials' samples by alternating least squares (A: the SVD
  solution of Wahba's problem; b: the normalised mean of R_pm^T A^T d), which
  never increases the summed squared chord. Five parameters; a rigid
  rotation cannot absorb a gain or a cross-talk, so whatever of those the
  tracker has shows in the held-out errors, which is what they are for.
  A fit needs movement along two independent directions (an isolated
  flexion and an isolated abduction); one posture or one movement plane
  leaves A or b undetermined and is refused (IDENTIFIABILITY_NOTE). An
  open-hand or neutral-posture calibration is exactly such a single
  posture: it fixes a zero, not the frame.

THE FINGER CMC
  LeapC's palm is one rigid frame: its finger metacarpals do not move
  against it (`observability`, measured on every frame of the dataset). A
  CMC column of a reference capture is imported and compared for the record,
  the tracker's motion against it is measured, and when the tracker does not
  move the verdict is "unobservable" and nothing is fitted: no calibration
  recovers motion the tracker does not report.

Pure apart from the loaders (which read files) and `write_report` /
`write_example` (which write into the folder they are given).
"""
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES
from xr_hand.kinematics import PARENT, absolute_to_relative, mat3_to_quat

from . import replay_core as rc
from . import tmc_estimate as te

FORMAT = "hand_reference_v1"
MAPPING_FORMAT = "hand_reference_mapping_v1"
REPORT_FORMAT = "hand_reference_report_v1"
SESSION_FORMAT = "hand_validation_session_v1"
HEADER_FILE = "reference.json"
MANIFEST_FILE = "validation_session.json"
META_SUFFIX = ".meta.json"

DATA_ORIGINS = ("measured", "synthetic", "software_cross_check")
ROLES = ("anatomical_reference", "independent_sensor", "software_cross_check")
DEVICE_KINDS = ("optical_motion_capture", "electromagnetic_tracker", "goniometer",
                "inertial_sensors", "instrumented_glove", "other")
SYNC_METHODS = ("shared_events", "hardware_trigger", "same_clock", "per_take")
ANGLE_UNITS = {"deg": 1.0, "rad": 180.0 / math.pi}
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
ORDERS = ("flexion_then_abduction", "abduction_then_flexion")
ZERO_IDS = ("isb_neutral", "clinical_zero", "device_static_trial", "other")
SAMPLE_COLUMNS = ("sample_id", "trial", "movement", "time_s", "time_end_s",
                  "tracker_file", "valid", "exclusion_reason")

PALM = JOINT_INDEX["PALM"]
THUMB_TIP = JOINT_INDEX["THUMB_TIP"]
INDEX_TIP = JOINT_INDEX["INDEX_TIP"]
HEAD = np.array([0.0, 0.0, -1.0])        # LeapC: a bone's -z runs from base to head

# Conditions chosen for this software (not accuracy thresholds; see the
# module docstring and docs/reference_validation.md).
MIN_SPREAD_DEG = 5.0          # RMS spread of the fit's directions along their 2nd axis
UNOBSERVABLE_DEG = 0.1        # a bone whose orientation against the palm moves less
MIN_HOLD_FRAMES = 10          # tracker frames a hold sample averages over
MAX_DRIFT = 0.01              # |clock scale - 1| beyond which events are mispaired
TAP_CLOSE_MM = 25.0           # thumb tip to index tip: a tap closes below this
TAP_OPEN_MM = 45.0            # ... and is over once the tips part beyond this
IDENTIFIABILITY_NOTE = (
    f"identifiability conditions chosen for this software, not accuracy thresholds: "
    f"the calibration samples' reference directions and the tracker's metacarpal "
    f"rotations must each spread at least {MIN_SPREAD_DEG:g} deg RMS along their "
    f"second principal axis (two independent movement directions); a bone is "
    f"unobservable when no frame's orientation against the palm leaves its mean by "
    f"{UNOBSERVABLE_DEG:g} deg; a hold averages at least {MIN_HOLD_FRAMES} tracker "
    f"frames; clock events implying a scale off 1 by more than {MAX_DRIFT:g} are "
    f"mispaired; a sync tap closes below {TAP_CLOSE_MM:g} mm between the thumb and "
    f"index tips and ends above {TAP_OPEN_MM:g} mm")
ACCURACY_THRESHOLD_NOTE = (
    "no accuracy threshold applied: none is sourced or agreed for these coordinates. "
    "Paper 3's coupling relations (Table 6, eq. 3-8) are model assumptions and its "
    "fingertip errors (Table 8) are reconstruction position errors, not angular "
    "accuracy thresholds; a threshold for accepting a mapping awaits agreement")

STATUSES = ("evaluated", "synthetic_demonstration", "cross_check")
STATUS_TEXT = {
    "evaluated": "evaluated against an independent anatomical reference",
    "synthetic_demonstration": "SYNTHETIC demonstration: exercises the software, measures nothing",
    "cross_check": "cross-check against a device that is not an anatomical reference"}
MAPPED_COLUMNS = ("T_TMC_fe_mapped", "T_TMC_aa_mapped")
MAPPED_STATUS_COLUMN = "T_TMC_mapped_status"
MAPPING_ID_COLUMN = "T_TMC_mapping"
ROW_STATUSES = ("evaluated_in_range", "synthetic_demonstration", "cross_check",
                "outside_tested_range")


class ReferenceError(ValueError):
    """A reference capture, tracker session or mapping that cannot be used;
    the message says what to change."""


# --- the joints ---------------------------------------------------------------------
@dataclass(frozen=True)
class JointSpec:
    joint: str
    bone: str                                   # the tracker bone (JOINT_NAMES)
    movements: Tuple[str, ...]
    positive: Dict[str, str]                    # canonical positive direction
    negative: Dict[str, str]
    paper3: Dict[str, str]                      # movement -> Paper 3 name


def _cmc(letter: str, finger: str) -> JointSpec:
    return JointSpec(f"{finger.lower()}_cmc", f"{finger}_METACARPAL",
                     ("flexion_extension",), {"flexion_extension": "palmar"},
                     {"flexion_extension": "dorsal"},
                     {"flexion_extension": f"{letter}_CMC"})


JOINTS: Dict[str, JointSpec] = {
    "thumb_tmc": JointSpec(
        "thumb_tmc", "THUMB_METACARPAL", ("flexion_extension", "abduction_adduction"),
        {"flexion_extension": "ulnar", "abduction_adduction": "palmar"},
        {"flexion_extension": "radial", "abduction_adduction": "dorsal"},
        {"flexion_extension": "T_TMC_fe", "abduction_adduction": "T_TMC_aa"}),
    **{s.joint: s for s in (_cmc("I", "INDEX"), _cmc("M", "MIDDLE"),
                            _cmc("R", "RING"), _cmc("L", "LITTLE"))},
}
CMC_JOINTS = tuple(j for j in JOINTS if j != "thumb_tmc")
SHORT = {"flexion_extension": "fe", "abduction_adduction": "aa"}


# --- the capture protocol (docs/validation_capture_protocol.md) --------------------------
@dataclass(frozen=True)
class CaptureItem:
    id: str
    joints: Tuple[str, ...]          # the joints its reference samples are compared on
    isolated: Optional[str]          # the coordinate an isolated movement moves
    duration_s: float
    instruction: str


PROTOCOL = "tmc_cmc_validation_v1"
TRIALS = 3                           # rounds of the trial items
SYNC_TAPS = 3
NEUTRAL_TEXT = ("the tmc_neutral posture: hand flat and open, palm to the camera, "
                "fingers straight and together, the thumb straight against the side of "
                "the index finger in the plane of the palm, nothing pressed "
                "(docs/tmc_calibration_protocol.md)")
SYNC_ITEMS = (
    CaptureItem("sync_start", (), None, 8.0,
                f"tap the thumb tip on the index fingertip {SYNC_TAPS} times, about one "
                "second apart, the fingers spread wide between taps; then hold the hand "
                "open and still"),
    CaptureItem("sync_end", (), None, 8.0,
                f"the same {SYNC_TAPS} thumb-index taps as at the start, then the hand "
                "open and still"))
TRIAL_ITEMS = (
    CaptureItem("tmc_neutral_hold", ("thumb_tmc",), None, 5.0,
                f"hold {NEUTRAL_TEXT}; keep it still"),
    CaptureItem("tmc_flexion_sweep", ("thumb_tmc",), "flexion_extension", 10.0,
                "keep the thumb in the plane of the palm, straight; from beside the "
                "index finger move it out to the side (extension, radial) as far as is "
                "comfortable, then across the palm toward the base of the little finger "
                "(flexion), and back; twice, slowly (about 2 s each way); do not lift it "
                "out of the plane of the palm"),
    CaptureItem("tmc_abduction_sweep", ("thumb_tmc",), "abduction_adduction", 10.0,
                "from beside the index finger lift the straight thumb away from the palm, "
                "at a right angle to it (palmar abduction), as far as is comfortable, and "
                "back to the index; twice, slowly; do not move it sideways in the plane "
                "of the palm"),
    CaptureItem("tmc_hold_extended", ("thumb_tmc",), None, 5.0,
                "hold the thumb out to the side in the plane of the palm (extension), "
                "about halfway to its limit, still"),
    CaptureItem("tmc_hold_flexed", ("thumb_tmc",), None, 5.0,
                "hold the thumb across the palm toward the base of the little finger "
                "(flexion), about halfway, still, not touching the palm"),
    CaptureItem("tmc_hold_abducted", ("thumb_tmc",), None, 5.0,
                "hold the thumb lifted away from the palm at a right angle to it (palmar "
                "abduction), about halfway, still"),
    CaptureItem("cmc_palm_arc", CMC_JOINTS, "flexion_extension", 8.0,
                "cup the palm: bring the bases of the thumb and the little finger toward "
                "each other so the palm arches, the fingers straight; then flatten the "
                "palm again; twice, slowly"),
)
CAPTURE_ITEMS = {i.id: i for i in SYNC_ITEMS + TRIAL_ITEMS}


def capture_sequence(trials: int = TRIALS) -> List[Tuple[CaptureItem, int]]:
    """(item, trial) in recording order: the start taps, then `trials` rounds of
    every trial item (so each trial holds both isolated sweeps, which a fit by
    trial needs), then the end taps."""
    seq = [(SYNC_ITEMS[0], 1)]
    for k in range(1, trials + 1):
        seq += [(item, k) for item in TRIAL_ITEMS]
    return seq + [(SYNC_ITEMS[1], 1)]


# --- canonical angles <-> directions ---------------------------------------------------
def _s(hand: str) -> float:
    if hand not in rc.SIDES:
        raise ValueError(f"hand must be one of {rc.SIDES}, got {hand!r}")
    return 1.0 if hand == "right" else -1.0


def tmc_rotation(fe: float, aa: float, order: str, hand: str) -> np.ndarray:
    """The canonical rotation of (fe, aa) degrees in `order` (module docstring)."""
    s = _s(hand)
    if order == "flexion_then_abduction":
        return jf.rot_y(s * fe) @ jf.rot_x(-aa)
    if order == "abduction_then_flexion":
        return jf.rot_x(-aa) @ jf.rot_y(s * fe)
    raise ValueError(f"rotation order must be one of {ORDERS}, got {order!r}")


def tmc_direction(fe: float, aa: float, order: str, hand: str) -> np.ndarray:
    """u = R e3: the metacarpal's head direction in the canonical frame."""
    return tmc_rotation(fe, aa, order, hand) @ np.array([0.0, 0.0, 1.0])


def tmc_angles(u, order: str, hand: str) -> Tuple[float, float]:
    """(fe, aa) degrees of a direction, the inverse of tmc_direction."""
    s = _s(hand)
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    if order == "flexion_then_abduction":
        aa = math.degrees(math.asin(max(-1.0, min(1.0, u[1]))))
        fe = s * math.degrees(math.atan2(u[0], u[2]))
    elif order == "abduction_then_flexion":
        fe = s * math.degrees(math.asin(max(-1.0, min(1.0, u[0]))))
        aa = math.degrees(math.atan2(u[1], u[2]))
    else:
        raise ValueError(f"rotation order must be one of {ORDERS}, got {order!r}")
    return fe, aa


def wrap180(x):
    """Angle differences into [-180, 180)."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def estimate_direction(r_tmc, hand: str) -> np.ndarray:
    """The unvalidated estimate's metacarpal head direction (R_tmc = R_rest^T
    R_pm, tmc_estimate) in the canonical axes, its zero the calibration's rest
    posture: e1 = -s y_rest, e2 = -s x_rest, e3 = -z_rest (the estimate's
    positive flexion runs toward the metacarpal's palmar side -y, its positive
    abduction toward the thumb side -s x; both pinned by tests/test_tmc_estimate)."""
    s = _s(hand)
    u = np.asarray(r_tmc, dtype=float) @ HEAD
    return np.array([-s * u[1], -s * u[0], -u[2]])


# --- rotations --------------------------------------------------------------------------
def bone_in_palm(rot, bone: str) -> np.ndarray:
    """R_palm^T R_bone from world rotations (26, 3, 3)."""
    rot = np.asarray(rot, dtype=float)
    return rot[PALM].T @ rot[JOINT_INDEX[bone]]


def _log(r) -> np.ndarray:
    """The rotation vector (radians) of a rotation matrix."""
    r = np.asarray(r, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))
    ang = math.acos(c)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if ang < 1e-9:
        return w / 2.0
    if math.pi - ang < 1e-6:                 # near 180: the axis from r + I
        m = (r + np.eye(3)) / 2.0
        axis = m[:, int(np.argmax(np.diag(m)))]
        return ang * axis / np.linalg.norm(axis)
    return ang * w / (2.0 * math.sin(ang))


def rotation_spread(rots) -> Dict[str, float]:
    """How much a set of rotations moves: the largest angle of any of them
    from their chordal mean, and the RMS spread along the two largest
    principal axes of their rotation vectors about it (degrees)."""
    rots = [np.asarray(r, dtype=float) for r in rots]
    mean = te.chordal_mean(rots)
    vec = np.array([_log(mean.T @ r) for r in rots])
    ev = np.sort(np.linalg.eigvalsh(np.cov(vec.T) if len(vec) > 1 else np.zeros((3, 3))))[::-1]
    return {"n": len(rots),
            "max_from_mean_deg": float(max(rc.rotation_angle_deg(mean, r) for r in rots)),
            "rms_axis1_deg": float(math.degrees(math.sqrt(max(ev[0], 0.0)))),
            "rms_axis2_deg": float(math.degrees(math.sqrt(max(ev[1], 0.0))))}


def direction_spread(dirs) -> Dict[str, float]:
    """RMS spread (degrees) of unit vectors along the two largest principal
    axes of their scatter about their mean direction."""
    d = np.asarray(dirs, dtype=float)
    ev = (np.sort(np.linalg.eigvalsh(np.cov(d.T)))[::-1] if len(d) > 1
          else np.zeros(3))
    return {"n": int(len(d)),
            "rms_axis1_deg": float(math.degrees(math.sqrt(max(ev[0], 0.0)))),
            "rms_axis2_deg": float(math.degrees(math.sqrt(max(ev[1], 0.0))))}


def kabsch(v, d) -> np.ndarray:
    """The rotation A minimising sum |A v_i - d_i|^2 (SVD, det +1)."""
    h = np.asarray(v, dtype=float).T @ np.asarray(d, dtype=float)
    u, _, vt = np.linalg.svd(h)
    sign = 1.0 if np.linalg.det(vt.T @ u.T) > 0 else -1.0
    return vt.T @ np.diag([1.0, 1.0, sign]) @ u.T


def _skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _expm(w) -> np.ndarray:
    """Rotation of the rotation vector w (radians), Rodrigues."""
    ang = float(np.linalg.norm(w))
    if ang < 1e-15:
        return np.eye(3) + _skew(w)
    k = _skew(np.asarray(w) / ang)
    return np.eye(3) + math.sin(ang) * k + (1.0 - math.cos(ang)) * k @ k


def _tangent(b) -> np.ndarray:
    """Two unit vectors orthogonal to b and to each other (3, 2)."""
    t = np.cross(b, [1.0, 0.0, 0.0] if abs(b[0]) < 0.9 else [0.0, 1.0, 0.0])
    t /= np.linalg.norm(t)
    return np.column_stack([t, np.cross(b, t)])


def fit_mapping(r_pm, d, iterations: int = 200, tol: float = 1e-13) -> dict:
    """A and b with d_i ~ A R_pm,i b (module docstring), from the calibration
    samples only: Levenberg-Marquardt on the five parameters (a rotation
    vector on A, two tangent components on b), started from the SVD solution
    of Wahba's problem with b = (0, 0, -1). Returns A, b, residuals and each
    parameter's standard deviation (sigma^2 (J^T J)^-1, degrees). Refused
    (ReferenceError) when the samples cannot determine them: fewer than two
    independent movement directions on either side."""
    r_pm = np.asarray(r_pm, dtype=float)
    d = np.asarray(d, dtype=float)
    if len(d) < 3:
        raise ReferenceError(f"{len(d)} calibration samples: a mapping needs at least 3 "
                             "(record an isolated flexion and an isolated abduction)")
    ref_spread = direction_spread(d)
    trk_spread = rotation_spread(r_pm)
    for side, s in (("reference directions", ref_spread),
                    ("tracker metacarpal rotations", trk_spread)):
        if s["rms_axis2_deg"] < MIN_SPREAD_DEG:
            raise ReferenceError(
                f"the calibration samples do not determine the mapping: the {side} "
                f"spread {s['rms_axis1_deg']:.2f} / {s['rms_axis2_deg']:.2f} deg RMS "
                f"along their two principal axes, under {MIN_SPREAD_DEG:g} deg on the "
                "second: one posture or one movement plane. Calibrate on trials that "
                "hold both an isolated flexion and an isolated abduction sweep "
                "(docs/validation_capture_protocol.md); a single reference posture "
                "fixes a zero, not the frame")
    def residual(a, b):
        return (np.einsum("ij,nj->ni", a, np.einsum("nij,j->ni", r_pm, b)) - d).ravel()

    def jacobian(a, b):
        v = np.einsum("nij,j->ni", r_pm, b)                  # R_i b
        t = _tangent(b)
        ja = np.stack([-a @ _skew(x) for x in v])            # d(A exp[w] v)/dw at 0
        jb = np.einsum("ij,njk,kl->nil", a, r_pm, t)         # A R_i T
        return np.concatenate([ja, jb], axis=2).reshape(-1, 5), t

    b = HEAD.copy()
    a = kabsch(np.einsum("nij,j->ni", r_pm, b), d)
    r = residual(a, b)
    cost, lam, it = float(r @ r), 1e-3, 0
    for it in range(1, iterations + 1):
        j, t = jacobian(a, b)
        g, h = j.T @ r, j.T @ j
        step = np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-12), -g)
        a_new = a @ _expm(step[:3])
        b_new = b + t @ step[3:]
        b_new /= np.linalg.norm(b_new)
        r_new = residual(a_new, b_new)
        new = float(r_new @ r_new)
        if new <= cost:
            a, b, r, lam = a_new, b_new, r_new, lam / 10.0
            done = cost - new < tol and float(np.linalg.norm(step)) < 1e-10
            cost = new
            if done:
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                break
    j, _ = jacobian(a, b)
    dof = max(1, j.shape[0] - 5)
    sigma2 = cost / dof
    try:
        cov = sigma2 * np.linalg.inv(j.T @ j)
        sd = np.degrees(np.sqrt(np.clip(np.diag(cov), 0.0, None)))
    except np.linalg.LinAlgError:
        sd = np.full(5, np.inf)
    pred = np.einsum("ij,nj->ni", a, np.einsum("nij,j->ni", r_pm, b))
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(pred, d), axis=1),
                                np.sum(pred * d, axis=1)))
    return {"A": a, "b": b, "iterations": it, "n": int(len(d)),
            "rms_residual_deg": float(math.sqrt(np.mean(ang * ang))),
            "max_residual_deg": float(ang.max()),
            "parameter_sd_deg": {"A_rotation": [float(x) for x in sd[:3]],
                                 "b_tangent": [float(x) for x in sd[3:]]},
            "b_from_tracker_axis_deg": float(math.degrees(math.acos(
                max(-1.0, min(1.0, float(b @ HEAD)))))),
            "reference_spread": ref_spread, "tracker_spread": trk_spread}


# --- files ------------------------------------------------------------------------------
def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _text(d: dict, key: str, where: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ReferenceError(f"{where}: '{key}' is missing or empty")
    return v.strip()


def _num(v) -> Optional[float]:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ReferenceError(f"{v!r} is not a number") from None
    if not math.isfinite(x):
        raise ReferenceError(f"{v!r} is not a finite number")
    return x


def _inside(folder: Path, rel: str, where: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ReferenceError(f"{where}: {rel!r} must be a path relative to the folder, "
                             "inside it")
    return folder / p


# --- the reference capture --------------------------------------------------------------
@dataclass
class Coordinate:
    column: str
    joint: str
    movement: str
    sign: float                  # +1: the file's positive is canonical, -1 flipped


@dataclass
class ReferenceSample:
    sample_id: str
    trial: int
    movement: str
    t0: Optional[float]          # seconds on the reference clock (None: per take)
    t1: Optional[float]          # end of a hold (None: an instant)
    tracker_file: Optional[str]
    valid: bool
    reason: str
    values: Dict[Tuple[str, str], Optional[float]]    # canonical degrees


@dataclass
class ReferenceCapture:
    folder: Path
    header: dict
    header_sha256: str
    data_origin: str
    role: str
    participant: str
    hand: str
    session: str
    device: dict
    calibration: dict
    tracker_calibration: Optional[dict]
    sync: dict
    time_scale: float            # clock units -> seconds
    angle_unit: str
    frames: Dict[str, dict]
    coordinates: List[Coordinate]
    movements: Dict[str, Optional[str]]     # id -> isolated movement (None: holds)
    movement_joints: Dict[str, Tuple[str, ...]]   # id -> the joints it is compared on
    samples: List[ReferenceSample]
    samples_sha256: str
    source_files: List[dict]

    @property
    def joints(self) -> List[str]:
        return sorted({c.joint for c in self.coordinates}, key=list(JOINTS).index)

    @property
    def status(self) -> str:
        if self.data_origin == "synthetic":
            return "synthetic_demonstration"
        if self.data_origin == "measured" and self.role == "anatomical_reference":
            return "evaluated"
        return "cross_check"


def load_reference(folder) -> ReferenceCapture:
    """Read and check a reference capture (FORMAT; docs/reference_validation.md).
    Every refusal is a ReferenceError naming the field and the fix."""
    folder = Path(folder)
    path = folder / HEADER_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ReferenceError(f"cannot read {path}: {e}") from None
    try:
        h = json.loads(text)
    except ValueError as e:
        raise ReferenceError(f"{path} is not JSON: {e}") from None
    w = HEADER_FILE
    if h.get("format") != FORMAT:
        raise ReferenceError(f"{w}: format must be {FORMAT!r}, got {h.get('format')!r}")
    origin, role = h.get("data_origin"), h.get("role")
    if origin not in DATA_ORIGINS:
        raise ReferenceError(f"{w}: data_origin must be one of {DATA_ORIGINS}, got {origin!r}")
    if role not in ROLES:
        raise ReferenceError(f"{w}: role must be one of {ROLES}, got {role!r}")
    device = h.get("device")
    if not isinstance(device, dict):
        raise ReferenceError(f"{w}: 'device' is missing (kind, make, model, serial, "
                             "software, sample_rate_hz)")
    if device.get("kind") not in DEVICE_KINDS:
        raise ReferenceError(f"{w}: device.kind must be one of {DEVICE_KINDS}, got "
                             f"{device.get('kind')!r}")
    for k in ("make", "model"):
        _text(device, k, f"{w} device")
    derived = device.get("derived_from_tracker")
    if not isinstance(derived, bool):
        raise ReferenceError(f"{w}: device.derived_from_tracker must be true or false "
                             "(is any value computed from the tracker's own skeleton?)")
    if (derived or origin == "software_cross_check" or role == "software_cross_check") \
            and not (derived and origin == role == "software_cross_check"):
        raise ReferenceError(
            f"{w}: a value computed from the tracker's own skeleton is a software "
            "cross-check, not an independent reference: set device.derived_from_tracker "
            "true and data_origin and role both to 'software_cross_check' (or, for an "
            "independent device, all three otherwise)")
    if role == "anatomical_reference" and device["kind"] == "instrumented_glove":
        raise ReferenceError(f"{w}: an instrumented glove reports its own model's angles, "
                             "not anatomical segment frames: role 'independent_sensor'")
    participant = _text(h.get("participant") or {}, "id", f"{w} participant")
    hand = h.get("hand")
    if hand not in rc.SIDES:
        raise ReferenceError(f"{w}: hand must be one of {rc.SIDES}, got {hand!r}")
    session = _text(h, "session", w)
    cal = h.get("calibration")
    if not isinstance(cal, dict):
        raise ReferenceError(f"{w}: 'calibration' is missing: say which calibration of "
                             "the reference device the data rest on (id, provenance)")
    _text(cal, "id", f"{w} calibration")
    _text(cal, "provenance", f"{w} calibration")
    if cal.get("hand") not in (None, hand):
        raise ReferenceError(f"{w}: the reference device's calibration is of the "
                             f"{cal.get('hand')} hand, the capture of the {hand} hand")
    tcal = h.get("tracker_calibration")
    if tcal is not None:
        if not isinstance(tcal, dict) or not isinstance(tcal.get("sha256"), str):
            raise ReferenceError(f"{w}: tracker_calibration must give at least the "
                                 "sha256 of the calibration.json it names")
        if tcal.get("hand") not in (None, hand):
            raise ReferenceError(f"{w}: tracker_calibration is of the {tcal.get('hand')} "
                                 f"hand, the capture of the {hand} hand")
    units = h.get("units")
    if not isinstance(units, dict) or "angle" not in units:
        raise ReferenceError(f"{w}: units.angle is missing: say whether the coordinates "
                             f"are in {sorted(ANGLE_UNITS)}")
    if units["angle"] not in ANGLE_UNITS:
        raise ReferenceError(f"{w}: units.angle must be one of {sorted(ANGLE_UNITS)}, "
                             f"got {units['angle']!r}")
    sync = h.get("sync")
    if not isinstance(sync, dict) or sync.get("method") not in SYNC_METHODS:
        raise ReferenceError(f"{w}: sync.method must be one of {SYNC_METHODS}, got "
                             f"{(sync or {}).get('method')!r}")
    time_scale = 1.0
    if sync["method"] != "per_take":
        clock = h.get("clock")
        if not isinstance(clock, dict) or clock.get("units") not in TIME_UNITS:
            raise ReferenceError(f"{w}: clock.units must be one of {sorted(TIME_UNITS)} "
                                 f"(the unit of time_s, time_end_s and the sync events)")
        _text(clock, "name", f"{w} clock")
        time_scale = TIME_UNITS[clock["units"]]
        u = sync.get("uncertainty_s")
        if not isinstance(u, (int, float)) or isinstance(u, bool) or u < 0 \
                or not math.isfinite(u):
            raise ReferenceError(f"{w}: sync.uncertainty_s must be a number >= 0: how far "
                                 "a reference time may be from the tracker time it is "
                                 "matched to")
        if sync["method"] in ("shared_events", "hardware_trigger"):
            ev = sync.get("events")
            if not isinstance(ev, list) or not ev:
                raise ReferenceError(f"{w}: sync.method {sync['method']!r} needs "
                                     "sync.events (at least one, two to measure drift)")
            for e in ev:
                if not isinstance(e, dict) or _num(e.get("reference_time_s")) is None:
                    raise ReferenceError(f"{w}: every sync event needs reference_time_s")
                has_us = isinstance(e.get("tracker_timestamp_us"), int)
                has_tap = (isinstance(e.get("tracker_file"), str)
                           and isinstance(e.get("tap_index"), int) and e["tap_index"] >= 1)
                if has_us == has_tap:
                    raise ReferenceError(f"{w}: sync event {e.get('id')!r} needs either "
                                         "tracker_timestamp_us or tracker_file + tap_index "
                                         "(1-based), not both and not neither")
    frames = h.get("frames")
    if not isinstance(frames, dict):
        raise ReferenceError(f"{w}: 'frames' is missing: the anatomical frame, zero, "
                             "rotation order and sign of every joint")
    coords_in = h.get("coordinates")
    if not isinstance(coords_in, list) or not coords_in:
        raise ReferenceError(f"{w}: 'coordinates' is missing: one entry per column "
                             "(column, joint, movement)")
    coords, seen_col, seen_jm = [], set(), set()
    for c in coords_in:
        col = _text(c, "column", f"{w} coordinate")
        joint, mov = c.get("joint"), c.get("movement")
        if joint not in JOINTS:
            raise ReferenceError(f"{w}: coordinate {col}: joint must be one of "
                                 f"{tuple(JOINTS)}, got {joint!r}")
        spec = JOINTS[joint]
        if mov not in spec.movements:
            raise ReferenceError(f"{w}: coordinate {col}: movement of {joint} must be one "
                                 f"of {spec.movements}, got {mov!r}")
        if col in seen_col or (joint, mov) in seen_jm or col in SAMPLE_COLUMNS:
            raise ReferenceError(f"{w}: coordinate {col} ({joint} {mov}) is given twice or "
                                 "uses a reserved column name")
        seen_col.add(col)
        seen_jm.add((joint, mov))
        coords.append((col, joint, mov))
    for joint in {j for _, j, _ in coords}:
        f = frames.get(joint)
        where = f"{w} frames.{joint}"
        if not isinstance(f, dict):
            raise ReferenceError(f"{where} is missing (proximal/distal segment and frame, "
                                 "zero, rotation_order, positive)")
        for k in ("proximal_segment", "distal_segment", "proximal_frame", "distal_frame"):
            _text(f, k, where)
        zero = f.get("zero")
        if not isinstance(zero, dict) or zero.get("id") not in ZERO_IDS:
            raise ReferenceError(f"{where}.zero.id must be one of {ZERO_IDS}")
        _text(zero, "text", f"{where}.zero")
        if joint == "thumb_tmc" and f.get("rotation_order") not in ORDERS:
            raise ReferenceError(f"{where}.rotation_order must be one of {ORDERS}: the "
                                 "two angles of a TMC direction depend on it (ambiguous "
                                 "without it)")
        pos = f.get("positive")
        if not isinstance(pos, dict):
            raise ReferenceError(f"{where}.positive is missing: the direction each "
                                 "coordinate's positive values move the distal bone")
    if "thumb_tmc" in {j for _, j, _ in coords} and \
            {m for _, j, m in coords if j == "thumb_tmc"} != set(JOINTS["thumb_tmc"].movements):
        raise ReferenceError(f"{w}: a thumb TMC direction needs both its flexion_extension "
                             "and its abduction_adduction column")
    coordinates = []
    for col, joint, mov in coords:
        spec = JOINTS[joint]
        word = frames[joint]["positive"].get(mov)
        if word == spec.positive[mov]:
            sign = 1.0
        elif word == spec.negative[mov]:
            sign = -1.0
        else:
            raise ReferenceError(
                f"{w}: frames.{joint}.positive.{mov} must be {spec.positive[mov]!r} or "
                f"{spec.negative[mov]!r} (the direction positive values move the "
                f"metacarpal head), got {word!r}: an ambiguous sign is not guessed")
        coordinates.append(Coordinate(col, joint, mov, sign))
    movements, movement_joints = {}, {}
    have = {c.joint for c in coordinates}
    for m in h.get("movements") or []:
        mid = _text(m, "id", f"{w} movement")
        joints = m.get("joints")
        if not isinstance(joints, list) or not joints or any(j not in have for j in joints):
            raise ReferenceError(f"{w}: movement {mid}: joints must list the joints it is "
                                 f"compared on, each with a coordinate column here "
                                 f"({sorted(have)}), got {joints!r}")
        iso = m.get("isolated")
        if iso is not None and any(iso not in JOINTS[j].movements for j in joints):
            raise ReferenceError(f"{w}: movement {mid}: isolated must be null or a movement "
                                 f"of every joint it lists, got {iso!r}")
        if mid in movements:
            raise ReferenceError(f"{w}: movement {mid} is given twice")
        movements[mid], movement_joints[mid] = iso, tuple(joints)
    if not movements:
        raise ReferenceError(f"{w}: 'movements' is missing: the movement ids the samples "
                             "name, each with its joint and isolated coordinate")
    sources = h.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ReferenceError(f"{w}: 'source_files' is missing: the reference device's own "
                             "exports these samples were made from, with their sha256")
    for s in sources:
        p = _inside(folder, _text(s, "path", f"{w} source file"), f"{w} source file")
        if not p.is_file():
            raise ReferenceError(f"{w}: source file {s['path']} is not in the folder")
        if sha256_of(p) != s.get("sha256"):
            raise ReferenceError(f"{w}: source file {s['path']} does not match its sha256 "
                                 "(changed since the header was written)")
    sm = h.get("samples")
    if not isinstance(sm, dict):
        raise ReferenceError(f"{w}: 'samples' is missing (file, sha256)")
    spath = _inside(folder, _text(sm, "file", f"{w} samples"), f"{w} samples")
    if not spath.is_file():
        raise ReferenceError(f"{w}: samples file {sm['file']} is not in the folder")
    digest = sha256_of(spath)
    if digest != sm.get("sha256"):
        raise ReferenceError(f"{w}: samples file {sm['file']} does not match its sha256 "
                             "(edited since the header was written); regenerate the "
                             "header from the device export")
    samples = _read_samples(spath, coordinates, movements, ANGLE_UNITS[units["angle"]],
                            sync["method"])
    return ReferenceCapture(
        folder=folder, header=h, header_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        data_origin=origin, role=role, participant=participant, hand=hand,
        session=session, device=device, calibration=cal, tracker_calibration=tcal,
        sync=sync, time_scale=time_scale, angle_unit=units["angle"], frames=frames,
        coordinates=coordinates, movements=movements, movement_joints=movement_joints,
        samples=samples, samples_sha256=digest, source_files=sources)


def _read_samples(path: Path, coordinates, movements, to_deg: float,
                  method: str) -> List[ReferenceSample]:
    name = path.name
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in list(SAMPLE_COLUMNS) + [c.column for c in coordinates]
                   if c not in header]
        if missing:
            raise ReferenceError(f"{name}: columns missing: {', '.join(missing)}")
        out, ids = [], set()
        for line, row in enumerate(reader, start=2):
            where = f"{name} line {line}"
            sid = (row["sample_id"] or "").strip()
            if not sid or sid in ids:
                raise ReferenceError(f"{where}: sample_id empty or repeated ({sid!r})")
            ids.add(sid)
            try:
                trial = int(row["trial"])
            except ValueError:
                raise ReferenceError(f"{where}: trial must be an integer >= 1") from None
            if trial < 1:
                raise ReferenceError(f"{where}: trial must be an integer >= 1")
            mov = (row["movement"] or "").strip()
            if mov and mov not in movements:
                raise ReferenceError(f"{where}: movement {mov!r} is not in the header's "
                                     f"movements {sorted(movements)}")
            if row["valid"] not in ("0", "1"):
                raise ReferenceError(f"{where}: valid must be 1 or 0")
            valid = row["valid"] == "1"
            reason = (row["exclusion_reason"] or "").strip()
            if not valid and not reason:
                raise ReferenceError(f"{where}: an invalid sample needs its exclusion_reason")
            try:
                t0, t1 = _num(row["time_s"]), _num(row["time_end_s"])
                values = {}
                for c in coordinates:
                    v = _num(row[c.column])
                    values[(c.joint, c.movement)] = (None if v is None
                                                     else c.sign * v * to_deg)
            except ReferenceError as e:
                raise ReferenceError(f"{where}: {e}") from None
            tfile = (row["tracker_file"] or "").strip() or None
            if method == "per_take":
                if tfile is None:
                    raise ReferenceError(f"{where}: sync.method per_take: every sample "
                                         "names its tracker_file (the take it was read on)")
            elif t0 is None:
                raise ReferenceError(f"{where}: time_s is empty")
            if t0 is not None and t1 is not None and t1 < t0:
                raise ReferenceError(f"{where}: time_end_s is before time_s")
            out.append(ReferenceSample(sid, trial, mov, t0, t1, tfile, valid, reason,
                                       values))
    if not out:
        raise ReferenceError(f"{name}: no samples")
    return out


# --- the tracker session ------------------------------------------------------------------
@dataclass
class TrackerTake:
    file: str
    item: str
    take: int
    rec: rc.Recording
    t0_us: int
    t1_us: int
    sidecar: dict


@dataclass
class TrackerSession:
    folder: Path
    manifest: dict
    manifest_sha256: str
    participant: str
    hand: str
    session: str
    device: str
    takes: List[TrackerTake]

    def take(self, file: str) -> Optional[TrackerTake]:
        return next((t for t in self.takes if t.file == file), None)


def load_tracker_session(folder) -> TrackerSession:
    """The tracker side of a validation capture: MANIFEST_FILE (written by
    scripts/leap/record_validation.py) and every take it lists, each checked
    against its sha256 and loaded with the manifest's hand (status != 0, every
    hand id; replay_core.load_recording)."""
    folder = Path(folder)
    path = folder / MANIFEST_FILE
    try:
        text = path.read_text(encoding="utf-8")
        m = json.loads(text)
    except (OSError, ValueError) as e:
        raise ReferenceError(f"cannot read the tracker session manifest {path}: {e}") from None
    if m.get("format") != SESSION_FORMAT:
        raise ReferenceError(f"{path}: format must be {SESSION_FORMAT!r}")
    hand = m.get("hand")
    if hand not in rc.SIDES:
        raise ReferenceError(f"{path}: hand must be one of {rc.SIDES}")
    participant = _text(m, "participant", str(path))
    session = _text(m, "session", str(path))
    takes = []
    for t in m.get("takes") or []:
        p = _inside(folder, _text(t, "file", str(path)), str(path))
        if not p.is_file():
            raise ReferenceError(f"{path}: take {t['file']} is not in the folder")
        if sha256_of(p) != t.get("sha256"):
            raise ReferenceError(f"tracker take {t['file']} does not match the manifest's "
                                 "sha256: it changed after the session was recorded")
        meta = {"session": session, "participant": participant, "item": t.get("item")}
        rec = rc.load_recording(p, hand, provenance=f"validation session {session}, "
                                f"{t['file']}", kind="file", name=str(t.get("item")),
                                take=int(t.get("take", 0)), metadata=meta)
        ts = [f.timestamp_us for f in rec.frames]
        takes.append(TrackerTake(t["file"], str(t.get("item")), int(t.get("take", 0)),
                                 rec, int(ts[0] if ts[0] is not None else 0),
                                 int(ts[-1] if ts[-1] is not None else 0),
                                 dict(t)))
    if not takes:
        raise ReferenceError(f"{path}: the manifest lists no takes")
    return TrackerSession(folder, m, hashlib.sha256(text.encode("utf-8")).hexdigest(),
                          participant, hand, session,
                          str(m.get("tracker_device") or "not reported"), takes)


def check_compatible(ref: ReferenceCapture, trk: TrackerSession) -> None:
    """The reference capture and the tracker session must be one capture of one
    hand of one participant (ReferenceError naming what differs)."""
    for what, a, b in (("hand", ref.hand, trk.hand),
                       ("participant", ref.participant, trk.participant),
                       ("session", ref.session, trk.session)):
        if a != b:
            raise ReferenceError(f"the reference capture is of {what} {a!r}, the tracker "
                                 f"session of {what} {b!r}: not one capture")


def check_tracker_calibration(ref: ReferenceCapture, cal: te.Calibration,
                              path=None) -> None:
    """The tracker calibration the unvalidated estimate is compared with: a known
    hand, the capture's hand (tmc_estimate.mismatch), and the file the reference
    header names when it names one."""
    reason = te.mismatch(cal, ref.hand)
    if reason:
        raise ReferenceError(f"tracker calibration not applied: {reason}")
    tc = ref.tracker_calibration
    if tc is not None and path is not None and sha256_of(path) != tc["sha256"]:
        raise ReferenceError(f"the reference header names the tracker calibration "
                             f"{tc.get('file')!r} (sha256 {tc['sha256'][:12]}...), not "
                             f"{path}: compare with the calibration the capture used")


# --- the clock ----------------------------------------------------------------------------
@dataclass
class ClockMap:
    """tracker seconds (timestamp_us / 1e6) = offset_s + scale * reference
    seconds, for every method but per_take (which needs no clock)."""
    method: str
    offset_s: float
    scale: float
    uncertainty_s: float
    events: List[dict]
    residuals_s: List[float]
    note: str

    def tracker_us(self, t_ref_s: float, shift_s: float = 0.0) -> float:
        return (self.offset_s + self.scale * t_ref_s + shift_s) * 1e6

    def as_json(self) -> dict:
        return {**asdict(self), "drift": self.scale - 1.0,
                "residual_rms_s": (float(np.sqrt(np.mean(np.square(self.residuals_s))))
                                   if self.residuals_s else None),
                "residual_max_s": (float(np.max(np.abs(self.residuals_s)))
                                   if self.residuals_s else None)}


def tap_times(rec: rc.Recording) -> List[dict]:
    """Thumb-index taps in a tracker take: every run of frames with the thumb
    and index tips closer than TAP_CLOSE_MM, entered from apart (> TAP_OPEN_MM)
    and left apart again, inside one segment; the tap is the frame of the
    smallest distance (its timestamp_us, frame_id, mm)."""
    seg_of = np.empty(len(rec.frames), dtype=int)
    for k, s in enumerate(rec.segments):
        seg_of[s.start:s.end + 1] = k
    taps, state, best = [], "unknown", None
    for i, f in enumerate(rec.frames):
        pos = rc.reconstruct(f).pos
        d = float(np.linalg.norm(pos[THUMB_TIP] - pos[INDEX_TIP]) * 1000.0)
        if best is not None and seg_of[i] != best[3]:
            best, state = None, "unknown"          # a tap cut by a discontinuity
        if d > TAP_OPEN_MM:
            if best is not None:
                taps.append({"timestamp_us": best[1], "frame_id": best[2],
                             "distance_mm": best[0]})
                best = None
            state = "open"
        elif d < TAP_CLOSE_MM and state in ("open", "closed"):
            state = "closed"
            if best is None or d < best[0]:
                best = (d, f.timestamp_us, f.frame_id, seg_of[i])
    return taps


def build_clock(ref: ReferenceCapture, trk: TrackerSession) -> Optional[ClockMap]:
    """The reference-to-tracker clock (ClockMap), or None for per_take. Events
    with a tracker_file and tap_index are resolved from that take's detected
    taps. With two or more events, offset and scale by least squares; every
    residual must stay within sync.uncertainty_s, and the scale within
    MAX_DRIFT of 1, else the events are refused as mispaired."""
    method = ref.sync["method"]
    if method == "per_take":
        return None
    u = float(ref.sync["uncertainty_s"])
    if method == "same_clock":                 # times reach here in seconds already
        return ClockMap(method, 0.0, 1.0, u, [], [],
                        "reference times are on the tracker's timestamp_us clock")
    events, taps = [], {}
    for e in ref.sync["events"]:
        t_ref = _num(e["reference_time_s"]) * ref.time_scale
        if "tracker_timestamp_us" in e:
            us = int(e["tracker_timestamp_us"])
            how = "given"
        else:
            take = trk.take(e["tracker_file"])
            if take is None:
                raise ReferenceError(f"sync event {e.get('id')!r}: tracker_file "
                                     f"{e['tracker_file']!r} is not a take of the session")
            if take.rec.timeline.source != "timestamp_us":
                raise ReferenceError(f"sync event {e.get('id')!r}: take {take.file} is not "
                                     f"on timestamp_us ({take.rec.timeline.reason})")
            found = taps.setdefault(take.file, tap_times(take.rec))
            k = int(e["tap_index"])
            if k > len(found):
                raise ReferenceError(
                    f"sync event {e.get('id')!r}: tap {k} of {take.file}, but "
                    f"{len(found)} taps were detected there (thumb-index tips under "
                    f"{TAP_CLOSE_MM:g} mm): check the sync take and the pairing")
            us = int(found[k - 1]["timestamp_us"])
            how = f"tap {k} of {take.file} (frame_id {found[k - 1]['frame_id']})"
        events.append({"id": e.get("id"), "reference_s": t_ref, "tracker_us": us,
                       "resolved": how})
    x = np.array([e["reference_s"] for e in events])
    y = np.array([e["tracker_us"] for e in events], dtype=float) / 1e6
    if len(events) == 1:
        scale, offset = 1.0, float(y[0] - x[0])
        note = "one event: offset only, drift not measured"
    else:
        if np.ptp(x) <= 0:
            raise ReferenceError("sync events all at one reference time")
        scale, offset = (float(v) for v in np.polyfit(x, y, 1))
        note = f"{len(events)} events: offset and drift by least squares"
    res = (y - (offset + scale * x)).tolist()
    if abs(scale - 1.0) > MAX_DRIFT:
        raise ReferenceError(f"the sync events imply a clock scale of {scale:.6f} (drift "
                             f"{scale - 1:+.2%}), beyond {MAX_DRIFT:g}: the events are "
                             "probably mispaired")
    worst = max((abs(r) for r in res), default=0.0)
    if worst > u:
        raise ReferenceError(f"the sync events disagree by up to {worst * 1e3:.1f} ms with "
                             f"one offset and drift, more than the declared uncertainty "
                             f"{u * 1e3:.1f} ms: check the pairing (tap order, missing taps)")
    return ClockMap(method, offset, scale, u, events, res, note)


# --- matching -----------------------------------------------------------------------------
@dataclass
class Match:
    sample: ReferenceSample
    joint: str
    used: bool
    reason: str
    take: Optional[str] = None
    item: Optional[str] = None
    tracker_us: Optional[Tuple[float, float]] = None
    frame_ids: Optional[Tuple[int, int]] = None
    n_frames: int = 0
    r_pm: Optional[np.ndarray] = None       # the joint's bone against the palm
    hold_spread_deg: Optional[float] = None


def _excl(s, joint, reason, **kw) -> Match:
    return Match(s, joint, False, reason, **kw)


def _take_at(trk: TrackerSession, us: float) -> Optional[TrackerTake]:
    return next((t for t in trk.takes if t.t0_us <= us <= t.t1_us), None)


def match_samples(ref: ReferenceCapture, trk: TrackerSession,
                  clock: Optional[ClockMap], shift_s: float = 0.0) -> List[Match]:
    """One Match per (sample, joint with a coordinate), used or excluded with
    its reason (docs/reference_validation.md, "Exclusion reasons"). A sample
    is placed on the tracker clock (ClockMap; shifted by shift_s for the
    sensitivity runs) and matched to the interpolated tracker pose inside one
    segment (replay_core.pose_at), never across a gap, an identity change or
    an invalid-status cut; a hold (time_end_s, or a per_take sample) is the
    chordal mean over its frames, which must lie in one segment."""
    out = []
    for s in ref.samples:
        joints = ref.movement_joints.get(s.movement) or tuple(ref.joints)
        for joint in joints:
            out.append(_match_one(ref, trk, clock, s, joint, shift_s))
    return out


def _match_one(ref, trk, clock, s: ReferenceSample, joint: str, shift_s: float) -> Match:
    spec = JOINTS[joint]
    if not s.valid:
        return _excl(s, joint, f"reference_invalid: {s.reason}")
    if not s.movement:
        return _excl(s, joint, "reference_unlabeled: the sample names no movement")
    missing = [m for m in spec.movements if s.values.get((joint, m)) is None]
    if missing:
        return _excl(s, joint, "reference_coordinate_missing: " + ", ".join(
            next(c.column for c in ref.coordinates if c.joint == joint and c.movement == m)
            for m in missing))
    if clock is None:                           # per_take: the whole take is the hold
        take = trk.take(s.tracker_file)
        if take is None:
            return _excl(s, joint, f"outside_tracker_takes: {s.tracker_file} is not a take "
                         "of the session")
        frames = list(range(len(take.rec.frames)))
    else:
        t0 = clock.tracker_us(s.t0 * ref.time_scale, shift_s)
        t1 = (t0 if s.t1 is None else clock.tracker_us(s.t1 * ref.time_scale, shift_s))
        take = _take_at(trk, t0)
        if take is None or (s.t1 is not None and _take_at(trk, t1) is not take):
            return _excl(s, joint, "outside_tracker_takes: no take covers "
                         f"{t0 / 1e6:.6f} s" + ("" if s.t1 is None else
                                                f" to {t1 / 1e6:.6f} s"))
        frames = None
    kw = {"take": take.file, "item": take.item}
    if take.item != s.movement:
        return _excl(s, joint, f"movement_mismatch: a {s.movement} sample in take "
                     f"{take.file} of item {take.item}", **kw)
    rec = take.rec
    bone = spec.bone
    if frames is None and s.t1 is None:         # an instant: the interpolated pose
        if rec.timeline.source != "timestamp_us":
            return _excl(s, joint, f"tracker_clock_fallback: {rec.timeline.reason}", **kw)
        t = (t0 - take.t0_us) / 1e6
        p = rc.pose_at(rec, t, "interpolate")
        if p.source not in ("recorded", "interpolated"):
            return _excl(s, joint, f"tracker_discontinuity: {p.note or p.source}", **kw)
        f0 = rec.frames[p.i0].frame_id
        f1 = rec.frames[p.i1 if p.i1 is not None else p.i0].frame_id
        return Match(s, joint, True, "", tracker_us=(t0, t0), frame_ids=(f0, f1),
                     n_frames=1 if p.source == "recorded" else 2,
                     r_pm=bone_in_palm(p.rot, bone), **kw)
    if frames is None:                           # a hold on the reference clock
        if rec.timeline.source != "timestamp_us":
            return _excl(s, joint, f"tracker_clock_fallback: {rec.timeline.reason}", **kw)
        tt = (t0 - take.t0_us) / 1e6, (t1 - take.t0_us) / 1e6
        frames = [i for i, x in enumerate(rec.timeline.t) if tt[0] <= x <= tt[1]]
        kw["tracker_us"] = (t0, t1)
    if len(frames) < MIN_HOLD_FRAMES:
        return _excl(s, joint, f"hold_too_short: {len(frames)} tracker frames, fewer than "
                     f"{MIN_HOLD_FRAMES}", **kw)
    segs = {k for k, g in enumerate(rec.segments)
            for i in (frames[0], frames[-1]) if g.start <= i <= g.end}
    if len(segs) != 1:
        cut = [e for e in rec.events if frames[0] < e.index <= frames[-1]
               and e.kind in ("gap", "id_change", "invalid_status")]
        return _excl(s, joint, "tracker_discontinuity inside the hold: " + "; ".join(
            f"{e.kind} at frame_id {e.frame_id_after}" for e in cut[:3]), **kw)
    rots = [bone_in_palm(rc.reconstruct(rec.frames[i]).rot, bone) for i in frames]
    mean = te.chordal_mean(rots)
    if "tracker_us" not in kw:
        kw["tracker_us"] = (float(rec.frames[frames[0]].timestamp_us or 0),
                            float(rec.frames[frames[-1]].timestamp_us or 0))
    return Match(s, joint, True, "", frame_ids=(rec.frames[frames[0]].frame_id,
                                                 rec.frames[frames[-1]].frame_id),
                 n_frames=len(frames), r_pm=mean,
                 hold_spread_deg=float(max(rc.rotation_angle_deg(mean, r) for r in rots)),
                 **kw)


# --- statistics ---------------------------------------------------------------------------
def error_stats(err, ref) -> dict:
    """Signed bias, MAE, RMSE and percentiles of wrapped errors (degrees), with
    the tested range of the reference values."""
    err = wrap180(err)
    ref = np.asarray(ref, dtype=float)
    if err.size == 0:
        return {"n": 0}
    a = np.abs(err)
    return {"n": int(err.size), "bias_deg": float(err.mean()), "mae_deg": float(a.mean()),
            "rmse_deg": float(math.sqrt(np.mean(err * err))),
            "abs_p50_deg": float(np.percentile(a, 50)),
            "abs_p90_deg": float(np.percentile(a, 90)),
            "abs_p95_deg": float(np.percentile(a, 95)), "abs_max_deg": float(a.max()),
            "signed_p05_deg": float(np.percentile(err, 5)),
            "signed_p95_deg": float(np.percentile(err, 95)),
            "reference_min_deg": float(ref.min()), "reference_max_deg": float(ref.max())}


def slope(x, y) -> Optional[float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) < 1.0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def crosstalk(rows: List[dict], isolated: str, movements: Sequence[str]) -> dict:
    """During an isolated movement of `isolated`: the isolated coordinate's own
    gain error (slope, deg per deg, of its wrapped error against its reference
    value: 0 for a unit gain, -0.1 for a tracker reading 0.9 of it), and how
    each other coordinate's error moves with it ("into": slope of its wrapped
    error against the isolated coordinate's reference value, the error's
    range, and the reference's own off-axis slope, i.e. how isolated the
    movement was). Slopes are None when the isolated coordinate moved less
    than 1 degree or there are fewer than 3 samples."""
    x = [r["ref"][isolated] for r in rows]
    e_iso = wrap180([r["pred"][isolated] - r["ref"][isolated] for r in rows])
    out = {"isolated": isolated, "n": len(rows),
           "isolated_range_deg": float(np.ptp(x)) if x else None,
           "gain_error_slope_deg_per_deg": slope(x, e_iso), "into": {}}
    for other in movements:
        if other == isolated:
            continue
        e = wrap180([r["pred"][other] - r["ref"][other] for r in rows])
        out["into"][other] = {
            "error_slope_deg_per_deg": slope(x, e),
            "error_range_deg": float(np.ptp(e)) if len(e) else None,
            "reference_offaxis_slope": slope(x, [r["ref"][other] for r in rows])}
    return out


def _groups(rows, key):
    g: Dict[object, List[dict]] = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return g


def evaluate(rows: List[dict], movements: Dict[str, Optional[str]],
             coords: Sequence[str]) -> dict:
    """Per movement and pooled ("all"): error_stats of every coordinate, the
    cross-talk of the isolated movements, and the repeatability across trials
    (each trial's bias; their SD and range when there are two or more)."""
    out = {}
    for mov, rs in sorted(_groups(rows, lambda r: r["movement"]).items()) + [("all", rows)]:
        d = {c: error_stats([r["pred"][c] - r["ref"][c] for r in rs],
                            [r["ref"][c] for r in rs]) for c in coords}
        iso = SHORT.get(movements.get(mov)) if mov != "all" else None
        if iso is not None:
            d["crosstalk"] = crosstalk(rs, iso, coords)
        trials = _groups(rs, lambda r: r["trial"])
        rep = {}
        for c in coords:
            per = {t: float(np.mean(wrap180([r["pred"][c] - r["ref"][c] for r in tr])))
                   for t, tr in sorted(trials.items())}
            vals = list(per.values())
            rep[c] = {"trials": len(per), "per_trial_bias_deg": per,
                      "sd_of_trial_bias_deg": (float(np.std(vals, ddof=1))
                                               if len(vals) > 1 else None),
                      "range_of_trial_bias_deg": (float(np.ptp(vals))
                                                  if len(vals) > 1 else None),
                      "note": ("" if len(vals) > 1 else
                               "one trial: repeatability not assessable")}
        d["repeatability"] = rep
        out[mov] = d
    return out


# --- the comparison ---------------------------------------------------------------------
def _tmc_rows(matches: List[Match], ref: ReferenceCapture, predict) -> List[dict]:
    rows = []
    for m in matches:
        if not m.used or m.joint != "thumb_tmc":
            continue
        pred = predict(m)
        if pred is None:
            continue
        rows.append({"sample_id": m.sample.sample_id, "trial": m.sample.trial,
                     "movement": m.sample.movement,
                     "ref": {"fe": m.sample.values[("thumb_tmc", "flexion_extension")],
                             "aa": m.sample.values[("thumb_tmc", "abduction_adduction")]},
                     "pred": {"fe": pred[0], "aa": pred[1]}})
    return rows


def compare(ref: ReferenceCapture, trk: TrackerSession, calibration_trials: Sequence[int],
            tmc_calibration: Optional[te.Calibration] = None,
            tmc_calibration_path=None) -> dict:
    """The whole comparison (docs/reference_validation.md): compatibility,
    clock, matching, the thumb TMC mapping fitted on the calibration trials,
    held-out metrics of the mapped values (and of the unvalidated estimate
    when a tracker calibration is given), the sync sensitivity, and the CMC
    observability. Returns the report dict with the private keys `_matches`,
    `_mapping` (the TmcMapping or None) for write_report."""
    check_compatible(ref, trk)
    if tmc_calibration is not None:
        check_tracker_calibration(ref, tmc_calibration, tmc_calibration_path)
    cal_trials = sorted({int(t) for t in calibration_trials})
    trials = sorted({s.trial for s in ref.samples})
    held_trials = [t for t in trials if t not in cal_trials]
    if not cal_trials or not set(cal_trials) & set(trials):
        raise ReferenceError(f"no calibration trial: the capture has trials {trials}; "
                             "give --calibration-trials (the trials the mapping is fitted on)")
    if not held_trials:
        raise ReferenceError(f"every trial ({trials}) is a calibration trial: nothing is "
                             "left to evaluate the mapping on; hold at least one trial out")
    clock = build_clock(ref, trk)
    matches = match_samples(ref, trk, clock)
    used = [m for m in matches if m.used]
    role = {m.sample.sample_id: ("calibration" if m.sample.trial in cal_trials else "heldout")
            for m in matches}
    cal_ids = sorted({m.sample.sample_id for m in used if role[m.sample.sample_id] == "calibration"})
    held_ids = sorted({m.sample.sample_id for m in used if role[m.sample.sample_id] == "heldout"})
    assert not set(cal_ids) & set(held_ids)
    excluded: Dict[str, int] = {}
    for m in matches:
        if not m.used:
            key = m.reason.split(":", 1)[0]
            excluded[key] = excluded.get(key, 0) + 1
    report = {
        "format": REPORT_FORMAT, "status": ref.status,
        "status_text": STATUS_TEXT[ref.status],
        "reference": {"folder": str(ref.folder), "header_sha256": ref.header_sha256,
                      "samples_sha256": ref.samples_sha256, "data_origin": ref.data_origin,
                      "role": ref.role, "participant": ref.participant, "hand": ref.hand,
                      "session": ref.session, "device": ref.device,
                      "calibration": ref.calibration, "angle_unit": ref.angle_unit,
                      "frames": ref.frames, "source_files": ref.source_files},
        "tracker_session": {"folder": str(trk.folder), "manifest_sha256": trk.manifest_sha256,
                            "participant": trk.participant, "hand": trk.hand,
                            "session": trk.session, "device": trk.device,
                            "takes": [{"file": t.file, "item": t.item, "take": t.take,
                                       "frames": len(t.rec.frames),
                                       "segments": len(t.rec.segments),
                                       "clock": t.rec.timeline.source,
                                       "sha256": t.sidecar.get("sha256")}
                                      for t in trk.takes]},
        "tracker_calibration": (None if tmc_calibration is None else {
            "kind": tmc_calibration.kind, "hand": tmc_calibration.hand,
            "session": tmc_calibration.session, "provenance": tmc_calibration.provenance,
            "file": None if tmc_calibration_path is None else str(tmc_calibration_path),
            "sha256": (None if tmc_calibration_path is None
                       else sha256_of(tmc_calibration_path))}),
        "sync": ({"method": "per_take", "note": "each sample is a hold read on a named "
                  "tracker take; no clock alignment, no sensitivity run"}
                 if clock is None else clock.as_json()),
        "split": {"unit": "trial", "calibration_trials": cal_trials,
                  "heldout_trials": held_trials, "calibration_samples": cal_ids,
                  "heldout_samples": held_ids},
        "samples": {"total": len(ref.samples), "matches": len(matches),
                    "used": len(used), "excluded": len(matches) - len(used),
                    "excluded_by_reason": dict(sorted(excluded.items()))},
        "thresholds": {"accuracy": ACCURACY_THRESHOLD_NOTE,
                       "identifiability": IDENTIFIABILITY_NOTE},
        "joints": {},
    }
    mapping = None
    if "thumb_tmc" in ref.joints:
        report["joints"]["thumb_tmc"], mapping = _compare_tmc(
            ref, trk, clock, matches, role, tmc_calibration, cal_trials)
    for joint in CMC_JOINTS:
        if joint in ref.joints:
            report["joints"][joint] = _compare_cmc(ref, matches, joint)
    report["_matches"], report["_mapping"] = matches, mapping
    report["_role"] = role
    return report


def _compare_tmc(ref, trk, clock, matches, role, tmc_cal, cal_trials):
    order = ref.frames["thumb_tmc"]["rotation_order"]
    hand = ref.hand
    tm = [m for m in matches if m.used and m.joint == "thumb_tmc"]
    out = {"rotation_order": order, "convention": "canonical (module docstring): fe "
           "positive ulnar across the palm, aa positive palmar",
           "observability": (rotation_spread([m.r_pm for m in tm]) if tm else None)}
    cal = [m for m in tm if role[m.sample.sample_id] == "calibration"]
    held = [m for m in tm if role[m.sample.sample_id] == "heldout"]
    out["counts"] = {"calibration": len(cal), "heldout": len(held)}
    if not held:
        raise ReferenceError("no usable held-out thumb TMC sample (every held-out sample "
                             "was excluded: see excluded_by_reason)")
    d = [tmc_direction(m.sample.values[("thumb_tmc", "flexion_extension")],
                       m.sample.values[("thumb_tmc", "abduction_adduction")], order, hand)
         for m in cal]
    try:
        fit = fit_mapping([m.r_pm for m in cal], d)
    except ReferenceError as e:
        out["mapping"] = {"refused": str(e)}
        fit = None
    mapping = None
    if fit is not None:
        out["mapping"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in fit.items()}

        def predict(m, a=fit["A"], b=fit["b"]):
            return tmc_angles(a @ m.r_pm @ b, order, hand)
        rows = _tmc_rows(held, ref, predict)
        out["heldout"] = {"mapped": evaluate(rows, ref.movements, ("fe", "aa"))}
        cal_rows = _tmc_rows(cal, ref, predict)
        out["calibration_fit"] = {"mapped": evaluate(cal_rows, ref.movements, ("fe", "aa"))}
        # tested = held out; the calibration range was fitted, not tested
        out["tested_range_deg"] = {c: [float(min(r["ref"][c] for r in rows)),
                                       float(max(r["ref"][c] for r in rows))]
                                   for c in ("fe", "aa")}
        out["calibration_range_deg"] = {c: [float(min(r["ref"][c] for r in cal_rows)),
                                            float(max(r["ref"][c] for r in cal_rows))]
                                        for c in ("fe", "aa")}
        if clock is not None:
            sens = {}
            for label, shift in (("minus", -clock.uncertainty_s), ("plus", clock.uncertainty_s)):
                sh = match_samples(ref, trk, clock, shift_s=shift)
                srows = _tmc_rows([m for m in sh if m.used and m.joint == "thumb_tmc"
                                   and role[m.sample.sample_id] == "heldout"], ref, predict)
                ev = evaluate(srows, ref.movements, ("fe", "aa"))["all"]
                sens[label] = {"shift_s": shift, "n": len(srows),
                               **{f"{c}_rmse_deg": ev[c].get("rmse_deg") for c in ("fe", "aa")},
                               **{f"{c}_bias_deg": ev[c].get("bias_deg") for c in ("fe", "aa")}}
            ev0 = out["heldout"]["mapped"]["all"]
            sens["zero"] = {"shift_s": 0.0, "n": ev0["fe"]["n"],
                            **{f"{c}_rmse_deg": ev0[c].get("rmse_deg") for c in ("fe", "aa")},
                            **{f"{c}_bias_deg": ev0[c].get("bias_deg") for c in ("fe", "aa")}}
            out["sync_sensitivity"] = sens
        mapping = TmcMapping.from_fit(ref, trk, fit, out, cal_trials,
                                      [m.sample.sample_id for m in cal],
                                      [m.sample.sample_id for m in held])
    if tmc_cal is not None:
        def estimate(m):
            r_tmc = tmc_cal.R_rest.T @ m.r_pm
            return tmc_angles(estimate_direction(r_tmc, hand), order, hand)
        out.setdefault("heldout", {})[f"estimate_{tmc_cal.kind}"] = evaluate(
            _tmc_rows(held, ref, estimate), ref.movements, ("fe", "aa"))
    return out, mapping


def _compare_cmc(ref, matches, joint) -> dict:
    ms = [m for m in matches if m.used and m.joint == joint]
    if not ms:
        return {"verdict": "no usable sample", "n": 0}
    spread = rotation_spread([m.r_pm for m in ms])
    refv = [m.sample.values[(joint, "flexion_extension")] for m in ms]
    # the tracker's own CMC reading: the metacarpal's flexion against the palm
    trk = [jf.joint_angles(m.r_pm, ref.hand)[0] for m in ms]
    unobservable = spread["max_from_mean_deg"] < UNOBSERVABLE_DEG
    return {"verdict": "unobservable" if unobservable else "observable_unvalidated",
            "explanation": (
                f"the tracker's {JOINTS[joint].bone} does not move against its palm "
                f"(largest departure from its mean {spread['max_from_mean_deg']:.4f} deg "
                f"over {spread['n']} samples) while the reference moves "
                f"{float(np.ptp(refv)):.2f} deg: the tracker does not report this motion, "
                "no mapping is fitted and the coordinate stays unavailable"
                if unobservable else
                "the tracker's metacarpal moves against its palm; no mapping is defined "
                "for the CMC here: the raw comparison is reported and the coordinate "
                "stays unavailable until a mapping and its held-out evaluation exist"),
            "tracker_spread": spread,
            "reference_range_deg": [float(min(refv)), float(max(refv))],
            "raw_tracker_minus_reference": error_stats(np.subtract(trk, refv), refv)}


# --- the mapping and its scope ---------------------------------------------------------
@dataclass
class TmcMapping:
    id: str
    status: str
    participant: str
    hand: str
    session: str
    tracker_device: str
    reference_device: dict
    rotation_order: str
    a: np.ndarray
    b: np.ndarray
    tested_range_deg: Dict[str, List[float]]
    heldout: dict
    calibration_trials: List[int]
    calibration_samples: List[str]
    heldout_samples: List[str]
    zero: dict
    provenance: dict

    @classmethod
    def from_fit(cls, ref, trk, fit, out, cal_trials, cal_ids, held_ids) -> "TmcMapping":
        a, b = fit["A"], fit["b"]
        key = json.dumps([ref.participant, ref.hand, ref.session, ref.samples_sha256,
                          np.round(a, 12).tolist(), np.round(b, 12).tolist()])
        allm = out["heldout"]["mapped"]["all"]
        return cls(
            id="tmcmap-" + hashlib.sha256(key.encode()).hexdigest()[:12],
            status=ref.status, participant=ref.participant, hand=ref.hand,
            session=ref.session, tracker_device=trk.device, reference_device=ref.device,
            rotation_order=ref.frames["thumb_tmc"]["rotation_order"], a=a, b=b,
            tested_range_deg=out["tested_range_deg"],
            heldout={c: {k: allm[c].get(k) for k in ("n", "bias_deg", "mae_deg", "rmse_deg",
                                                      "abs_p95_deg")} for c in ("fe", "aa")},
            calibration_trials=list(cal_trials), calibration_samples=sorted(cal_ids),
            heldout_samples=sorted(held_ids), zero=ref.frames["thumb_tmc"]["zero"],
            provenance={"data_origin": ref.data_origin, "role": ref.role,
                        "reference_folder": str(ref.folder),
                        "reference_header_sha256": ref.header_sha256,
                        "reference_samples_sha256": ref.samples_sha256,
                        "tracker_session_folder": str(trk.folder),
                        "tracker_manifest_sha256": trk.manifest_sha256})

    def to_json(self) -> dict:
        return {"format": MAPPING_FORMAT, "id": self.id, "status": self.status,
                "status_text": STATUS_TEXT[self.status], "joint": "thumb_tmc",
                "outputs": list(MAPPED_COLUMNS), "participant": self.participant,
                "hand": self.hand, "session": self.session,
                "tracker_device": self.tracker_device,
                "reference_device": self.reference_device,
                "rotation_order": self.rotation_order, "zero": self.zero,
                "convention": "canonical: fe positive ulnar across the palm, aa positive "
                              "palmar (leap_hand.reference_validation)",
                "A": np.asarray(self.a).tolist(), "b": np.asarray(self.b).tolist(),
                "tested_range_deg": self.tested_range_deg, "heldout": self.heldout,
                "calibration_trials": self.calibration_trials,
                "calibration_samples": self.calibration_samples,
                "heldout_samples": self.heldout_samples,
                "provenance": self.provenance,
                "scope": ("this participant, hand and tracker session only, inside the "
                          "tested range (the held-out samples' range); elsewhere the "
                          "mapped columns stay empty")}

    @classmethod
    def from_json(cls, d: dict) -> "TmcMapping":
        if d.get("format") != MAPPING_FORMAT:
            raise ReferenceError(f"a mapping file has format {MAPPING_FORMAT!r}, got "
                                 f"{d.get('format')!r}")
        if d.get("status") not in STATUSES:
            raise ReferenceError(f"mapping status must be one of {STATUSES}")
        a, b = np.asarray(d["A"], dtype=float), np.asarray(d["b"], dtype=float)
        if a.shape != (3, 3) or b.shape != (3,) or \
                not np.allclose(a.T @ a, np.eye(3), atol=1e-9) or np.linalg.det(a) < 0:
            raise ReferenceError("mapping A must be a 3x3 rotation and b a unit 3-vector")
        return cls(d["id"], d["status"], d["participant"], d["hand"], d["session"],
                   d.get("tracker_device", ""), d.get("reference_device", {}),
                   d["rotation_order"], a, b / np.linalg.norm(b), d["tested_range_deg"],
                   d.get("heldout", {}), d.get("calibration_trials", []),
                   d.get("calibration_samples", []), d.get("heldout_samples", []),
                   d.get("zero", {}), d.get("provenance", {}))

    def scope_row(self) -> str:
        """The scope, the tested range first (what a narrow panel keeps)."""
        r = self.tested_range_deg
        return (f"tested fe {r['fe'][0]:.0f}..{r['fe'][1]:.0f}, aa {r['aa'][0]:.0f}.."
                f"{r['aa'][1]:.0f} deg; {self.participant}, {self.hand} hand, session "
                f"{self.session}; {self.id}")

    def scope_text(self) -> str:
        r = self.tested_range_deg
        h = self.heldout
        return (f"{self.id}: participant {self.participant}, {self.hand} hand, session "
                f"{self.session}; tested fe {r['fe'][0]:.0f}..{r['fe'][1]:.0f}, aa "
                f"{r['aa'][0]:.0f}..{r['aa'][1]:.0f} deg; held-out MAE fe "
                f"{_f(h.get('fe', {}).get('mae_deg'))} / aa "
                f"{_f(h.get('aa', {}).get('mae_deg'))} deg (n "
                f"{h.get('fe', {}).get('n')})")


def _f(v, fmt=".1f") -> str:
    return "n/a" if v is None else format(v, fmt)


def load_mapping(path) -> TmcMapping:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ReferenceError(f"cannot read the mapping {path}: {e}") from None
    try:
        return TmcMapping.from_json(d)
    except KeyError as e:
        raise ReferenceError(f"mapping {path}: field {e} missing") from None


def recording_scope(rec: rc.Recording) -> Dict[str, Optional[str]]:
    """The participant and session a recording belongs to: its metadata's, or,
    for a loose file, those of its META_SUFFIX sidecar (written by
    record_validation.py); None where neither says."""
    md = rec.metadata
    out = {"session": md.get("session") or None,
           "participant": md.get("participant") or md.get("participant_id") or None}
    side = Path(rec.path).with_suffix(META_SUFFIX)
    if (out["session"] is None or out["participant"] is None) and side.is_file():
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        for k in ("session", "participant"):
            if out[k] is None and isinstance(meta.get(k), str):
                out[k] = meta[k]
    return out


def scope_mismatch(mapping: TmcMapping, hand_side: str, session: Optional[str],
                   participant: Optional[str]) -> Optional[str]:
    """Why a mapping is not applied to a recording, or None. Its scope is the
    participant, hand and tracker session it was evaluated on; nothing is
    extended to other sessions or participants by one success."""
    if hand_side != mapping.hand:
        return f"this recording is of the {hand_side} hand, the mapping of the {mapping.hand} hand"
    if session is not None and session != mapping.session:
        return (f"this recording is of session {session}, the mapping of session "
                f"{mapping.session}")
    if participant is not None and participant != mapping.participant:
        return (f"this recording is of participant {participant}, the mapping of "
                f"participant {mapping.participant}")
    unknown = [w for w, v in (("session", session), ("participant", participant)) if v is None]
    if unknown:
        return (f"this recording names no {' and no '.join(unknown)} (the mapping is of "
                f"participant {mapping.participant}, session {mapping.session})")
    return None


def apply_mapping(mapping: TmcMapping, rot) -> Dict[str, object]:
    """The mapped pair of one pose inside the scope: {T_TMC_fe_mapped,
    T_TMC_aa_mapped, status, note}. Outside the tested range both values are
    None and the status says so; the caller checks the scope first."""
    r_pm = bone_in_palm(rot, "THUMB_METACARPAL")
    fe, aa = tmc_angles(mapping.a @ r_pm @ mapping.b, mapping.rotation_order, mapping.hand)
    r = mapping.tested_range_deg
    if not (r["fe"][0] <= fe <= r["fe"][1] and r["aa"][0] <= aa <= r["aa"][1]):
        return {MAPPED_COLUMNS[0]: None, MAPPED_COLUMNS[1]: None,
                "status": "outside_tested_range",
                "note": (f"fe {fe:.1f}, aa {aa:.1f} deg outside the tested range fe "
                         f"{r['fe'][0]:.1f}..{r['fe'][1]:.1f}, aa {r['aa'][0]:.1f}.."
                         f"{r['aa'][1]:.1f}: not evaluated there")}
    status = {"evaluated": "evaluated_in_range"}.get(mapping.status, mapping.status)
    return {MAPPED_COLUMNS[0]: fe, MAPPED_COLUMNS[1]: aa, "status": status, "note": ""}


# --- the report on disk ---------------------------------------------------------------------
MATCH_COLUMNS = ["sample_id", "trial", "movement", "joint", "role", "used", "reason",
                 "take", "item", "tracker_t0_us", "tracker_t1_us", "frame_id_first",
                 "frame_id_last", "n_frames", "hold_spread_deg", "ref_fe_deg", "ref_aa_deg",
                 "mapped_fe_deg", "mapped_aa_deg", "error_fe_deg", "error_aa_deg"]


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def write_report(report: dict, out) -> dict:
    """report.json, report.md, matched_samples.csv and (when fitted) mapping.json
    into `out`; returns {name: path}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mapping = report.get("_mapping")
    role = report.get("_role", {})
    paths = {}
    order = (report["joints"].get("thumb_tmc") or {}).get("rotation_order")
    hand = report["reference"]["hand"]
    with open(out / "matched_samples.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(MATCH_COLUMNS)
        for m in report.get("_matches", []):
            s = m.sample
            ref_fe = s.values.get((m.joint, "flexion_extension"))
            ref_aa = s.values.get((m.joint, "abduction_adduction"))
            pred = (None, None)
            if m.used and m.joint == "thumb_tmc" and mapping is not None:
                pred = tmc_angles(mapping.a @ m.r_pm @ mapping.b, order, hand)
            err = [None if p is None or r is None else float(wrap180(p - r))
                   for p, r in zip(pred, (ref_fe, ref_aa))]
            us = m.tracker_us or (None, None)
            fid = m.frame_ids or (None, None)
            w.writerow([s.sample_id, s.trial, s.movement, m.joint, role.get(s.sample_id, ""),
                        int(m.used), m.reason, m.take or "", m.item or "",
                        "" if us[0] is None else int(round(us[0])),
                        "" if us[1] is None else int(round(us[1])),
                        "" if fid[0] is None else fid[0], "" if fid[1] is None else fid[1],
                        m.n_frames, "" if m.hold_spread_deg is None else
                        f"{m.hold_spread_deg:.4f}"]
                       + ["" if v is None else f"{v:.6f}"
                          for v in (ref_fe, ref_aa, *pred, *err)])
    paths["matched_samples.csv"] = out / "matched_samples.csv"
    if mapping is not None:
        (out / "mapping.json").write_text(json.dumps(_clean(mapping.to_json()), indent=2)
                                          + "\n", encoding="utf-8")
        paths["mapping.json"] = out / "mapping.json"
        report["mapping_file"] = str(out / "mapping.json")
        report["mapping_id"] = mapping.id
    (out / "report.json").write_text(json.dumps(_clean(report), indent=2) + "\n",
                                     encoding="utf-8")
    paths["report.json"] = out / "report.json"
    (out / "report.md").write_text(report_markdown(report), encoding="utf-8")
    paths["report.md"] = out / "report.md"
    return paths


def _stat_line(name, s) -> str:
    if not s or not s.get("n"):
        return f"| {name} | 0 | | | | | | |"
    return (f"| {name} | {s['n']} | {s['bias_deg']:+.2f} | {s['mae_deg']:.2f} | "
            f"{s['rmse_deg']:.2f} | {s['abs_p50_deg']:.2f} / {s['abs_p90_deg']:.2f} / "
            f"{s['abs_p95_deg']:.2f} / {s['abs_max_deg']:.2f} | "
            f"{s['reference_min_deg']:.1f} .. {s['reference_max_deg']:.1f} |")


def report_markdown(r: dict) -> str:
    ref, trk = r["reference"], r["tracker_session"]
    lines = [f"# Reference comparison: {ref['participant']}, {ref['hand']} hand, session "
             f"{ref['session']}", "",
             f"**Status: {r['status']}** ({r['status_text']}).", "",
             f"- Reference: {ref['device'].get('kind')} {ref['device'].get('make')} "
             f"{ref['device'].get('model')} (data_origin {ref['data_origin']}, role "
             f"{ref['role']}); calibration {ref['calibration'].get('id')}.",
             f"- Tracker: {trk['device']}; {len(trk['takes'])} takes.",
             f"- Sync: {r['sync'].get('method')}; " + (
                 f"offset {r['sync']['offset_s']:.6f} s, drift {r['sync']['drift']:+.2e}, "
                 f"declared uncertainty {r['sync']['uncertainty_s'] * 1e3:.1f} ms, "
                 f"residual max {_f((r['sync'].get('residual_max_s') or 0) * 1e3)} ms "
                 f"({r['sync']['note']})" if "offset_s" in r["sync"] else r["sync"]["note"]),
             f"- Split by trial: calibration {r['split']['calibration_trials']}, held out "
             f"{r['split']['heldout_trials']} ({len(r['split']['calibration_samples'])} / "
             f"{len(r['split']['heldout_samples'])} samples); no sample is in both.",
             f"- Samples: {r['samples']['total']}; matches {r['samples']['matches']}, used "
             f"{r['samples']['used']}, excluded {r['samples']['excluded']}: "
             + (", ".join(f"{k} {v}" for k, v in r["samples"]["excluded_by_reason"].items())
                or "none") + ".",
             f"- Thresholds: {r['thresholds']['accuracy']}.", ""]
    tmc = r["joints"].get("thumb_tmc")
    if tmc:
        lines += [f"## Thumb TMC (T_TMC_fe, T_TMC_aa: {tmc['rotation_order']}; "
                  f"{tmc['convention']})", ""]
        mp = tmc.get("mapping", {})
        if "refused" in mp:
            lines += [f"Mapping refused: {mp['refused']}", ""]
        else:
            lines += [f"Mapping fitted on {mp['n']} calibration samples: residual RMS "
                      f"{mp['rms_residual_deg']:.3f} deg (max {mp['max_residual_deg']:.3f}); "
                      f"long axis {mp['b_from_tracker_axis_deg']:.2f} deg from the tracker's.",
                      ""]
        for name, block in (tmc.get("heldout") or {}).items():
            lines += [f"### Held-out: {name}", "",
                      "| movement / coordinate | n | bias | MAE | RMSE | abs p50/p90/p95/max "
                      "| tested range |", "|---|---|---|---|---|---|---|"]
            for mov, d in block.items():
                for c in ("fe", "aa"):
                    lines.append(_stat_line(f"{mov} {c}", d.get(c)))
            lines.append("")
            for mov, d in block.items():
                ct = d.get("crosstalk")
                if ct:
                    lines.append(f"- {mov} (isolated {ct['isolated']}, range "
                                 f"{_f(ct['isolated_range_deg'], '.1f')} deg): gain error "
                                 f"slope {_f(ct['gain_error_slope_deg_per_deg'], '+.3f')} "
                                 "deg/deg")
                for other, x in ((ct or {}).get("into") or {}).items():
                    lines.append(f"- cross-talk in {mov} into {other}: error slope "
                                 f"{_f(x['error_slope_deg_per_deg'], '+.3f')} deg/deg, "
                                 f"error range {_f(x['error_range_deg'], '.2f')} deg, "
                                 f"reference off-axis slope "
                                 f"{_f(x['reference_offaxis_slope'], '+.3f')}")
                for c, rep in d.get("repeatability", {}).items():
                    if mov != "all":
                        lines.append(f"- repeatability {mov} {c}: {rep['trials']} trials, SD "
                                     f"of trial bias {_f(rep['sd_of_trial_bias_deg'], '.2f')} "
                                     f"deg {rep['note']}".rstrip())
            lines.append("")
        if tmc.get("sync_sensitivity"):
            s = tmc["sync_sensitivity"]
            lines += ["### Sync sensitivity (held-out RMSE with the clock shifted by the "
                      "declared uncertainty)", ""] + [
                f"- {k}: shift {v['shift_s'] * 1e3:+.1f} ms, n {v['n']}, fe RMSE "
                f"{_f(v['fe_rmse_deg'], '.3f')}, aa RMSE {_f(v['aa_rmse_deg'], '.3f')} deg"
                for k, v in s.items()] + [""]
    for joint in CMC_JOINTS:
        c = r["joints"].get(joint)
        if c:
            lines += [f"## {joint} ({JOINTS[joint].paper3['flexion_extension']}): "
                      f"{c['verdict']}", "", c.get("explanation", ""), ""]
    lines += ["## Scope", "", "These numbers hold for this participant, hand, tracker "
              "session and reference device, over the tested range only. They do not "
              "validate any other session or the historical dataset.", ""]
    return "\n".join(lines)


# --- observability over a dataset ------------------------------------------------------
OBSERVED_BONES = ("THUMB_METACARPAL", "INDEX_METACARPAL", "MIDDLE_METACARPAL",
                  "RING_METACARPAL", "LITTLE_METACARPAL")


def observability(rec: rc.Recording) -> Dict[str, dict]:
    """For each metacarpal: how far its orientation against the palm moves
    over the recording's selected frames (rotation_spread)."""
    rots = {b: [] for b in OBSERVED_BONES}
    for f in rec.frames:
        rot = rc.reconstruct(f).rot
        for b in OBSERVED_BONES:
            rots[b].append(bone_in_palm(rot, b))
    return {b: rotation_spread(v) for b, v in rots.items()}


# --- the synthetic example ----------------------------------------------------------------
SYN_BONE_M = {"PROXIMAL": 0.045, "INTERMEDIATE": 0.040, "DISTAL": 0.025, "TIP": 0.020}
SYN_SPREAD_M = {"THUMB": -0.030, "INDEX": -0.020, "MIDDLE": 0.0, "RING": 0.018,
                "LITTLE": 0.034}
SYN_THUMB_M = {"PROXIMAL": 0.035, "DISTAL": 0.030, "TIP": 0.022}
SYN_OFFSET = jf.rot_y(4.0) @ jf.rot_x(-3.0)               # the long axis 5 deg off -z
SYN_A = jf.rot_x(-20.0) @ jf.rot_y(35.0) @ jf.rot_z(70.0)   # palm frame -> canonical


def syn_q(hand: str) -> np.ndarray:
    """The synthetic tracker's metacarpal frame in the canonical frame, LeapC's
    way round (estimate_direction): x the thumb's flexion axis (-s e2), y
    dorsal of the thumb (-s e1), z toward the base (-e3), turned by SYN_OFFSET
    so that its long axis b = Q^T e3 is 5 degrees off (0, 0, -1)."""
    s = _s(hand)
    q0 = np.array([[0.0, -s, 0.0], [-s, 0.0, 0.0], [0.0, 0.0, -1.0]])
    return q0 @ SYN_OFFSET


def synthetic_pose(hand: str, fe: float, aa: float, order: str, *,
                   a=SYN_A, q=None, tip_gap_mm: Optional[float] = None,
                   finger_flex: float = 10.0, r_pm=None) -> Tuple[np.ndarray, np.ndarray]:
    """World (pos (26, 3) m, rot (26, 3, 3)) of a hand whose thumb metacarpal
    stands at canonical (fe, aa) in `order`: R_pm = A^T R(fe, aa) Q, so that
    A R_pm b = R(fe, aa) e3 with b = Q^T e3 (Q = syn_q(hand) by default). The
    finger metacarpals are rigid with the palm (as LeapC's), the fingers flexed
    `finger_flex` at each joint. `tip_gap_mm` places the thumb tip that far
    from the index tip (sync taps). `r_pm`, when given, is the thumb
    metacarpal's rotation against the palm as is (fe, aa, order, a, q unused)."""
    q = syn_q(hand) if q is None else q
    pos, rot = np.zeros((26, 3)), np.zeros((26, 3, 3))
    w = JOINT_INDEX["WRIST"]
    pos[w] = (0.02, 0.20, -0.04)
    rot[w] = jf.rot_z(15.0) @ jf.rot_x(-25.0)
    if r_pm is None:
        r_pm = a.T @ tmc_rotation(fe, aa, order, hand) @ q
    for i in [PALM] + list(range(2, 26)):                  # parents first
        name = JOINT_NAMES[i]
        digit, _, part = name.partition("_")
        par = PARENT[i]
        if name == "PALM":
            rot[i] = rot[par] @ jf.rot_x(-8.0)
            pos[i] = pos[par] + rot[par] @ np.array([0.0, 0.0, -0.045])
            continue
        if part == "METACARPAL":                            # rigid with the palm,
            rot[i] = rot[PALM] @ r_pm if digit == "THUMB" else rot[PALM]   # but the thumb
            off = (SYN_SPREAD_M[digit], 0.0, 0.020 if digit == "THUMB" else -0.010)
            pos[i] = pos[PALM] + rot[PALM] @ np.asarray(off)
            continue
        length = (SYN_THUMB_M if digit == "THUMB" else SYN_BONE_M)[part]
        flex = 0.0 if part == "TIP" else (finger_flex if digit != "THUMB" else 5.0)
        rot[i] = rot[par] @ jf.rotation_from_angles(flex, 0.0, 0.0, hand)
        pos[i] = pos[par] + rot[par] @ np.array([0.0, 0.0, -length])
    if tip_gap_mm is not None:
        t = JOINT_INDEX["THUMB_TIP"]
        n = np.array([1.0, 0.3, 0.2]) / np.linalg.norm([1.0, 0.3, 0.2])
        pos[t] = pos[INDEX_TIP] + n * tip_gap_mm / 1000.0
    return pos, rot


def synthetic_line(hand: str, hand_id: int, frame_id: int, t_us: int, pose,
                   status: int = 1, t0_us: int = 0) -> dict:
    """One JSONL line in the recorder's camera format (leap_hand.recorder)."""
    pos, rot = pose
    joints = absolute_to_relative(pos, [mat3_to_quat(r) for r in rot])
    capture = round(1_790_000_000.0 + (t_us - t0_us) / 1e6, 6)
    return {"wall_time": round(capture + 0.015, 6), "timestamp": t_us / 1e6,
            "timestamp_us": int(t_us), "packet_counter": frame_id, "hand_side": hand,
            "frame_id": frame_id, "status": status, "hand_id": hand_id,
            "framerate": 90.0, "capture_time": capture, "source": "leap",
            "space": "leap_desktop", "units": "m",
            "joints": [{"name": j.name, "x": j.x, "y": j.y, "z": j.z, "qw": j.qw,
                        "qx": j.qx, "qy": j.qy, "qz": j.qz} for j in joints],
            "abs26": [[round(float(c), 6) for c in p] for p in pos]}


def write_manifest(folder, *, participant: str, hand: str, session: str,
                   tracker_device: str, takes: List[dict], extra: Optional[dict] = None) -> Path:
    """MANIFEST_FILE for a tracker session: every take with its sha256 (computed
    here from the file on disk), item, take number and frame count."""
    folder = Path(folder)
    rows = []
    for t in takes:
        p = folder / t["file"]
        rows.append({**t, "sha256": sha256_of(p)})
    doc = {"format": SESSION_FORMAT, "participant": participant, "hand": hand,
           "session": session, "tracker_device": tracker_device, "takes": rows,
           **(extra or {})}
    path = folder / MANIFEST_FILE
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def write_reference(folder, header: dict, rows: List[dict], columns: List[str],
                    sources: Dict[str, str]) -> Path:
    """A reference capture on disk: samples.csv, the source files given as
    {name: text}, and HEADER_FILE with their sha256 filled in."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "samples.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SAMPLE_COLUMNS) + list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k))
                        for k in list(SAMPLE_COLUMNS) + list(columns)})
    src = []
    for name, text in sources.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text, encoding="utf-8")
        src.append({"path": name, "sha256": sha256_of(folder / name),
                    "description": "the generator's parameters (synthetic)"
                    if header.get("data_origin") == "synthetic" else "device export"})
    h = {**header, "samples": {"file": "samples.csv",
                               "sha256": sha256_of(folder / "samples.csv")},
         "source_files": header.get("source_files") or src}
    (folder / HEADER_FILE).write_text(json.dumps(h, indent=2) + "\n", encoding="utf-8")
    return folder


# --- the end-to-end example (synthetic) -------------------------------------------------
EXAMPLE = {
    "participant": "SYNTHETIC-P00", "hand": "left",
    "session": "synthetic_20990101_000000_left", "seed": 20261024,
    "durations_s": {"sync_start": 6.0, "sync_end": 6.0, "tmc_neutral_hold": 2.0,
                    "tmc_flexion_sweep": 6.0, "tmc_abduction_sweep": 6.0,
                    "tmc_hold_extended": 2.0, "tmc_hold_flexed": 2.0,
                    "tmc_hold_abducted": 2.0, "cmc_palm_arc": 4.0},
    "prep_s": 5.0, "tracker_t0_us": 5_000_000_000, "period_us": 11_111,
    # tracker seconds = offset + scale * reference seconds
    "clock_offset_s": 4995.25, "clock_scale": 1.0 + 40e-6, "sync_uncertainty_s": 0.015,
    "reference_rate_hz": 50.0,
    # what the tracker's thumb metacarpal encodes, against the truth
    "tracker_gain_fe": 0.93, "tracker_fe_from_aa": 0.06, "tracker_aa_from_fe": 0.08,
    "tracker_noise_deg": 0.4, "reference_noise_deg": 0.3, "tap_noise_s": 0.001,
    "gap": {"item": "tmc_flexion_sweep", "trial": 2, "from_s": 2.0, "to_s": 2.3},
    "invalid_line": {"item": "tmc_abduction_sweep", "trial": 3, "at_s": 3.0},
    "reference_invalid": {"item": "tmc_abduction_sweep", "trial": 2,
                          "from_s": 1.0, "to_s": 1.2},
    "reference_missing_aa": {"item": "tmc_flexion_sweep", "trial": 3, "at_s": 1.5},
    "calibration_trials": [1],
}


def example_truth(item: str, trial: int, tau: float, dur: float) -> Dict[str, float]:
    """The synthetic hand's true angles (canonical degrees) at time tau of a take."""
    w = 2.0 * math.pi * tau / (dur / 2.0)                   # two cycles per take
    out = {"fe": 4.0 + 0.3 * trial, "aa": 7.0 - 0.2 * trial, "ring": 0.0, "little": 0.0}
    if item == "tmc_flexion_sweep":
        out.update(fe=10.0 + 30.0 * math.sin(w), aa=8.0 + 0.5 * math.sin(w / 2.0))
    elif item == "tmc_abduction_sweep":
        out.update(fe=5.0, aa=8.0 + 16.0 * (1.0 - math.cos(w)))
    elif item == "tmc_hold_extended":
        out.update(fe=-15.0 + trial, aa=6.0)
    elif item == "tmc_hold_flexed":
        out.update(fe=30.0 - trial, aa=10.0)
    elif item == "tmc_hold_abducted":
        out.update(fe=5.0, aa=32.0 + trial)
    elif item == "cmc_palm_arc":
        out.update(ring=4.0 * (1.0 - math.cos(w)), little=6.0 * (1.0 - math.cos(w)))
    return out


def _tap_gap_mm(tau: float) -> float:
    return 60.0 - 55.0 * max(max(0.0, 1.0 - abs(tau - c) / 0.25) for c in (1.5, 3.0, 4.5))


def write_example(out) -> dict:
    """The end-to-end example in `out`: a synthetic tracker session (tracker_session/,
    takes in the recorder's format, sidecars, manifest), a neutral-posture
    tracker calibration (tracker_calibration.json) from its neutral holds, a
    synthetic reference capture (reference/, FORMAT, data_origin synthetic),
    and the comparison (comparison/: report.json, report.md,
    matched_samples.csv, mapping.json). Every file says synthetic. Returns
    a summary of the comparison."""
    out = Path(out)
    ex = EXAMPLE
    hand, session = ex["hand"], ex["session"]
    rng = np.random.default_rng(ex["seed"])
    order = "flexion_then_abduction"
    tdir = out / "tracker_session" / session
    tdir.mkdir(parents=True, exist_ok=True)
    t_us = ex["tracker_t0_us"]
    frame_id, takes, spans, neutral = 1000, [], [], []
    for item, trial in capture_sequence():
        dur = ex["durations_s"][item.id]
        n = int(dur * 1e6 / ex["period_us"])
        start = t_us
        lines = []
        for k in range(n):
            tau = k * ex["period_us"] / 1e6
            if (item.id, trial) == (ex["gap"]["item"], ex["gap"]["trial"]) \
                    and ex["gap"]["from_s"] <= tau < ex["gap"]["to_s"]:
                frame_id += 1
                continue                                     # frames lost: a gap
            truth = example_truth(item.id, trial, tau, dur)
            fe_t = (ex["tracker_gain_fe"] * truth["fe"] + ex["tracker_fe_from_aa"] * truth["aa"]
                    + rng.normal(0.0, ex["tracker_noise_deg"]))
            aa_t = (truth["aa"] + ex["tracker_aa_from_fe"] * truth["fe"]
                    + rng.normal(0.0, ex["tracker_noise_deg"]))
            pose = synthetic_pose(hand, fe_t, aa_t, order,
                                  tip_gap_mm=_tap_gap_mm(tau) if item.id.startswith("sync")
                                  else None)
            status = 1
            inv = ex["invalid_line"]
            if (item.id, trial) == (inv["item"], inv["trial"]) and \
                    abs(tau - inv["at_s"]) < ex["period_us"] / 2e6:
                status = 0
            lines.append(synthetic_line(hand, 41, frame_id, start + k * ex["period_us"],
                                        pose, status=status, t0_us=ex["tracker_t0_us"]))
            if item.id == "tmc_neutral_hold":
                neutral.append(bone_in_palm(pose[1], "THUMB_METACARPAL"))
            frame_id += 1
        name = f"{item.id}_take{trial}_{hand}.jsonl"
        (tdir / name).write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        meta = {"kind": "reference_validation_take", "protocol": PROTOCOL, "item": item.id,
                "take": trial, "joints": list(item.joints), "isolated": item.isolated,
                "instruction": item.instruction, "participant": ex["participant"],
                "hand": hand, "session": session, "mock": True, "synthetic": True,
                "file": name}
        (tdir / name).with_suffix(META_SUFFIX).write_text(json.dumps(meta, indent=2) + "\n",
                                                           encoding="utf-8")
        takes.append({"file": name, "item": item.id, "take": trial, "frames": len(lines)})
        spans.append((item.id, trial, start, start + (n - 1) * ex["period_us"], dur))
        t_us = start + int((dur + ex["prep_s"]) * 1e6)
    write_manifest(tdir, participant=ex["participant"], hand=hand, session=session,
                   tracker_device="SYNTHETIC tracker (leap_hand.reference_validation)",
                   takes=takes, extra={"protocol": PROTOCOL, "synthetic": True})
    # the neutral-posture tracker calibration (the unvalidated estimate's zero)
    r_rest = te.chordal_mean(neutral)
    cal = te.Calibration(
        R_rest=r_rest, session=session, n_transitions=TRIALS, n_frames=len(neutral),
        per_transition=[], agreement={"median_deg": 0.0, "p90_deg": 0.0, "max_deg": 0.0},
        provenance=f"{te.PROVENANCE_PREFIXES['neutral']} (SYNTHETIC example): chordal "
                   f"mean over the {len(neutral)} frames of the example's tmc_neutral_hold "
                   "takes", kind="neutral", hand=hand)
    cal_path = out / "tracker_calibration.json"
    cal_path.write_text(json.dumps(cal.to_json(), indent=2) + "\n", encoding="utf-8")
    # the reference capture
    to_ref = lambda us: (us / 1e6 - ex["clock_offset_s"]) / ex["clock_scale"]   # noqa: E731
    step = 1.0 / ex["reference_rate_hz"]
    rows, sid = [], 0
    for item_id, trial, a_us, b_us, dur in spans:
        item = CAPTURE_ITEMS[item_id]
        if not item.joints:
            continue
        r0, r1 = to_ref(a_us), to_ref(b_us)

        def row(tau, t0, t1=None, **kw):
            nonlocal sid
            sid += 1
            truth = example_truth(item_id, trial, tau, dur)
            noise = lambda: rng.normal(0.0, ex["reference_noise_deg"])   # noqa: E731
            r = {"sample_id": f"s{sid:05d}", "trial": trial, "movement": item_id,
                 "time_s": f"{t0:.6f}", "time_end_s": "" if t1 is None else f"{t1:.6f}",
                 "tracker_file": "", "valid": 1, "exclusion_reason": ""}
            if item_id == "cmc_palm_arc":
                r.update(ring_cmc_flex_deg=f"{truth['ring'] + noise():.4f}",
                         little_cmc_flex_deg=f"{truth['little'] + noise():.4f}")
            else:     # the file's flexion is positive radial: the canonical value negated
                r.update(tmc_flex_radial_deg=f"{-(truth['fe'] + noise()):.4f}",
                         tmc_abd_deg=f"{truth['aa'] + noise():.4f}")
            r.update(kw)
            rows.append(r)
        if item.isolated is None:                      # a hold: one interval sample
            row(dur / 2.0, r0 + 0.1 * (r1 - r0), r0 + 0.9 * (r1 - r0))
            continue
        t = r0 + 0.1
        while t < r1 - 0.1:
            tau = (t - r0) * ex["clock_scale"]
            kw = {}
            bad = ex["reference_invalid"]
            if (item_id, trial) == (bad["item"], bad["trial"]) and \
                    bad["from_s"] <= tau < bad["to_s"]:
                kw = {"valid": 0, "exclusion_reason": "marker occluded (synthetic)"}
            row(tau, t, **kw)
            mis = ex["reference_missing_aa"]
            if (item_id, trial) == (mis["item"], mis["trial"]) and abs(tau - mis["at_s"]) < step / 2:
                rows[-1]["tmc_abd_deg"] = ""
            t += step
        if item_id == "tmc_abduction_sweep" and trial == 1:
            # a sample labelled flexion that falls in this abduction take
            row(1.0, r0 + 1.0)
            rows[-1]["movement"] = "tmc_flexion_sweep"
        row(dur, r1 + 2.0)                           # after the take: no tracker frames
    events = []
    for take_id, (a_us, dur) in (("sync_start", (spans[0][2], spans[0][4])),
                                 ("sync_end", (spans[-1][2], spans[-1][4]))):
        for k, c in enumerate((1.5, 3.0, 4.5), start=1):
            # the tracker sees the tap at its closest frame; the reference at its own clock
            us = a_us + int(round(c * 1e6 / ex["period_us"])) * ex["period_us"]
            events.append({"id": f"{take_id}_tap{k}",
                           "reference_time_s": round(to_ref(us) + rng.normal(0.0, ex["tap_noise_s"]), 6),
                           "tracker_file": f"{take_id}_take1_{hand}.jsonl", "tap_index": k})
    header = {
        "format": FORMAT, "data_origin": "synthetic", "role": "anatomical_reference",
        "participant": {"id": ex["participant"]}, "hand": hand, "session": session,
        "device": {"kind": "optical_motion_capture", "make": "SYNTHETIC",
                   "model": "leap_hand.reference_validation.write_example", "serial": "none",
                   "software": "leap_hand.reference_validation", "sample_rate_hz":
                   ex["reference_rate_hz"], "derived_from_tracker": False},
        "calibration": {"id": "synthetic-ground-truth", "hand": hand, "provenance":
                        "the generator's ground truth: no device and no device calibration"},
        "tracker_calibration": {"file": "../tracker_calibration.json", "kind": "neutral",
                                "hand": hand, "sha256": sha256_of(cal_path)},
        "clock": {"name": "synthetic reference clock", "units": "s"},
        "sync": {"method": "shared_events", "uncertainty_s": ex["sync_uncertainty_s"],
                 "events": events, "note": "thumb-index taps at the start and the end"},
        "units": {"angle": "deg"},
        "frames": {
            "thumb_tmc": {"proximal_segment": "trapezium (synthetic)",
                          "distal_segment": "first metacarpal (synthetic)",
                          "proximal_frame": "the generator's canonical proximal frame",
                          "distal_frame": "the generator's metacarpal long axis",
                          "zero": {"id": "isb_neutral", "text": "the generator's zero; the "
                                   "neutral hold is at fe 4.3..4.9, aa 6.4..6.8 from it"},
                          "rotation_order": order,
                          "positive": {"flexion_extension": "radial",
                                       "abduction_adduction": "palmar"}},
            **{j: {"proximal_segment": "carpus (synthetic)",
                   "distal_segment": f"{j.split('_')[0]} metacarpal (synthetic)",
                   "proximal_frame": "the generator's carpal frame",
                   "distal_frame": "the generator's metacarpal frame",
                   "zero": {"id": "isb_neutral", "text": "flat palm"},
                   "positive": {"flexion_extension": "palmar"}}
               for j in ("ring_cmc", "little_cmc")}},
        "coordinates": [
            {"column": "tmc_flex_radial_deg", "joint": "thumb_tmc",
             "movement": "flexion_extension"},
            {"column": "tmc_abd_deg", "joint": "thumb_tmc", "movement": "abduction_adduction"},
            {"column": "ring_cmc_flex_deg", "joint": "ring_cmc", "movement": "flexion_extension"},
            {"column": "little_cmc_flex_deg", "joint": "little_cmc",
             "movement": "flexion_extension"}],
        "movements": [{"id": i.id, "isolated": i.isolated,
                       "joints": ["ring_cmc", "little_cmc"] if i.id == "cmc_palm_arc"
                       else list(i.joints)} for i in TRIAL_ITEMS],
        "notes": "SYNTHETIC: generated by leap_hand.reference_validation.write_example to "
                 "exercise the software path. It measures nothing.",
    }
    columns = ["tmc_flex_radial_deg", "tmc_abd_deg", "ring_cmc_flex_deg", "little_cmc_flex_deg"]
    params = json.dumps({k: v for k, v in ex.items()}, indent=2) + "\n"
    write_reference(out / "reference", header, rows, columns,
                    {"source/generator_parameters.json": params})
    ref = load_reference(out / "reference")
    trk = load_tracker_session(tdir)
    cal_back = te.Calibration.from_json(json.loads(cal_path.read_text(encoding="utf-8")))
    report = compare(ref, trk, ex["calibration_trials"], tmc_calibration=cal_back,
                     tmc_calibration_path=cal_path)
    paths = write_report(report, out / "comparison")
    return {"report": report, "paths": paths, "reference": ref, "tracker": trk}
