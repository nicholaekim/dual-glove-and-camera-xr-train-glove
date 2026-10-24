"""The thumb TMC pair ESTIMATED against the session's recorded open hand.

NOT A VALIDATED PAPER 3 COORDINATE. The numbers here are relative to the
open hand the participant held before each transition cue, read off the
tracker's thumb metacarpal against its palm. Paper 3 (Cobos et al. 2009)
gives its TMC zero only as a drawing (Figure 1, p.3), so these differ from
the paper's T_TMC_fe / T_TMC_aa by an unknown constant per coordinate. They
are kept under their own names (`*_est_openhand`), never in the 24, and
`paper3.compare` keeps the Table 6 relation T_TMC_fe ~ 11/10 T_MCP_fe
`unavailable` (tests/test_tmc_estimate.py checks it). Plan:
`docs/angle_coverage_plan.md`, "The thumb TMC estimate (tmc leaf)"; results
and measured values: `docs/tmc_estimate.md`.

MAPPING
  The paper's thumb chain (its Table 3, p.5) turns the TMC about an
  abduction axis first (joint 1, alpha = pi/2), then about a flexion axis
  parallel to the MCP and IP flexion axes (joints 2 to 4, alpha = 0). The
  tracker's thumb bone frames have parallel x axes (assumption A1), so its
  metacarpal x is that flexion axis. With R_rest the metacarpal's
  orientation against the palm in the open hand:

      R_pm     = R_palm^T R_metacarpal                 palm_to_metacarpal
      R_tmc    = R_rest^T R_pm
      (fe, aa) = joint_frames.paper_order_angles(R_tmc, hand)
                 R_tmc = Ry(beta) Rx(alpha) [Rz(gamma)], fe = -alpha,
                 aa = s beta, s = hand_sign (+1 right, -1 left)
      R_part   = Ry(s aa) Rx(-fe)                      abduction, then flexion
      residual = R_part^T R_tmc                        a turn about the bone (z)
      twist    = -s atan2(residual[1,0], residual[0,0])   (degrees)
      residual_deg = the full rotation angle of `residual`

  R_part reproduces R_tmc's bone axis (its z column) exactly, so the residual
  is a pure turn about the bone and residual_deg = |twist| up to rounding;
  both are reported so a reader can check it. The twist carries the sign
  `joint_frames.joint_angles` gives every twist (positive = the back of the
  bone turns toward the little finger, both hands), so
  `R_rest Ry(s aa) Rx(-fe) Rz(-s tw)` reads back as (fe, aa, tw).
  `estimate` returns the four as T_TMC_fe_est_openhand,
  T_TMC_aa_est_openhand, T_TMC_twist_est_openhand and
  T_TMC_residual_est_openhand_deg, plus the calibration's provenance.

SIGNS (pinned geometrically by tests/test_tmc_estimate.py)
  Positive flexion moves the metacarpal's bone direction (its -z) toward
  the rest frame's -y, the metacarpal's own palmar side, as for every joint
  of the repository; with this session's rest frame that is toward the
  palm's -y (palmar, 0.47 of the motion), mostly toward the palm's little-
  finger side (0.74: across the palm) and distally (0.49).
  Positive abduction moves the bone toward the rest frame's thumb-side x
  (+x on a left hand, -x on a right hand: the side `joint_frames` calls
  radial for every bone); with this session's rest frame that is mostly
  toward the palm's -y (0.81: palmar abduction, the thumb leaving the
  palm's plane on the palm side), toward the wrist (0.57) and slightly
  toward the palm's thumb side (0.14, radial). Both hold for a left hand
  and for its mirror image, a right hand. (Component sizes: the unit
  motion direction in the palm frame for the measured R_rest, left hand.)

CALIBRATION (`calibrate`)
  R_rest is the chordal mean (`chordal_mean`) of R_pm over the open-hand
  frames of the session's transitions: frames whose timeline time is before
  `movement_start_time_s - OPEN_HAND_MARGIN_S`, transitions with a blank
  movement start skipped, windows under MIN_FRAMES frames skipped. First one
  mean per transition, then the mean of those means, so every transition
  weighs once however long its window. The record (`Calibration`) keeps the
  count of transitions and frames, every per-transition spread, and the
  across-transition agreement (median, p90, max angle of the per-transition
  means to R_rest), which is the estimate's floor uncertainty.

  Time origins. `rec.timeline.t` is seconds from the first SELECTED frame
  on the camera clock (timestamp_us); the metadata's movement start is
  seconds from `first_frame_capture_time` on the capture clock. For these
  files the two origins coincide: the first selected frame's capture_time
  equals first_frame_capture_time on all 52 transitions (difference 0.000
  ms), and over the 42 open-hand windows origin and clocks together differ
  by at most 0.18 ms against an 11.1 ms frame period (`timeline_origin_ms`;
  the report writes it to assumption_checks.json on every run).

ASSUMPTIONS (measured on the real data by scripts/leap/tmc_estimate_report.py,
values in docs/tmc_estimate.md)
  A1 the thumb's hinge axes are parallel: the x axes of the metacarpal,
     proximal and distal frames (`axis_parallelism`, every frame).
  A2 the open-hand rest orientation is stable across the session (the
     agreement statistics above; a shifted transition shows in them).
  A3 the open hand is the zero. Paper 3 fixes no numeric zero (Figure 1 is
     a drawing, Table 5 gives ranges), so the estimate is relative to the
     recorded open hand and differs from the paper's by an unknown constant.
  A4 the tracker's thumb metacarpal is tracked, not a rigid part of its palm
     (`metacarpal_motion`: within-take spread of R_pm and its angle to R_rest).
  A5 the palm frame stands for the carpus: the trapezium is not tracked, so
     the TMC is measured against LeapC's rigid palm.

UNCERTAINTY
  At least the across-transition agreement of the rest frame (median
  4.4, p90 7.8, max 20.0 degrees on this session; the report measures it
  again every run), on top of an unknown constant offset from Paper 3's zero
  (A3) and the tracker's own error on the thumb, which this dataset has no
  reference to measure.

Pure apart from `calibrate` (and `metacarpal_motion` / `series`, which take
a loaded recording), which reads the dataset through `replay_core` and
writes nothing.
"""
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Sequence

import numpy as np

from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX

from . import replay_core as rc
from .replay_core import rotation_angle_deg  # noqa: F401  (part of this API)

PALM = JOINT_INDEX["PALM"]
THUMB_METACARPAL = JOINT_INDEX["THUMB_METACARPAL"]
THUMB_PROXIMAL = JOINT_INDEX["THUMB_PROXIMAL"]
THUMB_DISTAL = JOINT_INDEX["THUMB_DISTAL"]
OPEN_HAND_MARGIN_S = 0.1
MIN_FRAMES = 20
VALIDITIES = ("valid", "questionable", "invalid")

FE = "T_TMC_fe_est_openhand"
AA = "T_TMC_aa_est_openhand"
TWIST = "T_TMC_twist_est_openhand"
RESIDUAL = "T_TMC_residual_est_openhand_deg"
KEYS = (FE, AA, TWIST, RESIDUAL)
AXIS_KEYS = ("metacarpal_proximal_deg", "proximal_distal_deg",
             "metacarpal_palm_deg")


# --- rotations ----------------------------------------------------------------------
def palm_to_metacarpal(rot) -> np.ndarray:
    """The thumb metacarpal's orientation in the PALM joint's frame:
    rot[PALM]^T rot[THUMB_METACARPAL], from world rotations (26, 3, 3)."""
    rot = np.asarray(rot, dtype=float)
    return rot[PALM].T @ rot[THUMB_METACARPAL]


def chordal_mean(rots) -> np.ndarray:
    """The rotation nearest (Frobenius) to the mean of the matrices: the SVD
    projection of their average onto SO(3), det +1 enforced."""
    r = np.asarray(rots, dtype=float).reshape(-1, 3, 3)
    if len(r) == 0:
        raise ValueError("chordal_mean of no rotations")
    u, _s, vt = np.linalg.svd(r.mean(axis=0))
    d = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def _axis_angle_deg(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)),
                                       float(np.dot(a, b)))))


def axis_parallelism(rot) -> Dict[str, float]:
    """A1 on one frame: degrees between the x (hinge) axes of the thumb's
    metacarpal and proximal frames, of its proximal and distal frames, and
    between the metacarpal's x and the palm's x."""
    rot = np.asarray(rot, dtype=float)
    x = {j: rot[j][:, 0] for j in (PALM, THUMB_METACARPAL, THUMB_PROXIMAL,
                                   THUMB_DISTAL)}
    return {"metacarpal_proximal_deg": _axis_angle_deg(x[THUMB_METACARPAL],
                                                       x[THUMB_PROXIMAL]),
            "proximal_distal_deg": _axis_angle_deg(x[THUMB_PROXIMAL],
                                                   x[THUMB_DISTAL]),
            "metacarpal_palm_deg": _axis_angle_deg(x[THUMB_METACARPAL],
                                                   x[PALM])}


# --- the open-hand window ---------------------------------------------------------------
def _blank(v) -> bool:
    return v is None or str(v).strip() == ""


def open_hand_window(rec, margin_s: float = OPEN_HAND_MARGIN_S) -> List[int]:
    """Selected-frame indices of the open hand before the cue's movement:
    `rec.timeline.t < movement_start_time_s - margin_s`, the start read from
    `rec.metadata` (the pose_transitions.csv row). Blank or absent start:
    an empty list.

    The timeline starts at the recording's first selected frame (camera
    clock); the metadata times are seconds from `first_frame_capture_time`
    (capture clock). For these files the two origins coincide within one
    frame period (module docstring; `timeline_origin_ms` measures it).
    """
    start = rec.metadata.get("movement_start_time_s")
    if _blank(start):
        return []
    limit = float(start) - margin_s
    return [int(i) for i in np.flatnonzero(np.asarray(rec.timeline.t) < limit)]


def timeline_origin_ms(rec, margin_s: float = OPEN_HAND_MARGIN_S
                       ) -> Optional[dict]:
    """How far the metadata's time origin is from the timeline's, in ms.

    first_frame_ms: the first selected frame's capture_time minus the row's
    first_frame_capture_time. window_max_ms: over the open-hand window, the
    largest |t - (capture_time - first_frame_capture_time)|, i.e. the two
    origins and the two clocks together. None when the row or the frames
    lack the times.
    """
    ffc = rec.metadata.get("first_frame_capture_time")
    if _blank(ffc) or rec.frames[0].capture_time is None:
        return None
    ffc = float(ffc)
    window = open_hand_window(rec, margin_s)
    caps = [rec.frames[i].capture_time for i in window]
    diffs = [abs(float(rec.timeline.t[i]) - (c - ffc)) * 1e3
             for i, c in zip(window, caps) if c is not None]
    return {"first_frame_ms": (rec.frames[0].capture_time - ffc) * 1e3,
            "window_frames": len(diffs),
            "window_max_ms": max(diffs) if diffs else None}


# --- the calibration --------------------------------------------------------------------
@dataclass
class Calibration:
    """The open-hand rest frame and how well it is known (module docstring).

    per_transition: one dict per transition used (name, take, validity,
    n_frames, spread_median_deg, spread_max_deg, angle_to_session_mean_deg,
    rest_xyz_deg). agreement: median_deg, p90_deg, max_deg and worst
    {name, take} of angle_to_session_mean_deg. skipped: one dict per
    transition row not used (name, take, validity, reason).
    """
    R_rest: np.ndarray
    session: str
    n_transitions: int
    n_frames: int
    per_transition: List[dict]
    agreement: dict
    provenance: str
    skipped: List[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        """JSON-able dict; `from_json` reads it back. R_rest is written in
        full float precision, plus its xyz angles in degrees for reading."""
        d = asdict(self)
        d["R_rest"] = [[float(v) for v in row] for row in self.R_rest]
        d["R_rest_xyz_deg"] = [math.degrees(a) for a in jf.xyz_angles(self.R_rest)]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Calibration":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        kw["R_rest"] = np.asarray(d["R_rest"], dtype=float)
        return cls(**kw)


def _label(row: dict) -> dict:
    return {"name": row["to_grasp"], "take": int(row["take"]),
            "validity": row.get("validity", "")}


def calibrate(dataset_root, *, min_frames: int = MIN_FRAMES,
              validities: Sequence[str] = VALIDITIES,
              session: Optional[str] = None,
              margin_s: float = OPEN_HAND_MARGIN_S) -> Calibration:
    """R_rest from the open-hand frames of the dataset's transitions.

    Rows are skipped (and listed in `skipped` with the reason) when they
    belong to another session than `session`, their validity is not in
    `validities`, their movement start is blank, or their open-hand window
    has fewer than `min_frames` frames. Every remaining transition is one
    session; with more than one and no `session=`, a ValueError.
    """
    validities = tuple(validities)
    candidates, skipped = [], []
    for row in rc.list_transitions(dataset_root):
        label = _label(row)
        if session is not None and row.get("session", "") != session:
            skipped.append({**label, "reason": f"session {row.get('session', '')!r}, "
                                               f"not {session!r}"})
        elif label["validity"] not in validities:
            skipped.append({**label, "reason": f"validity {label['validity']!r} "
                                               f"not in {list(validities)}"})
        elif _blank(row.get("movement_start_time_s")):
            skipped.append({**label, "reason": "blank movement_start_time_s"})
        else:
            candidates.append(row)
    sessions = sorted({row.get("session", "") for row in candidates})
    if len(sessions) > 1:
        raise ValueError(f"transitions from {len(sessions)} sessions ({sessions}): "
                         "one rest frame per session, pass session=")
    used, means = [], []
    for row in candidates:
        label = _label(row)
        rec = rc.find_transition(dataset_root, row["to_grasp"], row["take"])
        window = open_hand_window(rec, margin_s)
        if len(window) < min_frames:
            skipped.append({**label, "reason": f"open-hand window {len(window)} "
                                               f"frames < {min_frames}"})
            continue
        pm = np.array([palm_to_metacarpal(rc.reconstruct(rec.frames[i]).rot)
                       for i in window])
        mean = chordal_mean(pm)
        spread = np.atleast_1d(rotation_angle_deg(mean, pm))
        means.append(mean)
        used.append({**label, "n_frames": len(window),
                     "spread_median_deg": float(np.median(spread)),
                     "spread_max_deg": float(np.max(spread)),
                     "rest_xyz_deg": [math.degrees(a) for a in jf.xyz_angles(mean)]})
    if not used:
        raise ValueError(f"no usable transition in {dataset_root} "
                         f"({len(skipped)} skipped)")
    r_rest = chordal_mean(means)
    angles = np.atleast_1d(rotation_angle_deg(r_rest, np.array(means)))
    for u, a in zip(used, angles):
        u["angle_to_session_mean_deg"] = float(a)
    worst = used[int(np.argmax(angles))]
    agreement = {"median_deg": float(np.median(angles)),
                 "p90_deg": float(np.percentile(angles, 90)),
                 "max_deg": float(np.max(angles)),
                 "worst": {"name": worst["name"], "take": worst["take"]}}
    session_name = session if session is not None else (sessions[0] if sessions else "")
    n_frames = sum(u["n_frames"] for u in used)
    provenance = (
        f"open-hand calibration of session {session_name}: R_rest = chordal mean "
        f"of R_palm^T R_thumb_metacarpal over the frames before "
        f"movement_start_time_s - {margin_s:g} s of {len(used)} transitions "
        f"({n_frames} frames, >= {min_frames} per transition, validity in "
        f"{'/'.join(validities)}); across-transition agreement median "
        f"{agreement['median_deg']:.1f}, p90 {agreement['p90_deg']:.1f}, max "
        f"{agreement['max_deg']:.1f} deg ({worst['name']} take {worst['take']}); "
        f"relative to the recorded open hand, not a validated Paper 3 coordinate")
    return Calibration(R_rest=r_rest, session=session_name,
                       n_transitions=len(used), n_frames=n_frames,
                       per_transition=used, agreement=agreement,
                       provenance=provenance, skipped=skipped)


# --- the estimate -----------------------------------------------------------------------
def estimate(pos, rot, hand_side: str, calibration: Calibration) -> dict:
    """The open-hand TMC estimate for one pose (module docstring, MAPPING).

    pos (26, 3) is accepted for symmetry with `paper3.frame_angles` and not
    used: the estimate depends on orientations only. Returns
    T_TMC_fe_est_openhand, T_TMC_aa_est_openhand, T_TMC_twist_est_openhand,
    T_TMC_residual_est_openhand_deg (degrees) and `calibration`, the
    calibration's provenance text.
    """
    s = jf.hand_sign(hand_side)
    r_tmc = np.asarray(calibration.R_rest, dtype=float).T @ palm_to_metacarpal(rot)
    fe, aa = jf.paper_order_angles(r_tmc, hand_side)
    part = jf.rot_y(s * aa) @ jf.rot_x(-fe)
    res = part.T @ r_tmc
    twist = -s * math.degrees(math.atan2(res[1, 0], res[0, 0]))
    return {FE: fe, AA: aa, TWIST: twist + 0.0,
            RESIDUAL: float(rotation_angle_deg(part, r_tmc)),
            "calibration": calibration.provenance}


def series(rec, calibration: Calibration) -> Dict[str, np.ndarray]:
    """`estimate` over every selected frame of a recording: KEYS -> arrays."""
    out = {k: np.empty(len(rec.frames)) for k in KEYS}
    for i, frame in enumerate(rec.frames):
        r = rc.reconstruct(frame)
        e = estimate(r.pos, r.rot, rec.hand_side, calibration)
        for k in KEYS:
            out[k][i] = e[k]
    return out


# --- assumption A4 ----------------------------------------------------------------------
def rotation_spread(rots, calibration: Optional[Calibration] = None) -> dict:
    """Spread of a stack of palm-to-metacarpal rotations: the angle of each
    to their chordal mean (median, max), and with a calibration the angle of
    that mean to R_rest and the largest angle of a single frame to R_rest."""
    r = np.asarray(rots, dtype=float).reshape(-1, 3, 3)
    mean = chordal_mean(r)
    spread = np.atleast_1d(rotation_angle_deg(mean, r))
    out = {"n_frames": len(r), "spread_median_deg": float(np.median(spread)),
           "spread_max_deg": float(np.max(spread)),
           "mean_to_rest_deg": None, "max_frame_to_rest_deg": None}
    if calibration is not None:
        rest = np.asarray(calibration.R_rest, dtype=float)
        out["mean_to_rest_deg"] = float(rotation_angle_deg(rest, mean))
        out["max_frame_to_rest_deg"] = float(np.max(np.atleast_1d(
            rotation_angle_deg(rest, r))))
    return out


def metacarpal_motion(rec, calibration: Optional[Calibration] = None) -> dict:
    """A4 on one recording: `rotation_spread` of R_palm^T R_metacarpal over
    every selected frame."""
    return rotation_spread([palm_to_metacarpal(rc.reconstruct(f).rot)
                            for f in rec.frames], calibration)
