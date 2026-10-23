"""Paper 3's joint relationships set against recorded camera angles.

Cobos et al. 2009, "Human Hand Descriptions and Gesture Recognition for
Object Manipulation" (Computer Methods in Biomechanics and Biomedical
Engineering; `paper3 (3).pdf`). Pages below are the printed page numbers,
which are also the PDF's. This is a research comparison, not a fidelity
check: the paper supplies no pass/fail threshold for any relationship, so
none is invented here; what comes out are residuals and their statistics.
Pure numpy: no files, no window. Poses come in already reconstructed, in any
world frame; this module reads no recordings.

WHAT THE PAPER GIVES, AND WHERE
  Table 5, p.12      active/passive ranges per joint: one column holds the
                     row's own range (flexion/extension or abduction/
                     adduction), one the hyperextension. No signed minimum or
                     maximum, so display only (TABLE5_NOTE).
  Eq. 3-7, p.13      inter-finger coupling measured with a Cyberglove while
                     ONE finger flexes alone (Figure 8, p.14). A grasp flexes
                     every finger, so eq. 3, 4, 5 and 7 are listed and never
                     applied; eq. 6 (ring and little abduction alike, "in most
                     cases") is reported as informational.
  Eq. 8, p.14        PIP ~ 3/2 DIP, "the most accepted intra-finger
                     constraint": the one relation a grasp with no Paper 3
                     class gets.
  Table 6, p.15      the intra-finger constraints that rebuild a circular
                     grasp. Its grey cells are the 9 independent angles of the
                     9-DoF description (pp.14-15), every white cell one
                     dependent angle.
  Table 7, p.16      prismatic grasps, thumb, index and middle. Section 5.1
                     (p.16) keeps Table 6 for ring and little and rebuilds a
                     prismatic grasp from Tables 6 and 7 together, so Table 7
                     replaces the five Table 6 flexion relations of thumb,
                     index and middle and every other Table 6 relation stands.
  Eq. 1-2, Tables 2-3, p.5   the D-H finger and thumb chains.
  Eq. 9-11, pp.16-17 fingertip relative error of a simplified description;
                     Table 8 (p.17) has the paper's 9-DoF values, 7.67 %
                     circular and 5.90 % prismatic (Cyberglove, five fingers).

CLASSES (grasp_class)
  p3_circular_* -> circular and p3_prismatic_* -> prismatic: recorded under
  the paper's own labels (its simplified Cutkosky table, Figure 4, p.8).
  Everything else -> unmapped: p1_* and p2_* were recorded under the Paper 1
  and Paper 2 taxonomies, and no record of the objects' shapes supports a
  circular/prismatic assignment.

ANGLES
  frame_angles reads the 24 names off xr_hand.joint_frames (table_from_world
  with the wrist reference, then dof24), so their axes and signs are that
  module's: flexion positive toward the palm, abduction positive toward the
  thumb, both hands. Six are not given by the camera (UNAVAILABLE =
  joint_frames.CAMERA_NOT_GIVEN: the four CMC, which LeapC's rigid palm does
  not measure, and the thumb's TMC pair, taken about other axes than the
  paper's): None for a frame, NaN in a series, and a relation that needs one
  is reported 'unavailable' with no numbers. The four finger MCP pairs also
  come in the paper's order (abduction first, as its D-H chain turns them,
  Table 2: joint_frames.paper_order_angles) as `<i>_MCP_fe_paper_order` and
  `<i>_MCP_aa_paper_order`; compare takes those as the primary MCP value and
  reports the main XYZ-order value beside them. The thumb MCP is a 1-DoF
  joint in the paper (Table 3) and has no second reading. How closely either
  reading lines up anatomically with the paper's Cyberglove angles is
  approximate.

D-H (p.5)
  The matrix printed on p.5 has sin(alpha) sin(alpha) in row 1 column 3 and
  -sin(alpha) cos(alpha) in row 2 column 3; the standard transform the paper
  cites (Denavit and Hartenberg 1955; Spong et al. 2006) has
  sin(alpha) sin(theta) and -sin(alpha) cos(theta): the print has alpha
  where theta belongs. Its rotation block is a rotation only when
  sin(alpha) = 0 or theta = alpha, and then it is the standard one; at
  Table 2's own alpha = +-pi/2 its first and third columns have the dot
  product cos(theta) (at theta = 0 they coincide). dh() is the standard
  matrix and the one used; dh_as_printed() keeps the misprint for the record.
  That is the one correction made to the paper.

EQ. 9-11 HERE
  fingertip_reconstruction_error rebuilds every frame from the 9 independent
  angles through Table 6 (circular) or Tables 6 + 7 (prismatic) and compares
  fingertips through the Table 2 chain, as section 5.1 does, with three
  differences a reader must know:
    * CMC is not given by the camera: 0 in the original and in the
      reconstruction alike, so the palm-arc part of the paper's error is not
      in these numbers.
    * u_i, the wrist-to-finger-base offset of eq. 1, has no value in the paper
      and is left out: positions are relative to each finger's base frame
      (D-H frame 0). u_i would cancel in |rp - op| (both tips move with it)
      but not in |rp|, so the percentage is relative to the distance from
      the finger base, not from the wrist. Eq. 9 is otherwise as printed,
      |rp| in the denominator: where the rebuilt finger curls back toward
      its base |rp| is small and delta passes 100 %.
    * The thumb is unavailable (T_TMC_aa is one of its two independent angles
      and its TMC pair is not given), so eq. 11 averages four fingers, not the
      paper's five.
  The error does not depend on the sign convention of the flexions or of the
  abduction: flipping either mirrors both tips, and eq. 9 is a ratio of
  lengths. So joint_frames' signed angles go into the chain as they are.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX
from xr_hand.kinematics import PARENT

CLASSES = ("circular", "prismatic", "unmapped")
STATUSES = ("applied", "not_applied", "informational", "unavailable")
FINGERS = ("I", "M", "R", "L")              # the paper's letters, thumb apart
UNAVAILABLE = jf.CAMERA_NOT_GIVEN
ANGLE_NAMES: List[str] = list(jf.DOF24_NAMES) + list(jf.MCP_PAPER_NAMES)


def grasp_class(name: str) -> Tuple[str, str]:
    """(class, reason) for a dataset grasp name (module docstring, CLASSES)."""
    name = str(name)
    for cls in ("circular", "prismatic"):
        if name.startswith(f"p3_{cls}_"):
            return cls, (f"recorded under Paper 3's own '{cls}' label "
                         "(its simplified Cutkosky table, Figure 4, p.8)")
    if name[:3] in ("p1_", "p2_"):
        origin = f"was recorded under the Paper {name[1]} taxonomy"
    else:
        origin = "is not one of Paper 3's circular or prismatic grasps"
    return "unmapped", (f"{name} {origin}; no object-shape record supports a "
                        "Cutkosky circular/prismatic assignment, so only "
                        "eq. 8 (p.14, general) is applied")


# --- the relationships ----------------------------------------------------------------
@dataclass(frozen=True)
class Relation:
    """lhs ~ coeff * rhs (kind 'ratio'), or lhs - coeff * rhs <= bound_deg
    (kind 'bound', eq. 5), between two of the 24 names. `applies_to` holds
    the grasp classes the relation is the paper's model for: empty = never
    applied (it needs one finger flexing alone), {'informational'} = shown
    for every class, never applied."""
    id: str
    source: str
    page: int
    lhs: str
    coeff: Fraction
    rhs: str
    applies_to: FrozenSet[str]
    condition: Optional[str]
    note: str
    kind: str = "ratio"
    bound_deg: Optional[float] = None


_C, _P, _U = "circular", "prismatic", "unmapped"
_T6_ONLY = ("circular only: for a prismatic grasp Table 7 replaces it "
            "(p.16, section 5.1)")
_T6_RL = ("circular and prismatic: ring and little keep Table 6 in a "
          "prismatic grasp (p.16, section 5.1)")
_T6_KEPT = ("circular and prismatic: Table 7 gives no replacement, and a "
            "prismatic grasp is rebuilt from Tables 6 and 7 together (p.16)")
_T7 = "prismatic only: thumb, index and middle (p.16, section 5.1)"
_EQ8 = ("the most accepted intra-finger constraint (p.14): the only relation "
        "a grasp with no Paper 3 class gets; for a circular grasp Table 6 "
        "gives the same; not for a prismatic grasp, where Table 7 has "
        "PIP ~ 2 DIP for index and middle")
_SINGLE = ("inter-finger coupling measured with a Cyberglove while one finger "
           "flexes alone (section 4.2.1, p.13; Figure 8, p.14); a grasp flexes "
           "every finger, so it is listed and never applied")


def _rel(tag, source, page, lhs, coeff, rhs, applies_to, note,
         condition=None, kind="ratio", bound=None) -> Relation:
    return Relation(f"{tag}:{lhs}:{rhs}", source, page, lhs, Fraction(coeff),
                    rhs, frozenset(applies_to), condition, note, kind, bound)


RELATIONS: List[Relation] = []
for _f in FINGERS:                                   # Table 6, p.15
    _cls, _why = ({_C, _P}, _T6_RL) if _f in "RL" else ({_C}, _T6_ONLY)
    RELATIONS += [
        _rel("T6", "Table 6", 15, f"{_f}_MCP_fe", Fraction(4, 3), f"{_f}_PIP",
             _cls, _why),
        _rel("T6", "Table 6", 15, f"{_f}_PIP", Fraction(3, 2), f"{_f}_DIP",
             _cls, _why + "; same coefficient as eq. 8 (p.14)"),
    ]
RELATIONS += [
    _rel("T6", "Table 6", 15, "T_MCP_fe", Fraction(4, 5), "T_IP", {_C},
         _T6_ONLY),
    _rel("T6", "Table 6", 15, "T_TMC_fe", Fraction(11, 10), "T_MCP_fe",
         {_C, _P}, _T6_KEPT),
    _rel("T6", "Table 6", 15, "I_CMC", 1, "M_CMC", {_C, _P}, _T6_KEPT),
    _rel("T6", "Table 6", 15, "M_CMC", Fraction(1, 2), "R_CMC", {_C, _P},
         _T6_KEPT),
    _rel("T6", "Table 6", 15, "R_CMC", Fraction(2, 3), "L_CMC", {_C, _P},
         _T6_KEPT),
    _rel("T6", "Table 6", 15, "M_MCP_aa", Fraction(1, 5), "I_MCP_aa", {_C, _P},
         _T6_KEPT),
    _rel("T6", "Table 6", 15, "R_MCP_aa", Fraction(1, 2), "L_MCP_aa", {_C, _P},
         _T6_KEPT),
    _rel("T7", "Table 7", 16, "T_MCP_fe", Fraction(6, 5), "T_IP", {_P}, _T7),
    _rel("T7", "Table 7", 16, "I_MCP_fe", Fraction(3, 2), "I_PIP", {_P}, _T7),
    _rel("T7", "Table 7", 16, "M_MCP_fe", Fraction(3, 2), "M_PIP", {_P}, _T7),
    _rel("T7", "Table 7", 16, "I_PIP", 2, "I_DIP", {_P}, _T7),
    _rel("T7", "Table 7", 16, "M_PIP", 2, "M_DIP", {_P}, _T7),
]
RELATIONS += [_rel("EQ8", "Eq. 8 (general)", 14, f"{_f}_PIP", Fraction(3, 2),
                   f"{_f}_DIP", {_U, _C}, _EQ8) for _f in FINGERS]
RELATIONS += [
    _rel("EQ3", "Eq. 3", 13, "R_MCP_fe", Fraction(2, 3), "M_MCP_fe", (),
         _SINGLE, "flexion of the middle finger alone, index and little not "
                  "flexed (pp.12-13, Figure 8a)"),
    _rel("EQ4", "Eq. 4", 13, "R_MCP_fe", Fraction(7, 12), "L_MCP_fe", (),
         _SINGLE, "flexion of the little finger alone (p.13, Figure 8b)"),
    _rel("EQ5a", "Eq. 5", 13, "R_MCP_fe", 1, "M_MCP_fe", (), _SINGLE,
         "flexion of the ring finger alone (p.13, Figure 8c)", "bound", 60.0),
    _rel("EQ5b", "Eq. 5", 13, "R_MCP_fe", 1, "L_MCP_fe", (), _SINGLE,
         "flexion of the ring finger alone (p.13, Figure 8c)", "bound", 50.0),
    _rel("EQ6", "Eq. 6", 13, "R_MCP_aa", 1, "L_MCP_aa", {"informational"},
         "Table 6 (p.15) has R_MCP_aa ~ 1/2 L_MCP_aa for grasps; eq. 6 is the "
         "inter-finger coupling, shown for information only",
         "ring and little abduction move together in most cases "
         "(p.13, Figure 8d)"),
    _rel("EQ7", "Eq. 7", 13, "M_MCP_fe", Fraction(1, 5), "I_MCP_fe", (),
         _SINGLE, "flexion of the index finger alone (p.13, Figure 8e)"),
]
RELATION_BY_ID: Dict[str, Relation] = {r.id: r for r in RELATIONS}
assert len(RELATION_BY_ID) == len(RELATIONS) == 30


def coefficient(relation_id: str) -> Fraction:
    """The printed coefficient of one relation, exactly (a Fraction)."""
    return RELATION_BY_ID[relation_id].coeff


# Table 5, p.12, as printed: name -> (the row's own range, its hyperextension).
_TABLE5_ROWS = {
    "T_TMC_fe": ("50°–90°", "15°"), "T_TMC_aa": ("45°–60°", "–"),
    "T_MCP_fe": ("75°–80°", "0°"), "T_IP": ("75°–80°", "5°–10°"),
    "I_CMC": ("5°", "0°"), "I_MCP_fe": ("90°", "30°–40°"),
    "I_MCP_aa": ("60°", "–"), "I_PIP": ("110°", "0°"), "I_DIP": ("80°–90°", "5°"),
    "M_CMC": ("5°", "0°"), "M_MCP_fe": ("90°", "30°–40°"),
    "M_MCP_aa": ("45°", "–"), "M_PIP": ("110°", "0°"), "M_DIP": ("80°–90°", "5°"),
    "R_CMC": ("10°", "0°"), "R_MCP_fe": ("90°", "30°–40°"),
    "R_MCP_aa": ("45°", "–"), "R_PIP": ("120°", "0°"), "R_DIP": ("80°–90°", "5°"),
    "L_CMC": ("15°", "0°"), "L_MCP_fe": ("90°", "30°–40°"),
    "L_MCP_aa": ("50°", "–"), "L_PIP": ("135°", "0°"), "L_DIP": ("90°", "5°"),
}
TABLE5: Dict[str, dict] = {
    name: {"flexion_range": None if name.endswith("_aa") else rng,
           "abduction_range": rng if name.endswith("_aa") else None,
           "hyperextension": hyper, "page": 12}
    for name, (rng, hyper) in _TABLE5_ROWS.items()}
TABLE5_NOTE = (
    "Table 5 (p.12) as printed, for display only. Its two columns are "
    "'Flexion/Extension or Abduction/adduction' (the row's own motion: "
    "flexion_range for a flexion row, abduction_range for an abduction row, "
    "None for the other key) and 'Hyperextension'. The table gives ranges "
    "without a signed minimum, maximum or zero convention, so no bound test "
    "is built from it. The scaphoid-trapezium row (0°, 0°) is not one of the "
    "24 and is left out.")


# --- angles ---------------------------------------------------------------------------
def frame_angles(pos, rot, hand_side: str) -> Dict[str, Optional[float]]:
    """The 24 paper angles (degrees) plus the 8 paper-order MCP values for one
    pose: pos (26, 3) metres and rot (26, 3, 3), in any world frame.

    The 24 come from joint_frames.table_from_world (wrist reference) and
    dof24; the six in UNAVAILABLE are None. `<i>_MCP_fe_paper_order` and
    `<i>_MCP_aa_paper_order` are paper_order_angles of each finger's
    proximal phalanx against its metacarpal, as joint_frames.mcp_paper_order
    computes them from a frame line.
    """
    pos = np.asarray(pos, dtype=float)
    rot = np.asarray(rot, dtype=float)
    if pos.shape != (26, 3) or rot.shape != (26, 3, 3):
        raise ValueError("need pos (26, 3) and rot (26, 3, 3), got "
                         f"{pos.shape} and {rot.shape}")
    out: Dict[str, Optional[float]] = dict(
        jf.dof24(jf.table_from_world(pos, rot, hand_side, "wrist")))
    for name in UNAVAILABLE:
        out[name] = None
    for letter, finger in jf.FINGERS[1:]:
        j = JOINT_INDEX[f"{finger}_PROXIMAL"]
        fe, aa = jf.paper_order_angles(rot[PARENT[j]].T @ rot[j], hand_side)
        out[f"{letter}_MCP_fe_paper_order"] = fe
        out[f"{letter}_MCP_aa_paper_order"] = aa
    return out


def series_angles(poses: Sequence, hand_side: str) -> Dict[str, np.ndarray]:
    """frame_angles over a sequence of (pos, rot): name -> float array, NaN
    where the camera gives nothing (UNAVAILABLE)."""
    cols: Dict[str, list] = {name: [] for name in ANGLE_NAMES}
    for pos, rot in poses:
        frame = frame_angles(pos, rot, hand_side)
        for name in ANGLE_NAMES:
            cols[name].append(np.nan if frame[name] is None else frame[name])
    return {name: np.asarray(v, dtype=float) for name, v in cols.items()}


def finger_bone_lengths_mm(pos) -> Dict[str, Tuple[float, float, float, float]]:
    """(L_Me, L_P, L_Mi, L_D) per finger letter in mm from one pose (26, 3) in
    metres: the Table 2 (p.5) link lengths, i.e. the distances METACARPAL ->
    PROXIMAL -> INTERMEDIATE -> DISTAL -> TIP. The shape
    fingertip_reconstruction_error takes."""
    pos = np.asarray(pos, dtype=float)
    parts = ("METACARPAL", "PROXIMAL", "INTERMEDIATE", "DISTAL", "TIP")
    out = {}
    for letter, finger in jf.FINGERS[1:]:
        idx = [JOINT_INDEX[f"{finger}_{p}"] for p in parts]
        out[letter] = tuple(float(np.linalg.norm(pos[b] - pos[a]) * 1000.0)
                            for a, b in zip(idx, idx[1:]))
    return out


# --- statistics -----------------------------------------------------------------------
_STAT_KEYS = ("mean", "mae", "rmse", "p05", "p50", "p95", "max_abs", "slope",
              "intercept", "r")


def residual_stats(lhs, rhs, coeff, bound: Optional[float] = None) -> dict:
    """Statistics of r = lhs - coeff * rhs (degrees) over the pairs where both
    are finite.

    Ratio relations: n, mean, mae, rmse, p05/p50/p95 (numpy's default linear
    interpolation), max_abs, and the empirical OLS line lhs = slope * rhs +
    intercept with Pearson r; those three are None below 3 points or when
    rhs (for r: either side) has no variance. With `bound` (eq. 5):
    n, bound_deg, fraction_satisfied (share of pairs with r <= bound) and
    max_excess (largest r - bound; <= 0 when every pair is within it).
    """
    y = np.asarray(lhs, dtype=float).ravel()
    x = np.asarray(rhs, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"lhs and rhs differ in length: {y.size} and {x.size}")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    r = y - float(coeff) * x
    if bound is not None:
        return {"n": n, "bound_deg": float(bound),
                "fraction_satisfied": float(np.mean(r <= bound)) if n else None,
                "max_excess": float(np.max(r - bound)) if n else None}
    out = {"n": n, **dict.fromkeys(_STAT_KEYS)}
    if n == 0:
        return out
    p05, p50, p95 = np.percentile(r, [5.0, 50.0, 95.0])
    out.update(mean=float(np.mean(r)), mae=float(np.mean(np.abs(r))),
               rmse=float(np.sqrt(np.mean(r * r))), p05=float(p05),
               p50=float(p50), p95=float(p95),
               max_abs=float(np.max(np.abs(r))))
    if n >= 3:
        dx, dy = x - x.mean(), y - y.mean()
        sxx, syy, sxy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
        # a constant series is not exactly constant after a float mean
        if sxx > n * (1e-9 * max(1.0, float(np.max(np.abs(x))))) ** 2:
            out["slope"] = sxy / sxx
            out["intercept"] = float(y.mean() - out["slope"] * x.mean())
            if syy > n * (1e-9 * max(1.0, float(np.max(np.abs(y))))) ** 2:
                out["r"] = sxy / float(np.sqrt(sxx * syy))
    return out


# --- comparison -----------------------------------------------------------------------
_CLASS_RULE = {
    _C: "a circular grasp is rebuilt with Table 6 (p.16, section 5.1)",
    _P: ("a prismatic grasp uses Table 7 for thumb, index and middle and "
         "Table 6 for the rest (p.16, section 5.1)"),
    _U: ("a grasp with no Paper 3 class gets only eq. 8, the general PIP/DIP "
         "constraint (p.14)"),
}
COMPARE_NOTE = (
    "MCP relations of the four fingers use the paper-order MCP pair "
    "(abduction first, as the paper's D-H chain turns them, Table 2 p.5; "
    "joint_frames.paper_order_angles) as the primary value; the main "
    "XYZ-order value (flexion first, joint_frames.joint_angles) is reported "
    "beside it as stats_all_main / stats_hold_main. The anatomical alignment "
    "of either with the paper's Cyberglove angles is approximate. The paper "
    "supplies no pass/fail threshold for any relationship: these are "
    "residuals, not verdicts. Statistics are given for every relation whose "
    "angles are available, whatever its status; only 'applied' relations are "
    "the paper's model for the class.")


def _check_class(cls: str) -> None:
    if cls not in CLASSES:
        raise ValueError(f"class must be one of {CLASSES}, got {cls!r}")


def _status(rel: Relation, cls: str) -> Tuple[str, str]:
    """(status, reason). 'unavailable' comes first: it is the data's, not the
    class's, and it means no numbers."""
    missing = [n for n in (rel.lhs, rel.rhs) if n in UNAVAILABLE]
    if missing:
        own = (f"; {rel.source} is the paper's model for class {cls}"
               if cls in rel.applies_to else "")
        return "unavailable", (f"{' and '.join(missing)} not given by the "
                               "camera (joint_frames.CAMERA_NOT_GIVEN), so no "
                               "numbers" + own)
    if "informational" in rel.applies_to:
        return "informational", (f"for information only: {rel.condition}; "
                                 f"{rel.note}")
    if not rel.applies_to:
        return "not_applied", f"conditional on {rel.condition}; {rel.note}"
    if cls in rel.applies_to:
        return "applied", f"{rel.source}, p.{rel.page}, class {cls}: {rel.note}"
    return "not_applied", f"not for class {cls}: {_CLASS_RULE[cls]}"


def _paper_name(name: str) -> Optional[str]:
    """The paper-order key of a finger MCP angle, None for any other angle."""
    if name[0] in FINGERS and name[1:] in ("_MCP_fe", "_MCP_aa"):
        return f"{name}_paper_order"
    return None


def _values(angles, name: str) -> np.ndarray:
    v = angles[name]
    return np.asarray(np.nan if v is None else v, dtype=float)


def _operands(rel: Relation, angles):
    """(lhs, rhs, mcp_order, main): the values a relation is tested on (the
    paper-order MCP reading where the angles carry it) and, when that is
    the case, the main-order pair beside them."""
    lp, rp = _paper_name(rel.lhs), _paper_name(rel.rhs)
    main = (_values(angles, rel.lhs), _values(angles, rel.rhs))
    if lp is None and rp is None:
        return main[0], main[1], None, None
    if all(p in angles for p in (lp, rp) if p):
        return (_values(angles, lp or rel.lhs), _values(angles, rp or rel.rhs),
                "paper_order", main)
    return main[0], main[1], "main", None


def _stats_pair(rel: Relation, y, x, mask):
    y, x = np.atleast_1d(y), np.atleast_1d(x)
    every = residual_stats(y, x, rel.coeff, rel.bound_deg)
    if mask is None:
        return every, None
    if mask.shape != y.shape:
        raise ValueError(f"hold_mask has {mask.size} entries for {y.size} "
                         "frames")
    return every, residual_stats(y[mask], x[mask], rel.coeff, rel.bound_deg)


def compare(angles: Dict[str, np.ndarray], cls: str, hold_mask=None) -> dict:
    """Every relation against an angle series (series_angles) for one class.

    Returns {'class', 'relations', 'summary', 'note'}. Each relation row:
    id, source, page, lhs, rhs, kind, coeff ('4/3'), coeff_float, bound_deg,
    condition, status, reason, mcp_order, stats_all, stats_hold (frames
    where hold_mask is True; None without a mask), stats_all_main and
    stats_hold_main (the main-order reading beside a paper-order one; None
    otherwise). mcp_order: 'paper_order' for a finger MCP relation, 'main'
    when the angles carry no paper-order values, None for the rest. Status,
    first match wins:
      unavailable    lhs or rhs in UNAVAILABLE: no numbers at all
      informational  eq. 6, every class
      applied        the class is in the relation's applies_to
      not_applied    the rest, with the reason (another class's table, or a
                     single-finger condition a grasp does not meet)
    summary counts the statuses; note is COMPARE_NOTE.
    """
    _check_class(cls)
    mask = None if hold_mask is None else np.asarray(hold_mask, bool).ravel()
    rows = []
    for rel in RELATIONS:
        status, reason = _status(rel, cls)
        row = {"id": rel.id, "source": rel.source, "page": rel.page,
               "lhs": rel.lhs, "rhs": rel.rhs, "kind": rel.kind,
               "coeff": str(rel.coeff), "coeff_float": float(rel.coeff),
               "bound_deg": rel.bound_deg, "condition": rel.condition,
               "status": status, "reason": reason, "mcp_order": None,
               "stats_all": None, "stats_hold": None,
               "stats_all_main": None, "stats_hold_main": None}
        if status != "unavailable":
            y, x, row["mcp_order"], main = _operands(rel, angles)
            row["stats_all"], row["stats_hold"] = _stats_pair(rel, y, x, mask)
            if main is not None:
                row["stats_all_main"], row["stats_hold_main"] = _stats_pair(
                    rel, main[0], main[1], mask)
        rows.append(row)
    summary = {s: sum(row["status"] == s for row in rows) for s in STATUSES}
    return {"class": cls, "relations": rows, "summary": summary,
            "note": COMPARE_NOTE}


def _frame_values(rel: Relation, y: float, x: float) -> dict:
    res = y - float(rel.coeff) * x
    sat = (bool(res <= rel.bound_deg)
           if rel.kind == "bound" and np.isfinite(res) else None)
    return {"lhs_value": y, "rhs_value": x, "residual": res, "satisfied": sat}


def frame_relation_values(angles_one_frame: dict, cls: str) -> List[dict]:
    """Per relation, the numbers of one frame (frame_angles) for a panel:
    id, source, page, lhs, rhs, kind, coeff, bound_deg, status, reason,
    mcp_order, lhs_value, rhs_value, residual (lhs - coeff * rhs, or the
    string 'unavailable'), satisfied (bound relations), and lhs_value_main /
    rhs_value_main / residual_main for a finger MCP relation."""
    _check_class(cls)
    out = []
    for rel in RELATIONS:
        status, reason = _status(rel, cls)
        row = {"id": rel.id, "source": rel.source, "page": rel.page,
               "lhs": rel.lhs, "rhs": rel.rhs, "kind": rel.kind,
               "coeff": str(rel.coeff), "bound_deg": rel.bound_deg,
               "status": status, "reason": reason, "mcp_order": None,
               "lhs_value": None, "rhs_value": None, "residual": "unavailable",
               "satisfied": None, "lhs_value_main": None,
               "rhs_value_main": None, "residual_main": None}
        if status != "unavailable":
            y, x, row["mcp_order"], main = _operands(rel, angles_one_frame)
            row.update(_frame_values(rel, float(y), float(x)))
            if main is not None:
                m = _frame_values(rel, float(main[0]), float(main[1]))
                row.update(lhs_value_main=m["lhs_value"],
                           rhs_value_main=m["rhs_value"],
                           residual_main=m["residual"])
        out.append(row)
    return out


# --- Denavit-Hartenberg (p.5) -----------------------------------------------------------
_FINGER_ALPHA = (np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0)      # Table 2, p.5
_THUMB_ALPHA = (np.pi / 2, 0.0, 0.0, 0.0)                   # Table 3, p.5


def dh(theta: float, a: float, alpha: float) -> np.ndarray:
    """The standard D-H link transform Rz(theta) Tx(a) Rx(alpha), radians,
    with d = 0 (p.4: the bones are aligned, so d is always null)."""
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -ca * s, sa * s, a * c],
                     [s, ca * c, -sa * c, a * s],
                     [0.0, sa, ca, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def dh_as_printed(theta: float, a: float, alpha: float) -> np.ndarray:
    """The p.5 matrix exactly as printed: row 1 column 3 sin(alpha) sin(alpha),
    row 2 column 3 -sin(alpha) cos(alpha), the rest as dh(). Its rotation
    block is a rotation only when sin(alpha) = 0 or theta = alpha, where it
    equals dh() (module docstring, D-H); kept for the record, never used."""
    m = dh(theta, a, alpha)
    m[0, 2] = np.sin(alpha) * np.sin(alpha)
    m[1, 2] = -np.sin(alpha) * np.cos(alpha)
    return m


def _chain_tip(angles_deg, links, alphas) -> np.ndarray:
    t = np.eye(4)
    for theta, a, alpha in zip(angles_deg, links, alphas):
        t = t @ dh(np.radians(float(theta)), a, alpha)
    return t[:3, 3].copy()


def finger_tip_position(angles5, lengths4) -> np.ndarray:
    """Fingertip (3,) in the finger's base frame (D-H frame 0), Table 2 (p.5).

    angles5 = (CMC, MCP_aa, MCP_fe, PIP, DIP) in degrees; lengths4 =
    (L_Me, L_P, L_Mi, L_D) in any unit, which the result keeps. Links:
    (CMC, L_Me, pi/2), (MCP_aa, 0, -pi/2), (MCP_fe, L_P, 0), (PIP, L_Mi, 0),
    (DIP, L_D, 0). The wrist offset u_i of eq. 1 is not applied.
    """
    if len(angles5) != 5 or len(lengths4) != 4:
        raise ValueError("need 5 angles and 4 lengths")
    l_me, l_p, l_mi, l_d = (float(v) for v in lengths4)
    return _chain_tip(angles5, (l_me, 0.0, l_p, l_mi, l_d), _FINGER_ALPHA)


def thumb_tip_position(angles4, lengths3) -> np.ndarray:
    """Thumb tip (3,) in the thumb's base frame, Table 3 (p.5).

    angles4 = (TMC_aa, TMC_fe, MCP_fe, IP) in degrees; lengths3 =
    (L_M, L_P, L_D). Links: (TMC_aa, 0, pi/2), (TMC_fe, L_M, 0),
    (MCP_fe, L_P, 0), (IP, L_D, 0). The wrist offset u_T of eq. 2 is not
    applied.
    """
    if len(angles4) != 4 or len(lengths3) != 3:
        raise ValueError("need 4 angles and 3 lengths")
    l_m, l_p, l_d = (float(v) for v in lengths3)
    return _chain_tip(angles4, (0.0, l_m, l_p, l_d), _THUMB_ALPHA)


# --- the 9-DoF description and eq. 9-11 ---------------------------------------------------
NINE_DOF = ("T_TMC_aa", "T_IP", "I_MCP_aa", "I_DIP", "M_DIP", "R_DIP",
            "L_CMC", "L_MCP_aa", "L_DIP")              # pp.14-15, Table 6 grey
TABLE8_9DOF_PCT = {_C: 7.67, _P: 5.90}                 # Table 8, p.17
# The relations that rebuild a class: Table 6 (circular), Tables 6 + 7
# (prismatic). Each of the 15 dependent angles is the lhs of exactly one.
_REBUILD = {cls: [r for r in RELATIONS if r.source in ("Table 6", "Table 7")
                  and cls in r.applies_to] for cls in (_C, _P)}
_NO_REBUILD = ("the 9-DoF description rebuilds circular and prismatic grasps "
               "only (Tables 6 and 7, pp.15-16); a grasp with no Paper 3 class "
               "has none")
ERROR_NOTE = (
    "Eq. 9-11 (pp.16-17) with the 9-DoF description (pp.14-15): the original "
    "tip from the measured angles (paper-order MCP pair, measured PIP and "
    "DIP), the reconstructed tip from the 9 independent angles through "
    "Table 6 (circular) or Tables 6 and 7 (prismatic), both through the "
    "Table 2 chain. CMC is not given by the camera and is 0 in both, so the "
    "palm-arc part of the paper's error is not in these numbers. Positions "
    "are relative to each finger's base frame: u_i of eq. 1 is unknown and "
    "left out; it would cancel in |rp - op| but not in |rp|. Eq. 9 divides "
    "by |rp| as printed, so where the rebuilt finger curls back toward its "
    "base the error passes 100 %. The thumb is "
    "unavailable (its TMC pair is not given), so eq. 11 averages four "
    "fingers, not the paper's five. Table 8's value (p.17) is the paper's "
    "Cyberglove result for five fingers: context, not a threshold.")


def _num(v) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _paper_view(frame: dict) -> dict:
    """The frame with each finger's MCP pair replaced by its paper-order
    reading where the frame carries one: the pair the D-H chain turns."""
    view = dict(frame)
    for f in FINGERS:
        for k in ("fe", "aa"):
            key = f"{f}_MCP_{k}_paper_order"
            if key in frame:
                view[f"{f}_MCP_{k}"] = frame[key]
    return view


def reconstruct_9dof(angles_frame: dict, cls: str) -> dict:
    """The 24 angles of one frame rebuilt from the 9 independent ones (pp.14-15).

    The independent angles are read from `angles_frame` (paper-order MCP
    abduction where the frame carries it); each dependent one is coeff x its
    rhs through _REBUILD[cls]. An independent angle the camera does not give
    (T_TMC_aa, L_CMC) is None and so is everything derived from it (the
    four CMC). Returns {'status': 'applied', 'class', 'values' (24 names),
    'derived_by' (name -> relation id or 'independent')}; for 'unmapped'
    {'status': 'not_applied', 'reason', 'values': {}, 'derived_by': {}}.
    """
    _check_class(cls)
    if cls == _U:
        return {"status": "not_applied", "class": cls, "reason": _NO_REBUILD,
                "values": {}, "derived_by": {}}
    view = _paper_view(angles_frame)
    values = {name: _num(view.get(name)) for name in NINE_DOF}
    derived_by = dict.fromkeys(NINE_DOF, "independent")
    pending = list(_REBUILD[cls])
    while pending:
        ready = [r for r in pending if r.rhs in values]
        if not ready:
            raise RuntimeError(f"Table 6/7 relations for {cls} do not resolve")
        for r in ready:
            v = values[r.rhs]
            values[r.lhs] = None if v is None else float(r.coeff) * v
            derived_by[r.lhs] = r.id
            pending.remove(r)
    return {"status": "applied", "class": cls,
            "values": {name: values[name] for name in jf.DOF24_NAMES},
            "derived_by": {name: derived_by[name] for name in jf.DOF24_NAMES}}


def _finger_lengths(bone_lengths_mm: dict, letter: str) -> Tuple[float, ...]:
    full = dict(jf.FINGERS)[letter]                 # 'I' -> 'INDEX'
    v = bone_lengths_mm.get(letter, bone_lengths_mm.get(full))
    if v is None or len(v) != 4:
        raise ValueError(f"bone_lengths_mm needs {letter!r} -> (L_Me, L_P, "
                         "L_Mi, L_D) in mm (finger_bone_lengths_mm)")
    return tuple(float(x) for x in v)


def fingertip_reconstruction_error(angles: Dict[str, np.ndarray],
                                   bone_lengths_mm: dict, cls: str) -> dict:
    """Eq. 9-11 (pp.16-17) of the 9-DoF description over an angle series.

    angles: series_angles output; bone_lengths_mm: finger letter ->
    (L_Me, L_P, L_Mi, L_D) in mm (finger_bone_lengths_mm). Per frame and
    finger (I, M, R, L): delta = |rp - op| / |rp| x 100 (eq. 9), op from
    the measured angles, rp from reconstruct_9dof, both with CMC = 0, frames
    with a missing angle skipped. Per finger n, mean_pct (eq. 10), p50_pct,
    p95_pct, max_pct; gesture_error_pct = mean of the finger means (eq. 11,
    four fingers); per_frame_pct = finger -> array (NaN where skipped);
    thumb 'unavailable'; paper_table8_pct for context. Module docstring and
    ERROR_NOTE give the assumptions. 'unmapped' -> status 'not_applied'.
    """
    _check_class(cls)
    if cls == _U:
        return {"status": "not_applied", "class": cls, "reason": _NO_REBUILD}
    series = {k: np.atleast_1d(np.asarray(v, dtype=float))
              for k, v in angles.items()}
    n = len(series["I_DIP"])
    lengths = {f: _finger_lengths(bone_lengths_mm, f) for f in FINGERS}
    per_frame = {f: np.full(n, np.nan) for f in FINGERS}
    for k in range(n):
        frame = {name: values[k] for name, values in series.items()}
        view = _paper_view(frame)
        rebuilt = reconstruct_9dof(frame, cls)["values"]
        for f in FINGERS:
            names = (f"{f}_MCP_aa", f"{f}_MCP_fe", f"{f}_PIP", f"{f}_DIP")
            orig = [_num(view.get(name)) for name in names]
            new = [rebuilt[name] for name in names]
            if None in orig or None in new:
                continue
            op = finger_tip_position([0.0] + orig, lengths[f])
            rp = finger_tip_position([0.0] + new, lengths[f])
            norm = float(np.linalg.norm(rp))
            if norm > 0.0:
                per_frame[f][k] = float(np.linalg.norm(rp - op)) / norm * 100.0
    fingers = {}
    for f in FINGERS:
        d = per_frame[f][np.isfinite(per_frame[f])]
        some = d.size > 0
        fingers[f] = {"n": int(d.size),
                      "mean_pct": float(d.mean()) if some else None,
                      "p50_pct": float(np.percentile(d, 50)) if some else None,
                      "p95_pct": float(np.percentile(d, 95)) if some else None,
                      "max_pct": float(d.max()) if some else None}
    averaged = [f for f in FINGERS if fingers[f]["mean_pct"] is not None]
    gesture = (float(np.mean([fingers[f]["mean_pct"] for f in averaged]))
               if averaged else None)
    return {"status": "applied", "class": cls,
            "description": "9 DoF (pp.14-15)", "independent": list(NINE_DOF),
            "fingers": fingers, "thumb": "unavailable",
            "thumb_reason": ("T_TMC_aa (independent) and T_TMC_fe are not "
                             "given by the camera (joint_frames."
                             "CAMERA_NOT_GIVEN)"),
            "gesture_error_pct": gesture, "fingers_averaged": averaged,
            "per_frame_pct": per_frame,
            "paper_table8_pct": TABLE8_9DOF_PCT[cls], "note": ERROR_NOTE}
