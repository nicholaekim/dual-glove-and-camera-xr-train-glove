"""Fit the glove's TEMPLATE hand to the operator's own hand.

The glove measures joint ANGLES. It does not measure bone LENGTHS: XR Trainer
hangs every angle it solves on one fixed template skeleton, the same one for
every wearer, so a `HandFrame` off the glove is the operator's pose on
somebody else's hand. The camera measures the real lengths, in millimetres,
every frame.

Fusion already aligns the two on the palm and takes each degree of freedom
from the sensor that measures it, and the size mismatch is what is left over.
It shows up twice, in numbers the fusion report has been printing all along:

  palm fit residual    3.0 mm median on sync_day1, 3.3 mm on sync_day2 — the
                       RMSE of a rigid 5-point palm fit, i.e. how far the two
                       palms are from being one object (`palm_fit_rmse_mm`).
  the pinch curl       with the rail override active the fused index takes the
                       camera's three bone DIRECTIONS on the glove's LENGTHS,
                       so the fused curl lands about 0.15 ABOVE the camera's:
                       curl is tip-to-wrist over palm length, and the template
                       index is about 11 % longer per palm length than this
                       operator's. The joint angles were exactly the camera's;
                       only the bones they hung on were wrong.

This module removes the mismatch at the source instead of correcting for it
downstream:

  measure_hand   median length of every bone of the 26-joint chain, over the
                 camera's own trusted frames of one hand.
  fit_template   rescale each of the glove hand's parent-relative joint
                 offsets to the camera's length for that segment, KEEPING
                 EVERY ROTATION.

WHY RESCALING THE OFFSETS AND NOTHING ELSE
  A `HandFrame` joint is a translation in its parent's frame plus a rotation.
  The translation's DIRECTION and the rotation together are the pose; the
  translation's LENGTH is the bone. So scaling each offset to the camera's
  length for that segment, with the quaternion untouched, changes lengths and
  nothing else: every joint angle in the hand is bit-identical afterwards, and
  forward kinematics reproduces the fitted lengths exactly (there is no
  least-squares step to leave a residual). `fit_template` is the only thing in
  this repo that changes a bone length, and it changes it to a MEASURED one.

WHAT STAYS ON THE RAW TEMPLATE, AND WHY
  The fusion report's `glove` column and the glove-only classifier stay on the
  RAW template hand, because that is what the glove alone gives: a glove-only
  pipeline has no camera to measure a hand with, and quoting a fitted number
  for it would be quoting a result the glove cannot produce. The fitted hand
  is the FUSED path's glove input. Both appear in the report, and the legend
  says which is which.

THE CURL METRIC MOVES, AND THE THRESHOLDS THAT READ IT HAVE TO FOLLOW
  Curl is tip-to-wrist over palm length. Fitting shortens the template's
  fingers by a few per cent each and shortens its palm too, so every glove
  curl in the pipeline changes by a few hundredths — including the ones
  `GateParams.curl_gate` and the learned rails are compared against. Two
  consequences, both handled by the caller rather than here:

    the rails must be LEARNED on the fitted values, because the override
    compares the fitted value against them (`fuse_poses.py` learns them from
    whatever it is about to fuse, which is what makes that automatic);

    a constant that no longer SEPARATES after fitting must be re-expressed as
    a fraction of that hand's own learned rail rather than retuned to a new
    constant — a new constant would be tuned on one operator's fitted hand
    and would be wrong for the next one by exactly the amount fitting exists
    to remove. Measured on both sessions, `curl_gate` (1.2, the midpoint
    between the glove's fists and its open palms) still separates after
    fitting, so it is still the constant it was; the report prints the fitted
    open and fist values it is being asked to separate.

THE MEASUREMENT IS TAKEN ON OPEN FRAMES WHEN THERE ARE ANY
  A bone is the same length in every pose, so in principle any frame will do.
  In practice a closed hand is self-occluded and the tracker INFERS the joints
  it cannot see, so its lengths are a solver's opinion. An open hand is the
  one shape both cameras see whole. Which frames are open is decided on the
  CAMERA's own curls against `RailOverrideParams.cam_open_curl` — a geometric
  fact about a straight finger, not the glove's claim about one — so the
  selection cannot be poisoned by the sensor under suspicion. If too few open
  frames exist the measurement falls back to every trusted frame and says so
  (`from_open`), because a measurement whose provenance is not printed is a
  constant with extra steps.
"""
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from xr_hand.joints import BONES, JOINT_NAMES, HandFrame, Joint
from xr_hand.keypoints21 import MP21_TO_OPENXR_IDX

from .features import flexion_features, median
from .fusion import DEFAULT_RAIL, FINGER_NAMES, RAIL_FINGERS, frame_trust

# (parent, child) index pairs of the 26-joint chain, in BONES order. This is
# every segment the hand has: WRIST -> PALM, the thumb's four, and for each
# finger the wrist-to-metacarpal and metacarpal-to-knuckle PALM segments
# followed by its three phalanges. Measuring `BONES` measures all of them, so
# there is no second list here to fall out of step with `joints.py`.
SEGMENTS: Tuple[Tuple[int, int], ...] = tuple(BONES)


def segment_name(parent: int, child: int) -> str:
    """`WRIST->INDEX_PROXIMAL` — the key a saved measurement is read by.

    Joint NAMES and not indices, so a saved file survives any renumbering of
    the layout and can be read by a human deciding whether to trust it.
    """
    return f"{JOINT_NAMES[parent]}->{JOINT_NAMES[child]}"


# Which segments belong to each finger, for the per-finger scale factor the
# report prints. The palm segments (wrist -> metacarpal -> knuckle) are left
# out: they are shared with the palm and the report has the palm fit for them.
FINGER_SEGMENTS: Dict[str, Tuple[Tuple[int, int], ...]] = {}
for _finger, _chain in (("thumb", ("THUMB_METACARPAL", "THUMB_PROXIMAL",
                                   "THUMB_DISTAL", "THUMB_TIP")),
                        ("index", ("INDEX_PROXIMAL", "INDEX_INTERMEDIATE",
                                   "INDEX_DISTAL", "INDEX_TIP")),
                        ("middle", ("MIDDLE_PROXIMAL", "MIDDLE_INTERMEDIATE",
                                    "MIDDLE_DISTAL", "MIDDLE_TIP")),
                        ("ring", ("RING_PROXIMAL", "RING_INTERMEDIATE",
                                  "RING_DISTAL", "RING_TIP")),
                        ("pinky", ("LITTLE_PROXIMAL", "LITTLE_INTERMEDIATE",
                                   "LITTLE_DISTAL", "LITTLE_TIP"))):
    _idx = [JOINT_NAMES.index(n) for n in _chain]
    FINGER_SEGMENTS[_finger] = tuple(zip(_idx, _idx[1:]))

# How far below its open reference a camera curl may sit and still count as a
# straight finger, for picking the frames a measurement is taken over. The
# rail override's `margin`, and for the same reason: it is the distance that
# separates sync_day1's genuinely straight fingers from its pinched ones.
OPEN_MARGIN = DEFAULT_RAIL.margin
# Fewer open frames than this and the measurement falls back to every trusted
# frame. 20 frames is about a fifth of a second of Leap tracking — enough for
# a median to mean something, few enough that any real open-palm take clears
# it many times over.
MIN_OPEN_FRAMES = 20


@dataclass(frozen=True)
class HandMeasurement:
    """One hand's real bone lengths, in METRES, as the camera measured them.

    Metres because everything else in this repo is metres; the saved JSON is
    in millimetres because that is the unit a hand is discussed in, and
    `to_json`/`from_json` are the only place the two meet.

    lengths   segment name -> median length over the frames used
    spreads   segment name -> median absolute deviation of that length. The
              measurement's own error bar: a segment whose length wanders by
              a millimetre frame to frame was not really measured.
    n_frames  how many frames the medians are over
    from_open how they were chosen: True = the camera's own open-hand frames,
              False = every trusted frame, because there were not enough open
              ones. Carried rather than inferred, so a report can print the
              provenance of the numbers it is about to rescale a hand with.
    """

    hand: str = ""
    lengths: Mapping[str, float] = field(default_factory=dict)
    spreads: Mapping[str, float] = field(default_factory=dict)
    n_frames: int = 0
    from_open: bool = False
    source: str = ""

    def length(self, parent: int, child: int) -> Optional[float]:
        return self.lengths.get(segment_name(parent, child))

    @property
    def n_segments(self) -> int:
        return len(self.lengths)

    @property
    def spread_mm(self) -> float:
        """The typical segment's spread, in mm — one number for the report."""
        if not self.spreads:
            return float("nan")
        return 1000.0 * median(list(self.spreads.values()))

    def to_json(self) -> dict:
        return {
            "hand": self.hand,
            "units": "mm",
            "n_frames": self.n_frames,
            "from_open_frames": self.from_open,
            "source": self.source,
            "lengths_mm": {k: round(1000.0 * v, 4)
                           for k, v in sorted(self.lengths.items())},
            "spread_mm": {k: round(1000.0 * v, 4)
                          for k, v in sorted(self.spreads.items())},
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n",
                        encoding="utf-8")
        return path


def measurement_filename(hand: str, stamp: Optional[str] = None) -> str:
    """`hand_left_20260921_120000.json` — the name a profile keeps one under."""
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    return f"hand_{str(hand).strip().lower() or 'unknown'}_{stamp}.json"


def from_json(data: Mapping) -> HandMeasurement:
    """A saved measurement -> a HandMeasurement, mm back to metres."""
    if not isinstance(data, Mapping):
        raise ValueError("a hand measurement is a JSON object")
    lengths = data.get("lengths_mm")
    if not isinstance(lengths, Mapping) or not lengths:
        raise ValueError("a hand measurement needs a non-empty 'lengths_mm'")
    spreads = data.get("spread_mm") or {}
    if not isinstance(spreads, Mapping):
        raise ValueError("'spread_mm' must be an object keyed by segment")
    unknown = sorted(k for k in lengths
                     if k not in {segment_name(a, b) for a, b in SEGMENTS})
    if unknown:
        raise ValueError(
            f"unknown segment(s) {', '.join(unknown)}; a segment is named "
            "PARENT->CHILD over the 26 OpenXR joints (see joints.JOINT_NAMES)")
    return HandMeasurement(
        hand=str(data.get("hand", "")),
        lengths={k: float(v) / 1000.0 for k, v in lengths.items()},
        spreads={k: float(v) / 1000.0 for k, v in spreads.items()},
        n_frames=int(data.get("n_frames", 0)),
        from_open=bool(data.get("from_open_frames", False)),
        source=str(data.get("source", "")))


def load_measurement(path) -> HandMeasurement:
    path = Path(path)
    return replace(from_json(json.loads(path.read_text(encoding="utf-8"))),
                   source=str(path))


# --- measuring the operator's hand -------------------------------------

def _abs26_of(row: Mapping) -> Optional[np.ndarray]:
    """One camera row's 26 joint positions, or None if it has none.

    `abs26` is the camera's RAW measurement — the 26 joints in camera space,
    before they were folded into parent-relative form — which is why it is
    what a length is read off. A MediaPipe row has no such thing (21
    normalised landmarks, no metacarpals and no metres), and a hand cannot be
    measured from it at all.
    """
    pts = row.get("abs26")
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=float)
    if arr.shape != (len(JOINT_NAMES), 3) or not np.isfinite(arr).all():
        return None
    return arr


def _is_open(abs26: np.ndarray, margin: float) -> bool:
    """Does the CAMERA read all four fingers of this frame as straight?

    On the camera's own curls against `cam_open_curl`, never on the glove's:
    the glove's claim that a hand is open is the claim the whole rail override
    exists because it cannot be trusted. Curl is
    `features.flexion_features` — tip-to-wrist over palm length, the same
    quantity every report in the repo prints — computed off the 21 of these 26
    joints that layout has a place for, and it is a RATIO, so it needs neither
    wrist-centring nor a scale.
    """
    curls = flexion_features([abs26[i] for i in MP21_TO_OPENXR_IDX])
    return all(curls[FINGER_NAMES.index(f)]
               >= DEFAULT_RAIL.open_curl(f) - margin for f in RAIL_FINGERS)


def measure_hand(cam_rows: Sequence[Mapping], hand: Optional[str] = None,
                 open_margin: float = OPEN_MARGIN,
                 min_open_frames: int = MIN_OPEN_FRAMES,
                 gates=None, source: str = "") -> Optional[HandMeasurement]:
    """The operator's real bone lengths, from the camera's trusted frames.

    `cam_rows` are the camera row dicts `scripts/fuse_poses.py` loads: each
    needs `abs26` (the 26 joints in camera space, metres) and, to be gated,
    the capture facts `frame_trust` reads. Pass `hand` to filter to one side;
    the lengths of two different hands must never land in one median.

    Every segment of the 26-joint chain is measured, which includes the
    wrist-to-metacarpal and metacarpal-to-knuckle PALM segments — those are
    the ones the palm fit is residual on, so leaving them out would fit the
    fingers of a hand whose palm was still the template's.

    Returns None when no frame carries a usable `abs26`: a camera that cannot
    measure a hand has not measured one, and the caller fuses unfitted.
    """
    side = None if hand is None else str(hand).strip().lower()
    trusted: List[Mapping] = []
    for row in cam_rows:
        if side is not None and str(row.get("hand_side", "")).lower() != side:
            continue
        pts = _abs26_of(row)
        if pts is None:
            continue
        # The same frame-level trust the gates use, when the row carries the
        # facts to judge it by. A row with no capture facts is kept: gating a
        # frame on evidence it does not produce would reject all of them.
        if row.get("palm_abs") is not None:
            ok, _why, _metrics = frame_trust(
                {"visible_time_us": row.get("visible_time_us"),
                 "hand_id_stable": row.get("hand_id_stable"),
                 "palm_abs": row.get("palm_abs"),
                 "palm_normal_abs": row.get("palm_normal_abs")}, gates)
            if not ok:
                continue
        trusted.append(pts)
    if not trusted:
        return None

    open_frames = [p for p in trusted if _is_open(p, open_margin)]
    from_open = len(open_frames) >= min_open_frames
    used = open_frames if from_open else trusted

    lengths: Dict[str, float] = {}
    spreads: Dict[str, float] = {}
    for parent, child in SEGMENTS:
        per_frame = [float(np.linalg.norm(p[child] - p[parent])) for p in used]
        mid = median(per_frame)
        if not math.isfinite(mid) or mid <= 0.0:
            continue
        name = segment_name(parent, child)
        lengths[name] = mid
        spreads[name] = median([abs(v - mid) for v in per_frame])
    if not lengths:
        return None
    return HandMeasurement(hand=side or "", lengths=lengths, spreads=spreads,
                           n_frames=len(used), from_open=from_open,
                           source=source)


def merge_measurements(parts: Iterable[HandMeasurement]
                       ) -> Optional[HandMeasurement]:
    """One measurement out of several — the median segment by segment.

    For a session read take by take: a per-take measurement of a hand is a
    perfectly good measurement of that hand, and the median over takes is less
    sensitive to one badly-tracked take than pooling every frame would be,
    which would let the longest take decide.

    The takes measured on OPEN frames win outright whenever there are any.
    "Use open frames when available, else all" is a decision about the SESSION
    and not about each take: a fist take has no open frame in it, falls back to
    its own self-occluded ones, and hands back lengths that are the tracker's
    guess at a finger it could not see. Measured on sync_day1's right hand,
    letting those takes into the median put the index at 0.86 of the template
    against 0.94 from the open takes alone — an 8 % error in the one number
    this whole module exists to get right.
    """
    parts = [p for p in parts if p is not None and p.lengths]
    if not parts:
        return None
    open_parts = [p for p in parts if p.from_open]
    used = open_parts or parts
    names = sorted({k for p in used for k in p.lengths})
    lengths = {n: median([p.lengths[n] for p in used if n in p.lengths])
               for n in names}
    spreads = {n: median([p.spreads.get(n, 0.0) for p in used
                          if n in p.lengths]) for n in names}
    return HandMeasurement(
        hand=used[0].hand, lengths=lengths, spreads=spreads,
        n_frames=sum(p.n_frames for p in used),
        from_open=bool(open_parts),
        source=(f"{len(used)} of {len(parts)} take(s)"
                if len(used) != len(parts) else f"{len(parts)} take(s)"))


# --- putting the measurement on the glove's template -------------------

def fit_template(glove, measurement: Optional[HandMeasurement]):
    """The glove's hand with the camera's bone LENGTHS and its own ANGLES.

    `glove` is a `HandFrame` or an iterable of them; the return has the same
    shape. Each non-root joint's parent-relative offset is rescaled to the
    measured length of its segment and its quaternion is copied through
    untouched, so:

      every joint angle is bit-identical to the glove's;
      forward kinematics of the result reproduces the measured lengths
      EXACTLY — there is no fit and therefore no residual;
      a segment the measurement does not name keeps the template's length,
      which is what makes a partial measurement usable rather than fatal.

    The WRIST is the root and keeps its pose: it carries where the hand is,
    which is not a length. A joint whose template offset is zero long is left
    alone — there is no direction to put a length along.

    With `measurement` None the frame is returned unchanged, so a caller can
    pass whatever it has and let this decide.
    """
    if isinstance(glove, HandFrame):
        return _fit_one(glove, measurement)
    return [_fit_one(f, measurement) for f in glove]


def _fit_one(frame: HandFrame, measurement: Optional[HandMeasurement]
             ) -> HandFrame:
    if measurement is None or not measurement.lengths:
        return frame
    joints: List[Joint] = list(frame.joints)
    if len(joints) != len(JOINT_NAMES):
        return frame
    out = list(joints)
    for parent, child in SEGMENTS:
        target = measurement.length(parent, child)
        if target is None or target <= 0.0:
            continue
        j = joints[child]
        current = math.sqrt(j.x * j.x + j.y * j.y + j.z * j.z)
        if current <= 1e-12:
            continue
        s = target / current
        out[child] = replace(j, x=j.x * s, y=j.y * s, z=j.z * s)
    return replace(frame, joints=out)


def template_lengths(frame: HandFrame) -> Dict[str, float]:
    """The template's own segment lengths, straight off the offsets, metres."""
    out: Dict[str, float] = {}
    for parent, child in SEGMENTS:
        j = frame.joints[child]
        out[segment_name(parent, child)] = math.sqrt(
            j.x * j.x + j.y * j.y + j.z * j.z)
    return out


def finger_scales(frame: HandFrame, measurement: Optional[HandMeasurement]
                  ) -> Dict[str, Optional[float]]:
    """Per finger: fitted total length over template total length.

    Summed over the finger's own phalanges, so it is the number the curl
    metric actually moves with — curl is a tip-to-wrist distance, and how
    much shorter the whole finger got is what changes it. None for a finger
    the measurement does not cover.
    """
    if measurement is None:
        return {f: None for f in FINGER_SEGMENTS}
    template = template_lengths(frame)
    out: Dict[str, Optional[float]] = {}
    for finger, segments in FINGER_SEGMENTS.items():
        names = [segment_name(a, b) for a, b in segments]
        if any(n not in measurement.lengths for n in names):
            out[finger] = None
            continue
        was = sum(template.get(n, 0.0) for n in names)
        now = sum(measurement.lengths[n] for n in names)
        out[finger] = (now / was) if was > 1e-12 else None
    return out
