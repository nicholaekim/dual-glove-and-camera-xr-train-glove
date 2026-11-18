"""A thumb-TMC calibration take: the neutral posture held under the camera,
read back as a `tmc_estimate.Calibration`.

WHAT THE TAKE IS
  `scripts/leap/record_poses.py --calibration tmc_neutral --hand <side>`
  records the posture of POSE_DEFINITION (docs/tmc_calibration_protocol.md)
  held still, `--takes` times, every frame kept, into
  recordings/protocol/tmc_neutral/<YYYYMMDD_HHMMSS>_<hand>/ (protocol_mock
  with --mock): one JSONL per take in the camera line format
  (`leap_hand.recorder`) with `pose` "tmc_neutral", and beside it one
  `<take>.meta.json` sidecar (META_SUFFIX) with the posture definition, the
  hand, the commanded hold (`hold_start`, `hold_end`, wall clock), the
  operator, a goniometer note, the mock flag and the tool commit. At the end
  of the session the recorder writes `calibration.json`
  (`Calibration.to_json`) from `calibrate` below.

WHAT THE LOADER DOES
  `load_take` reads one take through `replay_core.load_recording` (the hand
  the sidecar names, every hand id, status != 0) and takes every selected
  frame as held: the file starts at the REC beep, after the posture is made,
  and ends at the stop beep. The hold's duration is the span of the frames'
  own clock (`Recording.timeline`: timestamp_us when it is sound, else the
  fallbacks replay_core names), not the commanded seconds; the commanded span
  is kept beside it. The take's rest frame is the chordal mean of
  R_palm^T R_thumb_metacarpal over those frames
  (`tmc_estimate.palm_to_metacarpal`, `chordal_mean`), with the frames'
  spread about it.

  `calibrate(folder)` takes one mean per take and the chordal mean of those
  means as R_rest, so every take weighs once however long its hold, as the
  open-hand `tmc_estimate.calibrate` does per transition. Takes with fewer
  than `min_frames` frames or a hold shorter than `min_hold_s` are skipped and
  listed with the reason. The across-take agreement (median, p90 and max
  angle of a take's mean to R_rest) is the zero's floor uncertainty: it says
  how repeatably the posture was made. The record is a
  `tmc_estimate.Calibration` with `kind="neutral"` and `hand` the take's
  hand; its other field names come from the open-hand transitions: here
  `n_transitions` counts takes and `per_transition` holds one dict per take
  (`"name": "tmc_neutral"`). `calibration.json` carries the kind, and
  `Calibration.from_json` reads it back (a file written before the field
  existed is recognised by its provenance prefix, PROVENANCE_PREFIX).

WHAT THE ZERO IS, AND IS NOT
  R_rest from such a take is the thumb metacarpal's orientation against the
  tracker's palm in the clinical reference posture (thumb beside the index,
  in the plane of the palm). `tmc_estimate.estimate` against it gives angles
  from that posture instead of from the session's incidental open hand,
  under the `*_est_neutral` names (the estimate names its outputs by the
  calibration's kind), so that the two zeros never share a column;
  `estimate` here is that call with the kind checked first. The result is
  still a tracker-relative coordinate pair: changing the reference from the
  open hand to this posture turns R_tmc by a fixed rotation, which after the
  abduction-then-flexion decomposition shifts both angles by pose-dependent
  amounts, not by a constant (tmc_estimate, "WHAT A CHANGE OF REFERENCE
  DOES"). It is not the ISB neutral of the first carpometacarpal joint (Wu
  et al. 2005, section 4.4.1: the trapezium and first-metacarpal segment
  frames coincident, the trapezium frame of Cooney et al. 1981), which needs
  trapezium landmarks the camera does not see; and nothing here validates
  the mapping's scale or cross-talk against the true joint angles, which
  needs an independent sensor (docs/tmc_estimate.md, "A calibration take
  instead of the open hand").

Pure: reads files, writes nothing.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from xr_hand import joint_frames as jf

from . import replay_core as rc
from . import tmc_estimate as te

POSE = "tmc_neutral"
CALIBRATION_KIND = "tmc_neutral_calibration_take"
POSE_DEFINITION = {
    "id": "tmc_neutral_v1",
    "text": ("the hand flat and open with the palm toward the camera, the four "
             "fingers straight and together, the thumb straight and resting "
             "against the side of the index finger in the plane of the palm, "
             "nothing pressed"),
    "zero": ("the clinical reference posture for thumb CMC abduction and "
             "flexion (thumb beside the index, in the plane of the palm), held "
             "as a stand-in for the ISB neutral of the first carpometacarpal "
             "joint (Wu et al. 2005, section 4.4.1: trapezium and "
             "first-metacarpal segment frames coincident, the trapezium frame "
             "of Cooney et al. 1981), which needs trapezium landmarks the "
             "camera does not see"),
    "document": "docs/tmc_calibration_protocol.md",
}
META_SUFFIX = ".meta.json"
MIN_FRAMES = te.MIN_FRAMES          # 20, as the open-hand calibration
MIN_HOLD_S = 2.0                    # from the frames' timestamps, not commanded
KIND = "neutral"
PROVENANCE_PREFIX = te.PROVENANCE_PREFIXES[KIND]

FE, AA, TWIST, RESIDUAL = KEYS = te.keys_for(KIND)


@dataclass
class Take:
    """One held take, read. `hold_s` is from the frames' clock (`clock`);
    `hold_commanded_s` is the sidecar's hold_end - hold_start (wall clock),
    None when the sidecar lacks one of them."""
    path: Path
    meta: dict
    hand: str
    take: int
    n_frames: int
    hold_s: float
    hold_commanded_s: Optional[float]
    clock: str
    id_changes: int
    gaps: int
    mean: np.ndarray
    spread_median_deg: float
    spread_max_deg: float


def meta_path(jsonl_path) -> Path:
    """`<take>.jsonl` -> `<take>.meta.json`, beside it."""
    return Path(jsonl_path).with_suffix(META_SUFFIX)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def read_meta(path) -> dict:
    """The sidecar, checked to be a tmc_neutral calibration take of one hand."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path.name}: no sidecar; a calibration take needs "
                         f"<take>{META_SUFFIX} beside its JSONL")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("kind") != CALIBRATION_KIND or meta.get("pose") != POSE:
        raise ValueError(f"{path.name}: kind {meta.get('kind')!r}, pose "
                         f"{meta.get('pose')!r}: not a {CALIBRATION_KIND} of {POSE}")
    if meta.get("hand") not in rc.SIDES:
        raise ValueError(f"{path.name}: hand {meta.get('hand')!r}, expected one of {rc.SIDES}")
    if not _number(meta.get("take")):
        raise ValueError(f"{path.name}: no take number")
    return meta


def _definition_id(meta: dict) -> str:
    return str((meta.get("pose_definition") or {}).get("id", ""))


def load_take(jsonl_path) -> Take:
    """One take: its sidecar, the hand's frames, the hold from their clock,
    and the chordal mean of R_palm^T R_thumb_metacarpal over them."""
    path = Path(jsonl_path)
    meta = read_meta(meta_path(path))
    rec = rc.load_recording(path, meta["hand"], kind="calibration", name=POSE,
                            take=int(meta["take"]), metadata=meta)
    t = np.asarray(rec.timeline.t, dtype=float)
    pm = np.array([te.palm_to_metacarpal(rc.reconstruct(f).rot) for f in rec.frames])
    mean = te.chordal_mean(pm)
    spread = np.atleast_1d(te.rotation_angle_deg(mean, pm))
    start, end = meta.get("hold_start"), meta.get("hold_end")
    commanded = float(end) - float(start) if _number(start) and _number(end) else None
    return Take(path=path, meta=meta, hand=meta["hand"], take=int(meta["take"]),
                n_frames=len(rec.frames), hold_s=float(t[-1] - t[0]),
                hold_commanded_s=commanded, clock=rec.timeline.source,
                id_changes=sum(e.kind == "id_change" for e in rec.events),
                gaps=sum(e.kind == "gap" for e in rec.events),
                mean=mean, spread_median_deg=float(np.median(spread)),
                spread_max_deg=float(np.max(spread)))


def list_takes(folder) -> List[Path]:
    """The session folder's JSONL takes, sorted by name; every one needs its
    sidecar, or the folder is refused."""
    folder = Path(folder)
    paths = sorted(folder.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no .jsonl take in {folder}")
    missing = [p.name for p in paths if not meta_path(p).is_file()]
    if missing:
        raise ValueError(f"{folder}: take(s) without a {META_SUFFIX} sidecar: "
                         f"{', '.join(missing)}")
    return paths


def calibrate(folder, *, min_frames: int = MIN_FRAMES,
              min_hold_s: float = MIN_HOLD_S) -> te.Calibration:
    """R_rest from the held frames of a session folder's takes (module
    docstring). One hand and one posture definition per folder; takes with
    fewer than `min_frames` frames or a hold under `min_hold_s` seconds (from
    the frames' clock) are skipped and listed in `skipped`."""
    folder = Path(folder)
    takes = [load_take(p) for p in list_takes(folder)]
    hands = sorted({t.hand for t in takes})
    if len(hands) > 1:
        raise ValueError(f"{folder}: takes of both hands ({', '.join(hands)}); "
                         "one session calibrates one hand")
    definitions = sorted({_definition_id(t.meta) for t in takes})
    if len(definitions) > 1:
        raise ValueError(f"{folder}: takes of different posture definitions "
                         f"({', '.join(definitions)}); one session, one definition")
    used, skipped = [], []
    for t in takes:
        label = {"name": POSE, "take": t.take, "validity": "", "file": t.path.name}
        if t.n_frames < min_frames:
            skipped.append({**label, "reason": f"{t.n_frames} frames < {min_frames}"})
        elif t.hold_s < min_hold_s:
            skipped.append({**label, "reason": f"hold {t.hold_s:.2f} s < {min_hold_s:g} s "
                                               f"(from the frames' {t.clock})"})
        else:
            used.append(t)
    if not used:
        reasons = "; ".join(f"take {s['take']}: {s['reason']}" for s in skipped)
        raise ValueError(f"no usable take in {folder} ({len(skipped)} skipped: {reasons})")
    means = np.array([t.mean for t in used])
    r_rest = te.chordal_mean(means)
    angles = np.atleast_1d(te.rotation_angle_deg(r_rest, means))
    per = []
    for t, a in zip(used, angles):
        per.append({"name": POSE, "take": t.take, "validity": "held", "file": t.path.name,
                    "n_frames": t.n_frames, "hold_s": t.hold_s,
                    "hold_commanded_s": t.hold_commanded_s, "clock": t.clock,
                    "id_changes": t.id_changes, "gaps": t.gaps,
                    "spread_median_deg": t.spread_median_deg,
                    "spread_max_deg": t.spread_max_deg,
                    "angle_to_session_mean_deg": float(a),
                    "rest_xyz_deg": [math.degrees(x) for x in jf.xyz_angles(t.mean)]})
    worst = per[int(np.argmax(angles))]
    agreement = {"median_deg": float(np.median(angles)),
                 "p90_deg": float(np.percentile(angles, 90)),
                 "max_deg": float(np.max(angles)),
                 "worst": {"name": POSE, "take": worst["take"]}}
    n_frames = sum(t.n_frames for t in used)
    holds = [t.hold_s for t in used]
    notes = sorted({str(t.meta.get("goniometer")).strip() for t in used
                    if str(t.meta.get("goniometer") or "").strip()})
    definition = used[0].meta.get("pose_definition") or {}
    provenance = (
        f"{PROVENANCE_PREFIX} of session {folder.name} ({hands[0]} hand): R_rest = "
        f"chordal mean of R_palm^T R_thumb_metacarpal over the held frames of "
        f"{len(used)} take(s) ({n_frames} frames; holds {min(holds):.2f} to "
        f"{max(holds):.2f} s from the frames' timestamps, >= {min_frames} frames and "
        f">= {min_hold_s:g} s per take); zero = {definition.get('id', '?')}: "
        f"{definition.get('text', '?')}; across-take agreement median "
        f"{agreement['median_deg']:.1f}, p90 {agreement['p90_deg']:.1f}, max "
        f"{agreement['max_deg']:.1f} deg (take {worst['take']})"
        + ("; single take: the agreement is not measured" if len(used) == 1 else "")
        + (f"; goniometer: {' | '.join(notes)}" if notes else "")
        + "; a reference posture held under the camera: not the ISB bone-based "
          "neutral and not validated against an independent sensor")
    return te.Calibration(R_rest=r_rest, session=folder.name, n_transitions=len(used),
                          n_frames=n_frames, per_transition=per, agreement=agreement,
                          provenance=provenance, kind=KIND, hand=hands[0], skipped=skipped)


def summary_line(cal: te.Calibration) -> str:
    """The recorder's last line: takes, frames, the hold span and the agreement."""
    holds = [p["hold_s"] for p in cal.per_transition]
    return (f"TMC NEUTRAL CALIBRATION OK: takes={cal.n_transitions} frames={cal.n_frames} "
            f"hold_s={min(holds):.2f}..{max(holds):.2f} "
            f"agreement_max_deg={cal.agreement['max_deg']:.2f} skipped={len(cal.skipped)}")


def is_neutral(calibration) -> bool:
    """Was this Calibration made by `calibrate` here (its zero the posture),
    rather than by the open-hand `tmc_estimate.calibrate`? Read from its kind."""
    return getattr(calibration, "kind", None) == KIND


def estimate(pos, rot, hand_side: str, calibration: te.Calibration) -> dict:
    """`tmc_estimate.estimate` against a tmc_neutral calibration: its four
    numbers under the `*_est_neutral` names (KEYS) plus `calibration`, the
    provenance text, and `kind`. Refuses any other kind: the two zeros must
    never share a column."""
    if not is_neutral(calibration):
        raise ValueError("not a tmc_neutral calibration (kind "
                         f"{getattr(calibration, 'kind', None)!r}); use "
                         "tmc_estimate.estimate for an open-hand calibration")
    return te.estimate(pos, rot, hand_side, calibration)
