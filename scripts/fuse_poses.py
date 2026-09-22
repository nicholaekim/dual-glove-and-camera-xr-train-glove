"""Fuse simultaneous glove + camera takes and report what each sensor supplied.

Reads the paired recordings written by scripts/record_simultaneous.py, matches
frames on the capture clock, fuses each pair through cam_hand/fusion.py, and
prints a PER-DEGREE-OF-FREEDOM table: for every pose and hand, the glove's
value, the camera's value and the fused value for the five curls, the four
adjacent spreads and the thumb-index gap, followed by how often the camera
actually supplied each gated DOF and why it was refused the rest of the time.

That table is the headline because it is the thing that can be checked against
the hand that was in front of the sensors. A single accuracy number cannot say
WHICH degree of freedom fusion got right, and on the first real session it said
the opposite of the truth: fused scored below glove-only, because the fusion
was importing the camera's confidently wrong thumbs_up.

The leave-one-out nearest-centroid comparison is still printed, below, as a
secondary metric, and it is now leave-one-TAKE-out: no frame recorded in the
held-out take may train on it. With one take per pose that means the held-out
pose has no centroid at all, so the number is not meaningful and the report
says so rather than quoting it.

Two kinds of camera take, told apart by the file itself and never by a flag:

  recordings/sync/cam/     MediaPipe. A `world` key per line: 21 normalised
                           landmarks, a hand shape with no size. Aligned onto
                           the glove with rotation + scale (Umeyama).
  recordings/sync/leap/    Ultraleap. `source: "leap"`, the glove's own
                           26-joint schema plus camera extras, read back with
                           FrameRecorder.load + frame_to_keypoints21 — real
                           metres. Aligned by PALM BASIS — rotation and
                           wrist translation, scale 1 — because both sides are
                           already metric (plan section 3, fusion.py).

A session folder may hold both; each take is read the way its own first line
says, and the report names which camera every take came from.

  recordings/sync/rejected/  attempts the recorder threw out — the hand was
                             lost, the coverage was short, or both sensors
                             said the hand was not in the pose that was asked
                             for. They are kept (never deleted) so that every
                             exclusion can be counted and looked at, and they
                             are NOT fused: takes are found under `glove/`,
                             which is a sibling of `rejected/`, so a rejected
                             attempt cannot reach this report by accident.
                             `scripts/check_take_labels.py` skips it too.

Also reports the plumbing that has to be right for any of it to mean anything:
how many frames found a partner within the time window, and how often the
camera was confident enough to contribute.

TWO WAYS OF LEAVING EVIDENCE OUT, AND WHY THEY ARE DIFFERENT
  --exclude names takes that should never have been recorded — a mislabelled
            attempt, a take of the wrong pose. The take is dropped on BOTH
            sensors and the report names it, so "which 59 of the 60" is
            answerable from the report itself rather than from a folder
            somebody copied.
  --profile names DEGREES OF FREEDOM this glove is known to get wrong, per
            hand, and masks them out of the gates' evidence. That is a much
            stronger claim, so it is never made quietly: the report prints
            the fusion BOTH ways — ordinary and masked — and the reader can
            see what the mask bought. It defaults to `auto`, which applies
            `profiles/default.json` if it exists and the NK profile
            otherwise, because a mask that had to be remembered as a flag was
            a mask that mostly did not get applied. `--profile none` fuses
            unmasked.

THREE THINGS THIS REPORT CORRECTS FOR, EACH MEASURED AND EACH REFUSABLE
  --profile       the glove's known-bad fingers, above. Default `auto`.
  --glove-lag     the glove's solved hand TRAILS the camera — roughly 100 ms
                  on the left hand and 450-485 ms on the right, on this
                  laptop. Default `auto`: measured per hand from the SETTLE
                  clips the coached recorder writes beside each take, else
                  from takes whose camera curl moved enough, and applied only
                  where a trustworthy estimate exists. A session of held
                  poses reports "not measurable" and nothing is applied,
                  which is the right answer: a held pose is lag-insensitive.
  --fit-template  the glove reports XR TRAINER'S TEMPLATE HAND, whose bones
                  are not the operator's; the camera measures the real ones
                  in millimetres. Default `auto`: measure per hand from this
                  session's own camera frames, save the measurement beside
                  the report, and fuse with the template rescaled to it —
                  every rotation kept, so only the lengths change. The
                  `glove` column and the glove-only classifier stay on the
                  RAW template, because that is what the glove alone gives.

Usage:
  python scripts/fuse_poses.py                          # recordings/sync
  python scripts/fuse_poses.py recordings/sync --write  # also REPORT.txt
  python scripts/fuse_poses.py --export-csv fused.csv   # fused 21-kp CSV
  python scripts/fuse_poses.py --max-dt 0.1             # looser time matching
  python scripts/fuse_poses.py recordings/sync_day2 --exclude pinch_right_take1
  python scripts/fuse_poses.py recordings/sync_day1 --profile none \
      --glove-lag none --fit-template none             # the 2026-09-20 report
  python scripts/fuse_poses.py recordings/sync --glove-lag left:0.10,right:0.46
"""
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cam_hand.export21 import wrist_centered
from cam_hand.features import (
    ALL_COLS,
    ALL_NAMES,
    DESCRIPTIVE_ROWS,
    DOF_ROWS,
    FLEXION_COLS,
    FLEXION_NAMES,
    all_features,
    dof_values,
    flexion_features,
    loo_take_nearest_centroid,
    mean_vector,
    median,
    palm_normal,
)
from cam_hand.fusion import (
    AUTO,
    CAMERA_DOFS,
    DEFAULT_GATES,
    DEFAULT_RAIL,
    GATED_DOFS,
    NOT_MEASURABLE,
    RAIL_FINGERS,
    SENSORS,
    SPREAD_FINGERS,
    SRC_RAIL,
    GateParams,
    RailOverrideParams,
    RailOverrideTracker,
    curl_gates_from_rails,
    estimate_glove_lag,
    flag_hand_id_stability,
    fuse_skeletons,
    learn_flexion_scale,
    learn_rails,
    pair_by_time,
    pairing_clock,
    palm_fit_rmse_mm,
)
from cam_hand.landmarks import MP21_NAMES
from cam_hand.recorder import SETTLE_SUFFIX, CamRecorder, take_files
from cam_hand.template_fit import (
    MIN_FIT_FRAMES,
    finger_scales,
    fit_refusal,
    fit_template,
    load_measurement,
    measure_hand,
    merge_measurements,
)

from xr_hand.keypoints21 import MP21_TO_OPENXR_IDX, frame_to_keypoints21
from xr_hand.recorder import FrameRecorder

COORD_COLS = [f"{n}_{a}" for n in MP21_NAMES for a in ("x", "y", "z")]

LEAP = "leap"
MEDIAPIPE = "mediapipe"
# Where record_simultaneous puts each backend's camera takes. Both are read.
CAM_DIRS = ("cam", LEAP)

# --- where a reliability profile comes from when nobody names one ------
#
# Resolved against the REPO's profiles/ folder rather than the working
# directory: `python scripts\fuse_poses.py` is run from wherever the operator
# happens to be, and a default that silently depends on that is a default that
# silently stops applying.
PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
PROFILE_AUTO = "auto"
PROFILE_NONE = "none"
# In order. `default.json` is the operator's own glove pair, copied there so
# that swapping gloves or operators is one file to replace rather than a flag
# to remember; the NK profile is the fallback so that a checkout without a
# default.json still gets the masks that were measured on this hardware.
PROFILE_ORDER = ("default.json", "reality_glove_nk_2026-09.json")

# --- --glove-lag ------------------------------------------------------
LAG_AUTO = "auto"
LAG_NONE = "none"
# A MEASURED lag is applied only when several clips agree about it.
#
# `estimate_glove_lag` already refuses a clip whose camera did not move and a
# shift that does not explain the two traces, so one trustworthy estimate is a
# real measurement — of ONE transition, with nothing to check it against. A
# cross-correlation peak can be a real peak on a real transition and still be
# the wrong peak: a hand that closes and opens twice in a clip has a second
# maximum a whole cycle away, and the pairing shift is then a lie about every
# frame of the session rather than about that clip. The cheapest check there
# is, is another clip.
#
# So: at least MIN_LAG_CLIPS trustworthy estimates, agreeing to within
# MAX_LAG_MAD of their median (median absolute deviation, not the range: with
# a handful of clips one outlier must not be able to veto the other four).
# 60 ms is well under the 100 / 460 ms the two gloves measure on this laptop
# and comfortably over a 90 Hz camera frame's 11 ms plus a 60 Hz glove
# frame's 17 ms, so genuine estimates of one lag clear it and two different
# lags do not. Anything else reports the estimates and applies NOTHING: an
# unmeasured lag is not a zero lag, but pairing on the raw stamps is what the
# pipeline did before and is the only honest default.
MIN_LAG_CLIPS = 2
MAX_LAG_MAD = 0.060
NOT_CORROBORATED = "not corroborated"
NOT_AGREED = "estimates do not agree"

# --- --fit-template ---------------------------------------------------
FIT_AUTO = "auto"
FIT_NONE = "none"
# The pose whose frames the camera has actually measured a hand on, rather
# than inferred one: nothing is occluded on an open palm.
OPEN_POSE = "open_palm"
# Where `auto` leaves the measurement it took: beside the report, in the
# session it was measured from, because that is the session it describes.
FIT_FILE = "template_{hand}.json"


def load_glove(path: Path):
    """Glove JSONL -> dicts with the clocks, hand_side, pose, take, pts (21x3 m).

    `capture_time` — when the OSC packet arrived, if the file has it — is
    carried through, because that is the clock `pair_by_time` prefers.
    Recordings made before it existed simply do not have the key.

    `frame` is the parsed `HandFrame` itself, kept because `--fit-template`
    rescales the glove's parent-relative joint OFFSETS and then re-runs the
    forward kinematics: the 21 points here are the template's, and there is
    no way back from them to the offsets they came from.
    """
    with open(path, "r", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    out = []
    for d, (frame, wall) in zip(labels, FrameRecorder.load(path)):
        out.append({
            "wall_time": wall,
            "capture_time": d.get("capture_time"),
            "hand_side": frame.hand_side,
            "pose": d.get("pose", ""),
            "take": d.get("take", ""),
            "frame": frame,
            "pts": frame_to_keypoints21(frame),
        })
    return out


def camera_source(path: Path) -> str:
    """Which camera wrote this take: 'leap' or 'mediapipe'.

    Read off the first line's `source` key, not off the folder name, so a
    file that was moved, renamed or handed over still says what it is.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return (LEAP if json.loads(line).get("source") == LEAP
                        else MEDIAPIPE)
    return MEDIAPIPE


def load_leap_cam(path: Path):
    """Leap JSONL -> the same row dicts load_cam returns, in metres.

    `FrameRecorder.load` reads these files unchanged — that is the whole
    point of the Leap recorder's schema — and `frame_to_keypoints21` runs the
    same forward kinematics the glove side does, so both skeletons arrive in
    the 21-keypoint layout, wrist-centred, in real metres.

    There is no confidence to carry: LeapC reports `confidence` as a constant
    1.0, so gating on it would be gating on nothing. A line in the file is a
    frame the tracker actually reported, and `score` is 1.0 to say so.

    The camera-only timing keys ride along untouched — `capture_time` because
    `pair_by_time` pairs on it, `timestamp_us` and `frame_age_us` because
    they are what any later question about latency or cadence is answered
    from, and dropping them here would mean re-reading the file to ask.

    The CAPTURE FACTS the gates read ride along too, and they are facts about
    the capture rather than the tracker's opinion of it: how long LeapC has
    held this hand (`visible_time_us`), which track it is (`hand_id`, turned
    into `hand_id_stable` once the whole take is in hand), and where the palm
    sat in the module's field. `palm_normal_abs` is computed here, in ABSOLUTE
    leap space, because the wrist-centred `pts` have thrown away where the
    module is — and the module is the origin of that space, so the palm's
    absolute position is also the direction it was seen from.
    """
    with open(path, "r", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    out = []
    for d, (frame, wall) in zip(labels, FrameRecorder.load(path)):
        abs26 = d.get("abs26")
        normal = None
        if abs26 is not None:
            normal = palm_normal([abs26[i] for i in MP21_TO_OPENXR_IDX],
                                 frame.hand_side)
        out.append({
            "wall_time": wall,
            "capture_time": d.get("capture_time"),
            "timestamp_us": d.get("timestamp_us"),
            "frame_age_us": d.get("frame_age_us"),
            "hand_side": frame.hand_side,
            "pose": d.get("pose", ""),
            "take": d.get("take", ""),
            "score": 1.0,
            "hand_id": d.get("hand_id"),
            "visible_time_us": d.get("visible_time_us"),
            "palm_abs": d.get("palm_abs"),
            "palm_normal_abs": normal,
            # The camera's RAW 26 joint positions in camera space, which is
            # what a BONE LENGTH has to be read off: `pts` has dropped the
            # metacarpals, and `template_fit.measure_hand` measures the whole
            # 26-joint chain. Dropped again by `measure_take` as soon as the
            # medians are in hand — see there for why.
            "abs26": abs26,
            "pts": frame_to_keypoints21(frame),
        })
    return out


def load_cam(path: Path):
    """One camera take, whichever camera wrote it -> (rows, source)."""
    source = camera_source(path)
    if source == LEAP:
        return load_leap_cam(path), source
    out = []
    for d in CamRecorder.load(path):
        if "world" not in d:
            continue
        out.append({
            "wall_time": d["wall_time"],
            "capture_time": d.get("capture_time"),
            "hand_side": d["hand_side"],
            "pose": d.get("pose", ""),
            "take": d.get("take", ""),
            "score": d.get("score", 1.0),
            "pts": wrist_centered(d),
        })
    return out, source


class AmbiguousTake(RuntimeError):
    """One take name, a camera file under cam/ AND under leap/."""


def find_camera_take(input_dir: Path, name: str, camera: str = AUTO):
    """The camera file paired with a glove take, in cam/ or in leap/.

    With `camera` set to a folder, only that folder is looked in. In `auto`,
    a name present in BOTH folders raises: the two were recorded by different
    sensors, they align differently (scaled vs rigid), and quietly preferring
    whichever folder is listed first would pick one on an implementation
    detail and report numbers for a take nobody chose.
    """
    if camera != AUTO:
        candidate = input_dir / camera / name
        return candidate if candidate.is_file() else None

    found = [input_dir / folder / name for folder in CAM_DIRS
             if (input_dir / folder / name).is_file()]
    if len(found) > 1:
        raise AmbiguousTake(
            f"{name} exists under two cameras:\n    "
            + "\n    ".join(str(p) for p in found)
            + "\n  They are different sensors and are aligned differently, so "
              "pick one with --camera "
              f"{{{','.join(CAM_DIRS)}}} (or move the take you do not want).")
    return found[0] if found else None


def loo_table(samples, cols, label, lines):
    """samples: [(pose, hand, take, features)] -> print a scored row.

    Leave-one-TAKE-out: the held-out sample's whole take leaves the training
    set, so the other hand of the same five seconds cannot vote for it.
    """
    if not samples:
        lines.append(f"  {label:<14} (no samples)")
        return None
    ok, n, wrong = loo_take_nearest_centroid(
        [(s[0], s[2], s[3]) for s in samples], cols)
    lines.append(f"  {label:<14} {ok:>3}/{n} correct ({100.0 * ok / n:3.0f}%)")
    return {p for p, *_ in (samples[i] for _, _, i in wrong)}


def cam_meta_of(row, source):
    """The capture facts fusion gates on, or None if this camera has none.

    A MediaPipe frame has no absolute palm, no hand id and no visibility
    clock, so it returns None and is fused ungated, exactly as before. That is
    not a loophole: a gate needs evidence, and there is none to read.
    """
    if source != LEAP or row.get("palm_abs") is None:
        return None
    return {"visible_time_us": row.get("visible_time_us"),
            "hand_id_stable": row.get("hand_id_stable"),
            "palm_abs": row.get("palm_abs"),
            "palm_normal_abs": row.get("palm_normal_abs")}


# Which gated DOFs move each row of the per-DOF table. A curl of the four
# fingers is the glove's by construction and appears with no owner at all —
# the table shows it precisely because it is what the camera must NOT change.
ROW_OWNERS = {
    "curl thumb": ("thumb",),
    # A finger's curl is the glove's unless the rail-disagreement override
    # takes it, which is the whole point of listing them here: the row that
    # used to read "glove (by design)" now has to say when the design was
    # overruled.
    "curl index": ("curl index",),
    "curl middle": ("curl middle",),
    "curl ring": ("curl ring",),
    "curl pinky": ("curl pinky",),
    "spread index-middle": ("spread index", "spread middle"),
    "spread middle-ring": ("spread middle", "spread ring"),
    "spread ring-pinky": ("spread ring", "spread pinky"),
    "thumb-index gap": ("thumb", "spread index"),
}


def row_owner(label, sources, n_paired):
    """The 'from' column: how much of this row the camera actually supplied."""
    if label in DESCRIPTIVE_ROWS:
        return "descriptive"
    owners = ROW_OWNERS.get(label, ())
    if not owners:
        return "glove (by design)"
    if not n_paired:
        return "glove (no camera)"
    hits = rail_hits = 0
    for frame in sources:
        values = [frame.get(d) for d in owners]
        if any(v is not None and v.startswith("camera") for v in values):
            hits += 1
        if any(v == SRC_RAIL for v in values):
            rail_hits += 1
    if hits == 0:
        return "glove"
    what = SRC_RAIL if rail_hits else "camera"
    return f"{what} {100.0 * hits / n_paired:.0f}%"


def dof_table(per_pose, lines):
    """The headline: glove / camera / fused, per DOF, per pose and hand."""
    lines.append("Per-DOF values — median over the paired frames of each take")
    lines.append("  curls are tip-to-wrist over palm length; spreads are the "
                 "in-plane angle between")
    lines.append("  adjacent PROXIMAL bones, in degrees; the thumb-index gap "
                 "is tip-to-tip over palm length.")
    lines.append("  The camera column is blank where no camera frame paired "
                 "with that take at all.")
    lines.append("  'thumb-index gap' is where the thumb's contribution "
                 "actually shows. (The old 'spread")
    lines.append("  thumb-index' row, the angle of the thumb's BASE bone, is "
                 "gone: the camera supplies the")
    lines.append("  thumb's whole DIRECTION and it is grafted onto the glove "
                 "template's own CMC, so that")
    lines.append("  angle described neither hand.)")
    lines.append("  'thumb dir elevation/azimuth' are the CMC-to-tip "
                 "direction in the palm frame —")
    lines.append("  elevation is opposition, out of the palm plane. They are "
                 "marked DESCRIPTIVE because")
    lines.append("  the fused thumb is the camera's by construction: agreeing "
                 "with the camera column")
    lines.append("  restates what fusion did and validates nothing.")
    lines.append("")
    lines.append(f"  {'pose':<12} {'hand':<6} {'tk':<3} {'DOF':<20} "
                 f"{'glove':>8} {'camera':>8} {'fused':>8}   from")
    for (pose, hand, take) in sorted(per_pose):
        rows = per_pose[(pose, hand, take)]
        for k, (label, _kind, _i) in enumerate(DOF_ROWS):
            g = median([v[k] for v in rows["glove"]])
            f = median([v[k] for v in rows["fused"]])
            c = (median([v[k] for v in rows["camera"]])
                 if rows["camera"] else None)
            c_txt = f"{c:8.2f}" if c is not None else "       -"
            lines.append(f"  {pose:<12} {hand:<6} {take:<3} {label:<20} "
                         f"{g:8.2f} {c_txt} {f:8.2f}   "
                         f"{row_owner(label, rows['sources'], rows['paired'])}")
        lines.append("")


def camera_use_table(dof_used, dof_total, lines, indent="  "):
    """The one table that says how much of the hand the camera supplied.

    Pulled out of `gate_tables` so the reliability-profile comparison can
    print it twice — once ungated by the profile, once with it — from the
    same code. Two tables built two ways would be an invitation to compare
    numbers that are not comparable.
    """
    for dof in GATED_DOFS:
        n = dof_total.get(dof, 0)
        pct = 100.0 * dof_used.get(dof, 0) / n if n else 0.0
        lines.append(f"{indent}{dof:<16} {dof_used.get(dof, 0):>5}/{n:<5} "
                     f"{pct:5.1f}%")


def scale_lines(scale, gates, lines):
    """The endpoints the thumb vote's flexion fractions are taken on.

    Printed per hand, per finger, per SENSOR, because that is the claim: the
    two sensors do not share a scale, so each is put on its own before they
    are compared. A gate on a normalised quantity is only checkable if the
    normalisation is printed beside it — the raw tolerance it replaced was
    checkable because the units were the report's own.
    """
    lines.append("Thumb vote — flexion fractions (the gate compares the two "
                 "sensors on EACH sensor's")
    lines.append("  own endpoints: frac = (open - curl) / (open - flexed), "
                 "clipped to [-0.2, 1.2], and the")
    lines.append("  median |frac_glove - frac_camera| over the voting fingers "
                 "is what agree_tol_frac")
    lines.append("  thresholds. A raw curl difference could not be compared "
                 "across a template fit: the")
    lines.append("  fit moves every glove curl, so the same raw tolerance is "
                 "a different strictness.)")
    if scale is None or not scale:
        lines.append("  (no endpoints learned; the raw curl_agree_tol is in "
                     "force)")
        lines.append("")
        return
    lines.append(f"  {'hand':<6} {'finger':<7} {'sensor':<7} {'open':>7} "
                 f"{'flexed':>7} {'span':>7}   how")
    for hand in sorted(scale.hands):
        hs = scale.hands[hand]
        for finger in SPREAD_FINGERS:
            for sensor in SENSORS:
                got = hs.endpoints(sensor, finger)
                if got is None:
                    lines.append(f"  {hand:<6} {finger:<7} {sensor:<7} "
                                 f"{'-':>7} {'-':>7} {'-':>7}   (none)")
                    continue
                lines.append(f"  {hand:<6} {finger:<7} {sensor:<7} "
                             f"{got.open:7.3f} {got.flexed:7.3f} "
                             f"{got.span:7.3f}   {got.how}")
    lines.append("  glove open is the finger's learned RAIL on the hand being "
                 "fused (the fitted one when a")
    lines.append("  fit is in force); camera open is the median over the "
                 "session's open-palm-LIKE frames —")
    lines.append("  every one of index..pinky above cam_open_curl - margin, "
                 "the camera's own test, never")
    lines.append("  the glove's claim and never a pose label. Both flexed "
                 "ends are that sensor's 2nd")
    lines.append("  percentile over the session.")
    dropped = {(h, f): why for h, hs in sorted(scale.hands.items())
               for f, why in sorted(hs.dropped.items()) if f in SPREAD_FINGERS}
    if dropped:
        lines.append("  DROPPED from the vote (could not be normalised, so "
                     "they cast no vote this run):")
        for (hand, finger), why in dropped.items():
            lines.append(f"    {hand:<6} {finger:<7} {why}")
    else:
        lines.append(f"  every voting finger cleared min_glove_span "
                     f"{gates.min_glove_span:.2f} and min_cam_span "
                     f"{gates.min_cam_span:.2f}; none was dropped.")
    lines.append("")


def gate_tables(dof_used, dof_total, reasons, gates, rail_params, rails,
                unreliable, lines, scale=None):
    lines.append("Camera-use rate per gated DOF "
                 "(share of paired frames the camera actually supplied)")
    camera_use_table(dof_used, dof_total, lines)
    lines.append("  The four 'curl' rows are the rail-disagreement override, "
                 "and read 0% unless it is")
    lines.append("  enabled on that finger: a curl is the glove's by design "
                 "and the override is the")
    lines.append("  one thing that can take it.")
    lines.append("")
    lines.append("Why the glove kept a DOF (counted over DOF x paired frame)")
    if not reasons:
        lines.append("  (nothing was rejected)")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n:>6}  {why}")
    lines.append("")
    lines.append("Gate thresholds in force (empirical starting points from the "
                 "first real session,")
    lines.append("  not calibrated constants — every one is a named parameter)")
    for k, v in gates.described().items():
        lines.append(f"  {k:<22} {v}")
    lines.append("  Only fingers whose GLOVE curl is a measurement vote on "
                 "whether the camera has")
    lines.append("  the hand right. Excluded: a DISPUTED finger (on its rail "
                 "while the camera reads")
    lines.append("  it flexed), one the rail override has taken, and one "
                 "named --unreliable. A railed")
    lines.append("  finger the camera AGREES is extended still votes — that "
                 "agreement is evidence.")
    lines.append("  Below min_usable_fingers the glove casts no veto and the "
                 "camera's own geometry")
    lines.append("  decides.")
    lines.append(f"  {'unreliable fingers':<22} "
                 + ("; ".join(f"{h}: {', '.join(f)}"
                              for h, f in sorted(unreliable.items()))
                    if unreliable else "(none)"))
    lines.append("  ...and a finger whose curl range on either sensor was too "
                 "small to learn endpoints")
    lines.append("  from is dropped too — see the flexion-fraction table "
                 "below.")
    lines.append("")
    scale_lines(scale, gates, lines)
    lines.append("Rail-disagreement override (the glove's curl is a CONSTANT "
                 "at full extension;")
    lines.append("  when a trusted camera sees that finger flexed anyway, the "
                 "camera wins that curl)")
    if rail_params is None:
        lines.append("  DISABLED (--no-rail-override)")
        return
    for k, v in rail_params.described().items():
        lines.append(f"  {k:<22} {v}")
    lines.append("  rails learned from this session's own glove frames "
                 "(mode of the curl, per hand and finger):")
    if not rails:
        lines.append("    (none — no finger sat at the top of its range "
                     "often enough to teach one)")
    for (hand, finger), value in sorted(rails.items()):
        # Per hand: the override can be enabled on the right ring finger and
        # not on the left, and the table has to say which of the two this
        # row is.
        mark = ("  <- enabled" if finger in rail_params.fingers_for(hand)
                else "")
        lines.append(f"    {hand:<6} {finger:<7} {value:.3f}{mark}")


# --- what the command line asked for, as values ------------------------

def parse_unreliable(specs):
    """`['right:ring,pinky']` -> `{'right': ('ring', 'pinky')}`.

    Per hand, because a glove fails per hand: on sync_day1 the RIGHT glove's
    ring and pinky read partly extended through poses the left glove reports
    correctly (see `fuse_skeletons`).
    """
    out = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"--unreliable {spec}: expected HAND:FINGER[,FINGER], "
                             "e.g. right:ring,pinky")
        hand, _, fingers_txt = spec.partition(":")
        picked = tuple(f.strip().lower() for f in fingers_txt.split(",")
                       if f.strip())
        bad = [f for f in picked if f not in FLEXION_NAMES]
        if bad:
            raise SystemExit(f"--unreliable {spec}: {', '.join(bad)} is not a "
                             f"finger (choose from {', '.join(FLEXION_NAMES)})")
        out[hand.strip().lower()] = picked
    return out


def parse_rail_fingers(text):
    """`'index,ring'` -> `('index', 'ring')`, for BOTH hands."""
    picked = tuple(f.strip().lower() for f in str(text).split(",") if f.strip())
    bad = [f for f in picked if f not in RAIL_FINGERS]
    if bad:
        raise SystemExit(
            f"--rail-fingers: {', '.join(bad)} is not a finger the rail "
            f"override can act on (choose from {', '.join(RAIL_FINGERS)}).\n"
            "  The thumb is not among them: its whole direction is already "
            "the camera's when the thumb gate passes.")
    return picked


# Everything a profile may hold. A key outside this list is an error rather
# than something to ignore: `rail-fingers` written for `rail_fingers` would
# leave the override at its default and nothing in the report would say so.
PROFILE_KEYS = ("name", "comment", "unreliable", "rail_fingers")


def load_profile(path):
    """A reliability profile -> (unreliable, rail_fingers, name, comment).

    A profile is the operator's standing knowledge about THIS glove, written
    down once instead of retyped as flags every run:

        {
          "name": "Reality Glove, N Kim, September 2026",
          "comment": "why these fingers and not others",
          "unreliable":   {"right": ["middle", "ring", "pinky"]},
          "rail_fingers": {"right": ["index"], "left": ["index"]}
        }

    Both blocks are PER HAND, because that is how gloves fail: the same
    garment reports the right ring finger partly extended through poses the
    left one gets right, and a mask that covered both hands would throw away
    the good half of the evidence.

    `unreliable` names fingers whose GLOVE curl is not to be trusted on that
    hand; they stop voting on whether the camera has the hand right.
    `rail_fingers` names the fingers the rail-disagreement override may act
    on. Neither is a calibration: both are claims about hardware that a
    finger sweep is supposed to have established, which is why the file has a
    `comment` field and why the report prints it.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"--profile {path}: {e}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"--profile {path}: not valid JSON ({e})")
    if not isinstance(data, dict):
        raise SystemExit(f"--profile {path}: expected a JSON object with the "
                         f"keys {', '.join(PROFILE_KEYS)}")
    unknown = sorted(k for k in data if k not in PROFILE_KEYS)
    if unknown:
        raise SystemExit(
            f"--profile {path}: unknown key(s) {', '.join(unknown)}.\n"
            f"  A profile holds {', '.join(PROFILE_KEYS)}. A misspelled key "
            "would leave the mask off and\n  nothing would say so, so it is "
            "an error rather than something to ignore.")

    def per_hand(key, valid):
        block = data.get(key)
        if block is None:
            return {}
        if not isinstance(block, dict):
            raise SystemExit(
                f"--profile {path}: '{key}' must be an object keyed by hand, "
                'e.g. {"right": ["index"]} - a bare list cannot say which '
                "hand it is about.")
        out = {}
        for hand, fingers in block.items():
            if isinstance(fingers, str):
                fingers = fingers.split(",")
            if not isinstance(fingers, (list, tuple)):
                raise SystemExit(f"--profile {path}: '{key}' -> {hand} must be "
                                 "a list of finger names")
            picked = tuple(str(f).strip().lower() for f in fingers
                           if str(f).strip())
            bad = [f for f in picked if f not in valid]
            if bad:
                raise SystemExit(
                    f"--profile {path}: '{key}' names {', '.join(bad)} on the "
                    f"{hand} hand; choose from {', '.join(valid)}")
            side = str(hand).strip().lower()
            if side not in ("left", "right"):
                raise SystemExit(f"--profile {path}: '{key}' is keyed by "
                                 f"{hand!r}; the hands are 'left' and 'right'")
            out[side] = picked
        return out

    return (per_hand("unreliable", FLEXION_NAMES),
            per_hand("rail_fingers", RAIL_FINGERS),
            str(data.get("name", "")).strip(),
            str(data.get("comment", "")).strip())


def resolve_profile(spec, profiles_dir=None):
    """`--profile`'s value -> (path or None, how it was chosen).

    A profile is standing knowledge about THE GLOVE IN THE ROOM, and the
    measured masks for this hardware were sitting in `profiles/` being applied
    only when somebody remembered the flag. So the default is `auto`:

      profiles/default.json                    if it exists — the operator's
                                               own pair, the file to replace
                                               when the glove or the wearer
                                               changes
      profiles/reality_glove_nk_2026-09.json   else, if it exists — the
                                               profile these gates were
                                               measured against
      nothing                                  else

    `none` turns it off, and an explicit path is used as given (a missing one
    is an error from `load_profile`, not a silent fall back to auto — asking
    for a named profile and getting a different one is worse than stopping).

    The report always names the file and this string, because "which mask was
    applied" must be answerable from the report rather than from the command
    line somebody typed.
    """
    profiles_dir = Path(profiles_dir or PROFILES_DIR)
    text = str(spec).strip()
    if text.lower() == PROFILE_NONE:
        return None, "none (--profile none)"
    if text.lower() == PROFILE_AUTO:
        for name in PROFILE_ORDER:
            candidate = profiles_dir / name
            if candidate.is_file():
                return candidate, f"auto -> {candidate}"
        return None, (f"auto -> none: neither {' nor '.join(PROFILE_ORDER)} "
                      f"is in {profiles_dir}")
    return Path(text), "named on the command line"


def parse_glove_lag(text):
    """`--glove-lag`'s value -> "auto", "none", or `{'right': 0.46}`.

    Per hand, because the two gloves are separate garments on separate
    stretch sensors: measured on this laptop the left hand's solved hand
    trails the camera by about 100 ms and the right hand's by 450-485 ms.
    """
    raw = str(text).strip()
    if raw.lower() in (LAG_AUTO, LAG_NONE):
        return raw.lower()
    out = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(
                f"--glove-lag {text}: expected {LAG_NONE}, {LAG_AUTO}, or "
                "HAND:SECONDS[,HAND:SECONDS], e.g. 'left:0.10,right:0.46'")
        hand, _, value = part.partition(":")
        side = hand.strip().lower()
        if side not in ("left", "right"):
            raise SystemExit(f"--glove-lag {text}: {hand!r} is not a hand; "
                             "the hands are 'left' and 'right'")
        try:
            out[side] = float(value)
        except ValueError:
            raise SystemExit(f"--glove-lag {text}: {value!r} is not a number "
                             "of seconds")
    if not out:
        raise SystemExit(f"--glove-lag {text}: nothing to apply")
    return out


@dataclass
class LagRow:
    """One hand's glove lag: what was measured, from what, and what was used.

    `applied` is separate from `seconds` on purpose. A lag that was measured
    and a lag that was USED are different claims, and the report has to be
    able to say "0.46 s on the right, from 12 clips, applied" and also "not
    measurable, nothing applied" without the reader having to work out which
    from a single number.
    """

    hand: str
    seconds: float = 0.0
    applied: bool = False
    n_clips: int = 0
    n_trusted: int = 0
    source: str = ""
    why: str = ""
    # Every trustworthy estimate, in seconds, and how far they scatter around
    # their own median. Carried so that a refusal can PRINT the numbers it
    # refused: "two clips disagreed" is only checkable if the report says by
    # how much and about what.
    estimates: tuple = ()
    mad: float = 0.0

    def _estimates_text(self) -> str:
        if not self.estimates:
            return ""
        return ("; estimates "
                + ", ".join(f"{s * 1000.0:+.0f}" for s in self.estimates)
                + f" ms, MAD {self.mad * 1000.0:.0f} ms")

    def described(self) -> str:
        if self.source == "given":
            return f"{self.seconds * 1000.0:+.0f} ms as given, APPLIED"
        if self.applied:
            return (f"{self.seconds * 1000.0:+.0f} ms from {self.n_trusted} of "
                    f"{self.n_clips} {self.source} clip(s), MAD "
                    f"{self.mad * 1000.0:.0f} ms, APPLIED")
        return (f"{self.why or NOT_MEASURABLE} "
                f"({self.n_trusted} of {self.n_clips} {self.source} clip(s) "
                f"measurable{self._estimates_text()}), nothing applied")


def settle_clips(input_dir: Path, cam_dirs):
    """Every `<take>.settle.jsonl` pair in a session: (glove path, cam path).

    The settle clip is the open-palm -> pose TRANSITION the coached recorder
    writes beside each accepted take (`cam_hand.recorder.SETTLE_SUFFIX`). It
    is the only thing in a coached session that MOVES, so it is the only thing
    a lag can be measured on; the takes themselves are held poses, and two
    flat traces have no alignment.
    """
    out = []
    glove_dir = input_dir / "glove"
    if not glove_dir.is_dir():
        return out
    for gpath in sorted(glove_dir.glob("*" + SETTLE_SUFFIX)):
        for d in cam_dirs:
            cpath = d / gpath.name
            if cpath.is_file():
                out.append((gpath, cpath))
                break
    return out


def lag_from_clips(clips, hand: str, source: str,
                   min_clips: int = MIN_LAG_CLIPS,
                   max_mad: float = MAX_LAG_MAD) -> LagRow:
    """The median trustworthy lag over several clips, for one hand.

    `clips` is (glove rows, camera rows) per clip, already loaded. The median
    and not the mean: one clip in which the tracker lost the hand mid-
    transition produces a lag that is not wrong by a little.

    IT IS ONLY APPLIED WHEN SEVERAL CLIPS AGREE. One trustworthy estimate is
    a measurement of one transition with nothing to check it against, and a
    cross-correlation can find a real peak that is the wrong peak (see
    `MIN_LAG_CLIPS`). So `applied` needs `min_clips` trustworthy estimates
    scattering by no more than `max_mad` around their own median; short of
    that the row carries the estimates, the reason, and applies nothing.

    The median and the MAD are computed and reported either way, because
    "these two clips disagree" is only a checkable claim if the report says
    what they said.
    """
    row = LagRow(hand=hand, source=source, n_clips=len(clips))
    measured, refused = [], []
    for glove, cam in clips:
        got = estimate_glove_lag(glove, cam, hand=hand)
        if got.trusted:
            measured.append(got.seconds)
        elif got.n_glove and got.n_cam:
            refused.append(got)
    row.n_trusted = len(measured)
    row.estimates = tuple(sorted(float(v) for v in measured))
    if not measured:
        # The reason printed is the BEST candidate's, not the last clip's: a
        # session of 59 takes refuses most of them for having no frames of
        # this hand at all, and that says nothing about why the hand it does
        # have was not measurable. The best candidate is the clip whose camera
        # moved the most.
        best = max(refused, key=lambda e: e.cam_spread, default=None)
        row.why = (best.why if best is not None
                   else f"{NOT_MEASURABLE}: no clip holds this hand")
        return row
    row.seconds = float(median(measured))
    row.mad = float(median([abs(v - row.seconds) for v in measured]))
    if len(measured) < min_clips:
        row.why = (
            f"{NOT_CORROBORATED}: {len(measured)} trustworthy {source} "
            f"clip(s), need {min_clips}. One clip measures one transition "
            "and nothing checks it")
        return row
    if row.mad > max_mad:
        row.why = (
            f"{NOT_AGREED}: {len(measured)} trustworthy {source} clip(s) "
            f"scatter by {row.mad * 1000.0:.0f} ms around their median "
            f"{row.seconds * 1000.0:+.0f} ms, over the {max_mad * 1000.0:.0f} "
            "ms they must agree within")
        return row
    row.applied = True
    return row


def glove_lag_of(spec, input_dir: Path, cam_dirs, loaded):
    """`(per-hand LagRow, the mapping pair_by_time gets)` for this session.

    AUTO measures, per hand, from the SETTLE clips first — that is what they
    are written for — and falls back to the takes for a hand that has no
    APPLICABLE settle estimate, because a session recorded before the clips
    existed may still hold a take the hand moved during. A hand with no
    applicable estimate anywhere gets NOTHING applied: an unmeasured lag is
    not a zero lag, but pairing on the raw stamps is what the pipeline did
    before and is the only honest default. "Applicable" is stricter than
    "trustworthy" — see `lag_from_clips`: several clips have to agree before
    a measured lag is used at all.

    A manual value is applied exactly as given, and the report says so. The
    point of allowing it is to be able to CHECK a lag against a session
    (`--glove-lag right:0.46` on day 1's static takes moves nothing, which is
    what proves the shift is harmless on held poses).
    """
    hands = sorted({r["hand_side"] for entry in loaded for r in entry["glove"]})
    if spec == LAG_NONE:
        return {}, None
    if isinstance(spec, dict):
        rows = {h: LagRow(hand=h, seconds=v, applied=True, source="given")
                for h, v in spec.items()}
        return rows, dict(spec)

    clips = []
    for gpath, cpath in settle_clips(input_dir, cam_dirs):
        glove = load_glove(gpath)
        cam, _source = load_cam(cpath)
        if glove and cam:
            clips.append((glove, cam))
    takes = [(entry["glove"], entry["cam"]) for entry in loaded]

    rows, applied = {}, {}
    for hand in hands:
        row = lag_from_clips(clips, hand, "settle") if clips else None
        if row is None or not row.applied:
            # No settle clip, or none of this hand's clips moved enough: the
            # takes are the only other thing there is to measure on.
            fallback = lag_from_clips(takes, hand, "take")
            row = fallback if fallback.applied or row is None else row
        rows[hand] = row
        if row.applied:
            applied[hand] = row.seconds
    return rows, (applied or None)


def fit_measurements(spec, measure_parts, loaded, input_dir: Path):
    """Measure (or load) each hand, put it on the template, and report both.

    Returns `(measurements, per-finger scales, saved files, refusals)`. The
    fitted 21 points are written onto each glove row as `fit_pts`, so
    `fuse_all` has one thing to read and the raw template stays beside it
    untouched — the report needs both, and recomputing either later would
    mean keeping the measurement and the forward kinematics alive in two
    places.

    A hand whose session does not hold enough OPEN-palm frames is refused
    (`template_fit.fit_refusal`) and is not fitted at all: it appears in
    `refusals` with the reason, the report prints it, and that hand fuses on
    the raw template. A fit measured on frames the tracker had to infer would
    be the tracker's guess rescaled onto the template and then used for the
    whole session.
    """
    if spec.lower() == FIT_NONE:
        return {}, {}, [], {}

    measurements = {}
    refusals = {}
    saved = []
    hands = sorted({r["hand_side"] for entry in loaded for r in entry["glove"]})
    if spec.lower() == FIT_AUTO:
        for hand in hands:
            merged = merge_measurements(measure_parts.get(hand, []))
            why = fit_refusal(merged)
            if why:
                refusals[hand] = why
                continue
            measurements[hand] = merged
        for hand, m in sorted(measurements.items()):
            saved.append(m.save(input_dir / FIT_FILE.format(hand=hand)))
    else:
        try:
            loaded_m = load_measurement(spec)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            raise SystemExit(f"--fit-template {spec}: {e}")
        # A measurement names the hand it was taken from. With no hand named
        # it is applied to both, which is the only reading of a file that
        # does not say — and a measurement of the WRONG hand would be a
        # mirrored skeleton's lengths, so the file gets to decide, not us.
        named = [loaded_m.hand] if loaded_m.hand else hands
        why = fit_refusal(loaded_m)
        for hand in named:
            if why:
                refusals[hand] = f"{spec}: {why}"
            else:
                measurements[hand] = loaded_m

    if not measurements:
        return {}, {}, saved, refusals

    scales = {}
    for entry in loaded:
        for row in entry["glove"]:
            m = measurements.get(row["hand_side"])
            if m is None:
                continue
            fitted_frame = fit_template(row["frame"], m)
            row["fit_pts"] = frame_to_keypoints21(fitted_frame)
            if row["hand_side"] not in scales:
                scales[row["hand_side"]] = finger_scales(row["frame"], m)
    return measurements, scales, saved, refusals


def split_excluded(paths, patterns):
    """(kept, dropped) — dropped if any pattern appears in the file name.

    A SUBSTRING and not a glob, because what is being named is a take
    (`pinch_right_take1`) while the file it lives in carries a timestamp
    nobody remembers (`pinch_right_take1_20260920_200113.jsonl`).
    """
    kept, dropped = [], []
    for p in paths:
        if any(pat in p.name for pat in patterns):
            dropped.append(p)
        else:
            kept.append(p)
    return kept, dropped


@dataclass
class FusionRun:
    """One pass of the fusion over every loaded take, and what it produced.

    Two of these exist whenever a reliability profile is given: the same
    frames, the same gates and the same learned rails fused twice, once with
    the profile and once without. Making a pass a VALUE rather than a pile of
    locals inside `main` is what lets the report print both and show what the
    profile actually bought — which is the only honest way to argue for a
    mask that throws evidence away.
    """

    rail_params: object = None
    unreliable: dict = field(default_factory=dict)
    rails: dict = field(default_factory=dict)
    # The thumb vote's learned endpoints, per hand and finger and sensor.
    # Held on the run because the report prints them: a gate on a normalised
    # quantity is only checkable if the normalisation is printed too.
    scale: object = None
    glove_samples: list = field(default_factory=list)
    cam_samples: list = field(default_factory=list)
    fused_samples: list = field(default_factory=list)
    fused_rows: list = field(default_factory=list)
    per_pose: dict = field(default_factory=dict)
    dof_used: dict = field(default_factory=lambda: defaultdict(int))
    dof_total: dict = field(default_factory=lambda: defaultdict(int))
    reasons: dict = field(default_factory=lambda: defaultdict(int))
    palm_residuals: list = field(default_factory=list)
    # The same residual measured against the RAW template hand, collected only
    # when a fit is in force: it is the before of the fit's before/after, and
    # with no fit it would be the same list twice.
    palm_residuals_raw: list = field(default_factory=list)
    # (pose, raw, fitted) per metric-camera frame. Per POSE because the
    # camera's reported hand SIZE is pose-dependent — LeapC shrinks a closed
    # gloved hand it cannot see by about 12 % — so one median over a session
    # of mostly-closed poses hides which way the fit moved the frames where
    # the camera had actually measured the hand.
    palm_pose: list = field(default_factory=list)
    # Per (hand, finger): the glove curl actually fused, lowest and highest.
    # What `curl_gate` and the learned rails have to keep separating after a
    # fit has moved every curl by a few hundredths.
    curl_range: dict = field(default_factory=dict)
    # Per hand: the spread gate's per-finger curl threshold, carried across a
    # template fit by `curl_gates_from_rails`. Empty with no fit, where the
    # one constant is the threshold.
    curl_gates: dict = field(default_factory=dict)
    n_pairs: int = 0
    n_matched: int = 0
    n_cam_used: int = 0


def glove_pts(row, fitted: bool):
    """The glove points to FUSE with: the fitted hand, or the raw template.

    One reader, so "which hand went into the fusion" is decided in one place.
    The report's `glove` column and the glove-only classifier deliberately do
    NOT come through here — they are the raw template, because that is what a
    glove-only pipeline produces.
    """
    if fitted and row.get("fit_pts") is not None:
        return row["fit_pts"]
    return row["pts"]


def fuse_all(loaded, gates, rail_params, unreliable, max_dt, min_score,
             thumb_from_camera=True, export_csv=False, glove_lag=None,
             fitted=False):
    """Learn the rails, fuse every loaded take, collect what the report reads.

    The rails are learned HERE rather than once outside, although two runs
    over one session learn the same numbers: `learn_rails` reads
    `RailOverrideParams`, so a run given different parameters has to be free
    to learn different rails instead of silently inheriting another run's.

    With `fitted` the rails are learned on the FITTED glove's curls, which is
    not a detail: the override compares a frame's curl against the rail, and a
    rail learned on the template while the comparison is made on the fitted
    hand would be a float-equality test between two different hands and would
    never match. Same reason `curl_range` is collected on the fused input —
    `curl_gate` has to go on separating fists from open palms after the fit
    moved both.
    """
    run = FusionRun(rail_params=rail_params, unreliable=dict(unreliable))
    # The curls the fusion is about to compare, both sensors, once. The rails,
    # the spread gate's per-finger thresholds and the thumb vote's endpoints
    # are all read off them, and computing them separately three times would
    # be three chances for one of the three to be taken off a different hand.
    glove_curls = [(g["hand_side"],
                    flexion_features(np.asarray(glove_pts(g, fitted), float)))
                   for take in loaded for g in take["glove"]]
    cam_curls = [(c["hand_side"], flexion_features(np.asarray(c["pts"], float)))
                 for take in loaded for c in take["cam"]]
    # `--no-rail-override` switches off who may TAKE a curl, not whether the
    # gates know where a straight finger reads, so the rails the thumb vote's
    # open endpoint is built from are learned either way.
    rail_for_gates = rail_params or DEFAULT_RAIL
    gate_rails = learn_rails(glove_curls, rail_for_gates)
    if rail_params is not None:
        run.rails = gate_rails
    # Endpoints for the thumb vote's flexion fractions: the glove's rail and
    # 2nd percentile on the hand actually being fused, the camera's open
    # reference and 2nd percentile on the same session's camera frames. This
    # is what makes a fitted and an unfitted run comparable — see fusion.py.
    run.scale = learn_flexion_scale(glove_curls, cam_curls, gate_rails,
                                    gates=gates, rail_params=rail_for_gates)
    # `curl_gate` is a constant tuned on the TEMPLATE hand's curls, and the fit
    # moves every one of them. Learning the rails on both hands — the template
    # and the fitted one — is what lets the gate be carried across by the same
    # factor, so the fit provably changes no gate decision. Learned with
    # `DEFAULT_RAIL` rather than `rail_params` because this is not the
    # override: --no-rail-override switches off who may take a curl, not
    # whether the spread gate knows where a straight finger reads.
    if fitted:
        rail_for_gates = rail_params or DEFAULT_RAIL
        template_rails = learn_rails(
            ((g["hand_side"], flexion_features(np.asarray(g["pts"], float)))
             for take in loaded for g in take["glove"]), rail_for_gates)
        fitted_rails = learn_rails(
            ((g["hand_side"],
              flexion_features(np.asarray(glove_pts(g, True), float)))
             for take in loaded for g in take["glove"]), rail_for_gates)
        run.curl_gates = {
            hand: curl_gates_from_rails(gates, template_rails, fitted_rails,
                                        hand)
            for hand in {h for h, _f in template_rails}}

    for entry in loaded:
        glove, cam = entry["glove"], entry["cam"]
        source, clock = entry["source"], entry["clock"]
        with_scale = entry["with_scale"]
        gpath = Path(entry["name"])
        # A fresh tracker per take. The hysteresis counts CONSECUTIVE frames,
        # and consecutive across a take boundary is a fiction: the takes are
        # separate recordings seconds apart, so a run built at the end of one
        # must not still be armed at the start of the next.
        tracker = (RailOverrideTracker(run.rails, rail_params, gates)
                   if rail_params is not None else None)

        per_hand_g = defaultdict(list)
        per_hand_c = defaultdict(list)
        per_hand_f = defaultdict(list)
        pose = glove[0]["pose"]

        for g, c in pair_by_time(glove, cam, max_dt=max_dt, clock=clock,
                                 glove_lag=glove_lag or 0.0):
            run.n_pairs += 1
            hand = g["hand_side"]
            # RAW is what the glove alone gives and is what the `glove` column
            # and the glove-only classifier report; G is what is FUSED, which
            # is the fitted hand when a fit is in force. They are the same
            # array when it is not.
            raw = np.asarray(g["pts"], dtype=float)
            G = np.asarray(glove_pts(g, fitted), dtype=float)
            per_hand_g[hand].append(all_features(raw, hand_side=hand))
            slot = run.per_pose.setdefault(
                (pose, hand, str(g["take"])),
                {"glove": [], "camera": [], "fused": [], "sources": [],
                 "paired": 0})
            slot["glove"].append(dof_values(raw, hand_side=hand))
            curls_g = flexion_features(G)
            for finger, value in zip(FLEXION_NAMES, curls_g):
                lo, hi = run.curl_range.get((hand, finger), (value, value))
                run.curl_range[(hand, finger)] = (min(lo, value),
                                                  max(hi, value))
            if c is None:
                rail = (tracker.update(hand, curls_g)
                        if tracker is not None else None)
                fused, info = fuse_skeletons(G, None, min_score=min_score,
                                             with_scale=with_scale, gates=gates,
                                             rail=rail)
            else:
                run.n_matched += 1
                slot["paired"] += 1
                C = np.asarray(c["pts"], float)
                per_hand_c[hand].append(all_features(C, hand_side=hand))
                slot["camera"].append(dof_values(C, hand_side=hand))
                meta = cam_meta_of(c, source)
                rail = (tracker.update(hand, curls_g,
                                       flexion_features(C), meta)
                        if tracker is not None else None)
                fused, info = fuse_skeletons(
                    G, c["pts"], cam_score=c.get("score", 1.0),
                    min_score=min_score,
                    thumb_from_camera=thumb_from_camera,
                    with_scale=with_scale,
                    cam_meta=meta, gates=gates, rail=rail,
                    unreliable_fingers=unreliable.get(str(hand).lower(), ()),
                    curl_gates=run.curl_gates.get(hand),
                    scale=run.scale.for_hand(hand))
                for dof in GATED_DOFS:
                    run.dof_total[dof] += 1
                    if info["dof_source"][dof].startswith("camera"):
                        run.dof_used[dof] += 1
                for why in info["rejected"].values():
                    run.reasons[why] += 1
                slot["sources"].append(dict(info["dof_source"]))
                # The fit's before: the same residual against the template
                # hand that went in unfitted. Measured here and not inside
                # `fuse_skeletons`, which only ever sees one of the two.
                if not with_scale and info.get("kabsch_rmse_mm") is not None:
                    before = (palm_fit_rmse_mm(raw, C) if fitted
                              else info["kabsch_rmse_mm"])
                    if fitted:
                        run.palm_residuals_raw.append(before)
                    run.palm_pose.append((pose, before,
                                          info["kabsch_rmse_mm"]))
            if info["camera_used"]:
                run.n_cam_used += 1
            if info.get("kabsch_rmse_mm") is not None:
                run.palm_residuals.append(info["kabsch_rmse_mm"])
            per_hand_f[hand].append(all_features(fused, hand_side=hand))
            slot["fused"].append(dof_values(fused, hand_side=hand))
            if export_csv:
                run.fused_rows.append([pose, g["take"], hand, g["wall_time"],
                                       int(info["camera_used"])]
                                      + [f"{v:.6f}"
                                         for v in np.asarray(fused).ravel()])

        for hand, feats in per_hand_g.items():
            run.glove_samples.append((pose, hand, gpath.name, mean_vector(feats)))
        for hand, feats in per_hand_c.items():
            run.cam_samples.append((pose, hand, gpath.name, mean_vector(feats)))
        for hand, feats in per_hand_f.items():
            run.fused_samples.append((pose, hand, gpath.name, mean_vector(feats)))
    return run


def classifier_lines(run, lines):
    """The three leave-one-TAKE-out rows, for one fusion run."""
    loo_table(run.glove_samples, FLEXION_COLS, "glove only", lines)
    loo_table(run.cam_samples, ALL_COLS, "camera only", lines)
    loo_table(run.fused_samples, ALL_COLS, "fused", lines)


def mask_text(mapping):
    """`{'right': ('ring',)}` -> `right: ring`; empty -> `(none)`."""
    if not mapping:
        return "(none)"
    return "; ".join(f"{hand}: {', '.join(fingers) or '(none)'}"
                     for hand, fingers in sorted(mapping.items()))


def lag_lines(spec, rows, lines):
    """Per hand: the glove's measured lag, where it came from, and if it ran."""
    lines.append("Glove time lag (the glove's SOLVED hand trails the camera; "
                 "the pairing stamp is")
    lines.append("  shifted back by it, so a glove frame stamped t is matched "
                 "against the camera at")
    lines.append("  t - lag. Measured by cross-correlating the curl traces, "
                 "never assumed.)")
    if spec == LAG_NONE:
        lines.append("  --glove-lag none: the two streams are paired on their "
                     "raw stamps")
        return
    how = ("auto — measured from this session" if spec == LAG_AUTO
           else "given on the command line")
    lines.append(f"  {'source':<22} {how}")
    if not rows:
        lines.append("  (no hand had any camera frames to measure against)")
    for hand in sorted(rows):
        lines.append(f"  {hand:<22} {rows[hand].described()}")
    lines.append(f"  A measured lag is applied only when at least "
                 f"{MIN_LAG_CLIPS} clips measure one and their")
    lines.append(f"  estimates agree within {MAX_LAG_MAD * 1000.0:.0f} ms "
                 "(median absolute deviation). One clip is one")
    lines.append("  transition with nothing to check it against, and a "
                 "cross-correlation can find a real")
    lines.append("  peak that is the wrong one. Short of that the estimates "
                 "are printed and NOTHING is")
    lines.append("  applied. A value given on the command line is applied as "
                 "given — that is what it is")
    lines.append("  for, and the report says 'as given'.")
    lines.append("  A HELD pose is lag-insensitive — both sensors describe a "
                 "hand that is not moving —")
    lines.append("  so a session of held poses reports 'not measurable' and "
                 "applies nothing, which is")
    lines.append("  the right answer rather than a missing feature. The "
                 "transition a lag CAN be measured")
    lines.append("  on is the coached recorder's SETTLE clip, written beside "
                 "each take as")
    lines.append(f"  <take>{SETTLE_SUFFIX}.")
    lines.append("")


def fit_lines(spec, measurements, scales, run, gates, lines, refusals=None):
    """What the template fit measured, changed, and left alone."""
    refusals = refusals or {}
    lines.append("Template fit (the glove reports XR Trainer's TEMPLATE hand; "
                 "the camera measures the")
    lines.append("  operator's real bones. Each parent-relative joint offset "
                 "is rescaled to the measured")
    lines.append("  length, every rotation kept, so the joint ANGLES are "
                 "untouched and only the lengths")
    lines.append("  change. ONE fixed calibration per hand per session, "
                 "measured on open-palm frames")
    lines.append("  only and never refitted per pose or per frame.)")
    if spec == FIT_NONE:
        lines.append("  --fit-template none: the glove's template hand is "
                     "fused as recorded")
        lines.append("")
        return
    how = ("auto — measured from this session's own camera frames"
           if spec == FIT_AUTO else f"loaded from {spec}")
    lines.append(f"  {'source':<22} {how}")
    for hand in sorted(refusals):
        lines.append(f"  {hand:<22} NOT FITTED — {refusals[hand]}")
    if refusals:
        lines.append(f"  A hand is fitted only on at least {MIN_FIT_FRAMES} "
                     "OPEN-palm frames over the session (and")
        lines.append("  at least 20 in a take before that take measures "
                     "anything). There is no fallback to")
        lines.append("  the closed frames: a closed gloved hand is "
                     "self-occluded, so its lengths are the")
        lines.append("  tracker's guess at joints it could not see, and "
                     "rescaling the template onto that")
        lines.append("  guess would then apply it to the whole session. A "
                     "refused hand fuses UNFITTED.")
    if not measurements:
        lines.append("  NOT APPLIED to any hand: either no camera frame in "
                     "this session carries 26 metric")
        lines.append("  joints to measure a hand from (a MediaPipe take has "
                     "21 normalised landmarks and no")
        lines.append("  metacarpals), or every hand was refused above.")
        lines.append("")
        return
    for hand in sorted(measurements):
        m = measurements[hand]
        frames = ("open-palm frames" if m.from_open
                  else "frames NOT selected for being open (a file written "
                       "before that was required)")
        lines.append(f"  {hand:<22} {m.n_frames} {frames} from {m.source}, "
                     f"{m.n_segments} segments, typical spread "
                     f"{m.spread_mm:.2f} mm")
        per_finger = scales.get(hand) or {}
        lines.append("    scale per finger     "
                     + "  ".join(
                         f"{f} {v:.3f}" if v is not None else f"{f} -"
                         for f, v in per_finger.items()))
        # The gate that reads a glove curl, against the fitted values it now
        # has to separate, and the threshold it is being separated by.
        # Printed rather than asserted: if a fit ever moves the two ends onto
        # the same side of it, the report says so before anything else does.
        carried = run.curl_gates.get(hand)
        for finger in RAIL_FINGERS:
            got = run.curl_range.get((hand, finger))
            if got is None:
                continue
            lo, hi = got
            gate = (gates.curl_gate if carried is None
                    else carried[FLEXION_NAMES.index(finger)])
            verdict = ("separates" if lo < gate < hi
                       else "DOES NOT SEPARATE — see the template-fit notes")
            lines.append(f"    curl {finger:<15} fitted {lo:.2f} (most "
                         f"curled) .. {hi:.2f} (straightest); "
                         f"curl_gate {gate:.2f} {verdict}")
    if run.palm_pose:
        groups = [("every paired frame", None), ("open_palm takes", OPEN_POSE)]
        for label, pose in groups:
            rows = [(b, a) for p, b, a in run.palm_pose
                    if pose is None or p == pose]
            if not rows:
                continue
            before = sorted(b for b, _a in rows)
            after = sorted(a for _b, a in rows)
            lines.append(f"    palm fit RMSE, {label:<19} median "
                         f"{before[len(before) // 2]:.1f} -> "
                         f"{after[len(after) // 2]:.1f} mm, worst "
                         f"{before[-1]:.1f} -> {after[-1]:.1f} mm")
        lines.append("  The palm residual is a rigid 5-point fit, so it is "
                     "the one number a hand's overall")
        lines.append("  SIZE reaches — and the size the TRACKER reports is "
                     "pose-dependent. Measured on both")
        lines.append("  sessions, every segment of a closed gloved hand comes "
                     "back about 12 % shorter than the")
        lines.append("  same segment of the open hand, uniformly. That is "
                     "POSE-DEPENDENT SCALE VARIATION IN")
        lines.append("  THE TRACKER'S RECONSTRUCTED SKELETON — a closed hand "
                     "is self-occluded and the solver")
        lines.append("  infers the joints it cannot see, and it infers them "
                     "short. The hand did not change")
        lines.append("  size. So the SHAPE the camera reports is measurable "
                     "and the SIZE it reports is not.")
        lines.append("  The fit therefore lowers this residual on the "
                     "open-hand frames (where the camera's own")
        lines.append("  measurement is self-consistent) and raises it on the "
                     "closed ones, and no fixed skeleton")
        lines.append("  can be the right size for both. Nothing downstream "
                     "depends on the choice: the metric")
        lines.append("  path aligns on a basis of UNIT vectors "
                     "(`palm_frame_transfer`) and every reported quantity")
        lines.append("  is a ratio or an angle, so multiplying the fitted "
                     "hand by a constant changes this")
        lines.append("  residual and nothing else in the report.")
    if any(v is not None for v in run.curl_gates.values()):
        lines.append("  curl_gate is one of the two thresholds the fit moves: "
                     "it is a constant tuned on the template's")
        lines.append(f"  curls ({gates.curl_gate:.2f}, the midpoint between "
                     "the glove's fists and its open palms), and curl")
        lines.append("  is a LENGTH ratio. It is carried across as the same "
                     "FRACTION of that hand and finger's")
        lines.append("  own learned rail — its glove open-palm value — rather "
                     "than retuned to a new constant,")
        lines.append("  which would be tuned on one operator's fitted hand "
                     "and wrong for the next by exactly")
        lines.append("  the amount the fit removes. The threshold then moves "
                     "by the same factor the finger's own")
        lines.append("  open value moved by, so the camera-use table comes "
                     "back within half a point on index,")
        lines.append("  middle and pinky and several points BETTER on the "
                     "ring — against the 13-point loss the")
        lines.append("  bare constant cost. The OTHER threshold the fit "
                     "moves is the thumb vote's, and it is")
        lines.append("  re-expressed the same way: the two sensors are "
                     "compared as FLEXION FRACTIONS on their")
        lines.append("  own learned endpoints (agree_tol_frac), which is why "
                     "this run's thumb camera-use can")
        lines.append("  be compared against an unfitted one at all — see the "
                     "flexion-fraction table below.")
        lines.append("  Nothing else needed re-expressing: cam_open_curl and "
                     "the override's margin read the")
        lines.append("  CAMERA, and the override's tol compares the glove "
                     "against its own learned rail.")
    lines.append("  WHAT THE FIT IS NOT EVIDENCE FOR. After fitting, the "
                 "fused pinch index agrees with the")
    lines.append("  camera's to within a few thousandths. That is not an "
                 "independent validation of the")
    lines.append("  camera: the fused finger is BUILT from the camera's bone "
                 "directions, and the lengths it")
    lines.append("  is built on were measured by the same camera, so the "
                 "agreement restates the fit's")
    lines.append("  arithmetic. It says the fit removed the template's scale "
                 "error from that number, and")
    lines.append("  nothing about whether the camera had the finger right. "
                 "The only check on that is a")
    lines.append("  sensor the camera did not produce.")
    lines.append("  The 'glove' column of the table above and the 'glove "
                 "only' classifier row stay on the")
    lines.append("  RAW template hand: that is what the glove ALONE gives, "
                 "and a fitted number there")
    lines.append("  would quote a result the glove cannot produce without a "
                 "camera. The FUSED column is")
    lines.append("  the fitted glove. The rails are learned on the fitted "
                 "curls too, because that is what")
    lines.append("  the override compares against them. "
                 "`leap_hand.pose_check` is untouched: it reads the raw")
    lines.append("  take files at record time, where no camera measurement "
                 "exists yet.")
    lines.append("")


def profile_comparison(plain, masked, profile_path, name, comment, lines,
                       how=""):
    """Both fusions, side by side, so the profile has to earn its place.

    A mask is a claim that some of the evidence is worthless, and a report
    printing only the masked result would be unfalsifiable: the reader could
    not see whether throwing the evidence away helped, hurt or did nothing.
    So both runs are printed, over the same frames, with the two numbers a
    mask is supposed to move — how much of the hand the camera supplied, and
    how well the fused hand classifies.

    `how` says how this file came to be the one applied — a profile that
    applies BY DEFAULT has to be as visible in the report as one somebody
    typed, or the next reader will be comparing masked numbers against
    unmasked ones without knowing it.
    """
    lines.append("=" * 66)
    lines.append(f"Reliability profile — {profile_path}")
    if how:
        lines.append(f"  chosen by: {how}")
    if name:
        lines.append(f"  {name}")
    if comment:
        lines.append(f"  {comment}")
    lines.append(f"  {'unreliable fingers':<22} {mask_text(masked.unreliable)}")
    rail = ("DISABLED (--no-rail-override)" if masked.rail_params is None
            else masked.rail_params.described()["fingers"])
    lines.append(f"  {'rail override fingers':<22} {rail}")
    lines.append("  Same takes, same gates, same learned rails in both "
                 "fusions below; the profile is the")
    lines.append("  only difference. The ordinary run is what this command "
                 "would print without")
    lines.append("  --profile, so the gap between the two tables IS the "
                 "profile's effect.")
    for label, run in (("ordinary (no reliability profile)", plain),
                       ("with reliability profile", masked)):
        lines.append("")
        lines.append(f"  --- {label} ---")
        lines.append("    camera-use rate per gated DOF")
        camera_use_table(run.dof_used, run.dof_total, lines, indent="      ")
        lines.append("    leave-one-TAKE-out nearest centroid")
        sub = []
        classifier_lines(run, sub)
        lines.extend(f"    {line}" for line in sub)
    lines.append("")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fuse simultaneous glove+camera takes and compare all three.")
    p.add_argument("input", type=Path, nargs="?", default=Path("recordings") / "sync",
                   help="folder holding glove/ plus cam/ and/or leap/")
    p.add_argument("--max-dt", type=float, default=0.05,
                   help="max seconds between paired glove/camera frames")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="camera confidence below which the glove is kept as-is")
    p.add_argument("--no-thumb-camera", action="store_true",
                   help="do not take the thumb direction from the camera")
    p.add_argument("--curl-gate", type=float, default=DEFAULT_GATES.curl_gate,
                   help="glove curl above which a finger's spread may come "
                        f"from the camera (default {DEFAULT_GATES.curl_gate})")
    p.add_argument("--view-gate-deg", type=float,
                   default=DEFAULT_GATES.view_gate_deg,
                   help="max angle between palm normal and the ray to the "
                        f"module (default {DEFAULT_GATES.view_gate_deg})")
    p.add_argument("--agree-tol-frac", type=float,
                   default=DEFAULT_GATES.agree_tol_frac,
                   help="max median FLEXION-FRACTION disagreement over "
                        "index..pinky for the camera to own the thumb "
                        f"(default {DEFAULT_GATES.agree_tol_frac}). Each "
                        "curl is first put on its own sensor's, hand's and "
                        "finger's learned endpoints, so the number means the "
                        "same in a fitted and an unfitted run.")
    p.add_argument("--curl-agree-tol", type=float,
                   default=DEFAULT_GATES.curl_agree_tol,
                   help="the RAW curl tolerance the fraction replaced. Only "
                        "reached when no endpoints could be learned "
                        f"(default {DEFAULT_GATES.curl_agree_tol})")
    p.add_argument("--unreliable", action="append", default=[],
                   metavar="HAND:FINGER[,FINGER]",
                   help="fingers whose GLOVE curl is not to be trusted on that "
                        "hand, e.g. 'right:ring,pinky'. They are excluded from "
                        "the thumb gate's vote. Repeatable, once per hand; "
                        "overrides a --profile entry for the same hand.")
    p.add_argument("--no-rail-override", action="store_true",
                   help="do not let the camera take a finger's curl when the "
                        "glove is pinned at full extension and the camera "
                        "sees that finger flexed (default: the override is on)")
    p.add_argument("--rail-fingers", default=None,
                   help="comma-separated fingers the rail override may act on, "
                        f"on BOTH hands (any of {','.join(RAIL_FINGERS)}; "
                        f"default {','.join(DEFAULT_RAIL.fingers_for())}). "
                        "Overrides --profile. Per-hand lists come from a "
                        "profile, not from here.")
    p.add_argument("--profile", default=PROFILE_AUTO,
                   metavar="{auto,none,PATH.json}",
                   help="a reliability profile: per-hand 'unreliable' and "
                        "'rail_fingers' for this glove. The report then prints "
                        "the fusion BOTH ways, so the profile's effect is "
                        f"visible. Default {PROFILE_AUTO}: "
                        f"{' then '.join('profiles/' + n for n in PROFILE_ORDER)}"
                        f", whichever exists first. '{PROFILE_NONE}' fuses "
                        "unmasked. See profiles/.")
    p.add_argument("--glove-lag", default=LAG_AUTO,
                   metavar="{none,auto,left:SEC[,right:SEC]}",
                   help="seconds the glove's solved hand TRAILS the camera; "
                        "the glove's pairing stamp is shifted back by it. "
                        f"Default {LAG_AUTO}: measured per hand from this "
                        "session's SETTLE clips, else from takes whose camera "
                        "curl moved enough, and applied only where a "
                        "trustworthy estimate exists.")
    p.add_argument("--fit-template", default=FIT_AUTO,
                   metavar="{auto,none,PATH.json}",
                   help="rescale the glove's TEMPLATE bone lengths to the "
                        "operator's, as the camera measures them, keeping "
                        f"every rotation. Default {FIT_AUTO}: measured per "
                        "hand from this session's own camera frames and saved "
                        f"as <input>/{FIT_FILE.format(hand='<hand>')}. A path "
                        "loads a saved measurement instead.")
    p.add_argument("--exclude", action="append", default=[],
                   metavar="TEXT[,TEXT]",
                   help="skip takes whose file name contains this text, on "
                        "both the glove and the camera side, e.g. "
                        "'pinch_right_take1'. Repeatable, and comma-separated "
                        "lists are accepted. The excluded takes are named in "
                        "the report header.")
    p.add_argument("--camera", choices=(AUTO,) + CAM_DIRS, default=AUTO,
                   help="which camera folder to read (default: auto — both, "
                        "and a take name in both is an error)")
    p.add_argument("--export-csv", type=Path, default=None,
                   help="write the fused frames as a 21-keypoint CSV")
    p.add_argument("--write", action="store_true", help="write <input>/REPORT.txt")
    args = p.parse_args()

    glove_dir = args.input / "glove"
    wanted = CAM_DIRS if args.camera == AUTO else (args.camera,)
    cam_dirs = [args.input / d for d in wanted if (args.input / d).is_dir()]
    if not glove_dir.is_dir() or not cam_dirs:
        raise SystemExit(
            f"expected {glove_dir} and one of "
            + " or ".join(str(args.input / d) for d in wanted) + "\n"
            "Record them with: python scripts/record_simultaneous.py\n"
            "                  python scripts/record_simultaneous.py --camera leap")

    # `take_files` and not a plain glob: a session also holds each take's
    # SETTLE clip as a `.settle.jsonl` sibling, and a 1.5 s clip of a hand
    # CHANGING shape read as a take would put a transition into a table of
    # medians over held poses. The clips are found separately, by name, for
    # the lag measurement and nothing else.
    all_takes = take_files(glove_dir)
    if not all_takes:
        raise SystemExit(f"no glove recordings in {glove_dir}")

    # One pattern list, applied to the glove side AND to every camera folder,
    # so a take cannot be excluded on one sensor and quietly fused on the
    # other. The camera file is named after the glove take, so one substring
    # catches both — but they are matched and counted separately, because
    # that is the claim the report header makes.
    patterns = [t.strip() for spec in args.exclude
                for t in str(spec).split(",") if t.strip()]
    takes, dropped_glove = split_excluded(all_takes, patterns)
    dropped_cam = []
    for d in cam_dirs:
        dropped_cam += split_excluded(take_files(d), patterns)[1]
    unmatched = [pat for pat in patterns
                 if not any(pat in p.name for p in dropped_glove + dropped_cam)]
    if not takes:
        raise SystemExit(
            f"--exclude removed every take in {glove_dir} "
            f"({len(dropped_glove)} file(s)); nothing is left to fuse.")

    gates = GateParams(curl_gate=args.curl_gate,
                       view_gate_deg=args.view_gate_deg,
                       agree_tol_frac=args.agree_tol_frac,
                       curl_agree_tol=args.curl_agree_tol)

    # --- the two masks, and where each of them came from ----------------
    # Precedence is the same for both: an explicit flag beats the profile and
    # the profile beats the default. The "ordinary" run below is built from
    # the same flags with the PROFILE left out, which is what makes it an
    # honest comparison rather than a second set of defaults.
    cli_unreliable = parse_unreliable(args.unreliable)
    cli_rail = (None if args.rail_fingers is None
                else parse_rail_fingers(args.rail_fingers))
    profile_path, profile_how = resolve_profile(args.profile)
    profile_unreliable, profile_rail, profile_name, profile_comment = (
        load_profile(profile_path) if profile_path is not None
        else ({}, {}, "", ""))

    unreliable = dict(profile_unreliable)
    unreliable.update(cli_unreliable)

    def rail_params_for(rail_spec):
        return (None if args.no_rail_override
                else RailOverrideParams(fingers=rail_spec))

    if cli_rail is not None:
        rail_spec = cli_rail
    elif profile_rail:
        rail_spec = profile_rail
    else:
        rail_spec = DEFAULT_RAIL.fingers
    rail_params = rail_params_for(rail_spec)
    # What this command would do with no --profile at all.
    plain_rail_params = rail_params_for(
        cli_rail if cli_rail is not None else DEFAULT_RAIL.fingers)
    comparing = profile_path is not None
    lag_spec = parse_glove_lag(args.glove_lag)
    fit_spec = str(args.fit_template).strip()

    # --- pass 1: read every take ---------------------------------------
    # The rail override has to know each finger's rail BEFORE it can fuse a
    # frame, and a rail is only visible across a whole session: a take of
    # nothing but fists never shows one. So the reading is separated from the
    # fusing, and the rails are learned in between, from the same frames that
    # are about to be fused.
    loaded = []
    skipped = []
    by_source = defaultdict(int)          # camera -> takes read from it
    by_clock = defaultdict(int)           # pairing clock -> takes paired on it
    measure_parts = defaultdict(list)     # hand -> per-take HandMeasurement
    for gpath in takes:
        try:
            cpath = find_camera_take(args.input, gpath.name, args.camera)
        except AmbiguousTake as e:
            raise SystemExit(f"\n{e}\n")
        if cpath is None:
            skipped.append((gpath.name, "no matching camera file"))
            continue
        glove = load_glove(gpath)
        cam, source = load_cam(cpath)
        if not glove or not cam:
            skipped.append((gpath.name, "one side is empty"))
            continue
        by_source[source] += 1
        # Per take, because one session can hold takes recorded before
        # capture_time existed alongside takes recorded after.
        clock = pairing_clock(glove, cam)
        by_clock[clock] += 1
        # Needs the whole take at once: "did the id change 0.25 s ago" is a
        # question about the frames around this one, not about this one.
        flag_hand_id_stability(cam, gates, clock=clock)
        # Measure the hand from THIS take, then drop the raw 26-joint arrays.
        # Per take because a take is a self-contained measurement of the hand
        # and the median over takes is not at the mercy of the longest one;
        # dropped because keeping 26 positions on every frame of a 59-take
        # session is a hundred megabytes of numbers whose medians are already
        # in hand.
        if fit_spec.lower() == FIT_AUTO:
            for side in sorted({r["hand_side"] for r in cam}):
                got = measure_hand(cam, hand=side, gates=gates,
                                   source=gpath.name)
                if got is not None:
                    measure_parts[side].append(got)
        for r in cam:
            r.pop("abs26", None)
        loaded.append({"name": gpath.name, "glove": glove, "cam": cam,
                       "source": source, "clock": clock,
                       # Metric camera, rigid fit; normalised camera, fit scale.
                       "with_scale": source != LEAP})

    # --- the glove's time lag, per hand ---------------------------------
    lag_rows, glove_lag = glove_lag_of(lag_spec, args.input, cam_dirs, loaded)

    # --- the operator's hand, put on the glove's template ----------------
    measurements, scales, saved_measurements, fit_refusals = fit_measurements(
        fit_spec, measure_parts, loaded, args.input)
    fitted = bool(measurements)

    # --- pass 2: fuse ---------------------------------------------------
    run = fuse_all(loaded, gates, rail_params, unreliable,
                   max_dt=args.max_dt, min_score=args.min_score,
                   thumb_from_camera=not args.no_thumb_camera,
                   export_csv=args.export_csv is not None,
                   glove_lag=glove_lag, fitted=fitted)
    plain = (fuse_all(loaded, gates, plain_rail_params, cli_unreliable,
                      max_dt=args.max_dt, min_score=args.min_score,
                      thumb_from_camera=not args.no_thumb_camera,
                      glove_lag=glove_lag, fitted=fitted)
             if comparing else None)

    if not run.glove_samples:
        raise SystemExit("no usable paired takes found")

    lines = []
    lines.append("=" * 66)
    lines.append(f"Sensor fusion report — {len(takes)} takes from {args.input}")
    lines.append("")
    if patterns:
        lines.append("Excluded by --exclude (a named take is read on NEITHER "
                     "sensor)")
        lines.append(f"  patterns                {', '.join(patterns)}")
        for path in dropped_glove:
            folders = [d.name for d in cam_dirs if (d / path.name).is_file()]
            lines.append(f"    {path.name}  "
                         f"({' + '.join(['glove'] + folders)})")
        for path in dropped_cam:
            if path.name not in {p.name for p in dropped_glove}:
                lines.append(f"    {path.name}  ({path.parent.name} only — "
                             "no glove take of that name)")
        for pat in unmatched:
            lines.append(f"    {pat!r} matched nothing")
        lines.append(f"  takes left              {len(takes)} of "
                     f"{len(all_takes)}")
        lines.append("")
    lines.append("Camera")
    for source, n in sorted(by_source.items()):
        if source == LEAP:
            what = ("Ultraleap Stereo IR 170 — metric 3D joints in metres, "
                    "aligned by palm basis (rotation + wrist, no scale)")
        else:
            what = ("MediaPipe webcam — normalised landmarks, aligned with "
                    "rotation + scale")
        lines.append(f"  {n} take(s)   {what}")
    lines.append("")
    lines.append("Pairing (each glove frame matched to the nearest camera "
                 "frame of the same hand)")
    for clock, n in sorted(by_clock.items()):
        why = ("when each sensor actually had the frame"
               if clock == "capture_time"
               else "when each line was WRITTEN — the take has no "
                    "capture_time, so this is the best available")
        lines.append(f"  clock                   {clock} on {n} take(s): {why}")
    lines.append(f"  glove frames            {run.n_pairs}")
    lines.append(f"  matched within {args.max_dt * 1000:.0f} ms   "
                 f"{run.n_matched} "
                 f"({100.0 * run.n_matched / max(run.n_pairs, 1):.0f}%)")
    lines.append(f"  camera actually used    {run.n_cam_used} "
                 f"({100.0 * run.n_cam_used / max(run.n_pairs, 1):.0f}%)  "
                 f"[score >= {args.min_score}]")
    if skipped:
        lines.append(f"  skipped takes           {len(skipped)}")
        for name, why in skipped:
            lines.append(f"    {name}: {why}")
    if run.palm_residuals:
        ordered = sorted(run.palm_residuals)
        # NOT named `median`: that would rebind the imported function for the
        # whole of main() and make any earlier call to it an UnboundLocalError.
        median_mm = ordered[len(ordered) // 2]
        lines.append("")
        lines.append("Palm agreement (diagnostic — nothing is rejected on it)")
        lines.append(f"  median {median_mm:.1f} mm, worst {ordered[-1]:.1f} mm  "
                     "RMSE of a rigid 5-point palm fit"
                     + (" of the FITTED glove hand" if fitted else ""))
        lines.append("  This is the glove's TEMPLATE hand against the real "
                     "one, not a tracking error. It is")
        lines.append("  reported because it is what Phase 5 would measure; "
                     "the alignment itself does not use it,")
        lines.append("  which is why a large value here cannot bend a finger "
                     "direction.")
    lines.append("")
    lag_lines(lag_spec, lag_rows, lines)
    fit_lines(fit_spec, measurements, scales, run, gates, lines,
              refusals=fit_refusals)
    for path in saved_measurements:
        lines.append(f"  measurement saved       {path}")
    if saved_measurements:
        lines.append("")
    if comparing:
        profile_comparison(plain, run, profile_path, profile_name,
                           profile_comment, lines, how=profile_how)
    else:
        lines.append("Reliability profile — none in force "
                     f"({profile_how})")
        lines.append("  Nothing is masked out of the gates' evidence, and the "
                     "single fusion below is the")
        lines.append("  ordinary one. With a profile the report prints the "
                     "fusion BOTH ways instead.")
        lines.append("")
    lines.append("=" * 66)
    dof_table(run.per_pose, lines)
    gate_tables(run.dof_used, run.dof_total, run.reasons, gates,
                run.rail_params, run.rails, unreliable, lines,
                scale=run.scale)

    lines.append("")
    lines.append("=" * 66)
    lines.append("Secondary metric: leave-one-TAKE-out nearest centroid"
                 + (" (with the reliability profile)" if comparing else ""))
    classifier_lines(run, lines)
    lines.append("")
    n_takes = len({s[2] for s in run.glove_samples})
    per_pose_takes = defaultdict(set)
    for pose, _hand, take, _f in run.glove_samples:
        per_pose_takes[pose].add(take)
    if max((len(v) for v in per_pose_takes.values()), default=0) < 2:
        lines.append("  THIS NUMBER IS NOT MEANINGFUL on this session. Every "
                     "pose has exactly one take,")
        lines.append("  so holding that take out removes the pose's only "
                     "training samples and the true")
        lines.append("  label has no centroid left to be nearest to — every "
                     "held-out sample must be")
        lines.append("  wrong, whatever fusion did. It reads 0% for all three "
                     "rows and will keep")
        lines.append("  reading 0% until a second take of each pose is "
                     "recorded. Read the per-DOF")
        lines.append("  table above instead; it is measured per frame and "
                     "needs no held-out set.")
        lines.append("")
        lines.append(f"  (it was leave-one-SAMPLE-out before, which scored the "
                     f"{n_takes} takes against")
        lines.append("  the other hand of the same five seconds — a number "
                     "that looked meaningful and")
        lines.append("  was not.)")
    else:
        lines.append("  glove only uses the 5 flexion features (all it can "
                     "measure);")
        lines.append("  camera only and fused use flexion + spread.")
        lines.append(f"  feature order: {', '.join(ALL_NAMES)}")

    report = "\n".join(lines)
    print(report)

    if args.export_csv is not None:
        args.export_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pose", "take", "hand", "wall_time", "camera_used"] + COORD_COLS)
            w.writerows(run.fused_rows)
        print(f"\nwrote {args.export_csv}  ({len(run.fused_rows)} fused frames)")

    if args.write:
        out = args.input / "REPORT.txt"
        out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
