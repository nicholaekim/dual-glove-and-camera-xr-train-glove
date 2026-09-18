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

Usage:
  python scripts/fuse_poses.py                          # recordings/sync
  python scripts/fuse_poses.py recordings/sync --write  # also REPORT.txt
  python scripts/fuse_poses.py --export-csv fused.csv   # fused 21-kp CSV
  python scripts/fuse_poses.py --max-dt 0.1             # looser time matching
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cam_hand.export21 import wrist_centered
from cam_hand.features import (
    ALL_COLS,
    ALL_NAMES,
    DOF_ROWS,
    FLEXION_COLS,
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
    RAIL_FINGERS,
    SRC_RAIL,
    GateParams,
    RailOverrideParams,
    RailOverrideTracker,
    flag_hand_id_stability,
    fuse_skeletons,
    learn_rails,
    pair_by_time,
    pairing_clock,
)
from cam_hand.landmarks import MP21_NAMES
from cam_hand.recorder import CamRecorder

from xr_hand.keypoints21 import MP21_TO_OPENXR_IDX, frame_to_keypoints21
from xr_hand.recorder import FrameRecorder

COORD_COLS = [f"{n}_{a}" for n in MP21_NAMES for a in ("x", "y", "z")]

LEAP = "leap"
MEDIAPIPE = "mediapipe"
# Where record_simultaneous puts each backend's camera takes. Both are read.
CAM_DIRS = ("cam", LEAP)


def load_glove(path: Path):
    """Glove JSONL -> dicts with the clocks, hand_side, pose, take, pts (21x3 m).

    `capture_time` — when the OSC packet arrived, if the file has it — is
    carried through, because that is the clock `pair_by_time` prefers.
    Recordings made before it existed simply do not have the key.
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
    "spread thumb-index": ("thumb", "spread index"),
    "spread index-middle": ("spread index", "spread middle"),
    "spread middle-ring": ("spread middle", "spread ring"),
    "spread ring-pinky": ("spread ring", "spread pinky"),
    "thumb-index gap": ("thumb", "spread index"),
}


def row_owner(label, sources, n_paired):
    """The 'from' column: how much of this row the camera actually supplied."""
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
    lines.append("  'spread thumb-index' is the angle of the thumb's BASE "
                 "bone: the camera supplies the")
    lines.append("  thumb's whole direction, not its base angle, so that row "
                 "moves only incidentally —")
    lines.append("  'thumb-index gap' is where the thumb's contribution "
                 "actually shows.")
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


def gate_tables(dof_used, dof_total, reasons, gates, rail_params, rails, lines):
    lines.append("Camera-use rate per gated DOF "
                 "(share of paired frames the camera actually supplied)")
    for dof in GATED_DOFS:
        n = dof_total.get(dof, 0)
        pct = 100.0 * dof_used.get(dof, 0) / n if n else 0.0
        lines.append(f"  {dof:<16} {dof_used.get(dof, 0):>5}/{n:<5} {pct:5.1f}%")
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
    lines.append("")
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
        mark = "  <- enabled" if finger in rail_params.fingers else ""
        lines.append(f"    {hand:<6} {finger:<7} {value:.3f}{mark}")


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
    p.add_argument("--curl-agree-tol", type=float,
                   default=DEFAULT_GATES.curl_agree_tol,
                   help="max median curl disagreement over index..little for "
                        f"the camera to own the thumb "
                        f"(default {DEFAULT_GATES.curl_agree_tol})")
    p.add_argument("--no-rail-override", action="store_true",
                   help="do not let the camera take a finger's curl when the "
                        "glove is pinned at full extension and the camera "
                        "sees that finger flexed (default: the override is on)")
    p.add_argument("--rail-fingers",
                   default=",".join(DEFAULT_RAIL.fingers),
                   help="comma-separated fingers the rail override may act on "
                        f"(any of {','.join(RAIL_FINGERS)}; "
                        f"default {','.join(DEFAULT_RAIL.fingers)})")
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

    takes = sorted(glove_dir.glob("*.jsonl"))
    if not takes:
        raise SystemExit(f"no glove recordings in {glove_dir}")

    gates = GateParams(curl_gate=args.curl_gate,
                       view_gate_deg=args.view_gate_deg,
                       curl_agree_tol=args.curl_agree_tol)

    rail_params = None
    if not args.no_rail_override:
        picked = tuple(f.strip() for f in args.rail_fingers.split(",")
                       if f.strip())
        bad = [f for f in picked if f not in RAIL_FINGERS]
        if bad:
            raise SystemExit(
                f"--rail-fingers: {', '.join(bad)} is not a finger the rail "
                f"override can act on (choose from {', '.join(RAIL_FINGERS)}).\n"
                "  The thumb is not among them: its whole direction is already "
                "the camera's when the thumb gate passes.")
        rail_params = RailOverrideParams(fingers=picked)

    glove_samples, cam_samples, fused_samples = [], [], []
    fused_rows = []
    n_pairs = n_matched = n_cam_used = 0
    skipped = []
    by_source = defaultdict(int)          # camera -> takes read from it
    by_clock = defaultdict(int)           # pairing clock -> takes paired on it
    palm_residuals = []                   # diagnostic only, never a gate
    per_pose = {}                         # (pose, hand) -> the DOF table rows
    dof_used = defaultdict(int)           # gated DOF -> frames the camera won
    dof_total = defaultdict(int)          # gated DOF -> paired frames
    reasons = defaultdict(int)            # rejection reason -> count

    # --- pass 1: read every take ---------------------------------------
    # The rail override has to know each finger's rail BEFORE it can fuse a
    # frame, and a rail is only visible across a whole session: a take of
    # nothing but fists never shows one. So the reading is separated from the
    # fusing, and the rails are learned in between, from the same frames that
    # are about to be fused.
    loaded = []
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
        loaded.append({"name": gpath.name, "glove": glove, "cam": cam,
                       "source": source, "clock": clock,
                       # Metric camera, rigid fit; normalised camera, fit scale.
                       "with_scale": source != LEAP})

    # --- learn the rails ------------------------------------------------
    rails = {}
    if rail_params is not None:
        rails = learn_rails(
            ((g["hand_side"], flexion_features(np.asarray(g["pts"], float)))
             for take in loaded for g in take["glove"]),
            rail_params)

    # --- pass 2: fuse ---------------------------------------------------
    for entry in loaded:
        glove, cam = entry["glove"], entry["cam"]
        source, clock = entry["source"], entry["clock"]
        with_scale = entry["with_scale"]
        gpath = Path(entry["name"])
        # A fresh tracker per take. The hysteresis counts CONSECUTIVE frames,
        # and consecutive across a take boundary is a fiction: the takes are
        # separate recordings seconds apart, so a run built at the end of one
        # must not still be armed at the start of the next.
        tracker = (RailOverrideTracker(rails, rail_params, gates)
                   if rail_params is not None else None)

        per_hand_g = defaultdict(list)
        per_hand_c = defaultdict(list)
        per_hand_f = defaultdict(list)
        pose = glove[0]["pose"]

        for g, c in pair_by_time(glove, cam, max_dt=args.max_dt, clock=clock):
            n_pairs += 1
            hand = g["hand_side"]
            G = np.asarray(g["pts"], dtype=float)
            per_hand_g[hand].append(all_features(G, hand_side=hand))
            slot = per_pose.setdefault(
                (pose, hand, str(g["take"])),
                {"glove": [], "camera": [], "fused": [], "sources": [],
                 "paired": 0})
            slot["glove"].append(dof_values(G, hand_side=hand))
            if c is None:
                rail = (tracker.update(hand, flexion_features(G))
                        if tracker is not None else None)
                fused, info = fuse_skeletons(G, None, min_score=args.min_score,
                                             with_scale=with_scale, gates=gates,
                                             rail=rail)
            else:
                n_matched += 1
                slot["paired"] += 1
                C = np.asarray(c["pts"], float)
                per_hand_c[hand].append(all_features(C, hand_side=hand))
                slot["camera"].append(dof_values(C, hand_side=hand))
                meta = cam_meta_of(c, source)
                rail = (tracker.update(hand, flexion_features(G),
                                       flexion_features(C), meta)
                        if tracker is not None else None)
                fused, info = fuse_skeletons(
                    G, c["pts"], cam_score=c.get("score", 1.0),
                    min_score=args.min_score,
                    thumb_from_camera=not args.no_thumb_camera,
                    with_scale=with_scale,
                    cam_meta=meta, gates=gates, rail=rail)
                for dof in GATED_DOFS:
                    dof_total[dof] += 1
                    if info["dof_source"][dof].startswith("camera"):
                        dof_used[dof] += 1
                for why in info["rejected"].values():
                    reasons[why] += 1
                slot["sources"].append(dict(info["dof_source"]))
            if info["camera_used"]:
                n_cam_used += 1
            if info.get("kabsch_rmse_mm") is not None:
                palm_residuals.append(info["kabsch_rmse_mm"])
            per_hand_f[hand].append(all_features(fused, hand_side=hand))
            slot["fused"].append(dof_values(fused, hand_side=hand))
            if args.export_csv is not None:
                fused_rows.append([pose, g["take"], hand, g["wall_time"],
                                   int(info["camera_used"])]
                                  + [f"{v:.6f}" for v in np.asarray(fused).ravel()])

        for hand, feats in per_hand_g.items():
            glove_samples.append((pose, hand, gpath.name, mean_vector(feats)))
        for hand, feats in per_hand_c.items():
            cam_samples.append((pose, hand, gpath.name, mean_vector(feats)))
        for hand, feats in per_hand_f.items():
            fused_samples.append((pose, hand, gpath.name, mean_vector(feats)))

    if not glove_samples:
        raise SystemExit("no usable paired takes found")

    lines = []
    lines.append("=" * 66)
    lines.append(f"Sensor fusion report — {len(takes)} takes from {args.input}")
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
    lines.append(f"  glove frames            {n_pairs}")
    lines.append(f"  matched within {args.max_dt * 1000:.0f} ms   {n_matched} "
                 f"({100.0 * n_matched / max(n_pairs, 1):.0f}%)")
    lines.append(f"  camera actually used    {n_cam_used} "
                 f"({100.0 * n_cam_used / max(n_pairs, 1):.0f}%)  "
                 f"[score >= {args.min_score}]")
    if skipped:
        lines.append(f"  skipped takes           {len(skipped)}")
        for name, why in skipped:
            lines.append(f"    {name}: {why}")
    if palm_residuals:
        ordered = sorted(palm_residuals)
        # NOT named `median`: that would rebind the imported function for the
        # whole of main() and make any earlier call to it an UnboundLocalError.
        median_mm = ordered[len(ordered) // 2]
        lines.append("")
        lines.append("Palm agreement (diagnostic — nothing is rejected on it)")
        lines.append(f"  median {median_mm:.1f} mm, worst {ordered[-1]:.1f} mm  "
                     "RMSE of a rigid 5-point palm fit")
        lines.append("  This is the glove's TEMPLATE hand against the real "
                     "one, not a tracking error. It is")
        lines.append("  reported because it is what Phase 5 would measure; "
                     "the alignment itself does not use it,")
        lines.append("  which is why a large value here cannot bend a finger "
                     "direction.")
    lines.append("")
    lines.append("=" * 66)
    dof_table(per_pose, lines)
    gate_tables(dof_used, dof_total, reasons, gates, rail_params, rails, lines)

    lines.append("")
    lines.append("=" * 66)
    lines.append("Secondary metric: leave-one-TAKE-out nearest centroid")
    loo_table(glove_samples, FLEXION_COLS, "glove only", lines)
    loo_table(cam_samples, ALL_COLS, "camera only", lines)
    loo_table(fused_samples, ALL_COLS, "fused", lines)
    lines.append("")
    n_takes = len({s[2] for s in glove_samples})
    per_pose_takes = defaultdict(set)
    for pose, _hand, take, _f in glove_samples:
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
            w.writerows(fused_rows)
        print(f"\nwrote {args.export_csv}  ({len(fused_rows)} fused frames)")

    if args.write:
        out = args.input / "REPORT.txt"
        out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
