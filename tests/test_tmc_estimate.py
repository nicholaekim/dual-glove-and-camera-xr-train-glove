"""leap_hand.tmc_estimate and scripts/leap/tmc_estimate_report.py.

A synthetic thumb with a known rest frame and a known abduction-then-flexion
is recovered exactly; the signs are pinned geometrically; the rest frame is
recovered from synthetic transitions whose frames before the movement start
are the open hand; blank starts and short windows are skipped; a shifted
transition shows in the agreement statistics; Paper 3's Table 6 TMC
relation stays unavailable. The synthetic dataset is shaped like the real
one (metadata tables with a BOM, JSONL lines through
`absolute_to_relative`). Expected numbers are worked out here, not read back
from the code. The last tests run on the real dataset and skip without it.
"""
import contextlib
import csv
import importlib.util
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from leap_hand import paper3 as p3
from leap_hand import replay_core as rc
from leap_hand import tmc_estimate as te
from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES
from xr_hand.kinematics import PARENT, absolute_to_relative, mat3_to_quat

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "tmc_estimate_report", REPO / "scripts" / "leap" / "tmc_estimate_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

DATASET = Path(r"C:\Users\nkim2\OneDrive\Desktop\xr trainer"
               r"\grasp set 2026-10-08 organized")
needs_dataset = pytest.mark.skipif(not DATASET.exists(),
                                   reason="real dataset not on this machine")

# The session's measured rest frame, about: xyz angles (-28, -41, -79) deg.
R_REST = jf.rot_x(-28.0) @ jf.rot_y(-41.0) @ jf.rot_z(-79.0)
MIRROR = np.diag([-1.0, 1.0, 1.0])     # a right hand is a left one with x negated
W, P, TM = JOINT_INDEX["WRIST"], JOINT_INDEX["PALM"], JOINT_INDEX["THUMB_METACARPAL"]
SESSION = "20990101_000000_left"


def rest_for(hand):
    return R_REST if hand == "left" else MIRROR @ R_REST @ MIRROR


def cal_with(rest):
    return te.Calibration(R_rest=rest, session="synthetic", n_transitions=1,
                          n_frames=1, per_transition=[], agreement={},
                          provenance="synthetic rest")


def hand_pose(hand, fe=0.0, aa=0.0, tw=0.0, rest=None):
    """World (pos (26, 3) m, rot (26, 3, 3)). The palm is turned against the
    wrist (as on the camera, where WRIST is the forearm); the thumb
    metacarpal is palm @ rest @ Ry(s aa) Rx(-fe) Rz(-s tw); the other bones
    are chained along -z of their parents."""
    rest = rest_for(hand) if rest is None else rest
    s = jf.hand_sign(hand)
    pos, rot = np.zeros((26, 3)), np.zeros((26, 3, 3))
    pos[W] = (0.03, 0.18, -0.05)
    rot[W] = jf.rot_z(25.0) @ jf.rot_x(-30.0)
    rot[P] = rot[W] @ jf.rot_x(-12.0) @ jf.rot_y(5.0)
    pos[P] = pos[W] + rot[W] @ np.array([0.0, 0.0, -0.045])
    for i in range(2, 26):                      # parents come first
        name, par = JOINT_NAMES[i], PARENT[i]
        if i == TM:
            rot[i] = rot[P] @ rest @ jf.rot_y(s * aa) @ jf.rot_x(-fe) @ jf.rot_z(-s * tw)
            off = (-0.03 * -s, 0.0, -0.01)
        elif name.endswith("_METACARPAL"):
            rot[i], off = rot[P], (0.01 * (i - 11) / 5.0, 0.0, -0.01)
        else:
            rot[i] = rot[par] @ jf.rotation_from_angles(10.0, 0.0, 0.0, hand)
            off = (0.0, 0.0, -0.03)
        pos[i] = pos[par] + rot[par] @ np.asarray(off)
    return pos, rot


# --- one pose -----------------------------------------------------------------------------
PAIRS = [(0.0, 0.0), (20.0, 0.0), (0.0, 15.0), (-10.0, 25.0), (35.0, -12.0),
         (-30.0, -20.0), (60.0, 40.0), (110.0, 10.0)]


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("fe,aa", PAIRS)
def test_known_abduction_then_flexion_is_recovered(hand, fe, aa):
    pos, rot = hand_pose(hand, fe, aa)
    e = te.estimate(pos, rot, hand, cal_with(rest_for(hand)))
    assert e[te.FE] == pytest.approx(fe, abs=1e-9)
    assert e[te.AA] == pytest.approx(aa, abs=1e-9)
    assert e[te.TWIST] == pytest.approx(0.0, abs=1e-9)
    assert e[te.RESIDUAL] == pytest.approx(0.0, abs=1e-9)
    assert e["calibration"] == "synthetic rest"


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("tw", [7.0, -12.5])
def test_turn_about_the_bone_is_the_twist(hand, tw):
    pos, rot = hand_pose(hand, 25.0, -8.0, tw)
    e = te.estimate(pos, rot, hand, cal_with(rest_for(hand)))
    assert e[te.FE] == pytest.approx(25.0, abs=1e-9)    # a turn about z leaves
    assert e[te.AA] == pytest.approx(-8.0, abs=1e-9)    # the bone where it is
    assert e[te.TWIST] == pytest.approx(tw, abs=1e-9)
    assert e[te.RESIDUAL] == pytest.approx(abs(tw), abs=1e-9)


def test_estimate_uses_the_palm_not_the_wrist():
    pos, rot = hand_pose("left", 30.0, 5.0)
    rot2 = rot.copy()
    rot2[W] = jf.rot_y(40.0) @ rot[W]           # turn the forearm only
    a = te.estimate(pos, rot, "left", cal_with(R_REST))
    b = te.estimate(pos, rot2, "left", cal_with(R_REST))
    assert a[te.FE] == pytest.approx(b[te.FE], abs=1e-12)
    assert a[te.AA] == pytest.approx(b[te.AA], abs=1e-12)


# --- signs, geometrically ----------------------------------------------------------------
def bone_motion(hand, fe, aa):
    """Unit change of the metacarpal's bone direction (its -z) in the PALM
    frame when the thumb moves from rest to (fe, aa)."""
    _pos, rot0 = hand_pose(hand)
    _pos, rot1 = hand_pose(hand, fe, aa)
    b0 = -(rot0[P].T @ rot0[TM])[:, 2]
    b1 = -(rot1[P].T @ rot1[TM])[:, 2]
    d = b1 - b0
    return d / np.linalg.norm(d)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_positive_abduction_moves_toward_the_rest_frames_thumb_side_x(hand):
    rest = rest_for(hand)
    radial = 1.0 if hand == "left" else -1.0    # the side +x/-x means radial
    for aa in (3.0, -3.0):
        e = te.estimate(*hand_pose(hand, 0.0, aa), hand, cal_with(rest))
        d = bone_motion(hand, 0.0, aa)
        along_rest_x = float(d @ rest[:, 0])
        assert np.sign(e[te.AA]) == np.sign(aa)
        # positive abduction: the bone moves along the rest frame's radial x
        assert np.sign(along_rest_x * radial) == np.sign(e[te.AA])
        assert abs(along_rest_x) > 0.99
        # with this session's rest frame that is palmar (palm -y) and
        # slightly radial (palm +x on a left hand, -x on a right hand)
        assert np.sign(d[1]) == -np.sign(e[te.AA])
        assert np.sign(d[0] * radial) == np.sign(e[te.AA])


@pytest.mark.parametrize("hand", ["left", "right"])
def test_positive_flexion_moves_toward_the_palm(hand):
    rest = rest_for(hand)
    radial = 1.0 if hand == "left" else -1.0
    for fe in (3.0, -3.0):
        e = te.estimate(*hand_pose(hand, fe, 0.0), hand, cal_with(rest))
        d = bone_motion(hand, fe, 0.0)
        assert np.sign(e[te.FE]) == np.sign(fe)
        # toward the metacarpal's own palmar side (-y of the rest frame) ...
        assert float(d @ rest[:, 1]) * np.sign(e[te.FE]) < -0.99
        # ... which here is toward the palm's -y and across the palm (ulnar)
        assert np.sign(d[1]) == -np.sign(e[te.FE])
        assert np.sign(d[0] * radial) == -np.sign(e[te.FE])


# --- rotations ------------------------------------------------------------------------------
def test_chordal_mean_and_rotation_angle():
    r = jf.rot_x(10.0) @ jf.rot_z(-30.0)
    assert te.rotation_angle_deg(r, r @ jf.rot_y(12.0)) == pytest.approx(12.0, abs=1e-9)
    m = te.chordal_mean([r, r, r @ jf.rot_x(15.0)])
    phi = math.degrees(math.atan2(math.sin(math.radians(15.0)),
                                  2.0 + math.cos(math.radians(15.0))))
    assert te.rotation_angle_deg(r @ jf.rot_x(phi), m) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.det(m) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        te.chordal_mean([])


def test_axis_parallelism_of_a_parallel_chain():
    _pos, rot = hand_pose("left", 20.0, 10.0)
    a = te.axis_parallelism(rot)
    assert a["metacarpal_proximal_deg"] == pytest.approx(0.0, abs=1e-6)
    assert a["proximal_distal_deg"] == pytest.approx(0.0, abs=1e-6)
    expect = math.degrees(math.acos(float(np.clip(rot[TM][:, 0] @ rot[P][:, 0], -1, 1))))
    assert a["metacarpal_palm_deg"] == pytest.approx(expect, abs=1e-6)


# --- a synthetic dataset --------------------------------------------------------------------
T0_US = 5_000_000_000
PERIOD_US = 11_111
CAP0 = 1_790_000_000.0
STATIC_FE, STATIC_AA = 20.0, 10.0


def line(hand, hand_id, frame_id, t_us, pose):
    pos, rot = pose
    joints = absolute_to_relative(pos, [mat3_to_quat(r) for r in rot])
    capture = round(CAP0 + (t_us - T0_US) / 1e6, 6)
    return {"wall_time": round(capture + 0.015, 6), "timestamp_us": t_us,
            "hand_side": hand, "frame_id": frame_id, "status": 1,
            "hand_id": hand_id, "framerate": 90.0, "capture_time": capture,
            "source": "leap", "units": "m",
            "joints": [{"name": j.name, "x": j.x, "y": j.y, "z": j.z, "qw": j.qw,
                        "qx": j.qx, "qy": j.qy, "qz": j.qz} for j in joints],
            "abs26": [[round(float(c), 6) for c in p] for p in pos]}


def write_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:   # with a BOM
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def moving(t, move_at):
    """(fe, aa) of the thumb at time t: rest before move_at, then moving."""
    if t < move_at:
        return 0.0, 0.0
    u = min(1.0, (t - move_at) / 0.3)
    return 45.0 * u, -25.0 * u


# name, take, validity, frames, movement start (None = blank), data moves at,
# extra rotation of this transition's rest frame
TRANSITIONS = [
    ("p3_circular_power", 4, "valid", 90, 0.605, 0.605, None),
    ("p3_prismatic_precision", 4, "valid", 80, 0.405, 0.405, None),
    ("p1_lateral", 4, "questionable", 80, 0.505, 0.505, "shift"),
    ("p1_tip", 4, "invalid", 60, None, 0.3, None),
    ("p1_hook", 4, "valid", 60, 0.25, 0.25, None),
]
SHIFT = jf.rot_x(15.0)
STATIC_N, STATIC_MOVE_K = 40, 8
STATIC_HOLD_K, STATIC_REP_K = (12, 39), 20


def window_len(n, start):
    """The open-hand rule worked out here: t < start - 0.1 s."""
    return sum(k * PERIOD_US / 1e6 < start - 0.1 for k in range(n))


def make_dataset(root, shifted=True, sessions=(SESSION,) * 5):
    rows = []
    for i, (name, take, validity, n, start, move_at, extra) in enumerate(TRANSITIONS):
        rest = R_REST @ SHIFT if (extra == "shift" and shifted) else R_REST
        rel = f"transitions/{name}/take_0{take}/open_hand_to_{name}_left_take{take}.jsonl"
        t0 = T0_US + i * 10_000_000
        lines = []
        for k in range(n):
            fe, aa = moving(k * PERIOD_US / 1e6, move_at)
            lines.append(line("left", 30 + i, 5000 + 1000 * i + k, t0 + k * PERIOD_US,
                              hand_pose("left", fe, aa, rest=rest)))
        write_jsonl(root / rel, lines)
        rows.append({"participant_id": "P01", "hand": "left", "from_grasp": "open_hand",
                     "to_grasp": name, "take": take, "recording_file": rel,
                     "cue_time_s": 0.0,
                     "movement_start_time_s": "" if start is None else start,
                     "settled_time_s": "" if start is None else start + 0.4,
                     "validity": validity, "frames": n, "hand_ids": str(30 + i),
                     "first_frame_capture_time": lines[0]["capture_time"],
                     "session": sessions[i], "static_take_file": ""})
    write_table(root / "metadata" / "pose_transitions.csv", rows)

    rel = "grasps/p3_circular_power/take_01/p3_circular_power_left_take1.jsonl"
    lines = [line("left", 7, 1000 + k, T0_US + 90_000_000 + k * PERIOD_US,
                  hand_pose("left", *((STATIC_FE, STATIC_AA) if k >= STATIC_MOVE_K
                                      else (0.0, 0.0))))
             for k in range(STATIC_N)]
    write_jsonl(root / rel, lines)
    write_table(root / "metadata" / "grasps_summary.csv", [{
        "grasp_name": "p3_circular_power", "take": 1, "hand": "left", "hand_ids": "7",
        "raw_jsonl_file": rel, "hand_frame_count": STATIC_N,
        "static_interval_first_frame_id": 1000 + STATIC_HOLD_K[0],
        "static_interval_last_frame_id": 1000 + STATIC_HOLD_K[1],
        "representative_frame_id": 1000 + STATIC_REP_K, "session": SESSION}])
    return root


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("tmc") / "dataset")


def test_open_hand_window(dataset):
    rec = rc.find_transition(dataset, "p3_circular_power", 4)
    assert te.open_hand_window(rec) == list(range(window_len(90, 0.605)))
    assert len(te.open_hand_window(rec)) == 46
    blank = rc.find_transition(dataset, "p1_tip", 4)
    assert te.open_hand_window(blank) == []
    origin = te.timeline_origin_ms(rec)
    assert origin["first_frame_ms"] == pytest.approx(0.0, abs=1e-6)
    assert origin["window_max_ms"] < 0.01          # 1e-6 s capture rounding


def test_calibration_skips_blank_and_short_and_shows_the_shifted_one(dataset):
    cal = te.calibrate(dataset)
    used = [(u["name"], u["take"]) for u in cal.per_transition]
    assert used == [("p3_circular_power", 4), ("p3_prismatic_precision", 4),
                    ("p1_lateral", 4)]
    assert cal.n_transitions == 3
    assert cal.n_frames == (window_len(90, 0.605) + window_len(80, 0.405)
                            + window_len(80, 0.505)) == 46 + 28 + 37
    skipped = {(s["name"], s["take"]): s["reason"] for s in cal.skipped}
    assert set(skipped) == {("p1_tip", 4), ("p1_hook", 4)}
    assert "blank movement_start_time_s" in skipped[("p1_tip", 4)]
    assert f"{window_len(60, 0.25)} frames < 20" in skipped[("p1_hook", 4)]
    assert cal.session == SESSION
    # the chordal mean of {R, R, R Rx(15)} is R Rx(phi), tan(phi) = sin 15 / (2 + cos 15)
    phi = math.degrees(math.atan2(math.sin(math.radians(15.0)),
                                  2.0 + math.cos(math.radians(15.0))))
    assert te.rotation_angle_deg(R_REST @ jf.rot_x(phi), cal.R_rest) < 1e-6
    angles = [u["angle_to_session_mean_deg"] for u in cal.per_transition]
    assert angles == pytest.approx([phi, phi, 15.0 - phi], abs=1e-6)
    assert cal.agreement["worst"] == {"name": "p1_lateral", "take": 4}
    assert cal.agreement["max_deg"] == pytest.approx(15.0 - phi, abs=1e-6)
    assert cal.agreement["median_deg"] == pytest.approx(phi, abs=1e-6)
    for u in cal.per_transition:                  # identical frames in each window
        assert u["spread_max_deg"] < 1e-6
    assert "of 3 transitions (111 frames" in cal.provenance


def test_rest_frame_recovered_from_identical_rest_windows(dataset):
    cal = te.calibrate(dataset, validities=("valid",))
    assert [u["name"] for u in cal.per_transition] == ["p3_circular_power",
                                                       "p3_prismatic_precision"]
    assert te.rotation_angle_deg(R_REST, cal.R_rest) < 1e-6
    assert cal.agreement["max_deg"] < 1e-6
    reasons = {s["name"]: s["reason"] for s in cal.skipped}
    assert "validity 'questionable'" in reasons["p1_lateral"]
    assert "validity 'invalid'" in reasons["p1_tip"]


def test_min_frames_is_honoured(dataset):
    cal = te.calibrate(dataset, min_frames=10, validities=("valid",))
    assert [u["name"] for u in cal.per_transition] == [
        "p3_circular_power", "p3_prismatic_precision", "p1_hook"]


def test_more_than_one_session_needs_a_choice(tmp_path):
    root = make_dataset(tmp_path / "two", sessions=(SESSION, "other", SESSION,
                                                    SESSION, SESSION))
    with pytest.raises(ValueError, match="sessions"):
        te.calibrate(root)
    cal = te.calibrate(root, session=SESSION)
    assert [u["name"] for u in cal.per_transition] == ["p3_circular_power",
                                                       "p1_lateral"]
    assert any("session 'other'" in s["reason"] for s in cal.skipped)


def test_calibration_json_round_trip(dataset):
    cal = te.calibrate(dataset)
    back = te.Calibration.from_json(json.loads(json.dumps(cal.to_json())))
    assert np.array_equal(back.R_rest, cal.R_rest)
    assert back.provenance == cal.provenance
    assert back.agreement == cal.agreement
    assert back.per_transition == cal.per_transition
    pos, rot = hand_pose("left", 12.0, 3.0)
    assert te.estimate(pos, rot, "left", back) == te.estimate(pos, rot, "left", cal)


def test_series_and_metacarpal_motion(dataset):
    cal = te.calibrate(dataset, validities=("valid",))          # R_rest = R_REST
    rec = rc.find_static(dataset, "p3_circular_power", 1)
    s = te.series(rec, cal)
    assert set(s) == set(te.KEYS) and all(len(v) == STATIC_N for v in s.values())
    assert s[te.FE][:STATIC_MOVE_K] == pytest.approx(0.0, abs=1e-6)
    assert s[te.FE][STATIC_MOVE_K:] == pytest.approx(STATIC_FE, abs=1e-6)
    assert s[te.AA][STATIC_MOVE_K:] == pytest.approx(STATIC_AA, abs=1e-6)
    m = te.metacarpal_motion(rec, cal)
    moved = te.rotation_angle_deg(np.eye(3), jf.rot_y(-STATIC_AA) @ jf.rot_x(-STATIC_FE))
    assert m["n_frames"] == STATIC_N
    assert m["max_frame_to_rest_deg"] == pytest.approx(moved, abs=1e-6)
    assert 0.0 < m["mean_to_rest_deg"] < moved
    assert m["spread_median_deg"] < m["spread_max_deg"]
    assert te.metacarpal_motion(rec)["mean_to_rest_deg"] is None


# --- Paper 3 stays unavailable ------------------------------------------------------------
@pytest.mark.parametrize("cls", ["circular", "prismatic"])
def test_paper3_tmc_relation_stays_unavailable(cls):
    poses = [hand_pose("left", fe, 5.0) for fe in (10.0, 20.0, 30.0)]
    angles = p3.series_angles(poses, "left")
    est = {k: [] for k in te.KEYS}
    for pos, rot in poses:
        e = te.estimate(pos, rot, "left", cal_with(R_REST))
        for k in te.KEYS:
            est[k].append(e[k])
    angles.update({k: np.asarray(v) for k, v in est.items()})
    out = p3.compare(angles, cls)
    row = next(r for r in out["relations"] if r["id"] == "T6:T_TMC_fe:T_MCP_fe")
    assert p3.RELATION_BY_ID["T6:T_TMC_fe:T_MCP_fe"].lhs == "T_TMC_fe"
    assert row["status"] == "unavailable"
    assert row["stats_all"] is None and row["stats_hold"] is None
    assert np.isnan(angles["T_TMC_fe"]).all()       # the validated slot stays empty


# --- the report script ------------------------------------------------------------------
def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = report.main([str(a) for a in argv])
    return code, buf.getvalue()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_report_on_the_synthetic_dataset(tmp_path):
    root = make_dataset(tmp_path / "dataset", shifted=False)
    out = tmp_path / "out"
    code, text = run(["--dataset", root, "--out", out])
    assert code == 0, text
    assert text.strip().splitlines()[-1] == (
        "TMC ESTIMATE REPORT OK: transitions_used=3 recordings=6")
    for name in ("calibration.json", "assumption_checks.json",
                 "estimates_per_recording.csv", "estimates_per_frame.csv",
                 "estimates_by_class.csv"):
        assert (out / name).is_file(), name
    cal = te.Calibration.from_json(json.loads((out / "calibration.json").read_text()))
    assert cal.n_transitions == 3
    assert te.rotation_angle_deg(R_REST, cal.R_rest) < 1e-6

    per_rec = read_csv(out / "estimates_per_recording.csv")
    assert len(per_rec) == 6
    assert [r["kind"] for r in per_rec] == ["static"] + ["transition"] * 5
    static = per_rec[0]
    assert static["class"] == "circular" and int(static["n_frames"]) == STATIC_N
    assert int(static["rep_frame_id"]) == 1000 + STATIC_REP_K
    assert float(static[f"{te.FE}_rep"]) == pytest.approx(STATIC_FE, abs=1e-3)
    assert float(static[f"{te.AA}_rep"]) == pytest.approx(STATIC_AA, abs=1e-3)
    assert int(static["hold_frames"]) == STATIC_HOLD_K[1] - STATIC_HOLD_K[0] + 1
    for stat in ("mean", "min", "max"):
        assert float(static[f"{te.FE}_hold_{stat}"]) == pytest.approx(STATIC_FE, abs=1e-3)
        assert float(static[f"{te.AA}_hold_{stat}"]) == pytest.approx(STATIC_AA, abs=1e-3)
    assert float(static[f"{te.FE}_all_min"]) == pytest.approx(0.0, abs=1e-3)
    assert static["calibration"] == cal.provenance
    for r in per_rec[1:]:
        assert r["rep_frame_id"] == "" and r[f"{te.FE}_rep"] == ""
        assert r["hold_frames"] == "" and r[f"{te.FE}_hold_mean"] == ""
        assert r[f"{te.FE}_all_max"] != ""

    per_frame = read_csv(out / "estimates_per_frame.csv")
    assert len(per_frame) == STATIC_N + sum(t[3] for t in TRANSITIONS)
    assert list(per_frame[0]) == ["kind", "name", "take", "frame_index", "frame_id",
                                  "t_s", te.FE, te.AA, te.TWIST, te.RESIDUAL]

    checks = json.loads((out / "assumption_checks.json").read_text())
    assert set(checks) >= {"A1_parallel_hinge_axes", "A2_rest_stable",
                           "A3_open_hand_is_zero", "A4_metacarpal_tracked",
                           "A5_palm_is_carpus", "timeline_origin"}
    a1 = checks["A1_parallel_hinge_axes"]
    assert a1["frames"] == len(per_frame)
    assert a1["metacarpal_proximal_deg"]["max"] < 1e-3
    assert len(checks["A2_rest_stable"]["per_transition"]) == 3
    assert len(checks["A4_metacarpal_tracked"]["per_recording"]) == 6
    assert checks["timeline_origin"]["first_frame_ms_max_abs"] < 1e-3
    by_class = read_csv(out / "estimates_by_class.csv")
    assert [r["class"] for r in by_class] == ["circular"]
    assert float(by_class[0][f"{te.FE}_hold_mean_median"]) == pytest.approx(STATIC_FE, abs=1e-3)


def test_report_refuses_to_write_inside_the_dataset(tmp_path):
    root = make_dataset(tmp_path / "dataset", shifted=False)
    code, text = run(["--dataset", root, "--out", root / "out"])
    assert code == 2 and "refusing" in text
    assert not (root / "out").exists()


# --- the real dataset -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_cal():
    return te.calibrate(DATASET)


@needs_dataset
def test_real_calibration(real_cal):
    assert real_cal.n_transitions == 42
    assert real_cal.session == "20261001_101621_left"
    assert real_cal.agreement["median_deg"] < 8.0
    assert real_cal.agreement["max_deg"] < 25.0
    # the docstring's sign sentences, for this session's rest frame (left hand):
    # +aa moves the bone along +x_rest (palmar, slightly radial), +fe along -y_rest
    up_aa, up_fe = real_cal.R_rest[:, 0], -real_cal.R_rest[:, 1]
    assert up_aa[1] < -0.5 and up_aa[0] > 0.0
    assert up_fe[1] < 0.0 and up_fe[0] < -0.5


@needs_dataset
def test_real_thumb_hinge_axes_are_parallel():
    rec = rc.find_static(DATASET, "p3_circular_power", 1)
    a = [te.axis_parallelism(rc.reconstruct(f).rot) for f in rec.frames]
    assert np.median([x["metacarpal_proximal_deg"] for x in a]) < 5.0
    assert np.median([x["proximal_distal_deg"] for x in a]) < 5.0
