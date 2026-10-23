"""Replay core: the viewer must show what the tracker recorded, on its clock.

Every expectation here is derived without the module under test: rotations
from a Rodrigues formula and an axis-angle quaternion written in this file,
synthetic hands built bone by bone in the LeapC convention (a bone's local -z
runs from its joint to the next one, +y is dorsal), and clocks, gaps and
identity changes written by hand into small JSONL fixtures. The tests at the
bottom reproduce facts a separate scan measured in the real dataset and skip
on a machine that does not have it.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from leap_hand import replay_core as rc
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES, HandFrame
from xr_hand.kinematics import (
    absolute_to_relative,
    forward_kinematics_full,
    mat3_to_quat,
    quat_mul,
    quat_to_mat3,
)

WRIST = JOINT_INDEX["WRIST"]
PALM = JOINT_INDEX["PALM"]
FINGERS = ("INDEX", "MIDDLE", "RING", "LITTLE")
DATASET = Path(r"C:\Users\nkim2\OneDrive\Desktop\xr trainer"
               r"\grasp set 2026-10-08 organized")
needs_dataset = pytest.mark.skipif(not DATASET.exists(),
                                   reason="real dataset not on this machine")


# --- independent references ---------------------------------------------------
def rodrigues(axis, deg):
    """Axis-angle -> rotation matrix: R = I + sin(t) K + (1 - cos(t)) K^2."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    t = math.radians(deg)
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * (K @ K)


def quat_axis_angle(axis, deg):
    """Axis-angle -> XYZW quaternion: (k sin(t/2), cos(t/2))."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    h = math.radians(deg) / 2.0
    s = math.sin(h)
    return (k[0] * s, k[1] * s, k[2] * s, math.cos(h))


def angle_between_rotations_acos(ra, rb):
    """Rotation angle of ra^T rb from the trace (fine away from 0 and 180)."""
    c = (np.trace(np.asarray(ra).T @ np.asarray(rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# --- a synthetic hand in the LeapC convention -----------------------------------
def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def leap_basis(bone_dir, dorsal):
    """Columns x, y, z of a LeapC bone frame: -z along the bone, +y dorsal."""
    z = -unit(bone_dir)
    y = unit(dorsal - np.dot(dorsal, z) * z)
    return np.column_stack([np.cross(y, z), y, z])


def right_hand_local():
    """26 positions (m) and 26 rotations of a right hand in its own frame.

    Palm down (dorsal +y), fingers along -z, thumb toward -x, every finger a
    little flexed so no two bones are parallel. TIP rows reuse the distal
    bone's rotation and WRIST is a forearm slightly off the hand axis, as in
    `leap_hand.to_openxr`.
    """
    up = np.array([0.0, 1.0, 0.0])
    p = {"WRIST": np.zeros(3), "PALM": np.array([-0.004, -0.006, -0.048])}
    for k, f in enumerate(FINGERS):
        x0 = -0.024 + 0.017 * k
        base = np.array([0.4 * x0, -0.003, -0.012])
        mcp = np.array([x0, -0.002, -0.080 + 0.004 * k])
        p[f + "_METACARPAL"], p[f + "_PROXIMAL"] = base, mcp
        d = unit(mcp - base)
        cur = mcp
        for seg, length, flex in (("_INTERMEDIATE", 0.042 - 0.003 * k, 18 + 4 * k),
                                  ("_DISTAL", 0.026, 40), ("_TIP", 0.020, 62)):
            cur = cur + length * (rodrigues((1.0, 0.0, 0.0), -flex) @ d)
            p[f + seg] = cur
    d = unit((-0.65, -0.30, -0.70))
    bend_axis = np.cross(d, up)
    cur = np.array([-0.020, -0.012, -0.010])
    p["THUMB_METACARPAL"] = cur
    for name, length, bend in (("THUMB_PROXIMAL", 0.046, 0.0),
                               ("THUMB_DISTAL", 0.032, 15.0),
                               ("THUMB_TIP", 0.026, 35.0)):
        cur = cur + length * (rodrigues(bend_axis, bend) @ d)
        p[name] = cur

    r = {"WRIST": leap_basis((0.05, 0.02, -1.0), up),
         "PALM": leap_basis(p["MIDDLE_PROXIMAL"] - p["WRIST"], up)}
    for f in ("THUMB",) + FINGERS:
        chain = [n for n in JOINT_NAMES if n.startswith(f + "_")]
        for a, b in zip(chain[:-1], chain[1:]):
            r[a] = leap_basis(p[b] - p[a], up)
        r[chain[-1]] = r[chain[-2]]
    return (np.array([p[n] for n in JOINT_NAMES]),
            np.array([r[n] for n in JOINT_NAMES]))


RIGHT_LOCAL = right_hand_local()
G = rodrigues((0.3, 1.0, 0.2), 35.0)
T = np.array([0.04, 0.22, 0.03])
MIRROR = np.diag([-1.0, 1.0, 1.0])


def place(pos, rot, R=G, t=T):
    """The local hand moved rigidly into camera space."""
    return pos @ R.T + t, R @ rot


def mirror(pos, rot):
    """x -> -x: positions reflected, rotations conjugated (still proper)."""
    return pos @ MIRROR, MIRROR @ rot @ MIRROR


def joint_dicts(pos, rot):
    joints = absolute_to_relative(pos, [mat3_to_quat(r) for r in rot])
    return joints, [{"name": j.name, "x": j.x, "y": j.y, "z": j.z,
                     "qw": j.qw, "qx": j.qx, "qy": j.qy, "qz": j.qz}
                    for j in joints]


def hand_line(pos, rot, *, side="right", frame_id=1000, hand_id=7,
              ts_us=5_000_000_000, capture=1_790_000_000.0, wall=None,
              status=1, abs26="rounded", reverse_joints=False):
    """One JSONL line shaped like `leap_hand.recorder` writes it."""
    joints, jl = joint_dicts(pos, rot)
    if reverse_joints:
        jl = jl[::-1]                  # the reader must match joints by name
    line = {"wall_time": capture + 0.02 if wall is None else wall,
            "timestamp": ts_us / 1e6, "timestamp_us": ts_us,
            "packet_counter": frame_id, "hand_side": side,
            "frame_id": frame_id, "status": status, "hand_id": hand_id,
            "framerate": 90.0, "capture_time": capture, "joints": jl,
            "source": "leap", "space": "leap_desktop", "units": "m"}
    if abs26 == "rounded":             # what the recorder writes
        line["abs26"] = [[round(float(c), 6) for c in p] for p in pos]
    elif abs26 == "exact":
        line["abs26"] = [[float(c) for c in p] for p in pos]
    elif abs26 == "chain":             # the chain's own output, unrounded
        frame = HandFrame(timestamp=0.0, packet_counter=0, hand_side=side,
                          frame_id=0, status=1, joints=joints)
        line["abs26"] = [list(p) for p in forward_kinematics_full(frame)[0]]
    return line


def write_jsonl(path, lines):
    path.write_text("".join(json.dumps(x) + "\n" for x in lines),
                    encoding="utf-8")
    return path


BASE_LINE = hand_line(*place(*RIGHT_LOCAL))
TS0 = 5_000_000_000
PERIOD_US = 11_111


def clock_lines(n, *, frame_ids=None, hand_ids=None, extra_us=None):
    """n frames of one right hand at 90 Hz; the hand slides 1 mm per frame.

    extra_us[i] adds that many microseconds to every clock from frame i on,
    which is how a gap is written. capture_time follows the camera clock with
    a constant offset; wall_time trails it with writer jitter.
    """
    out, shift = [], 0
    for i in range(n):
        shift += (extra_us or {}).get(i, 0)
        ts = TS0 + i * PERIOD_US + shift
        cap = round(1_790_000_000.0 + (ts - TS0) / 1e6 + 0.0003, 6)
        line = json.loads(json.dumps(BASE_LINE))
        dx = 0.001 * i
        for j in line["joints"]:
            if j["name"] == "WRIST":   # the root: moving it moves the hand
                j["x"] += dx
        line["abs26"] = [[round(p[0] + dx, 6), p[1], p[2]]
                         for p in line["abs26"]]
        line.update(timestamp_us=ts, timestamp=ts / 1e6, capture_time=cap,
                    wall_time=cap + 0.02 + 0.001 * (i % 3),
                    frame_id=(frame_ids[i] if frame_ids else 1000 + i),
                    hand_id=(hand_ids[i] if hand_ids else 7))
        line["packet_counter"] = line["frame_id"]
        out.append(line)
    return out


def load(tmp_path, lines, side="right", ids=(7,), name="rec.jsonl", **kw):
    return rc.load_recording(write_jsonl(tmp_path / name, lines), side,
                             list(ids), **kw)


# --- the contract other leaves are written against ------------------------------
def test_contract_names_exist():
    for name in ("SourceFrame", "ClockReport", "Timeline", "Event", "Segment",
                 "Recording", "Reconstruction", "DisplayedPose", "read_jsonl",
                 "load_recording", "find_static", "find_transition",
                 "list_static", "list_transitions", "evaluate_clock",
                 "build_timeline", "find_segments", "reconstruct",
                 "reconstruct_quaternion_chain", "angular_distance_deg",
                 "rotation_angle_deg", "slerp", "frame_fidelity",
                 "recording_fidelity", "pose_at", "to_reference",
                 "PlaybackClock", "SchedulingLog", "TOLERANCES",
                 "CLOCK_PREFERENCE", "hold_mask"):
        assert hasattr(rc, name), name
    assert rc.CLOCK_PREFERENCE == ("timestamp_us", "capture_time", "wall_time")
    for key, value in (("position_m", 1e-6), ("chain_deg", 1e-6),
                       ("chain_m", 1e-9), ("bone_direction_deg_expectation", 0.05),
                       ("clock_interval_ms", 0.5), ("quat_norm", 1e-9)):
        assert rc.TOLERANCES[key]["value"] == value
        assert len(rc.TOLERANCES[key]["why"]) > 20


# --- rigid transforms against the in-test references ---------------------------
def test_rodrigues_reference_follows_the_right_hand_rule():
    np.testing.assert_allclose(rodrigues((0, 0, 1), 90) @ (1, 0, 0), (0, 1, 0), atol=1e-15)
    np.testing.assert_allclose(rodrigues((1, 0, 0), 90) @ (0, 1, 0), (0, 0, 1), atol=1e-15)
    np.testing.assert_allclose(rodrigues((0, 1, 0), 90) @ (0, 0, 1), (1, 0, 0), atol=1e-15)


@pytest.mark.parametrize("axis,deg", [((1, 0, 0), 90.0), ((0, 1, 0), 90.0),
                                      ((0, 0, 1), 90.0), ((1, -2, 0.5), 37.0)])
def test_quaternion_matrix_and_sandwich_match_rodrigues(axis, deg):
    q = quat_axis_angle(axis, deg)
    R = rodrigues(axis, deg)
    np.testing.assert_allclose(quat_to_mat3(*q), R, atol=1e-15)
    v = np.array([0.3, -1.2, 2.5])
    np.testing.assert_allclose(rc.quat_rotate(q, v), R @ v, atol=1e-14)


def test_composition_order_matches_matrix_product():
    qa, qb = quat_axis_angle((1, -2, 0.5), 37.0), quat_axis_angle((0.2, 0.1, 1), 120.0)
    Ra, Rb = rodrigues((1, -2, 0.5), 37.0), rodrigues((0.2, 0.1, 1), 120.0)
    np.testing.assert_allclose(quat_to_mat3(*quat_mul(qa, qb)), Ra @ Rb, atol=1e-15)
    v = np.array([1.0, 2.0, -0.5])
    np.testing.assert_allclose(rc.quat_rotate(quat_mul(qa, qb), v), Ra @ (Rb @ v), atol=1e-14)
    # and the other order is a different rotation, so the test can fail
    assert np.abs(quat_to_mat3(*quat_mul(qb, qa)) - Ra @ Rb).max() > 0.1


def test_angle_measures_agree_with_the_trace_formula_and_resolve_tiny_angles():
    a, b = (0.2, 0.1, 1.0), (1.0, -2.0, 0.5)
    qa, qb = quat_axis_angle(a, 50.0), quat_axis_angle(b, 120.0)
    Ra, Rb = rodrigues(a, 50.0), rodrigues(b, 120.0)
    expected = angle_between_rotations_acos(Ra, Rb)
    assert rc.angular_distance_deg(qa, qb) == pytest.approx(expected, abs=1e-9)
    assert rc.rotation_angle_deg(Ra, Rb) == pytest.approx(expected, abs=1e-9)
    assert rc.angular_distance_deg(qa, [-c for c in qa]) == pytest.approx(0.0, abs=1e-12)
    same_axis = rc.angular_distance_deg(quat_axis_angle(a, 10.0), quat_axis_angle(a, 75.0))
    assert same_axis == pytest.approx(65.0, abs=1e-9)
    # the 1e-6 degree gate needs angles far below what acos can resolve
    tiny = rc.angular_distance_deg((0, 0, 0, 1), quat_axis_angle(b, 1e-7))
    assert tiny == pytest.approx(1e-7, rel=1e-6)
    assert rc.rotation_angle_deg(np.eye(3), rodrigues(b, 1e-7)) == pytest.approx(1e-7, rel=1e-6)


def test_slerp_takes_the_short_way_at_constant_rate():
    q0, q1 = (0.0, 0.0, 0.0, 1.0), quat_axis_angle((0, 0, 1), 90.0)
    mid = rc.slerp(q0, q1, 0.5)
    assert rc.angular_distance_deg(mid, q0) == pytest.approx(45.0, abs=1e-12)
    assert rc.angular_distance_deg(mid, q1) == pytest.approx(45.0, abs=1e-12)
    np.testing.assert_allclose(quat_to_mat3(*mid), rodrigues((0, 0, 1), 45.0), atol=1e-15)
    np.testing.assert_allclose(rc.slerp(q0, [-c for c in q1], 0.5), mid, atol=1e-15)
    np.testing.assert_allclose(rc.slerp(q0, q1, 0.0), q0, atol=1e-15)
    assert rc.angular_distance_deg(rc.slerp(q0, q1, 1.0), q1) == pytest.approx(0.0, abs=1e-6)


# --- left and right synthetic hands ---------------------------------------------
@pytest.mark.parametrize("side", ["right", "left"])
def test_synthetic_hand_comes_back_for_both_sides(tmp_path, side):
    pos, rot = place(*RIGHT_LOCAL)
    if side == "left":
        pos, rot = mirror(pos, rot)
    # fixture self-check: proper rotations whose -z runs along each bone
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)
    i = JOINT_INDEX["INDEX_PROXIMAL"]
    np.testing.assert_allclose(-rot[i][:, 2], unit(pos[i + 1] - pos[i]), atol=1e-12)

    line = hand_line(pos, rot, side=side, abs26="exact",
                     reverse_joints=(side == "left"))
    rec = load(tmp_path, [line], side=side)
    assert rec.hand_side == side and rec.frames[0].hand_side == side
    r = rc.reconstruct(rec.frames[0])
    assert np.abs(r.pos - pos).max() < 1e-9
    assert rc.rotation_angle_deg(r.rot, rot).max() < 1e-9
    assert (r.quat[:, 3] >= 0).all()
    ff = rc.frame_fidelity(rec.frames[0])
    bones = ff["bone_direction_deg"]["per_bone"]
    assert len(bones) == 19 and "THUMB_METACARPAL" in bones and "LITTLE_DISTAL" in bones
    assert ff["bone_direction_deg"]["max"] < 1e-6
    assert ff["position_residual_m"]["max"] < 1e-12
    expected_mm = 1000.0 * np.linalg.norm(pos[i + 1] - pos[i])
    assert ff["bone_length_mm"]["INDEX_PROXIMAL"] == pytest.approx(expected_mm, abs=1e-9)


def test_bone_direction_check_notices_unmirrored_rotations(tmp_path):
    """Positive control: mirrored positions with the right hand's rotations."""
    pos, rot = place(*RIGHT_LOCAL)
    pos_l, _ = mirror(pos, rot)
    rec = load(tmp_path, [hand_line(pos_l, rot, side="left", abs26="exact")],
               side="left")
    assert rc.frame_fidelity(rec.frames[0])["bone_direction_deg"]["max"] > 10.0


# --- exact replay and the fidelity numbers ---------------------------------------
def test_exact_source_times_show_the_recorded_frame(tmp_path):
    rec = load(tmp_path, clock_lines(5))
    for i, f in enumerate(rec.frames):
        r = rc.reconstruct(f)
        for mode in ("hold", "interpolate"):
            shown = rc.pose_at(rec, rec.timeline.t[i], mode)
            assert shown.source == "recorded" and shown.i0 == i and shown.frame is f
            np.testing.assert_array_equal(shown.pos, r.pos)
            np.testing.assert_array_equal(shown.rot, r.rot)
            np.testing.assert_array_equal(shown.quat, r.quat)
    before = rc.pose_at(rec, -0.001, "interpolate")
    assert before.source == "unavailable" and before.pos is None
    after = rc.pose_at(rec, rec.timeline.t[-1] + 1.0, "interpolate")
    assert after.source == "held" and after.i0 == 4 and "last" in after.note


def test_position_residual_rounded_exact_and_absent(tmp_path):
    pos, rot = place(*RIGHT_LOCAL)
    lines = [hand_line(pos, rot, abs26="rounded", frame_id=1),
             hand_line(pos, rot, abs26="chain", frame_id=2, ts_us=TS0 + PERIOD_US),
             hand_line(pos, rot, abs26=None, frame_id=3, ts_us=TS0 + 2 * PERIOD_US)]
    lines[1]["capture_time"] += 0.011111
    lines[2]["capture_time"] += 0.022222
    rec = load(tmp_path, lines)
    rounded, chain, absent = (rc.frame_fidelity(f) for f in rec.frames)
    assert 0.0 < rounded["position_residual_m"]["max"] <= 1e-6
    # written as the chain's own unrounded output: the reader changed nothing
    assert chain["position_residual_m"]["max"] == 0.0
    assert set(chain["position_residual_m"]["per_joint"]) == {0.0}
    assert absent["position_residual_m"]["max"] is None
    assert absent["bone_direction_deg"]["max"] is None
    for ff in (rounded, chain, absent):
        assert ff["chain_deg"]["max"] <= 1e-9 and ff["chain_m"]["max"] <= 1e-12
        assert abs(ff["quat_norm"]["max"] - 1.0) < 1e-12
        assert abs(ff["rotation_det"]["min"] - 1.0) < 1e-12
        assert ff["rotation_orthogonality"]["max"] < 1e-12
        json.dumps(ff, allow_nan=False)
    assert rounded["bone_direction_deg"]["max"] < 0.05


def test_recording_fidelity_is_json_and_can_fail(tmp_path):
    good = rc.recording_fidelity(load(tmp_path, clock_lines(6), name="good.jsonl"))
    json.dumps(good, allow_nan=False)
    assert good["within_tolerance"] is True
    assert good["counts"]["frames_selected"] == 6
    assert good["frame_id"] == {"first": 1000, "last": 1005, "missing_in_range": 0}
    assert good["timeline"]["source"] == "timestamp_us"
    assert good["timeline"]["duration_s"] == pytest.approx(5 * PERIOD_US / 1e6, abs=1e-12)
    cc = good["timeline"]["cross_check"]                # the fixture's clocks, by hand:
    assert cc["capture_vs_timestamp_interval_max_ms"] < 0.001   # capture = camera + const
    assert cc["within_clock_interval_tolerance"] is True
    assert cc["writer_delay_max_ms"] == pytest.approx(22.0, abs=1e-3)    # 20 ms + 0..2
    assert cc["writer_delay_median_ms"] == pytest.approx(21.0, abs=1e-3)
    assert cc["camera_span_s"] == pytest.approx(5 * PERIOD_US / 1e6, abs=1e-12)
    assert cc["writer_span_s"] == pytest.approx(5 * PERIOD_US / 1e6 + 0.002, abs=1e-6)
    assert good["tolerances"]["position_m"]["value"] == 1e-6
    assert set(good["within_tolerance_detail"]) == {"position_m", "chain_deg", "chain_m"}

    lines = clock_lines(6)
    lines[3]["abs26"][JOINT_INDEX["INDEX_TIP"]][1] += 2e-6     # 2 um off
    bad = rc.recording_fidelity(load(tmp_path, lines, name="bad.jsonl"))
    assert bad["within_tolerance"] is False
    assert bad["within_tolerance_detail"]["position_m"]["ok"] is False
    assert bad["position_residual_m"]["worst_frame_id"] == 1003
    assert bad["position_residual_m"]["worst_joint"] == "INDEX_TIP"


# --- timestamp-based playback and speed -------------------------------------------
def test_playback_clock_speed_pause_and_seek():
    clk = rc.PlaybackClock(speed=2.0)
    clk.start(10.0, media_t=0.0)
    assert clk.playing
    assert clk.media_time(10.5) == pytest.approx(1.0)       # 0.5 s wall at 2x
    clk.set_speed(0.5, 10.5)
    assert clk.media_time(10.5) == pytest.approx(1.0)       # no jump at the change
    assert clk.media_time(11.5) == pytest.approx(1.5)
    clk.pause(11.5)
    assert not clk.playing
    assert clk.media_time(99.0) == pytest.approx(1.5)       # frozen
    clk.seek(4.0, 99.0)
    assert clk.media_time(120.0) == pytest.approx(4.0)      # a paused seek stays put
    clk.start(200.0)
    assert clk.media_time(201.0) == pytest.approx(4.5)      # resumes from there at 0.5x
    clk.seek(1.0, 201.0)
    assert clk.media_time(203.0) == pytest.approx(2.0)      # a playing seek keeps playing
    with pytest.raises(ValueError):
        clk.set_speed(0.0, 203.0)
    fake = iter([0.0, 3.0])
    clk2 = rc.PlaybackClock(speed=1.0, now=lambda: next(fake))
    clk2.start(media_t=0.0)
    assert clk2.media_time() == pytest.approx(3.0)          # injected clock used


def test_playback_is_driven_by_camera_time(tmp_path):
    rec = load(tmp_path, clock_lines(120))
    clk = rc.PlaybackClock(speed=2.0)
    clk.start(100.0, media_t=0.0)
    media = clk.media_time(100.5)
    shown = rc.pose_at(rec, media, "hold")
    assert shown.source == "held"
    assert shown.i0 == 1_000_000 // PERIOD_US                # latest frame at or before 1 s
    assert rec.timeline.t[shown.i0] <= media < rec.timeline.t[shown.i0 + 1]


def test_scheduling_log_statistics():
    log = rc.SchedulingLog()
    assert log.stats()["n"] == 0
    log.record(0.0, 0.001, frame_index=0)
    log.record(0.1, 0.103, frame_index=3)
    log.record(0.2, 0.2, frame_index=4)
    s = log.stats()
    assert s["n"] == 3
    assert s["mean_ms"] == pytest.approx(4.0 / 3.0)
    assert s["max_ms"] == pytest.approx(3.0)
    assert s["p95_ms"] == pytest.approx(2.8)                 # linear between 1 and 3 ms
    assert s["frames_skipped"] == 2


def test_hold_and_interpolate_between_frames_of_a_quarter_turn(tmp_path):
    pos_a, rot_a = place(*RIGHT_LOCAL)
    Rz = rodrigues((0, 0, 1), 90.0)
    pos_b, rot_b = pos_a @ Rz.T, Rz @ rot_a
    lines = [hand_line(pos_a, rot_a, frame_id=10),
             hand_line(pos_b, rot_b, frame_id=11, ts_us=TS0 + PERIOD_US,
                       capture=1_790_000_000.011111)]
    rec = load(tmp_path, lines)
    ra, rb = (rc.reconstruct(f) for f in rec.frames)
    t_mid = rec.timeline.t[1] / 2.0

    held = rc.pose_at(rec, t_mid, "hold")
    assert held.source == "held" and held.i0 == 0
    np.testing.assert_array_equal(held.pos, ra.pos)

    mid = rc.pose_at(rec, t_mid, "interpolate")
    assert mid.source == "interpolated" and (mid.i0, mid.i1) == (0, 1)
    assert mid.alpha == pytest.approx(0.5)
    np.testing.assert_allclose(mid.pos, (ra.pos + rb.pos) / 2.0, atol=1e-15)
    for j in range(26):
        assert rc.angular_distance_deg(mid.quat[j], ra.quat[j]) == pytest.approx(45.0, abs=1e-9)
        assert rc.angular_distance_deg(mid.quat[j], rb.quat[j]) == pytest.approx(45.0, abs=1e-9)
        np.testing.assert_allclose(mid.rot[j], rodrigues((0, 0, 1), 45.0) @ ra.rot[j], atol=1e-12)

    # off the midpoint, where swapped weights would show
    quarter = rc.pose_at(rec, rec.timeline.t[1] / 4.0, "interpolate")
    assert quarter.alpha == pytest.approx(0.25)
    np.testing.assert_allclose(quarter.pos, 0.75 * ra.pos + 0.25 * rb.pos, atol=1e-15)
    for j in range(26):
        np.testing.assert_allclose(quarter.rot[j], rodrigues((0, 0, 1), 22.5) @ ra.rot[j],
                                   atol=1e-12)


# --- repeated, nonmonotonic and missing clocks -------------------------------------
def test_evaluate_clock_counts():
    r = rc.evaluate_clock([1, 2, 2, None, 1, 5, float("nan")], field="x")
    assert (r.field, r.present, r.missing, r.repeated, r.nonmonotonic, r.valid) == \
        ("x", 5, 2, 1, 1, False)
    assert r.first_defect == 2
    assert rc.evaluate_clock([633838438806, 633838438806]).repeated == 1
    ok = rc.evaluate_clock([0.0, 0.5, 0.75])
    assert ok.valid and ok.first_defect is None


def test_repeated_timestamp_us_falls_back_to_capture_time(tmp_path):
    lines = clock_lines(6)
    lines[3]["timestamp_us"] = lines[2]["timestamp_us"]
    rec = load(tmp_path, lines)
    tl = rec.timeline
    assert tl.source == "capture_time" and tl.fallback is True
    assert "timestamp_us" in tl.reason and "repeated" in tl.reason
    r = tl.reports["timestamp_us"]
    assert (r.present, r.missing, r.repeated, r.nonmonotonic, r.valid) == (6, 0, 1, 0, False)
    assert tl.reports["capture_time"].valid and tl.reports["wall_time"].valid
    caps = np.array([x["capture_time"] for x in lines])
    np.testing.assert_allclose(tl.t, caps - caps[0], atol=1e-9)
    defects = [e for e in rec.events if e.kind == "clock_defect"]
    assert len(defects) == 1 and defects[0].index == 3


def test_defective_camera_clocks_fall_back_to_the_writer(tmp_path):
    lines = clock_lines(6)
    lines[4]["timestamp_us"] = lines[2]["timestamp_us"] - 5      # goes backwards
    lines[1]["capture_time"] = None                               # missing
    rec = load(tmp_path, lines)
    tl = rec.timeline
    assert tl.source == "wall_time" and tl.fallback
    assert (tl.reports["timestamp_us"].nonmonotonic, tl.reports["timestamp_us"].repeated) == (1, 0)
    assert (tl.reports["capture_time"].missing, tl.reports["capture_time"].present) == (1, 5)
    walls = np.array([x["wall_time"] for x in lines])
    np.testing.assert_allclose(tl.t, walls - walls[0], atol=1e-9)
    assert [e.kind for e in rec.events].count("clock_defect") == 1


def test_no_valid_clock_uses_frame_index_at_the_nominal_rate(tmp_path):
    lines = clock_lines(6)
    lines[2]["timestamp_us"] = lines[1]["timestamp_us"]          # repeated
    del lines[3]["capture_time"]                                  # absent key = missing
    lines[4]["wall_time"] = lines[0]["wall_time"]                 # goes backwards
    rec = load(tmp_path, lines)
    tl = rec.timeline
    assert tl.source == "frame_index_nominal" and tl.fallback
    np.testing.assert_allclose(tl.t, np.arange(6) / 90.0, atol=1e-15)
    assert tl.nominal_period_s == pytest.approx(1 / 90.0)
    rep = tl.reports
    assert (rep["timestamp_us"].repeated, rep["capture_time"].missing,
            rep["wall_time"].nonmonotonic) == (1, 1, 1)
    assert [e.kind for e in rec.events] == ["clock_defect"]
    hinted = rc.build_timeline(rec.frames, framerate_hint=60.0)
    np.testing.assert_allclose(hinted.t, np.arange(6) / 60.0, atol=1e-15)


# --- gaps, identity changes and dropped frames ---------------------------------------
def test_identity_change_splits_and_is_never_interpolated(tmp_path):
    rec = load(tmp_path, clock_lines(10, hand_ids=[7] * 5 + [8] * 5), ids=(7, 8))
    assert [(s.start, s.end, s.hand_id) for s in rec.segments] == [(0, 4, 7), (5, 9, 8)]
    assert [e.kind for e in rec.events] == ["id_change"]
    e = rec.events[0]
    assert (e.index, e.frame_id_before, e.frame_id_after) == (5, 1004, 1005)
    t_mid = (rec.timeline.t[4] + rec.timeline.t[5]) / 2.0
    shown = rc.pose_at(rec, t_mid, "interpolate")
    assert shown.source == "held" and shown.i0 == 4
    assert "discontinuity" in shown.note
    np.testing.assert_array_equal(shown.pos, rc.reconstruct(rec.frames[4]).pos)


def test_time_gap_splits_without_duplicate_events(tmp_path):
    ids = [1000 + i for i in range(5)] + [1032 + i for i in range(5)]
    rec = load(tmp_path, clock_lines(10, frame_ids=ids, extra_us={5: 300_000}))
    assert [(s.start, s.end) for s in rec.segments] == [(0, 4), (5, 9)]
    assert [e.kind for e in rec.events] == ["gap"]
    gap = rec.events[0]
    assert gap.dt_s == pytest.approx(0.311111, abs=1e-9)
    assert "27 frame ids skipped" in gap.detail      # counted here, not a 2nd event
    shown = rc.pose_at(rec, rec.timeline.t[4] + 0.1, "interpolate")
    assert shown.source == "held" and shown.i0 == 4 and "discontinuity" in shown.note


def test_dropped_frames_inside_a_segment_are_shown_and_interpolated(tmp_path):
    ids = [100, 101, 102, 103, 104, 107, 108, 109]
    rec = load(tmp_path, clock_lines(8, frame_ids=ids, extra_us={5: 2 * PERIOD_US}))
    assert len(rec.segments) == 1
    assert [e.kind for e in rec.events] == ["dropped_frames"]
    assert rec.events[0].index == 5
    assert "2 frame ids skipped" in rec.events[0].detail
    t_mid = (rec.timeline.t[4] + rec.timeline.t[5]) / 2.0
    shown = rc.pose_at(rec, t_mid, "interpolate")
    assert shown.source == "interpolated" and shown.alpha == pytest.approx(0.5)
    a, b = rc.reconstruct(rec.frames[4]), rc.reconstruct(rec.frames[5])
    np.testing.assert_allclose(shown.pos, (a.pos + b.pos) / 2.0, atol=1e-15)


# --- hand selection and exclusions --------------------------------------------------
def test_selection_counts_every_excluded_line(tmp_path):
    right = clock_lines(4)
    right[3]["hand_id"] = 9                                   # another id, same side
    left = [json.loads(json.dumps(x)) for x in right[:2]]
    for x in left:                                            # same tracking frame,
        x.update(hand_side="left", hand_id=6)                 # same timestamps
    invalid = json.loads(json.dumps(right[2]))
    invalid.update(status=0, frame_id=1099)
    trailing = json.loads(json.dumps(invalid))
    trailing.update(frame_id=1200)                            # after every good frame
    lines = [right[0], left[0], right[1], left[1], invalid, right[2], right[3], trailing]
    rec = load(tmp_path, lines, ids=(7,))
    assert rec.lines_total == 8
    assert rec.lines_excluded == {"other_hand": 2, "other_id": 1, "invalid_status": 2}
    assert [f.frame_id for f in rec.frames] == [1000, 1001, 1002]
    assert [f.line_no for f in rec.frames] == [1, 3, 6]
    assert rec.timeline.source == "timestamp_us"              # the other hand's repeats do not count
    inv = [(e.index, e.frame_id_after, e.detail[:7]) for e in rec.events
           if e.kind == "invalid_status"]
    assert inv == [(2, 1099, "line 5:"), (2, 1200, "line 8:")]  # drawn on a real frame
    every_id = rc.load_recording(tmp_path / "rec.jsonl", "right", None)
    assert len(every_id.frames) == 4 and every_id.lines_excluded["other_id"] == 0
    assert max(e.index for e in every_id.events) == 3
    with pytest.raises(ValueError):
        rc.load_recording(tmp_path / "rec.jsonl", "both", None)


def test_hand_ids_as_written_in_the_metadata_tables(tmp_path):
    """The tables store ids as "573 574": split, never read digit by digit."""
    path = write_jsonl(tmp_path / "ids.jsonl", clock_lines(4, hand_ids=[12, 12, 13, 13]))
    assert len(rc.load_recording(path, "right", "12 13").frames) == 4
    assert [f.hand_id for f in rc.load_recording(path, "right", "12").frames] == [12, 12]


def test_reference_frames_move_with_the_hand():
    pos1, rot1 = place(*RIGHT_LOCAL)
    pos2, rot2 = place(*RIGHT_LOCAL, R=rodrigues((-0.5, 0.2, 1.0), 120.0),
                       t=np.array([-0.1, 0.3, 0.1]))
    for ref in ("wrist", "palm"):
        a, b = rc.to_reference(pos1, rot1, ref), rc.to_reference(pos2, rot2, ref)
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)
    lp, lr = rc.to_reference(pos1, rot1, "wrist")
    local_pos, local_rot = RIGHT_LOCAL
    np.testing.assert_allclose(lp, local_pos @ local_rot[WRIST], atol=1e-12)  # R^T p
    np.testing.assert_allclose(lr[WRIST], np.eye(3), atol=1e-12)
    pp, pr = rc.to_reference(pos1, rot1, "palm")
    np.testing.assert_allclose(pp[WRIST], 0.0, atol=1e-15)
    np.testing.assert_allclose(pr[PALM], np.eye(3), atol=1e-12)
    cp, cr = rc.to_reference(pos1, rot1, "camera")
    np.testing.assert_array_equal(cp, pos1)
    np.testing.assert_array_equal(cr, rot1)
    with pytest.raises(ValueError):
        rc.to_reference(pos1, rot1, "elbow")


# --- the real dataset ---------------------------------------------------------------
@needs_dataset
def test_real_writer_stall_does_not_move_the_camera_clock():
    rec = rc.find_static(DATASET, "p3_circular_precision_thumb_2_fingers", 4)
    assert rec.provenance == ("metadata/grasps_summary.csv: grasp_name="
                              "p3_circular_precision_thumb_2_fingers take=4 "
                              "hand=left hand_ids=96")
    assert len(rec.frames) == 449 and rec.kind == "static"
    assert rec.timeline.source == "timestamp_us" and rec.timeline.fallback is False
    assert len(rec.segments) == 1 and rec.events == []
    at = {f.frame_id: f.index for f in rec.frames}
    a, b = rec.frames[at[151852]], rec.frames[at[151853]]
    assert b.wall_time - a.wall_time > 0.8
    assert b.timestamp_us - a.timestamp_us == 10_809
    t = rec.timeline.t
    assert t[at[151855]] - t[at[151749]] == pytest.approx(1.178128, abs=1e-9)
    writer = rec.frames[at[151855]].wall_time - rec.frames[at[151749]].wall_time
    assert writer == pytest.approx(1.999, abs=0.001)
    mask = rc.hold_mask(rec)
    assert mask.sum() == 107 and mask[at[151749]] and not mask[at[151749] - 1]
    fid = rc.recording_fidelity(rec)
    assert fid["within_tolerance"] is True
    assert fid["position_residual_m"]["max"] <= 1e-6
    json.dumps(fid, allow_nan=False)


@needs_dataset
def test_real_identity_change_gives_two_segments():
    rec = rc.find_static(DATASET, "p2_s03_power_grasp_finger_side", 1)
    assert rec.hand_ids == [573, 574] and len(rec.frames) == 409
    assert [(s.start, s.end, s.hand_id) for s in rec.segments] == [(0, 60, 573), (61, 408, 574)]
    kinds = sorted(e.kind for e in rec.events)
    assert kinds == ["gap", "id_change"]
    for e in rec.events:
        assert (e.index, e.frame_id_before, e.frame_id_after) == (61, 88967, 89008)
        assert e.dt_s == pytest.approx(0.455302, abs=1e-9)
        assert "40 frame ids skipped" in e.detail
    t_mid = (rec.timeline.t[60] + rec.timeline.t[61]) / 2.0
    shown = rc.pose_at(rec, t_mid, "interpolate")
    assert shown.source == "held" and shown.i0 == 60 and "discontinuity" in shown.note
    assert rc.recording_fidelity(rec)["within_tolerance"] is True


@needs_dataset
def test_real_metadata_tables_and_a_transition():
    assert len(rc.list_static(DATASET)) == 115
    assert len(rc.list_transitions(DATASET)) == 52
    rec = rc.find_transition(DATASET, "p3_circular_precision_thumb_2_fingers", 4)
    assert rec.kind == "transition"
    assert rec.provenance.startswith("metadata/pose_transitions.csv: to_grasp="
                                     "p3_circular_precision_thumb_2_fingers take=4 ")
    assert len(rec.frames) == int(rec.metadata["frames"])
    assert rec.segments[0].hand_id == 94 and rec.segments[-1].hand_id == 96
    assert "id_change" in [e.kind for e in rec.events]
    assert rc.hold_mask(rec) is None                          # no static interval
    with pytest.raises(LookupError):
        rc.find_static(DATASET, "no_such_grasp", 1)
