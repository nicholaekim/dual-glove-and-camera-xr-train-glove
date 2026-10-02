"""xr_hand.joint_frames and scripts/joint_frames_view.py.

Synthetic hands (xr_hand.mock, leap_hand.mock and hand-made frames built in
the real convention: bones along -z, dorsal +y) check the numbers; the
script is run on synthetic takes and sessions. The day-2 recordings are read
only when they are on disk.
"""
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from leap_hand.mock import MockLeapStream
from leap_hand.to_openxr import to_hand_frame
from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES
from xr_hand.kinematics import (PARENT, forward_kinematics_full, mat3_to_quat,
                                quat_to_mat3)
from xr_hand.mock import _REST, MockHandGenerator
from xr_hand.parser import parse_hand_message

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "joint_frames_view", REPO / "scripts" / "joint_frames_view.py")
view = importlib.util.module_from_spec(spec)
spec.loader.exec_module(view)

MM, DEG = 0.5, 0.5


# --- synthetic hands ------------------------------------------------------------------
def mock_glove_line(hand="right", frames=20):
    gen = MockHandGenerator(hand=hand)
    for _ in range(frames):
        raw = gen.next_frame()
    return jf.line_from_hand_frame(parse_hand_message(raw, hand_side_hint=hand))


# A hand in the real convention (glove and LeapC): every child along -z of
# its parent, +y dorsal, x = y cross z. Metacarpal bases in the wrist frame,
# right hand (+x toward the little finger); a left hand negates x.
BASES = {"THUMB": (-25.0, -12.0, -22.0), "INDEX": (-12.0, 3.0, -17.0),
         "MIDDLE": (0.0, 4.0, -17.5), "RING": (7.0, 2.0, -17.0),
         "LITTLE": (13.5, -4.0, -14.0)}
LENGTHS = {"THUMB": (36.0, 27.0, 32.0, 23.0),
           "INDEX": (65.0, 37.0, 22.0, 19.0),
           "MIDDLE": (64.0, 45.0, 23.0, 20.0),
           "RING": (60.0, 43.0, 22.0, 18.0),
           "LITTLE": (57.0, 33.0, 17.0, 14.0)}


def made_line(hand="right", rel=None):
    """A glove-convention line; `rel` maps joint name -> parent-relative 3x3."""
    rel = rel or {}
    s = 1.0 if hand == "right" else -1.0
    joints = []
    for name in JOINT_NAMES:
        r = rel.get(name, np.eye(3))
        if name == "WRIST":
            t = (0.0, 0.0, 0.0)
        elif name == "PALM":
            t = (0.0, 4.0, -50.0)
        else:
            finger, part = name.split("_")
            if part == "METACARPAL":
                b = BASES[finger]
                t = (s * b[0], b[1], b[2])
            else:
                k = ["PROXIMAL", "INTERMEDIATE", "DISTAL", "TIP"].index(part)
                t = (0.0, 0.0, -LENGTHS[finger][k])
        qx, qy, qz, qw = mat3_to_quat(r)
        joints.append({"name": name, "x": t[0] / 1000.0, "y": t[1] / 1000.0,
                       "z": t[2] / 1000.0, "qx": qx, "qy": qy, "qz": qz,
                       "qw": qw})
    return {"hand_side": hand, "joints": joints}


def absolute_line(line, rotation=None, offset=(0.0, 0.0, 0.0)):
    """The same hand as absolute abs26 + quat26, placed anywhere in a world."""
    pos, rots, hand = jf.world_pose(line, "glove")
    g = np.eye(3) if rotation is None else np.asarray(rotation)
    o = np.asarray(offset, dtype=float)
    return {"hand_side": hand, "units": "m",
            "abs26": [list(o + g @ p) for p in pos],
            "quat26": [list(mat3_to_quat(g @ r)) for r in rots]}


def row(rows, name):
    return rows[JOINT_INDEX[name]]


def assert_tables_equal(a, b, mm=MM, deg=DEG):
    assert [r["joint"] for r in a] == [r["joint"] for r in b] == JOINT_NAMES
    for ra, rb in zip(a, b):
        for k in ("x_mm", "y_mm", "z_mm", "bone_len_mm"):
            assert ra[k] == pytest.approx(rb[k], abs=mm), (ra["joint"], k)
        for k in ("flex_deg", "abd_deg", "twist_deg"):
            assert ra[k] == pytest.approx(rb[k], abs=deg), (ra["joint"], k)
        for k in ("axis_x", "axis_y", "axis_z"):
            assert np.allclose(ra[k], rb[k], atol=1e-3), (ra["joint"], k)
        qa = np.array([ra[k] for k in ("qx", "qy", "qz", "qw")])
        qb = np.array([rb[k] for k in ("qx", "qy", "qz", "qw")])
        assert abs(float(qa @ qb)) == pytest.approx(1.0, abs=1e-4)


# --- the table ---------------------------------------------------------------------------
def test_table_has_26_rows_in_joint_order_with_every_key():
    rows = jf.joint_table(mock_glove_line(), "glove")
    assert [r["joint"] for r in rows] == JOINT_NAMES
    for r in rows:
        assert list(r) == jf.ROW_KEYS
        for a in ("axis_x", "axis_y", "axis_z"):
            assert len(r[a]) == 3
            assert np.linalg.norm(r[a]) == pytest.approx(1.0, abs=1e-9)
        # the axes are a right-handed orthonormal frame
        assert np.allclose(np.cross(r["axis_x"], r["axis_y"]), r["axis_z"],
                           atol=1e-9)
    assert row(rows, "INDEX_TIP")["parent"] == "INDEX_DISTAL"
    assert row(rows, "PALM")["parent"] == "WRIST"


@pytest.mark.parametrize("hand", ["left", "right"])
def test_wrist_row_is_the_origin_with_identity(hand):
    for line, source in ((mock_glove_line(hand), "glove"),
                         (made_line(hand), "glove"),
                         (absolute_line(made_line(hand), jf.rot_x(30)
                                        @ jf.rot_z(-70), (0.1, 0.25, 0.04)),
                          "camera")):
        w = row(jf.joint_table(line, source), "WRIST")
        assert w["parent"] == ""
        assert (w["x_mm"], w["y_mm"], w["z_mm"]) == pytest.approx((0, 0, 0),
                                                                  abs=1e-9)
        assert (w["qx"], w["qy"], w["qz"], w["qw"]) == pytest.approx(
            (0, 0, 0, 1), abs=1e-9)
        assert np.allclose([w["axis_x"], w["axis_y"], w["axis_z"]], np.eye(3),
                           atol=1e-9)
        assert w["bone_len_mm"] == 0.0
        assert (w["flex_deg"], w["abd_deg"], w["twist_deg"]) == (0, 0, 0)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_bone_lengths_match_the_mock_template(hand):
    rows = jf.joint_table(mock_glove_line(hand), "glove")
    for r in rows:
        if r["joint"] == "WRIST":
            continue
        want = 1000.0 * math.dist((0, 0, 0), _REST[r["joint"]])
        assert r["bone_len_mm"] == pytest.approx(want, abs=1e-6), r["joint"]


def test_bone_lengths_of_a_hand_made_hand():
    rows = jf.joint_table(made_line("left", {
        "INDEX_PROXIMAL": jf.rot_x(-40)}), "glove")
    assert row(rows, "INDEX_PROXIMAL")["bone_len_mm"] == pytest.approx(65.0)
    assert row(rows, "INDEX_INTERMEDIATE")["bone_len_mm"] == pytest.approx(37.0)
    assert row(rows, "INDEX_METACARPAL")["bone_len_mm"] == pytest.approx(
        math.dist((0, 0, 0), BASES["INDEX"]))
    assert row(rows, "LITTLE_TIP")["bone_len_mm"] == pytest.approx(14.0)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_pure_90_degree_flexion_at_the_index_pip(hand):
    # the glove's own fist: -90 degrees about the joint's +x axis
    rows = jf.joint_table(made_line(hand, {"INDEX_INTERMEDIATE":
                                           jf.rot_x(-90)}), "glove")
    pip = row(rows, "INDEX_INTERMEDIATE")
    assert pip["flex_deg"] == pytest.approx(90.0, abs=DEG)
    assert pip["abd_deg"] == pytest.approx(0.0, abs=DEG)
    assert pip["twist_deg"] == pytest.approx(0.0, abs=DEG)
    # nothing else moved
    for r in rows:
        if r["joint"] != "INDEX_INTERMEDIATE":
            if r["joint"].endswith("_METACARPAL"):
                continue
            assert r["flex_deg"] == pytest.approx(0.0, abs=1e-6), r["joint"]
    # the fingertip went toward the palm, which is -y
    assert row(rows, "INDEX_TIP")["y_mm"] < row(rows, "INDEX_PROXIMAL")["y_mm"] - 30
    assert jf.dof24(rows)["I_PIP"] == pytest.approx(90.0, abs=DEG)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_pure_abduction_at_the_index_mcp(hand):
    # a rotation about the dorsal +y axis; toward the thumb is -x on a
    # right hand and +x on a left hand, i.e. +y rotation right, -y left
    toward_thumb = 20.0 if hand == "right" else -20.0
    rows = jf.joint_table(made_line(hand, {"INDEX_PROXIMAL":
                                           jf.rot_y(toward_thumb)}), "glove")
    mcp = row(rows, "INDEX_PROXIMAL")
    assert mcp["abd_deg"] == pytest.approx(20.0, abs=DEG)
    assert mcp["flex_deg"] == pytest.approx(0.0, abs=DEG)
    assert mcp["twist_deg"] == pytest.approx(0.0, abs=DEG)
    # the index tip moved toward the thumb's side of the hand
    tip, knuckle = row(rows, "INDEX_TIP"), row(rows, "INDEX_PROXIMAL")
    thumb_side = -1.0 if hand == "right" else 1.0
    assert (tip["x_mm"] - knuckle["x_mm"]) * thumb_side > 20.0
    assert jf.dof24(rows)["I_MCP_aa"] == pytest.approx(20.0, abs=DEG)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_pure_twist_about_the_bone(hand):
    rows = jf.joint_table(made_line(hand, {"MIDDLE_DISTAL": jf.rot_z(15)}),
                          "glove")
    dip = row(rows, "MIDDLE_DISTAL")
    assert dip["twist_deg"] == pytest.approx(-15.0 * jf.hand_sign(hand),
                                             abs=DEG)
    assert dip["flex_deg"] == pytest.approx(0.0, abs=DEG)
    assert dip["abd_deg"] == pytest.approx(0.0, abs=DEG)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_angles_round_trip_over_the_anatomical_range(hand):
    rng = np.random.default_rng(11)
    for _ in range(200):
        flex, abd, twist = rng.uniform(-30, 120), rng.uniform(-40, 40), \
            rng.uniform(-60, 60)
        got = jf.joint_angles(jf.rotation_from_angles(flex, abd, twist, hand),
                              hand)
        assert got == pytest.approx((flex, abd, twist), abs=1e-6)


def test_the_locked_angle_still_returns_the_whole_rotation():
    a, b, c = jf.xyz_angles(jf.rot_y(90) @ jf.rot_x(0))
    assert math.degrees(b) == pytest.approx(90.0)
    assert np.allclose(jf.rot_x(math.degrees(a)) @ jf.rot_y(math.degrees(b))
                       @ jf.rot_z(math.degrees(c)), jf.rot_y(90), atol=1e-9)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_a_mirrored_hand_reads_the_same_angles(hand):
    """The left hand is the right hand's mirror: same anatomical angles."""
    angles = {"INDEX_PROXIMAL": (40.0, 12.0, 5.0),
              "INDEX_INTERMEDIATE": (70.0, 0.0, 0.0),
              "THUMB_PROXIMAL": (25.0, -6.0, 8.0),
              "LITTLE_PROXIMAL": (30.0, -15.0, -4.0)}
    rel = {j: jf.rotation_from_angles(*a, hand) for j, a in angles.items()}
    rows = jf.joint_table(made_line(hand, rel), "glove")
    for j, (flex, abd, twist) in angles.items():
        r = row(rows, j)
        assert (r["flex_deg"], r["abd_deg"], r["twist_deg"]) == pytest.approx(
            (flex, abd, twist), abs=1e-6), j


@pytest.mark.parametrize("hand", ["left", "right"])
def test_camera_path_and_glove_path_give_the_same_table(hand):
    """Absolute joints anywhere in a camera's space == the relative glove
    joints of the same hand, row for row."""
    glove = mock_glove_line(hand, frames=37)       # mid-curl
    placed = absolute_line(glove, jf.rot_z(25) @ jf.rot_x(-60) @ jf.rot_y(140),
                           (-0.08, 0.24, 0.05))
    assert_tables_equal(jf.joint_table(glove, "glove"),
                        jf.joint_table(placed, "camera"))
    made = made_line(hand, {"INDEX_PROXIMAL": jf.rotation_from_angles(
        45, 10, 3, hand), "RING_INTERMEDIATE": jf.rot_x(-80),
        "THUMB_METACARPAL": jf.rot_z(-70) @ jf.rot_y(-35)})
    placed = absolute_line(made, jf.rot_y(-100), (0.2, 0.3, -0.1))
    assert_tables_equal(jf.joint_table(made, "glove"),
                        jf.joint_table(placed, "camera"))


@pytest.mark.parametrize("hand", ["left", "right"])
def test_leap_hand_absolute_and_its_recorded_file_line_agree(hand):
    """A live LeapHand (absolute) and the line the recorder writes for it
    (parent-relative, WRIST in camera space) give one table."""
    stream = MockLeapStream(pose="fist", seed=3, dropout_every=0,
                            reacquire_every=0)
    lh = next(h for side, h in stream.generate(3) if side == hand)
    live = jf.joint_table(jf.line_from_leap_hand(lh), "camera")
    filed = jf.joint_table(jf.line_from_hand_frame(to_hand_frame(lh)),
                           "camera")
    assert_tables_equal(live, filed, mm=1e-6, deg=1e-6)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_camera_mock_signs(hand):
    """LeapC's convention through the Leap mock: a fist flexes positive,
    an open palm reads zero, and a spread is toward the thumb on both
    hands (the mock mirrors the left hand properly)."""
    def table(pose):
        stream = MockLeapStream(pose=pose, seed=1, dropout_every=0,
                                reacquire_every=0, cycle_seconds=1.0)
        hands = [h for side, h in stream.generate(46) if side == hand]
        return jf.joint_table(jf.line_from_leap_hand(hands[-1]), "camera")
    fist, open_palm = jf.dof24(table("fist")), jf.dof24(table("open_palm"))
    for f in "IMRL":
        assert fist[f"{f}_PIP"] == pytest.approx(math.degrees(1.05), abs=DEG)
        assert fist[f"{f}_MCP_fe"] == pytest.approx(math.degrees(0.95),
                                                    abs=DEG)
        assert open_palm[f"{f}_PIP"] == pytest.approx(0.0, abs=DEG)
    spread = row(table("open_palm"), "INDEX_METACARPAL")
    assert spread["abd_deg"] == pytest.approx(math.degrees(0.20), abs=DEG)
    little = row(table("open_palm"), "LITTLE_METACARPAL")
    assert little["abd_deg"] == pytest.approx(-math.degrees(0.24), abs=DEG)


def test_glove_mock_curl_is_positive_flexion():
    """xr_hand.mock curls about -x (its own cartoon axes); still positive."""
    gen = MockHandGenerator(hand="left")
    for _ in range(40):                           # curl rising from 0
        raw = gen.next_frame()
    rows = jf.joint_table(jf.line_from_hand_frame(
        parse_hand_message(raw, hand_side_hint="left")), "glove")
    curl = 0.5 * (1 - math.cos(40 / 60.0 * 1.5))
    want = math.degrees(curl * math.pi / 4)
    for name in ("INDEX_PROXIMAL", "INDEX_INTERMEDIATE", "RING_DISTAL"):
        assert row(rows, name)["flex_deg"] == pytest.approx(want, abs=1e-4)


def test_palm_reference_removes_a_bent_forearm():
    """On the camera the WRIST quaternion is the forearm: bend it and the
    wrist-frame CMC moves with it, the palm-frame CMC does not."""
    hand = "left"
    rel = {"INDEX_INTERMEDIATE": jf.rot_x(-60)}
    straight = absolute_line(made_line(hand, rel), jf.rot_y(20), (0, .25, 0))
    bent = json.loads(json.dumps(straight))
    forearm = quat_to_mat3(*bent["quat26"][JOINT_INDEX["WRIST"]]) @ \
        jf.rot_x(-25)                              # forearm 25 deg off the hand
    bent["quat26"][JOINT_INDEX["WRIST"]] = list(mat3_to_quat(forearm))
    assert jf.wrist_palm_deg(straight, "camera") == pytest.approx(0, abs=1e-6)
    assert jf.wrist_palm_deg(bent, "camera") == pytest.approx(25, abs=1e-6)
    w_straight = jf.dof24(jf.joint_table(straight, "camera"))
    w_bent = jf.dof24(jf.joint_table(bent, "camera"))
    p_bent = jf.dof24(jf.joint_table(bent, "camera", reference="palm"))
    assert w_bent["I_CMC"] == pytest.approx(w_straight["I_CMC"] - 25, abs=DEG)
    assert p_bent["I_CMC"] == pytest.approx(w_straight["I_CMC"], abs=1e-6)
    # joint-to-joint angles never depend on the reference
    assert p_bent["I_PIP"] == pytest.approx(60.0, abs=1e-6)
    assert w_bent["I_PIP"] == pytest.approx(60.0, abs=1e-6)
    # on the glove the two references are the same thing
    glove = made_line(hand, rel)
    assert_tables_equal(jf.joint_table(glove, "glove"),
                        jf.joint_table(glove, "glove", reference="palm"),
                        mm=1e-9, deg=1e-9)


def test_to_world_puts_the_table_back_where_it_was():
    line = made_line("right", {"MIDDLE_PROXIMAL": jf.rot_x(-50)})
    rows = jf.joint_table(line, "glove")
    g = jf.rot_z(40) @ jf.rot_x(10)
    pos, axes = jf.to_world(rows, (0.05, 0.2, 0.0), g)
    again = jf.table_from_world(pos, list(axes), "right")
    assert_tables_equal(rows, again, mm=1e-6, deg=1e-6)


# --- the paper's 24 ----------------------------------------------------------------------
def test_dof24_has_the_papers_24_names():
    rows = jf.joint_table(mock_glove_line("left"), "glove")
    dof = jf.dof24(rows)
    assert len(dof) == 24 and list(dof) == jf.DOF24_NAMES
    assert set(dof) == {"T_TMC_fe", "T_TMC_aa", "T_MCP_fe", "T_IP"} | {
        f"{f}_{j}" for f in "IMRL"
        for j in ("CMC", "MCP_fe", "MCP_aa", "PIP", "DIP")}
    assert dof["I_PIP"] == row(rows, "INDEX_INTERMEDIATE")["flex_deg"]
    assert dof["L_MCP_aa"] == row(rows, "LITTLE_PROXIMAL")["abd_deg"]
    assert dof["T_TMC_aa"] == row(rows, "THUMB_METACARPAL")["abd_deg"]
    assert dof["T_IP"] == row(rows, "THUMB_DISTAL")["flex_deg"]
    assert all(isinstance(v, float) for v in dof.values())


# --- small things ---------------------------------------------------------------------------
def test_csv_rows_and_columns():
    rows = jf.joint_table(made_line("left"), "glove")
    cells = jf.csv_rows(rows)
    assert len(cells) == 26
    assert all(len(c) == len(jf.CSV_COLUMNS) for c in cells)
    assert not any(c.startswith("-0.00") and not c.strip("-0.")
                   for line in cells for c in line)
    assert jf.CSV_COLUMNS[9:12] == ["axis_x_x", "axis_x_y", "axis_x_z"]


def test_source_detection_and_errors():
    assert jf.detect_source(made_line()) == "glove"
    assert jf.detect_source({"source": "leap", "joints": []}) == "camera"
    assert jf.detect_source(absolute_line(made_line())) == "camera"
    # numpy arrays straight off a LeapHand read the same as lists
    arr = absolute_line(made_line())
    arr = dict(arr, abs26=np.array(arr["abs26"]), quat26=np.array(arr["quat26"]))
    assert jf.detect_source(arr) == "camera"
    assert_tables_equal(jf.joint_table(arr, "camera"),
                        jf.joint_table(made_line(), "glove"))
    with pytest.raises(ValueError):
        jf.joint_table(made_line(), "webcam")
    with pytest.raises(ValueError):
        jf.joint_table(dict(made_line(), hand_side="both"), "glove")
    broken = made_line()
    broken["joints"] = broken["joints"][:-1]
    with pytest.raises(ValueError, match="LITTLE_TIP"):
        jf.joint_table(broken, "glove")
    with pytest.raises(ValueError):
        jf.joint_table(made_line(), "glove", reference="elbow")
    # millimetre lines are scaled, metre lines are not
    mm = made_line()
    for j in mm["joints"]:
        j.update(x=j["x"] * 1000, y=j["y"] * 1000, z=j["z"] * 1000)
    mm["units"] = "mm"
    assert_tables_equal(jf.joint_table(mm, "glove"),
                        jf.joint_table(made_line(), "glove"), mm=1e-9)


def test_parent_table_is_the_bone_table():
    rows = jf.joint_table(made_line(), "glove")
    for i, r in enumerate(rows):
        assert r["parent"] == ("" if i not in PARENT
                               else JOINT_NAMES[PARENT[i]])


# --- real data, when it is on this machine ------------------------------------------
DAY2 = REPO / "recordings" / "sync_day2"


@pytest.mark.skipif(not (DAY2 / "glove").is_dir(), reason="day-2 recordings not here")
def test_day2_fist_and_open_palm_signs():
    def pip(folder, pattern):
        path = sorted((DAY2 / folder).glob(pattern))[0]
        line, _i = view.medoid_line(view.read_lines(path), prefer="left")
        return jf.dof24(jf.joint_table(line, folder if folder == "glove"
                                       else "camera"))["I_PIP"]
    fist_g, fist_c = pip("glove", "fist_left_take1_*.jsonl"), \
        pip("leap", "fist_left_take1_*.jsonl")
    open_g = pip("glove", "open_palm_left_take1_*.jsonl")
    assert fist_g > 60 and open_g < 20
    assert abs(fist_g - fist_c) < 25


# --- the script ----------------------------------------------------------------------------
def write_take(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d) + "\n" for d in lines),
                    encoding="utf-8")
    return path


def glove_take(path: Path, hand="left", n=30):
    gen = MockHandGenerator(hand=hand)
    lines = []
    for k in range(n):
        line = jf.line_from_hand_frame(parse_hand_message(
            gen.next_frame(), hand_side_hint=hand))
        line.update(wall_time=1_790_000_000.0 + k / 60.0, status=1,
                    timestamp=float(k), packet_counter=k, frame_id=k)
        lines.append(line)
    return write_take(path, lines)


def read_csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_take_export_writes_the_csv_the_png_and_prints_24(tmp_path, capsys):
    take = glove_take(tmp_path / "in" / "index_left_take1_20260928_140000.jsonl")
    out = tmp_path / "out"
    assert view.main(["--take", str(take), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    table = read_csv(out / f"{take.stem}.csv")
    assert table[0] == jf.CSV_COLUMNS and len(table) == 27
    assert [r[0] for r in table[1:]] == JOINT_NAMES
    png = out / f"{take.stem}.png"
    assert png.read_bytes()[:4] == b"\x89PNG"
    assert "glove, left hand" in text and "medoid of the whole take" in text
    for name in ("TMC_fe", "MCP_aa", "PIP", "DIP", "CMC", "IP"):
        assert name in text


def test_take_export_frame_n_and_a_camera_take(tmp_path):
    stream = MockLeapStream(pose="fist", seed=2, dropout_every=0,
                            reacquire_every=0)
    lines = []
    for k, (side, lh) in enumerate(stream.generate(20)):
        line = jf.line_from_hand_frame(to_hand_frame(lh))
        line.update(wall_time=lh.capture_time, status=1, source="leap",
                    units="m", abs26=lh.abs26, hand_id=lh.hand_id)
        lines.append(line)
    take = write_take(tmp_path / "leap" / "grip_left_take1.jsonl", lines)
    info = view.export_take(take, tmp_path / "o", frame="5")
    assert info["source"] == "camera" and info["index"] == 5
    assert info["hand"] == lines[5]["hand_side"]
    assert len(read_csv(info["csv"])) == 27
    with pytest.raises(ValueError, match="lines 0 to 39"):
        view.export_take(take, tmp_path / "o", frame="40")
    assert view.main(["--take", str(tmp_path / "missing.jsonl")]) == 1


def test_the_drawing_primitives_cover_every_joint_and_axis():
    rows = jf.joint_table(made_line("left"), "glove")
    prims = view.layout(rows, jf.dof24(rows), "t", "s", "left")
    axis_lines = [p for p in prims if p[0] == "line" and p[5] in view.AXIS_RGB]
    assert len(axis_lines) >= 2 * 26 * 3             # two views, 3 axes each
    texts = " ".join(p[3] for p in prims if p[0] == "text")
    for name in ("TMC", "PIP", "DIP", "CMC", "MCP_aa", "TMC_aa"):
        assert name in texts


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    import test_protocol_check as fx
    root = tmp_path_factory.mktemp("jf_sessions")
    grasps = fx.make_grasp_session(root / "a", takes_per_item=1)
    flexion = fx.make_flexion_session(root / "b", [
        fx.FlexTake("index"),
        fx.FlexTake("middle", peaks=[0.3] * 5, accepted=False,
                    reason="span too small")], stills=True)
    return {"grasps": grasps, "flexion": flexion}


def test_session_export_grasps(sessions, capsys):
    session = sessions["grasps"]
    assert view.main(["--session", str(session)]) == 0
    out = session / "joint_frames"
    from cam_hand import protocol_check as pc
    takes = [t for t in pc.load_session(session).takes if t.accepted]
    assert takes
    pages = {p["stem"]: p for p in view.export_session(session)["pages"]}
    for t in takes:
        assert len(read_csv(out / f"{t.name}.csv")) == 27
        assert (out / f"{t.name}.png").is_file()
        meta = json.loads(t.path("meta").read_text(encoding="utf-8"))
        # the page is the recorder's own summary frame, found again
        assert pages[t.name]["wall_time"] == pytest.approx(
            meta["medoid_wall_time"])
    pdf = out / "joint_frames.pdf"
    assert pdf.read_bytes()[:5] == b"%PDF-"
    import pypdf
    assert len(pypdf.PdfReader(str(pdf)).pages) == len(takes)
    assert "the recorder's static-interval medoid" in capsys.readouterr().out


def test_the_summary_frame_is_the_meta_line_first():
    """The recorder's `medoid_line` is the summary frame when that line has
    the meta's wall time; else the wall-time search of the meta's hand id;
    else the recompute. Two hands share every wall time in a take, so the
    label alone could pick the other hand's line."""
    stream = MockLeapStream(pose="fist", seed=2, dropout_every=0,
                            reacquire_every=0)
    lines = []
    for k, (side, lh) in enumerate(stream.generate(40)):
        line = jf.line_from_hand_frame(to_hand_frame(lh))
        line.update(wall_time=1_790_000_000.0 + (k // 2) / 90.0, status=1,
                    source="leap", units="m", abs26=lh.abs26,
                    hand_id=lh.hand_id)
        lines.append(line)
    right = next(i for i in range(20, 40) if lines[i]["hand_side"] == "right")
    left = right - 1                     # the same frame's other hand
    assert lines[left]["hand_side"] == "left"
    assert lines[left]["wall_time"] == lines[right]["wall_time"]
    t = lines[right]["wall_time"]
    meta = {"medoid_line": right, "medoid_wall_time": t,
            "medoid_hand_id": lines[right]["hand_id"],
            "static_interval": [t - 0.05, t + 0.05]}
    line, i, how = view.static_medoid_line(lines, meta, "left")
    assert i == right and line is lines[right]
    assert how == "the recorder's static-interval medoid (meta line)"
    # a line number that no longer matches its time: the hand id decides
    line, i, how = view.static_medoid_line(lines, {**meta, "medoid_line": 3},
                                           "left")
    assert i == right and how == "the recorder's static-interval medoid (meta)"
    # no line number, no id: the operator's label, as before
    old = {"medoid_wall_time": t, "static_interval": meta["static_interval"]}
    line, i, how = view.static_medoid_line(lines, old, "left")
    assert i == left and how == "the recorder's static-interval medoid (meta)"
    # a line number out of range is ignored
    line, i, _how = view.static_medoid_line(lines, {**meta, "medoid_line": 400},
                                            "left")
    assert i == right


def test_session_export_flexion_has_glove_and_camera(sessions):
    session = sessions["flexion"]
    result = view.export_session(session)
    out = session / "joint_frames"
    names = sorted(p.name for p in out.iterdir())
    from cam_hand import protocol_check as pc
    (take,) = [t for t in pc.load_session(session).takes if t.accepted]
    assert f"{take.name}.csv" in names and f"camera_{take.name}.csv" in names
    assert f"{take.name}.png" in names and "joint_frames.pdf" in names
    # the rejected attempt is not exported
    assert not [n for n in names if "middle" in n]
    assert [p["source"] for p in result["pages"]] == ["glove", "camera"]
    import pypdf
    reader = pypdf.PdfReader(str(out / "joint_frames.pdf"))
    assert len(reader.pages) == 2
    text = reader.pages[0].extract_text()
    assert "INDEX_INTERMEDIATE" in text and "flex" in text
    # no raster image inside the PDF: it is drawn, not pasted
    assert not any(page.images for page in reader.pages)


def test_session_export_then_the_hand_in(sessions, tmp_path):
    """--session, then the packager: the CSVs and the PDF go in, no PNG."""
    pkg_spec = importlib.util.spec_from_file_location(
        "package_professor_set", REPO / "scripts" / "package_professor_set.py")
    pkg = importlib.util.module_from_spec(pkg_spec)
    pkg_spec.loader.exec_module(pkg)
    view.export_session(sessions["grasps"])
    view.export_session(sessions["flexion"])
    out = tmp_path / "handin"
    assert pkg.main(["--out", str(out), "--grasps", str(sessions["grasps"]),
                     "--flexion", str(sessions["flexion"])]) == 0
    g = sorted(p.name for p in (out / "grasps" / "joint_frames").iterdir())
    assert "joint_frames.pdf" in g
    assert {"cylindrical_left_take1.csv", "tip_pinch_left_take1.csv"} <= set(g)
    f = out / "finger_flexion" / "left" / "joint_frames"
    assert len(list(f.glob("*.csv"))) == 2 and (f / "joint_frames.pdf").is_file()
    assert not list(out.rglob("*.png")) and pkg.images_in(out) == []
    rows = read_csv(out / "grasps" / "joint_frames" / "cylindrical_left_take1.csv")
    assert len(rows) == 27


def test_live_mock_headless(capsys):
    assert view.main(["--live", "--mock", "--glove", "--seconds", "1.5",
                      "--no-window"]) == 0
    text = capsys.readouterr().out
    n = int(text.rsplit("frames drawn:", 1)[1].split()[0])
    assert n >= 1


def test_live_view_keys_and_glove_table():
    lv = view.LiveView(glove_on=True)
    stream = MockLeapStream(pose="fist", seed=4, dropout_every=0,
                            reacquire_every=0)
    for side, lh in stream.generate(2):
        lv.add_camera(lh)
    lv.add_glove(parse_hand_message(MockHandGenerator("left").next_frame(),
                                    hand_side_hint="left"))
    project = view.Projector(None)
    img = lv.compose(project)
    assert img.shape == (view.IMAGE_SIZE, view.IMAGE_SIZE + view.PANEL_W, 3)
    assert lv.keys(ord("g")) and lv.source == "glove"
    assert lv.keys(ord("n")) and view.FINGER_CYCLE[lv.finger] == "THUMB"
    lv.compose(project)
    assert lv.drawn == 2
    assert not lv.keys(ord("q"))


def test_live_view_draws_both_hands_and_h_switches_the_table():
    """Both tracked hands are drawn; the h key moves the table to the other
    hand; --hand left limits the view to that hand."""
    lv = view.LiveView()
    stream = MockLeapStream(pose="open", seed=2, dropout_every=0,
                            reacquire_every=0)
    for side, lh in stream.generate(2):
        lv.add_camera(lh)
    assert set(lv.camera_hands()) == {"left", "right"}
    project = view.Projector(None)
    first = lv.camera_hand().hand_side
    img_one = lv.compose(project)
    assert lv.keys(ord("h"))
    second = lv.camera_hand().hand_side
    assert {first, second} == {"left", "right"}
    img_two = lv.compose(project)
    # the bright (table) hand changed, so the picture differs
    assert (img_one != img_two).any()
    # with the table on one hand, both skeletons are still in the picture:
    # a hand drawn dimmer still carries its own colour somewhere
    dim_left = tuple(int(v * 0.6) for v in view.LIVE_HAND_BGR["left"])
    dim_right = tuple(int(v * 0.6) for v in view.LIVE_HAND_BGR["right"])
    picture = img_two[:, :view.IMAGE_SIZE]
    dim = dim_left if second == "right" else dim_right
    assert (picture == dim).all(axis=2).any()
    only = view.LiveView(prefer="left")
    for side, lh in stream.generate(2):
        only.add_camera(lh)
    assert set(only.camera_hands()) == {"left"}
    assert only.camera_hand().hand_side == "left"
    assert only.keys(ord("h")) and only.camera_hand().hand_side == "left"
