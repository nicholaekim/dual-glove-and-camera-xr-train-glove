"""leap_hand.angle_coverage: the joint-by-movement table and all_angles.

The synthetic hand is built here in the real (LeapC and glove) convention:
every child at a fixed offset along -z of its parent, its orientation the
parent's turned by `joint_frames.rotation_from_angles(flex, abd, twist)`, so
each joint's flexion, abduction and twist are known numbers the module must
give back. The whole hand is turned and moved in the world, so nothing
depends on the camera's axes.
"""
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from leap_hand import angle_coverage as ac
from leap_hand import paper3 as p3
from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES
from xr_hand.kinematics import PARENT, mat3_to_quat

REPO = Path(__file__).resolve().parents[1]
SIX = ("T_TMC_fe", "T_TMC_aa", "I_CMC", "M_CMC", "R_CMC", "L_CMC")
LETTERS = (("I", "INDEX"), ("M", "MIDDLE"), ("R", "RING"), ("L", "LITTLE"))
NON_ROOT = [n for n in JOINT_NAMES if n not in ("WRIST", "PALM")]

# (flex, abd, twist) in degrees of each joint against its parent
INJECTED = {"PALM": (20.0, 10.0, 5.0),
            "THUMB_METACARPAL": (30.0, 40.0, -60.0),
            "THUMB_PROXIMAL": (12.0, 6.0, 2.5),
            "THUMB_DISTAL": (25.0, -4.0, 1.0),
            "THUMB_TIP": (0.0, 0.0, 3.0)}
for _k, (_l, _f) in enumerate(LETTERS):
    INJECTED.update({f"{_f}_METACARPAL": (4.0 + _k, 2.0 - _k, 1.0 + _k),
                     f"{_f}_PROXIMAL": (35.0 + 7 * _k, 8.0 - 5 * _k, 3.0 - _k),
                     f"{_f}_INTERMEDIATE": (50.0 + 4 * _k, 4.0 - 2 * _k, -2.0),
                     f"{_f}_DISTAL": (25.0 + _k, -3.0 + _k, 1.5),
                     f"{_f}_TIP": (0.0, 0.0, -1.0 - _k)})
assert set(INJECTED) == set(JOINT_NAMES) - {"WRIST"}
SPREAD_M = {"THUMB": -0.030, "INDEX": -0.020, "MIDDLE": 0.0, "RING": 0.018,
            "LITTLE": 0.034}
BONE_M = {"PROXIMAL": 0.045, "INTERMEDIATE": 0.040, "DISTAL": 0.025, "TIP": 0.020}


def hand(side):
    """World (pos (26, 3) m, rot (26, 3, 3)) with INJECTED at every joint."""
    pos, rot = np.zeros((26, 3)), np.zeros((26, 3, 3))
    w = JOINT_INDEX["WRIST"]
    pos[w] = (0.05, 0.21, -0.03)
    rot[w] = jf.rot_z(25.0) @ jf.rot_x(-30.0) @ jf.rot_y(15.0)
    for i in [JOINT_INDEX["PALM"]] + list(range(2, 26)):        # parents first
        name = JOINT_NAMES[i]
        digit, _, part = name.partition("_")
        off = ((0.0, 0.0, -0.045) if name == "PALM" else
               (SPREAD_M[digit], 0.0, -0.010) if part == "METACARPAL" else
               (0.0, 0.0, -BONE_M[part]))
        par = PARENT[i]
        rot[i] = rot[par] @ jf.rotation_from_angles(*INJECTED[name], side)
        pos[i] = pos[par] + rot[par] @ np.asarray(off)
    return pos, rot


def line_of(pos, rot, side):
    """The same pose as an absolute frame line (what joint_table reads)."""
    return {"hand_side": side, "units": "m", "abs26": pos.tolist(),
            "quat26": [list(mat3_to_quat(r)) for r in rot]}


@pytest.fixture(params=["left", "right"])
def pose(request):
    side = request.param
    pos, rot = hand(side)
    return side, pos, rot, ac.all_angles(pos, rot, side)


def short(movement):
    return {"flexion/extension": "fe", "flexion/extension (palm arc)": "fe",
            "abduction/adduction": "aa", "radial/ulnar deviation": "dev",
            "twist about the bone axis": "twist"}[movement]


# --- the table -----------------------------------------------------------------------
def test_coverage_holds_every_joint_movement_pair_of_the_plan():
    """thumb 3 joints x 2 + four fingers x 4 joints x 2 + wrist 2 + twist 1 =
    41 pairs, plus the two estimated thumb TMC entries: 43."""
    expected = [("thumb TMC", "fe", "unavailable", ()),
                ("thumb TMC", "fe", "estimated", ("T_TMC_fe_est_openhand",)),
                ("thumb TMC", "aa", "unavailable", ()),
                ("thumb TMC", "aa", "estimated", ("T_TMC_aa_est_openhand",)),
                ("thumb MCP", "fe", "calculated", ("T_MCP_fe",)),
                ("thumb MCP", "aa", "outside_paper3", ("T_MCP_aa_tracker",)),
                ("thumb IP", "fe", "calculated", ("T_IP",)),
                ("thumb IP", "aa", "outside_paper3", ("T_IP_aa_tracker",))]
    for letter, finger in LETTERS:
        f = finger.lower()
        expected += [
            (f"{f} CMC", "fe", "unavailable", ()),
            (f"{f} CMC", "aa", "unavailable", ()),
            (f"{f} MCP", "fe", "calculated",
             (f"{letter}_MCP_fe", f"{letter}_MCP_fe_paper_order")),
            (f"{f} MCP", "aa", "calculated",
             (f"{letter}_MCP_aa", f"{letter}_MCP_aa_paper_order")),
            (f"{f} PIP", "fe", "calculated", (f"{letter}_PIP",)),
            (f"{f} PIP", "aa", "outside_paper3", (f"{letter}_PIP_aa_tracker",)),
            (f"{f} DIP", "fe", "calculated", (f"{letter}_DIP",)),
            (f"{f} DIP", "aa", "outside_paper3", (f"{letter}_DIP_aa_tracker",))]
    expected += [("wrist", "fe", "outside_paper3", ("WRIST_fe_tracker",)),
                 ("wrist", "dev", "outside_paper3", ("WRIST_dev_tracker",))]
    got = [(e.joint, short(e.movement), e.status, e.output_columns)
           for e in ac.COVERAGE]
    assert got[:-1] == expected
    twist = ac.COVERAGE[-1]
    assert (short(twist.movement), twist.status) == ("twist", "outside_paper3")
    assert twist.output_columns == tuple(f"{n}_twist_tracker" for n in NON_ROOT)
    assert len(twist.output_columns) == 24 and twist.method == (
        "xyz third angle (joint_frames.joint_angles twist_deg)")
    assert len(ac.COVERAGE) == 43 == len(expected) + 1
    assert all(e.status in ac.STATUS for e in ac.COVERAGE)


def test_every_entry_names_its_frame_zero_sign_and_method():
    for e in ac.COVERAGE:
        for field in ("reference_frame", "zero_position", "sign_convention",
                      "method"):
            assert getattr(e, field), (e.joint, e.movement, field)
        if e.status == "unavailable":
            assert e.output_columns == () and e.requirement, (e.joint, e.movement)
        if e.status in ("calculated", "outside_paper3"):
            assert e.output_columns
    signs = {e.sign_convention for e in ac.COVERAGE}
    assert "flexion positive toward the palm (both hands)" in signs
    assert "abduction positive toward the thumb side, radial (both hands)" in signs


def test_unavailable_paper3_names_carry_the_plans_requirements():
    by_name = {e.paper3_name: e for e in ac.COVERAGE
               if e.status == "unavailable" and e.paper3_name}
    assert tuple(by_name) == SIX == ac.UNAVAILABLE_COLUMNS
    for name in ("T_TMC_fe", "T_TMC_aa"):
        assert "Cheze et al. 2008" in by_name[name].requirement
        assert "Cyberglove" in by_name[name].requirement
    for name in SIX[2:]:
        assert "metacarpal against the carpus" in by_name[name].requirement
        assert "rigid palm" in by_name[name].requirement
    cmc_aa = [e for e in ac.COVERAGE if e.joint.endswith("CMC")
              and short(e.movement) == "aa"]
    assert len(cmc_aa) == 4
    for e in cmc_aa:
        assert e.status == "unavailable" and e.paper3_name is None
        assert e.note == "outside Paper 3's model and not measured: rigid palm"


def test_the_estimate_is_kept_apart_from_the_validated_coordinates():
    est = [e for e in ac.COVERAGE if e.status == "estimated"]
    assert [e.output_columns for e in est] == [("T_TMC_fe_est_openhand",),
                                               ("T_TMC_aa_est_openhand",)]
    assert ac.ESTIMATE_COLUMNS == ("T_TMC_fe_est_openhand", "T_TMC_aa_est_openhand")
    for e in est:
        assert e.paper3_name is None and e.dataset_column is None
        assert e.reference_frame == "palm frame via the session's open-hand rest frame"
        assert e.zero_position == "the session's open-hand pose"
        assert e.method.startswith("paper_order on R_rest^T (R_palm^T R_metacarpal)")
        assert "NOT a validated Paper 3 coordinate" in e.note
        assert "unknown constant" in e.note
    assert not set(ac.ESTIMATE_COLUMNS) & set(ac.ALL_ANGLE_COLUMNS)


# --- all_angles on a synthetic hand ----------------------------------------------------
def test_every_calculated_and_outside_column_is_a_finite_number(pose):
    _, _, _, a = pose
    assert list(a) == ac.ALL_ANGLE_COLUMNS
    for e in ac.COVERAGE:
        if e.status in ("calculated", "outside_paper3"):
            for c in e.output_columns:
                assert isinstance(a[c], float) and math.isfinite(a[c]), c


def test_the_six_unavailable_are_none_never_zero(pose):
    _, _, _, a = pose
    for name in SIX:
        assert a[name] is None, name
    assert [c for c, v in a.items() if v is None] == list(SIX)


def test_paper3_columns_are_joint_frames_on_the_same_pose(pose):
    side, pos, rot, a = pose
    line = line_of(pos, rot, side)
    d24 = jf.dof24(jf.joint_table(line, "camera"))
    paper = jf.mcp_paper_order(line, "camera")
    for name in jf.DOF24_NAMES:
        if name not in SIX:
            assert a[name] == pytest.approx(d24[name], abs=1e-9), name
    for name in jf.MCP_PAPER_NAMES:
        assert a[name] == pytest.approx(paper[name], abs=1e-9), name
    for letter, _ in LETTERS:
        assert a[f"{letter}_MCP_fe"] == pytest.approx(d24[f"{letter}_MCP_fe"], abs=1e-9)
        assert a[f"{letter}_MCP_fe_paper_order"] == pytest.approx(
            paper[f"{letter}_MCP_fe_paper_order"], abs=1e-9)
    frame = p3.frame_angles(pos, rot, side)
    assert {k: a[k] for k in frame} == frame                 # exactly paper3's numbers


def test_angles_give_back_the_injected_rotations(pose):
    """Independent of the table code: the numbers the hand was built with."""
    side, _, _, a = pose
    inj = INJECTED
    assert a["T_MCP_fe"] == pytest.approx(inj["THUMB_PROXIMAL"][0], abs=1e-9)
    assert a["T_MCP_aa_tracker"] == pytest.approx(inj["THUMB_PROXIMAL"][1], abs=1e-9)
    assert a["T_IP"] == pytest.approx(inj["THUMB_DISTAL"][0], abs=1e-9)
    assert a["T_IP_aa_tracker"] == pytest.approx(inj["THUMB_DISTAL"][1], abs=1e-9)
    for letter, finger in LETTERS:
        fe, aa, _ = inj[f"{finger}_PROXIMAL"]
        assert a[f"{letter}_MCP_fe"] == pytest.approx(fe, abs=1e-9)
        assert a[f"{letter}_MCP_aa"] == pytest.approx(aa, abs=1e-9)
        # the paper's order reads the same bone: sin(fe_p) = sin(fe) cos(aa),
        # tan(aa_p) = tan(aa) / cos(fe) (joint_frames, WHERE THIS IS NOT ...)
        fe_p, aa_p = a[f"{letter}_MCP_fe_paper_order"], a[f"{letter}_MCP_aa_paper_order"]
        r = math.radians
        assert math.sin(r(fe_p)) == pytest.approx(math.sin(r(fe)) * math.cos(r(aa)),
                                                  abs=1e-12)
        assert math.tan(r(aa_p)) == pytest.approx(math.tan(r(aa)) / math.cos(r(fe)),
                                                  abs=1e-12)
        assert abs(aa_p - aa) > 0.1 or aa == 0.0           # two different numbers
        for part, col in (("INTERMEDIATE", "PIP"), ("DISTAL", "DIP")):
            assert a[f"{letter}_{col}"] == pytest.approx(inj[f"{finger}_{part}"][0],
                                                         abs=1e-9)
            assert a[f"{letter}_{col}_aa_tracker"] == pytest.approx(
                inj[f"{finger}_{part}"][1], abs=1e-9)
    assert a["WRIST_fe_tracker"] == pytest.approx(inj["PALM"][0], abs=1e-9)
    assert a["WRIST_dev_tracker"] == pytest.approx(inj["PALM"][1], abs=1e-9)
    for joint in NON_ROOT:
        assert a[f"{joint}_twist_tracker"] == pytest.approx(inj[joint][2], abs=1e-9)


def test_tracker_columns_are_the_rows_of_the_joint_table(pose):
    side, pos, rot, a = pose
    rows = {r["joint"]: r for r in jf.table_from_world(pos, rot, side, "wrist")}
    assert a["T_MCP_aa_tracker"] == rows["THUMB_PROXIMAL"]["abd_deg"]
    assert a["T_IP_aa_tracker"] == rows["THUMB_DISTAL"]["abd_deg"]
    for letter, finger in LETTERS:
        assert a[f"{letter}_PIP_aa_tracker"] == rows[f"{finger}_INTERMEDIATE"]["abd_deg"]
        assert a[f"{letter}_DIP_aa_tracker"] == rows[f"{finger}_DISTAL"]["abd_deg"]
    assert a["WRIST_fe_tracker"] == rows["PALM"]["flex_deg"]
    assert a["WRIST_dev_tracker"] == rows["PALM"]["abd_deg"]
    for joint in NON_ROOT:
        assert a[f"{joint}_twist_tracker"] == rows[joint]["twist_deg"]


def test_all_angles_refuses_a_wrong_shape():
    pos, rot = hand("left")
    with pytest.raises(ValueError):
        ac.all_angles(pos[:25], rot, "left")
    with pytest.raises(ValueError):
        ac.all_angles(pos, rot, "both")


# --- CSV, status and the doc --------------------------------------------------------------
def test_write_coverage_csv_has_one_row_per_entry(tmp_path):
    path = ac.write_coverage_csv(tmp_path / "sub" / "angle_coverage.csv")
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == ac.COVERAGE_CSV_COLUMNS == [
        "joint", "movement", "paper3_name", "status", "output_columns",
        "reference_frame", "zero_position", "sign_convention", "method", "note",
        "requirement", "dataset_column"]
    assert len(rows) == 43 == len(ac.coverage_rows())
    by = [dict(zip(header, r)) for r in rows]
    mcp = next(r for r in by if r["paper3_name"] == "I_MCP_fe")
    assert mcp["output_columns"] == "I_MCP_fe; I_MCP_fe_paper_order"
    assert mcp["status"] == "calculated"
    tmc = next(r for r in by if r["paper3_name"] == "T_TMC_aa")
    assert tmc["output_columns"] == "" and tmc["status"] == "unavailable"
    assert tmc["requirement"]
    assert by[-1]["output_columns"].count("_twist_tracker") == 24


def test_status_of_maps_every_column():
    for c in ac.ALL_ANGLE_COLUMNS:
        expected = ("unavailable" if c in SIX else
                    "outside_paper3" if c.endswith("_tracker") else "calculated")
        assert ac.status_of(c) == expected, c
    for c in ac.ESTIMATE_COLUMNS:
        assert ac.status_of(c) == "estimated"
    assert ac.status_of("I_MCP_aa_paper_order") == "calculated"
    with pytest.raises(KeyError):
        ac.status_of("I_MCP_twist")


def test_the_doc_holds_the_table_the_code_renders():
    doc = (REPO / "docs" / "angle_coverage.md").read_text(encoding="utf-8")
    assert ac.coverage_markdown() in doc
    assert ac.coverage_markdown().count("\n") == 43 + 1         # header, rule, 43 rows
