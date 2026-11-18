"""leap_hand.paper3: the Paper 3 (Cobos et al. 2009) relationship layer.

Every expectation is derived without the module under test: coefficients are
read off the paper (Table 6 p.15, Table 7 p.16, eq. 3-8 pp.13-14),
statistics are worked by hand, D-H chains are composed here from elementary
rotations and translations, and synthetic hands are built in joint_frames'
convention (each child along -z of its parent, local rotations from
jf.rotation_from_angles).
"""
import ast
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from leap_hand import paper3 as p3
from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES
from xr_hand.kinematics import PARENT, mat3_to_quat

DATASET_GRASPS = """
p1_cylindrical p1_hook p1_lateral p1_palmar p1_spherical p1_tip
p2_s01_power_grasp p2_s02_power_grasp_ball p2_s03_power_grasp_finger_side
p2_s04_fingertip_grasp p2_s05_fingertip_grasp_side_support p2_s06_tripod_grasp
p2_s07_directional_power_side p2_s08_directional_power_tips
p2_s09_directional_power_side_and_tips p2_s10_power_grasp_dexterous
p3_circular_power p3_circular_precision_thumb_1_finger
p3_circular_precision_thumb_2_fingers p3_circular_precision_thumb_4_fingers
p3_prismatic_power_heavy_wrap p3_prismatic_power_medium_wrap
p3_prismatic_precision
""".split()

SIX = {"T_TMC_fe", "T_TMC_aa", "I_CMC", "M_CMC", "R_CMC", "L_CMC"}
FINGERS = "IMRL"
LENGTHS = {"I": (45.0, 40.0, 25.0, 20.0), "M": (44.0, 45.0, 28.0, 22.0),
           "R": (42.0, 42.0, 27.0, 21.0), "L": (38.0, 33.0, 20.0, 19.0)}

T6_UNAVAILABLE = {"T6:T_TMC_fe:T_MCP_fe", "T6:I_CMC:M_CMC", "T6:M_CMC:R_CMC",
                  "T6:R_CMC:L_CMC"}
T6_REPLACED_FOR_PRISMATIC = {"T6:T_MCP_fe:T_IP", "T6:I_MCP_fe:I_PIP",
                             "T6:I_PIP:I_DIP", "T6:M_MCP_fe:M_PIP",
                             "T6:M_PIP:M_DIP"}
T6_KEPT_FOR_PRISMATIC = {"T6:R_MCP_fe:R_PIP", "T6:R_PIP:R_DIP",
                         "T6:L_MCP_fe:L_PIP", "T6:L_PIP:L_DIP",
                         "T6:M_MCP_aa:I_MCP_aa", "T6:R_MCP_aa:L_MCP_aa"}
T7 = {"T7:T_MCP_fe:T_IP", "T7:I_MCP_fe:I_PIP", "T7:M_MCP_fe:M_PIP",
      "T7:I_PIP:I_DIP", "T7:M_PIP:M_DIP"}
EQ8 = {f"EQ8:{f}_PIP:{f}_DIP" for f in FINGERS}
CONDITIONAL = {"EQ3:R_MCP_fe:M_MCP_fe", "EQ4:R_MCP_fe:L_MCP_fe",
               "EQ5a:R_MCP_fe:M_MCP_fe", "EQ5b:R_MCP_fe:L_MCP_fe",
               "EQ7:M_MCP_fe:I_MCP_fe"}
EQ6 = "EQ6:R_MCP_aa:L_MCP_aa"


# --- synthetic hands and angle series -------------------------------------------------
BONE_M = {"PROXIMAL": 0.045, "INTERMEDIATE": 0.040, "DISTAL": 0.025,
          "TIP": 0.020}
SPREAD_M = {"THUMB": -0.030, "INDEX": -0.020, "MIDDLE": 0.0, "RING": 0.018,
            "LITTLE": 0.034}
POSED = {   # joint -> (flex, abd, twist) degrees against its parent
    "THUMB_METACARPAL": (20.0, 35.0, 40.0), "THUMB_PROXIMAL": (25.0, 3.0, 0.0),
    "THUMB_DISTAL": (30.0, 0.0, 0.0),
    "INDEX_METACARPAL": (4.0, 2.0, 0.0), "INDEX_PROXIMAL": (30.0, 10.0, 0.0),
    "INDEX_INTERMEDIATE": (45.0, 0.0, 0.0), "INDEX_DISTAL": (20.0, 0.0, 0.0),
    "MIDDLE_PROXIMAL": (60.0, -8.0, 2.0), "MIDDLE_INTERMEDIATE": (70.0, 0.0, 0.0),
    "MIDDLE_DISTAL": (35.0, 0.0, 0.0),
    "RING_PROXIMAL": (50.0, -12.0, 0.0), "RING_INTERMEDIATE": (65.0, 1.0, 0.0),
    "RING_DISTAL": (30.0, 0.0, 0.0),
    "LITTLE_PROXIMAL": (40.0, -15.0, 0.0), "LITTLE_INTERMEDIATE": (55.0, 0.0, 0.0),
    "LITTLE_DISTAL": (25.0, 0.0, 0.0),
}
WRIST_ROT = jf.rot_z(25.0) @ jf.rot_x(-30.0) @ jf.rot_y(15.0)


def build_pose(hand="right", local=None, wrist_rot=None):
    """(pos (26,3) m, rot (26,3,3)) chained the way the recorder's files are:
    each joint sits at a fixed offset in its parent's frame (finger bones
    along -z, metacarpal bases spread along x) and turns by
    `local[name]` = (flex, abd, twist) through jf.rotation_from_angles."""
    local = local or {}
    pos = np.zeros((26, 3))
    rot = np.zeros((26, 3, 3))
    w = JOINT_INDEX["WRIST"]
    pos[w] = (0.03, 0.18, -0.05)
    rot[w] = np.eye(3) if wrist_rot is None else wrist_rot
    for i in [JOINT_INDEX["PALM"]] + list(range(2, 26)):
        name = JOINT_NAMES[i]
        digit, _, part = name.partition("_")
        if name == "PALM":
            off = (0.0, 0.0, -0.045)
        elif part == "METACARPAL":
            off = (SPREAD_M[digit], 0.0, -0.010)
        else:
            off = (0.0, 0.0, -BONE_M[part])
        par = PARENT[i]
        local_rot = jf.rotation_from_angles(*local.get(name, (0.0, 0.0, 0.0)),
                                            hand)
        rot[i] = rot[par] @ local_rot
        pos[i] = pos[par] + rot[par] @ np.asarray(off)
    return pos, rot


def table6_series(n=8, seed=3):
    """Angle series (the dict series_angles returns) that satisfies Table 6
    exactly: DIP drawn, PIP = 3/2 DIP, MCP_fe = 4/3 PIP, M_aa = I_aa / 5,
    R_aa = L_aa / 2, T_MCP_fe = 4/5 T_IP; NaN for the six; paper-order MCP
    values equal to the main-order ones."""
    rng = np.random.default_rng(seed)
    a = {}
    for f in FINGERS:
        a[f"{f}_DIP"] = rng.uniform(5.0, 50.0, n)
        a[f"{f}_PIP"] = 1.5 * a[f"{f}_DIP"]
        a[f"{f}_MCP_fe"] = 4.0 / 3.0 * a[f"{f}_PIP"]
    a["I_MCP_aa"] = rng.uniform(-10.0, 15.0, n)
    a["L_MCP_aa"] = rng.uniform(-5.0, 20.0, n)
    a["M_MCP_aa"] = a["I_MCP_aa"] / 5.0
    a["R_MCP_aa"] = a["L_MCP_aa"] / 2.0
    a["T_IP"] = rng.uniform(5.0, 40.0, n)
    a["T_MCP_fe"] = 0.8 * a["T_IP"]
    for name in SIX:
        a[name] = np.full(n, np.nan)
    for f in FINGERS:
        for k in ("fe", "aa"):
            a[f"{f}_MCP_{k}_paper_order"] = a[f"{f}_MCP_{k}"].copy()
    return a


def statuses(result):
    return {row["id"]: row["status"] for row in result["relations"]}


def rows_by_id(result):
    return {row["id"]: row for row in result["relations"]}


# --- elementary D-H, written independently of the module -------------------------------
def rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def tx(a):
    m = np.eye(4)
    m[0, 3] = a
    return m


def link(theta_deg, a, alpha):
    """One D-H link: turn theta about z, move a along the new x, twist alpha
    about that x (d = 0, Table 2 p.5)."""
    return rz(math.radians(theta_deg)) @ tx(a) @ rx(alpha)


def chain_tip(links):
    t = np.eye(4)
    for spec in links:
        t = t @ link(*spec)
    return t[:3, 3]


# --- A. class mapping -------------------------------------------------------------------
def test_grasp_class_maps_the_23_dataset_names():
    assert len(DATASET_GRASPS) == 23
    classes = {g: p3.grasp_class(g) for g in DATASET_GRASPS}
    by = {c: sorted(g for g, (cls, _) in classes.items() if cls == c)
          for c in ("circular", "prismatic", "unmapped")}
    assert by["circular"] == ["p3_circular_power",
                              "p3_circular_precision_thumb_1_finger",
                              "p3_circular_precision_thumb_2_fingers",
                              "p3_circular_precision_thumb_4_fingers"]
    assert by["prismatic"] == ["p3_prismatic_power_heavy_wrap",
                               "p3_prismatic_power_medium_wrap",
                               "p3_prismatic_precision"]
    assert len(by["unmapped"]) == 16
    for g in by["unmapped"]:
        reason = classes[g][1]
        assert f"Paper {g[1]}" in reason and "Cutkosky" in reason
    assert p3.grasp_class("fist")[0] == "unmapped"


# --- B. the relations -------------------------------------------------------------------
def test_table6_and_table7_coefficients_are_the_printed_fractions():
    c = p3.coefficient
    for f in FINGERS:
        assert c(f"T6:{f}_MCP_fe:{f}_PIP") == Fraction(4, 3)     # Table 6
        assert c(f"T6:{f}_PIP:{f}_DIP") == Fraction(3, 2)        # Table 6
        assert c(f"EQ8:{f}_PIP:{f}_DIP") == Fraction(3, 2)       # eq. 8
    for f in "IM":
        assert c(f"T7:{f}_MCP_fe:{f}_PIP") == Fraction(3, 2)     # Table 7
        assert c(f"T7:{f}_PIP:{f}_DIP") == Fraction(2)           # Table 7
    assert c("T6:T_MCP_fe:T_IP") == Fraction(4, 5)
    assert float(c("T6:T_MCP_fe:T_IP")) == 0.8
    assert c("T7:T_MCP_fe:T_IP") == Fraction(6, 5)
    assert float(c("T7:T_MCP_fe:T_IP")) == 1.2
    assert c("T6:T_TMC_fe:T_MCP_fe") == Fraction(11, 10)
    assert c("T6:I_CMC:M_CMC") == 1
    assert c("T6:M_CMC:R_CMC") == Fraction(1, 2)
    assert c("T6:R_CMC:L_CMC") == Fraction(2, 3)
    assert c("T6:M_MCP_aa:I_MCP_aa") == Fraction(1, 5)
    assert c("T6:R_MCP_aa:L_MCP_aa") == Fraction(1, 2)
    assert c("EQ3:R_MCP_fe:M_MCP_fe") == Fraction(2, 3)
    assert c("EQ4:R_MCP_fe:L_MCP_fe") == Fraction(7, 12)
    assert c(EQ6) == 1
    assert c("EQ7:M_MCP_fe:I_MCP_fe") == Fraction(1, 5)
    bounds = {r.id: r.bound_deg for r in p3.RELATIONS if r.kind == "bound"}
    assert bounds == {"EQ5a:R_MCP_fe:M_MCP_fe": 60.0,
                      "EQ5b:R_MCP_fe:L_MCP_fe": 50.0}
    assert all(isinstance(r.coeff, Fraction) for r in p3.RELATIONS)


def test_relations_cite_their_pages_and_use_the_24_names():
    by_source = {}
    for r in p3.RELATIONS:
        by_source.setdefault(r.source, []).append(r)
        assert r.lhs in jf.DOF24_NAMES and r.rhs in jf.DOF24_NAMES
        assert p3.RELATION_BY_ID[r.id] is r
    assert len(p3.RELATION_BY_ID) == len(p3.RELATIONS) == 30
    assert {s: len(v) for s, v in by_source.items()} == {
        "Table 6": 15, "Table 7": 5, "Eq. 8 (general)": 4, "Eq. 3": 1,
        "Eq. 4": 1, "Eq. 5": 2, "Eq. 6": 1, "Eq. 7": 1}
    assert {s: {r.page for r in v} for s, v in by_source.items()} == {
        "Table 6": {15}, "Table 7": {16}, "Eq. 8 (general)": {14},
        "Eq. 3": {13}, "Eq. 4": {13}, "Eq. 5": {13}, "Eq. 6": {13},
        "Eq. 7": {13}}
    for r in by_source["Table 6"]:
        if r.lhs.endswith("_PIP"):
            assert "eq. 8" in r.note.lower()
    for r in by_source["Eq. 8 (general)"]:
        assert r.applies_to == frozenset({"unmapped", "circular"})
        assert "the most accepted intra-finger constraint" in r.note


def test_table5_is_transcribed_for_display():
    t5 = p3.TABLE5
    assert set(t5) == set(jf.DOF24_NAMES)
    assert t5["I_PIP"]["flexion_range"] == "110°"
    assert t5["I_PIP"]["hyperextension"] == "0°"
    assert t5["I_MCP_aa"]["abduction_range"] == "60°"
    assert t5["I_MCP_aa"]["flexion_range"] is None
    assert t5["I_MCP_fe"]["flexion_range"] == "90°"
    assert t5["I_MCP_fe"]["hyperextension"] == "30°–40°"
    assert t5["T_TMC_aa"]["abduction_range"] == "45°–60°"
    assert t5["T_IP"]["hyperextension"] == "5°–10°"
    assert t5["R_PIP"]["flexion_range"] == "120°"
    assert t5["L_PIP"]["flexion_range"] == "135°"
    assert t5["L_CMC"]["flexion_range"] == "15°"
    assert all(row["page"] == 12 for row in t5.values())
    assert "display" in p3.TABLE5_NOTE


def test_paper3_does_not_import_replay_core():
    tree = ast.parse(Path(p3.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {node.module or ""} | {a.name for a in node.names}
    assert not any("replay_core" in name for name in imported)


# --- C. angles --------------------------------------------------------------------------
@pytest.mark.parametrize("hand", ["right", "left"])
def test_frame_angles_match_joint_frames_on_a_frame_line(hand):
    pos, rot = build_pose(hand, POSED, WRIST_ROT)
    got = p3.frame_angles(pos, rot, hand)
    line = {"hand_side": hand, "units": "m", "abs26": pos.tolist(),
            "quat26": [list(mat3_to_quat(r)) for r in rot]}
    want = jf.dof24(jf.joint_table(line, "camera"))      # wrist reference
    want_paper = jf.mcp_paper_order(line, "camera")
    assert set(got) == set(jf.DOF24_NAMES) | set(jf.MCP_PAPER_NAMES)
    for name in jf.DOF24_NAMES:
        if name in SIX:
            assert got[name] is None
        else:
            assert isinstance(got[name], float)
            assert got[name] == pytest.approx(want[name], abs=1e-9)
    for name in jf.MCP_PAPER_NAMES:
        assert got[name] == pytest.approx(want_paper[name], abs=1e-9)
    assert sum(v is None for v in got.values()) == 6


def test_frame_angles_read_back_the_built_rotations():
    pos, rot = build_pose("right", POSED, WRIST_ROT)
    a = p3.frame_angles(pos, rot, "right")
    assert a["I_MCP_fe"] == pytest.approx(30.0, abs=1e-9)
    assert a["I_MCP_aa"] == pytest.approx(10.0, abs=1e-9)
    assert a["I_PIP"] == pytest.approx(45.0, abs=1e-9)
    assert a["I_DIP"] == pytest.approx(20.0, abs=1e-9)
    assert a["M_MCP_aa"] == pytest.approx(-8.0, abs=1e-9)
    assert a["T_MCP_fe"] == pytest.approx(25.0, abs=1e-9)
    assert a["T_IP"] == pytest.approx(30.0, abs=1e-9)
    # abduction first: sin(fe_p) = sin(fe) cos(aa), tan(aa_p) = tan(aa) / cos(fe)
    built = {"I": (30.0, 10.0), "M": (60.0, -8.0), "R": (50.0, -12.0),
             "L": (40.0, -15.0)}
    for f, (fe, aa) in built.items():
        fe_r, aa_r = math.radians(fe), math.radians(aa)
        fe_p = math.degrees(math.asin(math.sin(fe_r) * math.cos(aa_r)))
        aa_p = math.degrees(math.atan(math.tan(aa_r) / math.cos(fe_r)))
        assert a[f"{f}_MCP_fe_paper_order"] == pytest.approx(fe_p, abs=1e-9)
        assert a[f"{f}_MCP_aa_paper_order"] == pytest.approx(aa_p, abs=1e-9)


def test_series_angles_gives_nan_for_the_six_and_numbers_elsewhere():
    poses = [build_pose("left", {**POSED,
                                 "INDEX_INTERMEDIATE": (k * 10.0, 0.0, 0.0)},
                        WRIST_ROT) for k in range(3)]
    s = p3.series_angles(poses, "left")
    assert set(s) == set(jf.DOF24_NAMES) | set(jf.MCP_PAPER_NAMES)
    for name, values in s.items():
        assert values.shape == (3,)
        assert (np.isnan(values).all() if name in SIX
                else np.isfinite(values).all())
    assert s["I_PIP"] == pytest.approx([0.0, 10.0, 20.0], abs=1e-9)


def test_a_relation_touching_an_unavailable_angle_has_no_numbers():
    assert p3.UNAVAILABLE == jf.CAMERA_NOT_GIVEN
    for cls in ("circular", "prismatic", "unmapped"):
        for row in p3.compare(table6_series(), cls)["relations"]:
            touches = row["lhs"] in SIX or row["rhs"] in SIX
            assert (row["status"] == "unavailable") == touches
            if touches:
                assert row["stats_all"] is None and row["stats_hold"] is None
    pos, rot = build_pose("right", POSED)
    values = {v["id"]: v for v in p3.frame_relation_values(
        p3.frame_angles(pos, rot, "right"), "circular")}
    assert values["T6:R_CMC:L_CMC"]["residual"] == "unavailable"
    assert values["T6:T_TMC_fe:T_MCP_fe"]["status"] == "unavailable"


def test_finger_bone_lengths_follow_the_table2_links():
    pos, _rot = build_pose("right", POSED, WRIST_ROT)
    lengths = p3.finger_bone_lengths_mm(pos)
    assert set(lengths) == set(FINGERS)
    for f in FINGERS:     # metacarpal, proximal, middle, distal (L_Me L_P L_Mi L_D)
        assert lengths[f] == pytest.approx((45.0, 40.0, 25.0, 20.0), abs=1e-9)


# --- D. residual statistics -------------------------------------------------------------
def test_residual_stats_against_hand_computed_values():
    rhs = np.array([0.0, 10.0, 20.0, 30.0, 40.0, np.nan, 7.0])
    offsets = np.array([5.0, 2.0, 1.0, 2.0, 5.0, 0.0, 0.0])
    lhs = 4.0 / 3.0 * rhs + offsets
    lhs[6] = np.nan                      # a pair with one side missing is dropped
    st = p3.residual_stats(lhs, rhs, Fraction(4, 3))
    # residuals 5 2 1 2 5, sorted 1 2 2 5 5
    assert st["n"] == 5
    assert st["mean"] == pytest.approx(3.0)
    assert st["mae"] == pytest.approx(3.0)
    assert st["rmse"] == pytest.approx(math.sqrt(59.0 / 5.0))
    # numpy's default linear percentile: position (n - 1) p in the sorted list
    assert st["p05"] == pytest.approx(1.2)      # 1 + 0.2 * (2 - 1)
    assert st["p50"] == pytest.approx(2.0)
    assert st["p95"] == pytest.approx(5.0)      # 5 + 0.8 * (5 - 5)
    assert st["max_abs"] == pytest.approx(5.0)
    # offsets - 3 = (2, -1, -2, -1, 2) is orthogonal to rhs - 20, so the OLS
    # slope is the coefficient and the intercept the mean offset
    assert st["slope"] == pytest.approx(4.0 / 3.0)
    assert st["intercept"] == pytest.approx(3.0)
    sxx, sxy, syy = 1000.0, 4.0 / 3.0 * 1000.0, 16.0 / 9.0 * 1000.0 + 14.0
    assert st["r"] == pytest.approx(sxy / math.sqrt(sxx * syy))


def test_bound_stats():
    lhs = np.array([70.0, 130.0, 100.0, 125.0, np.nan])
    rhs = np.array([60.0, 60.0, 50.0, 60.0, 10.0])
    # differences 10, 70, 50, 65: two of four within 60, the worst 10 over it
    assert p3.residual_stats(lhs, rhs, 1, bound=60.0) == {
        "n": 4, "bound_deg": 60.0, "fraction_satisfied": 0.5,
        "max_excess": 10.0}


def test_residual_stats_degenerate_inputs():
    two = p3.residual_stats([1.0, 2.0], [1.0, 2.0], 1)
    assert two["n"] == 2 and two["slope"] is None and two["r"] is None
    flat = p3.residual_stats([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], Fraction(3, 2))
    assert flat["n"] == 3 and flat["mean"] == pytest.approx(1.85)
    assert flat["slope"] is None and flat["intercept"] is None
    assert flat["r"] is None
    empty = p3.residual_stats([np.nan], [1.0], 1)
    assert empty["n"] == 0 and empty["mean"] is None and empty["p50"] is None


# --- E. compare per class ---------------------------------------------------------------
def test_circular_applies_table6_in_full():
    res = p3.compare(table6_series(), "circular")
    st = statuses(res)
    t6 = {i for i in st if i.startswith("T6:")}
    assert len(t6) == 15 and len(t6 - T6_UNAVAILABLE) == 11
    assert {i for i in t6 if st[i] == "applied"} == t6 - T6_UNAVAILABLE
    assert {i for i in t6 if st[i] == "unavailable"} == T6_UNAVAILABLE
    assert all(st[i] == "not_applied" for i in T7 | CONDITIONAL)
    assert all(st[i] == "applied" for i in EQ8)
    assert st[EQ6] == "informational"
    assert res["class"] == "circular"
    assert res["summary"] == {"applied": 15, "not_applied": 10,
                              "informational": 1, "unavailable": 4}
    for row in res["relations"]:        # the series satisfies Table 6 exactly
        if row["status"] == "applied":
            assert row["stats_all"]["max_abs"] < 1e-12


def test_prismatic_takes_table7_for_thumb_index_middle_and_table6_for_ring_little():
    a = table6_series()
    res = p3.compare(a, "prismatic")
    st = statuses(res)
    assert {i for i, s in st.items() if s == "applied"} == (
        T7 | T6_KEPT_FOR_PRISMATIC)
    assert all(st[i] == "not_applied" for i in T6_REPLACED_FOR_PRISMATIC)
    assert all(st[i] == "unavailable" for i in T6_UNAVAILABLE)
    assert all(st[i] == "not_applied" for i in EQ8 | CONDITIONAL)
    assert res["summary"] == {"applied": 11, "not_applied": 14,
                              "informational": 1, "unavailable": 4}
    row = rows_by_id(res)["T7:I_PIP:I_DIP"]     # 1.5 DIP - 2 DIP = -0.5 DIP
    assert row["stats_all"]["mean"] == pytest.approx(-0.5 * a["I_DIP"].mean(),
                                                     abs=1e-12)
    assert row["coeff"] == "2" and row["coeff_float"] == 2.0
    assert rows_by_id(res)["T6:I_MCP_fe:I_PIP"]["coeff"] == "4/3"


def test_unmapped_applies_only_eq8():
    res = p3.compare(table6_series(), "unmapped")
    st = statuses(res)
    assert {i for i, s in st.items() if s == "applied"} == EQ8
    assert st[EQ6] == "informational"
    assert res["summary"] == {"applied": 4, "not_applied": 21,
                              "informational": 1, "unavailable": 4}
    for row in res["relations"]:
        if row["id"] in EQ8:
            assert row["source"] == "Eq. 8 (general)" and row["page"] == 14
        elif row["status"] == "not_applied" and row["id"] not in CONDITIONAL:
            assert "unmapped" in row["reason"]


@pytest.mark.parametrize("cls", ["circular", "prismatic", "unmapped"])
def test_single_finger_equations_are_never_applied_and_eq6_is_informational(cls):
    a = table6_series()
    rows = rows_by_id(p3.compare(a, cls))
    for i in CONDITIONAL:
        rel = p3.RELATION_BY_ID[i]
        assert rel.applies_to == frozenset() and rel.condition
        assert rows[i]["status"] == "not_applied"
        assert rel.condition in rows[i]["reason"]
    assert p3.RELATION_BY_ID[EQ6].applies_to == frozenset({"informational"})
    assert rows[EQ6]["status"] == "informational"
    assert rows[EQ6]["stats_all"]["n"] == 8
    diff = a["R_MCP_fe"] - a["M_MCP_fe"]
    bound = rows["EQ5a:R_MCP_fe:M_MCP_fe"]["stats_all"]
    assert bound["fraction_satisfied"] == pytest.approx(np.mean(diff <= 60.0))


def test_mcp_relations_use_the_paper_order_pair_first_and_report_the_main_order():
    a = table6_series()
    a["I_MCP_fe_paper_order"] = a["I_MCP_fe"] + 2.0
    a["R_MCP_aa_paper_order"] = a["R_MCP_aa"] - 1.0
    res = p3.compare(a, "circular")
    rows = rows_by_id(res)
    mcp = rows["T6:I_MCP_fe:I_PIP"]
    assert mcp["mcp_order"] == "paper_order"
    assert mcp["stats_all"]["mean"] == pytest.approx(2.0)
    assert mcp["stats_all_main"]["mean"] == pytest.approx(0.0, abs=1e-12)
    aa = rows["T6:R_MCP_aa:L_MCP_aa"]
    assert aa["mcp_order"] == "paper_order"
    assert aa["stats_all"]["mean"] == pytest.approx(-1.0)
    assert rows["T6:I_PIP:I_DIP"]["mcp_order"] is None
    assert rows["T6:I_PIP:I_DIP"]["stats_all_main"] is None
    assert rows["T6:T_MCP_fe:T_IP"]["mcp_order"] is None   # thumb MCP: 1 DoF
    assert "paper" in res["note"] and "threshold" in res["note"]
    # without the paper-order values the main order is primary, and says so
    main_only = {k: v for k, v in a.items() if not k.endswith("_paper_order")}
    row = rows_by_id(p3.compare(main_only, "circular"))["T6:I_MCP_fe:I_PIP"]
    assert row["mcp_order"] == "main"
    assert row["stats_all"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert row["stats_all_main"] is None


def test_hold_mask_restricts_the_hold_statistics():
    a = table6_series(n=6)
    a["I_PIP"] = a["I_PIP"] + np.array([0.0, 0.0, 3.0, 3.0, 9.0, 9.0])
    hold = np.array([False, False, True, True, False, False])
    row = rows_by_id(p3.compare(a, "circular", hold_mask=hold))["T6:I_PIP:I_DIP"]
    assert row["stats_hold"]["n"] == 2
    assert row["stats_hold"]["mean"] == pytest.approx(3.0)
    assert row["stats_all"]["n"] == 6
    assert row["stats_all"]["mean"] == pytest.approx(4.0)
    assert rows_by_id(p3.compare(a, "circular"))["T6:I_PIP:I_DIP"][
        "stats_hold"] is None
    with pytest.raises(ValueError):
        p3.compare(a, "circular", hold_mask=hold[:3])


def test_unknown_class_is_rejected():
    with pytest.raises(ValueError):
        p3.compare(table6_series(), "spherical")
    with pytest.raises(ValueError):
        p3.frame_relation_values({}, "power")


def test_frame_relation_values_for_the_viewer_panel():
    a = table6_series(n=1)
    frame = {k: (None if np.isnan(v[0]) else float(v[0])) for k, v in a.items()}
    frame["I_PIP"] += 1.5
    vals = {v["id"]: v for v in p3.frame_relation_values(frame, "circular")}
    assert len(vals) == len(p3.RELATIONS)
    v = vals["T6:I_PIP:I_DIP"]
    assert v["status"] == "applied"
    assert v["lhs_value"] == pytest.approx(frame["I_PIP"])
    assert v["rhs_value"] == pytest.approx(frame["I_DIP"])
    assert v["residual"] == pytest.approx(1.5)
    assert vals["T6:I_CMC:M_CMC"]["residual"] == "unavailable"
    b = vals["EQ5a:R_MCP_fe:M_MCP_fe"]
    diff = frame["R_MCP_fe_paper_order"] - frame["M_MCP_fe_paper_order"]
    assert b["kind"] == "bound" and b["residual"] == pytest.approx(diff)
    assert b["satisfied"] == (diff <= 60.0)


def test_series_angles_feed_compare_and_the_error():
    poses = [build_pose("left", {**POSED, "RING_DISTAL": (10.0 + 5 * k, 0.0, 0.0)},
                        WRIST_ROT) for k in range(4)]
    angles = p3.series_angles(poses, "left")
    res = p3.compare(angles, "prismatic", hold_mask=[True, True, False, False])
    assert res["summary"]["unavailable"] == 4
    row = rows_by_id(res)["T6:R_PIP:R_DIP"]
    assert row["stats_all"]["n"] == 4 and row["stats_hold"]["n"] == 2
    err = p3.fingertip_reconstruction_error(
        angles, p3.finger_bone_lengths_mm(poses[0][0]), "prismatic")
    assert all(err["fingers"][f]["n"] == 4 for f in FINGERS)


# --- F. Denavit-Hartenberg --------------------------------------------------------------
@pytest.mark.parametrize("theta_deg,alpha", [
    (37.0, math.pi / 2), (37.0, -math.pi / 2), (37.0, 0.7), (-50.0, 0.7),
    (120.0, -1.1)])
def test_dh_is_rz_tx_rx_and_a_proper_rotation(theta_deg, alpha):
    m = p3.dh(math.radians(theta_deg), 0.03, alpha)
    assert np.allclose(m, link(theta_deg, 0.03, alpha), rtol=0, atol=1e-15)
    r = m[:3, :3]
    assert np.abs(r.T @ r - np.eye(3)).max() < 1e-12
    assert abs(np.linalg.det(r) - 1.0) < 1e-12


@pytest.mark.parametrize("theta_deg,alpha", [
    (37.0, math.pi / 2), (37.0, -math.pi / 2), (-50.0, 0.7), (120.0, -1.1)])
def test_dh_as_printed_is_not_a_rotation(theta_deg, alpha):
    r = p3.dh_as_printed(math.radians(theta_deg), 0.03, alpha)[:3, :3]
    assert np.abs(r.T @ r - np.eye(3)).max() > 0.1


def test_dh_as_printed_reproduces_page_5_literally():
    th, al = math.radians(37.0), 0.7
    printed, standard = p3.dh_as_printed(th, 0.03, al), p3.dh(th, 0.03, al)
    assert printed[0, 2] == pytest.approx(math.sin(al) * math.sin(al))
    assert printed[1, 2] == pytest.approx(-math.sin(al) * math.cos(al))
    others = np.ones((4, 4), bool)
    others[0, 2] = others[1, 2] = False
    assert np.array_equal(printed[others], standard[others])
    # the misprint puts alpha where theta belongs in column 3, so the two
    # agree when alpha = 0 and when theta = alpha (why 37 deg against
    # alpha = 0.7 rad, 0.646 rad apart, is a weak test point)
    assert np.array_equal(p3.dh_as_printed(th, 0.03, 0.0), p3.dh(th, 0.03, 0.0))
    assert np.allclose(p3.dh_as_printed(0.7, 0.03, 0.7), p3.dh(0.7, 0.03, 0.7),
                       rtol=0, atol=1e-15)


def test_finger_tip_with_all_angles_zero_lies_along_x():
    # joints 1 and 2 twist the frame +pi/2 then -pi/2 about x with a = 0
    # between them: no net turn, so the four links line up along x
    assert np.allclose(p3.finger_tip_position([0.0] * 5, LENGTHS["I"]),
                       [45.0 + 40.0 + 25.0 + 20.0, 0.0, 0.0], rtol=0, atol=1e-12)
    assert np.allclose(p3.thumb_tip_position([0.0] * 4, (40.0, 30.0, 25.0)),
                       [95.0, 0.0, 0.0], rtol=0, atol=1e-12)


def test_finger_tip_with_the_mcp_flexed_90_degrees():
    L = LENGTHS["I"]
    # by hand: joints 1-2 leave the frame unturned at (L_Me, 0, 0); joint 3
    # turns 90 degrees about z, so the three phalanges run along +y
    expected = np.array([L[0], L[1] + L[2] + L[3], 0.0])
    by_matrices = chain_tip([(0.0, L[0], math.pi / 2), (0.0, 0.0, -math.pi / 2),
                             (90.0, L[1], 0.0), (0.0, L[2], 0.0),
                             (0.0, L[3], 0.0)])
    assert np.allclose(by_matrices, expected, rtol=0, atol=1e-12)
    assert np.allclose(p3.finger_tip_position((0.0, 0.0, 90.0, 0.0, 0.0), L),
                       expected, rtol=0, atol=1e-12)


def test_tip_chains_match_the_elementary_composition():
    rng = np.random.default_rng(7)
    for _ in range(5):
        a5, l4 = rng.uniform(-30, 90, 5), rng.uniform(15, 50, 4)
        want = chain_tip([(a5[0], l4[0], math.pi / 2), (a5[1], 0.0, -math.pi / 2),
                          (a5[2], l4[1], 0.0), (a5[3], l4[2], 0.0),
                          (a5[4], l4[3], 0.0)])
        assert np.allclose(p3.finger_tip_position(a5, l4), want, rtol=0,
                           atol=1e-12)
        a4, l3 = rng.uniform(-30, 90, 4), rng.uniform(15, 50, 3)
        want = chain_tip([(a4[0], 0.0, math.pi / 2), (a4[1], l3[0], 0.0),
                          (a4[2], l3[1], 0.0), (a4[3], l3[2], 0.0)])
        assert np.allclose(p3.thumb_tip_position(a4, l3), want, rtol=0,
                           atol=1e-12)


def test_relative_error_does_not_depend_on_the_sign_conventions():
    # flipping every flexion mirrors both tips (y -> -y), flipping the
    # abduction mirrors them (z -> -z); eq. 9 is a ratio of lengths
    L = LENGTHS["M"]
    orig = np.array([0.0, 7.0, 50.0, 60.0, 30.0])
    rec = np.array([0.0, 1.4, 60.0, 45.0, 30.0])

    def ratio(o, r):
        op, rp = p3.finger_tip_position(o, L), p3.finger_tip_position(r, L)
        return np.linalg.norm(rp - op) / np.linalg.norm(rp)

    base = ratio(orig, rec)
    assert base > 0.01
    flex = np.array([-1.0, 1.0, -1.0, -1.0, -1.0])
    abd = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    assert ratio(orig * flex, rec * flex) == pytest.approx(base, rel=1e-12)
    assert ratio(orig * abd, rec * abd) == pytest.approx(base, rel=1e-12)


# --- G. the 9-DoF description and eq. 9-11 ---------------------------------------------
def test_reconstruct_9dof_derives_the_15_dependent_angles():
    frame = {"T_IP": 25.0, "I_DIP": 10.0, "M_DIP": 20.0, "R_DIP": 12.0,
             "L_DIP": 8.0, "I_MCP_aa": 10.0, "L_MCP_aa": 6.0,
             "T_TMC_aa": None, "L_CMC": None}
    circ = p3.reconstruct_9dof(frame, "circular")
    v = circ["values"]
    assert circ["status"] == "applied" and len(v) == 24
    assert v["T_MCP_fe"] == pytest.approx(20.0)        # 4/5 x 25
    assert v["T_TMC_fe"] == pytest.approx(22.0)        # 11/10 x 20
    assert v["I_PIP"] == pytest.approx(15.0)           # 3/2 x 10
    assert v["I_MCP_fe"] == pytest.approx(20.0)        # 4/3 x 15
    assert v["M_MCP_fe"] == pytest.approx(40.0)        # 4/3 x 3/2 x 20
    assert v["M_MCP_aa"] == pytest.approx(2.0)         # 1/5 x 10
    assert v["R_MCP_aa"] == pytest.approx(3.0)         # 1/2 x 6
    assert v["L_PIP"] == pytest.approx(12.0) and v["L_MCP_fe"] == pytest.approx(16.0)
    for name in ("T_TMC_aa", "L_CMC", "R_CMC", "M_CMC", "I_CMC"):
        assert v[name] is None
    assert circ["derived_by"]["I_PIP"] == "T6:I_PIP:I_DIP"
    assert circ["derived_by"]["L_DIP"] == "independent"
    pris = p3.reconstruct_9dof(frame, "prismatic")
    w = pris["values"]
    assert w["T_MCP_fe"] == pytest.approx(30.0)        # 6/5 x 25 (Table 7)
    assert w["I_PIP"] == pytest.approx(20.0)           # 2 x 10 (Table 7)
    assert w["I_MCP_fe"] == pytest.approx(30.0)        # 3/2 x 20 (Table 7)
    assert w["R_PIP"] == pytest.approx(18.0)           # Table 6 kept for ring
    assert w["R_MCP_fe"] == pytest.approx(24.0)
    assert pris["derived_by"]["M_PIP"] == "T7:M_PIP:M_DIP"
    assert p3.reconstruct_9dof(frame, "unmapped")["status"] == "not_applied"
    # the independent MCP abduction is read in the paper's order when given
    paper = p3.reconstruct_9dof({**frame, "I_MCP_aa_paper_order": 12.0},
                                "circular")["values"]
    assert paper["I_MCP_aa"] == 12.0 and paper["M_MCP_aa"] == pytest.approx(2.4)


def test_fingertip_error_uses_the_paper_order_mcp_pair():
    a = table6_series(n=6)
    a["I_MCP_fe_paper_order"] = a["I_MCP_fe"] + 5.0   # main order still fits
    res = p3.fingertip_reconstruction_error(a, LENGTHS, "circular")
    assert res["fingers"]["I"]["mean_pct"] > 0.1
    for f in "MRL":
        assert res["fingers"][f]["mean_pct"] < 1e-9


def test_fingertip_error_is_zero_when_table6_holds_exactly():
    res = p3.fingertip_reconstruction_error(table6_series(n=6), LENGTHS,
                                            "circular")
    assert res["status"] == "applied" and res["thumb"] == "unavailable"
    for f in FINGERS:
        assert res["fingers"][f]["n"] == 6
        assert res["fingers"][f]["mean_pct"] < 1e-9
    assert res["gesture_error_pct"] < 1e-9
    assert res["fingers_averaged"] == ["I", "M", "R", "L"]


def test_fingertip_error_under_prismatic_moves_index_and_middle_only():
    a = table6_series(n=6)
    res = p3.fingertip_reconstruction_error(a, LENGTHS, "prismatic")
    for f in "IM":
        assert res["fingers"][f]["mean_pct"] > 1.0
    for f in "RL":
        assert res["fingers"][f]["mean_pct"] < 1e-9
    assert res["thumb"] == "unavailable"
    # eq. 9 by hand for the index on frame 0, through this file's own chain
    dip, aa, L = a["I_DIP"][0], a["I_MCP_aa_paper_order"][0], LENGTHS["I"]

    def tip(fe, pip):
        return chain_tip([(0.0, L[0], math.pi / 2), (aa, 0.0, -math.pi / 2),
                          (fe, L[1], 0.0), (pip, L[2], 0.0), (dip, L[3], 0.0)])

    op = tip(2.0 * dip, 1.5 * dip)      # measured: the Table 6 shape
    rp = tip(3.0 * dip, 2.0 * dip)      # Table 7: PIP = 2 DIP, MCP = 3/2 PIP
    delta = np.linalg.norm(rp - op) / np.linalg.norm(rp) * 100.0
    assert res["per_frame_pct"]["I"][0] == pytest.approx(delta, rel=1e-9)
    assert res["fingers"]["I"]["mean_pct"] == pytest.approx(
        np.mean(res["per_frame_pct"]["I"]))                       # eq. 10
    assert res["gesture_error_pct"] == pytest.approx(               # eq. 11
        sum(res["fingers"][f]["mean_pct"] for f in FINGERS) / 4)


def test_fingertip_error_not_applied_to_unmapped():
    res = p3.fingertip_reconstruction_error(table6_series(), LENGTHS, "unmapped")
    assert res["status"] == "not_applied" and res["reason"]
