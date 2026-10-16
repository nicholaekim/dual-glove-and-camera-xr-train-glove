"""xr_hand.joint_key: the 26-row key of the joint-frame tables.

The paper's names are checked against `joint_frames.DOF24_SOURCE`, the
landmark labels against the text that `scripts/glove/export_prof_format.py`
(the writer of the hand-in's keypoint files) writes, read back with
`cam_hand.prof_format`; nothing the key says is copied into the test as a
second list of the same strings.
"""
import csv
import importlib.util
import re
from collections import Counter
from pathlib import Path

import pytest

from cam_hand import prof_format
from cam_hand.export21 import PROF_LABELS
from xr_hand import joint_key as jk
from xr_hand.joint_frames import (CAMERA_NOT_GIVEN, DOF24_NAMES,
                                  DOF24_SOURCE, PAPER_JOINT)
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES
from xr_hand.kinematics import PARENT
from xr_hand.mock import MockHandGenerator
from xr_hand.parser import parse_hand_message

REPO = Path(__file__).resolve().parents[1]
SESSION = (REPO / "recordings" / "protocol" / "grasps"
           / "20261001_101621_left")
COLUMN = {"flex_deg": "paper_angle_flex", "abd_deg": "paper_angle_abd"}


def exporter():
    """The writer of the keypoint text files, imported by path."""
    path = REPO / "scripts" / "glove" / "export_prof_format.py"
    spec = importlib.util.spec_from_file_location("glove_prof_format_k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def by_row():
    return {r["row"]: r for r in jk.key_rows()}


# --- the rows ----------------------------------------------------------------------
def test_26_rows_in_joint_names_order_with_the_key_columns():
    rows = jk.key_rows()
    assert len(rows) == 26 == len(JOINT_NAMES)
    assert [r["row"] for r in rows] == JOINT_NAMES
    assert [r["number"] for r in rows] == list(range(1, 27))
    for r in rows:
        assert list(r) == jk.KEY_COLUMNS
        for k in jk.KEY_COLUMNS:
            if k not in ("paper_angle_flex", "paper_angle_abd"):
                assert str(r[k]).strip(), (r["row"], k)


def test_each_of_the_24_paper_names_once_where_dof24_source_reads_it():
    rows = by_row()
    names = Counter(r[c] for r in rows.values() for c in COLUMN.values()
                    if r[c])
    assert set(names) == set(DOF24_NAMES)
    assert all(n == 1 for n in names.values()), names
    for name, (joint, column) in DOF24_SOURCE.items():
        assert rows[joint][COLUMN[column]] == name, name


def test_the_six_camera_does_not_give_say_not_given_and_why():
    rows = by_row()
    no_rows = {DOF24_SOURCE[n][0] for n in CAMERA_NOT_GIVEN}
    for name in CAMERA_NOT_GIVEN:
        joint = DOF24_SOURCE[name][0]
        assert rows[joint]["given"] == "no: not measured by this tracker"
        assert PAPER_JOINT[joint] in jk.NOT_GIVEN_WHY
    assert {n for n, r in rows.items()
            if r["given"] == "no: not measured by this tracker"} == no_rows
    assert set(jk.NOT_GIVEN_WHY) == {PAPER_JOINT[j] for j in no_rows}
    assert "rigid" in jk.NOT_GIVEN_WHY["CMC"]
    assert "bend of the wrist" in jk.NOT_GIVEN_WHY["CMC"]
    assert "hand's axes, not the thumb's" in jk.NOT_GIVEN_WHY["TMC"]


def test_given_counts():
    rows = jk.key_rows()
    per = Counter()
    names = Counter()
    for r in rows:
        per[r["given"]] += 1
        names[r["given"]] += sum(1 for c in COLUMN.values() if r[c])
    assert set(per) == {jk.GIVEN_YES, jk.GIVEN_NO, jk.NOT_OF_24}
    assert names[jk.GIVEN_YES] == len(DOF24_NAMES) - len(CAMERA_NOT_GIVEN) \
        == 18
    assert names[jk.GIVEN_NO] == len(CAMERA_NOT_GIVEN) == 6
    assert names[jk.NOT_OF_24] == 0
    assert (per[jk.GIVEN_YES], per[jk.GIVEN_NO], per[jk.NOT_OF_24]) \
        == (14, 5, 7)
    not_24 = {r["row"] for r in rows if r["given"] == jk.NOT_OF_24}
    assert not_24 == {"PALM", "WRIST"} | {n for n in JOINT_NAMES
                                          if n.endswith("_TIP")}


def test_rows_are_anatomically_right():
    rows = by_row()
    ip = rows["INDEX_PROXIMAL"]
    assert ip["anatomical_joint"] == "index MCP (metacarpophalangeal)"
    assert ip["bone_before"] == "index metacarpal"
    assert ip["bone_after"] == "index proximal phalanx"
    assert ip["flex_deg_means"] == \
        "bending of the proximal phalanx toward the palm"
    assert ip["abd_deg_means"] == "spread of the finger toward the thumb side"
    assert (ip["paper_angle_flex"], ip["paper_angle_abd"], ip["given"]) == \
        ("I_MCP_fe", "I_MCP_aa", "yes")
    for f in ("INDEX", "MIDDLE", "RING", "LITTLE"):
        digit = f.lower()
        mc = rows[f"{f}_METACARPAL"]
        assert mc["anatomical_joint"].startswith(f"{digit} CMC")
        assert mc["bone_before"] == "forearm (the WRIST joint's frame)"
        assert mc["bone_after"] == f"{digit} metacarpal"
        assert "seen from the forearm" in mc["flex_deg_means"]
        assert "seen from the forearm" in mc["abd_deg_means"]
        assert (mc["paper_angle_flex"], mc["paper_angle_abd"]) == \
            (f"{f[0]}_CMC", "")
        assert rows[f"{f}_PROXIMAL"]["anatomical_joint"].startswith(
            f"{digit} MCP")
        assert rows[f"{f}_INTERMEDIATE"]["anatomical_joint"].startswith(
            f"{digit} PIP")
        assert rows[f"{f}_DISTAL"]["anatomical_joint"].startswith(
            f"{digit} DIP")
        assert rows[f"{f}_DISTAL"]["bone_after"] == f"{digit} distal phalanx"
    tm = rows["THUMB_METACARPAL"]
    assert tm["anatomical_joint"].startswith("thumb TMC")
    assert tm["bone_before"] == "forearm (the WRIST joint's frame)"
    assert tm["bone_after"] == "thumb metacarpal"
    assert (tm["paper_angle_flex"], tm["paper_angle_abd"]) == \
        ("T_TMC_fe", "T_TMC_aa")
    assert rows["THUMB_PROXIMAL"]["anatomical_joint"].startswith("thumb MCP")
    assert rows["THUMB_DISTAL"]["anatomical_joint"].startswith("thumb IP")
    palm = rows["PALM"]
    assert palm["anatomical_joint"] == "wrist (the palm against the forearm)"
    assert (palm["paper_angle_flex"], palm["paper_angle_abd"]) == ("", "")
    assert palm["given"] == "not one of the 24"
    wrist = rows["WRIST"]
    assert wrist["anatomical_joint"].startswith("root")
    assert "no angles" in wrist["anatomical_joint"]
    assert wrist["landmark_in_text_files"] == "Wrist"
    for name in JOINT_NAMES:
        if name.endswith("_TIP"):
            assert rows[name]["anatomical_joint"].endswith(
                "fingertip (not a joint; angles 0 by construction)")
            assert rows[name]["flex_deg_means"].startswith(
                "0 by construction")


def test_bone_before_is_the_parent_rows_bone():
    rows = by_row()
    for i, name in enumerate(JOINT_NAMES):
        par = PARENT.get(i)
        if par is None:
            assert name == "WRIST"
            continue
        parent = JOINT_NAMES[par]
        if parent == "WRIST":
            assert rows[name]["bone_before"] == jk.FOREARM
        elif not parent.endswith("_TIP"):
            assert rows[name]["bone_before"] == rows[parent]["bone_after"]


def test_naming_rule_states_the_equivalences():
    rule = jk.NAMING_RULE
    assert "bone that starts at its joint" in rule
    assert len(re.findall(r"\.(\s|$)", rule)) <= 2
    for part in ("*_METACARPAL = CMC", "*_PROXIMAL = MCP",
                 "*_INTERMEDIATE = PIP", "*_DISTAL = DIP",
                 "THUMB_METACARPAL = TMC", "THUMB_PROXIMAL = MCP",
                 "THUMB_DISTAL = IP", "PALM = the wrist", "WRIST = the root",
                 "*_TIP = the fingertip, not a joint"):
        assert part in rule, part


# --- README lines ------------------------------------------------------------------
def test_readme_lines_fit_the_page_and_carry_the_rule_and_every_row():
    lines = jk.readme_lines()
    assert all(len(x) <= 96 for x in lines), max(lines, key=len)
    assert jk.NAMING_RULE in " ".join(x.strip() for x in lines if x.strip())
    table = [x for x in lines if re.match(r"^\s+\d+\s+[A-Z]", x)]
    assert len(table) == 26
    rows = jk.key_rows()
    for line, r in zip(table, rows):
        cells = line.split()
        assert cells[0] == str(r["number"]) and cells[1] == r["row"]
        assert jk.short_joint(r["row"]) in line
        assert line.rstrip().endswith(r["landmark_in_text_files"])
        for c in COLUMN.values():
            if r[c]:
                assert r[c] in line
    text = "\n".join(lines)
    for name in JOINT_NAMES:
        assert re.search(rf"\b{name}\b", text), name


# --- the files ----------------------------------------------------------------------
def test_csv_round_trips_and_is_deterministic(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    jk.write_key_csv(a)
    jk.write_key_csv(b)
    assert a.read_bytes() == b.read_bytes()
    with open(a, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == jk.KEY_COLUMNS
        back = list(reader)
    rows = jk.key_rows()
    assert len(back) == 26
    for got, want in zip(back, rows):
        assert got == {k: str(v) for k, v in want.items()}


def test_pdf_is_one_page_with_every_number_and_row(tmp_path):
    from pypdf import PdfReader

    a, b = tmp_path / "a.pdf", tmp_path / "other_name.pdf"
    footer = "joint key test footer, built 2026-10-06"
    jk.write_key_pdf(a, footer)
    jk.write_key_pdf(b, footer)
    assert a.read_bytes() == b.read_bytes()
    reader = PdfReader(str(a))
    assert len(reader.pages) == 1
    box = reader.pages[0].mediabox
    assert (round(float(box.width)), round(float(box.height))) == (595, 842)
    text = reader.pages[0].extract_text()
    tokens = set(re.findall(r"\b\d+\b", text))
    assert {str(n) for n in range(1, 27)} <= tokens
    for r in jk.key_rows():
        assert r["row"] in text or jk.short_joint(r["row"]) in text
    assert footer in text
    assert "numbers = row of the joint-frame table" in text
    for letter in "TIMRL":
        assert re.search(rf"(^|\s){letter}(\s|$)", text), letter


def test_the_drawing_is_a_left_palm_and_no_number_is_covered():
    m = jk.HAND_MM
    tips = [m[f"{f}_TIP"][0] for f in ("THUMB", "INDEX", "MIDDLE", "RING",
                                        "LITTLE")]
    assert tips == sorted(tips)                 # thumb on the left
    for f in ("INDEX", "MIDDLE", "RING", "LITTLE"):
        ys = [m[f"{f}_{k}"][1] for k in ("METACARPAL", "PROXIMAL",
                                         "INTERMEDIATE", "DISTAL", "TIP")]
        assert ys == sorted(ys)                 # fingers up
    assert max(p[1] for p in jk.FOREARM_MM) <= m["WRIST"][1]
    assert min(p[1] for p in jk.FOREARM_MM) < m["WRIST"][1]
    for scale in (1.8, 2.09, 2.4):
        lay = jk.hand_layout(100.0, 300.0, scale)
        assert len(lay["plates"]) == 26
        assert jk.layout_clashes(lay) == []
    # the clash test catches what it is for
    lay = jk.hand_layout(100.0, 300.0, 2.09)
    lay["plates"]["PALM"] = lay["plates"]["WRIST"]
    assert any("overlaps plate" in c for c in jk.layout_clashes(lay))
    lay = jk.hand_layout(100.0, 300.0, 2.09)
    x, y = lay["dots"]["INDEX_DISTAL"]
    lay["plates"]["INDEX_DISTAL"] = (x - 4, y - 4, x + 4, y + 4)
    clashes = jk.layout_clashes(lay)
    assert any("covers the dot" in c for c in clashes)
    assert any("bone line" in c for c in clashes)


def test_the_thumb_its_letter_and_its_numbers_are_on_one_side():
    lay = jk.hand_layout(100.0, 300.0, 2.09)
    dots, plates = lay["dots"], lay["plates"]
    thumb = [n for n in JOINT_NAMES if n.startswith("THUMB_")]
    assert [JOINT_INDEX[n] + 1 for n in thumb] == [3, 4, 5, 6]
    little_x = min(dots[n][0] for n in JOINT_NAMES if n.startswith("LITTLE_"))
    index_x = min(dots[n][0] for n in JOINT_NAMES if n.startswith("INDEX_"))
    for n in thumb:
        assert dots[n][0] < little_x
        # the thumb's plates sit on its outer (radial) side
        assert plates[n][2] <= dots[n][0]
    assert dots["THUMB_TIP"][0] < index_x
    assert lay["letters"]["T"][0] == dots["THUMB_TIP"][0]
    assert lay["letters"]["L"][0] == dots["LITTLE_TIP"][0]


def joint_frames_view():
    path = REPO / "scripts" / "joint_frames_view.py"
    spec = importlib.util.spec_from_file_location("joint_frames_view_k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def left_to_right(xy, rows):
    """The five metacarpal bases (TMC, CMC) from left to right on a view."""
    names = [r["joint"] for r in rows]
    bases = ["THUMB_METACARPAL", "INDEX_METACARPAL", "MIDDLE_METACARPAL",
             "RING_METACARPAL", "LITTLE_METACARPAL"]
    return sorted(bases, key=lambda n: xy[names.index(n)][0])


def test_the_drawing_has_the_handedness_of_the_joint_frame_pages():
    """The key's hand is drawn as the palm side view of joint_frames.pdf
    draws a left hand (`_view_points`, the projection that page uses), so
    the two pages agree: on both the thumb is on the left."""
    from xr_hand import joint_frames as jf

    view = joint_frames_view()
    rows_key = [{"joint": n} for n in JOINT_NAMES]
    key_xy = [jk.HAND_MM[n] for n in JOINT_NAMES]
    want = left_to_right(key_xy, rows_key)
    assert want[0] == "THUMB_METACARPAL" and want[-1] == "LITTLE_METACARPAL"

    gen = MockHandGenerator(hand="left")
    for _ in range(20):
        raw = gen.next_frame()
    line = jf.line_from_hand_frame(parse_hand_message(raw,
                                                      hand_side_hint="left"))
    rows = jf.joint_table(line, "glove")
    xy, _tips = view._view_points(rows, "palm")
    assert left_to_right(xy, rows) == want
    # and a right hand comes out the other way round on the same view
    gen = MockHandGenerator(hand="right")
    for _ in range(20):
        raw = gen.next_frame()
    line = jf.line_from_hand_frame(parse_hand_message(raw,
                                                      hand_side_hint="right"))
    rows = jf.joint_table(line, "glove")
    xy, _tips = view._view_points(rows, "palm")
    assert left_to_right(xy, rows) == list(reversed(want))


@pytest.mark.skipif(not (SESSION / "joint_frames").is_dir(),
                    reason="the 2026-10-01 grasp session is not on disk")
def test_the_drawing_has_the_handedness_of_the_real_left_hand():
    view = joint_frames_view()
    path = sorted((SESSION / "joint_frames").glob("*_left_take*.csv"))[0]
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            row = {"joint": r["joint"]}
            for k in ("x_mm", "y_mm", "z_mm"):
                row[k] = float(r[k])
            for a in "xyz":
                row[f"axis_{a}"] = [float(r[f"axis_{a}_{c}"]) for c in "xyz"]
            rows.append(row)
    xy, _tips = view._view_points(rows, "palm")
    rows_key = [{"joint": n} for n in JOINT_NAMES]
    assert left_to_right(xy, rows) == left_to_right(
        [jk.HAND_MM[n] for n in JOINT_NAMES], rows_key)


# --- the landmarks, from the writer of the text files -------------------------------
def written_block():
    gen = MockHandGenerator(hand="left")
    for _ in range(5):
        raw = gen.next_frame()
    frame = parse_hand_message(raw, hand_side_hint="left")
    return exporter().frame_block(frame, 1790000000.0)


def labels_in(text):
    """{index: "index (label)"} of every landmark line, read with the
    reader's own pattern, and whether the block has a Wrist line."""
    out = {}
    wrist = False
    for line in text.splitlines():
        m = prof_format.POINT_RE.match(line.strip())
        if m:
            out[int(m.group("idx"))] = f"{m.group('idx')} ({m.group('label')})"
        elif line.startswith(jk.WRIST_LABEL + ": "):
            wrist = True
    return out, wrist


def test_landmark_labels_are_the_ones_the_text_files_carry():
    block = written_block()
    frames = list(prof_format.parse_text(block))
    assert len(frames) == 1 and len(frames[0].points) == \
        prof_format.N_LANDMARKS
    labels, wrist = labels_in(block)
    assert wrist
    assert sorted(labels) == list(range(prof_format.N_LANDMARKS))
    rows = by_row()
    joints = {idx: joint for idx, _label, joint in exporter().LANDMARKS}
    for idx, label in labels.items():
        assert rows[joints[idx]]["landmark_in_text_files"] == label
        assert label == f"{idx} ({PROF_LABELS[idx]})"
    in_text = set(joints.values())
    for name in JOINT_NAMES:
        mark = rows[name]["landmark_in_text_files"]
        if name == "WRIST":
            assert mark == jk.WRIST_LABEL
        elif name in in_text:
            assert mark in labels.values()
        else:
            assert name.endswith("_METACARPAL") and name != "THUMB_METACARPAL"
            assert mark == "not in the text files"
    assert JOINT_INDEX["WRIST"] not in [JOINT_INDEX[j] for j in in_text]


@pytest.mark.skipif(not (SESSION / "keypoints").is_dir(),
                    reason="the 2026-10-01 grasp session is not on disk")
def test_landmark_labels_match_a_real_keypoint_file():
    path = sorted((SESSION / "keypoints").glob("*_keypoints.txt"))[0]
    labels, wrist = labels_in(path.read_text(encoding="utf-8"))
    assert wrist
    marks = {r["landmark_in_text_files"] for r in jk.key_rows()}
    assert set(labels.values()) <= marks
    assert len(labels) == prof_format.N_LANDMARKS
