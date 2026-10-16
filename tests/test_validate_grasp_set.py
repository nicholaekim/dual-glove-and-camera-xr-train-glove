"""The rebuild command and the check of the folder it writes.

One session is recorded with the real recorder on the mock hand (so its
static interval and summary frame are what the recorder's own rule gives),
rebuilt with `scripts/rebuild_grasp_set.py`, and checked. Then the folder
is damaged one way at a time, each a thing the check exists to catch, and
the check has to say which.
"""
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from leap_hand import grasp_handin as gh
from leap_hand import grasp_validate as gv

REPO = Path(__file__).resolve().parents[1]


def load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digests(folder: Path, skip=("joint_frames",)) -> dict:
    return {p.relative_to(folder).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()
            and p.relative_to(folder).parts[0] not in skip}


def fixture_protocol(root: Path) -> Path:
    """root/grasps_fixture.json: two grasps, cylindrical and tip_pinch."""
    protocol = root / "grasps_fixture.json"
    protocol.write_text(json.dumps({
        "name": "grasps", "version": 1, "status": "test fixture",
        "description": "two grasps for the validator's tests",
        "takes_per_item": 2, "duration_s": 5.0, "prep_s": 5.0,
        "items": [
            {"id": "cylindrical", "label": "Cylindrical", "source": "p1 Test",
             "figure": "Figure 1, panel 1", "shape": "wrap a cylinder",
             "object_implied": True},
            {"id": "tip_pinch", "label": "Tip pinch", "source": "p1 Test",
             "figure": "Figure 1, panel 6", "shape": "thumb tip to index tip",
             "object_implied": False}]}), encoding="utf-8")
    return protocol


def record_mock(*args) -> None:
    """The real recorder on the mock hand, its beeps silenced, no window."""
    rp = load("scripts/leap/record_poses.py", "record_poses_for_validate")
    rp.beep = lambda *a, **k: None
    argv, sys.argv = sys.argv, ["record_poses.py", "--mock", "--hand", "left",
                                "--auto-accept", "--no-panel", "--no-open",
                                *map(str, args)]
    try:
        rp.main()
    finally:
        sys.argv = argv


def make_set_a(root: Path) -> Path:
    """A mock Set A session of two grasps (protocol root/grasps_fixture.json:
    cylindrical, tip_pinch), recorded by the real recorder on the mock hand
    into root/rec, one take each with the other hand in view, its target
    then raised to two takes. Its folder passes the check. Also used by
    tests/test_rebuild_grasp_set.py."""
    protocol = fixture_protocol(root)
    record_mock("--protocol", protocol, "--takes", "1", "--duration", "2.5",
                "--prep", "1.2", "--still", "hand", "--out-dir", root / "rec",
                "--operator", "T Est", "--mock-other-hand")
    (session,) = [p for p in (root / "rec").iterdir() if p.is_dir()]
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    assert sum(1 for t in meta["takes"] if t["accepted"]) == 2
    meta["takes_per_item"] = 2          # as --resume --takes 2 leaves it
    (session / "session.json").write_text(json.dumps(meta, indent=2),
                                          encoding="utf-8")
    return session


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """`make_set_a`'s session, and the folder the rebuild command makes of
    it."""
    root = tmp_path_factory.mktemp("validate")
    session = make_set_a(root)
    raw = digests(session)
    rebuild = load("scripts/rebuild_grasp_set.py", "rebuild_for_validate")
    out = root / "deliver" / "grasp set"
    args = ["--session", str(session), "--out", str(out), "--operator",
            "T Est", "--allow-mock"]
    rc = rebuild.main(args)
    return {"session": session, "out": out, "rc": rc, "rebuild": rebuild,
            "args": args, "raw": raw}


def damaged(built, tmp_path) -> Path:
    copy = tmp_path / "copy"
    shutil.copytree(built["out"], copy)
    return copy


def rows_of(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_rows(path: Path, header, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def test_the_rebuild_writes_a_folder_that_passes_its_check(built, capsys):
    assert built["rc"] == 0
    out = built["out"]
    rep = gv.validate(out, built["session"])
    assert rep.errors == []
    f = rep.facts
    assert (f["grasp_types"], f["recordings"], f["target"],
            f["missing_takes"]) == (2, 2, 4, 2)
    assert f["takes_per_grasp"] == {"cylindrical": 1, "tip_pinch": 1}
    assert f["missing_by_grasp"] == {"cylindrical": 1, "tip_pinch": 1}
    # two different questions, two different answers
    assert f["static_complete"] is True and f["multi_complete"] is False
    assert f["ready"] == 2
    assert f["other_hand"] == 2                    # --mock-other-hand
    assert f["rule_same"] == f["rule_checked"] == 2
    assert f["pdf_pages"] == 2 and f["transitions"] == 2
    # the mock stills carry the skeleton: a picture of each, and the key
    assert (f["images"], f["missing_images"], f["image_pdf_pages"],
            f["key_rows"]) == (2, 0, 2, 26)
    for key in ("bad_duration", "missing_keypoints", "missing_all_frames",
                "missing_joint_frames", "missing_angles", "nan_angles",
                "bad_xyz", "no_source", "label_errors"):
        assert f[key] == 0, key
    lo, mid, hi = f["duration"]
    assert 2.2 < lo <= mid <= hi < 2.6             # the 2.5 s takes
    text = gv.render(rep)
    assert "STATIC-POSE TASK: COMPLETE (2 of 2 grasp types" in text
    assert ("FULL MULTI-TAKE DATA COLLECTION: INCOMPLETE (2 of 4 recordings "
            "against our replication target; 2 still to record)") in text
    assert text.rstrip().endswith("FOLDER CHECK: PASSED")
    # the rebuild left the same report in the folder, without a time
    assert (out / gh.VALIDATION_FILE).read_text(encoding="utf-8") == text
    assert "Checked:" not in text
    cli = load("scripts/validate_grasp_set.py", "validate_cli")
    assert cli.main([str(out), "--session", str(built["session"])]) == 0
    printed = capsys.readouterr().out
    assert "Checked: " in printed and "FOLDER CHECK: PASSED" in printed
    assert cli.main([str(out / "not there")]) == 2


def test_the_rebuild_reads_the_recordings_and_changes_none(built, capsys):
    """Run again: no file of the folder changes, and every file of the
    session outside its derived joint_frames folder has the bytes it had."""
    stamps = {p: p.stat().st_mtime_ns for p in built["out"].rglob("*")
              if p.is_file()}
    assert built["rebuild"].main(built["args"]) == 0
    printed = capsys.readouterr().out
    assert ": 0 new, 0 changed, " in printed
    assert {p: p.stat().st_mtime_ns for p in built["out"].rglob("*")
            if p.is_file()} == stamps
    assert not (built["out"].parent / "_superseded").exists()
    assert digests(built["session"]) == built["raw"]
    assert "FOLDER CHECK: PASSED" in printed


def test_the_folder_says_it_is_a_mock_and_holds_what_the_readme_lists(built):
    out = built["out"]
    text = (out / "README.txt").read_text(encoding="utf-8")
    assert "MOCK DATA" in text
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*")
                   if p.is_file())
    assert names == sorted([
        "README.txt", "validation_report.txt",
        "grasps/grasps_summary.csv", "grasps/grasp_types_audit.csv",
        "grasps/joint_angles_24.csv",
        "grasps/cylindrical_left_take1.jsonl",
        "grasps/cylindrical_left_take1_keypoints.txt",
        "grasps/cylindrical_left_all_frames.txt",
        "grasps/tip_pinch_left_take1.jsonl",
        "grasps/tip_pinch_left_take1_keypoints.txt",
        "grasps/tip_pinch_left_all_frames.txt",
        "grasps/joint_frames/cylindrical_left_take1.csv",
        "grasps/joint_frames/tip_pinch_left_take1.csv",
        "grasps/joint_frames/joint_frames.pdf",
        "grasps/joint_frames/joint_key.csv",
        "grasps/joint_frames/joint_key.pdf",
        "grasps/images/cylindrical_left_take1.png",
        "grasps/images/tip_pinch_left_take1.png",
        "grasps/images/grasp_images.pdf",
        "transitions/pose_transitions.csv",
        "transitions/open_hand_to_cylindrical_left_take1.jsonl",
        "transitions/open_hand_to_tip_pinch_left_take1.jsonl"])
    # the other hand is in the recording and nowhere else
    _h, summary = rows_of(out / "grasps" / gh.SUMMARY_FILE)
    for r in summary:
        assert r["other_hand_detected"] == "true"
        assert float(r["other_hand_frame_fraction"]) > 0.9
        assert float(r["other_hand_median_distance_cm"]) > 15.0
        assert r["hand_selection"] == "followed id" and r["identity_flag"] == ""
        assert int(r["jsonl_row_count"]) > int(r["hand_frame_count"])
        blocks = gv.read_blocks(out / r["all_frames_file"])
        assert len(blocks) == int(r["hand_frame_count"])
        assert {b[1] for b in blocks} == {"left"}


def test_a_missing_keypoints_file_fails_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    (copy / "grasps" / "tip_pinch_left_take1_keypoints.txt").unlink()
    rep = gv.validate(copy)
    assert not rep.passed and rep.facts["missing_keypoints"] == 1
    assert any("tip_pinch take 1: no representative keypoints file" in e
               for e in rep.errors)
    # the grasp has no usable static configuration left
    assert rep.facts["static_complete"] is False
    text = gv.render(rep)
    assert "STATIC-POSE TASK: INCOMPLETE (1 of 2" in text
    assert "FOLDER CHECK: FAILED" in text


def test_a_nan_angle_fails_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / "joint_frames" / "cylindrical_left_take1.csv"
    header, rows = rows_of(path)
    rows[8]["flex_deg"] = "nan"
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert not rep.passed and rep.facts["nan_angles"] >= 1
    assert any("NaN or infinite number" in e for e in rep.errors)


def test_a_wrong_duration_fails_the_check(built, tmp_path):
    """A duration that is not the span of the timestamps: here the number
    of the take's lines over the frame rate, the mistake the summary was
    rebuilt to stop."""
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / gh.SUMMARY_FILE
    header, rows = rows_of(path)
    true = float(rows[0]["recording_duration_s"])
    rows[0]["recording_duration_s"] = f"{int(rows[0]['jsonl_row_count']) / 90.0:.3f}"
    assert abs(float(rows[0]["recording_duration_s"]) - true) > 1.0
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert not rep.passed and rep.facts["bad_duration"] == 1
    assert any("the timestamps give" in e for e in rep.errors)
    # and a missing one
    rows[0]["recording_duration_s"] = ""
    write_rows(path, header, rows)
    assert gv.validate(copy).facts["bad_duration"] == 1


def test_a_frame_missing_from_a_per_pose_dataset_fails_the_check(built,
                                                                 tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / "cylindrical_left_all_frames.txt"
    blocks = path.read_text(encoding="utf-8").strip("\n").split("\n\n")
    path.write_text("\n\n".join(blocks[:-1]) + "\n", encoding="utf-8")
    rep = gv.validate(copy)
    assert any("cylindrical_left_all_frames.txt has" in e for e in rep.errors)
    # a frame of another hand put in its place
    other = blocks[-1].replace("Hand ID: left", "Hand ID: right").replace(
        "Frame ", "Frame 9", 1)
    path.write_text("\n\n".join(blocks[:-1] + [other]) + "\n",
                    encoding="utf-8")
    rep = gv.validate(copy)
    assert any("is not the labelled hand's frames" in e for e in rep.errors)


def test_a_keypoints_file_of_another_frame_fails_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    blocks = (copy / "grasps" / "cylindrical_left_all_frames.txt").read_text(
        encoding="utf-8").strip("\n").split("\n\n")
    (copy / "grasps" / "cylindrical_left_take1_keypoints.txt").write_text(
        blocks[-1] + "\n", encoding="utf-8")            # the last frame
    rep = gv.validate(copy)
    assert any("is not the representative frame of the recording" in e
               for e in rep.errors)


def test_a_summary_that_hides_the_other_hand_fails_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / gh.SUMMARY_FILE
    header, rows = rows_of(path)
    rows[1]["other_hand_detected"] = "false"
    rows[1]["other_hand_frame_fraction"] = "0"
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert sum("other_hand" in e for e in rep.errors) == 2


def test_a_duration_on_a_take_that_was_not_timed_fails_the_check(built,
                                                                 tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "transitions" / gh.TRANSITIONS_FILE
    header, rows = rows_of(path)
    # the mock closes at the instant of the cue: no reaction time, so the
    # rule calls it moving at the cue and gives it no duration
    assert [r["validity"] for r in rows] == ["invalid", "invalid"]
    assert all("already changing at the cue" in r["notes"]
               and r["transition_duration_s"] == "" for r in rows)
    rows[0]["transition_duration_s"] = "0.4"
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert any("a transition_duration_s on a row marked invalid" in e
               for e in rep.errors)
    # a row for a recording that is not there
    rows[0]["transition_duration_s"] = ""
    rows[1]["recording_file"] = "transitions/open_hand_to_hook_left_take1.jsonl"
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert any("open_hand_to_hook_left_take1.jsonl" in e for e in rep.errors)


def test_a_transition_row_that_says_mock_fails_unless_allowed(built,
                                                              tmp_path):
    """A row whose notes say MOCK is a take of a synthetic hand: the folder
    fails for it alone, and passes with allow_mock, which leaves a note. The
    marker is no reason for an invalid row, and the README must say MOCK
    DATA at its top whatever the check is told."""
    copy = damaged(built, tmp_path)
    path = copy / "transitions" / gh.TRANSITIONS_FILE
    header, rows = rows_of(path)
    assert not any("MOCK" in r["notes"] for r in rows)
    own = rows[0]["notes"]
    rows[0]["notes"] = f"{gh.MOCK_NOTE}: {own}"
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    (error,) = rep.errors
    assert error.startswith(f"1 row of {gh.TRANSITIONS_FILE} says MOCK")
    assert "cylindrical take 1" in error and "--allow-mock" in error
    allowed = gv.validate(copy, allow_mock=True)
    assert allowed.passed and allowed.facts["mock_transitions"] == 1
    assert any("allowed by --allow-mock" in n for n in allowed.notes)
    text = gv.render(allowed)
    assert "2: 0 valid, 0 questionable; 1 of them MOCK (synthetic hand)" in text
    assert text.rstrip().endswith("FOLDER CHECK: PASSED")
    rows[0]["notes"] = gh.MOCK_NOTE
    write_rows(path, header, rows)
    assert gv.validate(copy, allow_mock=True).errors == [
        "open_hand to cylindrical take 1: invalid, with no reason given"]
    rows[0]["notes"] = f"{gh.MOCK_NOTE}: {own}"
    write_rows(path, header, rows)
    readme = copy / "README.txt"
    readme.write_text(readme.read_text(encoding="utf-8").replace(
        "MOCK DATA", "DATA"), encoding="utf-8")
    (error,) = gv.validate(copy, allow_mock=True).errors
    assert error.startswith("README.txt does not say MOCK DATA at its top")


def test_a_transition_number_in_the_static_summary_fails_the_check(built,
                                                                   tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / gh.SUMMARY_FILE
    header, rows = rows_of(path)
    header = header + ["transition_s"]
    for r in rows:
        r["transition_s"] = "1.7"
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert any("unexpected columns" in e for e in rep.errors)


def test_a_stray_file_and_a_cut_readme_fail_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    (copy / "grasps" / "scratch.txt").write_text("x", encoding="utf-8")
    readme = copy / "README.txt"
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace("17. POSE TRANSITION TIMING",
                                   "17. TIMING"), encoding="utf-8")
    rep = gv.validate(copy)
    assert any("grasps/scratch.txt: in the folder, in no table" in e
               for e in rep.errors)
    assert any("section \"17. POSE TRANSITION TIMING\" is missing" in e
               for e in rep.errors)
    # a file at the top of the folder is not the check's business
    (copy / "grasps" / "scratch.txt").unlink()
    readme.write_text(text, encoding="utf-8")
    (copy / "a note.md").write_text("x", encoding="utf-8")
    assert gv.validate(copy).passed


def test_a_recording_under_the_wrong_label_fails_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / gh.SUMMARY_FILE
    header, rows = rows_of(path)
    rows[0]["raw_jsonl_file"] = rows[1]["raw_jsonl_file"]
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert rep.facts["label_errors"] >= 1
    assert any("carry another label" in e for e in rep.errors)


def test_a_session_with_more_accepted_takes_than_the_folder_fails(built,
                                                                  tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    meta = json.loads((built["session"] / "session.json").read_text(
        encoding="utf-8"))
    meta["takes"].append(dict(meta["takes"][0], take=2,
                              name="cylindrical_left_take2_x"))
    (session / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    rep = gv.validate(built["out"], session)
    assert any("session.json has 3 accepted take(s), the summary 2" in e
               for e in rep.errors)


def test_a_folder_without_its_tables_is_reported_not_crashed_on(tmp_path):
    (tmp_path / "grasps").mkdir()
    rep = gv.validate(tmp_path)
    assert not rep.passed and rep.facts == {}
    text = gv.render(rep)
    assert "could not be read" in text and "FOLDER CHECK: FAILED" in text


def test_a_folder_deeper_than_260_characters_is_checked_all_the_same(built,
                                                                    tmp_path):
    """The hand-in folder sits under a long OneDrive path and its files have
    long names. Past 260 characters Windows says a file is not there unless
    it is asked the long way; the check asks the long way, as the packager
    writes."""
    deep = tmp_path / ("x" * 100) / ("y" * 100) / "grasp set for the professor"
    shutil.copytree(built["out"], gh.long_path(deep))
    longest = max((p for p in gh.long_path(deep).rglob("*") if p.is_file()),
                  key=lambda p: len(str(p)))
    assert len(str(longest)) > 264
    rep = gv.validate(deep)
    assert rep.errors == []
    assert rep.facts["recordings"] == 2 and rep.facts["pdf_pages"] == 2
    assert gv.render(rep).splitlines()[1 + 1] == "Folder: grasp set for the professor"
    # the rebuild writes and checks a folder that deep too
    out = tmp_path / ("p" * 100) / ("q" * 100) / "grasp set"
    args = [a if a != str(built["out"]) else str(out) for a in built["args"]]
    assert built["rebuild"].main(args) == 0
    assert (gh.long_path(out) / gh.VALIDATION_FILE).read_text(
        encoding="utf-8").rstrip().endswith("FOLDER CHECK: PASSED")


# --- the README's formulas, computed a second time --------------------------------------
def test_the_check_computes_the_tables_and_the_stillness_a_second_time(built):
    """Every joint-frame table, static interval and hand size of the folder
    agrees with the check's own implementation of the README's formulas,
    and the report says how many it compared."""
    rep = gv.validate(built["out"])
    f = rep.facts
    assert f["formula_same"] == f["formula_checked"] == 2
    assert f["stillness_same"] == f["stillness_checked"] == 2
    assert f["angles_same"] == f["angles_checked"] == 2
    lo, mid, hi = f["static_speed"]
    assert 0.0 <= lo <= mid <= hi
    text = gv.render(rep)
    assert "joint-frame table = the README's formulas, again" in text
    assert ("stillness, shape and hand size = the README's formulas, again"
            in text)
    assert "angle table = the README's formulas, again" in text
    assert ("of the paper's 24 angles, not given by this tracker" in text
            and "6 per recording: T_TMC_fe, T_TMC_aa, I_CMC" in text)
    assert ("TRANSITION TIMING: MISSING (0 valid and 0 questionable of 2 "
            "recording(s) of a change of pose") in text
    assert "none between two grasp types)" in text
    assert "expected recording target (our replication target)" in text
    assert "missing takes (against the replication target)" in text


def test_the_readme_formulas_are_the_librarys_on_both_hands():
    """`readme_table` shares no code with `xr_hand.joint_frames`; on a left
    and on a right hand, flexed, spread and turned, the two give one table."""
    import numpy as np
    from leap_hand.mock import MockLeapStream
    from leap_hand.to_openxr import to_hand_frame
    from xr_hand import joint_frames as jf
    for pose in ("fist", "open_palm"):
        stream = MockLeapStream(pose=pose, seed=5, dropout_every=0,
                                reacquire_every=0, cycle_seconds=1.0)
        hands = list(stream.generate(46))
        for side in ("left", "right"):
            lh = [h for s, h in hands if s == side][-1]
            # the line the recorder writes: the chain, and the tracker's own
            # absolute joint positions beside it
            line = jf.line_from_hand_frame(to_hand_frame(lh))
            line["abs26"] = [[float(v) for v in p] for p in lh.abs26]
            want = jf.joint_table(line, "camera")
            got = gv.readme_table(line)
            assert list(got) == [r["joint"] for r in want]
            for r in want:
                g = got[r["joint"]]
                assert g["parent"] == r["parent"]
                assert np.allclose(g["xyz"], [r["x_mm"], r["y_mm"], r["z_mm"]],
                                   atol=1e-6), (side, r["joint"])
                assert np.allclose(g["axes"], np.array(
                    [r["axis_x"], r["axis_y"], r["axis_z"]]).T, atol=1e-9)
                assert g["bone"] == pytest.approx(r["bone_len_mm"], abs=1e-6)
                assert g["angles"] == pytest.approx(
                    (r["flex_deg"], r["abd_deg"], r["twist_deg"]), abs=1e-6)
            # every filled column of the angle table, the same way
            own = gv.readme_angle_row(line)
            lib = dict(jf.dof24(want))
            lib.update(jf.palm_reference_angles(line, "camera"))
            lib.update(jf.mcp_paper_order(line, "camera"))
            for name in gh.GIVEN_NAMES + jf.MCP_PAPER_NAMES:
                assert own[name] == pytest.approx(lib[name], abs=1e-6), name
            for column, key in gh.METACARPAL_COLUMNS.items():
                assert own[column] == pytest.approx(lib[key], abs=1e-6), column
            assert own["forearm_to_palm_deg"] == pytest.approx(
                jf.wrist_palm_deg(line, "camera"), abs=1e-6)
            assert set(own) == set(gh.ANGLE_COLUMNS[5:]) - set(gh.NOT_GIVEN)
            # a fist bends: the comparison is not of zeros
            if pose == "fist":
                assert got["INDEX_INTERMEDIATE"]["angles"][0] > 30.0
            # the CSV of the table passes, and a cell off by 0.05 does not
            cells = jf.csv_rows(want)
            table = [dict(zip(jf.CSV_COLUMNS, c)) for c in cells]
            cell, axis, wrong = gv.table_gap(line, table)
            assert wrong == "" and cell <= gv.CELL_TOL and axis <= gv.AXIS_TOL
            table[8]["flex_deg"] = f"{float(table[8]['flex_deg']) + 0.05:.2f}"
            assert gv.table_gap(line, table)[0] > gv.CELL_TOL


def test_rigid_fit_puts_a_moved_shape_back_and_never_mirrors_it():
    import numpy as np
    rng = np.random.default_rng(3)
    shape = rng.normal(size=(26, 3))
    c, s = np.cos(0.7), np.sin(0.7)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    moved = (turn @ shape.T).T + np.array([0.3, -0.2, 0.1])
    r, t = gv.rigid_fit(moved, shape)
    assert np.allclose((r @ moved.T).T + t, shape, atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0)
    mirrored = shape * np.array([-1.0, 1.0, 1.0])
    r, _t = gv.rigid_fit(mirrored, shape)
    assert np.linalg.det(r) == pytest.approx(1.0)      # a rotation still


def test_a_table_cell_that_is_not_the_formulas_fails_the_check(built,
                                                               tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / "joint_frames" / "cylindrical_left_take1.csv"
    header, rows = rows_of(path)
    rows[8]["flex_deg"] = f"{float(rows[8]['flex_deg']) + 0.05:.2f}"
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert not rep.passed
    assert any("is not what the README's formulas give" in e
               for e in rep.errors)
    assert rep.facts["formula_same"] == 1 and rep.facts["formula_checked"] == 2


def test_a_stillness_or_size_that_is_not_the_recordings_fails_the_check(
        built, tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / gh.SUMMARY_FILE
    header, rows = rows_of(path)
    true = dict(rows[0])
    for column, off in (("static_interval_mean_speed_mm_s", 1.0),
                        ("static_interval_pose_speed_mm_s", 1.0),
                        ("static_interval_shape_rms_max_mm", 0.5),
                        ("take_shape_rms_max_mm", 0.5),
                        ("frames_within_5mm_percent", 1.0),
                        ("shape_rms_start_mm", 0.5),
                        ("index_metacarpal_mm", 1.0)):
        rows[0] = dict(true)
        rows[0][column] = f"{float(true[column]) + off:.2f}"
        write_rows(path, header, rows)
        rep = gv.validate(copy)
        assert not rep.passed, column
        assert any(f"{column} is" in e and "the README's formula gives" in e
                   for e in rep.errors), column
        assert rep.facts["stillness_same"] == 1


def test_the_audit_counts_the_recordings_without_a_note(built, tmp_path):
    """The mock hand moves, so both recordings carry a note and both grasp
    types are named; an audit that says otherwise fails the check."""
    out = built["out"]
    _h, audit = rows_of(out / "grasps" / gh.AUDIT_FILE)
    _h, summary = rows_of(out / "grasps" / gh.SUMMARY_FILE)
    for a in audit:
        clean = sum(1 for r in summary if r["grasp_name"] == a["grasp_name"]
                    and not r["notes"])
        assert int(a["recordings_without_a_note"]) == clean
    rep = gv.validate(out)
    noted = sorted(a["grasp_name"] for a in audit
                   if a["recordings_without_a_note"] == "0")
    assert sorted(rep.facts["all_noted"]) == noted
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    if noted:
        assert "Worth making all the same: a recording of" in flat
        assert ("grasp types whose every recording carries a note"
                in gv.render(rep))
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / gh.AUDIT_FILE
    header, rows = rows_of(path)
    rows[0]["recordings_without_a_note"] = str(
        int(rows[0]["recordings_without_a_note"]) + 1)
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert any("recordings_without_a_note is" in e for e in rep.errors)


def test_an_angle_this_tracker_does_not_give_must_stay_empty(built, tmp_path):
    """joint_angles_24.csv: the six not given are empty, every other column
    is what the check's own formulas give, and either kind of damage fails."""
    out = built["out"]
    header, rows = rows_of(out / "grasps" / gh.ANGLES_FILE)
    assert header == gh.ANGLE_COLUMNS
    for r in rows:
        assert [r[n] for n in gh.NOT_GIVEN] == [""] * 6
        assert all(r[n] != "" for n in gh.GIVEN_NAMES)
        assert all(r[c] != "" for c in gh.METACARPAL_COLUMNS)
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / gh.ANGLES_FILE
    rows[0]["I_CMC"] = rows[0]["I_metacarpal_flex_wrist_ref"]
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert any("I_CMC filled in, which this tracker does not give" in e
               for e in rep.errors)
    rows[0]["I_CMC"] = ""
    rows[0]["M_MCP_aa_paper_order"] = \
        f"{float(rows[0]['M_MCP_aa_paper_order']) + 0.05:.2f}"
    write_rows(path, header, rows)
    rep = gv.validate(copy)
    assert any("is not what the README's formulas give" in e
               for e in rep.errors)
    assert rep.facts["angles_same"] == 1 and rep.facts["angles_checked"] == 2


def test_the_frame_numbers_of_the_static_interval_cut_the_per_pose_file(built):
    """static_interval_first_frame_id and ..._last_frame_id are 'Frame'
    numbers of the all-frames file, and the frames between them span the
    static interval's length."""
    out = built["out"]
    _h, summary = rows_of(out / "grasps" / gh.SUMMARY_FILE)
    for r in summary:
        first, last = (int(r["static_interval_first_frame_id"]),
                       int(r["static_interval_last_frame_id"]))
        ids = [b[0] for b in gv.read_blocks(out / r["all_frames_file"])]
        assert first in ids and last in ids and first < last
        inside = [i for i in ids if first <= i <= last]
        # 90 frames a second: the interval's length in frames, give or take
        want = float(r["static_interval_duration_s"]) * 90.0
        assert abs(len(inside) - want) <= 0.15 * want
        assert first <= int(r["representative_frame_id"]) <= last


def test_a_readme_that_drops_a_column_or_a_status_fails_the_check(built,
                                                                  tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "README.txt"
    text = path.read_text(encoding="utf-8")
    assert "    hand_ids: " in text and "TRANSITION TIMING: " in text
    path.write_text(text.replace("    hand_ids: ", "    hand ids: "),
                    encoding="utf-8")
    rep = gv.validate(copy)
    assert any("does not explain the summary column(s) hand_ids" in e
               for e in rep.errors)
    path.write_text(text.replace("TRANSITION TIMING: ", "TIMING: "),
                    encoding="utf-8")
    rep = gv.validate(copy)
    assert any('does not give its "TRANSITION TIMING:" status' in e
               for e in rep.errors)


def test_the_readme_opens_with_where_each_thing_asked_for_is(built, tmp_path):
    """The ten things the hand-in was asked to make easy to find, in the
    words they were asked in, each with a file of the folder; and the
    check fails when one of them is dropped."""
    out = built["out"]
    text = (out / "README.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    top = lines[:lines.index("CONTENTS")]
    at = top.index(gh.ASKED_TITLE)
    assert at > 0 and len(gh.ASKED_FOR) == 10
    assert gh.ASKED_FOR[-2:] == ("Images of the grasp types",
                                 "Which joint each angle corresponds to")
    for name in gh.ASKED_FOR:
        at = top.index(f"  {name}", at + 1)        # all there, in order
    flat = " ".join(" ".join(top).split())
    for needle in (
            "grasps\\grasps_summary.csv, column recording_duration_s",
            "transitions\\pose_transitions.csv, column "
            "transition_duration_s. Recorded separately, never taken from "
            "the static recordings",
            "grasps\\<grasp>_left_take<N>.jsonl, 26 joints, metres, camera "
            "frame",
            "grasps\\joint_frames\\<grasp>_left_take<N>.csv, columns "
            "flex_deg abd_deg twist_deg",
            "18 given and 6 that this tracker does not measure left empty",
            "This file, section 12",
            "grasps\\<grasp>_left_all_frames.txt: one file per grasp type, "
            "2 files",
            "drawn on every page of grasps\\joint_frames\\joint_frames.pdf",
            "one frame per recording, taken from the stillest part of it, "
            "not the last frame (section 7)",
            "grasps\\images\\<grasp>_left_take<N>.png: one infrared picture "
            "per recording, cut to the recorded hand, and "
            "grasps\\images\\grasp_images.pdf, one page per grasp type "
            "(section 5)",
            "grasps\\joint_frames\\joint_key.csv and joint_key.pdf, and the "
            "table in section 11"):
        assert needle in flat, needle
    # every file the map names is in the folder, with the column it names
    header, _rows = rows_of(out / "grasps" / "grasps_summary.csv")
    assert "recording_duration_s" in header
    header, _rows = rows_of(out / "transitions" / "pose_transitions.csv")
    assert "transition_duration_s" in header
    assert (out / "grasps" / "joint_angles_24.csv").is_file()
    assert (out / "grasps" / "joint_frames" / "joint_frames.pdf").is_file()
    assert list((out / "grasps").glob("*_all_frames.txt"))
    assert list((out / "grasps").glob("*_keypoints.txt"))
    assert max(len(x) for x in top if ":\\" not in x) <= 96
    # dropped from the top, the check says which
    copy = damaged(built, tmp_path)
    path = copy / "README.txt"
    path.write_text(text.replace("  Joint angles\n", "  Angles\n", 1),
                    encoding="utf-8")
    rep = gv.validate(copy)
    assert any("does not say at its top where to find: Joint angles" in e
               for e in rep.errors)


# --- the pictures and the joint key ---------------------------------------------
def test_a_removed_picture_fails_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    (copy / "grasps" / "images" / "tip_pinch_left_take1.png").unlink()
    rep = gv.validate(copy)
    assert any("tip_pinch take 1: its picture "
               "grasps/images/tip_pinch_left_take1.png is missing" in e
               for e in rep.errors)
    assert rep.facts["images"] == 1


def test_an_audit_that_miscounts_the_pictures_fails_the_check(built,
                                                              tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / gh.AUDIT_FILE
    header, rows = rows_of(path)
    assert [r["recordings_with_an_image"] for r in rows] == ["1", "1"]
    rows[0]["recordings_with_an_image"] = "0"
    write_rows(path, header, rows)
    assert any("cylindrical: recordings_with_an_image is '0', the summary "
               "names 1 picture(s)" in e for e in gv.validate(copy).errors)


def test_a_picture_that_is_not_a_png_fails_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    (copy / "grasps" / "images" / "cylindrical_left_take1.png").write_text(
        "not a picture", encoding="utf-8")
    rep = gv.validate(copy)
    assert any("cylindrical take 1: grasps/images/cylindrical_left_take1.png "
               "does not open as a picture" in e for e in rep.errors)
    # and one that opens but is all black
    from PIL import Image
    Image.new("RGB", (40, 60)).save(
        copy / "grasps" / "images" / "cylindrical_left_take1.png")
    rep = gv.validate(copy)
    assert any("cylindrical_left_take1.png is all black" in e
               for e in rep.errors)


def test_a_stray_image_or_a_pdf_without_its_pages_fails_the_check(built,
                                                                  tmp_path):
    copy = damaged(built, tmp_path)
    images = copy / "grasps" / "images"
    (copy / "photo.png").write_bytes(
        (images / "tip_pinch_left_take1.png").read_bytes())
    rep = gv.validate(copy)
    assert any("photo.png: an image in the folder that no summary row names"
               in e for e in rep.errors)
    (copy / "photo.png").unlink()
    (images / "grasp_images.pdf").unlink()
    rep = gv.validate(copy)
    assert any("grasp_images.pdf is missing" in e for e in rep.errors)


def test_a_key_row_swapped_fails_the_check(built, tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "grasps" / "joint_frames" / gh.KEY_CSV
    header, rows = rows_of(path)
    assert len(rows) == 26
    # two rows exchanged: the rows are no longer the tables' order
    swapped = list(rows)
    swapped[7], swapped[8] = swapped[8], swapped[7]
    write_rows(path, header, swapped)
    rep = gv.validate(copy)
    assert any("joint_key.csv: its rows are not the 26 joints" in e
               for e in rep.errors)
    # the names kept in order but one angle moved to another row
    moved = [dict(r) for r in rows]
    i = next(k for k, r in enumerate(moved) if r["row"] == "INDEX_PROXIMAL")
    j = next(k for k, r in enumerate(moved) if r["row"] == "INDEX_DISTAL")
    moved[i]["paper_angle_flex"], moved[j]["paper_angle_flex"] = (
        moved[j]["paper_angle_flex"], moved[i]["paper_angle_flex"])
    write_rows(path, header, moved)
    rep = gv.validate(copy)
    assert any("joint_key.csv: I_MCP_fe is not in the paper_angle_flex of "
               "row INDEX_PROXIMAL" in e for e in rep.errors)
    write_rows(path, header, rows)
    assert gv.validate(copy).passed
    (copy / "grasps" / "joint_frames" / gh.KEY_PDF).unlink()
    assert any("joint_key.pdf is missing" in e
               for e in gv.validate(copy).errors)


def test_a_readme_that_says_no_object_was_held_fails_the_check(built,
                                                              tmp_path):
    copy = damaged(built, tmp_path)
    path = copy / "README.txt"
    text = path.read_text(encoding="utf-8")
    assert "No object was held" not in text and gv.validate(copy).passed
    end = text.index("\n2. DATASET AT A GLANCE\n") + 1      # the section's head
    path.write_text(text[:end] + "  No object was held.\n\n" + text[end:],
                    encoding="utf-8")
    rep = gv.validate(copy)
    assert any("section 1 says \"No object was held\"" in e
               for e in rep.errors)
    end = text.index("\n17. POSE TRANSITION TIMING\n") + 1
    path.write_text(text[:end] + "  It was held open beside the camera.\n\n"
                    + text[end:], encoding="utf-8")
    assert any("section 16 says the other hand was \"held open\"" in e
               for e in gv.validate(copy).errors)
    path.write_text(text.replace("Images: one infrared picture",
                                 "Pictures: one infrared picture"),
                    encoding="utf-8")
    assert any("does not say that the folder holds the infrared pictures" in e
               for e in gv.validate(copy).errors)
    rule = " ".join(__import__("xr_hand.joint_key", fromlist=["x"])
                    .NAMING_RULE.split())
    first = rule.split()[:6]
    lines = text.splitlines()
    cut = [x for x in lines if not (" ".join(x.split()).startswith(
        " ".join(first)))]
    path.write_text("\n".join(cut) + "\n", encoding="utf-8")
    assert any("section 11 does not carry the joint key" in e
               for e in gv.validate(copy).errors)


# --- what the pictures show, read by eye -----------------------------------------
def test_a_picture_not_read_by_eye_is_counted_apart_and_the_counts_add_up(
        built, tmp_path, monkeypatch):
    """Still notes that read one of the two pictures: section 1 counts the
    other as not yet read (never as showing nothing), its row carries the
    note, and the rebuilt folder passes its check. A README whose counts do
    not add up to the summary's rows, or that leaves an unread picture out
    of its line, fails it. With both pictures read the line is gone."""
    session = built["session"]
    notes = {"session": session.name, "read_on": "2026-10-06",
             "confirmed_by_operator": False, "in_or_against_the_hand": {},
             "in_the_other_hand": {}, "cannot_tell": [],
             "read": ["cylindrical_left_take1"]}
    monkeypatch.setattr(gh, "load_notes", lambda s: notes)
    out = tmp_path / "unread" / "grasp set"
    args = ["--session", str(session), "--out", str(out), "--operator",
            "T Est", "--allow-mock"]
    assert built["rebuild"].main(args) == 0          # its check passed
    text = (out / "README.txt").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    # the head the folder check reads section 1's counts under
    # (gv.OBJECTS_HEAD): the reading was made on the uncut stills
    assert gv.OBJECTS_HEAD in flat
    for needle in ("What was seen in the hands, read by eye on 2026-10-06 "
                   "(not confirmed by the operator) from the recorder's "
                   "uncut stills, which show both hands and the room and "
                   "are kept in the session folder, not in this folder; the "
                   "pictures in this folder (section 5) are cut to the "
                   "recorded hand, so what is reported for the other hand "
                   "cannot be seen in them:",
                   "0 recordings with something in or against the recorded "
                   "hand. 0 with something in the other hand. 0 where the "
                   "picture does not let one tell. the other 1 with a "
                   "picture: nothing seen in the hand. 1 with a picture not "
                   "yet read by eye: tip_pinch_left_take1.",
                   "The picture has not been read by eye (section 5): 1"):
        assert needle in flat, needle
    _h, summary = rows_of(out / "grasps" / gh.SUMMARY_FILE)
    assert [r["notes"].endswith("the infrared picture has not been read by "
                                "eye") for r in summary] == [False, True]
    rep = gv.validate(out, session)
    assert rep.errors == []
    # the counts do not add up: one more "nothing seen" than there is
    readme = out / "README.txt"
    readme.write_text(text.replace(
        "the other 1 with a picture: nothing seen",
        "the other 2 with a picture: nothing seen"), encoding="utf-8")
    assert any("README.txt section 1: 0 in or against the recorded hand + 0 "
               "in the other hand + 0 cannot tell + 2 nothing seen + 1 not "
               "yet read + 0 without a picture = 3, the summary has 2 "
               "recordings" in e for e in gv.validate(out).errors)
    # the unread picture counted as showing nothing, its line dropped
    lines = text.splitlines()
    at = next(k for k, x in enumerate(lines)
              if "with a picture not yet read by eye" in x)
    readme.write_text("\n".join(
        x.replace("the other 1 with", "the other 2 with")
        for x in lines[:at] + lines[at + 1:]) + "\n", encoding="utf-8")
    errors = gv.validate(out).errors
    assert any("section 1 does not count among the pictures not yet read "
               "by eye: tip_pinch_left_take1" in e for e in errors), errors
    assert not any("the summary has 2 recordings" in e for e in errors)
    readme.write_text(text, encoding="utf-8")
    assert gv.validate(out).passed
    # both read: no line, no note, the two counted as showing nothing
    notes["read"] = ["cylindrical_left_take1", "tip_pinch_left_take1"]
    out = tmp_path / "read" / "grasp set"
    args[3] = str(out)
    assert built["rebuild"].main(args) == 0
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert "the other 2 with a picture: nothing seen in the hand." in flat
    assert "not yet read" not in flat and "not been read" not in flat
    _h, summary = rows_of(out / "grasps" / gh.SUMMARY_FILE)
    assert not any("not been read" in r["notes"] for r in summary)


# --- a take kept after the hand was lost while forming (rules of 2026-10-07) -------
def section(text: str, number: int) -> str:
    """README section `number`, flattened: from its "N. TITLE" line to the
    next numbered section's."""
    lines = text.splitlines()
    heads = [k for k, x in enumerate(lines)
             if x[:1].isdigit() and ". " in x[:4] and x.split(". ", 1)[1][:1].isupper()]
    start = next(k for k in heads if lines[k].startswith(f"{number}. "))
    after = [k for k in heads if k > start]
    return " ".join(" ".join(lines[start:after[0] if after else None]).split())


@pytest.fixture(scope="module")
def built_lost(tmp_path_factory):
    """`make_set_a`'s protocol recorded by the real recorder on the mock
    hand: cylindrical as usual, then the same session resumed for tip_pinch
    with the hand lost 0.6 s after the GO beep and back 0.5 s later
    (--mock-lose-forming 1 --mock-back-after 0.5); rebuilt as `built` is."""
    root = tmp_path_factory.mktemp("validate_lost")
    protocol = fixture_protocol(root)
    common = ["--protocol", protocol, "--takes", "1", "--duration", "2.5",
              "--prep", "1.2", "--still", "hand", "--out-dir", root / "rec",
              "--operator", "T Est", "--mock-other-hand"]
    record_mock(*common, "--items", "cylindrical")
    (session,) = [p for p in (root / "rec").iterdir() if p.is_dir()]
    record_mock(*common, "--items", "tip_pinch", "--resume", session,
                "--mock-lose-forming", "1", "--mock-back-after", "0.5")
    rebuild = load("scripts/rebuild_grasp_set.py", "rebuild_for_validate_lost")
    out = root / "deliver" / "grasp set"
    rc = rebuild.main(["--session", str(session), "--out", str(out),
                       "--operator", "T Est", "--allow-mock"])
    return {"session": session, "out": out, "rc": rc}


def test_a_take_kept_after_a_loss_while_forming_is_said_so_and_not_timed(
        built_lost):
    """The README of a mock build with such a take: the per-take note in
    the summary and section 15 with its count line, the rule change in
    section 5 (from the metas), the forming file counted among the invalid
    in section 17 with that reason and no duration in the table; and the
    folder check, which times every forming file again, agrees."""
    from leap_hand import transition as tr

    assert built_lost["rc"] == 0
    session, out = built_lost["session"], built_lost["out"]
    data = json.loads((session / "session.json").read_text(encoding="utf-8"))
    takes = {t["item"]: t for t in data["takes"]}
    assert [(t["item"], t["attempt"], t["accepted"]) for t in data["takes"]] == [
        ("cylindrical", 1, True), ("tip_pinch", 1, True)]
    metas = {i: json.loads((session / t["files"]["meta"]).read_text(
        encoding="utf-8")) for i, t in takes.items()}
    assert metas["tip_pinch"]["coaching"]["forming_lost_count"] == 1
    assert metas["cylindrical"]["coaching"]["forming_lost_count"] == 0
    rep = gv.validate(out, session)
    assert rep.errors == []
    note = ("the hand was lost 1 time while forming the grasp and found again "
            "before the hold; the forming file is not timed")
    assert note == gh.forming_lost_notes(metas["tip_pinch"])[0]
    _h, summary = rows_of(out / "grasps" / gh.SUMMARY_FILE)
    by = {r["grasp_name"]: r for r in summary}
    assert note in by["tip_pinch"]["notes"]
    assert "lost" not in by["cylindrical"]["notes"]
    _h, rows = rows_of(out / gh.TRANSITIONS_DIR / gh.TRANSITIONS_FILE)
    moves = {r["to_grasp"]: r for r in rows}
    lost = moves["tip_pinch"]
    assert lost["validity"] == "invalid"
    assert lost["notes"] == tr.lost_note(1)
    assert (lost["transition_duration_s"], lost["movement_start_time_s"],
            lost["settled_time_s"], lost["reaction_time_s"]) == ("", "", "", "")
    assert "lost while forming" not in moves["cylindrical"]["notes"]
    # the copy in the folder carries the mark on every line
    copied = [json.loads(x) for x in (out / lost["recording_file"]).read_text(
        encoding="utf-8").splitlines() if x.strip()]
    assert copied and all(x[tr.FORMING_LOST_KEY] == 1 for x in copied)
    text = (out / "README.txt").read_text(encoding="utf-8")
    assert max(len(x) for x in text.splitlines()) <= 96
    s5, s15, s17 = (section(text, k) for k in (5, 15, 17))
    assert ("All 2 coached recordings were made under the recorder's rules of "
            "2026-10-07 (recorder_rules in each recording's meta)") in s5
    assert ("Since that change a hand lost while the grasp was being formed "
            "no longer rejects the attempt: the recorder waits for the tracker "
            "to find the hand again (up to 4 s each time; no later than "
            "8 s after the GO beep), holds it tracked for 1 s, and then "
            "records the static hold under the same acquisition gate") in s5
    assert ("The tracker's next hand is treated as the same hand when it has "
            "the same tracker id, or a new id seen in a frame without the "
            "hand, within 10 cm of where the hand was lost and not set aside "
            "before as another hand (in view at OPEN HAND, seen beside it, or "
            "further away): a rule of id and distance, not a proof of "
            "identity.") in s5
    assert "same hand again" not in s5
    # section 16 says the same rule as a rule (an outside review, 2026-10-07)
    assert ("\"followed id\", 2 recordings: the recorder followed the hand it "
            "acquired at OPEN HAND by the tracker's hand_id. A new id appearing "
            "where that hand was, in a frame without it, is treated as the "
            "same hand found again and followed: a rule of id and distance "
            "(within 10 cm, the recorder's FOLLOW_RADIUS_M), not a proof of "
            "identity. An id seen in the same frame, or more than 10 cm away, "
            "is treated as another hand.") in section(text, 16)
    assert "is the same hand found again" not in " ".join(text.split())
    assert ("The hand the tracker acquired at OPEN HAND is followed by its "
            "tracker id through the recording, by the rule of id and distance "
            "in section 16, so the frames measured are the ones that rule "
            "gives that hand, whatever else was in view.") in s5
    assert "1 recording here was kept that way" in s5
    assert "goes back to OPEN HAND up to 10 times" in s5
    assert ("the READY cue is 2 short beeps (520 Hz, 70 ms, at 30 % of full "
            "scale) with KEEP HAND OPEN on screen, the GO cue one 450 ms beep "
            "at 1760 Hz and 100 % of full scale") in s5
    assert ("Lost while forming the grasp, found again before the hold "
            "(sections 5, 17): 1 tip_pinch_left_take1: " + note) in s15
    assert ("1 of the invalid because the hand was lost while forming the "
            "grasp") in s17
    assert ("open_hand to tip_pinch, take 1: invalid: " + tr.lost_note(1)) in s17
    assert "1 not read, the hand lost while forming the grasp" in " ".join(
        text.split())


def test_the_rule_change_is_said_only_for_takes_recorded_under_it(built):
    """The section 5 sentence comes from the metas: the validator's own
    session, all recorded by today's recorder with no loss, says the rules
    and that no recording here lost the hand while forming; the same
    session with its metas as they were before 2026-10-07 says nothing of
    the change."""
    import copy

    from cam_hand import protocol_check as pc

    gs = gh.collect(pc.load_session(built["session"]))
    s5 = section(gh.readme_text(gs, "T Est", "now", "cmd"), 5)
    assert "Since that change a hand lost while the grasp" in s5
    assert "In none of the recordings here was the hand lost while forming." in s5
    for t in gs.takes:
        t.meta = copy.deepcopy(t.meta)
        t.meta.pop("recorder_rules", None)
        t.meta["coaching"].pop("forming_lost_count", None)
    s5 = section(gh.readme_text(gs, "T Est", "now", "cmd"), 5)
    assert "Since that change" not in s5 and "recorder_rules" not in s5
    assert "Coached procedure (2 recordings)" in s5
