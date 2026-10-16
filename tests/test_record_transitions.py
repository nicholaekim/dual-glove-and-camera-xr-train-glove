"""`scripts/leap/record_transitions.py`, end to end on a synthetic hand.

The mock hand (`TransitionActor`) holds one curl, and a known time after the
cue moves to another over a known time; the recording must hold the whole
change with frames from before the cue, and the timing must find the two
moments that were planted. Then the session goes into the professor's
transition table through the packager.
"""
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

import test_protocol_check as fx
from leap_hand import grasp_handin as gh
from leap_hand import grasp_validate as gv
from leap_hand import transition as tr

REPO = Path(__file__).resolve().parents[1]


def load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = load("scripts/leap/record_transitions.py", "record_transitions_t")
FAST = ["--start-hold", "0.6", "--cue-wait", "0.3", "0.5", "--hold", "0.6",
        "--seed", "1", "--quiet", "--mock", "--hand", "left"]


def session_of(out: Path) -> Path:
    (folder,) = [p for p in out.iterdir() if p.is_dir()]
    return folder


def lines_of(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    out = tmp_path_factory.mktemp("transitions")
    rc = rt.main(FAST + ["--pairs", "p1_hook>p1_tip", "--takes", "2",
                         "--out", str(out), "--operator", "T Est"])
    return rc, session_of(out)


def test_pairs_name_grasps_of_the_protocol():
    known = ["p1_hook", "p1_tip", "p1_lateral"]
    assert rt.parse_pairs("p1_hook>p1_tip, p1_tip > p1_lateral", known) == [
        ("p1_hook", "p1_tip"), ("p1_tip", "p1_lateral")]
    # the open hand is a pose a change can start from or end in
    assert rt.parse_pairs("open_hand>p1_hook,p1_hook>open_hand", known) == [
        ("open_hand", "p1_hook"), ("p1_hook", "open_hand")]
    for bad, why in (("p1_hook", "is not <from>><to>"),
                     ("p1_hook>fist", "is not a grasp of the protocol"),
                     ("p1_hook>p1_hook", "changes nothing"),
                     (" , ", "no pair given")):
        with pytest.raises(ValueError, match=why):
            rt.parse_pairs(bad, known)


def test_a_mock_session_writes_the_recordings_and_their_record(recorded):
    rc, session = recorded
    assert rc == 0
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    assert (meta["set"], meta["hand"], meta["mock"], meta["operator"]) == (
        "transitions", "left", True, "T Est")
    assert meta["pairs"] == [["p1_hook", "p1_tip"]] and meta["ended"]
    assert ("0.1 s or longer, of pose speed at or above 25 mm/s"
            in meta["movement_start_method"])
    assert "stays below 10 mm/s for 0.3 s" in meta["settled_method"]
    takes = meta["takes"]
    assert [(t["from_grasp"], t["to_grasp"], t["take"], t["accepted"])
            for t in takes] == [("p1_hook", "p1_tip", 1, True),
                                ("p1_hook", "p1_tip", 2, True)]
    for t in takes:
        assert (session / t["files"]["leap"]).is_file()
        assert (session / t["files"]["meta"]).is_file()
        assert t["name"].startswith(f"p1_hook_to_p1_tip_left_take{t['take']}_")


def test_the_recording_holds_the_whole_change_from_before_the_cue(recorded):
    _rc, session = recorded
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    for t in meta["takes"]:
        take = json.loads((session / t["files"]["meta"]).read_text(
            encoding="utf-8"))
        lines = lines_of(session / t["files"]["leap"])
        cue = take["cue_wall_time"]
        seen = [d["capture_time"] for d in lines]
        # a second of the starting pose, held still, before the cue
        assert cue - 1.05 <= min(seen) <= cue - 0.9
        before = [d for d in lines if d["capture_time"] < cue]
        assert len(before) >= 80
        assert max(tr.joint_speeds(before)[1][5:-5]) < tr.SETTLE_SPEED_MM_S
        # and on past the settle by the hold
        g = take["timing"]
        assert max(seen) - min(seen) >= g["settled_time_s"] + 0.5
        # the label and the take on every line, in the camera line format
        assert {d["pose"] for d in lines} == {"p1_hook_to_p1_tip"}
        assert {d["take"] for d in lines} == {t["take"]}
        assert all(len(d["abs26"]) == 26 and len(d["joints"]) == 26
                   and d["source"] == "leap" for d in lines)
        assert take["start_hand_id"] in {d["hand_id"] for d in lines}
        assert take["frames"] == len(lines)


def test_the_timing_finds_the_planted_start_and_settle(recorded):
    """The actor reacts 0.25 s after the cue and moves for 0.5 s, eased at
    both ends. Both moments are threshold crossings, so the start reads a
    little late and the settle a little early; the README says so."""
    _rc, session = recorded
    actor = rt.TransitionActor()
    assert (actor.react_s, actor.move_s) == (0.25, 0.5)
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    for t in meta["takes"]:
        take = json.loads((session / t["files"]["meta"]).read_text(
            encoding="utf-8"))
        g = take["timing"]
        assert g["validity"] == tr.VALID and take["accepted"] is True
        assert g["cue_time_s"] == pytest.approx(1.0, abs=0.06)
        assert g["reaction_time_s"] == pytest.approx(0.25, abs=0.1)
        assert g["transition_duration_s"] == pytest.approx(0.5, abs=0.15)
        assert g["transition_duration_s"] == pytest.approx(
            g["settled_time_s"] - g["movement_start_time_s"], abs=1e-6)
        assert g["speed_at_cue_mm_s"] < tr.POSE_MOVE_SPEED_MM_S
        # the mock hand changes pose in place: nothing carried it
        assert g["hand_speed_at_cue_mm_s"] < tr.MOVE_SPEED_MM_S
        assert g["pose_change_mm"] > 10.0
        # the file alone gives the same numbers again
        again = tr.recorded_timing(lines_of(session / t["files"]["leap"]),
                                   take)
        assert again.validity == tr.VALID
        assert again.movement_start_time_s == pytest.approx(
            g["movement_start_time_s"], abs=0.012)
        assert again.settled_time_s == pytest.approx(g["settled_time_s"],
                                                     abs=0.012)


def test_the_session_table_has_the_hand_ins_columns(recorded):
    _rc, session = recorded
    with open(session / gh.TRANSITIONS_FILE, encoding="utf-8",
              newline="") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    assert header == gh.TRANSITION_COLUMNS and len(rows) == 2
    for r in rows:
        assert (r["participant_id"], r["hand"], r["from_grasp"],
                r["to_grasp"], r["validity"]) == ("P01", "left", "p1_hook",
                                                  "p1_tip", "valid")
        assert (session / r["recording_file"]).is_file()
        assert float(r["transition_duration_s"]) == pytest.approx(
            float(r["settled_time_s"]) - float(r["movement_start_time_s"]),
            abs=0.0011)
        assert r["static_take_file"] == ""


def test_thresholds_are_options_and_go_into_the_record(tmp_path):
    out = tmp_path / "t"
    assert rt.main(FAST + ["--pairs", "open_hand>p1_spherical", "--takes", "1",
                           "--out", str(out), "--start-speed", "30",
                           "--start-sustain", "0.05", "--settle-speed", "15",
                           "--settle-window", "0.2", "--cue-still", "0.05"]
                   ) == 0
    session = session_of(out)
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    assert meta["thresholds"] == {
        "start_speed_mm_s": 30.0, "start_sustain_s": 0.05,
        "settle_speed_mm_s": 15.0, "settle_window_s": 0.2,
        "cue_still_s": 0.05}
    take = json.loads((session / meta["takes"][0]["files"]["meta"]).read_text(
        encoding="utf-8"))
    assert take["thresholds"] == meta["thresholds"]
    g = take["timing"]
    assert ("0.05 s or longer, of pose speed at or above 30 mm/s"
            in g["movement_start_method"])
    assert "below 15 mm/s for 0.2 s" in g["settled_method"]
    assert take["from_grasp"] == "open_hand"
    # a lower start speed is crossed sooner
    assert g["reaction_time_s"] < 0.31


def test_a_hand_that_moves_before_the_cue_is_sent_back_then_rejected(
        tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "EARLY_MOVES_MAX", 2)
    said = []
    args = rt.argparse.Namespace(
        pairs="p1_hook>p1_tip", hand="left", takes=1,
        protocol="protocols/grasps.json", operator=None, participant="P01",
        out=tmp_path / "t", start_hold=0.4, cue_wait=[0.4, 0.4], pre=1.0,
        hold=0.5, timeout=3.0, start_timeout=10.0, retries=0, seed=1,
        start_speed=None, start_sustain=None, settle_speed=None,
        settle_window=None, cue_still=None, mock=True, mock_move_early=True)
    rc = rt.run(args, beeper=lambda *_a: None, say=said.append)
    assert rc == 1
    session = session_of(tmp_path / "t")
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    (take,) = meta["takes"]
    assert take["accepted"] is False
    assert take["reason"] == "the hand moved before the cue 2 times"
    # the cue never sounded: no recording was written
    assert take["files"]["leap"] is None
    assert not list((session / "leap").iterdir())
    assert sum("MOVED BEFORE THE BEEP" in x for x in said) == 2
    assert not any("CHANGE NOW" in x for x in said)
    with open(session / gh.TRANSITIONS_FILE, encoding="utf-8",
              newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["validity"] == "invalid" and row["transition_duration_s"] == ""
    assert row["notes"] == "the hand moved before the cue 2 times"


def test_a_missing_camera_and_a_bad_pair_are_said_not_crashed_on(tmp_path,
                                                                 capsys):
    assert rt.main(["--mock", "--quiet", "--hand", "left", "--pairs",
                    "p1_hook>nothing", "--out", str(tmp_path)]) == 2
    assert "is not a grasp of the protocol" in capsys.readouterr().out
    assert rt.main(["--mock", "--quiet", "--hand", "left", "--pairs",
                    "p1_hook>p1_tip", "--protocol", str(tmp_path / "no.json"),
                    "--out", str(tmp_path)]) == 2
    assert "cannot read the protocol file" in capsys.readouterr().out


def test_the_takes_join_the_professors_transition_table(recorded, tmp_path,
                                                        capsys):
    """`--transitions <session>` on the packager: the kept takes are copied
    into transitions\\, timed again and listed, and the README stops saying
    that no change between two grasps was recorded."""
    _rc, session = recorded
    pkg = load("scripts/package_professor_set.py", "package_for_transitions")
    grasps = fx.make_grasp_session(tmp_path / "a")
    out = tmp_path / "out"
    # a mock session is left out unless asked for, like a mock grasp session
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--transitions", str(session)]) == 0
    assert "mock transition session, not handed in" in capsys.readouterr().out
    assert sorted(p.name for p in (out / "transitions").iterdir()) == [
        gh.TRANSITIONS_FILE]
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--transitions", str(session), "--allow-mock"]) == 0
    names = sorted(p.name for p in (out / "transitions").iterdir())
    assert names == ["p1_hook_to_p1_tip_left_take1.jsonl",
                     "p1_hook_to_p1_tip_left_take2.jsonl", gh.TRANSITIONS_FILE]
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    assert (out / "transitions" / names[0]).read_bytes() == (
        session / meta["takes"][0]["files"]["leap"]).read_bytes()
    with open(out / "transitions" / gh.TRANSITIONS_FILE, encoding="utf-8",
              newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["from_grasp"], r["to_grasp"], r["take"], r["validity"],
             r["recording_file"]) for r in rows] == [
        ("p1_hook", "p1_tip", "1", "valid",
         "transitions/p1_hook_to_p1_tip_left_take1.jsonl"),
        ("p1_hook", "p1_tip", "2", "valid",
         "transitions/p1_hook_to_p1_tip_left_take2.jsonl")]
    assert all(r["session"] == session.name and r["static_take_file"] == ""
               for r in rows)
    # synthetic rows never read like real ones: MOCK heads their notes, the
    # README's top and their lines in section 17
    assert all(r["notes"].startswith("MOCK (synthetic hand)") for r in rows)
    readme = (out / "README.txt").read_text(encoding="utf-8")
    top = readme.split("\nOperator:", 1)[0]
    assert "MOCK DATA" in top and session.name in top
    flat = " ".join(readme.split())
    assert "What exists, one held pose into another: 2 recordings" in flat
    assert ("MOCK (synthetic hand): p1_hook to p1_tip, take 1: valid: the "
            "pose began to change") in flat
    assert ("2 of them are MOCK: a synthetic hand acting the change out, "
            f"recorded with --mock (session {session.name})") in flat
    assert "it has been run on synthetic hands only" in flat
    assert "What does not exist: a recording of the change from one GRASP" \
        not in flat
    assert ("2 recordings, 2 containing a change of pose: 2 valid, 0 "
            "questionable; 2 of them between two grasp types") in flat
    # the check of the folder accepts the table and its recordings when
    # told the folder is a rehearsal, and refuses the MOCK rows otherwise
    rep = gv.validate(out, allow_mock=True)
    assert not [e for e in rep.errors if "transition" in e.lower()
                or "_to_" in e or "MOCK" in e]
    assert rep.facts["transitions"] == 2
    assert rep.facts["valid_transitions"] == 2
    assert rep.facts["mock_transitions"] == 2
    refused = gv.validate(out).errors
    assert [e for e in refused if "MOCK" in e] == [
        e for e in refused if e not in rep.errors]
    assert any(e.startswith(f"2 rows of {gh.TRANSITIONS_FILE} say MOCK")
               for e in refused)
    # a folder that is not a transition session is named, not packaged
    assert pkg.main(["--out", str(tmp_path / "o2"), "--grasps", str(grasps),
                     "--transitions", str(grasps)]) == 0
    assert "not a transition session" in capsys.readouterr().out


def test_a_real_transition_session_says_no_mock_anywhere(recorded, tmp_path):
    """The same takes in a session that is not mock (a copy whose
    session.json says "mock": false), with a real grasp session and no
    --allow-mock: no "MOCK" in any text file of the folder, the rows' notes
    are the timing's own, and the check has nothing to say about MOCK."""
    _rc, session = recorded
    real = tmp_path / "real" / session.name
    shutil.copytree(session, real)
    meta = json.loads((real / "session.json").read_text(encoding="utf-8"))
    meta["mock"] = False
    (real / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    pkg = load("scripts/package_professor_set.py", "package_for_real_rows")
    grasps = fx.make_grasp_session(tmp_path / "a")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--grasps", str(grasps),
                     "--transitions", str(real)]) == 0
    with open(out / "transitions" / gh.TRANSITIONS_FILE, encoding="utf-8",
              newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["session"] == real.name]
    assert len(rows) == 2
    texts = [p for p in out.rglob("*")
             if p.suffix in (".txt", ".csv", ".json", ".jsonl")]
    assert texts and not [p.name for p in texts if "MOCK" in p.read_text(
        encoding="utf-8", errors="replace")]
    flat = " ".join((out / "README.txt").read_text(encoding="utf-8").split())
    assert "p1_hook to p1_tip, take 1: valid: the pose began to change" in flat
    assert "it has been run on synthetic hands only" not in flat
    rep = gv.validate(out)
    assert rep.facts["mock_transitions"] == 0
    assert not [e for e in rep.errors if "MOCK" in e]
    assert gv.validate(out, allow_mock=True).errors == rep.errors


def test_the_last_line_says_how_the_session_reaches_the_table(tmp_path):
    """A real session in recordings/protocol/transitions is picked up by
    the rebuild launcher by itself; a --mock one must be named with
    --transitions and --allow-mock; a real one put elsewhere with --out must
    be named with --transitions."""
    default = rt.REPO / rt.OUT / "20261008_100000_left"
    line = rt.rebuild_hint(default, mock=False)
    assert line == (
        "to put them in the professor's table: run the rebuild launcher "
        "(rebuild_professor_grasp_set.cmd on the Desktop); it picks this "
        "session up by itself from recordings\\protocol\\transitions")
    assert "--transitions" not in line and "--allow-mock" not in line
    rehearsal = rt.REPO / rt.MOCK_OUT / "20261008_100000_left"
    line = rt.rebuild_hint(rehearsal, mock=True)
    assert line.startswith("this is a MOCK session (synthetic hand)")
    assert line.endswith(f"--transitions \"{rehearsal}\" --allow-mock")
    # mock wins wherever the session is
    assert rt.rebuild_hint(default, mock=True).endswith("--allow-mock")
    elsewhere = tmp_path / "t" / "20261008_100000_left"
    line = rt.rebuild_hint(elsewhere, mock=False)
    assert f"--transitions \"{elsewhere}\"" in line
    assert "--allow-mock" not in line and "picks up by itself only" in line


def test_a_mock_run_ends_with_the_mock_hint(tmp_path):
    said = []
    args = rt.argparse.Namespace(
        pairs="p1_hook>p1_tip", hand="left", takes=1,
        protocol="protocols/grasps.json", operator=None, participant="P01",
        out=tmp_path / "t", start_hold=0.6, cue_wait=[0.3, 0.5], pre=1.0,
        hold=0.6, timeout=8.0, start_timeout=10.0, retries=0, seed=1,
        start_speed=None, start_sustain=None, settle_speed=None,
        settle_window=None, cue_still=None, mock=True, mock_move_early=False)
    assert rt.run(args, beeper=lambda *_a: None, say=said.append) == 0
    session = session_of(tmp_path / "t")
    assert said[-1] == rt.rebuild_hint(session, mock=True)
    assert said[-1].endswith(f"--transitions \"{session}\" --allow-mock")
