"""`scripts/rebuild_grasp_set.py` finds the transition sessions by itself, and
the two Desktop launchers of the transition recorder.

The rebuild looks through `--transitions-dir` (by default the `transitions`
folder beside the session's own set folder: recordings/protocol/transitions
for a session in recordings/protocol/grasps) and adds every session of
`scripts/leap/record_transitions.py` in it to the packager's --transitions,
mock ones only with --allow-mock, after the folders given with
--transitions and none twice. So the rebuild launcher on the Desktop, whose
command never changes, picks up a transition session recorded with the
camera, and a rehearsal never. A rehearsal taken with --allow-mock says
MOCK at the README's top, in its rows' notes and in section 17, and the
check refuses the folder unless it is told --allow-mock too.
"""
import csv
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import test_protocol_check as fx
from leap_hand import grasp_handin as gh
from leap_hand import grasp_validate as gv

REPO = Path(__file__).resolve().parents[1]


def load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rb = load("scripts/rebuild_grasp_set.py", "rebuild_grasp_set_t")
rt = load("scripts/leap/record_transitions.py", "record_transitions_rb_t")


def session(root: Path, name: str, kind: str = "transitions", mock=False,
            accepted=(True, True, False)) -> Path:
    """A session folder with the session.json keys the discovery reads."""
    folder = root / name
    folder.mkdir(parents=True)
    meta = {"set": kind, "hand": "left", "mock": mock,
            "takes": [{"name": f"take{i}", "accepted": a}
                      for i, a in enumerate(accepted, 1)]}
    (folder / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


@pytest.fixture
def folder(tmp_path):
    """Two real transition sessions (given out of name order), a mock one,
    a grasp session, a folder without session.json and a loose file."""
    d = tmp_path / "transitions"
    b = session(d, "20261008_120000_left")
    a = session(d, "20261008_090000_left", accepted=(True,))
    m = session(d, "20261008_100000_left", mock=True)
    session(d, "20261001_101621_left", kind="grasps")
    (d / "20261008_130000_left" / "leap").mkdir(parents=True)
    (d / "notes.txt").write_text("not a session", encoding="utf-8")
    return d, a, b, m


def test_real_sessions_are_found_in_name_order_and_mock_ones_skipped(folder):
    d, a, b, m = folder
    said = []
    assert rb.discover_transition_sessions(d, False, said.append) == [a, b]
    assert said == [
        "  passed over 20261001_101621_left: not a transition session "
        "(set \"grasps\")",
        "  transition session 20261008_090000_left: 1 kept take(s)",
        "  skipped 20261008_100000_left: a mock transition session "
        "(synthetic hand, recorded with --mock); taken only with --allow-mock",
        "  transition session 20261008_120000_left: 2 kept take(s)",
        "  passed over 20261008_130000_left: no readable session.json"]
    # --allow-mock takes the rehearsal too, and says it is one
    said = []
    assert rb.discover_transition_sessions(d, True, said.append) == [a, m, b]
    assert ("  transition session 20261008_100000_left: 2 kept take(s), "
            "MOCK (synthetic hand)") in said


def test_every_mock_flag_shape_counts_as_mock(tmp_path):
    """The packager's rule (`protocol_check.mock_flags`): "mock": true, a
    dict of sensors, or "mock_flags"."""
    d = tmp_path / "t"
    session(d, "a")
    for name, flags in (("b", {"mock": {"leap": True}}),
                        ("c", {"mock_flags": {"camera": True}})):
        f = session(d, name)
        meta = json.loads((f / "session.json").read_text(encoding="utf-8"))
        meta.update(flags)
        (f / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    assert [p.name for p in rb.discover_transition_sessions(
        d, False, lambda _x: None)] == ["a"]
    assert [p.name for p in rb.discover_transition_sessions(
        d, True, lambda _x: None)] == ["a", "b", "c"]


def test_an_absent_or_empty_folder_is_said_in_one_line(tmp_path):
    said = []
    absent = tmp_path / "nothing here"
    assert rb.discover_transition_sessions(absent, False, said.append) == []
    assert said == [f"  no transition sessions in {absent} (no such folder)"]
    said = []
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rb.discover_transition_sessions(empty, False, said.append) == []
    assert said == [f"  no transition sessions in {empty}"]
    # only a rehearsal in it: why it was skipped, then that none was taken
    said = []
    only = tmp_path / "only mock"
    session(only, "20261008_100000_left", mock=True)
    assert rb.discover_transition_sessions(only, False, said.append) == []
    assert said[-1] == f"  no transition sessions in {only} to add"
    assert said[0].startswith("  skipped 20261008_100000_left: a mock")


def test_sessions_given_with_transitions_come_first_and_are_not_repeated(
        folder, tmp_path):
    d, a, b, m = folder
    elsewhere = session(tmp_path / "elsewhere", "20261009_080000_left")
    # the same folder as one found, written another way
    given_b = d / "x" / ".." / b.name
    said = []
    out = rb.transition_folders([given_b, elsewhere], d, False, said.append)
    assert out == [given_b, elsewhere, a]
    assert ("  (20261008_120000_left is also given with --transitions: "
            "added once)") in said
    # a mock session given by name is passed on as before: the packager
    # leaves it out without --allow-mock and says so
    assert rb.transition_folders([m], d, False, lambda _x: None) == [m, a, b]


def test_the_default_folder_is_beside_the_sessions_set_folder():
    grasps = REPO / "recordings" / "protocol" / "grasps" / "20261001_101621_left"
    assert rb.default_transitions_dir(grasps) == (
        REPO / "recordings" / "protocol" / "transitions")
    rehearsal = REPO / "recordings" / "protocol_mock" / "grasps" / "x_left"
    assert rb.default_transitions_dir(rehearsal) == (
        REPO / "recordings" / "protocol_mock" / "transitions")


def stub_scripts(monkeypatch, calls, rc=1):
    """Stand-ins for the joint-frame export and the packager: the packager's
    arguments are kept, and its exit code `rc` ends the rebuild there."""
    def script(name):
        if name == "joint_frames_view":
            return types.SimpleNamespace(export_session=lambda *_a: {
                "skipped": [], "pdf": Path("joint_frames.pdf"), "pages": [1]})
        assert name == "package_professor_set"
        return types.SimpleNamespace(main=lambda call: calls.append(call)
                                     or rc)
    monkeypatch.setattr(rb, "script", script)


def transitions_of(call):
    i = call.index("--transitions")
    rest = call[i + 1:]
    end = next((k for k, x in enumerate(rest) if x.startswith("--")),
               len(rest))
    return rest[:end]


def test_the_rebuild_hands_the_found_sessions_to_the_packager(
        folder, tmp_path, monkeypatch, capsys):
    d, a, b, m = folder
    grasps = session(tmp_path / "protocol" / "grasps", "20261001_101621_left",
                     kind="grasps")
    given = session(tmp_path / "elsewhere", "20261009_080000_left")
    calls = []
    stub_scripts(monkeypatch, calls)
    out = tmp_path / "out"
    assert rb.main(["--session", str(grasps), "--out", str(out),
                    "--transitions", str(given), "--transitions-dir",
                    str(d)]) == 1
    (call,) = calls
    assert transitions_of(call) == [str(given), str(a), str(b)]
    assert "--allow-mock" not in call
    printed = capsys.readouterr().out
    assert "skipped 20261008_100000_left: a mock transition session" in printed
    # with --allow-mock the rehearsal goes in too
    calls.clear()
    assert rb.main(["--session", str(grasps), "--out", str(out),
                    "--transitions-dir", str(d), "--allow-mock"]) == 1
    assert transitions_of(calls[0]) == [str(a), str(m), str(b)]
    assert "--allow-mock" in calls[0]
    # no --transitions-dir: the transitions folder beside the session's set
    # folder, which is absent here, said in one line; no --transitions passed
    calls.clear()
    assert rb.main(["--session", str(grasps), "--out", str(out)]) == 1
    assert "--transitions" not in calls[0]
    assert (f"no transition sessions in {tmp_path / 'protocol' / 'transitions'}"
            " (no such folder)") in capsys.readouterr().out
    # and once a session is recorded there, the same command picks it up
    real = session(tmp_path / "protocol" / "transitions",
                   "20261010_100000_left")
    calls.clear()
    assert rb.main(["--session", str(grasps), "--out", str(out)]) == 1
    assert transitions_of(calls[0]) == [str(real)]


def test_a_recorded_session_beside_the_grasps_reaches_the_table(tmp_path,
                                                               capsys):
    """End to end, no stand-ins: a session recorded by the transition
    recorder (mock hand) into the transitions folder beside a grasp session
    is found by the unchanged rebuild command and becomes rows of the
    table; without --allow-mock it is skipped. (The fixture's grasp session
    lacks what a real one has, source-paper labels among them, so its
    folder check fails on those; none of its errors is about transitions.)"""
    grasps = fx.make_grasp_session(tmp_path / "a", takes_per_item=1)
    beside = grasps.parent.parent / gh.TRANSITION_SET
    assert rt.main(["--start-hold", "0.6", "--cue-wait", "0.3", "0.5",
                    "--hold", "0.6", "--seed", "1", "--quiet", "--mock",
                    "--hand", "left", "--pairs", "p1_hook>p1_tip",
                    "--takes", "1", "--out", str(beside)]) == 0
    (recorded,) = [p for p in beside.iterdir() if p.is_dir()]
    capsys.readouterr()

    def built(out: Path):
        """The rows of the recorded session in the table, and the check's
        lines about transitions (told --allow-mock, as the build was)."""
        assert (out / "README.txt").is_file()
        with open(out / gh.TRANSITIONS_DIR / gh.TRANSITIONS_FILE,
                  encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if r["session"] == recorded.name]
        rep = gv.validate(out, allow_mock=True)
        assert not [e for e in rep.errors if "transition" in e.lower()
                    or "_to_" in e or "MOCK" in e]
        return rows, capsys.readouterr().out

    out = tmp_path / "out" / "with"
    rb.main(["--session", str(grasps), "--out", str(out), "--allow-mock"])
    rows, printed = built(out)
    assert [(r["from_grasp"], r["to_grasp"], r["take"]) for r in rows] == [
        ("p1_hook", "p1_tip", "1")]
    assert rows[0]["notes"].startswith("MOCK (synthetic hand)")
    assert (out / rows[0]["recording_file"]).is_file()
    assert ("TRANSITION TIMING: PARTIAL (1 valid and 0 questionable of 1 "
            "recording(s) of a change of pose; 1 of them between two grasp "
            "types)") in printed
    out = tmp_path / "out" / "without"
    rb.main(["--session", str(grasps), "--out", str(out)])
    rows, printed = built(out)
    assert rows == []
    assert f"skipped {recorded.name}: a mock transition session" in printed
    assert "none between two grasp types" in printed
    assert "MOCK" not in (out / "README.txt").read_text(encoding="utf-8")


# --- what the README says from the takes, measured, not assumed (2026-10-08) -----
def _still_take(item, n, offset, start=100.0, end=105.0):
    """A take as `gh.still_shares` reads it: its still written `offset` s
    after its meta's take_start."""
    return types.SimpleNamespace(
        item=item, ref=types.SimpleNamespace(take=n), still=Path("s.png"),
        still_offset_s=offset, meta={"take_start": start, "take_end": end})


def test_the_stills_time_names_every_still_not_about_the_middle():
    takes = [_still_take("a", 1, 2.5), _still_take("a", 2, 2.6),
             _still_take("b", 1, 3.3), _still_take("c", 2, 1.5)]
    gs = types.SimpleNamespace(takes=takes)
    assert [p for _t, p in gh.still_shares(gs)] == [50, 52, 66, 30]
    assert gh._still_share_text(gs) == (
        "; that is 30 to 66 % of the way through its recording (from the "
        "recorder's start of it to its end): about the middle (40 to 60 %) "
        "in 2 of the 4, earlier in 1: c take 2 (30 %); later in 1: b take 1 "
        "(66 %)")
    gs.takes = takes[:2]
    assert gh._still_share_text(gs).endswith(
        "50 to 52 % of the way through its recording (from the recorder's "
        "start of it to its end): about the middle (40 to 60 %) in all 2")
    # the share is of each take's own recording, from its meta
    gs.takes = [_still_take("a", 1, 2.5, end=110.0)]
    assert [p for _t, p in gh.still_shares(gs)] == [25]
    # no start or end in the meta, or no still: no share, and no text
    gs.takes = [_still_take("a", 1, 2.5)]
    gs.takes[0].meta = {"take_start": 100.0}
    assert gh.still_shares(gs) == [] and gh._still_share_text(gs) == ""


def _day_take(item, n, day):
    return types.SimpleNamespace(item=item, ref=types.SimpleNamespace(take=n),
                                 recorded_day=day)


def _coached(item, n):
    return types.SimpleNamespace(coached=True, to_grasp=item, take=n)


def test_the_thresholds_say_which_recordings_they_were_read_from():
    read_on = gh.tr.THRESHOLDS_READ_ON
    gs = types.SimpleNamespace(takes=[
        _day_take("a", 1, "2026-10-02"), _day_take("b", 1, "2026-10-01"),
        _day_take("a", 2, "2026-10-07"), _day_take("b", 2, "2026-10-08")])
    early = [_coached("a", 1)]
    assert gh._thresholds_origin(gs, early) == (
        "The thresholds were read from these same recordings and have been "
        "tried on no others.")
    late = [_coached("a", 1), _coached("a", 2), _coached("b", 2)]
    assert [m.take for m in gh._after_thresholds(gs, late)] == [2, 2]
    assert gh._thresholds_origin(gs, late) == (
        f"The thresholds were read on {read_on} from the recordings made by "
        "then: 1 recording of these (2026-10-02) and the held poses of the 2 "
        "takes kept by then. The other 2 here (2026-10-07, 2026-10-08) were "
        "recorded after that; the thresholds have been tried on no "
        "recordings besides these.")
    assert gh._thresholds_origin(gs, late[1:2]).endswith(
        "The other 1 here (2026-10-07) was recorded after that; the "
        "thresholds have been tried on no recordings besides these.")
    assert gh._move_day(gs, _coached("b", 2)) == "2026-10-08"
    assert gh._move_day(gs, types.SimpleNamespace(
        coached=False, to_grasp="b", take=2)) is None


# --- a mock transition session says so, and the check refuses it ----------------
@pytest.fixture(scope="module")
def mock_rows(tmp_path_factory):
    """The validator tests' mock Set A session (`make_set_a`, a folder that
    passes its check), a mock transition session recorded beside it by the
    real recorder, and the folder the unchanged rebuild command makes of
    both with --allow-mock, finding the transition session by itself."""
    import test_validate_grasp_set as tv
    root = tmp_path_factory.mktemp("mock_rows")
    grasps = tv.make_set_a(root)
    beside = rb.default_transitions_dir(grasps)
    assert rt.main(["--start-hold", "0.6", "--cue-wait", "0.3", "0.5",
                    "--hold", "0.6", "--seed", "1", "--quiet", "--mock",
                    "--hand", "left", "--protocol",
                    str(root / "grasps_fixture.json"), "--pairs",
                    "cylindrical>tip_pinch", "--takes", "1", "--out",
                    str(beside)]) == 0
    (recorded,) = [p for p in beside.iterdir() if p.is_dir()]
    out = root / "out" / "grasp set"
    rc = rb.main(["--session", str(grasps), "--out", str(out), "--operator",
                  "T Est", "--allow-mock"])
    return {"rc": rc, "out": out, "grasps": grasps, "recorded": recorded}


def test_a_mock_transition_session_says_mock_and_is_refused_unless_allowed(
        mock_rows, tmp_path):
    out, recorded = mock_rows["out"], mock_rows["recorded"]
    assert mock_rows["rc"] == 0                  # its own check was allowed
    # the README's banner, above the operator line, names the session
    lines = (out / "README.txt").read_text(encoding="utf-8").splitlines()
    top = " ".join(lines[:next(k for k, x in enumerate(lines)
                               if x.startswith("Operator:"))])
    assert "MOCK DATA" in top and recorded.name in top
    assert mock_rows["grasps"].name in top       # the grasp session is mock too
    # every line fits the page (a path aside), the header's too
    assert not [x for x in lines if len(x) > gh.WIDTH
                and ":\\" not in x and ":/" not in x]
    op = next(k for k, x in enumerate(lines) if x.startswith("Operator:"))
    head = " ".join([lines[op]] + [x.strip() for x in lines[op + 1:op + 3]
                                   if x.startswith("    ")])
    assert "Operator: T Est" in head and " Recorded: " in head
    # its row's notes, and its line in section 17
    with open(out / gh.TRANSITIONS_DIR / gh.TRANSITIONS_FILE,
              encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mine = [r for r in rows if r["session"] == recorded.name]
    assert [(r["from_grasp"], r["to_grasp"], r["validity"]) for r in mine] == [
        ("cylindrical", "tip_pinch", "valid")]
    assert mine[0]["notes"].startswith("MOCK (synthetic hand)")
    # the grasp session's own forming rows are not transition-session takes
    assert not [r for r in rows if r not in mine and "MOCK" in r["notes"]]
    flat = " ".join(" ".join(lines).split())
    s17 = flat.split("17. POSE TRANSITION TIMING", 2)[-1].split(
        "18. CURRENT DATASET COMPLETENESS")[0]
    assert "MOCK (synthetic hand): cylindrical to tip_pinch, take 1: valid" \
        in s17
    assert "1 of them is MOCK" in s17
    # the check: refused for the MOCK row alone, passed with --allow-mock
    rep = gv.validate(out, mock_rows["grasps"])
    (error,) = rep.errors
    assert error.startswith(f"1 row of {gh.TRANSITIONS_FILE} says MOCK")
    assert "cylindrical to tip_pinch take 1" in error
    allowed = gv.validate(out, mock_rows["grasps"], allow_mock=True)
    assert allowed.errors == [] and allowed.facts["mock_transitions"] == 1
    assert (out / gh.VALIDATION_FILE).read_text(
        encoding="utf-8").rstrip().endswith("FOLDER CHECK: PASSED")
    cli = load("scripts/validate_grasp_set.py", "validate_cli_mock_rows")
    assert cli.main([str(out)]) == 1
    assert cli.main([str(out), "--allow-mock"]) == 0
    # a copy whose README lost its MOCK DATA line fails even when allowed
    copy = tmp_path / "copy"
    shutil.copytree(gh.long_path(out), gh.long_path(copy))
    readme = copy / "README.txt"
    readme.write_text(readme.read_text(encoding="utf-8").replace(
        "MOCK DATA", "DATA"), encoding="utf-8")
    (error,) = gv.validate(copy, allow_mock=True).errors
    assert error.startswith("README.txt does not say MOCK DATA at its top")


# --- the Desktop launchers ---------------------------------------------------
DESKTOP = Path(r"C:\Users\nkim2\OneDrive\Desktop")
CD = r'cd /d "C:\Users\nkim2\OneDrive\Desktop\non glove xr trainer"'
RECORDER = r".venv\Scripts\python.exe scripts\leap\record_transitions.py "
LAUNCHERS = {
    "rehearse_transitions.cmd":
        "Rehearsal of the transition recorder (no hardware): mock camera",
    "record_transitions_left.cmd":
        "Transitions, camera only, bare LEFT hand"}
GRASP_IDS = [it["id"] for it in json.loads(
    (REPO / "protocols" / "grasps.json").read_text(encoding="utf-8"))["items"]]
on_desktop = pytest.mark.skipif(
    not all((DESKTOP / n).is_file() for n in LAUNCHERS),
    reason="the Desktop launchers are on the recording computer only")


def launcher_lines(name: str):
    data = (DESKTOP / name).read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf"), "a BOM"
    assert data.endswith(b"\r\n") and data.count(b"\n") == data.count(
        b"\r\n"), "not CRLF"
    return data.decode("ascii")[:-2].split("\r\n")


def recorder_args(lines):
    (line,) = [x for x in lines if x.startswith(RECORDER)]
    return shlex.split(line[len(RECORDER):])


@on_desktop
@pytest.mark.parametrize("name", sorted(LAUNCHERS))
def test_a_launcher_names_its_setup_and_runs_the_recorder(name):
    lines = launcher_lines(name)
    assert lines[:2] == ["@echo off", CD]
    assert lines[2].startswith(f"echo {LAUNCHERS[name]}")
    assert lines[-3:] == ["echo.", "echo Finished. This window can be closed.",
                          "pause"]
    assert (REPO / "scripts" / "leap" / "record_transitions.py").is_file()
    args = recorder_args(lines)
    assert args[args.index("--hand") + 1] == "left"
    assert args[args.index("--operator") + 1] == "N Kim"


@on_desktop
def test_the_rehearsal_is_a_mock_session_of_pairs_the_recorder_reads():
    args = recorder_args(launcher_lines("rehearse_transitions.cmd"))
    assert "--mock" in args and args[args.index("--takes") + 1] == "2"
    assert rt.parse_pairs(args[args.index("--pairs") + 1], GRASP_IDS) == [
        ("open_hand", "p1_cylindrical"), ("p1_cylindrical", "p1_spherical")]


@on_desktop
def test_the_recording_launcher_takes_its_pairs_from_the_command_line():
    lines = launcher_lines("record_transitions_left.cmd")
    args = recorder_args(lines)
    assert "--mock" not in args and args[-2:] == ["--pairs", "%*"]
    assert args[args.index("--takes") + 1] == "3"
    # with no argument: two example lines, then out before the recorder
    start = lines.index('if "%~1"=="" (')
    end = lines.index(")", start)
    assert end < lines.index(next(x for x in lines if x.startswith(RECORDER)))
    block = lines[start + 1:end]
    assert "  exit /b 1" in block
    examples = [x.rsplit("'\"", 1)[1].rstrip("\"'") for x in block
                if x.endswith("\"'")]
    assert len(examples) == 2
    # the pairs in single quotes around double quotes: Windows PowerShell
    # drops plain double quotes, and cmd then reads ">" as a redirection
    assert all(f"'\"{e}\"'" in "\n".join(block) for e in examples)
    dry, between = (rt.parse_pairs(e, GRASP_IDS) for e in examples)
    assert dry == [("open_hand", "p1_cylindrical")]
    assert len(between) >= 2 and not any(rt.OPEN_HAND in p for p in between)


@on_desktop
@pytest.mark.skipif(sys.platform != "win32", reason="a Windows launcher")
def test_the_recording_launcher_without_pairs_records_nothing():
    done = subprocess.run(
        ["cmd", "/d", "/c", str(DESKTOP / "record_transitions_left.cmd")],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    out = done.stdout + done.stderr
    assert done.returncode == 1
    assert "Nothing was recorded" in out and "open_hand>p1_cylindrical" in out
    assert "Finished" not in out
