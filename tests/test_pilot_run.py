"""`scripts/pilot_run.py`: one command per step of docs/pilot_hand_on_object.md.

Gate R1 of GATES_pilot_run.md. The recorder is never started here:
`subprocess.run` is replaced by a fake that records argv and cwd, prints a
marker, writes the accepted takes a recorder call asks for (fewer where a
test says a grasp falls short), and returns 0 (or the exit codes a test
sets). Every test uses `tmp_path` as
`--root`, or points the script's REPO at `tmp_path` to pin the lines printed
for the default `recordings\\pilot_hoo`.
"""
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "pilot_hand_on_object.md"
GRASPS = ("cylindrical_object", "spherical_object", "tip_object")
OBJECT_RUNS = ("B1", "C", "B2")

# run, folder, takes per grasp, hint, note before the grasp and glove text
RUNS = [
    ("A", "A_mimed", 5, False, "condition A mimed, no object"),
    ("B1", "B1_object", 3, False,
     "condition B1 object: OBJ-CYL, OBJ-BALL, OBJ-PEN as in objects.txt"),
    ("C", "C_object_hint", 5, True,
     "condition C object with hand_on_object: OBJ-CYL, OBJ-BALL, OBJ-PEN as "
     "in objects.txt"),
    ("B2", "B2_object", 2, False,
     "condition B2 object after C: OBJ-CYL, OBJ-BALL, OBJ-PEN as in "
     "objects.txt"),
    ("A2", "A_mimed_again", 1, False,
     "condition A repeated at the end, drift check"),
]
FOLDER = {r[0]: r[1] for r in RUNS}
EXPECTED = {r[0]: r[2] for r in RUNS}
NOTE = {r[0]: r[4] for r in RUNS}
ORDER = [r[0] for r in RUNS]

OBJECTS = (
    "# id | what it is | diameter_mm | length_mm | mass_g\n"
    "\n"
    "OBJ-CYL | tin of tomatoes | 66 | 110 | 420\n"
    "   \n"
    "OBJ-BALL | tennis ball | 67 | 67 | 58.5\n"
    "# the pen was not weighed\n"
    "OBJ-PEN | blue ballpoint pen | 8.5 | 145 | not measured\n"
)
OBJECT_LINES = [ln for ln in OBJECTS.splitlines() if ln.startswith("OBJ-")]
OBJECT_OF = dict(zip(GRASPS, OBJECT_LINES))      # CYL, BALL, PEN


def load_pilot_run():
    path = REPO / "scripts" / "pilot_run.py"
    spec = importlib.util.spec_from_file_location(
        "repo_script_pilot_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pr():
    return load_pilot_run()


class Calls(list):
    """(argv, kwargs) of every subprocess.run call; `codes` are the exit
    codes returned in turn, 0 after them; `short` {grasp: k} makes the
    recorder keep k takes fewer than asked for that grasp (as when the
    operator quits with q)."""

    def __init__(self):
        super().__init__()
        self.codes, self.short = (), {}


@pytest.fixture
def calls(pr, monkeypatch):
    got = Calls()

    def fake_run(argv, **kwargs):
        argv = list(argv)
        got.append((argv, kwargs))
        if "--poses" in argv:            # the recorder: keep the takes
            arg = lambda name: argv[argv.index(name) + 1]   # noqa: E731
            grasp, takes, first = (arg("--poses"), int(arg("--takes")),
                                   int(arg("--first-take")))
            leap = Path(kwargs["cwd"]) / arg("--out-dir") / "leap"
            leap.mkdir(parents=True, exist_ok=True)
            for take in range(first,
                              first + takes - got.short.get(grasp, 0)):
                stem = f"{grasp}_left_take{take}_20261022_{len(got):02d}0000"
                (leap / f"{stem}.meta.json").write_text(json.dumps(
                    {"accepted": True, "pose": grasp, "take": take}),
                    encoding="utf-8")
            print(f"[recorder {grasp}]")
        else:
            print("[recorder]")
        n = len(got)
        return SimpleNamespace(
            returncode=got.codes[n - 1] if n <= len(got.codes) else 0)

    monkeypatch.setattr(pr.subprocess, "run", fake_run)
    return got


def take_meta(root: Path, run: str, pose="cylindrical_object", take=1,
              accepted=True, rejected=False):
    leap = root / FOLDER[run] / ("rejected/leap" if rejected else "leap")
    leap.mkdir(parents=True, exist_ok=True)
    stem = f"{pose}_left_take{take}_20261021_10{take:04d}"
    (leap / f"{stem}.meta.json").write_text(
        json.dumps({"accepted": accepted, "pose": pose, "take": take}),
        encoding="utf-8")


def fill(root: Path, run: str, counts=None, accepted=True):
    """Kept takes 1..n per grasp in `run`'s folder, n the protocol's unless
    `counts` says otherwise."""
    counts = counts or {g: EXPECTED[run] for g in GRASPS}
    for grasp, n in counts.items():
        for take in range(1, n + 1):
            take_meta(root, run, grasp, take, accepted=accepted)


def lock_text(lines=OBJECT_LINES) -> str:
    body = "\n".join(lines)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"sha256 {digest}\n{body}\n"


def prepare(root: Path, run: str, gloves="left", objects=OBJECTS):
    """A pilot folder ready for `run`: the glove setup, objects.txt, every
    earlier run complete, and objects.lock once B1 is behind (B1's start
    writes it)."""
    root.mkdir(parents=True, exist_ok=True)
    if gloves is not None:
        (root / "glove_setup.txt").write_text(gloves + "\n", encoding="utf-8")
    if objects is not None:
        (root / "objects.txt").write_text(objects, encoding="utf-8")
    for earlier in ORDER[:ORDER.index(run)]:
        fill(root, earlier)
    if ORDER.index(run) > ORDER.index("B1"):
        (root / "objects.lock").write_text(lock_text(), encoding="utf-8")


def note_of(run, grasp, gloves="left glove only"):
    middle = (f"grasp {grasp} with {OBJECT_OF[grasp]}" if run in OBJECT_RUNS
              else f"grasp {grasp}, nothing in the hand")
    return f"{NOTE[run]}; {middle}; {gloves}"


def expected_argv(root, run, grasp, takes, first):
    argv = [sys.executable, "scripts\\record_simultaneous.py",
            "--camera", "leap", "--hand", "left", "--poses", grasp,
            "--takes", str(takes), "--first-take", str(first)]
    if run in OBJECT_RUNS:                 # the amendment of 2026-10-23
        argv += ["--keep-all-tracking"]
    if run == "C":
        argv += ["--hint", "hand_on_object"]
    return argv + ["--out-dir", str(Path(root) / FOLDER[run]),
                   "--note", note_of(run, grasp)]


def shell_line(root, run, grasp, takes, first):
    return (".venv\\Scripts\\python.exe scripts\\record_simultaneous.py "
            f"--camera leap --hand left --poses {grasp} --takes {takes} "
            f"--first-take {first}"
            + (" --keep-all-tracking" if run in OBJECT_RUNS else "")
            + (" --hint hand_on_object" if run == "C" else "")
            + f" --out-dir {Path(root) / FOLDER[run]}"
            + f' --note "{note_of(run, grasp)}"')


def main(pr, *argv):
    return pr.main([str(a) for a in argv])


# --- 1. the commands of every run ----------------------------------------------
@pytest.mark.parametrize("run", ORDER)
def test_every_run_records_each_grasp_with_its_exact_command(
        pr, calls, capsys, tmp_path, run):
    prepare(tmp_path, run)
    n = EXPECTED[run]
    assert main(pr, run, "--root", tmp_path) == 0
    out = capsys.readouterr().out
    assert [c[0] for c in calls] == [expected_argv(tmp_path, run, g, n, 1)
                                     for g in GRASPS]
    assert all(Path(kwargs["cwd"]) == REPO for _, kwargs in calls)
    lines = out.splitlines()
    assert lines[:3] == [shell_line(tmp_path, run, g, n, 1) for g in GRASPS]
    assert lines[3] == ""
    assert "Gloves on + camera." in out
    assert f"{3 * n} takes ({n} per grasp)" in out
    assert "Right hand on the table, outside the camera's view." in out
    assert "Do not recalibrate the glove." in out
    # each grasp announced (with its object) just before its own recording
    at = []
    for g in GRASPS:
        block = lines.index(f"NEXT GRASP: {g} ({n} take{'s' * (n != 1)}, "
                            "numbered from 1)")
        if run in OBJECT_RUNS:
            ident = OBJECT_OF[g].split(" |")[0]
            assert lines[block + 1] == f"USE {ident}: {OBJECT_OF[g]}"
        at += [block, lines.index(f"[recorder {g}]")]
    assert at == sorted(at)
    if run in OBJECT_RUNS:
        assert "Hold, do not squeeze." in out
        for line in OBJECT_LINES:
            assert f"  {line}" in lines
    else:
        assert "Nothing in the hand" in out and "USE OBJ" not in out
    assert ("asked for hand_on_object" in out) == (run == "C")
    assert f"run {run} now: " in out


def options(line: str):
    """A recorder line's words before --note without --poses and
    --first-take, its --poses, and its note."""
    head, _, note = line.strip().partition(' --note "')
    words, kept, poses = head.split(), [], None
    i = 0
    while i < len(words):
        if words[i] in ("--poses", "--first-take"):
            poses = words[i + 1] if words[i] == "--poses" else poses
            i += 2
        else:
            kept.append(words[i])
            i += 1
    return kept, poses, note[:-1]


@pytest.mark.parametrize("run", ORDER)
def test_every_command_is_the_protocols_step(pr, calls, capsys, tmp_path,
                                             monkeypatch, run):
    """With the default root, the three lines of a fresh run are the
    protocol's step command split by grasp: the same options, its --poses
    one grasp each, its note extended by the grasp and the glove setup."""
    monkeypatch.setattr(pr, "REPO", tmp_path)
    prepare(tmp_path / "recordings" / "pilot_hoo", run)
    assert main(pr, run, "--print") == 0
    lines = capsys.readouterr().out.splitlines()[:3]
    (doc_line,) = [ln for ln in DOC.read_text(encoding="utf-8").splitlines()
                   if f"--out-dir recordings\\pilot_hoo\\{FOLDER[run]} " in ln]
    doc_options, doc_poses, doc_note = options(doc_line)
    assert doc_poses == ",".join(GRASPS)
    got = [options(ln) for ln in lines]
    assert [g[1] for g in got] == list(GRASPS)
    flag = "--keep-all-tracking"           # the object runs, from 2026-10-23
    for (opts, _, note), grasp in zip(got, GRASPS):
        assert (flag in opts) == (run in OBJECT_RUNS)
        # the protocol document may not show the amendment's flag yet
        if flag not in doc_options:
            opts = [w for w in opts if w != flag]
        assert opts == doc_options
        assert note.startswith(doc_note.split(";")[0] + f"; grasp {grasp}")
        assert note.endswith("; left glove only")


def test_default_root_lines_exactly(pr, calls, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "REPO", tmp_path)
    root = tmp_path / "recordings" / "pilot_hoo"
    prepare(root, "A")
    assert main(pr, "A", "--print") == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        r".venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap"
        r" --hand left --poses cylindrical_object --takes 5 --first-take 1"
        r" --out-dir recordings\pilot_hoo\A_mimed --note "
        r'"condition A mimed, no object; grasp cylindrical_object, nothing in'
        r' the hand; left glove only"')
    fill(root, "A")
    fill(root, "B1")
    (root / "objects.lock").write_text(lock_text(), encoding="utf-8")
    assert main(pr, "C", "--print") == 0
    assert capsys.readouterr().out.splitlines()[2] == (
        r".venv\Scripts\python.exe scripts\record_simultaneous.py --camera leap"
        r" --hand left --poses tip_object --takes 5 --first-take 1"
        r" --keep-all-tracking"
        r" --hint hand_on_object --out-dir recordings\pilot_hoo\C_object_hint"
        r' --note "condition C object with hand_on_object: OBJ-CYL, OBJ-BALL,'
        r" OBJ-PEN as in objects.txt; grasp tip_object with OBJ-PEN | blue"
        r' ballpoint pen | 8.5 | 145 | not measured; left glove only"')
    assert calls == []


# --- 2. --print ------------------------------------------------------------------
def test_print_runs_nothing_and_writes_no_marker(pr, calls, capsys, tmp_path):
    prepare(tmp_path, "B1")
    assert main(pr, "B1", "--print", "--root", tmp_path) == 0
    assert capsys.readouterr().out.splitlines() == [
        shell_line(tmp_path, "B1", g, 3, 1) for g in GRASPS]
    assert calls == []
    assert not (tmp_path / "runs_started.txt").exists()
    assert not (tmp_path / "objects.lock").exists()


# --- 3. the glove setup ----------------------------------------------------------
def test_no_glove_setup_refuses_and_names_the_option(pr, calls, capsys,
                                                     tmp_path):
    prepare(tmp_path, "A", gloves=None)
    assert main(pr, "A", "--root", tmp_path) == 2
    out = capsys.readouterr().out
    assert "--gloves left" in out and "--gloves both" in out
    assert "glove_setup.txt" in out
    assert calls == []
    assert not (tmp_path / "glove_setup.txt").exists()


def test_gloves_written_once_and_kept(pr, calls, capsys, tmp_path):
    root = tmp_path / "fresh"          # does not exist yet
    assert main(pr, "A", "--gloves", "left", "--root", root) == 0
    assert (root / "glove_setup.txt").read_text(encoding="utf-8").strip() \
        == "left"
    assert calls[-1][0][-1].endswith("; left glove only")
    # the same setup again is fine (A recorded afresh)
    shutil.rmtree(root / "A_mimed")
    assert main(pr, "A", "--gloves", "left", "--root", root) == 0
    # a different one is refused, and the file is unchanged
    capsys.readouterr()
    n = len(calls)
    assert main(pr, "A", "--gloves", "both", "--root", root) == 2
    out = capsys.readouterr().out
    assert "one glove setup" in out
    assert '"left"' in out and '"both"' in out
    assert len(calls) == n
    assert (root / "glove_setup.txt").read_text(encoding="utf-8").strip() \
        == "left"


def test_both_gloves_note(pr, calls, tmp_path):
    prepare(tmp_path, "B1", gloves=None)
    assert main(pr, "B1", "--gloves", "both", "--root", tmp_path) == 0
    assert calls[0][0][-1] == note_of("B1", GRASPS[0], "both gloves connected")
    assert main(pr, "C", "--root", tmp_path) == 0      # B1 is complete
    assert calls[-1][0][-1] == note_of("C", GRASPS[2], "both gloves connected")


def test_glove_setup_file_with_another_word_refuses(pr, calls, capsys,
                                                    tmp_path):
    prepare(tmp_path, "A", gloves="right")
    assert main(pr, "A", "--root", tmp_path) == 2
    assert "glove_setup.txt" in capsys.readouterr().out
    assert calls == []


def test_refused_run_writes_nothing(pr, calls, tmp_path):
    prepare(tmp_path, "B1", gloves=None, objects=None)
    assert main(pr, "B1", "--gloves", "left", "--root", tmp_path) == 2
    for name in ("glove_setup.txt", "runs_started.txt", "objects.lock"):
        assert not (tmp_path / name).exists(), name


# --- 4. objects.txt --------------------------------------------------------------
@pytest.mark.parametrize("run", OBJECT_RUNS)
def test_object_runs_need_objects_txt(pr, calls, capsys, tmp_path, run):
    prepare(tmp_path, run, objects=None)
    assert main(pr, run, "--root", tmp_path) == 2
    out = capsys.readouterr().out
    assert "objects.txt" in out and "not measured" in out
    assert calls == []


@pytest.mark.parametrize("run", ["A", "A2"])
def test_mimed_runs_do_not_need_objects_txt(pr, calls, tmp_path, run):
    prepare(tmp_path, run, objects=None)
    assert main(pr, run, "--root", tmp_path) == 0
    assert len(calls) == 3


def bad_objects(pr, capsys, tmp_path, text):
    prepare(tmp_path, "B1", objects=text)
    rc = main(pr, "B1", "--root", tmp_path)
    return rc, capsys.readouterr().out


def test_missing_identifier(pr, calls, capsys, tmp_path):
    text = OBJECTS.replace(OBJECT_LINES[2] + "\n", "")
    rc, out = bad_objects(pr, capsys, tmp_path, text)
    assert rc == 2 and "OBJ-PEN" in out and calls == []


def test_line_with_four_fields(pr, calls, capsys, tmp_path):
    text = OBJECTS.replace("OBJ-BALL | tennis ball | 67 | 67 | 58.5",
                           "OBJ-BALL | tennis ball | 67 | 58.5")
    rc, out = bad_objects(pr, capsys, tmp_path, text)
    assert rc == 2 and "OBJ-BALL" in out and calls == []


@pytest.mark.parametrize("old,new,word", [
    ("| 66 | 110 |", "| abc | 110 |", "diameter"),
    ("| 66 | 110 |", "| 66 | 0 |", "length"),
    ("| 66 | 110 |", "| 66 | -3 |", "length"),
    ("| 110 | 420", "| 110 | heavy", "mass"),
])
def test_bad_numbers(pr, calls, capsys, tmp_path, old, new, word):
    rc, out = bad_objects(pr, capsys, tmp_path, OBJECTS.replace(old, new))
    assert rc == 2 and word in out and "OBJ-CYL" in out and calls == []


def test_unknown_or_repeated_identifier(pr, calls, capsys, tmp_path):
    rc, out = bad_objects(pr, capsys, tmp_path,
                          OBJECTS + "OBJ-MUG | mug | 80 | 95 | 300\n")
    assert rc == 2 and "OBJ-MUG" in out
    rc, out = bad_objects(pr, capsys, tmp_path,
                          OBJECTS + "OBJ-PEN | red pen | 9 | 140 | 12\n")
    assert rc == 2 and "OBJ-PEN" in out
    assert calls == []


def test_objects_txt_saved_with_a_bom(pr, calls, tmp_path):
    """Notepad may save with a byte-order mark before OBJ-CYL."""
    prepare(tmp_path, "B1", objects=None)
    (tmp_path / "objects.txt").write_text(
        "\n".join(OBJECT_LINES) + "\n", encoding="utf-8-sig")
    assert main(pr, "B1", "--root", tmp_path) == 0


# --- 5. objects.lock -------------------------------------------------------------
def test_objects_are_locked_when_B1_starts(pr, calls, capsys, tmp_path):
    prepare(tmp_path, "B1")
    assert main(pr, "B1", "--root", tmp_path) == 0
    lock = tmp_path / "objects.lock"
    assert lock.read_text(encoding="utf-8") == lock_text()
    # spacing, comments, blank lines and the order of the lines may change
    respaced = ("# measured again, same objects\n\n"
                "OBJ-PEN|blue ballpoint pen|8.5|145|not measured\n"
                "  OBJ-CYL  |  tin of tomatoes |66 | 110|420  \n\n"
                "OBJ-BALL | tennis ball | 67 | 67 | 58.5\n")
    (tmp_path / "objects.txt").write_text(respaced, encoding="utf-8")
    assert main(pr, "C", "--root", tmp_path) == 0
    # a changed measurement refuses C and B2
    (tmp_path / "objects.txt").write_text(
        OBJECTS.replace("| 110 | 420", "| 110 | 400"), encoding="utf-8")
    capsys.readouterr()
    n = len(calls)
    assert main(pr, "C", "--root", tmp_path) == 2
    out = capsys.readouterr().out
    assert "objects.txt changed since B1 started (objects.lock)" in out
    assert "new --root" in out
    assert len(calls) == n
    assert lock.read_text(encoding="utf-8") == lock_text()


def test_B1_top_up_is_checked_against_its_lock(pr, calls, capsys, tmp_path):
    prepare(tmp_path, "B1")
    calls.short = {"tip_object": 1}
    assert main(pr, "B1", "--root", tmp_path) == 3        # lock written
    (tmp_path / "objects.txt").write_text(
        OBJECTS.replace("tennis ball", "squash ball"), encoding="utf-8")
    assert main(pr, "B1", "--root", tmp_path) == 2
    assert "changed since B1 started" in capsys.readouterr().out


def test_C_without_a_lock_is_refused(pr, calls, capsys, tmp_path):
    prepare(tmp_path, "C")
    (tmp_path / "objects.lock").unlink()
    assert main(pr, "C", "--root", tmp_path) == 2
    assert "objects.lock" in capsys.readouterr().out
    assert calls == []


# --- 6. order, completion, top-up -----------------------------------------------
def test_run_before_its_predecessor_is_complete_is_refused(pr, calls, capsys,
                                                          tmp_path):
    prepare(tmp_path, "A")
    fill(tmp_path, "A", {"cylindrical_object": 5, "spherical_object": 3})
    assert main(pr, "B1", "--root", tmp_path) == 2
    out = capsys.readouterr().out
    assert "run B1 comes after A; " in out
    assert ("A_mimed is incomplete (cylindrical_object 5, spherical_object 3, "
            "tip_object 0, needs 5 per grasp)") in out
    assert calls == []


def test_every_unmet_precondition_is_named_in_order(pr, calls, capsys,
                                                    tmp_path):
    """An empty pilot folder: B1 names the glove setup, objects.txt and the
    order, one paragraph each, in that order."""
    assert main(pr, "B1", "--print", "--root", tmp_path) == 2
    paragraphs = capsys.readouterr().out.strip().split("\n\n")
    assert len(paragraphs) == 3
    assert paragraphs[0].startswith("no glove setup yet")
    assert paragraphs[1].startswith(f"no {tmp_path / 'objects.txt'} yet")
    assert paragraphs[2].startswith("run B1 comes after A")
    assert calls == []


def test_rejected_takes_do_not_complete_a_run(pr, calls, capsys, tmp_path):
    prepare(tmp_path, "A")
    fill(tmp_path, "A", accepted=False)
    assert main(pr, "B1", "--root", tmp_path) == 2
    assert "tip_object 0, needs 5 per grasp" in capsys.readouterr().out


def test_complete_run_is_refused(pr, calls, capsys, tmp_path):
    prepare(tmp_path, "B1")                        # A complete, B1 not begun
    assert main(pr, "A", "--root", tmp_path) == 2
    out = capsys.readouterr().out
    assert "run A is complete (" in out
    assert "nothing to record. Takes beyond the protocol are not added." in out
    assert calls == []


def test_run_after_a_later_run_has_started_is_refused(pr, calls, capsys,
                                                      tmp_path):
    prepare(tmp_path, "A")
    fill(tmp_path, "A", {"cylindrical_object": 2})
    # a rejected attempt alone starts B1
    take_meta(tmp_path, "B1", accepted=False, rejected=True)
    assert main(pr, "A", "--root", tmp_path) == 2
    said = ("run A cannot be recorded after B1 has started: the protocol's "
            "order is A, B1, C, B2, A2")
    assert said in capsys.readouterr().out
    # so does its line in runs_started.txt
    shutil.rmtree(tmp_path / "B1_object")
    (tmp_path / "runs_started.txt").write_text(
        "A 2026-10-21T10:00:00-04:00\nB1 2026-10-21T10:30:00-04:00\n",
        encoding="utf-8")
    assert main(pr, "A", "--root", tmp_path) == 2
    assert said in capsys.readouterr().out
    assert calls == []
    # with neither, the short run A is topped up
    (tmp_path / "runs_started.txt").unlink()
    assert main(pr, "A", "--root", tmp_path) == 0


def test_top_up_records_only_the_missing_takes(pr, calls, capsys, tmp_path):
    prepare(tmp_path, "A")
    fill(tmp_path, "A", {"cylindrical_object": 5, "spherical_object": 3,
                         "tip_object": 5})
    assert main(pr, "A", "--root", tmp_path) == 0
    assert [c[0] for c in calls] == [
        expected_argv(tmp_path, "A", "spherical_object", 2, 4)]
    out = capsys.readouterr().out
    assert "NEXT GRASP: spherical_object (2 takes, numbered from 4)" in out
    assert "2 of the 15 takes still missing" in out


def test_top_up_numbers_on_from_the_highest_kept_take(pr, calls, tmp_path):
    """Take 3 failed every retry (only a rejected meta); takes 1, 2 and 4 are
    kept: the missing two are numbered 5 and 6, never a second take 4."""
    prepare(tmp_path, "A")
    fill(tmp_path, "A", {"cylindrical_object": 5, "tip_object": 5})
    for take in (1, 2, 4):
        take_meta(tmp_path, "A", "spherical_object", take)
    take_meta(tmp_path, "A", "spherical_object", 3, accepted=False,
              rejected=True)
    assert main(pr, "A", "--root", tmp_path) == 0
    assert [c[0] for c in calls] == [
        expected_argv(tmp_path, "A", "spherical_object", 2, 5)]


def test_runs_started_gets_one_line_per_run_start(pr, calls, tmp_path):
    def labels():
        path = tmp_path / "runs_started.txt"
        rows = path.read_text(encoding="utf-8").splitlines()
        for row in rows:
            label, stamp = row.split(" ")
            assert datetime.fromisoformat(stamp).tzinfo is not None
        return [row.split(" ")[0] for row in rows]

    calls.short = {"tip_object": 1}
    assert main(pr, "A", "--gloves", "left", "--root", tmp_path) == 3
    assert labels() == ["A"]
    assert len(calls) == 3                      # one line, three grasps
    assert main(pr, "A", "--print", "--root", tmp_path) == 0
    assert labels() == ["A"]
    calls.short = {}
    assert main(pr, "A", "--root", tmp_path) == 0        # a top-up
    assert labels() == ["A", "A"]
    assert main(pr, "B1", "--root", tmp_path) == 2       # no objects.txt
    assert labels() == ["A", "A"]


def test_the_sequence_stops_at_the_first_failing_grasp(pr, calls, capsys,
                                                       tmp_path):
    prepare(tmp_path, "A")
    calls.codes = (0, 5)
    assert main(pr, "A", "--root", tmp_path) == 5
    assert len(calls) == 2
    assert "run A now" not in capsys.readouterr().out


def test_a_grasp_left_short_stops_the_run(pr, calls, capsys, tmp_path):
    """q in the recorder exits 0 with the grasp short: the run stops there
    instead of moving on, and the next run of the step tops it up first."""
    prepare(tmp_path, "A")
    calls.short = {"spherical_object": 1}
    assert main(pr, "A", "--root", tmp_path) == 3
    assert [c[0] for c in calls] == [
        expected_argv(tmp_path, "A", "cylindrical_object", 5, 1),
        expected_argv(tmp_path, "A", "spherical_object", 5, 1)]
    out = capsys.readouterr().out
    assert ("spherical_object is still short (4/5): stopped here so the "
            "grasps stay in order; run this step again to top it up.") in out
    assert "[recorder tip_object]" not in out
    assert "run A now" not in out
    calls.short = {}
    assert main(pr, "A", "--root", tmp_path) == 0
    assert [c[0] for c in calls[2:]] == [
        expected_argv(tmp_path, "A", "spherical_object", 1, 5),
        expected_argv(tmp_path, "A", "tip_object", 5, 1)]
    assert "run A now: cylindrical_object 5/5, spherical_object 5/5, " \
           "tip_object 5/5; complete." in capsys.readouterr().out


# --- 7. compare and close --------------------------------------------------------
def test_compare_is_refused_until_A2_is_complete(pr, calls, capsys, tmp_path):
    prepare(tmp_path, "A2")
    fill(tmp_path, "A2", {"cylindrical_object": 1})
    assert main(pr, "compare", "--root", tmp_path) == 2
    out = capsys.readouterr().out
    assert ("the pilot is not finished: A2 is incomplete (cylindrical_object "
            "1, spherical_object 0, tip_object 0, needs 1 per grasp)") in out
    assert 'pilot_run.py close --reason "<why>"' in out
    assert calls == []


def test_compare_runs_once_A2_is_complete(pr, calls, capsys, tmp_path):
    fill(tmp_path, "A2")             # no glove setup or objects.txt needed
    assert main(pr, "compare", "--root", tmp_path) == 0
    (argv, kwargs), = calls
    assert argv == [sys.executable, "scripts\\pilot_compare.py",
                    "--root", str(tmp_path)]
    assert Path(kwargs["cwd"]) == REPO
    assert capsys.readouterr().out.splitlines()[0] == (
        f".venv\\Scripts\\python.exe scripts\\pilot_compare.py --root "
        f"{tmp_path}")


def test_compare_print_default_root(pr, calls, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "REPO", tmp_path)
    fill(tmp_path / "recordings" / "pilot_hoo", "A2")
    assert main(pr, "compare", "--print") == 0
    assert capsys.readouterr().out.splitlines() == [
        r".venv\Scripts\python.exe scripts\pilot_compare.py --root "
        r"recordings\pilot_hoo"]
    assert calls == []


def test_close_lets_an_unfinished_pilot_be_compared(pr, calls, capsys,
                                                    tmp_path):
    prepare(tmp_path, "C")
    fill(tmp_path, "C", {"cylindrical_object": 2})
    closed = tmp_path / "pilot_closed.txt"
    for reason in (None, "   "):
        extra = () if reason is None else ("--reason", reason)
        assert main(pr, "close", "--root", tmp_path, *extra) == 2
        assert "--reason" in capsys.readouterr().out
    assert main(pr, "close", "--reason", "the right glove failed", "--print",
                "--root", tmp_path) == 0
    assert not closed.exists()
    capsys.readouterr()
    assert main(pr, "close", "--reason", "the right glove failed",
                "--root", tmp_path) == 0
    text = closed.read_text(encoding="utf-8")
    assert capsys.readouterr().out == text
    rows = text.splitlines()
    assert rows[0].startswith("pilot closed as incomplete ")
    datetime.fromisoformat(rows[0].rsplit(" ", 1)[1])
    assert ("A (A_mimed): cylindrical_object 5/5, spherical_object 5/5, "
            "tip_object 5/5") in rows
    assert ("C (C_object_hint): cylindrical_object 2/5, spherical_object 0/5,"
            " tip_object 0/5") in rows
    assert ("A2 (A_mimed_again): cylindrical_object 0/1, spherical_object "
            "0/1, tip_object 0/1") in rows
    assert rows[-1] == "reason: the right glove failed"
    assert main(pr, "compare", "--root", tmp_path) == 0
    assert len(calls) == 1
    # a second close is refused: the first record stays as written
    assert main(pr, "close", "--reason", "again", "--root", tmp_path) == 2
    assert closed.read_text(encoding="utf-8") == text


def test_close_is_refused_when_the_pilot_is_complete(pr, calls, capsys,
                                                     tmp_path):
    fill(tmp_path, "A2")
    assert main(pr, "close", "--reason", "x", "--root", tmp_path) == 2
    assert "nothing to close: the pilot is complete" in capsys.readouterr().out
    assert not (tmp_path / "pilot_closed.txt").exists()


# --- 8. the launchers ------------------------------------------------------------
LAUNCHERS = {
    "pilot_step3_gloves_left_only.cmd": "A --gloves left --print",
    "pilot_step3_gloves_both.cmd": "A --gloves both --print",
    "pilot_step4_A_mimed.cmd": "A",
    "pilot_step5_B1_object.cmd": "B1",
    "pilot_step6_C_object_hint.cmd": "C",
    "pilot_step7_B2_object.cmd": "B2",
    "pilot_step8_A_mimed_again.cmd": "A2",
    "pilot_step9_compare.cmd": "compare",
}


def test_launchers():
    folder = REPO / "launchers" / "pilot"
    assert sorted(p.name for p in folder.glob("*.cmd")) == sorted(LAUNCHERS)
    for name, args in LAUNCHERS.items():
        data = (folder / name).read_bytes()
        assert data.count(b"\n") == data.count(b"\r\n"), name   # CRLF only
        lines = data.decode("ascii").splitlines()
        assert lines[0] == "@echo off"
        assert lines[1] == ('cd /d "C:\\Users\\nkim2\\OneDrive\\Desktop\\'
                            'non glove xr trainer"')
        assert (f".venv\\Scripts\\python.exe scripts\\pilot_run.py {args}"
                in lines), name
        assert lines[-3:] == ["echo.", "echo Finished. This window can be "
                              "closed.", "pause"]
        hardware = "No hardware." if args == "compare" else "Gloves on + camera"
        assert any(ln.startswith("echo ") and hardware in ln
                   for ln in lines), name


def test_ctrl_c_during_a_run_ends_quietly(pr, capsys, tmp_path, monkeypatch):
    """Ctrl+C reaches the recorder, which deletes the interrupted take; here
    it is only kept off the screen: one line, exit 130, no traceback."""
    def interrupted(argv, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(pr.subprocess, "run", interrupted)
    assert pr.main(["A", "--gloves", "left", "--root", str(tmp_path)]) == 130
    out = capsys.readouterr().out
    assert "stopped with Ctrl+C" in out and "Traceback" not in out
