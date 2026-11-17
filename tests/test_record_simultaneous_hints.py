"""`scripts/record_simultaneous.py --hint` and `--note`, on mock sensors.

Gate H4 of GATES_hand_on_object.md: every `<take>.meta.json`, kept or
rejected, says which tracking hint the camera was asked for
(`tracking_hints`), whether the service confirmed it (`hints_confirmed`) and
the Hyperion flag as read (`license_hyperion`); and `note` holds `--note`
exactly as typed. The mock leap stream reports a requested hint the way a
licensed service would (`leap_hand.mock.MockLeapStream`), so a mock run
writes what a hinted camera run writes. Whether the real service accepts the
hint is `tests/test_leap_hand.py -k hint`.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from leap_hand.mock import MockLeapStream
from test_pose_check import MOCK_PARAMS

REPO = Path(__file__).resolve().parents[1]
HINT_KEYS = ("tracking_hints", "hints_confirmed", "license_hyperion")
HINTED = {"tracking_hints": ["hand_on_object"], "hints_confirmed": True,
          "license_hyperion": True}
UNHINTED = {"tracking_hints": [], "hints_confirmed": False,
            "license_hyperion": None}
# punctuation, a decimal and a non-ASCII dash: stored exactly as typed
NOTE = "condition C, object + hint: tennis ball, 6.7 cm — left hand"


def load_sync():
    path = REPO / "scripts" / "record_simultaneous.py"
    spec = importlib.util.spec_from_file_location(
        "repo_script_record_simultaneous_hints", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(sync, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["record_simultaneous.py", *map(str, argv)])
    sync.main()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hint_keys(meta: dict) -> dict:
    return {k: meta[k] for k in HINT_KEYS}


def posed_session(sync, tmp_path, monkeypatch, leap_pose: str, glove_curl: float,
                  hints=(), **kwargs):
    """A coached session whose two mock sensors hold one shape, so the pose
    check can refuse an attempt for real (as in tests/test_pose_check.py),
    with the leap mock asked for `hints` the way main() asks it."""
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    monkeypatch.setattr(sync, "WRONG_POSE_SECONDS", 0.05)
    leap = MockLeapStream(pose=leap_pose, noise_mm=0.0, dropout_every=0,
                          reacquire_every=0, hints=hints)
    leap.start()
    glove = sync.MockGloveSource()
    for gen in glove.gens.values():
        gen.t = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * glove_curl))) / 1.5
        gen.dt = 0.0
    return sync.CoachedLeapSession(
        leap, glove, hz=None, out_dir=tmp_path / "sync", hand="left",
        settle=0.0, acquire_timeout=10.0, pose_check=True,
        pose_check_params=MOCK_PARAMS, **kwargs)


def test_hint_mock_run_writes_the_hint_and_the_note_into_every_meta(
        tmp_path, monkeypatch, capsys):
    """H4: the whole main(), --mock-glove --mock-leap, shortest takes."""
    sync = load_sync()
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    asked = []
    real_open = sync.open_stream

    def spy(**kwargs):
        asked.append(kwargs)
        return real_open(**kwargs)
    monkeypatch.setattr(sync, "open_stream", spy)

    out = tmp_path / "sync"
    run_main(sync, monkeypatch, "--camera", "leap", "--hand", "left",
             "--mock-glove", "--mock-leap", "--poses", "fist,open_palm",
             "--takes", "1", "--duration", "0.5", "--settle", "0",
             "--prep", "0", "--retries", "6", "--out-dir", out,
             "--hint", "hand_on_object", "--note", NOTE)

    assert asked == [{"mock": True, "mode": "desktop",
                      "hints": ("hand_on_object",)}]
    kept = sorted((out / "leap").glob("*.meta.json"))
    assert len(kept) == 2, "one kept take per pose"
    # the default mock drops frames and changes hand id on its own clock, so
    # an attempt may have been set aside: its meta must say the same
    aside = sorted((out / "rejected" / "leap").glob("*.meta.json"))
    for path in kept + aside:
        meta = read(path)
        assert meta["accepted"] is (path in kept), path.name
        assert hint_keys(meta) == HINTED, path.name
        assert meta["note"] == NOTE, path.name
    assert "hint: hand_on_object" in capsys.readouterr().out


def test_hint_and_note_reach_the_meta_of_a_rejected_attempt(tmp_path,
                                                             monkeypatch):
    """A refused attempt is described in the same terms as a kept one."""
    sync = load_sync()
    session = posed_session(sync, tmp_path, monkeypatch, leap_pose="fist",
                            glove_curl=1.0, hints=("hand_on_object",),
                            note=NOTE, retries=0)
    session.run_take("open_palm", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()

    out = tmp_path / "sync"
    assert list((out / "leap").glob("*.meta.json")) == [], "nothing was kept"
    (path,) = sorted((out / "rejected" / "leap").glob("*.meta.json"))
    meta = read(path)
    assert meta["accepted"] is False and "expected OPEN PALM" in meta["why"]
    assert hint_keys(meta) == HINTED
    assert meta["note"] == NOTE
    # the printed redo records under the same hint and note
    assert session.redo_command(["open_palm"], 1).endswith(
        f'--takes 1 --hint hand_on_object --note "{NOTE}"')


def test_no_hint_and_no_note_read_as_none_in_the_meta(tmp_path, monkeypatch):
    sync = load_sync()
    session = posed_session(sync, tmp_path, monkeypatch, leap_pose="fist",
                            glove_curl=1.0, retries=0)
    session.run_take("fist", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()
    (path,) = sorted((tmp_path / "sync" / "leap").glob("*.meta.json"))
    meta = read(path)
    assert meta["accepted"] is True
    assert hint_keys(meta) == UNHINTED
    assert meta["note"] is None
    assert session.redo_command(["fist"], 1).endswith("--takes 1")


def test_hint_condition_cannot_be_mixed_into_one_output_folder(tmp_path,
                                                               monkeypatch):
    """The pilot's conditions B and C differ only by --hint: a folder that
    already holds takes of one is refused to a run of the other, either way."""
    sync = load_sync()
    plain = posed_session(sync, tmp_path / "B", monkeypatch, leap_pose="fist",
                          glove_curl=1.0, retries=0)
    plain.run_take("fist", 1, 1, 1, 1, duration=0.6, prep=0)
    plain.finish()
    hinted = posed_session(sync, tmp_path / "C", monkeypatch, leap_pose="fist",
                           glove_curl=1.0, hints=("hand_on_object",), retries=0)
    hinted.run_take("fist", 1, 1, 1, 1, duration=0.6, prep=0)
    hinted.finish()

    base = ["--camera", "leap", "--hand", "left", "--mock-glove", "--mock-leap",
            "--poses", "fist", "--takes", "1"]
    for folder, extra, already in (
            (tmp_path / "B" / "sync", ["--hint", "hand_on_object"],
             "no tracking hint"),
            (tmp_path / "C" / "sync", [], "tracking hint hand_on_object")):
        before = sorted(p.name for p in folder.rglob("*"))
        with pytest.raises(SystemExit) as e:
            run_main(sync, monkeypatch, *base, *extra, "--out-dir", folder)
        assert "one tracking condition" in str(e.value)
        assert f"recorded with {already}" in str(e.value)
        assert sorted(p.name for p in folder.rglob("*")) == before


@pytest.mark.parametrize("argv, said", [
    # the uncoached leap session and the webcam session write no meta.json
    (["--camera", "leap", "--hand", "both", "--hint", "hand_on_object"],
     "--hint needs --camera leap --hand left|right"),
    (["--camera", "0", "--hint", "hand_on_object"],
     "--hint needs --camera leap --hand left|right"),
    (["--camera", "leap", "--hand", "both", "--note", "a pen"],
     "--note needs --camera leap --hand left|right"),
    # a name the service would ignore silently is refused here instead
    # (argparse's own refusal: exit status 2)
    (["--camera", "leap", "--hand", "left", "--hint", "hand_on_objects"], 2),
])
def test_hint_and_note_are_refused_where_they_would_be_lost(argv, said, tmp_path,
                                                            monkeypatch):
    sync = load_sync()
    out = tmp_path / "sync"
    with pytest.raises(SystemExit) as e:
        run_main(sync, monkeypatch, "--mock-glove", "--mock-leap"
                 if "leap" in argv else "--no-preview", *argv, "--out-dir", out)
    if isinstance(said, str):
        assert said in str(e.value)
    else:
        assert e.value.code == said
    assert not out.exists(), "refused before anything was recorded"


def test_every_meta_carries_the_commanded_hold(tmp_path, monkeypatch):
    """`hold_start` / `hold_end`: the REC hold as commanded (its start plus
    --duration), which scripts/pilot_compare.py ends its window on; written
    for a kept and a refused attempt alike."""
    sync = load_sync()
    session = posed_session(sync, tmp_path, monkeypatch, leap_pose="fist",
                            glove_curl=1.0, retries=0)
    session.run_take("fist", 1, 1, 1, 2, duration=0.6, prep=0)        # kept
    session.run_take("open_palm", 1, 1, 2, 2, duration=0.6, prep=0)   # refused
    session.finish()

    out = tmp_path / "sync"
    (path,) = sorted((out / "leap").glob("*.meta.json"))
    meta = read(path)
    assert meta["accepted"] is True
    assert meta["hold_end"] - meta["hold_start"] == pytest.approx(0.6,
                                                                  abs=1e-6)
    first = json.loads((out / "leap" / meta["file"]).read_text(
        encoding="utf-8").splitlines()[0])
    assert abs(meta["hold_start"] - first["wall_time"]) < 0.5
    (path,) = sorted((out / "rejected" / "leap").glob("*.meta.json"))
    refused = read(path)
    assert refused["accepted"] is False
    assert "hold_start" in refused and "hold_end" in refused


def test_first_take_numbers_the_takes_from_n(tmp_path, monkeypatch):
    """`--first-take 3`: a top-up of a run that already holds takes 1 and 2
    (scripts/pilot_run.py) is numbered on from 3, file and meta alike."""
    sync = load_sync()
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    out = tmp_path / "sync"
    run_main(sync, monkeypatch, "--camera", "leap", "--hand", "left",
             "--mock-glove", "--mock-leap", "--poses", "fist",
             "--takes", "1", "--first-take", "3", "--duration", "0.5",
             "--settle", "0", "--prep", "0", "--retries", "6",
             "--out-dir", out)
    (path,) = sorted((out / "leap").glob("*.meta.json"))
    meta = read(path)
    assert meta["accepted"] is True and meta["take"] == 3
    assert "_take3_" in path.name and "_take3_" in meta["file"]
    for aside in (out / "rejected" / "leap").glob("*.meta.json"):
        assert read(aside)["take"] == 3, aside.name


# --- --keep-all-tracking: the object runs' acceptance (amendment 2026-10-23) ---
TRACKING_KEYS = ("acceptance_policy", "protocol_amendment_date",
                 "hold_completed", "cam_tracked_fraction", "cam_max_gap_ms",
                 "cam_gaps_over_100ms", "cam_reacquisitions", "tracking_note")


def lossy_session(sync, tmp_path, monkeypatch, keep_all: bool, **kwargs):
    """A coached session on a mock camera that loses the hand during every
    hold and finds it again under a new id. The mock's hand id changes every
    100 frames (`reacquire_every`, visible_time back to zero) and the last 50
    of every 100 are dropped (`dropout_every` / `dropout_frames`): a hand is
    acquirable (visible >= 0.5 s) only in frames 45 to 49 of each id, so the
    hold starts there; the hand is gone from frame 50 (0.56 s, past LOST_S)
    and back under the next id at frame 100, about 0.6 s into a 1 s hold."""
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    leap = MockLeapStream(pose="fist", noise_mm=0.0, dropout_every=100,
                          dropout_frames=50, reacquire_every=100)
    leap.start()
    return sync.CoachedLeapSession(
        leap, sync.MockGloveSource(), hz=None, out_dir=tmp_path / "sync",
        hand="left", settle=0.0, acquire_timeout=10.0, pose_check=False,
        retries=0, keep_all_tracking=keep_all, **kwargs)


def only_meta(folder: Path) -> dict:
    (path,) = sorted(folder.glob("*.meta.json"))
    return read(path)


def test_keep_all_tracking_keeps_a_take_that_lost_the_hand(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    """With the flag the REC hold runs its full duration whatever the camera
    does, the take is kept, and the loss is written down instead."""
    sync = load_sync()
    session = lossy_session(sync, tmp_path, monkeypatch, keep_all=True)
    session.run_take("cylindrical_object", 1, 1, 1, 1, duration=1.0, prep=0)
    session.finish()
    session.print_summary()

    out = tmp_path / "sync"
    assert not (out / "rejected").exists(), "nothing was rejected"
    meta = only_meta(out / "leap")
    assert meta["accepted"] is True
    assert meta["acceptance_policy"] == "object_keep_all_tracking"
    assert meta["protocol_amendment_date"] == "2026-10-23"
    assert meta["hold_completed"] is True
    assert 0.0 < meta["cam_tracked_fraction"] < 1.0
    assert meta["cam_reacquisitions"] >= 1
    assert meta["cam_max_gap_ms"] > 500 and meta["cam_gaps_over_100ms"] >= 1
    # the strict rule's own words, as information on a kept take
    assert meta["why"] == ""
    assert "the left hand was lost for" in meta["tracking_note"]
    assert "re-acquired the hand (id 2001 -> 2002)" in meta["tracking_note"]
    assert meta["tracking_note"].endswith("during the take")
    assert meta["coverage"] == meta["cam_tracked_fraction"]
    assert meta["hand_id"] == 2001, "the id acquired at ACQUIRE"
    # the hold ran to the end: the glove file spans the commanded duration
    glove = [json.loads(s) for s in (out / "glove" / meta["file"]).read_text(
        encoding="utf-8").splitlines()]
    assert glove[-1]["wall_time"] - glove[0]["wall_time"] >= 1.0 - 0.1
    assert meta["hold_end"] - meta["hold_start"] == pytest.approx(1.0,
                                                                  abs=1e-6)
    # the re-acquired hand is followed, so the camera file has its frames too
    cam = [json.loads(s) for s in (out / "leap" / meta["file"]).read_text(
        encoding="utf-8").splitlines()]
    assert {d["hand_id"] for d in cam} == {2001, 2002}
    text = capsys.readouterr().out
    assert "kept (tracking lost " in text and "1 re-acquisition)" in text
    assert "--keep-all-tracking" in text            # the printed summary
    assert session.redo_command(["cylindrical_object"], 1).endswith(
        "--takes 1 --keep-all-tracking")


def test_without_the_flag_the_same_camera_rejects_the_attempt(tmp_path,
                                                              monkeypatch):
    sync = load_sync()
    session = lossy_session(sync, tmp_path, monkeypatch, keep_all=False)
    session.run_take("cylindrical_object", 1, 1, 1, 1, duration=1.0, prep=0)
    session.finish()

    out = tmp_path / "sync"
    assert list((out / "leap").glob("*.meta.json")) == [], "nothing kept"
    meta = only_meta(out / "rejected" / "leap")
    assert meta["accepted"] is False
    assert meta["why"] == "the left hand was lost for 0.5 s during the take"
    assert meta["tracking_note"] == meta["why"]
    assert meta["acceptance_policy"] == "strict"
    assert meta["protocol_amendment_date"] is None
    assert meta["hold_completed"] is False
    # over the COMMANDED hold, so the part the early end cut off counts as lost
    assert meta["cam_tracked_fraction"] < 0.5
    assert meta["cam_reacquisitions"] == 0
    assert session.redo_command(["cylindrical_object"], 1).endswith(
        "--takes 1")


def test_a_clean_strict_take_has_the_tracking_keys(tmp_path, monkeypatch):
    sync = load_sync()
    session = posed_session(sync, tmp_path, monkeypatch, leap_pose="fist",
                            glove_curl=1.0, retries=0)
    session.run_take("fist", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()
    meta = only_meta(tmp_path / "sync" / "leap")
    assert set(TRACKING_KEYS) <= set(meta)
    assert meta["accepted"] is True
    assert meta["acceptance_policy"] == "strict"
    assert meta["protocol_amendment_date"] is None
    assert meta["hold_completed"] is True
    assert meta["cam_tracked_fraction"] == 1.0
    assert meta["cam_reacquisitions"] == 0
    assert meta["cam_gaps_over_100ms"] == 0 and meta["cam_max_gap_ms"] < 100
    assert meta["tracking_note"] == "" and meta["why"] == ""


def test_keep_all_tracking_through_main_writes_the_policy(tmp_path,
                                                          monkeypatch):
    """The flag on the command line reaches every meta, kept or rejected."""
    sync = load_sync()
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    out = tmp_path / "sync"
    run_main(sync, monkeypatch, "--camera", "leap", "--hand", "left",
             "--mock-glove", "--mock-leap", "--poses", "cylindrical_object",
             "--takes", "1", "--duration", "0.5", "--settle", "0",
             "--prep", "0", "--retries", "6", "--keep-all-tracking",
             "--out-dir", out)
    metas = [read(p) for p in sorted(out.rglob("*.meta.json"))]
    assert [m["accepted"] for m in metas].count(True) == 1
    for meta in metas:
        assert meta["acceptance_policy"] == "object_keep_all_tracking"
        assert meta["protocol_amendment_date"] == "2026-10-23"


@pytest.mark.parametrize("argv", [
    ["--camera", "0", "--keep-all-tracking"],
    ["--camera", "leap", "--hand", "both", "--keep-all-tracking"],
])
def test_keep_all_tracking_is_refused_where_hint_is(argv, tmp_path,
                                                    monkeypatch):
    sync = load_sync()
    out = tmp_path / "sync"
    with pytest.raises(SystemExit) as e:
        run_main(sync, monkeypatch, "--mock-glove", "--mock-leap"
                 if "leap" in argv else "--no-preview", *argv, "--out-dir", out)
    assert "--keep-all-tracking needs --camera leap --hand left|right" in str(
        e.value)
    assert not out.exists(), "refused before anything was recorded"


class SpyView:
    """Stands in for the CameraView: keeps every caption and its ref."""

    def __init__(self):
        self.calls = []

    def caption(self, text, band=None, ref=None):
        self.calls.append((text, ref))

    def snapshot(self, path):
        return None


def test_a_pose_phase_sends_the_reference_picture_of_that_pose(tmp_path,
                                                                monkeypatch):
    """G27: while FIST is asked for, the window is sent the reference folder's
    fist.png; with the file gone it is sent nothing. The folder here is a
    stand-in: the professor's photographs are not in the repository
    (.gitignore), and the recorder passes the path on without opening it."""
    sync = load_sync()
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    picture = ref_dir / "fist.png"
    picture.write_bytes(b"stand-in")
    monkeypatch.setattr(sync, "REFERENCE_DIR", ref_dir)
    session = posed_session(sync, tmp_path, monkeypatch, leap_pose="fist",
                            glove_curl=1.0, retries=0)
    session.view = SpyView()
    session.run_take("fist", 1, 1, 1, 1, duration=0.3, prep=0)
    session.finish()
    calls = session.view.calls
    held = [ref for text, ref in calls if text.startswith("HOLD: FIST")]
    acquiring = [ref for text, ref in calls if text.startswith("OPEN PALM first")]
    assert held and set(held) == {str(picture)}
    assert acquiring and set(acquiring) == {str(picture)}
    # no file on disk: the recorder passes nothing
    monkeypatch.setattr(sync, "REFERENCE_DIR", tmp_path / "none")
    session.view.calls.clear()
    session._phase, session._pose = "REC", "fist"
    session._show_hud(0.0)
    assert session.view.calls[-1][1] is None
