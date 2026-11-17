"""`scripts/glove/record_poses.py`: the glove-only pose recorder, on its mock glove.

Gates R9 and R10 of GATES_glove_recording.md. The glove recording uses no
camera (the user's correction of 13 Nov 2026); it takes its poses from a pose
list, keeps one glove's frames, and saves a picture of the glove's own
application (XR Train) for every take. The application window is stood in for
by a function that writes a small PNG; the real window is gates R3 and R4.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "glove" / "record_poses.py"
APP = "Fake Glove App"
APP_KEYS = ("app_screenshot", "app_screenshot_missing_reason", "app_screenshot_captured_at",
            "app_screenshot_method", "app_screenshot_window", "app_screenshot_process",
            "app_screenshot_size", "app_screenshot_black")


def load_script():
    spec = importlib.util.spec_from_file_location("repo_script_glove_record_poses", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rec(monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "beep", lambda *a, **k: None)
    return module


def run_main(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["record_poses.py", *map(str, argv)])
    module.main()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frames(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fake_capture(calls=None, **override):
    """A stand-in for `xr_hand.app_screenshot.capture_window`: an 8x6 white PNG."""
    from PIL import Image

    def capture(title, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        at = time.time()
        Image.new("RGB", (8, 6), (255, 255, 255)).save(path, format="PNG")
        if calls is not None:
            calls.append((title, path.name))
        return {"file": path.name, "missing_reason": None, "captured_at": round(at, 6),
                "method": "fake", "window": title, "process": r"C:\fake\app.exe",
                "width": 8, "height": 6, "black": False, **override}
    return capture


def pose_list(tmp_path) -> Path:
    path = tmp_path / "list.json"
    path.write_text(json.dumps({"name": "t", "items": [
        {"id": "a_pose", "label": "first shape", "shape": "all four fingers a quarter curled"},
        {"id": "b_pose", "label": "second shape", "shape": "index straight, three hooked"},
        {"id": "c_pose", "label": "third shape"},
    ]}), encoding="utf-8")
    return path


FAST = ("--mock", "--duration", "0.4", "--prep", "0", "--hz", "0")


def test_pose_list_hand_and_a_picture_for_every_take(rec, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rec, "capture_window", fake_capture(calls))
    out = tmp_path / "glove"
    run_main(rec, monkeypatch, *FAST, "--hand", "left", "--pose-list", pose_list(tmp_path),
             "--start-at", "b_pose", "--takes", "2", "--out-dir", out,
             "--app-screenshot", APP, "--note", "calibrated once, 11:02")

    takes = sorted(out.glob("*.jsonl"))
    assert [t.name.rsplit("_", 2)[0] for t in takes] == [
        "b_pose_left_take1", "b_pose_left_take2", "c_pose_left_take1", "c_pose_left_take2"]
    for take in takes:
        rows = frames(take)
        assert rows and {r["hand_side"] for r in rows} == {"left"}, "only the left glove's frames"
        meta = read(take.with_name(take.stem + ".meta.json"))
        assert meta["file"] == take.name and meta["frames"] == len(rows)
        assert meta["pose"] == rows[0]["pose"] and meta["take"] == rows[0]["take"]
        assert meta["sensors"] == "glove only, no camera"
        assert meta["hand"] == "left" and meta["hands_recorded"] == "left"
        assert meta["mock"] is True and meta["note"] == "calibrated once, 11:02"
        assert meta["frames_per_second_kept"] is None, "--hz 0 keeps every frame"
        assert meta["hold_end"] - meta["hold_start"] == pytest.approx(0.4, abs=1e-5)
        assert meta["hold_end"] <= meta["recording_stopped"] < meta["hold_end"] + 0.3
        # every frame was written inside the recording
        assert meta["hold_start"] - 0.05 <= rows[0]["wall_time"] <= rows[-1]["wall_time"] \
            <= meta["recording_stopped"]
        # the picture: named like the take, taken at the midpoint of the recording
        assert {k for k in meta if k.startswith("app_")} == set(APP_KEYS)
        assert Path(meta["app_screenshot"]) == Path("xr_train") / f"{take.stem}.png"
        assert (out / meta["app_screenshot"]).is_file()
        assert meta["app_screenshot_missing_reason"] is None
        mid = (meta["hold_start"] + meta["hold_end"]) / 2
        assert meta["hold_start"] <= meta["app_screenshot_captured_at"] <= meta["hold_end"]
        assert abs(meta["app_screenshot_captured_at"] - mid) < 0.15
        assert meta["app_screenshot_window"] == APP and meta["app_screenshot_size"] == [8, 6]
        assert meta["app_screenshot_black"] is False
    b1, c1 = read(takes[0].with_name(takes[0].stem + ".meta.json")), \
        read(takes[2].with_name(takes[2].stem + ".meta.json"))
    assert (b1["label"], b1["shape"]) == ("second shape", "index straight, three hooked")
    assert (c1["label"], c1["shape"]) == ("third shape", None)
    assert sorted(p.name for p in (out / "xr_train").glob("*.png")) == [
        t.stem + ".png" for t in takes], "no stray working names"
    said = capsys.readouterr().out
    assert "sensors: glove only, no camera   hand: left" in said
    assert "--- Pose 1/2: B POSE  (take 1/2) ---" in said
    assert "    second shape: index straight, three hooked" in said
    assert "    third shape\n" in said and "first shape" not in said
    assert f"app screenshots: the {APP!r} window" in said
    assert "APP SCREENSHOTS: 4 of 4 takes have one" in said
    assert calls[0] == (APP, "probe.png") and len(calls) == 5, "the probe, then one per take"


def test_first_take_auto_numbers_after_the_takes_on_disk(rec, tmp_path, monkeypatch):
    out = tmp_path / "glove"
    base = (*FAST, "--hand", "left", "--pose-list", pose_list(tmp_path), "--out-dir", out)
    run_main(rec, monkeypatch, *base, "--start-at", "c_pose", "--takes", "2")
    run_main(rec, monkeypatch, *base, "--start-at", "c_pose", "--takes", "1",
             "--first-take", "auto")
    run_main(rec, monkeypatch, *base, "--start-at", "b_pose", "--takes", "1",
             "--first-take", "auto")
    names = sorted(t.name.rsplit("_", 2)[0] for t in out.glob("*.jsonl"))
    # run 2 added c_pose take 3; run 3 began at b_pose (take 1) and went on to
    # c_pose, which was numbered after the three already there
    assert names == ["b_pose_left_take1", "c_pose_left_take1", "c_pose_left_take2",
                     "c_pose_left_take3", "c_pose_left_take4"]
    assert rec.next_take(out, "c_pose") == 5 and rec.next_take(out, "a_pose") == 1
    # a name that only starts like the pose is another pose's take
    (out / "c_pose_extra_left_take9_20260101_000000.jsonl").write_text("", encoding="utf-8")
    assert rec.next_take(out, "c_pose") == 5


@pytest.mark.parametrize("capture, reason", [
    (lambda title, path: {"file": None, "missing_reason": "window minimised", "captured_at": None,
                          "method": None, "window": title, "process": "x.exe",
                          "width": None, "height": None, "black": None},
     "window minimised"),
    (lambda title, path: (_ for _ in ()).throw(RuntimeError("no display")),
     "capture error: RuntimeError: no display"),
    (lambda title, path: {"file": Path(path).name, "missing_reason": None, "captured_at": 1.0,
                          "method": "fake", "window": title, "process": None,
                          "width": 1, "height": 1, "black": False},
     "the capture reported a file that is not there"),
])
def test_a_failed_capture_never_fails_the_take(capture, reason, rec, tmp_path, capsys):
    out = tmp_path / "glove"
    session = rec.Session(rec.MockSource(), hz=None, out_dir=out, hand="left",
                          app_window=APP, app_capture=capture, mock=True)
    session.run_take("fist", 1, 1, 1, 1, duration=0.4, prep=0)
    session.print_summary()
    (take,) = out.glob("*.jsonl")
    meta = read(take.with_name(take.stem + ".meta.json"))
    assert meta["frames"] > 0, "the take stands without its picture"
    assert meta["app_screenshot"] is None
    assert meta["app_screenshot_missing_reason"] == reason
    assert not list(out.rglob("*.png"))
    said = capsys.readouterr().out
    assert f"! no {APP} picture: {reason}" in said
    assert "APP SCREENSHOTS: 0 of 1 takes have one" in said
    assert f"no screenshot for {take.name}: {reason}" in said


def test_a_black_picture_is_kept_and_counted_as_black(rec, tmp_path, capsys):
    out = tmp_path / "glove"
    session = rec.Session(rec.MockSource(), hz=None, out_dir=out, app_window=APP,
                          app_capture=fake_capture(black=True), mock=True)
    session.run_take("fist", 1, 1, 1, 1, duration=0.4, prep=0)
    session.print_summary()
    (take,) = out.glob("*.jsonl")
    meta = read(take.with_name(take.stem + ".meta.json"))
    assert meta["app_screenshot_black"] is True and (out / meta["app_screenshot"]).is_file()
    said = capsys.readouterr().out
    assert "(BLACK: not a picture of the hand)" in said
    assert "1 of them BLACK, not a picture of the hand" in said


def test_a_take_with_no_frames_keeps_neither_file_nor_picture(rec, tmp_path, monkeypatch, capsys):
    out = tmp_path / "glove"
    source = rec.MockSource()
    monkeypatch.setattr(source, "drain", lambda max_items=64: [])
    session = rec.Session(source, hz=None, out_dir=out, hand="left", app_window=APP,
                          app_capture=fake_capture(), mock=True)
    session.run_take("fist", 1, 1, 1, 1, duration=0.4, prep=0)
    session.print_summary()
    assert [p for p in out.rglob("*") if p.is_file()] == [], "no take, no meta, no picture"
    said = capsys.readouterr().out
    assert "FAILED: no frames captured" in said and "0/1 takes captured" in said
    assert "APP SCREENSHOTS: 0 of 0 takes have one" in said


@pytest.mark.parametrize("probe, said", [
    ({"file": None, "missing_reason": "window not found"}, "cannot be captured (window not found)"),
    ({"file": None, "missing_reason": "window minimised"}, "cannot be captured (window minimised)"),
    ({"file": "probe.png", "missing_reason": None, "black": True}, "captures as a black picture"),
])
def test_the_recorder_refuses_to_start_when_the_window_cannot_be_captured(
        probe, said, rec, tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "capture_window", lambda title, path: dict(probe))
    monkeypatch.setattr(rec, "window_titles", lambda: ["Notepad", "Calculator"])
    out = tmp_path / "glove"
    with pytest.raises(SystemExit) as e:
        run_main(rec, monkeypatch, *FAST, "--poses", "fist", "--takes", "1",
                 "--out-dir", out, "--app-screenshot", APP)
    assert said in str(e.value) and APP in str(e.value)
    assert not out.exists(), "refused before anything was recorded"


@pytest.mark.parametrize("argv, said", [
    (["--start-at", "b_pose"], "--start-at needs --pose-list"),
    (["--pose-list", "LIST", "--start-at", "zz"], "--start-at zz: not in"),
    (["--pose-list", "MISSING"], "--pose-list: cannot read"),
    (["--poses", "fist", "--first-take", "soon"], "--first-take must be a number or 'auto'"),
    (["--poses", "fist", "--first-take", "0"], "--first-take must be 1 or more"),
    (["--poses", "fist", "--takes", "0"], "--takes must be 1 or more"),
])
def test_flags_that_cannot_work_are_refused(argv, said, rec, tmp_path, monkeypatch):
    listing = pose_list(tmp_path)
    argv = [str(listing) if a == "LIST" else str(tmp_path / "nope.json") if a == "MISSING" else a
            for a in argv]
    out = tmp_path / "glove"
    with pytest.raises(SystemExit) as e:
        run_main(rec, monkeypatch, *FAST, *argv, "--out-dir", out)
    assert said in str(e.value)
    assert not out.exists()


def test_without_the_new_flags_it_writes_what_it_always_wrote(rec, tmp_path, monkeypatch, capsys):
    def never(*_a):
        raise AssertionError("no capture without --app-screenshot")
    monkeypatch.setattr(rec, "capture_window", never)
    out = tmp_path / "glove"
    run_main(rec, monkeypatch, *FAST, "--poses", "fist", "--takes", "1", "--out-dir", out)
    (take,) = out.iterdir()
    assert take.name.startswith("fist_both_take1_") and take.suffix == ".jsonl"
    assert {r["hand_side"] for r in frames(take)} == {"left", "right"}, "both gloves, as before"
    said = capsys.readouterr().out
    assert "Hold: all fingers curled into a tight fist" in said
    assert "APP SCREENSHOTS" not in said


def test_the_real_recording_set_loads_and_is_shown(rec):
    from xr_hand.recorder import load_pose_list
    ids, text = load_pose_list(REPO / "protocols" / "glove_planar_poses_recording_set.json")
    assert ids == ["ps01", "pd01", "pf07", "pr07", "pc01", "pc05", "pm01", "pt01", "pt02", "pt03"]
    assert text["pf07"] == ("little straight, three hooked",
                            "index, middle and ring bent at PIP and DIP with MCP straight, little straight")


@pytest.mark.parametrize("content, said", [
    ("{not json", "cannot read"),
    ('{"items": []}', "has no items"),
    ('{"items": [{"id": "Has Space"}]}', "needs an id made of lower-case letters"),
    ('{"items": [{"id": "a"}, {"id": "a"}]}', "appears twice"),
])
def test_a_bad_pose_list_says_what_is_wrong(content, said, tmp_path):
    from xr_hand.recorder import load_pose_list
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=said):
        load_pose_list(path)


# --- no camera ---------------------------------------------------------------------------
def test_no_camera_in_the_glove_recorder_script():
    """The glove recording uses no camera: the script reaches for nothing of
    the camera packages and has no camera flag."""
    source = SCRIPT.read_text(encoding="utf-8")
    for needle in ("leap_hand", "cam_hand", "import cv2", "--camera", "LeapC", "open_camera"):
        assert needle not in source, needle
    assert "this recorder opens no camera" in source
    assert load_script().SENSORS == "glove only, no camera"
    # in a fresh interpreter, loading the script brings in no camera package
    import subprocess
    probe = ("import importlib.util, sys; "
             f"spec = importlib.util.spec_from_file_location('g', r'{SCRIPT}'); "
             "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
             "print(sorted(n for n in sys.modules "
             "if n.split('.')[0] in ('leap_hand', 'cam_hand', 'cv2', 'mediapipe')))")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          cwd=REPO, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]", done.stdout


def test_no_camera_in_the_launcher():
    launcher = (REPO / "launchers" / "record_glove_poses.cmd").read_text(encoding="utf-8")
    command = next(ln for ln in launcher.splitlines() if "python.exe" in ln)
    assert "scripts\\glove\\record_poses.py" in command
    assert "--hand left" in command and "--hz 0" in command
    assert "--pose-list protocols\\glove_planar_poses_recording_set.json %START%" in command
    assert "--out-dir recordings\\glove_planar_left " in command
    assert '--app-screenshot "XR Train"' in command and "--first-take auto" in command
    for needle in ("record_simultaneous", "--camera", "leap"):
        assert needle not in command.lower(), needle
    lowered = launcher.lower()
    assert "no camera" in lowered
    for needle in ("above the module", "lenses up", "ultraleap", "gloves on + camera"):
        assert needle not in lowered, needle


def test_the_desktop_copy_of_the_launcher_is_the_same_file():
    """The operator starts the recording from a copy of the launcher beside the
    repository folder, on the Desktop. A checkout elsewhere has no such copy."""
    desktop = REPO.parent / "record_glove_poses.cmd"
    if not desktop.is_file():
        pytest.skip("no copy of the launcher beside this checkout")
    launcher = (REPO / "launchers" / "record_glove_poses.cmd").read_text(encoding="utf-8")
    assert desktop.read_text(encoding="utf-8") == launcher, "the Desktop copy is the same file"
