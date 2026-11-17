"""`scripts/record_simultaneous.py --app-screenshot`, `--pose-list`, `--start-at`,
on mock sensors.

Gate R2 of GATES_glove_recording.md: every attempt of a coached session gets a
picture of the named application window at the midpoint of its hold, named
like the take, following the take when it is kept or rejected, with the
`app_screenshot*` keys in its meta.json; a capture that fails never fails the
take; without the flag nothing changes. The application window is stood in for
by a function that writes a small PNG, so no application has to be running
(the real window is gates R3 and R4).
"""
import json
import time
from pathlib import Path

import pytest

from test_record_simultaneous_hints import load_sync, posed_session, read, run_main

APP = "Fake Glove App"
APP_KEYS = ("app_screenshot", "app_screenshot_missing_reason", "app_screenshot_captured_at",
            "app_screenshot_method", "app_screenshot_window", "app_screenshot_process",
            "app_screenshot_size", "app_screenshot_black")


def fake_capture(calls=None):
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
                "width": 8, "height": 6, "black": False}
    return capture


def app_keys(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k.startswith("app_")}


BASE = ("--camera", "leap", "--hand", "left", "--mock-glove", "--mock-leap",
        "--takes", "1", "--duration", "0.5", "--settle", "0", "--prep", "0",
        "--retries", "6")


def test_every_kept_take_gets_a_screenshot_named_like_it(tmp_path, monkeypatch, capsys):
    sync = load_sync()
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(sync, "capture_window", fake_capture(calls))
    out = tmp_path / "sync"
    run_main(sync, monkeypatch, *BASE, "--poses", "fist,open_palm",
             "--out-dir", out, "--app-screenshot", APP)

    kept = sorted((out / "leap").glob("*.meta.json"))
    assert len(kept) == 2, "one kept take per pose"
    for path in kept:
        meta = read(path)
        stem = meta["file"][:-len(".jsonl")]
        assert set(app_keys(meta)) == set(APP_KEYS)
        assert Path(meta["app_screenshot"]) == Path("xr_train") / f"{stem}.png"
        assert (out / meta["app_screenshot"]).is_file()
        assert meta["app_screenshot_missing_reason"] is None
        assert meta["hold_start"] <= meta["app_screenshot_captured_at"] <= meta["hold_end"]
        # the midpoint of the hold, give or take the frame loop
        mid = (meta["hold_start"] + meta["hold_end"]) / 2
        assert abs(meta["app_screenshot_captured_at"] - mid) < 0.2
        assert meta["app_screenshot_window"] == APP
        assert meta["app_screenshot_method"] == "fake"
        assert meta["app_screenshot_size"] == [8, 6]
        assert meta["app_screenshot_black"] is False
    assert sorted(p.name for p in (out / "xr_train").glob("*.png")) == sorted(
        read(p)["file"][:-len(".jsonl")] + ".png" for p in kept), "no stray working names"
    # an attempt the mock's own dropouts set aside is described the same way
    for path in sorted((out / "rejected" / "leap").glob("*.meta.json")):
        meta = read(path)
        assert set(app_keys(meta)) == set(APP_KEYS)
        if meta["app_screenshot"]:
            assert Path(meta["app_screenshot"]).parts[:2] == ("rejected", "xr_train")
            assert (out / meta["app_screenshot"]).is_file()
        else:
            assert meta["app_screenshot_missing_reason"]
    said = capsys.readouterr().out
    assert f"app screenshots: the {APP!r} window" in said
    assert "APP SCREENSHOTS: 2 of 2 kept takes have one" in said
    # the probe at start-up, then one call per attempt that reached its midpoint
    assert calls[0] == (APP, "probe.png") and len(calls) >= 3


def test_a_rejected_attempts_screenshot_moves_with_it(tmp_path, monkeypatch):
    sync = load_sync()
    session = posed_session(sync, tmp_path, monkeypatch, leap_pose="fist", glove_curl=1.0,
                            retries=0, app_window=APP, app_capture=fake_capture())
    session.run_take("open_palm", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()

    out = tmp_path / "sync"
    assert list((out / "leap").glob("*.meta.json")) == [], "nothing was kept"
    (path,) = sorted((out / "rejected" / "leap").glob("*.meta.json"))
    meta = read(path)
    assert meta["accepted"] is False and "expected OPEN PALM" in meta["why"]
    target = meta["file"][:-len(".jsonl")]
    assert Path(meta["app_screenshot"]) == Path("rejected") / "xr_train" / f"{target}.png"
    assert (out / meta["app_screenshot"]).is_file()
    assert not list((out / "xr_train").glob("*.png")) if (out / "xr_train").is_dir() else True
    assert session.redo_command(["open_palm"], 1).endswith(f'--takes 1 --app-screenshot "{APP}"')


@pytest.mark.parametrize("capture, reason", [
    (lambda title, path: {"file": None, "missing_reason": "window minimised", "captured_at": None,
                          "method": None, "window": title, "process": "x.exe",
                          "width": None, "height": None, "black": None},
     "window minimised"),
    (lambda title, path: (_ for _ in ()).throw(RuntimeError("no display")),
     "capture error: RuntimeError: no display"),
    # says there is a file, writes none
    (lambda title, path: {"file": Path(path).name, "missing_reason": None, "captured_at": 1.0,
                          "method": "fake", "window": title, "process": None,
                          "width": 1, "height": 1, "black": False},
     "the capture reported a file that is not there"),
])
def test_a_failed_capture_never_fails_the_take(capture, reason, tmp_path, monkeypatch, capsys):
    sync = load_sync()
    session = posed_session(sync, tmp_path, monkeypatch, leap_pose="fist", glove_curl=1.0,
                            retries=0, app_window=APP, app_capture=capture)
    session.run_take("fist", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()
    session.print_summary()

    out = tmp_path / "sync"
    (path,) = sorted((out / "leap").glob("*.meta.json"))
    meta = read(path)
    assert meta["accepted"] is True, "the take stands without its screenshot"
    assert meta["app_screenshot"] is None
    assert meta["app_screenshot_missing_reason"] == reason
    assert meta["app_screenshot_window"] == APP
    assert not list(out.rglob("*.png"))
    said = capsys.readouterr().out
    assert "APP SCREENSHOTS: 0 of 1 kept takes have one" in said
    assert f"no screenshot for {meta['file']}: {reason}" in said


def test_a_black_screenshot_is_counted_as_black(tmp_path, monkeypatch, capsys):
    sync = load_sync()
    white = fake_capture()

    def black(title, path):
        return {**white(title, path), "black": True}
    session = posed_session(sync, tmp_path, monkeypatch, leap_pose="fist", glove_curl=1.0,
                            retries=0, app_window=APP, app_capture=black)
    session.run_take("fist", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()
    session.print_summary()
    (path,) = sorted((tmp_path / "sync" / "leap").glob("*.meta.json"))
    assert read(path)["app_screenshot_black"] is True
    assert "1 of them BLACK, not a picture of the hand" in capsys.readouterr().out


def test_without_the_flag_nothing_changes(tmp_path, monkeypatch):
    sync = load_sync()
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)

    def never(*_a):
        raise AssertionError("no capture without --app-screenshot")
    monkeypatch.setattr(sync, "capture_window", never)
    out = tmp_path / "sync"
    run_main(sync, monkeypatch, *BASE, "--poses", "fist", "--out-dir", out)
    metas = sorted(out.rglob("*.meta.json"))
    assert metas
    for path in metas:
        assert app_keys(read(path)) == {}, "no app_screenshot key at all"
    assert not (out / "xr_train").exists() and not (out / "rejected" / "xr_train").exists()


@pytest.mark.parametrize("probe, said", [
    ({"file": None, "missing_reason": "window not found"}, "cannot be captured (window not found)"),
    ({"file": None, "missing_reason": "window minimised"}, "cannot be captured (window minimised)"),
    ({"file": "probe.png", "missing_reason": None, "black": True}, "captures as a black picture"),
])
def test_the_recorder_refuses_to_start_when_the_window_cannot_be_captured(
        probe, said, tmp_path, monkeypatch):
    sync = load_sync()
    monkeypatch.setattr(sync, "capture_window", lambda title, path: dict(probe))
    monkeypatch.setattr(sync, "window_titles", lambda: ["Notepad", "Calculator"])
    out = tmp_path / "sync"
    with pytest.raises(SystemExit) as e:
        run_main(sync, monkeypatch, *BASE, "--poses", "fist", "--out-dir", out,
                 "--app-screenshot", APP)
    assert said in str(e.value) and APP in str(e.value)
    assert not out.exists(), "refused before anything was recorded"


@pytest.mark.parametrize("argv, said", [
    (["--camera", "leap", "--hand", "both", "--app-screenshot", APP],
     "--app-screenshot needs --camera leap --hand left|right"),
    (["--camera", "leap", "--hand", "both", "--pose-list", "LIST"],
     "--pose-list needs --camera leap --hand left|right"),
    (["--camera", "leap", "--hand", "left", "--start-at", "b_pose"],
     "--start-at needs --pose-list"),
    (["--camera", "leap", "--hand", "left", "--pose-list", "LIST", "--start-at", "zz"],
     "--start-at zz: not in"),
])
def test_the_new_flags_are_refused_where_they_cannot_work(argv, said, tmp_path, monkeypatch):
    sync = load_sync()
    listing = pose_list(tmp_path)
    argv = [str(listing) if a == "LIST" else a for a in argv]
    out = tmp_path / "sync"
    with pytest.raises(SystemExit) as e:
        run_main(sync, monkeypatch, "--mock-glove", "--mock-leap", *argv, "--out-dir", out)
    assert said in str(e.value)
    assert not out.exists()


def pose_list(tmp_path) -> Path:
    path = tmp_path / "list.json"
    path.write_text(json.dumps({"name": "t", "items": [
        {"id": "a_pose", "label": "first shape", "shape": "all four fingers a quarter curled"},
        {"id": "b_pose", "label": "second shape", "shape": "index straight, three hooked"},
        {"id": "c_pose", "label": "third shape"},
    ]}), encoding="utf-8")
    return path


def test_pose_list_supplies_the_poses_and_shows_label_and_shape(tmp_path, monkeypatch, capsys):
    sync = load_sync()
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    out = tmp_path / "sync"
    run_main(sync, monkeypatch, *BASE, "--pose-list", pose_list(tmp_path),
             "--start-at", "b_pose", "--out-dir", out)
    kept = sorted(read(p)["file"] for p in (out / "leap").glob("*.meta.json"))
    assert [k.split("_left_")[0] for k in kept] == ["b_pose", "c_pose"], "a_pose is before --start-at"
    said = capsys.readouterr().out
    assert "--- Pose 1/2: B POSE" in said
    assert "second shape: index straight, three hooked" in said
    assert "NOW: B POSE (second shape)" in said
    assert "      third shape\n" in said, "a pose without a shape shows its label alone"
    assert "first shape" not in said


@pytest.mark.parametrize("content, said", [
    ("{not json", "cannot read"),
    ('{"items": []}', "has no items"),
    ('{"items": [{"id": "Has Space"}]}', "needs an id made of lower-case letters"),
    ('{"items": [{"id": "a"}, {"id": "a"}]}', "appears twice"),
])
def test_a_bad_pose_list_is_refused(content, said, tmp_path, monkeypatch):
    sync = load_sync()
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        sync.load_pose_list(path)
    assert said in str(e.value)


def test_the_real_glove_pose_list_loads():
    sync = load_sync()
    ids, text = sync.load_pose_list(Path(sync.__file__).resolve().parents[1]
                                    / "protocols" / "glove_planar_poses_recording_set.json")
    assert len(ids) == 10 and ids[0] == "ps01" and ids[-1] == "pt03"
    assert text["pf07"] == ("little straight, three hooked",
                            "index, middle and ring bent at PIP and DIP with MCP straight, little straight")
