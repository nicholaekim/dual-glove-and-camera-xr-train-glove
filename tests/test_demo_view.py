"""The demo window: the projection, the badges, the pose guess, the replay
player, and scripts/demo.py end to end on a synthetic session and on the
mock sensors."""
import importlib.util
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cam_hand.demo_view import (
    BG_BGR,
    BONES,
    MM_PER_PX,
    CentroidClassifier,
    SideState,
    badge_text,
    canvas_size,
    draw_hand,
    even_size,
    pad_to,
    palm_view_mm,
    to_pixels,
)
from cam_hand.features import loo_take_nearest_centroid
from cam_hand.fusion import GATED_DOFS, SRC_CAMERA, SRC_GLOVE, SRC_RAIL

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _demo():
    return _load("demo_script", ROOT / "scripts" / "demo.py")


# --- the projection ----------------------------------------------------------

def _flat_open_hand(scale=1.0):
    """A flat open hand in metres, in the plane z = 0: wrist at the origin,
    fingers along +y, the thumb out to -x, the pinky at +x."""
    pts = np.zeros((21, 3))
    knuckles = {"index": -0.030, "middle": -0.008, "ring": 0.013,
                "pinky": 0.032}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for finger, x in knuckles.items():
        k = base[finger]
        pts[k] = [x, 0.090, 0.0]
        for j, length in enumerate((0.045, 0.028, 0.022), start=1):
            pts[k + j] = pts[k + j - 1] + [0.0, length, 0.0]
    thumb_dir = np.array([-0.7, 0.7, 0.0]) / math.sqrt(0.98)
    pts[1] = [-0.020, 0.020, 0.0]
    for j, length in enumerate((0.040, 0.032, 0.028), start=2):
        pts[j] = pts[j - 1] + thumb_dir * length
    return pts * scale


def _rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _pixel_bones(px):
    return np.array([np.linalg.norm(px[a] - px[b]) for a, b in BONES])


@pytest.mark.parametrize("side", ["left", "right"])
def test_an_open_palm_has_its_fingertips_above_the_wrist(side):
    hand = _flat_open_hand()
    for seed in range(5):
        moved = hand @ _rotation(seed).T + [0.3, -0.1, 0.4]
        px = to_pixels(palm_view_mm(moved, side), (160, 250))
        wrist_y = px[0, 1]
        for tip in (8, 12, 16, 20):
            assert px[tip, 1] < wrist_y - 150          # well above, y down
        assert np.array_equal(px[0], [160, 250])


def test_the_thumb_is_on_the_viewers_right_for_a_right_palm():
    view_r = palm_view_mm(_flat_open_hand(), "right")
    view_l = palm_view_mm(_flat_open_hand(), "left")
    assert view_r[4, 0] > 0 > view_r[20, 0]
    assert view_l[4, 0] < 0 < view_l[20, 0]


def test_bone_lengths_in_pixels_are_the_same_in_every_panel_and_rotation():
    """One fixed scale for all three hands: the same hand drawn in any panel,
    turned any way, has the same bones on screen, and a larger hand has
    proportionally longer ones (nothing is normalised away)."""
    hand = _flat_open_hand()
    img = np.zeros((300, 1000, 3), np.uint8)
    drawn = [draw_hand(img, hand @ _rotation(seed).T, "right",
                       (120 + 330 * k, 280), (255, 255, 255))
             for k, seed in enumerate((1, 2, 3))]
    lengths = [_pixel_bones(px) for px in drawn]
    for other in lengths[1:]:
        assert np.max(np.abs(other - lengths[0])) <= 1.5   # pixel rounding
    # A flat 45 mm bone (index MCP to PIP) is 45 / MM_PER_PX pixels.
    assert abs(np.linalg.norm(drawn[0][5] - drawn[0][6])
               - 45.0 / MM_PER_PX) <= 1.0
    bigger = _pixel_bones(to_pixels(palm_view_mm(_flat_open_hand(1.1),
                                                 "right"), (0, 0)))
    ratio = bigger.sum() / _pixel_bones(to_pixels(
        palm_view_mm(hand, "right"), (0, 0))).sum()
    assert abs(ratio - 1.1) < 0.01


# --- the badges --------------------------------------------------------------

def test_badge_text_says_where_each_finger_came_from():
    sources = {dof: SRC_GLOVE for dof in GATED_DOFS}
    sources.update({"thumb": SRC_CAMERA, "spread index": SRC_CAMERA,
                    "spread pinky": SRC_CAMERA, "curl index": SRC_RAIL})
    lines = badge_text(sources, rail_active=("index",))
    assert lines == [
        "thumb   direction: camera",
        "index   curl: camera (override)   spread: camera",
        "middle   curl: glove   spread: glove",
        "ring   curl: glove   spread: glove",
        "pinky   curl: glove   spread: camera",
        "override: index",
    ]
    assert badge_text({dof: SRC_GLOVE for dof in GATED_DOFS})[-1] == \
        "override: none"
    # Every gated DOF has its own word on some line: thumb + 4 x (curl,
    # spread).
    words = sum(line.count(": ") for line in lines[:-1])
    assert words == len(GATED_DOFS)


# --- the pose guess ----------------------------------------------------------

def _samples(seed=0):
    """(pose, hand, take, features): three poses, three takes each, both
    hands, clustered around a pose centre in the 11 report features."""
    rng = np.random.default_rng(seed)
    centres = {"fist": np.full(11, 0.5), "open_palm": np.full(11, 2.0),
               "pinch": np.r_[np.full(5, 1.2), np.full(6, 0.3)]}
    out = []
    for pose, c in centres.items():
        for take in range(3):
            for hand in ("left", "right"):
                out.append((pose, hand, f"{pose}_take{take}",
                            list(c + rng.normal(scale=0.08, size=11))))
    return out, centres


def test_the_centroid_classifier_picks_the_nearest_pose_with_a_margin():
    samples, centres = _samples()
    clf = CentroidClassifier(samples)
    assert clf.poses == ["fist", "open_palm", "pinch"]
    g = clf.guess(list(centres["pinch"] + 0.05))
    assert g.pose == "pinch"
    assert g.margin > 0 and g.runner_up in ("fist", "open_palm")
    assert math.isclose(
        g.distance,
        float(np.linalg.norm(centres["pinch"] + 0.05
                             - clf.centroids()["pinch"])))
    assert clf.guess(None) is None


def test_leaving_a_take_out_agrees_with_the_reports_classifier():
    """Held out take by take, the demo's guess is the report's
    (`loo_take_nearest_centroid`), including where both get it wrong."""
    samples, _ = _samples(seed=3)
    samples.append(("pinch", "right", "odd_take", [2.0] * 11))   # a wrong one
    clf = CentroidClassifier(samples)
    _ok, _n, wrong = loo_take_nearest_centroid(
        [(p, t, f) for p, _h, t, f in samples])
    wrong_at = {i: predicted for _true, predicted, i in wrong}
    assert wrong_at, "the odd take should be misread by both"
    for i, (pose, _hand, take, feats) in enumerate(samples):
        assert clf.guess(feats, exclude_take=take).pose == \
            wrong_at.get(i, pose)


def test_a_side_goes_grey_after_half_a_second_without_a_frame():
    hand = _flat_open_hand().tolist()
    frame = SimpleNamespace(t_glove=10.0, pts=hand, glove_in=np.array(hand),
                            cam_in=None, dof_source={}, rail_active=())
    state = SideState("right")
    state.update(frame)
    assert state.seen("glove", 10.4) and not state.seen("glove", 10.6)
    assert not state.seen("camera", 10.0)
    assert not state.camera_fresh(10.0)
    state.update(replace_frame(frame, t_glove=10.1, cam_in=np.array(hand)))
    assert state.seen("camera", 10.5) and state.camera_fresh(10.3)
    assert state.glove_hz(10.1) == 2.0
    assert len(state.mean_features()) == 11


def replace_frame(frame, **changes):
    d = dict(vars(frame))
    d.update(changes)
    return SimpleNamespace(**d)


# --- the replay player -------------------------------------------------------

def test_the_replay_player_keeps_the_recordings_pace_and_loops():
    demo = _demo()

    def take(name, n):
        return demo.Take(name, name, 1, ("right",),
                         [SimpleNamespace(t_glove=100.0 + k * 0.1,
                                          hand="right") for k in range(n)])

    player = demo.ReplayPlayer([take("a", 5), take("b", 3)])
    got, changed = player.advance(0.0)
    assert len(got) == 1 and not changed             # the first frame, t=100.0
    got, _ = player.advance(0.2)
    assert [round(f.t_glove, 1) for f in got] == [100.1, 100.2]
    player.faster()
    got, _ = player.advance(0.1)                     # 0.2 s of take at x2
    assert [round(f.t_glove, 1) for f in got] == [100.3, 100.4]
    got, changed = player.advance(0.2)               # past the take's gap
    assert changed and player.take.name == "b"
    player.paused = True
    assert player.advance(1.0) == ([], False)
    player.paused = False
    player.next_take()
    assert player.take.name == "a"                   # loops at the end
    for _ in range(5):
        player.step()
    frames, changed = player.step()
    assert changed and player.take.name == "b" and frames[0].t_glove == 100.0


# --- end to end --------------------------------------------------------------

def _session_with_two_poses(root):
    """The live-fusion tests' synthetic one-take session, plus a copy of it
    labelled as another pose, so the pose guess has two centroids."""
    helpers = _load("live_fusion_test_helpers",
                    ROOT / "tests" / "test_live_fusion.py")
    helpers._write_session(root)
    other = "other_right_take2_20260922_120100.jsonl"
    for folder in ("glove", "leap"):
        src = root / folder / helpers.TAKE
        lines = []
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                d.update({"pose": "other", "take": 2})
                lines.append(json.dumps(d))
        (root / folder / other).write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    return root, helpers.LAG


def test_replay_of_a_synthetic_session_writes_a_snapshot(tmp_path, capsys):
    demo = _demo()
    root, lag = _session_with_two_poses(tmp_path / "session")
    png = tmp_path / "shots" / "replay.png"
    code = demo.main(["--replay", str(root), "--profile", "none",
                      "--glove-lag", f"right:{lag}", "--fit-template", "none",
                      "--no-window", "--snapshot", str(png),
                      "--frames", "150"])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "Pose centroids from" in text and "Playing 2 take(s)" in text
    assert "Drew 150 frame(s)." in text
    img = cv2.imdecode(np.frombuffer(png.read_bytes(), np.uint8),
                       cv2.IMREAD_COLOR)
    assert img.shape[1::-1] == canvas_size(1)
    assert img.std() > 10                             # something was drawn


def test_replay_to_a_video_keeps_the_recordings_pace(tmp_path, capsys):
    """Every take once, 30 video frames per second of recording (times the
    speed) plus the gap after each take, and a picture in the first frame."""
    demo = _demo()
    root, lag = _session_with_two_poses(tmp_path / "session")
    flags = ["--replay", str(root), "--profile", "none",
             "--glove-lag", f"right:{lag}", "--fit-template", "none"]
    video = tmp_path / "clips" / "replay.mp4"
    code = demo.main(flags + ["--video", str(video), "--speed", "2"])
    text = capsys.readouterr().out
    assert code == 0, text
    assert video.is_file() and "Video: " in text

    s = demo.settings_from_args(demo.build_parser().parse_args(flags))
    takes, _samples = demo.fuse_session(root, s, log=lambda *a: None)
    step = 2.0 / demo.VIDEO_FPS
    expected = sum(math.ceil((t.frames[-1].t_glove - t.frames[0].t_glove
                              + demo.TAKE_GAP_S) / step) for t in takes)
    cap = cv2.VideoCapture(str(video))
    try:
        assert cap.isOpened()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert abs(count - expected) <= len(takes), (count, expected)
        ok, first = cap.read()
    finally:
        cap.release()
    assert ok
    assert first.shape[1::-1] == even_size(canvas_size(1))
    assert first.std() > 10                           # not a blank frame


def test_a_video_frame_is_padded_to_an_even_size():
    assert even_size((1131, 562)) == (1132, 562)
    img = np.full((3, 5, 3), 200, np.uint8)
    out = pad_to(img, even_size(img.shape[1::-1]))
    assert out.shape == (4, 6, 3)
    assert (out[:3, :5] == 200).all() and (out[3] == BG_BGR).all()


def _quiet_live(demo, monkeypatch):
    """No beeps, a short countdown, and File Explorer never opened."""
    opened = []
    monkeypatch.setattr(demo, "beep", lambda *a, **k: None)
    monkeypatch.setattr(demo, "COUNTDOWN_S", 0.1)
    monkeypatch.setattr(demo, "open_folder", opened.append)
    return opened


def test_live_on_the_mock_sensors_saves_everything_in_one_folder(
        tmp_path, monkeypatch, capsys):
    """--out PATH: the fused frames, the warm-up and summary records, the
    fitted template, the video of every frame drawn, the last frame and a
    README naming each of them, all beside PATH; the folder is the last
    line printed, and --no-window never opens it."""
    demo = _demo()
    root, _lag = _session_with_two_poses(tmp_path / "session")
    opened = _quiet_live(demo, monkeypatch)
    png = tmp_path / "live.png"
    out = tmp_path / "run" / "live.jsonl"
    code = demo.main(["--live", "--mock-glove", "--mock-leap", "--no-view",
                      "--no-window", "--snapshot", str(png), "--frames", "30",
                      "--hand", "right", "--fit-template", "auto",
                      "--acquire-timeout", "5", "--warmup-open", "3",
                      "--warmup-fist", "1.2", "--out", str(out),
                      "--classifier-from", str(root)])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "Drew 30 frame(s)." in text
    assert "Pose centroids from" in text
    folder = out.parent
    assert text.rstrip().splitlines()[-1] == (
        f"Data for this demo: {folder.resolve()}")
    assert opened == []
    names = ["live.jsonl", "live.warmup.txt", "live.summary.txt",
             "template_right.json", "demo.mp4", "snapshot.png", "README.txt"]
    assert sorted(p.name for p in folder.iterdir()) == sorted(names)
    # The same record beside --out as scripts/fuse_live.py writes.
    kept = (out.parent / "live.warmup.txt").read_text(encoding="utf-8")
    assert " demo.py --live  hands right  profile " in kept.splitlines()[0]
    assert "RIGHT hand: acquired after " in kept
    assert "Warm-up learned:" in kept and "  right: " in kept
    n = len(out.read_text(encoding="utf-8").splitlines())
    summary = (out.parent / "live.summary.txt").read_text(encoding="utf-8")
    assert summary.startswith("Run started ")
    assert f"  right: {n} glove frames fused, " in summary
    assert "Drew 30 frame(s)." in summary
    # The summary says whether ffmpeg re-encoded the video.
    assert ("re-encoded by ffmpeg from OpenCV's mp4v file" in summary
            or "mp4v (ffmpeg is not on PATH" in summary)
    for path in (png, folder / "snapshot.png"):
        img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8),
                           cv2.IMREAD_COLOR)
        assert img.shape[1::-1] == canvas_size(1)
    cap = cv2.VideoCapture(str(folder / "demo.mp4"))
    try:
        assert cap.isOpened()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = cap.read()
    finally:
        cap.release()
    assert abs(count - 30) <= 2 and ok
    assert first.shape[1::-1] == even_size(canvas_size(1))

    readme = (folder / "README.txt").read_text(encoding="utf-8")
    for name in names:
        assert f"\n  {name}  " in readme, name
    assert "\nHands: right\n" in readme
    assert " s of fusion (" in readme
    assert f"Frames: {n} fused frames saved, 30 drawn in the window, " \
           "30 in demo.mp4" in readme
    assert f"  right: {n} glove frames fused, " in readme
    for dof in GATED_DOFS:
        assert f"      {dof} " in readme
    assert "why the camera was refused" not in readme


def test_live_with_no_save_writes_nothing(tmp_path, monkeypatch, capsys):
    demo = _demo()
    _quiet_live(demo, monkeypatch)
    # The unfitted mock glove's index spans less than the real glove's floor
    # (see test_live_fusion._low_glove_floor); a plumbing test lowers it.
    monkeypatch.setattr(demo, "DEFAULT_GATES",
                        replace(demo.DEFAULT_GATES, min_glove_span=0.30))
    monkeypatch.setattr(demo, "DEMO_DIR", tmp_path / "demo")
    monkeypatch.chdir(tmp_path)
    code = demo.main(["--live", "--mock-glove", "--mock-leap", "--no-view",
                      "--no-window", "--frames", "30", "--hand", "right",
                      "--fit-template", "none", "--acquire-timeout", "5",
                      "--warmup-open", "1", "--warmup-fist", "1.2",
                      "--classifier-from", "none", "--no-save"])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "Drew 30 frame(s)." in text
    assert list(tmp_path.iterdir()) == []
    assert "Data for this demo" not in text


def test_the_default_demo_folder_is_named_by_minute_and_hands(tmp_path,
                                                              monkeypatch):
    demo = _demo()
    when = datetime(2026, 9, 24, 14, 5, 59)
    assert (demo.demo_folder("left", when, base=tmp_path)
            == tmp_path / "2026-09-24_1405_left")
    (tmp_path / "2026-09-24_1405_left").mkdir()
    assert (demo.demo_folder("left", when, base=tmp_path).name
            == "2026-09-24_1405_left_2")
    assert demo.DEMO_DIR == ROOT / "recordings" / "demo"
    monkeypatch.setattr(demo, "DEMO_DIR", tmp_path)
    parse = demo.build_parser().parse_args
    assert (demo.live_out_path(parse(["--live", "--hand", "both"]), when)
            == tmp_path / "2026-09-24_1405_both" / "both.jsonl")
    assert (demo.live_out_path(parse(["--live", "--out", "x/run.jsonl"]))
            == Path("x/run.jsonl"))
    assert demo.live_out_path(parse(["--live", "--no-save"])) is None
    assert demo.main(["--live", "--no-save", "--out", "x.jsonl"]) == 1


def test_without_ffmpeg_the_video_stays_as_opencv_wrote_it(tmp_path,
                                                          monkeypatch):
    demo = _demo()
    monkeypatch.setattr(demo.shutil, "which", lambda name: None)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"mp4v")
    what = demo.reencode_h264(video, "mp4v")
    assert what == "mp4v (ffmpeg is not on PATH, so not re-encoded)"
    assert video.read_bytes() == b"mp4v"


def test_exactly_one_mode_is_asked_for(capsys):
    demo = _demo()
    assert demo.main([]) == 1
    assert demo.main(["--live", "--replay", "x"]) == 1
    assert "exactly one" in capsys.readouterr().out
