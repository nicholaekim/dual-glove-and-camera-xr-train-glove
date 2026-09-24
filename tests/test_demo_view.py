"""The demo window: the projection, the badges, the pose guess, the replay
player, and scripts/demo.py end to end on a synthetic session and on the
mock sensors."""
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cam_hand.demo_view import (
    BONES,
    MM_PER_PX,
    CentroidClassifier,
    SideState,
    badge_text,
    canvas_size,
    draw_hand,
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


def test_live_on_the_mock_sensors_writes_a_snapshot(tmp_path, monkeypatch,
                                                     capsys):
    demo = _demo()
    root, _lag = _session_with_two_poses(tmp_path / "session")
    monkeypatch.setattr(demo, "beep", lambda *a, **k: None)
    monkeypatch.setattr(demo, "COUNTDOWN_S", 0.1)
    # The unfitted mock glove's index spans less than the real glove's floor
    # (see test_live_fusion._low_glove_floor); a plumbing test lowers it.
    monkeypatch.setattr(demo, "DEFAULT_GATES",
                        replace(demo.DEFAULT_GATES, min_glove_span=0.30))
    png = tmp_path / "live.png"
    out = tmp_path / "run" / "live.jsonl"
    code = demo.main(["--live", "--mock-glove", "--mock-leap", "--no-view",
                      "--no-window", "--snapshot", str(png), "--frames", "30",
                      "--hand", "right", "--fit-template", "none",
                      "--acquire-timeout", "5", "--warmup-open", "1",
                      "--warmup-fist", "1.2", "--out", str(out),
                      "--classifier-from", str(root)])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "Drew 30 frame(s)." in text
    assert "Pose centroids from" in text
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
    img = cv2.imdecode(np.frombuffer(png.read_bytes(), np.uint8),
                       cv2.IMREAD_COLOR)
    assert img.shape[1::-1] == canvas_size(1)


def test_exactly_one_mode_is_asked_for(capsys):
    demo = _demo()
    assert demo.main([]) == 1
    assert demo.main(["--live", "--replay", "x"]) == 1
    assert "exactly one" in capsys.readouterr().out
