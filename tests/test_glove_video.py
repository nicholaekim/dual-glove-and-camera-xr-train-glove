"""A take's glove data drawn as a video (`cam_hand.glove_video`,
`scripts/render_glove_video.py`): the right frame and the right cue for
each instant, a picture that changes when the finger does, and a file a
player can read with as many pictures as the take's seconds ask for.

The sessions are `test_protocol_check`'s synthetic glove-only ones.
"""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

import test_protocol_check as fx
from cam_hand import glove_video as gv
from cam_hand import protocol_check as pc
from cam_hand.fusion import FINGER_CHAINS

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "render_glove_video", REPO / "scripts" / "render_glove_video.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture(scope="module")
def flexion(tmp_path_factory):
    root = tmp_path_factory.mktemp("video")
    return pc.load_session(fx.make_flexion_session(
        root / "b", [fx.FlexTake("index"),
                     fx.FlexTake("ring", accepted=False, reason="redo")],
        camera=False, stills=False))


def loaded(session, k=0):
    t = session.takes[k]
    return t, gv.load_take(t.path("glove"), pc.read_events(t.path("events")),
                           session.hand)


def test_the_take_is_read_in_time_order_with_its_cues(flexion):
    t, take = loaded(flexion)
    assert len(take.times) == len(take.local_mm) == len(take.angles) > 100
    assert take.times == sorted(take.times)
    events = pc.read_events(t.path("events"))
    assert take.t0 == events[0]["t"] and take.t1 > take.t0
    assert take.seconds == pytest.approx(take.t1 - take.t0)
    assert take.take_fingers() == ["index"]
    assert take.local_mm[0].shape == (21, 3)
    assert np.allclose(take.local_mm[0][0], 0.0)


def test_each_instant_has_the_newest_frame_and_the_cue_in_force(flexion):
    _t, take = loaded(flexion)
    assert take.index_at(take.times[0] - 0.001) is None
    assert take.index_at(take.times[5]) == 5
    assert take.index_at(take.times[5] + 1e-4) == 5
    assert take.index_at(take.times[-1] + 9.0) == len(take.times) - 1
    first, second = take.cues[0], take.cues[1]
    assert take.cue_at(first["t"] - 0.01) is None
    assert take.cue_at(first["t"]) is first
    assert take.cue_at(second["t"] - 1e-6) is first
    assert take.cue_at(second["t"]) is second
    assert take.cue_at(take.t1) is None and take.cue_at(take.t1 + 1) is None
    assert gv.cue_words(take, None) == ""
    assert gv.cue_words(take, second) == f"cycle 1/5:  {second['label']}"


@pytest.mark.parametrize("hand", ["left", "right"])
def test_a_bent_finger_points_to_the_palm_side_in_the_side_view(tmp_path,
                                                                hand):
    session = pc.load_session(fx.make_flexion_session(
        tmp_path / hand, [fx.FlexTake("index")], hand=hand, camera=False,
        stills=False))
    _t, take = loaded(session)
    hold = next(c for c in take.cues if c["phase"] == "hold")
    k = take.index_at(hold["t"] + hold["hold_s"] * 0.9)
    side = gv.side_xy(take.local_mm[k], take.side_sign)
    tip, knuckle = FINGER_CHAINS["index"][-1], FINGER_CHAINS["index"][0]
    assert side[tip, 0] - side[knuckle, 0] > 20        # mm, to the right
    # the middle finger was not bent: it stays in the palm's plane
    mid = FINGER_CHAINS["middle"][-1]
    assert abs(side[mid, 0] - side[FINGER_CHAINS["middle"][0], 0]) < 5
    # the palm view puts the thumb on the same side for a hand of either
    # side as `demo_view.palm_view_mm` does
    from cam_hand.demo_view import palm_view_mm
    from xr_hand.keypoints21 import frame_to_keypoints21
    frames, _bad = pc.read_frames(session.takes[0].path("glove"))
    pts = np.asarray(frame_to_keypoints21(frames[0][1]), float)
    want = palm_view_mm(pts, hand)
    got = gv.palm_xy(gv.local_points_mm(frames[0][1]), hand)
    assert np.allclose(got, want, atol=1e-6)


def test_the_picture_changes_when_the_finger_bends_and_says_what_it_is(
        flexion):
    t, take = loaded(flexion)
    hold = next(c for c in take.cues if c["phase"] == "hold")
    rest = next(c for c in take.cues if c["phase"] == "rest")
    bent = gv.render_frame(take, hold["t"] + hold["hold_s"] * 0.9, t.name)
    straight = gv.render_frame(take, rest["t"] + rest["hold_s"] * 0.9, t.name)
    assert bent.shape == straight.shape == (gv.HEIGHT, gv.WIDTH, 3)
    assert bent.dtype == np.uint8
    x, y, w, h = gv.SIDE_PANEL
    changed = np.any(bent[y:y + h, x:x + w] != straight[y:y + h, x:x + w],
                     axis=2)
    assert changed.sum() > 2000
    # the caption strip along the bottom is drawn on every picture, alike
    assert np.array_equal(bent[688:, :], straight[688:, :])
    assert np.any(bent[688:, :] != np.array(gv.BG_BGR, np.uint8))
    assert "Not camera footage" in gv.CAPTION
    # before the first glove frame the picture says so and draws no hand
    early = gv.render_frame(take, take.times[0] - 5.0, t.name)
    px, py, pw, ph = gv.PALM_PANEL
    drawn = np.any(early[py + 60:py + ph, px:px + pw]
                   != np.array(gv.PANEL_BGR, np.uint8), axis=2)
    assert drawn.sum() < 4000                  # only the words


def test_the_heavy_finger_is_the_takes_in_set_b_and_the_cues_in_set_c(
        flexion, tmp_path):
    _t, take = loaded(flexion)
    rest = next(c for c in take.cues if c["phase"] == "rest")
    assert gv.heavy_fingers(take, rest) == ["index"]
    assert gv.heavy_fingers(take, None) == ["index"]
    seq = pc.load_session(fx.make_sequence_session(
        tmp_path / "c", [fx.SeqTake()], camera=False, stills=False))
    _t2, stake = loaded(seq)
    assert gv.heavy_fingers(stake, stake.cues[0]) == []
    assert gv.heavy_fingers(stake, stake.cues[1]) == ["thumb", "index"]
    assert gv.heavy_fingers(stake, None) == []
    assert gv.cue_words(stake, stake.cues[1]) == (
        f"step 2/{len(stake.cues)}:  thumb + index")


def test_frame_count_is_one_picture_per_tick_from_zero_to_the_end():
    assert gv.frame_count(0.0, 30) == 1
    assert gv.frame_count(1.0, 30) == 31
    assert gv.frame_count(24.0, 30) == 721
    assert gv.frame_count(0.99, 10) == 10


def test_the_file_plays_and_lasts_as_long_as_the_take(flexion, tmp_path):
    t, take = loaded(flexion)
    dest = tmp_path / "out" / "take.mp4"
    info = gv.render_take(dest, t.path("glove"),
                          pc.read_events(t.path("events")), flexion.hand,
                          t.name, fps=5, h264=False)
    assert info["path"] == dest and dest.stat().st_size > 10_000
    assert info["pictures"] == gv.frame_count(take.seconds, 5)
    assert info["glove_frames"] == len(take.times)
    assert info["seconds"] == pytest.approx(take.seconds)
    cap = cv2.VideoCapture(str(dest))
    try:
        assert cap.isOpened()
        n = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            n += 1
            assert img.shape == (gv.HEIGHT, gv.WIDTH, 3)
        assert n == info["pictures"]
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(5.0, abs=0.01)
    finally:
        cap.release()


def test_the_command_draws_the_accepted_takes_and_only_them(flexion,
                                                            tmp_path, capsys):
    out = tmp_path / "videos"
    assert cli.main([str(flexion.path), "--out", str(out), "--fps", "2",
                     "--no-h264"]) == 0
    accepted = [t for t in flexion.takes if t.accepted]
    assert sorted(p.name for p in out.rglob("*.mp4")) == sorted(
        f"{t.name}.mp4" for t in accepted)
    assert "1 of 1 take(s) drawn." in capsys.readouterr().out
    # with --rejected the rejected attempt goes into its own folder
    assert cli.main([str(flexion.path), "--out", str(out), "--fps", "2",
                     "--no-h264", "--rejected"]) == 0
    rejected = [t for t in flexion.takes if not t.accepted]
    assert [p.name for p in (out / "rejected").iterdir()] == [
        f"{rejected[0].name}.mp4"]
    # the default folder is the session's own videos folder
    assert cli.main([str(flexion.path), "--fps", "2", "--no-h264",
                     "--items", "index"]) == 0
    assert (flexion.path / cli.VIDEOS_DIR / f"{accepted[0].name}.mp4"
            ).is_file()


def test_the_command_refuses_what_it_cannot_draw(flexion, tmp_path, capsys):
    assert cli.main([str(tmp_path / "nothing")]) == 2
    assert cli.main([str(flexion.path), "--items", "thumb"]) == 1
    assert "No take of: thumb" in capsys.readouterr().out
    grasps = fx.make_grasp_session(tmp_path / "a")
    assert cli.main([str(grasps)]) == 2
    assert "only the glove sets" in capsys.readouterr().out
