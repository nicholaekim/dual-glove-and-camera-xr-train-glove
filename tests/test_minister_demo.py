"""The visitor demo window (scripts/minister_demo.py): draws the webcam, sensor
and glove panels and both number tables from synthetic sources, says in words
when a source is missing, and runs headless."""
import importlib.util
import py_compile
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "minister_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("repo_script_minister_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lime(img, y0, y1, x0, x1, m):
    region = img[y0:y1, x0:x1]
    return int(np.all(region == np.array(m.LIME, np.uint8), axis=2).sum())


def _panel_lime(canvas, x0, w, m):
    return _lime(canvas, m.PANEL_Y, m.PANEL_Y + m.PANEL_H - 40, x0, x0 + w, m)     # above the status line


def _all_on(m, t=0.0):
    return m.compose(m.mock_webcam_frame(t), [m.mock_mp_hand(t)], m.mock_ir_image(t), m.mock_leap_hands(t),
                     m.mock_to_px, m.mock_glove_hands(t), (True, ""), (True, ""), (True, ""))


def test_script_compiles():
    py_compile.compile(str(SCRIPT), doraise=True)


def test_every_panel_draws_a_synthetic_hand():
    m = _load()
    canvas = _all_on(m)
    assert canvas.shape == (m.CANVAS_H, m.CANVAS_W, 3)
    assert _panel_lime(canvas, m.WEB_X, m.WEB_W, m) > 200          # the webcam skeleton
    assert _panel_lime(canvas, m.IR_X, m.IR_W, m) > 200            # the sensor skeletons
    assert _panel_lime(canvas, m.GLOVE_X, m.GLOVE_W, m) > 200      # the glove hands


def test_missing_sources_draw_words_not_a_crash():
    m = _load()
    none = {"left": None, "right": None}
    canvas = m.compose(None, [], None, [], m.mock_to_px, none, (False, "no webcam found"),
                       (False, "tracking service not running"), (False, "port 9002 busy"))
    assert canvas.shape == (m.CANVAS_H, m.CANVAS_W, 3)
    for x0, w in ((m.WEB_X, m.WEB_W), (m.IR_X, m.IR_W), (m.GLOVE_X, m.GLOVE_W)):
        assert _panel_lime(canvas, x0, w, m) == 0


def test_status_lines_are_plain_words():
    m = _load()
    none = {"left": None, "right": None}
    assert m.webcam_status([], True)[0] == "no hand in the picture yet"
    assert m.webcam_status([m.mock_mp_hand()], True)[0] == "hand found, 21 points"
    assert m.webcam_status([], False, "no webcam found")[0] == "webcam not available: no webcam found"
    assert m.leap_status([], True)[0] == "no hand over the sensor yet"
    assert m.leap_status([m.mock_leap_hand(dx=0.0)], True)[0] == "hand tracked in 3D   22 cm above the sensor"
    assert m.leap_status(m.mock_leap_hands(), True)[0] == "both hands tracked in 3D"
    assert m.leap_status([], False, "no sensor found on USB")[0] == "sensor not connected: no sensor found on USB"
    assert m.glove_status(none, True)[0] == "no glove data yet: start XR Train and connect a glove"
    assert m.glove_status({"left": None, "right": m.mock_glove_pts("right")}, True)[0] == "right glove streaming"
    assert m.glove_status(m.mock_glove_hands(), True)[0] == "both gloves streaming"
    assert m.glove_status(none, False, "port 9002 busy")[0] == "glove stream not available: port 9002 busy"


def test_sensor_points_are_palm_and_five_tips_in_centimetres():
    m = _load()
    pts = m.sensor_points(m.mock_leap_hand(dx=0.0))
    assert list(pts) == ["palm", "thumb tip", "index tip", "middle tip", "ring tip", "pinky tip"]
    assert pts["palm"] == (0.0, 22.0, 0.0)                         # the mock palm sits 220 mm up
    assert all(abs(p[1] - 22.2) < 0.05 for n, p in pts.items() if n != "palm")


def test_glove_points_are_knuckle_middle_and_five_tips_from_the_wrist():
    m = _load()
    pts21 = [(0.001 * k, 0.002 * k, 0.0) for k in range(21)]       # point k at (k mm, 2k mm)
    pts = m.glove_points(pts21)
    assert pts["palm"] == (1.1, 2.2, 0.0)                          # knuckles 5, 9, 13, 17 average to 11
    assert pts["index tip"] == (0.8, 1.6, 0.0)
    assert pts["pinky tip"] == (2.0, 4.0, 0.0)


def _values(panel, m):
    """Pixels in the colour the live numbers are printed in."""
    return int(np.all(panel == np.array(m.VALUE, np.uint8), axis=2).sum())


def test_each_table_lists_only_the_hands_it_has():
    m = _load()
    left, right = m.mock_leap_hands()
    assert m.hands_by_side([left, right]) == {"left": left, "right": right}
    assert m.hands_by_side([right]) == {"left": None, "right": right}
    both = m.table_rows({"left": m.sensor_points(left), "right": m.sensor_points(right)})
    assert [r[0] for r in both] == ["hand", "x", "y", "z", "hand", "x", "y", "z"]
    assert both[0][1] == "left" and both[4][1] == "right"
    assert both[2][1] == ["22.0", "22.2", "22.2", "22.2", "22.2", "22.2"]   # the y row: every mock point 22 cm up
    only_right = m.table_rows({"left": None, "right": m.sensor_points(right)})
    assert [r[0] for r in only_right] == ["hand", "x", "y", "z"] and only_right[0][1] == "right"
    assert m.table_rows({"left": None, "right": None}) == []
    # with bends: three more rows per hand, a dash in the palm column, whole degrees
    with_bends = m.table_rows({"left": None, "right": m.sensor_points(right)},
                              {"left": None, "right": m.joint_bends(m.leap_pts21(right))})
    assert [r[0] for r in with_bends] == ["hand", "x", "y", "z", "knuckle", "middle", "end"]
    assert with_bends[4][1][0] == "-" and len(with_bends[4][1]) == 6
    assert all(t.lstrip("-").isdigit() for t in with_bends[5][1][1:])
    # and the drawn card: numbers only when there is a hand, one line of words otherwise
    # (a stray anti-aliased pixel of the white title can match the number colour, hence < 5)
    assert _values(m.sensor_table([right]), m) > 100
    assert _values(m.sensor_table([]), m) < 5
    assert _values(m.glove_table({"left": m.mock_glove_pts("left"), "right": None}), m) > 100
    assert _values(m.glove_table({"left": None, "right": None}), m) < 5


def test_joint_frames_are_right_handed_unit_triads_with_z_along_the_bone():
    m = _load()
    pts_mm = np.asarray(m.mock_glove_pts("right"), float) * 1000.0
    frames = m.joint_frames21(pts_mm, "right")
    assert len(frames) == 21
    for i, (o, x, y, z) in enumerate(frames):
        for v in (x, y, z):
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6
        assert abs(np.dot(x, y)) < 1e-6 and abs(np.dot(y, z)) < 1e-6 and abs(np.dot(z, x)) < 1e-6
        assert np.allclose(np.cross(y, z), x)                         # right-handed
        parent = m.MP_PARENT[i]
        if parent is not None:
            assert np.allclose(z, m._unit(pts_mm[parent] - pts_mm[i]))  # toward the wrist


def test_axes_are_drawn_on_the_glove_and_sensor_hands_and_can_be_switched_off():
    m = _load()
    red = np.array(m.AXIS_COLOURS["x"], np.uint8)
    glove_on = m.glove_panel(m.mock_glove_hands(), axes=True)
    glove_off = m.glove_panel(m.mock_glove_hands(), axes=False)
    assert int(np.all(glove_on == red, axis=2).sum()) > 50
    assert int(np.all(glove_off == red, axis=2).sum()) == 0
    sensor_on = m.leap_panel(m.mock_ir_image(), m.mock_leap_hands(), m.mock_to_px, axes=True)
    sensor_off = m.leap_panel(m.mock_ir_image(), m.mock_leap_hands(), m.mock_to_px, axes=False)
    assert int(np.all(sensor_on == red, axis=2).sum()) > 50
    assert int(np.all(sensor_off == red, axis=2).sum()) == 0


def test_joint_angles_read_zero_straight_and_ninety_at_a_right_angle():
    m = _load()
    straight = m.mock_glove_pts("right", t=-np.pi / 2)             # curl 0: every finger a straight line
    angles = m.joint_angles21(np.asarray(straight) * 1000.0)
    assert sorted(angles) == [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]   # no wrist, no tips
    assert all(abs(a) < 1e-3 for i, a in angles.items() if i not in (1, 5, 9, 13, 17))   # within a finger
    pts = [(0.0, 0.0, 0.0)] * 21
    pts[5], pts[6], pts[7] = (0.0, 0.08, 0.0), (0.0, 0.12, 0.0), (0.04, 0.12, 0.0)     # index bent 90 at the middle joint
    assert abs(m.joint_angles21(np.asarray(pts) * 1000.0)[6] - 90.0) < 1e-3
    bends = m.joint_bends(np.asarray(pts) * 1000.0)
    assert list(bends) == ["knuckle", "middle", "end"] and abs(bends["middle"][1] - 90.0) < 1e-3   # index = column 1
    panel_on = m.glove_panel(m.mock_glove_hands(), axes=False, angles=True)
    panel_off = m.glove_panel(m.mock_glove_hands(), axes=False, angles=False)
    black = np.array((0, 0, 0), np.uint8)
    assert int(np.all(panel_on == black, axis=2).sum()) > 200       # the dark boxes behind the numbers
    assert int(np.all(panel_off == black, axis=2).sum()) == 0


def test_mock_projection_keeps_both_hands_inside_the_sensor_picture():
    m = _load()
    for hand in m.mock_leap_hands():
        for chain in hand["chains"]:
            for p in chain:
                px = m.mock_to_px(p)
                assert px is not None
                assert 0 <= px[0] < m.IR_W and 0 <= px[1] < m.IR_W, (hand["side"], px)


def test_mock_glove_source_feeds_both_hands_through_the_real_parser():
    import time
    m = _load()
    src = m.GloveSource(mock=True)
    try:
        time.sleep(0.2)
        src.poll()
        hands = src.hands()
    finally:
        src.close()
    for side in ("left", "right"):
        assert hands[side] is not None and len(hands[side]) == 21
        assert hands[side][0] == (0.0, 0.0, 0.0)                   # wrist-centred


def test_headless_mock_run_writes_a_snapshot(tmp_path):
    png = tmp_path / "demo.png"
    r = subprocess.run([sys.executable, str(SCRIPT), "--mock", "--no-window", "--frames", "5", "--snapshot", str(png)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "glove: synthetic" in r.stdout and "Drew 5 frame(s)." in r.stdout
    assert png.is_file() and png.stat().st_size > 10_000
