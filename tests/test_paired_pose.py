"""The paired-pose tool: one glove + camera take -> each sensor's frame nearest
the photo's moment, the same 21 points in one frame, and where the two disagree.

Synthetic hands come from tests/test_fusion.py (`make_hand`, and the two
writers that put the same hand through the real glove and Leap recorders).
"""
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from cam_hand.prof_format import parse_text
from leap_hand.paired_pose import (INDETERMINATE, LABEL, OPPOSITE, SAME, SIMULTANEITY,
                                   STILL_CAPTURE, TIMING_LABEL,
                                   STILL_MM_S, STILL_MTIME, DegeneratePalm, TakeFrame,
                                   axis_definition, bone_lengths, check_orthonormal,
                                   compare_points, frame_nearest, handedness, palm_side_mm,
                                   same_handedness, stillness, take_moment, to_palm_frame,
                                   window_speed_mm_s)
from test_fusion import _write_glove_take, _write_leap_take, make_hand

REPO = Path(__file__).resolve().parents[1]
DAY2 = REPO / "recordings" / "sync_day2"
DAY2_TAKE = "fist_left_take1_20260920_195141"


def _script():
    path = REPO / "scripts" / "paired_pose_compare.py"
    spec = importlib.util.spec_from_file_location("repo_script_paired_pose_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _hand_mm(**kw) -> np.ndarray:
    return np.asarray(make_hand(**kw), float) * 1000.0


# --- the nearest frame to the photo's moment ------------------------------------
def _frames(captures, walls=None):
    walls = walls or [c + 0.5 for c in captures]
    return [TakeFrame(frame=None, wall_time=w, capture_time=c, line={})
            for c, w in zip(captures, walls)]


def test_frame_nearest_picks_the_nearest_frame_on_capture_time():
    frames = _frames([10.00, 10.02, 10.04])
    near = frame_nearest(frames, 10.031)
    assert near.index == 2 and near.clock == "capture_time"
    assert near.dt == pytest.approx(0.009)          # frame time minus the moment
    near = frame_nearest(frames, 10.029)
    assert near.index == 1 and near.dt == pytest.approx(-0.009)


def test_frame_nearest_refuses_beyond_max_dt():
    frames = _frames([10.00, 10.02, 10.04])
    assert frame_nearest(frames, 10.2) is None                  # 160 ms away
    assert frame_nearest(frames, 10.2, max_dt_s=0.2).index == 2
    assert frame_nearest([], 10.0) is None


def test_frame_nearest_falls_back_to_wall_time_and_says_so():
    frames = _frames([10.00, None, 10.04], walls=[20.00, 20.02, 20.04])
    near = frame_nearest(frames, 20.019)
    assert near.clock == "wall_time" and near.index == 1        # one frame lacks it: no mixing


# --- the common frame -----------------------------------------------------------
def test_palm_frame_coordinates_do_not_depend_on_where_the_hand_is_or_how_it_is_turned():
    pts = _hand_mm(spread_deg=12.0, curl=0.4, thumb_lift_deg=25.0)
    local, basis = to_palm_frame(pts)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) == pytest.approx(1.0)
    assert np.allclose(local[0], 0.0)
    for seed in range(5):
        R = _rotation(seed)
        moved = pts @ R.T + np.array([-37.0, 266.0, 71.0]) * (seed + 1)
        assert np.allclose(to_palm_frame(moved)[0], local, atol=1e-9, rtol=0)


def test_palm_frame_axes_follow_the_named_landmarks():
    local, _ = to_palm_frame(_hand_mm(curl=0.3))
    assert local[9][1] == pytest.approx(0.0, abs=1e-9)          # middle MCP on +x, y = n = 0
    assert local[9][2] == pytest.approx(0.0, abs=1e-9) and local[9][0] > 0
    assert local[5][1] < 0 < local[17][1]                       # y runs index -> pinky
    assert "middle" in axis_definition().lower() and "pinky" in axis_definition().lower()


def test_a_mirrored_hand_flips_same_handedness_and_a_moved_one_does_not():
    pts = _hand_mm(spread_deg=8.0, curl=0.5)
    mirrored = pts * np.array([-1.0, 1.0, 1.0])
    moved = pts @ _rotation(3).T + np.array([5.0, 200.0, -40.0])
    g = to_palm_frame(pts)[0]
    assert same_handedness(g, to_palm_frame(moved)[0]) is True
    assert same_handedness(g, to_palm_frame(mirrored)[0]) is False
    # what the verdict is read from: the points' side of the palm plane
    assert palm_side_mm(to_palm_frame(mirrored)[0]) == pytest.approx(-palm_side_mm(g))


def test_a_degenerate_palm_is_refused_with_the_angle_and_the_basis_is_orthonormal():
    pts = _hand_mm(spread_deg=10.0, curl=0.4)
    for seed in range(5):                                       # the guard in the code path
        basis = to_palm_frame(pts @ _rotation(seed).T, "glove")[1]
        assert np.abs(basis.T @ basis - np.eye(3)).max() < 1e-9
    flat = pts.copy()
    flat[17] = flat[5] + 0.8 * (flat[9] - flat[0])              # P - I along x: angle 0
    with pytest.raises(DegeneratePalm, match=r"degenerate palm frame \(camera, angle 0\.0 deg\)"):
        to_palm_frame(flat, "camera")
    near = pts.copy()                                           # 5 degrees off x: still refused
    x = (near[9] - near[0]) / np.linalg.norm(near[9] - near[0])
    side = np.cross(x, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    near[17] = near[5] + 40.0 * (np.cos(np.radians(5)) * x + np.sin(np.radians(5)) * side)
    with pytest.raises(DegeneratePalm, match=r"\(glove, angle 5\.0 deg\)"):
        to_palm_frame(near, "glove")
    gone = pts.copy()
    gone[9] = np.nan
    with pytest.raises(DegeneratePalm, match="MIDDLE_FINGER_MCP missing or not finite"):
        to_palm_frame(gone, "glove")
    with pytest.raises(AssertionError, match="not orthonormal"):
        check_orthonormal(np.diag([1.0, 1.0, 1.0 + 1e-6]), "glove")


def test_handedness_is_same_opposite_or_indeterminate():
    pts = _hand_mm(spread_deg=8.0, curl=0.5)
    g = to_palm_frame(pts)[0]
    assert handedness(g, to_palm_frame(pts @ _rotation(4).T + 30.0)[0]) == SAME
    assert handedness(g, to_palm_frame(pts * np.array([1.0, -1.0, 1.0]))[0]) == OPPOSITE
    flat = to_palm_frame(_hand_mm(spread_deg=8.0, curl=0.0))[0]  # every point in the palm plane
    assert abs(palm_side_mm(flat)) < 1e-9
    assert handedness(g, flat) == INDETERMINATE and handedness(flat, g) == INDETERMINATE
    assert INDETERMINATE == "indeterminate (margin under 5 mm)"
    assert OPPOSITE == "opposite (suspected mirrored tracker fit)"


def test_stillness_on_a_still_and_a_moving_synthetic_sequence():
    t = np.arange(0.0, 1.0, 1 / 90.0)
    pts = _hand_mm(curl=0.3)
    still = np.repeat(pts[None], t.size, axis=0)
    carried = still + (t * 300.0)[:, None, None]                # the whole hand at 520 mm/s
    assert window_speed_mm_s(t, still, 0.5) == pytest.approx(0.0)
    assert window_speed_mm_s(t, carried, 0.5) == pytest.approx(0.0, abs=1e-9)
    moving = still.copy()
    moving[:, 1:, 2] += (t * 40.0)[:, None]                     # every finger point 40 mm/s
    assert window_speed_mm_s(t, moving, 0.5) == pytest.approx(40.0 * 20 / 21, rel=1e-6)
    assert np.isnan(window_speed_mm_s(t[:1], still[:1], 0.0))
    assert STILL_MM_S == 10.0
    assert stillness(2.0, 3.0) == "still"
    assert stillness(38.0, 3.0) == "moving (glove)"
    assert stillness(2.0, 11.0) == "moving (camera)"
    assert stillness(38.0, 11.0) == "moving (both)"


# --- the comparison -------------------------------------------------------------
def test_compare_points_keeps_the_wrist_out_of_the_statistics():
    g = to_palm_frame(_hand_mm(curl=0.2))[0]
    c = g.copy()
    c[8] = c[8] + np.array([0.0, 0.0, 10.0])        # index tip 10 mm off
    c[0] = c[0] + np.array([500.0, 0.0, 0.0])       # would dominate if it counted
    rows, summary = compare_points(g, c)
    assert len(rows) == 21 and rows[8]["name"] == "INDEX_FINGER_TIP"
    assert rows[8]["distance_mm"] == pytest.approx(10.0)
    assert [r["index"] for r in rows] == list(range(21))
    assert set(rows[0]) >= {"gx", "gy", "gz", "cx", "cy", "cz"}
    assert summary["n_points"] == 20
    assert summary["max_mm"] == pytest.approx(10.0) and summary["max_point"] == "INDEX_FINGER_TIP"
    assert summary["median_mm"] == pytest.approx(0.0)


def test_twenty_bones_with_their_lengths():
    bones = bone_lengths(_hand_mm(spread_deg=5.0, curl=0.3))
    assert len(bones) == 20
    names = [name for name, _ in bones]
    assert names[0] == "WRIST-THUMB_CMC" and "INDEX_FINGER_MCP-INDEX_FINGER_PIP" in names
    in_finger = [mm for name, mm in bones if not name.startswith("WRIST-")]
    assert len(in_finger) == 15 and np.allclose(in_finger, 25.0)    # make_hand's bone


# --- the moment -----------------------------------------------------------------
def _meta(session: Path, stem: str, **fields) -> Path:
    meta = {"accepted": True, "hand": "left", "pose": "fist", "take": 1, "hand_id": 7,
            "still": f"stills\\{stem}.jpg", "file": f"{stem}.jsonl", **fields}
    path = session / "leap" / f"{stem}.meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def _still(session: Path, stem: str, mtime=None) -> Path:
    from PIL import Image
    path = session / "stills" / f"{stem}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (96, 96), 90).save(path)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_the_moment_is_the_photo_s_capture_stamp_then_its_write_time_then_none(tmp_path):
    _meta(tmp_path, "a", still_capture_time=1790000123.25)
    still_a = _still(tmp_path, "a", mtime=1790000999.0)
    m = take_moment(tmp_path, "a")
    assert (m.t, m.source, m.still) == (1790000123.25, STILL_CAPTURE, still_a)

    _meta(tmp_path, "b", still_capture_time=None)               # an old session: no stamp
    still_b = _still(tmp_path, "b", mtime=1790000456.5)
    m = take_moment(tmp_path, "b")
    assert (m.t, m.source, m.still) == (1790000456.5, STILL_MTIME, still_b)
    assert STILL_MTIME == "still file write time (no capture stamp in this take)"

    _meta(tmp_path, "c")                                        # no stamp, no file
    assert take_moment(tmp_path, "c") is None
    _meta(tmp_path, "d", still=None)
    assert take_moment(tmp_path, "d") is None


# --- the script, end to end on synthetic sessions -------------------------------
def _synthetic_take(session: Path, stem: str, hand="left", glove=True, still=True,
                    stamp=True):
    leap = _write_leap_take(session / "leap" / f"{stem}.jsonl", 6.0, 0.35, n=8, hand=hand)
    if glove:
        _write_glove_take(session / "glove" / f"{stem}.jsonl", 6.0, 0.35, n=8, hand=hand)
    walls = [json.loads(s)["wall_time"] for s in leap.read_text(encoding="utf-8").splitlines()]
    moment = walls[len(walls) // 2]
    _meta(session, stem, hand=hand, still_capture_time=moment if stamp else None,
          **({} if still else {"still": None}))
    if still:
        _still(session, stem)
    return moment


def _rows(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_script_writes_both_sensors_nearest_the_moment_in_the_professor_format(tmp_path):
    session, out = tmp_path / "session", tmp_path / "out"
    _synthetic_take(session, "fist_left_take1_x")
    assert _script().main([str(session), "--out", str(out)]) == 0
    take = out / "fist_left_take1_x"
    for name in ("camera.txt", "glove.txt"):
        text = (take / name).read_text(encoding="utf-8")
        blocks = list(parse_text(text))
        assert len(blocks) == 1 and len(blocks[0].points) == 21     # ONE block, 21 landmarks
        assert "Wrist:" in text
    rows = _rows(take / "points_common_frame.csv")
    assert len(rows) == 21 and rows[0]["name"] == "WRIST"
    assert len(_rows(take / "bone_lengths.csv")) == 20
    # the same synthetic hand through both recorders: the same points
    assert max(float(r["distance_mm"]) for r in rows) < 1e-3
    notes = (take / "notes.txt").read_text(encoding="utf-8")
    assert STILL_CAPTURE in notes and "wall_time" in notes       # the clock it fell back to
    assert (take / "figure.png").is_file()
    (row,) = _rows(out / "comparison.csv")
    assert row["status"] == "ok" and row["moment_source"] == STILL_CAPTURE
    assert row["handedness"] == "same" and row["n_points"] == "20"
    assert float(row["glove_z_margin_mm"]) == pytest.approx(float(row["camera_z_margin_mm"]))
    assert row["stillness"] == "still" and float(row["camera_speed_mm_s"]) < 1e-6
    for s in (LABEL, TIMING_LABEL, SIMULTANEITY, "timing uncertainty", "transformation:",
              "visible_time_us 5000000"):
        assert s in notes, s
    assert _script().check_outputs(out) == 0
    assert (out / "figures.pdf").is_file() and (out / "README.md").is_file()


def test_takes_that_cannot_be_processed_are_listed_with_the_reason(tmp_path):
    session, out = tmp_path / "session", tmp_path / "out"
    _synthetic_take(session, "fist_left_take1_a")
    _synthetic_take(session, "fist_left_take2_b", glove=False)
    _synthetic_take(session, "fist_left_take3_c", still=False, stamp=False)
    _meta(session, "fist_left_take4_d", accepted=False)
    assert _script().main([str(session), "--out", str(out)]) == 0
    status = {r["take"]: r["status"] for r in _rows(out / "comparison.csv")}
    assert status["fist_left_take1_a"] == "ok"
    assert "no glove file" in status["fist_left_take2_b"]
    assert "no moment" in status["fist_left_take3_c"]
    assert "not accepted" in status["fist_left_take4_d"]
    assert not (out / "fist_left_take2_b" / "figure.png").exists()


def test_a_take_asked_for_by_name_that_is_not_there_is_listed(tmp_path):
    session, out = tmp_path / "session", tmp_path / "out"
    _synthetic_take(session, "fist_left_take1_a")
    assert _script().main([str(session), "--takes", "fist_left_take1_a", "nope",
                           "--out", str(out)]) == 0
    status = {r["take"]: r["status"] for r in _rows(out / "comparison.csv")}
    assert status == {"fist_left_take1_a": "ok", "nope": "no meta.json for this take"}


# --- the real day-2 take ----------------------------------------------------------
def test_real_day2_take_runs_end_to_end(tmp_path):
    need = [DAY2 / "leap" / f"{DAY2_TAKE}.jsonl", DAY2 / "glove" / f"{DAY2_TAKE}.jsonl",
            DAY2 / "leap" / f"{DAY2_TAKE}.meta.json", DAY2 / "stills" / f"{DAY2_TAKE}.jpg"]
    missing = [p for p in need if not p.is_file()]
    if missing:
        pytest.skip(f"the day-2 recording is not on this machine: {missing[0]} is missing")
    assert _script().main([str(DAY2), "--takes", DAY2_TAKE, "--out", str(tmp_path)]) == 0
    take = tmp_path / DAY2_TAKE
    for name in ("camera.txt", "glove.txt", "points_common_frame.csv", "bone_lengths.csv",
                 "notes.txt", "figure.png"):
        assert (take / name).is_file(), name
    (row,) = _rows(tmp_path / "comparison.csv")
    assert row["status"] == "ok"
    assert row["moment_source"] == STILL_MTIME
    assert abs(float(row["cam_dt_ms"])) <= 50 and abs(float(row["glove_dt_ms"])) <= 50
    assert 0 < float(row["median_mm"]) <= float(row["max_mm"])
    cam = list(parse_text((take / "camera.txt").read_text(encoding="utf-8")))
    assert len(cam) == 1 and len(cam[0].points) == 21
    for name in ("comparison.csv", "figures.pdf", "README.md"):
        assert (tmp_path / name).is_file(), name
