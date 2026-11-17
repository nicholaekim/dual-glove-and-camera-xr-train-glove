"""Replay viewer on the Agg backend: controls, playback on the camera clock,
source-frame stepping, interpolation, references, overlay, timeline markers,
HUD and Paper 3 text, screenshots and the timing log.

The recording is built here, without the module under test: a synthetic left
hand in the LeapC bone convention (-z along each bone, +y dorsal), moved
rigidly from frame to frame, converted with `absolute_to_relative`, written as
JSONL lines shaped like the recorder's (abs26 rounded to 1e-6 m) and read back
with `replay_core.load_recording`. 30 frames at 90 Hz with one dropped frame
id, a hand-id change (two segments), a writer stall and a declared static
interval. The last test runs the scenario screenshots on the real dataset and
skips on a machine that does not have it.
"""
import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from leap_hand import angle_coverage as ac
from leap_hand import paper3
from leap_hand import replay_core as rc
from leap_hand import tmc_calibration_take as tct
from leap_hand import tmc_estimate as te
from leap_hand.replay_viewer import (ANG_TITLE, ANG_UNAVAILABLE, C_WARN, CLOCK_OK,
                                     EST_TITLE, EST_TITLES, P3_TITLE, RECORDED_NA,
                                     RECT_3D, RECT_HUD, TICK_MS, ReplayViewer)
from PIL import Image
from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES
from xr_hand.kinematics import absolute_to_relative, mat3_to_quat

REPO = Path(__file__).resolve().parents[1]
DATASET = Path(r"C:\Users\nkim2\OneDrive\Desktop\xr trainer"
               r"\grasp set 2026-10-08 organized")
needs_dataset = pytest.mark.skipif(not DATASET.exists(),
                                   reason="real dataset not on this machine")
spec = importlib.util.spec_from_file_location(
    "replay_view_script", REPO / "scripts" / "leap" / "replay_view.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

N = 30
PERIOD_US = 11_111
TS0 = 7_000_000_000
DROP_AT = 8            # one frame id skipped before this index (a dropped frame)
ID_CHANGE_AT = 15      # hand_id 11 -> 12 from this index on (a new segment)
STALL_AT = 20          # the writer stalls 0.3 s before this line is written
WRIST = JOINT_INDEX["WRIST"]
FINGERS = ("INDEX", "MIDDLE", "RING", "LITTLE")


# --- a synthetic left hand ----------------------------------------------------
def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rodrigues(axis, deg):
    k = unit(axis)
    t = math.radians(deg)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * (K @ K)


def leap_basis(bone_dir, dorsal=(0.0, 1.0, 0.0)):
    """Columns x, y, z: -z along the bone, +y dorsal (a proper rotation)."""
    z = -unit(bone_dir)
    d = np.asarray(dorsal, dtype=float)
    y = unit(d - np.dot(d, z) * z)
    return np.column_stack([np.cross(y, z), y, z])


def left_hand_local():
    """26 positions (m) and rotations of a left hand, palm down, fingers along
    -z, thumb toward +x, every finger a little flexed."""
    p = {"WRIST": np.zeros(3), "PALM": np.array([0.004, -0.006, -0.048])}
    for k, f in enumerate(FINGERS):
        x0 = 0.024 - 0.017 * k
        base = np.array([0.4 * x0, -0.003, -0.012])
        mcp = np.array([x0, -0.002, -0.080 + 0.004 * k])
        p[f + "_METACARPAL"], p[f + "_PROXIMAL"] = base, mcp
        d, cur = unit(mcp - base), mcp
        for seg, length, flex in (("_INTERMEDIATE", 0.042 - 0.003 * k, 18 + 4 * k),
                                  ("_DISTAL", 0.026, 40), ("_TIP", 0.020, 62)):
            cur = cur + length * (rodrigues((1.0, 0.0, 0.0), -flex) @ d)
            p[f + seg] = cur
    d = unit((0.65, -0.30, -0.70))
    bend = np.cross(d, (0.0, 1.0, 0.0))
    cur = np.array([0.020, -0.012, -0.010])
    p["THUMB_METACARPAL"] = cur
    for name, length, deg in (("THUMB_PROXIMAL", 0.046, 0.0),
                              ("THUMB_DISTAL", 0.032, -15.0),
                              ("THUMB_TIP", 0.026, -35.0)):
        cur = cur + length * (rodrigues(bend, deg) @ d)
        p[name] = cur
    r = {"WRIST": leap_basis((-0.05, 0.02, -1.0)),
         "PALM": leap_basis(p["MIDDLE_PROXIMAL"] - p["WRIST"])}
    for f in ("THUMB",) + FINGERS:
        chain = [n for n in JOINT_NAMES if n.startswith(f + "_")]
        for a, b in zip(chain[:-1], chain[1:]):
            r[a] = leap_basis(p[b] - p[a])
        r[chain[-1]] = r[chain[-2]]
    return (np.array([p[n] for n in JOINT_NAMES]),
            np.array([r[n] for n in JOINT_NAMES]))


LOCAL = left_hand_local()


def frame_line(k):
    """Line k: the hand turned 2 deg a frame about y and slid 1 mm in x."""
    pos, rot = LOCAL
    R = rodrigues((0.0, 1.0, 0.0), 2.0 * k)
    pos_w, rot_w = pos @ R.T + np.array([0.001 * k - 0.05, 0.20, 0.02]), R @ rot
    joints = absolute_to_relative(pos_w, [mat3_to_quat(m) for m in rot_w])
    ts = TS0 + (k + (k >= DROP_AT)) * PERIOD_US
    capture = round(1_790_000_000.0 + (ts - TS0) / 1e6 + 0.0004, 6)
    frame_id = 1000 + k + (k >= DROP_AT)
    return {"wall_time": capture + 0.02 + (0.3 if k >= STALL_AT else 0.0),
            "timestamp": ts / 1e6, "timestamp_us": ts, "packet_counter": frame_id,
            "hand_side": "left", "frame_id": frame_id, "status": 1,
            "hand_id": 11 if k < ID_CHANGE_AT else 12, "framerate": 90.0,
            "capture_time": capture,
            "joints": [{"name": j.name, "x": j.x, "y": j.y, "z": j.z, "qw": j.qw,
                        "qx": j.qx, "qy": j.qy, "qz": j.qz} for j in joints],
            "abs26": [[round(float(c), 6) for c in q] for q in pos_w],
            "source": "leap", "space": "leap_desktop", "units": "m"}


LINES = [frame_line(k) for k in range(N)]
METADATA = {"static_interval_first_frame_id": "1020",
            "static_interval_last_frame_id": "1025",
            "representative_frame_id": "1022"}
PROVENANCE = "test fixture: synthetic left hand, 30 frames"


def write_jsonl(path, lines):
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return path


@pytest.fixture
def jsonl(tmp_path):
    return write_jsonl(tmp_path / "synthetic_left.jsonl", LINES)


@pytest.fixture
def rec(jsonl):
    return rc.load_recording(jsonl, "left", [11, 12], provenance=PROVENANCE,
                             kind="static", name="p3_circular_power", take=1,
                             metadata=METADATA)


@pytest.fixture
def viewer(rec, tmp_path):
    v = ReplayViewer(rec, dpi=100, screenshot_dir=tmp_path / "screens")
    yield v
    v.close()


def key(v, name):
    v._on_key(SimpleNamespace(key=name))


def latest_at_or_before(t, media):
    """Independent of pose_at: the last index whose time is <= media."""
    return max(i for i, ti in enumerate(t) if ti <= media)


def test_fixture_has_the_events_the_tests_rely_on(rec):
    assert len(rec.frames) == N and rec.timeline.source == "timestamp_us"
    assert [(e.kind, e.index) for e in rec.events] == [
        ("dropped_frames", DROP_AT), ("id_change", ID_CHANGE_AT)]
    assert [(s.start, s.end, s.hand_id) for s in rec.segments] == [
        (0, ID_CHANGE_AT - 1, 11), (ID_CHANGE_AT, N - 1, 12)]
    assert rc.hold_mask(rec).sum() == 6


# --- construction and exact source frames ---------------------------------------
def test_builds_on_agg_paused_in_hold_mode_at_a_recorded_frame(viewer):
    assert matplotlib.get_backend().lower() == "agg"
    v = viewer
    assert (v.mode, v.reference, v.playing, v.speed) == ("hold", "camera", False, 1.0)
    assert (v.overlay, v.axes_on, v.frame_index) == (False, False, 0)
    assert v.displayed.source == "recorded" and v.displayed.i0 == 0
    assert v.hud_panel.rows > 15 and not v.hud_panel.truncated
    assert v.p3_panel.rows > 30 and not v.p3_panel.truncated
    assert v.ang_panel.rows > 45 and not v.ang_panel.truncated


def test_seek_and_step_land_on_exact_source_frames(viewer, rec):
    v = viewer
    for i in (0, 5, 17, N - 1):
        v.seek_frame(i)
        assert v.frame_index == i and v.slider.val == i
        assert v.displayed.source == "recorded" and v.displayed.i0 == i
        assert v.media_t == rec.timeline.t[i]
        np.testing.assert_array_equal(v.displayed.pos, rc.reconstruct(rec.frames[i]).pos)
        assert f"frame_id {rec.frames[i].frame_id}" in v.hud_text()
        assert "displayed pose: recorded" in v.hud_text()
    v.seek_frame(5)
    v.step(+1)
    assert (v.frame_index, v.displayed.source) == (6, "recorded")
    v.step(-1)
    v.step(-1)
    assert (v.frame_index, v.displayed.source) == (4, "recorded")
    v.seek_frame(N - 1)
    v.step(+1)
    assert v.frame_index == N - 1                      # clamped at the end
    v.seek_frame(-7)
    assert v.frame_index == 0
    v.seek_frame(10_000)
    assert v.frame_index == N - 1


def test_step_back_from_between_two_frames_lands_on_the_earlier_one(viewer, rec):
    t = rec.timeline.t
    viewer.seek_time((t[3] + t[4]) / 2)
    assert viewer.displayed.source == "held" and viewer.frame_index == 3
    viewer.step(-1)
    assert viewer.frame_index == 3 and viewer.displayed.source == "recorded"


# --- playback on the camera clock -------------------------------------------------
def test_tick_shows_the_latest_frame_at_or_before_the_media_time(viewer, rec):
    v, t = viewer, rec.timeline.t
    v.toggle_play(now=100.0)
    assert v.playing
    v.tick(now=100.1)
    assert v.media_t == pytest.approx(0.1)
    assert v.frame_index == latest_at_or_before(t, 0.1)
    assert v.displayed.source == "held"                 # 0.1 s is not a frame time
    v.set_speed(2.0, now=100.1)
    v.tick(now=100.15)                                  # 0.05 s of wall time at 2x
    assert v.media_t == pytest.approx(0.2)
    assert v.frame_index == latest_at_or_before(t, 0.2)
    v.toggle_play(now=100.15)                           # pause
    frozen = (v.media_t, v.frame_index)
    v.tick(now=200.0)
    assert not v.playing and (v.media_t, v.frame_index) == frozen
    assert v.timing_summary()["n"] == 2


def test_playing_to_the_end_pauses_on_the_last_frame_and_play_restarts(viewer, rec):
    v = viewer
    v.toggle_play(now=0.0)
    v.tick(now=60.0)
    assert not v.playing and v.frame_index == N - 1
    assert v.displayed.source == "recorded" and v.media_t == rec.timeline.t[-1]
    v.toggle_play(now=61.0)
    assert v.playing and v.frame_index == 0 and v.media_t == 0.0
    v.toggle_play(now=161.0)               # paused past the end, before any tick
    assert not v.playing and v.media_t == rec.timeline.t[-1]
    assert v.frame_index == N - 1 and v.displayed.source == "recorded"


def test_lateness_and_skipped_frames_are_logged_per_tick(viewer, rec):
    v, t = viewer, rec.timeline.t
    v.toggle_play(now=10.0)                # frame 0 on screen
    v.tick(now=10.1)                       # due at TICK_MS of media time, came at 0.1
    a = v.frame_index
    v.tick(now=10.2)                       # due at 0.1 + TICK_MS, came at 0.2
    b = v.frame_index
    s = v.timing_summary()
    assert s["n"] == 2 and 1 < a < b
    late = 100.0 - TICK_MS
    assert s["mean_ms"] == pytest.approx(late) and s["max_ms"] == pytest.approx(late)
    skipped = (a - 1) + (b - a - 1)        # frames never on screen since frame 0
    assert s["frames_skipped"] == skipped
    v.seek_frame(25, now=10.2)             # a jump while playing: a new run
    v.tick(now=10.21)
    assert v.frame_index == latest_at_or_before(t, t[25] + 0.01) == 25
    assert v.timing_summary()["frames_skipped"] == skipped    # the jump is not counted
    assert v.timing_summary()["runs"] == 2


# --- interpolation ------------------------------------------------------------------
def test_interpolation_is_off_by_default_and_never_crosses_the_id_change(viewer, rec):
    v, t = viewer, rec.timeline.t
    v.seek_time((t[2] + t[3]) / 2)
    assert v.displayed.source == "held" and v.displayed.i0 == 2       # hold mode
    key(v, "i")
    assert v.mode == "interpolate"
    d = v.displayed
    assert d.source == "interpolated" and (d.i0, d.i1) == (2, 3)
    assert d.alpha == pytest.approx(0.5)
    assert "displayed pose: interpolated (2..3 alpha 0.500)" in v.hud_text()
    v.seek_time((t[DROP_AT - 1] + t[DROP_AT]) / 2)                    # dropped frame
    assert v.displayed.source == "interpolated"
    assert "dropped frame ids" in v.displayed.note
    v.seek_time((t[ID_CHANGE_AT - 1] + t[ID_CHANGE_AT]) / 2)          # id change
    d = v.displayed
    assert d.source == "held" and d.i0 == ID_CHANGE_AT - 1
    assert "discontinuity" in d.note
    assert "displayed pose: held (discontinuity (id_change)" in v.hud_text()
    v.set_mode("hold")
    v.seek_time((t[2] + t[3]) / 2)
    assert v.displayed.source == "held"


# --- overlay, axes and references ----------------------------------------------------
def test_overlay_shows_abs26_at_source_frames_and_hides_between_them(viewer, rec):
    v, t = viewer, rec.timeline.t
    v.seek_frame(4)
    assert v.shown["recorded"] is None and not v._rec_markers.get_visible()
    key(v, "o")
    assert v.overlay and v._rec_markers.get_visible() and v._residuals.get_visible()
    np.testing.assert_array_equal(v.shown["recorded"], rec.frames[4].abs_recorded)
    residual = np.linalg.norm(v.shown["pos"] - v.shown["recorded"], axis=1)
    ff = rc.frame_fidelity(rec.frames[4])
    np.testing.assert_allclose(residual, ff["position_residual_m"]["per_joint"],
                               atol=1e-15)
    assert 0.0 < residual.max() <= 1e-6                  # abs26 rounding only
    v.set_reference("wrist")                             # same residual, moved
    residual_w = np.linalg.norm(v.shown["pos"] - v.shown["recorded"], axis=1)
    np.testing.assert_allclose(residual_w, residual, atol=1e-12)
    v.set_reference("camera")
    v.set_mode("interpolate")
    v.seek_time((t[2] + t[3]) / 2)
    assert v.displayed.source == "interpolated"
    assert v.shown["recorded"] is None and not v._rec_markers.get_visible()
    assert not v._residuals.get_visible()
    assert RECORDED_NA in v.hud_text()
    key(v, "o")
    assert not v.overlay and RECORDED_NA not in v.hud_text()


def test_axes_toggle_draws_one_triad_per_joint(viewer):
    v = viewer
    assert not any(c.get_visible() for c in v._triads)
    key(v, "a")
    assert v.axes_on and all(c.get_visible() for c in v._triads)
    for k, coll in enumerate(v._triads):
        segs = np.asarray(coll._segments3d)
        assert segs.shape == (26, 2, 3)
        np.testing.assert_allclose(segs[:, 1] - segs[:, 0],
                                   0.01 * v.shown["rot"][:, :, k], atol=1e-15)
    key(v, "a")
    assert not v.axes_on


def test_reference_cycle_labels_and_limits(viewer, rec):
    v = viewer
    hud = v.hud_text()
    assert ("reference: camera space (LeapC desktop): +x right, +y up, "
            "+z toward operator, metres") in hud
    limits = (v.ax.get_xlim(), v.ax.get_ylim(), v.ax.get_zlim())
    every = np.concatenate([rc.reconstruct(f).pos for f in rec.frames])
    for k, (lo, hi) in enumerate(limits):
        assert lo < every[:, k].min() and hi > every[:, k].max()
    for i in (3, 12, 27):                                 # never re-centred
        v.seek_frame(i)
        assert (v.ax.get_xlim(), v.ax.get_ylim(), v.ax.get_zlim()) == limits
    key(v, "r")
    assert v.reference == "wrist"
    assert ("reference: wrist-relative: origin WRIST joint, axes = WRIST "
            "orientation (LeapC forearm on the camera)") in v.hud_text()
    np.testing.assert_allclose(v.shown["pos"][WRIST], 0.0, atol=1e-15)
    lo, hi = v.ax.get_xlim()
    assert lo == pytest.approx(-hi)
    key(v, "r")
    assert v.reference == "palm"
    assert ("reference: palm-relative: origin WRIST joint, axes = PALM "
            "orientation") in v.hud_text()
    key(v, "r")
    assert v.reference == "camera"
    assert (v.ax.get_xlim(), v.ax.get_ylim(), v.ax.get_zlim()) == limits
    with pytest.raises(ValueError):
        v.set_reference("elbow")


# --- timeline and events ------------------------------------------------------------
def test_timeline_has_one_marker_per_event_and_the_hold_band(viewer, rec, jsonl):
    markers = viewer.timeline_markers
    assert len(markers) == len(rec.events) + 1
    assert sorted(k for k, _, _ in markers) == sorted(
        [e.kind for e in rec.events] + ["hold"])
    assert [(k, i) for k, i, _ in markers if k != "hold"] == [
        (e.kind, e.index) for e in rec.events]
    assert all(arts for _, _, arts in markers)
    plain = rc.load_recording(jsonl, "left", [11, 12])   # no static interval
    v2 = ReplayViewer(plain, dpi=100)
    try:
        assert len(v2.timeline_markers) == len(plain.events)
    finally:
        v2.close()


def test_next_and_previous_event_pause_on_the_event_frames(viewer):
    v = viewer
    v.toggle_play(now=0.0)
    key(v, "n")
    assert not v.playing and v.frame_index == DROP_AT
    assert v.displayed.source == "recorded"
    assert "event at this frame: dropped_frames" in v.hud_text()
    v.next_event()
    assert v.frame_index == ID_CHANGE_AT
    assert "event at this frame: id_change" in v.hud_text()
    v.next_event()
    assert v.frame_index == ID_CHANGE_AT and "no later event" in v.hud_text()
    key(v, "p")
    assert v.frame_index == DROP_AT
    v.prev_event()
    assert v.frame_index == DROP_AT and "no earlier event" in v.hud_text()


def test_speed_and_play_keys(viewer):
    v = viewer
    for k, expected in (("+", 2.0), ("+", 4.0), ("+", 4.0), ("-", 2.0)):
        key(v, k)
        assert v.speed == expected
    for _ in range(6):
        key(v, "-")
    assert v.speed == 0.1
    assert [h["speed"] for h in v.speed_history][:3] == [1.0, 2.0, 4.0]
    key(v, " ")
    assert v.playing and v.buttons["play"].label.get_text() == "Pause"
    key(v, " ")
    assert not v.playing and v.buttons["play"].label.get_text() == "Play"
    key(v, "right")
    assert v.displayed.source == "recorded"
    with pytest.raises(ValueError):
        v.set_speed(0.0)


# --- HUD and Paper 3 text ------------------------------------------------------------
def test_hud_names_the_line_the_clock_and_the_fidelity_comparisons(viewer, rec):
    v = viewer
    v.seek_frame(STALL_AT)
    hud = v.hud_text()
    f = rec.frames[STALL_AT]
    for text in (rec.path.name, PROVENANCE, CLOCK_OK, "kind static",
                 "name p3_circular_power", "take 1", "hand left", "hand ids 11 12",
                 f"frame {STALL_AT}/{N}", f"frame_id {f.frame_id}", "hand_id 12",
                 "status 1", "segment 2/2",
                 f"wall_time {f.wall_time!r}", f"capture_time {f.capture_time!r}",
                 f"timestamp_us {f.timestamp_us}",
                 "FK vs recorded abs26",
                 "matrix chain vs quaternion chain (same data, two code paths)",
                 "bone -z axis vs recorded bone vector",
                 "units: positions m", "inside the static interval (hold)",
                 "writer stall", "speed 1x   mode hold"):
        assert text in hud, text
    ff = rc.frame_fidelity(f)
    assert (f"position residual max {ff['position_residual_m']['max'] * 1e6:.3f} µm"
            in hud)
    assert "Paper 3" not in hud                          # the research panel is apart
    v.seek_frame(STALL_AT - 1)
    assert "writer stall" not in v.hud_text()


def test_hud_shows_a_clock_fallback_in_the_warning_colour(tmp_path):
    lines = json.loads(json.dumps(LINES))
    lines[5]["timestamp_us"] = lines[4]["timestamp_us"]     # repeated camera stamp
    path = write_jsonl(tmp_path / "fallback.jsonl", lines)
    rec = rc.load_recording(path, "left", [11, 12])
    assert rec.timeline.fallback and rec.timeline.source == "capture_time"
    v = ReplayViewer(rec, dpi=100)
    try:
        clock = [(text, color) for text, color in v._hud_lines
                 if text.startswith("clock:")]
        assert len(clock) == 1
        assert clock[0][0].startswith("clock: capture_time (FALLBACK")
        assert rec.timeline.reason in clock[0][0]
        assert clock[0][1] == C_WARN
        assert "clock_defect" in [k for k, _, _ in v.timeline_markers]
    finally:
        v.close()


def test_paper3_panel_is_separate_and_marks_the_unavailable_angles(viewer, rec):
    v = viewer
    v.seek_frame(7)
    text = v.paper3_text()
    lines = text.splitlines()
    assert lines[0] == P3_TITLE
    assert lines[1].startswith("class: circular (")
    for name in paper3.UNAVAILABLE:
        assert any(name in ln and "unavailable" in ln for ln in lines), name
    assert paper3.COMPARE_NOTE.split(". ")[0] in text
    r = rc.reconstruct(rec.frames[7])
    rows = paper3.frame_relation_values(
        paper3.frame_angles(r.pos, r.rot, "left"), "circular")
    assert sum(ln.split(" ")[0] in {x["id"] for x in rows} for ln in lines) == len(rows)
    for row in rows:
        line = next(ln for ln in lines if ln.split(" ")[0] == row["id"])
        assert row["status"] in line
        if row["residual"] == "unavailable":
            assert line.endswith("unavailable")
        else:
            assert f"{row['residual']:+7.1f}".strip() in line
            assert f"{row['lhs_value']:7.1f}".strip() in line
    assert "fidelity" not in text.replace(P3_TITLE, "")   # numbers kept apart


# --- the angle panel -------------------------------------------------------------------
SIX = ("T_TMC_fe", "T_TMC_aa", "I_CMC", "M_CMC", "R_CMC", "L_CMC")
DIGITS = ("THUMB", "INDEX", "MIDDLE", "RING", "LITTLE")


def deg(v):
    """How the angle panel prints a value (0.1 deg, signed, no '-0.0')."""
    text = f"{v:+.1f}"
    return "0.0" if float(text) == 0.0 else text


def row_of(lines, column):
    """The one panel line whose column field is `column`."""
    hits = [ln for ln in lines if f" {column} " in ln]
    assert len(hits) == 1, (column, hits)
    return hits[0]


def check_angle_rows(lines, angles):
    for e in ac.COVERAGE:
        if e.status not in ("calculated", "outside_paper3") or e.joint.startswith("every"):
            continue
        for column in e.output_columns:
            line = row_of(lines, column)
            assert line.endswith(ac.STATUS_LABELS[e.status]), line
            assert f" {deg(angles[column])} " in line, (column, line)
    for digit in DIGITS:                          # twist on the digit's own line
        line = next(ln for ln in lines if ln.startswith(digit + " "))
        assert " twist " in line
        for joint in ac.TWIST_JOINTS:
            if joint.startswith(digit + "_"):
                bone = joint.split("_", 1)[1][:4]
                value = deg(angles[f"{joint}_twist_tracker"])
                assert f"{bone:<4} {value:>5}" in line, (joint, line)


def test_angle_panel_lists_each_digits_angles_with_column_and_status(viewer, rec):
    v = viewer
    v.seek_frame(7)
    lines = v.angles_text().splitlines()
    assert lines[0] == ANG_TITLE == ("Angles of the displayed source frame (degrees; "
                                     "calculated = tracker coordinate)")
    assert lines.count("flexion + toward the palm; abduction + toward the thumb "
                       "side; both hands") == 1
    r = rc.reconstruct(rec.frames[7])
    angles = ac.all_angles(r.pos, r.rot, "left")
    check_angle_rows(lines, angles)
    for letter in "IMRL":                         # the MCP twice, labelled by order
        for k in ("fe", "aa"):
            xyz = row_of(lines, f"{letter}_MCP_{k}")
            paper = row_of(lines, f"{letter}_MCP_{k}_paper_order")
            assert xyz.startswith(f"MCP {k} (xyz, flexion first) ")
            assert paper.startswith(f"MCP {k} (paper order, abduction first) ")
            assert xyz.endswith(" calculated") and paper.endswith(" calculated")
    text = "\n".join(lines)
    for label in (" calculated", " outside Paper 3"):
        assert label in text
    assert ANG_UNAVAILABLE == ("unavailable: T_TMC_fe, T_TMC_aa, I_CMC, M_CMC, R_CMC, "
                               "L_CMC (not measured)")
    assert not [ln for ln in lines if "mapped" in ln]    # no mapping: no mapped block
    assert ANG_UNAVAILABLE in lines
    for name in SIX:                              # named, never given a number
        assert [ln for ln in lines if name in [w.strip(",") for w in ln.split()]
                ] == [ANG_UNAVAILABLE], name
    k = lines.index(EST_TITLE)                    # the estimate, apart and last
    assert EST_TITLE == ("estimated (open-hand calibration; not a validated Paper 3 "
                         "coordinate)")
    assert k > lines.index(ANG_UNAVAILABLE) and lines[k - 1] == ""
    assert lines[k + 1:] == ["T_TMC_fe_est_openhand: not computed (no calibration loaded)",
                             "T_TMC_aa_est_openhand: not computed (no calibration loaded)"]
    assert "fidelity" not in text
    panels = v.panel_texts()
    assert panels["angles"] == text and set(panels) == {"hud", "angles", "paper3"}
    for other in (v.hud_text(), v.paper3_text()):
        assert "_est_openhand" not in other and "_tracker" not in other


def test_angle_panel_on_an_interpolated_pose(viewer, rec):
    v, t = viewer, rec.timeline.t
    v.set_mode("interpolate")
    v.seek_time((t[2] + t[3]) / 2)
    d = v.displayed
    assert d.source == "interpolated"
    lines = v.angles_text().splitlines()
    assert (f"interpolated between frame_id {rec.frames[2].frame_id} and "
            f"{rec.frames[3].frame_id} (not a recorded pose)") in lines
    check_angle_rows(lines, ac.all_angles(d.pos, d.rot, "left"))
    assert not v.ang_panel.truncated


def test_a_tmc_provider_feeds_only_the_estimated_block(rec, tmp_path):
    calls = []

    def provider(pos, rot, hand_side):
        calls.append((np.array(pos), np.array(rot), hand_side))
        return {"T_TMC_fe_est_openhand": 12.345, "T_TMC_aa_est_openhand": None,
                "provenance": "fake open-hand calibration, 3 transitions"}

    v = ReplayViewer(rec, dpi=100, tmc_provider=provider,
                     screenshot_dir=tmp_path / "screens")
    try:
        v.seek_frame(4)
        lines = v.angles_text().splitlines()
        k = lines.index(EST_TITLE)
        assert lines[k + 1:] == [
            "T_TMC_fe_est_openhand: +12.3 deg (unvalidated estimate)",
            "T_TMC_aa_est_openhand: not computed (the provider gave no value)",
            "provenance: fake open-hand calibration, 3 transitions"]
        pos, rot, side = calls[-1]
        r = rc.reconstruct(rec.frames[4])
        np.testing.assert_array_equal(pos, r.pos)
        np.testing.assert_array_equal(rot, r.rot)
        assert side == "left"
        assert ANG_UNAVAILABLE in lines            # the validated pair stays unavailable
        for other in (v.hud_text(), v.paper3_text()):
            assert "fake open-hand" not in other and "_est_openhand" not in other
        assert not v.ang_panel.truncated
        png = v.screenshot(tmp_path / "with_estimate.png")
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        v.close()


# --- output files ---------------------------------------------------------------------
def test_screenshot_timeline_crop_and_timing_log(viewer, tmp_path):
    v = viewer
    png = v.screenshot(tmp_path / "shot.png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    crop = v.screenshot_timeline(tmp_path / "timeline.png")
    assert crop.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    key(v, "s")                                         # the key's stamped file
    assert len(list((tmp_path / "screens").glob("p3_circular_power_take1_*.png"))) == 1
    v.toggle_play(now=0.0)
    v.tick(now=0.05)
    v.tick(now=0.10)
    out = v.write_timing_log(tmp_path / "timing" / "log.json", label="unit test")
    data = json.loads(out.read_text(encoding="utf-8"))
    for k in ("n", "mean_ms", "p95_ms", "max_ms", "frames_skipped", "backend",
              "speed_history", "what_is_logged", "timer_interval_ms", "ticks"):
        assert k in data, k
    assert data["backend"].lower() == "agg" and data["label"] == "unit test"
    assert data["n"] == 2 == len(data["ticks"])
    assert data["recording"]["name"] == "p3_circular_power"
    assert data["ticks"][0]["lateness_ms"] == pytest.approx(50.0 - TICK_MS)
    for k in ("rendered_updates", "frames_shown", "wall_duration_s", "rendered_per_s",
              "frames_shown_per_s", "runs_detail"):
        assert k in data, k


# --- rendering counts and the drawn pixels (Codex finding 1, 2026-10-24) ---------------
def test_timing_summary_counts_rendered_updates_and_frames_shown_per_run(viewer, rec):
    v, t = viewer, rec.timeline.t
    v.fig.canvas.draw()                              # a full draw: the background exists
    assert v._bg is not None
    before = v.timing_summary()["rendered_updates"]
    v.toggle_play(now=20.0)                          # run 1 from frame 0
    for k in range(1, 6):
        v.tick(now=20.0 + 0.02 * k)                  # a new frame every 20 ms of wall time
    s = v.timing_summary()
    (run,) = s["runs_detail"]
    assert run["run"] == 1 and run["ticks"] == 5 and run["speed"] == 1.0
    assert run["rendered_updates"] == 5              # one blit per tick, counted
    assert run["frames_shown"] == 5                  # every tick landed on a new frame
    assert run["frames_skipped"] == s["frames_skipped"] > 0   # 20 ms steps over 11 ms frames
    assert run["wall_duration_s"] == pytest.approx(0.1)
    assert run["rendered_per_s"] == pytest.approx(50.0) == run["frames_shown_per_s"]
    assert s["wall_duration_s"] == pytest.approx(0.1) and s["frames_shown"] == 5
    assert s["rendered_updates"] >= before + 5       # play's own draw also counts
    # a tick that lands on the frame already shown renders but shows nothing new
    v.tick(now=20.1001)
    (run,) = v.timing_summary()["runs_detail"]
    assert run["ticks"] == 6 and run["rendered_updates"] == 6 and run["frames_shown"] == 5
    # a seek starts a second run; the first run's record is kept
    v.seek_frame(25, now=20.2)
    v.tick(now=20.21)
    runs = v.timing_summary()["runs_detail"]
    assert [r["run"] for r in runs] == [1, 2] and runs[1]["ticks"] == 1
    assert runs[0]["ticks"] == 6                     # unchanged
    v.toggle_play(now=20.3)                          # pause: the run is closed
    v.tick(now=20.4)                                 # paused: no tick is recorded
    assert v.timing_summary()["runs_detail"][1]["ticks"] == 1


def test_screenshot_shows_the_hand_and_the_atlas_rendered_text(viewer, tmp_path):
    """The hand's artists are animated and drawn on top of the background;
    the panels are composed from the glyph atlas. Both must reach the pixels
    of a saved figure (and of the blitted window, which uses the same code)."""
    v = viewer
    png = v.screenshot(tmp_path / "shot.png")
    img = np.asarray(Image.open(png).convert("RGB")).astype(int)
    H, W = img.shape[:2]

    def region(rect):
        x0, x1 = int(rect[0] * W), int((rect[0] + rect[2]) * W)
        y1, y0 = int((1 - rect[1]) * H), int((1 - rect[1] - rect[3]) * H)
        return img[y0:y1, x0:x1]
    hand = region(RECT_3D)
    pane = np.array([10, 10, 18])                   # the 3D pane colour (0.04, 0.04, 0.07)
    bright = (np.abs(hand - pane).sum(axis=2) > 60).sum()
    assert bright > 500, bright                      # bones, joints and tips are drawn
    hud = region(RECT_HUD)
    text_px = (hud.max(axis=2) > 120).sum()          # light glyphs on the dark panel
    assert text_px > 2000, text_px
    assert v.hud_panel.rows > 20 and not v.hud_panel.truncated
    assert v.ang_panel.rows > 40 and v.p3_panel.rows > 10
    # the glyph atlas holds what the panels used: digits, letters, the degree-ish signs
    atlas = next(iter(v.hud_panel._atlases.values()))
    assert {"0", "9", "a", "e", ".", "µ"} <= set(atlas.ids)
    assert atlas.strip("ab").shape == (atlas.h, 2 * atlas.adv)
    assert atlas.strip("").shape == (atlas.h, 0)


# --- the script ---------------------------------------------------------------------
def test_script_headless_screenshot_and_argument_checks(jsonl, tmp_path, capsys):
    png = tmp_path / "out" / "view.png"
    log = tmp_path / "out" / "timing.json"
    assert script.main(["--file", str(jsonl), "--hand", "left", "--hand-ids", "11 12",
                        "--mode", "interpolate", "--reference", "palm", "--overlay",
                        "--axes", "--start-frame", "5", "--headless",
                        "--screenshot", str(png), "--timing-log", str(log)]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads(log.read_text(encoding="utf-8"))["n"] == 0
    assert "screenshot:" in capsys.readouterr().out
    for argv in ([], ["--grasp", "p3_circular_power"],
                 ["--dataset", str(tmp_path), "--grasp", "p3_circular_power",
                  "--take", "1", "--hand-ids", "11"],
                 ["--file", str(jsonl)],
                 ["--file", str(jsonl), "--hand", "left", "--headless"],
                 ["--scenario-screenshots", str(tmp_path)],
                 ["--dataset", str(tmp_path), "--scenario-screenshots", str(tmp_path),
                  "--file", str(jsonl)]):
        with pytest.raises(SystemExit) as e:
            script.main(argv)
        assert e.value.code == 2, argv
    assert script.main(["--dataset", str(tmp_path), "--grasp", "nothing", "--take",
                        "1", "--headless", "--screenshot", str(png)]) == 1


# --- the estimated block from the script ------------------------------------------------
EST_NONE_LINES = ["T_TMC_fe_est_openhand: not computed (no calibration loaded)",
                  "T_TMC_aa_est_openhand: not computed (no calibration loaded)"]


def write_calibration(path, rec, fe, aa, session="synthetic", kind="openhand", hand=None,
                      provenance="hand-made test calibration", drop=()):
    """A hand-made Calibration (to_json) whose rest frame reads the fixture's
    palm-to-metacarpal rotation (the same in every frame: the hand moves
    rigidly) as (fe, aa) on this left hand: R_rest = R_pm (Ry(-aa) Rx(-fe))^T.
    `drop` removes keys from the written JSON (a file of an older format)."""
    pm = te.palm_to_metacarpal(rc.reconstruct(rec.frames[0]).rot)
    cal = te.Calibration(R_rest=pm @ (jf.rot_y(-aa) @ jf.rot_x(-fe)).T, session=session,
                         n_transitions=3, n_frames=75, per_transition=[],
                         agreement={"median_deg": 4.4, "p90_deg": 7.8, "max_deg": 20.0,
                                    "worst": {"name": "x", "take": 1}},
                         provenance=provenance, kind=kind, hand=hand)
    doc = {k: v for k, v in cal.to_json().items() if k not in drop}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def run_headless(jsonl, png, monkeypatch, *extra):
    """(exit code, angle panel lines) of the script, headless at source frame
    5 of the fixture file; the lines as the viewer had them when the script
    closed it."""
    seen = []
    close = ReplayViewer.close

    def capture(self):
        seen.append(self.angles_text().splitlines())
        close(self)
    monkeypatch.setattr(ReplayViewer, "close", capture)
    code = script.main(["--file", str(jsonl), "--hand", "left", "--hand-ids", "11 12",
                        "--start-frame", "5", "--headless", "--screenshot", str(png),
                        *map(str, extra)])
    return code, (seen[-1] if seen else None)


def test_script_builds_the_estimate_from_a_calibration_file(jsonl, rec, tmp_path,
                                                            monkeypatch, capsys):
    cal_path = write_calibration(tmp_path / "calibration.json", rec, 12.0, -7.0, hand="left")
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch,
                               "--tmc-calibration", cal_path)
    assert code == 0
    k = lines.index(EST_TITLE)
    assert lines[k + 1:] == [
        "T_TMC_fe_est_openhand: +12.0 deg (unvalidated estimate)",
        "T_TMC_aa_est_openhand: -7.0 deg (unvalidated estimate)",
        "T_TMC_twist_est_openhand: 0.0 deg (residual 0.0 deg)",
        "provenance: open hand, 3 transitions; agreement median 4.4/p90 7.8/max 20.0 deg"]
    assert ANG_UNAVAILABLE in lines              # the validated pair stays unavailable
    out = capsys.readouterr().out
    assert f"TMC calibration: {cal_path}" in out and "hand-made test calibration" in out
    # the numbers are tmc_estimate's on the shown frame
    r = rc.reconstruct(rec.frames[5])
    e = te.estimate(r.pos, r.rot, "left", script.load_calibration(cal_path))
    assert (deg(e["T_TMC_fe_est_openhand"]), deg(e["T_TMC_aa_est_openhand"])) == (
        "+12.0", "-7.0")


def test_script_without_a_calibration_says_not_computed(jsonl, rec, tmp_path,
                                                        monkeypatch, capsys):
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch, "--no-tmc")
    assert code == 0 and lines[lines.index(EST_TITLE) + 1:] == EST_NONE_LINES
    assert "TMC estimate: off (--no-tmc)" in capsys.readouterr().out
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch)   # --file alone
    assert code == 0 and lines[lines.index(EST_TITLE) + 1:] == EST_NONE_LINES
    assert "no --dataset, no --tmc-calibration" in capsys.readouterr().out
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch,
                               "--tmc-calibration", tmp_path / "missing.json")
    assert code == 1 and lines is None
    assert "cannot read the TMC calibration" in capsys.readouterr().out
    cal_path = write_calibration(tmp_path / "calibration.json", rec, 1.0, 2.0)
    with pytest.raises(SystemExit) as e:
        script.main(["--file", str(jsonl), "--hand", "left", "--no-tmc",
                     "--tmc-calibration", str(cal_path)])
    assert e.value.code == 2


def test_a_calibration_of_another_session_is_not_applied(jsonl, rec, tmp_path):
    other = rc.load_recording(jsonl, "left", [11, 12], provenance=PROVENANCE,
                              kind="static", name="p3_circular_power", take=1,
                              metadata={**METADATA, "session": "S1"})
    cal = script.load_calibration(
        write_calibration(tmp_path / "calibration.json", rec, 12.0, -7.0, session="S0",
                          hand="left"))
    v = ReplayViewer(other, dpi=100, tmc_provider=script.tmc_provider(cal, other),
                     screenshot_dir=tmp_path / "screens")
    try:
        lines = v.angles_text().splitlines()
        assert lines[lines.index(EST_TITLE) + 1:] == [
            "T_TMC_fe_est_openhand: not computed (the provider gave no value)",
            "T_TMC_aa_est_openhand: not computed (the provider gave no value)",
            "provenance: calibration of session S0, this recording is of session S1: "
            "not applied"]
    finally:
        v.close()


# --- the calibration kind through the real CLI path (Codex finding 4, 2026-10-24) -------
NEUTRAL_BLOCK = [
    EST_TITLES["neutral"],
    "T_TMC_fe_est_neutral: +12.0 deg (unvalidated estimate)",
    "T_TMC_aa_est_neutral: -7.0 deg (unvalidated estimate)",
    "T_TMC_twist_est_neutral: 0.0 deg (residual 0.0 deg)",
    "provenance: neutral posture take, 3 takes; agreement median 4.4/p90 7.8/max 20.0 deg"]


def test_script_routes_a_neutral_calibration_to_the_neutral_block(jsonl, rec, tmp_path,
                                                                 monkeypatch, capsys):
    cal_path = write_calibration(tmp_path / "neutral.json", rec, 12.0, -7.0,
                                 kind="neutral", hand="left",
                                 provenance="tmc_neutral calibration of session X (left hand)")
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch,
                               "--tmc-calibration", cal_path)
    assert code == 0
    assert lines[lines.index(EST_TITLES["neutral"]):] == NEUTRAL_BLOCK
    assert EST_TITLE not in lines and not any("_est_openhand" in l for l in lines)
    assert ANG_UNAVAILABLE in lines                 # the validated pair stays unavailable
    out = capsys.readouterr().out
    assert "kind: neutral (tmc_neutral calibration take" in out
    assert "hand left; outputs T_TMC_fe_est_neutral, T_TMC_aa_est_neutral" in out


def test_script_shows_a_real_calibration_take_under_the_neutral_names(tmp_path, monkeypatch,
                                                                      capsys):
    """End to end with the real writer: synthetic tmc_neutral takes written the
    way the recorder writes them, read by tmc_calibration_take.calibrate, saved
    as calibration.json, loaded by the script, applied to a loose camera file
    whose thumb sits at (20, 10) from the same rest frame."""
    from test_tmc_calibration_take import N_HOLD, write_take
    from test_tmc_estimate import hand_pose, line as est_line
    from test_tmc_estimate import write_jsonl as est_write_jsonl
    folder = tmp_path / "20990101_000000_left"
    write_take(folder, 1, N_HOLD)
    write_take(folder, 2, N_HOLD)
    cal = tct.calibrate(folder)
    cal_path = folder / "calibration.json"
    cal_path.write_text(json.dumps(cal.to_json()), encoding="utf-8")
    assert json.loads(cal_path.read_text(encoding="utf-8"))["kind"] == "neutral"
    loose = tmp_path / "loose_left.jsonl"
    pose = hand_pose("left", 20.0, 10.0)
    est_write_jsonl(loose, [est_line("left", 5, 100 + k, 5_000_000_000 + k * 11_111, pose)
                            for k in range(30)])
    seen = []
    close = ReplayViewer.close

    def capture(self):
        seen.append(self.angles_text().splitlines())
        close(self)
    monkeypatch.setattr(ReplayViewer, "close", capture)
    code = script.main(["--file", str(loose), "--hand", "left", "--headless",
                        "--screenshot", str(tmp_path / "view.png"),
                        "--tmc-calibration", str(cal_path)])
    assert code == 0
    lines = seen[-1]
    k = lines.index(EST_TITLES["neutral"])
    assert lines[k + 1:k + 4] == ["T_TMC_fe_est_neutral: +20.0 deg (unvalidated estimate)",
                                  "T_TMC_aa_est_neutral: +10.0 deg (unvalidated estimate)",
                                  "T_TMC_twist_est_neutral: 0.0 deg (residual 0.0 deg)"]
    assert lines[k + 4].startswith("provenance: neutral posture take, 2 takes; agreement")
    assert not any("_est_openhand" in l or "open hand" in l for l in lines)
    out = capsys.readouterr().out
    assert "kind: neutral" in out and tct.PROVENANCE_PREFIX in out
    # the standalone neutral estimator agrees with what the window shows
    e = tct.estimate(*pose, "left", script.load_calibration(cal_path))
    assert (deg(e[tct.FE]), deg(e[tct.AA])) == ("+20.0", "+10.0")
    # and the other hand's recording does not get it
    right = rc.load_recording(loose, "left", None, kind="file")
    right.hand_side = "right"
    prov = script.tmc_provider(script.load_calibration(cal_path), right)
    got = prov(*pose, "right")
    assert got["kind"] == "neutral" and got["T_TMC_fe_est_neutral"] is None
    assert got["provenance"] == ("calibration of the left hand, this recording is of the "
                                 "right hand: not applied")


def test_older_calibration_files_are_read_by_their_provenance_or_refused(jsonl, rec, tmp_path,
                                                                        monkeypatch, capsys):
    """A calibration.json written before `kind` existed: the first words of
    the provenance say which writer made it; anything else is refused with
    the fix spelled out, never guessed."""
    open_hand = write_calibration(tmp_path / "old_openhand.json", rec, 12.0, -7.0,
                                  provenance="open-hand calibration of session X: ...",
                                  drop=("kind", "hand"))
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch,
                               "--tmc-calibration", open_hand)
    k = lines.index(EST_TITLE)                       # the kind is read; the hand is not known
    assert code == 0 and lines[k + 1] == ("T_TMC_fe_est_openhand: not computed (the provider "
                                          "gave no value)")
    assert lines[k + 3] == f"provenance: {te.HAND_UNKNOWN}: not applied"
    out = capsys.readouterr().out
    assert "hand not recorded in the file (no --dataset to recover it from)" in out
    assert "declare it with --tmc-calibration-hand left|right" in out
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch,
                               "--tmc-calibration", open_hand, "--tmc-calibration-hand", "left")
    assert code == 0 and lines[lines.index(EST_TITLE) + 1].startswith(
        "T_TMC_fe_est_openhand: +12.0 deg")
    assert "hand left (declared with --tmc-calibration-hand)" in capsys.readouterr().out
    neutral = write_calibration(tmp_path / "old_neutral.json", rec, 12.0, -7.0,
                                provenance="tmc_neutral calibration of session X (left hand)",
                                drop=("kind", "hand"))
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch,
                               "--tmc-calibration", neutral)
    k = lines.index(EST_TITLES["neutral"])
    assert code == 0 and lines[k + 1].endswith("not computed (the provider gave no value)")
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch,
                               "--tmc-calibration", neutral, "--tmc-calibration-hand", "left")
    assert code == 0 and lines[lines.index(EST_TITLES["neutral"]) + 1].startswith(
        "T_TMC_fe_est_neutral: +12.0 deg")
    unknown = write_calibration(tmp_path / "old_unknown.json", rec, 12.0, -7.0,
                                drop=("kind", "hand"))
    code, lines = run_headless(jsonl, tmp_path / "view.png", monkeypatch,
                               "--tmc-calibration", unknown)
    out = capsys.readouterr().out
    assert code == 1 and lines is None
    assert "cannot read the TMC calibration" in out and 'no "kind"' in out
    assert '"kind": "openhand"' in out and '"kind": "neutral"' in out


def right_handed(jsonl):
    """The fixture's frames read back as a right-hand recording (the data is
    left-hand; only the label matters for the routing checks)."""
    rec = rc.load_recording(jsonl, "left", [11, 12], kind="file")
    rec.hand_side = "right"
    return rec


@pytest.mark.parametrize("kind", te.KINDS)
def test_an_older_file_without_a_hand_is_applied_to_no_hand_unless_declared(
        jsonl, rec, tmp_path, monkeypatch, capsys, kind):
    """Codex's recheck of 2026-10-24: the default calibration file predates
    the hand field. Such a file is never applied to a right-hand recording,
    never assumed to fit a left one, and applies only once its hand is
    declared (or recovered from the dataset's table); a declaration that
    contradicts a recorded hand is refused."""
    fe, aa = ac.ESTIMATE_COLUMNS_BY_KIND[kind]
    title = EST_TITLES[kind]
    path = write_calibration(tmp_path / f"old_{kind}.json", rec, 12.0, -7.0, kind=kind,
                             provenance=f"{te.PROVENANCE_PREFIXES[kind]} of session X",
                             drop=("hand",))
    cal = script.load_calibration(path)
    assert cal.kind == kind and cal.hand is None
    for recording in (rec, right_handed(jsonl)):
        r = rc.reconstruct(recording.frames[0])
        got = script.tmc_provider(cal, recording)(r.pos, r.rot, recording.hand_side)
        assert got[fe] is None and got[aa] is None and got["kind"] == kind
        assert got["provenance"] == f"{te.HAND_UNKNOWN}: not applied"
    code, lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-calibration", path)
    k = lines.index(title)
    assert code == 0 and lines[k + 1] == f"{fe}: not computed (the provider gave no value)"
    assert lines[k + 2] == f"{aa}: not computed (the provider gave no value)"
    assert lines[k + 3] == f"provenance: {te.HAND_UNKNOWN}: not applied"
    out = capsys.readouterr().out
    assert "hand not recorded in the file" in out and "--tmc-calibration-hand" in out
    code, lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-calibration", path,
                               "--tmc-calibration-hand", "left")
    assert code == 0 and lines[lines.index(title) + 1] == f"{fe}: +12.0 deg (unvalidated estimate)"
    assert "hand left (declared with --tmc-calibration-hand)" in capsys.readouterr().out
    code, lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-calibration", path,
                               "--tmc-calibration-hand", "right")
    assert code == 0 and lines[lines.index(title) + 3] == (
        "provenance: calibration of the right hand, this recording is of the left hand: "
        "not applied")
    typed = write_calibration(tmp_path / f"typed_{kind}.json", rec, 12.0, -7.0, kind=kind,
                              hand="left")
    code, lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-calibration",
                               typed, "--tmc-calibration-hand", "right")
    assert code == 1 and lines is None
    assert "records the left hand; the declared right hand contradicts it" in (
        capsys.readouterr().out)


@pytest.mark.parametrize("kind", te.KINDS)
def test_a_known_hand_applies_only_to_recordings_of_that_hand(jsonl, rec, tmp_path, kind):
    fe = ac.ESTIMATE_COLUMNS_BY_KIND[kind][0]
    recordings = {"left": rec, "right": right_handed(jsonl)}
    for cal_hand in ("left", "right"):
        cal = script.load_calibration(write_calibration(
            tmp_path / f"{kind}_{cal_hand}.json", rec, 12.0, -7.0, kind=kind, hand=cal_hand))
        for rec_hand, recording in recordings.items():
            r = rc.reconstruct(recording.frames[0])
            got = script.tmc_provider(cal, recording)(r.pos, r.rot, rec_hand)
            assert got["kind"] == kind
            if rec_hand == cal_hand:
                assert got[fe] is not None and "not applied" not in got["provenance"]
            else:
                assert got[fe] is None
                assert got["provenance"] == (f"calibration of the {cal_hand} hand, this "
                                             f"recording is of the {rec_hand} hand: not applied")


def test_a_calibration_of_another_hand_is_not_applied(jsonl, rec, tmp_path):
    cal = script.load_calibration(
        write_calibration(tmp_path / "right.json", rec, 12.0, -7.0, hand="right"))
    v = ReplayViewer(rec, dpi=100, tmc_provider=script.tmc_provider(cal, rec),
                     screenshot_dir=tmp_path / "screens")
    try:
        lines = v.angles_text().splitlines()
        assert lines[lines.index(EST_TITLE) + 1:] == [
            "T_TMC_fe_est_openhand: not computed (the provider gave no value)",
            "T_TMC_aa_est_openhand: not computed (the provider gave no value)",
            "provenance: calibration of the right hand, this recording is of the left "
            "hand: not applied"]
    finally:
        v.close()


def test_the_default_calibration_then_the_computed_one(rec, tmp_path, monkeypatch):
    """With --dataset: the default calibration.json when it exists, else
    tmc_estimate.calibrate over the dataset (timed), else not computed."""
    args = script.build_parser().parse_args(
        ["--dataset", str(tmp_path), "--grasp", "g", "--take", "1"])
    path = write_calibration(tmp_path / "default.json", rec, 1.0, 2.0)
    monkeypatch.setattr(script, "DEFAULT_TMC", path)
    cal, line = script.tmc_calibration(args)
    assert line.startswith(f"TMC calibration: {path}; hand not recorded in the file (the "
                           "calibration lists no transitions to match): the estimate is not "
                           "applied; declare it with --tmc-calibration-hand")
    assert cal.hand is None and cal.provenance == "hand-made test calibration"
    # an older default file whose transitions the dataset's table names: the hand
    # comes from the table, the one place it is recorded
    from test_tmc_estimate import write_table
    write_table(tmp_path / "metadata" / "pose_transitions.csv", [
        {"participant_id": "P01", "hand": "left", "from_grasp": "open_hand", "to_grasp": "g",
         "take": 1, "recording_file": "x.jsonl", "validity": "valid", "frames": 10,
         "hand_ids": "1", "session": "S"}])
    older = te.Calibration(R_rest=np.eye(3), session="S", n_transitions=1, n_frames=10,
                           per_transition=[{"name": "g", "take": 1, "n_frames": 10}],
                           agreement={"median_deg": 0.0, "p90_deg": 0.0, "max_deg": 0.0,
                                      "worst": {"name": "g", "take": 1}},
                           provenance="open-hand calibration of session S", kind="openhand")
    (tmp_path / "older.json").write_text(
        json.dumps({k: v for k, v in older.to_json().items() if k != "hand"}),
        encoding="utf-8")
    monkeypatch.setattr(script, "DEFAULT_TMC", tmp_path / "older.json")
    cal, line = script.tmc_calibration(args)
    assert cal.hand == "left"
    assert line.endswith("hand left (recovered from metadata/pose_transitions.csv: 1 of 1 "
                         "transitions of session S are left-hand recordings)")
    args_declared = script.build_parser().parse_args(
        ["--dataset", str(tmp_path), "--grasp", "g", "--take", "1",
         "--tmc-calibration-hand", "right"])
    cal, line = script.tmc_calibration(args_declared)          # the declaration wins
    assert cal.hand == "right" and line.endswith("hand right (declared with --tmc-calibration-hand)")
    monkeypatch.setattr(script, "DEFAULT_TMC", path)            # a typed file: agrees or raises
    assert script.tmc_calibration(args_declared)[0].hand == "right"
    typed = write_calibration(tmp_path / "typed.json", rec, 1.0, 2.0, hand="left")
    monkeypatch.setattr(script, "DEFAULT_TMC", typed)
    cal, line = script.tmc_calibration(args_declared)   # the default file: reported, replay opens
    assert cal is None and "records the left hand; the declared right hand contradicts it" in line
    assert script.tmc_calibration(args)[0].hand == "left"
    cal, line = script.tmc_calibration(args)
    assert line.endswith("hand left (recorded in the file)")
    monkeypatch.setattr(script, "DEFAULT_TMC", tmp_path / "absent.json")
    calls = []
    monkeypatch.setattr(script.te, "calibrate", lambda root: calls.append(root) or cal)
    computed, line = script.tmc_calibration(args)
    assert computed is cal and calls == [tmp_path]
    assert line.startswith("TMC calibration: computed from the dataset's transitions in ")
    assert line.endswith(f"s (no {tmp_path / 'absent.json'})")

    def fails(root):
        raise ValueError("no usable transition")
    monkeypatch.setattr(script.te, "calibrate", fails)
    assert script.tmc_calibration(args) == (
        None, "TMC estimate: not computed (calibration failed: ValueError: no usable "
              "transition)")
    broken = tmp_path / "broken.json"                # a default that cannot be read
    broken.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(script, "DEFAULT_TMC", broken)
    cal, line = script.tmc_calibration(args)
    assert cal is None and line.startswith(f"TMC estimate: not computed (cannot read {broken}")


@needs_dataset
def test_scenario_screenshots_on_the_real_dataset(tmp_path, capsys):
    out_dir = tmp_path / "screens"
    assert script.main(["--dataset", str(DATASET),
                        "--scenario-screenshots", str(out_dir)]) == 0
    out = capsys.readouterr().out
    m = re.search(r"SCREENSHOTS WRITTEN (\d+) files to (.+)", out)
    assert m and Path(m.group(2).strip()) == out_dir
    pngs = sorted(p.name for p in out_dir.glob("*.png"))
    assert int(m.group(1)) == len(pngs) == 19
    for stem in ("p3_circular_power_take1", "p3_prismatic_precision_take1",
                 "p3_circular_precision_thumb_2_fingers_take4",
                 "p2_s03_power_grasp_finger_side_take1"):
        for state in ("camera", "wrist", "overlay", "axes"):
            assert f"{stem}_{state}.png" in pngs
    assert "p3_circular_precision_thumb_2_fingers_take4_writer_stall.png" in pngs
    assert "p2_s03_power_grasp_finger_side_take1_interpolate_gap.png" in pngs
    assert "p2_s03_power_grasp_finger_side_take1_timeline.png" in pngs
    assert "interpolation at the gap: held (discontinuity (id_change, gap)" in out
    assert "TMC calibration: " in out          # the screenshots carry the estimated block
    timing = json.loads((out_dir / "timing_headless_agg.json").read_text(encoding="utf-8"))
    # how many ticks fit in 2 s depends on the machine's load: at least one
    assert timing["label"] == "headless Agg draw loop" and timing["n"] >= 1
    assert timing["backend"].lower() == "agg" and len(timing["ticks"]) == timing["n"]


# --- the reference-mapped block (leap_hand.reference_validation) --------------------
from leap_hand import reference_validation as rvl   # noqa: E402


def write_mapping(path, rec, fe, aa, *, status="synthetic_demonstration", hand="left",
                  session="S1", participant="P01", order="abduction_then_flexion",
                  tested=((-10.0, 40.0), (0.0, 30.0))):
    """A hand-made mapping whose A turns the fixture's thumb metacarpal head
    direction (R_pm (0, 0, -1), the same in every frame) onto the canonical
    direction of (fe, aa) written out here for a left hand (s = -1): in the
    abduction-then-flexion order u = (sin(-fe), sin aa cos fe, cos aa cos fe).
    So the block must read (fe, aa) exactly."""
    pm = te.palm_to_metacarpal(rc.reconstruct(rec.frames[0]).rot)
    v = pm @ np.array([0.0, 0.0, -1.0])
    f, a = math.radians(fe), math.radians(aa)
    u = np.array([math.sin(-f), math.sin(a) * math.cos(f), math.cos(a) * math.cos(f)])
    axis = np.cross(v, u)
    A = rodrigues(axis, math.degrees(math.atan2(np.linalg.norm(axis), float(v @ u))))
    doc = {"format": rvl.MAPPING_FORMAT, "id": "tmcmap-test", "status": status,
           "participant": participant, "hand": hand, "session": session,
           "rotation_order": order, "A": A.tolist(), "b": [0.0, 0.0, -1.0],
           "tested_range_deg": {"fe": list(tested[0]), "aa": list(tested[1])},
           "heldout": {"fe": {"n": 120, "mae_deg": 1.25}, "aa": {"n": 120, "mae_deg": 0.75}}}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def sidecar(jsonl, **fields):
    jsonl.with_suffix(rvl.META_SUFFIX).write_text(json.dumps(fields), encoding="utf-8")


def mapped_rows(lines):
    k = next(i for i, ln in enumerate(lines) if ln.startswith("mapped ("))
    return lines[k], lines[k + 1], k


@pytest.mark.parametrize("status,tag", [
    ("synthetic_demonstration", "SYNTHETIC, not measured"),
    ("evaluated", "reference-evaluated"),
    ("cross_check", "cross-check only")])
def test_mapping_block_shows_values_in_scope_with_what_the_reference_was(
        jsonl, rec, tmp_path, monkeypatch, capsys, status, tag):
    sidecar(jsonl, participant="P01", session="S1")
    m = write_mapping(tmp_path / "mapping.json", rec, 20.0, 10.0, status=status)
    code, lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-mapping", m)
    assert code == 0
    first, second, k = mapped_rows(lines)
    assert first == f"mapped ({tag}): T_TMC_fe_mapped +20.0 T_TMC_aa_mapped +10.0"
    assert second == ("tested fe -10..40, aa 0..30 deg; P01, left hand, session S1; "
                      "tmcmap-test")
    assert lines[k - 1] == ANG_UNAVAILABLE                  # right under the unavailable six
    assert lines[k + 2] == "" and lines[k + 3] == EST_TITLE   # the estimate apart, last
    out = capsys.readouterr().out
    assert "TMC mapping:" in out and "tmcmap-test" in out
    assert "applied to this recording (inside its tested range only)" in out
    assert ANG_UNAVAILABLE in lines                         # the Paper 3 pair stays unavailable


def test_mapping_is_not_applied_outside_its_scope(jsonl, rec, tmp_path, monkeypatch, capsys):
    m = write_mapping(tmp_path / "mapping.json", rec, 20.0, 10.0)
    cases = [({"participant": "P01", "session": "S2"},
              "this recording is of session S2, the mapping of session S1"),
             ({"participant": "P02", "session": "S1"},
              "this recording is of participant P02, the mapping of participant P01"),
             (None, "this recording names no session and no participant")]
    for fields, reason in cases:
        side = jsonl.with_suffix(rvl.META_SUFFIX)
        side.unlink(missing_ok=True)
        if fields is not None:
            sidecar(jsonl, **fields)
        code, lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-mapping", m)
        first, second, _k = mapped_rows(lines)
        assert code == 0
        assert first == "mapped (SYNTHETIC, not measured): not applied to this recording"
        assert second.startswith(reason), second
        assert not [ln for ln in lines if "T_TMC_fe_mapped" in ln]
        assert f"not applied to this recording: {reason}" in capsys.readouterr().out
    sidecar(jsonl, participant="P01", session="S1")
    m = write_mapping(tmp_path / "right.json", rec, 20.0, 10.0, hand="right")
    code, lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-mapping", m)
    assert mapped_rows(lines)[1] == ("this recording is of the left hand, the mapping "
                                     "of the right hand")


def test_mapping_values_outside_the_tested_range_stay_empty(jsonl, rec, tmp_path,
                                                            monkeypatch):
    sidecar(jsonl, participant="P01", session="S1")
    m = write_mapping(tmp_path / "mapping.json", rec, 20.0, 10.0,
                      tested=((-10.0, 15.0), (0.0, 30.0)))
    code, lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-mapping", m)
    first, second, _k = mapped_rows(lines)
    assert first == "mapped (SYNTHETIC, not measured): empty here"
    assert second.startswith("fe 20.0, aa 10.0 deg outside the tested range fe -10.0..15.0")


def test_mapping_file_that_is_not_a_mapping_is_refused(jsonl, tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something else"}), encoding="utf-8")
    code, _lines = run_headless(jsonl, tmp_path / "v.png", monkeypatch, "--tmc-mapping", bad)
    assert code == 1
    assert "cannot read the TMC mapping: a mapping file has format" in capsys.readouterr().out


def test_status_rows_fit_the_real_window_in_the_worst_case(rec, jsonl, tmp_path):
    """Interpolated pose, an estimate with its twist and provenance, and the
    mapped block, on the TkAgg window's 804 px height at 100 dpi: nothing cut."""
    sidecar(jsonl, participant="P01", session="S1")
    cal = script.load_calibration(write_calibration(tmp_path / "c.json", rec, 12.0, -7.0,
                                                    hand="left"))
    mapping = rvl.load_mapping(write_mapping(tmp_path / "m.json", rec, 20.0, 10.0))
    mprov, how = script.mapping_provider(mapping, rec)
    assert how.startswith("applied")
    v = ReplayViewer(rec, mode="interpolate", figsize=(15.5, 8.04), dpi=100,
                     tmc_provider=script.tmc_provider(cal, rec), mapping_provider=mprov,
                     screenshot_dir=tmp_path / "s")
    try:
        t = rec.timeline.t
        v.seek_time((t[2] + t[3]) / 2)
        assert v.displayed.source == "interpolated"
        v.screenshot(tmp_path / "worst.png")
        lines = v.angles_text().splitlines()
        assert any(ln.startswith("mapped (SYNTHETIC") for ln in lines)
        assert lines[-1].startswith("provenance: ")
        assert not v.ang_panel.truncated, v.ang_panel.rows
    finally:
        v.close()
