"""The pose check: the window's caption, the verdict rule, the audit, the retry.

The failure all of this is about, measured on `recordings/sync_coached_20260917`
(36 takes, 2026-09-17): nine takes hold a pose that is not the one in their
filename, and both sensors agree on the wrong one. The camera window's caption
was `f"{phase}  {extra}{secs}"` and `extra` is empty during SETTLE and REC, so
the window the operator was watching never once named the pose.

The numbers in `test_the_real_session_*` are the medians of that session,
copied out of the per-DOF report. They are the regression test with teeth: any
change to the thresholds that stops reproducing those verdicts is a change
that has to be argued for.
"""
import importlib.util
import json
import math
from pathlib import Path

import pytest

from leap_hand.pose_check import (
    AMBIGUOUS,
    CURLED,
    DEFAULT_PARAMS,
    EXTENDED,
    MISMATCH,
    OK,
    UNCHECKED,
    WARN,
    PoseCheckParams,
    SensorReading,
    check_pose,
    finger_state,
    name_shape,
    read_take,
    short_summary,
)
from leap_hand.protocol import (
    CameraView,
    parse_status,
    pose_label,
    status_text,
    stream_health,
    view_caption,
)


# --- A. the window names the pose, in every phase ---------------------------
def test_acquire_says_open_first_and_what_is_coming():
    line = view_caption("ACQUIRE", "fist", take=2, takes=3)
    assert line == "OPEN PALM first   next: FIST (take 2/3)"
    assert len(line) < 60


def test_acquire_keeps_the_need_extras():
    line = view_caption("ACQUIRE", "fist", "need: TOO LOW", take=1, takes=3)
    assert line.startswith("OPEN PALM first   next: FIST (take 1/3)")
    assert line.endswith("need: TOO LOW")


def test_settle_and_rec_name_the_pose():
    assert view_caption("SETTLE", "fist", seconds_left=1.0) == "NOW: FIST   1s"
    assert view_caption("REC", "fist", seconds_left=4.0) == "HOLD: FIST   REC 4s"


def test_the_open_palm_is_told_to_stay():
    """Every take is acquired OPEN, so "NOW: OPEN PALM" would read as "move"."""
    assert view_caption("SETTLE", "open_palm") == "NOW: OPEN PALM (stay)"
    assert view_caption("REC", "open_palm", seconds_left=3.0) == (
        "HOLD: OPEN PALM (stay)   REC 3s")


def test_a_timed_schedule_does_not_say_stay():
    """scripts/leap/gate.py can hold open_palm AFTER a fist: that means open."""
    assert view_caption("REC", "open_palm", seconds_left=3.0, stay=False) == (
        "HOLD: OPEN PALM   REC 3s")
    assert pose_label("open_palm", stay=False) == "OPEN PALM"


def test_the_wrong_pose_line():
    assert view_caption("WRONG POSE", "fist", "saw OPEN HAND, want FIST") == (
        "WRONG POSE: saw OPEN HAND, want FIST")


def test_seconds_are_never_negative_and_a_missing_pose_is_not_a_crash():
    assert view_caption("REC", "", seconds_left=-2.0) == "HOLD   REC 0s"
    assert view_caption("", "") == ""


def test_every_phase_of_a_real_take_fits_the_window():
    """768 px at font scale 0.75 is about 60 characters."""
    for pose in ("open_palm", "fist", "index_point", "thumbs_up", "peace",
                 "pinch"):
        for line in (view_caption("ACQUIRE", pose, take=3, takes=3),
                     view_caption("SETTLE", pose, seconds_left=1.0),
                     view_caption("REC", pose, seconds_left=5.0)):
            assert pose.split("_")[0].upper() in line
            assert len(line) <= 60, line


# --- the status file the window is driven through ---------------------------
def test_status_text_round_trips():
    text = status_text("REC  FIST", band=(18, 28), snap="C:\\tmp\\a.jpg")
    assert parse_status(text) == ("REC  FIST", (18.0, 28.0), "C:\\tmp\\a.jpg")


def test_status_without_the_optional_lines():
    assert parse_status(status_text("NOW: FIST")) == ("NOW: FIST", None, None)
    assert status_text("NOW: FIST") == "NOW: FIST"


def test_a_torn_or_odd_status_file_is_ignored_not_fatal():
    caption, band, snap = parse_status("HOLD\nband=oops\nsnap=\nother=1")
    assert (caption, band, snap) == ("HOLD", None, None)
    assert parse_status("") == ("", None, None)


def test_snapshot_is_a_noop_without_a_window():
    v = CameraView(hand="left", enabled=False).start()
    assert v.snapshot("anywhere.jpg") is None       # must not raise or write
    assert not Path("anywhere.jpg").exists()


def test_snapshot_asks_for_one_still_without_losing_the_caption(tmp_path):
    v = CameraView(hand="left", band=(18, 28), enabled=True)
    v._status = tmp_path / "status.txt"             # as start() would set it
    v.caption("REC  HOLD: FIST   REC 4s", band=(18, 28))
    v.snapshot(tmp_path / "stills" / "fist_take1.jpg")
    lines = v._status.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "REC  HOLD: FIST   REC 4s"
    assert lines[-1] == f"snap={tmp_path / 'stills' / 'fist_take1.jpg'}"
    caption, _band, snap = parse_status(v._status.read_text(encoding="utf-8"))
    assert caption == "REC  HOLD: FIST   REC 4s"
    assert Path(snap).name == "fist_take1.jpg"
    v._proc = None
    v.close()


# --- B. the verdict rule ----------------------------------------------------
# The reference session's medians, per sensor, in thumb..pinky order.
GLOVE_OPEN = (1.43, 1.97, 2.07, 1.97, 1.71)
GLOVE_FIST = (0.95, 0.66, 0.62, 0.66, 0.76)
CAM_OPEN = (1.30, 1.75, 1.85, 1.71, 1.47)
CAM_FIST = (1.20, 0.97, 0.88, 0.90, 0.93)


def _glove(curls, frames=24, gap=0.99):
    return SensorReading(frames=frames, curls=tuple(curls), gap=gap)


def _camera(curls, frames=450, gap=0.97, view=30.0):
    return SensorReading(frames=frames, curls=tuple(curls), gap=gap,
                         view_angle_deg=view)


def test_a_finger_is_wrong_only_when_BOTH_sensors_contradict():
    check = check_pose("fist", _glove(GLOVE_OPEN), _camera(CAM_OPEN))
    assert check.verdict == MISMATCH
    assert check.seen == "OPEN HAND"
    assert "expected FIST" in check.description
    assert short_summary(check) == "saw OPEN HAND, want FIST"


def test_one_sensor_disagreeing_is_a_warning_and_never_a_failure():
    """Right-hand peace, measured: glove ring 0.65, camera ring 1.66."""
    check = check_pose("peace",
                       _glove((1.11, 1.97, 2.07, 0.68, 0.92)),
                       _camera((1.15, 1.81, 1.80, 1.66, 0.85)))
    assert check.verdict == WARN
    assert "camera alone disagrees: ring extended" in check.description
    assert "the take stands" in check.description
    ring = next(f for f in check.fingers if f.finger == "ring")
    assert (ring.glove, ring.camera) == (CURLED, EXTENDED)
    assert ring.verdict == WARN


def test_a_reading_in_the_ambiguous_band_cannot_fail_anything():
    """Halfway between open and fist is not evidence of either."""
    mid = tuple((a + b) / 2 for a, b in zip(GLOVE_OPEN, GLOVE_FIST))
    check = check_pose("fist", _glove(mid), _camera(CAM_FIST))
    assert check.verdict == OK
    assert all(f.glove == AMBIGUOUS for f in check.fingers)


def test_an_edge_on_camera_casts_no_vote_so_nothing_can_fail():
    check = check_pose("fist", _glove(GLOVE_OPEN),
                       _camera(CAM_OPEN, view=78.0))
    assert check.verdict == WARN
    assert check.camera_votes is False
    assert "edge-on" in check.camera_silent_because
    assert all(f.camera == "no data" for f in check.fingers)


def test_a_camera_that_barely_saw_the_take_casts_no_vote():
    check = check_pose("fist", _glove(GLOVE_OPEN), _camera(CAM_OPEN, frames=4))
    assert check.verdict == WARN
    assert check.camera_votes is False
    assert "4 frame(s)" in check.camera_silent_because


def test_an_unknown_pose_is_unchecked():
    check = check_pose("rock_on", _glove(GLOVE_FIST), _camera(CAM_FIST))
    assert check.verdict == UNCHECKED
    assert check.fingers == ()
    assert "no expected hand shape" in check.description


def test_the_glove_can_never_fail_a_pinch():
    """The measured dead zone: a real pinch reads as an exact open palm.

    All six pinch takes of the reference session are pinches by the camera
    (thumb-index gap 0.10-0.35 palm lengths) while the glove reports its open
    palm to the second decimal, because stretch sensors cannot see thumb
    opposition at all. Failing those takes would delete the pinch data.
    """
    check = check_pose("pinch", _glove(GLOVE_OPEN, gap=0.99),
                       _camera((1.30, 1.30, 1.83, 1.73, 1.51), gap=0.10))
    assert check.verdict == OK
    assert all(f.expected is None and f.verdict == UNCHECKED
               for f in check.fingers)
    assert check.as_dict()["pinch_camera_gap"] == 0.1


def test_a_pinch_the_camera_doubts_is_a_warning_never_a_mismatch():
    """The camera may not select the pinch takes it will later be scored on."""
    check = check_pose("pinch", _glove(GLOVE_OPEN),
                       _camera(CAM_OPEN, gap=0.97))
    assert check.verdict == WARN
    assert check.as_dict()["pinch_camera_gap"] == 0.97
    assert "never rejected" in check.description


def test_a_take_with_no_sensors_at_all_is_unchecked():
    check = check_pose("fist", SensorReading(), SensorReading())
    assert check.verdict == UNCHECKED
    assert "neither sensor" in check.description


def test_finger_state_bands():
    assert finger_state(2.0, 1.32, 0.30) == EXTENDED
    assert finger_state(0.66, 1.32, 0.30) == CURLED
    assert finger_state(1.32, 1.32, 0.30) == AMBIGUOUS
    assert finger_state(None, 1.32, 0.30) == "no data"


def test_name_shape_reads_the_agreed_fingers():
    assert name_shape((CURLED,) * 5) == "FIST"
    assert name_shape((EXTENDED,) * 5) == "OPEN HAND"
    assert name_shape((EXTENDED, CURLED, CURLED, CURLED, CURLED)) == "THUMBS UP"
    assert name_shape((None,) * 5) == "ANOTHER SHAPE"


# --- the real session, take by take -----------------------------------------
# (labelled pose, glove medians, camera medians, expected verdict)
REAL_TAKES = [
    ("fist_left_1", "fist", (1.426, 1.974, 2.073, 1.970, 1.707),
     (1.309, 1.764, 1.870, 1.740, 1.485), MISMATCH),
    ("fist_left_2", "fist", (0.952, 0.662, 0.624, 0.661, 0.762),
     (1.188, 0.979, 0.878, 0.891, 0.920), OK),
    ("fist_right_1", "fist", (1.426, 1.975, 2.073, 1.970, 1.707),
     (1.336, 1.767, 1.835, 1.695, 1.442), MISMATCH),
    ("index_point_left_1", "index_point", (0.952, 0.662, 0.624, 0.655, 0.749),
     (1.213, 1.004, 0.908, 0.930, 0.979), MISMATCH),
    ("index_point_right_1", "index_point", (1.227, 1.974, 0.784, 0.680, 0.990),
     (1.201, 1.787, 0.923, 0.923, 0.846), OK),
    ("thumbs_up_left_1", "thumbs_up", (0.952, 0.662, 0.624, 0.655, 0.782),
     (1.234, 1.012, 0.903, 0.925, 0.987), MISMATCH),
    ("thumbs_up_left_3", "thumbs_up", (1.426, 0.662, 0.624, 0.714, 1.091),
     (1.108, 1.721, 1.028, 1.069, 0.923), WARN),
    ("thumbs_up_right_1", "thumbs_up", (1.322, 0.662, 0.812, 0.864, 1.107),
     (1.396, 0.906, 0.884, 0.926, 0.900), OK),
    ("peace_left_1", "peace", (1.398, 0.662, 0.624, 0.680, 0.729),
     (1.427, 0.880, 0.754, 0.880, 0.908), MISMATCH),
    ("peace_left_2", "peace", (0.952, 1.974, 2.073, 0.680, 0.766),
     (1.089, 1.786, 1.856, 1.259, 0.831), OK),
    ("peace_right_1", "peace", (1.105, 1.974, 2.072, 0.680, 0.917),
     (1.155, 1.807, 1.803, 1.662, 0.849), WARN),
    ("open_palm_right_1", "open_palm", (1.426, 1.974, 2.073, 1.971, 1.709),
     (1.309, 1.751, 1.797, 1.661, 1.388), OK),
]


@pytest.mark.parametrize("name,pose,glove,cam,want", REAL_TAKES)
def test_the_real_session_reproduces_the_hand_that_was_in_front_of_it(
        name, pose, glove, cam, want):
    assert check_pose(pose, _glove(glove), _camera(cam)).verdict == want, name


def test_the_three_wrong_fists_are_named_as_open_hands():
    check = check_pose("fist", _glove((1.426, 1.974, 2.073, 1.970, 1.707)),
                       _camera((1.309, 1.764, 1.870, 1.740, 1.485)))
    assert check.description == (
        "both sensors saw an OPEN HAND, expected FIST "
        "(thumb, index, middle, ring, pinky extended)")


def test_the_thumbs_up_takes_that_were_really_fists_are_caught_on_the_thumb():
    """The tightest call in the whole check, and the reason it is honest.

    The camera's thumb curl separates a fist from an open hand by 0.045 of a
    palm length (1.238 against 1.283). It is only usable because a mismatch
    needs the GLOVE to contradict the pose too, and the glove's thumb
    separates by 0.37.
    """
    check = check_pose("thumbs_up", _glove((0.952, 0.662, 0.624, 0.655, 0.782)),
                       _camera((1.234, 1.012, 0.903, 0.925, 0.987)))
    assert check.verdict == MISMATCH
    thumb = check.fingers[0]
    assert (thumb.glove, thumb.camera) == (CURLED, CURLED)
    assert check.seen == "FIST"


def test_widening_a_margin_can_only_lose_a_mismatch_never_invent_one():
    """The direction the margins are allowed to be wrong in.

    A wider ambiguous band silences a sensor; it cannot make it say the
    opposite thing. So no take that stands under the shipped thresholds may
    start failing under wider ones — which is why erring wide is the safe
    error when a false failure costs a re-recorded session.
    """
    wide = PoseCheckParams(camera_margin=(0.40, 0.40, 0.40, 0.40, 0.40),
                           glove_margin=(0.40, 0.40, 0.40, 0.40, 0.40))
    for name, pose, glove, cam, want in REAL_TAKES:
        if want == MISMATCH:
            continue
        assert check_pose(pose, _glove(glove), _camera(cam),
                          wide).verdict != MISMATCH, name


# --- stream health ----------------------------------------------------------
def test_stream_health_measures_the_rate_and_the_worst_hole():
    times = [0.0, 0.02, 0.04, 1.5, 1.52]
    health = stream_health(times)
    assert health["frames"] == 5
    assert health["rate_hz"] == pytest.approx(4 / 1.52, abs=0.01)
    assert health["max_gap_ms"] == pytest.approx(1460.0)
    assert health["gaps_over"] == 1


def test_stream_health_counts_a_hole_at_the_head_of_the_take():
    health = stream_health([1.0, 1.02, 1.04], t0=0.0, t1=1.05)
    assert health["max_gap_ms"] == pytest.approx(1000.0)
    assert health["gaps_over"] == 1


def test_stream_health_of_a_silent_sensor():
    health = stream_health([], t0=0.0, t1=5.0)
    assert health["frames"] == 0 and health["rate_hz"] is None
    assert health["max_gap_ms"] == pytest.approx(5000.0)


# --- C. the audit, on files ------------------------------------------------
def _load_repo_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"repo_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The mock hands are a cartoon on their own scale: the glove mock's fully open
# index reads 1.43 palm lengths where a real glove reads 1.97, and its fist
# reads 1.08 where a real one reads 0.66. Measured against the shipped
# thresholds every mock finger is ambiguous, which is the honest answer and a
# useless test. So the file-level tests below run the machinery with
# parameters centred on the mock's own open palm and fist — the shipped
# numbers are tested against the real session's medians, above.
MOCK_PARAMS = PoseCheckParams(
    glove_mid=(1.005, 1.254, 1.376, 1.278, 1.145),
    glove_margin=(0.05, 0.10, 0.10, 0.10, 0.05),
    camera_mid=(1.492, 1.668, 1.776, 1.646, 1.424),
    camera_margin=(0.10, 0.20, 0.20, 0.20, 0.10),
    min_camera_frames=30, min_glove_frames=5)


def _frozen_glove_frames(hand: str, curl: float, n: int):
    """`n` identical glove frames of a hand held at `curl` (0 open, 1 closed).

    The mock's curl is a sine of its own clock; pinning `t` and stopping `dt`
    holds one shape, which is what a take of one pose is supposed to be.
    """
    from xr_hand.mock import MockHandGenerator
    from xr_hand.parser import parse_hand_message

    gen = MockHandGenerator(hand=hand)
    gen.t = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * curl))) / 1.5
    gen.dt = 0.0
    return [parse_hand_message(gen.next_frame(), hand_side_hint=hand)
            for _ in range(n)]


def _write_take(folder: Path, pose: str, hand: str, take: int,
                glove_curl: float, leap_pose: str, n: int = 60) -> str:
    """One synthetic take: the same name under glove/ and leap/."""
    from leap_hand.mock import MockLeapStream
    from leap_hand.recorder import LeapRecorder
    from xr_hand.recorder import FrameRecorder

    name = f"{pose}_{hand}_take{take}_20260101_000000.jsonl"
    grec = FrameRecorder(pose=pose, take=take)
    grec.start(folder / "glove" / name)
    for frame in _frozen_glove_frames(hand, glove_curl, n):
        grec.record(frame)
    grec.stop()

    lrec = LeapRecorder(pose=pose, take=take)
    lrec.start(folder / "leap" / name)
    stream = MockLeapStream(pose=leap_pose, noise_mm=0.0, dropout_every=0,
                            reacquire_every=0)
    for side, lh in stream.generate(n):
        if side == hand:
            lrec.record(lh)
    lrec.stop()
    return name


def test_the_audit_reads_a_folder_and_names_the_takes_that_are_wrong(tmp_path):
    check = _load_repo_script("check_take_labels")
    folder = tmp_path / "sync"
    _write_take(folder, "fist", "left", 1, glove_curl=1.0, leap_pose="fist")
    _write_take(folder, "fist", "left", 2, glove_curl=0.0,
                leap_pose="open_palm")            # an open hand labelled fist
    _write_take(folder, "open_palm", "left", 1, glove_curl=0.0,
                leap_pose="open_palm")

    rows = check.audit(folder, "leap", MOCK_PARAMS)
    verdicts = {r["stem"].split("_2026")[0]: r["check"].verdict for r in rows}
    assert verdicts["fist_left_take1"] == OK
    assert verdicts["fist_left_take2"] == MISMATCH
    assert verdicts["open_palm_left_take1"] == OK
    wrong = next(r for r in rows if r["check"].verdict == MISMATCH)
    assert wrong["check"].seen == "OPEN HAND"
    assert wrong["pose"] == "fist" and wrong["hand"] == "left"


def test_the_audit_prints_the_command_that_re_records_the_whole_pose(tmp_path):
    check = _load_repo_script("check_take_labels")
    folder = tmp_path / "sync"
    _write_take(folder, "fist", "left", 1, 0.0, "open_palm")
    _write_take(folder, "fist", "left", 2, 1.0, "fist")
    _write_take(folder, "peace", "right", 1, 0.0, "open_palm")

    rows = check.audit(folder, "leap", MOCK_PARAMS)
    commands = check.redo_commands(rows)
    assert len(commands) == 2
    left = next(c for c in commands if c.startswith("  left"))
    assert "--hand left --poses fist" in left
    # take numbering is per pose, so BOTH takes of fist are re-recorded
    assert "--takes 2" in left
    text = check.report(rows, MOCK_PARAMS, folder)
    assert "Take numbering is per pose" in text
    assert "mismatch: 2" in text


def test_the_audit_ignores_attempts_the_recorder_already_rejected(tmp_path):
    check = _load_repo_script("check_take_labels")
    folder = tmp_path / "sync"
    _write_take(folder, "fist", "left", 1, 1.0, "fist")
    _write_take(folder / "rejected", "fist", "left", 1, 0.0, "open_palm")
    rows = check.audit(folder, "leap", MOCK_PARAMS)
    assert len(rows) == 1, "rejected/ holds attempts that were already counted"
    assert rows[0]["check"].verdict == OK


def test_the_audit_reports_glove_stream_health(tmp_path):
    check = _load_repo_script("check_take_labels")
    folder = tmp_path / "sync"
    _write_take(folder, "fist", "left", 1, 1.0, "fist")
    rows = check.audit(folder, "leap", MOCK_PARAMS)
    assert rows[0]["glove_health"]["frames"] == 60
    assert rows[0]["camera_health"]["rate_hz"] == pytest.approx(90.0, rel=0.05)


def test_read_take_keeps_only_the_hand_it_was_asked_for(tmp_path):
    from leap_hand.mock import MockLeapStream
    from leap_hand.recorder import LeapRecorder

    path = tmp_path / "both.jsonl"
    rec = LeapRecorder(pose="fist", take=1)
    rec.start(path)
    stream = MockLeapStream(pose="fist", noise_mm=0.0, dropout_every=0,
                            reacquire_every=0)
    for _side, lh in stream.generate(10):
        rec.record(lh)
    rec.stop()
    assert read_take(path).frames == 20
    assert read_take(path, "left").frames == 10


def test_read_take_of_a_missing_file_is_empty_not_an_error():
    assert read_take(None).frames == 0
    assert read_take("no-such-file.jsonl").present is False


# --- the retry, with mock sensors -------------------------------------------
def _posed_session(sync, tmp_path, monkeypatch, leap_pose: str,
                   glove_curl: float, **kwargs):
    """A coached session whose two mock sensors HOLD one shape.

    The pose check is off for mocks in `main()`, because neither mock's hand
    follows the pose being called — the leap mock cycles four cartoon poses on
    its own timer and the glove mock's curl is a sine. Pinned like this they
    do hold a shape, so the retry path can be driven for real.
    """
    from leap_hand.mock import MockLeapStream

    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    monkeypatch.setattr(sync, "WRONG_POSE_SECONDS", 0.05)
    leap = MockLeapStream(pose=leap_pose, noise_mm=0.0, dropout_every=0,
                          reacquire_every=0)
    leap.start()
    glove = sync.MockGloveSource()
    for gen in glove.gens.values():
        gen.t = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * glove_curl))) / 1.5
        gen.dt = 0.0
    session = sync.CoachedLeapSession(
        leap, glove, hz=None, out_dir=tmp_path / "sync", hand="left",
        settle=0.0, acquire_timeout=10.0, pose_check=True,
        pose_check_params=MOCK_PARAMS, **kwargs)
    return session


def test_a_mock_holding_the_wrong_pose_is_rejected_and_the_take_retried(
        tmp_path, monkeypatch):
    """Both mocks hold a closed hand while an OPEN PALM is asked for."""
    sync = _load_repo_script("record_simultaneous")
    session = _posed_session(sync, tmp_path, monkeypatch, leap_pose="fist",
                             glove_curl=1.0, retries=1)
    session.run_take("open_palm", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()

    result = session.takes[0]
    assert result.attempts == 2, "the first attempt must have been retried"
    assert result.complete is False
    assert len(session.pose_rejects) == 2
    assert session.pose_rejects[0] == ("open_palm", "FIST")

    out = tmp_path / "sync"
    assert list((out / "leap").glob("*.jsonl")) == [], "nothing was accepted"
    assert list((out / "glove").glob("*.jsonl")) == []
    # ...and nothing was deleted either: the sensors that vetoed the take are
    # the ones being evaluated, so the exclusion has to be auditable.
    rejected = sorted((out / "rejected" / "leap").glob("*.jsonl"))
    assert len(rejected) == 2
    assert rejected[0].name.endswith("_attempt1.jsonl")
    assert rejected[1].name.endswith("_attempt2.jsonl")
    assert len(list((out / "rejected" / "glove").glob("*.jsonl"))) == 2

    meta = json.loads(rejected[0].with_name(rejected[0].stem + ".meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["accepted"] is False
    assert meta["pose"] == "open_palm" and meta["hand"] == "left"
    assert meta["pose_check"]["verdict"] == MISMATCH
    assert meta["pose_check"]["seen"] == "FIST"
    assert "expected OPEN PALM" in meta["why"]


def test_the_same_mocks_holding_the_right_pose_are_accepted(tmp_path,
                                                            monkeypatch):
    """The check has to let a correct take through, or it proves nothing."""
    sync = _load_repo_script("record_simultaneous")
    session = _posed_session(sync, tmp_path, monkeypatch, leap_pose="fist",
                             glove_curl=1.0, retries=0)
    session.run_take("fist", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()

    result = session.takes[0]
    assert result.complete is True and result.attempts == 1
    assert session.pose_rejects == []
    out = tmp_path / "sync"
    assert len(list((out / "leap").glob("*.jsonl"))) == 1
    assert not (out / "rejected").exists()

    meta = json.loads(next((out / "leap").glob("*.meta.json"))
                      .read_text(encoding="utf-8"))
    assert meta["accepted"] is True
    assert meta["pose_check"]["verdict"] in (OK, WARN)
    assert meta["glove_rate_hz"] > 30, "the glove is no longer throttled"
    assert meta["glove_max_gap_ms"] is not None
    assert meta["still"] is None, "there is no window behind a mock"


def test_the_check_can_be_turned_off(tmp_path, monkeypatch):
    sync = _load_repo_script("record_simultaneous")
    session = _posed_session(sync, tmp_path, monkeypatch, leap_pose="fist",
                             glove_curl=1.0, retries=0)
    session.pose_check = False
    session.run_take("open_palm", 1, 1, 1, 1, duration=0.6, prep=0)
    session.finish()
    assert session.takes[0].complete is True
    meta = json.loads(next((tmp_path / "sync" / "leap").glob("*.meta.json"))
                      .read_text(encoding="utf-8"))
    assert meta["pose_check"]["verdict"] == UNCHECKED
