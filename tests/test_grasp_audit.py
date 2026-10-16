"""`leap_hand.grasp_audit`: what a kept grasp take holds, from its own lines.

The point of the module is that nothing is taken from a count of lines, so
the takes here are built to make a count of lines wrong: a second hand in
every frame, a second hand in a few frames, a hand lost and found again.
"""
import numpy as np
import pytest

from leap_hand import grasp_audit as ga
from leap_hand.grasp_audit import BY_LABEL, FOLLOWED, take_facts

HZ = 90.0
T0 = 1_790_000_000.0


def line(k: int, hand_id: int = 7, side: str = "left", x_m: float = 0.0,
         t: float = None, item: str = "p1_hook") -> dict:
    """One tracked hand at frame k: 26 joints 20 cm up, `x_m` to the side."""
    t = T0 + k / HZ if t is None else t
    palm = [x_m, 0.20, 0.0]
    abs26 = [[x_m + 0.001 * j, 0.20, 0.002 * j] for j in range(26)]
    return {"wall_time": t + 0.004, "capture_time": t,
            "timestamp_us": int(t * 1e6), "frame_id": 1000 + k,
            "hand_side": side, "hand_id": hand_id, "status": 1,
            "framerate": HZ, "palm_abs": palm, "abs26": abs26, "item": item}


def take(n: int = 450, **kw) -> list:
    return [line(k, **kw) for k in range(n)]


def test_one_hand_duration_is_the_span_of_its_timestamps():
    lines = take(450)
    f = take_facts(lines, {"take_start": T0, "take_end": T0 + 5.0}, "left")
    assert f.jsonl_row_count == f.unique_frame_count == f.hand_frame_count == 450
    assert f.recording_duration_s == pytest.approx(449 / HZ)
    assert f.camera_clock_duration_s == pytest.approx(449 / HZ, abs=1e-5)
    assert f.recorder_window_s == pytest.approx(5.0)
    assert f.frame_rate_hz == HZ
    assert not f.other_hand_detected and f.other_hand_frame_fraction == 0.0
    assert f.hand_selection == BY_LABEL and f.hand_ids == [7]
    assert f.identity_flag == "" and f.timing_notes == []


def test_a_second_hand_doubles_the_lines_and_changes_no_duration():
    """900 lines, 450 frames, the same five seconds."""
    lines = []
    for k in range(450):
        lines.append(line(k, hand_id=7, side="left", x_m=-0.15))
        lines.append(line(k, hand_id=8, side="right", x_m=0.18))
    one = take_facts(take(450), {}, "left")
    two = take_facts(lines, {"operator_hand_ids": [7], "other_hand_ids": [8]},
                     "left")
    assert two.jsonl_row_count == 900
    assert two.unique_frame_count == two.hand_frame_count == 450
    assert two.recording_duration_s == pytest.approx(one.recording_duration_s)
    assert two.other_hand_detected and two.other_hand_frames == 450
    assert two.other_hand_frame_fraction == 1.0
    assert two.other_hand_labels == {"right": 450}
    assert two.other_hand_median_distance_cm == pytest.approx(33.0)
    assert two.hand_selection == FOLLOWED
    # another hand 33 cm away is not a doubt about whose hand this is
    assert two.identity_flag == "" and two.timing_notes == []
    # (the one note is about the meta, which here names no summary frame)
    assert two.notes == ["the take's meta names no summary frame"]


def test_the_followed_ids_decide_not_the_label():
    """The tracker called the operator's hand "right": the ids still pick it."""
    lines = []
    for k in range(90):
        lines.append(line(k, hand_id=7, side="right", x_m=-0.15))
        lines.append(line(k, hand_id=8, side="left", x_m=0.18))
    mine, how = ga.operator_lines(lines, {"operator_hand_ids": [7]}, "left")
    assert how == FOLLOWED and {d["hand_id"] for _i, d in mine} == {7}
    assert [i for i, _d in mine] == list(range(0, 180, 2))
    # with no ids, the label rule: the operator's own label
    mine, how = ga.operator_lines(lines, {}, "left")
    assert how == BY_LABEL and {d["hand_id"] for _i, d in mine} == {8}


def test_a_second_fit_on_the_same_hand_is_flagged_as_one_hand_twice():
    lines = take(450)
    for k in range(100, 112):               # 12 frames, 2 mm away, as "right"
        lines.append(line(k, hand_id=9, side="right", x_m=0.002))
    lines.sort(key=lambda d: (d["frame_id"], d["hand_id"]))
    f = take_facts(lines, {}, "left")
    assert f.jsonl_row_count == 462 and f.unique_frame_count == 450
    assert f.hand_frame_count == 450
    assert f.other_hand_frames == 12
    assert f.other_hand_frame_fraction == pytest.approx(12 / 450)
    assert "one hand fitted twice" in f.identity_flag
    assert f.abnormal


def test_a_hand_lost_and_found_again_is_a_hole_and_an_id_change():
    lines = [line(k, hand_id=7) for k in range(60)]
    lines += [line(k, hand_id=8) for k in range(100, 450)]   # 0.44 s gone
    f = take_facts(lines, {"take_start": T0, "take_end": T0 + 5.0}, "left")
    assert f.hand_frame_count == 410 and f.hand_ids == [7, 8]
    assert f.tracker_id_changes == 1
    assert f.largest_gap_s == pytest.approx(41 / HZ, abs=1e-5)
    assert any("a hole of 0.46 s" in n for n in f.timing_notes)
    assert "changed the hand's id 1 time(s)" in f.identity_flag
    # the duration is still first to last, the hole inside it
    assert f.recording_duration_s == pytest.approx(449 / HZ)


def test_a_label_flip_on_the_followed_hand_is_flagged():
    lines = [line(k, side="left") for k in range(200)]
    lines += [line(k, side="right") for k in range(200, 450)]
    f = take_facts(lines, {"operator_hand_ids": [7]}, "left")
    assert f.tracker_labels == {"left": 200, "right": 250}
    assert "labelled the hand left on 200 frame(s), right on 250" in f.identity_flag


def test_a_take_tracked_late_says_its_span_is_short():
    lines = [line(k) for k in range(20, 450)]        # the first 0.22 s missing
    f = take_facts(lines, {"take_start": T0, "take_end": T0 + 5.01}, "left")
    assert f.recording_duration_s == pytest.approx(429 / HZ)
    assert len(f.timing_notes) == 1
    assert "tracked data spans 4.77 s of the 5.01 s" in f.timing_notes[0]
    # one frame period short is the ordinary case, and gets no note
    f = take_facts(take(450), {"take_start": T0, "take_end": T0 + 5.01},
                   "left")
    assert f.timing_notes == []


def test_the_summary_frame_and_the_static_interval_are_from_the_first_frame():
    lines = take(450)
    meta = {"static_interval": [lines[90]["wall_time"],
                                lines[90]["wall_time"] + 2.0],
            "medoid_wall_time": lines[201]["wall_time"], "medoid_line": 201}
    f = take_facts(lines, meta, "left")
    assert f.static_interval_start_s == pytest.approx(1.0)
    assert f.static_interval_end_s == pytest.approx(3.0)
    assert f.static_interval_duration_s == pytest.approx(2.0)
    assert f.representative_frame_index == 201
    assert f.representative_frame_id == 1201
    assert f.representative_frame_time_s == pytest.approx(201 / HZ)
    assert f.representative_label == "left"
    # a meta written before the line number was stored: found by its time
    del meta["medoid_line"]
    assert take_facts(lines, meta, "left").representative_frame_index == 201
    # a line number that does not carry the meta's time is not believed
    meta["medoid_line"] = 5
    assert take_facts(lines, meta, "left").representative_frame_index == 201


def test_a_summary_frame_that_is_not_in_the_file_is_said():
    f = take_facts(take(90), {"medoid_wall_time": T0 - 50.0}, "left")
    assert f.representative_frame_index is None
    assert "summary frame was not found" in f.notes[-1]


def test_a_summary_frame_under_the_other_label_is_flagged():
    lines = [line(k, side="left") for k in range(300)]
    lines += [line(k, side="right") for k in range(300, 450)]
    meta = {"operator_hand_ids": [7], "medoid_line": 400,
            "medoid_wall_time": lines[400]["wall_time"]}
    f = take_facts(lines, meta, "left")
    assert "representative frame carries the tracker's right label" \
        in f.identity_flag


def test_no_line_of_the_hand_gives_no_numbers():
    f = take_facts([], {}, "left")
    assert f.jsonl_row_count == 0 and f.recording_duration_s is None
    assert f.identity_flag == "no line of the labelled hand in the file"
    f = take_facts(take(30, side="right"), {"operator_hand_ids": [99]}, "left")
    assert f.hand_frame_count == 0 and f.recording_duration_s is None


def test_lines_without_a_frame_id_each_count_as_a_frame():
    lines = take(30)
    for d in lines:
        del d["frame_id"]
    f = take_facts(lines, {}, "left")
    assert f.unique_frame_count == 30


def test_spread_is_min_median_max_of_what_is_there():
    assert ga.spread([3.0, None, 1.0, 2.0]) == (1.0, 2.0, 3.0)
    assert ga.spread([None]) is None
    assert ga.spread([4.84, 4.97, 5.0, 4.9]) == pytest.approx(
        (4.84, float(np.median([4.84, 4.97, 5.0, 4.9])), 5.0))


# --- how still the static interval was, how big the hand, which side the camera saw ---
RNG = np.random.default_rng(7)
SHAPE = RNG.uniform(-0.04, 0.04, size=(26, 3)) + np.array([0.0, 0.20, 0.0])
I_MC, I_PX = 6, 7          # INDEX_METACARPAL, INDEX_PROXIMAL in the joint order


def hand_line(k: int, shift_m: float = 0.0, lift_m: float = 0.0,
              bone_mm: float = 60.0) -> dict:
    """A hand with a shape (26 scattered points 20 cm up): moved `shift_m`
    sideways as a whole, its last five joints lifted `lift_m`, and its index
    metacarpal `bone_mm` long."""
    d = line(k)
    pos = SHAPE.copy()
    pos[I_PX] = pos[I_MC] + np.array([0.0, 0.0, -bone_mm / 1000.0])
    pos[:, 0] += shift_m
    pos[21:, 1] += lift_m
    d["abs26"] = pos.tolist()
    return d


def held(lines, start=90, medoid=180) -> dict:
    return {"static_interval": [lines[start]["wall_time"],
                                lines[start]["wall_time"] + 2.0],
            "medoid_wall_time": lines[medoid]["wall_time"],
            "medoid_line": medoid}


def test_a_hand_that_does_not_move_is_still_in_both_senses():
    lines = [hand_line(k) for k in range(450)]
    f = take_facts(lines, held(lines), "left")
    assert f.static_interval_mean_speed_mm_s == pytest.approx(0.0, abs=1e-9)
    assert f.static_interval_shape_rms_mm == pytest.approx(0.0, abs=1e-6)
    assert f.static_interval_shape_rms_max_mm == pytest.approx(0.0, abs=1e-6)
    assert f.pose_notes == [] and not f.abnormal


def test_a_hand_that_drifts_is_moving_but_keeps_its_shape():
    """30 mm/s sideways as a whole: the stillest window moved, the shape in
    it did not, and the note says both."""
    lines = [hand_line(k, shift_m=0.030 * k / HZ) for k in range(450)]
    f = take_facts(lines, held(lines), "left")
    assert f.static_interval_mean_speed_mm_s == pytest.approx(30.0, abs=0.1)
    assert f.static_interval_shape_rms_max_mm == pytest.approx(0.0, abs=1e-6)
    assert f.static_interval_mean_speed_mm_s > ga.STATIC_MOVING_MM_S
    (note,) = f.pose_notes
    assert note.startswith("the hand was moving during its stillest 2 s "
                           "(mean joint speed 30 mm/s)")
    assert "its shape stayed within 0.0 mm RMS" in note
    assert f.abnormal and note in f.notes


def test_a_hand_that_changes_shape_is_not_said_to_have_kept_it():
    """Five joints lifted 60 mm over the window: little speed on average,
    a different shape. And with a drift on top the note that it moved does
    not go on to say its shape held."""
    def lifted(k):
        return 0.060 * min(1.0, max(0.0, (k - 90) / 180.0))
    lines = [hand_line(k, lift_m=lifted(k)) for k in range(450)]
    f = take_facts(lines, held(lines), "left")
    assert f.static_interval_mean_speed_mm_s < ga.STATIC_MOVING_MM_S
    assert f.static_interval_shape_rms_max_mm > ga.SHAPE_CHANGE_MM
    assert (f.static_interval_shape_rms_mm
            < f.static_interval_shape_rms_max_mm)
    (note,) = f.pose_notes
    assert note.startswith("the hand's shape changed inside the static "
                           "interval (up to ")
    both = [hand_line(k, shift_m=0.030 * k / HZ, lift_m=lifted(k))
            for k in range(450)]
    f = take_facts(both, held(both), "left")
    moving, changed = f.pose_notes
    assert moving.startswith("the hand was moving during its stillest 2 s")
    assert "stayed within" not in moving
    assert changed.startswith("the hand's shape changed")


def test_the_stillness_is_of_the_window_and_against_the_summary_frame():
    """Movement outside the static interval does not count; the shape is
    measured against the meta's summary frame, wherever that is."""
    def shift(k):
        return 0.0 if 90 <= k <= 270 else 0.5 * k / HZ
    lines = [hand_line(k, shift_m=shift(k)) for k in range(450)]
    f = take_facts(lines, held(lines), "left")
    assert f.static_interval_mean_speed_mm_s == pytest.approx(0.0, abs=1e-9)
    speed, rms, worst = ga.stillness(lines[90:271], lines[180])
    assert (speed, rms, worst) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    # no summary frame: the speed still, no shape to compare
    assert ga.stillness(lines[90:271], None) == (pytest.approx(0.0), None,
                                                 None)
    assert ga.stillness(lines[:1], lines[0]) == (None, None, None)


def test_the_hand_size_is_the_index_metacarpal_and_its_change_in_the_take():
    lines = [hand_line(k, bone_mm=60.0 + 4.0 * (k % 2)) for k in range(450)]
    f = take_facts(lines, held(lines), "left")
    assert f.index_metacarpal_mm == pytest.approx(62.0, abs=1e-6)
    assert f.index_metacarpal_range_mm == pytest.approx(4.0, abs=1e-6)
    # one joint 4 mm out is a size the tracker changed, not a note
    assert f.pose_notes == []


def test_the_back_of_the_hand_to_the_camera_is_said_as_the_trackers_fit():
    lines = [hand_line(k) for k in range(450)]
    meta = dict(held(lines), view_angle_deg=140.2)
    f = take_facts(lines, meta, "left")
    (note,) = f.pose_notes
    assert note.startswith(ga.BACK_OF_HAND_NOTE)
    assert "140 degrees" in note and "tracker" in note
    assert f.abnormal
    # edge-on or palm to the lens is the ordinary case
    for view in (0.0, 45.0, 90.0):
        meta["view_angle_deg"] = view
        assert take_facts(lines, meta, "left").pose_notes == []


# --- how much of a whole take is the pose (second review, 2026-10-05) -------------------
def test_a_take_that_begins_in_another_shape_says_so_and_counts_its_frames():
    """The first 0.4 s with five joints 60 mm out (a hand still closing),
    then the pose: the stillest two seconds are the pose, the start is not,
    and the frames outside it are counted."""
    def lift(k):
        return 0.060 * max(0.0, 1.0 - k / 36.0)
    lines = [hand_line(k, lift_m=lift(k)) for k in range(450)]
    f = take_facts(lines, held(lines), "left")
    assert f.static_interval_shape_rms_max_mm == pytest.approx(0.0, abs=1e-6)
    assert f.shape_frames == 450
    # 60 mm on 5 of 26 joints is 26.3 mm RMS before the fit takes some out
    assert 15.0 < f.take_shape_rms_max_mm < 26.4
    assert f.shape_rms_start_mm > ga.NOT_THE_POSE_MM
    assert f.shape_rms_end_mm == pytest.approx(0.0, abs=1e-6)
    assert 0 < f.frames_over_10mm < f.frames_over_5mm < 36
    assert f.frames_within_5mm_percent == pytest.approx(
        100.0 * (450 - f.frames_over_5mm) / 450)
    (note,) = f.extent_notes
    assert note.startswith("the hand was not yet in the representative "
                           "shape when the recording began (first 0.2 s: ")
    assert f.pose_notes == [] and f.notes == [note] and f.abnormal
    # the same at the other end
    lines = [hand_line(k, lift_m=lift(449 - k)) for k in range(450)]
    f = take_facts(lines, held(lines), "left")
    (note,) = f.extent_notes
    assert note.startswith("the hand had left the representative shape by "
                           "the end of the recording (last 0.2 s: ")
    assert f.shape_rms_start_mm == pytest.approx(0.0, abs=1e-6)
    # a take that holds its shape throughout has nothing to say
    still = [hand_line(k) for k in range(450)]
    f = take_facts(still, held(still), "left")
    assert f.extent_notes == [] and f.frames_within_5mm_percent == 100.0
    assert (f.frames_over_5mm, f.frames_over_10mm) == (0, 0)


def test_the_static_interval_names_its_tracker_frames():
    lines = [hand_line(k) for k in range(450)]
    f = take_facts(lines, held(lines, start=90), "left")
    # line k carries frame id 1000 + k; two seconds at 90 Hz
    assert f.static_interval_first_frame_id == 1090
    assert f.static_interval_last_frame_id == 1270
    assert (f.static_interval_first_frame_id <= f.representative_frame_id
            <= f.static_interval_last_frame_id)
    # no static interval in the meta: no frame numbers
    f = take_facts(lines, {}, "left")
    assert f.static_interval_first_frame_id is None
    assert f.static_interval_last_frame_id is None


def test_the_pose_speed_of_a_drifting_hand_is_near_zero():
    """30 mm/s across the camera with the pose held: the window's own score
    says 30, its pose speed says the pose did not move."""
    lines = [hand_line(k, shift_m=0.030 * k / HZ) for k in range(450)]
    f = take_facts(lines, held(lines), "left")
    assert f.static_interval_mean_speed_mm_s == pytest.approx(30.0, abs=0.1)
    assert f.static_interval_pose_speed_mm_s == pytest.approx(0.0, abs=0.01)
    # five finger joints (none of the palm's) moving 30 mm/s in a hand that
    # stays put: the two speeds are the same number
    fingers = [8, 9, 10, 13, 14]      # index and middle, past the knuckle
    lines = []
    for k in range(450):
        d = hand_line(k)
        pos = np.array(d["abs26"])
        pos[fingers, 1] += 0.060 * min(1.0, max(0.0, (k - 90) / 180.0))
        d["abs26"] = pos.tolist()
        lines.append(d)
    f = take_facts(lines, held(lines), "left")
    assert f.static_interval_mean_speed_mm_s == pytest.approx(30.0 * 5 / 26,
                                                              rel=0.02)
    assert f.static_interval_pose_speed_mm_s == pytest.approx(
        f.static_interval_mean_speed_mm_s, rel=0.02)


# --- the other hand's shape ----------------------------------------------------
def mock_lines(side: str, pose: str, n: int = 450, hand_id: int = 7,
               seed: int = 3) -> list:
    """`n` lines of one mock hand in `pose`, as the recorder writes them
    (joints and abs26, so a joint-frame table can be made of each)."""
    import test_protocol_check as fx
    from leap_hand.mock import MockLeapStream

    stream = MockLeapStream(hz=HZ, pose=pose, seed=seed, dropout_every=0,
                            reacquire_every=0, sides=(side,))
    stream._t0_us = int(T0 * 1e6)
    out = []
    for _s, lh in stream.generate(n):
        d = fx.leap_line(lh, lh.capture_time + 0.003, {})
        d["hand_id"] = hand_id
        out.append(d)
    return out


def both(own: list, other: list) -> list:
    return [d for pair in zip(own, other) for d in pair] + own[len(other):]


WINDOW = {"take_start": T0, "take_end": T0 + 5.0, "operator_hand_ids": [7]}


def test_the_other_hands_finger_curl_is_measured_against_the_labelled_hands():
    """A fist with an open hand beside it: the other hand is far less
    curled. The number is the four fingers' MCP + PIP + DIP flexion of each
    hand at its line nearest the middle of the window, other minus own."""
    from xr_hand import joint_frames as jf

    fist = mock_lines("left", "fist")
    open_ = mock_lines("right", "open_palm", hand_id=8)
    f = take_facts(both(fist, open_), WINDOW, "left")
    mid = T0 + 2.5
    near = lambda rows: min(rows, key=lambda d: abs(d["wall_time"] - mid))  # noqa: E731

    def curl(line):
        table = {r["joint"]: r for r in jf.joint_table(line, "camera")}
        return sum(table[f"{g}_{s}"]["flex_deg"]
                   for g in ("INDEX", "MIDDLE", "RING", "LITTLE")
                   for s in ("PROXIMAL", "INTERMEDIATE", "DISTAL"))
    assert f.own_finger_curl_deg == pytest.approx(curl(near(fist)))
    assert f.other_finger_curl_deg == pytest.approx(curl(near(open_)))
    assert f.other_hand_finger_curl_minus_own_deg == pytest.approx(
        f.other_finger_curl_deg - f.own_finger_curl_deg)
    assert f.other_hand_finger_curl_minus_own_deg < -300.0
    # the same pose on both hands: the two curls agree, whichever the side
    same = take_facts(both(fist, mock_lines("right", "fist", hand_id=8)),
                      WINDOW, "left")
    assert abs(same.other_hand_finger_curl_minus_own_deg) < 30.0


def test_an_other_hand_on_under_half_the_frames_is_not_measured():
    fist = mock_lines("left", "fist")
    some = mock_lines("right", "open_palm", hand_id=8)
    f = take_facts(both(fist, some[:200]), WINDOW, "left")      # 200 of 450
    assert f.other_hand_detected and f.other_hand_frame_fraction < 0.5
    assert (f.own_finger_curl_deg, f.other_finger_curl_deg,
            f.other_hand_finger_curl_minus_own_deg) == (None, None, None)
    f = take_facts(both(fist, some[:230]), WINDOW, "left")      # 230 of 450
    assert f.other_hand_finger_curl_minus_own_deg is not None
    # without another hand, nothing; without a window, the hand's own span
    assert take_facts(fist, WINDOW, "left").other_finger_curl_deg is None
    assert take_facts(both(fist, some), {"operator_hand_ids": [7]},
                      "left").other_hand_finger_curl_minus_own_deg < -300.0


def test_the_back_of_hand_note_leaves_both_readings_open():
    lines = [hand_line(k) for k in range(450)]
    f = take_facts(lines, dict(held(lines), view_angle_deg=143.0), "left")
    (note,) = f.pose_notes
    assert "either the hand was held that way" in note
    assert "or the fit is upside down; which was not determined" in note
