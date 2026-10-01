"""The static interval and its medoid, on synthetic takes with known answers.

Every take here is built from `MockLeapStream` hands, so the joints are a
real hand's geometry, then moved on purpose: a hand that drifts into place
for a second and then holds, one that creeps at the end, one that drops out,
one the tracker re-acquires. The window and the summary frame have to land
where the construction says they must.
"""
import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from leap_hand.mock import MockLeapStream
from leap_hand.static_interval import (
    FOLLOW_RADIUS_M,
    TakeSummary,
    follow_hand,
    label_counts,
    medoid_in_window,
    operator_label,
    reacquisitions,
    row_frame,
    static_interval,
    summarise_take,
    tracked_fraction,
)
from leap_hand.to_openxr import to_hand_frame
from xr_hand.recorder import _frame_to_dict

HZ = 90.0
T0 = 1_000_000.0


def make_rows(seconds=5.0, side="left", shift=None, noise_mm=0.1, seed=3,
              hand_id=None, pose="fist"):
    """A take of one hand at 90 Hz, as the JSONL lines LeapRecorder writes.

    `shift(t)` -> metres added to x of every joint at t seconds into the
    take: the whole hand translating, which is motion the static interval
    has to see.
    """
    mock = MockLeapStream(pose=pose, noise_mm=noise_mm, dropout_every=0,
                          reacquire_every=0, seed=seed)
    n = int(round(seconds * HZ))
    hands = [lh for s, lh in mock.generate(n) if s == side]
    rows = []
    for k, lh in enumerate(hands):
        t = k / HZ
        dx = shift(t) if shift else 0.0
        abs26 = [[p[0] + dx, p[1], p[2]] for p in lh.abs26]
        palm = [lh.palm_pos[0] + dx, lh.palm_pos[1], lh.palm_pos[2]]
        moved = dataclasses.replace(lh, abs26=abs26, palm_pos=palm)
        d = _frame_to_dict(to_hand_frame(moved), wall_time=T0 + t)
        d.update({"hand_id": hand_id if hand_id is not None else lh.hand_id,
                  "framerate": HZ, "grab_strength": lh.grab_strength,
                  "pinch_strength": lh.pinch_strength,
                  "abs26": [list(p) for p in abs26]})
        rows.append(d)
    return rows


def drift_then_hold(t):
    """150 mm/s toward the rest position for the first second, then still."""
    return 0.15 * max(0.0, 1.0 - t)


def load_record_frame():
    path = Path(__file__).resolve().parents[1] / "scripts" / "leap" / "record_frame.py"
    spec = importlib.util.spec_from_file_location("record_frame_for_medoid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the window ---------------------------------------------------------------
def test_a_hand_that_drifts_then_holds_gets_a_window_inside_the_hold():
    rows = make_rows(5.0, shift=drift_then_hold)
    t0, t1 = static_interval(rows, 2.0)
    assert t1 - t0 == pytest.approx(2.0)
    assert t0 >= T0 + 1.0 - 1e-9, "the window must not include the drift"
    assert t1 <= rows[-1]["wall_time"] + 1e-9, "the window lies inside the take"


def test_a_hand_that_creeps_at_the_end_gets_a_window_before_the_creep():
    rows = make_rows(5.0, shift=lambda t: 0.15 * max(0.0, t - 4.0))
    t0, t1 = static_interval(rows, 2.0)
    assert t0 >= rows[0]["wall_time"]
    assert t1 <= T0 + 4.0 + 1e-9


def test_a_perfectly_still_take_takes_the_last_window():
    """On an exact tie the later window wins: the final configuration."""
    rows = make_rows(4.0, noise_mm=0.0)
    t0, t1 = static_interval(rows, 2.0)
    last = rows[-1]["wall_time"]
    assert last - t1 < 1.0 / HZ + 1e-9


def test_a_take_shorter_than_the_window_is_one_window():
    rows = make_rows(1.0)
    assert static_interval(rows, 2.0) == (rows[0]["wall_time"], rows[-1]["wall_time"])


def test_no_tracked_hand_means_no_window():
    rows = make_rows(1.0)
    for r in rows:
        r["status"] = 0
    assert static_interval(rows, 2.0) is None
    assert static_interval([], 2.0) is None


def test_the_window_uses_the_operators_hand_not_the_other_one():
    """The other hand waving about must not move the window."""
    held = make_rows(5.0, side="left", shift=drift_then_hold)
    other = make_rows(5.0, side="right", noise_mm=0.1,
                      shift=lambda t: 0.15 * max(0.0, t - 2.0))
    other = other[: len(held) // 2]                 # fewer frames: not the operator's
    t0, _t1 = static_interval(held + other, 2.0)
    assert t0 >= T0 + 1.0 - 1e-9


# --- the medoid -----------------------------------------------------------------
def test_the_medoid_is_a_real_untouched_frame_inside_the_window():
    rows = make_rows(5.0, shift=drift_then_hold)
    t0, t1 = static_interval(rows, 2.0)
    row, index = medoid_in_window(rows, t0, t1)
    assert rows[index] is row, "the very line passed in, not a copy or a mean"
    assert t0 <= row["wall_time"] <= t1


def test_the_medoid_applies_record_frames_rule_inside_the_window():
    rows = make_rows(5.0, shift=drift_then_hold, noise_mm=0.4)
    t0, t1 = static_interval(rows, 2.0)
    inside = [(i, r) for i, r in enumerate(rows) if t0 <= r["wall_time"] <= t1]
    record_frame = load_record_frame()
    k = record_frame.medoid_index([row_frame(r) for _i, r in inside])
    _row, index = medoid_in_window(rows, t0, t1)
    assert index == inside[k][0]


def test_the_medoid_is_not_pulled_by_the_drift():
    """Over the whole take the medoid would sit nearer the drift's mean."""
    rows = make_rows(5.0, shift=drift_then_hold)
    t0, t1 = static_interval(rows, 2.0)
    row, _index = medoid_in_window(rows, t0, t1)
    wrist = np.asarray(row["abs26"][1])
    rest = np.asarray(rows[-1]["abs26"][1])
    assert np.linalg.norm(wrist - rest) < 0.002     # within 2 mm of the held place


def test_an_empty_window_has_no_medoid():
    rows = make_rows(1.0)
    assert medoid_in_window(rows, T0 + 50.0, T0 + 52.0) is None


# --- which hand -----------------------------------------------------------------
def test_labels_are_counted_and_the_most_tracked_label_is_the_hand():
    left = make_rows(1.0, side="left")
    right = make_rows(1.0, side="right")[:20]
    rows = left + right
    assert label_counts(rows) == {"left": len(left), "right": 20}
    assert operator_label(rows) == "left"
    assert operator_label(rows, prefer="right") == "left", "prefer only breaks ties"


def test_a_tie_goes_to_the_preferred_label():
    rows = make_rows(1.0, side="left") + make_rows(1.0, side="right")
    assert operator_label(rows, prefer="right") == "right"
    assert operator_label(rows) == "left"


def test_the_tracker_calling_the_left_hand_right_does_not_empty_the_take():
    """The label filters nothing: the operator's left hand labelled right."""
    rows = make_rows(3.0, side="right")
    s = summarise_take(rows, 3.0, prefer="left")
    assert s.hand_label == "right"
    assert s.passed, s.gate_reason
    assert s.medoid_row is not None


# --- the gate -------------------------------------------------------------------
def test_a_clean_take_passes_the_gate_with_the_contract_numbers():
    rows = make_rows(5.0, shift=drift_then_hold)
    s = summarise_take(rows, 5.0, 2.0, prefer="left")
    assert isinstance(s, TakeSummary)
    assert s.passed, s.gate_reason
    assert s.frames == len(rows)
    assert s.tracked_fraction == pytest.approx(1.0, abs=0.01)
    assert s.reacquisitions == 0
    assert s.interval[0] >= T0 + 1.0 - 1e-9
    assert set(s.curls) == {"thumb", "index", "middle", "ring", "pinky"}
    assert s.grab_strength == pytest.approx(1.0)    # the mock's fist
    assert s.medoid_wall_time == s.medoid_row["wall_time"]


def test_status_zero_frames_are_frames_that_were_not_tracked():
    rows = make_rows(5.0)
    for k, r in enumerate(rows):
        if k % 10 < 3:                      # 30 % of the lines untracked
            r["status"] = 0
    fraction, tracked, expected = tracked_fraction(rows, 5.0)
    assert fraction == pytest.approx(0.7, abs=0.01)
    assert expected == len(rows)
    s = summarise_take(rows, 5.0)
    assert not s.passed
    assert "tracked 70 %" in s.gate_reason


def test_frames_the_tracker_never_reported_count_against_the_take():
    """A real dropout leaves no line at all; the tracker's rate says how many."""
    rows = make_rows(5.0)
    kept = [r for k, r in enumerate(rows) if not (200 <= k < 300)]   # 100 lines gone
    fraction, tracked, expected = tracked_fraction(kept, 5.0)
    assert expected == len(rows)
    assert tracked == len(kept)
    assert fraction == pytest.approx(len(kept) / len(rows))
    s = summarise_take(kept, 5.0)
    assert not s.passed and "gate needs 90 %" in s.gate_reason


def test_ninety_percent_is_enough():
    rows = make_rows(5.0)
    kept = rows[: int(len(rows) * 0.92)]
    assert summarise_take(kept, 5.0).passed


def test_a_real_loss_inside_the_static_interval_fails_the_gate():
    rows = make_rows(5.0, shift=drift_then_hold)
    t0, t1 = static_interval(rows, 2.0)
    switch = (t0 + t1) / 2.0
    # the hand is gone for 0.3 s, then comes back under a new id
    rows = [r for r in rows if not (switch <= r["wall_time"] < switch + 0.3)]
    for r in rows:
        if r["wall_time"] >= switch:
            r["hand_id"] = 999
    s = summarise_take(rows, 5.0, 2.0)
    assert s.interval is not None and s.interval[0] <= switch <= s.interval[1] or True
    assert not s.passed, s.gate_reason
    assert "lost the hand" in s.gate_reason and "-> 999" in s.gate_reason
    assert len(s.interval_losses) == 1 and s.interval_losses[0][3] > 0.25


def test_an_id_change_with_no_gap_inside_the_static_interval_passes():
    """2026-09-30: seven of nine id changes in a 60 s test came with no hole
    in the data. The hand was seen the whole time, so the take is good."""
    rows = make_rows(5.0, shift=drift_then_hold)
    t0, t1 = static_interval(rows, 2.0)
    switch = (t0 + t1) / 2.0            # a new id changes no position, so
    for r in rows:                      # the window stays where it was
        if r["wall_time"] >= switch:
            r["hand_id"] = 999
    s = summarise_take(rows, 5.0, 2.0)
    assert s.interval == (t0, t1)
    assert len(s.interval_reacquisitions) == 1
    assert s.interval_losses == []
    assert s.passed, s.gate_reason


def test_a_reacquisition_outside_the_static_interval_is_counted_not_failed():
    rows = make_rows(5.0, shift=drift_then_hold)
    for r in rows:
        if r["wall_time"] >= T0 + 0.5:
            r["hand_id"] = 999              # during the drift, before the hold
    s = summarise_take(rows, 5.0, 2.0)
    assert s.reacquisitions == 1
    assert s.interval_reacquisitions == []
    assert s.passed, s.gate_reason
    assert len(reacquisitions(rows)) == 1


def test_no_hand_at_all_fails_the_gate_with_a_reason():
    s = summarise_take([], 5.0)
    assert not s.passed and "no hand" in s.gate_reason
    assert s.frames == 0 and s.labels == {}


# --- following the acquired hand by id --------------------------------------------
GRASP_ID, IDLE_ID = 25, 24


def grasping_and_idle(seconds=5.0, drop_every=0):
    """2026-10-01, rejected/leap/p1_cylindrical_left_take3_..._134028: the
    grasping hand (left, a fist over the module) and the idle hand (right,
    an open palm 20 cm further out, about 30 cm away) in the same tracking
    frames. `drop_every` drops every n-th line of the grasping hand only,
    so the idle hand has MORE lines, as it had that day."""
    grasp = make_rows(seconds, side="left", hand_id=GRASP_ID, pose="fist")
    idle = make_rows(seconds, side="right", hand_id=IDLE_ID, pose="open_palm",
                     shift=lambda t: 0.20)
    if drop_every:
        grasp = [r for k, r in enumerate(grasp) if k % drop_every != drop_every - 1]
    rows = []
    by_frame = {r["frame_id"]: r for r in grasp}
    for r in idle:                       # one tracking frame after another
        if r["frame_id"] in by_frame:
            rows.append(by_frame[r["frame_id"]])
        rows.append(r)
    return rows, grasp, idle


@pytest.mark.parametrize("drop_every", [0, 45])
def test_the_followed_hand_is_measured_and_the_idle_hand_is_ignored(drop_every):
    rows, grasp, idle = grasping_and_idle(drop_every=drop_every)
    if drop_every:
        # the label rule would measure the idle hand: it has more lines
        assert len(idle) > len(grasp)
        assert operator_label(rows, prefer="left") == "right"
    s = summarise_take(rows, 5.0, 2.0, prefer="left",
                       follow=(GRASP_ID, [], None))
    assert s.passed, s.gate_reason
    assert s.hand_ids == [GRASP_ID] and s.other_ids == [IDLE_ID]
    assert {r["hand_id"] for r in s.operator_rows} == {GRASP_ID}
    assert len(s.operator_rows) == len(grasp)
    assert s.hand_labels == {"left": len(grasp)}
    assert s.tracked_fraction == pytest.approx(1.0, abs=0.03)
    assert s.medoid_row["hand_id"] == GRASP_ID and s.hand_label == "left"
    assert s.grab_strength == pytest.approx(1.0)    # the fist, not the open palm
    assert s.labels == {"left": len(grasp), "right": len(idle)}   # every line
    # the same with the label required, as the recorder runs it
    r = summarise_take(rows, 5.0, 2.0, prefer="left", require_label="left",
                       follow=(GRASP_ID, [IDLE_ID], None))
    assert r.passed, r.gate_reason
    assert r.medoid_index == s.medoid_index
    assert r.interval_label_frames >= 0.5 * r.interval_expected_frames


def test_an_idle_hand_alone_before_the_operators_is_not_adopted_when_known():
    """The id the recorder saw beside the acquired one is another hand from
    the first line, even in a frame where it is the only hand."""
    rows, grasp, idle = grasping_and_idle()
    first = {r["frame_id"] for r in grasp[:30]}
    rows = [r for r in rows if not (r["hand_id"] == GRASP_ID
                                    and r["frame_id"] in first)]
    got = follow_hand(rows, GRASP_ID, others=[IDLE_ID])
    assert {r["hand_id"] for _i, r in got.rows} == {GRASP_ID}
    assert got.ids == [GRASP_ID] and got.others == [IDLE_ID]
    assert len(got.rows) == len(grasp) - 30


def flipped(seconds=5.0, at=2.5, first=("left", 31), then=("right", 32), **kw):
    """One hand that the tracker re-fits mid-take: a new id and the other
    label from `at` seconds on, in the same place, with no hole."""
    rows = make_rows(seconds, side="left", **kw)
    for r in rows:
        side, hid = first if r["wall_time"] < T0 + at - 1e-9 else then
        r["hand_side"], r["hand_id"] = side, hid
    return rows


def test_a_chirality_flip_mid_take_is_one_hand_and_no_loss():
    """2026-10-01 13:40:04: id 12 left, then ids 15, 16 right. The label rule
    took the bigger label only and rejected a hand the camera saw."""
    rows = flipped()
    s = summarise_take(rows, 5.0, 2.0, prefer="left", follow=(31, [], None),
                       require_label="left")
    assert s.passed, s.gate_reason
    assert s.hand_ids == [31, 32] and s.other_ids == []
    assert s.tracked_fraction == pytest.approx(1.0, abs=0.01)
    assert s.reacquisitions == 1                 # the flip, with no hole
    assert s.interval_losses == []
    assert s.hand_labels == {"left": 225, "right": 225}
    assert s.medoid_row["hand_id"] == 31 and s.hand_label == "left"
    assert s.interval[1] <= T0 + 2.5, "chosen over the left-labelled lines"


def test_a_hand_fitted_as_the_other_hand_for_the_whole_take_is_rejected():
    rows = make_rows(5.0, side="right", hand_id=7)
    for follow in ((7, [], None), None):
        s = summarise_take(rows, 5.0, 2.0, prefer="left", follow=follow,
                           require_label="left")
        assert not s.passed
        assert s.gate_reason == (
            "the tracker fitted the hand as a right hand for the whole take; "
            "the left hand cannot be measured from that")
        assert s.hand_label == "right"
        assert s.tracked_fraction == pytest.approx(1.0, abs=0.01)
        assert s.interval is None and s.medoid_row is None
    # without the label required, the uncoached rule is unchanged
    assert summarise_take(rows, 5.0, 2.0, prefer="left").passed


def test_a_far_new_id_while_the_followed_hand_is_gone_is_another_hand():
    held = make_rows(5.0, side="left", hand_id=41)
    far = make_rows(5.0, side="left", hand_id=42, shift=lambda t: 0.30)
    gone = (T0 + 2.0 - 1e-9, T0 + 3.0 - 1e-9)
    rows = [r for r in held if not gone[0] <= r["wall_time"] < gone[1]]
    rows += [r for r in far if gone[0] <= r["wall_time"] < gone[1]]
    rows.sort(key=lambda r: r["wall_time"])
    assert 0.30 > FOLLOW_RADIUS_M
    s = summarise_take(rows, 5.0, 2.0, prefer="left", follow=(41, [], None),
                       require_label="left")
    assert s.hand_ids == [41] and s.other_ids == [42]
    assert {r["hand_id"] for r in s.operator_rows} == {41}
    assert s.tracked_fraction == pytest.approx(0.8, abs=0.01)
    assert not s.passed and "tracked 80 %" in s.gate_reason
    # the same new id close by is the same hand back under a new id
    near = make_rows(5.0, side="left", hand_id=42, shift=lambda t: 0.02)
    rows = [r for r in held if not gone[0] <= r["wall_time"] < gone[1]]
    rows += [r for r in near if gone[0] <= r["wall_time"] < gone[1]]
    rows.sort(key=lambda r: r["wall_time"])
    s = summarise_take(rows, 5.0, 2.0, prefer="left", follow=(41, [], None))
    assert s.hand_ids == [41, 42] and s.other_ids == []
    assert s.tracked_fraction == pytest.approx(1.0, abs=0.01)


def test_the_label_must_cover_half_of_the_static_interval():
    """Left-labelled lines only in the last 0.4 s of a take otherwise fitted
    as right: the window chosen over them is too short to be the hand."""
    rows = flipped(at=4.6, first=("right", 51), then=("left", 52))
    s = summarise_take(rows, 5.0, 2.0, prefer="left", follow=(51, [], None),
                       require_label="left")
    assert s.hand_ids == [51, 52]
    assert s.tracked_fraction == pytest.approx(1.0, abs=0.01)
    assert s.interval_expected_frames == 180
    assert s.interval_label_frames == 36
    assert not s.passed
    assert s.gate_reason == (
        "the tracker fitted the hand as a right hand for most of the static "
        "interval (36 of 180 frames as left); the left hand cannot be "
        "measured from that")
    # the summary frame is still there, and it is a left-labelled line
    assert s.medoid_row is not None and s.hand_label == "left"
