"""`leap_hand.transition`: the seconds from the close cue to a settled hand.

The settle rule is tested on synthetic tracks whose settle time is known by
construction: an open hand held still at the cue (the operator's reaction
time), a close that moves every joint, then the grasp held with the
camera's own jitter on it. Then the forming summary on a mock hand acted
out by `CoachedActor`, followed by id beside the other hand, and the
thresholds against the real hold data they were chosen from.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from leap_hand import transition as tr
from leap_hand.mock import CoachedActor, MockLeapStream
from leap_hand.recorder import LeapRecorder

REPO = Path(__file__).resolve().parents[1]
REAL_SESSION = REPO / "recordings" / "protocol" / "grasps" / "20261001_101621_left"
HZ = 90.0
T0 = 1_790_300_000.0


def track(settle_at, react=0.3, length=2.0, travel_mm=60.0, jitter_mm=0.3,
          hz=HZ, burst=1, holes=(), seed=1):
    """One hand's lines: still until `react`, moving until `settle_at`
    (None: moving to the end), then still. Every joint moves `travel_mm`
    over the close along its own direction, which is several times the
    threshold's speed; `jitter_mm` is per-axis noise on every joint, about
    the camera's own. `burst` lines share one `wall_time`, the way the
    writer stamps a backlog; `holes` are (start, end) seconds with no line.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.08, 0.08, size=(26, 3)) + [0.0, 0.25, 0.0]
    direction = rng.normal(size=(26, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    end = length if settle_at is None else settle_at
    rows = []
    for k in range(int(length * hz)):
        t = k / hz
        if any(a <= t < b for a, b in holes):
            continue
        w = 0.0 if t < react else min(1.0, (t - react) / (end - react))
        pos = base + direction * (travel_mm / 1000.0) * w
        pos = pos + rng.normal(scale=jitter_mm / 1000.0, size=pos.shape)
        wall = T0 + (k // burst) * burst / hz + 0.004
        rows.append({"wall_time": wall, "capture_time": T0 + t,
                     "hand_side": "left", "hand_id": 7, "frame_id": k,
                     "status": 1, "framerate": hz,
                     "abs26": [[round(float(c), 6) for c in p] for p in pos]})
    return rows


# --- the settle rule ----------------------------------------------------------------
@pytest.mark.parametrize("settle_at", [0.6, 1.0, 1.45])
def test_settle_time_is_the_known_settle_within_a_tenth(settle_at):
    got = tr.settle_time(track(settle_at), T0)
    assert got is not None
    assert got == pytest.approx(settle_at, abs=0.1)


def test_settle_time_reads_the_capture_clock_not_the_write_bursts():
    """Ten lines to one wall_time, as a drained backlog is written: the
    capture clock still times the stop."""
    got = tr.settle_time(track(1.0, burst=10), T0)
    assert got == pytest.approx(1.0, abs=0.1)


def test_settle_time_is_none_when_the_hand_never_settles():
    assert tr.settle_time(track(None), T0) is None
    # settled, but too late for a whole 0.3 s window before the end
    assert tr.settle_time(track(1.85), T0, t_end=T0 + 2.0) is None
    # settled in the data, but after the forming phase ended
    assert tr.settle_time(track(1.0), T0, t_end=T0 + 1.1) is None


def test_settle_time_needs_the_hand_to_move_first():
    """The open hand is still at the cue too: that is not the grasp."""
    still = track(None, react=5.0)
    assert np.nanmax(tr.joint_speeds(still)[1]) < tr.SETTLE_SPEED_MM_S
    assert tr.settle_time(still, T0) is None


def test_settle_time_counts_from_the_cue_and_ignores_frames_before_it():
    rows = track(1.0)
    # the cue 0.2 s into the track: frames before it do not count
    got = tr.settle_time(rows, T0 + 0.2)
    assert got == pytest.approx(0.8, abs=0.1)


def test_settle_window_must_not_span_a_hole():
    """Quiet from 1.0 s, but the hand is gone 1.1 to 1.3 s: it is first seen
    still for a whole window after the hole."""
    got = tr.settle_time(track(1.0, holes=[(1.1, 1.3)]), T0)
    assert got is not None and got >= 1.3 - 0.06


def test_settle_thresholds_are_the_named_constants():
    import inspect

    sig = inspect.signature(tr.settle_time)
    assert sig.parameters["speed_thresh_mm_s"].default == tr.SETTLE_SPEED_MM_S == 20.0
    assert sig.parameters["window_s"].default == tr.SETTLE_WINDOW_S == 0.3
    assert sig.parameters["step_s"].default == tr.SPEED_STEP_S == 0.1
    # the slower the settle threshold, the later the hand counts as settled
    rows = track(1.0)
    assert tr.settle_time(rows, T0, speed_thresh_mm_s=60.0) <= tr.settle_time(rows, T0)


# --- the four shapes the first real forming files showed --------------------------------
def profile(moves, length=3.0, jitter_mm=0.3, hz=HZ, seed=3, carry=(),
            pose_jitter_mm=0.05):
    """One hand's lines changing pose at a known mean joint speed: `moves`
    are (start s, end s, mm/s). The palm's eleven joints stay where they
    are and the other fifteen move along their own directions, fast enough
    that the mean over the 26 is the speed given: the pose speed and the
    speed over the camera are then the same number. `carry` are (start s,
    end s, mm/s) of the hand carried as a whole along one line with its
    pose held, which only the speed over the camera sees. `jitter_mm` is
    noise on the hand as a whole (per axis; about 7 mm/s of speed over the
    camera, the camera's own) and `pose_jitter_mm` on every joint by itself
    (about 1 mm/s of pose speed, a held real hand's). The 0.25 s medians of
    the real takes, built back: a close at 80 to 190 mm/s, a twitch or an
    adjustment at 20 to 35 (or a fast one), a hold at 20 or less."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.08, 0.08, size=(26, 3)) + [0.0, 0.25, 0.0]
    direction = rng.normal(size=(26, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    direction[tr.PALM_INDEX] = 0.0
    direction *= 26.0 / (26 - len(tr.PALM_INDEX))
    rows, travel, carried = [], 0.0, 0.0
    for k in range(int(length * hz)):
        t = k / hz
        if k:
            mid = t - 0.5 / hz
            travel += sum(v for a, b, v in moves if a <= mid < b) / hz
            carried += sum(v for a, b, v in carry if a <= mid < b) / hz
        pos = base + direction * travel / 1000.0
        pos = pos + np.array([0.0, 0.0, carried / 1000.0])
        pos = pos + rng.normal(scale=jitter_mm / 1000.0, size=3)
        pos = pos + rng.normal(scale=pose_jitter_mm / 1000.0, size=pos.shape)
        rows.append({"wall_time": T0 + t + 0.004, "capture_time": T0 + t,
                     "hand_side": "left", "hand_id": 7, "frame_id": k,
                     "status": 1, "framerate": hz,
                     "abs26": [[round(float(c), 6) for c in p] for p in pos]})
    return rows


def test_settle_after_a_clean_close():
    got = tr.settle(profile([(0.3, 0.8, 150.0)]), T0)
    assert got.peak_speed_mm_s == pytest.approx(150.0, rel=0.15)
    assert 0.3 <= got.peak_at_s <= 0.8
    assert got.transition_s == pytest.approx(0.8, abs=0.1)
    assert got.last_move_s == pytest.approx(0.8, abs=0.1)
    assert got.note == "" and got.moved_again_s is None


@pytest.mark.parametrize("twitch_mm_s", [35.0, 80.0])
def test_settle_after_a_twitch_and_a_pause_is_the_close_not_the_pause(twitch_mm_s):
    """p1_hook take 5: a twitch on the beep, a pause longer than the quiet
    window, then the close. The first rule settled in the pause (0.18 s);
    the settle is after the close, the fastest movement, whether the
    twitch was slow or itself above the move speed."""
    rows = profile([(0.05, 0.2, twitch_mm_s), (0.7, 1.1, 130.0)])
    got = tr.settle(rows, T0)
    assert 0.7 <= got.peak_at_s <= 1.1
    assert got.transition_s == pytest.approx(1.1, abs=0.1)
    assert got.last_move_s == pytest.approx(1.1, abs=0.1)
    assert got.note == ""
    assert tr.settle_time(rows, T0) == got.transition_s
    # the trap is there: the pause is a whole quiet window after the twitch,
    # where the first rule (settled after the first frame above 20) stopped
    t, speed = tr.joint_speeds(rows)
    pause = (t - T0 >= 0.3) & (t - T0 <= 0.6)
    assert np.nanmax(speed[pause]) < tr.SETTLE_SPEED_MM_S
    assert np.nanmax(speed[t - T0 <= 0.2]) >= tr.SETTLE_SPEED_MM_S


def test_settle_after_a_close_then_an_adjustment_says_it_moved_again():
    """p1_hook take 4: settled after the close, then the grip adjusted. The
    transition is the settle after the close; last_move_s and the note say
    when the hand moved again."""
    got = tr.settle(profile([(0.2, 0.8, 160.0), (2.0, 2.2, 90.0)]), T0)
    assert 0.2 <= got.peak_at_s <= 0.8
    assert got.transition_s == pytest.approx(0.8, abs=0.1)
    assert got.moved_again_s == pytest.approx(2.0, abs=0.1)
    assert got.last_move_s == pytest.approx(2.2, abs=0.1)
    assert got.note == f"settled, then moved again at {got.moved_again_s:.2f} s"
    assert got.last_move_s > got.transition_s + tr.SETTLE_WINDOW_S


def test_settle_with_the_hand_already_closed_has_no_transition_and_says_why():
    """p1_lateral takes 4 and 5: the operator closed on the acquisition
    beep. After the cue the hand only shifts a little (at 30 mm/s for 0.2 s)
    and holds: no frame at the move speed, so nothing to time."""
    got = tr.settle(profile([(0.0, 0.2, 30.0)]), T0)
    assert got.peak_speed_mm_s < tr.MOVE_SPEED_MM_S
    assert got.transition_s is None and got.last_move_s is None
    assert got.note == (f"no close after the cue (peak "
                        f"{got.peak_speed_mm_s:.0f} mm/s): the hand was already "
                        "in the grasp")
    assert tr.settle_time(profile([(0.0, 0.2, 30.0)]), T0) is None


def test_settle_never_found_before_the_hold_says_so():
    got = tr.settle(profile([(0.2, 0.6, 150.0), (0.6, 3.0, 35.0)]), T0)
    assert got.peak_speed_mm_s >= tr.MOVE_SPEED_MM_S
    assert got.transition_s is None
    assert got.note == "never settled before the hold"
    assert got.last_move_s == pytest.approx(0.6, abs=0.1)


def test_settle_with_no_frames_says_there_was_nothing_to_measure():
    got = tr.settle([], T0)
    assert got.peak_speed_mm_s is None and got.transition_s is None
    assert got.note == tr.NOTE_NO_FRAMES


def test_settle_move_threshold_sits_between_the_real_shapes():
    """MOVE_SPEED_MM_S against the numbers in the module docstring: above
    the fastest frame of the two takes already closed (38 and 43 mm/s),
    under a third of the slowest real close's peak (180), and above the
    0.25 s medians of a twitch or an adjustment (20 to 35)."""
    assert tr.MOVE_SPEED_MM_S == 50.0
    assert 43.0 < tr.MOVE_SPEED_MM_S < 180.0 / 3
    assert 35.0 < tr.MOVE_SPEED_MM_S
    assert tr.transition_rule() == {"speed_mm_s": 20.0, "window_s": 0.3,
                                    "step_s": 0.1, "move_speed_mm_s": 50.0}


def test_transition_coaching_numbers_are_rounded_as_the_meta_has_them():
    s = tr.FormingSummary(transition_s=1.68912, peak_speed_mm_s=397.84,
                          peak_at_s=0.56612, last_move_s=None,
                          transition_note="")
    assert s.coaching_numbers() == {
        "transition_s": 1.689, "peak_speed_mm_s": 397.8, "peak_at_s": 0.566,
        "last_move_s": None, "transition_note": "",
        "transition_rule": tr.transition_rule()}
    assert list(s.coaching_numbers())[0] == "transition_s"


# --- the forming summary on an acted mock hand -----------------------------------------
def forming_rows(tmp_path, close_s=0.4, seconds=1.5, lose=False):
    """MAKE THE GRASP acted by the mock, both hands, written by LeapRecorder
    and read back: the forming file's lines."""
    actor = CoachedActor(close_s=close_s, operator="left",
                         lose_forming=1 if lose else 0)
    s = MockLeapStream(dropout_every=0, reacquire_every=0, script=actor)
    s.start()
    actor.phase("open", 0)
    actor.phase("form", 0)
    path = tmp_path / "forming.jsonl"
    rec = LeapRecorder(hz=None)
    rec.start(path)
    hands = s.generate(int(seconds * HZ))
    for _side, lh in hands:
        rec.record(lh)
    rec.stop()
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    left = next(lh for side, lh in hands if side == "left")
    right = next(lh for side, lh in hands if side == "right")
    t0 = left.capture_time
    return rows, t0, left, right


def test_summarise_forming_times_the_acted_close(tmp_path):
    rows, t0, left, right = forming_rows(tmp_path, close_s=0.4)
    s = tr.summarise_forming(rows, t0, t0 + 1.5,
                             follow=(left.hand_id, [right.hand_id],
                                     list(left.palm_pos)),
                             require_label="left", prefer="left")
    assert s.frames == len(rows) == 2 * int(1.5 * HZ)
    assert s.hand_ids == [left.hand_id]
    assert s.tracked_fraction == pytest.approx(1.0, abs=0.02)
    # a smoothstep close of 0.4 s slows below 20 mm/s just before its end
    assert s.transition_s is not None
    assert 0.25 <= s.transition_s <= 0.45


def test_summarise_forming_without_a_settle_is_null(tmp_path):
    """The hand lost while it closes (palm turned edge-on, then gone): no
    quiet window after the motion, so no transition, and the tracked
    fraction says how much of the phase the hand was there."""
    rows, t0, left, right = forming_rows(tmp_path, close_s=1.0, lose=True)
    s = tr.summarise_forming(rows, t0, t0 + 1.5,
                             follow=(left.hand_id, [right.hand_id],
                                     list(left.palm_pos)),
                             require_label="left")
    assert s.transition_s is None
    assert s.tracked_fraction < 0.5


# --- a forming file in which the hand was lost (recorder rules of 2026-10-07) ----------
def test_a_forming_file_marked_lost_is_read_by_no_rule():
    """The same clean close, timed as it is (the control), then with every
    line marked `forming_lost_count` 1, as the recorder marks the forming
    file of a take kept after the hand was lost while forming: every rule
    says invalid or null with the reason, and gives no time."""
    rows = profile([(0.3, 0.8, 150.0)])
    clean = tr.transition_timing(rows, T0)
    assert clean.validity == tr.VALID and clean.transition_duration_s
    assert clean.forming_lost_count == 0
    marked = [dict(r, forming_lost_count=1) for r in rows]
    assert tr.forming_lost_count(marked) == 1 and tr.forming_lost_count(rows) == 0
    g = tr.transition_timing(marked, T0)
    assert g.validity == tr.INVALID and g.forming_lost_count == 1
    assert g.notes == tr.lost_note(1) == (
        "the hand was lost while forming the grasp (1 time): a change of "
        "pose is not timed across a gap in the tracking")
    for value in (g.movement_start_time_s, g.settled_time_s,
                  g.transition_duration_s, g.reaction_time_s,
                  g.cue_to_settled_s, g.peak_speed_mm_s, g.pose_change_mm):
        assert value is None
    # which lines it was is still said
    assert g.frames == len(rows) and g.hand_ids == [7]
    assert g.first_frame_time == pytest.approx(T0)
    assert g.cue_time_s == pytest.approx(0.0, abs=1e-9)
    # a check that times the file again from some of its lines agrees
    again = tr.transition_timing(marked[::2], T0)
    assert again.validity == tr.INVALID and again.movement_start_time_s is None
    # the recorder's own summary: no transition, the note, the tracked fraction
    s = tr.summarise_forming(marked, T0, T0 + 3.0)
    assert s.transition_s is None and s.peak_speed_mm_s is None
    assert s.transition_note == tr.lost_note(1)
    assert s.tracked_fraction == pytest.approx(1.0, abs=0.02)
    assert tr.summarise_forming(rows, T0, T0 + 3.0).transition_s is not None
    # the hand-in's reading, from the meta's count alone or from the lines
    meta = {"hand": "left", "take_start": T0 + 3.0,
            "coaching": {"form_start": T0, "acquired": [{"hand_id": 7}],
                         "other_hands": []}}
    assert tr.forming_timing(rows, meta).validity == tr.VALID
    two = dict(meta, coaching=dict(meta["coaching"], forming_lost_count=2))
    g = tr.forming_timing(rows, two)
    assert g.validity == tr.INVALID and g.notes == tr.lost_note(2)
    assert "(2 times)" in g.notes and g.transition_duration_s is None
    assert tr.forming_timing(marked, meta).notes == tr.lost_note(1)


def test_the_mock_hand_lost_while_forming_can_come_back_under_a_new_id():
    """`CoachedActor(back_after_s=...)`: the hand turns edge-on, is gone for
    that long, and is back in the grasp, palm to the lens, under a new id
    that it keeps through the hold; without it, gone until the next OPEN
    HAND, as before."""
    for back in (0.4, None):
        actor = CoachedActor(operator="left", lose_forming=1, close_s=0.3,
                             lose_at_s=0.6, back_after_s=back)
        s = MockLeapStream(dropout_every=0, reacquire_every=0, script=actor,
                           sides=("left",))
        s.start()
        actor.phase("open", 0)
        actor.phase("ready", 45)
        actor.phase("form", 135)
        # 2 s after the GO cue: the hold, or (never back) the attempt given up
        actor.phase("hold" if back else "idle", 315)
        hands = [lh for _sd, lh in s.generate(400)]
        form = [lh for lh in hands if 135 <= lh.frame_id < 315]
        t = [(lh.frame_id - 135) / HZ for lh in form]
        first = hands[0].hand_id
        assert all(lh.hand_id == first for lh, ti in zip(form, t) if ti < 0.6)
        assert not [ti for ti in t if 0.61 <= ti < 0.6 + (back or 0.0) - 0.01]
        hold = [lh for lh in hands if lh.frame_id >= 315]
        if back is None:
            assert not [ti for ti in t if ti >= 0.61] and not hold
            continue
        found = [lh for lh, ti in zip(form, t) if ti >= 0.6 + back]
        assert found and {lh.hand_id for lh in found} == {first + 1}
        assert all(lh.grab_strength == pytest.approx(0.6) for lh in found)
        assert hold and {lh.hand_id for lh in hold} == {first + 1}
        assert actor.lost == 1


# --- the thresholds against the data they were read from --------------------------------
def recorded_on_2026_10_02(meta: dict) -> bool:
    """One of the six forming takes these numbers were read from: recorded
    on 2026-10-02 (the take's name carries its stamp). The session is
    resumed into, and later takes, with forming files of their own, are not
    the data the docstring's numbers describe."""
    return "_20261002_" in str(meta.get("name") or "")


@pytest.mark.skipif(not (REAL_SESSION / "forming").is_dir(),
                    reason="the 2026-10-02 forming files are not on this machine")
def test_transition_of_the_six_real_forming_takes():
    """The module docstring's numbers, from the forming files themselves
    (the hand, the cue and the end exactly as
    scripts/leap/recompute_transition.py reads them): the three clean
    closes keep their first-rule values, hook take 5 is timed after its
    close instead of in the pause, and the lateral takes, closed before the
    cue, have no transition and say why."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recompute_transition_t", REPO / "scripts" / "leap" / "recompute_transition.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    got = {}
    for p in cli.forming_metas(REAL_SESSION):
        meta = json.loads(p.read_text(encoding="utf-8"))
        if not recorded_on_2026_10_02(meta):
            continue                     # the session has grown since
        got[f"{meta['item']} {meta['take']}"] = cli.forming_numbers(REAL_SESSION, meta)
    assert len(got) == 6
    for key, first in (("p1_cylindrical 4", 1.69), ("p1_cylindrical 5", 2.80),
                       ("p1_hook 4", 2.15)):
        assert got[key]["transition_s"] == pytest.approx(first, abs=0.3), key
        assert got[key]["peak_speed_mm_s"] >= 180.0
    assert got["p1_hook 4"]["transition_note"].startswith("settled, then moved again at 3.")
    assert got["p1_hook 4"]["last_move_s"] > 4.0
    hook5 = got["p1_hook 5"]
    assert hook5["transition_s"] >= 0.9 and hook5["transition_note"] == ""
    assert 0.5 <= hook5["peak_at_s"] <= 1.0 and hook5["last_move_s"] <= 1.0
    for key in ("p1_lateral 4", "p1_lateral 5"):
        assert got[key]["transition_s"] is None and got[key]["last_move_s"] is None
        assert got[key]["peak_speed_mm_s"] < tr.MOVE_SPEED_MM_S
        assert got[key]["transition_note"].startswith("no close after the cue (peak ")



@pytest.mark.skipif(not REAL_SESSION.is_dir(),
                    reason="the 2026-10-01 Set A session is not on this machine")
def test_settle_threshold_is_a_few_times_the_real_static_motion():
    """The numbers in the module docstring: over the static intervals of the
    63 kept takes of 20261001_101621_left the followed hand's median speed
    is 6.4 mm/s, and 20 mm/s is three times that."""
    medians = []
    for mp in sorted((REAL_SESSION / "meta").glob("*.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        # the 63 takes the threshold was read from: the session's kept takes
        # before 2026-10-02, when the forming files (and 6 more takes) came
        if not m.get("accepted") or (m.get("files") or {}).get("forming"):
            continue
        ids = set(m.get("operator_hand_ids") or [])
        a, b = m["static_interval"]
        rows = [json.loads(x) for x in
                (REAL_SESSION / m["files"]["leap"]).read_text(
                    encoding="utf-8").splitlines() if x.strip()]
        rows = [r for r in rows if r.get("hand_side") == "left"
                and (not ids or r.get("hand_id") in ids)
                and a <= r["wall_time"] <= b]
        speeds = tr.joint_speeds(rows)[1]
        medians.append(float(np.nanmedian(speeds)))
    assert len(medians) == 63
    typical = float(np.median(medians))
    assert typical == pytest.approx(6.4, abs=0.1)
    assert 2.5 * typical <= tr.SETTLE_SPEED_MM_S <= 4.0 * typical


# --- from the start of the movement to the settle --------------------------------------
# `transition_timing` on tracks whose movement is known by construction
# (`profile`): the cue is put where the test wants it, so the same track is a
# hand that waited for the cue or a hand that was already moving at it.
def test_timing_of_a_clean_change_start_settle_and_the_times_between():
    rows = profile([(0.5, 1.0, 150.0)])
    g = tr.transition_timing(rows, T0 + 0.2)
    assert g.validity == tr.VALID and g.notes == ""
    # every *_time_s is from the first frame of the recording
    assert g.first_frame_time == pytest.approx(T0)
    assert g.cue_time_s == pytest.approx(0.2, abs=1e-6)
    assert g.movement_start_time_s == pytest.approx(0.5, abs=0.06)
    assert g.settled_time_s == pytest.approx(1.0, abs=0.1)
    assert g.reaction_time_s == pytest.approx(
        g.movement_start_time_s - g.cue_time_s)
    assert g.transition_duration_s == pytest.approx(
        g.settled_time_s - g.movement_start_time_s)
    assert g.transition_duration_s == pytest.approx(0.5, abs=0.12)
    assert g.cue_to_settled_s == pytest.approx(
        g.settled_time_s - g.cue_time_s)
    assert g.peak_speed_mm_s == pytest.approx(150.0, rel=0.15)
    assert g.speed_at_cue_mm_s < tr.POSE_MOVE_SPEED_MM_S
    # nothing carried the hand: over the camera it moved as its pose did
    assert g.hand_speed_at_cue_mm_s < tr.MOVE_SPEED_MM_S
    assert g.hand_peak_speed_mm_s == pytest.approx(150.0, rel=0.2)
    # fifteen joints went 26/15 of 150 mm/s for 0.5 s; the RMS over the 26
    assert g.pose_change_mm == pytest.approx(
        75.0 * (26 / 15) * np.sqrt(15 / 26), rel=0.1)
    assert g.earlier_move_time_s is None and g.moved_again_time_s is None
    assert g.frames == len(rows) and g.hand_ids == [7]
    assert g.end_time_s == pytest.approx(rows[-1]["capture_time"] - T0)


def test_timing_of_a_file_that_begins_at_the_cue():
    """A forming file: its first frame a few milliseconds after the cue."""
    rows = profile([(0.3, 0.8, 150.0)])
    g = tr.transition_timing(rows, T0 - 0.004)
    assert g.validity == tr.VALID
    assert g.cue_time_s == pytest.approx(-0.004, abs=1e-6)
    assert g.reaction_time_s == pytest.approx(0.3, abs=0.07)
    # a hand that changes pose in place: the recorder's own rule, on the
    # speed over the camera, finds the same settle within a frame or two
    assert g.cue_to_settled_s == pytest.approx(
        tr.settle(rows, T0 - 0.004).transition_s, abs=0.12)


def test_timing_a_pose_already_changing_at_the_cue_has_no_start_and_no_duration():
    """A change that began before the cue: its start is not in what came
    after, and nothing is timed from it."""
    rows = profile([(0.0, 0.5, 150.0)])
    g = tr.transition_timing(rows, T0 + 0.2)
    assert g.validity == tr.INVALID
    assert g.notes.startswith("the pose was already changing at the cue (")
    assert g.speed_at_cue_mm_s >= tr.POSE_MOVE_SPEED_MM_S
    assert g.movement_start_time_s is None and g.reaction_time_s is None
    assert g.transition_duration_s is None
    # the cue and the settle were both recorded: that one time is kept
    assert g.cue_to_settled_s == pytest.approx(0.3, abs=0.1)


def test_timing_movement_just_before_the_cue_counts_as_changing_at_it():
    """Frames from before the cue are looked at too, when the file has them."""
    rows = profile([(0.12, 0.19, 150.0), (0.8, 1.3, 150.0)])
    g = tr.transition_timing(rows, T0 + 0.2)
    assert g.validity == tr.INVALID and "already changing at the cue" in g.notes
    # the same hand with the cue a second later was still at it
    g = tr.transition_timing(profile([(0.12, 0.19, 150.0), (1.8, 2.3, 150.0)],
                                     length=3.5), T0 + 1.2)
    assert g.validity == tr.VALID
    assert g.movement_start_time_s == pytest.approx(1.8, abs=0.06)


def test_timing_a_pose_that_never_changes_is_invalid_and_says_so():
    g = tr.transition_timing(profile([]), T0 + 0.2)
    assert g.validity == tr.INVALID
    assert g.notes.startswith("no change of pose after the cue (fastest "
                              "frame ")
    assert "the pose at the end is 0 mm RMS from the pose at the cue" in g.notes
    assert g.movement_start_time_s is None and g.settled_time_s is None
    assert g.transition_duration_s is None and g.cue_to_settled_s is None
    assert g.peak_speed_mm_s < tr.POSE_MOVE_SPEED_MM_S
    assert g.pose_change_mm < 1.0


def test_timing_a_hand_carried_with_its_pose_held_has_not_changed_pose():
    """300 mm/s across the camera for half a second, the pose held: the
    speed over the camera sees a movement, the pose speed none, and nothing
    is timed. This is what the first real take was mostly doing."""
    rows = profile([], carry=[(0.5, 1.0, 300.0)])
    t, over_camera = tr.joint_speeds(rows)
    pose = tr.pose_speeds(t, [np.array(r["abs26"]) for r in rows])
    assert np.nanmax(over_camera) == pytest.approx(300.0, rel=0.1)
    assert np.nanmax(pose) < tr.POSE_SETTLE_SPEED_MM_S
    g = tr.transition_timing(rows, T0 + 0.2)
    assert g.validity == tr.INVALID
    assert g.notes.startswith("no change of pose after the cue")
    assert g.hand_peak_speed_mm_s == pytest.approx(300.0, rel=0.1)
    assert g.pose_change_mm < 1.0
    # the recorder's own rule, on the speed over the camera, calls it a close
    assert tr.settle(rows, T0 + 0.2).transition_s is not None


def test_timing_a_change_made_while_the_hand_is_carried_is_timed_on_the_pose():
    """Carried from before the cue until 0.7 s, the pose changing from 0.9
    to 1.4 s: the start and the length are the pose's, and the note says
    the hand as a whole was moving at the cue."""
    rows = profile([(0.9, 1.4, 150.0)], carry=[(0.05, 0.7, 300.0)])
    g = tr.transition_timing(rows, T0 + 0.2)
    assert g.validity == tr.VALID
    assert g.movement_start_time_s == pytest.approx(0.9, abs=0.06)
    assert g.transition_duration_s == pytest.approx(0.5, abs=0.12)
    assert g.speed_at_cue_mm_s < tr.POSE_MOVE_SPEED_MM_S
    assert g.hand_speed_at_cue_mm_s == pytest.approx(300.0, rel=0.1)
    assert g.notes.startswith("the hand as a whole was moving over the "
                              "camera at the cue (")
    assert "with its pose still" in g.notes
    # the same change with the hand at rest reads the same
    still = tr.transition_timing(profile([(0.9, 1.4, 150.0)]), T0 + 0.2)
    assert still.notes == ""
    assert still.transition_duration_s == pytest.approx(
        g.transition_duration_s, abs=0.03)


def test_timing_a_twitch_shorter_than_the_dwell_is_not_the_start():
    rows = profile([(0.5, 0.54, 200.0), (1.2, 1.7, 150.0)])
    g = tr.transition_timing(rows, T0 + 0.2, start_sustain_s=0.15)
    assert g.validity == tr.VALID and g.earlier_move_time_s is None
    assert g.movement_start_time_s == pytest.approx(1.2, abs=0.06)
    assert g.settled_time_s == pytest.approx(1.7, abs=0.1)
    # the dwell is a setting. With almost none the twitch is a movement of
    # its own, but still not the start: the start is the main movement's,
    # and the take says its pose moved before it
    g = tr.transition_timing(rows, T0 + 0.2, start_sustain_s=0.02)
    assert g.validity == tr.QUESTIONABLE
    assert g.movement_start_time_s == pytest.approx(1.2, abs=0.06)
    assert g.earlier_move_time_s == pytest.approx(0.5, abs=0.06)
    assert g.notes.startswith("the pose moved before its main movement (at ")
    assert g.transition_duration_s == pytest.approx(0.5, abs=0.12)


def test_timing_the_main_movement_is_the_fastest_sustained_one():
    """A slow stir of the pose and its return, a pause, then the change:
    the change is timed, the stir is said. The take the rule was written on
    did this while the hand was carried sideways."""
    rows = profile([(0.4, 0.6, 60.0), (1.2, 1.7, 150.0)], length=3.2)
    g = tr.transition_timing(rows, T0 + 0.1)
    assert g.validity == tr.QUESTIONABLE
    assert g.earlier_move_time_s == pytest.approx(0.4, abs=0.06)
    assert g.movement_start_time_s == pytest.approx(1.2, abs=0.06)
    assert g.peak_time_s == pytest.approx(1.45, abs=0.3)
    assert g.transition_duration_s == pytest.approx(0.5, abs=0.12)
    assert g.notes.startswith("the pose moved before its main movement (at ")
    assert g.notes.endswith("the main movement is timed")
    # the stretches themselves
    t, _over = tr.joint_speeds(rows)
    pose = tr.pose_speeds(t, [np.array(r["abs26"]) for r in rows])
    runs = tr.sustained_runs(t, pose, T0 + 0.1)
    assert len(runs) == 2
    assert [float(t[a] - T0) for a, _b in runs] == pytest.approx([0.4, 1.2],
                                                                 abs=0.06)
    assert tr.movement_start_index(t, pose, T0 + 0.1) == runs[0][0]


def test_timing_a_grip_adjusted_after_the_settle_is_questionable():
    rows = profile([(0.5, 1.0, 150.0), (2.0, 2.3, 120.0)], length=3.2)
    g = tr.transition_timing(rows, T0 + 0.2)
    assert g.validity == tr.QUESTIONABLE
    assert g.transition_duration_s == pytest.approx(0.5, abs=0.12)
    assert g.moved_again_time_s == pytest.approx(2.0, abs=0.1)
    assert g.last_move_time_s == pytest.approx(2.3, abs=0.1)
    assert g.notes.startswith("settled, then moved again at ")
    assert "the first settle is timed" in g.notes


def test_timing_a_hand_that_never_settles_has_a_start_and_no_duration():
    g = tr.transition_timing(profile([(0.5, 3.0, 150.0)]), T0 + 0.2)
    assert g.validity == tr.INVALID and g.notes == tr.NOTE_NEVER_SETTLED
    assert g.movement_start_time_s == pytest.approx(0.5, abs=0.06)
    assert g.settled_time_s is None and g.transition_duration_s is None


def test_timing_thresholds_are_arguments_and_the_methods_name_them():
    rows = profile([(0.5, 1.0, 150.0)])
    # a start speed above the movement: no movement at all
    g = tr.transition_timing(rows, T0 + 0.2, start_speed_mm_s=250.0)
    assert g.validity == tr.INVALID
    assert g.notes.startswith("no change of pose")
    assert ("0.1 s or longer, of pose speed at or above 250 mm/s"
            in g.movement_start_method)
    assert "rigid fit of the palm's 11 joints" in g.movement_start_method
    # a longer dwell for the settle: the settle is the same moment here,
    # and the method says which dwell was used
    g = tr.transition_timing(rows, T0 + 0.2, settle_window_s=0.8,
                             settle_speed_mm_s=30.0)
    assert g.validity == tr.VALID
    assert "below 30 mm/s for 0.8 s" in g.settled_method
    # a dwell longer than the hold that is left: never settled
    g = tr.transition_timing(rows, T0 + 0.2, settle_window_s=5.0)
    assert g.validity == tr.INVALID and g.notes == tr.NOTE_NEVER_SETTLED
    # the defaults are the named constants
    import inspect
    sig = inspect.signature(tr.transition_timing).parameters
    assert sig["start_speed_mm_s"].default == tr.POSE_MOVE_SPEED_MM_S == 25.0
    assert sig["start_sustain_s"].default == tr.START_SUSTAIN_S == 0.1
    assert (sig["settle_speed_mm_s"].default == tr.POSE_SETTLE_SPEED_MM_S
            == 10.0)
    assert sig["settle_window_s"].default == tr.SETTLE_WINDOW_S == 0.3
    assert sig["cue_still_s"].default == tr.CUE_STILL_S == 0.1
    assert tr.start_method() == tr.transition_timing(
        rows, T0 + 0.2).movement_start_method


def test_timing_a_hand_not_seen_at_the_cue_cannot_be_called_still():
    rows = [r for r in profile([(0.8, 1.3, 150.0)])
            if abs(r["capture_time"] - (T0 + 0.3)) > 0.16]
    g = tr.transition_timing(rows, T0 + 0.3)
    assert g.validity == tr.INVALID
    assert g.notes.startswith("no frame of the hand within 0.1 s of the cue")


def test_timing_reads_nothing_after_its_end_and_nothing_with_no_frames():
    rows = profile([(0.5, 1.0, 150.0)])
    g = tr.transition_timing(rows, T0 + 0.2, t_end=T0 + 0.9)
    assert g.validity == tr.INVALID and g.settled_time_s is None
    assert g.end_time_s == pytest.approx(0.9, abs=1e-6)
    assert g.frames == sum(1 for r in rows if r["capture_time"] <= T0 + 0.9)
    for empty in ([], rows[:1]):
        g = tr.transition_timing(empty, T0)
        assert g.validity == tr.INVALID and g.notes == tr.NOTE_NO_FRAMES
    # frames, but none after the cue
    g = tr.transition_timing(rows, T0 + 99.0)
    assert g.validity == tr.INVALID and g.notes == tr.NOTE_NO_FRAMES


def test_movement_start_index_wants_the_whole_dwell_without_a_hole():
    t = T0 + np.arange(90) / HZ
    speed = np.full(90, 5.0)
    speed[30:36] = 120.0                 # 6 frames: 0.056 s
    speed[50:70] = 120.0                 # 20 frames: 0.21 s
    assert tr.movement_start_index(t, speed, T0) == 50
    assert tr.movement_start_index(t, speed, T0, start_sustain_s=0.04) == 30
    assert tr.movement_start_index(t, speed, t[60]) == 60
    assert tr.movement_start_index(t, speed, t[65]) is None
    # a hole inside the run breaks it
    holed = np.delete(t, np.arange(55, 66))
    assert tr.movement_start_index(holed, np.delete(speed, np.arange(55, 66)),
                                   T0) is None


def test_measured_rows_follow_the_id_and_keep_the_operators_label():
    mine = profile([(0.5, 1.0, 150.0)], length=1.5)
    other = [dict(r, hand_id=8, hand_side="right",
                  abs26=[[x + 0.3, y, z] for x, y, z in r["abs26"]])
             for r in mine]
    for r in mine + other:
        r["palm_abs"] = r["abs26"][0]
    both = [r for pair in zip(mine, other) for r in pair]
    got = tr.measured_rows(both, "left", (7, [8], None))
    assert [r["hand_id"] for r in got] == [7] * len(mine)
    # a stretch the tracker fitted as the other hand is left out
    for r in mine[40:60]:
        r["hand_side"] = "right"
    assert len(tr.measured_rows(both, "left", (7, [8], None))) == len(mine) - 20
    # without a hand to follow: the label with the most lines
    assert {r["hand_id"] for r in tr.measured_rows(other, "left", None)} == set()
    assert len(tr.measured_rows(mine, "left", None)) == len(mine) - 20


def test_recorded_timing_reads_the_cue_and_the_hand_from_the_meta():
    rows = profile([(1.5, 2.0, 150.0)], length=3.5)
    for r in rows:
        r["palm_abs"] = r["abs26"][0]
    meta = {"hand": "left", "cue_wall_time": T0 + 1.2, "start_hand_id": 7,
            "other_hand_ids": [], "start_palm": rows[100]["palm_abs"]}
    g = tr.recorded_timing(rows, meta)
    assert g.validity == tr.VALID
    assert g.cue_time_s == pytest.approx(1.2, abs=1e-6)
    assert g.reaction_time_s == pytest.approx(0.3, abs=0.07)
    assert g.transition_duration_s == pytest.approx(0.5, abs=0.12)
    # thresholds pass through
    assert tr.recorded_timing(rows, meta, start_speed_mm_s=400.0
                              ).validity == tr.INVALID
    # no cue in the meta: nothing is guessed
    g = tr.recorded_timing(rows, {"hand": "left", "start_hand_id": 7})
    assert g.validity == tr.INVALID and "no cue time" in g.notes


@pytest.mark.skipif(not (REAL_SESSION / "forming").is_dir(),
                    reason="the 2026-10-02 forming files are not on this machine")
def test_timing_of_the_six_real_forming_takes():
    """What the pose rule reads off them: four hands changed pose after the
    cue (two cleanly, one with a stir of the pose before it, one adjusting
    its grip after), two were already in the grasp. The first rule, on the
    speed over the camera, had passed one and given it 1.50 s; most of that
    was the hand being carried sideways."""
    got, raw = {}, {}
    for mp in sorted((REAL_SESSION / "meta").glob("*.json")):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        forming = (meta.get("files") or {}).get("forming")
        if not forming or not recorded_on_2026_10_02(meta):
            continue
        rows = [json.loads(x) for x in (REAL_SESSION / forming).read_text(
            encoding="utf-8").splitlines() if x.strip()]
        key = f"{meta['item']} {meta['take']}"
        got[key], raw[key] = tr.forming_timing(rows, meta), (rows, meta)
    assert len(got) == 6
    want = {"p1_cylindrical 4": (tr.QUESTIONABLE, 0.76, 0.80),
            "p1_cylindrical 5": (tr.VALID, 0.69, 1.11),
            "p1_hook 4": (tr.QUESTIONABLE, 0.81, 1.04),
            "p1_hook 5": (tr.VALID, 0.57, 0.92)}
    for key, (validity, reaction, duration) in want.items():
        g = got[key]
        assert g.validity == validity, key
        assert g.reaction_time_s == pytest.approx(reaction, abs=0.03), key
        assert g.transition_duration_s == pytest.approx(duration, abs=0.03), key
        assert g.speed_at_cue_mm_s < tr.POSE_MOVE_SPEED_MM_S, key
        assert 25.0 < g.pose_change_mm < 60.0, key
    stirred = got["p1_cylindrical 4"]
    assert stirred.notes.startswith("the pose moved before its main movement")
    assert stirred.earlier_move_time_s == pytest.approx(0.28, abs=0.03)
    # the sideways shift the first rule timed: nearly 400 mm/s over the
    # camera, with a pose that peaked at 85
    assert stirred.hand_peak_speed_mm_s == pytest.approx(398.0, abs=2.0)
    assert stirred.peak_speed_mm_s == pytest.approx(85.0, abs=2.0)
    assert got["p1_hook 4"].notes.startswith("settled, then moved again at ")
    for key in ("p1_cylindrical 5", "p1_hook 4", "p1_hook 5"):
        assert "the hand as a whole was moving over the camera at the cue" \
            in got[key].notes, key
    for key in ("p1_lateral 4", "p1_lateral 5"):
        g = got[key]
        assert g.validity == tr.INVALID
        assert g.notes.startswith("no change of pose after the cue"), key
        assert g.cue_to_settled_s is None and g.pose_change_mm < 10.0
    # the fast part alone, at twice the thresholds: 0.3 to 0.5 s
    for key in want:
        fast = tr.forming_timing(*raw[key], start_speed_mm_s=50.0,
                                 settle_speed_mm_s=20.0)
        assert 0.28 <= fast.transition_duration_s <= 0.48, key


# --- the rule on a movement whose length is known --------------------------------------
def test_eased_move_lines_are_the_movement_they_say():
    lines, cue = tr.eased_move_lines(100.0, move_s=0.5, hz=90.0, before_s=1.0,
                                     react_s=0.3, hold_s=1.5)
    t = np.array([d["capture_time"] for d in lines])
    pos = np.array([d["abs26"] for d in lines]) * 1000.0
    assert pos.shape == (len(lines), 26, 3)
    assert cue - t[0] == pytest.approx(1.0)
    moved = pos - pos[0]
    palm = np.zeros(26, dtype=bool)
    palm[tr.PALM_INDEX] = True
    # the palm's eleven joints never move; the other fifteen go the same
    # 100 mm along one line, and nowhere else
    assert np.allclose(moved[:, palm], 0.0)
    x = moved[:, ~palm, 0]
    assert np.allclose(x, x[:, :1]) and np.allclose(moved[:, ~palm, 1:], 0.0)
    travel = x[:, 0]
    assert travel[0] == 0.0 and travel[-1] == pytest.approx(100.0)
    assert np.all(np.diff(travel) >= 0.0)
    # from the last frame still at the start to the first frame at the end
    start = t[travel <= 1e-9][-1]
    end = t[travel >= 100.0 - 1e-9][0]
    assert start - cue == pytest.approx(0.3, abs=1.5 / 90.0)
    assert end - start == pytest.approx(0.5, abs=1.5 / 90.0)
    # carried as a whole as well: every joint the same way, on another line
    carried, _cue = tr.eased_move_lines(100.0, carry_mm=150.0)
    extra = np.array([d["abs26"] for d in carried]) * 1000.0 - pos
    assert np.allclose(extra[:, :, :2], 0.0)
    assert np.allclose(extra[-1, :, 2], 150.0)


def test_the_rule_reads_a_known_half_second_short_when_the_change_is_small():
    """The numbers the hand-in's README gives for what the rule does to a
    0.50 s change of pose: nearly all of it for a large change, less for a
    small one, nothing at all for a very small one; and the same whether
    or not the hand is carried while it changes."""
    read = {mm: tr.known_move_reading(float(mm)) for mm in (20, 40, 80, 160)}
    assert all(g.validity == tr.VALID for g in read.values())
    got = {mm: g.transition_duration_s for mm, g in read.items()}
    assert got[20] == pytest.approx(0.34, abs=0.012)
    assert got[40] == pytest.approx(0.43, abs=0.012)
    assert got[80] == pytest.approx(0.49, abs=0.012)
    assert got[160] == pytest.approx(0.51, abs=0.012)
    assert got[20] < got[40] < got[80] <= got[160]
    # the size of the change, as the table's pose_change_mm has it: the RMS
    # over the 26 joints of fifteen that went the distance
    for mm, g in read.items():
        assert g.pose_change_mm == pytest.approx(mm * np.sqrt(15 / 26),
                                                 rel=0.02)
    # the start is read late when the change is small: the reaction grows
    assert read[20].reaction_time_s > read[160].reaction_time_s >= 0.3 - 0.012
    small = tr.known_move_reading(10.0)
    assert small.validity == tr.INVALID
    assert small.transition_duration_s is None
    assert "no change of pose after the cue" in small.notes
    # the thresholds are the rule's arguments here too
    fine = tr.known_move_reading(10.0, start_speed_mm_s=8.0,
                                 settle_speed_mm_s=3.0)
    assert fine.validity == tr.VALID and fine.transition_duration_s > 0.2
    # carried 150 mm across the camera while it changes: the same readings
    for mm in (20, 40, 80, 160):
        carried = tr.known_move_reading(float(mm), carry_mm=150.0)
        assert carried.validity == tr.VALID
        assert carried.transition_duration_s == pytest.approx(got[mm],
                                                              abs=0.012)
        assert carried.hand_peak_speed_mm_s > 400.0


def test_laid_on_takes_out_the_whole_hands_movement_and_nothing_else():
    rng = np.random.default_rng(5)
    hand = rng.uniform(-0.08, 0.08, size=(26, 3)) + [0.0, 0.25, 0.0]
    c, s_ = np.cos(0.6), np.sin(0.6)
    turn = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    moved = (turn @ hand.T).T + [0.1, -0.05, 0.2]
    assert np.allclose(tr.laid_on(moved, hand), hand, atol=1e-9)
    assert tr.pose_distance_mm(moved, hand) == pytest.approx(0.0, abs=1e-6)
    # fifteen joints 10 mm out, then the whole hand moved: 10 x sqrt(15/26)
    bent = hand.copy()
    fingers = [k for k in range(26) if k not in tr.PALM_INDEX]
    bent[fingers, 1] += 0.010
    bent = (turn @ bent.T).T + [0.1, -0.05, 0.2]
    assert tr.pose_distance_mm(bent, hand) == pytest.approx(
        10.0 * np.sqrt(15 / 26), abs=1e-6)
    assert len(tr.PALM_JOINTS) == len(tr.PALM_INDEX) == 11


def test_the_joints_the_hand_is_laid_on_can_be_chosen():
    """The palm's eleven are not a rigid body: the WRIST joint shifts with
    the wrist's bend. Laid on the ten without it, a wrist that moves alone
    is one joint of 26 moving; laid on all eleven, the fit follows the wrist
    a little and every joint is left out of place."""
    rng = np.random.default_rng(6)
    hand = rng.uniform(-0.08, 0.08, size=(26, 3)) + [0.0, 0.25, 0.0]
    bent = hand.copy()
    bent[tr.JOINT_INDEX["WRIST"], 1] += 0.011
    no_wrist = [tr.JOINT_INDEX[n] for n in tr.PALM_JOINTS_NO_WRIST]
    assert len(no_wrist) == 10 and tr.JOINT_INDEX["WRIST"] not in no_wrist
    assert tr.pose_distance_mm(bent, hand, no_wrist) == pytest.approx(
        11.0 / np.sqrt(26), abs=1e-6)
    left = np.linalg.norm(tr.laid_on(bent, hand) - hand, axis=1) * 1000.0
    assert left[tr.JOINT_INDEX["WRIST"]] < 11.0 and np.median(left) > 0.3
    t = np.array([0.0, 0.1])
    assert tr.pose_speeds(t, [hand, bent], 0.2, no_wrist)[0] == pytest.approx(
        11.0 / 26 / 0.1, abs=1e-6)
    # the rule reads a change of the fingers the same either way, and its
    # method says how many joints the hand was laid on
    a = tr.known_move_reading(80.0)
    b = tr.known_move_reading(80.0, palm_joints=tr.PALM_JOINTS_NO_WRIST)
    assert a.validity == b.validity == tr.VALID
    assert b.transition_duration_s == pytest.approx(a.transition_duration_s,
                                                    abs=1e-9)
    assert "the palm's 11 joints" in a.movement_start_method
    assert "the palm's 10 joints" in b.movement_start_method
