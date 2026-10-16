"""The glove's joint angles over a take (`cam_hand.angle_series`): one row per
frame with its cue, one row per held cue with its medians.

The hand is `test_protocol_check`'s synthetic glove hand, whose phalanges are
all bent by one known angle per finger, so the angle a row must carry is the
angle the frame was built with.
"""
import csv
import json
import math

import pytest

import test_protocol_check as fx
from cam_hand import angle_series as asr
from cam_hand import protocol_check as pc
from xr_hand.recorder import _frame_to_dict

T0 = 1_790_000_000.0


def line(hand, angles, t, counter=0):
    """One glove line of `hand` with each finger's phalanges bent by its
    angle in `angles`, in degrees (the fixture's hand takes radians)."""
    rad = {f: math.radians(a) for f, a in angles.items()}
    return _frame_to_dict(fx.hand_frame(hand, rad, counter), wall_time=t)


def cue(t, step, label, flexed, hold_s, **more):
    return {"t": t, "kind": "cue", "step": step, "label": label,
            "flexed": flexed, "hold_s": hold_s, **more}


def test_the_24_names_and_the_nine_the_glove_does_not_measure():
    assert len(asr.ANGLES) == 24 and len(set(asr.ANGLES)) == 24
    assert len(asr.MODEL_ONLY) == 9 and len(asr.FLEXIONS) == 15
    assert set(asr.MODEL_ONLY) | set(asr.FLEXIONS) == set(asr.ANGLES)
    # the nine are every CMC and every abduction, and nothing else
    assert set(asr.MODEL_ONLY) == {a for a in asr.ANGLES
                                   if a.endswith("_CMC") or a.endswith("_aa")}
    assert asr.FRAME_COLUMNS[:3] == ["frame_id", "wall_time", "t_s"]
    assert asr.FRAME_COLUMNS[-24:] == asr.ANGLES
    assert asr.CUE_ROW_COLUMNS[-24:] == asr.ANGLES


@pytest.mark.parametrize("hand", ["left", "right"])
def test_a_frame_carries_the_angles_it_was_built_with(hand):
    got = asr.angles_of(line(hand, {"index": 40.0, "ring": 70.0}, T0), hand)
    for name in ("I_MCP_fe", "I_PIP", "I_DIP"):
        assert got[name] == pytest.approx(40.0, abs=1e-6), name
    for name in ("R_MCP_fe", "R_PIP", "R_DIP"):
        assert got[name] == pytest.approx(70.0, abs=1e-6), name
    for name in ("M_MCP_fe", "M_PIP", "L_DIP", "T_IP"):
        assert got[name] == pytest.approx(0.0, abs=1e-6), name


def test_each_frame_gets_the_cue_in_force_and_none_outside_the_take():
    events = [
        {"t": T0, "kind": "take_start"},
        cue(T0 + 1.0, 0, "bend the index", ["index"], 1.0,
            cycle=1, phase="bend"),
        cue(T0 + 2.0, 1, "hold", ["index"], 0.5, cycle=1, phase="hold"),
        {"t": T0 + 2.5, "kind": "take_end"},
        {"t": T0 + 3.0, "kind": "decision", "accepted": True},
    ]
    times = [0.5, 1.0, 1.99, 2.0, 2.49, 2.5, 2.9]
    lines = [line("left", {"index": 10.0 * k}, T0 + t, k)
             for k, t in enumerate(times)]
    rows = asr.frame_rows(lines, events, "left")
    assert [r["cue_step"] for r in rows] == ["", 0, 0, 1, 1, "", ""]
    assert [r["cue_phase"] for r in rows] == [
        "", "bend", "bend", "hold", "hold", "", ""]
    assert rows[1]["flexed"] == "index" and rows[1]["cue_cycle"] == 1
    assert rows[0]["cue_label"] == "" and rows[0]["flexed"] == ""
    # a cue's own instant belongs to that cue, take_end's to no cue
    assert rows[3]["cue_label"] == "hold"
    assert [r["t_s"] for r in rows] == pytest.approx(times)
    assert [r["frame_id"] for r in rows] == list(range(len(times)))
    assert [r["I_PIP"] for r in rows] == pytest.approx(
        [10.0 * k for k in range(len(times))], abs=1e-6)


def test_lines_of_the_other_glove_and_broken_lines_are_left_out(tmp_path):
    path = tmp_path / "take.jsonl"
    good = [line("left", {"index": 30.0}, T0 + 0.1, 1),
            line("right", {"index": 60.0}, T0 + 0.2, 2),
            line("left", {"index": 50.0}, T0 + 0.3, 3)]
    path.write_text("\n".join(json.dumps(d) for d in good)
                    + '\n{"wall_time": 1.0, "joints": [\n', encoding="utf-8")
    lines, bad = asr.read_glove_lines(path, "left")
    assert [d["frame_id"] for d in lines] == [1, 3] and bad == 1
    rows = asr.take_rows(path, [], "left")
    assert [r["I_PIP"] for r in rows] == pytest.approx([30.0, 50.0], abs=1e-6)
    # no events at all: time runs from the first frame, no cue anywhere
    assert rows[0]["t_s"] == 0.0 and {r["cue_step"] for r in rows} == {""}


def test_set_b_has_a_row_for_each_hold_and_rest_over_the_whole_phase():
    events = [{"t": T0, "kind": "take_start"}]
    step = 0
    for c in range(2):
        c0 = T0 + 0.5 + c * 3.0
        for phase, dur, flexed in (("bend", 1.0, ["ring"]),
                                   ("hold", 0.5, ["ring"]),
                                   ("straighten", 1.0, []),
                                   ("rest", 0.5, [])):
            events.append(cue(c0, step, phase, flexed, dur, cycle=c + 1,
                              phase=phase))
            step += 1
            c0 += dur
    # 100 Hz: the ring at 60 through each hold, at 5 through each rest, and
    # somewhere else while it moves
    lines = []
    for k in range(700):
        t = T0 + k * 0.01
        x = (t - T0 - 0.5) % 3.0 if t >= T0 + 0.5 else -1.0
        ring = (60.0 if 1.0 <= x < 1.5 else 5.0 if 2.5 <= x < 3.0 else 33.0)
        lines.append(line("left", {"ring": ring, "pinky": ring / 4}, t, k))
    rows = asr.frame_rows(lines, events, "left")
    held = asr.cue_rows(rows, events, asr.FLEXION, 2.5, 1.5)
    assert [(r["phase"], r["cycle"]) for r in held] == [
        ("hold", 1), ("rest", 1), ("hold", 2), ("rest", 2)]
    assert [r["step"] for r in held] == [1, 3, 5, 7]
    assert [r["R_PIP"] for r in held] == pytest.approx(
        [60.0, 5.0, 60.0, 5.0], abs=1e-6)
    # the neighbour is in the same row: how far it went along
    assert [r["L_PIP"] for r in held] == pytest.approx(
        [15.0, 1.25, 15.0, 1.25], abs=1e-6)
    assert held[0]["flexed"] == "ring" and held[1]["flexed"] == ""
    assert held[0]["cue_t_s"] == pytest.approx(1.5)
    assert held[0]["window_start_s"] == pytest.approx(1.5)
    assert held[0]["window_end_s"] == pytest.approx(2.0)
    assert 49 <= held[0]["frames"] <= 51


def test_set_c_reads_the_checkers_window_of_every_step(tmp_path):
    session = pc.load_session(fx.make_sequence_session(
        tmp_path / "s", [fx.SeqTake()], camera=False))
    take = session.takes[0]
    events = pc.read_events(take.path("events"))
    hold, window = pc.check_window_s(session)
    rows = asr.take_rows(take.path("glove"), events, session.hand)
    held = asr.cue_rows(rows, events, asr.SEQUENCES, hold, window)
    steps = pc.step_windows(pc.load_take(session, take), hold, window)
    assert len(held) == len(steps) == len(fx.SEQ4_STEPS)
    origin = events[0]["t"]
    for r, s in zip(held, steps):
        assert r["step"] == s["step"] and r["label"] == s["label"]
        assert r["window_start_s"] + origin == pytest.approx(s["window"][0])
        assert r["window_end_s"] + origin == pytest.approx(s["window"][1])
        assert r["frames"] > 0
    # the open hand, then thumb + index, then all five
    assert held[0]["I_PIP"] < 5 and held[0]["flexed"] == ""
    assert held[1]["I_PIP"] > 40 and held[1]["M_PIP"] < 5
    assert min(held[5][a] for a in ("I_PIP", "M_PIP", "R_PIP", "L_PIP")) > 40


def test_a_scaled_run_keeps_the_same_share_of_each_hold():
    events = [cue(T0, 0, "open hand", [], 0.625),
              cue(T0 + 0.625, 1, "index", ["index"], 0.625)]
    got = asr.held_windows(events, asr.SEQUENCES, 2.5, 1.5)
    assert got[0]["window"] == pytest.approx((T0 + 0.25, T0 + 0.625))
    assert got[1]["window"] == pytest.approx((T0 + 0.875, T0 + 1.25))


def test_a_window_with_no_frame_keeps_its_row_with_empty_angles():
    events = [cue(T0, 0, "open hand", [], 2.5)]
    rows = asr.frame_rows([line("left", {}, T0 + 0.1)], events, "left")
    held = asr.cue_rows(rows, events, asr.SEQUENCES, 2.5, 1.5)
    assert len(held) == 1 and held[0]["frames"] == 0
    assert all(held[0][a] == "" for a in asr.ANGLES)


def test_the_csv_is_the_rows_to_two_decimals(tmp_path):
    events = [{"t": T0, "kind": "take_start"},
              cue(T0 + 0.5, 0, "ring + little", ["ring", "pinky"], 2.5)]
    lines = [line("left", {"ring": 12.3456}, T0 + 0.25, 7),
             line("left", {"ring": 45.0}, T0 + 0.75, 8)]
    rows = asr.frame_rows(lines, events, "left")
    dest = tmp_path / "take.csv"
    assert asr.write_csv(dest, asr.FRAME_COLUMNS, rows) == 2
    with open(dest, encoding="utf-8", newline="") as f:
        got = list(csv.DictReader(f))
    assert list(got[0]) == asr.FRAME_COLUMNS
    assert got[0]["R_PIP"] == "12.35" and got[0]["cue_step"] == ""
    assert got[0]["t_s"] == "0.2500" and got[0]["frame_id"] == "7"
    assert got[1]["cue_label"] == "ring + little"
    assert got[1]["flexed"] == "ring pinky" and got[1]["R_DIP"] == "45.00"
    assert float(got[1]["wall_time"]) == pytest.approx(T0 + 0.75, abs=1e-5)


def _digit_rows(index, n=600, seed=1):
    """Rows whose index joints are `index(k, rng)` -> (MCP, PIP, DIP)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        mcp, pip, dip = index(k, rng)
        row = {a: 0.0 for a in asr.ANGLES}
        row.update(I_MCP_fe=mcp, I_PIP=pip, I_DIP=dip)
        rows.append(row)
    return rows


def test_three_joints_driven_by_one_value_are_told_from_three_that_are_not():
    # one value v spread over the joints by a fixed mapping, the way a hand
    # model does it: every joint is a function of v, none a plain multiple
    def driven(k, rng):
        v = 90.0 * (0.5 - 0.5 * math.cos(k / 400.0))
        return min(v, 60.0), v, 0.6 * v + 0.002 * v * v

    def free(k, rng):
        return tuple(rng.uniform(0.0, 90.0, 3))

    rows = _digit_rows(driven, n=6000)
    rms, joint, span, counted = asr.one_value(rows, "index")
    assert rms < 0.5 and span > 150 and counted > 5000
    assert joint in asr.DIGIT_FLEXIONS["index"]
    rms_free, _j, _span, _n = asr.one_value(_digit_rows(free, n=6000),
                                           "index")
    assert rms_free > 10.0
    # a digit that did not move is not judged, and neither is no frame
    assert asr.one_value(rows, "ring") is None
    assert asr.one_value([], "index") is None
    assert set(asr.DIGIT_FLEXIONS) == {"thumb", "index", "middle", "ring",
                                       "little"}
    assert {a for cols in asr.DIGIT_FLEXIONS.values() for a in cols} == set(
        asr.FLEXIONS)
