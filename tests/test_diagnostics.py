"""The arithmetic behind the two glove diagnostics, on data whose answer is known.

Two kinds of check, and both are needed:

  synthetic   a sweep built from a KNOWN transfer function with a KNOWN lag
              and a KNOWN saturation point, and a hold with a KNOWN drift.
              If the analysis cannot recover a number that was put there on
              purpose, nothing it says about a real hand means anything.
  recorded    the two real files the tools were written from, when they are
              on this machine. They are the reason the analysis is shaped the
              way it is — one sweep that began with the finger already bent,
              one hold whose glove sat on its open-palm value for five
              seconds — so the numbers they produce are pinned here.
"""
import py_compile
from pathlib import Path

import numpy as np
import pytest

from leap_hand.diagnostics import (
    FINGERS,
    PoseHold,
    StreamLog,
    bending_mask,
    camera_pose_match,
    coverage_gaps,
    drift_lines,
    estimate_lag,
    hold_drift,
    open_reference,
    rail_saturation,
    stream_summary_line,
    transfer_curve,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "leap"
# The scratch folder the two tools were written in. Outside the repo, and not
# every checkout has it, so the recorded-data checks skip rather than fail.
SCRATCH = Path(r"C:\Users\nkim2\OneDrive\Desktop\xr trainer")
SWEEP_CSV = SCRATCH / "dead_zone_right_ring_20260920_200435.csv"
HOLD_CSV = SCRATCH / "hold_drift_right_fist_20260918_190912.csv"


# --- a sweep whose answer is known -------------------------------------------
RAIL = 1.97            # the glove's open-palm constant
SATURATION = 1.45      # camera curl above which the glove stops answering
LAG = 0.12             # seconds the glove trails the camera
CAM_LOW, CAM_HIGH = 0.90, 1.80


def triangle(t, low, high, period):
    """A slow, even sweep: straight down and straight back, over and over."""
    phase = np.mod(np.asarray(t, float), period) / period
    rising = phase < 0.5
    return np.where(rising,
                    low + (high - low) * (phase / 0.5),
                    high - (high - low) * ((phase - 0.5) / 0.5))


def glove_of(cam):
    """A glove that tracks the camera until it hits its rail, then stops.

    Linear below the saturation point and pinned at the rail above it —
    a caricature of the one failure the rail-disagreement override exists
    for, with the numbers chosen rather than measured so the analysis has
    something to be checked against.
    """
    cam = np.asarray(cam, float)
    slope = (RAIL - 0.70) / (SATURATION - CAM_LOW)
    return np.minimum(RAIL, RAIL - slope * (SATURATION - cam))


def synthetic_sweep(seconds=40.0, period=10.0, lag=LAG):
    """(glove t, glove curl, camera t, camera curl) for one known sweep."""
    ct = np.arange(0.0, seconds, 1.0 / 90.0)          # the camera runs at 90 Hz
    gt = np.arange(0.0, seconds, 1.0 / 60.0)          # the glove at 60
    cv = triangle(ct, CAM_LOW, CAM_HIGH, period)
    gv = glove_of(triangle(gt - lag, CAM_LOW, CAM_HIGH, period))
    return gt, gv, ct, cv


def test_a_synthetic_sweep_gives_back_its_lag():
    gt, gv, ct, cv = synthetic_sweep()
    lag = estimate_lag(gt, gv, ct, cv)
    assert lag is not None
    # the search grid is 5 ms, so one step of slack
    assert lag.seconds == pytest.approx(LAG, abs=0.006)
    assert lag.ms == pytest.approx(120.0, abs=6.0)
    assert lag.correlation > 0.9


def test_a_synthetic_sweep_gives_back_its_rail_saturation():
    """The whole point of binning by the camera: no open reference needed."""
    gt, gv, ct, cv = synthetic_sweep()
    lag = estimate_lag(gt, gv, ct, cv)
    cam_at_glove = np.interp(gt - lag.seconds, ct, cv)

    bins = transfer_curve(cam_at_glove, gv)
    found = rail_saturation(bins, RAIL)
    assert found == pytest.approx(SATURATION, abs=0.06)

    # below the saturation point the glove is still answering, and the curve
    # says so: each bin's median is lower than the one above it
    below = [b.glove for b in bins if b.high <= SATURATION]
    assert below == sorted(below)
    assert below[0] < RAIL - 0.5


def test_one_bin_brushing_the_rail_is_not_saturation():
    """A rail is a place the glove STAYS, not one it touches in passing."""
    gt, gv, ct, cv = synthetic_sweep()
    cam_at_glove = np.interp(gt - LAG, ct, cv)
    bins = transfer_curve(cam_at_glove, gv)

    # forge a dip well above the saturation point: the glove leaves the rail
    # again in the top bin, so nothing below it can count as saturated
    from dataclasses import replace
    spiked = list(bins)
    spiked[-1] = replace(spiked[-1], glove=1.10)
    assert rail_saturation(spiked, RAIL) is None


def test_bending_is_read_off_the_camera_not_the_glove():
    """Which way the finger is moving is a fact about the hand."""
    cam = triangle(np.arange(0.0, 10.0, 0.01), CAM_LOW, CAM_HIGH, 10.0)
    down = bending_mask(cam)
    # curl falls as the finger closes, so the first half (rising curl) is
    # straightening and the second half is bending
    assert down[:400].mean() < 0.05
    assert down[600:900].mean() > 0.95


def test_a_sweep_with_a_hole_in_it_names_the_missing_bins():
    """The sweep's own acceptance test: a binned curve hides its own holes."""
    cam = np.concatenate([np.linspace(0.90, 1.20, 400),
                          np.linspace(1.60, 1.80, 400)])     # 1.2-1.6 skipped
    gaps = coverage_gaps(cam, 0.90, 1.80, min_samples=5)
    missing = [(round(a, 2), round(b, 2)) for a, b, _n in gaps]
    assert (1.20, 1.30) in missing
    assert (1.50, 1.60) in missing
    assert (0.90, 1.00) not in missing
    assert all(n < 5 for _a, _b, n in gaps)

    # a sweep that covered the range has nothing to report
    assert coverage_gaps(np.linspace(0.90, 1.80, 2000), 0.90, 1.80) == []


def test_lag_is_not_guessed_from_a_signal_that_did_not_move():
    """Two flat lines correlate perfectly at every shift, and mean nothing."""
    t = np.arange(0.0, 20.0, 0.01)
    flat = np.full_like(t, 1.5)
    moving = triangle(t, CAM_LOW, CAM_HIGH, 8.0)
    assert estimate_lag(t, flat, t, moving) is None
    assert estimate_lag(t, moving, t, flat) is None
    assert estimate_lag(t[:3], moving[:3], t[:3], moving[:3]) is None


# --- a hold whose drift is known ---------------------------------------------
OPEN_CURLS = [1.43, RAIL, 2.07, 1.97, 1.71]     # the glove's open-palm values
STUCK_UNTIL = 6.0                                # it answers the fist this late
SETTLED, RELAXED = 0.70, 0.95                    # and then creeps this far


def synthetic_hold(seconds=60.0, rate=60.0):
    """A glove stuck on its open value, then creeping — the day-2 failure."""
    t = np.arange(0.0, seconds, 1.0 / rate)
    ramp = SETTLED + (RELAXED - SETTLED) * (t - STUCK_UNTIL) / (
        seconds - STUCK_UNTIL)
    index = np.where(t < STUCK_UNTIL, OPEN_CURLS[1], ramp)
    curls = np.tile(np.asarray(OPEN_CURLS, float), (t.size, 1))
    curls[:, 1] = index
    return t, curls


def test_a_hold_with_known_drift_is_reported():
    t, curls = synthetic_hold()
    rows = hold_drift(t, curls, open_curls=OPEN_CURLS)
    index = {r.finger: r for r in rows}["index"]

    # the baseline starts where the glove began answering, not at the beep
    assert index.baseline_at == pytest.approx(STUCK_UNTIL, abs=0.05)
    assert index.stuck_s == pytest.approx(3.0, abs=0.05)
    # medians of the first and last five seconds of the answering part
    assert index.drift == pytest.approx(0.227, abs=0.01)
    assert index.first < index.last                     # it relaxed toward open


def test_without_the_open_reference_the_same_hold_reads_backwards():
    """Why `open_curls` exists, stated as a test rather than as a comment.

    A baseline taken blindly at `skip` lands on the OPEN hand the glove was
    still reporting, so the glove CLOSING comes out as the glove creeping
    open — which is what the 2026-09-18 right-fist file does.
    """
    t, curls = synthetic_hold()
    blind = {r.finger: r for r in hold_drift(t, curls)}["index"]
    assert blind.first == pytest.approx(RAIL, abs=0.01)
    assert blind.drift < -0.9


def test_a_finger_that_never_leaves_its_open_value_reads_no_drift():
    """An open-palm hold is a legitimate measurement, not a missing one."""
    t = np.arange(0.0, 30.0, 1.0 / 60.0)
    curls = np.tile(np.asarray(OPEN_CURLS, float), (t.size, 1))
    rows = hold_drift(t, curls, open_curls=OPEN_CURLS)
    assert all(r.measured for r in rows)
    assert all(abs(r.drift) < 1e-9 for r in rows)
    assert all(r.stuck_s == 0.0 for r in rows)
    assert all(r.baseline_at == 3.0 for r in rows)


def test_a_momentary_blip_does_not_move_the_baseline():
    """The glove climbing back onto its rail passes the tolerance in passing.

    On hold_drift_right_fist_20260918_190912.csv that transient is 70 ms
    long and, taken as the hand, drops the whole baseline window back onto
    the open palm.
    """
    t, curls = synthetic_hold()
    blip = (t >= 3.4) & (t < 3.47)
    curls[blip, 1] = 1.90                    # 0.07 off the rail, for 70 ms
    index = {r.finger: r for r in hold_drift(t, curls,
                                             open_curls=OPEN_CURLS)}["index"]
    assert index.baseline_at == pytest.approx(STUCK_UNTIL, abs=0.05)


def test_the_summary_flags_a_glove_that_moved_while_the_camera_did_not():
    t, curls = synthetic_hold()
    steady = np.tile(np.asarray([1.18, 0.97, 0.86, 0.88, 0.96], float),
                     (t.size, 1))
    lines = drift_lines(hold_drift(t, curls, open_curls=OPEN_CURLS),
                        hold_drift(t, steady, open_curls=list(steady[0])))
    text = "\n".join(lines)
    assert "glove drifts, camera does not" in text
    # ...on the index and on nothing else: the others never moved
    flagged = [line.split()[0] for line in lines if "drifts" in line]
    assert flagged == ["index"]


def test_open_reference_finds_the_glove_s_own_straightest_reading():
    t, curls = synthetic_hold()
    found = open_reference(curls)
    assert found == pytest.approx(OPEN_CURLS, abs=1e-6)


# --- is the camera showing the pose that was asked for? ----------------------
# Medians measured off recordings/sync_day2, both sensors, right hand.
CAM_FIST = [1.17, 0.98, 0.86, 0.88, 0.96]
CAM_OPEN = [1.27, 1.75, 1.83, 1.69, 1.43]


def test_the_camera_confirms_a_fist_and_refuses_an_open_palm():
    """The day-2 failure, as the gate now sees it."""
    ok, why = camera_pose_match("fist", CAM_FIST)
    assert ok and why == ""

    ok, why = camera_pose_match("fist", CAM_OPEN)
    assert not ok
    assert "index extended" in why           # and it says which finger


def test_an_open_palm_is_confirmed_although_the_camera_thumb_is_borderline():
    """The camera's thumb band is 0.045 wide; a real open palm straddles it."""
    ok, why = camera_pose_match("open_palm", CAM_OPEN)
    assert ok, why
    # 1.27 is inside the ambiguous band, so this passes on the OTHER four
    borderline = list(CAM_OPEN)
    borderline[0] = 1.26
    assert camera_pose_match("open_palm", borderline)[0]


def test_two_unclear_fingers_is_a_hand_that_is_not_in_the_pose_yet():
    half = [1.26, 1.36, 1.37, 1.69, 1.43]     # index and middle mid-band
    ok, why = camera_pose_match("open_palm", half)
    assert not ok
    assert "cannot tell about" in why and "index" in why


def test_pinch_is_judged_on_the_thumb_index_gap_and_nothing_else():
    """A fist and a pinch have almost the same curls; the thumb tip differs."""
    assert camera_pose_match("pinch", CAM_FIST, gap=0.22)[0]
    ok, why = camera_pose_match("pinch", CAM_FIST, gap=0.97)
    assert not ok and "0.97" in why
    assert not camera_pose_match("pinch", CAM_FIST, gap=None)[0]


def test_an_unknown_pose_is_refused_rather_than_assumed():
    ok, why = camera_pose_match("jazz_hands", CAM_OPEN)
    assert not ok and "jazz_hands" in why
    assert not camera_pose_match("fist", None)[0]


def test_a_pose_has_to_be_held_without_a_break():
    """One lucky frame is not the pose, and neither is a run with a hole."""
    hold = PoseHold(seconds=1.0)
    assert hold.update(100.0, True) is False
    assert hold.update(100.9, True) is False
    assert hold.update(101.0, True) is True
    # a single miss starts the clock again
    assert hold.update(101.1, False) is False
    assert hold.update(101.2, True) is False
    assert hold.update(102.2, True) is True


# --- the glove's stream, second by second ------------------------------------

def test_stream_log_buckets_by_second_and_counts_counter_jumps():
    log = StreamLog()
    t, counter = 1000.0, 0
    for k in range(180):                      # 3 s at 60 Hz
        if k == 90:
            t += 0.4                          # a 400 ms hole, with the
            counter += 7                      # counter 8 further on: 7 lost
        log.add(t, counter)
        t += 1.0 / 60.0
        counter += 1

    rows = log.rows()
    # 3 s of packets plus the 400 ms hole, so the last second is a short one
    assert [r.second for r in rows] == [0, 1, 2, 3]
    assert sum(r.frames for r in rows) == 180
    assert max(r.max_gap_ms for r in rows) == pytest.approx(417, abs=2)
    assert sum(r.counter_jumps for r in rows) == 1
    assert sum(r.lost_packets for r in rows) == 7

    summary = log.summary()
    assert summary["frames"] == 180
    assert summary["counter_jumps"] == 1
    assert summary["lost_packets"] == 7
    assert summary["gaps_over"] == 1          # the 400 ms hole, over 100 ms
    assert summary["max_gap_ms"] == pytest.approx(417, abs=2)
    # 180 packets over 3.38 s: the hole is charged against the rate
    assert 52 < summary["rate_hz"] < 56

    line = stream_summary_line(summary)
    assert "180 packets" in line and "1 counter jump" in line


def test_a_clean_stream_reports_no_gaps_and_no_jumps():
    log = StreamLog()
    for k in range(120):
        log.add(1000.0 + k / 60.0, k)
    summary = log.summary()
    assert summary["gaps_over"] == 0 and summary["counter_jumps"] == 0
    assert summary["rate_hz"] == pytest.approx(60.0, abs=0.1)
    assert [r.frames for r in log.rows()] == [60, 60]


def test_a_stream_log_with_no_counter_still_measures_the_rate():
    """A sensor with no packet counter is not a sensor with no stream."""
    log = StreamLog()
    for k in range(90):
        log.add(1000.0 + k / 90.0)
    assert log.summary()["counter_jumps"] == 0
    assert log.summary()["rate_hz"] == pytest.approx(90.0, abs=0.1)


# --- the two scripts ---------------------------------------------------------

@pytest.mark.parametrize("name", ["hold_test", "finger_sweep"])
def test_the_diagnostic_scripts_compile_and_parse_their_arguments(name):
    path = SCRIPTS / f"{name}.py"
    assert path.is_file()
    py_compile.compile(str(path), doraise=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["--hand", "right"])
    assert args.hand == "right"
    assert args.out == Path("results") / "diagnostics"
    # both keep the flags the scratch versions had
    assert module.parse_args(["--hand", "left", "--no-view"]).no_view is True


# --- the recorded files the tools were written from --------------------------

def _read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.skipif(not SWEEP_CSV.is_file(),
                    reason=f"{SWEEP_CSV.name} is not on this machine")
def test_the_day2_right_ring_sweep_saturates_between_1_4_and_1_5():
    """The sweep that began with the finger already bent, re-read.

    Its open reference is wrong, which is why the old tool's "N % of full
    bend" number is worthless — and why this analysis does not use one.
    """
    rows = _read_csv(SWEEP_CSV)
    glove = np.array([float(r["glove_curl"]) for r in rows])
    cam = np.array([float(r["camera_curl_at_same_instant"]) for r in rows])

    rail = open_reference(glove.reshape(-1, 1))[0]
    assert rail == pytest.approx(1.971, abs=0.005)

    bins = transfer_curve(cam, glove)
    saturation = rail_saturation(bins, rail)
    assert saturation is not None
    assert 1.4 <= saturation <= 1.5

    # and below it the glove really is still answering
    lower = {round(b.low, 1): b.glove for b in bins}
    assert lower[1.0] < lower[1.2] < lower[1.4]


@pytest.mark.skipif(not HOLD_CSV.is_file(),
                    reason=f"{HOLD_CSV.name} is not on this machine")
def test_the_2026_09_18_right_fist_hold_shows_the_glove_creeping_open():
    """Glove drifts, camera does not — on the file that motivated the tool."""
    rows = _read_csv(HOLD_CSV)
    fingers = list(FINGERS)

    def series(source):
        picked = [r for r in rows if r["source"] == source]
        return ([float(r["t"]) for r in picked],
                [[float(r[f]) for f in fingers] for r in picked])

    gt, gv = series("glove")
    ct, cv = series("camera")
    g_rows = hold_drift(gt, gv, open_curls=open_reference(gv))
    c_rows = hold_drift(ct, cv, open_curls=open_reference(cv))
    drift = {r.finger: r.drift for r in g_rows}
    steady = {r.finger: r.drift for r in c_rows}

    # the glove sat on its open-palm value for ~2.6 s after the 3 s skip,
    # although the camera saw the fist from the first frame
    assert {r.finger: r.stuck_s for r in g_rows}["index"] == pytest.approx(
        2.6, abs=0.2)
    # and then crept open over the minute, on every finger
    assert drift["index"] == pytest.approx(0.29, abs=0.03)
    assert drift["ring"] == pytest.approx(0.17, abs=0.03)
    assert all(v > 0.1 for k, v in drift.items() if k != "thumb")
    # the camera, holding the same hand, did not
    assert all(abs(v) < 0.08 for v in steady.values())
