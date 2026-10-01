"""`leap_hand.tracking_quality`: what a loss is, why it happened, and the report.

The classifier is the evidence for "the camera does not register sometimes",
so every cause is tested with a synthetic series built to produce exactly
that cause and nothing else, the loss definition is pinned at its edge
(a 100 ms hole is not a loss, a 150 ms hole is), and the 60 second test is
run end to end on the scripted mock the way the operator runs it. The
coaching is pinned too: six short moves, the window's phrase, bar, status
and skeleton, a voice that fails silently, and a console that prints one
setup line, one line per move and the report, nothing repeated.
"""
import csv
import gc
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from leap_hand import tracking_quality as tq
from leap_hand.tracking_quality import HandState, Sample

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "leap" / "tracking_quality.py"
HZ = 90.0


def load_script():
    spec = importlib.util.spec_from_file_location("leap_script_tracking_quality", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hand(height_cm=30.0, x_cm=0.0, view=10.0, grab=0.0, hand_id=7, side="left",
         z_cm=0.0):
    return HandState(side=side, hand_id=hand_id,
                     palm=(x_cm / 100.0, height_cm / 100.0, z_cm / 100.0),
                     view_deg=view, grab=grab, pinch=0.0, visible_s=1.0)


def series(n_before=180, gap_s=0.3, n_after=90, state=None, before_state=None,
           framerate=90.0, image_mean=6.0, image_saturated=0.0, status=("streaming",),
           gap_image_mean=None, gap_image_saturated=None, lead_framerate=None,
           lead_status=None, back_id=None, path=None):
    """A hand held for `n_before` frames, gone for `gap_s`, back for `n_after`.

    Empty frames fill the gap, as the live stream delivers them. `path(t)`
    overrides the hand frame by frame (for speed); `before_state` is the
    state of every frame before the gap unless `path` is given.
    """
    before_state = before_state or state or hand()
    out = []
    t = 0.0
    for k in range(n_before):
        h = path(t) if path else before_state
        out.append(Sample(t=t, hands=(h,), framerate=lead_framerate or framerate,
                          image_mean=22.0, image_saturated=image_saturated,
                          status=lead_status or status, frame_id=k))
        t = round(t + 1.0 / HZ, 9)
    last = out[-1].t
    back = last + gap_s
    k = n_before
    t = last + 1.0 / HZ
    while t < back - 1e-9:
        out.append(Sample(t=t, hands=(), framerate=framerate,
                          image_mean=image_mean if gap_image_mean is None else gap_image_mean,
                          image_saturated=(image_saturated if gap_image_saturated is None
                                           else gap_image_saturated),
                          status=status, frame_id=k))
        k += 1
        t += 1.0 / HZ
    after = hand(hand_id=back_id) if back_id is not None else hand()
    for j in range(n_after):
        out.append(Sample(t=back + j / HZ, hands=(after,), framerate=framerate,
                          image_mean=22.0, image_saturated=0.0, status=status,
                          frame_id=k + j))
    return out


def only_loss(samples, **kw):
    losses = tq.find_losses(samples, prefer="left", **kw)
    assert len(losses) == 1, [(l.start, l.duration, l.causes) for l in losses]
    return losses[0]


# --- what a loss is -----------------------------------------------------------------
def test_a_clean_series_has_no_losses():
    samples = series(n_before=270, gap_s=1.0 / HZ, n_after=0)
    report = tq.summarise(samples, prefer="left")
    assert report.losses == []
    assert report.tracked_fraction == pytest.approx(1.0)
    assert tq.verdict(report).startswith("VERDICT: no losses")


def test_a_100_ms_gap_is_not_a_loss_and_a_150_ms_gap_is():
    assert tq.find_losses(series(gap_s=0.100), prefer="left") == []
    loss = only_loss(series(gap_s=0.150))
    assert loss.duration == pytest.approx(0.150, abs=1e-6)
    assert loss.recovered and not loss.new_id


def test_a_new_hand_id_is_a_loss_even_with_no_gap():
    loss = only_loss(series(gap_s=1.0 / HZ, back_id=99))
    assert loss.new_id and not loss.gap
    assert loss.kind == "new hand id with no gap"


def test_a_hand_never_seen_is_not_lost_but_one_that_leaves_is():
    empty = [Sample(t=k / HZ, hands=()) for k in range(90)]
    assert tq.find_losses(empty, t_end=1.0) == []
    report = tq.summarise(empty)
    assert report.tracked_frames == 0
    assert tq.verdict(report).startswith("VERDICT: no hand was tracked")
    # tracked, then gone until the end of the run: a loss that never recovered
    gone = series(gap_s=0.5, n_after=0)
    losses = tq.find_losses(gone, prefer="left", t_end=gone[-1].t)
    assert len(losses) == 1 and not losses[0].recovered


# --- one synthetic series per cause ------------------------------------------------
@pytest.mark.parametrize("kwargs, cause", [
    (dict(state=hand(height_cm=49.0)), tq.TOO_HIGH),
    (dict(state=hand(height_cm=10.0)), tq.TOO_LOW),
    (dict(state=hand(height_cm=30.0, x_cm=56.0)), tq.OFF_CENTRE),
    (dict(state=hand(view=75.0)), tq.PALM_AWAY),
    (dict(state=hand(grab=0.95, view=50.0)), tq.CLOSED_BELOW),
    (dict(lead_framerate=60.0), tq.LOW_FPS),
    (dict(gap_image_mean=120.0), tq.BRIGHT),
    (dict(gap_image_saturated=0.05), tq.BRIGHT),
    (dict(lead_status=("streaming", "smudged")), tq.DEVICE),
    (dict(), tq.UNEXPLAINED),
])
def test_each_cause_is_found_by_a_series_built_for_it(kwargs, cause):
    loss = only_loss(series(**kwargs))
    assert loss.causes == [cause]
    assert loss.fix and loss.fix == tq.fix_for(cause, loss.status)


def test_moving_fast_is_the_palm_speed_just_before_the_loss():
    def path(t):          # still, then 1 m/s sideways over the last 0.15 s
        start = 179 / HZ - 0.15
        return hand(x_cm=100.0 * max(0.0, t - start))
    loss = only_loss(series(path=path))
    assert loss.speed == pytest.approx(1.0, rel=0.1)
    assert loss.causes == [tq.FAST]


def test_the_thresholds_sit_where_the_docstring_says():
    def causes(**kw):
        return tq.causes_for(hand(**kw))
    assert causes(height_cm=45.0) == [tq.UNEXPLAINED]
    assert causes(height_cm=45.5) == [tq.TOO_HIGH]
    assert causes(height_cm=12.0) == [tq.UNEXPLAINED]
    assert causes(height_cm=11.5) == [tq.TOO_LOW]
    # 60 degrees off the axis is the edge: tan(59) and tan(61) at 30 cm
    assert causes(x_cm=30.0 * 1.664) == [tq.UNEXPLAINED]
    assert causes(x_cm=30.0 * 1.804) == [tq.OFF_CENTRE]
    assert causes(view=60.0) == [tq.UNEXPLAINED]
    assert causes(view=61.0) == [tq.PALM_AWAY]
    # a fist facing the lens is not "closed hand from below"
    assert causes(grab=0.95, view=20.0) == [tq.UNEXPLAINED]
    assert causes(grab=0.8, view=50.0) == [tq.UNEXPLAINED]
    assert tq.causes_for(hand(), speed=0.5) == [tq.UNEXPLAINED]
    assert tq.causes_for(hand(), speed=0.51) == [tq.FAST]
    assert tq.causes_for(hand(), min_framerate=80.0) == [tq.UNEXPLAINED]
    assert tq.causes_for(hand(), min_framerate=79.0) == [tq.LOW_FPS]


def test_a_loss_keeps_every_cause_and_the_room_comes_first():
    loss = only_loss(series(state=hand(height_cm=49.0, grab=0.9, view=70.0),
                            lead_status=("streaming", "robust")))
    assert loss.causes == [tq.DEVICE, tq.TOO_HIGH, tq.CLOSED_BELOW, tq.PALM_AWAY]
    assert tq.cause_label(loss.cause, loss.status) == (
        "device status (infrared interference, robust mode)")
    assert "infrared" in loss.fix


def test_device_status_is_decoded_from_the_raw_value():
    assert tq.decode_device_status(0x0) == ()
    assert tq.decode_device_status(0x1) == ("streaming",)
    assert tq.decode_device_status(0x9) == ("streaming", "smudged")
    assert tq.decode_device_status(0x15) == ("streaming", "robust", "low_resource")
    # the failure codes are whole values, not flags; signed or not
    assert tq.decode_device_status(0xE8010003) == ("bad_transport",)
    assert tq.decode_device_status(-0x17FEFFFD) == ("bad_transport",)
    assert tq.decode_device_status(0xE8010001) == ("bad_calibration",)
    assert tq.status_warnings(("streaming",)) == ()


def test_the_ranking_counts_every_cause_and_ties_go_to_the_room():
    a = only_loss(series(state=hand(height_cm=49.0)))
    b = only_loss(series(state=hand(height_cm=49.0, view=70.0)))
    c = only_loss(series(state=hand(view=70.0)))
    ranked = tq.rank_causes([a, b, c])
    assert [(cause, n) for cause, n, _s in ranked] == [(tq.TOO_HIGH, 2), (tq.PALM_AWAY, 2)]


# --- the report -----------------------------------------------------------------------
def two_losses():
    first = series(state=hand(height_cm=49.0), gap_s=0.4)
    second = series(state=hand(height_cm=49.0), gap_s=0.2)
    shift = first[-1].t + 1.0 / HZ
    return first + [Sample(t=s.t + shift, hands=s.hands, framerate=s.framerate,
                           image_mean=s.image_mean, image_saturated=s.image_saturated,
                           status=s.status) for s in second]


def test_the_report_names_each_loss_and_the_fix_to_try_first():
    report = tq.summarise(two_losses(), prefer="left")
    lines = tq.format_report(report, setup="camera only, bare left hand",
                             notes=["Mock run."])
    text = "\n".join(lines)
    assert "Losses (gone longer than 100 ms, or a new hand id): 2" in text
    assert "hand 49 cm up" in text and "why:   too high" in text
    assert "Fix to try first: keep the palm 25 to 35 cm above the module" in text
    assert lines[-1] == ("VERDICT: too high (2 of 2 losses): keep the palm 25 to 35 cm "
                         "above the module")
    assert chr(0x2014) not in text                   # no em dashes
    assert "Mock run." in text


def test_the_report_says_what_it_could_not_measure():
    samples = [Sample(t=s.t, hands=s.hands, framerate=s.framerate)
               for s in series(state=hand(height_cm=49.0))]
    text = "\n".join(tq.format_report(tq.summarise(samples, prefer="left")))
    assert "Image brightness: not measured" in text
    assert "Device status: not reported" in text


def test_the_csv_round_trip_gives_the_same_losses():
    samples = two_losses()
    rows = tq.series_rows(samples, prefer="left")
    assert set(rows[0]) == set(tq.SERIES_COLUMNS)
    back = tq.samples_from_series_rows(rows)
    a = tq.find_losses(samples, prefer="left")
    b = tq.find_losses(back, prefer="left")
    assert [(l.cause, round(l.duration, 3)) for l in a] == [
        (l.cause, round(l.duration, 3)) for l in b]


# --- the live line ----------------------------------------------------------------------
def test_the_live_line_counts_a_loss_once_and_names_its_cause():
    live = tq.LiveLossTracker(prefer="left")
    started = [live.add(s) for s in series(state=hand(height_cm=49.0), gap_s=0.5)]
    assert sum(1 for x in started if x is not None) == 1
    assert live.lost_in_window() == 1
    assert re.fullmatch(r"tracking: 9\d Hz, lost 1 time this minute, last: too high",
                        live.line())


def test_the_live_line_forgets_losses_older_than_a_minute():
    live = tq.LiveLossTracker(prefer="left")
    samples = series(state=hand(height_cm=49.0))
    for s in samples:
        live.add(s)
    assert live.lost_in_window() == 1
    t = samples[-1].t
    while t < 63.0:                         # a minute of steady tracking later
        t += 1.0 / HZ
        live.add(Sample(t=t, hands=(hand(),), framerate=90.0))
    assert live.lost_in_window() == 0 and live.total_losses == 1
    assert live.line().endswith("lost 0 times this minute, last: too high")


def test_the_mock_scenario_acts_out_every_cause_live_and_after():
    script = load_script()
    src = script.MockSource(60.0, "left")
    samples = [script.to_sample(r) for r in src.frames(int(60 * HZ))]
    losses = tq.find_losses(samples, prefer="left")
    # every slot but the hold ends in a loss; the 60 ms blip is not one
    assert len(losses) == len(tq.MockScenario.SLOTS) - 1
    assert {l.cause for l in losses} == set(tq.CAUSES)
    assert [l.cause for l in losses].count(tq.TOO_HIGH) == 2
    live = tq.LiveLossTracker(prefer="left")
    started = [x for x in (live.add(s) for s in samples) if x is not None]
    assert [l.causes for l in started] == [l.causes for l in losses]


# --- recorded takes -----------------------------------------------------------------------
def recorded_rows(tmp_path, drop=range(60, 87), height_mm=490.0, frames=90):
    """A take through the real recorder: a scripted mock hand, lost at 49 cm."""
    from leap_hand.mock import MockLeapStream
    from leap_hand.recorder import LeapRecorder

    def script(i):
        out = {"origin_mm": (0.0, height_mm, 40.0)}
        if i in drop:
            out["drop"] = True
        return out

    mock = MockLeapStream(dropout_every=0, reacquire_every=0, sides=("left",),
                          script=script)
    rec = LeapRecorder(hz=None)
    rec.start(tmp_path / "take.jsonl")
    for _side, lh in mock.generate(frames):
        rec.record(lh)
    rec.stop()
    rows = [json.loads(x) for x in (tmp_path / "take.jsonl").read_text().splitlines()]
    for r in rows:                     # a live writer: a few ms behind the capture
        r["wall_time"] = r["capture_time"] + 0.004
    return rows


def test_take_losses_read_the_recorded_lines(tmp_path):
    rows = recorded_rows(tmp_path)
    t_start = rows[0]["wall_time"]
    t_stop = t_start + 1.0
    got = tq.take_losses(rows, "left", t_start, t_stop)
    assert got.clock == "capture_time"
    assert len(got.losses) == 1
    d = got.to_list()[0]
    assert d["cause"] == tq.TOO_HIGH and d["height_cm"] == pytest.approx(49.0, abs=0.5)
    assert d["duration_s"] == pytest.approx(28 / HZ, abs=0.002)
    assert d["start_s"] == pytest.approx(59 / HZ, abs=0.01)
    assert {"start_s", "duration_s", "kind", "height_cm", "offset_cm", "view_angle_deg",
            "grab_strength", "speed_m_s", "causes", "fix"} <= set(d)


def test_a_file_saved_at_5_per_second_is_not_a_loss_every_line(tmp_path):
    rows = recorded_rows(tmp_path, drop=(), frames=450)
    slow = rows[::18]                                  # 5 lines a second
    got = tq.take_losses(slow, "left")
    assert got.losses == [] and got.gap_s == pytest.approx(2.5 * 18 / HZ)
    holed = [r for r in slow if not 1.1 < r["capture_time"] < 1.9]
    got = tq.take_losses(holed, "left")
    assert len(got.losses) == 1 and got.losses[0].duration == pytest.approx(1.0, abs=0.01)


def test_the_reject_reason_names_the_losses_like_the_example():
    losses = []
    for gap, height in ((0.4, 30.0), (1.4, 49.0), (0.3, 30.0)):
        loss = only_loss(series(state=hand(height_cm=height), gap_s=gap))
        losses.append(loss)
    text = tq.take_reason("tracked 72 % ...", 0.72, 0.9, [], True, losses)
    assert text == ("tracked 72 percent: lost 3 times, longest 1.4 s with the hand at "
                    "49 cm (too high: keep the palm 25 to 35 cm above the module); "
                    "the gate needs 90 percent")
    reacq = tq.take_reason("re-acquired", 0.95, 0.9, [(1.0, 5, 6, 0.4)], True, losses[:1])
    assert reacq.startswith("the hand was lost for 0.40 s inside the static interval "
                            "(id 5 -> 6): lost 1 time, for 0.4 s with the hand at 30 cm")
    assert tq.take_reason("", 1.0, 0.9, [], True, []) == ""
    assert tq.take_reason("no hand was tracked during the take", 0.0, 0.9, [], False,
                          []) == "no hand was tracked during the take"


# --- the coaching: six moves, a beep and a voice, one window -------------------------------
SIX_MOVES = (
    "Hand open, hold still",
    "Slowly up, then back down",
    "Slowly left, then right",
    "Turn the palm away, then back",
    "Slow fist, then open",
    "Three grasp shapes, slowly",
)


def test_six_short_moves_split_the_run_evenly():
    script = load_script()
    assert script.MOVES == SIX_MOVES
    assert all(len(p.split()) <= 6 for p in script.MOVES + (script.GET_READY,))
    assert script.scaled_steps(60.0) == [(10.0 * k, p) for k, p in enumerate(SIX_MOVES)]
    assert [script.move_index(t, 60.0) for t in (0.0, 9.99, 10.0, 35.0, 59.99, 60.0)] == [
        0, 0, 1, 3, 5, 5]
    # --seconds 5: every move is 5/6 s, and every one of them still comes up
    assert {script.move_index(k * 5.0 / 6 + 0.01, 5.0) for k in range(6)} == set(range(6))
    assert script.setup_line("left") == (
        "Camera only, bare left hand. Module flat on the desk, lenses up.")
    assert script.move_line(1) == "Move 2 of 6: Slowly up, then back down"


def test_the_window_shows_the_phrase_the_bar_the_status_and_the_skeleton():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from leap_hand.mock import MockLeapStream
    script = load_script()
    lh = [h for _s, h in MockLeapStream(hz=HZ, dropout_every=0, reacquire_every=0,
                                       sides=("left",)).generate(3)][-1]

    def to_px(p):                       # a plain pinhole, the mock has no LeapC
        x, y, z = p
        return None if y <= 5 else (int(384 - 700 * (x - 32) / y), int(384 + 700 * z / y))

    top, size = script.VIEW_TOP, script.VIEW_SIZE
    picture = np.full((384, 384), 4, np.uint8)
    frame = script.compose_frame(picture, [lh], SIX_MOVES[3], 0.5, "hand seen   tracked 99 %",
                                 to_px, hand_now=True)
    assert frame.shape == (top + size + script.VIEW_BOTTOM, size, 3)
    white = (frame >= 200).all(axis=2)
    assert white[:80].sum() > 2000                          # the phrase, large, at the top
    scale = script.phrase_scale(cv2)
    assert scale >= 1.2
    for p in SIX_MOVES:                                     # and every phrase fits
        assert cv2.getTextSize(p, 0, scale, 3)[0][0] <= size - 48
    amber = tuple(script._AMBER)
    bar_y = 98                                              # the countdown bar, half left
    assert tuple(frame[bar_y, 24 + 180]) == amber
    assert tuple(frame[bar_y, 24 + 540]) != amber
    empty = script.compose_frame(picture, [], SIX_MOVES[3], 0.0, "NO HAND", to_px)
    assert not (empty[84:112] == amber).all(axis=2).any()
    assert (frame[top + size:] > 100).any(axis=2).sum() > 200   # the status line
    cyan = ((frame[..., 0] > 200) & (frame[..., 1] > 170) & (frame[..., 2] < 90))
    no_hand = ((empty[..., 0] > 200) & (empty[..., 1] > 170) & (empty[..., 2] < 90))
    assert cyan[top:top + size].sum() > 200 and no_hand.sum() == 0   # the skeleton


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_the_voice_fails_silently(capfd):
    script = load_script()
    off = script.Voice(["no-such-program-for-the-voice"])
    assert not off.on
    off.say("Hand open, hold still")
    off.close()
    gone = script.Voice([sys.executable, "-c", "import sys; sys.stderr.write('x'); sys.exit(3)"])
    gone._proc.wait(timeout=30)                 # a voice that died at start-up
    gone.say("Hand open, hold still")
    gone.say("Slowly up, then back down")
    assert not gone.on
    gone.close()
    del gone
    gc.collect()                                # nothing left to complain at exit
    assert script.Voice(None).on is False
    out, err = capfd.readouterr()
    assert out == "" and err == ""


@pytest.mark.skipif(shutil.which("powershell") is None, reason="no Windows PowerShell")
def test_the_voice_script_runs_in_powershell():
    """The real PowerShell loop, with the sound sent nowhere."""
    script = load_script()
    loop = script.VOICE_LOOP.replace("$voice.Speak($line)",
                                     "$voice.SetOutputToNull(); $voice.Speak($line)")
    assert loop != script.VOICE_LOOP
    voice = script.Voice(list(script.VOICE_COMMAND[:-1]) + [loop])
    if not voice.on:
        pytest.skip("PowerShell would not start")
    proc = voice._proc
    for phrase in SIX_MOVES:
        voice.say(phrase)
    voice.close()
    assert proc.wait(timeout=60) == 0


def test_the_doc_lists_the_six_moves():
    text = (REPO / "docs" / "tracking_quality.md").read_text(encoding="utf-8")
    for k, phrase in enumerate(SIX_MOVES, 1):
        assert re.search(rf"^{k}\. {re.escape(phrase)}\b", text, re.M), phrase
    assert chr(0x2014) not in text


# --- the 60 second test, end to end on the mock ---------------------------------------------
def test_the_mock_run_end_to_end(tmp_path):
    out = tmp_path / "diag"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--mock", "--seconds", "5", "--no-view",
         "--no-voice", "--no-beep", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    printed = [x for x in done.stdout.splitlines() if x.strip()]
    assert printed[-1].startswith("VERDICT: too high (2 of 7 losses)"), printed[-5:]
    csvs = list(out.glob("tracking_quality_*.csv"))
    txts = list(out.glob("tracking_quality_*.txt"))
    assert len(csvs) == 1 and len(txts) == 1
    report = txts[0].read_text(encoding="utf-8").splitlines()
    assert report[-1] == printed[-1]
    assert any(x.startswith("Mock run:") for x in report)
    # The console: one setup line, one line per move, then the report. No
    # plan, no status line repeated many times a second (universal newlines
    # turn a "\r" rewrite into lines of its own, so it would show up here).
    before = printed[:next(i for i, x in enumerate(printed)
                           if x.startswith("Per-frame series:"))]
    assert before == ["Camera only, bare hand. Module flat on the desk, lenses up."] + [
        f"Move {k} of 6: {p}" for k, p in enumerate(SIX_MOVES, 1)]
    for phrase in SIX_MOVES:
        assert done.stdout.count(phrase) == 1, phrase
    lines = done.stdout.splitlines()
    assert len(lines) == len(before) + 4 + len(report), lines[:20]
    assert "s left" not in done.stdout
    with open(csvs[0], encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == pytest.approx(5 * HZ, abs=10)
    assert list(rows[0]) == list(tq.SERIES_COLUMNS)
    assert {r["tracked"] for r in rows} == {"0", "1"}

    again = subprocess.run([sys.executable, str(SCRIPT), "--from", str(csvs[0])],
                           cwd=REPO, capture_output=True, text=True, timeout=120)
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stdout.strip().splitlines()[-1] == printed[-1]
