"""Live fusion: the camera buffer, the warm-up, and the whole command on mocks.

The fusion itself is `fuse_skeletons` and friends, tested in test_fusion.py
and test_drift_anchor.py. What is new live, and tested here, is everything
around it: pairing a lagged glove frame against a short camera buffer,
deciding hand-id stability as frames arrive (and deciding it the way the
offline `flag_hand_id_stability` does), the ACQUIRE gate in front of each
hand's warm-up, learning rails and endpoints from that warm-up instead of a
session, and `scripts/fuse_live.py` end to end.
"""
import importlib.util
import json
import socket
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cam_hand.fusion import (
    GATED_DOFS,
    SENSOR_CAMERA,
    SENSOR_GLOVE,
    RailOverrideParams,
    flag_hand_id_stability,
)
from cam_hand.live_fusion import (
    FIT_NONE,
    MIN_OPEN_CAMERA_FRAMES,
    PHASE_FIST,
    PHASE_OPEN,
    STAGE_ACQUIRE,
    STAGE_COUNTDOWN,
    STAGE_NOT_ACQUIRED,
    AcquireStatus,
    CameraBuffer,
    Warmup,
    acquire_status,
    countdown_text,
)
from leap_hand.protocol import HandReading

ROOT = Path(__file__).resolve().parents[1]
HZ = 90.0


def _row(t, hand="right", hand_id=1):
    return {"t": t, "hand_side": hand, "hand_id": hand_id,
            "pts": [[0.0, 0.0, 0.0]] * 21, "visible_time_us": 1_000_000,
            "palm_abs": [0.0, 0.22, 0.0], "palm_normal_abs": [0.0, -1.0, 0.0],
            "abs26": [[0.0, 0.0, 0.0]] * 26}


# --- the camera buffer ------------------------------------------------------

def test_buffer_pairs_a_lagged_glove_frame_with_the_nearest_camera_frame():
    buf = CameraBuffer()
    t0 = 1000.0
    for k in range(200):                       # 2.2 s of a 90 Hz camera
        buf.add(_row(t0 + k / HZ))
    newest = t0 + 199 / HZ

    # The right glove trails by 0.47 s: its frame is paired with the camera
    # frame from 0.47 s earlier, not with the newest one.
    t_glove, lag = t0 + 2.0, 0.47
    got = buf.nearest("right", t_glove - lag, max_dt=0.05)
    assert got is not None
    assert abs(got["t"] - (t_glove - lag)) <= 0.5 / HZ + 1e-9
    unlagged = buf.nearest("right", t_glove, max_dt=0.05)
    assert unlagged["t"] - got["t"] == pytest.approx(lag, abs=1.0 / HZ)

    # max_dt is a hard limit either side of the buffer.
    assert buf.nearest("right", newest + 0.04, max_dt=0.05)["t"] == newest
    assert buf.nearest("right", newest + 0.06, max_dt=0.05) is None
    assert buf.nearest("right", t0 - 0.06, max_dt=0.05) is None
    # A hand the camera never reported pairs with nothing.
    assert buf.nearest("left", t_glove - lag, max_dt=0.05) is None


def test_buffer_forgets_old_rows_and_drops_abs26_after_the_warmup():
    buf = CameraBuffer(keep_s=3.0, keep_abs26=True)
    kept = buf.add(_row(10.0))
    assert kept["abs26"] is not None           # the warm-up measures bones
    buf.keep_abs26 = False
    assert buf.add(_row(10.5))["abs26"] is None
    buf.add(_row(14.0))                         # 10.0 and 10.5 now > 3 s old
    assert buf.nearest("right", 10.0, max_dt=0.6) is None
    assert buf.latest_t("right") == 14.0
    assert len(buf) == 1


def test_hand_id_stable_is_decided_as_frames_arrive_as_offline():
    """False for `hand_id_settle_s` (0.25 s) after the id changes, per hand,
    and the same answer `flag_hand_id_stability` gives over the whole take."""
    buf = CameraBuffer()
    frames = [(0.00, "right", 1), (0.50, "right", 1), (1.00, "right", 2),
              (1.10, "left", 7), (1.20, "right", 2), (1.24, "right", 2),
              (1.25, "right", 2), (1.30, "right", 2), (1.40, "left", 8),
              (1.60, "left", 8), (1.70, "left", 8)]
    live = [buf.add(_row(t, hand, hid))["hand_id_stable"]
            for t, hand, hid in frames]
    assert live == [True, True, False, True, False, False, True, True,
                    False, False, True]

    offline = [{"capture_time": t, "wall_time": t, "hand_side": hand,
                "hand_id": hid} for t, hand, hid in frames]
    flag_hand_id_stability(offline)
    assert live == [r["hand_id_stable"] for r in offline]


# --- the warm-up --------------------------------------------------------------

# Glove curls, thumb to pinky. An open glove reads the SAME bits every frame
# (the sensor is against its stop), which is what makes it a rail.
OPEN_GLOVE = [1.10, 1.97, 2.05, 1.90, 1.70]
# Camera open curls clear `cam_open_curl - margin` on every finger, so they
# count as open-palm-like frames for the camera's open reference.
OPEN_CAMERA = [1.40, 1.80, 1.85, 1.72, 1.50]


def _fist(k, base=0.70, step=0.005):
    return [base + step * k + 0.01 * i for i in range(5)]


def _warmup(right_camera_open=30, right_fist_base=0.70):
    w = Warmup(("left", "right"), open_s=0.1, fist_s=0.1)
    w.begin(PHASE_OPEN)
    for _ in range(60):
        w.add_glove("right", OPEN_GLOVE)
    for _ in range(right_camera_open):
        w.add_camera({"hand_side": "right"}, curls=OPEN_CAMERA)
    w.begin(PHASE_FIST)
    for k in range(60):
        w.add_glove("right", _fist(k, base=right_fist_base))
        # The left glove never held a straight hand: every reading differs by
        # more than the rail tolerance, so no value can be a rail.
        w.add_glove("left", _fist(k, base=0.50, step=0.01))
    for k in range(30):
        w.add_camera({"hand_side": "right"}, curls=_fist(k, base=0.90))
        w.add_camera({"hand_side": "left"}, curls=_fist(k, base=0.90))
    w.begin(None)
    return w


def test_warmup_learns_rails_and_endpoints_from_open_then_fist():
    w = _warmup()
    result = w.learn(rail_params=RailOverrideParams(fingers=("index",)),
                     fit=FIT_NONE)

    assert "right" not in result.refused
    for finger, value in zip(("thumb", "index", "middle", "ring", "pinky"),
                             OPEN_GLOVE):
        assert result.gate_rails[("right", finger)] == pytest.approx(value)
    # The override's rails are the gates' rails when the override is on.
    assert result.rails[("right", "index")] == pytest.approx(1.97)
    assert result.override_enabled["right"] == ("index",)

    hs = result.scale.for_hand("right")
    glove = hs.endpoints(SENSOR_GLOVE, "index")
    camera = hs.endpoints(SENSOR_CAMERA, "index")
    assert glove.open == pytest.approx(1.97)
    assert glove.flexed < 1.0                   # the fist's end
    assert camera.open == pytest.approx(1.80)   # median of the open frames
    assert camera.flexed < 1.0
    assert hs.normalisable("index")
    assert result.measurements == {} and result.curl_gates == {}
    text = "\n".join(result.lines())
    assert "index 1.970" in text


def test_warmup_refuses_a_hand_never_seen_open():
    """Every cause is named, each with what the operator can do about it."""
    result = _warmup().learn(fit=FIT_NONE)
    assert set(result.refused) == {"left"}
    why = result.refused["left"]
    assert why.startswith("LEFT hand refused: ")
    assert result.causes["left"] == [
        "the LEFT glove never read the index finger straight and still (no "
        "rail in 60 frames): keep the fingers flat and still for the whole "
        "OPEN PALM phase",
        "the camera never saw the LEFT hand during the open palm: hold it 18 "
        "to 28 cm above the module, palm to the lens"]
    # With the override off there are no override rails, but the gates still
    # know where a straight finger reads.
    assert result.rails == {}
    assert ("right", "index") in result.gate_rails
    text = "\n".join(result.lines())
    assert "  LEFT hand refused:\n    - the LEFT glove never read" in text


def test_warmup_refuses_a_hand_the_camera_barely_saw_open():
    result = _warmup(right_camera_open=5).learn(fit=FIT_NONE)
    assert result.causes["right"] == [
        "the camera saw the RIGHT hand on only 5 frames of the open palm "
        f"(need {MIN_OPEN_CAMERA_FRAMES}): hold it 18 to 28 cm above the "
        "module, palm to the lens, for the whole phase"]


def test_warmup_refuses_a_fist_that_did_not_close_the_hand():
    """Open seen by both sensors, but the glove's index only got from 1.97
    to about 1.62: a span under `min_glove_span` (0.40), which would make
    every flexion fraction noise."""
    result = _warmup(right_fist_base=1.60).learn(fit=FIT_NONE)
    glove_span, camera_span = result.index_spans("right")
    assert glove_span < 0.40 <= camera_span
    assert result.causes["right"] == [
        f"the RIGHT glove did not register the fist (index span "
        f"{glove_span:.3f}, need 0.40): close a full fist during the FIST "
        "phase, and check the glove is calibrated in XR Trainer"]
    # The same warm-up with a real fist is not refused.
    assert "right" not in _warmup().learn(fit=FIT_NONE).refused


# --- the ACQUIRE gate, and one hand after the other ---------------------------

def _reading(hand="left", visible_s=1.0, height_cm=23.0):
    return HandReading(hand_side=hand, hand_id=1,
                       visible_time_us=int(visible_s * 1e6),
                       height_cm=height_cm, lateral_cm=2.0,
                       view_angle_deg=10.0)


def test_acquire_status_names_what_is_still_missing():
    nothing = acquire_status("left", 0, 0, None, False)
    assert not nothing.ok
    assert nothing.line() == (
        "ACQUIRE LEFT  glove: 0 LEFT packets in the last second, need 10 (is "
        "XR Trainer streaming?)  camera: no LEFT hand (raise it to 18 to 28 cm "
        "above the module)")
    other = acquire_status("left", 60, 60, None, True)
    assert other.line(12.3) == ("ACQUIRE LEFT  glove ok  camera: RIGHT hand "
                                "seen, need LEFT  (12 s left)")
    assert other.caption_text == "RIGHT hand seen, need LEFT"
    assert other.refusal(60) == ("the LEFT hand was not acquired in 60 s: "
                                 "camera: RIGHT hand seen, need LEFT")
    only_right = acquire_status("left", 0, 60, _reading(), False)
    assert only_right.camera_ok and not only_right.glove_ok
    assert only_right.glove_text.startswith("glove: no LEFT packets, only "
                                            "RIGHT")
    young_low = acquire_status("left", 60, 0, _reading(visible_s=0.2,
                                                       height_cm=12.4), False)
    assert young_low.camera_text == (
        "camera: LEFT hand seen, hold it still (0.2 of 0.5 s tracked), raise "
        "it to 18 to 28 cm (now 12 cm)")
    assert acquire_status("left", 10, 0, _reading(), False).ok
    assert acquire_status("left", 9, 0, _reading(), False).glove_ok is False
    assert countdown_text("left", 2.0) == "LEFT hand acquired: open palm in 2"
    assert countdown_text("left", 0.6) == "LEFT hand acquired: open palm in 1"


def test_acquire_gate_passes_at_once_on_the_mock_sensors():
    """The mock hand is open, in the band and facing the lens from its first
    frame, so the gate waits only for the half second of tracking."""
    from cam_hand.live_fusion import CameraSource, GloveSource

    camera = CameraSource(("right",), mock=True).start()
    glove = GloveSource(("right",), mock=True).start()
    seen = []

    def poll(_w):
        camera.drain()
        glove.drain()

    def acquire(hand, now):
        seen.append(acquire_status(hand, glove.recent_packets(hand, now),
                                   glove.recent_packets("left", now),
                                   camera.reading(hand, now),
                                   camera.seen_recently("left", now)))
        return seen[-1]

    try:
        camera.coach("open_palm")
        w = Warmup(("right",), open_s=0.0, fist_s=0.0, acquire_s=3.0,
                   countdown_s=0.0)
        t0 = time.time()
        w.run(poll, lambda *a: None, acquire=acquire)
        took = time.time() - t0
    finally:
        camera.stop()
        glove.stop()
    assert not w.not_acquired
    assert seen[-1].ok and seen[-1].camera_text == "camera ok"
    assert took < 1.5


def _clocked():
    """A clock that advances 50 ms per sleep, so a warm-up runs instantly."""
    now = [0.0]

    def sleep(_dt):
        now[0] += 0.05
    return (lambda: now[0]), sleep


def _status(hand, ok):
    return AcquireStatus(hand=hand, glove_ok=True, glove_text="glove ok",
                         camera_ok=ok,
                         camera_text="camera ok" if ok else
                         f"camera: no {hand.upper()} hand",
                         caption_text="")


def test_warmup_runs_each_hand_in_turn_and_learns_from_its_own_phases():
    clock, sleep = _clocked()
    cues = []
    calls = Counter()

    def poll(w):
        # Both gloves stream the whole time. Each frame is tagged with the
        # stage and the hand whose warm-up was running when it arrived.
        for hand in ("left", "right"):
            w.add_glove(hand, OPEN_GLOVE, frame=(w.phase, w.active))

    def acquire(hand, _now):
        calls[hand] += 1
        return _status(hand, calls[hand] > 3)

    w = Warmup(("left", "right"), open_s=0.2, fist_s=0.2, acquire_s=5.0,
               countdown_s=0.1)
    w.run(poll, lambda stage, hand, s: cues.append((stage, hand)),
          acquire=acquire, clock=clock, sleep=sleep)
    assert cues == [(STAGE_ACQUIRE, "left"), (STAGE_COUNTDOWN, "left"),
                    (PHASE_OPEN, "left"), (PHASE_FIST, "left"),
                    (STAGE_ACQUIRE, "right"), (STAGE_COUNTDOWN, "right"),
                    (PHASE_OPEN, "right"), (PHASE_FIST, "right")]
    # Nothing from the gate or the countdown, nothing of the other hand.
    assert {h for h, _c, _tag in w.glove} == {"left", "right"}
    for hand, _curls, (phase, active) in w.glove:
        assert hand == active and phase in (PHASE_OPEN, PHASE_FIST)


def test_a_hand_the_gate_times_out_on_is_refused_and_the_next_one_goes_on():
    clock, sleep = _clocked()
    cues = []
    w = Warmup(("left", "right"), open_s=0.2, fist_s=0.2, acquire_s=0.3,
               countdown_s=0.1)
    w.run(lambda _w: None, lambda stage, hand, s: cues.append((stage, hand)),
          acquire=lambda hand, _now: _status(hand, hand == "right"),
          clock=clock, sleep=sleep)
    assert cues[:2] == [(STAGE_ACQUIRE, "left"), (STAGE_NOT_ACQUIRED, "left")]
    assert cues[2:] == [(STAGE_ACQUIRE, "right"), (STAGE_COUNTDOWN, "right"),
                        (PHASE_OPEN, "right"), (PHASE_FIST, "right")]
    result = w.learn(fit=FIT_NONE)
    assert result.not_acquired == ("left",)
    assert result.refused["left"] == ("LEFT hand refused: the LEFT hand was "
                                      "not acquired in 0.3 s: camera: no LEFT "
                                      "hand")
    assert "  left: not acquired, so no warm-up" in result.lines()


# --- the command, end to end on the mocks ---------------------------------------

def _live_module():
    """scripts/fuse_live.py, which is not on a package path."""
    path = ROOT / "scripts" / "fuse_live.py"
    spec = importlib.util.spec_from_file_location("fuse_live_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quick(live, monkeypatch):
    """No beeps and a short countdown: a mock session is not a rehearsal."""
    monkeypatch.setattr(live, "beep", lambda *a, **k: None)
    monkeypatch.setattr(live, "COUNTDOWN_S", 0.1)


def _low_glove_floor(live, monkeypatch):
    """The UNFITTED mock glove's index spans only 0.36 of a palm length at
    its fist, under the 0.40 a real glove clears by far (template open
    1.71-2.07, fist 0.63-0.80). A plumbing test lowers that one floor on the
    script's own gates; the production default is untouched."""
    from dataclasses import replace

    from cam_hand.fusion import DEFAULT_GATES
    monkeypatch.setattr(live, "DEFAULT_GATES",
                        replace(DEFAULT_GATES, min_glove_span=0.30))


def _camera_without(live, monkeypatch, hand):
    """The mock camera, minus every frame of `hand`."""
    real = live.CameraSource

    class Without(real):
        def start(self):
            super().start()
            drain = self.stream.drain
            self.stream.drain = lambda n=16: [(s, lh) for s, lh in drain(n)
                                              if s != hand]
            return self

    monkeypatch.setattr(live, "CameraSource", Without)


def _osc_address_and_args(datagram):
    from pythonosc.osc_message import OscMessage
    msg = OscMessage(datagram)
    return msg.address, list(msg.params)


def test_mock_session_fuses_to_jsonl_and_osc(tmp_path, monkeypatch):
    live = _live_module()
    _quick(live, monkeypatch)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    out = tmp_path / "live.jsonl"
    try:
        # The mock glove starts open on the OPEN PALM cue and reaches its
        # fist about 2 s later; the fitted mock below passes the glove's
        # span floor as it stands, this unfitted one needs it lowered.
        _low_glove_floor(live, monkeypatch)
        code = live.main(["--mock-glove", "--mock-leap", "--no-view",
                          "--seconds", "2", "--hand", "right",
                          "--out", str(out), "--fit-template", "none",
                          "--acquire-timeout", "5",
                          "--warmup-open", "1", "--warmup-fist", "1.2",
                          "--osc-out", f"127.0.0.1:{port}"])
        assert code == 0

        frames = [json.loads(line) for line in
                  out.read_text(encoding="utf-8").splitlines()]
        assert len(frames) > 30                 # 2 s of a 60 Hz glove
        for d in frames:
            assert d["hand"] == "right"
            assert len(d["pts"]) == 21
            assert all(len(p) == 3 for p in d["pts"])
            assert set(d["dof_source"]) == set(GATED_DOFS)
        assert any(d["t_cam"] is not None for d in frames)

        sock.settimeout(2.0)
        got = {}
        for _ in range(50):
            address, args = _osc_address_and_args(sock.recv(65536))
            got.setdefault(address, args)
            if len(got) == 2:
                break
    finally:
        sock.close()
    keypoints = got["/fused/right/keypoints21"]
    assert len(keypoints) == 63
    assert all(isinstance(v, float) for v in keypoints)
    sources = got["/fused/right/sources"]
    assert [s.split("=", 1)[0] for s in sources] == list(GATED_DOFS)


def test_mock_session_with_no_open_palm_stops_with_exit_code_2(tmp_path,
                                                                monkeypatch,
                                                                capsys):
    live = _live_module()
    _quick(live, monkeypatch)
    out = tmp_path / "never.jsonl"
    code = live.main(["--mock-glove", "--mock-leap", "--no-view",
                      "--seconds", "1", "--hand", "right",
                      "--out", str(out), "--fit-template", "none",
                      "--acquire-timeout", "5",
                      "--warmup-open", "0", "--warmup-fist", "0.3"])
    assert code == 2
    text = capsys.readouterr().out
    assert ("the camera never saw the RIGHT hand during the open palm: hold "
            "it 18 to 28 cm above the module, palm to the lens") in text
    assert "No hand passed the warm-up" in text
    assert not out.exists()                     # nothing was fused


def test_mock_session_that_never_closes_the_fist_is_refused(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """One second of the mock glove is a finger that barely moved."""
    live = _live_module()
    _quick(live, monkeypatch)
    code = live.main(["--mock-glove", "--mock-leap", "--no-view",
                      "--seconds", "1", "--hand", "right",
                      "--fit-template", "none",
                      "--acquire-timeout", "5",
                      "--warmup-open", "0.5", "--warmup-fist", "0.5"])
    assert code == 2
    assert ("the RIGHT glove did not register the fist (index span "
            in capsys.readouterr().out)


def test_mock_session_saves_its_bone_measurement_beside_the_output(
        tmp_path, monkeypatch, capsys):
    """--fit-template auto with --out leaves template_<hand>.json, in the
    offline format, for a later --fit-template PATH. The drift anchor is on
    here so the live path through it runs at least once end to end."""
    from cam_hand.template_fit import load_measurement

    live = _live_module()
    _quick(live, monkeypatch)
    out = tmp_path / "session" / "live.jsonl"
    code = live.main(["--mock-glove", "--mock-leap", "--no-view",
                      "--seconds", "0.5", "--hand", "right",
                      "--out", str(out), "--fit-template", "auto",
                      "--drift-anchor", "on",
                      "--acquire-timeout", "5",
                      "--warmup-open", "3", "--warmup-fist", "1.2"])
    assert code == 0
    saved = out.parent / "template_right.json"
    assert saved.is_file(), capsys.readouterr().out
    m = load_measurement(saved)
    assert m.hand == "right" and m.from_open and m.n_frames >= 200
    assert out.read_text(encoding="utf-8").strip()


def test_mock_session_times_out_when_the_camera_never_sees_the_hand(
        tmp_path, monkeypatch, capsys):
    """The camera reports only the RIGHT hand: the LEFT hand's gate says so
    on the HUD, times out, and with no hand left the program exits 2."""
    live = _live_module()
    _quick(live, monkeypatch)
    _camera_without(live, monkeypatch, "left")
    out = tmp_path / "never.jsonl"
    code = live.main(["--mock-glove", "--mock-leap", "--no-view",
                      "--seconds", "1", "--hand", "left",
                      "--out", str(out), "--fit-template", "none",
                      "--acquire-timeout", "0.5"])
    text = capsys.readouterr().out
    assert code == 2, text
    assert "ACQUIRE LEFT  " in text
    assert "camera: RIGHT hand seen, need LEFT" in text
    assert ("LEFT hand refused: the LEFT hand was not acquired in 0.5 s: "
            "camera: RIGHT hand seen, need LEFT") in text
    assert "LEFT hand: OPEN PALM" not in text   # no phase ever started
    assert not out.exists()


def test_mock_session_goes_on_with_the_other_hand_after_a_timeout(
        tmp_path, monkeypatch, capsys):
    live = _live_module()
    _quick(live, monkeypatch)
    _low_glove_floor(live, monkeypatch)
    _camera_without(live, monkeypatch, "left")
    out = tmp_path / "right_only.jsonl"
    code = live.main(["--mock-glove", "--mock-leap", "--no-view",
                      "--seconds", "0.5", "--hand", "both",
                      "--out", str(out), "--fit-template", "none",
                      "--acquire-timeout", "0.5", "--warmup-open", "1",
                      "--warmup-fist", "1.2"])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "Going on to the RIGHT hand." in text
    assert ("Fusing the RIGHT hand only: the LEFT hand was refused (above)."
            in text)
    hands = {json.loads(line)["hand"]
             for line in out.read_text(encoding="utf-8").splitlines()}
    assert hands == {"right"}


def test_mock_session_warms_up_the_left_hand_then_the_right(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    """--hand both: acquire, countdown, open, fist for LEFT, then the same
    for RIGHT, in the console and in the camera window's caption."""
    live = _live_module()
    _quick(live, monkeypatch)
    _low_glove_floor(live, monkeypatch)
    captions = []

    class View:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

        def caption(self, text, band=None):
            captions.append(text)

        def close(self):
            pass

    monkeypatch.setattr(live, "CameraView", View)
    out = tmp_path / "both.jsonl"
    code = live.main(["--mock-glove", "--mock-leap", "--no-view",
                      "--seconds", "0.5", "--hand", "both",
                      "--out", str(out), "--fit-template", "none",
                      "--acquire-timeout", "5",
                      "--warmup-open", "1",
                      "--warmup-fist", "1.2"])
    text = capsys.readouterr().out
    assert code == 0, text

    def in_order(lines, wanted):
        at = 0
        for want in wanted:
            while at < len(lines) and not lines[at].startswith(want):
                at += 1
            assert at < len(lines), f"{want!r} missing or out of order"
            at += 1

    console = text.replace("\r", "\n").splitlines()
    in_order(console, [
        "Warm-up, one hand at a time (LEFT, then RIGHT)",
        "LEFT hand: ACQUIRE. Hold the LEFT hand open",
        "LEFT hand acquired: open palm in 1",
        "LEFT hand: OPEN PALM flat to the camera for 1 s",
        "LEFT hand: FIST for 1.2 s",
        "RIGHT hand: ACQUIRE. Hold the RIGHT hand open",
        "RIGHT hand acquired: open palm in 1",
        "RIGHT hand: OPEN PALM flat to the camera for 1 s",
        "RIGHT hand: FIST for 1.2 s",
        "Warm-up learned:",
        "Fusing for 0.5 s."])
    in_order(captions, [
        "LEFT hand: ACQUIRE", "LEFT hand acquired: open palm in",
        "LEFT hand: OPEN PALM", "LEFT hand: FIST",
        "RIGHT hand: ACQUIRE", "RIGHT hand acquired: open palm in",
        "RIGHT hand: OPEN PALM", "RIGHT hand: FIST", "LIVE FUSION"])
    hands = {json.loads(line)["hand"]
             for line in out.read_text(encoding="utf-8").splitlines()}
    assert hands == {"left", "right"}


# --- replay: the live path against fuse_all ------------------------------------

LAG = 0.47
TAKE = "mixed_right_take1_20260922_120000.jsonl"


def _fuse_module():
    """scripts/fuse_poses.py, which is not on a package path."""
    path = ROOT / "scripts" / "fuse_poses.py"
    spec = importlib.util.spec_from_file_location("fuse_poses_live_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_session(root, seconds=2.0):
    """A one-take session in the recorders' own formats, from the mocks.

    The camera is the mock Leap (`open_stream(mock=True)`), written by the
    real `LeapRecorder`, with its artefacts turned up so two seconds hold
    everything the pairing has to get right: a dropout every 60 frames, a
    re-acquisition (new hand id, visible time back to zero) at frame 100,
    and the four poses cycling every 40 frames. The glove is `MockGlove`'s
    generator, written line for line as `StampedFrameRecorder` writes it,
    arriving `LAG` seconds after the camera frames its first rail frames
    should pair with, so the override has a rail to act on while the camera
    sees a fist, and its last quarter second has no camera at all.
    """
    from leap_hand.live import MockGlove
    from leap_hand.recorder import LeapRecorder
    from leap_hand.stream import open_stream
    from xr_hand.parser import parse_hand_message
    from xr_hand.recorder import _frame_to_dict

    stream = open_stream(mock=True)
    stream.dropout_every, stream.dropout_frames = 60, 6
    stream.reacquire_every = 100
    stream.cycle_frames = 40
    rec = LeapRecorder(pose="mixed", take=1)
    rec.start(root / "leap" / TAKE)
    for _side, lh in stream.generate(int(seconds * 90)):
        rec.record(lh)
    rec.stop()
    t0 = stream._t0_us / 1e6                    # capture time of frame 0

    gen = MockGlove("right").gen
    glove_path = root / "glove" / TAKE
    glove_path.parent.mkdir(parents=True, exist_ok=True)
    with open(glove_path, "w", encoding="utf-8") as f:
        for k in range(int(seconds * 60)):
            t = round(t0 + 0.44 + LAG + k / 60.0, 6)
            frame = parse_hand_message(gen.next_frame(), hand_side_hint="right")
            d = _frame_to_dict(frame, wall_time=t)
            d.update({"pose": "mixed", "take": 1, "capture_time": t})
            f.write(json.dumps(d) + "\n")
    return root


@pytest.mark.parametrize("variant", ["plain", "fitted_with_anchor"])
def test_replay_through_the_live_path_matches_fuse_all(tmp_path, monkeypatch,
                                                       variant):
    """Frame for frame: the same fused points to 1e-9 m, the same dof_source
    on every paired frame.

    What is compared is the per-frame sequence, with the learning held
    equal. `fuse_all` learns its rails, endpoints and curl gates from the
    takes it is given, where the live path is handed them (by a warm-up,
    normally); that difference is inherent, so the live path is handed
    `learn_from_session`'s values and the test first checks they ARE
    `fuse_all`'s. Everything after that (pairing on the lagged stamp,
    hand-id stability decided as rows arrive, the rail tracker, the drift
    anchor's learn-then-apply, `fuse_skeletons` with `gate_curls`) runs
    through `LiveFusion.step` on one side and through `fuse_all` on the
    other.
    """
    from cam_hand.fusion import (
        DEFAULT_GATES,
        DriftAnchorParams,
        pairing_clock,
    )
    from cam_hand.live_fusion import learn_from_session, replay_take
    from cam_hand.template_fit import HandMeasurement, template_lengths

    fp = _fuse_module()
    root = _write_session(tmp_path)
    glove = fp.load_glove(root / "glove" / TAKE)
    cam = fp.load_leap_cam(root / "leap" / TAKE)
    clock = pairing_clock(glove, cam)
    assert clock == "capture_time"
    gates = DEFAULT_GATES
    flag_hand_id_stability(cam, gates, clock=clock)
    for r in cam:
        r.pop("abs26", None)
    loaded = [{"name": TAKE, "glove": glove, "cam": cam, "source": fp.LEAP,
               "clock": clock, "with_scale": False}]
    rail_params = RailOverrideParams(fingers=("index", "middle"))
    unreliable = {"right": ("ring",)}
    lag = {"right": LAG}

    measurements = {}
    anchor = None
    if variant == "fitted_with_anchor":
        lengths = {k: 0.93 * v
                   for k, v in template_lengths(glove[0]["frame"]).items()}
        path = tmp_path / "template_right.json"
        HandMeasurement(hand="right", lengths=lengths, spreads={},
                        n_frames=500, from_open=True).save(path)
        measurements, _s, _saved, refusals = fp.fit_measurements(
            str(path), {}, loaded, tmp_path)
        assert set(measurements) == {"right"} and not refusals
        anchor = DriftAnchorParams(min_frames=5)

    offline = []
    real = fp.fuse_skeletons

    def listen(glove_pts, cam_pts, *a, **k):
        fused, info = real(glove_pts, cam_pts, *a, **k)
        offline.append((np.array(fused, float), dict(info["dof_source"]),
                        cam_pts is not None))
        return fused, info

    monkeypatch.setattr(fp, "fuse_skeletons", listen)
    run = fp.fuse_all(loaded, gates, rail_params, unreliable, max_dt=0.05,
                      min_score=0.5, glove_lag=lag,
                      fitted=bool(measurements), anchor=anchor)

    learned = learn_from_session(glove, cam, gates=gates,
                                 rail_params=rail_params,
                                 measurements=measurements)
    assert learned.rails == run.rails
    assert learned.scale == run.scale
    assert learned.curl_gates == run.curl_gates
    if measurements:
        assert learned.curl_gates.get("right")   # the fit moved the gates

    live = replay_take(glove, cam, learned, gates=gates,
                       rail_params=rail_params, unreliable=unreliable,
                       anchor=anchor, lag=lag, max_dt=0.05, clock=clock)

    # fuse_all fuses a take's glove rows in stamp order, one call each, and
    # replay_take returns them in the same order.
    assert len(offline) == len(live) == len(glove)
    for k, ((pts, sources, paired), frame) in enumerate(zip(offline, live)):
        assert (frame.t_cam is not None) == paired, k
        assert np.max(np.abs(pts - np.asarray(frame.pts))) <= 1e-9, k
        if paired:
            assert frame.dof_source == sources, k

    # ...and the take exercised what it was built to exercise.
    paired = [f for f in live if f.t_cam is not None]
    assert 0 < len(paired) < len(live)
    assert any(s.startswith("camera") for f in paired
               for s in f.dof_source.values())
    assert any(f.rail_active for f in live)     # the override armed
    ids = {c["hand_id"] for c in cam if c["hand_side"] == "right"}
    assert len(ids) == 2                        # one re-acquisition
    if anchor is not None:
        assert any(f.corrections for f in live)


def test_replay_command_agrees_with_fuse_all_on_a_recorded_session(
        tmp_path, monkeypatch, capsys):
    live = _live_module()
    root = _write_session(tmp_path / "session")
    for extra in ([], ["--drift-anchor", "on"]):
        code = live.main(["--replay", str(root), "--profile", "none",
                          "--glove-lag", f"right:{LAG}",
                          "--fit-template", "none"] + extra)
        text = capsys.readouterr().out
        assert code == 0, text
        assert "identical to fuse_all's" in text
        assert "Live path agrees with fuse_all on every take." in text
        assert TAKE[:40] in text


def test_anchor_forgets_once_per_camera_gap_longer_than_a_second():
    """Stale is no camera frame for STALE_S (0.3 s); the anchor is reset once
    the hand has been stale for ANCHOR_RESET_S (1 s) more, once per gap."""
    from cam_hand.fusion import DriftAnchorParams
    from cam_hand.live_fusion import LiveFusion
    from leap_hand.live import MockGlove
    from xr_hand.keypoints21 import frame_to_keypoints21
    from xr_hand.parser import parse_hand_message

    learned = _warmup().learn(fit=FIT_NONE)
    buf = CameraBuffer()
    fusion = LiveFusion(learned, buf, anchor=DriftAnchorParams(),
                        lag={"right": 0.0})
    resets = []
    real_reset = fusion.anchor.reset
    fusion.anchor.reset = lambda: (resets.append(1), real_reset())
    gen = MockGlove("right").gen

    def glove_at(t):
        frame = parse_hand_message(gen.next_frame(), hand_side_hint="right")
        return {"t": t, "hand_side": "right", "frame": frame,
                "pts": frame_to_keypoints21(frame)}

    def camera_at(t):
        row = _row(t)
        row["pts"] = glove_at(t)["pts"]
        buf.add(row)

    camera_at(0.0)
    for t, expected in ((0.05, 0), (1.0, 0), (1.40, 1), (1.6, 1)):
        fusion.step(glove_at(t))
        assert len(resets) == expected, t
    camera_at(1.7)
    fusion.step(glove_at(1.72))                 # camera back: re-armed
    fusion.step(glove_at(3.10))                 # a second gap of 1.4 s
    assert len(resets) == 2
    assert fusion.anchor_resets["right"] == 2

    never = LiveFusion(learned, CameraBuffer(), anchor=DriftAnchorParams(),
                       anchor_reset_s=None)
    never.step(glove_at(10.0))
    assert not never.anchor_resets
