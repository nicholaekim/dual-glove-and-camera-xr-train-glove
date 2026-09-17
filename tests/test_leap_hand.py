"""The Ultraleap backend, exercised without an Ultraleap.

No SDK and no camera exist on the machine this was written on, so the tests
stand in for both: a duck-typed `hand`/`event` pair covers the mapping from
LeapC's object model, and `MockLeapStream` covers everything downstream. The
one thing that cannot be checked here is whether real LeapC objects match the
shape assumed below — that is hardware day, and `scripts/leap/check_setup.py`
is the first thing to run.
"""
import math
from enum import Enum
from pathlib import Path

import pytest

from leap_hand.mock import MockLeapStream
from leap_hand.recorder import LeapRecorder
from leap_hand.stats import analyse_file, fingertip_jitter
from leap_hand.stream import LeapStream, LeapUnavailable
from leap_hand.to_openxr import (
    JOINT_SOURCES,
    MM_TO_FRAME_UNITS,
    hand_side,
    leap_hand_from_api,
    to_hand_frame,
)
from xr_hand.joints import JOINT_NAMES
from xr_hand.keypoints21 import frame_to_keypoints21
from xr_hand.kinematics import (
    absolute_to_relative,
    forward_kinematics,
    forward_kinematics_full,
    mat3_to_quat,
    quat_canonical,
    quat_normalize,
)
from xr_hand.recorder import FrameRecorder


# --- a fake LeapC hand ------------------------------------------------------
# Duck-typed to the bits of the SDK's object model to_openxr reads. Values are
# deliberately unique per joint so a mis-wired row of the mapping table cannot
# accidentally pass.
class FakeHandType(Enum):
    """Stand-in for leap.HandType; `str()` gives 'FakeHandType.Left'."""
    Left = 0
    Right = 1


class FakeVector:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class FakeQuaternion:
    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = (float(v) for v in (x, y, z, w))


class FakeBone:
    def __init__(self, tag: float):
        # prev/next are 10 mm apart so a prev/next mix-up is visible.
        self.prev_joint = FakeVector(tag, tag + 1.0, tag + 2.0)
        self.next_joint = FakeVector(tag + 10.0, tag + 11.0, tag + 12.0)
        self.rotation = FakeQuaternion(0.0, 0.0, math.sin(tag / 100.0),
                                       math.cos(tag / 100.0))
        self.width = 8.0


class FakeDigit:
    def __init__(self, base: float):
        self.bones = [FakeBone(base + 100.0 * b) for b in range(4)]


class FakePalm:
    def __init__(self):
        self.position = FakeVector(1.0, 250.0, 3.0)
        self.orientation = FakeQuaternion(0.0, 0.0, 0.0, 1.0)
        self.width = 85.0


class FakeHand:
    def __init__(self, side: str = "right", hand_id: int = 42):
        self.id = hand_id
        self.type = FakeHandType.Left if side == "left" else FakeHandType.Right
        self.visible_time = 1_500_000
        self.pinch_strength = 0.25
        self.grab_strength = 0.75
        self.pinch_distance = 30.0
        self.confidence = 1.0            # constant 1.0; must never be read
        self.palm = FakePalm()
        self.arm = FakeBone(5000.0)
        self.digits = [FakeDigit(1000.0 * (d + 1)) for d in range(5)]


class FakeEvent:
    def __init__(self, hands):
        self.hands = hands
        self.timestamp = 123_456_789
        self.tracking_frame_id = 7
        self.framerate = 89.7


def fake_event(side: str = "right"):
    hand = FakeHand(side)
    return hand, FakeEvent([hand])


def mm(v: FakeVector):
    """The metres a FakeVector's millimetres should become."""
    return [v.x * MM_TO_FRAME_UNITS, v.y * MM_TO_FRAME_UNITS, v.z * MM_TO_FRAME_UNITS]


# --- mapping ----------------------------------------------------------------
def test_mapping_fills_all_26_joints_in_openxr_order():
    hand, event = fake_event("right")
    lh = leap_hand_from_api(hand, event)

    assert len(lh.abs26) == len(lh.quat26) == len(JOINT_NAMES) == 26
    assert len(JOINT_SOURCES) == 26
    frame = to_hand_frame(lh)
    assert [j.name for j in frame.joints] == JOINT_NAMES

    assert lh.hand_side == "right"
    assert lh.hand_id == 42
    assert lh.frame_id == 7 and lh.timestamp_us == 123_456_789
    assert lh.framerate == pytest.approx(89.7)
    assert lh.visible_time_us == 1_500_000


def test_palm_and_wrist_rows():
    hand, event = fake_event()
    lh = leap_hand_from_api(hand, event)
    assert lh.abs26[0] == pytest.approx(mm(hand.palm.position))     # PALM
    assert lh.palm_pos == pytest.approx(mm(hand.palm.position))
    # WRIST is the far end of the arm bone, not its start.
    assert lh.abs26[1] == pytest.approx(mm(hand.arm.next_joint))


def test_thumb_chain_skips_the_zero_length_metacarpal():
    """Leap's thumb bones[0] has no length: the chain starts at bones[1]."""
    hand, event = fake_event()
    lh = leap_hand_from_api(hand, event)
    thumb = hand.digits[0].bones

    assert lh.abs26[2] == pytest.approx(mm(thumb[1].prev_joint))   # METACARPAL
    assert lh.abs26[3] == pytest.approx(mm(thumb[2].prev_joint))   # PROXIMAL
    assert lh.abs26[4] == pytest.approx(mm(thumb[3].prev_joint))   # DISTAL
    assert lh.abs26[5] == pytest.approx(mm(thumb[3].next_joint))   # TIP
    # bones[0] must not appear anywhere in the 26.
    assert mm(thumb[0].prev_joint) not in [list(p) for p in lh.abs26]


def test_finger_rows_and_tips():
    """Fingers use bones[0..3]; every TIP is its distal bone's next_joint."""
    hand, event = fake_event()
    lh = leap_hand_from_api(hand, event)
    for f, digit_idx in enumerate((1, 2, 3, 4)):        # index, middle, ring, little
        bones = hand.digits[digit_idx].bones
        base = 6 + f * 5
        for k in range(4):                              # METACARPAL..DISTAL
            assert lh.abs26[base + k] == pytest.approx(mm(bones[k].prev_joint))
        assert lh.abs26[base + 4] == pytest.approx(mm(bones[3].next_joint))
        # A TIP has no bone of its own: it reuses the distal bone's rotation.
        assert lh.quat26[base + 4] == lh.quat26[base + 3]


def test_units_are_metres_like_the_glove():
    hand, event = fake_event()
    lh = leap_hand_from_api(hand, event)
    # 250 mm above the module -> 0.25 in the frame's unit.
    assert lh.palm_pos[1] == pytest.approx(0.25)


def test_hand_side_from_the_enum():
    assert hand_side(FakeHand("left")) == "left"
    assert hand_side(FakeHand("right")) == "right"


def test_hand_frame_carries_the_leap_identifiers():
    hand, event = fake_event("left")
    frame = to_hand_frame(leap_hand_from_api(hand, event))
    assert frame.hand_side == "left"
    assert frame.frame_id == 7 and frame.packet_counter == 7
    assert frame.status == 1
    assert frame.timestamp == pytest.approx(123.456789)


# --- the round trip ---------------------------------------------------------
def test_absolute_to_relative_round_trip_on_a_fake_hand():
    """abs -> parent-relative -> forward kinematics must return the input."""
    hand, event = fake_event()
    lh = leap_hand_from_api(hand, event)
    world = forward_kinematics(to_hand_frame(lh))
    for got, want in zip(world, lh.abs26):
        assert got == pytest.approx(want, abs=1e-6)


def test_round_trip_on_every_mock_pose():
    stream = MockLeapStream(pose=None, noise_mm=0.4)
    stream.start()
    for _, lh in stream.generate(400):
        world = forward_kinematics(to_hand_frame(lh))
        for got, want in zip(world, lh.abs26):
            assert got == pytest.approx(want, abs=1e-6)


def test_round_trip_through_forward_kinematics_full():
    """The other direction: a glove frame -> absolute -> back, unchanged."""
    from xr_hand.joints import HandFrame
    from xr_hand.mock import MockHandGenerator
    from xr_hand.parser import parse_hand_message

    gen = MockHandGenerator(hand="right")
    for _ in range(25):
        raw = gen.next_frame()
    original = parse_hand_message(raw, hand_side_hint="right")

    positions, rotations = forward_kinematics_full(original)
    quats = [mat3_to_quat(r) for r in rotations]
    rebuilt = HandFrame(
        timestamp=original.timestamp, packet_counter=original.packet_counter,
        hand_side="right", frame_id=original.frame_id, status=original.status,
        joints=absolute_to_relative(positions, quats),
    )
    for got, want in zip(forward_kinematics(rebuilt), positions):
        assert got == pytest.approx(want, abs=1e-6)


def test_absolute_to_relative_keeps_the_wrist_absolute():
    """WRIST is the root (PARENT comes from BONES), so it keeps world pose."""
    hand, event = fake_event()
    lh = leap_hand_from_api(hand, event)
    joints = absolute_to_relative(lh.abs26, lh.quat26)
    wrist = joints[JOINT_NAMES.index("WRIST")]
    assert [wrist.x, wrist.y, wrist.z] == pytest.approx(lh.abs26[1])


def test_absolute_to_relative_rejects_the_wrong_length():
    with pytest.raises(ValueError):
        absolute_to_relative([[0.0, 0.0, 0.0]] * 25, [[0.0, 0.0, 0.0, 1.0]] * 25)


def test_absolute_to_relative_rejects_a_reordered_name_list():
    """PARENT is indexed by position, so another order would reparent joints."""
    pos = [[0.0, 0.0, 0.0]] * 26
    quat = [[0.0, 0.0, 0.0, 1.0]] * 26
    with pytest.raises(ValueError):
        absolute_to_relative(pos, quat, names=list(reversed(JOINT_NAMES)))
    assert len(absolute_to_relative(pos, quat, names=list(JOINT_NAMES))) == 26


def test_quat_canonical_keeps_a_sequence_continuous():
    q = quat_normalize((0.1, 0.2, 0.3, 0.9))
    flipped = tuple(-c for c in q)
    # Same rotation, opposite sign: against a reference, the sign that stays
    # on the reference's side wins.
    assert quat_canonical(flipped, q) == pytest.approx(q)
    assert quat_canonical(q, q) == pytest.approx(q)
    # With no reference, w >= 0 decides, so a lone quaternion still has a sign.
    assert quat_canonical(flipped)[3] >= 0.0
    # Either sign is the same rotation - that is why this is safe to do.
    from xr_hand.kinematics import quat_to_mat3
    assert quat_to_mat3(*flipped) == pytest.approx(quat_to_mat3(*q))


def test_mock_quaternions_never_flip_sign_between_frames():
    """A generator that flipped sign mid-sweep would fake a discontinuity."""
    stream = MockLeapStream(pose=None, noise_mm=0.0)
    stream.start()
    previous = {}
    for side, lh in stream.generate(400):
        if side in previous:
            for k, (was, now) in enumerate(zip(previous[side], lh.quat26)):
                dot = sum(a * b for a, b in zip(was, now))
                assert dot >= 0.0, f"{side} joint {k} flipped sign"
        previous[side] = lh.quat26


# --- the mock stream --------------------------------------------------------
def test_mock_stream_yields_both_hands():
    stream = MockLeapStream()
    stream.start()
    try:
        items = stream.generate(10)
        assert {side for side, _ in items} == {"left", "right"}
        assert len(items) == 20
        assert all(lh.hand_side == side for side, lh in items)
        assert all(len(lh.abs26) == 26 for _, lh in items)
    finally:
        stream.stop()


def test_mock_stream_drain_has_the_receiver_shape():
    """(hand_side, payload) items, like xr_hand.receiver.QueueItem."""
    import time
    stream = MockLeapStream()
    stream.start()
    try:
        time.sleep(0.15)
        items = stream.drain(64)
        assert items, "the mock produced nothing in 150 ms"
        for side, lh in items:
            assert side in ("left", "right")
            assert lh.hand_side == side
    finally:
        stream.stop()


def test_mock_injects_dropouts_and_reacquisitions():
    stream = MockLeapStream()
    stream.start()
    items = stream.generate(600)
    assert len(items) < 600 * 2, "no dropout was injected"
    ids = [lh.hand_id for side, lh in items if side == "right"]
    assert len(set(ids)) > 1, "no re-acquisition was injected"


def test_mock_poses_differ():
    """A fist's fingertips sit closer to the wrist than an open palm's."""
    def tip_spread(pose: str) -> float:
        s = MockLeapStream(pose=pose, noise_mm=0.0)
        s.start()
        lh = dict(s.generate(1))["right"]
        wrist = lh.abs26[1]
        return sum(math.dist(t, wrist) for t in lh.tips())

    assert tip_spread("fist") < tip_spread("open_palm")


# --- recording, and the tools that read it ----------------------------------
def _record_mock(tmp_path: Path, frames: int = 600, **kwargs) -> Path:
    stream = MockLeapStream(**kwargs)
    stream.start()
    rec = LeapRecorder(pose="open_palm", take=1)
    path = tmp_path / "open_palm_both_take1.jsonl"
    rec.start(path)
    for _, lh in stream.generate(frames):
        rec.record(lh)
    rec.stop()
    return path


def test_recording_loads_through_the_glove_loader(tmp_path: Path):
    """The whole point: FrameRecorder.load reads Leap files unchanged."""
    path = _record_mock(tmp_path, frames=20)
    frames = list(FrameRecorder.load(path))
    assert len(frames) == 40                       # 20 events x 2 hands
    frame, wall = frames[0]
    assert frame.hand_side in ("left", "right")
    assert [j.name for j in frame.joints] == JOINT_NAMES
    assert wall > 0


def test_recording_keeps_the_glove_keys_and_adds_the_leap_extras(tmp_path: Path):
    import json
    path = _record_mock(tmp_path, frames=5)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    for key in ("wall_time", "timestamp", "packet_counter", "hand_side",
                "frame_id", "status", "joints"):
        assert key in row, f"glove key {key} missing"
    assert row["source"] == "leap"
    assert row["space"] == "leap_desktop"
    assert row["units"] == "m"
    for key in ("hand_id", "visible_time_us", "framerate", "pinch_strength",
                "grab_strength", "palm_abs", "abs26", "timestamp_us"):
        assert key in row
    assert len(row["abs26"]) == 26 and len(row["palm_abs"]) == 3
    assert row["pose"] == "open_palm" and row["take"] == 1


def test_recording_carries_both_clocks(tmp_path: Path):
    """wall_time is the shared clock (fuse_poses pairs on it); timestamp is LeapC's."""
    import json
    import time

    before = time.time()
    path = _record_mock(tmp_path, frames=5)
    after = time.time()
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()]

    for row in rows:
        # wall_time must be real wall clock, taken at the write.
        assert before <= row["wall_time"] <= after
        # timestamp is the LeapC clock in seconds, matching timestamp_us.
        assert row["timestamp"] == pytest.approx(row["timestamp_us"] / 1e6)

    # ...and the LeapC clock advances by the frame interval, not by however
    # long the writer took.
    per_hand = [r for r in rows if r["hand_side"] == "right"]
    if len(per_hand) > 1:
        step = per_hand[1]["timestamp"] - per_hand[0]["timestamp"]
        assert step == pytest.approx(1.0 / 90.0, abs=1e-4)


def test_keypoints21_from_a_recorded_leap_frame(tmp_path: Path):
    path = _record_mock(tmp_path, frames=5)
    frame, _ = next(iter(FrameRecorder.load(path)))
    pts = frame_to_keypoints21(frame)
    assert len(pts) == 21
    assert pts[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)   # wrist at origin
    # A hand is roughly 20 cm: nothing should be metres away from the wrist.
    assert max(math.dist(p, (0.0, 0.0, 0.0)) for p in pts) < 0.30


def test_recorder_throttles_each_hand_independently(tmp_path: Path, monkeypatch):
    """Fake clock: the real one races the throttle interval (see test_pipeline)."""
    import leap_hand.recorder as recorder_mod
    now = [1000.0]
    monkeypatch.setattr(recorder_mod.time, "time", lambda: now[0])

    stream = MockLeapStream()
    stream.start()
    hands = dict(stream.generate(1))
    rec = LeapRecorder(hz=5.0)                  # one frame per hand per 0.2 s
    rec.start(tmp_path / "t.jsonl")
    rec.record(hands["right"])                  # first right: kept
    rec.record(hands["left"])                   # other hand, own schedule: kept
    now[0] += 0.1
    rec.record(hands["right"])                  # too soon: dropped
    now[0] += 0.15
    rec.record(hands["right"])                  # 0.25 s after its last: kept
    rec.stop()
    assert rec.count == 3
    assert rec.hands_seen == {"left", "right"}


def test_recorder_refuses_to_record_before_start():
    stream = MockLeapStream()
    stream.start()
    lh = stream.generate(1)[0][1]
    with pytest.raises(RuntimeError):
        LeapRecorder().record(lh)


# --- stats ------------------------------------------------------------------
def test_stats_on_a_mock_recording(tmp_path: Path):
    path = _record_mock(tmp_path, frames=600)
    rows = analyse_file(path)
    assert {r.hand_side for r in rows} == {"left", "right"}
    for s in rows:
        assert s.frames > 0
        assert math.isfinite(s.detection_rate) and 0.0 < s.detection_rate <= 1.0
        assert math.isfinite(s.mean_framerate) and s.mean_framerate > 0
        assert math.isfinite(s.jitter_mm) and s.jitter_mm >= 0.0
        assert math.isfinite(s.span_s) and s.span_s > 0
        assert s.frame_age_ms is None            # no LeapC clock behind a mock
        # 600 frames at 90 Hz, re-acquisition every 450: one id change.
        assert s.reacquisitions == 1
        # 20 dropped frames in every 300 -> about 93% of frames present.
        assert 0.85 < s.detection_rate < 1.0


def test_stats_jitter_grows_with_noise(tmp_path: Path):
    quiet = analyse_file(_record_mock(tmp_path / "a", frames=200,
                                      pose="open_palm", noise_mm=0.1))
    noisy = analyse_file(_record_mock(tmp_path / "b", frames=200,
                                      pose="open_palm", noise_mm=2.0))
    assert max(s.jitter_mm for s in quiet) < min(s.jitter_mm for s in noisy)


def test_jitter_is_zero_for_a_still_hand():
    rows = [{"wall_time": 100.0 + i * 0.01, "abs26": [[0.0, 0.0, 0.0]] * 26}
            for i in range(20)]
    assert fingertip_jitter(rows) == pytest.approx(0.0)


# --- the detection-rate denominator -----------------------------------------
# The reviewer's rule, and the bug it exists to stop. A full-rate file's
# timestamps jitter, so the 10th-percentile gap reads FASTER than the tracker
# ever ran; dividing by that invents dropouts. See leap_hand.stats.choose_rate.
def _write_leap_rows(path: Path, n: int, hz: float, framerate: float,
                     jitter_s: float = 0.0, seed: int = 3,
                     drop: range = range(0)) -> Path:
    """A minimal leap-format JSONL: one hand, chosen cadence, chosen jitter."""
    import json
    import random

    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        if i in drop:
            continue                      # a real dropout: the frame is absent
        t = i / hz + (rng.uniform(-jitter_s, jitter_s) if jitter_s else 0.0)
        lines.append(json.dumps({
            "source": "leap", "timestamp": t, "wall_time": 1.7e9 + t,
            "hand_side": "left", "hand_id": 11, "framerate": framerate,
            "pose": "bare", "take": 1,
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_choose_rate_picks_framerate_only_for_a_full_rate_file():
    from leap_hand.stats import choose_rate

    assert choose_rate(101.0, 90.0) == (90.0, "framerate")   # jittered 90 Hz
    assert choose_rate(90.0, 90.0) == (90.0, "framerate")
    assert choose_rate(5.0, 90.0) == (5.0, "cadence")        # --hz 5
    assert choose_rate(60.0, 90.0) == (60.0, "cadence")      # genuinely slower
    assert choose_rate(90.0, 0.0) == (90.0, "cadence")       # no framerate key


def test_jittered_full_rate_file_is_not_penalised_for_its_own_jitter(tmp_path: Path):
    """The 2026-09-16 bare-hand bug: 90 Hz file, cadence reads ~101 Hz.

    Every frame the tracker produced is in the file, so detection is 100%.
    Against the jittered cadence it would score about 89%.
    """
    path = _write_leap_rows(tmp_path / "bare_left_take1.jsonl",
                            n=900, hz=90.0, framerate=90.0, jitter_s=0.0011)
    s = analyse_file(path)[0]

    assert s.sample_hz > 95.0, "the jitter should make the cadence read fast"
    assert s.rate_source == "framerate"
    assert s.rate_hz == pytest.approx(90.0)
    assert s.detection_rate > 0.99
    # and the old denominator really would have cost it the plan's threshold
    assert s.frames / (s.span_s * s.sample_hz + 1) < 0.92


def test_a_throttled_file_is_still_measured_against_its_own_cadence(tmp_path: Path):
    """--hz 5 against a 90 Hz tracker: the rate is 5, or the file scores 6%."""
    path = _write_leap_rows(tmp_path / "poses_left_take1.jsonl",
                            n=50, hz=5.0, framerate=90.0)
    s = analyse_file(path)[0]
    assert s.rate_source == "cadence"
    assert s.rate_hz == pytest.approx(5.0, rel=1e-6)
    assert s.detection_rate > 0.98


def test_real_dropouts_still_show_up_at_full_rate(tmp_path: Path):
    """The rule must not turn every file into 100%: a gap is still a gap."""
    path = _write_leap_rows(tmp_path / "gappy_left_take1.jsonl",
                            n=900, hz=90.0, framerate=90.0, jitter_s=0.0011,
                            drop=range(300, 390))       # 1 s with no hand
    s = analyse_file(path)[0]
    assert s.rate_source == "framerate"
    assert s.detection_rate == pytest.approx(0.90, abs=0.02)


def test_the_table_footnote_names_the_denominator_it_used(tmp_path: Path):
    from leap_hand.stats import analyse_paths, format_table

    full = _write_leap_rows(tmp_path / "full_left_take1.jsonl",
                            n=200, hz=90.0, framerate=90.0, jitter_s=0.0011)
    table = format_table(analyse_paths([full]))
    assert "90.0*" in table                      # the row says which it used
    assert "tracking framerate" in table
    assert "no file here was throttled" in table

    slow = _write_leap_rows(tmp_path / "slow_left_take1.jsonl",
                            n=50, hz=5.0, framerate=90.0)
    table = format_table(analyse_paths([slow]))
    assert "5.0 " in table
    assert "own cadence" in table


# --- the no-hardware path ---------------------------------------------------
def test_missing_bindings_explain_the_next_step(monkeypatch):
    """The first error every new machine hits must name the fix, not traceback.

    `sys.modules['leap'] = None` makes `import leap` raise ImportError even
    where the bindings are installed, so this covers the empty-machine path
    on any machine.
    """
    import sys
    monkeypatch.setitem(sys.modules, "leap", None)

    with pytest.raises(LeapUnavailable) as e:
        LeapStream().start()
    message = str(e.value)
    assert "setup_bindings.ps1" in message
    assert "check_setup.py" in message
    assert "--mock" in message


def test_stream_rejects_an_unknown_tracking_mode():
    with pytest.raises(ValueError):
        LeapStream(mode="sideways")


def test_hand_side_falls_back_to_the_string_form(monkeypatch):
    """Without the bindings, `str(hand.type)` is all there is to go on."""
    import sys
    monkeypatch.setitem(sys.modules, "leap", None)
    assert hand_side(FakeHand("left")) == "left"
    assert hand_side(FakeHand("right")) == "right"


# --- IR images --------------------------------------------------------------
# The gate's evidence path: LeapC's raw buffer -> numpy -> PNG + sidecar. The
# fakes below are shaped like `leap.Image` / `leap.ImageEvent`: a thin wrapper
# with everything real behind `c_data`.
class FakeImageProperties:
    def __init__(self, width, height, bpp):
        self.width, self.height, self.bpp = width, height, bpp


class FakeImageCData:
    def __init__(self, properties, data, offset=0):
        self.properties = properties
        self.data = data
        self.offset = offset


class FakeImage:
    """Like leap.Image: only `c_data` reaches the pixels."""
    def __init__(self, c_data):
        self.c_data = c_data
        self.matrix_version = 1


class FakeImageEvent:
    def __init__(self, images, frame_id=11, timestamp=987_654):
        self.image = images

        class _Info:
            pass

        class _CData:
            pass

        info = _Info()
        info.frame_id = frame_id
        info.timestamp = timestamp
        self.c_data = _CData()
        self.c_data.info = info


def _fake_image(width, height, bpp=1, offset=0, fill=None, cffi=False):
    """One FakeImage over `width*height*bpp` bytes, optionally real cffi data."""
    payload = bytes(fill if fill is not None
                    else [(i * 7) % 256 for i in range(width * height * bpp)])
    raw = bytes(offset) + payload
    if cffi:
        ffi = pytest.importorskip("leapc_cffi").ffi
        raw = ffi.new("uint8_t[]", raw)
    return FakeImage(FakeImageCData(FakeImageProperties(width, height, bpp),
                                    raw, offset))


def test_image_to_numpy_reads_the_leapc_buffer_through_cffi():
    """The hardware path: a real ffi buffer, an offset, and a copy."""
    ffi = pytest.importorskip("leapc_cffi").ffi
    from leap_hand.images import image_to_numpy

    width, height, offset = 4, 3, 5
    image = _fake_image(width, height, offset=offset, cffi=True)
    array = image_to_numpy(image)

    assert array.shape == (height, width)          # (rows, columns)
    assert array.dtype.name == "uint8"
    assert array.ravel().tolist() == [(i * 7) % 256 for i in range(width * height)]

    # The copy is the point: LeapC reuses this buffer for the next frame, so
    # overwriting it must not reach an array we already handed out.
    before = array.copy()
    for i in range(offset, offset + width * height):
        image.c_data.data[i] = 0
    assert array.tolist() == before.tolist()
    assert ffi.buffer(image.c_data.data)[offset] == b"\x00"   # really cleared


def test_image_to_numpy_without_the_bindings():
    """Plain bytes work too, so the conversion is testable on any machine."""
    from leap_hand.images import image_to_numpy
    array = image_to_numpy(_fake_image(5, 2))
    assert array.shape == (2, 5)
    assert array[0, 0] == 0 and array[0, 1] == 7


def test_image_to_numpy_keeps_the_planes_of_a_wider_pixel():
    from leap_hand.images import image_to_numpy
    assert image_to_numpy(_fake_image(4, 3, bpp=3)).shape == (3, 4, 3)


def test_image_to_numpy_rejects_an_empty_frame():
    from leap_hand.images import image_to_numpy
    with pytest.raises(ValueError):
        image_to_numpy(_fake_image(0, 0))


def test_image_sampler_keeps_only_the_newest_pair():
    from leap_hand.images import ImageSampler

    sampler = ImageSampler()
    for frame_id in (1, 2, 3):
        sampler.on_image_event(FakeImageEvent(
            [_fake_image(4, 3), _fake_image(4, 3)], frame_id=frame_id,
            timestamp=1000 * frame_id))

    pair = sampler.latest()
    assert pair.frame_id == 3 and pair.timestamp_us == 3000
    assert pair.width == 4 and pair.height == 3 and pair.bpp == 1
    assert [side for side, _ in pair.sides()] == ["L", "R"]
    assert sampler.received == 3 and sampler.skipped == 2 and sampler.errors == 0
    assert sampler.latest() is None          # taken once, gone


def test_image_sampler_survives_a_bad_event():
    """A frame we cannot read is counted, not raised: the run keeps going."""
    from leap_hand.images import ImageSampler

    sampler = ImageSampler()
    sampler.on_image_event(FakeImageEvent([_fake_image(0, 0),
                                           _fake_image(0, 0)]))
    assert sampler.errors == 1 and sampler.latest() is None


def test_hand_trail_pairs_an_image_with_the_nearest_tracking_event():
    from leap_hand.images import HandTrail

    trail = HandTrail()
    assert trail.nearest(0) == ([], None)

    stream = MockLeapStream(noise_mm=0.0)
    stream.start()
    items = stream.generate(3)
    for _side, lh in items:
        trail.add(lh)

    first_ts = items[0][1].timestamp_us
    hands, dt_ms = trail.nearest(first_ts + 100)
    assert {h["hand_side"] for h in hands} == {"left", "right"}   # one event
    assert all(len(h["palm_pos"]) == 3 for h in hands)
    assert dt_ms == pytest.approx(-0.1)            # the event is 100 us older

    trail.clear()
    assert trail.nearest(first_ts) == ([], None)


def test_hand_trail_refuses_to_pair_a_still_with_a_stale_hand():
    """A photo of an untracked glove must not inherit an old hand."""
    from leap_hand.images import HandTrail

    trail = HandTrail()
    stream = MockLeapStream(noise_mm=0.0)
    stream.start()
    for _side, lh in stream.generate(1):
        trail.add(lh)
        last_ts = lh.timestamp_us

    # Ten seconds later the tracker has seen nothing since. The nearest event
    # is still that one, and it says nothing about this image.
    assert trail.nearest(last_ts + 10_000_000) == ([], None)
    assert trail.nearest(last_ts + 10_000)[0]          # 10 ms away: a real pair


def test_write_snapshot_writes_two_pngs_and_a_sidecar(tmp_path: Path):
    """A still is only evidence with the tracker's verdict beside it."""
    import json

    import cv2

    from leap_hand.images import MockImageSampler, write_snapshot

    pair = MockImageSampler(size=(32, 24)).latest()
    hands = [{"hand_side": "right", "hand_id": 7, "palm_pos": [0.0, 0.25, 0.0],
              "visible_time_us": 1_000_000}]
    snap = write_snapshot(tmp_path, "glove", 2, pair, hands=hands,
                          tracking_dt_ms=1.5)

    left = tmp_path / "glove_002_L.png"
    assert left.is_file() and (tmp_path / "glove_002_R.png").is_file()
    assert cv2.imread(str(left), cv2.IMREAD_UNCHANGED).shape == (24, 32)

    side = json.loads((tmp_path / "glove_002.json").read_text(encoding="utf-8"))
    assert side["width"] == 32 and side["height"] == 24 and side["bpp"] == 1
    assert side["frame_id"] == pair.frame_id
    assert side["timestamp_us"] == pair.timestamp_us
    assert side["wall_time"] > 0
    assert side["hand_count"] == 1 and side["hands"][0]["hand_id"] == 7
    assert "right" in side["tracking"]
    assert snap.saw_hand and snap.png_paths and snap.json_path


def test_write_snapshot_says_so_when_nothing_was_tracked(tmp_path: Path):
    from leap_hand.images import MockImageSampler, write_snapshot
    snap = write_snapshot(tmp_path, "glove", 0,
                          MockImageSampler(size=(16, 16)).latest())
    assert not snap.saw_hand
    assert snap.tracking_text == "no hand tracked"


def test_open_sampler_refuses_a_mock_stream_without_mock():
    from leap_hand.images import open_sampler
    stream = MockLeapStream()
    stream.start()
    with pytest.raises(LeapUnavailable):
        open_sampler(stream, mock=False)        # no connection behind a mock


# --- the gate verdict -------------------------------------------------------
# Synthetic stats, so the plan's thresholds are checked against numbers chosen
# to sit either side of them. A threshold that is wrong by a factor of ten is
# invisible in a live run and obvious here.
def _hand_stats(side: str, detection: float, reacquisitions: int,
                jitter_mm: float, span_s: float = 20.0, frames: int = 1800):
    from leap_hand.stats import HandStats
    return HandStats(
        file="gate.jsonl", hand_side=side, frames=frames, span_s=span_s,
        sample_hz=90.0, expected_frames=frames, detection_rate=detection,
        reacquisitions=reacquisitions, mean_framerate=89.9, jitter_mm=jitter_mm,
        frame_age_ms=9.5, pose="gate", rate_hz=89.9, rate_source="framerate",
    )


def _condition(name: str, rows, snapshots: int = 3):
    from leap_hand.gate import ConditionResult
    return ConditionResult(condition=name, stats=list(rows),
                           snapshots=snapshots,
                           snapshots_with_hand=snapshots if rows else 0,
                           folder=f"recordings/leap/gate/{name}")


def test_verdict_path_a_when_the_gloved_hand_tracks():
    from leap_hand.gate import format_report, verdict

    results = [
        _condition("bare", [_hand_stats("left", 1.00, 0, 0.50),
                            _hand_stats("right", 1.00, 0, 0.60)]),
        # 95% detected, one loss in 20 s (0.5 per 10 s), jitter 1.6x bare.
        _condition("glove", [_hand_stats("left", 0.95, 1, 0.80),
                             _hand_stats("right", 0.93, 1, 0.90)]),
    ]
    v = verdict(results)
    assert v.path == "A" and v.path_a
    assert v.passing == ["glove"]
    assert v.lines[0].startswith("Path A: yes because")
    assert "glove" in v.lines[0]
    report = format_report(results, v, title="test")
    assert "Path A: yes because" in report
    assert "bare" in report and "glove" in report


def test_verdict_path_b_when_the_glove_defeats_the_tracker():
    from leap_hand.gate import format_report, verdict

    results = [
        _condition("bare", [_hand_stats("left", 1.00, 0, 0.50),
                            _hand_stats("right", 1.00, 0, 0.50)]),
        # Every threshold missed: 21% of frames, 4.5 losses per 10 s, 6x jitter.
        _condition("glove", [_hand_stats("left", 0.21, 9, 3.00),
                             _hand_stats("right", 0.18, 11, 3.40)]),
    ]
    v = verdict(results)
    assert v.path == "B" and not v.path_a and v.passing == []
    assert v.lines[0].startswith("Path A: no because")
    assert "detection" in v.lines[0]
    assert "re-acquisitions" in v.lines[0]
    assert "jitter" in v.lines[0]
    assert any(line.startswith("Path B:") for line in v.lines)
    assert "Path A: no because" in format_report(results, v)


def test_verdict_each_threshold_alone_is_enough_to_fail():
    from leap_hand.gate import verdict

    bare = _condition("bare", [_hand_stats("right", 1.00, 0, 0.50)])
    only_detection = _condition("glove", [_hand_stats("right", 0.79, 0, 0.50)])
    only_reacq = _condition("glove", [_hand_stats("right", 1.00, 3, 0.50)])
    only_jitter = _condition("glove", [_hand_stats("right", 1.00, 0, 1.01)])
    for bad in (only_detection, only_reacq, only_jitter):
        assert verdict([bare, bad]).path == "B"
    # ...and the same numbers just inside every threshold pass.
    good = _condition("glove", [_hand_stats("right", 0.80, 2, 1.00)])
    assert verdict([bare, good]).path == "A"


def test_verdict_credits_a_mitigation_and_names_it():
    """If the liner is what works, the protocol keeps the liner — and says so."""
    from leap_hand.gate import verdict

    results = [
        _condition("bare", [_hand_stats("right", 1.00, 0, 0.50)]),
        _condition("glove", [_hand_stats("right", 0.10, 8, 4.00)]),
        _condition("glove_liner", [_hand_stats("right", 0.97, 0, 0.70)]),
    ]
    v = verdict(results)
    assert v.path == "A" and v.passing == ["glove_liner"]
    assert "glove_liner" in v.lines[0] and "plain glove did not" in v.lines[0]


def test_verdict_uses_the_bare_hand_of_the_same_side_as_the_baseline():
    """A left hand jitters differently; comparing across sides blames the glove."""
    from leap_hand.gate import jitter_baselines, verdict

    results = [
        _condition("bare", [_hand_stats("left", 1.0, 0, 2.00),
                            _hand_stats("right", 1.0, 0, 0.50)]),
        # 1.6 mm is 0.8x the left bare hand, but 3.2x the right one.
        _condition("glove", [_hand_stats("left", 1.0, 0, 1.60)]),
    ]
    assert jitter_baselines(results)["left"] == pytest.approx(2.0)
    assert verdict(results).path == "A"


def test_verdict_says_so_when_no_hand_was_tracked_at_all():
    """The smoke-test case: the run is not a result about the glove."""
    from leap_hand.gate import format_report, verdict

    results = [_condition("bare", [], snapshots=1)]
    v = verdict(results)
    assert v.path == "B"
    assert v.lines[0].startswith("Path A: no because no hand was tracked")
    report = format_report(results, v)
    assert "no hand was seen" in report


def test_verdict_refuses_to_judge_a_bare_only_run_that_did_see_a_hand():
    from leap_hand.gate import verdict
    v = verdict([_condition("bare", [_hand_stats("right", 1.0, 0, 0.5)])])
    assert v.path == "B"
    assert "no gloved condition was recorded" in v.lines[0]


def test_verdict_skips_the_jitter_threshold_without_a_bare_run():
    from leap_hand.gate import verdict
    v = verdict([_condition("glove", [_hand_stats("right", 0.99, 0, 9.9)])])
    assert v.path == "A"                       # detection and losses both pass
    assert any("jitter threshold could not be applied" in line
               for line in v.lines)


# --- the scripts ------------------------------------------------------------
def _load_script(name: str):
    """Import scripts/leap/<name>.py, which is not on a package path."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "leap" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"leap_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_script_runs_end_to_end_on_the_mock(tmp_path: Path, monkeypatch):
    """The whole Phase 2 protocol, minus the camera and the human."""
    import json
    import sys

    gate = _load_script("gate")
    monkeypatch.setattr(gate, "beep", lambda *a, **k: None)   # quiet, and fast

    out_dir = tmp_path / "recordings"
    report_path = tmp_path / "results" / "REPORT.txt"
    monkeypatch.setattr(sys, "argv", [
        "gate.py", "--mock", "--conditions", "bare,glove",
        "--seconds", "1", "--prep", "0", "--snapshots", "1",
        "--out-dir", str(out_dir), "--report", str(report_path),
    ])
    gate.main()

    report = report_path.read_text(encoding="utf-8")
    assert "Path A:" in report
    sidecars = []
    for condition in ("bare", "glove"):
        folder = out_dir / condition
        takes = list(folder.glob("*.jsonl"))
        assert len(takes) == 1, f"{condition} recorded no take"
        assert (folder / f"{condition}_000_L.png").is_file()
        assert (folder / f"{condition}_000_R.png").is_file()

        sidecar = json.loads(
            (folder / f"{condition}_000.json").read_text(encoding="utf-8"))
        assert sidecar["width"] == sidecar["height"] == 384
        assert "--mock" in sidecar["note"]
        # A still taken inside the mock's injected dropout legitimately sees
        # nothing; what must never happen is half a hand or a stale one.
        assert ({h["hand_side"] for h in sidecar["hands"]}
                in (set(), {"left", "right"}))
        sidecars.append(sidecar)

        rows = analyse_file(takes[0])
        assert {r.hand_side for r in rows} == {"left", "right"}
        for side in ("left", "right"):
            assert f"{condition:<16} {side:<5}" in report
    assert any(s["hand_count"] == 2 for s in sidecars)
    assert "no hand was seen" not in report


def test_gate_script_refuses_raw_capture_on_the_mock(monkeypatch):
    import sys
    gate = _load_script("gate")
    monkeypatch.setattr(sys, "argv", ["gate.py", "--mock", "--raw"])
    with pytest.raises(SystemExit):
        gate.main()


# --- the gate, recomputed from disk -----------------------------------------
def _mock_gate_run(tmp_path: Path, monkeypatch, conditions: str,
                   seconds: str = "1") -> tuple:
    """Run the mock gate once and return (out_dir, report_path)."""
    import sys

    gate = _load_script("gate")
    monkeypatch.setattr(gate, "beep", lambda *a, **k: None)
    out_dir = tmp_path / "recordings"
    report = tmp_path / "results" / "REPORT.txt"
    monkeypatch.setattr(sys, "argv", [
        "gate.py", "--mock", "--conditions", conditions,
        "--seconds", seconds, "--prep", "0", "--snapshots", "1",
        "--out-dir", str(out_dir), "--report", str(report),
    ])
    gate.main()
    return out_dir, report


def test_recompute_rebuilds_the_report_from_a_copy_of_a_run(tmp_path: Path,
                                                            monkeypatch):
    """No camera, no recording: the same numbers, out of the same files.

    Recomputed on a COPY, so this also proves the report does not depend on
    anything outside the folder — that is what lets a measurement change be
    re-applied to a session recorded days ago.
    """
    import shutil
    import sys

    out_dir, report = _mock_gate_run(tmp_path, monkeypatch, "bare,glove")
    original = report.read_text(encoding="utf-8")

    copy_dir = tmp_path / "copy"
    shutil.copytree(out_dir, copy_dir)
    new_report = tmp_path / "copy_results" / "REPORT.txt"

    gate = _load_script("gate")
    monkeypatch.setattr(sys, "argv", [
        "gate.py", "--recompute",
        "--out-dir", str(copy_dir), "--report", str(new_report),
    ])
    gate.main()
    rebuilt = new_report.read_text(encoding="utf-8")

    assert "recomputed:" in rebuilt and "no camera" in rebuilt
    for condition in ("bare", "glove"):
        for side in ("left", "right"):
            row = f"{condition:<16} {side:<5}"
            assert row in rebuilt
            # identical numbers to the live run: same files, same maths
            assert _row_of(rebuilt, row) == _row_of(original, row)
    assert rebuilt.count("IR still(s), 1 with a tracked hand") >= 1
    assert "Path A:" in rebuilt


def _row_of(report: str, prefix: str) -> str:
    for line in report.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"no row starting {prefix!r} in\n{report}")


def test_recompute_measures_the_newest_take_and_names_the_others(tmp_path: Path):
    from leap_hand.gate import scan_out_dir

    folder = tmp_path / "glove_20cm"
    old = _write_leap_rows(folder / "glove_20cm_left_take1_20260916_100000.jsonl",
                           n=100, hz=90.0, framerate=90.0)
    new = _write_leap_rows(folder / "glove_20cm_left_take1_20260916_223000.jsonl",
                           n=400, hz=90.0, framerate=90.0)
    # mtimes deliberately the wrong way round: the filename stamp decides.
    import os
    os.utime(old, (2e9, 2e9))
    os.utime(new, (1e9, 1e9))

    result = scan_out_dir(tmp_path)[0]
    assert result.condition == "glove_20cm"
    assert result.recordings == [str(new)]
    assert result.frames == 400
    assert new.name in result.note and old.name in result.note
    assert "not measured" in result.note


def test_recompute_takes_any_condition_name_and_judges_it_as_a_glove(
        tmp_path: Path):
    """The reviewer's extra runs: custom folders, same thresholds."""
    from leap_hand.gate import format_report, scan_out_dir, verdict

    for name, n in (("bare", 900), ("glove_right", 900), ("glove_50cm", 300)):
        _write_leap_rows(tmp_path / name / f"{name}_right_take1_20260916_220000.jsonl",
                         n=n, hz=90.0, framerate=90.0, jitter_s=0.0011,
                         drop=range(0) if n == 900 else range(100, 280))
    results = scan_out_dir(tmp_path)

    assert [r.condition for r in results] == ["bare", "glove_50cm", "glove_right"]
    v = verdict(results)
    assert v.path == "A"
    assert "glove_right" in v.passing        # full detection
    assert "glove_50cm" not in v.passing     # 40% of its frames missing
    report = format_report(results, v)
    assert "glove_right" in report and "glove_50cm" in report


def test_recompute_reports_a_condition_folder_with_no_take(tmp_path: Path):
    from leap_hand.gate import scan_out_dir

    (tmp_path / "glove_day2").mkdir(parents=True)
    result = scan_out_dir(tmp_path)[0]
    assert result.stats == [] and not result.saw_hand
    assert "no JSONL take" in result.note


def test_recompute_refuses_an_empty_folder(tmp_path: Path, monkeypatch):
    import sys
    gate = _load_script("gate")
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(sys, "argv", [
        "gate.py", "--recompute", "--out-dir", str(tmp_path / "nope"),
        "--report", str(tmp_path / "R.txt"),
    ])
    with pytest.raises(SystemExit):
        gate.main()


def test_check_setup_warns_about_an_empty_scene_but_still_exits_zero():
    """'Nobody was holding a hand up' is not a broken machine."""
    check_setup = _load_script("check_setup")
    ready = [
        check_setup.Check("import leap", check_setup.PASS, "here"),
        check_setup.Check(check_setup.STREAMING, check_setup.PASS,
                          "540 events, tracking 89.8 Hz"),
        check_setup.Check(check_setup.HAND_SEEN, check_setup.WARN,
                          "0 hands (none)", "hold a hand above the module"),
    ]
    assert check_setup.report(ready) == 0

    broken = list(ready)
    broken[1] = check_setup.Check(check_setup.STREAMING, check_setup.FAIL,
                                  "no tracking events", "check the panel")
    assert check_setup.report(broken) == 2


def test_check_setup_watches_for_six_seconds_by_default():
    """A device plugged in seconds ago sends nothing for the first few."""
    check_setup = _load_script("check_setup")
    assert check_setup.build_parser().parse_args([]).seconds == 6.0


def test_check_setup_without_the_bindings_skips_the_hand_line(monkeypatch):
    """No bindings: 'device streaming' fails, 'hand seen' has nothing to say."""
    import sys

    check_setup = _load_script("check_setup")
    monkeypatch.setitem(sys.modules, "leap", None)   # make `import leap` fail

    checks = check_setup.check_live(0.1)
    assert [c.name for c in checks] == [check_setup.STREAMING,
                                        check_setup.HAND_SEEN]
    assert checks[0].status == check_setup.FAIL
    assert checks[1].status == check_setup.SKIP
    assert check_setup.report(checks) == 2


# --- Path A: the simultaneous recorder with the leap backend ----------------
def _load_repo_script(name: str):
    """Import scripts/<name>.py, which is not on a package path."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"repo_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_sync_run(tmp_path: Path, monkeypatch, extra=()) -> tuple:
    """A whole --camera leap session on mock glove + mock leap."""
    import sys

    sync = _load_repo_script("record_simultaneous")
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    out_dir = tmp_path / "sync"
    monkeypatch.setattr(sys, "argv", [
        "record_simultaneous.py", "--camera", "leap",
        "--mock-glove", "--mock-leap", "--poses", "fist", "--takes", "1",
        "--duration", "2", "--prep", "0", "--out-dir", str(out_dir), *extra,
    ])
    sync.main()
    return sync, out_dir


def test_leap_backend_writes_a_pair_of_takes_that_pair_by_time_matches(
        tmp_path: Path, monkeypatch):
    """The Path A deliverable: two files, one name, one clock, matched frames."""
    import json

    from cam_hand.fusion import pair_by_time

    _sync, out_dir = _mock_sync_run(tmp_path, monkeypatch)

    glove_takes = sorted((out_dir / "glove").glob("*.jsonl"))
    leap_takes = sorted((out_dir / "leap").glob("*.jsonl"))
    assert len(glove_takes) == len(leap_takes) == 1
    # the same stem on both sides is what fuse_poses pairs takes on
    assert glove_takes[0].name == leap_takes[0].name
    assert not (out_dir / "cam").exists(), "the leap backend owns recordings/sync/leap"

    def rows(path):
        return [json.loads(line) for line
                in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    glove, cam = rows(glove_takes[0]), rows(leap_takes[0])
    assert glove and cam
    assert {d["source"] for d in cam} == {"leap"}     # how fuse_poses knows
    assert all("abs26" in d and len(d["abs26"]) == 26 for d in cam)
    assert all("joints" in d for d in cam)            # still the glove schema
    assert {d["pose"] for d in cam} == {"fist"}

    pairs = pair_by_time(glove, cam, max_dt=0.05)
    matched = [(g, c) for g, c in pairs if c is not None]
    assert len(matched) >= 0.8 * len(pairs), (
        f"only {len(matched)}/{len(pairs)} glove frames found a camera frame")
    assert all(g["hand_side"] == c["hand_side"] for g, c in matched)
    # both files are stamped with time.time() at the write, so the pairs are
    # tens of milliseconds apart, not hundreds
    assert max(abs(c["wall_time"] - g["wall_time"]) for g, c in matched) < 0.05


def test_leap_backend_keeps_every_camera_frame_by_default(tmp_path: Path,
                                                          monkeypatch):
    """--hz throttles the glove; the camera stays dense so pairing holds."""
    _sync, out_dir = _mock_sync_run(tmp_path, monkeypatch, ["--hz", "5"])
    glove = sorted((out_dir / "glove").glob("*.jsonl"))[0]
    cam = sorted((out_dir / "leap").glob("*.jsonl"))[0]
    n_glove = len(glove.read_text(encoding="utf-8").splitlines())
    n_cam = len(cam.read_text(encoding="utf-8").splitlines())
    assert n_cam > 5 * n_glove, f"{n_cam} camera frames vs {n_glove} glove"


def test_leap_backend_rejects_a_camera_name_that_is_neither(monkeypatch):
    import sys

    sync = _load_repo_script("record_simultaneous")
    monkeypatch.setattr(sys, "argv",
                        ["record_simultaneous.py", "--camera", "webcam"])
    with pytest.raises(SystemExit):
        sync.main()

    monkeypatch.setattr(sys, "argv",
                        ["record_simultaneous.py", "--mock-leap"])
    with pytest.raises(SystemExit):
        sync.main()


# --- the professor-frame replication ----------------------------------------
def test_record_frame_accepts_every_spelling_of_a_frame_id():
    record_frame = _load_script("record_frame")
    for raw in ("128166", "frame_128166", "frame_128166_DONE",
                "frame_128166_NA", " frame_128166_left "):
        assert record_frame.frame_name(raw) == "frame_128166"
    with pytest.raises(SystemExit):
        record_frame.frame_name("open_palm")


def test_record_frame_finds_the_reference_image_or_says_it_is_absent(
        tmp_path: Path):
    record_frame = _load_script("record_frame")
    root = tmp_path / "frames"
    (root / "frame_99").mkdir(parents=True)
    png = root / "frame_99" / "frame_99.png"
    png.write_bytes(b"\x89PNG\r\n")
    assert record_frame.find_reference("frame_99", root) == png
    assert record_frame.find_reference("frame_1234", root) is None
    assert record_frame.find_reference("frame_99", tmp_path / "nope") is None


def test_medoid_is_a_real_recorded_frame_not_an_average(tmp_path: Path):
    """The whole reason it is a medoid: bone lengths must survive."""
    record_frame = _load_script("record_frame")
    path = _record_mock(tmp_path, frames=120)
    frames = [f for f, _w in FrameRecorder.load(path)
              if f.hand_side == "right"]
    i = record_frame.medoid_index(frames)
    assert 0 <= i < len(frames)
    chosen = frame_to_keypoints21(frames[i])
    # it IS one of the frames, identical to the one at that index
    assert chosen == frame_to_keypoints21(frames[i])
    # and it is closer to the take's mean than the worst frame is
    def dist(f):
        pts = [c for p in frame_to_keypoints21(f) for c in p]
        return sum((a - b) ** 2 for a, b in zip(pts, mean))
    rows = [[c for p in frame_to_keypoints21(f) for c in p] for f in frames]
    mean = [sum(r[k] for r in rows) / len(rows) for k in range(len(rows[0]))]
    assert dist(frames[i]) == min(dist(f) for f in frames)


def test_record_frame_writes_the_professor_format_on_the_mock(tmp_path: Path,
                                                              monkeypatch):
    """The pipeline end to end: record, pick the medoid, write his format."""
    import sys

    from cam_hand.prof_format import load_file

    record_frame = _load_script("record_frame")
    monkeypatch.setattr(record_frame, "beep", lambda *a, **k: None)
    out_dir = tmp_path / "prof_frames"
    monkeypatch.setattr(sys, "argv", [
        "record_frame.py", "128166", "--mock", "--prep", "0",
        "--seconds", "1", "--out-dir", str(out_dir),
        "--reference", str(tmp_path / "no_reference_here"),
    ])
    record_frame.main()

    out_file = out_dir / "frame_128166_keypoints.txt"
    assert out_file.is_file()
    # one block per hand, 21 landmarks each, parsed by the reader that also
    # reads the professor's own files
    blocks = load_file(out_file)
    assert sorted(b.hand for b in blocks) == ["left", "right"]
    for b in blocks:
        assert len(b.points) == 21
        # millimetres in camera space: a hand is tens to hundreds of mm out,
        # never metres (that would mean the unit conversion was skipped)
        assert max(abs(c) for p in b.points for c in p) < 2000.0

    # the take it came from is kept beside it
    takes = list((out_dir / "frame_128166").glob("*.jsonl"))
    assert len(takes) == 1
    assert len(analyse_file(takes[0])) == 2       # both hands recorded


def test_record_frame_keeps_only_the_hand_you_asked_for(tmp_path: Path,
                                                        monkeypatch):
    import sys

    from cam_hand.prof_format import load_file

    record_frame = _load_script("record_frame")
    monkeypatch.setattr(record_frame, "beep", lambda *a, **k: None)
    out_dir = tmp_path / "prof_frames"
    monkeypatch.setattr(sys, "argv", [
        "record_frame.py", "frame_77", "--mock", "--prep", "0",
        "--seconds", "1", "--hand", "left", "--out-dir", str(out_dir),
    ])
    record_frame.main()
    blocks = load_file(out_dir / "frame_77_keypoints.txt")
    assert [b.hand for b in blocks] == ["left"]


# --- the installed bindings, if they are here -------------------------------
# These check the assumptions the hardware path is written against. They are
# skipped on a machine without the bindings, and they never touch a device.
class TestAgainstTheRealBindings:
    @pytest.fixture(autouse=True)
    def leap(self):
        return pytest.importorskip("leap", reason="the leap bindings are not installed")

    def test_hand_type_is_an_enum(self):
        import leap
        assert isinstance(leap.HandType.Left, leap.HandType)
        # ...and a foreign enum must not read as a left hand.
        assert hand_side(FakeHand("right")) == "right"

    def test_tracking_modes_exist(self):
        import leap
        for name in ("Desktop", "HMD", "ScreenTop"):
            assert hasattr(leap.TrackingMode, name)

    def test_connection_lifecycle_methods_exist(self):
        import leap
        for name in ("add_listener", "remove_listener", "connect", "disconnect",
                     "set_tracking_mode"):
            assert hasattr(leap.Connection, name), name

    def test_listener_callbacks_exist(self):
        import leap
        for name in ("on_connection_event", "on_connection_lost_event",
                     "on_device_event", "on_tracking_event"):
            assert hasattr(leap.Listener, name), name

    def test_get_now_is_available_for_frame_age(self):
        import leap
        assert isinstance(leap.get_now(), int)

    def test_cannot_open_device_error_is_where_the_stream_looks_for_it(self):
        import leap
        assert hasattr(leap.exceptions, "LeapCannotOpenDeviceError")

    def test_the_mapping_reads_attributes_that_actually_exist(self):
        """Every attribute to_openxr touches is a real property of the SDK type."""
        from leap.datatypes import Bone, Digit, Hand, Palm
        for name in ("id", "type", "visible_time", "pinch_strength",
                     "grab_strength", "palm", "digits", "arm"):
            assert hasattr(Hand, name), f"Hand.{name}"
        for name in ("prev_joint", "next_joint", "rotation"):
            assert hasattr(Bone, name), f"Bone.{name}"
        assert hasattr(Digit, "bones")
        for name in ("position", "orientation"):
            assert hasattr(Palm, name), f"Palm.{name}"

    def test_raw_recording_types_exist(self):
        import leap
        assert hasattr(leap, "Recording") and hasattr(leap, "Recorder")
        assert issubclass(leap.Recorder, leap.Listener)

    def test_the_images_policy_flag_is_in_leap_enums_only(self):
        """images.py depends on this: leap.PolicyFlag is an AttributeError."""
        import leap
        assert hasattr(leap.enums.PolicyFlag, "Images")
        assert not hasattr(leap, "PolicyFlag")
        assert hasattr(leap.Connection, "set_policy_flags")
        assert hasattr(leap.Listener, "on_image_event")

    def test_an_image_only_exposes_its_pixels_through_c_data(self):
        """Why image_to_numpy goes through c_data instead of the wrapper."""
        from leap.datatypes import Image
        assert hasattr(Image, "matrix_version")
        for name in ("properties", "data", "offset"):
            assert not hasattr(Image, name), f"Image.{name} exists now"
        from leap.events import ImageEvent
        assert hasattr(ImageEvent, "image")
