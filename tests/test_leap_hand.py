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
                "grab_strength", "palm_abs", "abs26"):
        assert key in row
    assert len(row["abs26"]) == 26 and len(row["palm_abs"]) == 3
    assert row["pose"] == "open_palm" and row["take"] == 1


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
        # 600 frames at 90 Hz with a re-acquisition every 180: 3 id changes.
        assert s.reacquisitions == 3
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
