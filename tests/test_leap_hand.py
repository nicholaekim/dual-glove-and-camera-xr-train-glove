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
    """A whole --camera leap session on mock glove + mock leap.

    `--hand both` is the uncoached protocol these tests were written against:
    both hands kept, no acquire phase. The coached one-hand protocol has its
    own tests further down.
    """
    import sys

    sync = _load_repo_script("record_simultaneous")
    monkeypatch.setattr(sync, "beep", lambda *a, **k: None)
    out_dir = tmp_path / "sync"
    monkeypatch.setattr(sys, "argv", [
        "record_simultaneous.py", "--camera", "leap", "--hand", "both",
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
    # pairs are matched on capture_time, so that is the clock the 50 ms bound
    # holds on. wall_time is when a line was WRITTEN: on a loaded machine one
    # late write out of hundreds is normal, so it only has to be near, and the
    # typical pair close.
    def at(d):
        return d.get("capture_time", d["wall_time"])
    assert max(abs(at(c) - at(g)) for g, c in matched) <= 0.05
    wall = sorted(abs(c["wall_time"] - g["wall_time"]) for g, c in matched)
    assert wall[len(wall) // 2] < 0.05 and wall[-1] < 0.5


def test_leap_backend_keeps_every_camera_frame_by_default(tmp_path: Path,
                                                          monkeypatch):
    """--hz throttles the glove; the camera stays dense so pairing holds."""
    _sync, out_dir = _mock_sync_run(tmp_path, monkeypatch, ["--hz", "5"])
    glove = sorted((out_dir / "glove").glob("*.jsonl"))[0]
    cam = sorted((out_dir / "leap").glob("*.jsonl"))[0]
    n_glove = len(glove.read_text(encoding="utf-8").splitlines())
    n_cam = len(cam.read_text(encoding="utf-8").splitlines())
    assert n_cam > 5 * n_glove, f"{n_cam} camera frames vs {n_glove} glove"


def test_the_start_beep_cannot_backdate_the_head_of_a_take(tmp_path: Path,
                                                           monkeypatch):
    """A blocking beep must not leave a quarter second of stale frames.

    Both sensor threads keep queueing while `winsound.Beep` blocks, and both
    recorders stamp a frame with the time of the WRITE — so a beep that runs
    after the files are open puts backdated frames at the head of every take.
    The fix is an ordering, and this is what holds it: with a beep that
    really does block, nothing written may predate the end of that beep.
    """
    import json
    import sys
    import time as _time

    sync = _load_repo_script("record_simultaneous")
    beeps = []

    def slow_beep(freq=880, ms=180):
        end = _time.time() + ms / 1000.0
        while _time.time() < end:            # a real beep blocks; so does this
            _time.sleep(0.005)
        beeps.append(_time.time())

    monkeypatch.setattr(sync, "beep", slow_beep)
    out_dir = tmp_path / "sync"
    monkeypatch.setattr(sys, "argv", [
        "record_simultaneous.py", "--camera", "leap", "--hand", "both",
        "--mock-glove", "--mock-leap", "--poses", "fist", "--takes", "1",
        "--duration", "1", "--prep", "0", "--out-dir", str(out_dir),
    ])
    sync.main()

    assert len(beeps) >= 2                   # start beep, then the stop beep
    path = sorted((out_dir / "leap").glob("*.jsonl"))[0]
    rows = [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows

    # The backlog is visible as a burst: every frame the beep queued is
    # drained in one pass and written within a millisecond of the next, all
    # stamped with the same instant, although the camera captured them over a
    # quarter of a second. A 250 ms beep at 90 Hz x 2 hands is about 45 of
    # them; after the fix the first drain finds what one tick's worth is.
    t0 = min(r["wall_time"] for r in rows)
    in_first_20ms = sum(1 for r in rows if r["wall_time"] - t0 < 0.020)
    assert in_first_20ms < 10, (
        f"{in_first_20ms} frames share the first 20 ms of the take — that is "
        "the beep's backlog, written as if it had just been captured")


def test_both_sides_record_a_capture_time_and_it_is_used(tmp_path: Path,
                                                         monkeypatch):
    """The clock the two files are actually paired on, end to end."""
    import json

    from cam_hand.fusion import pairing_clock
    from xr_hand.recorder import FrameRecorder as PlainFrameRecorder

    _sync, out_dir = _mock_sync_run(tmp_path, monkeypatch)
    glove_path = sorted((out_dir / "glove").glob("*.jsonl"))[0]
    cam_path = sorted((out_dir / "leap").glob("*.jsonl"))[0]

    def rows(path):
        return [json.loads(line) for line
                in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    glove, cam = rows(glove_path), rows(cam_path)
    for name, side in (("glove", glove), ("leap", cam)):
        assert all(r.get("capture_time") is not None for r in side), name
        # it is a real wall-clock instant, at or before the write
        for r in side:
            assert 0.0 <= r["wall_time"] - r["capture_time"] < 5.0, name
    # the camera keeps the keys a later latency question is answered from
    assert all(r.get("timestamp_us") is not None for r in cam)
    assert all("frame_age_us" in r for r in cam)
    assert pairing_clock(glove, cam) == "capture_time"

    # the added key must not stop any existing reader: this is the loader
    # every glove tool in the repo uses
    frames = list(PlainFrameRecorder.load(glove_path))
    assert len(frames) == len(glove)
    assert frames[0][0].hand_side in ("left", "right")


def test_the_glove_recorder_adds_capture_time_without_changing_the_default():
    """StampedFrameRecorder adds one key; FrameRecorder's output is untouched."""
    import json
    import tempfile

    from xr_hand.mock import MockHandGenerator
    from xr_hand.parser import parse_hand_message
    from xr_hand.recorder import FrameRecorder as PlainFrameRecorder

    sync = _load_repo_script("record_simultaneous")
    frame = parse_hand_message(MockHandGenerator(hand="right").next_frame(),
                               hand_side_hint="right")

    with tempfile.TemporaryDirectory() as tmp:
        stamped = Path(tmp) / "stamped.jsonl"
        plain = Path(tmp) / "plain.jsonl"

        rec = sync.StampedFrameRecorder(pose="fist", take=1)
        rec.start(stamped)
        rec.record(frame, capture_time=1234.5)
        rec.record(frame)                      # no stamp given -> no key
        rec.stop()

        ref = PlainFrameRecorder(pose="fist", take=1)
        ref.start(plain)
        ref.record(frame)
        ref.stop()

        lines = [json.loads(x) for x
                 in stamped.read_text(encoding="utf-8").splitlines() if x.strip()]
        base = json.loads(plain.read_text(encoding="utf-8").splitlines()[0])

    assert lines[0]["capture_time"] == pytest.approx(1234.5)
    assert "capture_time" not in lines[1]
    assert "capture_time" not in base, "the plain recorder must be unchanged"
    # everything else is identical, key for key
    assert set(lines[0]) - {"capture_time"} == set(base)


def test_the_osc_queue_item_carries_arrival_time_and_still_unpacks():
    """Frozen scripts do `for hand, raw in drain(64)`; that must keep working."""
    import time as _time

    from xr_hand.receiver import OSCHandReceiver, QueueItem

    item = QueueItem("left", [1.0, 2.0], recv_time=99.5)
    hand, raw = item                            # the two-value unpack
    assert hand == "left" and raw == [1.0, 2.0]
    assert item.recv_time == 99.5
    assert item == ("left", [1.0, 2.0])         # still equal to a plain tuple
    assert QueueItem("left", [1.0]).recv_time == 0.0   # default

    # and the receiver really stamps it, on the thread that took the packet
    rx = OSCHandReceiver()
    before = _time.time()
    rx._enqueue("right", "/addr", [0.0] * 187)
    after = _time.time()
    queued = rx.drain(4)[0]
    assert before <= queued.recv_time <= after
    assert queued[0] == "right"


def test_a_take_needs_one_hand_on_BOTH_sensors_not_frames_on_each():
    """camera=left + glove=right is plenty of frames and exactly zero pairs."""
    sync = _load_repo_script("record_simultaneous")

    class FakeRec:
        def __init__(self, count, hands):
            self.count, self.hands_seen = count, set(hands)

    msg = sync.describe_mismatch(FakeRec(300, {"left"}), FakeRec(60, {"right"}))
    assert "left" in msg and "right" in msg and "no hand in common" in msg
    # the two silences need different fixes, so they get different words
    assert "camera captured nothing" in sync.describe_mismatch(
        FakeRec(0, set()), FakeRec(60, {"right"}))
    assert "glove captured nothing" in sync.describe_mismatch(
        FakeRec(300, {"left"}), FakeRec(0, set()))


def test_readiness_needs_a_shared_hand_and_names_the_disagreement():
    sync = _load_repo_script("record_simultaneous")
    session = sync.SyncSession(cap=None, tracker=None, glove_source=None,
                               hz=5.0, out_dir=Path("recordings") / "sync")
    session.glove_sides, session.cam_sides = {"right"}, {"left"}
    msg = session.not_ready_message(glove_ok=400, cam_ok=200)
    assert "no hand in common" in msg and "right" in msg and "left" in msg
    # a genuinely silent sensor still gets the old, correct advice
    session.glove_sides, session.cam_sides = set(), set()
    assert "Check XR Trainer" in session.not_ready_message(0, 200)


def test_the_mediapipe_session_still_owns_cam_and_its_own_recorder():
    """The refactor that made the camera pluggable must not have moved it."""
    from cam_hand.recorder import CamRecorder

    sync = _load_repo_script("record_simultaneous")
    session = sync.SyncSession(cap=None, tracker=None, glove_source=None,
                               hz=5.0, out_dir=Path("recordings") / "sync")
    assert session.cam_dir == Path("recordings") / "sync" / "cam"
    assert session.glove_dir == Path("recordings") / "sync" / "glove"
    assert isinstance(session.make_cam_recorder("fist", 1), CamRecorder)
    assert session.dots is True         # it still prints progress dots
    assert session.show is True         # and still opens its preview window


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


# --- the coached one-hand protocol ------------------------------------------
# The measured failure these cover (2026-09-17, left gloved hand): every pose
# but open_palm was recorded with a hand id that changed mid-take, or with the
# skeleton labelled as the other hand, and nothing in the recorder noticed.
class ScriptedLeap:
    """A Leap source whose hands a test decides, second by second.

    `MockLeapStream` is the right stand-in for "a camera is running", but its
    artefacts are on its own schedule: to test that a re-acquisition triggers
    a retry, the re-acquisition has to happen at a known moment. `plan(t)`
    returns `(side, hand_id, visible_time_us)` for every hand in view at
    `t` seconds into the run, and an empty list is a hand that is not there.

    Hands are stamped with the drain time, so a frame loop that stalls shows
    up as a burst of identical capture times — which is what the beep test
    looks for.
    """

    def __init__(self, plan, hz: float = 90.0):
        self.plan = plan
        self.hz = float(hz)
        self.frames = 0
        self.framerate = self.hz
        self.device_serial = "SCRIPTED"
        self._t0 = None
        self._next = 0.0
        self._shape = MockLeapStream(noise_mm=0.0, dropout_every=0,
                                     reacquire_every=0, pose="open_palm")

    def start(self):
        import time as _time
        self._t0 = _time.time()
        self._next = self._t0

    def stop(self):
        pass

    def drain(self, max_items: int = 16):
        import time as _time
        if self._t0 is None:
            self.start()
        now = _time.time()
        out = []
        while self._next <= now and len(out) < max_items:
            t = self._next - self._t0
            self._next += 1.0 / self.hz
            self.frames += 1
            shapes = dict(self._shape.generate(1))
            for side, hand_id, visible_us in self.plan(t):
                lh = shapes[side]
                lh.hand_id = hand_id
                lh.visible_time_us = int(visible_us)
                lh.capture_time = now
                lh.framerate = self.hz
                out.append((side, lh))
        return out


def _coached_run(tmp_path: Path, monkeypatch, plan, extra=(), beep=None):
    """A whole coached `--hand left` session over a scripted camera."""
    import sys

    sync = _load_repo_script("record_simultaneous")
    monkeypatch.setattr(sync, "beep", beep or (lambda *a, **k: None))
    monkeypatch.setattr(sync, "open_stream",
                        lambda **kw: _started(ScriptedLeap(plan)))
    out_dir = tmp_path / "sync"
    monkeypatch.setattr(sys, "argv", [
        "record_simultaneous.py", "--camera", "leap", "--hand", "left",
        "--mock-glove", "--mock-leap", "--poses", "fist", "--takes", "1",
        "--out-dir", str(out_dir), *extra,
    ])
    sync.main()
    return sync, out_dir


def _started(source):
    source.start()
    return source


def _takes(out_dir: Path, folder: str):
    """The TAKES in a session folder — settle clips are not takes.

    `cam_hand.recorder.take_files` is the one enumeration a reader of a
    session should use, and these tests are a reader of a session: a coached
    take now has a `.settle.jsonl` sibling holding its open-palm -> pose
    transition, and counting it as a take would mean every assertion about
    "one take on each side" was really about two files.
    """
    from cam_hand.recorder import take_files

    return take_files(out_dir / folder)


def _settles(out_dir: Path, folder: str):
    """...and the settle clips, which is the other half of the same question."""
    from cam_hand.recorder import SETTLE_SUFFIX

    return sorted((out_dir / folder).glob("*" + SETTLE_SUFFIX))


def _meta(out_dir: Path) -> dict:
    import json
    files = sorted((out_dir / "leap").glob("*.meta.json"))
    assert len(files) == 1, f"expected one meta.json, got {files}"
    return json.loads(files[0].read_text(encoding="utf-8"))


def _rows(path: Path):
    import json
    return [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_camera_hand_of_the_wrong_chirality_is_never_written(
        tmp_path: Path, monkeypatch):
    """--hand left must not be able to produce a right-handed line.

    The tracker re-acquiring a gloved hand from a closed pose sometimes
    returns the mirror image labelled as the other hand. That take fuses
    against the other glove and produces a plausible, wrong result, so the
    rejection has to happen before the frame reaches the recorder — not as a
    filter someone remembers to run afterwards.
    """
    def plan(t):
        return [("left", 4001, 5_000_000), ("right", 9001, 5_000_000)]

    _sync, out_dir = _coached_run(tmp_path, monkeypatch, plan,
                                  ["--duration", "1", "--settle", "0.2"])
    cam = _takes(out_dir, "leap")
    glove = _takes(out_dir, "glove")
    assert len(cam) == len(glove) == 1
    assert {r["hand_side"] for r in _rows(cam[0])} == {"left"}
    assert {r["hand_side"] for r in _rows(glove[0])} == {"left"}
    assert "_left_" in cam[0].name and cam[0].name == glove[0].name

    meta = _meta(out_dir)
    assert meta["hand"] == "left"
    assert meta["rejected_chirality"] > 0, (
        "the right hand was in view for the whole take and must be counted")
    assert meta["second_hand_frames"] == 0


def test_an_id_change_during_the_settle_retries_and_leaves_no_file(
        tmp_path: Path, monkeypatch):
    """The failure that ruined every gloved pose but open_palm.

    The hand is acquired open on id 5001; the tracker lets go a second later
    and comes back as 5002. That attempt must be abandoned — nothing written,
    nothing renamed — and the take retried from the acquire prompt.
    """
    def plan(t):
        if t < 1.0:
            return [("left", 5001, int(t * 1e6))]
        return [("left", 5002, int((t - 1.0) * 1e6))]

    _sync, out_dir = _coached_run(
        tmp_path, monkeypatch, plan,
        ["--duration", "0.8", "--settle", "1.5", "--retries", "1"])

    meta = _meta(out_dir)
    assert meta["attempts"] == 2, "the first attempt must have been retried"
    assert meta["hand_id"] == 5002, "the retry re-acquires under the new id"
    assert meta["accepted"] is True
    # exactly one take on each side: the session folder holds the take that
    # stood and nothing else
    assert len(_takes(out_dir, "leap")) == 1
    assert len(_takes(out_dir, "glove")) == 1
    assert {r["hand_id"] for r in _rows(_takes(out_dir, "leap")[0])} == {5002}
    # Nothing under rejected/ either: this attempt died during the SETTLE,
    # before either file was opened, so there is no recording to set aside.
    # (An attempt that fails once recording has started is moved there — see
    # test_a_take_the_hand_flickered_through_is_not_complete.)
    assert not (out_dir / "rejected").exists()


def test_a_take_the_hand_flickered_through_is_not_complete(tmp_path: Path,
                                                           monkeypatch):
    """One unbroken hand id is not enough; it has to be there the whole time.

    Here the id never changes — the hand simply keeps dropping out for a third
    of a second at a time. Coverage is what catches it, and a take under 90 %
    is set aside under `rejected/` rather than written into the session and
    blamed on the glove later.
    """
    def plan(t):
        if 1.2 <= t < 1.55 or 1.9 <= t < 2.25 or 2.6 <= t < 2.95:
            return []
        return [("left", 6001, 5_000_000)]

    _sync, out_dir = _coached_run(
        tmp_path, monkeypatch, plan,
        ["--duration", "2.5", "--settle", "0", "--retries", "0",
         "--acquire-timeout", "3"])

    assert _takes(out_dir, "leap") == [], "an incomplete take is not kept"
    assert _takes(out_dir, "glove") == []
    assert sorted((out_dir / "leap").glob("*.meta.json")) == []
    # ...and it is still on disk, under rejected/, with the reason attached
    aside = _takes(out_dir / "rejected", "leap")
    assert len(aside) == 1
    import json as _json
    why = _json.loads(aside[0].with_name(aside[0].stem + ".meta.json")
                      .read_text(encoding="utf-8"))
    assert why["accepted"] is False and "covered only" in why["why"]


def test_a_clean_take_records_its_coverage_and_how_it_was_got(tmp_path: Path,
                                                              monkeypatch):
    """meta.json is the take's provenance: hand, id, coverage, height, angle."""
    def plan(t):
        return [("left", 7001, 5_000_000)]

    _sync, out_dir = _coached_run(tmp_path, monkeypatch, plan,
                                  ["--duration", "1", "--settle", "0.2"])
    meta = _meta(out_dir)
    assert meta["coverage"] == pytest.approx(1.0, abs=0.02)
    assert meta["attempts"] == 1
    assert meta["hand_id"] == 7001 and meta["hand"] == "left"
    assert meta["pose"] == "fist" and meta["take"] == 1
    # the mock hand hovers 25 cm up with the palm toward the module, which is
    # inside the coached band and well inside the 40 degree view gate
    assert 18.0 <= meta["median_height_cm"] <= 28.0
    assert meta["median_view_angle_deg"] < 40.0
    assert meta["band_cm"] == [18.0, 28.0]
    assert meta["file"].endswith(".jsonl") and "_left_" in meta["file"]
    assert (out_dir / "leap" / meta["file"]).exists()


def test_the_hud_beeps_cannot_stall_the_frame_loop(tmp_path: Path,
                                                   monkeypatch):
    """The coached protocol beeps in the middle of a live capture clock.

    `winsound.Beep` blocks for its whole duration. The old protocol beeped
    before the files were open and threw the backlog away, which cannot work
    here: the pose cue lands between the acquire and the take, and the stop
    beep lands at its end. A blocking beep on this thread would be a quarter
    second of hand that nobody recorded, so the beeps go to a thread — and
    this is what holds that. Nothing may pile up at one instant.
    """
    import time as _time

    beeps = []

    def slow_beep(freq=880, ms=180):
        end = _time.time() + 0.25            # a real beep blocks; so does this
        while _time.time() < end:
            _time.sleep(0.005)
        beeps.append(_time.time())

    def plan(t):
        return [("left", 8001, 5_000_000)]

    _sync, out_dir = _coached_run(tmp_path, monkeypatch, plan,
                                  ["--duration", "1.5", "--settle", "0.6"],
                                  beep=slow_beep)
    assert len(beeps) >= 2, "the pose cue and the stop beep both really ran"

    rows = _rows(_takes(out_dir, "leap")[0])
    assert len(rows) > 50
    times = sorted(r["capture_time"] for r in rows)
    # A stalled loop drains its whole backlog in one pass, so every frame the
    # beep queued shares one capture instant. A 250 ms beep at 90 Hz is about
    # 22 of them.
    t0 = times[0]
    assert sum(1 for t in times if t - t0 < 0.020) < 8, (
        "frames piled up at the head of the take — the beep blocked the loop")
    assert max(b - a for a, b in zip(times, times[1:])) < 0.10, (
        "the frame loop went quiet for longer than a beep")
    # and the take is still the length it was asked for, not beep-stretched
    assert 1.4 <= times[-1] - times[0] <= 1.9
    assert _meta(out_dir)["coverage"] == pytest.approx(1.0, abs=0.02)


def test_an_accepted_take_keeps_its_settle_clip_beside_it(tmp_path: Path,
                                                          monkeypatch):
    """The transition into the pose is recorded, and NOT into the take.

    A coached take is REC only, so every tool downstream can take a median
    over it and mean "the held pose". That leaves the open-palm -> pose
    transition unrecorded, and it is the only moving hand the session
    produces — the only thing `cam_hand.fusion.estimate_glove_lag` can
    measure the glove's time lag on. So the settle phase goes to a sibling
    file on both sides, named after the take and stamped `phase: "settle"`,
    and the take itself is untouched.
    """
    def plan(t):
        return [("left", 7101, 5_000_000)]

    _sync, out_dir = _coached_run(tmp_path, monkeypatch, plan,
                                  ["--duration", "1", "--settle", "0.6"])
    for folder in ("leap", "glove"):
        takes, settles = _takes(out_dir, folder), _settles(out_dir, folder)
        assert len(takes) == len(settles) == 1, folder
        # same take name on both, so a clip is found beside its take by name
        assert settles[0].name == takes[0].name.replace(".jsonl",
                                                        ".settle.jsonl")
        assert "_left_" in settles[0].name, "renamed with the take"
        rows = _rows(settles[0])
        assert rows, f"{folder} settle clip is empty"
        assert all(r["phase"] == "settle" for r in rows)
        assert {r["pose"] for r in rows} == {"fist"}
        # ...and the take says nothing about a phase: it is REC and only REC
        assert all("phase" not in r for r in _rows(takes[0]))
    # both sides of the clip pair, and on the clock they will be paired on
    from cam_hand.fusion import pairing_clock
    glove = _rows(_settles(out_dir, "glove")[0])
    cam = _rows(_settles(out_dir, "leap")[0])
    assert pairing_clock(glove, cam) == "capture_time"
    assert {r["source"] for r in cam} == {"leap"}
    # the clip is the settle window, not the take: it ended before REC began
    assert max(r["capture_time"] for r in cam) <= min(
        r["capture_time"] for r in _rows(_takes(out_dir, "leap")[0])) + 0.05


def test_a_rejected_attempt_takes_its_settle_clip_with_it(tmp_path: Path,
                                                          monkeypatch):
    """A clip whose take was refused is set aside with it, never left behind.

    Same rule as the take itself: an exclusion has to stay countable, and a
    clip left in the session folder would be measured as the transition into
    a pose nobody kept.
    """
    def plan(t):
        if 1.2 <= t < 1.55 or 1.9 <= t < 2.25 or 2.6 <= t < 2.95:
            return []
        return [("left", 7201, 5_000_000)]

    _sync, out_dir = _coached_run(
        tmp_path, monkeypatch, plan,
        ["--duration", "2.5", "--settle", "0.4", "--retries", "0",
         "--acquire-timeout", "3"])

    for folder in ("leap", "glove"):
        assert _takes(out_dir, folder) == [], "an incomplete take is not kept"
        assert _settles(out_dir, folder) == [], "...nor is its settle clip"
        aside = _settles(out_dir / "rejected", folder)
        assert len(aside) == 1, f"the clip must be under rejected/{folder}"
        assert "_attempt1" in aside[0].name
        assert all(r["phase"] == "settle" for r in _rows(aside[0]))


def test_an_attempt_lost_during_the_settle_leaves_no_clip(tmp_path: Path,
                                                          monkeypatch):
    """Nothing to pair a clip with means no clip.

    The hand is acquired on one id and the tracker re-acquires it under
    another during the settle, so the attempt is abandoned before a take file
    is ever opened. There is no verdict on a recording to audit — the same
    case the interrupted-take branch deletes rather than sets aside — so the
    clip goes too, and `rejected/` stays a folder of judged attempts.
    """
    def plan(t):
        if t < 1.0:
            return [("left", 7301, int(t * 1e6))]
        return [("left", 7302, int((t - 1.0) * 1e6))]

    _sync, out_dir = _coached_run(
        tmp_path, monkeypatch, plan,
        ["--duration", "0.8", "--settle", "1.5", "--retries", "1"])

    assert _meta(out_dir)["attempts"] == 2, "the first attempt was retried"
    assert not (out_dir / "rejected").exists()
    for folder in ("leap", "glove"):
        assert len(_takes(out_dir, folder)) == 1
        # one clip, the surviving attempt's — not two
        assert len(_settles(out_dir, folder)) == 1


def test_a_settle_clip_is_where_the_glove_lag_is_measured(tmp_path: Path,
                                                          monkeypatch):
    """End to end: the recorder writes clips, fuse_poses finds them by name.

    The mock hand does not actually change shape on cue, so this is about the
    plumbing — that the pair is discovered, loaded and offered to the
    estimator — and not about the lag it comes back with. What the estimator
    does with a moving hand is `tests/test_fusion.py`'s job, on a synthetic
    pair with a planted lag.
    """
    def plan(t):
        return [("left", 7401, 5_000_000)]

    _sync, out_dir = _coached_run(tmp_path, monkeypatch, plan,
                                  ["--duration", "1", "--settle", "0.6"])
    fuse = _load_repo_script("fuse_poses")
    clips = fuse.settle_clips(out_dir, [out_dir / "leap"])
    assert len(clips) == 1
    gpath, cpath = clips[0]
    assert gpath.parent.name == "glove" and cpath.parent.name == "leap"
    assert gpath.name == cpath.name
    # ...and the take enumeration does not see them
    assert len(fuse.take_files(out_dir / "glove")) == 1
    row = fuse.lag_from_clips(
        [(fuse.load_glove(gpath), fuse.load_cam(cpath)[0])], "left", "settle")
    assert row.n_clips == 1
    assert row.applied or "not measurable" in row.why


def test_the_leap_backend_insists_on_being_told_which_hand(monkeypatch):
    """A session that does not say which hand cannot enforce chirality."""
    import sys

    sync = _load_repo_script("record_simultaneous")
    monkeypatch.setattr(sys, "argv",
                        ["record_simultaneous.py", "--camera", "leap"])
    with pytest.raises(SystemExit) as e:
        sync.main()
    assert "--hand" in str(e.value) and "both" in str(e.value)


def test_the_acquire_gate_is_the_one_the_analysis_uses():
    """Every reason a hand is not ready yet, and the all-clear."""
    from leap_hand.protocol import (
        NOT_FACING,
        NOT_TRACKED,
        OFF_AXIS,
        TOO_HIGH,
        TOO_LOW,
        TOO_YOUNG,
        WRONG_HAND,
        HandReading,
        acquire_failures,
    )

    band = (18.0, 28.0)

    def reading(**kw):
        d = dict(hand_side="left", hand_id=1, visible_time_us=800_000,
                 height_cm=23.0, lateral_cm=5.0, view_angle_deg=10.0)
        d.update(kw)
        return HandReading(**d)

    assert acquire_failures(None, "left", band) == [NOT_TRACKED]
    assert acquire_failures(reading(hand_side="right"), "left",
                            band) == [WRONG_HAND]
    assert acquire_failures(reading(), "left", band) == []
    assert TOO_YOUNG in acquire_failures(reading(visible_time_us=400_000),
                                         "left", band)
    assert TOO_LOW in acquire_failures(reading(height_cm=14.0), "left", band)
    assert TOO_HIGH in acquire_failures(reading(height_cm=33.0), "left", band)
    # 40 degrees is the angle cam_hand.fusion stops trusting a camera DOF at;
    # coaching to anything looser just moves the failure downstream
    assert NOT_FACING in acquire_failures(reading(view_angle_deg=55.0),
                                          "left", band)
    assert NOT_FACING in acquire_failures(reading(view_angle_deg=None),
                                          "left", band)
    # the idle other hand, 20 cm off to the side of a hand 23 cm up
    assert OFF_AXIS in acquire_failures(reading(lateral_cm=20.0), "left", band)


def test_coverage_measures_the_hole_not_the_frame_count():
    """Coverage has to mean the same thing at 90 Hz and at --leap-hz 5."""
    from leap_hand.protocol import coverage, longest_gap

    dense = [i / 90.0 for i in range(int(2.0 * 90))]
    assert coverage(dense, 0.0, 2.0) == pytest.approx(1.0, abs=0.01)
    # the same two seconds saved at 5 Hz is the same hand and scores the same,
    # which is the whole reason this is not frames-recorded / frames-expected:
    # that ratio would call the throttled file 6 % covered
    sparse = [i / 5.0 for i in range(10)]
    assert coverage(sparse, 0.0, 2.0) == pytest.approx(1.0, abs=0.01)
    # below the gap tolerance the saved frames really are further apart than a
    # loss, and the measurement says so rather than pretending
    assert coverage([i / 2.0 for i in range(4)], 0.0, 2.0) < 0.2
    # half a second missing out of two is exactly a quarter of the take
    holed = [t for t in dense if not 0.5 <= t < 1.0]
    assert coverage(holed, 0.0, 2.0) == pytest.approx(0.75, abs=0.02)
    assert longest_gap(holed, 0.0, 2.0) == pytest.approx(0.5, abs=0.02)
    assert coverage([], 0.0, 2.0) == 0.0
    assert longest_gap([], 0.0, 2.0) == 2.0


def test_the_hud_line_says_which_hand_and_how_high():
    """The operator's eyes are on the camera; the line has one second to work."""
    from leap_hand.protocol import HandReading, hud_line

    band = (18.0, 28.0)
    good = HandReading(hand_side="left", hand_id=1, visible_time_us=900_000,
                       height_cm=23.0, lateral_cm=4.0, view_angle_deg=12.0)
    line = hud_line("REC", 3.2, good, "left", band, 61.0)
    assert "REC" in line and "3.2s" in line and "left YES" in line
    assert "23.0 cm" in line and "OK" in line and "12 deg" in line
    assert "61.0/s" in line

    low = hud_line("ACQUIRE", None, HandReading("left", 1, 900_000, 13.0, 4.0,
                                                12.0), "left", band, 0.0)
    assert "TOO LOW" in low
    high = hud_line("ACQUIRE", None, HandReading("left", 1, 900_000, 40.0, 4.0,
                                                 12.0), "left", band, 0.0)
    assert "TOO HIGH" in high
    # only the other chirality in view is a different problem from no hand
    assert "WRONG HAND" in hud_line("ACQUIRE", None, None, "left", band, 60.0,
                                    saw_other_hand=True)
    assert "left no" in hud_line("ACQUIRE", None, None, "left", band, 60.0)


def test_the_presentation_hints_reach_the_acquire_prompt():
    """The poses that failed get told how to be presented, not just named."""
    from leap_hand.protocol import acquire_prompt

    thumbs = " ".join(acquire_prompt("thumbs_up", "left"))
    assert "OPEN PALM" in thumbs and "LEFT" in thumbs
    assert "forearm" in thumbs and "30-45" in thumbs
    assert "palm facing the lens" in " ".join(acquire_prompt("pinch", "right"))
    assert "close slowly" in " ".join(acquire_prompt("fist", "left"))
    # a pose with no known trap is prompted without inventing advice
    assert len(acquire_prompt("peace", "left")) == 1


# --- the coached gate: per scheduled pose -----------------------------------
def _write_scheduled_take(path: Path, plan: str = "open_palm:1,fist:1",
                          present=None, side: str = "left", hand_id: int = 1,
                          hz: float = 90.0, condition: str = "glove",
                          ids=None):
    """A take recorded under `plan`, with `present(t)` deciding each frame.

    `pose_t` is the capture clock offset into the schedule, so a test can put
    a hole exactly inside one pose window and nowhere else.
    """
    from leap_hand.protocol import parse_schedule

    rec = LeapRecorder(pose=condition, take=1, pose_plan=plan)
    rec.start(path)
    rec.schedule_t0 = 0.0
    shape = MockLeapStream(noise_mm=0.0, dropout_every=0, reacquire_every=0,
                           pose="open_palm")
    for i in range(int(parse_schedule(plan).total * hz)):
        t = i / hz
        hands = dict(shape.generate(1))
        if present is not None and not present(t):
            continue
        lh = hands[side]
        lh.hand_id = ids(t) if ids else hand_id
        lh.visible_time_us = 5_000_000
        lh.framerate = hz
        lh.capture_time = t                    # schedule_t0 is 0, so pose_t = t
        rec.record(lh)
    rec.stop()
    return path


def _pose(**kw):
    """A `PoseStats` row with plausible defaults, for judging in isolation."""
    from leap_hand.stats import PoseStats

    d = dict(file="f.jsonl", condition="glove", hand_side="left",
             pose="fist", frames=450, scheduled_s=5.0, detection_rate=0.95,
             longest_loss_s=0.05, reacquisitions=0)
    d.update(kw)
    return PoseStats(**d)


def test_a_pose_passes_on_its_own_merit_or_level_with_the_bare_hand():
    """Absolute OR relative, on both clauses, and nothing without a bare row."""
    from leap_hand.gate import judge_pose

    bare = _pose(condition="bare", detection_rate=0.98, longest_loss_s=0.04)

    passed, why = judge_pose(_pose(detection_rate=0.93), bare)
    assert passed is True and "93.0%" in why[0]

    # well under 80 %, but the bare hand was no better: that is not about the
    # glove, and the pose passes
    poor_bare = _pose(condition="bare", detection_rate=0.55,
                      longest_loss_s=3.0)
    assert judge_pose(_pose(detection_rate=0.50, longest_loss_s=2.5),
                      poor_bare)[0] is True

    # the fist case: far below 80 % and far below a bare hand that managed it
    passed, why = judge_pose(_pose(detection_rate=0.41, longest_loss_s=2.3),
                             bare)
    assert passed is False
    assert any("41.0%" in r and "below bare" in r for r in why)
    assert any("2.30 s" in r for r in why)

    # a long loss alone fails it, even with the detection rate up
    assert judge_pose(_pose(detection_rate=0.99, longest_loss_s=1.6),
                      bare)[0] is False
    # ...unless the bare hand lost it for just as long
    assert judge_pose(_pose(detection_rate=0.99, longest_loss_s=1.6),
                      _pose(condition="bare", longest_loss_s=2.0))[0] is True


def test_a_pose_with_no_bare_partner_is_reported_missing_never_guessed():
    """The 2026-09-17 failure: four gloved fists, no bare fist anywhere."""
    from leap_hand.gate import NO_BARE, judge_pose

    passed, why = judge_pose(_pose(pose="fist", detection_rate=0.94), None)
    assert passed is None, "an unpaired pose must not be scored either way"
    assert "no bare fist" in why[0] and "left hand" in why[0]
    assert "94.0%" in why[0], "its own numbers are still reported"

    from leap_hand.gate import PoseVerdict

    assert PoseVerdict(condition="glove", hand_side="left", pose="fist",
                       glove=_pose(), passed=None).status == NO_BARE


def test_the_paired_verdict_pairs_by_pose_and_by_hand():
    """Fist against fist, left against left — and 20 cm against 20 cm."""
    from leap_hand.gate import ConditionResult, pose_verdicts

    bare = ConditionResult(condition="bare", pose_stats=[
        _pose(condition="bare", pose="open_palm", detection_rate=0.98),
        _pose(condition="bare", pose="fist", detection_rate=0.96),
        _pose(condition="bare", pose="fist", hand_side="right",
              detection_rate=0.30, longest_loss_s=4.0),
    ])
    glove = ConditionResult(condition="glove", pose_stats=[
        _pose(pose="open_palm", detection_rate=0.97),
        _pose(pose="fist", detection_rate=0.35, longest_loss_s=2.2),
        _pose(pose="fist", hand_side="right", detection_rate=0.31,
              longest_loss_s=3.5),
        _pose(pose="pinch", detection_rate=0.88),
    ])
    got = {(v.pose, v.hand_side): v for v in pose_verdicts([bare, glove])}
    assert got[("open_palm", "left")].passed is True
    assert got[("fist", "left")].passed is False
    # the right hand's bare fist was just as bad, so the glove is not blamed
    assert got[("fist", "right")].passed is True
    # no bare pinch at all: reported, not guessed
    assert got[("pinch", "left")].passed is None
    assert got[("pinch", "left")].bare is None
    # bare rows are not judged against themselves
    assert all(v.condition == "glove" for v in pose_verdicts([bare, glove]))


def test_a_distance_run_is_paired_against_the_bare_run_at_that_distance():
    """Detection falls off with height; 20 cm vs 35 cm measures the height."""
    from leap_hand.gate import ConditionResult, pose_verdicts

    results = [
        ConditionResult(condition="bare", pose_stats=[
            _pose(condition="bare", detection_rate=0.99)]),
        ConditionResult(condition="bare_50cm", pose_stats=[
            _pose(condition="bare_50cm", detection_rate=0.45)]),
        ConditionResult(condition="glove_50cm", pose_stats=[
            _pose(condition="glove_50cm", detection_rate=0.44)]),
    ]
    v = pose_verdicts(results)[0]
    assert v.baseline_condition == "bare_50cm"
    assert v.passed is True, "44 % against a bare 45 % is not a glove failure"


def test_the_height_band_comes_from_the_condition_name():
    """`glove_20cm` says its own band; that is the point of a distance sweep."""
    from leap_hand.protocol import DEFAULT_BAND, band_for_condition, parse_band

    assert band_for_condition("glove") == DEFAULT_BAND
    assert band_for_condition("glove_20cm") == (15.0, 25.0)
    assert band_for_condition("bare_50cm") == (45.0, 55.0)
    # an explicit --band wins over the name
    assert band_for_condition("glove_20cm", (10.0, 40.0)) == (10.0, 40.0)
    assert parse_band("18,28") == (18.0, 28.0)
    with pytest.raises(ValueError):
        parse_band("28,18")


def test_per_pose_stats_measure_each_window_against_its_own_seconds(
        tmp_path: Path):
    """A pose the tracker saw for half a second must not score 100 %."""
    from leap_hand.stats import analyse_poses

    # open_palm 0-2 s fully tracked; fist 2-4 s lost for a second in the middle
    path = _write_scheduled_take(
        tmp_path / "glove_left_take1.jsonl", plan="open_palm:2,fist:2",
        present=lambda t: not (2.5 <= t < 3.5))
    rows = {s.pose: s for s in analyse_poses(path)}
    assert set(rows) == {"open_palm", "fist"}
    assert rows["open_palm"].detection_rate == pytest.approx(1.0, abs=0.02)
    assert rows["open_palm"].longest_loss_s < 0.05
    assert rows["fist"].detection_rate == pytest.approx(0.5, abs=0.03)
    assert rows["fist"].longest_loss_s == pytest.approx(1.0, abs=0.05)
    assert rows["fist"].scheduled_s == 2.0
    # the mock hand hovers 25 cm up with the palm toward the module
    assert rows["fist"].median_height_cm == pytest.approx(25.0, abs=0.5)
    assert rows["fist"].facing_pct == pytest.approx(100.0, abs=0.1)
    assert rows["fist"].in_band_pct == pytest.approx(100.0, abs=0.1)


def test_a_pose_the_tracker_never_saw_reports_the_whole_window_as_lost(
        tmp_path: Path):
    from leap_hand.stats import analyse_poses

    path = _write_scheduled_take(
        tmp_path / "glove_left_take1.jsonl", plan="open_palm:2,fist:2",
        present=lambda t: t < 2.0)
    rows = {s.pose: s for s in analyse_poses(path)}
    assert rows["fist"].frames == 0
    assert rows["fist"].detection_rate == 0.0
    assert rows["fist"].longest_loss_s == pytest.approx(2.0, abs=0.05)


def test_a_reacquisition_is_counted_inside_the_pose_it_happened_in(
        tmp_path: Path):
    from leap_hand.stats import analyse_poses

    path = _write_scheduled_take(
        tmp_path / "glove_left_take1.jsonl", plan="open_palm:2,fist:2",
        ids=lambda t: 11 if t < 3.0 else 12)
    rows = {s.pose: s for s in analyse_poses(path)}
    assert rows["open_palm"].reacquisitions == 0
    assert rows["fist"].reacquisitions == 1


def test_an_unscheduled_take_has_no_per_pose_rows_and_says_so(tmp_path: Path):
    """--recompute on last week's data: the old table, and why it is the old
    table."""
    import sys

    from leap_hand.stats import analyse_poses

    gate = _load_script("gate")
    for condition in ("bare", "glove"):
        folder = tmp_path / "gate" / condition
        folder.mkdir(parents=True)
        rec = LeapRecorder(pose=condition, take=1)      # no pose_plan
        rec.start(folder / f"{condition}_left_take1_20260916_120000.jsonl")
        shape = MockLeapStream(noise_mm=0.0, dropout_every=0,
                               reacquire_every=0, pose="open_palm")
        for _ in range(200):
            lh = dict(shape.generate(1))["left"]
            lh.visible_time_us = 5_000_000
            rec.record(lh)
        rec.stop()
        assert analyse_poses(rec.path) == [], "no plan, no per-pose rows"

    report = tmp_path / "R.txt"
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "argv", [
        "gate.py", "--recompute", "--out-dir", str(tmp_path / "gate"),
        "--report", str(report)])
    try:
        gate.main()
    finally:
        monkey.undo()

    text = report.read_text(encoding="utf-8")
    assert "Per pose" in text
    assert "was recorded under a --schedule" in text
    assert "Paired verdict per pose" not in text
    # the per-condition table is still the whole report it always was
    assert "thresholds: detection >= 80%" in text
    assert "bare" in text and "glove" in text


def test_the_gate_records_the_schedule_on_every_frame(tmp_path: Path,
                                                      monkeypatch):
    """A scheduled mock run, end to end: plan and offset on each line."""
    import json
    import sys

    gate = _load_script("gate")
    monkeypatch.setattr(gate, "beep", lambda *a, **k: None)
    out_dir = tmp_path / "gate"
    monkeypatch.setattr(sys, "argv", [
        "gate.py", "--mock", "--conditions", "bare,glove", "--prep", "0",
        "--schedule", "open_palm:1,fist:1", "--snapshots", "0",
        "--out-dir", str(out_dir), "--report", str(tmp_path / "R.txt"),
    ])
    gate.main()

    take = sorted((out_dir / "glove").glob("*.jsonl"))[0]
    rows = [json.loads(line) for line
            in take.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert {r["pose_plan"] for r in rows} == {"open_palm:1,fist:1"}
    # `pose_t` is on the CAPTURE clock, so the first frames drained after the
    # file opened can be a few milliseconds older than the schedule's start.
    # Those belong to no window and are simply outside every pose's rows.
    assert all(-0.1 <= r["pose_t"] < 2.1 for r in rows)
    assert any(r["pose_t"] >= 1.0 for r in rows), "the fist window was recorded"
    # the condition label is untouched: it is what the folder and the
    # per-condition table are keyed on
    assert {r["pose"] for r in rows} == {"glove"}

    text = (tmp_path / "R.txt").read_text(encoding="utf-8")
    assert "Per pose (schedule: open_palm:1,fist:1" in text
    assert "Paired verdict per pose" in text
    assert "open_palm" in text and "fist" in text


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
    import numpy as np

    from cam_hand.align import align_points
    from cam_hand.fusion import PALM_IDX

    i = record_frame.medoid_index(frames)
    assert 0 <= i < len(frames)
    # it IS one of the recorded frames — a real hand with real bone lengths,
    # never the mean, whose bones are shorter than any frame's
    assert frame_to_keypoints21(frames[i]) == frame_to_keypoints21(frames[i])

    # and it is the one closest to the take's mean once orientation is taken
    # out of the comparison, which is the criterion medoid_index applies
    pts = [np.asarray(frame_to_keypoints21(f), float) for f in frames]
    mean = np.mean(np.stack(pts), axis=0)

    def aligned_dist(p):
        moved, _r, _e, _s = align_points(p, mean, with_scale=False,
                                         subset=PALM_IDX)
        return float(((moved - mean) ** 2).sum())

    assert aligned_dist(pts[i]) == pytest.approx(
        min(aligned_dist(p) for p in pts))


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


# --- the medoid must rank POSE, not orientation -----------------------------
def _rotated_take(angles, pose=(15.0, 0.30), outlier=None, outlier_at=None):
    """Frames of one hand: a slow rotation about +z, optionally one bad pose.

    Built from test_fusion's 26-joint synthetic hand so the geometry is the
    one the fusion tests already reason about.
    """
    import math

    import numpy as np
    from test_fusion import _hand26

    from xr_hand.joints import HandFrame
    from xr_hand.kinematics import absolute_to_relative

    identity = [[0.0, 0.0, 0.0, 1.0]] * 26
    frames = []
    for i, deg in enumerate(angles):
        spread, curl = outlier if (outlier and i == outlier_at) else pose
        pts = np.asarray(_hand26(spread, curl), dtype=float)
        wrist = pts[JOINT_NAMES.index("WRIST")].copy()
        a = math.radians(deg)
        rot = np.array([[math.cos(a), -math.sin(a), 0.0],
                        [math.sin(a), math.cos(a), 0.0],
                        [0.0, 0.0, 1.0]])
        pts = (rot @ (pts - wrist).T).T + wrist
        frames.append(HandFrame(
            timestamp=float(i), packet_counter=i, hand_side="right",
            frame_id=i, status=0,
            joints=absolute_to_relative([list(p) for p in pts], identity)))
    return frames


def test_the_medoid_ignores_slow_rotation_and_rejects_the_bad_pose():
    """A hand held still still turns; that must not decide which frame wins.

    Twenty frames of one pose over a 10-degree drift, plus one frame in a
    plainly wrong pose at the middle of the sweep. Scored on raw
    wrist-centred coordinates, identical poses span a 48x range of "distance
    from the mean" purely because of the drift — at 100 mm from the wrist
    5 degrees is 8.7 mm, against the 0.20 mm jitter the gate measured. After
    a rigid palm alignment they are all equal, and only the pose is left to
    rank.
    """
    import numpy as np

    from cam_hand.align import align_points
    from cam_hand.fusion import PALM_IDX

    record_frame = _load_script("record_frame")
    angles = list(np.linspace(0.0, 10.0, 20))
    bad = 10
    frames = _rotated_take(angles, outlier=(0.0, 0.75), outlier_at=bad)

    chosen = record_frame.medoid_index(frames)
    assert chosen != bad, "the medoid picked the frame in the wrong pose"

    pts = [np.asarray(frame_to_keypoints21(f), float) for f in frames]
    mean = np.mean(np.stack(pts), axis=0)
    good = [i for i in range(len(frames)) if i != bad]

    def naive(i):
        return float(((pts[i] - mean) ** 2).sum())

    def aligned(i):
        moved, _r, _e, _s = align_points(pts[i], mean, with_scale=False,
                                         subset=PALM_IDX)
        return float(((moved - mean) ** 2).sum())

    # identical poses, scored purely on where they sit in the drift
    assert max(naive(i) for i in good) / min(naive(i) for i in good) > 10.0
    # ...and scored on pose alone, they are the same frame as far as this cares
    assert max(aligned(i) for i in good) / min(aligned(i) for i in good) < 1.01
    # the outlier is the worst of the take, by a wide margin
    assert aligned(bad) > 10.0 * max(aligned(i) for i in good)


def test_the_medoid_exports_the_original_frame_not_an_aligned_copy(tmp_path: Path):
    """Alignment picks the winner. What gets written is the measurement."""
    import numpy as np

    from cam_hand.prof_format import load_file

    record_frame = _load_script("record_frame")
    frames = _rotated_take(list(np.linspace(0.0, 10.0, 8)))

    take = tmp_path / "frame_99_right_take1.jsonl"
    rec = FrameRecorder(pose="frame_99", take=1)
    rec.start(take)
    for f in frames:
        rec.record(f)
    rec.stop()

    out = tmp_path / "frame_99_keypoints.txt"
    chosen = record_frame.write_prof_file(take, out)
    assert set(chosen) == {"right"}
    index, total = chosen["right"]
    assert total == len(frames)

    # the file holds exactly the frame that won, unrotated: its coordinates
    # are the ones the recorder stored, not a copy turned to face the mean
    winner, wall = list(FrameRecorder.load(take))[index]
    exporter = record_frame.glove_exporter()
    expected = exporter.frame_block(winner, wall)
    assert out.read_text(encoding="utf-8").strip() == expected

    block = load_file(out)[0]
    assert block.frame == winner.packet_counter and len(block.points) == 21
