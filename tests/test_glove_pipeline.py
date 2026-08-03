"""Sanity tests covering parser + validator against the mock generator."""
from xr_hand.joints import EXPECTED_VALUE_COUNT
from xr_hand.mock import MockHandGenerator
from xr_hand.parser import PacketParseError, parse_hand_message
from xr_hand.validator import StreamMonitor, validate_raw_message


def test_clean_mock_packet_is_valid():
    gen = MockHandGenerator(hand="right")
    raw = gen.next_frame()
    assert len(raw) == EXPECTED_VALUE_COUNT
    result = validate_raw_message(raw, prev_counter=None)
    assert result.is_valid
    assert result.errors == []


def test_wrong_length_is_error():
    gen = MockHandGenerator(hand="right")
    gen.wrong_length = True
    raw = gen.next_frame()
    result = validate_raw_message(raw)
    assert not result.is_valid
    assert any("length mismatch" in e for e in result.errors)


def test_frozen_counter_warns_via_stream_monitor():
    gen = MockHandGenerator(hand="right")
    gen.freeze_counter = True
    mon = StreamMonitor("right", freeze_threshold=3, drop_threshold=10)
    warnings = []
    for _ in range(5):
        raw = gen.next_frame()
        warnings.extend(mon.update(int(raw[0])))
    assert any("frozen" in w for w in warnings)


def test_stream_monitor_ignores_normal_batching():
    """Single repeats and small skips (real-device behaviour) should not warn."""
    mon = StreamMonitor("test", freeze_threshold=20, drop_threshold=10)
    sequence = [100, 100, 101, 102, 102, 103, 105, 105, 106]
    warnings = []
    for c in sequence:
        warnings.extend(mon.update(c))
    assert warnings == []


def test_stream_monitor_flags_backwards():
    mon = StreamMonitor("test")
    mon.update(100)
    warnings = mon.update(50)
    assert any("backwards" in w for w in warnings)


def test_zero_joint_warns():
    gen = MockHandGenerator(hand="right")
    gen.zero_joint = "MIDDLE_PROXIMAL"
    raw = gen.next_frame()
    result = validate_raw_message(raw)
    assert any("MIDDLE_PROXIMAL" in w and "all-zero" in w for w in result.warnings)


def test_nan_is_error():
    gen = MockHandGenerator(hand="right")
    raw = gen.next_frame()
    raw[10] = float("nan")
    result = validate_raw_message(raw)
    assert not result.is_valid


def test_parse_round_trip():
    gen = MockHandGenerator(hand="left")
    raw = gen.next_frame()
    frame = parse_hand_message(raw, hand_side_hint="left")
    assert frame.hand_side == "left"
    assert len(frame.joints) == 26
    assert frame.joints[0].name == "PALM"
    assert frame.joints[10].name == "INDEX_TIP"


def test_parse_rejects_wrong_length():
    try:
        parse_hand_message([0.0] * 100)
    except PacketParseError:
        return
    assert False, "should have raised"
