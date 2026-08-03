"""Round-trip tests for FrameRecorder + pose-labeling helpers."""
import json
import tempfile
from pathlib import Path

from xr_hand.mock import MockHandGenerator
from xr_hand.parser import parse_hand_message
from xr_hand.recorder import (
    FrameRecorder,
    finalize_pose_name,
    hand_tag,
    pose_filename,
    slugify,
)
from xr_hand.validator import validate_raw_message


def _make_frames(hand: str, n: int):
    gen = MockHandGenerator(hand=hand)
    frames = []
    prev = None
    for _ in range(n):
        raw = gen.next_frame()
        assert validate_raw_message(raw, prev).is_valid
        frame = parse_hand_message(raw, hand_side_hint=hand)
        prev = frame.packet_counter
        frames.append(frame)
    return frames


def test_round_trip():
    original = _make_frames("right", 30)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)

    rec = FrameRecorder()
    rec.start(path)
    for frame in original:
        rec.record(frame)
    rec.stop()
    assert rec.count == 30

    loaded = [frame for frame, _ in FrameRecorder.load(path)]
    assert len(loaded) == 30
    assert loaded[0].hand_side == "right"
    assert loaded[0].joints[0].name == "PALM"
    assert loaded[-1].packet_counter == original[-1].packet_counter

    for orig, back in zip(original, loaded):
        assert orig.packet_counter == back.packet_counter
        assert orig.hand_side == back.hand_side
        for jo, jb in zip(orig.joints, back.joints):
            assert jo.name == jb.name
            assert abs(jo.x - jb.x) < 1e-9


def test_both_hands_interleaved():
    left_frames = _make_frames("left", 10)
    right_frames = _make_frames("right", 10)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)

    rec = FrameRecorder()
    rec.start(path)
    for l, r in zip(left_frames, right_frames):
        rec.record(l)
        rec.record(r)
    rec.stop()
    assert rec.count == 20

    loaded = [frame for frame, _ in FrameRecorder.load(path)]
    sides = [f.hand_side for f in loaded]
    assert sides[::2] == ["left"] * 10
    assert sides[1::2] == ["right"] * 10


def test_wall_times_are_monotonic():
    frames = _make_frames("right", 20)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)

    rec = FrameRecorder()
    rec.start(path)
    for frame in frames:
        rec.record(frame)
    rec.stop()

    wall_times = [wt for _, wt in FrameRecorder.load(path)]
    assert all(b >= a for a, b in zip(wall_times, wall_times[1:]))


def test_pose_and_take_stamped_into_every_frame():
    frames = _make_frames("left", 5)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)

    rec = FrameRecorder(pose="fist", take=2)
    rec.start(path)
    for frame in frames:
        rec.record(frame)
    rec.stop()
    assert rec.hands_seen == {"left"}

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 5
    assert all(d["pose"] == "fist" and d["take"] == 2 for d in lines)

    # labeled files must still load through the normal path
    loaded = [frame for frame, _ in FrameRecorder.load(path)]
    assert len(loaded) == 5
    assert loaded[0].hand_side == "left"


def test_unlabeled_recording_has_no_pose_keys():
    frames = _make_frames("right", 3)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)

    rec = FrameRecorder()
    rec.start(path)
    for frame in frames:
        rec.record(frame)
    rec.stop()

    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert all("pose" not in d and "take" not in d for d in lines)


def test_hand_tag():
    assert hand_tag({"left"}) == "left"
    assert hand_tag({"right"}) == "right"
    assert hand_tag({"left", "right"}) == "both"
    assert hand_tag(set()) == "nohand"


def test_slugify():
    assert slugify("Open Palm!") == "open_palm"
    assert slugify("thumbs-up") == "thumbs_up"
    assert slugify("  fist  ") == "fist"


def test_keypoints21_mapping():
    from xr_hand.joints import JOINT_NAMES
    from xr_hand.keypoints21 import (
        MP21_NAMES,
        MP21_TO_OPENXR,
        MP21_TO_OPENXR_IDX,
        frame_to_keypoints21,
    )

    assert len(MP21_NAMES) == 21
    assert len(set(MP21_TO_OPENXR_IDX)) == 21  # no duplicate source joints
    # spot-check anatomical landmarks
    pairs = dict(MP21_TO_OPENXR)
    assert pairs["WRIST"] == "WRIST"
    assert pairs["THUMB_CMC"] == "THUMB_METACARPAL"
    assert pairs["INDEX_FINGER_MCP"] == "INDEX_PROXIMAL"
    assert pairs["PINKY_TIP"] == "LITTLE_TIP"
    # dropped joints are exactly PALM + the four non-thumb metacarpals
    used = {xr for _, xr in MP21_TO_OPENXR}
    dropped = set(JOINT_NAMES) - used
    assert dropped == {"PALM", "INDEX_METACARPAL", "MIDDLE_METACARPAL",
                       "RING_METACARPAL", "LITTLE_METACARPAL"}

    frame = _make_frames("right", 1)[0]
    pts = frame_to_keypoints21(frame)
    assert len(pts) == 21
    assert pts[0] == (0.0, 0.0, 0.0)  # wrist-centred
    # fingertips must not collapse onto the wrist
    assert all(abs(pts[i][0]) + abs(pts[i][1]) + abs(pts[i][2]) > 0.01
               for i in (4, 8, 12, 16, 20))


def test_finalize_pose_name_inserts_hand():
    with tempfile.TemporaryDirectory() as tmp:
        name = pose_filename("fist", 1)
        assert name.startswith("fist_take1_") and name.endswith(".jsonl")
        path = Path(tmp) / name
        path.write_text("{}\n")
        final = finalize_pose_name(path, {"left"})
        assert final.name.startswith("fist_left_take1_")
        assert final.exists() and not path.exists()
