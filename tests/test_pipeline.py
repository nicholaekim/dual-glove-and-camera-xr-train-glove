"""Recording, export and the tracker-format reader."""
import json
from pathlib import Path

import pytest

from cam_hand.export21 import palm_point, prof_block, wrist_centered
from cam_hand.landmarks import MP21_NAMES, CamHand
from cam_hand.prof_format import parse_text
from cam_hand.recorder import CamRecorder, finalize_pose_name, hand_tag, slugify


def fake_hand(side="right", offset=0.0):
    img = [[float(i), float(i) * 2, 0.01 * i] for i in range(21)]
    world = [[0.01 * i + offset, 0.02 * i, 0.003 * i] for i in range(21)]
    return CamHand(hand_side=side, score=0.9, img=img, world=world)


def test_landmark_order_matches_the_glove_pipeline():
    """Both projects must speak the same 21-keypoint order or nothing lines up."""
    from xr_hand.keypoints21 import MP21_NAMES as glove_names
    assert MP21_NAMES == glove_names


def test_record_and_load_round_trip(tmp_path: Path):
    rec = CamRecorder(pose="fist", take=2)
    rec.start(tmp_path / "t.jsonl")
    rec.record(fake_hand("right"), ts_ms=1000, frame_size=(640, 480))
    rec.record(fake_hand("left"), ts_ms=1033, frame_size=(640, 480))
    rec.stop()

    frames = list(CamRecorder.load(tmp_path / "t.jsonl"))
    assert len(frames) == 2 == rec.count
    assert rec.hands_seen == {"left", "right"}
    for d in frames:
        assert d["pose"] == "fist" and d["take"] == 2
        assert len(d["world"]) == 21 and len(d["img"]) == 21
        assert d["frame_w"] == 640 and d["frame_h"] == 480


def test_recorder_throttles_each_hand_independently(tmp_path: Path, monkeypatch):
    """Each hand keeps its own schedule, so one hand cannot starve the other.

    The clock is faked. Driving this with the real clock races the throttle
    interval against however long a JSON write plus flush happens to take,
    which passes or fails depending on how busy the machine is.
    """
    import cam_hand.recorder as recorder_mod
    now = [1000.0]
    monkeypatch.setattr(recorder_mod.time, "time", lambda: now[0])

    rec = CamRecorder(hz=5.0)                       # one frame per hand per 0.2 s
    rec.start(tmp_path / "t.jsonl")
    rec.record(fake_hand("right"), 0, (640, 480))   # first right: kept
    rec.record(fake_hand("left"), 0, (640, 480))    # other hand, own schedule: kept
    now[0] += 0.1
    rec.record(fake_hand("right"), 0, (640, 480))   # 0.1 s later: too soon, dropped
    now[0] += 0.15
    rec.record(fake_hand("right"), 0, (640, 480))   # 0.25 s after its last: kept
    rec.stop()

    assert rec.count == 3
    sides = [d["hand_side"] for d in CamRecorder.load(tmp_path / "t.jsonl")]
    assert sides == ["right", "left", "right"]


def test_recorder_writes_one_valid_json_line_per_frame(tmp_path: Path):
    """Line-per-frame + flush is what makes an interrupted take still usable."""
    rec = CamRecorder()
    rec.start(tmp_path / "t.jsonl")
    for _ in range(3):
        rec.record(fake_hand(), 0, (640, 480))
    rec.stop()
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["hand_side"] == "right" for line in lines)


def test_wrist_centering_puts_the_wrist_at_the_origin():
    d = {"world": [[1.0 + 0.01 * i, 2.0, 3.0] for i in range(21)]}
    pts = wrist_centered(d)
    assert pts[0] == [0.0, 0.0, 0.0]
    assert pts[5][0] == pytest.approx(0.05)


def test_palm_point_is_the_wrist_to_middle_knuckle_midpoint():
    pts = [[0.0, 0.0, 0.0]] + [[1.0, 1.0, 1.0]] * 20
    pts[9] = [2.0, 4.0, 6.0]
    assert palm_point(pts) == [1.0, 2.0, 3.0]


def test_prof_block_matches_the_tracker_layout():
    d = {"world": [[0.001 * i, 0.002 * i, 0.003 * i] for i in range(21)],
         "hand_side": "left", "wall_time": 1_700_000_000.25}
    block = prof_block(d, frame_no=7)
    lines = block.splitlines()
    assert lines[0].startswith("Frame 7 | Hand ID: left | Time: ")
    assert lines[1] == "Wrist: (0.00, 0.00, 0.00)"
    assert lines[2].startswith("0 (Palm): ")
    assert lines[-1].startswith("20 (Pinky): ")
    assert len(lines) == 23          # header + wrist + 21 landmarks


def test_prof_reader_accepts_both_header_styles():
    theirs = """Frame 102287 | Hand ID 1232 (right)
""" + "\n".join(f"{i} (X): ({i}.5, {i}.25, -{i}.75)" for i in range(21))
    ours = """Frame 7 | Hand ID: left | Time: 2026-07-31T10:00:00.000
Wrist: (0.00, 0.00, 0.00)
""" + "\n".join(f"{i} (X): ({i}.0, 0.0, 0.0)" for i in range(21))

    a = list(parse_text(theirs))
    assert len(a) == 1 and a[0].frame == 102287 and a[0].hand == "right"
    assert a[0].hand_id == "1232"
    assert a[0].points[20] == [20.5, 20.25, -20.75]

    b = list(parse_text(ours))
    assert len(b) == 1 and b[0].hand == "left" and b[0].points[3] == [3.0, 0.0, 0.0]


def test_prof_reader_splits_multiple_frames_and_skips_partial_ones():
    good = "\n".join(f"{i} (X): (0.0, 0.0, 0.0)" for i in range(21))
    text = (f"Frame 1 | Hand ID 9 (left)\n{good}\n\n"
            f"Frame 2 | Hand ID 9 (left)\n0 (X): (1.0, 1.0, 1.0)\n\n"
            f"Frame 3 | Hand ID 9 (right)\n{good}\n")
    frames = list(parse_text(text))
    assert [f.frame for f in frames] == [1, 3]      # frame 2 was incomplete


def test_pose_filenames_are_slugged_and_tagged(tmp_path: Path):
    assert slugify("Open Palm!") == "open_palm"
    assert hand_tag({"left", "right"}) == "both"
    assert hand_tag(set()) == "nohand"

    p = tmp_path / "fist_take1_20260731_120000.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    final = finalize_pose_name(p, {"left"})
    assert final.name == "fist_left_take1_20260731_120000.jsonl"
    assert final.is_file()
