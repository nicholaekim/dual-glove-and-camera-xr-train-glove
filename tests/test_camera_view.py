"""CameraView: the launcher for the live camera window (no window is opened here)."""
import py_compile
from pathlib import Path

import importlib.util
from types import SimpleNamespace

from leap_hand.protocol import (IMAGES_OFF, NO_TRACKED_HAND, STILL_UNKNOWN, CameraView,
                                crop_hand, full_still_path, hand_crop_box, image_banner,
                                move_still, parse_status, paste_reference, reference_label,
                                reference_path, save_still, status_text, skipped_still_path,
                                skipped_still_text, still_capture_meta, still_capture_path,
                                still_capture_record, still_status, write_still_capture)


def test_viewer_script_exists_and_compiles():
    assert CameraView.SCRIPT.exists()
    py_compile.compile(str(CameraView.SCRIPT), doraise=True)


def test_disabled_view_is_a_noop():
    v = CameraView(hand="left", enabled=False).start()
    assert v.enabled is False and v._proc is None
    v.caption("anything")          # must not raise or create files
    v.close()


def test_caption_writes_text_and_band_and_skips_repeats(tmp_path):
    v = CameraView(hand="right", band=(18, 28), enabled=True)
    v._status = tmp_path / "status.txt"      # as start() would set, without spawning the window
    v.caption("REC  FIST   4s", band=(15, 25))
    assert v._status.read_text(encoding="utf-8").splitlines() == ["REC  FIST   4s", "band=15,25"]
    stamp = v._status.stat().st_mtime_ns
    v.caption("REC  FIST   4s", band=(15, 25))   # unchanged: not rewritten
    assert v._status.stat().st_mtime_ns == stamp
    v._proc = None
    v.close()
    assert not (tmp_path / "status.txt").exists()


def test_unknown_hand_means_both():
    assert CameraView(hand=None, enabled=False).hand == "both"


# --- the reference picture of the pose being asked for --------------------------

def test_status_round_trips_the_reference_picture():
    ref = "C:\\repo\\reference_poses\\crossed_fingers.png"
    text = status_text("HOLD: CROSSED FINGERS   REC 5s", band=(18, 28), ref=ref)
    assert parse_status(text) == ("HOLD: CROSSED FINGERS   REC 5s", (18.0, 28.0), None, ref)
    assert parse_status(status_text("ACQUIRE"))[3] is None          # no ref: None


def test_view_client_sends_the_reference_and_drops_it_when_none(tmp_path):
    v = CameraView(hand="left", band=(18, 28), enabled=True)
    v._status = tmp_path / "status.txt"
    ref = tmp_path / "fist.png"
    v.caption("HOLD: FIST   REC 4s", band=(18, 28), ref=ref)
    assert parse_status(v._status.read_text(encoding="utf-8"))[3] == str(ref)
    v.snapshot(tmp_path / "stills" / "fist_take1.jpg")              # a still keeps the ref
    assert parse_status(v._status.read_text(encoding="utf-8"))[2:] == \
        (str(tmp_path / "stills" / "fist_take1.jpg"), str(ref))
    v.caption("idle", band=(18, 28))                                # no ref: none sent
    assert "ref=" not in v._status.read_text(encoding="utf-8")
    v._proc = None
    v.close()


def test_reference_path_only_for_a_pose_phase_and_an_existing_file(tmp_path):
    (tmp_path / "fist.png").write_bytes(b"png")
    for phase in ("ACQUIRE", "SETTLE", "REC"):
        assert reference_path(tmp_path, phase, "fist") == str(tmp_path / "fist.png")
    assert reference_path(tmp_path, "idle", "fist") is None
    assert reference_path(tmp_path, "REC", "") is None
    assert reference_path(tmp_path, "REC", "pinch") is None           # no file: nothing
    assert reference_label(tmp_path / "crossed_fingers.png") == "TARGET: CROSSED FINGERS"


def test_paste_reference_draws_only_the_top_right_and_leaves_the_clean_frame():
    import numpy as np
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (768, 768, 3), dtype=np.uint8)
    clean = frame.copy()                         # what compose() keeps for the still
    before = frame.copy()
    ref_img = np.full((180, 240, 3), 200, np.uint8)
    said = []

    def put(f, text, org):
        said.append((text, org))
        x, y = org
        f[y - 17:y, x:758] = 255                 # a label-sized mark, like cv2.putText

    box = paste_reference(frame, ref_img, "TARGET: FIST", put)
    assert box == (518, 124, 758, 304)
    x0, y0, x1, y1 = box
    assert np.array_equal(frame[y0:y1, x0:x1], ref_img)
    assert said == [("TARGET: FIST", (518, 114))]   # over the picture, one line above
    changed = np.argwhere((frame != before).any(axis=2))
    assert changed[:, 0].min() >= 124 - 10 - 17 and changed[:, 0].max() < y1
    assert changed[:, 1].min() >= 384 and changed[:, 1].max() < x1   # the right half only
    assert np.array_equal(clean, before)         # a copy taken before is untouched
    # a long label runs left past the picture, never off the frame
    said.clear()
    paste_reference(before.copy(), ref_img, "TARGET: THUMB OPPOSITION", put)
    assert said[0][1] == (758 - 17 * 24, 114)
    # a picture wider than the frame is not drawn at all
    assert paste_reference(before.copy(), np.zeros((10, 800, 3), np.uint8), "x", put) is None


# --- the "camera images are off" banner --------------------------------------
# Every still in recordings/sync_day2/stills/ carries that warning printed over
# a perfectly good IR picture of the hand, because the viewer decided it from
# the reply to the policy request instead of from whether an image arrived.

def test_no_banner_while_images_are_arriving():
    started = 1000.0
    # an image 0.2 s ago: the picture is there, so the window says nothing
    assert image_banner(1010.0, 1009.8, started) == ""
    # still nothing at the edge of the stale window
    assert image_banner(1010.0, 1007.0, started, stale_s=3.0) == ""


def test_banner_only_after_a_few_seconds_with_no_image():
    started = 1000.0
    # a window that has only just opened accuses nobody
    assert image_banner(1001.0, None, started) == ""
    # no image at all, grace period over: the policy really is off
    assert image_banner(1005.0, None, started) == IMAGES_OFF
    assert "Allow Images" in IMAGES_OFF


def test_images_that_stop_are_reported_as_stopping_not_as_a_setting():
    """A different fault needs a different sentence: the setting IS on."""
    banner = image_banner(1030.0, 1020.0, started=1000.0)
    assert banner and banner != IMAGES_OFF
    assert "stopped" in banner and "10 s" in banner


# --- the per-take still, cropped to the hand -----------------------------------
# A still of the whole 768 px frame has the operator's face in it, so the
# viewer saves only the box hand_crop_box gives around the tracked hand.

def _inside(box, size):
    x0, y0, x1, y1 = box
    return 0 <= x0 < x1 <= size and 0 <= y0 < y1 <= size


def _contains(box, points):
    x0, y0, x1, y1 = box
    return all(x0 <= x < x1 and y0 <= y < y1 for x, y in points)


def test_crop_of_a_normal_hand_has_the_margin_on_every_side():
    pts = [(300, 300), (400, 300), (300, 450), (400, 450)]
    box = hand_crop_box(pts, 768)
    assert box == (247, 247, 454, 504)
    assert all(isinstance(v, int) for v in box)
    x0, y0, x1, y1 = box
    pad = 0.35 * 151                             # the larger side, 300..450 inclusive
    assert 300 - x0 >= pad and x1 - 401 >= pad
    assert 300 - y0 >= pad and y1 - 451 >= pad


def test_crop_of_a_small_hand_is_at_least_min_side_and_stays_in_the_frame():
    pts = [(500, 600), (520, 610)]
    box = hand_crop_box(pts, 768)
    x0, y0, x1, y1 = box
    assert x1 - x0 >= 192 and y1 - y0 >= 192
    assert _inside(box, 768) and _contains(box, pts)
    # the same small hand at the bottom edge: slid back in, not cut short
    pts = [(10, 760), (30, 765)]
    box = hand_crop_box(pts, 768)
    x0, y0, x1, y1 = box
    assert x0 == 0 and y1 == 768
    assert x1 - x0 >= 192 and y1 - y0 >= 192
    assert _contains(box, pts)


def test_crop_near_a_corner_is_clamped_to_the_frame():
    pts = [(5, 5), (120, 140)]
    box = hand_crop_box(pts, 768)
    assert box[:2] == (0, 0)
    assert _inside(box, 768) and _contains(box, pts)
    # a hand filling the frame: the margin runs off every edge and is cut
    assert hand_crop_box([(0, 0), (767, 767)], 768) == (0, 0, 768, 768)
    # a frame smaller than min_side: the whole frame, never past it
    assert hand_crop_box(pts, 100) == (0, 0, 100, 100)


def test_no_hand_in_the_picture_means_no_crop():
    assert hand_crop_box([], 768) is None
    # projected points that all fall outside the image are not a hand to crop to
    assert hand_crop_box([(-5, 3), (800, 10)], 768) is None


# The still is a photo: the crop is the frame's own pixels in hand_crop_box's
# box, with nothing drawn on it (no skeleton, no ring, no caption).

def test_crop_hand_is_the_frame_region_untouched():
    import numpy as np
    rng = np.random.default_rng(0)
    plain = rng.integers(0, 256, (768, 768, 3), dtype=np.uint8)
    source = plain.copy()
    pts = [(300, 300), (400, 300), (300, 450), (400, 450)]
    crop, box = crop_hand(plain, pts)
    assert box == hand_crop_box(pts, 768)
    x0, y0, x1, y1 = box
    assert np.array_equal(crop, source[y0:y1, x0:x1])
    assert int(np.count_nonzero(crop != source[y0:y1, x0:x1])) == 0
    assert np.array_equal(plain, source)               # the frame is not drawn on either
    crop[:] = 0                                        # a copy, not a view into the frame
    assert np.array_equal(plain, source)


def test_crop_hand_with_no_hand_in_the_picture_is_none():
    import numpy as np
    plain = np.zeros((768, 768, 3), np.uint8)
    assert crop_hand(plain, []) == (None, None)
    assert crop_hand(plain, [(-5, 3), (800, 10)]) == (None, None)


# --- a still that is missing, and why ----------------------------------------
# A still skipped because no hand was tracked leaves `<stem>.skipped.txt`
# behind; no JPEG and no note is a failure nobody recorded. The meta.json says
# which, so the two are not confused when the takes are audited.

def test_still_status_tells_skipped_from_unknown(tmp_path):
    still = tmp_path / "stills" / "fist_take1_x.jpg"
    assert still_status(still) == (None, STILL_UNKNOWN)
    note = skipped_still_path(still)
    assert note.name == "fist_take1_x.skipped.txt"
    note.parent.mkdir(parents=True)
    note.write_text(skipped_still_text(NO_TRACKED_HAND, 1790000000.0), encoding="utf-8")
    assert note.read_text(encoding="utf-8").splitlines()[0] == "still_missing_reason=no_tracked_hand"
    assert still_status(still) == (None, NO_TRACKED_HAND)
    still.write_bytes(b"jpeg")                   # a JPEG is the still, whatever else is there
    assert still_status(still) == ("fist_take1_x.jpg", None)
    note.write_text("garbled", encoding="utf-8")
    still.unlink()
    assert still_status(still) == (None, STILL_UNKNOWN)


def test_the_note_moves_with_the_still(tmp_path):
    src = tmp_path / "stills" / "fist_take1_x.jpg"
    dst = tmp_path / "rejected" / "stills" / "fist_left_take1_x_attempt1.jpg"
    move_still(src, dst)                         # nothing there: nothing created
    assert not (tmp_path / "rejected").exists()
    src.parent.mkdir(parents=True)
    skipped_still_path(src).write_text(skipped_still_text(NO_TRACKED_HAND, 0.0), encoding="utf-8")
    move_still(src, dst)
    assert not skipped_still_path(src).exists()
    assert still_status(dst) == (None, NO_TRACKED_HAND)


def _recorder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "record_simultaneous.py"
    spec = importlib.util.spec_from_file_location("repo_script_record_simultaneous", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorder_writes_why_a_still_is_missing(tmp_path):
    sync = _recorder()
    session = sync.CoachedLeapSession
    still = tmp_path / "stills" / "fist_take1_x.jpg"
    still.parent.mkdir(parents=True)
    skipped_still_path(still).write_text(skipped_still_text(NO_TRACKED_HAND, 0.0), encoding="utf-8")
    assert session._rename_still(still, "fist_left_take1_x") == ("", NO_TRACKED_HAND)
    assert (tmp_path / "stills" / "fist_left_take1_x.skipped.txt").is_file()

    me = SimpleNamespace(hand="left", _pose="fist", _take=1, band=(18, 28), settle=1.5)
    def meta(**kw):
        return session._meta_dict(me, sync.Attempt(**kw), accepted=True, file="f.jsonl",
                                  still=kw.get("still", ""), attempts=1)
    skipped = meta(still_missing=NO_TRACKED_HAND)
    assert skipped["still"] is None and skipped["still_missing_reason"] == "no_tracked_hand"
    lost = meta()                                # no JPEG, no note
    assert lost["still"] is None and lost["still_missing_reason"] == "unknown"
    kept = meta(still="fist_left_take1_x.jpg")
    assert kept["still"] == str(Path("stills") / "fist_left_take1_x.jpg")
    assert kept["still_missing_reason"] is None


# --- when the still was taken ---------------------------------------------------
# The still is the photo a paired-pose comparison is timed by, so the viewer
# writes `<still>.json` beside it with the image frame's own capture instant.

NO_CAPTURE = {"still_capture_time": None, "still_timestamp_us": None,
              "still_frame_id": None, "still_capture_time_source": None}


def test_the_capture_record_comes_from_the_image_frame_or_says_it_does_not():
    rec = still_capture_record(saved_at=1790000000.5, capture_time=1790000000.42,
                               timestamp_us=194619959785, frame_id=3543)
    assert rec == {"capture_time": 1790000000.42, "capture_time_source": "leapc_image_timestamp",
                   "timestamp_us": 194619959785, "frame_id": 3543, "saved_at": 1790000000.5,
                   "full": None}
    fallback = still_capture_record(saved_at=1790000000.5)
    assert fallback["capture_time"] == 1790000000.5
    assert fallback["capture_time_source"] == "saved_at"
    assert fallback["timestamp_us"] is None and fallback["frame_id"] is None


def test_the_capture_sidecar_is_written_beside_the_still_and_read_into_meta_keys(tmp_path):
    still = tmp_path / "stills" / "fist_take1_x.jpg"
    assert still_capture_path(still) == tmp_path / "stills" / "fist_take1_x.jpg.json"
    assert still_capture_meta(still) == NO_CAPTURE               # nothing there: all null
    write_still_capture(still, still_capture_record(saved_at=20.0, capture_time=19.96,
                                                    timestamp_us=555, frame_id=9))
    assert still_capture_meta(still) == {"still_capture_time": 19.96, "still_timestamp_us": 555,
                                         "still_frame_id": 9,
                                         "still_capture_time_source": "leapc_image_timestamp"}
    still_capture_path(still).write_text("{torn", encoding="utf-8")
    assert still_capture_meta(still) == NO_CAPTURE               # unreadable: null, not a crash


def test_the_capture_sidecar_moves_with_the_still(tmp_path):
    src = tmp_path / "stills" / "fist_take1_x.jpg"
    dst = tmp_path / "rejected" / "stills" / "fist_left_take1_x_attempt1.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"jpeg")
    write_still_capture(src, still_capture_record(saved_at=5.0))
    move_still(src, dst)
    assert not still_capture_path(src).exists()
    assert still_capture_meta(dst)["still_capture_time"] == 5.0


def test_recorder_copies_the_still_capture_into_the_meta(tmp_path):
    sync = _recorder()
    session = sync.CoachedLeapSession
    stills = tmp_path / "stills"
    still = stills / "fist_take1_x.jpg"
    still.parent.mkdir(parents=True)
    still.write_bytes(b"jpeg")
    write_still_capture(still, still_capture_record(saved_at=31.0, capture_time=30.95,
                                                    timestamp_us=77, frame_id=12))
    assert session._rename_still(still, "fist_left_take1_x") == ("fist_left_take1_x.jpg", "")
    assert still_capture_path(stills / "fist_left_take1_x.jpg").is_file()

    me = SimpleNamespace(hand="left", _pose="fist", _take=1, band=(18, 28), settle=1.5,
                         stills_dir=stills, rejected_dir=tmp_path / "rejected")
    def meta(accepted=True, **kw):
        return session._meta_dict(me, sync.Attempt(**kw), accepted=accepted, file="f.jsonl",
                                  still=kw.get("still", ""), attempts=1)
    kept = meta(still="fist_left_take1_x.jpg")
    assert (kept["still_capture_time"], kept["still_timestamp_us"], kept["still_frame_id"],
            kept["still_capture_time_source"]) == (30.95, 77, 12, "leapc_image_timestamp")
    # a rejected attempt's still lives under rejected/stills, and so does its sidecar
    move_still(stills / "fist_left_take1_x.jpg",
               tmp_path / "rejected" / "stills" / "fist_left_take1_x_attempt1.jpg")
    refused = meta(accepted=False, still="fist_left_take1_x_attempt1.jpg")
    assert refused["still_capture_time"] == 30.95
    # no still, or an old session without sidecars: the keys are there and null
    for got in (meta(), meta(still="fist_left_take2_y.jpg")):
        assert {k: got[k] for k in NO_CAPTURE} == NO_CAPTURE


# --- the uncropped frame kept beside the cropped still ------------------------------
# The viewer crops the still to the hand; the whole frame it was cut from is
# kept too, `<stem>.full.jpg`, from the same frame object and so the same moment.

def _fake_imwrite(written):
    def imwrite(path, array):
        written.append((path, array))
        Path(path).write_bytes(b"jpeg")
        return True
    return imwrite


def test_save_still_writes_crop_full_frame_and_one_sidecar_for_both(tmp_path):
    import numpy as np
    plain = np.arange(768 * 768 * 3, dtype=np.uint32).reshape(768, 768, 3).astype(np.uint8)
    crop = plain[100:300, 200:400].copy()
    still = tmp_path / "stills" / "fist_take1_x.jpg"
    still.parent.mkdir(parents=True)
    written = []
    save_still(still, crop, (19.96, 555, 9), 20.0, _fake_imwrite(written),
               full=plain, cut_from=plain)
    assert full_still_path(still) == tmp_path / "stills" / "fist_take1_x.full.jpg"
    # the full frame is the very array the crop came from, written uncropped
    assert [Path(p).name for p, _ in written] == ["fist_take1_x.full.jpg", "fist_take1_x.jpg"]
    assert written[0][1] is plain and written[1][1] is crop
    rec = still_capture_meta(still)
    assert rec["still_capture_time"] == 19.96 and rec["still_frame_id"] == 9
    import json
    side = json.loads(still_capture_path(still).read_text(encoding="utf-8"))
    assert side["full"] == "fist_take1_x.full.jpg" and side["capture_time"] == 19.96


def test_save_still_refuses_a_crop_from_another_frame_object(tmp_path):
    import numpy as np
    import pytest
    plain = np.zeros((768, 768, 3), np.uint8)
    other = plain.copy()                         # equal pixels, another frame object
    still = tmp_path / "fist_take1_x.jpg"
    written = []
    with pytest.raises(ValueError):
        save_still(still, plain[:10, :10].copy(), (1.0, 2, 3), 4.0, _fake_imwrite(written),
                   full=plain, cut_from=other)
    assert written == [] and not still_capture_path(still).exists()
    # the whole-frame still (--still full) has no separate full frame: `full` null
    save_still(still, plain, (1.0, 2, 3), 4.0, _fake_imwrite(written))
    import json
    assert json.loads(still_capture_path(still).read_text(encoding="utf-8"))["full"] is None
    assert not full_still_path(still).exists()


def test_the_full_frame_moves_with_the_still_and_the_sidecar_names_it(tmp_path):
    import json
    src = tmp_path / "stills" / "fist_take1_x.jpg"
    src.parent.mkdir(parents=True)
    written = []
    save_still(src, b"crop", (5.0, 1, 2), 6.0, _fake_imwrite(written), full=b"f", cut_from=b"f")
    # accepted: renamed in place, the full frame and the sidecar with it
    kept = tmp_path / "stills" / "fist_left_take1_x.jpg"
    move_still(src, kept)
    assert full_still_path(kept).is_file() and not full_still_path(src).exists()
    side = json.loads(still_capture_path(kept).read_text(encoding="utf-8"))
    assert side["full"] == "fist_left_take1_x.full.jpg" and side["capture_time"] == 5.0
    # rejected: moved under rejected/stills, again together
    rej = tmp_path / "rejected" / "stills" / "fist_left_take1_x_attempt1.jpg"
    move_still(kept, rej)
    assert full_still_path(rej).is_file() and not full_still_path(kept).exists()
    assert json.loads(still_capture_path(rej).read_text(encoding="utf-8"))["full"] == \
        "fist_left_take1_x_attempt1.full.jpg"


def test_recorder_meta_names_the_full_frame_or_null(tmp_path):
    sync = _recorder()
    session = sync.CoachedLeapSession
    stills = tmp_path / "stills"
    stills.mkdir()
    save_still(stills / "fist_take1_x.jpg", b"crop", (30.95, 77, 12), 31.0,
               _fake_imwrite([]), full=b"f", cut_from=b"f")
    assert session._rename_still(stills / "fist_take1_x.jpg", "fist_left_take1_x") == \
        ("fist_left_take1_x.jpg", "")
    me = SimpleNamespace(hand="left", _pose="fist", _take=1, band=(18, 28), settle=1.5,
                         stills_dir=stills, rejected_dir=tmp_path / "rejected")

    def meta(accepted=True, **kw):
        return session._meta_dict(me, sync.Attempt(**kw), accepted=accepted, file="f.jsonl",
                                  still=kw.get("still", ""), attempts=1)
    assert meta(still="fist_left_take1_x.jpg")["still_full"] == \
        str(Path("stills") / "fist_left_take1_x.full.jpg")
    move_still(stills / "fist_left_take1_x.jpg",
               tmp_path / "rejected" / "stills" / "fist_left_take1_x_attempt1.jpg")
    assert meta(accepted=False, still="fist_left_take1_x_attempt1.jpg")["still_full"] == \
        str(Path("rejected") / "stills" / "fist_left_take1_x_attempt1.full.jpg")
    assert meta()["still_full"] is None                      # no still, no full frame
    (stills / "fist_left_take2_y.jpg").write_bytes(b"jpeg")  # a still without a full frame
    assert meta(still="fist_left_take2_y.jpg")["still_full"] is None


def test_first_take_auto_continues_after_every_take_on_disk(tmp_path):
    sync = _recorder()
    assert sync.next_take(tmp_path, "open_palm", "left") == 1
    (tmp_path / "leap").mkdir()
    (tmp_path / "leap" / "open_palm_left_take1_20261104_100000.meta.json").write_text("{}")
    (tmp_path / "leap" / "open_palm_right_take7_20261104_100000.meta.json").write_text("{}")
    (tmp_path / "leap" / "fist_left_take9_20261104_100000.meta.json").write_text("{}")
    assert sync.next_take(tmp_path, "open_palm", "left") == 2
    # a take whose every attempt was rejected still holds its number
    (tmp_path / "rejected" / "glove").mkdir(parents=True)
    (tmp_path / "rejected" / "glove" /
     "open_palm_left_take2_20261104_100100_attempt3.jsonl").write_text("")
    assert sync.next_take(tmp_path, "open_palm", "left") == 3
    assert sync.next_take(tmp_path, "fist", "left") == 10
    assert sync.next_take(tmp_path, "pinch", "left") == 1
