"""CameraView: the launcher for the live camera window (no window is opened here)."""
import py_compile
from pathlib import Path

import importlib.util
from types import SimpleNamespace

from leap_hand.protocol import (IMAGES_OFF, NO_TRACKED_HAND, STILL_UNKNOWN, CameraView,
                                hand_crop_box, image_banner, move_still, skipped_still_path,
                                skipped_still_text, still_status)


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
