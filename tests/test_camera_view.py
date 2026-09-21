"""CameraView: the launcher for the live camera window (no window is opened here)."""
import py_compile
from pathlib import Path

from leap_hand.protocol import IMAGES_OFF, CameraView, image_banner


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
