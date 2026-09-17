"""CameraView: the launcher for the live camera window (no window is opened here)."""
import py_compile
from pathlib import Path

from leap_hand.protocol import CameraView


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
