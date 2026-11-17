"""`xr_hand.app_screenshot`: a picture of another application's window.

Gate R1 of GATES_glove_recording.md. The window under test is a Tk window the
test opens itself, so nothing here needs XR Train to be running; that is gate
R3. Skipped off Windows and where Tk cannot open a window.
"""
import sys
import time

import pytest

from xr_hand import app_screenshot as shot

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows window capture")
TITLE = "app-screenshot-test-window-7c1e"


@pytest.fixture(scope="module")
def _root():
    """One Tk window for the whole file: this Python's Tcl sometimes cannot
    start a second interpreter in the same process, and a skipped capture test
    proves nothing."""
    tk = pytest.importorskip("tkinter")
    root, error = None, None
    for _ in range(3):
        try:
            root = tk.Tk()
            break
        except tk.TclError as e:
            error = e
            time.sleep(0.3)
    if root is None:
        pytest.skip(f"Tk cannot open a window here: {error}")
    root.title(TITLE)
    root.geometry("320x240+60+60")
    root.configure(bg="#d06010")
    yield root
    root.destroy()


@pytest.fixture
def window(_root):
    """The orange window, restored, on top, mapped and drawn."""
    _root.deiconify()
    _root.attributes("-topmost", True)
    for _ in range(6):
        _root.update_idletasks()
        _root.update()
        time.sleep(0.05)
    return _root


def test_a_missing_window_gives_a_reason_and_no_file(tmp_path):
    out = tmp_path / "none.png"
    rec = shot.capture_window("no such window 9f3a2b", out)
    assert rec["file"] is None and rec["missing_reason"] == shot.NOT_FOUND
    assert rec["window"] == "no such window 9f3a2b"
    assert rec["captured_at"] is None and rec["black"] is None
    assert not out.exists()


def test_the_window_is_saved_with_its_time_method_size_and_owner(window, tmp_path):
    from PIL import Image
    out = tmp_path / "shots" / "take.png"
    t0 = time.time()
    rec = shot.capture_window(TITLE, out)
    t1 = time.time()
    assert rec["missing_reason"] is None and rec["file"] == "take.png"
    assert out.is_file(), "the folder is made and the PNG written"
    assert t0 <= rec["captured_at"] <= t1
    assert rec["method"] in ("PrintWindow", "screen")
    assert rec["window"] == TITLE
    assert rec["process"] and "python" in rec["process"].lower()
    assert rec["black"] is False
    with Image.open(out) as im:
        assert im.format == "PNG" and im.size == (rec["width"], rec["height"])
        assert im.width >= 320 and im.height >= 240
        r, g, b = im.convert("RGB").getpixel((im.width // 2, im.height // 2))
        assert r > 150 and g < 150 and b < 90, "the window's own orange, not something else"


def test_the_title_must_match_exactly_never_in_part(window):
    assert shot.find_window(TITLE) is not None
    assert shot.find_window(TITLE.upper()) is not None, "case does not matter"
    assert shot.find_window(TITLE[:-1]) is None, "a shorter title is another window"
    assert shot.find_window(TITLE + " - Google Chrome") is None
    assert TITLE in shot.window_titles()


def test_a_minimised_window_is_not_captured(window, tmp_path):
    window.attributes("-topmost", False)
    window.iconify()
    for _ in range(5):
        window.update()
        time.sleep(0.05)
    out = tmp_path / "min.png"
    rec = shot.capture_window(TITLE, out)
    assert rec["file"] is None and rec["missing_reason"] == shot.MINIMISED
    assert rec["process"], "the window was found; it is just not drawing"
    assert not out.exists()


def _solid(colour):
    from PIL import Image

    def fake(*_args):
        return Image.new("RGB", (40, 30), colour), time.time()
    return fake


def test_a_black_capture_is_saved_and_flagged_not_passed_off(window, tmp_path, monkeypatch):
    monkeypatch.setattr(shot, "_print_window", _solid((0, 0, 0)))
    monkeypatch.setattr(shot, "_grab_screen", _solid((0, 0, 0)))
    out = tmp_path / "black.png"
    rec = shot.capture_window(TITLE, out)
    assert rec["file"] == "black.png" and out.is_file()
    assert rec["black"] is True and rec["method"] == "PrintWindow"


def test_the_screen_copy_is_used_only_when_the_window_is_on_top(window, tmp_path, monkeypatch):
    monkeypatch.setattr(shot, "_print_window", _solid((0, 0, 0)))
    monkeypatch.setattr(shot, "_grab_screen", _solid((250, 250, 250)))
    # covered: the screen there shows another window, so it is not used
    monkeypatch.setattr(shot, "_uncovered", lambda *a: False)
    rec = shot.capture_window(TITLE, tmp_path / "covered.png")
    assert rec["method"] == "PrintWindow" and rec["black"] is True
    # on top: the screen copy is this window
    monkeypatch.setattr(shot, "_uncovered", lambda *a: True)
    rec = shot.capture_window(TITLE, tmp_path / "ontop.png")
    assert rec["method"] == "screen" and rec["black"] is False


def test_the_real_uncovered_check_sees_a_topmost_window(window):
    from ctypes import wintypes as wt
    import ctypes
    hwnd = shot.find_window(TITLE)
    rect = wt.RECT()
    ctypes.windll.user32.GetWindowRect(wt.HWND(hwnd), ctypes.byref(rect))
    assert shot._uncovered(hwnd, rect.left, rect.top, rect.right, rect.bottom)


def test_an_error_inside_the_capture_is_a_reason_not_an_exception(window, tmp_path, monkeypatch):
    def boom(*_a):
        raise OSError("device context lost")
    monkeypatch.setattr(shot, "_print_window", boom)
    rec = shot.capture_window(TITLE, tmp_path / "x.png")
    assert rec["file"] is None
    assert rec["missing_reason"] == "capture error: OSError: device context lost"


def test_is_black_reads_the_brightest_pixel():
    from PIL import Image
    assert shot.is_black(Image.new("RGB", (4, 4), (0, 0, 0)))
    assert shot.is_black(Image.new("RGB", (4, 4), (shot.BLACK_MAX,) * 3))
    im = Image.new("RGB", (4, 4), (0, 0, 0))
    im.putpixel((1, 1), (0, 200, 0))
    assert not shot.is_black(im)
