"""A screenshot of another application's window, for the take it belongs to.

The recorder saves what the two sensors streamed; this saves what the glove's
own application (StretchSense XR Train) SHOWED at that moment, as a PNG, so a
recorded pose can be set beside the hand model the application drew.

    capture_window("XR Train", path) -> dict

The window is found by its exact title (case-insensitive). An exact match is
required on purpose: "XR Train" is also part of a browser tab or a folder
window named "xr trainer", and a screenshot of the wrong window that looks
like evidence is worse than no screenshot. The window is drawn with
PrintWindow(PW_RENDERFULLCONTENT), which asks the compositor for the window's
own content and so works while another window (the recorder's console, the
camera window) covers it. If that comes back black, the same rectangle is
copied off the screen instead, which needs the window uncovered. A capture
that is still black is saved and flagged `black: true`; it is never reported
as a picture of the hand. A minimised window is not captured (it is not
drawing).

The dict says what happened and is what goes into a take's meta.json:

    file            the PNG's name, None when there is none
    missing_reason  why there is none, None when there is one
    captured_at     time.time() just before the window was drawn
    method          "PrintWindow" or "screen"
    window          the title asked for
    process         the executable that owns the window
    width, height   pixels
    black           True when no pixel is brighter than BLACK_MAX

Run as a module to try it:  python -m xr_hand.app_screenshot "XR Train" out.png
"""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
from typing import List, Optional

NOT_WINDOWS = "not a Windows machine"
NOT_FOUND = "window not found"
MINIMISED = "window minimised"
NO_AREA = "window has no area"
FAILED = "the window could not be drawn"

BLACK_MAX = 8                    # brightest pixel of a capture counted as black
PW_RENDERFULLCONTENT = 0x00000002
_DPI_PER_MONITOR_V2 = -4         # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2


def empty_record(title: str, reason: Optional[str] = None) -> dict:
    return {"file": None, "missing_reason": reason, "captured_at": None,
            "method": None, "window": title, "process": None,
            "width": None, "height": None, "black": None}


def meta_keys(record: dict, window: str, folder, unknown: str = "unknown") -> dict:
    """The meta.json keys about one take's application screenshot, the same
    for every recorder. `folder` is where the PNG lies, relative to the
    session folder; `record` is `capture_window`'s."""
    name = record.get("file")
    return {
        "app_screenshot": str(Path(folder) / name) if name else None,
        "app_screenshot_missing_reason": (None if name else
                                          record.get("missing_reason") or unknown),
        # time.time() just before the window was drawn
        "app_screenshot_captured_at": record.get("captured_at"),
        "app_screenshot_method": record.get("method"),
        "app_screenshot_window": window,
        "app_screenshot_process": record.get("process"),
        "app_screenshot_size": ([record["width"], record["height"]]
                                if record.get("width") else None),
        # True when the picture has no pixel brighter than near-black: it is
        # kept, and it is not a picture of the hand
        "app_screenshot_black": record.get("black"),
    }


def _api():
    """user32, gdi32 and kernel32 with the handle-sized signatures set."""
    from ctypes import wintypes as wt
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetWindowDC.argtypes, user32.GetWindowDC.restype = [wt.HWND], wt.HDC
    user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
    user32.PrintWindow.argtypes = [wt.HWND, wt.HDC, wt.UINT]
    user32.PrintWindow.restype = wt.BOOL
    user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
    user32.IsIconic.argtypes = [wt.HWND]
    user32.IsWindowVisible.argtypes = [wt.HWND]
    user32.GetWindowTextLengthW.argtypes = [wt.HWND]
    user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    gdi32.CreateCompatibleDC.argtypes, gdi32.CreateCompatibleDC.restype = [wt.HDC], wt.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wt.HBITMAP
    gdi32.SelectObject.argtypes, gdi32.SelectObject.restype = [wt.HDC, wt.HGDIOBJ], wt.HGDIOBJ
    gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wt.HDC]
    gdi32.GetDIBits.argtypes = [wt.HDC, wt.HBITMAP, wt.UINT, wt.UINT,
                                ctypes.c_void_p, ctypes.c_void_p, wt.UINT]
    kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.CloseHandle.argtypes = [wt.HANDLE]
    kernel32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR,
                                                    ctypes.POINTER(wt.DWORD)]
    return user32, gdi32, kernel32


def window_titles() -> List[str]:
    """Titles of the visible top-level windows, for the 'not found' message."""
    return [t for _, t in _windows()]


def _windows():
    from ctypes import wintypes as wt
    user32, _, _ = _api()
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def each(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if buf.value.strip():
                    found.append((hwnd, buf.value.strip()))
        return True

    user32.EnumWindows(each, 0)
    return found


def find_window(title: str) -> Optional[int]:
    """The first visible top-level window titled exactly `title`
    (case-insensitive), or None. Never a partial match (see the module text)."""
    if sys.platform != "win32":
        return None
    wanted = title.strip().lower()
    for hwnd, text in _windows():
        if text.lower() == wanted:
            return hwnd
    return None


def _process_path(hwnd) -> Optional[str]:
    from ctypes import wintypes as wt
    user32, _, kernel32 = _api()
    pid = wt.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(0x1000, False, pid.value)   # QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        size = wt.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value
        return None
    finally:
        kernel32.CloseHandle(handle)


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


def _print_window(hwnd, width: int, height: int):
    """(PIL image or None, the time just before the window was drawn)."""
    from PIL import Image
    user32, gdi32, _ = _api()
    hdc_win = user32.GetWindowDC(hwnd)
    if not hdc_win:
        return None, time.time()
    hdc_mem = gdi32.CreateCompatibleDC(hdc_win)
    bitmap = gdi32.CreateCompatibleBitmap(hdc_win, width, height)
    old = gdi32.SelectObject(hdc_mem, bitmap)
    try:
        at = time.time()
        if not user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT):
            return None, at
        header = _BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        header.biWidth, header.biHeight = width, -height      # top-down rows
        header.biPlanes, header.biBitCount, header.biCompression = 1, 32, 0
        pixels = ctypes.create_string_buffer(width * height * 4)
        rows = gdi32.GetDIBits(hdc_mem, bitmap, 0, height, pixels,
                               ctypes.byref(header), 0)
        if rows != height:
            return None, at
        return Image.frombuffer("RGB", (width, height), pixels.raw, "raw", "BGRX", 0, 1), at
    finally:
        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_win)


def _grab_screen(left: int, top: int, right: int, bottom: int):
    """(PIL image or None, the time just before the copy) of a screen rectangle."""
    from PIL import ImageGrab
    at = time.time()
    try:
        return ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True).convert("RGB"), at
    except OSError:
        return None, at


def _uncovered(hwnd, left: int, top: int, right: int, bottom: int) -> bool:
    """Is `hwnd` the top window at the centre and the four quarter points of
    its rectangle? Only then does a copy of the screen there show it."""
    from ctypes import wintypes as wt
    user32, _, _ = _api()
    user32.WindowFromPoint.argtypes, user32.WindowFromPoint.restype = [wt.POINT], wt.HWND
    user32.GetAncestor.argtypes, user32.GetAncestor.restype = [wt.HWND, wt.UINT], wt.HWND
    width, height = right - left, bottom - top
    for fx, fy in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        point = wt.POINT(int(left + fx * width), int(top + fy * height))
        top_window = user32.WindowFromPoint(point)
        if not top_window or user32.GetAncestor(top_window, 2) != hwnd:   # GA_ROOT
            return False
    return True


def is_black(image) -> bool:
    return image.convert("L").getextrema()[1] <= BLACK_MAX


def capture_window(title: str, path) -> dict:
    """Save the window titled `title` to `path` (PNG). Never raises: what went
    wrong is the record's `missing_reason`."""
    record = empty_record(title)
    if sys.platform != "win32":
        record["missing_reason"] = NOT_WINDOWS
        return record
    from ctypes import wintypes as wt
    path = Path(path)
    restore = None
    try:
        user32, _, _ = _api()
        # Real pixels, not the size a DPI-unaware process is told: for this
        # thread only, and put back afterwards.
        try:
            user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
            user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            restore = user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(_DPI_PER_MONITOR_V2))
        except AttributeError:
            restore = None
        hwnd = find_window(title)
        if hwnd is None:
            record["missing_reason"] = NOT_FOUND
            return record
        record["process"] = _process_path(hwnd)
        if user32.IsIconic(hwnd):
            record["missing_reason"] = MINIMISED
            return record
        rect = wt.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            record["missing_reason"] = NO_AREA
            return record
        image, at = _print_window(hwnd, width, height)
        method = "PrintWindow"
        # The screen copy shows whatever is on top at that rectangle, so it is
        # used only when this window is the one on top there.
        if (image is None or is_black(image)) and _uncovered(
                hwnd, rect.left, rect.top, rect.right, rect.bottom):
            grabbed, grabbed_at = _grab_screen(rect.left, rect.top, rect.right, rect.bottom)
            if grabbed is not None and (image is None or not is_black(grabbed)):
                image, at, method = grabbed, grabbed_at, "screen"
        if image is None:
            record["missing_reason"] = FAILED
            return record
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
        record.update(file=path.name, captured_at=round(at, 6), method=method,
                      width=image.width, height=image.height, black=is_black(image))
        return record
    except Exception as e:                      # a screenshot never stops a recording
        record["missing_reason"] = f"capture error: {type(e).__name__}: {e}"
        return record
    finally:
        if restore:
            try:
                user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(restore))
            except Exception:
                pass


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print('usage: python -m xr_hand.app_screenshot "<window title>" <out.png>')
        return 2
    record = capture_window(argv[0], argv[1])
    if not record["file"]:
        print(f"APP SCREENSHOT FAIL: {record['missing_reason']} (window {argv[0]!r}); "
              f"open windows: {', '.join(window_titles()) if sys.platform == 'win32' else 'n/a'}")
        return 1
    print(f"APP SCREENSHOT OK: file={argv[1]} method={record['method']} "
          f"size={record['width']}x{record['height']} "
          f"black={'yes' if record['black'] else 'no'} process={record['process']}")
    return 1 if record["black"] else 0


if __name__ == "__main__":
    sys.exit(main())
