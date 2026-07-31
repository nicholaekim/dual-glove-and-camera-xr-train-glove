"""Webcam helpers (Windows-friendly).

cv2.VideoCapture on Windows defaults to MSMF, which can take several seconds
to open and occasionally ignores resolution requests; DirectShow usually opens
fast. Try DSHOW first, fall back to the default backend.
"""
import logging
import time
from typing import Optional, Tuple

log = logging.getLogger(__name__)


def open_camera(index: int = 0, width: int = 640, height: int = 480,
                fps: int = 30):
    """Open a webcam and return the cv2.VideoCapture, or raise RuntimeError."""
    import cv2

    for backend, name in ((cv2.CAP_DSHOW, "DSHOW"), (None, "default")):
        cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = cap.read()
        if ok and frame is not None:
            log.info("camera %d open via %s: %dx%d", index, name,
                     frame.shape[1], frame.shape[0])
            return cap
        cap.release()
    raise RuntimeError(
        f"could not open camera {index}. Is a webcam connected, and does "
        "Windows camera privacy allow desktop apps? (Settings > Privacy > Camera)")


def read_frame(cap) -> Tuple[Optional[object], int]:
    """Read one frame; returns (frame_bgr | None, monotonic ts in ms)."""
    ok, frame = cap.read()
    ts_ms = int(time.monotonic() * 1000)
    return (frame if ok else None), ts_ms
