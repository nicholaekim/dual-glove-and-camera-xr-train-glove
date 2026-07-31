"""Optional image boosting before hand detection.

Included because it is the obvious first thing to try when a hand is not
detected, and because trying it is cheap. Be aware of the measured result
though: MediaPipe still detects a bare hand at 0.97 confidence after the image
is darkened to gamma 0.25, so underexposure is NOT usually why detection
fails. If a gloved hand is missed, the cause is almost certainly that the
model was trained on bare skin and does not recognise the glove as a hand —
no amount of gamma will change that.

Use scripts/tune_detection.py to find out whether it helps in your setup
rather than assuming it does.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _gamma_lut(gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    return np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                    dtype=np.uint8)


def boost(frame_bgr, gamma: float = 1.0, clahe_clip: float = 0.0):
    """Brighten (gamma < 1) and/or raise local contrast (clahe_clip > 0).

    Returns the frame unchanged when both are at their no-op defaults, so
    callers can pass user options straight through.
    """
    import cv2

    if gamma == 1.0 and clahe_clip <= 0.0:
        return frame_bgr
    out = frame_bgr
    if gamma != 1.0:
        out = cv2.LUT(out, _gamma_lut(gamma))
    if clahe_clip > 0.0:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return out
