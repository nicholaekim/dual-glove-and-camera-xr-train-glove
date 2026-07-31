"""Optional image boosting before hand detection. OFF by default, and the
evidence says leave it off.

It is the obvious thing to reach for when a hand is not detected, so it exists
and can be swept — but measured on this setup it HURT badly. A bare hand
detected in 100% of frames dropped to 0% under gamma 0.55 + CLAHE 2.5 (68% at
the loosest threshold). Brightening an already well-exposed frame blows out
the skin texture the model depends on.

Nor is underexposure usually the problem in the first place: MediaPipe still
detects a bare hand at 0.97 confidence after the image is darkened to gamma
0.25. When a gloved hand is missed, the cause is that the model was trained on
bare skin and does not recognise the glove as a hand; no amount of gamma
changes that.

Only turn this on if scripts/tune_detection.py measures it helping in a
genuinely dark setup.
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
