"""Make a bare hand look gloved, so its keypoint labels can be reused.

The problem this solves: training a detector to see the gloved hand needs
images of a gloved hand with 21 keypoints marked on them, and marking those by
hand is thousands of clicks. But MediaPipe detects the BARE hand in 100% of
frames, so bare-hand video is a free source of perfectly labelled keypoints.
Repainting the hand region to look like the glove leaves the geometry — and
therefore the labels — exactly unchanged.

The mask comes from the landmarks themselves: thick strokes along the bone
connections plus a filled palm polygon, dilated and feathered. It is an
approximation of the hand silhouette, not a segmentation, which is adequate
because the goal is to hide skin texture rather than to matte the hand
perfectly.

Everything scales with palm length in pixels, so it behaves the same whether
the hand is near or far from the camera.

Sanity check worth repeating whenever the parameters change: run MediaPipe on
the output. If the synthetic glove is convincing, detection should COLLAPSE
the way it does on the real glove. A synthetic glove that MediaPipe still sees
is not simulating the problem.
"""
from typing import Sequence

import numpy as np

from .draw import CONNECTIONS

# Fingers are slimmer than the palm; both are expressed as a fraction of the
# wrist-to-middle-knuckle distance so the look is scale invariant.
FINGER_W = 0.22
PALM_PAD = 0.16
PALM_POLY = [0, 1, 2, 5, 9, 13, 17]     # wrist, thumb base, knuckles


def _palm_px(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts[9] - pts[0])) or 1.0


def hand_mask(shape, landmarks_px: Sequence[Sequence[float]],
              feather: bool = True) -> np.ndarray:
    """Soft 0..1 mask covering the hand, built from its 21 landmarks."""
    import cv2

    h, w = shape[:2]
    pts = np.asarray(landmarks_px, dtype=float)[:, :2]
    scale = _palm_px(pts)
    mask = np.zeros((h, w), np.uint8)

    stroke = max(2, int(round(scale * FINGER_W)))
    for a, b in CONNECTIONS:
        cv2.line(mask, tuple(np.round(pts[a]).astype(int)),
                 tuple(np.round(pts[b]).astype(int)), 255, stroke, cv2.LINE_AA)
    for p in pts:
        cv2.circle(mask, tuple(np.round(p).astype(int)),
                   max(2, stroke // 2), 255, -1, cv2.LINE_AA)

    palm = np.round(pts[PALM_POLY]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(palm), 255)
    pad = max(1, int(round(scale * PALM_PAD)))
    mask = cv2.dilate(mask, np.ones((pad, pad), np.uint8))
    if feather:
        k = max(3, (pad // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def apply_glove(frame_bgr, landmarks_px, rng=None, base=(38, 36, 34),
                keep_shading: float = 0.45, dots: bool = True,
                jitter: bool = True):
    """Repaint the hand region as dark fabric. Returns a new BGR image.

    keep_shading retains some of the original luminance so the fingers still
    read as three-dimensional rather than as a flat silhouette — without it a
    detector could learn to key on an unnaturally uniform blob.
    """
    import cv2

    rng = rng or np.random.default_rng()
    pts = np.asarray(landmarks_px, dtype=float)[:, :2]
    scale = _palm_px(pts)
    mask = hand_mask(frame_bgr.shape, landmarks_px)[..., None]

    colour = np.array(base, dtype=np.float32)
    if jitter:                       # vary fabric colour and lighting per frame
        colour = np.clip(colour + rng.normal(0, 7, 3), 8, 90)

    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    shading = (grey / 255.0 - 0.5) * 2.0 * keep_shading
    fabric = np.clip(colour[None, None, :] * (1.0 + shading), 0, 255)

    if dots:                         # the glove's grip texture, lightly
        tex = np.zeros(frame_bgr.shape[:2], np.uint8)
        step = max(3, int(round(scale * 0.09)))
        ys, xs = np.mgrid[0:frame_bgr.shape[0]:step, 0:frame_bgr.shape[1]:step]
        for y, x in zip(ys.ravel(), xs.ravel()):
            cv2.circle(tex, (int(x), int(y)), 1, 255, -1)
        fabric = np.clip(fabric + (tex[..., None] / 255.0) * 26.0, 0, 255)

    out = frame_bgr.astype(np.float32) * (1 - mask) + fabric * mask
    return out.astype(np.uint8)
