"""Make a bare hand look like the StretchSense glove, keeping its labels.

The problem this solves: training a detector to see the gloved hand needs
gloved images with 21 keypoints marked, and marking those by hand is thousands
of clicks. Bare-hand images with ground-truth labels are plentiful, and
repainting a hand moves no landmark — so the labels transfer unchanged.

The first version of this repaint failed in practice: a model trained on it
scored 0/250 on the real glove. Comparing real captured glove frames against
that synthetic one showed why, and this version is drawn from the real frames:

  bare fingertips    the glove is open-tipped; skin shows from roughly the
                     last knuckle out. v1 painted the tips over.
  separated fingers  fabric follows each finger; v1 dilated everything into
                     one mitten silhouette.
  sensor puck        a round module on the back of the hand.
  wrist strap        a dark band across the wrist and watch area.
  ribbed knit        directional fabric texture, not a dot grid.

Everything scales with palm length in pixels and jitters per frame (colour,
coverage, puck/strap presence) so the detector cannot key on one exact look.

Sanity check whenever this changes: MediaPipe should still COLLAPSE on the
output (the real glove scores ~1%). If MediaPipe sees the synthetic glove
fine, it is not simulating the problem.
"""
from typing import Sequence

import numpy as np

# (chain of landmark indices per finger, knuckle -> tip)
FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
PALM_POLY = [0, 1, 2, 5, 9, 13, 17]

FINGER_W = 0.20      # finger stroke width, fraction of palm length
PALM_PAD = 0.07      # small smoothing dilation — must NOT merge fingers


def _palm_px(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts[9] - pts[0])) or 1.0


def _pt(p) -> tuple:
    return int(round(p[0])), int(round(p[1]))


def glove_mask(shape, pts: np.ndarray, rng) -> np.ndarray:
    """Soft 0..1 mask of the FABRIC only: palm + fingers minus bare tips."""
    import cv2

    h, w = shape[:2]
    scale = _palm_px(pts)
    mask = np.zeros((h, w), np.uint8)
    stroke = max(2, int(round(scale * FINGER_W)))

    cv2.fillConvexPoly(mask, cv2.convexHull(
        np.round(pts[PALM_POLY]).astype(np.int32)), 255)

    for chain in FINGERS.values():
        # cover knuckle -> mid-distal; the tip segment stays mostly bare like
        # the real open-tipped glove, with per-finger jitter in coverage
        cover = rng.uniform(0.15, 0.45)
        end = pts[chain[2]] + cover * (pts[chain[3]] - pts[chain[2]])
        cv2.line(mask, _pt(pts[chain[0]]), _pt(pts[chain[1]]), 255, stroke,
                 cv2.LINE_AA)
        cv2.line(mask, _pt(pts[chain[1]]), _pt(pts[chain[2]]), 255, stroke,
                 cv2.LINE_AA)
        cv2.line(mask, _pt(pts[chain[2]]), _pt(end), 255,
                 max(2, int(stroke * 0.9)), cv2.LINE_AA)

    pad = max(1, int(round(scale * PALM_PAD)))
    mask = cv2.dilate(mask, np.ones((pad, pad), np.uint8))
    k = max(3, (pad // 2) * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0


def _fabric(frame_bgr, pts, rng, keep_shading: float):
    """Ribbed dark-knit layer the same size as the frame."""
    import cv2

    h, w = frame_bgr.shape[:2]
    scale = _palm_px(pts)
    # jitter brightness much more than hue: the real fabric is neutral grey,
    # so per-channel jitter must stay small or frames drift green/purple
    lum = rng.normal(0, 9)
    base = np.clip(np.array([40, 38, 36], np.float32) + lum
                   + rng.normal(0, 2.5, 3), 12, 95)

    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    shading = (grey / 255.0 - 0.5) * 2.0 * keep_shading
    fabric = np.clip(base[None, None, :] * (1.0 + shading), 0, 255)

    # ribbing running roughly along the fingers
    d = pts[9] - pts[0]
    ang = np.arctan2(d[1], d[0]) + rng.normal(0, 0.15)
    period = max(3.0, scale * rng.uniform(0.06, 0.11))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = (xx * np.cos(ang + np.pi / 2) + yy * np.sin(ang + np.pi / 2)) / period
    ribs = (np.sin(phase * 2 * np.pi) * 9.0)[..., None]
    noise = rng.normal(0, 5.0, (h, w, 1))
    return np.clip(fabric + ribs + noise, 0, 255)


def _draw_puck(fabric, pts, rng):
    """The round sensor module on the back of the hand."""
    import cv2

    scale = _palm_px(pts)
    mid_mcp = (pts[9] + pts[13]) / 2.0
    centre = pts[0] + rng.uniform(0.45, 0.6) * (mid_mcp - pts[0])
    r = max(3, int(round(scale * rng.uniform(0.24, 0.30))))
    tone = float(rng.uniform(22, 34))
    cv2.circle(fabric, _pt(centre), r, (tone, tone, tone + 2), -1, cv2.LINE_AA)
    cv2.circle(fabric, _pt(centre), r, (tone + 26,) * 3, 2, cv2.LINE_AA)
    glyph = (centre[0] + rng.uniform(-0.2, 0.2) * r,
             centre[1] + rng.uniform(-0.2, 0.2) * r)
    cv2.circle(fabric, _pt(glyph), max(1, r // 5), (tone + 45,) * 3, -1,
               cv2.LINE_AA)


def _strap_mask_and_tone(shape, pts, rng):
    """Dark band across the wrist, drawn beyond the hand silhouette."""
    import cv2

    h, w = shape[:2]
    scale = _palm_px(pts)
    d = pts[9] - pts[0]
    n = d / (np.linalg.norm(d) or 1.0)
    perp = np.array([-n[1], n[0]])
    centre = pts[0] - n * scale * rng.uniform(0.05, 0.2)
    half_len = scale * rng.uniform(0.55, 0.75)
    half_th = scale * rng.uniform(0.14, 0.22)
    corners = np.array([
        centre + perp * half_len + n * half_th,
        centre - perp * half_len + n * half_th,
        centre - perp * half_len - n * half_th,
        centre + perp * half_len - n * half_th,
    ])
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
    k = max(3, int(scale * 0.05) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    return mask[..., None], float(rng.uniform(16, 26))


def apply_glove(frame_bgr, landmarks_px, rng=None, keep_shading: float = 0.5,
                puck_prob: float = 0.9, strap_prob: float = 0.85):
    """Repaint one hand as the open-tipped sensor glove. Returns a new image.

    landmarks_px: 21 x [x, y, ...] image-space landmarks of the hand.
    Labels never move: only pixel appearance changes.
    """
    rng = rng or np.random.default_rng()
    pts = np.asarray(landmarks_px, dtype=float)[:, :2]

    mask = glove_mask(frame_bgr.shape, pts, rng)[..., None]
    fabric = _fabric(frame_bgr, pts, rng, keep_shading)
    if rng.random() < puck_prob:
        _draw_puck(fabric, pts, rng)

    out = frame_bgr.astype(np.float32) * (1 - mask) + fabric * mask

    if rng.random() < strap_prob:
        smask, tone = _strap_mask_and_tone(frame_bgr.shape, pts, rng)
        band = np.full_like(out, tone)
        band += rng.normal(0, 4.0, out.shape)
        out = out * (1 - smask) + np.clip(band, 0, 255) * smask

    return out.astype(np.uint8)


# kept for callers of the old API
def hand_mask(shape, landmarks_px, feather: bool = True):
    rng = np.random.default_rng(0)
    return glove_mask(shape, np.asarray(landmarks_px, dtype=float)[:, :2], rng)
