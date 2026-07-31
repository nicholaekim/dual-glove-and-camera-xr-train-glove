"""Fusion maths: the invariants that make a fused hand trustworthy."""
import math

import numpy as np
import pytest

from cam_hand.align import umeyama
from cam_hand.features import all_features, flexion_features, spread_features
from cam_hand.fusion import (
    FINGER_CHAINS,
    fuse_skeletons,
    pair_by_time,
    palm_frame,
    rotation_between,
)


def make_hand(spread_deg=0.0, curl=0.0, thumb_lift_deg=0.0):
    """A synthetic 21-keypoint right hand, wrist at the origin, metres.

    Fingers leave their knuckles along +y, fanned in the xy (palm) plane by
    `spread_deg`; `curl` bends them toward -z, which is what the glove senses.
    """
    pts = np.zeros((21, 3))
    knuckle_x = {"thumb": -0.035, "index": -0.02, "middle": 0.0,
                 "ring": 0.02, "pinky": 0.038}
    knuckle_y = {"thumb": 0.03, "index": 0.085, "middle": 0.09,
                 "ring": 0.085, "pinky": 0.075}
    fan = {"thumb": -2.0, "index": -1.0, "middle": 0.0, "ring": 1.0, "pinky": 2.0}
    bone = 0.025
    for finger, chain in FINGER_CHAINS.items():
        base = np.array([knuckle_x[finger], knuckle_y[finger], 0.0])
        pts[chain[0]] = base
        a = math.radians(spread_deg * fan[finger])
        c = math.radians(curl * 60.0)
        step = np.array([math.sin(a) * math.cos(c),
                         math.cos(a) * math.cos(c),
                         -math.sin(c)]) * bone
        for k in range(1, 4):
            pts[chain[k]] = pts[chain[k - 1]] + step
    if thumb_lift_deg:
        chain = FINGER_CHAINS["thumb"]
        th = math.radians(thumb_lift_deg)
        R = np.array([[1, 0, 0],
                      [0, math.cos(th), -math.sin(th)],
                      [0, math.sin(th), math.cos(th)]])
        base = pts[chain[0]].copy()
        pts[chain] = (R @ (pts[chain] - base).T).T + base
    return pts


def bone_lengths(pts):
    out = []
    for chain in FINGER_CHAINS.values():
        for a, b in zip(chain, chain[1:]):
            out.append(float(np.linalg.norm(np.asarray(pts)[b] - np.asarray(pts)[a])))
    return out


# --- alignment ---------------------------------------------------------

def test_umeyama_recovers_known_transform():
    src = make_hand(spread_deg=10.0)
    th = math.radians(37.0)
    R_true = np.array([[math.cos(th), -math.sin(th), 0],
                       [math.sin(th), math.cos(th), 0],
                       [0, 0, 1]])
    dst = (1.7 * (R_true @ src.T)).T + np.array([0.3, -0.2, 0.05])
    R, s, t = umeyama(src, dst)
    assert s == pytest.approx(1.7, abs=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose((s * (R @ src.T)).T + t, dst, atol=1e-9)


def test_umeyama_never_reflects():
    """A mirrored hand must not be 'aligned' by flipping it."""
    src = make_hand(spread_deg=12.0)
    dst = src.copy()
    dst[:, 0] *= -1.0
    R, _s, _t = umeyama(src, dst)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_rotation_between_handles_antiparallel():
    a = np.array([0.0, 1.0, 0.0])
    R = rotation_between(a, -a)
    assert np.allclose(R @ a, -a, atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


# --- fusion ------------------------------------------------------------

def test_fusion_preserves_every_bone_length():
    """Rotating chains about their knuckles cannot stretch or shrink a finger."""
    glove = make_hand(spread_deg=0.0, curl=0.4)
    cam = make_hand(spread_deg=20.0, curl=0.4)
    fused, info = fuse_skeletons(glove, cam)
    assert info["camera_used"]
    assert bone_lengths(fused) == pytest.approx(bone_lengths(glove), abs=1e-9)


def test_fusion_adopts_camera_spread():
    """Fingers held together by the glove must fan out to the camera's spread."""
    glove = make_hand(spread_deg=0.0, curl=0.2)
    cam = make_hand(spread_deg=22.0, curl=0.2)
    fused, _ = fuse_skeletons(glove, cam)

    def tip_gap(p):
        return float(np.linalg.norm(np.asarray(p)[8] - np.asarray(p)[12]))

    assert tip_gap(glove) < tip_gap(fused)
    assert tip_gap(fused) == pytest.approx(tip_gap(cam), rel=0.12)


def test_fusion_keeps_glove_curl():
    """Camera curl is ignored: flexion must still come from the glove.

    The exact invariant is the finger's angle out of the palm plane — that is
    what the algorithm carries over untouched. Wrist-to-fingertip distance is
    only approximately preserved, because abducting a finger swings it about a
    knuckle that is offset from the wrist, which moves the tip a little even
    at constant curl.
    """
    glove = make_hand(spread_deg=0.0, curl=0.9)
    cam = make_hand(spread_deg=15.0, curl=0.0)     # camera thinks it is open
    fused, _ = fuse_skeletons(glove, cam, thumb_from_camera=False)

    n, _x, _y = palm_frame(glove)
    for finger, chain in FINGER_CHAINS.items():
        def out_of_plane(p):
            d = np.asarray(p)[chain[-1]] - np.asarray(p)[chain[0]]
            return float(np.dot(d / np.linalg.norm(d), n))
        assert out_of_plane(fused) == pytest.approx(out_of_plane(glove), abs=1e-9), finger

    # and the fused hand must still read as a curled hand, not the open one
    # the camera saw
    fused_f = np.asarray(flexion_features(fused))
    assert (np.abs(fused_f - np.asarray(flexion_features(glove))).max()
            < np.abs(fused_f - np.asarray(flexion_features(cam))).max() / 3)


def test_fusion_takes_thumb_opposition_from_camera():
    """The pose the glove cannot see must come through from the camera."""
    glove = make_hand(curl=0.3)
    cam = make_hand(curl=0.3, thumb_lift_deg=55.0)
    fused, _ = fuse_skeletons(glove, cam, thumb_from_camera=True)
    out_glove = spread_features(glove)[-1]
    out_cam = spread_features(cam)[-1]
    out_fused = spread_features(fused)[-1]
    assert abs(out_fused - out_cam) < abs(out_glove - out_cam)


def test_fusion_falls_back_to_glove_when_camera_is_absent_or_unsure():
    glove = make_hand(spread_deg=0.0, curl=0.5)
    cam = make_hand(spread_deg=25.0, curl=0.5)

    fused, info = fuse_skeletons(glove, None)
    assert not info["camera_used"] and "no camera" in info["reason"]
    assert np.allclose(fused, glove - glove[0])

    fused, info = fuse_skeletons(glove, cam, cam_score=0.2, min_score=0.5)
    assert not info["camera_used"] and "score" in info["reason"]
    assert np.allclose(fused, glove - glove[0])


def test_fusion_is_scale_invariant():
    """A camera hand of the wrong size still contributes only its angles."""
    glove = make_hand(spread_deg=0.0, curl=0.3)
    cam_small = make_hand(spread_deg=18.0, curl=0.3) * 0.6
    fused, _ = fuse_skeletons(glove, cam_small)
    assert bone_lengths(fused) == pytest.approx(bone_lengths(glove), abs=1e-9)


def test_fused_output_is_wrist_centred():
    glove = make_hand(curl=0.3)
    fused, _ = fuse_skeletons(glove, make_hand(spread_deg=10.0, curl=0.3))
    assert np.allclose(fused[0], 0.0, atol=1e-12)


def test_palm_frame_axes_are_orthonormal():
    n, x, y = palm_frame(make_hand(spread_deg=8.0))
    for v in (n, x, y):
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-9)
    assert float(np.dot(n, x)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.dot(n, y)) == pytest.approx(0.0, abs=1e-9)


# --- chirality ---------------------------------------------------------

def test_features_match_between_mirrored_hands():
    """The same gesture on the other hand must give the same numbers.

    Without the hand_side correction the signed thumb feature flips, and a
    classifier sees one gesture as two.
    """
    right = make_hand(spread_deg=12.0, curl=0.3, thumb_lift_deg=40.0)
    left = right.copy()
    left[:, 0] *= -1.0
    assert all_features(left, hand_side="left") == pytest.approx(
        all_features(right, hand_side="right"), abs=1e-9)


# --- time alignment ----------------------------------------------------

def test_pair_by_time_picks_the_nearest_same_hand_frame():
    glove = [{"wall_time": 10.00, "hand_side": "right"},
             {"wall_time": 10.10, "hand_side": "right"},
             {"wall_time": 10.20, "hand_side": "left"}]
    cam = [{"wall_time": 9.99, "hand_side": "right", "tag": "a"},
           {"wall_time": 10.12, "hand_side": "right", "tag": "b"},
           {"wall_time": 10.19, "hand_side": "left", "tag": "c"}]
    pairs = pair_by_time(glove, cam, max_dt=0.05)
    assert [c["tag"] for _g, c in pairs] == ["a", "b", "c"]


def test_pair_by_time_never_crosses_hands_and_drops_far_frames():
    glove = [{"wall_time": 5.0, "hand_side": "left"}]
    cam = [{"wall_time": 5.001, "hand_side": "right"}]
    assert pair_by_time(glove, cam, max_dt=0.05) == [(glove[0], None)]

    glove = [{"wall_time": 5.0, "hand_side": "left"}]
    cam = [{"wall_time": 5.5, "hand_side": "left"}]
    assert pair_by_time(glove, cam, max_dt=0.05) == [(glove[0], None)]
