"""Fusion maths: the invariants that make a fused hand trustworthy."""
import math
import time

import numpy as np
import pytest

from cam_hand.align import umeyama
from cam_hand.features import all_features, flexion_features, spread_features
from cam_hand.fusion import (
    CAMERA_DOFS,
    DEFAULT_GATES,
    FINGER_CHAINS,
    GateParams,
    R_CURLED,
    R_DISAGREE,
    R_FIELD,
    R_HAND_ID,
    R_NO_FRAME,
    R_VIEW,
    R_VISIBLE,
    flag_hand_id_stability,
    frame_trust,
    fuse_skeletons,
    pair_by_time,
    palm_field_angle_deg,
    palm_frame,
    proximal_direction,
    rotation_between,
    viewing_angle_deg,
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


# --- a leap camera take, end to end ------------------------------------
# Deliverable 4: a synthetic glove take and a synthetic LEAP take of the same
# hand, written as real files, read back by the real loaders in
# scripts/fuse_poses.py, and fused. The invariant is the whole reason the two
# sensors are recorded together: the fused hand must have the camera's spread
# and the glove's curl.

def _hand26(spread_deg: float, curl: float, origin=(0.0, 0.25, 0.0)):
    """A 26-joint OpenXR hand, absolute metres, built from make_hand's 21.

    The five metacarpals the 21-point layout has no place for are put on the
    segment from the wrist to their knuckle, which is where a metacarpal is.
    Rotations are identity throughout: `absolute_to_relative` then reduces to
    plain differences and `forward_kinematics` reproduces these positions
    exactly, so the test measures fusion rather than a quaternion convention.
    """
    from xr_hand.joints import JOINT_INDEX
    from xr_hand.keypoints21 import MP21_TO_OPENXR

    pts21 = make_hand(spread_deg=spread_deg, curl=curl)
    abs26 = [None] * 26
    for k, (_mp, xr) in enumerate(MP21_TO_OPENXR):
        abs26[JOINT_INDEX[xr]] = np.asarray(pts21[k], float)
    wrist = abs26[JOINT_INDEX["WRIST"]]
    for finger in ("INDEX", "MIDDLE", "RING", "LITTLE"):
        knuckle = abs26[JOINT_INDEX[f"{finger}_PROXIMAL"]]
        abs26[JOINT_INDEX[f"{finger}_METACARPAL"]] = wrist + 0.5 * (knuckle - wrist)
    abs26[JOINT_INDEX["PALM"]] = wrist + 0.5 * (
        abs26[JOINT_INDEX["MIDDLE_PROXIMAL"]] - wrist)
    o = np.asarray(origin, float)
    return [list(p + o) for p in abs26]


def _write_glove_take(path, spread_deg, curl, n=6, hand="right"):
    """A real glove JSONL, through absolute_to_relative + FrameRecorder."""
    from xr_hand.joints import HandFrame
    from xr_hand.kinematics import absolute_to_relative
    from xr_hand.recorder import FrameRecorder

    abs26 = _hand26(spread_deg, curl)
    quats = [[0.0, 0.0, 0.0, 1.0]] * 26
    rec = FrameRecorder(pose="pinch", take=1)
    rec.start(path)
    for i in range(n):
        rec.record(HandFrame(timestamp=float(i), packet_counter=i,
                             hand_side=hand, frame_id=i, status=0,
                             joints=absolute_to_relative(abs26, quats)))
        time.sleep(0.004)
    rec.stop()
    return path


def _write_leap_take(path, spread_deg, curl, n=6, hand="right"):
    """A real leap JSONL, through LeapHand + LeapRecorder."""
    from leap_hand.recorder import LeapRecorder
    from leap_hand.types import LeapHand

    abs26 = _hand26(spread_deg, curl)
    quats = [[0.0, 0.0, 0.0, 1.0]] * 26
    rec = LeapRecorder(pose="pinch", take=1)
    rec.start(path)
    for i in range(n):
        rec.record(LeapHand(
            hand_side=hand, hand_id=1, timestamp_us=1_000_000 + i * 11_111,
            frame_id=i, framerate=90.0, visible_time_us=5_000_000,
            pinch_strength=0.0, grab_strength=0.0,
            palm_pos=abs26[0], palm_quat=[0.0, 0.0, 0.0, 1.0],
            abs26=abs26, quat26=quats))
        time.sleep(0.004)
    rec.stop()
    return path


def _fuse_module():
    """scripts/fuse_poses.py, which is not on a package path."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "fuse_poses.py"
    spec = importlib.util.spec_from_file_location("fuse_poses_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_leap_take_fuses_with_a_glove_take_of_the_same_hand(tmp_path):
    """Curl from the glove, spread from the Ultraleap, on real files."""
    from cam_hand.features import flexion_features, spread_features

    fuse = _fuse_module()
    name = "pinch_right_take1_20260916_230000.jsonl"
    _write_glove_take(tmp_path / "glove" / name, spread_deg=0.0, curl=0.8)
    _write_leap_take(tmp_path / "leap" / name, spread_deg=24.0, curl=0.0)

    # the camera file is found in leap/ and recognised by its own contents
    cpath = fuse.find_camera_take(tmp_path, name)
    assert cpath == tmp_path / "leap" / name
    assert fuse.camera_source(cpath) == "leap"

    glove = fuse.load_glove(tmp_path / "glove" / name)
    cam, source = fuse.load_cam(cpath)
    assert source == "leap" and len(glove) == len(cam) == 6
    # metres, wrist-centred, the same layout on both sides
    assert np.allclose(cam[0]["pts"][0], 0.0, atol=1e-9)
    assert 0.05 < float(np.linalg.norm(np.asarray(cam[0]["pts"][12]))) < 0.30

    pairs = pair_by_time(glove, cam, max_dt=0.05)
    matched = [(g, c) for g, c in pairs if c is not None]
    assert len(matched) == len(glove), "both takes share one wall clock"

    g, c = matched[0]
    fused, info = fuse_skeletons(g["pts"], c["pts"], with_scale=False)
    assert info["camera_used"] and info["alignment"] == "palm_frame"

    G, C = np.asarray(g["pts"], float), np.asarray(c["pts"], float)

    # spread: the fused fingers must fan out like the camera's, not like the
    # glove's, which recorded them held together
    def gaps(p):
        return np.asarray(spread_features(p, hand_side="right")[:4])
    assert np.abs(gaps(fused) - gaps(C)).max() < np.abs(gaps(G) - gaps(C)).max() / 3

    # curl: out of the palm plane, every finger keeps the glove's angle
    n, _x, _y = palm_frame(G)
    for finger, chain in FINGER_CHAINS.items():
        def out_of_plane(p):
            d = np.asarray(p)[chain[-1]] - np.asarray(p)[chain[0]]
            return float(np.dot(d / np.linalg.norm(d), n))
        if finger == "thumb":
            continue                     # opposition is the camera's by design
        assert out_of_plane(fused) == pytest.approx(out_of_plane(G), abs=1e-9)

    # ...and the four fingers still read as the curled hand the glove
    # measured, not the open one the camera saw. The thumb is excluded on
    # purpose: its whole direction is the camera's, because opposition is
    # what the glove cannot see.
    f = np.asarray(flexion_features(fused))[1:]
    assert (np.abs(f - np.asarray(flexion_features(G))[1:]).max()
            < np.abs(f - np.asarray(flexion_features(C))[1:]).max() / 10)
    # the thumb went the other way, which is the point of taking it from the camera
    assert (abs(flexion_features(fused)[0] - flexion_features(C)[0])
            < abs(flexion_features(fused)[0] - flexion_features(G)[0]))
    # bones are the glove's, untouched: fusion rotates, it never rescales
    assert bone_lengths(fused) == pytest.approx(bone_lengths(G), abs=1e-9)


def test_a_metric_camera_is_aligned_rigidly_and_a_normalised_one_is_not():
    """plan section 3: no scale when both sides are already millimetres."""
    from cam_hand.fusion import camera_into_glove_frame

    glove = make_hand(spread_deg=0.0, curl=0.3)
    half = make_hand(spread_deg=0.0, curl=0.3) * 0.5      # same hand, half size

    rigid = camera_into_glove_frame(glove, half, with_scale=False)
    scaled = camera_into_glove_frame(glove, half, with_scale=True)
    # the rigid fit keeps the camera's own size; the scaled one adopts the glove's
    assert bone_lengths(rigid) == pytest.approx(bone_lengths(half), abs=1e-9)
    assert bone_lengths(scaled) == pytest.approx(bone_lengths(glove), abs=1e-9)

    # but the fused hand is the same either way: only directions are used
    a, _ = fuse_skeletons(glove, half, with_scale=False)
    b, _ = fuse_skeletons(glove, half, with_scale=True)
    assert np.allclose(a, b, atol=1e-9)


def test_a_mediapipe_take_is_still_read_as_before(tmp_path):
    """Adding the leap path must not change what a webcam file means."""
    import json

    fuse = _fuse_module()
    name = "fist_right_take1_20260916_230000.jsonl"
    world = [[x, y, z] for x, y, z in make_hand(spread_deg=10.0, curl=0.2)]
    path = tmp_path / "cam" / name
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps({
        "wall_time": 1.7e9 + i * 0.03, "ts_ms": i * 33, "hand_side": "right",
        "score": 0.97, "frame_w": 640, "frame_h": 480, "pose": "fist",
        "take": 1, "img": [[0.0, 0.0, 0.0]] * 21, "world": world,
    }) for i in range(4)) + "\n", encoding="utf-8")

    assert fuse.find_camera_take(tmp_path, name) == path
    assert fuse.camera_source(path) == "mediapipe"
    rows, source = fuse.load_cam(path)
    assert source == "mediapipe" and len(rows) == 4
    assert rows[0]["score"] == pytest.approx(0.97)
    assert np.allclose(rows[0]["pts"][0], 0.0, atol=1e-12)   # wrist-centred


def test_the_same_hand_fuses_the_same_through_either_camera_loader(tmp_path):
    """One geometry, written twice: as a webcam take and as a leap take.

    The leap path must be the same pipeline with a different reader and a
    rigid fit, not a second implementation that happens to run. If these two
    ever disagree, one of the loaders is bending the data.
    """
    fuse = _fuse_module()
    name = "spread_right_take1_20260916_230000.jsonl"
    cam_pts = make_hand(spread_deg=22.0, curl=0.1)

    _write_glove_take(tmp_path / "glove" / name, spread_deg=0.0, curl=0.6)
    _write_leap_take(tmp_path / "leap" / name, spread_deg=22.0, curl=0.1)
    glove = fuse.load_glove(tmp_path / "glove" / name)
    leap_rows, _ = fuse.load_cam(tmp_path / "leap" / name)

    # the same 21 points the leap file holds, read as a MediaPipe take would be
    leap_pts = np.asarray(leap_rows[0]["pts"], float)
    assert np.allclose(leap_pts, cam_pts, atol=1e-6), (
        "the leap loader must return the geometry that was recorded")

    G = glove[0]["pts"]
    through_leap, info_leap = fuse_skeletons(G, leap_pts, with_scale=False)
    through_cam, info_cam = fuse_skeletons(G, cam_pts, with_scale=True)
    assert info_leap["alignment"] == "palm_frame"
    assert info_cam["alignment"] == "similarity"
    assert np.allclose(through_leap, through_cam, atol=1e-9)


# --- which clock pairs the two streams ---------------------------------
# `wall_time` is a WRITER stamp. Both recorders drain a queue and write the
# burst it held, so frames captured hundreds of ms apart can share a
# wall_time. `capture_time` is when the sensor had the frame. These pin that
# the right one is chosen, and that choosing it actually changes the answer.

def test_pairing_clock_needs_capture_time_on_every_frame():
    from cam_hand.fusion import pairing_clock

    full = [{"wall_time": 1.0, "capture_time": 0.9},
            {"wall_time": 1.1, "capture_time": 1.0}]
    assert pairing_clock(full, full) == "capture_time"
    # one stream without it: a mixture would compare two different clocks
    old = [{"wall_time": 1.0}, {"wall_time": 1.1}]
    assert pairing_clock(full, old) == "wall_time"
    assert pairing_clock(old, full) == "wall_time"
    # one FRAME without it is enough, and so is an explicit null
    partial = [{"wall_time": 1.0, "capture_time": 0.9}, {"wall_time": 1.1}]
    assert pairing_clock(full, partial) == "wall_time"
    nulled = [{"wall_time": 1.0, "capture_time": None}]
    assert pairing_clock(full, nulled) == "wall_time"
    assert pairing_clock([], []) == "wall_time"


def test_a_written_burst_pairs_wrong_on_wall_time_and_right_on_capture_time():
    """The bug, in eight frames.

    The camera queued four frames 30 ms apart and they were all written in
    one pass, a millisecond apart. On the writer's clock every one of them
    looks equally close to the glove frame, so the FIRST wins by scan order;
    on the capture clock the one actually taken at that moment wins.
    """
    glove = [{"wall_time": 100.000, "capture_time": 100.000,
              "hand_side": "right"}]
    cam = [{"wall_time": 100.050 + i * 0.001, "capture_time": 99.940 + i * 0.030,
            "hand_side": "right", "tag": i} for i in range(4)]
    #        capture times: 99.940, 99.970, 100.000, 100.030
    #        the glove frame was captured at 100.000, so tag 2 is the partner

    on_wall = pair_by_time(glove, cam, max_dt=0.05, clock="wall_time")[0][1]
    on_capture = pair_by_time(glove, cam, max_dt=0.05, clock="capture_time")[0][1]
    assert on_capture["tag"] == 2, "capture_time must pick the frame of that instant"
    assert on_wall["tag"] != 2, "wall_time cannot tell these four apart"
    # auto picks the right one, because every frame here has a capture_time
    assert pair_by_time(glove, cam, max_dt=0.05)[0][1]["tag"] == 2


def test_wall_time_pairing_can_exceed_the_window_it_reports():
    """Worse than picking wrong: it picks a frame outside --max-dt and says ok."""
    glove = [{"wall_time": 100.0, "capture_time": 100.0, "hand_side": "left"}]
    cam = [{"wall_time": 100.001, "capture_time": 99.80, "hand_side": "left"}]

    paired = pair_by_time(glove, cam, max_dt=0.05, clock="wall_time")[0][1]
    assert paired is not None                       # "matched within 50 ms"
    assert abs(paired["capture_time"] - glove[0]["capture_time"]) > 0.05
    # on the real clock it is correctly refused
    assert pair_by_time(glove, cam, max_dt=0.05)[0][1] is None


def test_old_recordings_without_capture_time_still_pair():
    """Files recorded before capture_time existed must keep working."""
    glove = [{"wall_time": 10.00, "hand_side": "right"},
             {"wall_time": 10.10, "hand_side": "right"}]
    cam = [{"wall_time": 9.99, "hand_side": "right", "tag": "a"},
           {"wall_time": 10.12, "hand_side": "right", "tag": "b"}]
    assert [c["tag"] for _g, c in pair_by_time(glove, cam, max_dt=0.05)] == ["a", "b"]


def test_one_take_name_under_two_cameras_is_an_error_not_a_coin_flip(tmp_path):
    """Two sensors, two alignments: picking by folder order picks by accident."""
    fuse = _fuse_module()
    name = "fist_right_take1_20260916_230000.jsonl"
    _write_leap_take(tmp_path / "leap" / name, spread_deg=10.0, curl=0.2)
    (tmp_path / "cam").mkdir()
    (tmp_path / "cam" / name).write_text("{}\n", encoding="utf-8")

    with pytest.raises(fuse.AmbiguousTake) as e:
        fuse.find_camera_take(tmp_path, name)
    assert "cam" in str(e.value) and "leap" in str(e.value)
    assert "--camera" in str(e.value)            # and it says how to resolve it

    # naming a camera resolves it, and only looks where it was told
    assert fuse.find_camera_take(tmp_path, name, "leap") == tmp_path / "leap" / name
    assert fuse.find_camera_take(tmp_path, name, "cam") == tmp_path / "cam" / name
    assert fuse.find_camera_take(tmp_path, "absent.jsonl", "leap") is None


# --- the metric path does not let palm SIZE become rotation -------------
# The glove reports the XR Trainer template hand; the camera measures the
# real one. Kabsch over five palm points trades rotation against that size
# difference, and the tilt it invents lands in the spread the camera is
# supposed to be supplying.

def test_a_camera_hand_10_percent_larger_still_gives_its_spread(tmp_path):
    """The stated case: the real hand is bigger than the template."""
    from cam_hand.features import spread_features

    glove = make_hand(spread_deg=0.0, curl=0.5)
    cam = make_hand(spread_deg=24.0, curl=0.5) * 1.10      # same pose, bigger

    fused, info = fuse_skeletons(glove, cam, with_scale=False)
    assert info["alignment"] == "palm_frame"

    def gaps(p):
        # normalised by palm length, so a bigger hand is comparable at all
        return np.asarray(spread_features(p, hand_side="right")[:4])

    assert np.abs(gaps(fused) - gaps(cam)).max() < np.abs(
        gaps(glove) - gaps(cam)).max() / 5
    # bones stay the glove's: fusion rotates chains, it never rescales them
    assert bone_lengths(fused) == pytest.approx(bone_lengths(glove), abs=1e-9)
    # and the size disagreement is reported rather than absorbed
    assert info["kabsch_rmse_mm"] > 1.0


def _tilt_deg(R) -> float:
    """The rotation angle of R, in degrees."""
    return math.degrees(math.acos(
        min(1.0, max(-1.0, (np.trace(np.asarray(R)) - 1.0) / 2.0))))


def test_palm_size_mismatch_does_not_tilt_the_camera_hand():
    """The mechanism, isolated: a reshaped palm must not rotate the hand.

    Every camera hand here holds the SAME orientation as the glove hand and
    differs only in palm proportions — a template-vs-real difference, not a
    pose difference. The transfer must therefore be a pure placement, and on
    the palm-basis path it is: exactly zero degrees, because the basis is
    built from unit vectors and dimensions cannot reach it.

    The 5-point rigid fit this replaced tilts by a few tenths of a degree,
    in a fixed direction for a given mismatch. Not large — the point is that
    it is systematic, it is entirely an artefact of the glove's template
    hand, and it lands in the one quantity the camera is there to supply.
    Note the last case: a hand that is evenly BIGGER does not fool Kabsch at
    all. Only a change of PROPORTIONS does, which is exactly what a template
    hand on a real hand is.
    """
    from cam_hand.align import umeyama
    from cam_hand.fusion import PALM_IDX, palm_basis, palm_frame_transfer

    glove = make_hand(spread_deg=0.0, curl=0.3)
    reshapes = {"wider palm": (1.20, 1.00), "longer palm": (1.00, 1.20),
                "wider and shorter": (1.20, 0.90),
                "narrower palm": (0.85, 1.00)}
    worst_kabsch = 0.0
    for label, (sx, sy) in reshapes.items():
        cam = make_hand(spread_deg=0.0, curl=0.3)
        cam[:, 0] *= sx
        cam[:, 1] *= sy

        moved = palm_frame_transfer(glove, cam)
        # placed, not turned: the shape that arrives is the shape that leaves
        assert np.allclose(moved - moved[0], cam - cam[0], atol=1e-9), label
        assert np.allclose(moved[0], glove[0], atol=1e-12), label   # wrist on wrist
        assert _tilt_deg(palm_basis(glove) @ palm_basis(cam).T) < 1e-9, label

        R, _s, _t = umeyama(cam[PALM_IDX], glove[PALM_IDX], with_scale=False)
        worst_kabsch = max(worst_kabsch, _tilt_deg(R))

    assert worst_kabsch > 0.25, (
        "if the 5-point fit no longer tilts on a reshaped palm, this test has "
        "stopped testing anything")

    # ...and a hand that is simply bigger, same proportions, fools neither.
    even = make_hand(spread_deg=0.0, curl=0.3) * 1.10
    R, _s, _t = umeyama(even[PALM_IDX], glove[PALM_IDX], with_scale=False)
    assert _tilt_deg(R) < 1e-6


def test_the_kabsch_residual_is_reported_and_never_gates_a_frame():
    """A palm that does not match the template is a fact, not a rejection."""
    glove = make_hand(spread_deg=0.0, curl=0.4)
    cam = make_hand(spread_deg=20.0, curl=0.4) * 1.5      # wildly wrong size

    fused, info = fuse_skeletons(glove, cam, with_scale=False)
    assert info["kabsch_rmse_mm"] > 10.0     # tens of mm out
    assert info["camera_used"] is True       # ...and still used, by design
    assert info["fingers_adjusted"]
    # the MediaPipe path has no metric residual to report
    _f, mp = fuse_skeletons(glove, cam, with_scale=True)
    assert mp["kabsch_rmse_mm"] is None and mp["alignment"] == "similarity"


# --- gating: when the camera may and may not supply a DOF ---------------
# The first real simultaneous session is the source of every number here.
# The two cases that matter are pinch (the camera is right, the glove is
# blind) and thumbs_up (the glove is right, the camera is confidently wrong),
# and a gate that cannot tell them apart is not a gate.

def facing_meta(view_deg=0.0, **over):
    """Camera metadata for a hand held above the module, palm turned by view_deg.

    The module is the origin, so a palm at (0, h, 0) is seen along -y; tilting
    the normal away from that ray by `view_deg` is exactly what the gate
    measures.
    """
    meta = {"visible_time_us": 5_000_000, "hand_id_stable": True,
            "palm_abs": [0.0, 0.25, 0.0],
            "palm_normal_abs": [math.sin(math.radians(view_deg)),
                                -math.cos(math.radians(view_deg)), 0.0]}
    meta.update(over)
    return meta


def hand_with_curls(curls, spread_deg=0.0):
    """A synthetic hand whose five flexion features are exactly `curls`.

    `make_hand`'s `curl` is a bend ANGLE; the gate reads the FEATURE
    (tip-to-wrist over palm length). Each finger is therefore bent as far as
    the angle can take it toward its target and its bones are then scaled
    about the knuckle to land on it — a real hand is both longer and more
    foldable than this toy one, reaching 2.07 palm lengths open and 0.62
    curled where the toy spans 0.97 to 1.83. Knuckles never move, so palm
    length, and with it every other finger's feature, is unaffected.
    """
    pts = make_hand(spread_deg=spread_deg, curl=0.0)
    wrist = pts[0].copy()
    palm = float(np.linalg.norm(pts[9] - wrist))
    for k, chain in enumerate(FINGER_CHAINS.values()):
        target = curls[k] * palm
        lo, hi = 0.0, 3.0          # past 90 degrees: a real fist folds under
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            trial = make_hand(spread_deg=spread_deg, curl=mid)
            if float(np.linalg.norm(trial[chain[-1]] - wrist)) > target:
                lo = mid
            else:
                hi = mid
        seg = make_hand(spread_deg=spread_deg, curl=0.5 * (lo + hi))[chain]
        knuckle = seg[0].copy()
        # |knuckle - wrist + s*v| = target, solved rather than searched: with
        # the tip folded back past the wrist the distance is not monotonic in
        # s and a bisection would find the wrong side of the turn.
        v = seg[-1] - knuckle
        w = knuckle - wrist
        qa = float(v @ v)
        qb = 2.0 * float(w @ v)
        qc = float(w @ w) - target * target
        disc = qb * qb - 4.0 * qa * qc
        s = (-qb + math.sqrt(disc)) / (2.0 * qa) if disc >= 0 else 1.0
        pts[chain] = knuckle + s * (seg - knuckle)
    assert np.allclose(flexion_features(pts), curls, atol=1e-6), (
        flexion_features(pts), curls)
    return pts


# Medians measured in recordings/sync, take 1, over the 5 s each pose was held.
OPEN_GLOVE_CURLS = [1.43, 1.97, 2.07, 1.97, 1.71]
FIST_GLOVE_CURLS = [0.95, 0.73, 0.72, 0.71, 0.73]
PINCH_GLOVE_CURLS = OPEN_GLOVE_CURLS          # identical, to the last digit
PINCH_CAM_CURLS = [1.28, 1.31, 1.79, 1.69, 1.49]
THUMBSUP_GLOVE_CURLS = [1.43, 0.66, 0.62, 0.70, 1.02]
THUMBSUP_CAM_CURLS = [1.18, 1.68, 1.82, 1.44, 1.22]


def test_viewing_angle_is_zero_when_the_palm_faces_the_module():
    # palm 25 cm above the module, normal pointing straight back down at it
    assert viewing_angle_deg([0.0, 0.25, 0.0], [0.0, -1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    # edge-on: the normal is perpendicular to the ray
    assert viewing_angle_deg([0.0, 0.25, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(90.0, abs=1e-9)
    # back of the hand
    assert viewing_angle_deg([0.0, 0.25, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(180.0, abs=1e-9)
    # and it is the RAY, not the y axis: an off-axis palm is measured from
    # where it actually is
    assert viewing_angle_deg([0.25, 0.25, 0.0], [0.0, -1.0, 0.0]) == pytest.approx(45.0, abs=1e-6)


def test_the_central_field_gate_is_lateral_offset_against_height():
    assert palm_field_angle_deg([0.0, 0.3, 0.0]) == pytest.approx(0.0, abs=1e-9)
    assert palm_field_angle_deg([0.3, 0.3, 0.0]) == pytest.approx(45.0, abs=1e-6)
    ok, reasons, _m = frame_trust(facing_meta(palm_abs=[0.10, 0.30, 0.0]))
    assert ok and reasons == []
    ok, reasons, _m = frame_trust(facing_meta(palm_abs=[0.40, 0.30, 0.0]))
    assert not ok and R_FIELD in reasons
    # a palm level with or below the module is outside it by construction
    assert palm_field_angle_deg([0.1, 0.0, 0.0]) >= 90.0


def test_frame_trust_rejects_a_fresh_or_unsettled_track():
    ok, reasons, metrics = frame_trust(facing_meta(visible_time_us=299_999))
    assert not ok and R_VISIBLE in reasons
    assert frame_trust(facing_meta(visible_time_us=300_000))[0]
    ok, reasons, _m = frame_trust(facing_meta(hand_id_stable=False))
    assert not ok and R_HAND_ID in reasons
    # the numbers it measured are reported whether or not it passed
    assert metrics["view_angle_deg"] == pytest.approx(0.0, abs=1e-9)


def test_hand_id_stability_is_flagged_per_hand_over_time():
    rows = [{"hand_side": "right", "capture_time": t, "hand_id": hid}
            for t, hid in [(0.0, 7), (0.1, 7), (0.2, 9), (0.3, 9),
                           (0.5, 9), (0.6, 9)]]
    # a left hand interleaved, whose own id never changes
    rows += [{"hand_side": "left", "capture_time": t, "hand_id": 3}
             for t in (0.05, 0.25, 0.55)]
    flag_hand_id_stability(rows, DEFAULT_GATES, clock="capture_time")
    right = [r["hand_id_stable"] for r in rows if r["hand_side"] == "right"]
    assert right == [True, True, False, False, True, True], (
        "the change at 0.2 s must blank 0.25 s of frames and no more")
    assert all(r["hand_id_stable"] for r in rows if r["hand_side"] == "left"), (
        "the other hand's re-acquisition is not this hand's problem")


# --- the spread gate ---------------------------------------------------

def test_a_curled_finger_keeps_the_gloves_spread_and_an_open_one_does_not():
    """The glove decides whether the camera is allowed to bend the finger.

    A finger curled into the palm has no abduction left to see and its
    proximal bone points at the camera end-on, so its azimuth there is noise.
    """
    open_glove = hand_with_curls(OPEN_GLOVE_CURLS)
    cam = make_hand(spread_deg=25.0, curl=0.0)
    fused, info = fuse_skeletons(open_glove, cam, with_scale=False,
                                 cam_meta=facing_meta())
    assert all(info["dof_source"][f"spread {f}"] == "camera"
               for f in ("index", "middle", "ring", "pinky"))
    assert not np.allclose(fused, open_glove - open_glove[0], atol=1e-6)

    curled = hand_with_curls(FIST_GLOVE_CURLS)
    fused, info = fuse_skeletons(curled, cam, with_scale=False,
                                 cam_meta=facing_meta())
    for f in ("index", "middle", "ring", "pinky"):
        assert info["dof_source"][f"spread {f}"] == "glove"
        assert info["rejected"][f"spread {f}"] == R_CURLED


def test_an_edge_on_view_rejects_every_camera_dof():
    """thumbs_up in the real session: 70-78 degrees, and wrong about everything."""
    glove = make_hand(spread_deg=0.0, curl=0.0)
    cam = make_hand(spread_deg=25.0, curl=0.0)
    fused, info = fuse_skeletons(glove, cam, with_scale=False,
                                 cam_meta=facing_meta(view_deg=75.0))
    assert info["view_angle_deg"] == pytest.approx(75.0, abs=1e-6)
    assert set(info["rejected"]) == set(CAMERA_DOFS)
    assert all(v == R_VIEW for v in info["rejected"].values())
    assert np.allclose(fused, glove - glove[0], atol=1e-12)
    # ...and moving the gate past it lets the same frame straight back in
    loose = GateParams(view_gate_deg=80.0)
    _f, info = fuse_skeletons(glove, cam, with_scale=False, gates=loose,
                              cam_meta=facing_meta(view_deg=75.0))
    assert info["rejected"] == {}


# --- the thumb gate: the pinch / thumbs_up pair -------------------------
# Both hands are built from the real session's curl numbers, which is the
# whole argument for the rule: the same camera, the same tracker, one frame
# worth taking and one not, told apart without asking the tracker.

def test_pinch_passes_the_thumb_gate_and_thumbs_up_is_rejected():
    """One rule, both real cases, and it has to get both right.

    pinch      the glove is numerically its own open palm, and the camera
               agrees with it about index..little to within 0.28 — it is
               looking at the same hand, so its thumb is worth having.
    thumbs_up  the camera has the four curled fingers nearly straight. A
               camera that wrong about the fingers has the hand's orientation
               wrong, and orientation error moves the thumb most of all.
    """
    pinch_g = hand_with_curls(PINCH_GLOVE_CURLS)
    pinch_c = hand_with_curls(PINCH_CAM_CURLS)
    _f, info = fuse_skeletons(pinch_g, pinch_c, with_scale=False,
                              cam_meta=facing_meta(view_deg=40.0))
    assert info["curl_disagreement"] < DEFAULT_GATES.curl_agree_tol
    assert info["dof_source"]["thumb"] == "camera", info["rejected"]

    up_g = hand_with_curls(THUMBSUP_GLOVE_CURLS)
    up_c = hand_with_curls(THUMBSUP_CAM_CURLS)
    _f, info = fuse_skeletons(up_g, up_c, with_scale=False,
                              cam_meta=facing_meta(view_deg=40.0))
    assert info["curl_disagreement"] > DEFAULT_GATES.curl_agree_tol
    assert info["dof_source"]["thumb"] == "glove"
    assert info["rejected"]["thumb"] == R_DISAGREE

    # the real thumbs_up frames were ALSO edge-on, so the frame is refused
    # twice over — either rule alone is enough
    _f, info = fuse_skeletons(up_g, up_c, with_scale=False,
                              cam_meta=facing_meta(view_deg=75.0))
    assert info["rejected"]["thumb"] == R_VIEW


def test_the_thumb_gate_ignores_the_thumb_it_is_deciding_about():
    """Only index..little vote. A disagreeing thumb is the reason to look."""
    glove = hand_with_curls([1.43, 1.97, 2.07, 1.97, 1.71])
    cam = hand_with_curls([0.70, 1.95, 2.05, 1.95, 1.70])   # thumb miles off
    _f, info = fuse_skeletons(glove, cam, with_scale=False,
                              cam_meta=facing_meta(view_deg=10.0))
    assert info["curl_disagreement"] < 0.05
    assert info["dof_source"]["thumb"] == "camera"


# --- azimuth is read off the proximal bone ------------------------------

def test_azimuth_comes_from_the_proximal_bone_not_the_curled_tip():
    """A finger whose TIP is bent sideways must not look abducted.

    The index is left straight out of its knuckle and only its last two bones
    are swung 40 degrees across the palm. Knuckle -> tip then reports a large
    azimuth that the knuckle never produced; knuckle -> PIP reports the truth.
    """
    from cam_hand.features import proximal_azimuth_deg

    pts = make_hand(spread_deg=0.0, curl=0.0)
    chain = FINGER_CHAINS["index"]
    pivot = pts[chain[1]].copy()                  # bend at the PIP
    th = math.radians(40.0)
    R = np.array([[math.cos(th), -math.sin(th), 0.0],
                  [math.sin(th), math.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    pts[chain[2:]] = (R @ (pts[chain[2:]] - pivot).T).T + pivot

    straight = make_hand(spread_deg=0.0, curl=0.0)
    n, x, y = palm_frame(pts)

    def tip_az(p):
        d = np.asarray(p)[chain[-1]] - np.asarray(p)[chain[0]]
        d = d / np.linalg.norm(d)
        return math.degrees(math.atan2(float(np.dot(d, y)), float(np.dot(d, x))))

    # the tip moved a long way; the proximal bone did not move at all
    assert abs(tip_az(pts) - tip_az(straight)) > 15.0
    assert proximal_azimuth_deg(pts, "index") == pytest.approx(
        proximal_azimuth_deg(straight, "index"), abs=1e-9)
    assert np.allclose(proximal_direction(pts, "index"),
                       proximal_direction(straight, "index"), atol=1e-9)

    # and fusing against this camera hand must not swing the glove's index:
    # there is no abduction here, only flexion, which is the glove's already
    fused, _info = fuse_skeletons(straight, pts, with_scale=False)
    d = fused[chain[1]] - fused[chain[0]]
    got = math.degrees(math.atan2(float(np.dot(d / np.linalg.norm(d), y)),
                                  float(np.dot(d / np.linalg.norm(d), x))))
    assert got == pytest.approx(
        proximal_azimuth_deg(straight, "index"), abs=1e-6)


def test_adopting_a_camera_azimuth_leaves_every_curl_untouched():
    """The chain turns about the palm normal, so curl survives exactly."""
    glove = make_hand(spread_deg=0.0, curl=0.7)
    cam = make_hand(spread_deg=25.0, curl=0.0)
    fused, _info = fuse_skeletons(glove, cam, thumb_from_camera=False,
                                  with_scale=False)
    n, _x, _y = palm_frame(glove)
    for chain in FINGER_CHAINS.values():
        for a, b in zip(chain, chain[1:]):
            def out(p):
                d = np.asarray(p)[b] - np.asarray(p)[a]
                return float(np.dot(d / np.linalg.norm(d), n))
            assert out(fused) == pytest.approx(out(glove), abs=1e-12)


# --- bookkeeping -------------------------------------------------------

def test_every_frame_reports_where_each_dof_came_from_and_why():
    glove = make_hand(spread_deg=0.0, curl=0.0)
    cam = make_hand(spread_deg=25.0, curl=0.0)

    _f, info = fuse_skeletons(glove, cam, with_scale=False,
                              cam_meta=facing_meta())
    assert info["gated"] is True
    assert set(info["dof_source"]) == set(CAMERA_DOFS)
    assert set(info["dof_source"].values()) == {"camera"}
    assert info["rejected"] == {}
    # every DOF is either sourced from the camera or carries a reason it is not
    for dof in CAMERA_DOFS:
        assert (info["dof_source"][dof] == "camera") != (dof in info["rejected"])

    # a frame-level failure names itself on every DOF, and nothing is dropped
    fused, info = fuse_skeletons(glove, cam, with_scale=False,
                                 cam_meta=facing_meta(visible_time_us=1000))
    assert info["frame_reasons"] == [R_VISIBLE]
    assert set(info["rejected"].values()) == {R_VISIBLE}
    assert np.allclose(fused, glove, atol=1e-12), "a rejected frame is the glove"

    # so does having no camera frame at all
    _f, info = fuse_skeletons(glove, None)
    assert set(info["rejected"].values()) == {R_NO_FRAME}


def test_an_ungated_mediapipe_frame_is_fused_exactly_as_before():
    """No capture facts to read is not a reason to reject a whole sensor.

    A MediaPipe take has no absolute palm, no hand id and no visibility
    clock. It passes cam_meta=None, every camera DOF is taken, and `gated`
    says so rather than the report inventing a 0% camera-use rate for it.
    """
    glove = make_hand(spread_deg=0.0, curl=0.9)     # a fist: would be gated out
    cam = make_hand(spread_deg=25.0, curl=0.9)
    fused, info = fuse_skeletons(glove, cam, with_scale=True)
    assert info["gated"] is False
    assert info["rejected"] == {}
    assert set(info["dof_source"].values()) == {"camera"}
    assert info["fingers_adjusted"] == list(FINGER_CHAINS)
    assert not np.allclose(fused, glove - glove[0], atol=1e-6)


def test_gate_thresholds_are_named_parameters_and_are_reported():
    described = DEFAULT_GATES.described()
    assert described == {"curl_gate": 1.2, "view_gate_deg": 50.0,
                         "curl_agree_tol": 0.35, "min_visible_time_us": 300_000,
                         "hand_id_settle_s": 0.25, "field_half_angle_deg": 45.0}
    # and they are honoured, not just stored
    glove = make_hand(spread_deg=0.0, curl=0.0)
    cam = make_hand(spread_deg=25.0, curl=0.0)
    strict = GateParams(curl_gate=9.9)
    _f, info = fuse_skeletons(glove, cam, with_scale=False, gates=strict,
                              cam_meta=facing_meta())
    assert info["rejected"]["spread index"] == R_CURLED
