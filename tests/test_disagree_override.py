"""The trusted-camera DISAGREEMENT override (`RailOverrideParams(mode="disagree")`).

The failure it exists for, seen live on 2026-09-25: the glove drifted open
within a session while a trusted, palm-facing camera saw a full fist. No
finger sat on its rail, so the rail rule could not fire and the fused hand
followed the glove.
"""
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from cam_hand.features import flexion_features
from cam_hand.fusion import (
    DEFAULT_GATES,
    FINGER_CHAINS,
    FINGER_NAMES,
    MODE_DISAGREE,
    MODE_RAIL,
    RAIL_FINGERS,
    R_DIS_AGREE,
    R_DIS_ARMING,
    R_DIS_NO_SCALE,
    R_RAIL_EXTENDED,
    R_RAIL_OFF,
    R_VIEW,
    R_VISIBLE,
    RailOverrideParams,
    RailOverrideTracker,
    bend_finger_to_curl,
    fuse_skeletons,
    learn_flexion_scale,
    learn_rails,
)

HAND = "right"


def _fuse_module():
    """scripts/fuse_poses.py, which is not on a package path."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "fuse_poses.py"
    spec = importlib.util.spec_from_file_location("fuse_poses_disagree", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def open_hand():
    """A flat synthetic right hand, wrist at the origin, metres, fingers +y."""
    pts = np.zeros((21, 3))
    knuckle = {"thumb": (-0.035, 0.03), "index": (-0.02, 0.085),
               "middle": (0.0, 0.09), "ring": (0.02, 0.085),
               "pinky": (0.038, 0.075)}
    for finger, chain in FINGER_CHAINS.items():
        x, y = knuckle[finger]
        for k, joint in enumerate(chain):
            pts[joint] = (x, y + 0.025 * k, 0.0)
    return pts


OPEN = flexion_features(open_hand())
# A fist on the same hand: every finger folded to about 0.75 palm lengths.
FIST = [OPEN[0], 0.75, 0.72, 0.75, 0.78]


def hand_at(curls):
    """`open_hand` with each of index..pinky bent to its curl in `curls`."""
    pts = open_hand()
    for finger in RAIL_FINGERS:
        i = FINGER_NAMES.index(finger)
        pts = bend_finger_to_curl(pts, finger, curls[i])
    return pts


def at_fraction(frac):
    """Curls with every finger `frac` of the way from open (0) to fist (1)."""
    return [OPEN[0]] + [o - frac * (o - f)
                        for o, f in zip(OPEN[1:], FIST[1:])]


def facing_meta(view_deg=0.0, **over):
    """A trusted camera frame: palm above the module, turned by view_deg."""
    meta = {"visible_time_us": 5_000_000, "hand_id_stable": True,
            "palm_abs": [0.0, 0.25, 0.0],
            "palm_normal_abs": [math.sin(math.radians(view_deg)),
                                -math.cos(math.radians(view_deg)), 0.0]}
    meta.update(over)
    return meta


def session():
    """Rails and endpoints from 100 open and 100 fist frames on both sensors."""
    glove = [(HAND, list(OPEN))] * 100 + [(HAND, list(FIST))] * 100
    rails = learn_rails(glove)
    scale = learn_flexion_scale(glove, glove, rails)
    return rails, scale


def disagree_params(**over):
    kw = dict(mode=MODE_DISAGREE, enter_frames=5, exit_frames=3)
    kw.update(over)
    return RailOverrideParams(**kw)


def run_take(glove_fracs, cam_frac=1.0, meta=None, **over):
    """Drive a tracker and `fuse_skeletons` over one take.

    Returns per frame (decision, fused curls, glove curls, camera curls).
    """
    rails, scale = session()
    tracker = RailOverrideTracker(rails, disagree_params(**over), scale=scale)
    cam_curls = at_fraction(cam_frac)
    C = hand_at(cam_curls)
    c_feat = flexion_features(C)
    out = []
    for f in glove_fracs:
        G = hand_at(at_fraction(f))
        g_feat = flexion_features(G)
        m = facing_meta() if meta is None else meta
        d = tracker.update(HAND, g_feat, c_feat, m)
        fused, _info = fuse_skeletons(G, C, with_scale=False, cam_meta=m,
                                      rail=d, scale=scale.for_hand(HAND))
        out.append((d, flexion_features(fused), g_feat, c_feat))
    return out


# --- the failure it was built for ----------------------------------------

def test_a_glove_drifting_open_under_a_trusted_fist_is_overruled_then_released():
    """Fist on both sensors, the glove drifts open, then recovers.

    The fused curls must follow the camera from exactly `enter_frames` frames
    after the gap first reaches `disagree_frac`, and go back to the glove
    `exit_frames` frames after the glove recovers.
    """
    params = disagree_params()
    fist = [1.0] * 8
    drift = [1.0 - 0.1 * k for k in range(1, 10)]          # 0.9 .. 0.1
    held_open = [0.1] * 10
    recovered = [1.0] * 8
    fracs = fist + drift + held_open + recovered
    frames = run_take(fracs)

    first_gap = next(k for k, f in enumerate(fracs)
                     if 1.0 - f >= params.disagree_frac - 1e-9)
    armed = first_gap + params.enter_frames - 1
    back = len(fist) + len(drift) + len(held_open)
    released = back + params.exit_frames - 1

    for k, (d, fused, glove, cam) in enumerate(frames):
        active = armed <= k < released
        assert set(d.active) == (set(RAIL_FINGERS) if active else set()), k
        for finger in RAIL_FINGERS:
            i = FINGER_NAMES.index(finger)
            want = cam[i] if active else glove[i]
            assert fused[i] == pytest.approx(want, abs=1e-6), (k, finger)
    # the gap really was the camera seeing the fist the glove lost
    d, fused, glove, cam = frames[armed + 3]
    assert all(d.gaps[f] >= params.disagree_frac for f in RAIL_FINGERS)
    assert all(abs(glove[i] - cam[i]) > 0.3 for i in range(1, 5))
    # before it armed, the reason given is that it was still arming
    assert frames[armed - 1][0].rejected["curl index"] == R_DIS_ARMING
    assert frames[0][0].rejected["curl index"] == R_DIS_AGREE


def test_no_rail_is_needed_but_a_railed_reading_still_qualifies():
    """The glove reporting its open-palm rail under a trusted fist fires too."""
    frames = run_take([0.0] * 6)
    assert set(frames[-1][0].active) == set(RAIL_FINGERS)


def test_a_gap_between_release_and_entry_holds_the_override():
    """Hysteresis on the gap: 0.25 is under entry but over release."""
    params = disagree_params()
    frames = run_take([0.1] * 6 + [0.75] * 20 + [0.95] * 4)
    assert set(frames[5][0].active) == set(RAIL_FINGERS)
    # gap 0.25: never enough to arm, but enough to stay armed
    assert all(set(f[0].active) == set(RAIL_FINGERS) for f in frames[6:26])
    # gap 0.05: released after exit_frames
    tail = [set(f[0].active) for f in frames[26:]]
    assert tail[params.exit_frames - 2] == set(RAIL_FINGERS)
    assert tail[params.exit_frames - 1] == set()


# --- when it must not fire -------------------------------------------------

def test_no_firing_when_the_palm_is_turned_away():
    frames = run_take([0.1] * 20, meta=facing_meta(view_deg=80.0))
    assert all(f[0].active == () for f in frames)
    assert frames[-1][0].rejected["curl index"] == R_VIEW
    assert frames[-1][0].gaps == {}, "an untrusted frame reports no gap"


def test_no_firing_when_the_camera_frame_is_not_trusted():
    frames = run_take([0.1] * 20, meta=facing_meta(visible_time_us=1000))
    assert all(f[0].active == () for f in frames)
    assert frames[-1][0].rejected["curl index"] == R_VISIBLE


def test_no_firing_under_the_threshold():
    """Glove at 0.70 of a fist under a camera fist: a 0.30 gap is not enough."""
    frames = run_take([0.70] * 30)
    assert all(f[0].active == () for f in frames)
    assert frames[-1][0].rejected["curl ring"] == R_DIS_AGREE
    assert frames[-1][0].gaps["ring"] == pytest.approx(0.30, abs=1e-6)


def test_no_firing_without_endpoints():
    """No scale, no fractions, no gap: "disagree" cannot fire blind."""
    rails, _scale = session()
    tracker = RailOverrideTracker(rails, disagree_params(enter_frames=1))
    d = tracker.update(HAND, at_fraction(0.0), at_fraction(1.0), facing_meta())
    assert d.active == ()
    assert d.rejected["curl index"] == R_DIS_NO_SCALE


def test_the_reverse_direction_needs_both_ways():
    """Glove calls the finger curled, the camera sees it straight."""
    default = run_take([1.0] * 10, cam_frac=0.0)
    assert all(f[0].active == () for f in default)
    assert default[-1][0].gaps["index"] == pytest.approx(-1.0, abs=1e-6)
    both = run_take([1.0] * 10, cam_frac=0.0, disagree_both_ways=True)
    assert set(both[-1][0].active) == set(RAIL_FINGERS)


def test_an_agreeing_open_palm_never_fires():
    """False-fire check: both sensors open, jittered, 300 frames, both ways."""
    rails, scale = session()
    rng = random.Random(7)
    for both in (False, True):
        tracker = RailOverrideTracker(
            rails, disagree_params(disagree_both_ways=both, enter_frames=1),
            scale=scale)
        for _ in range(300):
            g = [c - abs(rng.gauss(0, 0.03)) for c in OPEN]
            c = [c - abs(rng.gauss(0, 0.03)) for c in OPEN]
            assert tracker.update(HAND, g, c, facing_meta()).active == ()


def test_the_report_counts_false_fires_per_finger_on_open_palm_and_fist():
    """`record_override` and `override_lines` over two synthetic takes.

    An agreeing open palm must report zero; a fist whose glove drifted open
    is reported as firing (the table does not know which fires were right,
    which is why fist is watched).
    """
    fuse = _fuse_module()
    rails, scale = session()
    params = disagree_params()
    run = fuse.FusionRun(rail_params=params)
    hs = scale.for_hand(HAND)
    for pose, glove_frac, cam_frac in (("open_palm", 0.0, 0.0),
                                       ("fist", 0.1, 1.0)):
        tracker = RailOverrideTracker(rails, params, scale=scale)
        G = hand_at(at_fraction(glove_frac))
        C = hand_at(at_fraction(cam_frac))
        g, c = flexion_features(G), flexion_features(C)
        for _ in range(12):
            d = tracker.update(HAND, g, c, facing_meta())
            fused, info = fuse_skeletons(G, C, with_scale=False,
                                         cam_meta=facing_meta(), rail=d,
                                         scale=hs)
            fuse.record_override(run, HAND, pose, d, info, facing_meta(),
                                 DEFAULT_GATES, hs, g, c,
                                 flexion_features(fused))
    lines = []
    fuse.override_lines(run, lines)
    text = "\n".join(lines)
    false_fire = text.split("FALSE-FIRE check")[1].splitlines()
    fired = 12 - params.enter_frames + 1

    def row(pose):
        return next(line.split() for line in false_fire
                    if line.split()[:2] == [HAND, pose])

    assert row("open_palm")[2:] == ["12", "0", "0", "0", "0", "0"]
    assert row("fist")[2:] == ["12"] + [str(fired)] * 4 + [str(4 * fired)]
    assert f"false fires: {4 * fired} of {2 * 12 * 4} finger-frames" in text
    # the residual: exactly the fired frames moved onto the camera
    res = run.residual[(HAND, "index")]
    assert res["fired"] == fired
    assert sum(1 for b, a in zip(res["before"], res["after"])
               if a < b - 0.5) == fired


# --- "rail" mode is unchanged ---------------------------------------------

RAIL_VALUE = 1.9740


def _rail_frames(seed=3, segments=80):
    """A long mixed sequence of short runs: on and off the rail, flexed and
    straight camera, trusted and untrusted frames, missing camera frames."""
    rng = random.Random(seed)
    metas = [facing_meta(), facing_meta(), facing_meta(view_deg=70.0),
             facing_meta(visible_time_us=10), None]
    out = []
    for _ in range(segments):
        on_rail = rng.random() < 0.6
        flexed = rng.random() < 0.6
        meta = rng.choice(metas)
        missing = rng.random() < 0.1
        for _ in range(rng.randint(1, 8)):
            idx = (RAIL_VALUE + rng.uniform(-0.002, 0.002) if on_rail
                   else 1.2 + rng.random() * 0.7)
            glove = [1.2, idx, 1.2 + rng.random() * 0.8, 1.2, 1.2]
            cam = [1.2, (1.2 if flexed else 1.6) + rng.random() * 0.25,
                   1.8, 1.8, 1.8]
            out.append((glove, None if missing else cam, meta))
    return out


def test_rail_mode_ignores_the_scale_and_every_disagree_parameter():
    """Decision for decision, the rail rule is what it was.

    The tracker built the old way (no scale, default parameters) and one
    handed a scale and wildly different disagree settings must agree on
    every frame of a long mixed sequence, and so must the fused hands.
    """
    rails = {("right", "index"): RAIL_VALUE, ("right", "middle"): 2.07}
    fingers = ("index", "middle")
    old = RailOverrideTracker(rails, RailOverrideParams(fingers=fingers,
                                                        enter_frames=3,
                                                        exit_frames=2))
    _r, scale = session()
    new = RailOverrideTracker(
        rails, RailOverrideParams(fingers=fingers, enter_frames=3,
                                  exit_frames=2, mode=MODE_RAIL,
                                  disagree_frac=0.01, disagree_release=0.0,
                                  disagree_both_ways=True,
                                  disagree_fingers=("pinky",)),
        scale=scale)
    n_active = 0
    for glove, cam, meta in _rail_frames():
        a = old.update("right", glove, cam, meta)
        b = new.update("right", glove, cam, meta)
        assert (a.active, dict(a.rejected), a.disputed) == (
            b.active, dict(b.rejected), b.disputed)
        n_active += len(a.active)
        if cam is not None:
            G, C = hand_at(glove), hand_at(cam)
            fa, ia = fuse_skeletons(G, C, with_scale=False, cam_meta=meta,
                                    rail=a)
            fb, ib = fuse_skeletons(G, C, with_scale=False, cam_meta=meta,
                                    rail=b)
            assert np.array_equal(fa, fb)
            assert ia["dof_source"] == ib["dof_source"]
    assert n_active > 20, "the sequence must actually exercise the override"


def test_rail_mode_reasons_are_the_rail_rules():
    tracker = RailOverrideTracker({("right", "index"): RAIL_VALUE},
                                  RailOverrideParams(enter_frames=3))
    off = tracker.update("right", [1.2, 1.5, 1.2, 1.2, 1.2],
                         [1.2, 1.2, 1.8, 1.8, 1.8], facing_meta())
    assert off.rejected["curl index"] == R_RAIL_OFF
    straight = tracker.update("right", [1.2, RAIL_VALUE, 1.2, 1.2, 1.2],
                              [1.2, 1.8, 1.8, 1.8, 1.8], facing_meta())
    assert straight.rejected["curl index"] == R_RAIL_EXTENDED


# --- parameters, profile and report ---------------------------------------

def test_disagree_parameters_are_named_checked_and_reported():
    p = RailOverrideParams()
    assert (p.mode, p.disagree_frac, p.disagree_release,
            p.disagree_both_ways) == (MODE_RAIL, 0.35, 0.20, False)
    assert p.fingers_for("left") == ("index",)
    d = RailOverrideParams(mode=MODE_DISAGREE)
    assert d.fingers_for("left") == RAIL_FINGERS
    assert d.fingers_for() == RAIL_FINGERS
    per_hand = RailOverrideParams(mode=MODE_DISAGREE,
                                  disagree_fingers={"left": ("ring",)})
    assert per_hand.fingers_for("left") == ("ring",)
    assert per_hand.fingers_for("right") == ()

    rail_text = p.described()
    assert rail_text["mode"] == MODE_RAIL
    assert "disagree_frac" not in rail_text
    assert rail_text["fingers"] == "index (both hands)"
    dis_text = d.described()
    assert dis_text["mode"] == MODE_DISAGREE
    assert dis_text["disagree_frac"] == 0.35
    assert dis_text["fingers"] == "index, middle, ring, pinky (both hands)"

    with pytest.raises(ValueError):
        RailOverrideParams(mode="sometimes")
    with pytest.raises(ValueError):
        RailOverrideParams(disagree_frac=0.15, disagree_release=0.20)


def test_a_profile_may_name_the_override_mode(tmp_path):
    fuse = _fuse_module()

    def profile(**data):
        path = tmp_path / f"p{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    assert fuse.profile_override_mode(profile()) is None
    assert fuse.profile_override_mode(
        profile(override_mode="disagree")) == MODE_DISAGREE
    # the five-value return other scripts unpack is unchanged
    assert len(fuse.load_profile(profile(override_mode="rail"))) == 5
    with pytest.raises(SystemExit):
        fuse.load_profile(profile(override_mode="always"))
    assert fuse.profile_override_mode(None) is None


def test_the_report_names_the_mode():
    fuse = _fuse_module()
    assert fuse.mode_text(None).startswith("off")
    assert fuse.mode_text(RailOverrideParams()).startswith("rail")
    text = fuse.mode_text(RailOverrideParams(mode=MODE_DISAGREE,
                                             disagree_both_ways=True))
    assert text.startswith("disagree") and "either direction" in text
