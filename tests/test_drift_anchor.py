"""The camera drift anchor (EXPERIMENTAL, off by default) and the finger
bend both glove corrections share.

A held glove finger creeps (0.05 to 0.26 curl per minute on the Reality
Glove) and the camera does not, but the camera can only be believed on some
frames. These tests pin down what makes the anchor safe to try: it learns
only from frames that can teach it, it corrects every frame without ever
changing a bone length, it forgets everything at a take boundary (the
glove's error is mostly pose-dependent), and with the anchor off, or with
nothing to correct, the fusion is exactly what it was.
"""
import numpy as np
import pytest

from cam_hand.features import flexion_features
from cam_hand.fusion import (
    DEFAULT_ANCHOR,
    FINGER_CHAINS,
    FINGER_NAMES,
    RAIL_FINGERS,
    SENSOR_CAMERA,
    SENSOR_GLOVE,
    DriftAnchor,
    DriftAnchorParams,
    Endpoints,
    GateParams,
    HandScale,
    bend_finger_to_curl,
    fuse_skeletons,
)

from test_fusion import (
    FIST_CAM_CURLS,
    FIST_GLOVE_CURLS,
    OPEN_CAM_CURLS,
    OPEN_GLOVE_CURLS,
    _fuse_module,
    bone_lengths,
    curled_hand,
    facing_meta,
    hand_with_curls,
)

HZ = 60.0                   # the glove's frame rate
TOL = 0.005                 # RailOverrideParams.tol, the on-rail test


def _scale(hand="right", dropped=None):
    """Endpoints on both sensors: glove open = its rail, camera its own."""
    ends = {}
    for i, f in enumerate(FINGER_NAMES):
        ends[(SENSOR_GLOVE, f)] = Endpoints(open=OPEN_GLOVE_CURLS[i],
                                            flexed=FIST_GLOVE_CURLS[i], n=100)
        ends[(SENSOR_CAMERA, f)] = Endpoints(open=OPEN_CAM_CURLS[i],
                                             flexed=FIST_CAM_CURLS[i], n=100)
    return HandScale(hand=hand, ends=ends, dropped=dict(dropped or {}))


RAILS = {("right", f): OPEN_GLOVE_CURLS[i] for i, f in enumerate(FINGER_NAMES)}


def _at(open_, fist, frac):
    """Curls at one flexion fraction on one sensor's endpoints."""
    return [o - frac * (o - c) for o, c in zip(open_, fist)]


def _glove(frac):
    return _at(OPEN_GLOVE_CURLS, FIST_GLOVE_CURLS, frac)


def _cam(frac):
    return _at(OPEN_CAM_CURLS, FIST_CAM_CURLS, frac)


# Everything that fuses POINTS uses `curled_hand`, whose fingers roll up bone
# by bone like a real finger (1.79 open to 0.85 closed on the index), on both
# sensors. `hand_with_curls` reaches the glove's real numbers by folding a
# straight finger past vertical, and the spread transfer then reads that
# finger's azimuth as pointing backwards, which is a fact about the toy and
# would drown what these tests measure.
TOY_OPEN = flexion_features(curled_hand(curl=0.0))
TOY_FIST = flexion_features(curled_hand(curl=1.0))
TOY_RAILS = {("right", f): TOY_OPEN[i] for i, f in enumerate(FINGER_NAMES)}


def _toy_scale(hand="right"):
    ends = {}
    for i, f in enumerate(FINGER_NAMES):
        for sensor in (SENSOR_GLOVE, SENSOR_CAMERA):
            ends[(sensor, f)] = Endpoints(open=TOY_OPEN[i], flexed=TOY_FIST[i],
                                          n=100)
    return HandScale(hand=hand, ends=ends)


def _toy_span(finger):
    i = FINGER_NAMES.index(finger)
    return TOY_OPEN[i] - TOY_FIST[i]


def _creep(hand, curls):
    """`hand` with each of the four fingers bent to its entry in `curls`."""
    for finger in RAIL_FINGERS:
        hand = bend_finger_to_curl(hand, finger,
                                   curls[FINGER_NAMES.index(finger)])
    return hand


def _teach(anchor, t0, n, residual, glove_frac=0.5, hand="right", **kw):
    """`n` trusted frames at HZ whose camera reads `residual` more flexed."""
    for k in range(n):
        anchor.learn(hand, t0 + k / HZ, _glove(glove_frac),
                     _cam(glove_frac + residual), facing_meta(), _scale(hand),
                     RAILS, TOL, **kw)
    return t0 + (n - 1) / HZ


# --- bending a finger to a curl ----------------------------------------

def _turned(before, after, a, b):
    """Angle, radians, the bone a -> b turned through between two hands."""
    u = before[b] - before[a]
    v = after[b] - after[a]
    c = float(u @ v) / float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.arccos(min(1.0, max(-1.0, c))))


def test_bend_finger_to_curl_hits_the_target_and_keeps_every_bone():
    for base in (curled_hand(curl=0.3), hand_with_curls(_glove(0.5))):
        before = flexion_features(base)
        frozen = base.copy()
        for finger in RAIL_FINGERS:
            i = FINGER_NAMES.index(finger)
            for target in (before[i] - 0.3, before[i] - 0.05,
                           before[i] + 0.05):
                out = bend_finger_to_curl(base, finger, target)
                after = flexion_features(out)
                assert after[i] == pytest.approx(target, abs=1e-3)
                # a rigid rotation about the knuckle: no bone changes length
                assert bone_lengths(out) == pytest.approx(bone_lengths(base),
                                                          abs=1e-12)
                # ...and nothing but that finger's PIP, DIP and tip moved
                chain = FINGER_CHAINS[finger]
                others = [k for k in range(21) if k not in chain[1:]]
                assert np.array_equal(out[others], base[others])
                # the bend is SHARED: the proximal bone turns by half of
                # what the distal bone turns, 0.5 of t against all of t
                assert _turned(base, out, chain[0], chain[1]) == \
                    pytest.approx(0.5 * _turned(base, out, chain[2],
                                                chain[3]), abs=1e-6)
                assert [c for k, c in enumerate(after) if k != i] == \
                    pytest.approx([c for k, c in enumerate(before) if k != i],
                                  abs=1e-12)
        assert np.array_equal(base, frozen), "pure: the input is not touched"


def test_folding_goes_toward_the_palm_on_either_hand():
    """The knuckle-built normal flips with the hand; the fold must not.

    These toy hands fold toward -z. A mirror image (x -> -x) is the other
    hand, whose knuckle normal points the other way, and its fingers must
    still fold toward -z.
    """
    right = curled_hand(curl=0.3)
    left = right * np.array([-1.0, 1.0, 1.0])
    for hand in (right, left):
        tip = FINGER_CHAINS["index"][-1]
        target = flexion_features(hand)[1] - 0.2
        out = bend_finger_to_curl(hand, "index", target)
        assert out[tip][2] < hand[tip][2] - 0.01
        assert flexion_features(out)[1] == pytest.approx(target, abs=1e-6)


def test_an_unreachable_curl_gets_the_nearest_reachable_one():
    straight = hand_with_curls(OPEN_GLOVE_CURLS)
    before = flexion_features(straight)[1]
    # past straight: nothing in range reads longer, so the finger stays
    out = bend_finger_to_curl(straight, "index", before + 1.0)
    assert flexion_features(out)[1] == pytest.approx(before, abs=1e-9)
    # past any fist: it folds as far as the range allows, and not the wrong way
    out = bend_finger_to_curl(straight, "index", 0.0)
    got = flexion_features(out)[1]
    assert got < before - 0.5
    assert bone_lengths(out) == pytest.approx(bone_lengths(straight), abs=1e-12)


# --- learning only from frames that can teach ----------------------------

def test_the_offset_needs_min_frames_and_is_the_windowed_median():
    anchor = DriftAnchor(DriftAnchorParams(deadband=0.0))
    t = _teach(anchor, 0.0, DEFAULT_ANCHOR.min_frames - 1, 0.10)
    assert anchor.offset("right", "index", t) is None
    t = _teach(anchor, t + 1 / HZ, 1, 0.10)
    assert anchor.offset("right", "index", t) == pytest.approx(0.10, abs=1e-9)

    # 20 s at 0.10, then 10 s at 0.30: the window only remembers its last
    # window_s, which is inside the 0.30 stretch
    anchor = DriftAnchor(DriftAnchorParams(deadband=0.0))
    t = _teach(anchor, 0.0, int(20 * HZ), 0.10)
    t = _teach(anchor, t + 1 / HZ, int(10 * HZ), 0.30)
    assert anchor.offset("right", "index", t) == pytest.approx(0.30, abs=1e-9)
    assert anchor.learned[("right", "index")] == int(30 * HZ)


def test_the_deadband_is_continuous():
    anchor = DriftAnchor()
    for residual, applied in ((0.02, 0.0), (0.05, 0.02), (-0.05, -0.02),
                              (0.30, 0.27)):
        a = DriftAnchor()
        t = _teach(a, 0.0, 40, residual)
        assert a.offset("right", "ring", t) == pytest.approx(applied, abs=1e-9)
    assert anchor.params.deadband == 0.03


def test_disputed_on_rail_and_untrusted_frames_do_not_teach():
    anchor = DriftAnchor()
    scale = _scale()
    glove = _glove(0.5)
    glove[1] = OPEN_GLOVE_CURLS[1]          # index sits exactly on its rail
    cam = _cam(0.6)

    taught = anchor.learn("right", 0.0, glove, cam, facing_meta(), scale,
                          RAILS, TOL, disputed=("middle",))
    assert taught == ("ring", "pinky")
    assert anchor.learned[("right", "middle")] == 0
    assert anchor.learned[("right", "index")] == 0

    # nothing about a frame the camera cannot be believed on is learned
    for meta in (facing_meta(visible_time_us=1_000),    # a fresh track
                 facing_meta(hand_id_stable=False),     # just re-acquired
                 facing_meta(view_deg=70.0),            # edge-on: thumbs_up
                 None):                                 # no capture facts
        assert anchor.learn("right", 0.1, glove, cam, meta, scale, RAILS,
                            TOL) == ()
    assert anchor.learn("right", 0.1, glove, None, facing_meta(), scale,
                        RAILS, TOL) == ()
    assert anchor.learn("right", 0.1, glove, cam, facing_meta(), None,
                        RAILS, TOL) == ()

    # a finger the endpoints could not normalise has no fraction to compare
    dropped = _scale(dropped={"ring": "the glove's curl range is too small"})
    assert "ring" not in anchor.learn("right", 0.2, glove, cam, facing_meta(),
                                      dropped, RAILS, TOL)
    # a finger with no learned rail counts as off it
    no_index_rail = {k: v for k, v in RAILS.items() if k[1] != "index"}
    assert "index" in anchor.learn("right", 0.3, glove, cam, facing_meta(),
                                   scale, no_index_rail, TOL)
    # the thumb is never the anchor's: the thumb gate owns it
    assert anchor.learned[("right", "thumb")] == 0


# --- correcting every frame --------------------------------------------

def _drifting_session(anchors, seconds=60.0, creep=0.2, fuse_every=6):
    """A held half fist: a stable camera, and a glove creeping `creep` curl
    more open over `seconds`, fed at HZ to each anchor in `anchors`.

    Returns, per anchor, (t, true curl, raw curl, fused curl) for the index
    on every `fuse_every`-th frame (the anchor itself sees every frame; only
    the check is thinned, to keep the test fast), and the last glove hand and
    curls.
    """
    scale = _toy_scale()
    C = curled_hand(curl=0.5)
    truth = flexion_features(C)
    meta = facing_meta()
    out = [[] for _a in anchors]
    for k in range(int(seconds * HZ)):
        t = k / HZ
        G = _creep(C, [c + creep * t / seconds for c in truth])
        curls = flexion_features(G)
        for a, anchor in enumerate(anchors):
            anchor.learn("right", t, curls, truth, meta, scale, TOY_RAILS, TOL)
            corrected, _moved = anchor.apply("right", t, G, curls, scale)
            if k % fuse_every:
                continue
            assert bone_lengths(corrected) == pytest.approx(bone_lengths(G),
                                                            abs=1e-12)
            fused, _info = fuse_skeletons(corrected, C, with_scale=False,
                                          cam_meta=meta, scale=scale)
            out[a].append((t, truth[1], curls[1], flexion_features(fused)[1]))
    return out, G, curls


def test_a_creeping_glove_is_held_on_the_truth_by_a_stable_camera():
    creep, seconds = 0.2, 60.0
    anchor = DriftAnchor(DriftAnchorParams(deadband=0.0))
    banded = DriftAnchor()
    (frames, frames_banded), _G, _c = _drifting_session(
        [anchor, banded], seconds, creep, fuse_every=6)
    t_end, truth, raw, _fused = frames[-1]
    assert raw - truth > 0.19, "the raw glove ends 0.2 open"
    settled = [abs(f - tr) for t, tr, _r, f in frames
               if t >= anchor.params.window_s]
    assert max(settled) <= 0.03, max(settled)

    # with the deadband the correction stops short by at most the band (in
    # curl units) plus the median's half-window lag behind a steady creep
    anchor, frames = banded, frames_banded
    lag = anchor.params.window_s / 2 * creep / seconds
    bound = anchor.params.deadband * _toy_span("index") + lag + 0.005
    settled = [abs(f - tr) for t, tr, _r, f in frames
               if t >= anchor.params.window_s]
    assert max(settled) <= bound, (max(settled), bound)
    rows = {(r["hand"], r["finger"]): r for r in anchor.summary()}
    assert rows[("right", "index")]["learned"] == int(seconds * HZ)
    assert rows[("right", "index")]["corrected"] > 0


def test_frames_with_no_camera_keep_the_last_offset_until_hold_s():
    anchor = DriftAnchor()
    (frames,), G, curls = _drifting_session([anchor], 30.0, 0.2,
                                            fuse_every=10**9)
    t_end = (int(30.0 * HZ) - 1) / HZ
    held = anchor.offset("right", "index", t_end)
    assert held is not None and held > 0.0
    scale = _toy_scale()

    # no camera for a minute: nothing is learned, the offset still applies
    t = t_end + 60.0
    assert anchor.offset("right", "index", t) == held
    corrected, moved = anchor.apply("right", t, G, curls, scale)
    assert moved["index"] == pytest.approx(-held * _toy_span("index"),
                                           abs=1e-6)
    assert flexion_features(corrected)[1] == pytest.approx(
        curls[1] - held * _toy_span("index"), abs=1e-6)

    # past hold_s after its last trusted frame the offset describes a glove
    # that no longer exists, and nothing is applied
    t = t_end + anchor.params.hold_s + 0.1
    assert anchor.offset("right", "index", t) is None
    corrected, moved = anchor.apply("right", t, G, curls, scale)
    assert moved == {} and np.array_equal(corrected, np.asarray(G))


def test_reset_forgets_every_offset_and_keeps_the_counters():
    """An offset learned in one pose is wrong in the next.

    Without a reset the anchor holds an offset while a new window fills.
    After one, which `fuse_all` does at every take boundary and a live
    caller must do whenever the hand is lost, there is no offset at all
    until the new take has taught its own.
    """
    anchor = DriftAnchor(DriftAnchorParams(deadband=0.0))
    t = _teach(anchor, 0.0, 120, 0.20)
    assert anchor.offset("right", "index", t) == pytest.approx(0.20)
    t = _teach(anchor, t + 30.0, 5, 0.0)
    assert anchor.offset("right", "index", t) == pytest.approx(0.20)

    anchor.reset()
    assert anchor.offset("right", "index", t) is None
    t = _teach(anchor, t + 1.0, anchor.params.min_frames - 1, 0.0)
    assert anchor.offset("right", "index", t) is None
    t = _teach(anchor, t + 1 / HZ, 1, 0.0)
    assert anchor.offset("right", "index", t) == pytest.approx(0.0)
    # the report's counters survive: they describe the whole run
    assert anchor.learned[("right", "index")] == 120 + 5 + anchor.params.min_frames


def test_the_anchor_changes_what_is_fused_and_not_what_the_gates_know():
    """A large offset folds an open glove hand; the gates must not notice.

    Without `gate_curls` the spread gate would read the folded fingers as
    curled and keep the glove's spread on exactly the frames the anchor had
    just corrected, and the thumb vote would read the correction as a glove
    disagreeing with the camera. With it, every camera-owned DOF comes from
    the same sensor as it did without the anchor, and the fused fingers
    still carry the correction.
    """
    gates = GateParams()
    anchor = DriftAnchor()
    t = _teach(anchor, 0.0, 60, 0.80)               # the camera: far more flexed
    G = curled_hand(curl=0.0, spread_deg=5.0)       # an open hand, glove
    C = curled_hand(curl=0.0, spread_deg=15.0)      # the same, camera
    curls = flexion_features(G)
    scale = _toy_scale()
    corrected, moved = anchor.apply("right", t, G, curls, scale)
    assert set(moved) == set(RAIL_FINGERS)
    assert flexion_features(corrected)[1] < gates.curl_gate, "folded below"

    kw = dict(with_scale=False, cam_meta=facing_meta(), scale=scale,
              gates=gates)
    plain, info_plain = fuse_skeletons(G, C, **kw)
    fused, info = fuse_skeletons(corrected, C, gate_curls=curls, **kw)
    assert info["dof_source"] == info_plain["dof_source"]
    assert all(info["dof_source"][f"spread {f}"] == "camera"
               for f in RAIL_FINGERS)
    assert info["dof_source"]["thumb"] == "camera"
    assert info["flex_disagreement"] == info_plain["flex_disagreement"]
    # ...while the fused hand is the corrected one
    assert flexion_features(fused)[1] < flexion_features(plain)[1] - 0.3
    assert bone_lengths(fused) == pytest.approx(bone_lengths(G), abs=1e-12)

    # the counterfactual: gates reading the bent fingers refuse the camera
    _f, info_bent = fuse_skeletons(corrected, C, **kw)
    assert info_bent["dof_source"]["spread index"] == "glove"
    assert info_bent["dof_source"]["thumb"] == "glove"


# --- wired into the session fusion ---------------------------------------

def _leap_row(hand, t, pts):
    meta = facing_meta()
    return {"hand_side": hand, "capture_time": t, "wall_time": t,
            "score": 1.0, "pts": pts, "palm_abs": meta["palm_abs"],
            "palm_normal_abs": meta["palm_normal_abs"],
            "visible_time_us": meta["visible_time_us"],
            "hand_id_stable": True}


def _take(name, pose, t0, seconds, curl, creep=0.0, hz=30.0):
    """One loaded take of a right hand held at `curled_hand(curl)`, both
    sensors at `hz`; the glove creeps `creep` curl open over the take."""
    C = curled_hand(curl=curl)
    truth = flexion_features(C)
    glove, cam = [], []
    for k in range(int(seconds * hz)):
        t = t0 + k / hz
        G = (_creep(C, [c + creep * k / hz / seconds for c in truth])
             if creep else C)
        glove.append({"hand_side": "right", "pose": pose, "take": 1,
                      "capture_time": t, "wall_time": t, "pts": G})
        cam.append(_leap_row("right", t + 0.001, C))
    return {"name": name, "glove": glove, "cam": cam, "source": "leap",
            "clock": "capture_time", "with_scale": False}


def _session():
    """Three takes: an open palm and a fist to learn the endpoints from, and
    a held half fist whose glove creeps open within the take."""
    return [
        _take("half_right_take1.jsonl", "half", 100.0, 20.0, 0.5, creep=0.2),
        _take("open_right_take1.jsonl", "open", 0.0, 3.0, 0.0),
        _take("fist_right_take1.jsonl", "fist", 50.0, 3.0, 1.0),
    ]


def _fuse(fuse, loaded, anchor):
    return fuse.fuse_all(loaded, fuse.DEFAULT_GATES, fuse.DEFAULT_RAIL, {},
                         max_dt=0.05, min_score=0.5, export_csv=True,
                         anchor=anchor)


def test_with_nothing_to_correct_the_anchor_changes_no_fused_sample():
    """`--drift-anchor off` is the fusion as it was, and an anchor that has
    nothing to apply (here: a deadband no offset can clear) reproduces it
    sample for sample."""
    fuse = _fuse_module()
    loaded = _session()
    off = _fuse(fuse, loaded, None)
    assert off.anchor is None
    idle = _fuse(fuse, loaded, DriftAnchorParams(deadband=10.0))
    assert idle.anchor is not None and sum(idle.anchor.learned.values()) > 0
    for a, b in ((off.glove_samples, idle.glove_samples),
                 (off.cam_samples, idle.cam_samples),
                 (off.fused_samples, idle.fused_samples)):
        assert [s[:3] for s in a] == [s[:3] for s in b]
        assert [s[3] for s in a] == [s[3] for s in b]
    assert off.fused_rows == idle.fused_rows
    assert dict(off.dof_used) == dict(idle.dof_used)


def test_the_anchor_moves_the_fused_hand_and_never_the_glove_column():
    fuse = _fuse_module()
    loaded = _session()
    off = _fuse(fuse, loaded, None)
    # The session creeps 0.2 in 20 s, three times the fastest creep measured,
    # to keep it small; a 4 s window and no deadband keep the anchor's own
    # lag well inside the margin asserted below.
    on = _fuse(fuse, loaded, DriftAnchorParams(window_s=4.0, deadband=0.0))
    # the raw glove is what a glove-only pipeline gives, anchor or not
    assert [s[3] for s in off.glove_samples] == [s[3] for s in on.glove_samples]
    truth = flexion_features(curled_hand(curl=0.5))[1]

    def index_of(run):
        slot = next(v for k, v in run.per_pose.items() if k[0] == "half")
        return float(np.median([v[1] for v in slot["fused"]]))

    # the held half fist crept 0.2 open; the anchor pulls the fused index
    # back toward what the camera sees
    assert index_of(off) - truth > 0.08
    assert abs(index_of(on) - truth) < 0.5 * abs(index_of(off) - truth)
    rows = {(r["hand"], r["finger"]): r for r in on.anchor.summary()}
    assert rows[("right", "index")]["corrected"] > 0
    assert rows[("right", "index")]["median_offset"] > 0.0


def test_the_anchor_keeps_constant_memory_over_a_long_run():
    """A multi-hour live run must not grow the anchor's report state.

    Counts, sums and maxima are running values; only the most recent
    ANCHOR_MEDIAN_KEEP offsets are kept for the median.
    """
    from cam_hand import fusion

    G = curled_hand(curl=0.5)
    curls = flexion_features(G)
    keep = 50
    old, fusion.ANCHOR_MEDIAN_KEEP = fusion.ANCHOR_MEDIAN_KEEP, keep
    try:
        anchor = DriftAnchor()
        t = _teach(anchor, 0.0, 60, 0.20)
        for k in range(3 * keep):
            anchor.apply("right", t, G, curls, _toy_scale())
    finally:
        fusion.ANCHOR_MEDIAN_KEEP = old
    key = ("right", "index")
    assert anchor.in_force[key] == 3 * keep
    assert len(anchor._recent[key]) == keep
    row = {(r["hand"], r["finger"]): r for r in anchor.summary()}[key]
    assert row["in_force"] == 3 * keep
    assert row["median_offset"] == pytest.approx(0.20)
    assert row["mean_offset"] == pytest.approx(0.20)
    assert row["sd_offset"] == pytest.approx(0.0, abs=1e-6)


def test_reset_can_forget_one_hand_and_keep_the_other():
    """A live caller loses one hand at a time."""
    anchor = DriftAnchor(DriftAnchorParams(deadband=0.0))
    t = _teach(anchor, 0.0, 60, 0.20, hand="right")
    t = _teach(anchor, 0.0, 60, 0.10, hand="left")
    assert anchor.offset("left", "index", t) == pytest.approx(0.10)

    anchor.reset("right")
    assert anchor.offset("right", "index", t) is None
    assert anchor.offset("left", "index", t) == pytest.approx(0.10)
    # counters survive a reset of either kind
    assert anchor.learned[("right", "index")] == 60
    anchor.reset()
    assert anchor.offset("left", "index", t) is None
    assert anchor.learned[("left", "index")] == 60
