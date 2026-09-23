"""Camera-referenced glove recalibration.

The glove's curl error on the recorded sessions is mostly a function of the
POSE, and a pose-dependent error is what a calibration fixes: each finger's
camera flexion fraction as a ridge regression on the glove's fractions. These
tests pin down that `cross` untangles a channel that also answers to its
neighbour where `own` cannot, that it is evaluated so no take is corrected
by a model that saw it, that it refuses to fit on too little, and that with
it off (or refusing) the fusion is exactly what it was.
"""
import numpy as np
import pytest

from cam_hand.features import flexion_features
from cam_hand.fusion import (
    FINGER_NAMES,
    RAIL_FINGERS,
    SENSOR_CAMERA,
    SENSOR_GLOVE,
    GloveRecalibration,
    RecalibrationParams,
)

from test_drift_anchor import _creep, _leap_row, _scale, _toy_scale
from test_fusion import (
    _fuse_module,
    bone_lengths,
    curled_hand,
    facing_meta,
)

MIDDLE = FINGER_NAMES.index("middle")
INDEX = FINGER_NAMES.index("index")


def _curls(scale, sensor, fracs):
    """Curls on one sensor's endpoints for five flexion fractions."""
    out = []
    for finger, f in zip(FINGER_NAMES, fracs):
        ends = scale.endpoints(sensor, finger)
        out.append(ends.open - f * ends.span)
    return out


def _mixed(true):
    """What this glove reports: the middle channel reads half its own
    finger and half the index's."""
    g = list(true)
    g[MIDDLE] = 0.5 * true[MIDDLE] + 0.5 * true[INDEX]
    return g


def _observe_session(recal, groups=5, per_group=200, seed=0, hand="right"):
    rng = np.random.default_rng(seed)
    scale = _scale(hand)
    for group in range(groups):
        for _ in range(per_group):
            true = rng.uniform(0.0, 1.0, size=5)
            recal.observe(group, hand,
                          _curls(scale, SENSOR_GLOVE, _mixed(true)),
                          _curls(scale, SENSOR_CAMERA, true),
                          facing_meta(), scale)
    return scale


def _pooled(recal, finger, hand="right"):
    before, after = [], []
    for (h, _pose, f), (b, a) in recal.residuals.items():
        if h == hand and f == finger:
            before += b
            after += a
    return float(np.median(before)), float(np.median(after))


def test_cross_untangles_a_neighbour_that_own_cannot():
    """The middle channel reads half the index: only `cross` can undo it."""
    results = {}
    for variant in ("own", "cross"):
        recal = GloveRecalibration(RecalibrationParams(variant=variant))
        _observe_session(recal)
        recal.evaluate({k: "p" for k in range(5)})
        results[variant] = _pooled(recal, "middle")
        # every take was scored by the model that never saw it
        for k in range(5):
            assert recal.model("right", exclude=k).n_frames == 800
        assert recal.model("right").n_frames == 1000
        # and an honest channel is fine either way
        assert _pooled(recal, "index")[1] < 0.02

    before, after_cross = results["cross"]
    _before, after_own = results["own"]
    assert before > 0.1
    assert after_cross < 0.05, results
    assert after_own > 0.1, results
    # the model it learned is the inverse of the mixing: 2 middle - 1 index
    recal = GloveRecalibration(RecalibrationParams(variant="cross"))
    _observe_session(recal)
    bias, weights = recal.model("right").coef["middle"]
    # (a little under, by design: the ridge shrinks toward zero)
    assert weights["middle"] == pytest.approx(2.0, abs=0.1)
    assert weights["index"] == pytest.approx(-1.0, abs=0.1)
    assert abs(bias) < 0.05
    own = GloveRecalibration(RecalibrationParams(variant="own"))
    _observe_session(own)
    assert set(own.model("right").coef["middle"][1]) == {"middle"}


def test_too_few_trusted_frames_refuse_with_a_reason():
    recal = GloveRecalibration(RecalibrationParams(min_frames=50))
    scale = _observe_session(recal, groups=1, per_group=49)
    model = recal.model("right")
    assert model.refused and "49" in model.refused and "50" in model.refused
    G = curled_hand(curl=0.3)
    pts, moved = recal.apply("right", None, G, flexion_features(G), scale)
    assert moved == {} and np.array_equal(pts, G)
    # one more frame and it fits
    _observe_session(recal, groups=1, per_group=1, seed=1)
    assert not recal.model("right").refused


def test_only_trusted_frames_are_learned_and_on_rail_ones_are():
    recal = GloveRecalibration()
    scale = _scale()
    open_g = _curls(scale, SENSOR_GLOVE, [0.0] * 5)      # on every rail
    flexed_c = _curls(scale, SENSOR_CAMERA, [0.6] * 5)   # camera sees flexion
    # a saturated glove reading IS an input the model has to see
    assert recal.observe(0, "right", open_g, flexed_c, facing_meta(), scale)
    for meta in (facing_meta(visible_time_us=1_000),
                 facing_meta(hand_id_stable=False),
                 facing_meta(view_deg=70.0),
                 None):
        assert not recal.observe(0, "right", open_g, flexed_c, meta, scale)
    assert not recal.observe(0, "right", open_g, None, facing_meta(), scale)
    assert recal.n_frames("right") == 1


def test_a_corrected_finger_keeps_its_bones_and_an_owned_one_is_untouched():
    recal = GloveRecalibration(RecalibrationParams(variant="own"))
    toy = _toy_scale()
    rng = np.random.default_rng(3)
    for k in range(120):
        true = rng.uniform(0.0, 1.0, size=5)
        glove = [min(1.0, f + 0.3) for f in true]           # reads too flexed
        recal.observe(k % 3, "right", _curls(toy, SENSOR_GLOVE, glove),
                      _curls(toy, SENSOR_CAMERA, true), facing_meta(), toy)
    G = curled_hand(curl=0.6)
    curls = flexion_features(G)
    pts, moved = recal.apply("right", None, G, curls, toy)
    assert set(moved) == set(RAIL_FINGERS)
    assert all(v > 0.05 for v in moved.values()), "opened, not folded"
    assert bone_lengths(pts) == pytest.approx(bone_lengths(G), abs=1e-12)
    # a finger the rail override owns this frame is never touched
    pts, moved = recal.apply("right", None, G, curls, toy, skip=("index",))
    assert "index" not in moved
    chain = [5, 6, 7, 8]
    assert np.array_equal(pts[chain], np.asarray(G)[chain])


def test_a_railed_index_neither_teaches_its_model_nor_is_corrected():
    """The rail rule, on sync_day1's pinch in miniature.

    The glove's index sits on its rail in some frames while the camera sees
    it flexed. A rail reading means anything from straight to flexed, so
    those frames must not teach the index model a bias (without the rule
    they did, and the bias then bent every open palm's index), and an
    on-rail index frame is returned uncorrected: that dispute is the rail
    override's.
    """
    toy = _toy_scale()
    rail = toy.endpoints(SENSOR_GLOVE, "index").open

    def session(rails):
        recal = GloveRecalibration(RecalibrationParams(variant="cross"),
                                   rails=rails, rail_tol=0.005)
        rng = np.random.default_rng(7)
        for k in range(600):
            true = rng.uniform(0.05, 1.0, size=5)
            glove = list(true)
            if k % 3 == 0:              # the pinch frames: glove on its rail
                true[INDEX] = 0.6
                glove[INDEX] = 0.0
            recal.observe(k % 4, "right", _curls(toy, SENSOR_GLOVE, glove),
                          _curls(toy, SENSOR_CAMERA, true), facing_meta(), toy)
        return recal

    ruled = session({("right", "index"): rail})
    assert ruled.railed[("right", "index")] == 200
    assert ruled.railed[("right", "middle")] == 0
    unruled = session({})               # no learned rail: always off it
    assert unruled.railed[("right", "index")] == 0

    probe = {f: 0.3 for f in FINGER_NAMES}
    # with the rule the index model is the identity it was shown off the rail
    assert ruled.model("right").predict("index", probe) == pytest.approx(
        0.3, abs=0.02)
    bias, weights = ruled.model("right").coef["index"]
    assert abs(bias) < 0.02 and weights["index"] == pytest.approx(1.0, abs=0.03)
    # without it the railed frames drag it toward "flexed" everywhere
    assert unruled.model("right").predict("index", probe) > 0.35
    # ...while the railed index is still an INPUT to its neighbours' models
    assert "index" in ruled.model("right").coef["middle"][1]

    # an on-rail index is returned uncorrected; the other fingers are not
    G = _hand_at_curls(toy, [0.0, 0.0, 0.4, 0.4, 0.4])
    curls = flexion_features(G)
    assert abs(curls[INDEX] - rail) <= 0.005
    pts, moved = unruled.apply("right", None, G, curls, toy)
    assert "index" in moved, "the unruled model would have bent it"
    pts, moved = ruled.apply("right", None, G, curls, toy)
    assert "index" not in moved
    assert np.array_equal(pts[[5, 6, 7, 8]], np.asarray(G)[[5, 6, 7, 8]])
    # held out, a railed frame's residual is left as it was
    ruled.evaluate({k: "p" for k in range(4)})
    before, after = ruled.residuals[("right", "p", "index")]
    railed = [(b, a) for b, a in zip(before, after) if b > 0.5]
    assert railed and all(a == b for b, a in railed)


def _hand_at_curls(scale, fracs):
    """The toy hand with its four fingers at glove fractions of `scale`."""
    return _creep(curled_hand(curl=0.0), _curls(scale, SENSOR_GLOVE, fracs))


# --- wired into the session fusion ---------------------------------------

TOY_OPEN = flexion_features(curled_hand(curl=0.0))
TOY_FIST = flexion_features(curled_hand(curl=1.0))


def _hand_at(fracs):
    """The toy hand with each of the four fingers at a flexion fraction."""
    curls = [o - f * (o - c) for o, c, f in zip(TOY_OPEN, TOY_FIST, fracs)]
    return _creep(curled_hand(curl=0.0), curls)


def _take(k, pose, index, middle, seconds=2.0, hz=30.0):
    """A held pose: the camera sees it as it is, the glove's middle channel
    reads half the middle and half the index."""
    true = [0.0, index, middle, middle, middle]
    C = _hand_at(true)
    G = _hand_at(_mixed(true))
    t0 = 100.0 * k
    glove, cam = [], []
    for i in range(int(seconds * hz)):
        t = t0 + i / hz
        glove.append({"hand_side": "right", "pose": pose, "take": 1,
                      "capture_time": t, "wall_time": t, "pts": G})
        cam.append(_leap_row("right", t + 0.001, C))
    return {"name": f"{pose}_right_take1.jsonl", "glove": glove, "cam": cam,
            "source": "leap", "clock": "capture_time", "with_scale": False}


def _session():
    shapes = [("open", 0.0, 0.0), ("fist", 1.0, 1.0), ("a", 0.2, 0.8),
              ("b", 0.8, 0.2), ("c", 0.5, 0.5), ("d", 0.9, 0.6),
              ("e", 0.3, 0.3), ("f", 0.6, 0.9)]
    # The open take is the longest so its glove reading is each finger's
    # most common one: that spike is how `learn_rails` finds a rail, and
    # three poses sharing one mixed middle reading would otherwise outvote it.
    return [_take(k, *s, seconds=8.0 if s[0] == "open" else 2.0)
            for k, s in enumerate(shapes)]


def _fuse(fuse, loaded, recalibrate=None):
    return fuse.fuse_all(loaded, fuse.DEFAULT_GATES, fuse.DEFAULT_RAIL, {},
                         max_dt=0.05, min_score=0.5, export_csv=True,
                         recalibrate=recalibrate)


def test_recalibration_off_or_refusing_is_the_fusion_as_it_was():
    fuse = _fuse_module()
    loaded = _session()
    off = _fuse(fuse, loaded)
    assert off.recal is None
    idle = _fuse(fuse, loaded, RecalibrationParams(min_frames=10 ** 9))
    assert idle.recal is not None and idle.recal.n_frames("right") > 0
    for a, b in ((off.glove_samples, idle.glove_samples),
                 (off.cam_samples, idle.cam_samples),
                 (off.fused_samples, idle.fused_samples)):
        assert [s[:3] for s in a] == [s[:3] for s in b]
        assert [s[3] for s in a] == [s[3] for s in b]
    assert off.fused_rows == idle.fused_rows
    assert dict(off.dof_used) == dict(idle.dof_used)


def test_each_take_is_corrected_by_a_model_that_never_saw_it():
    fuse = _fuse_module()
    loaded = _session()
    off = _fuse(fuse, loaded)
    on = _fuse(fuse, loaded, RecalibrationParams(variant="cross"))
    recal = on.recal
    total = recal.n_frames("right")
    for k, entry in enumerate(loaded):
        assert (recal.model("right", exclude=k).n_frames
                == total - len(entry["glove"]))
    # the glove column is the glove's, and the gates decided on it
    assert [s[3] for s in off.glove_samples] == [s[3] for s in on.glove_samples]
    assert dict(off.dof_used) == dict(on.dof_used)

    def middle(run, pose):
        slot = next(v for key, v in run.per_pose.items() if key[0] == pose)
        return (float(np.median([v[MIDDLE] for v in slot["fused"]])),
                float(np.median([v[MIDDLE] for v in slot["camera"]])))

    # the mixed poses: the held-out model moves the fused middle finger onto
    # what the camera sees
    for pose in ("a", "b", "d", "f"):
        fused_off, cam = middle(off, pose)
        fused_on, _cam = middle(on, pose)
        assert abs(fused_on - cam) < 0.05, (pose, fused_on, cam)
        assert abs(fused_on - cam) < 0.5 * abs(fused_off - cam), pose
    assert sum(recal.corrected.values()) > 0


def test_the_anchor_and_the_recalibration_are_one_at_a_time():
    fuse = _fuse_module()
    with pytest.raises(ValueError):
        fuse.fuse_all(_session()[:2], fuse.DEFAULT_GATES, fuse.DEFAULT_RAIL,
                      {}, max_dt=0.05, min_score=0.5,
                      anchor=fuse.DriftAnchorParams(),
                      recalibrate=RecalibrationParams())


# --- a whole-session hold-out ---------------------------------------------

def _longer_fingers(pts, s):
    """The four fingers' bones scaled by `s` about their knuckles: the same
    hand on a different glove ruler (a different template, or a refit)."""
    from cam_hand.fusion import FINGER_CHAINS

    P = np.array(pts, float)
    for finger in RAIL_FINGERS:
        chain = FINGER_CHAINS[finger]
        knuckle = P[chain[0]].copy()
        P[chain] = knuckle + s * (P[chain] - knuckle)
    return P


def _other_session(shapes, t0, finger_scale=1.0):
    out = []
    for k, shape in enumerate(shapes):
        entry = _take(k, *shape, seconds=8.0 if shape[0] == "open" else 2.0)
        for rows in (entry["glove"], entry["cam"]):
            for row in rows:
                row["capture_time"] += t0
                row["wall_time"] += t0
        for row in entry["glove"]:
            row["pts"] = _longer_fingers(row["pts"], finger_scale)
        out.append(entry)
    return out


def test_a_model_fitted_on_one_session_corrects_another_on_its_own_ruler():
    """`--recalibrate-from`: fit on session A, apply to every take of B.

    B's poses are not A's, and B's glove fingers are 10 % longer, so its
    rails and endpoints differ from A's: the coefficients carry across and
    the endpoints are re-learned, as a live warm-up would deploy it. There
    is no leave-one-take-out, since nothing in B was in the fit.
    """
    fuse = _fuse_module()
    params = RecalibrationParams(variant="cross")
    A = _other_session([("open", 0.0, 0.0), ("fist", 1.0, 1.0),
                        ("a", 0.2, 0.8), ("b", 0.8, 0.2), ("c", 0.5, 0.5),
                        ("d", 0.9, 0.6)], 0.0)
    B = _other_session([("open", 0.0, 0.0), ("fist", 1.0, 1.0),
                        ("e", 0.3, 0.3), ("f", 0.6, 0.9), ("g", 0.7, 0.4),
                        ("h", 0.1, 0.6)], 5000.0, finger_scale=1.1)
    from_a = fuse.recalibration_from_session(
        A, fuse.DEFAULT_GATES, fuse.DEFAULT_RAIL, params, max_dt=0.05,
        source="session A")
    run = fuse.fuse_all(B, fuse.DEFAULT_GATES, fuse.DEFAULT_RAIL, {},
                        max_dt=0.05, min_score=0.5, recalibrate=params,
                        recal_from=from_a)
    recal = run.recal
    assert recal.cross_session and recal.source == "session A"
    # every take of B is corrected by the one model fitted on all of A
    for k in range(len(B)):
        assert recal.model("right", exclude=k) is from_a.model("right")
    assert recal.model("right").n_frames == from_a.n_frames("right")
    # ...on B's own rails and endpoints, which are not A's
    rails_b, scale_b, _c = fuse.session_scale(B, fuse.DEFAULT_GATES,
                                              fuse.DEFAULT_RAIL, False)
    rails_a, scale_a, _c = fuse.session_scale(A, fuse.DEFAULT_GATES,
                                              fuse.DEFAULT_RAIL, False)
    assert recal.rails == rails_b and recal.rails != rails_a
    open_a = scale_a.for_hand("right").endpoints(SENSOR_GLOVE, "middle").open
    open_b = scale_b.for_hand("right").endpoints(SENSOR_GLOVE, "middle").open
    assert open_b - open_a > 0.05
    # the mixed middle channel is untangled on B's poses
    for pose in ("f", "g", "h"):
        before, after = recal.residuals[("right", pose, "middle")]
        assert np.median(before) > 0.1 and np.median(after) < 0.05, pose

    with pytest.raises(ValueError):
        fuse.fuse_all(B, fuse.DEFAULT_GATES, fuse.DEFAULT_RAIL, {},
                      max_dt=0.05, min_score=0.5, recal_from=from_a)
    with pytest.raises(ValueError):
        GloveRecalibration(RecalibrationParams(variant="own"),
                           fitted_on=from_a)
