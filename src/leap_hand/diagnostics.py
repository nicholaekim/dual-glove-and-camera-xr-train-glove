"""The arithmetic behind the two glove diagnostics, with no I/O in it.

`scripts/leap/hold_test.py` and `scripts/leap/finger_sweep.py` ask the glove
two questions the fusion report cannot answer, because both need the operator
to do something specific in front of the camera:

  hold_test      does the glove's curl DRIFT while a pose is held still? The
                 camera is the reference that does not creep.
  finger_sweep   what does the glove report as one finger bends slowly all
                 the way down and back? Where does it stop responding, how
                 far behind the camera is it, and does it read the same
                 bending as straightening?

Both scripts are a live loop around the functions here. Everything in this
module is values in, values out — no sensors, no clock of its own, no files —
so the same code that judges a session live re-judges a CSV written months
ago, and the unit tests can hand it a synthetic sweep with a KNOWN lag and a
KNOWN saturation point and check that both come back.

THE MEASUREMENT IS THE SAME ONE THE REPORTS PRINT
  Curl is `cam_hand.features.flexion_features`: fingertip-to-wrist over palm
  length, higher = straighter. Not re-derived here. A number in a drift
  summary and a row in `scripts/fuse_poses.py`'s per-DOF table have to be the
  same quantity or they cannot be argued about together, and the scratch
  versions of these tools computed their own from the 26-joint layout, which
  is how two numbers that looked comparable came to be only nearly so.

WHY A BINNED TRANSFER CURVE, AND WHY BINNED BY THE **CAMERA**
  The obvious summary of a sweep is "the glove sat on its rail until the
  finger was N % bent", which is what the scratch `measure_dead_zone.py`
  printed. That percentage is measured against the finger's OPEN reading, so
  a sweep that began with the finger already half bent — which is exactly
  what `dead_zone_right_ring_20260920_200435.csv` is — has no valid zero and
  the percentage is meaningless.

  Binning the glove's curl by the CAMERA's curl needs no such reference. The
  camera measures the finger's actual shape frame by frame; the question
  "what does the glove say when the camera says 1.3?" is answerable from any
  sweep that visited 1.3, however it started. The rail-saturation point then
  falls out of the same table: the camera curl above which the glove's answer
  stops changing.

...WHICH ONLY WORKS IF THE CAMERA FOLLOWED THE FINGER
  Binning by the camera makes the camera the measuring axis, so a run in
  which the camera's own curl barely moved has no axis to bin against.
  `dead_zone_right_ring_20260920_212254.csv` is exactly that: an isolated
  right-ring bend that the glove read from 0.87 to 1.97 while the camera's
  ring curl stayed inside half a unit, because a gloved ring folding on its
  own is a shape the tracker does not resolve. The curve built on it came out
  non-monotonic and a saturation point was quoted off it anyway.

  So `camera_range` is checked BEFORE anything is binned, per finger, and a
  finger it fails gets no transfer curve and no saturation point — see
  `FingerSweep`, which simply does not carry them. The fix for such a run is
  not a slower sweep; it is the WHOLE-HAND sweep (`--finger all`), where
  every finger moves through its full range and the camera resolves all of
  them, with `analyse_sweep` reading the four (or five) transfer curves out
  of the one recording.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cam_hand.features import FLEXION_NAMES

from .pose_check import (
    CURLED,
    DEFAULT_PARAMS,
    EXPECTED_FINGERS,
    EXTENDED,
    PoseCheckParams,
    finger_state,
)
from .protocol import stream_health

FINGERS: Tuple[str, ...] = tuple(FLEXION_NAMES)

# Default shape of a hold: drop this many seconds of transient, then compare
# the first and last `DRIFT_WINDOW` seconds of what is left.
DRIFT_SKIP = 3.0
DRIFT_WINDOW = 5.0
# How close a glove curl has to be to its open-palm reading to count as
# "the glove has not moved off its open value yet". The glove's open-palm
# output is bit-exact to three decimals (see cam_hand.fusion on rails), so
# this is a float-equality test with room for the last digit.
OPEN_TOL = 0.01
# ...and how long it has to STAY off that value to count as having left it.
# The same idea as the rail override's `enter_frames`: a glove climbing back
# onto its rail passes within a hair of the tolerance for a few frames on the
# way, and on hold_drift_right_fist_20260918_190912.csv that 70 ms of
# transient at t = 3.02 is enough to place the whole baseline window back on
# the open hand. A departure that lasts half a second is the hand.
OPEN_SETTLE = 0.5

# Sweep analysis.
BIN_WIDTH = 0.1
MIN_BIN_SAMPLES = 5
RAIL_TOL = 0.05
# How much of its own range the CAMERA has to have covered on a finger before
# anything is reported about that finger. A full bend spans about 0.8 to 1.7
# of camera curl, so 0.5 is a comfortably bent finger and still well clear of
# a finger that only wobbled. See `camera_range` for why this is a separate
# check from coverage.
MIN_CAMERA_RANGE = 0.50
# Measured between these percentiles rather than min to max: one mistracked
# frame at either end can invent a third of a unit of range out of nothing,
# and the guard exists precisely to catch the runs where the real range was
# small.
RANGE_PERCENTILES = (5.0, 95.0)
# Cross-correlation search window, seconds: the glove may lead the camera a
# little (negative) and is expected to trail it (positive).
LAG_MIN = -0.2
LAG_MAX = 1.0
LAG_STEP = 0.005


# --- holding a pose still: does the glove creep? -----------------------------

@dataclass(frozen=True)
class DriftRow:
    """One finger's verdict over one hold."""

    finger: str
    first: Optional[float]          # median curl over the baseline window
    last: Optional[float]           # median curl over the closing window
    drift: Optional[float]          # last - first
    baseline_at: float              # when the baseline window started
    stuck_s: float                  # how long the curl sat on its open value

    @property
    def measured(self) -> bool:
        return self.drift is not None


def _median(values) -> Optional[float]:
    arr = np.asarray(list(values), dtype=float)
    return None if arr.size == 0 else float(np.median(arr))


def open_reference(curls: Sequence[Sequence[float]],
                   tol: float = OPEN_TOL) -> List[float]:
    """Each finger's straightest reading, as the glove reports it.

    The top of the sensor's range — the rail, in `cam_hand.fusion`'s sense.
    Taken as the median of every sample within `tol` of the maximum rather
    than the maximum itself, so one noisy frame cannot set it.

    A live run does not need this: `hold_test` records the operator's actual
    open palm before the pose is called and passes THAT. It exists for
    re-reading an old CSV, which has no recorded baseline in it.
    """
    arr = np.asarray(curls, dtype=float)
    out = []
    for i in range(arr.shape[1]):
        top = float(arr[:, i].max())
        near = arr[np.abs(arr[:, i] - top) <= tol, i]
        out.append(float(np.median(near)))
    return out


def _left_open_at(times: np.ndarray, off_open: np.ndarray, skip: float,
                  settle: float) -> Optional[float]:
    """When the curl left its open value FOR GOOD, at or after `skip`.

    The first instant from which `off_open` stays true for `settle` seconds
    without a break, reported as the start of that run rather than the end of
    it — the hand left the open shape when the run began, and the run length
    is only the evidence that it stayed left. None if no departure ever
    lasted that long.
    """
    run_start: Optional[float] = None
    for k in range(times.size):
        if times[k] < skip:
            continue
        if not off_open[k]:
            run_start = None
            continue
        if run_start is None:
            run_start = float(times[k])
        elif float(times[k]) - run_start >= settle:
            return run_start
    return None


def hold_drift(times: Sequence[float], curls: Sequence[Sequence[float]],
               skip: float = DRIFT_SKIP, window: float = DRIFT_WINDOW,
               open_curls: Optional[Sequence[float]] = None,
               open_tol: float = OPEN_TOL, open_settle: float = OPEN_SETTLE,
               names: Sequence[str] = FINGERS) -> List[DriftRow]:
    """Per finger: the curl at the start of the hold, at the end, and the gap.

    `times` are seconds from the start of the hold and `curls` one row per
    frame. The closing window is always the last `window` seconds. The
    BASELINE window is the part with a decision in it:

      * it starts at `skip`, which throws away the transient while the hand
        is still changing shape;
      * and then, if `open_curls` is given, it is pushed further to the first
        sample at which that finger has actually LEFT its open-palm reading.

    That second rule is not a nicety. On
    `hold_drift_right_fist_20260918_190912.csv` the camera sees the fist from
    the first frame while the glove sits pinned at its open-palm value for
    5.6 s — so a baseline taken at 3 s is a median of the OPEN hand, and the
    drift comes out as -0.90 on the index: the glove closing, reported as the
    glove creeping open. Measured from where the glove starts answering, the
    same file reads +0.29 on the index and +0.17 on the ring, which is the
    creep the test is for. `stuck_s` reports how long that took, because a
    glove that takes 5 s to notice a fist is itself a finding.

    A finger that never leaves its open value keeps the plain `skip`
    baseline: an open-palm hold is a legitimate thing to measure, and
    "it never moved" must come out as no drift rather than as no data.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(curls, dtype=float)
    if t.size == 0 or v.size == 0:
        return [DriftRow(n, None, None, None, skip, 0.0) for n in names]
    order = np.argsort(t)
    t, v = t[order], v[order, :]
    t_end = float(t[-1])

    rows: List[DriftRow] = []
    for i, name in enumerate(names[:v.shape[1]]):
        start = float(skip)
        stuck = 0.0
        if open_curls is not None:
            off_open = np.abs(v[:, i] - float(open_curls[i])) > open_tol
            left = _left_open_at(t, off_open, float(skip), float(open_settle))
            if left is not None:
                stuck = max(0.0, left - float(skip))
                start = left
        base = (t >= start) & (t < start + window)
        tail = t > t_end - window
        first = _median(v[base, i])
        last = _median(v[tail, i])
        drift = None if (first is None or last is None) else last - first
        rows.append(DriftRow(name, first, last, drift, start, stuck))
    return rows


def drift_lines(glove: Sequence[DriftRow], camera: Sequence[DriftRow],
                flag_drift: float = 0.15, flag_steady: float = 0.08
                ) -> List[str]:
    """The hold summary, in the columns the scratch tool printed them in.

    A finger is called out when the GLOVE moved and the CAMERA did not: that
    combination is the only one that can only be the glove, since the camera
    is the sensor with nothing to creep.
    """
    out = [f"  {'finger':7s} {'glove first5s':>14s} {'glove last5s':>13s} "
           f"{'drift':>7s} | {'cam first5s':>12s} {'cam last5s':>11s} "
           f"{'drift':>7s}"]
    by_name = {row.finger: row for row in camera}
    for g in glove:
        c = by_name.get(g.finger)
        if not g.measured or c is None or not c.measured:
            out.append(f"  {g.finger:7s} {'(not measured)':>14s}")
            continue
        flag = ("  <-- glove drifts, camera does not"
                if abs(g.drift) > flag_drift and abs(c.drift) < flag_steady
                else "")
        out.append(f"  {g.finger:7s} {g.first:14.3f} {g.last:13.3f} "
                   f"{g.drift:+7.3f} | {c.first:12.3f} {c.last:11.3f} "
                   f"{c.drift:+7.3f}{flag}")
    return out


# --- is the camera showing the pose that was asked for? ----------------------

MAX_UNCLEAR = 1


def camera_pose_match(pose: str, curls: Sequence[float],
                      gap: Optional[float] = None,
                      params: PoseCheckParams = DEFAULT_PARAMS,
                      max_unclear: int = MAX_UNCLEAR) -> Tuple[bool, str]:
    """(is the camera seeing `pose` right now, why not).

    The finger states and the bands are `leap_hand.pose_check`'s, unchanged,
    so "the camera says this is a fist" means the same thing live as it does
    when `scripts/check_take_labels.py` audits the folder afterwards. Only
    the GLOVE is left out, deliberately: it is the instrument under test, and
    the failure being guarded against — the glove sitting on its open-palm
    value while the hand is closed — is precisely the case where it would
    vote yes.

    The rule is "nothing contradicts the pose, and nearly everything
    confirms it":

      * a finger the camera reads as the OPPOSITE state fails outright. This
        is what catches the day-2 right-fist hold, 60 s of open palm under
        the label `fist`: the camera reads index 1.75, extended, against an
        expected curled.
      * up to `max_unclear` fingers may sit in the AMBIGUOUS band. One is
        allowed because the camera's THUMB band is 0.045 wide by measurement
        (see `PoseCheckParams`) and a genuine open palm reads 1.25-1.33 on
        it, i.e. either side of the boundary — refusing to start on that
        would leave the operator holding an open hand the window keeps
        rejecting. Two unclear fingers is a hand that is not yet in the
        pose, and waiting is the right answer.

    `pinch` has no expected finger shape at all — a fist and a pinch have
    almost the same curls — so it is judged on the thumb-index tip gap, the
    one number that separates them.
    """
    want = EXPECTED_FINGERS.get(str(pose))
    if want is None:
        return False, f"{pose} has no expected hand shape to look for"
    if curls is None or len(curls) < 5:
        return False, "the camera is not seeing that hand"
    if all(w is None for w in want):
        if gap is None:
            return False, "no thumb-index gap from the camera yet"
        if gap > params.pinch_gap_max:
            return (False, f"thumb and index {gap:.2f} palm lengths apart, "
                           f"need under {params.pinch_gap_max:g}")
        return True, ""

    wrong: List[str] = []
    unclear: List[str] = []
    for i, finger in enumerate(FINGERS):
        if want[i] is None:
            continue
        state = finger_state(curls[i], params.camera_mid[i],
                             params.camera_margin[i])
        if state == want[i]:
            continue
        if state in (EXTENDED, CURLED):
            wrong.append(f"{finger} {state}")
        else:
            unclear.append(finger)
    if wrong:
        return False, "camera sees " + ", ".join(wrong)
    if len(unclear) > max_unclear:
        return False, ("camera cannot tell about " + ", ".join(unclear))
    return True, ""


class PoseHold:
    """Has the camera shown the pose CONTINUOUSLY for long enough?

    One frame is not evidence, and neither is a run of frames with a gap in
    it: the day-2 right-fist hold recorded 60 s of open palm because nothing
    ever checked the shape, and a checker that armed on a single lucky frame
    would have let the same take through. `update` returns True once the
    match has held without a break for `seconds`.
    """

    def __init__(self, seconds: float = 1.5):
        self.seconds = float(seconds)
        self.since: Optional[float] = None

    def update(self, now: float, matched: bool) -> bool:
        if not matched:
            self.since = None
            return False
        if self.since is None:
            self.since = float(now)
        return (float(now) - self.since) >= self.seconds

    @property
    def held(self) -> float:
        return 0.0 if self.since is None else self.seconds


# --- one finger swept slowly: the transfer curve -----------------------------

@dataclass(frozen=True)
class CurveBin:
    """One 0.1-wide slice of CAMERA curl, and what the glove said inside it."""

    low: float
    high: float
    n: int
    glove: Optional[float]                  # median glove curl in this bin
    n_bending: int = 0
    glove_bending: Optional[float] = None
    n_straightening: int = 0
    glove_straightening: Optional[float] = None

    @property
    def centre(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def hysteresis(self) -> Optional[float]:
        """Bending minus straightening: the glove's path dependence."""
        if self.glove_bending is None or self.glove_straightening is None:
            return None
        return self.glove_bending - self.glove_straightening


def bending_mask(cam: Sequence[float]) -> np.ndarray:
    """True where the finger is CLOSING, from the camera's own trace.

    Curl is tip-to-wrist over palm length, so bending makes it fall. Read off
    the camera and never off the glove: which way the finger is moving is a
    fact about the hand, and the glove's answer to it is the thing under
    test.
    """
    c = np.asarray(cam, dtype=float)
    if c.size < 2:
        return np.zeros(c.shape, dtype=bool)
    return np.gradient(c) < 0


def transfer_curve(cam: Sequence[float], glove: Sequence[float],
                   bin_width: float = BIN_WIDTH,
                   bending: Optional[Sequence[bool]] = None
                   ) -> List[CurveBin]:
    """The glove's answer as a function of the CAMERA's curl, binned.

    One row per `bin_width` slice of camera curl that has any samples in it,
    in ascending order, with the glove's median overall and separately while
    bending and while straightening. A gap between those last two is
    hysteresis — the same finger shape read two different ways depending on
    where it came from.
    """
    c = np.asarray(cam, dtype=float)
    g = np.asarray(glove, dtype=float)
    if c.size == 0 or c.size != g.size:
        return []
    down = (bending_mask(c) if bending is None
            else np.asarray(bending, dtype=bool))

    lo = np.floor(float(c.min()) / bin_width) * bin_width
    hi = np.ceil(float(c.max()) / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width / 2.0, bin_width)
    out: List[CurveBin] = []
    for a, b in zip(edges, edges[1:]):
        inside = (c >= a) & (c < b)
        if not inside.any():
            continue
        out.append(CurveBin(
            low=float(a), high=float(b), n=int(inside.sum()),
            glove=_median(g[inside]),
            n_bending=int((inside & down).sum()),
            glove_bending=_median(g[inside & down]),
            n_straightening=int((inside & ~down).sum()),
            glove_straightening=_median(g[inside & ~down])))
    return out


def rail_saturation(bins: Sequence[CurveBin], rail: float,
                    tol: float = RAIL_TOL) -> Optional[float]:
    """The camera curl at which the glove stops answering: its rail.

    The lowest bin whose glove median is within `tol` of the rail AND from
    which every higher bin is too. The "and every higher bin too" is what
    keeps one bin that happens to brush the rail — a single stationary moment
    part way through a sweep — from being reported as saturation: a rail is a
    place the glove stays, not one it touches.

    Returned as that bin's CENTRE, because the bin is the resolution of the
    answer and quoting an edge would imply more precision than there is.
    None when the glove never reaches its rail in this sweep.
    """
    usable = [b for b in bins if b.glove is not None]
    found = None
    for b in reversed(usable):
        if abs(b.glove - float(rail)) <= tol:
            found = b
        else:
            break
    return None if found is None else found.centre


def coverage_gaps(cam: Sequence[float], low: float, high: float,
                  bin_width: float = BIN_WIDTH,
                  min_samples: int = MIN_BIN_SAMPLES
                  ) -> List[Tuple[float, float, int]]:
    """Which `bin_width` slices between `low` and `high` are under-sampled.

    The sweep's own acceptance test. A transfer curve is only as good as the
    range the finger actually visited, and a sweep done too fast, or one that
    never straightened the finger fully, leaves holes the binned curve
    quietly interpolates over. Returns `(low, high, n)` for every bin with
    fewer than `min_samples` in it — empty means the sweep covered the range.
    """
    c = np.asarray(cam, dtype=float)
    lo, hi = (float(low), float(high)) if low <= high else (float(high),
                                                            float(low))
    lo = np.floor(lo / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width / 2.0, bin_width)
    gaps: List[Tuple[float, float, int]] = []
    for a, b in zip(edges, edges[1:]):
        n = int(((c >= a) & (c < b)).sum())
        if n < min_samples:
            gaps.append((float(a), float(b), n))
    return gaps


# --- did the camera follow the finger at all? --------------------------------

@dataclass(frozen=True)
class CameraRange:
    """How much camera curl one finger covered, and whether that is enough.

    The guard that `coverage_gaps` is not. Coverage asks whether the range
    the finger DID visit was sampled evenly; this asks whether there was a
    range at all. A sweep can pass coverage perfectly while the camera watched
    the finger move a tenth of a unit, and the binned curve built on it is
    then a picture of tracking noise with a saturation point read off it.
    """

    finger: str
    low: float                      # camera curl at the low percentile
    high: float                     # ...and at the high one
    needed: float                   # the threshold it is being held to
    n: int                          # samples the range was measured over

    @property
    def span(self) -> float:
        return self.high - self.low

    @property
    def followed(self) -> bool:
        """Did the camera see this finger move enough to say anything?"""
        return self.n > 0 and self.span >= self.needed


def camera_range(cam: Sequence[float], finger: str = "",
                 min_range: float = MIN_CAMERA_RANGE,
                 percentiles: Tuple[float, float] = RANGE_PERCENTILES
                 ) -> CameraRange:
    """The camera's own range over one finger, as a pass/fail on the run.

    This exists because of `dead_zone_right_ring_20260920_212254.csv`. The
    operator bent the right ring through most of the glove's range — the
    glove swung 0.87 to 1.97 — and the camera's ring curl stayed inside half
    a unit the whole time, because an isolated ring bend on a gloved hand is
    a shape the tracker does not resolve. Every later number in that run was
    computed anyway: a transfer curve binned by a camera axis that barely
    moved, non-monotonic, with a rail-saturation point quoted off it to two
    decimals. None of it was a measurement, and nothing in the tool said so.

    So: measure the camera's range first, and when it is too small, report
    that and STOP. A finger the camera did not follow has no transfer curve,
    no saturation point and no lag worth printing, and the honest output is
    the raw CSV plus a sentence saying why there is nothing else.
    """
    c = np.asarray(cam, dtype=float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return CameraRange(finger=str(finger), low=0.0, high=0.0,
                           needed=float(min_range), n=0)
    low, high = (float(v) for v in np.percentile(c, list(percentiles)))
    return CameraRange(finger=str(finger), low=low, high=high,
                       needed=float(min_range), n=int(c.size))


def camera_range_verdict(r: CameraRange) -> List[str]:
    """The sentence printed instead of a transfer curve, when it comes to it."""
    name = r.finger or "finger"
    if r.n == 0:
        return [f"camera did not follow the {name}: it was never tracked; "
                f"this run is descriptive only, no transfer curve or "
                f"saturation point is reported"]
    return [f"camera did not follow the {name}: its range was {r.span:.2f} "
            f"(camera curl {r.low:.2f} to {r.high:.2f}), need {r.needed:.2f}; "
            f"this run is descriptive only, no transfer curve or saturation "
            f"point is reported"]


@dataclass(frozen=True)
class LagEstimate:
    """How far the glove's trace trails the camera's, and how well they match."""

    seconds: float
    correlation: float

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0


def estimate_lag(glove_t: Sequence[float], glove_v: Sequence[float],
                 cam_t: Sequence[float], cam_v: Sequence[float],
                 step: float = LAG_STEP, lag_min: float = LAG_MIN,
                 lag_max: float = LAG_MAX,
                 min_std: float = 0.01) -> Optional[LagEstimate]:
    """The shift that best aligns the two traces. Positive = the glove trails.

    Both series are resampled onto one uniform grid and z-scored, so the
    match is about SHAPE and neither the glove's different range nor its
    offset can influence it. Returns None when either signal barely moved:
    a cross-correlation of two flat lines has a maximum, and it means
    nothing.
    """
    gt = np.asarray(glove_t, dtype=float)
    gv = np.asarray(glove_v, dtype=float)
    ct = np.asarray(cam_t, dtype=float)
    cv = np.asarray(cam_v, dtype=float)
    if gt.size < 2 or ct.size < 2:
        return None
    start = max(gt[0], ct[0]) + step
    stop = min(gt[-1], ct[-1]) - step
    if stop - start < max(lag_max - lag_min, 4 * step):
        return None
    grid = np.arange(start, stop, step)
    G = np.interp(grid, gt, gv)
    C = np.interp(grid, ct, cv)
    if float(G.std()) < min_std or float(C.std()) < min_std:
        return None

    lags = np.arange(int(round(lag_min / step)), int(round(lag_max / step)) + 1)
    best_k, best = None, -np.inf
    for k in lags:
        # k > 0: the glove sample k steps LATER lines up with this camera
        # sample, i.e. the glove is behind by k steps.
        a = G[max(0, k):G.size + min(0, k)]
        b = C[max(0, -k):C.size + min(0, -k)]
        if a.size < 10:
            continue
        # Pearson over THIS overlap, not a z-score taken once over the whole
        # signal. Every lag drops `k` samples off one end, and on a periodic
        # sweep those samples sit at a turning point where the z-scored
        # product is far from average — so a fixed normalisation makes the
        # score depend on how much was truncated, which is largest at k = 0.
        # Measured on the synthetic sweep in tests/test_diagnostics.py that
        # bias is 0.006, nearly twice the real difference between the right
        # lag and no lag at all, and it puts the peak at zero every time.
        a = a - a.mean()
        b = b - b.mean()
        den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if den <= 0.0:
            continue
        score = float((a * b).sum() / den)
        if score > best:
            best_k, best = int(k), score
    if best_k is None:
        return None
    return LagEstimate(seconds=best_k * step, correlation=best)


# --- a whole sweep, finger by finger -----------------------------------------

@dataclass(frozen=True, eq=False)
class FingerSweep:
    """One finger's share of a sweep: the paired traces, and what they say.

    `camera` is the camera's curl at the instant each glove sample describes,
    already shifted by this finger's own measured lag — every finger of a
    whole-hand sweep gets its own alignment, because the four stretch sensors
    are not obliged to answer at the same speed.

    `bins` and `saturation` are empty and None whenever `span.followed` is
    False. That is the guard, carried in the result rather than left to the
    caller to remember: a finger the camera did not follow has no transfer
    curve to hand out, so there is none in the object.
    """

    finger: str
    times: np.ndarray                    # glove sample times
    glove: np.ndarray                    # the glove's curl at each
    camera: np.ndarray                   # the camera's, at the same instants
    rail: float                          # the glove's open-palm constant
    lag: Optional[LagEstimate]
    span: CameraRange
    bins: List[CurveBin]
    saturation: Optional[float]

    @property
    def measured(self) -> bool:
        """Is there a transfer curve in here, or only a description?"""
        return self.span.followed


def analyse_finger(glove_t: Sequence[float], glove_v: Sequence[float],
                   cam_t: Sequence[float], cam_v: Sequence[float],
                   rail: float, finger: str = "",
                   bin_width: float = BIN_WIDTH,
                   min_range: float = MIN_CAMERA_RANGE,
                   rail_tol: float = RAIL_TOL) -> FingerSweep:
    """Lag, camera range, transfer curve and saturation for ONE finger.

    The order matters and is the whole point of the rewrite: the lag is
    measured, the traces are paired on it, the camera's range is checked, and
    only then — if the camera moved enough to have been watching a finger
    bend — is a curve binned and a saturation point read off it.
    """
    gt = np.asarray(glove_t, dtype=float)
    gv = np.asarray(glove_v, dtype=float)
    ct = np.asarray(cam_t, dtype=float)
    cv = np.asarray(cam_v, dtype=float)
    lag = estimate_lag(gt, gv, ct, cv)
    shift = 0.0 if lag is None else lag.seconds
    if gt.size == 0 or ct.size == 0:
        cam_at_glove = np.zeros(gt.shape, dtype=float)
    else:
        cam_at_glove = np.interp(gt - shift, ct, cv)
    span = camera_range(cam_at_glove, finger=finger, min_range=min_range)
    if not span.followed:
        return FingerSweep(finger=finger, times=gt, glove=gv,
                           camera=cam_at_glove, rail=float(rail), lag=lag,
                           span=span, bins=[], saturation=None)
    bins = transfer_curve(cam_at_glove, gv, bin_width=bin_width)
    return FingerSweep(finger=finger, times=gt, glove=gv, camera=cam_at_glove,
                       rail=float(rail), lag=lag, span=span, bins=bins,
                       saturation=rail_saturation(bins, rail, tol=rail_tol))


def analyse_sweep(glove_t: Sequence[float],
                  glove_curls: Sequence[Sequence[float]],
                  cam_t: Sequence[float],
                  cam_curls: Sequence[Sequence[float]],
                  rails: Sequence[float],
                  fingers: Optional[Sequence[str]] = None,
                  names: Sequence[str] = FINGERS,
                  bin_width: float = BIN_WIDTH,
                  min_range: float = MIN_CAMERA_RANGE,
                  rail_tol: float = RAIL_TOL) -> List[FingerSweep]:
    """`analyse_finger` over every finger of a whole-hand sweep.

    `glove_curls` and `cam_curls` are one row per frame, one column per
    finger of `names`, on their own clocks. `fingers` picks which subset to
    analyse; the default is all of them.

    A single-finger run records all five columns too, so the same call
    re-reads an isolated ring sweep for what the index was doing — which is
    the only way to tell "the camera did not follow the ring" apart from
    "the camera was not tracking the hand".
    """
    gv = np.asarray(glove_curls, dtype=float)
    cv = np.asarray(cam_curls, dtype=float)
    if gv.ndim != 2 or cv.ndim != 2:
        return []
    wanted = list(names) if fingers is None else list(fingers)
    out: List[FingerSweep] = []
    for name in wanted:
        i = list(names).index(name)
        if i >= gv.shape[1] or i >= cv.shape[1]:
            continue
        out.append(analyse_finger(glove_t, gv[:, i], cam_t, cv[:, i],
                                  float(rails[i]), finger=name,
                                  bin_width=bin_width, min_range=min_range,
                                  rail_tol=rail_tol))
    return out


# --- the glove's stream, second by second ------------------------------------

@dataclass(frozen=True)
class StreamSecond:
    """One second of the glove's stream, as the sidecar CSV records it."""

    second: int
    frames: int
    max_gap_ms: float
    counter_jumps: int
    lost_packets: int


class StreamLog:
    """Packet arrivals and packet counters, bucketed into whole seconds.

    Recorded continuously through both diagnostics, because a drift or a lag
    measured over a stream that stopped for two seconds is an artefact of the
    stream. The glove's own packet counter is what separates the two possible
    stories: frames missing with the counter still consecutive means WE
    stopped draining, and frames missing with the counter jumping means the
    packets never arrived.

    `add` is called once per packet with the arrival time (`QueueItem.
    recv_time`, stamped on the OSC server thread) and the counter out of the
    packet header.
    """

    def __init__(self, gap_s: float = 0.1):
        self.gap_s = float(gap_s)
        self.t0: Optional[float] = None
        self.times: List[float] = []
        self._last_counter: Optional[int] = None
        self._buckets: Dict[int, Dict[str, float]] = {}
        self._last_in_bucket: Dict[int, float] = {}
        self.jumps = 0
        self.lost = 0

    def add(self, t: float, counter: Optional[int] = None) -> None:
        t = float(t)
        if self.t0 is None:
            self.t0 = t
        self.times.append(t)
        second = int(t - self.t0)
        row = self._buckets.setdefault(
            second, {"frames": 0, "max_gap_ms": 0.0, "jumps": 0, "lost": 0})
        row["frames"] += 1
        previous = self.times[-2] if len(self.times) > 1 else None
        if previous is not None:
            row["max_gap_ms"] = max(row["max_gap_ms"], (t - previous) * 1000.0)
        if counter is not None:
            if self._last_counter is not None:
                delta = int(counter) - self._last_counter
                if delta != 1:
                    row["jumps"] += 1
                    self.jumps += 1
                    if delta > 1:
                        row["lost"] += delta - 1
                        self.lost += delta - 1
            self._last_counter = int(counter)

    def rows(self) -> List[StreamSecond]:
        return [StreamSecond(second=s, frames=int(r["frames"]),
                             max_gap_ms=round(r["max_gap_ms"], 1),
                             counter_jumps=int(r["jumps"]),
                             lost_packets=int(r["lost"]))
                for s, r in sorted(self._buckets.items())]

    def summary(self, t0: Optional[float] = None,
                t1: Optional[float] = None) -> dict:
        """Rate, worst hole, holes over the threshold, and counter jumps.

        The first four come from `protocol.stream_health`, which every other
        take in this repo is already measured with; only the counter columns
        are new here.
        """
        out = dict(stream_health(self.times, t0, t1, gap_s=self.gap_s))
        out["counter_jumps"] = self.jumps
        out["lost_packets"] = self.lost
        return out


def stream_summary_line(summary: dict) -> str:
    """One line: `glove 59.9 Hz, worst gap 34 ms, 0 gaps over 100 ms, ...`."""
    rate = summary.get("rate_hz")
    gap = summary.get("max_gap_ms")
    return (f"glove stream: {summary.get('frames', 0)} packets, "
            f"{'--' if rate is None else f'{rate:.1f}'} Hz, "
            f"worst gap {'--' if gap is None else f'{gap:.0f}'} ms, "
            f"{summary.get('gaps_over', 0)} gap(s) over "
            f"{summary.get('gaps_over_ms', 100)} ms, "
            f"{summary.get('counter_jumps', 0)} counter jump(s) "
            f"({summary.get('lost_packets', 0)} packet(s) never arrived)")
