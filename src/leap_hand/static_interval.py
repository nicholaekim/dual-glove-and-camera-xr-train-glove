"""Where a grasp take holds still, and the one real frame that summarises it.

Set A of the professor's protocol (docs/grasp_and_flexion_protocol_plan.md,
D4) asks for the FINAL configuration of each grasp, not the reach into it.
A five-second take still contains the reach: the hand drifts into the pose
for the first second or so, settles, and may creep at the end. A summary
taken over the whole take is pulled toward the drift. So each take gets an
explicit static interval, the `static_s` window (2 s by default) with the
least joint motion inside the take, and the one-frame summary is chosen
inside that window only.

Why the mean speed is a ratio of sums. Joint speed per frame would divide a
displacement by the gap between two `wall_time`s, and `wall_time` is a
WRITER clock: LeapC hands queue up on its polling thread and are written in
bursts, so two frames captured 10 ms apart can carry write times a
microsecond apart and a per-frame speed explodes on noise. Summing the joint
path over the window and dividing by the window's elapsed time is the same
quantity (the mean speed) without that division by near zero. `wall_time`
is the clock used, not `capture_time`, because it is on every line of every
recording (glove files have no `capture_time`) and it is the clock the
contract's `static_interval` and `medoid_wall_time` are written in, so a
reader can select the window's rows with no conversion.

Positions are the camera's own absolute joint positions (`abs26`) when the
line has them, forward kinematics of `joints` otherwise. Absolute on
purpose: a hand translating or turning into place IS the motion this is
looking for, and wrist-centred coordinates would hide it.

Why the summary is a medoid. Same rule as `scripts/leap/record_frame.py`
(`medoid_index`), reimplemented here because a script is not importable
from a package: the real frame closest to the window's mean after a RIGID
alignment on the palm landmarks, never the mean itself. Averaged joint
positions shorten every bone, so the "average" hand is a hand nobody has,
and without the alignment "closest to the mean" means "held at the average
angle" rather than "in the typical pose". The alignment decides the winner
and nothing else: the frame handed back is the original line, untouched.

Which hand. The tracker's left/right label is its opinion, not a fact: on
2026-09-23 it called the operator's left hand "right" in 20 of 21 poses. So
the label never decides whose hand a frame is.

The coached recorder follows the hand it acquired, by the tracker's hand id
(`follow_hand`). OPEN HAND acquires one hand over the module; the take is
that id, plus every new id the tracker gives the same hand while it is
absent under the old one and within `FOLLOW_RADIUS_M` of where it was (a
chirality flip or a re-acquisition). An id seen in the same tracking frame
as the followed hand, or further away, is another hand and stays another
hand. On 2026-10-01 both failures were measured: a flip from left to right
mid-take split one hand's frames over two labels and the take was rejected
as 54 % tracked, and the operator's idle right hand, held in view to help
the tracker, had more frames than the grasping hand and was measured
instead. Following the id fixes both. The label then only says whether the
tracker fitted the hand as the operator's chirality: a skeleton fitted as
the other hand is a mirrored model of it, so a take is measured on the
frames labelled as the operator's hand (`require_label`), and rejected when
too few of them are left.

Without a hand to follow (the uncoached recorder, and files written before
the coached one), the label rule remains: the operator's hand is the label
with the most tracked frames (record_frame's rule, the hand that was
actually held over the camera, whatever it was called). A tie goes to the
label the caller prefers, then to the one that sorts first, so a rerun
picks the same one.

Tracked fraction. LeapC reports nothing at all for a frame in which it lost
the hand, so the file cannot count its own holes. The denominator is the
number of frames the tracker produced over the take, its own measured
`framerate` times the recorded duration; a line with `status` 0 counts as a
frame that was there but not tracked. The acquisition gate (plan section 4)
is at least 90 % tracked and no re-acquisition (a change of LeapC hand id)
inside the static interval. It is the minimum for a usable take, not the
acceptance: that is the operator's call on the still and the numbers. A
followed hand is counted under any label: a chirality flip with no hole in
the data is neither a loss nor a missing frame.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cam_hand.align import align_points
from cam_hand.features import FLEXION_NAMES, flexion_features
from cam_hand.fusion import PALM_IDX
from xr_hand.keypoints21 import frame_to_keypoints21
from xr_hand.kinematics import forward_kinematics
from xr_hand.recorder import _dict_to_frame

DEFAULT_STATIC_S = 2.0
MIN_TRACKED_FRACTION = 0.90
LOSS_GAP_S = 0.100   # an id change after a hole longer than this is a real loss
                     # (the same number as leap_hand.tracking_quality.LOSS_GAP_S)
# A new hand id whose palm is within this of the followed hand's last palm,
# in a frame without the followed hand, is the same hand re-fitted (a
# chirality flip or a re-acquisition). Anything further away, or present in
# the same frame as the followed hand, is another hand. Metres.
FOLLOW_RADIUS_M = 0.10
# With a label required (`summarise_take(require_label=...)`), at least this
# share of the static interval's frames must carry it.
MIN_LABEL_SHARE = 0.5


# --- one line of a take -------------------------------------------------------
def row_time(row: dict) -> float:
    """The line's `wall_time`, the clock the static interval is written in."""
    return float(row["wall_time"])


def is_tracked(row: dict) -> bool:
    """A line counts as tracked unless it says `status` 0.

    The camera never writes status 0 (a lost hand is simply absent, see
    `leap_hand.to_openxr.to_hand_frame`), but the glove's convention allows
    it and a reader must not count such a line as a measurement.
    """
    return row.get("status", 1) != 0


def joint_positions(row: dict) -> np.ndarray:
    """26 x 3 absolute joint positions in metres for one line."""
    if row.get("abs26"):
        return np.asarray(row["abs26"], dtype=float)
    frame, _wall = _dict_to_frame(row)
    return np.asarray(forward_kinematics(frame), dtype=float)


def row_frame(row: dict):
    """The line as the glove's `HandFrame`, exactly as `FrameRecorder.load` reads it."""
    return _dict_to_frame(row)[0]


# --- which hand ---------------------------------------------------------------
def label_counts(rows: Sequence[dict]) -> Dict[str, int]:
    """Lines per tracker label, every line counted: `tracker_hand_labels`."""
    out: Dict[str, int] = {}
    for row in rows:
        side = str(row.get("hand_side"))
        out[side] = out.get(side, 0) + 1
    return {side: out[side] for side in sorted(out)}


def operator_label(rows: Sequence[dict], prefer: Optional[str] = None
                   ) -> Optional[str]:
    """The tracker label whose frames are the operator's hand, or None.

    The label with the most tracked lines; see the module docstring for why
    it is never the label that matches the operator's hand by name. `prefer`
    only breaks a tie.
    """
    counts: Dict[str, int] = {}
    for row in rows:
        if is_tracked(row):
            side = str(row.get("hand_side"))
            counts[side] = counts.get(side, 0) + 1
    if not counts:
        return None
    most = max(counts.values())
    tied = sorted(side for side, n in counts.items() if n == most)
    return prefer if prefer in tied else tied[0]


def hand_rows(rows: Sequence[dict], hand: Optional[str] = None
              ) -> List[Tuple[int, dict]]:
    """(index in `rows`, line) for the tracked lines of one hand, in time order.

    `hand` is a tracker label; None means `operator_label(rows)`. The index
    is the line's position in what the caller passed, so the caller can find
    the untouched original again.
    """
    side = operator_label(rows) if hand is None else hand
    picked = [(i, r) for i, r in enumerate(rows)
              if is_tracked(r) and str(r.get("hand_side")) == side]
    picked.sort(key=lambda ir: (row_time(ir[1]), ir[0]))
    return picked


def _rows(frames: Sequence[dict], hand: Optional[str],
          picked: Optional[Sequence[Tuple[int, dict]]]
          ) -> List[Tuple[int, dict]]:
    """`picked` when the caller already chose the operator's lines (a
    followed hand), else the label rule, `hand_rows(frames, hand)`."""
    return hand_rows(frames, hand) if picked is None else list(picked)


# --- following one hand by its id ---------------------------------------------
def row_hand_id(row: dict) -> Optional[int]:
    """The line's LeapC hand id, or None when it has none (glove lines)."""
    hid = row.get("hand_id")
    return None if hid is None else int(hid)


def row_palm(row: dict) -> Optional[np.ndarray]:
    """The palm position in metres: `palm_abs`, else the first joint.

    None when the line has neither, which `follow_hand` counts as near:
    with nothing to measure, the id is the only evidence there is.
    """
    if row.get("palm_abs"):
        return np.asarray(row["palm_abs"], dtype=float)[:3]
    if not (row.get("abs26") or row.get("joints")):
        return None
    return joint_positions(row)[0]


def _distance(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=float)
                                - np.asarray(b, dtype=float)))


@dataclass
class Followed:
    """The operator's hand, followed by id through one take."""

    # (index in `frames`, line): the tracked lines of the operator's hand
    # only, one per tracking frame, in time order
    rows: List[Tuple[int, dict]] = field(default_factory=list)
    ids: List[int] = field(default_factory=list)      # in order of adoption
    others: List[int] = field(default_factory=list)   # sorted


def follow_hand(frames: Sequence[dict], start_id: int,
                others: Sequence[int] = (),
                start_palm: Optional[Sequence[float]] = None,
                radius_m: float = FOLLOW_RADIUS_M,
                prefer: Optional[str] = None) -> Followed:
    """The tracked lines of the hand acquired as `start_id`, by id.

    `others` are ids already known to be another hand (in view beside the
    acquired one when it was acquired), `start_palm` the acquired hand's
    last palm position in metres, or None. The lines of one tracking frame
    share a `frame_id` (a line without one is a frame of its own), and the
    frames are walked in time order:

      a frame with a line of an id already followed: the one of those
        nearest the last palm is the hand, and every other id in the frame
        is another hand, because two hands seen at once cannot both be the
        operator's;
      a frame without one: of the ids not known to be another hand, the one
        nearest the last palm (no palm yet: the `prefer` label, else the
        first) is the same hand under a new id when it is within `radius_m`
        (a line with no palm counts as near) and is followed from then on;
        the rest, and that one when it is further, are other hands.

    A hand lost and back far from where it was is therefore not taken back:
    its absence counts against the take, which is the honest number.
    """
    groups: Dict[object, List[Tuple[int, dict]]] = {}
    for i, row in enumerate(frames):
        if not is_tracked(row):
            continue
        fid = row.get("frame_id")
        key = ("line", i) if fid is None else ("frame", fid)
        groups.setdefault(key, []).append((i, row))
    ordered = sorted(groups.values(),
                     key=lambda g: min((row_time(r), i) for i, r in g))

    mine = {int(start_id)}
    ids = [int(start_id)]
    other = {int(h) for h in others} - mine
    last_palm = None if start_palm is None else np.asarray(start_palm, dtype=float)
    out: List[Tuple[int, dict]] = []

    def nearest(cands: List[Tuple[int, dict]]) -> Tuple[int, dict]:
        if last_palm is None:
            return cands[0]
        best, best_d = cands[0], None
        for c in cands:
            d = _distance(row_palm(c[1]), last_palm)
            if d is not None and (best_d is None or d < best_d):
                best, best_d = c, d
        return best

    for group in ordered:
        own = [(i, r) for i, r in group if row_hand_id(r) in mine]
        if own:
            taken = nearest(own)
            for _i, r in group:
                hid = row_hand_id(r)
                if hid is not None and hid not in mine:
                    other.add(hid)
        else:
            cands = [(i, r) for i, r in group if row_hand_id(r) not in other]
            if not cands:
                continue
            if last_palm is None:
                pick = next((c for c in cands if prefer is not None
                             and str(c[1].get("hand_side")) == prefer), cands[0])
            else:
                pick = nearest(cands)
            d = _distance(row_palm(pick[1]), last_palm)
            near = d is None or d <= radius_m
            for c in cands:
                hid = row_hand_id(c[1])
                if hid is None:
                    continue
                if c is pick and near:
                    if hid not in mine:
                        mine.add(hid)
                        ids.append(hid)
                else:
                    other.add(hid)
            if not near:
                continue
            taken = pick
        out.append(taken)
        palm = row_palm(taken[1])
        if palm is not None:
            last_palm = palm
    return Followed(rows=out, ids=ids, others=sorted(other - mine))


# --- the static interval ------------------------------------------------------
def _step_motion(picked: Sequence[Tuple[int, dict]]) -> np.ndarray:
    """Mean joint displacement (m) between each line and the one before it.

    Element k is the step from line k-1 to line k; element 0 is 0.
    """
    steps = np.zeros(len(picked))
    prev = None
    for k, (_i, row) in enumerate(picked):
        pos = joint_positions(row)
        if prev is not None and prev.shape == pos.shape:
            steps[k] = float(np.linalg.norm(pos - prev, axis=1).mean())
        prev = pos
    return steps


def static_interval(frames: Sequence[dict], seconds: float = DEFAULT_STATIC_S,
                    hand: Optional[str] = None,
                    picked: Optional[Sequence[Tuple[int, dict]]] = None
                    ) -> Optional[Tuple[float, float]]:
    """(t0, t1) on the `wall_time` clock: the stillest `seconds` of the take.

    `frames` are the take's lines as dicts (the JSONL, parsed). Only the
    tracked lines of `hand` (a tracker label; None = the label with the most
    tracked lines) are used. The window starts on a line and ends `seconds`
    later, and lies entirely inside the span of those lines; its score is the
    joint path travelled inside it divided by the time it covers, i.e. the
    mean joint speed (see the module docstring). The lowest score wins, and
    on an exact tie the later window, because the professor wants the final
    configuration.

    A take shorter than `seconds` has one candidate, the whole take, and gets
    it. None when there is no tracked line of that hand at all. `picked`
    (`hand_rows`-shaped) replaces the label rule with lines the caller chose,
    a followed hand's.
    """
    picked = _rows(frames, hand, picked)
    if not picked:
        return None
    times = np.array([row_time(r) for _i, r in picked])
    if len(picked) == 1 or times[-1] - times[0] <= seconds:
        return float(times[0]), float(times[-1])

    steps = _step_motion(picked)
    path = np.cumsum(steps)                  # path[k] = motion from line 0 to k
    best = None
    best_score = None
    j = 0
    for i in range(len(picked)):
        end = times[i] + seconds
        if end > times[-1]:
            break
        if j < i:
            j = i
        while j + 1 < len(picked) and times[j + 1] <= end:
            j += 1
        span = times[j] - times[i]
        if span <= 0:
            continue
        score = (path[j] - path[i]) / span
        if best_score is None or score <= best_score:
            best, best_score = i, score
    if best is None:                          # every window was one burst
        return float(times[0]), float(times[-1])
    return float(times[best]), float(times[best] + seconds)


def window_rows(frames: Sequence[dict], t0: float, t1: float,
                hand: Optional[str] = None,
                picked: Optional[Sequence[Tuple[int, dict]]] = None
                ) -> List[Tuple[int, dict]]:
    """(index, line) of the tracked lines of `hand` (or of `picked`) with
    t0 <= wall_time <= t1."""
    return [(i, r) for i, r in _rows(frames, hand, picked)
            if t0 <= row_time(r) <= t1]


# --- the summary frame --------------------------------------------------------
def medoid_index(hand_frames) -> int:
    """`scripts/leap/record_frame.py`'s `medoid_index`, the same rule exactly.

    The index of the frame closest to the frames' mean after a rigid
    (rotation and translation, scale 1) alignment onto that mean on the palm
    landmarks. `hand_frames` are `HandFrame`s.
    """
    pts = [np.asarray(frame_to_keypoints21(f), dtype=float) for f in hand_frames]
    if len(pts) == 1:
        return 0
    mean = np.mean(np.stack(pts), axis=0)
    best, best_d = 0, None
    for i, p in enumerate(pts):
        aligned, _rmse, _err, _s = align_points(p, mean, with_scale=False,
                                                subset=PALM_IDX)
        d = float(((aligned - mean) ** 2).sum())
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def medoid_in_window(frames: Sequence[dict], t0: float, t1: float,
                     hand: Optional[str] = None,
                     picked: Optional[Sequence[Tuple[int, dict]]] = None
                     ) -> Optional[Tuple[dict, int]]:
    """(the original line, its index in `frames`) of the window's medoid.

    record_frame's rule applied to the tracked lines of `hand` inside
    [t0, t1] only. The line handed back is the very object that was passed
    in, untouched. None when the window holds no tracked line of that hand.
    `picked` replaces the label rule, as in `static_interval`.
    """
    inside = window_rows(frames, t0, t1, hand, picked)
    if not inside:
        return None
    k = medoid_index([row_frame(r) for _i, r in inside])
    index, row = inside[k]
    return row, index


# --- the acquisition gate -----------------------------------------------------
def reacquisitions(frames: Sequence[dict], t0: Optional[float] = None,
                   t1: Optional[float] = None, hand: Optional[str] = None,
                   picked: Optional[Sequence[Tuple[int, dict]]] = None
                   ) -> List[Tuple[float, int, int]]:
    """Every change of LeapC hand id between consecutive lines of one hand.

    (wall_time of the first line under the new id, old id, new id, gap_s):
    gap_s is the time since the previous line of that hand, so a change
    that arrives with no hole in the data (the tracker re-labelled a hand it
    never stopped seeing) has a gap of one frame, and a change after the
    hand was really gone has a gap of that absence. With a window, only
    changes whose new id first appears strictly after t0 and at or before
    t1 count: a window that starts on a hand re-acquired just before it
    holds one id throughout.

    The 60 s tracking test of 2026-09-30 is why the gap is carried: 99 % of
    the minute tracked, nine id changes, seven of them with no hole at all.
    Rejecting a take for those would reject a hand the camera saw the
    whole time. A followed hand (`picked`) changes id on a chirality flip
    too, with no hole, and that is counted the same way.
    """
    out = []
    prev = None
    prev_t = None
    for _i, row in _rows(frames, hand, picked):
        hid = row.get("hand_id")
        t = row_time(row)
        if prev is not None and hid is not None and prev != hid:
            if (t0 is None or t > t0) and (t1 is None or t <= t1):
                gap = 0.0 if (prev_t is None or t is None) else max(0.0, t - prev_t)
                out.append((t, int(prev), int(hid), gap))
        if hid is not None:
            prev = hid
        if t is not None:
            prev_t = t
    return out


def real_losses(changes, gap_s: float):
    """The id changes that came with the hand actually gone: gap above gap_s."""
    return [c for c in changes if c[3] > gap_s]


def tracked_fraction(frames: Sequence[dict], duration_s: float,
                     hand: Optional[str] = None,
                     picked: Optional[Sequence[Tuple[int, dict]]] = None
                     ) -> Tuple[float, int, int]:
    """(fraction, tracked lines, frames expected) for one hand over one take.

    Expected is the tracker's own median `framerate` times `duration_s`, or
    the number of lines of that hand when no line carries a framerate
    (synthetic data); never fewer than the lines actually written, so a
    tracker that under-reports its rate cannot push the fraction above 1.
    With `picked` (a followed hand's tracked lines, any label) those are
    the hand's lines.
    """
    if picked is None:
        side = operator_label(frames) if hand is None else hand
        lines = [r for r in frames if str(r.get("hand_side")) == side]
    else:
        lines = [r for _i, r in picked]
    tracked = sum(1 for r in lines if is_tracked(r))
    rates = [float(r["framerate"]) for r in frames
             if r.get("framerate") and float(r["framerate"]) > 0]
    if rates and duration_s > 0:
        expected = int(round(float(np.median(rates)) * float(duration_s)))
    else:
        expected = len(lines)
    expected = max(expected, len(lines))
    if expected <= 0:
        return 0.0, tracked, expected
    return tracked / expected, tracked, expected


@dataclass
class TakeSummary:
    """Everything the recorder writes to a take's meta, measured from its lines."""

    frames: int = 0                                   # lines, every label
    labels: Dict[str, int] = field(default_factory=dict)
    # the tracker's label on the summary frame, else the majority label of
    # the operator's lines
    hand_label: Optional[str] = None
    hand_ids: List[int] = field(default_factory=list)       # the operator's hand
    other_ids: List[int] = field(default_factory=list)      # other hands in view
    hand_labels: Dict[str, int] = field(default_factory=dict)   # operator's lines
    # the operator's tracked lines, any label, in time order (not for the meta)
    operator_rows: List[dict] = field(default_factory=list, repr=False)
    tracked_frames: int = 0
    expected_frames: int = 0
    tracked_fraction: float = 0.0
    reacquisitions: int = 0                           # whole take
    interval: Optional[Tuple[float, float]] = None
    interval_reacquisitions: List[Tuple[float, int, int, float]] = field(default_factory=list)
    # the ones with the hand really gone (gap above LOSS_GAP_S); only these fail
    interval_losses: List[Tuple[float, int, int, float]] = field(default_factory=list)
    # with a label required: its frames in the static interval, and the
    # frames the tracker produced over the interval's length
    interval_label_frames: Optional[int] = None
    interval_expected_frames: Optional[int] = None
    medoid_index: Optional[int] = None                # line number in the file, from 0
    medoid_row: Optional[dict] = None
    grab_strength: Optional[float] = None
    pinch_strength: Optional[float] = None
    curls: Optional[Dict[str, float]] = None
    gate_reason: str = ""                             # "" = passed the gate

    @property
    def passed(self) -> bool:
        return not self.gate_reason

    @property
    def medoid_wall_time(self) -> Optional[float]:
        return None if self.medoid_row is None else row_time(self.medoid_row)


def _median_rate(frames: Sequence[dict]) -> Optional[float]:
    rates = [float(r["framerate"]) for r in frames
             if r.get("framerate") and float(r["framerate"]) > 0]
    return float(np.median(rates)) if rates else None


def _other_label(rows: Sequence[dict], label: str) -> Optional[str]:
    """The most common tracker label among `rows` that is not `label`."""
    counts = {k: n for k, n in label_counts(rows).items() if k != label}
    if not counts:
        return None
    most = max(counts.values())
    return sorted(k for k, n in counts.items() if n == most)[0]


def summarise_take(frames: Sequence[dict], duration_s: float,
                   static_s: float = DEFAULT_STATIC_S,
                   prefer: Optional[str] = None,
                   min_tracked: float = MIN_TRACKED_FRACTION,
                   follow: Optional[Tuple[int, Sequence[int],
                                          Optional[Sequence[float]]]] = None,
                   require_label: Optional[str] = None) -> TakeSummary:
    """Measure one take and pass the acquisition gate on it.

    `follow` is (hand id, ids of other hands, last palm position in metres
    or None) of the hand the coached recorder acquired: the operator's lines
    are then `follow_hand`'s, whatever their label. Without it they are the
    label rule's, `hand_rows(frames, operator_label(frames, prefer))`, and
    `prefer` only breaks a tie between two labels with the same number of
    frames. `tracked_fraction` and the whole take's `reacquisitions` are
    over the operator's lines, any label.

    `require_label` is the operator's hand when the tracker's label must
    match it: the static interval and the summary frame are then chosen
    from the lines carrying that label only, because a skeleton fitted as
    the other hand is a mirrored model of it, and the take is rejected when
    there is no such line, or when they are under half of the frames the
    tracker produced over the static interval's length (`static_s`, or the
    take when it is shorter). Id changes and losses inside the interval are
    still judged on the operator's lines, any label.

    The summary frame's `grab_strength` and `pinch_strength` are LeapC's
    own values on that line, and `curls` are
    `cam_hand.features.flexion_features` of its 21 keypoints, the
    per-finger curl every other report in the repo uses.
    """
    s = TakeSummary(frames=len(frames), labels=label_counts(frames))
    tracked_ids = sorted({hid for r in frames if is_tracked(r)
                          for hid in [row_hand_id(r)] if hid is not None})
    if follow is not None:
        start_id, others, start_palm = follow
        got = follow_hand(frames, start_id, others or (), start_palm,
                          prefer=prefer)
        mine = got.rows
        s.hand_ids = list(got.ids)
        # "in view": the other hands that are actually in this take's lines
        s.other_ids = [h for h in got.others if h in tracked_ids]
    else:
        label = operator_label(frames, prefer)
        mine = [] if label is None else hand_rows(frames, label)
        seen: List[int] = []
        for _i, r in mine:
            hid = row_hand_id(r)
            if hid is not None and hid not in seen:
                seen.append(hid)
        s.hand_ids = seen
        s.other_ids = [h for h in tracked_ids if h not in seen]
    s.operator_rows = [r for _i, r in mine]
    s.hand_labels = label_counts(s.operator_rows)
    if not mine:
        s.gate_reason = "no hand was tracked during the take"
        return s
    majority = operator_label(s.operator_rows, prefer)
    s.hand_label = majority
    if follow is not None:
        s.tracked_fraction, s.tracked_frames, s.expected_frames = tracked_fraction(
            frames, duration_s, picked=mine)
    else:
        s.tracked_fraction, s.tracked_frames, s.expected_frames = tracked_fraction(
            frames, duration_s, majority)
    s.reacquisitions = len(reacquisitions(frames, picked=mine))
    measured = (mine if require_label is None else
                [(i, r) for i, r in mine
                 if str(r.get("hand_side")) == require_label])
    s.interval = static_interval(frames, static_s, picked=measured)
    if s.interval is not None:
        t0, t1 = s.interval
        s.interval_reacquisitions = reacquisitions(frames, t0, t1, picked=mine)
        s.interval_losses = real_losses(s.interval_reacquisitions, LOSS_GAP_S)
        if require_label is not None:
            # The interval's nominal length, not its span: lines with the
            # label only at the very end of a take span less than static_s,
            # and the window chosen over them is that short too.
            length = (min(float(static_s), float(duration_s))
                      if duration_s > 0 else t1 - t0)
            rate = _median_rate(frames)
            s.interval_expected_frames = (
                int(round(rate * length)) if rate else
                len(window_rows(frames, t0, t1, picked=mine)))
            s.interval_label_frames = len(window_rows(frames, t0, t1,
                                                      picked=measured))
        found = medoid_in_window(frames, t0, t1, picked=measured)
        if found is not None:
            s.medoid_row, s.medoid_index = found
            row = s.medoid_row
            s.grab_strength = (None if row.get("grab_strength") is None
                               else round(float(row["grab_strength"]), 4))
            s.pinch_strength = (None if row.get("pinch_strength") is None
                                else round(float(row["pinch_strength"]), 4))
            curls = flexion_features(frame_to_keypoints21(row_frame(row)))
            s.curls = {name: round(float(c), 4)
                       for name, c in zip(FLEXION_NAMES, curls)}
            s.hand_label = str(row.get("hand_side"))

    if s.tracked_fraction < min_tracked:
        s.gate_reason = (f"tracked {s.tracked_fraction * 100:.0f} % of the "
                         f"take's frames ({s.tracked_frames} of "
                         f"{s.expected_frames}); the gate needs "
                         f"{min_tracked * 100:.0f} %")
    elif s.interval_losses:
        t, old, new, gap = s.interval_losses[0]
        s.gate_reason = (f"the tracker lost the hand for {gap:.2f} s inside the "
                         f"static interval and re-acquired it (id {old} -> {new})")
    elif require_label is not None and not measured:
        s.gate_reason = (f"the tracker fitted the hand as a {majority} hand for "
                         f"the whole take; the {require_label} hand cannot be "
                         "measured from that")
    elif (require_label is not None and s.interval_expected_frames is not None
          and s.interval_label_frames is not None
          and s.interval_label_frames
          < MIN_LABEL_SHARE * s.interval_expected_frames):
        t0, t1 = s.interval
        inside = [r for _i, r in window_rows(frames, t0, t1, picked=mine)]
        other = (_other_label(inside, require_label)
                 or _other_label(s.operator_rows, require_label) or "different")
        s.gate_reason = (f"the tracker fitted the hand as a {other} hand for "
                         f"most of the static interval ({s.interval_label_frames}"
                         f" of {s.interval_expected_frames} frames as "
                         f"{require_label}); the {require_label} hand cannot be "
                         "measured from that")
    elif s.medoid_row is None:
        s.gate_reason = "no tracked frame inside the static interval"
    return s
