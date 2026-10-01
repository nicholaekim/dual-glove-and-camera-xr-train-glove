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
the label never decides whose hand a frame is. When the tracker reports two
hands, the label is only used to keep them apart, and the operator's hand is
the label with the most tracked frames (record_frame's rule, the hand that
was actually held over the camera, whatever it was called). A tie goes to
the label the caller prefers, then to the one that sorts first, so a rerun
picks the same one.

Tracked fraction. LeapC reports nothing at all for a frame in which it lost
the hand, so the file cannot count its own holes. The denominator is the
number of frames the tracker produced over the take, its own measured
`framerate` times the recorded duration; a line with `status` 0 counts as a
frame that was there but not tracked. The acquisition gate (plan section 4)
is at least 90 % tracked and no re-acquisition (a change of LeapC hand id)
inside the static interval. It is the minimum for a usable take, not the
acceptance: that is the operator's call on the still and the numbers.
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
                    hand: Optional[str] = None
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
    it. None when there is no tracked line of that hand at all.
    """
    picked = hand_rows(frames, hand)
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
                hand: Optional[str] = None) -> List[Tuple[int, dict]]:
    """(index, line) of the tracked lines of `hand` with t0 <= wall_time <= t1."""
    return [(i, r) for i, r in hand_rows(frames, hand)
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
                     hand: Optional[str] = None
                     ) -> Optional[Tuple[dict, int]]:
    """(the original line, its index in `frames`) of the window's medoid.

    record_frame's rule applied to the tracked lines of `hand` inside
    [t0, t1] only. The line handed back is the very object that was passed
    in, untouched. None when the window holds no tracked line of that hand.
    """
    inside = window_rows(frames, t0, t1, hand)
    if not inside:
        return None
    k = medoid_index([row_frame(r) for _i, r in inside])
    index, row = inside[k]
    return row, index


# --- the acquisition gate -----------------------------------------------------
def reacquisitions(frames: Sequence[dict], t0: Optional[float] = None,
                   t1: Optional[float] = None, hand: Optional[str] = None
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
    whole time.
    """
    out = []
    prev = None
    prev_t = None
    for _i, row in hand_rows(frames, hand):
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
                     hand: Optional[str] = None) -> Tuple[float, int, int]:
    """(fraction, tracked lines, frames expected) for one hand over one take.

    Expected is the tracker's own median `framerate` times `duration_s`, or
    the number of lines of that hand when no line carries a framerate
    (synthetic data); never fewer than the lines actually written, so a
    tracker that under-reports its rate cannot push the fraction above 1.
    """
    side = operator_label(frames) if hand is None else hand
    lines = [r for r in frames if str(r.get("hand_side")) == side]
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
    hand_label: Optional[str] = None                  # label taken as the operator's hand
    tracked_frames: int = 0
    expected_frames: int = 0
    tracked_fraction: float = 0.0
    reacquisitions: int = 0                           # whole take
    interval: Optional[Tuple[float, float]] = None
    interval_reacquisitions: List[Tuple[float, int, int, float]] = field(default_factory=list)
    # the ones with the hand really gone (gap above LOSS_GAP_S); only these fail
    interval_losses: List[Tuple[float, int, int, float]] = field(default_factory=list)
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


def summarise_take(frames: Sequence[dict], duration_s: float,
                   static_s: float = DEFAULT_STATIC_S,
                   prefer: Optional[str] = None,
                   min_tracked: float = MIN_TRACKED_FRACTION) -> TakeSummary:
    """Measure one take and pass the acquisition gate on it.

    `prefer` is the operator's hand, used only to break a tie between two
    labels with the same number of frames. The summary frame's
    `grab_strength` and `pinch_strength` are LeapC's own values on that line,
    and `curls` are `cam_hand.features.flexion_features` of its 21
    keypoints, the per-finger curl every other report in the repo uses.
    """
    s = TakeSummary(frames=len(frames), labels=label_counts(frames))
    s.hand_label = operator_label(frames, prefer)
    if s.hand_label is None:
        s.gate_reason = "no hand was tracked during the take"
        return s
    s.tracked_fraction, s.tracked_frames, s.expected_frames = tracked_fraction(
        frames, duration_s, s.hand_label)
    s.reacquisitions = len(reacquisitions(frames, hand=s.hand_label))
    s.interval = static_interval(frames, static_s, s.hand_label)
    if s.interval is not None:
        t0, t1 = s.interval
        s.interval_reacquisitions = reacquisitions(frames, t0, t1, s.hand_label)
        s.interval_losses = real_losses(s.interval_reacquisitions, LOSS_GAP_S)
        found = medoid_in_window(frames, t0, t1, s.hand_label)
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

    if s.tracked_fraction < min_tracked:
        s.gate_reason = (f"tracked {s.tracked_fraction * 100:.0f} % of the "
                         f"take's frames ({s.tracked_frames} of "
                         f"{s.expected_frames}); the gate needs "
                         f"{min_tracked * 100:.0f} %")
    elif s.interval_losses:
        t, old, new, gap = s.interval_losses[0]
        s.gate_reason = (f"the tracker lost the hand for {gap:.2f} s inside the "
                         f"static interval and re-acquired it (id {old} -> {new})")
    elif s.medoid_row is None:
        s.gate_reason = "no tracked frame inside the static interval"
    return s
