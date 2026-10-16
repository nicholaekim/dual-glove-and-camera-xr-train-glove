"""How long the hand took to change from the open palm into the grasp.

The professor asked for the "time to change from poses" as well as the poses
themselves. Set A's coached take already has the moment the change is asked
for: MAKE THE GRASP starts with the GO cue, one long, loud, high beep, the
operator's cue to close the open hand into the grasp
(`scripts/leap/record_poses.py`, `cue_go`, which plays `BEEP_GO`).
The recorder now keeps every frame from that cue until the take's recording
starts (`forming/<take>.jsonl`), and this module reads one number out of
them: `transition_s`, the seconds from the cue to the first moment the hand
has settled after its close.

Settled. The hand's mean joint speed, over its 26 joints (`abs26`, the
camera's own absolute positions, as `leap_hand.static_interval` uses them;
not the 21 landmarks, which are derived from the same joints), stays below
`SETTLE_SPEED_MM_S` for `SETTLE_WINDOW_S` without a break, after the close.
The close is the hand's main movement after the cue, its fastest frame
(`peak_speed_mm_s`, at `peak_at_s`), and the quiet window is searched for
after it (`settle`). Anchored there because the hand is still at other
times too: at the cue it is the open palm OPEN HAND acquired, waiting for
the operator's reaction, and a twitch on the beep followed by a pause
before the real close is not the grasp either. `transition_s` is the start
of the first quiet window after the peak, measured from the cue. Beside it
`last_move_s`, the last frame at `MOVE_SPEED_MM_S` or faster before the
forming phase ended (the end of the hand's last real movement), and
`transition_note`, which is "" for a plain close and otherwise one of:

  "no close after the cue (peak X mm/s): the hand was already in the grasp"
      the fastest frame after the cue is below `MOVE_SPEED_MM_S`. The
      operator closed before the cue, and there is nothing to time:
      transition_s None (and last_move_s None).
  "settled, then moved again at T s"
      a frame at `MOVE_SPEED_MM_S` or faster after the quiet window, the
      grip adjusted after it had settled: transition_s is still the settle
      after the close, T the first frame of the new movement, and
      last_move_s the end of the last one.
  "never settled before the hold"
      no quiet window after the close before the forming phase ended (or
      the hand was lost while it closed): transition_s None.

(and, when not one frame of the hand after the cue could be measured, "no
frame of the hand after the cue to measure", with every number None.)

A hand lost while forming (since 2026-10-07). The recorder used to throw
an attempt away when the tracker lost the hand while it closed; now it
waits for the hand, and keeps the take when the hold is clean. The forming
file of such a take has a gap in the tracking inside the change of pose,
and nothing here can say when a change that was not seen began or ended.
So the recorder writes `FORMING_LOST_KEY` (how many times the hand was
lost) on every line of that file, and every reading of it says so instead
of a number: `summarise_forming` a null `transition_s` with the note
`NOTE_LOST_WHILE_FORMING`, `transition_timing` INVALID with that note and
no start, settle or duration. The mark is on the lines, not only in the
meta, so the file read on its own (the hand-in folder holds it without its
meta) and every check that times it again say the same.

From the start of the change of pose, not from the cue, and of the pose,
not of the hand over the camera. `transition_s` holds the operator's
reaction to the beep and every movement of the hand as a whole.
`transition_timing` (2026-10-05) takes both out. It reads the pose speed:
the same centred difference, after the later of the two frames has been
laid on the earlier by the rigid fit of the palm's joints, so a hand carried
across the camera with its pose held reads near zero (`pose_speeds`). The
movement is the stretch of pose speed at `POSE_MOVE_SPEED_MM_S` or faster,
`START_SUSTAIN_S` or longer, that holds the fastest frame after the cue;
its first frame is the movement start, the settle is the first frame after
its fastest from which the pose speed stays under `POSE_SETTLE_SPEED_MM_S`
for `SETTLE_WINDOW_S`, and `transition_duration_s` is the settle minus the
start, with the reaction time and the cue-to-settle time beside it. It also
asks whether the pose was still at the cue: a frame at the start speed
within `CUE_STILL_S` of the cue is a change that began before it, and such
a take gets no start and no duration (`INVALID`, with the reason).

A first version of this rule, the same day, used the speed over the camera.
It passed one of the six forming files of 2026-10-02, p1_cylindrical take
4, and gave it 1.50 s; an independent review showed that most of that was
the open hand being carried 25 mm sideways and back (398 mm/s over the
camera, the pose within 2 mm RMS of the open hand until 0.7 s), and that
the three takes it had called "already moving at the cue" were hands being
carried with their pose still. On the pose speed the six read: four changes
of pose, starting 0.57 to 0.81 s after the cue and lasting 0.80 to 1.11 s
(p1_cylindrical 4 and 5, p1_hook 4 and 5; two valid, take 4 of each
questionable: a stir of the pose before the main movement, a grip adjusted
after it), and two takes whose hand was already in the grasp (p1_lateral 4
and 5, the pose at the end 8 and 9 mm RMS from the pose at the cue). The
length depends on where "settled" is put: at twice the thresholds (50 and
20 mm/s) the same four changes read 0.30 to 0.46 s, the fast part of the
close alone; the pose goes on creeping for half a second after it. Every
threshold is an argument; the meta's `transition_s` and the recorder's own
checks, which watch the hand over the camera, are untouched by any of this.

The speed. Per frame, a centred difference over `SPEED_STEP_S` (0.1 s): the
mean joint displacement between the frames at the two ends of the 0.1 s
around it, divided by the time between them. Frame to frame the camera's
step is mostly its own jitter, and a 0.1 s base keeps the jitter's share
small without smearing a stop by more than half a step: an abrupt stop is
found at most 0.05 s late. The clock is `capture_time`, when the camera saw
the hand, on the same wall clock as the cue (`time.time()` at the beep);
`wall_time` only where a line has no capture time (a replay). Not
`wall_time` first, as the static interval has it, because that is a writer
clock: hands written in one burst share it to the microsecond, and a
per-frame speed divided by it explodes (see `leap_hand.static_interval`).

The thresholds were read from the hold data, there being no forming data on
disk when they were chosen: the 63 kept takes of the session
`recordings/protocol/grasps/20261001_101621_left`, each take's 2 s static
interval (its "least joint motion", about 180 frames at 90 Hz), the frames of
the followed left hand, the speed exactly as above. Per take, the median
speed over the interval: 6.4 mm/s across takes (90th percentile 14.7, the
largest 34.0). Per take, the 95th percentile: 13.0 mm/s across takes (90th
percentile 34.6, the largest 78.0). The interval's mean speed as the static
interval scores it (joint path over elapsed `wall_time`): 8.8 mm/s (90th
percentile 20.8, the largest 36.2). A hand held still therefore moves its
joints at about 6 mm/s, and 20 mm/s is three times that, still above the
typical take's 95th percentile. Checked the other way, 62 of those 63 hold
takes contain a 0.3 s window entirely below 20 mm/s (the one that does not,
p3_circular_precision_thumb_1_finger take 2, moved at a median 20.9 mm/s
even in its stillest 2 s: a hand that unsteady gets no transition_s rather
than a wrong one).

`MOVE_SPEED_MM_S`, 50 mm/s, was read from the first forming files, the six
takes of the same session recorded on 2026-10-02 (p1_cylindrical, p1_hook
and p1_lateral, takes 4 and 5), the followed hand's speed exactly as above.
Per 0.25 s after the cue, the median speed: a real close runs at 80 to 190
mm/s, a twitch on the beep or a grip adjusted afterwards at 20 to 35, and a
hand holding the grasp at 20 or less. Per frame, the fastest after the cue:
180, 191, 268 and 398 mm/s for the four takes that closed on the cue, 38 and
43 for the two lateral takes whose hand was already closed (the operator
closed on the acquisition beep, which then sounded right before the high
one). 50 is above both of those and under a third of the slowest real
close's peak. The first rule, settled after the first frame above 20 mm/s,
timed those six at 1.69, 2.80, 2.15, 0.18, 0.24 and 0.41 s: right for the
first three, but hook take 5's 0.18 s was a twitch at the cue and a pause
before its close at 0.5 to 1.0 s, and the two lateral numbers timed jitter
of a hand that had nothing left to close. The peak-anchored rule gives the
first three the same values, hook take 5 2.10 s (the close ends 0.98 s
after the cue, `last_move_s`, and the hand then drifts at a median 22 to 24
mm/s until about 2 s), and the lateral takes no transition with the note.
Hook take 4 settled at 2.15 s and adjusted its grip at 3.1 and 3.8 s, which
its note now says. And the motion it has to see: the mock hand closing from
open to grab 0.6 in 0.5 s (`leap_hand.mock.CoachedActor`'s default) moves
its joints at 67 mm/s on average over the close, 100 mm/s at the peak, the
speed of the slower real closes; in 1 s, 37 and 55 mm/s, barely a close.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cam_hand.align import umeyama
from xr_hand.joints import JOINT_INDEX

from .static_interval import (LOSS_GAP_S, follow_hand, hand_rows, is_tracked,
                              joint_positions, operator_label,
                              tracked_fraction)

SETTLE_SPEED_MM_S = 20.0   # mean joint speed below this counts as settled
SETTLE_WINDOW_S = 0.3      # ... without a break for this long
SPEED_STEP_S = 0.1         # the base of each frame's centred speed
MOVE_SPEED_MM_S = 50.0     # a peak below this after the cue is no close

# `transition_note`, beside the number (`settle`)
NOTE_NO_CLOSE = ("no close after the cue (peak {peak:.0f} mm/s): the hand was "
                 "already in the grasp")
NOTE_MOVED_AGAIN = "settled, then moved again at {at:.2f} s"
NOTE_NEVER_SETTLED = "never settled before the hold"
NOTE_NO_FRAMES = "no frame of the hand after the cue to measure"

# A forming file in which the hand was lost (module docstring): the key the
# recorder writes on every one of its lines, the number of losses, and what
# every reading of such a file says in place of a time.
FORMING_LOST_KEY = "forming_lost_count"
NOTE_LOST_WHILE_FORMING = ("the hand was lost while forming the grasp ({n} "
                           "{times}): a change of pose is not timed across a "
                           "gap in the tracking")


def forming_lost_count(frames: Sequence[dict]) -> int:
    """How many times the hand was lost while forming the grasp, as the
    recorder marked it on the lines (`FORMING_LOST_KEY`): the largest mark,
    0 on lines without one."""
    n = 0
    for r in frames:
        v = r.get(FORMING_LOST_KEY)
        if v:
            n = max(n, int(v))
    return n


def lost_note(n: int) -> str:
    """`NOTE_LOST_WHILE_FORMING` for `n` losses."""
    return NOTE_LOST_WHILE_FORMING.format(n=int(n),
                                          times="time" if n == 1 else "times")


def frame_time(row: dict) -> float:
    """When the camera saw the hand: `capture_time`, else `wall_time`."""
    t = row.get("capture_time")
    return float(row["wall_time"] if t is None else t)


def centred_speeds(t: np.ndarray, pos: Sequence[np.ndarray],
                   step_s: float = SPEED_STEP_S, start: int = 0) -> np.ndarray:
    """The mean joint speed in mm/s of frames `start` onward, from their
    times `t` (seconds, ascending) and 26 x 3 joint positions in metres.

    The centred difference over `step_s` described in the module docstring;
    NaN for a frame with no other frame within half a step on either side
    (a lone frame between two holes), which no quiet window can then
    contain. `start` lets a live watch (the recorder's READY) work out only
    the frames it has not seen yet.
    """
    half = float(step_s) / 2.0
    speeds = np.full(max(0, len(t) - start), np.nan)
    for k in range(start, len(t)):
        a = int(np.searchsorted(t, t[k] - half, side="left"))
        b = int(np.searchsorted(t, t[k] + half, side="right")) - 1
        if b <= a or t[b] - t[a] <= 0 or pos[a].shape != pos[b].shape:
            continue
        step = float(np.linalg.norm(pos[b] - pos[a], axis=1).mean())
        speeds[k - start] = step / (t[b] - t[a]) * 1000.0
    return speeds


def joint_speeds(frames: Sequence[dict], step_s: float = SPEED_STEP_S
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(times, mean joint speed in mm/s) for one hand's lines, in time order
    (`centred_speeds` of their `frame_time` and `abs26`)."""
    rows = sorted(frames, key=frame_time)
    if not rows:
        return np.zeros(0), np.zeros(0)
    t = np.array([frame_time(r) for r in rows])
    return t, centred_speeds(t, [joint_positions(r) for r in rows], step_s)


@dataclass
class Settle:
    """The change into the grasp, as `settle` reads it from one hand's lines.

    Every time is in seconds from the cue. `peak_speed_mm_s` and
    `peak_at_s` are the fastest frame (None with no measured frame);
    `transition_s` the start of the first quiet window after it, None when
    there was no close or no window; `last_move_s` the last frame at
    `MOVE_SPEED_MM_S` or faster, the end of the hand's last real movement,
    None when it never moved that fast; `moved_again_s` the first such
    frame after the quiet window, when there is one; and `note` says why a
    transition is missing or what came after it ("" when nothing did).
    """

    peak_speed_mm_s: Optional[float] = None
    peak_at_s: Optional[float] = None
    transition_s: Optional[float] = None
    last_move_s: Optional[float] = None
    moved_again_s: Optional[float] = None
    note: str = ""


def _search(t: np.ndarray, speed: np.ndarray, t0: float,
            speed_thresh_mm_s: float, window_s: float, gap_s: float,
            move_thresh_mm_s: float) -> Settle:
    """`settle` on a speed series that is already worked out: `t` and
    `speed` are the frames the close is looked for in, in time order, and
    every time handed back is counted from `t0`."""
    out = Settle()
    if len(t) < 2 or np.all(np.isnan(speed)):
        out.note = NOTE_NO_FRAMES
        return out
    peak = int(np.nanargmax(speed))
    out.peak_speed_mm_s = float(speed[peak])
    out.peak_at_s = float(t[peak] - t0)
    moving = np.flatnonzero(speed >= move_thresh_mm_s)   # NaN never is
    if len(moving):
        out.last_move_s = float(t[moving[-1]] - t0)
    if out.peak_speed_mm_s < move_thresh_mm_s:
        out.note = NOTE_NO_CLOSE.format(peak=out.peak_speed_mm_s)
        return out
    quiet = speed < speed_thresh_mm_s           # NaN is never quiet
    # Half a frame of slack, so a window of 27 frames at 90 Hz is 0.3 s.
    slack = 0.5 * float(np.median(np.diff(t)))
    n = len(t)
    i = peak + 1
    found = None
    while i < n and found is None:
        if not quiet[i]:
            i += 1
            continue
        if window_s - slack <= 0:
            found = i
            break
        j = i
        while j + 1 < n and quiet[j + 1] and t[j + 1] - t[j] <= gap_s:
            j += 1
            if t[j] - t[i] >= window_s - slack:
                found = i
                break
        if found is None:
            if j + 1 >= n:
                break                    # quiet to the end, but not for long enough
            i = j + 1
    if found is None:
        out.note = NOTE_NEVER_SETTLED
        return out
    out.transition_s = float(t[found] - t0)
    again = moving[moving > found]
    if len(again):
        out.moved_again_s = float(t[again[0]] - t0)
        out.note = NOTE_MOVED_AGAIN.format(at=out.moved_again_s)
    return out


def settle(frames: Sequence[dict], t0: float,
           speed_thresh_mm_s: float = SETTLE_SPEED_MM_S,
           window_s: float = SETTLE_WINDOW_S,
           t_end: Optional[float] = None,
           step_s: float = SPEED_STEP_S,
           gap_s: float = LOSS_GAP_S,
           move_thresh_mm_s: float = MOVE_SPEED_MM_S) -> Settle:
    """The close after the cue `t0`, and when the hand settled after it.

    `frames` are one hand's lines (the followed hand's, see
    `summarise_forming`); only those seen at or after `t0` and, with
    `t_end`, at or before it count, so a window that would run past the end
    of the forming phase is no window. The hand's main movement is its
    fastest frame. Below `move_thresh_mm_s` there was no close after the
    cue at all: the hand was already in the grasp (it closed on the earlier
    beep) and there is nothing to time. Otherwise `transition_s` is the
    time of the first frame after the peak from which every frame for
    `window_s` is slower than `speed_thresh_mm_s`, with no hole longer than
    `gap_s` between two of them (a hand the camera lost for a moment is not
    seen to be still); a twitch and a pause before the close are not the
    settle, because the search starts at the close. A frame at
    `move_thresh_mm_s` after that window is the hand moving again (a grip
    adjusted), which the note says; it is at least `window_s` after the
    transition by construction, the window being quiet throughout.
    """
    rows = [r for r in frames if is_tracked(r) and frame_time(r) >= t0
            and (t_end is None or frame_time(r) <= t_end)]
    t, speed = joint_speeds(rows, step_s)
    return _search(t, speed, t0, speed_thresh_mm_s, window_s, gap_s,
                   move_thresh_mm_s)


def settle_time(frames: Sequence[dict], t0: float,
                speed_thresh_mm_s: float = SETTLE_SPEED_MM_S,
                window_s: float = SETTLE_WINDOW_S,
                t_end: Optional[float] = None,
                step_s: float = SPEED_STEP_S,
                gap_s: float = LOSS_GAP_S,
                move_thresh_mm_s: float = MOVE_SPEED_MM_S) -> Optional[float]:
    """Seconds from `t0` (the close cue) until the hand has settled after
    its close, or None: `settle(...).transition_s`."""
    return settle(frames, t0, speed_thresh_mm_s, window_s, t_end, step_s,
                  gap_s, move_thresh_mm_s).transition_s


def transition_rule() -> dict:
    """The thresholds a meta's `coaching.transition_rule` records."""
    return {"speed_mm_s": SETTLE_SPEED_MM_S, "window_s": SETTLE_WINDOW_S,
            "step_s": SPEED_STEP_S, "move_speed_mm_s": MOVE_SPEED_MM_S}


@dataclass
class FormingSummary:
    """What a take's meta says about its forming file (`coaching`)."""

    frames: int = 0                      # lines, every hand
    tracked_frames: int = 0              # the followed hand's
    expected_frames: int = 0
    tracked_fraction: Optional[float] = None
    transition_s: Optional[float] = None
    hand_ids: List[int] = field(default_factory=list)   # the ids followed
    peak_speed_mm_s: Optional[float] = None
    peak_at_s: Optional[float] = None
    last_move_s: Optional[float] = None
    transition_note: str = ""

    def coaching_numbers(self) -> dict:
        """The meta's `coaching` transition keys, rounded as the recorder
        writes them (`scripts/leap/recompute_transition.py` rewrites
        exactly these)."""
        def r(v, nd):
            return None if v is None else round(float(v), nd)
        return {"transition_s": r(self.transition_s, 3),
                "peak_speed_mm_s": r(self.peak_speed_mm_s, 1),
                "peak_at_s": r(self.peak_at_s, 3),
                "last_move_s": r(self.last_move_s, 3),
                "transition_note": self.transition_note,
                "transition_rule": transition_rule()}


def summarise_forming(rows: Sequence[dict], t0: float, t_end: float,
                      follow: Optional[Tuple[int, Sequence[int],
                                             Optional[Sequence[float]]]] = None,
                      require_label: Optional[str] = None,
                      prefer: Optional[str] = None) -> FormingSummary:
    """The forming file's frame count, tracked fraction and transition.

    `follow` is the hand OPEN HAND acquired (hand id, other hands' ids, its
    palm in metres), followed by id exactly as the take is
    (`static_interval.follow_hand`); without one, the label with the most
    tracked lines. `tracked_fraction` is what
    `static_interval.tracked_fraction` computes for a take: the followed
    hand's tracked lines over the frames the tracker produced from the cue
    `t0` to the start of the hold `t_end` (its median `framerate` times
    that), never more than 1. The transition is measured on the followed
    hand's lines that carry `require_label`, the operator's hand, for the
    reason the static interval is: a skeleton fitted as the other hand is a
    mirrored model, and a jump between the two is not motion. The peak,
    the last movement and the note are `settle`'s, on the same lines. A
    file whose lines say the hand was lost while forming
    (`FORMING_LOST_KEY`) gets its frame count and tracked fraction and no
    transition: the note `NOTE_LOST_WHILE_FORMING`, every time None.
    """
    s = FormingSummary(frames=len(rows), hand_ids=[])
    if follow is not None:
        start_id, others, palm = follow
        got = follow_hand(rows, start_id, others or (), palm, prefer=prefer)
        mine, s.hand_ids = got.rows, list(got.ids)
    else:
        label = operator_label(rows, prefer)
        mine = [] if label is None else hand_rows(rows, label)
        s.hand_ids = sorted({int(r["hand_id"]) for _i, r in mine
                             if r.get("hand_id") is not None})
    if not rows:
        s.transition_note = NOTE_NO_FRAMES
        return s
    frac, s.tracked_frames, s.expected_frames = tracked_fraction(
        rows, max(0.0, float(t_end) - float(t0)), picked=mine)
    s.tracked_fraction = min(1.0, frac)
    lost = forming_lost_count(rows)
    if lost:
        s.transition_note = lost_note(lost)
        return s
    measured = [r for _i, r in mine
                if require_label is None or str(r.get("hand_side")) == require_label]
    got = settle(measured, t0, t_end=t_end)
    s.transition_s = got.transition_s
    s.peak_speed_mm_s, s.peak_at_s = got.peak_speed_mm_s, got.peak_at_s
    s.last_move_s, s.transition_note = got.last_move_s, got.note
    return s


# --- from the start of the change of pose to the settled pose -----------------
# `transition_s` above runs from the cue and is measured on the joints as the
# camera saw them, so it holds the operator's reaction to the beep and every
# movement of the hand as a whole. The time a hand takes to change pose is
# neither: it is the movement of the pose alone, from that movement's start.
# `transition_timing` measures that. Every number is an argument of it: none
# is fixed in the rule.
#
# The pose speed. A hand can cross the camera without changing shape, and in
# the first forming files it did. The one take a rule on the camera-frame
# speed passed (p1_cylindrical take 4, 2026-10-02) shifted 25 mm sideways
# and back while still open, at up to 398 mm/s, and that shift was most of
# the "1.50 s" it was given; the pose itself changed between about 0.8 and
# 1.3 s after the cue (an independent review found this on 2026-10-05). So
# the whole hand's movement is taken out before anything is timed: of the
# two frames a speed is taken between, the later is laid on the earlier by
# the rotation and translation that best fit the palm's joints
# (`PALM_JOINTS`), and the speed is the mean over the 26 joints of what is
# left (`pose_speeds`). The speed over the camera is kept beside it
# (`hand_speed_at_cue_mm_s`, `hand_peak_speed_mm_s`): it says what the hand
# as a whole was doing, and decides nothing.
#
# The palm's joints are the wrist, the palm centre, the bases of the digits
# and the knuckles: the part of the hand that bending the fingers does not
# move. They are not a rigid body. Measured on the 69 kept takes of the
# session 20261001_101621_left with the tracker's change of hand size taken
# out, the eleven stay within 0.35 mm RMS of themselves over a take at the
# median (4.0 mm in the worst take); but from one take to another the WRIST
# joint sits up to 21 mm elsewhere against the palm, following the bend of
# the wrist (a linear fit on the PALM row's three angles explains 97 and
# 99.5 % of its two large components), and the tracker's hand model changes
# in proportion as well as in size. Both are read as pose movement.
#
# Which joints the hand is laid on matters little: without the WRIST
# (`PALM_JOINTS_NO_WRIST`) the four timed closes of that session keep their
# validity and their durations to within one frame (0.011 s), and a held
# pose still reads 1.2 mm/s (2026-10-05). `transition_timing` takes the set
# as `palm_joints`, so a hand-in can show that for its own recordings.
PALM_JOINTS = ("WRIST", "PALM", "THUMB_METACARPAL", "INDEX_METACARPAL",
               "MIDDLE_METACARPAL", "RING_METACARPAL", "LITTLE_METACARPAL",
               "INDEX_PROXIMAL", "MIDDLE_PROXIMAL", "RING_PROXIMAL",
               "LITTLE_PROXIMAL")
PALM_INDEX = [JOINT_INDEX[n] for n in PALM_JOINTS]
PALM_JOINTS_NO_WRIST = tuple(n for n in PALM_JOINTS if n != "WRIST")

# The thresholds on the pose speed, read from the same session (2026-10-05):
#   a held pose        inside the 2 s static intervals of the 69 kept takes,
#                      the median pose speed of a take is 1.2 mm/s at the
#                      median over takes (4.2 in the least steady take), its
#                      fastest frame 3.3 at the median;
#   a pose settled     0.8 to 3.1 mm/s at the median over the last 0.5 s of
#   after its change   the five forming files that ended still;
#   a change of pose   its fastest frame 71, 85, 143 and 171 mm/s in the
#                      four forming files that closed after the cue;
#   a pose held while  5 to 20 mm/s within 0.1 s of the cue in the six
#   the hand travels   forming files (the fit is not perfect in motion).
# Settled is below 10 mm/s: eight times a held pose's typical speed, above
# every held take's median. The movement is 25 mm/s or faster: above what
# travelling does to a held pose, about a third of the slowest change's
# peak. Neither has been tried on recordings other than the ones they were
# read from.
POSE_MOVE_SPEED_MM_S = 25.0
POSE_SETTLE_SPEED_MM_S = 10.0
# The day the two were read (above): a recording made after it was not among
# those they were read from (the README says which these are).
THRESHOLDS_READ_ON = "2026-10-05"
START_SUSTAIN_S = 0.1      # at the start speed or faster for this long: a movement
CUE_STILL_S = 0.1          # a frame that fast within this of the cue: changing already

VALID = "valid"
QUESTIONABLE = "questionable"
INVALID = "invalid"

NOTE_MOVING_AT_CUE = ("the pose was already changing at the cue ({speed:.0f} "
                      "mm/s within {window:g} s of it): the change began "
                      "before the cue, so its start cannot be timed")
NOTE_NO_MOVEMENT = ("no change of pose after the cue (fastest frame "
                    "{peak:.0f} mm/s)")
NOTE_NO_START = ("no change of pose lasting {sustain:g} s at {speed:g} mm/s "
                 "or faster after the cue")
NOTE_NOT_SEEN_AT_CUE = ("no frame of the hand within {window:g} s of the cue: "
                        "whether its pose was still cannot be checked")
NOTE_EARLIER = ("the pose moved before its main movement (at {at:.2f} s, for "
                "{length:.2f} s): the main movement is timed")
NOTE_HAND_MOVING = ("the hand as a whole was moving over the camera at the "
                    "cue ({speed:.0f} mm/s) with its pose still")
NOTE_POSE_CHANGE = "the pose at the end is {change:.0f} mm RMS from the pose at the cue"


def start_method(start_speed_mm_s: float = POSE_MOVE_SPEED_MM_S,
                 start_sustain_s: float = START_SUSTAIN_S,
                 step_s: float = SPEED_STEP_S,
                 palm_joints: Optional[Sequence[str]] = None) -> str:
    """The movement-start rule in words, with the numbers it was run with."""
    joints = PALM_JOINTS if palm_joints is None else palm_joints
    return (f"first frame of the stretch, {start_sustain_s:g} s or longer, "
            f"of pose speed at or above {start_speed_mm_s:g} mm/s that holds "
            f"the fastest frame after the cue (pose speed: mean over the 26 "
            f"joints, centred difference over {step_s:g} s, after the later "
            f"frame is laid on the earlier by the rigid fit of the palm's "
            f"{len(joints)} joints)")


def settled_method(settle_speed_mm_s: float = POSE_SETTLE_SPEED_MM_S,
                   settle_window_s: float = SETTLE_WINDOW_S,
                   step_s: float = SPEED_STEP_S) -> str:
    """The settle rule in words, with the numbers it was run with."""
    return (f"first frame after the movement's fastest frame from which the "
            f"pose speed (as for the start, centred difference over "
            f"{step_s:g} s) stays below {settle_speed_mm_s:g} mm/s for "
            f"{settle_window_s:g} s")


def laid_on(pos: np.ndarray, ref: np.ndarray,
            index: Optional[Sequence[int]] = None) -> np.ndarray:
    """The hand `pos` (26 x 3) moved onto the hand `ref` by the rotation and
    translation that best fit its palm joints to `ref`'s (least squares, no
    scaling, no mirroring): where its joints would be had the hand as a
    whole not moved between the two. `index`: the joints to fit on, when
    they are not `PALM_INDEX`."""
    index = PALM_INDEX if index is None else list(index)
    r, _s, move = umeyama(pos[index], ref[index], with_scale=False)
    return (r @ pos.T).T + move


def pose_distance_mm(pos, ref, index: Optional[Sequence[int]] = None
                     ) -> float:
    """How different two poses are: the RMS distance over the 26 joints, in
    mm, once the first hand is `laid_on` the second."""
    a, b = np.asarray(pos, dtype=float), np.asarray(ref, dtype=float)
    return float(np.sqrt(((laid_on(a, b, index) - b) ** 2)
                         .sum(axis=1).mean())) * 1000.0


def pose_speeds(t: np.ndarray, pos: Sequence[np.ndarray],
                step_s: float = SPEED_STEP_S,
                index: Optional[Sequence[int]] = None) -> np.ndarray:
    """The mean joint speed in mm/s of every frame with the whole hand's
    movement taken out.

    As `centred_speeds`: the two frames are the earliest and the latest
    within half a step of the frame (at an end of the recording, the frames
    there are on that side), NaN where there is no such pair. But the later
    of the two is first `laid_on` the earlier, so a hand carried across the
    camera with its pose held reads near zero and a hand closing in place
    reads what its fingers did.
    """
    half = float(step_s) / 2.0
    speeds = np.full(len(t), np.nan)
    for k in range(len(t)):
        a = int(np.searchsorted(t, t[k] - half, side="left"))
        b = int(np.searchsorted(t, t[k] + half, side="right")) - 1
        if (b <= a or t[b] - t[a] <= 0 or pos[a].shape != (26, 3)
                or pos[b].shape != (26, 3)):
            continue
        step = float(np.linalg.norm(laid_on(pos[b], pos[a], index) - pos[a],
                                    axis=1).mean())
        speeds[k] = step / (t[b] - t[a]) * 1000.0
    return speeds


@dataclass
class Timing:
    """One cued change of pose, as `transition_timing` reads it.

    Every `*_time_s` is in seconds from the first frame of the hand in the
    recording, the recording's own clock; `first_frame_time` is that frame
    on the wall clock, so a reader can find the rows. The three durations
    are differences of those times:

      reaction_time_s        movement start minus cue
      transition_duration_s  settled minus movement start
      cue_to_settled_s       settled minus cue

    Every speed is the pose speed (`pose_speeds`) except the two `hand_*`,
    which are the speed over the camera (`centred_speeds`), the hand as a
    whole included. `pose_change_mm` is how far the pose at the settle (at
    the end of the frames, with no settle) is from the pose at the cue
    (`pose_distance_mm`): the size of the change that was timed, which a
    duration read off speed thresholds depends on.

    `validity` is VALID for a pose that was still at the cue, changed in
    one sustained movement after it and settled; QUESTIONABLE for one that
    did that and also moved before its main movement or again after
    settling (the main movement and its first settle are timed, the note
    says what else happened); INVALID otherwise, with the reason in `notes`
    and no `transition_duration_s`. `forming_lost_count` is how many times
    the hand was lost while forming the grasp (`FORMING_LOST_KEY`): above
    0 the recording is INVALID and nothing is read off it.
    """

    first_frame_time: Optional[float] = None
    cue_time_s: Optional[float] = None
    speed_at_cue_mm_s: Optional[float] = None
    hand_speed_at_cue_mm_s: Optional[float] = None
    movement_start_time_s: Optional[float] = None
    settled_time_s: Optional[float] = None
    reaction_time_s: Optional[float] = None
    transition_duration_s: Optional[float] = None
    cue_to_settled_s: Optional[float] = None
    peak_speed_mm_s: Optional[float] = None
    peak_time_s: Optional[float] = None
    hand_peak_speed_mm_s: Optional[float] = None
    pose_change_mm: Optional[float] = None
    last_move_time_s: Optional[float] = None
    moved_again_time_s: Optional[float] = None
    earlier_move_time_s: Optional[float] = None
    end_time_s: Optional[float] = None      # where the analysis stopped reading
    frames: int = 0
    hand_ids: List[int] = field(default_factory=list)   # the tracker ids measured
    validity: str = INVALID
    notes: str = ""
    movement_start_method: str = ""
    settled_method: str = ""
    forming_lost_count: int = 0


def movement_start_index(t: np.ndarray, speed: np.ndarray, t0: float,
                         start_speed_mm_s: float = POSE_MOVE_SPEED_MM_S,
                         start_sustain_s: float = START_SUSTAIN_S,
                         gap_s: float = LOSS_GAP_S) -> Optional[int]:
    """Index of the first frame at or after `t0` from which the speed stays
    at or above `start_speed_mm_s` for `start_sustain_s`, or None.

    "Stays" as the settle has it: every frame of the run is that fast and
    no two of them are more than `gap_s` apart, with half a frame of slack
    on the length. A twitch shorter than the dwell is not a start.
    """
    runs = sustained_runs(t, speed, t0, start_speed_mm_s, start_sustain_s,
                          gap_s)
    return runs[0][0] if runs else None


def sustained_runs(t: np.ndarray, speed: np.ndarray, t0: float,
                   start_speed_mm_s: float = POSE_MOVE_SPEED_MM_S,
                   start_sustain_s: float = START_SUSTAIN_S,
                   gap_s: float = LOSS_GAP_S) -> List[Tuple[int, int]]:
    """(first, last) index of every stretch of frames, from `t0` on, whose
    speed stays at or above `start_speed_mm_s` for `start_sustain_s` or
    longer, in time order.

    Every frame of a stretch is that fast (NaN never is) and no two of them
    are more than `gap_s` apart; half a frame of slack on the length, so
    nine frames at 90 Hz are 0.1 s. A stretch that is under way at `t0`
    starts at the first frame at or after it.
    """
    n = len(t)
    if n < 2:
        return []
    fast = speed >= start_speed_mm_s                  # NaN never is
    slack = 0.5 * float(np.median(np.diff(t)))
    runs = []
    i = int(np.searchsorted(t, t0, side="left"))
    while i < n:
        if not fast[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fast[j + 1] and t[j + 1] - t[j] <= gap_s:
            j += 1
        if t[j] - t[i] >= start_sustain_s - slack:
            runs.append((i, j))
        i = j + 1
    return runs


def _settle_after(t: np.ndarray, speed: np.ndarray, peak: int,
                  settle_speed_mm_s: float, window_s: float,
                  gap_s: float) -> Optional[int]:
    """Index of the first frame after `peak` from which the speed stays
    below `settle_speed_mm_s` for `window_s` (no hole over `gap_s`, half a
    frame of slack), or None: `_search`'s quiet window, from a given peak."""
    quiet = speed < settle_speed_mm_s                 # NaN is never quiet
    slack = 0.5 * float(np.median(np.diff(t)))
    n = len(t)
    i = peak + 1
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        if window_s - slack <= 0:
            return i
        j = i
        while j + 1 < n and quiet[j + 1] and t[j + 1] - t[j] <= gap_s:
            j += 1
            if t[j] - t[i] >= window_s - slack:
                return i
        if j + 1 >= n:
            return None                  # quiet to the end, but not for long enough
        i = j + 1
    return None


def transition_timing(frames: Sequence[dict], cue_time: float,
                      t_end: Optional[float] = None,
                      start_speed_mm_s: float = POSE_MOVE_SPEED_MM_S,
                      start_sustain_s: float = START_SUSTAIN_S,
                      settle_speed_mm_s: float = POSE_SETTLE_SPEED_MM_S,
                      settle_window_s: float = SETTLE_WINDOW_S,
                      cue_still_s: float = CUE_STILL_S,
                      step_s: float = SPEED_STEP_S,
                      gap_s: float = LOSS_GAP_S,
                      palm_joints: Optional[Sequence[str]] = None,
                      forming_lost: int = 0) -> Timing:
    """The start of a cued change of pose, its settle and the times between.

    `frames` are one hand's lines, from before the cue when the recording
    has them (a forming file starts at the cue) to the end of the hold;
    with `t_end`, lines after it do not count. `cue_time` is the cue on the
    frames' clock (`frame_time`). Everything is decided on the pose speed
    (`pose_speeds`) of all of them, so a frame just after the cue is
    measured against the frames just before it when there are any.

      still at the cue  no frame at `start_speed_mm_s` or faster within
                        `cue_still_s` of the cue, before or after. A hand
                        cannot answer a beep in less than that, so a pose
                        changing that fast there began to change before the
                        cue, and neither its start nor its length can be
                        read from a recording of what came after.
      the movement      the stretch of frames at `start_speed_mm_s` or
                        faster, `start_sustain_s` or longer, that holds the
                        fastest such frame after the cue (`sustained_runs`):
                        the hand's main change of pose. Its first frame is
                        the movement start.
      settled           after the movement's fastest frame, the first frame
                        from which the pose speed stays below
                        `settle_speed_mm_s` for `settle_window_s`.

    Anchored on the main movement, because a pose can stir and come back
    before the hand commits to the change (the take this was written on did,
    0.3 s after its cue, while the hand was carried sideways): such an
    earlier stretch is not the start, and the take that has one is
    QUESTIONABLE with the note saying when. So is a take whose pose moves
    again after settling (a grip adjusted): its first settle is timed.

    A pose that was changing at the cue still gets its `cue_to_settled_s`
    when it settled, because that is a time between two things that were
    both recorded; it gets no start, no reaction time and no
    `transition_duration_s`. The cue is the moment the beep was started on
    the recording computer: the delay until the sound is heard is not
    measured, and it is part of `reaction_time_s`, never of
    `transition_duration_s`.

    `palm_joints`: the joints the hand is laid on for the pose speed, when
    they are not `PALM_JOINTS` (to see how much the choice matters).

    A recording in which the hand was lost while the grasp was formed (its
    lines marked `FORMING_LOST_KEY`, or `forming_lost` above 0 from the
    take's meta) is INVALID with `NOTE_LOST_WHILE_FORMING` and is not read:
    no start, no settle, no duration, no speeds; only which lines it was
    (`frames`, `hand_ids`, the first frame, the cue and the end).
    """
    index = (None if palm_joints is None
             else [JOINT_INDEX[n] for n in palm_joints])
    out = Timing(
        movement_start_method=start_method(start_speed_mm_s, start_sustain_s,
                                           step_s, palm_joints),
        settled_method=settled_method(settle_speed_mm_s, settle_window_s,
                                      step_s))
    out.forming_lost_count = max(int(forming_lost or 0),
                                 forming_lost_count(frames))
    rows = sorted((r for r in frames if is_tracked(r)
                   and (t_end is None or frame_time(r) <= t_end)),
                  key=frame_time)
    out.frames = len(rows)
    if len(rows) < 2:
        out.notes = (lost_note(out.forming_lost_count)
                     if out.forming_lost_count else NOTE_NO_FRAMES)
        return out
    t = np.array([frame_time(r) for r in rows])
    zero = float(t[0])
    out.first_frame_time = zero
    out.cue_time_s = float(cue_time) - zero
    out.end_time_s = (float(t[-1]) if t_end is None else float(t_end)) - zero
    for r in rows:
        hid = r.get("hand_id")
        if hid is not None and int(hid) not in out.hand_ids:
            out.hand_ids.append(int(hid))
    if out.forming_lost_count:
        out.notes = lost_note(out.forming_lost_count)
        return out
    pos = [joint_positions(r) for r in rows]
    speed = pose_speeds(t, pos, step_s, index)
    hand = centred_speeds(t, pos, step_s)
    after = int(np.searchsorted(t, cue_time, side="left"))
    if after >= len(t) or np.all(np.isnan(speed[after:])):
        out.notes = NOTE_NO_FRAMES
        return out

    near = (t >= cue_time - cue_still_s) & (t <= cue_time + cue_still_s)
    seen_at_cue = bool(np.any(near & ~np.isnan(speed)))
    if seen_at_cue:
        out.speed_at_cue_mm_s = float(np.nanmax(speed[near]))
        if np.any(near & ~np.isnan(hand)):
            out.hand_speed_at_cue_mm_s = float(np.nanmax(hand[near]))
    moving_at_cue = (out.speed_at_cue_mm_s is not None
                     and out.speed_at_cue_mm_s >= start_speed_mm_s)

    runs = [] if moving_at_cue else sustained_runs(
        t, speed, cue_time, start_speed_mm_s, start_sustain_s, gap_s)
    main = max(runs, key=lambda r: float(np.nanmax(speed[r[0]:r[1] + 1]))
               ) if runs else None
    first = after if main is None else main[0]
    stretch = speed[first:] if main is None else speed[main[0]:main[1] + 1]
    peak = first + int(np.nanargmax(stretch))
    out.peak_speed_mm_s = float(speed[peak])
    out.peak_time_s = float(t[peak]) - zero
    moving = np.flatnonzero(speed[first:] >= start_speed_mm_s) + first
    if len(moving):
        out.last_move_time_s = float(t[moving[-1]]) - zero
    moved = out.peak_speed_mm_s >= start_speed_mm_s
    settled = _settle_after(t, speed, peak, settle_speed_mm_s,
                            settle_window_s, gap_s) if moved else None
    if settled is not None:
        out.settled_time_s = float(t[settled]) - zero
        out.cue_to_settled_s = out.settled_time_s - out.cue_time_s
        again = moving[moving > settled]
        if len(again):
            out.moved_again_time_s = float(t[again[0]]) - zero
    if main is None and not moving_at_cue:
        # a twitch shorter than the dwell is not a movement: nothing is
        # timed from it, and nothing settled after it
        settled = None
        out.settled_time_s = out.cue_to_settled_s = None
        out.moved_again_time_s = None
    last = len(t) - 1 if settled is None else settled
    out.pose_change_mm = pose_distance_mm(pos[last], pos[after], index)
    seen = ~np.isnan(hand[after:last + 1])
    if np.any(seen):
        out.hand_peak_speed_mm_s = float(np.nanmax(hand[after:last + 1]))

    notes: List[str] = []
    if (out.hand_speed_at_cue_mm_s is not None and not moving_at_cue
            and out.hand_speed_at_cue_mm_s >= MOVE_SPEED_MM_S):
        notes.append(NOTE_HAND_MOVING.format(speed=out.hand_speed_at_cue_mm_s))
    change = NOTE_POSE_CHANGE.format(change=out.pose_change_mm)

    def done(*reasons: str) -> Timing:
        out.notes = "; ".join(list(reasons) + notes)
        return out

    if not seen_at_cue:
        return done(NOTE_NOT_SEEN_AT_CUE.format(window=cue_still_s))
    if moving_at_cue:
        return done(NOTE_MOVING_AT_CUE.format(speed=out.speed_at_cue_mm_s,
                                              window=cue_still_s))
    if not moved:
        return done(NOTE_NO_MOVEMENT.format(peak=out.peak_speed_mm_s), change)
    if main is None:
        return done(NOTE_NO_START.format(sustain=start_sustain_s,
                                         speed=start_speed_mm_s), change)
    out.movement_start_time_s = float(t[main[0]]) - zero
    out.reaction_time_s = out.movement_start_time_s - out.cue_time_s
    if settled is None:
        return done(NOTE_NEVER_SETTLED)
    out.transition_duration_s = out.settled_time_s - out.movement_start_time_s
    doubts = []
    earlier = [r for r in runs if r[1] < main[0]]
    if earlier:
        out.earlier_move_time_s = float(t[earlier[0][0]]) - zero
        doubts.append(NOTE_EARLIER.format(
            at=out.earlier_move_time_s,
            length=float(t[earlier[0][1]] - t[earlier[0][0]])))
    if out.moved_again_time_s is not None:
        doubts.append(
            NOTE_MOVED_AGAIN.format(at=out.moved_again_time_s)
            + f" (last movement ended at {out.last_move_time_s:.2f} s): "
            "the first settle is timed, the grip was adjusted after it")
    out.validity = QUESTIONABLE if doubts else VALID
    return done(*doubts)


def eased_move_lines(travel_mm: float, move_s: float = 0.5, hz: float = 90.0,
                     before_s: float = 1.0, react_s: float = 0.3,
                     hold_s: float = 1.5, carry_mm: float = 0.0
                     ) -> Tuple[List[dict], float]:
    """(lines, cue time) of a synthetic change of pose whose length is known.

    A hand at rest; a cue `before_s` in; `react_s` later every joint that is
    not of the palm (15 of the 26) is carried `travel_mm` along one line in
    exactly `move_s`, eased in and out (smoothstep), the palm's joints
    staying where they are; then still for `hold_s`. With `carry_mm` the
    whole hand is also carried that far along another line over the same
    time, which the pose speed must not see. One line per frame at `hz`,
    with the fields the timing reads. `known_move_reading` runs the rule on
    it: the difference between its answer and `move_s` is what the rule
    does to a movement, measured, not argued.
    """
    rng = np.random.default_rng(11)
    base = rng.uniform(-0.05, 0.05, size=(26, 3)) + np.array([0.0, 0.25, 0.0])
    fingers = np.array([k not in PALM_INDEX for k in range(26)])
    cue = 1000.0 + float(before_s)
    lines = []
    for i in range(int(round((before_s + react_s + move_s + hold_s) * hz))):
        t = i / float(hz)
        w = min(1.0, max(0.0, (t - before_s - react_s) / move_s))
        w = w * w * (3.0 - 2.0 * w)
        pos = base.copy()
        pos[fingers, 0] += travel_mm / 1000.0 * w
        pos[:, 2] += carry_mm / 1000.0 * w
        lines.append({"capture_time": 1000.0 + t, "wall_time": 1000.0 + t,
                      "abs26": pos.tolist(), "hand_id": 1, "status": 1})
    return lines, cue


def known_move_reading(travel_mm: float, move_s: float = 0.5,
                       carry_mm: float = 0.0, **thresholds) -> Timing:
    """What `transition_timing` reads off `eased_move_lines`."""
    lines, cue = eased_move_lines(travel_mm, move_s, carry_mm=carry_mm)
    return transition_timing(lines, cue, **thresholds)


def forming_follow(meta: dict) -> Optional[Tuple[int, List[int], None]]:
    """The hand a coached take's forming file is measured on, from its meta:
    the last open hand OPEN HAND acquired (`coaching.acquired`) and the
    other hands the coaching saw. None for a meta without one."""
    coaching = meta.get("coaching") if isinstance(meta.get("coaching"),
                                                  dict) else {}
    acquired = coaching.get("acquired") or []
    if acquired and acquired[-1].get("hand_id") is not None:
        return (int(acquired[-1]["hand_id"]),
                [int(i) for i in coaching.get("other_hands") or []], None)
    return None


def measured_rows(rows: Sequence[dict], hand: Optional[str],
                  follow: Optional[Tuple[int, Sequence[int],
                                         Optional[Sequence[float]]]]
                  ) -> List[dict]:
    """The lines of a recording a change of pose is timed on.

    `follow` is (hand id, other hands' ids, the hand's palm in metres or
    None): that hand followed by id (`static_interval.follow_hand`), so a
    new id the tracker gives it is still the same hand and another hand in
    view is never measured. Without one, the tracker label with the most
    lines. Of those lines, only the ones the tracker fitted as `hand`: a
    skeleton fitted as the other hand is a mirrored model of it, and the
    jump between the two would read as a movement.
    """
    if follow is not None:
        start_id, others, palm = follow
        mine = [r for _i, r in follow_hand(rows, int(start_id), others or (),
                                           palm, prefer=hand).rows]
    else:
        label = operator_label(rows, hand)
        mine = [] if label is None else [r for _i, r in hand_rows(rows, label)]
    return [r for r in mine if hand is None or str(r.get("hand_side")) == hand]


def forming_timing(rows: Sequence[dict], meta: dict, **thresholds) -> Timing:
    """`transition_timing` of one coached take's forming file.

    The hand, the cue and the end are the recorder's own, as
    `scripts/leap/recompute_transition.py` reads them: the acquired hand
    followed by id on its lines labelled as the operator's hand, the cue
    `coaching.form_start`, the end the start of the hold (`take_start`).
    `thresholds` are `transition_timing`'s keyword arguments. A hand lost
    while forming is read from the file's lines and from the meta's
    `coaching.forming_lost_count`, either of them (`transition_timing`).
    """
    coaching = meta.get("coaching") if isinstance(meta.get("coaching"),
                                                  dict) else {}
    lost = max(int(coaching.get(FORMING_LOST_KEY) or 0),
               forming_lost_count(rows))
    measured = measured_rows(rows, meta.get("hand"), forming_follow(meta))
    if coaching.get("form_start") is None:
        out = Timing(movement_start_method=start_method(),
                     settled_method=settled_method(), frames=len(measured),
                     forming_lost_count=lost)
        out.notes = "no cue time in the take's meta (coaching.form_start)"
        return out
    t_end = meta.get("take_start")
    return transition_timing(measured, float(coaching["form_start"]),
                             None if t_end is None else float(t_end),
                             forming_lost=lost, **thresholds)


def recorded_timing(rows: Sequence[dict], meta: dict, **thresholds) -> Timing:
    """`transition_timing` of a take of `scripts/leap/record_transitions.py`.

    Its meta names the cue (`cue_wall_time`), the operator's hand (`hand`)
    and the hand that was held still at the cue (`start_hand_id`,
    `other_hand_ids`, `start_palm`); the recording runs from before the cue
    to the end of the hold, so nothing is cut off.
    """
    follow = None
    if meta.get("start_hand_id") is not None:
        follow = (int(meta["start_hand_id"]),
                  [int(i) for i in meta.get("other_hand_ids") or []],
                  meta.get("start_palm"))
    measured = measured_rows(rows, meta.get("hand"), follow)
    if meta.get("cue_wall_time") is None:
        out = Timing(movement_start_method=start_method(),
                     settled_method=settled_method(), frames=len(measured))
        out.notes = "no cue time in the take's meta (cue_wall_time)"
        return out
    return transition_timing(measured, float(meta["cue_wall_time"]), None,
                             **thresholds)
