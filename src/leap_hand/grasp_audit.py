"""What a kept grasp take holds, measured from its own lines.

The hand-in folder used to describe a take by one number, the lines in its
JSONL, and a reader took 450 lines for five seconds. That only holds for a
take with one hand in view. The camera writes one line per tracked hand per
tracking frame, so a take recorded with the operator's other hand beside
the module has two lines per frame (up to 900 in five seconds), and one in
which the tracker briefly fitted a second hand has a few more. The audit of
2026-10-05 found another hand in 42 of the 69 kept takes of the session
`recordings/protocol/grasps/20261001_101621_left`. So nothing here is
inferred from a count of lines:

  duration   the last minus the first `wall_time` of the labelled hand's
             lines (`recording_duration_s`), the clock the static interval
             and the summary frame are written in. Beside it the same span
             on the camera's own frame clock (`timestamp_us`, LeapC's
             microseconds: the interval clock inside one sensor, see
             `leap_hand.to_openxr.to_hand_frame`), and the recorder's own
             window, `take_end - take_start`. The three differ by a few
             hundredths of a second for reasons that are the pipeline's, not
             the hand's: `wall_time` is stamped when a line is written and
             LeapC's hands arrive in bursts, the first frame of a take is
             up to one frame period after the window opens, and a frame
             already queued when it opens was captured a little before it.
  frames     `unique_frame_count`, the distinct tracking frames in the file;
             `hand_frame_count`, the lines of the labelled hand (one per
             frame in which it was tracked); and `jsonl_row_count`, every
             line, which is the only one of the three that counts the other
             hand.

Whose lines. The labelled hand is the hand the files are named after, the
operator's. A take the coached recorder made names the tracker ids it
followed (`operator_hand_ids` in the meta, `leap_hand.static_interval.
follow_hand`): the hand's lines are the lines of those ids, whatever the
tracker called them. A take recorded before that has no ids, and the hand
is the tracker label `cam_hand.protocol_check.choose_camera_label` picks,
the rule the recorder itself measured that take with. `operator_lines` is
the one place that decides it, for the summary, the all-frames text files
and every check, so they cannot disagree.

Another hand in the file does not spoil a take: it is the operator's other
hand in view beside the module, or for a few frames a second model the
tracker fitted on the same hand. It is counted (`other_hand_frames`, the
fraction of the take's frames it appears in, and how far it was from the
labelled hand), and of its shape only one number is taken (below). What
does need saying is when the labelled hand's own
identity is in doubt, and `identity_flag` says it: the tracker changed the
hand's id during the take (it lost the hand and found it again), its lines
carry both labels (a skeleton fitted as the other hand is a mirrored model),
the summary frame carries the other label, or the "other hand" sat within
`leap_hand.static_interval.FOLLOW_RADIUS_M` of the labelled one, which is
one hand fitted twice rather than two hands.

How still "static" was. The static interval is the stillest two seconds of
a take, and nothing ever required it to be still: the recorder's gate has
no speed limit, only tracking. So each take is measured, twice, because a
hand can drift over the camera without changing shape:

  static_interval_mean_speed_mm_s   the window's own score, the mean speed
             of the 26 joints in the camera's frame (the joint path over the
             window divided by its time): the whole hand's movement.
  static_interval_shape_rms_mm      the shape alone: every frame of the
             window fitted rigidly (rotation and translation,
             `cam_hand.align.umeyama` without scale) onto the take's summary
             frame, and the RMS distance of the 26 joints after the fit;
             the median over the window, and beside it the largest.

A window that moved faster than `STATIC_MOVING_MM_S` gets a note, and so
does one whose shape strayed more than `SHAPE_CHANGE_MM` from the summary
frame. Both numbers are thresholds for a note, not for acceptance.

How big the tracker made the hand. LeapC fits the size of the hand as well
as its pose, and fits it anew: the same operator's index metacarpal is 58 to
75 mm long across the 69 takes of the 2026-10-01 session, and changes by a
few millimetres inside some takes. `index_metacarpal_mm` (the median over
the take's frames) and `index_metacarpal_range_mm` (largest minus smallest)
say so per take. Angles do not depend on it; millimetres do.

How much of a recording is the pose. A recording runs five seconds and the
pose is formed in it, more or less: in six of the 69 kept takes of the
2026-10-01 session the first 0.2 s is more than 10 mm RMS (26 joints, rigid
fit) from the summary frame's shape, a hand still closing or a fit still
settling; in two the last 0.2 s is. So the whole take is measured against
its summary frame, not only its stillest two seconds (`whole_take_shape`):
`take_shape_rms_max_mm` the furthest any frame is from it,
`frames_within_5mm_percent` the share of the frames within
`SHAPE_CHANGE_MM` of it, `shape_rms_start_mm` and `shape_rms_end_mm` the
mean over the first and the last `EDGE_S`. An end further than
`NOT_THE_POSE_MM` gets a note (`extent_notes`). The frame-by-frame files
hold every frame all the same; `static_interval_first_frame_id` and
`static_interval_last_frame_id` are the tracker frame numbers the static
interval runs between, so a reader can cut them.

Which side of the hand the camera saw. `view_angle_deg` of the meta is the
recorder's own figure for the summary frame, the angle between the palm's
normal and the line from the palm to the camera: 0 with the palm square to
the lens, above 90 with the back of the hand toward it. Past
`BACK_OF_HAND_DEG` the fingers are on the far side of the hand from the
camera, and the take says so. It is the tracker's fit that is measured: a
fit that has the hand the wrong way up reads the same, and nothing here can
tell the two apart. In the 2026-10-01 session an upside-down fit is likely
for some of them: six of the seven are takes of closed hands recorded
without the open-hand start, every other take of the same grasps was
fitted palm side, and two takes of one grasp made 12 s apart were fitted
opposite ways. So the note says both readings and leaves it open.

What shape the other hand was in. Where another hand is in view on at
least `OTHER_HAND_MIN_SHARE` of a take's frames, both hands are measured
once, at the line of each nearest the middle of the recorder's window
(`take_start`, `take_end`; the labelled hand's own first and last frame
when the meta has no window): the sum over the index, middle, ring and
little fingers of the flexion of their PROXIMAL, INTERMEDIATE and DISTAL
rows of the joint-frame table (`xr_hand.joint_frames.joint_table`, camera,
wrist reference: MCP + PIP + DIP of four fingers; the thumb is left out),
the other hand's minus the labelled hand's
(`other_hand_finger_curl_minus_own_deg`). It says how far the two hands
were curled alike, not whether they made the same grasp.

Pure: lines in, numbers out. No files, no SDK.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cam_hand import protocol_check as pc
from cam_hand.align import umeyama
from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX

from .static_interval import (FOLLOW_RADIUS_M, LOSS_GAP_S, is_tracked,
                              joint_positions, row_palm)
from .transition import frame_time, pose_speeds

FOLLOWED = "followed id"       # `hand_selection`: the recorder followed the hand
BY_LABEL = "tracker label"     # ... or the take predates that
# A take whose tracked data stops this far short of the recorder's own
# window lost the hand at one end of it, and says so in its notes.
SHORT_BY_S = 0.10
# Notes, not gates (module docstring): a static interval faster than this
# drifted (the settle speed of `leap_hand.transition`); a shape further than
# this from the summary frame changed; a palm turned further than this from
# the lens showed the camera the back of the hand.
STATIC_MOVING_MM_S = 20.0
SHAPE_CHANGE_MM = 5.0
BACK_OF_HAND_DEG = 90.0
# The first or the last EDGE_S of a take further than this (mm RMS) from the
# summary frame's shape: the recording holds the hand in another shape there.
NOT_THE_POSE_MM = 10.0
EDGE_S = 0.2
# How that last note begins, for a reader that sorts the notes by kind.
BACK_OF_HAND_NOTE = "the tracker's fit has the back of the hand toward the camera"
# The other hand's shape is measured only where it is in view on at least
# this share of a take's frames (module docstring), over these rows of the
# joint-frame table: MCP, PIP and DIP of the four fingers.
OTHER_HAND_MIN_SHARE = 0.5
CURL_FINGERS = ("INDEX", "MIDDLE", "RING", "LITTLE")
CURL_ROWS = tuple(f"{f}_{s}" for f in CURL_FINGERS
                  for s in ("PROXIMAL", "INTERMEDIATE", "DISTAL"))


# --- whose lines ------------------------------------------------------------
def operator_lines(lines: Sequence[dict], meta: Optional[dict], hand: str
                   ) -> Tuple[List[Tuple[int, dict]], str]:
    """((index in `lines`, line) of the labelled hand, in file order; how).

    `how` is `FOLLOWED` when the meta names the ids the recorder followed,
    `BY_LABEL` otherwise (module docstring). File order is time order, and
    it is the order the all-frames text file is written in.
    """
    ids = (meta or {}).get("operator_hand_ids")
    if isinstance(ids, list) and ids:
        wanted = {int(i) for i in ids}
        return ([(i, d) for i, d in enumerate(lines)
                 if d.get("hand_id") is not None
                 and int(d["hand_id"]) in wanted], FOLLOWED)
    counts: Dict[str, int] = {}
    for d in lines:
        counts[d.get("hand_side")] = counts.get(d.get("hand_side"), 0) + 1
    label, _note = pc.choose_camera_label(counts, hand)
    return ([(i, d) for i, d in enumerate(lines)
             if d.get("hand_side") == label], BY_LABEL)


def frame_key(index: int, line: dict):
    """What tells one tracking frame from the next: the camera's frame id,
    and for a line without one, the line itself."""
    fid = line.get("frame_id")
    return ("line", index) if fid is None else ("frame", int(fid))


def representative_line(lines: Sequence[dict], meta: Optional[dict],
                        mine: Sequence[Tuple[int, dict]]
                        ) -> Optional[Tuple[int, dict]]:
    """(index, line) of the take's summary frame, found again in its lines.

    The recorder stores the line's number in the file (`medoid_line`) and
    its `wall_time` (`medoid_wall_time`). The numbered line is the one when
    it exists and carries that time; otherwise the labelled hand's line with
    that time (a meta written before the number was stored). None when the
    meta names neither, or names a frame the file does not hold.
    """
    meta = meta or {}
    t = meta.get("medoid_wall_time")
    n = meta.get("medoid_line")
    if isinstance(n, int) and not isinstance(n, bool) and 0 <= n < len(lines):
        try:
            same = t is None or abs(float(lines[n].get("wall_time"))
                                    - float(t)) <= 1e-6
        except (TypeError, ValueError):
            same = False
        if same:
            return n, lines[n]
    if t is None:
        return None
    for i, d in mine:
        try:
            if abs(float(d.get("wall_time")) - float(t)) <= 1e-6:
                return i, d
        except (TypeError, ValueError):
            continue
    return None


# --- one take ---------------------------------------------------------------
@dataclass
class TakeFacts:
    """One take, as `take_facts` measured it. Times in seconds; a value the
    lines cannot give is None, never a guess."""

    jsonl_row_count: int = 0
    unique_frame_count: int = 0
    hand_frame_count: int = 0
    first_wall_time: Optional[float] = None
    last_wall_time: Optional[float] = None
    recording_duration_s: Optional[float] = None
    camera_clock_duration_s: Optional[float] = None
    recorder_window_s: Optional[float] = None
    frame_rate_hz: Optional[float] = None
    largest_gap_s: Optional[float] = None
    # the static interval and the summary frame, from the first line of the
    # labelled hand
    static_interval_start_s: Optional[float] = None
    static_interval_end_s: Optional[float] = None
    static_interval_duration_s: Optional[float] = None
    representative_frame_index: Optional[int] = None    # line in the JSONL, from 0
    representative_frame_id: Optional[int] = None       # the camera's frame id
    representative_frame_time_s: Optional[float] = None
    representative_label: Optional[str] = None          # the tracker's label on it
    # how still the static interval was: in the camera's frame, and in shape
    static_interval_mean_speed_mm_s: Optional[float] = None
    # the same window's pose speed (`transition.pose_speeds`, the whole
    # hand's movement taken out), its median: how fast a held pose moves
    static_interval_pose_speed_mm_s: Optional[float] = None
    static_interval_shape_rms_mm: Optional[float] = None        # median over the window
    static_interval_shape_rms_max_mm: Optional[float] = None
    # the tracker's frame numbers the static interval runs between
    static_interval_first_frame_id: Optional[int] = None
    static_interval_last_frame_id: Optional[int] = None
    # the whole take's shape against the summary frame's
    take_shape_rms_max_mm: Optional[float] = None
    frames_within_5mm_percent: Optional[float] = None
    shape_frames: int = 0                   # the lines it was measured on
    frames_over_5mm: int = 0                # of them, further than SHAPE_CHANGE_MM
    frames_over_10mm: int = 0               # and further than NOT_THE_POSE_MM
    shape_rms_start_mm: Optional[float] = None                  # mean, first EDGE_S
    shape_rms_end_mm: Optional[float] = None                    # mean, last EDGE_S
    # how big the tracker made the hand
    index_metacarpal_mm: Optional[float] = None                 # median over the take
    index_metacarpal_range_mm: Optional[float] = None
    # the other hand
    other_hand_detected: bool = False
    other_hand_frames: int = 0
    other_hand_frame_fraction: float = 0.0
    other_hand_labels: Dict[str, int] = field(default_factory=dict)
    other_hand_median_distance_cm: Optional[float] = None
    # the four fingers' MCP + PIP + DIP flexion of each hand at the middle
    # of the recording, and the other's minus the labelled hand's (module
    # docstring); None where the other hand is in view on fewer than
    # OTHER_HAND_MIN_SHARE of the frames
    own_finger_curl_deg: Optional[float] = None
    other_finger_curl_deg: Optional[float] = None
    other_hand_finger_curl_minus_own_deg: Optional[float] = None
    # the labelled hand's identity
    hand_selection: str = ""
    hand_ids: List[int] = field(default_factory=list)
    tracker_id_changes: int = 0
    tracker_labels: Dict[str, int] = field(default_factory=dict)
    identity_flag: str = ""
    # what is unusual about the take's timing (a short span, a hole), about
    # the pose it holds (a static interval that moved, a shape that changed,
    # the back of the hand to the camera), about how much of it is the pose
    # (an end in another shape), and then everything a reader should know:
    # those, the identity flag, a summary frame not found
    timing_notes: List[str] = field(default_factory=list)
    pose_notes: List[str] = field(default_factory=list)
    extent_notes: List[str] = field(default_factory=list)
    # what the take's infrared picture shows in or near the hand, read by
    # eye (`leap_hand.grasp_handin` adds them from the still notes: the
    # lines of a take cannot say it)
    picture_notes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def abnormal(self) -> bool:
        """Something a reader should know before using the take."""
        return bool(self.notes)


def _span(values: Sequence[Optional[float]]) -> Optional[float]:
    got = [float(v) for v in values if v is not None]
    return (max(got) - min(got)) if got else None


def _camera_clock(rows: Sequence[dict]) -> List[Optional[float]]:
    """Each line's time on the camera's own clock, in seconds: LeapC's
    `timestamp_us`, else `capture_time`. None for a line with neither."""
    out: List[Optional[float]] = []
    for d in rows:
        if d.get("timestamp_us") is not None:
            out.append(float(d["timestamp_us"]) / 1e6)
        elif d.get("capture_time") is not None:
            out.append(float(d["capture_time"]))
        else:
            out.append(None)
    return out


def stillness(rows: Sequence[dict], reference: Optional[dict]
              ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(mean joint speed in mm/s, median shape RMS in mm, largest shape RMS)
    of a window's lines, in time order; the shape against `reference`, the
    take's summary frame (module docstring). None where it cannot be told:
    under two lines, or no reference."""
    if len(rows) < 2:
        return None, None, None
    pos = np.array([joint_positions(r) for r in rows], dtype=float)
    t = np.array([float(r["wall_time"]) for r in rows])
    span = float(t[-1] - t[0])
    path = float(np.linalg.norm(pos[1:] - pos[:-1], axis=2).mean(axis=1).sum())
    speed = None if span <= 0 else path / span * 1000.0
    if reference is None:
        return speed, None, None
    ref = np.asarray(joint_positions(reference), dtype=float)
    rms = []
    for q in pos:
        r, _s, tr = umeyama(q, ref, with_scale=False)
        fitted = (r @ q.T).T + tr
        rms.append(float(np.sqrt(((fitted - ref) ** 2).sum(axis=1).mean()))
                   * 1000.0)
    return speed, float(np.median(rms)), float(np.max(rms))


def shape_rms_mm(rows: Sequence[dict], reference: dict) -> np.ndarray:
    """For every line, how far its hand's shape is from `reference`'s: the
    RMS distance of the 26 joints in mm after the rotation and translation
    that fit it best (`cam_hand.align.umeyama` without scale), as
    `stillness` measures the static interval."""
    ref = np.asarray(joint_positions(reference), dtype=float)
    out = []
    for d in rows:
        q = np.asarray(joint_positions(d), dtype=float)
        r, _s, tr = umeyama(q, ref, with_scale=False)
        out.append(float(np.sqrt((((r @ q.T).T + tr - ref) ** 2)
                                 .sum(axis=1).mean())) * 1000.0)
    return np.array(out)


def whole_take_shape(rows: Sequence[dict], reference: Optional[dict]
                     ) -> Tuple[Optional[float], Optional[float],
                                Optional[float], Optional[float]]:
    """(largest shape distance in mm, percent of the lines within
    `SHAPE_CHANGE_MM`, mean over the first `EDGE_S`, mean over the last
    `EDGE_S`) of a take's lines, in time order, against `reference`, its
    summary frame (`shape_rms_mm`). None where it cannot be told."""
    if reference is None or not rows:
        return None, None, None, None
    d = shape_rms_mm(rows, reference)
    t = np.array([float(r["wall_time"]) for r in rows])
    return (float(d.max()), 100.0 * float((d <= SHAPE_CHANGE_MM).mean()),
            float(d[t <= t[0] + EDGE_S].mean()),
            float(d[t >= t[-1] - EDGE_S].mean()))


def finger_curl_deg(line: dict) -> Optional[float]:
    """The sum of the flexion of the `CURL_ROWS` (MCP, PIP and DIP of the
    index, middle, ring and little fingers) of one line's joint-frame table,
    in degrees; None for a line the table cannot be made of."""
    try:
        table = jf.joint_table(line, "camera", "wrist")
    except (ValueError, KeyError, TypeError, IndexError):
        return None
    by = {r["joint"]: r for r in table}
    return float(sum(float(by[n]["flex_deg"]) for n in CURL_ROWS))


def _nearest(rows: Sequence[dict], t: float) -> Optional[dict]:
    """The line of `rows` whose wall_time is nearest `t` (the earlier of two
    as near), or None."""
    best, gap = None, None
    for d in rows:
        try:
            g = abs(float(d["wall_time"]) - t)
        except (KeyError, TypeError, ValueError):
            continue
        if gap is None or g < gap:
            best, gap = d, g
    return best


def other_hand_curl(own: Sequence[dict], others: Sequence[dict],
                    meta: Optional[dict], share: float
                    ) -> Tuple[Optional[float], Optional[float],
                               Optional[float]]:
    """(the labelled hand's finger curl, the other hand's, other minus own)
    at the middle of the recording (module docstring); all None where the
    other hand is in view on less than `OTHER_HAND_MIN_SHARE` of the frames
    (`share`), or either line's table cannot be made."""
    if not own or not others or share < OTHER_HAND_MIN_SHARE:
        return None, None, None
    meta = meta or {}
    t0, t1 = meta.get("take_start"), meta.get("take_end")
    try:
        mid = (float(t0) + float(t1)) / 2.0
    except (TypeError, ValueError):
        walls = [float(d["wall_time"]) for d in own]
        mid = (min(walls) + max(walls)) / 2.0
    a, b = _nearest(own, mid), _nearest(others, mid)
    if a is None or b is None:
        return None, None, None
    mine, theirs = finger_curl_deg(a), finger_curl_deg(b)
    if mine is None or theirs is None:
        return None, None, None
    return mine, theirs, theirs - mine


def take_facts(lines: Sequence[dict], meta: Optional[dict], hand: str
               ) -> TakeFacts:
    """Measure one take from its lines and its meta (module docstring)."""
    meta = meta or {}
    f = TakeFacts(jsonl_row_count=len(lines))
    frames = {frame_key(i, d) for i, d in enumerate(lines)}
    f.unique_frame_count = len(frames)
    mine, f.hand_selection = operator_lines(lines, meta, hand)
    mine = [(i, d) for i, d in mine if is_tracked(d)]
    f.hand_frame_count = len(mine)
    rates = [float(d["framerate"]) for d in lines
             if d.get("framerate") and float(d["framerate"]) > 0]
    f.frame_rate_hz = float(np.median(rates)) if rates else None
    t0, t1 = meta.get("take_start"), meta.get("take_end")
    if t0 is not None and t1 is not None:
        f.recorder_window_s = float(t1) - float(t0)
    elif meta.get("duration_s") is not None:
        f.recorder_window_s = float(meta["duration_s"])
    if not mine:
        f.identity_flag = "no line of the labelled hand in the file"
        f.notes.append(f.identity_flag)
        return f

    rows = [d for _i, d in mine]
    walls = [float(d["wall_time"]) for d in rows]
    f.first_wall_time, f.last_wall_time = min(walls), max(walls)
    f.recording_duration_s = f.last_wall_time - f.first_wall_time
    clock = _camera_clock(rows)
    f.camera_clock_duration_s = (_span(clock) if None not in clock else None)
    steps = np.diff(np.sort(np.array(
        walls if None in clock else clock, dtype=float)))
    f.largest_gap_s = float(steps.max()) if len(steps) else 0.0

    interval = meta.get("static_interval")
    if (isinstance(interval, (list, tuple)) and len(interval) >= 2
            and None not in interval[:2]):
        a, b = float(interval[0]), float(interval[1])
        f.static_interval_start_s = a - f.first_wall_time
        f.static_interval_end_s = b - f.first_wall_time
        f.static_interval_duration_s = b - a
    found = representative_line(lines, meta, mine)
    if found is not None:
        index, row = found
        f.representative_frame_index = index
        fid = row.get("frame_id")
        f.representative_frame_id = None if fid is None else int(fid)
        f.representative_frame_time_s = (float(row["wall_time"])
                                         - f.first_wall_time)
        f.representative_label = str(row.get("hand_side"))
    if f.static_interval_start_s is not None:
        # the lines the recorder chose the window over: the labelled hand's,
        # under the operator's own label
        a = f.first_wall_time + f.static_interval_start_s
        b = f.first_wall_time + f.static_interval_end_s
        window = [d for d in rows if str(d.get("hand_side")) == hand
                  and a - 1e-6 <= float(d["wall_time"]) <= b + 1e-6
                  and (d.get("abs26") or d.get("joints"))]
        (f.static_interval_mean_speed_mm_s, f.static_interval_shape_rms_mm,
         f.static_interval_shape_rms_max_mm) = stillness(
            window, None if found is None else found[1])
        if len(window) >= 3:
            v = pose_speeds(
                np.array([frame_time(d) for d in window]),
                [np.asarray(joint_positions(d), dtype=float) for d in window])
            if np.any(~np.isnan(v)):
                f.static_interval_pose_speed_mm_s = float(np.nanmedian(v))
        ids = [d.get("frame_id") for d in window]
        if ids and ids[0] is not None and ids[-1] is not None:
            f.static_interval_first_frame_id = int(ids[0])
            f.static_interval_last_frame_id = int(ids[-1])
    shaped = [d for d in rows if str(d.get("hand_side")) == hand
              and (d.get("abs26") or d.get("joints"))]
    (f.take_shape_rms_max_mm, f.frames_within_5mm_percent,
     f.shape_rms_start_mm, f.shape_rms_end_mm) = whole_take_shape(
        shaped, None if found is None else found[1])
    if found is not None and shaped:
        d = shape_rms_mm(shaped, found[1])
        f.shape_frames = len(d)
        f.frames_over_5mm = int((d > SHAPE_CHANGE_MM).sum())
        f.frames_over_10mm = int((d > NOT_THE_POSE_MM).sum())
    sized = [np.asarray(joint_positions(d), dtype=float) for d in rows
             if d.get("abs26") or d.get("joints")]
    if sized:
        a, b = JOINT_INDEX["INDEX_METACARPAL"], JOINT_INDEX["INDEX_PROXIMAL"]
        lengths = np.array([np.linalg.norm(p[b] - p[a]) for p in sized]) * 1000.0
        f.index_metacarpal_mm = float(np.median(lengths))
        f.index_metacarpal_range_mm = float(lengths.max() - lengths.min())

    # the other hand: every tracked line that is not the labelled hand's
    own = {i for i, _d in mine}
    others = [(i, d) for i, d in enumerate(lines)
              if i not in own and is_tracked(d)]
    other_frames = {frame_key(i, d) for i, d in others}
    f.other_hand_frames = len(other_frames)
    f.other_hand_detected = bool(others)
    f.other_hand_frame_fraction = (len(other_frames) / len(frames)
                                   if frames else 0.0)
    for _i, d in others:
        side = str(d.get("hand_side"))
        f.other_hand_labels[side] = f.other_hand_labels.get(side, 0) + 1
    f.other_hand_labels = dict(sorted(f.other_hand_labels.items()))
    if others:
        mine_at = {frame_key(i, d): d for i, d in mine}
        dists = []
        for i, d in others:
            m = mine_at.get(frame_key(i, d))
            a, b = row_palm(d), (None if m is None else row_palm(m))
            if a is not None and b is not None:
                dists.append(float(np.linalg.norm(a - b)) * 100.0)
        f.other_hand_median_distance_cm = (float(np.median(dists))
                                           if dists else None)
    (f.own_finger_curl_deg, f.other_finger_curl_deg,
     f.other_hand_finger_curl_minus_own_deg) = other_hand_curl(
        rows, [d for _i, d in others], meta, f.other_hand_frame_fraction)

    # the labelled hand's own identity
    for d in rows:
        side = str(d.get("hand_side"))
        f.tracker_labels[side] = f.tracker_labels.get(side, 0) + 1
        hid = d.get("hand_id")
        if hid is not None and int(hid) not in f.hand_ids:
            f.hand_ids.append(int(hid))
    f.tracker_labels = dict(sorted(f.tracker_labels.items()))
    prev = None
    for d in rows:
        hid = d.get("hand_id")
        if hid is None:
            continue
        if prev is not None and int(hid) != prev:
            f.tracker_id_changes += 1
        prev = int(hid)

    flags = []
    if f.tracker_id_changes:
        flags.append(f"the tracker changed the hand's id "
                     f"{f.tracker_id_changes} time(s) during the take")
    if len(f.tracker_labels) > 1:
        flags.append("the tracker labelled the hand "
                     + ", ".join(f"{k} on {n} frame(s)"
                                 for k, n in f.tracker_labels.items()))
    if f.representative_label not in (None, hand):
        flags.append(f"the representative frame carries the tracker's "
                     f"{f.representative_label} label")
    if (f.other_hand_median_distance_cm is not None
            and f.other_hand_median_distance_cm < FOLLOW_RADIUS_M * 100.0):
        flags.append(f"a second hand model sat "
                     f"{f.other_hand_median_distance_cm:.1f} cm from the "
                     f"labelled hand on {f.other_hand_frames} frame(s): one "
                     "hand fitted twice, not two hands")
    f.identity_flag = "; ".join(flags)

    if (f.recorder_window_s is not None
            and f.recorder_window_s - f.recording_duration_s > SHORT_BY_S):
        f.timing_notes.append(
            f"tracked data spans {f.recording_duration_s:.2f} s of the "
            f"{f.recorder_window_s:.2f} s the recorder kept open: the hand "
            "was not tracked at one end of the take")
    if f.largest_gap_s is not None and f.largest_gap_s > LOSS_GAP_S:
        f.timing_notes.append(
            f"a hole of {f.largest_gap_s:.2f} s in the labelled hand's "
            "frames (hand lost and found again)")
    speed = f.static_interval_mean_speed_mm_s
    shape = f.static_interval_shape_rms_max_mm
    if speed is not None and speed > STATIC_MOVING_MM_S:
        f.pose_notes.append(
            f"the hand was moving during its stillest "
            f"{f.static_interval_duration_s:g} s (mean joint speed "
            f"{speed:.0f} mm/s)"
            + ("" if shape is None or shape > SHAPE_CHANGE_MM else
               f"; its shape stayed within {shape:.1f} mm RMS of the "
               "representative frame"))
    if shape is not None and shape > SHAPE_CHANGE_MM:
        f.pose_notes.append(
            f"the hand's shape changed inside the static interval (up to "
            f"{shape:.1f} mm RMS from the representative frame)")
    view = meta.get("view_angle_deg")
    if view is not None and float(view) > BACK_OF_HAND_DEG:
        f.pose_notes.append(
            f"{BACK_OF_HAND_NOTE} (palm normal {float(view):.0f} degrees "
            "from the line to the lens): either the hand was held that way, "
            "its fingers on the far side where the tracker places them by "
            "its hand model, or the fit is upside down; which was not "
            "determined")
    if (f.shape_rms_start_mm is not None
            and f.shape_rms_start_mm > NOT_THE_POSE_MM):
        f.extent_notes.append(
            f"the hand was not yet in the representative shape when the "
            f"recording began (first {EDGE_S:g} s: "
            f"{f.shape_rms_start_mm:.0f} mm RMS from it)")
    if (f.shape_rms_end_mm is not None
            and f.shape_rms_end_mm > NOT_THE_POSE_MM):
        f.extent_notes.append(
            f"the hand had left the representative shape by the end of the "
            f"recording (last {EDGE_S:g} s: {f.shape_rms_end_mm:.0f} mm RMS "
            "from it)")
    f.notes += f.timing_notes + f.pose_notes + f.extent_notes
    if f.identity_flag:
        f.notes.append(f.identity_flag)
    if found is None:
        named = (meta.get("medoid_wall_time") is not None
                 or meta.get("medoid_line") is not None)
        f.notes.append("the meta's summary frame was not found in the file"
                       if named else "the take's meta names no summary frame")
    return f


# --- across takes -----------------------------------------------------------
def spread(values: Sequence[Optional[float]]
           ) -> Optional[Tuple[float, float, float]]:
    """(minimum, median, maximum) of the values that are there, or None."""
    got = sorted(float(v) for v in values if v is not None)
    if not got:
        return None
    return got[0], float(np.median(got)), got[-1]
