"""Is the hand in this take actually holding the pose the take is labelled with?

Written after the session of 2026-09-17 (`recordings/sync_coached_20260917`,
36 takes): 9 of them hold a pose that is not the one in the filename, and
BOTH sensors agree on the wrong pose. `fist_left_take1` is an open palm,
`index_point_left_take1..3` are fists, `thumbs_up_left_take1..2` are fists,
`peace_left_take1` is a thumbs up. Nothing in the recorder noticed, because
nothing in the recorder ever looked at the SHAPE of the hand — only at
whether one continuous hand id covered the take. The cause was the live
camera window never naming the pose (see `view_caption`), but a label the
operator can get wrong is a label that has to be checked.

What is measured, and why it is this and not something new: the curl of each
finger, tip-to-wrist over palm length, straight out of
`cam_hand.features.flexion_features` — the same number the per-DOF table in
`scripts/fuse_poses.py` prints, so a verdict here and a row there are the
same quantity and can be argued about together. One median per finger over
the whole take, per sensor.

The rule that keeps this from throwing away good takes:

    a finger is WRONG only when BOTH sensors are decisive and BOTH
    contradict the pose. One sensor disagreeing is a `warn`, never a
    failure.

That is not caution for its own sake; it is what the same session measured.
The glove has a dead zone at the pinch: all six pinch takes are real pinches
by the camera (thumb-index tip gap 0.10-0.35 palm lengths against 0.97 for
an open palm) while the glove reports an exact open palm, because the
stretch sensors cannot see thumb opposition at all. And the camera has its
own: on the right-hand peace takes the glove reads the ring finger curled
(0.65) and the camera reads it extended (1.62-1.66). Either sensor alone
would fail takes that are fine. Both of them wrong the same way is an
operator error.

So `pinch` is checked on the CAMERA ONLY, by the thumb-index gap, and never
failed on the glove; and a camera that is looking at the hand edge-on
(median viewing angle above `camera_view_max_deg`) or that barely saw it
casts no vote at all, which means nothing can fail and at most a `warn` is
raised.

`pinch` goes further: it can never reach `mismatch` at all. The gap is
measured, reported in the verdict and written to `<take>.meta.json` as
`pinch_camera_gap`, but it does not reject the take — because the camera is
the sensor whose pinch performance this whole dataset exists to measure, and
a set of pinch takes that the camera itself selected would make any later
"the camera sees the pinch" result circular. The same reasoning is why a
rejected attempt is MOVED to `rejected/` rather than deleted (see
`CoachedLeapSession._attempt`): two sensors under evaluation are acting as
the arbiter here, so every take they exclude has to stay on disk where it can
be counted and argued with.

Everything here is pure: values in, a verdict out. `read_take` is the one
function that touches a file, and it reads a take exactly the way
`fuse_poses.py` does — `FrameRecorder.load` then `frame_to_keypoints21` —
so the numbers in a verdict and the numbers in the report match to the
digit.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from cam_hand.features import FLEXION_NAMES, flexion_features, thumb_index_gap
from xr_hand.keypoints21 import frame_to_keypoints21
from xr_hand.recorder import FrameRecorder

from .protocol import median, row_view_angle_deg

# --- what a finger is doing --------------------------------------------------
EXTENDED = "extended"
CURLED = "curled"
AMBIGUOUS = "ambiguous"      # between the two bands: the sensor casts no vote
NO_DATA = "no data"          # this sensor did not see the take at all

# --- what a whole take is worth ----------------------------------------------
OK = "ok"                    # nothing contradicted the pose
WARN = "warn"                # one sensor contradicted it; the other did not
MISMATCH = "mismatch"        # both sensors contradicted it: the wrong pose
UNCHECKED = "unchecked"      # nothing could be judged (unknown pose, no data)

GLOVE = "glove"
CAMERA = "camera"

# The hand shape each pose is: thumb, index, middle, ring, pinky.
# None is "not checked" — `pinch` is the whole reason that exists, because the
# glove cannot see it and the fingers say nothing useful about it either
# (a fist and a pinch have almost the same finger curls; what differs is
# where the THUMB TIP is, which is `pinch_gap_max` below).
EXPECTED_FINGERS: Dict[str, Tuple[Optional[str], ...]] = {
    "open_palm": (EXTENDED, EXTENDED, EXTENDED, EXTENDED, EXTENDED),
    "fist": (CURLED, CURLED, CURLED, CURLED, CURLED),
    "index_point": (CURLED, EXTENDED, CURLED, CURLED, CURLED),
    "thumbs_up": (EXTENDED, CURLED, CURLED, CURLED, CURLED),
    "peace": (CURLED, EXTENDED, EXTENDED, CURLED, CURLED),
    "pinch": (None, None, None, None, None),
}

# What a hand of that shape is CALLED when it turns up where it was not asked
# for. Deliberately not `pose.replace("_", " ")`: "saw an OPEN HAND, expected
# OPEN PALM" would be an unreadable verdict, and the shape is what was seen
# while the pose is what was asked for.
SEEN_NAMES = {
    "open_palm": "OPEN HAND",
    "fist": "FIST",
    "index_point": "INDEX POINT",
    "thumbs_up": "THUMBS UP",
    "peace": "PEACE",
}
OPEN_HAND = "OPEN HAND"
CLOSED_HAND = "CLOSED HAND"
ANOTHER_SHAPE = "ANOTHER SHAPE"


@dataclass(frozen=True)
class PoseCheckParams:
    """Where the curl bands sit, per sensor and per finger.

    EMPIRICAL, like `cam_hand.fusion.GateParams`, and from the same kind of
    place: the reference open palm and fist of one real session
    (`recordings/sync_coached_20260917`, 36 takes, both hands), not
    calibrated constants. Every one is named, passed through and printed by
    `scripts/check_take_labels.py` so the next session can argue with it.

    A finger is `extended` above `mid + margin`, `curled` below
    `mid - margin`, and `ambiguous` in between — where the sensor says
    nothing and therefore cannot fail a take.

      mid      midway between that sensor's open palm and its fist:
                 glove   open 1.43 1.97 2.07 1.97 1.71
                         fist 0.95 0.66 0.62 0.66 0.76
                 camera  open 1.30 1.75 1.85 1.71 1.47
                         fist 1.20 0.97 0.88 0.90 0.93
               The camera's thumb mid is 1.26 rather than the arithmetic
               1.25: the observed boundary is curled <= 1.238 (a fist held
               under a `thumbs_up` label) against extended >= 1.283 (a true
               open palm), and 1.26 is the middle of THAT.
      margin   how far from the mid a reading has to be to count. Wide where
               the sensor separates the two states cleanly, narrow where it
               does not. Widening a margin can only ever lose a real
               mismatch; it cannot invent one, which is the direction to err
               in when a false failure means re-recording a good take.

    Observed separation on that session (worst case over every take whose
    true finger state is known):

      finger   glove curled<=  extended>=    camera curled<=  extended>=
      thumb          0.95         1.32              1.238        1.283
      index          0.70         1.97              1.012        1.723
      middle         0.82         2.07              1.028        1.783
      ring           0.86         1.97              1.259*       1.623*
      pinky          1.11         1.71              1.008        1.388

    * the camera's ring finger is the one DOF the two sensors genuinely
      disagree about (right-hand peace: glove 0.65, camera 1.62-1.66), which
      is why its margin is wide enough to abstain on both of those numbers
      rather than pick a side.

    The camera's thumb is the tight one: 0.045 of separation, against 0.37
    for the glove's. 0.02 of margin leaves about 0.002 either side, which is
    stated here rather than hidden because it is the single number a future
    session is most likely to have to move. It is safe only because a
    failure needs the GLOVE to contradict the pose too, and the glove's
    thumb separates by 0.37.
    """

    glove_mid: Tuple[float, float, float, float, float] = (1.19, 1.32, 1.35,
                                                           1.32, 1.24)
    glove_margin: Tuple[float, float, float, float, float] = (0.14, 0.30, 0.30,
                                                              0.30, 0.20)
    camera_mid: Tuple[float, float, float, float, float] = (1.26, 1.36, 1.37,
                                                            1.31, 1.20)
    camera_margin: Tuple[float, float, float, float, float] = (0.02, 0.30, 0.30,
                                                               0.30, 0.20)
    # Thumb tip to index tip over palm length. A real pinch measures 0.10-0.35
    # and an open palm 0.97, so 0.6 is the middle of a gap nothing sits in.
    pinch_gap_max: float = 0.6
    # Above this median viewing angle the camera is looking at the hand
    # edge-on and its fingertips are guesses, so it casts no vote. 65 is
    # deliberately looser than the fusion gate's 50: this gate only has to
    # exclude a camera that cannot see the shape at all, and every take of
    # the reference session sits at or below 58 degrees.
    camera_view_max_deg: float = 65.0
    min_camera_frames: int = 30
    min_glove_frames: int = 5

    def described(self) -> Dict[str, object]:
        return asdict(self)


DEFAULT_PARAMS = PoseCheckParams()


# --- one sensor's view of one take -------------------------------------------
@dataclass(frozen=True)
class SensorReading:
    """What one sensor measured over one take, reduced to medians.

    `view_angle_deg` is the camera's only; a glove file has no palm geometry
    in absolute space and `read_take` leaves it None, which is also how a
    glove is stopped from ever being silenced by the edge-on rule.
    """

    frames: int = 0
    curls: Optional[Tuple[float, float, float, float, float]] = None
    gap: Optional[float] = None
    view_angle_deg: Optional[float] = None
    # When each frame was captured, for `protocol.stream_health`. Kept out of
    # every serialised form: it is a few hundred floats and the health summary
    # is what anyone reads.
    times: Tuple[float, ...] = ()

    @property
    def present(self) -> bool:
        return self.frames > 0 and self.curls is not None


def reading_from_points(points: Sequence[Sequence[Sequence[float]]],
                        view_angles: Sequence[float] = (),
                        times: Sequence[float] = ()) -> SensorReading:
    """Per-frame 21-keypoint hands -> the medians the check reads.

    `flexion_features` and `thumb_index_gap` are imported, never re-derived:
    the verdict has to be the same quantity the per-DOF report prints or the
    two cannot be compared.
    """
    frames = [list(p) for p in points]
    if not frames:
        return SensorReading()
    curls = [flexion_features(p) for p in frames]
    gaps = [thumb_index_gap(p) for p in frames]
    angles = [a for a in view_angles if a is not None]
    return SensorReading(
        frames=len(frames),
        curls=tuple(median([c[i] for c in curls]) for i in range(5)),
        gap=median(gaps),
        view_angle_deg=median(angles) if angles else None,
        times=tuple(float(t) for t in times if t is not None),
    )


def read_take(path, hand: Optional[str] = None) -> SensorReading:
    """One recorded JSONL take -> a `SensorReading`. Glove or camera, alike.

    Both sensors write the same schema (`leap_hand.recorder` exists to make
    that true), so one reader covers both and the two sides of a take are
    measured by identical code. `hand` keeps only that chirality, for the
    uncoached files that hold two.
    """
    if path is None:
        return SensorReading()
    path = Path(path)
    if not path.is_file():
        return SensorReading()
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    points: List[list] = []
    angles: List[float] = []
    times: List[float] = []
    for row, (frame, wall) in zip(rows, FrameRecorder.load(path)):
        if hand is not None and frame.hand_side != hand:
            continue
        points.append(frame_to_keypoints21(frame))
        angle = row_view_angle_deg(row)
        if angle is not None:
            angles.append(angle)
        # `capture_time` is when the sensor had the frame; `wall_time` is when
        # the line was written, near-identical across a drained burst. Stream
        # health measured on the writer's clock would report a stall that was
        # the recorder's, not the glove's — so the capture clock wins wherever
        # the file has one.
        stamp = row.get("capture_time")
        times.append(float(stamp) if stamp is not None else float(wall))
    return reading_from_points(points, angles, times)


# --- classifying one finger --------------------------------------------------
def finger_state(curl: Optional[float], mid: float, margin: float) -> str:
    """`extended` / `curled` / `ambiguous` for one median curl."""
    if curl is None:
        return NO_DATA
    if curl > mid + margin:
        return EXTENDED
    if curl < mid - margin:
        return CURLED
    return AMBIGUOUS


def finger_states(reading: SensorReading, mids, margins) -> Tuple[str, ...]:
    if not reading.present:
        return (NO_DATA,) * 5
    return tuple(finger_state(reading.curls[i], mids[i], margins[i])
                 for i in range(5))


@dataclass(frozen=True)
class FingerCheck:
    """One finger of one take, both sensors, and what that is worth."""

    finger: str
    expected: Optional[str]
    glove: str
    camera: str
    glove_curl: Optional[float]
    camera_curl: Optional[float]
    verdict: str

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("glove_curl", "camera_curl"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d


@dataclass(frozen=True)
class PoseCheck:
    """The whole verdict on one take."""

    pose: str
    verdict: str
    description: str
    seen: str = ""
    fingers: Tuple[FingerCheck, ...] = ()
    glove_curls: Optional[Tuple[float, ...]] = None
    camera_curls: Optional[Tuple[float, ...]] = None
    camera_gap: Optional[float] = None
    camera_votes: bool = False
    camera_silent_because: str = ""
    glove_frames: int = 0
    camera_frames: int = 0
    camera_view_deg: Optional[float] = None

    @property
    def failed(self) -> bool:
        return self.verdict == MISMATCH

    def as_dict(self) -> dict:
        """The `pose_check` block of `<take>.meta.json`."""
        def curls(v):
            return None if v is None else [round(float(x), 3) for x in v]

        return {
            "pose": self.pose,
            "verdict": self.verdict,
            "description": self.description,
            "seen": self.seen,
            "glove_curls": curls(self.glove_curls),
            "camera_curls": curls(self.camera_curls),
            # Named for what it is FOR. It is reported and never enforced: see
            # the module docstring on why the camera may not select the pinch
            # takes it is later going to be scored on.
            "pinch_camera_gap": (None if self.camera_gap is None
                                 else round(float(self.camera_gap), 3)),
            "camera_votes": self.camera_votes,
            "camera_silent_because": self.camera_silent_because,
            "camera_view_angle_deg": (None if self.camera_view_deg is None
                                      else round(float(self.camera_view_deg), 1)),
            "glove_frames": self.glove_frames,
            "camera_frames": self.camera_frames,
            "fingers": [f.as_dict() for f in self.fingers],
        }


def camera_silence(camera: SensorReading, params: PoseCheckParams) -> str:
    """Why the camera casts no vote on this take, or "" if it does.

    A silenced camera cannot fail anything: the verdict rule needs both
    sensors, so an edge-on take can only ever come back `ok`, `warn` or
    `unchecked`. That is the point — a hand seen from the side has fingertips
    the tracker inferred rather than saw, and this check is not entitled to
    throw a take away on an inference.
    """
    if not camera.present:
        return "the camera recorded nothing"
    if camera.frames < params.min_camera_frames:
        return (f"the camera saw only {camera.frames} frame(s) "
                f"(need {params.min_camera_frames})")
    angle = camera.view_angle_deg
    if angle is not None and angle > params.camera_view_max_deg:
        return (f"the camera saw the hand edge-on ({angle:.0f} deg, "
                f"over {params.camera_view_max_deg:.0f})")
    return ""


def _named(shape: str) -> str:
    """`FIST` -> `a FIST`; `ANOTHER SHAPE` -> `ANOTHER SHAPE`."""
    if not shape or shape == ANOTHER_SHAPE:
        return ANOTHER_SHAPE
    return f"{'an' if shape[:1] in 'AEIOU' else 'a'} {shape}"


def _observed(states_g: Sequence[str], states_c: Sequence[str],
              camera_votes: bool) -> Tuple[Optional[str], ...]:
    """What the two sensors AGREE each finger was doing, None where they do not.

    The agreed state is the only thing a shape can honestly be named from:
    the whole module exists because either sensor alone is confidently wrong
    on some DOF.
    """
    out: List[Optional[str]] = []
    for i in range(5):
        g, c = states_g[i], states_c[i]
        if not camera_votes:
            out.append(g if g in (EXTENDED, CURLED) else None)
        elif g == c and g in (EXTENDED, CURLED):
            out.append(g)
        elif g in (EXTENDED, CURLED) and c not in (EXTENDED, CURLED):
            out.append(None)
        elif c in (EXTENDED, CURLED) and g not in (EXTENDED, CURLED):
            out.append(None)
        else:
            out.append(None)
    return tuple(out)


def name_shape(observed: Sequence[Optional[str]]) -> str:
    """The hand shape these agreed finger states are, as a name.

    Exact when one known pose fits every finger both sensors were sure
    about, coarse when several fit, and `ANOTHER SHAPE` when none does. The
    coarse answer is still the useful one in a caption the operator reads in
    a second: OPEN HAND versus CLOSED HAND is the distinction that says which
    way they got it wrong.
    """
    known = [(pose, want) for pose, want in EXPECTED_FINGERS.items()
             if any(w is not None for w in want)]
    fits = [pose for pose, want in known
            if all(o is None or want[i] is None or o == want[i]
                   for i, o in enumerate(observed))]
    sure = [o for o in observed if o is not None]
    if len(fits) == 1 and sure:
        return SEEN_NAMES.get(fits[0], fits[0].replace("_", " ").upper())
    if not sure:
        return ANOTHER_SHAPE
    if sum(1 for o in sure if o == EXTENDED) >= 4:
        return OPEN_HAND
    if sum(1 for o in sure if o == CURLED) >= 4:
        return CLOSED_HAND
    return ANOTHER_SHAPE


def _fingers_phrase(checks: Sequence[FingerCheck], which: str) -> str:
    """`index, middle, ring, pinky extended` — the fingers that were wrong."""
    groups: Dict[str, List[str]] = {}
    for c in checks:
        if c.verdict != which:
            continue
        seen = c.camera if c.camera in (EXTENDED, CURLED) else c.glove
        groups.setdefault(seen, []).append(c.finger)
    return "; ".join(f"{', '.join(names)} {state}"
                     for state, names in groups.items())


def _warn_phrase(checks: Sequence[FingerCheck], title: str) -> str:
    """`the camera alone disagrees: ring extended, thumb curled (PEACE ...)`.

    Grouped by which sensor it was, because that is the actionable half: the
    glove disagreeing about the thumb and the camera disagreeing about the
    ring finger are two different known blind spots, not one vague doubt.
    """
    by_sensor: Dict[str, List[str]] = {}
    for c in checks:
        if c.verdict != WARN:
            continue
        if c.glove in (EXTENDED, CURLED) and c.glove != c.expected:
            by_sensor.setdefault(GLOVE, []).append(f"{c.finger} {c.glove}")
        if c.camera in (EXTENDED, CURLED) and c.camera != c.expected:
            by_sensor.setdefault(CAMERA, []).append(f"{c.finger} {c.camera}")
    parts = []
    for sensor, bits in by_sensor.items():
        other = CAMERA if sensor == GLOVE else GLOVE
        parts.append(f"the {sensor} alone disagrees: {', '.join(bits)} "
                     f"(the {other} agrees with {title})")
    return "; ".join(parts)


def check_pose(pose: str, glove: SensorReading, camera: SensorReading,
               params: PoseCheckParams = DEFAULT_PARAMS) -> PoseCheck:
    """Did this take hold `pose`? The whole rule, in one function.

    Returns a verdict, the per-finger detail behind it, both sensors' median
    curls and one line of English. Nothing here reads a file or a clock, so
    a live session and a folder audit months later reach the same answer from
    the same numbers.
    """
    want = EXPECTED_FINGERS.get(str(pose))
    title = str(pose).replace("_", " ").upper()
    silent = camera_silence(camera, params)
    votes = not silent
    shared = dict(
        pose=str(pose),
        glove_curls=glove.curls, camera_curls=camera.curls,
        camera_gap=camera.gap, camera_votes=votes,
        camera_silent_because=silent,
        glove_frames=glove.frames, camera_frames=camera.frames,
        camera_view_deg=camera.view_angle_deg,
    )

    if want is None:
        return PoseCheck(
            verdict=UNCHECKED,
            description=f"{title} has no expected hand shape, so the hand was "
                        "not checked",
            **shared)

    states_g = finger_states(glove, params.glove_mid, params.glove_margin)
    states_c = finger_states(camera, params.camera_mid, params.camera_margin)
    glove_votes = (glove.present and glove.frames >= params.min_glove_frames)

    checks: List[FingerCheck] = []
    for i, finger in enumerate(FLEXION_NAMES):
        expected = want[i]
        g = states_g[i] if glove_votes else NO_DATA
        c = states_c[i] if votes else NO_DATA
        if expected is None:
            verdict = UNCHECKED
        else:
            g_bad = g in (EXTENDED, CURLED) and g != expected
            c_bad = c in (EXTENDED, CURLED) and c != expected
            # THE RULE: one sensor is never enough to throw a take away.
            verdict = MISMATCH if (g_bad and c_bad) else (
                WARN if (g_bad or c_bad) else OK)
        checks.append(FingerCheck(
            finger=finger, expected=expected, glove=g, camera=c,
            glove_curl=glove.curls[i] if glove.present else None,
            camera_curl=camera.curls[i] if camera.present else None,
            verdict=verdict))
    fingers = tuple(checks)
    seen = name_shape(_observed(states_g if glove_votes else (NO_DATA,) * 5,
                                states_c, votes))

    # `pinch` is the glove's blind spot, so it is judged on the camera alone
    # and on the one number that separates it: where the thumb tip is. And it
    # is judged only as far as a WARNING — the camera cannot be allowed to
    # choose which pinch takes survive to be scored on the camera.
    if all(w is None for w in want):
        if not votes:
            return PoseCheck(verdict=UNCHECKED, seen=seen, fingers=fingers,
                             description=f"{title} can only be judged by the "
                                         f"camera, and {silent}", **shared)
        gap = camera.gap
        if gap is None:
            return PoseCheck(verdict=UNCHECKED, seen=seen, fingers=fingers,
                             description=f"{title} needs the camera's "
                                         "thumb-index gap and there is none",
                             **shared)
        if gap > params.pinch_gap_max:
            return PoseCheck(
                verdict=WARN, seen=seen, fingers=fingers,
                description=(f"the camera saw the thumb and index "
                             f"{gap:.2f} palm lengths apart, expected "
                             f"{title} (under {params.pinch_gap_max:g}) — "
                             "reported, never rejected: the camera may not "
                             "pick the pinch takes it will be scored on"),
                **shared)
        return PoseCheck(
            verdict=OK, seen=seen, fingers=fingers,
            description=(f"the camera saw the thumb and index {gap:.2f} palm "
                         f"lengths apart — a {title}"), **shared)

    wrong = [c for c in fingers if c.verdict == MISMATCH]
    if wrong:
        return PoseCheck(
            verdict=MISMATCH, seen=seen, fingers=fingers,
            description=(f"both sensors saw {_named(seen)}, expected "
                         f"{title} ({_fingers_phrase(fingers, MISMATCH)})"),
            **shared)

    warned = [c for c in fingers if c.verdict == WARN]
    if warned:
        return PoseCheck(
            verdict=WARN, seen=seen, fingers=fingers,
            description=f"{_warn_phrase(fingers, title)}, so the take stands",
            **shared)

    if not glove_votes and not votes:
        return PoseCheck(verdict=UNCHECKED, seen=seen, fingers=fingers,
                         description="neither sensor could be read, so the "
                                     "hand was not checked", **shared)
    if not votes:
        return PoseCheck(verdict=OK, seen=seen, fingers=fingers,
                         description=f"the glove agrees with {title}; "
                                     f"{silent}", **shared)
    return PoseCheck(verdict=OK, seen=seen, fingers=fingers,
                     description=f"both sensors agree with {title}",
                     **shared)


def check_take(pose: str, glove_path, camera_path, hand: Optional[str] = None,
               params: PoseCheckParams = DEFAULT_PARAMS) -> PoseCheck:
    """`check_pose` straight off the two files a take wrote."""
    return check_pose(pose, read_take(glove_path, hand),
                      read_take(camera_path, hand), params)


def short_summary(check: PoseCheck) -> str:
    """`saw OPEN HAND, want FIST` — the caption line, for a window at 768 px."""
    return (f"saw {check.seen or ANOTHER_SHAPE}, want "
            f"{str(check.pose).replace('_', ' ').upper()}")
