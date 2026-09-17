"""Quality numbers for a Leap recording: detection, re-acquisitions, jitter.

These are the measurements the Phase 2 gate turns on (does the IR camera see
a hand inside the black StretchSense glove?), so the definitions matter more
than the code:

  detection rate   frames actually in the file / frames the file should hold.
      "Should hold" is the span times a rate, and *which* rate is the whole
      question. There are two candidates:

        cadence     the recording's own sustained pace: the 10th-percentile
                    gap between consecutive frames of that hand. The only
                    honest denominator for a throttled file — one written
                    with `--hz 5` must be measured against 5, not 90, or it
                    scores 6%.
        framerate   `event.framerate`, the rate LeapC reports it was tracking
                    at. The right denominator for a file that kept every
                    frame, because then every gap IS one tracking interval,
                    and the 10th percentile of them is not the pace: it is
                    the shortest gap the timestamp jitter happened to produce.

      `choose_rate` decides. If the file reports a framerate and its cadence
      is above `FRAMERATE_CADENCE_FLOOR` x that framerate, the recorder was
      keeping everything and the framerate is used; otherwise the cadence is.
      That threshold separates "same rate, jittered" from "throttled to a
      fraction of it", and nothing useful sits in between — a `--hz` worth
      passing is far below the tracking rate.

      This is not a detail. On the 2026-09-16 gate run a 90 Hz bare-hand file
      whose LeapC timestamps jittered read a 10th-percentile cadence of
      101 Hz; that inflated the denominator by 12% and pushed detection from
      about 89% down to 79% — under the plan's 80% threshold, on the very row
      every gloved condition is compared against. The rate actually used is
      carried on every row (`rate_hz`, `rate_source`) and named in the table
      footnote, because a detection rate whose denominator is not stated is
      not a measurement.

      Dropouts show up either way, as gaps far wider than the baseline, which
      is what this is measuring.

  re-acquisitions  how many times `hand_id` changed. The tracker gives a new
      id whenever it loses a hand and finds it again, so this counts recovered
      losses, while the detection rate counts unrecovered ones.

  jitter           the spread of the five fingertips over the steadiest two
      seconds in the file, in millimetres. Taking the steadiest window rather
      than the whole take keeps deliberate motion out of a noise figure.

  frame age        `leap.get_now() - event.timestamp` at receipt, recorded per
      frame by the live stream. Frame age, not end-to-end latency: it does
      not include exposure, our conversion, or the write. Absent (None) in
      mock and replayed recordings.

Reading a stat line: a gloved-hand run with a detection rate near 1.0, few
re-acquisitions and jitter close to the bare-hand run is Path A in the plan;
anything much worse is Path B.

**Per pose.** A take recorded under a `--schedule` carries `pose_plan` and
`pose_t` on every frame, and `analyse_poses` then measures each of the above
inside each scheduled pose window instead of over the whole take. That is the
difference between "the glove scored 62 %" and "the glove scored 96 % open and
11 % in a fist", and it is the only form in which the gloved and bare runs can
honestly be compared — the 2026-09-17 session lost tracking four times while
the hand was a fist and had no bare-hand fist anywhere to compare against.
A take with no `pose_plan` has no per-pose analysis and says so; nothing is
inferred from the condition name.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .protocol import (
    DEFAULT_BAND,
    IN_BAND,
    VIEW_ANGLE_MAX_DEG,
    Schedule,
    band_flag,
    longest_gap,
    parse_schedule,
    row_height_cm,
    row_view_angle_deg,
)
from .protocol import median as _median

# abs26 indices of the five fingertips (thumb, index, middle, ring, little).
TIP_IDX = (5, 10, 15, 20, 25)
JITTER_WINDOW_S = 2.0
MIN_JITTER_FRAMES = 5
MM = 1000.0

# A file whose cadence reaches this fraction of the reported tracking
# framerate kept every frame; below it, the recorder was throttling. See the
# module docstring — the gap between the two cases is an order of magnitude,
# so the exact value only has to be clear of the jitter, not tuned.
FRAMERATE_CADENCE_FLOOR = 0.85

RATE_FRAMERATE = "framerate"
RATE_CADENCE = "cadence"


def choose_rate(sample_hz: float, mean_framerate: float) -> tuple:
    """(rate, which) — the denominator the detection rate is measured against.

    `sample_hz` is the file's own cadence, `mean_framerate` what LeapC said it
    was tracking at. Returns the framerate when the file evidently kept every
    frame (its cadence is within `FRAMERATE_CADENCE_FLOOR` of the framerate,
    from either side — timestamp jitter pushes it *above* as readily as
    below), and the cadence otherwise.
    """
    if mean_framerate > 0 and sample_hz > FRAMERATE_CADENCE_FLOOR * mean_framerate:
        return mean_framerate, RATE_FRAMERATE
    return sample_hz, RATE_CADENCE


@dataclass
class HandStats:
    """One row of the stats table: one hand of one recording."""
    file: str
    hand_side: str
    frames: int
    span_s: float
    sample_hz: float
    expected_frames: int
    detection_rate: float
    reacquisitions: int
    mean_framerate: float
    jitter_mm: float
    frame_age_ms: Optional[float]
    pose: str = ""
    # The denominator behind `detection_rate`: the rate itself, and which of
    # the two candidates it is. Defaulted so older call sites still build a
    # row; `_hand_stats` always fills both.
    rate_hz: float = 0.0
    rate_source: str = RATE_CADENCE

    @property
    def ok(self) -> bool:
        """The plan's Path A thresholds: 80%+ detected, under 1 loss per 10 s."""
        losses_per_10s = self.reacquisitions / max(self.span_s / 10.0, 1e-9)
        return self.detection_rate >= 0.80 and losses_per_10s <= 1.0

    @property
    def rate_flag(self) -> str:
        """'*' when the tracking framerate was the denominator, else ' '."""
        return "*" if self.rate_source == RATE_FRAMERATE else " "


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..1). numpy-free, tiny inputs."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def frame_times(rows: Sequence[dict]) -> List[float]:
    """Seconds per frame, from the best clock the file has.

    Leap files carry `timestamp`, the LeapC clock in seconds — the time the
    camera saw the hand, free of any jitter our writer added. Its epoch is
    arbitrary, which is fine: only differences are ever used. Files from any
    other source (a glove `timestamp` is a tick counter, not seconds) fall
    back to `wall_time`.
    """
    if all(r.get("source") == "leap" and r.get("timestamp") is not None
           for r in rows):
        return [float(r["timestamp"]) for r in rows]
    return [float(r["wall_time"]) for r in rows]


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def fingertip_jitter(rows: List[dict], window_s: float = JITTER_WINDOW_S) -> float:
    """Fingertip spread over the steadiest `window_s`, in millimetres.

    Per tip: the root-sum-square of its per-axis standard deviation inside the
    window; averaged over the five tips; minimised over every window start.
    Returns 0.0 when the rows carry no `abs26` (nothing to measure).
    """
    usable = [r for r in rows if r.get("abs26")]
    if len(usable) < 2:
        return 0.0
    times = frame_times(usable)

    def window_jitter(lo: int, hi: int) -> float:
        per_tip = []
        for t in TIP_IDX:
            axes = []
            for axis in range(3):
                axes.append(_std([usable[k]["abs26"][t][axis] for k in range(lo, hi)]))
            per_tip.append(math.sqrt(sum(a * a for a in axes)))
        return sum(per_tip) / len(per_tip) * MM

    best = None
    for lo in range(len(usable)):
        hi = lo
        while hi < len(usable) and times[hi] - times[lo] <= window_s:
            hi += 1
        if hi - lo < MIN_JITTER_FRAMES:
            continue
        j = window_jitter(lo, hi)
        best = j if best is None else min(best, j)
    if best is None:                      # recording shorter than one window
        best = window_jitter(0, len(usable))
    return best


def _hand_stats(path: Path, hand_side: str, rows: List[dict]) -> HandStats:
    times = frame_times(rows)
    span = max(times) - min(times) if len(times) > 1 else 0.0

    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    baseline = _percentile(gaps, 0.10) if gaps else 0.0
    sample_hz = 1.0 / baseline if baseline > 0 else 0.0

    ids = [r.get("hand_id") for r in rows]
    reacquisitions = sum(1 for a, b in zip(ids, ids[1:]) if a != b and b is not None)

    rates = [float(r["framerate"]) for r in rows if r.get("framerate")]
    mean_framerate = sum(rates) / len(rates) if rates else 0.0
    rate_hz, rate_source = choose_rate(sample_hz, mean_framerate)
    expected = int(round(span * rate_hz)) + 1 if rate_hz else len(rows)
    detection = len(rows) / expected if expected else 1.0

    ages = [float(r["frame_age_us"]) for r in rows if r.get("frame_age_us") is not None]
    poses = {r.get("pose", "") for r in rows}

    return HandStats(
        file=path.name,
        hand_side=hand_side,
        frames=len(rows),
        span_s=span,
        sample_hz=sample_hz,
        expected_frames=expected,
        detection_rate=min(detection, 1.0),
        reacquisitions=reacquisitions,
        mean_framerate=mean_framerate,
        jitter_mm=fingertip_jitter(rows),
        frame_age_ms=(sum(ages) / len(ages) / 1000.0) if ages else None,
        pose=sorted(poses)[0] if poses else "",
        rate_hz=rate_hz,
        rate_source=rate_source,
    )


def read_rows(path: str | Path) -> List[dict]:
    """Every JSONL line of a recording, as raw dicts (extras included)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyse_file(path: str | Path) -> List[HandStats]:
    """One `HandStats` per hand side present in one recording."""
    path = Path(path)
    rows = read_rows(path)
    by_hand: Dict[str, List[dict]] = {}
    for r in rows:
        by_hand.setdefault(r.get("hand_side", "?"), []).append(r)
    return [_hand_stats(path, side, hand_rows)
            for side, hand_rows in sorted(by_hand.items()) if hand_rows]


def analyse_paths(paths: Sequence[str | Path]) -> List[HandStats]:
    """Stats for every .jsonl under each path (files or folders)."""
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])
    out: List[HandStats] = []
    for f in files:
        out.extend(analyse_file(f))
    return out


# --- per scheduled pose ------------------------------------------------------
@dataclass
class PoseStats:
    """One row of the per-pose table: one pose, one hand, one recording."""

    file: str
    condition: str
    hand_side: str
    pose: str
    frames: int
    scheduled_s: float
    detection_rate: float
    longest_loss_s: float
    reacquisitions: int
    median_height_cm: Optional[float] = None
    in_band_pct: Optional[float] = None
    facing_pct: Optional[float] = None
    median_view_deg: Optional[float] = None
    rate_hz: float = 0.0
    rate_source: str = RATE_CADENCE

    @property
    def rate_flag(self) -> str:
        return "*" if self.rate_source == RATE_FRAMERATE else " "


def schedule_of(rows: Sequence[dict]) -> Optional[Schedule]:
    """The schedule these rows were recorded under, or None.

    None means "this take cannot be read per pose", which is a fact about the
    file, not a reason to guess: a take recorded before schedules existed has
    no pose boundaries in it, and inventing them from the condition name would
    manufacture exactly the comparison this module exists to make honest.
    """
    plans = {r.get("pose_plan") for r in rows if r.get("pose_plan")}
    if len(plans) != 1:
        return None
    try:
        return parse_schedule(plans.pop())
    except ValueError:                           # pragma: no cover - bad file
        return None


def _in_window(t, window) -> bool:
    return t is not None and window.start <= float(t) < window.end


def _pose_stats(path: Path, condition: str, side: str, rows: List[dict],
                schedule: Schedule, pose: str, rate_hz: float,
                rate_source: str, band: Tuple[float, float]) -> PoseStats:
    windows = schedule.windows_for(pose)
    frames: List[dict] = []
    worst_loss = 0.0
    reacquisitions = 0
    for w in windows:
        here = sorted((r for r in rows if _in_window(r.get("pose_t"), w)),
                      key=lambda r: float(r["pose_t"]))
        frames.extend(here)
        # The window's own edges bound the gap, so a pose the tracker never
        # saw at all reports the whole window as one loss rather than zero.
        worst_loss = max(worst_loss,
                         longest_gap([float(r["pose_t"]) for r in here],
                                     w.start, w.end))
        ids = [r.get("hand_id") for r in here]
        # Only inside a window: the id may legitimately differ across two
        # windows of the same pose, with another pose held in between.
        reacquisitions += sum(1 for a, b in zip(ids, ids[1:])
                              if a != b and b is not None)

    scheduled = schedule.seconds_of(pose)
    expected = scheduled * rate_hz
    heights = [h for h in (row_height_cm(r) for r in frames) if h is not None]
    angles = [a for a in (row_view_angle_deg(r) for r in frames)
              if a is not None]
    return PoseStats(
        file=path.name,
        condition=condition,
        hand_side=side,
        pose=pose,
        frames=len(frames),
        scheduled_s=scheduled,
        detection_rate=min(len(frames) / expected, 1.0) if expected else 0.0,
        longest_loss_s=worst_loss,
        reacquisitions=reacquisitions,
        median_height_cm=_median(heights),
        in_band_pct=(100.0 * sum(1 for h in heights
                                 if band_flag(h, band) == IN_BAND) / len(heights)
                     if heights else None),
        facing_pct=(100.0 * sum(1 for a in angles if a < VIEW_ANGLE_MAX_DEG)
                    / len(angles) if angles else None),
        median_view_deg=_median(angles),
        rate_hz=rate_hz,
        rate_source=rate_source,
    )


def analyse_poses(path: str | Path, band: Tuple[float, float] = DEFAULT_BAND,
                  condition: str = "") -> List[PoseStats]:
    """One `PoseStats` per (hand, scheduled pose). Empty if unscheduled.

    The detection denominator is the one the whole-take row already chose
    (`choose_rate`), applied to the seconds the SCHEDULE allotted the pose —
    not to the span of the frames that happen to be there, which would score a
    pose the tracker saw for half a second at 100 %.
    """
    path = Path(path)
    rows = read_rows(path)
    schedule = schedule_of(rows)
    if schedule is None:
        return []
    by_hand: Dict[str, List[dict]] = {}
    for r in rows:
        by_hand.setdefault(r.get("hand_side", "?"), []).append(r)
    out: List[PoseStats] = []
    for side, hand_rows in sorted(by_hand.items()):
        base = _hand_stats(path, side, hand_rows)
        for pose in schedule.poses:
            out.append(_pose_stats(path, condition or path.parent.name, side,
                                   hand_rows, schedule, pose, base.rate_hz,
                                   base.rate_source, band))
    return out


_HEADER = (f"{'file':<38} {'hand':<5} {'pose':<10} {'n':>5} {'span':>6} "
           f"{'rate':>7} {'cad':>6} {'fps':>6} {'det%':>6} {'reacq':>6} "
           f"{'jit_mm':>7} {'age_ms':>7}")


def format_table(rows: Sequence[HandStats]) -> str:
    """The stats table, plus a totals line. Plain text, for terminal or file."""
    lines = [_HEADER, "-" * len(_HEADER)]
    for s in rows:
        age = "      -" if s.frame_age_ms is None else f"{s.frame_age_ms:7.1f}"
        lines.append(
            f"{s.file[:38]:<38} {s.hand_side:<5} {s.pose[:10]:<10} {s.frames:>5} "
            f"{s.span_s:>6.1f} {s.rate_hz:>6.1f}{s.rate_flag} "
            f"{s.sample_hz:>6.1f} {s.mean_framerate:>6.1f} "
            f"{100.0 * s.detection_rate:>6.1f} {s.reacquisitions:>6} "
            f"{s.jitter_mm:>7.2f} {age}"
        )
    if rows:
        n = len(rows)
        span = sum(s.span_s for s in rows)
        lines += [
            "-" * len(_HEADER),
            f"{n} hand-recordings   "
            f"mean detection {100.0 * sum(s.detection_rate for s in rows) / n:.1f}%   "
            f"mean tracking rate {sum(s.mean_framerate for s in rows) / n:.1f} Hz   "
            f"re-acquisitions {sum(s.reacquisitions for s in rows)} "
            f"({sum(s.reacquisitions for s in rows) / max(span / 10.0, 1e-9):.2f} per 10 s)   "
            f"mean jitter {sum(s.jitter_mm for s in rows) / n:.2f} mm",
            f"Path A thresholds met by {sum(1 for s in rows if s.ok)}/{n} "
            "(detection >= 80%, re-acquisitions <= 1 per 10 s)",
        ]
    else:
        lines.append("(no recordings)")
    lines += rate_footnote(rows)
    lines.append(
        "det% = frames present / span x rate; jit_mm = fingertip spread over "
        "the steadiest 2 s; age_ms = frame age at receipt."
    )
    return "\n".join(lines)


def rate_footnote(rows: Sequence[HandStats],
                  has_cadence_column: bool = True) -> List[str]:
    """The lines that say which denominator each row's det% used.

    Shared by this table and the gate report, because a detection rate is
    only readable next to the rate it was divided by. `has_cadence_column`
    is False for the gate table, which has no `cad` column to explain.
    """
    used = {s.rate_source for s in rows}
    tail = ("cad = that cadence (10th-percentile gap); "
            if has_cadence_column else "")
    lines = [
        "rate = the denominator of det%, marked * when it is the LeapC "
        "tracking framerate (the file kept every frame) and unmarked when it",
        "is the file's own cadence (a --hz throttled file); "
        f"{tail}fps = framerate LeapC reported.",
    ]
    if used == {RATE_FRAMERATE}:
        lines.append("Every row above used the tracking framerate: no file "
                     "here was throttled.")
    elif used == {RATE_CADENCE}:
        lines.append("Every row above used the file's own cadence: no file "
                     "here reported a framerate it kept up with.")
    return lines
