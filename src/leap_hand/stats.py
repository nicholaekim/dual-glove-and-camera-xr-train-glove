"""Quality numbers for a Leap recording: detection, re-acquisitions, jitter.

These are the measurements the Phase 2 gate turns on (does the IR camera see
a hand inside the black StretchSense glove?), so the definitions matter more
than the code:

  detection rate   frames actually in the file / frames the file should hold.
      "Should hold" is the span times the recording's own sustained cadence,
      taken as the 10th-percentile gap between consecutive frames of that
      hand. For an unthrottled recording that cadence is the tracking rate;
      for one written with `--hz 5` it is 5. Dropouts show up as gaps far
      wider than the baseline, which is exactly what this is measuring. The
      raw tracking `framerate` is reported next to it, never used as the
      denominator — otherwise a deliberately downsampled file would score 6%.

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
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# abs26 indices of the five fingertips (thumb, index, middle, ring, little).
TIP_IDX = (5, 10, 15, 20, 25)
JITTER_WINDOW_S = 2.0
MIN_JITTER_FRAMES = 5
MM = 1000.0


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

    @property
    def ok(self) -> bool:
        """The plan's Path A thresholds: 80%+ detected, under 1 loss per 10 s."""
        losses_per_10s = self.reacquisitions / max(self.span_s / 10.0, 1e-9)
        return self.detection_rate >= 0.80 and losses_per_10s <= 1.0


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
    expected = int(round(span * sample_hz)) + 1 if sample_hz else len(rows)
    detection = len(rows) / expected if expected else 1.0

    ids = [r.get("hand_id") for r in rows]
    reacquisitions = sum(1 for a, b in zip(ids, ids[1:]) if a != b and b is not None)

    rates = [float(r["framerate"]) for r in rows if r.get("framerate")]
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
        mean_framerate=sum(rates) / len(rates) if rates else 0.0,
        jitter_mm=fingertip_jitter(rows),
        frame_age_ms=(sum(ages) / len(ages) / 1000.0) if ages else None,
        pose=sorted(poses)[0] if poses else "",
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


_HEADER = (f"{'file':<38} {'hand':<5} {'pose':<10} {'n':>5} {'span':>6} "
           f"{'rate':>6} {'fps':>6} {'det%':>6} {'reacq':>6} {'jit_mm':>7} {'age_ms':>7}")


def format_table(rows: Sequence[HandStats]) -> str:
    """The stats table, plus a totals line. Plain text, for terminal or file."""
    lines = [_HEADER, "-" * len(_HEADER)]
    for s in rows:
        age = "      -" if s.frame_age_ms is None else f"{s.frame_age_ms:7.1f}"
        lines.append(
            f"{s.file[:38]:<38} {s.hand_side:<5} {s.pose[:10]:<10} {s.frames:>5} "
            f"{s.span_s:>6.1f} {s.sample_hz:>6.1f} {s.mean_framerate:>6.1f} "
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
    lines.append(
        "rate = the file's own sample cadence (10th-percentile gap); "
        "fps = tracking framerate reported by LeapC;"
    )
    lines.append(
        "det% = frames present / span x rate; jit_mm = fingertip spread over "
        "the steadiest 2 s; age_ms = frame age at receipt."
    )
    return "\n".join(lines)
