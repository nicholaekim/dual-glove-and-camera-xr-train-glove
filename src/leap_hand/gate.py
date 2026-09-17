"""The Phase 2 gate, as a decision: Path A or Path B.

`docs/ultraleap_ir170_plan.md` section 2 asks one question — does the IR
tracker see a hand inside the black StretchSense glove? — and answers it with
three numbers per condition, measured by `leap_hand.stats`:

    detection rate          >= 80 %              frames the hand was reported
    re-acquisitions         <= 1 per 10 s        losses the tracker recovered
    fingertip jitter        <= 2x the bare hand  noise at rest, millimetres

A gloved condition that meets all three is Path A: simultaneous capture, the
glove supplying flexion and the camera supplying palm pose, spread, thumb and
wrist. Anything less is Path B: sequential capture, each pose recorded twice,
glove on and glove off. Both paths reuse every line of Phase 1, so this
decides the protocol, not the code.

This module holds the judgment and the report text; `scripts/leap/gate.py`
runs the protocol that produces the input. They are split because the verdict
is the part worth unit-testing against synthetic numbers — a threshold that
is wrong by a factor of ten is invisible in a live run and obvious in a test.

Two deliberate choices in the reading of the thresholds:

  * The jitter baseline is the **bare hand of the same side** where the bare
    run has it, otherwise the mean over all bare rows. A left hand jitters
    differently from a right one, and comparing across sides would blame the
    glove for it.
  * The mitigations (a liner glove, white tape) count. If the plain glove
    fails but `glove_liner` passes, that is still Path A — with the liner in
    the protocol, and the report says so rather than burying it.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .stats import HandStats, rate_footnote

# The conditions of the plan's protocol, in the order they are run: each one
# adds something to the hand, so the operator never has to take anything off.
DEFAULT_CONDITIONS = ("bare", "glove", "glove_liner", "glove_tape")
BARE = "bare"

# What the operator does before each condition. Printed and waited on between
# runs, so the session is self-documenting for whoever repeats it.
CONDITION_INSTRUCTIONS: Dict[str, str] = {
    "bare": "Bare hand. Nothing on it. This is the reference every other "
            "condition is measured against.",
    "glove": "Put the StretchSense glove on. Nothing over it yet.",
    "glove_liner": "Pull a thin white cotton liner glove OVER the "
                   "StretchSense glove.",
    "glove_tape": "Take the liner off. Stick small strips of plain white "
                  "tape on the fingertips and knuckles of the StretchSense "
                  "glove.",
    "glove_retroreflective": "Replace the white tape with retro-reflective "
                             "tape. Last, because it can saturate the IR "
                             "image.",
}

# Plan section 2, decision table.
DETECTION_MIN = 0.80
REACQ_PER_10S_MAX = 1.0
JITTER_FACTOR_MAX = 2.0

THRESHOLD_TEXT = (f"detection >= {DETECTION_MIN * 100:.0f}%, "
                  f"re-acquisitions <= {REACQ_PER_10S_MAX:g} per 10 s, "
                  f"jitter <= {JITTER_FACTOR_MAX:g}x the bare hand")


@dataclass
class ConditionResult:
    """Everything one condition of the gate produced."""

    condition: str
    stats: List[HandStats] = field(default_factory=list)
    snapshots: int = 0
    snapshots_with_hand: int = 0
    recordings: List[str] = field(default_factory=list)
    folder: str = ""
    note: str = ""

    @property
    def frames(self) -> int:
        return sum(s.frames for s in self.stats)

    @property
    def saw_hand(self) -> bool:
        return bool(self.stats) and self.frames > 0


def reacq_per_10s(s: HandStats) -> float:
    """Re-acquisitions scaled to the plan's per-10-seconds unit."""
    return s.reacquisitions / max(s.span_s / 10.0, 1e-9)


def jitter_baselines(results: Sequence[ConditionResult]) -> Dict[str, float]:
    """Bare-hand jitter per side, plus `""` for the all-rows mean.

    Empty when there is no usable bare run, which the verdict reports as an
    un-checked threshold rather than silently passing it.
    """
    bare = [s for r in results if r.condition == BARE for s in r.stats
            if s.frames > 0]
    if not bare:
        return {}
    out: Dict[str, float] = {}
    for side in {s.hand_side for s in bare}:
        rows = [s.jitter_mm for s in bare if s.hand_side == side]
        out[side] = sum(rows) / len(rows)
    out[""] = sum(s.jitter_mm for s in bare) / len(bare)
    return out


def baseline_for(side: str, baselines: Dict[str, float]) -> Optional[float]:
    """The bare-hand jitter this row is judged against, or None."""
    if not baselines:
        return None
    return baselines.get(side, baselines.get(""))


def row_failures(s: HandStats, baselines: Dict[str, float]) -> List[str]:
    """Which of the plan's thresholds this hand-row misses. Empty = passes."""
    bad: List[str] = []
    if s.detection_rate < DETECTION_MIN:
        bad.append(f"detection {s.detection_rate * 100:.1f}% "
                   f"< {DETECTION_MIN * 100:.0f}%")
    per10 = reacq_per_10s(s)
    if per10 > REACQ_PER_10S_MAX:
        bad.append(f"{per10:.2f} re-acquisitions per 10 s "
                   f"> {REACQ_PER_10S_MAX:g}")
    base = baseline_for(s.hand_side, baselines)
    if base is not None and base > 0 and s.jitter_mm > JITTER_FACTOR_MAX * base:
        bad.append(f"jitter {s.jitter_mm:.2f} mm "
                   f"> {JITTER_FACTOR_MAX:g}x bare ({base:.2f} mm)")
    return bad


def jitter_ratio(s: HandStats, baselines: Dict[str, float]) -> Optional[float]:
    base = baseline_for(s.hand_side, baselines)
    if base is None or base <= 0:
        return None
    return s.jitter_mm / base


def condition_failures(result: ConditionResult,
                       baselines: Dict[str, float]) -> List[str]:
    """Why this condition is not Path A. Empty = every hand passed."""
    if not result.saw_hand:
        return ["no hand was tracked at all"]
    bad: List[str] = []
    for s in result.stats:
        for reason in row_failures(s, baselines):
            bad.append(f"{s.hand_side}: {reason}")
    return bad


@dataclass
class Verdict:
    """The gate's answer, already phrased for the report."""

    path: str                                   # "A" or "B"
    passing: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    baselines: Dict[str, float] = field(default_factory=dict)

    @property
    def path_a(self) -> bool:
        return self.path == "A"


def verdict(results: Sequence[ConditionResult]) -> Verdict:
    """Path A or Path B from the plan's thresholds, with the reasons.

    The first line is always `Path A: yes because ...` or
    `Path A: no because ...`, so the answer survives being read by eye, by
    grep, or by whoever opens the report six months from now.
    """
    baselines = jitter_baselines(results)
    gloved = [r for r in results if r.condition != BARE]
    any_hand = any(r.saw_hand for r in results)

    if not any_hand:
        return Verdict(
            path="B", baselines=baselines,
            lines=[
                "Path A: no because no hand was tracked in any condition — "
                "the run recorded no hands at all, so none of the three "
                "thresholds could be measured.",
                "This is not a result about the glove. Re-run the gate with "
                "a hand over the module.",
            ],
        )

    if not gloved:
        conditions = ", ".join(r.condition for r in results) or "none"
        return Verdict(
            path="B", baselines=baselines,
            lines=[
                "Path A: no because no gloved condition was recorded "
                f"(conditions in this run: {conditions}) — the gate compares "
                "a gloved hand against the bare one, and there is nothing to "
                "compare.",
                "Re-run with --conditions bare,glove (plus any mitigation) "
                "before reading anything into this.",
            ],
        )

    passing = [r.condition for r in gloved if not condition_failures(r, baselines)]
    lines: List[str] = []

    if passing:
        best = "glove" if "glove" in passing else passing[0]
        why = (f"the {best} condition met all three thresholds "
               f"({THRESHOLD_TEXT})")
        if best != "glove":
            why += (" — the plain glove did not, so the protocol has to keep "
                    f"the {best.replace('glove_', '')} on")
        lines.append(f"Path A: yes because {why}.")
        lines.append("Simultaneous capture: the glove supplies flexion, the "
                     "camera supplies palm pose, spread, thumb and wrist "
                     "(plan section 2, Path A).")
    else:
        worst = []
        for r in gloved:
            reasons = condition_failures(r, baselines)
            worst.append(f"{r.condition} ({'; '.join(reasons)})")
        lines.append("Path A: no because no gloved condition met all three "
                     f"thresholds ({THRESHOLD_TEXT}): " + ", ".join(worst) + ".")
        lines.append("Path B: sequential capture — record each pose twice, "
                     "glove on and glove off, under one protocol and one "
                     "naming scheme (plan section 2, Path B).")

    if not baselines:
        lines.append("Note: no bare-hand run in this session, so the jitter "
                     "threshold could not be applied — only detection and "
                     "re-acquisitions were checked.")
    return Verdict(path="A" if passing else "B", passing=passing, lines=lines,
                   baselines=baselines)


# --- the report -------------------------------------------------------------
_HEADER = (f"{'condition':<16} {'hand':<5} {'n':>5} {'span':>6} {'fps':>6} "
           f"{'rate':>7} {'det%':>6} {'reacq':>6} {'/10s':>6} {'jit_mm':>7} "
           f"{'jit_x':>6} {'snaps':>6} {'pass':>5}")


def _row(result: ConditionResult, s: HandStats,
         baselines: Dict[str, float]) -> str:
    ratio = jitter_ratio(s, baselines)
    return (f"{result.condition[:16]:<16} {s.hand_side:<5} {s.frames:>5} "
            f"{s.span_s:>6.1f} {s.mean_framerate:>6.1f} "
            f"{s.rate_hz:>6.1f}{s.rate_flag} "
            f"{s.detection_rate * 100:>6.1f} {s.reacquisitions:>6} "
            f"{reacq_per_10s(s):>6.2f} {s.jitter_mm:>7.2f} "
            f"{'     -' if ratio is None else f'{ratio:>6.2f}'} "
            f"{result.snapshots:>6} "
            f"{'yes' if not row_failures(s, baselines) else 'NO':>5}")


def format_report(results: Sequence[ConditionResult], v: Verdict,
                  title: str = "", seconds: Optional[float] = None) -> str:
    """The whole of `results/leap_gate/REPORT.txt`, as one string."""
    lines = ["Ultraleap Stereo IR 170 — Phase 2 gate "
             "(docs/ultraleap_ir170_plan.md section 2)"]
    if title:
        lines.append(title)
    if seconds is not None:
        lines.append(f"seconds recorded per condition: {seconds:g}")
    lines += ["", _HEADER, "-" * len(_HEADER)]

    for result in results:
        if result.stats:
            for s in sorted(result.stats, key=lambda s: s.hand_side):
                lines.append(_row(result, s, v.baselines))
        else:
            lines.append(f"{result.condition[:16]:<16} {'-':<5} {0:>5} "
                         f"{0.0:>6.1f} {0.0:>6.1f} {0.0:>6.1f}  {0.0:>6.1f} "
                         f"{0:>6} {0.0:>6.2f} {0.0:>7.2f} {'     -':>6} "
                         f"{result.snapshots:>6} {'NO':>5}")
    lines.append("-" * len(_HEADER))
    lines.append(f"thresholds: {THRESHOLD_TEXT}")
    lines.append("det% = frames present / span x rate; "
                 "jit_mm = fingertip spread over the steadiest 2 s;")
    lines.append("jit_x = that jitter divided by the bare hand's "
                 "(same side where the bare run has it).")
    lines += rate_footnote([s for r in results for s in r.stats])

    lines += ["", "What the camera saw"]
    for result in results:
        if result.saw_hand:
            sides = ", ".join(sorted({s.hand_side for s in result.stats}))
            what = (f"{result.frames} frames, hands: {sides}")
        else:
            what = "no hand was seen"
        stills = (f"{result.snapshots} IR still(s), "
                  f"{result.snapshots_with_hand} with a tracked hand")
        lines.append(f"  {result.condition:<16} {what}; {stills}")
        if result.folder:
            lines.append(f"  {'':<16} files: {result.folder}")
        if result.note:
            lines.append(f"  {'':<16} note: {result.note}")

    lines += ["", "Verdict"]
    lines += [f"  {line}" for line in v.lines]

    lines += ["", "Evidence",
              "  The JSONL takes, the IR stills and their sidecars are in the "
              "per-condition folders listed above.",
              "  The stills are the professor-facing evidence: each PNG has a "
              "sidecar saying whether the",
              "  tracker reported a hand at that instant, so a photo of a "
              "clearly visible glove next to",
              "  'no hand tracked' is itself the finding."]
    return "\n".join(lines)
