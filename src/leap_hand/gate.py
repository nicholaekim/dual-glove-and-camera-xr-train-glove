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

Condition names are open. `bare` is the reserved name — the reference every
other row is measured against, along with the per-distance `bare_<N>cm` runs a
distance sweep needs — and **everything else is a gloved condition**, judged
against the bare hand of the same side. That is what lets
the reviewer's extra runs (`glove_right`, `glove_both`, `glove_20cm`,
`glove_35cm`, `glove_50cm`, `glove_day2`) go through the same table, the same
thresholds and the same verdict as the four the plan names, with no code
change: the side comes from the frames, not from the folder name.

**Per pose, and paired.** The three thresholds above are measured over a whole
condition, which is only meaningful if both conditions did the same thing —
and the 2026-09-17 session proved what happens when they do not: the bare
baseline scored 62 % because the hand turned edge-on for five seconds, and
four gloved runs lost tracking while the hand was a fist with no bare-hand
fist anywhere to compare against. So a take recorded under a `--schedule`
carries the plan on every frame, `stats.analyse_poses` measures each pose
separately, and `pose_verdicts` pairs each gloved pose against the BARE hand
doing the SAME pose on the SAME side. A pose with no bare partner is reported
as missing and left unjudged; nothing about it is inferred.

`scan_out_dir` rebuilds these results from recordings already on disk, which
is what `scripts/leap/gate.py --recompute` runs on. A report is then a pure
function of the files in the folder, so a fix to a measurement (see
`stats.choose_rate`) can be re-applied to last night's data without asking
anyone to hold a hand over the camera again.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .protocol import band_for_condition, band_text, baseline_condition
from .stats import (
    HandStats,
    PoseStats,
    analyse_paths,
    analyse_poses,
    rate_footnote,
    read_rows,
    schedule_of,
)

log = logging.getLogger(__name__)

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

# Conditions that are the plain glove plus something. Only these get the "the
# plain glove did not, so keep the X on" sentence in the verdict; a custom
# name like `glove_20cm` is a different setup, not a mitigation, and saying
# "keep the 20cm on" would be nonsense.
MITIGATIONS: Dict[str, str] = {
    "glove_liner": "liner",
    "glove_tape": "white tape",
    "glove_retroreflective": "retro-reflective tape",
}


def is_reference(condition: str) -> bool:
    """`bare`, and the per-distance bare runs `bare_20cm`, `bare_50cm`, ...

    A distance sweep needs a bare run at each distance, or the gloved rows at
    50 cm are compared against a bare hand at 25 cm and the report measures
    the height rather than the glove. Those runs are references, not
    conditions under test, whatever else follows the prefix.
    """
    return condition == BARE or condition.startswith(BARE + "_")


def is_glove(condition: str) -> bool:
    """Everything that is not a bare reference is a gloved condition."""
    return not is_reference(condition)


def instruction_for(condition: str) -> str:
    """What the operator does before this condition, custom names included."""
    hint = CONDITION_INSTRUCTIONS.get(condition)
    if hint:
        return hint
    return (f"Glove condition '{condition}': the StretchSense glove on. The "
            "name says the rest — which hand, how far above the module, "
            "which day.")

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
    # Per scheduled pose. Empty for a take recorded without a `--schedule`,
    # which is the whole of what `--recompute` has to cope with on the old
    # data: no pose boundaries in the file, so no per-pose row, and the report
    # says so rather than attributing the frames to a pose by their position.
    pose_stats: List[PoseStats] = field(default_factory=list)
    schedule_text: str = ""

    @property
    def frames(self) -> int:
        return sum(s.frames for s in self.stats)

    @property
    def saw_hand(self) -> bool:
        return bool(self.stats) and self.frames > 0

    @property
    def scheduled(self) -> bool:
        return bool(self.pose_stats)


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
    gloved = [r for r in results if is_glove(r.condition)]
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
        mitigation = MITIGATIONS.get(best)
        if mitigation and any(r.condition == "glove" for r in gloved):
            why += (" — the plain glove did not, so the protocol has to keep "
                    f"the {mitigation} on")
        if len(passing) > 1:
            why += f"; also passing: {', '.join(c for c in passing if c != best)}"
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


# --- the paired per-pose verdict ---------------------------------------------
# The whole-condition thresholds above answer "is the glove trackable"; these
# answer "which poses is it trackable in", which is the question the 2026-09-17
# session could not answer because the gloved and bare runs held different
# poses for different lengths of time. Pairing is per POSE and per HAND, and a
# pose with no bare partner is reported as missing — the one thing that must
# never happen here is a gloved fist being quietly compared against a bare
# open palm.
POSE_DETECTION_MIN = 0.80
POSE_DETECTION_MARGIN = 0.10      # points below bare that still count as level
POSE_LOSS_MAX_S = 1.0

PASS = "PASS"
FAIL = "FAIL"
NO_BARE = "NO BARE"

POSE_THRESHOLD_TEXT = (
    f"a pose passes when glove detection >= {POSE_DETECTION_MIN * 100:.0f}% "
    f"OR within {POSE_DETECTION_MARGIN * 100:.0f} points of bare, AND its "
    f"longest loss is <= {POSE_LOSS_MAX_S:g} s OR no longer than bare's")


@dataclass
class PoseVerdict:
    """One gloved pose, judged against the bare hand doing the same pose."""

    condition: str
    hand_side: str
    pose: str
    glove: PoseStats
    bare: Optional[PoseStats] = None
    baseline_condition: str = ""
    passed: Optional[bool] = None            # None = not judged, no bare
    reasons: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.passed is None:
            return NO_BARE
        return PASS if self.passed else FAIL


def judge_pose(glove: PoseStats, bare: Optional[PoseStats]) -> tuple:
    """(passed, reasons) for one gloved pose row. `passed` None = no bare.

    Both clauses are "absolute OR relative": a glove that tracks well enough
    on its own passes, and so does one that tracks no worse than the bare hand
    did — because a pose the camera cannot see bare is not evidence about the
    glove. Without a bare row neither relative half can be evaluated, so the
    row is not judged at all rather than judged on the absolute half alone.
    """
    det = glove.detection_rate
    loss = glove.longest_loss_s
    if bare is None:
        note = (f"glove detection {det * 100:.1f}% and longest loss "
                f"{loss:.2f} s were measured, but there is no bare "
                f"{glove.pose} for the {glove.hand_side} hand to pair them "
                "with, so this pose is not judged")
        return None, [note]

    reasons: List[str] = []
    det_ok = det >= POSE_DETECTION_MIN or det >= bare.detection_rate - POSE_DETECTION_MARGIN
    if not det_ok:
        reasons.append(
            f"detection {det * 100:.1f}% is below {POSE_DETECTION_MIN * 100:.0f}% "
            f"and {(bare.detection_rate - det) * 100:.1f} points below bare "
            f"({bare.detection_rate * 100:.1f}%)")
    loss_ok = loss <= POSE_LOSS_MAX_S or loss <= bare.longest_loss_s
    if not loss_ok:
        reasons.append(
            f"longest loss {loss:.2f} s is over {POSE_LOSS_MAX_S:g} s and "
            f"longer than bare's ({bare.longest_loss_s:.2f} s)")
    if not reasons:
        reasons.append(
            f"detection {det * 100:.1f}% (bare {bare.detection_rate * 100:.1f}%), "
            f"longest loss {loss:.2f} s (bare {bare.longest_loss_s:.2f} s)")
    return (det_ok and loss_ok), reasons


def pose_verdicts(results: Sequence[ConditionResult]) -> List[PoseVerdict]:
    """Every gloved pose row, paired with its bare partner where there is one.

    A distance run is paired against the bare run at the same distance where
    one exists (`glove_20cm` -> `bare_20cm`): detection falls off with height,
    so comparing 20 cm against 35 cm measures the height, not the glove.
    """
    available = [r.condition for r in results]
    bare_rows: Dict[tuple, PoseStats] = {
        (r.condition, s.hand_side, s.pose): s
        for r in results if is_reference(r.condition)
        for s in r.pose_stats
    }
    out: List[PoseVerdict] = []
    for r in results:
        if not is_glove(r.condition):
            continue
        base_name = baseline_condition(r.condition, available, BARE) or ""
        for s in r.pose_stats:
            bare = bare_rows.get((base_name, s.hand_side, s.pose))
            passed, reasons = judge_pose(s, bare)
            out.append(PoseVerdict(condition=r.condition,
                                   hand_side=s.hand_side, pose=s.pose,
                                   glove=s, bare=bare,
                                   baseline_condition=base_name,
                                   passed=passed, reasons=reasons))
    return out


# --- rebuilding a run from the files it left behind --------------------------
# Every take this pipeline writes carries a `_YYYYmmdd_HHMMSS` stamp, so the
# name is the recording time and survives copying, syncing and restoring —
# none of which mtime does. mtime is only the tiebreak.
_STAMP_RE = re.compile(r"_(\d{8}_\d{6})")


def take_recency(path: Path) -> tuple:
    """Sort key over takes of one condition: newest last."""
    m = _STAMP_RE.search(path.name)
    try:
        mtime = path.stat().st_mtime
    except OSError:                              # pragma: no cover - race
        mtime = 0.0
    return (m.group(1) if m else "", mtime, path.name)


def read_snapshots(folder: Path) -> tuple:
    """(stills, stills with a tracked hand) from the sidecars in a folder.

    A sidecar is a JSON dict with an `images` list — `write_snapshot` writes
    one beside every PNG pair. Anything else in the folder is ignored rather
    than guessed at, and an unreadable sidecar is skipped with a warning: a
    miscounted still would end up in the report as evidence.
    """
    total = with_hand = 0
    for path in sorted(folder.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("unreadable snapshot sidecar %s (%s); not counted",
                        path.name, e)
            continue
        if not isinstance(d, dict) or "images" not in d:
            continue
        total += 1
        if d.get("hand_count", 0) > 0:
            with_hand += 1
    return total, with_hand


def scan_condition(folder: Path, band=None) -> ConditionResult:
    """One condition folder -> the `ConditionResult` the report wants.

    The newest take is the one measured. Older takes in the same folder are
    named in the note instead of being merged in: they are usually aborted
    runs, and averaging an aborted run into a good one hides both.
    """
    condition = folder.name
    snapshots, with_hand = read_snapshots(folder)
    result = ConditionResult(condition=condition, snapshots=snapshots,
                             snapshots_with_hand=with_hand,
                             folder=str(folder))

    takes = sorted(folder.glob("*.jsonl"), key=take_recency)
    if not takes:
        result.note = ("no JSONL take in this folder"
                       + (f"; {snapshots} IR still(s) kept" if snapshots else ""))
        return result

    newest = takes[-1]
    result.recordings = [str(newest)]
    result.stats = analyse_paths([newest])
    attach_pose_stats(result, newest, band)
    notes = [f"measured from the most recent take, {newest.name}"]
    if len(takes) > 1:
        notes.append("older takes in this folder, not measured: "
                     + ", ".join(t.name for t in takes[:-1]))
    if not result.stats:
        notes.append("that take holds no frames: the tracker reported no hand")
    elif not result.scheduled:
        notes.append("recorded without a pose schedule, so it has no per-pose "
                     "rows: the frames carry no pose boundaries and inventing "
                     "them would make up the comparison")
    result.note = "; ".join(notes)
    return result


def attach_pose_stats(result: ConditionResult, take: Path, band=None) -> None:
    """Fill in the per-pose rows of one condition, if its take has a schedule."""
    used = band_for_condition(result.condition, band)
    result.pose_stats = analyse_poses(take, band=used,
                                      condition=result.condition)
    if result.pose_stats:
        try:
            schedule = schedule_of(read_rows(take))
        except (OSError, ValueError):             # pragma: no cover - race
            schedule = None
        result.schedule_text = schedule.text if schedule else ""


def scan_out_dir(out_dir: Path, conditions: Optional[Sequence[str]] = None,
                 band=None) -> List[ConditionResult]:
    """Rebuild every condition of a gate run from `out_dir`, bare first.

    Conditions are the sub-folder names — whatever they were called, so the
    reviewer's extra runs need no list here. `conditions`, when given, keeps
    only those (and keeps the caller's order).
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    folders = {p.name: p for p in sorted(out_dir.iterdir()) if p.is_dir()}
    if conditions is not None:
        names = [c for c in conditions if c in folders]
    else:
        # bare first: it is the reference, and the table reads top-down.
        names = ([BARE] if BARE in folders else []) + sorted(
            n for n in folders if n != BARE)
    return [scan_condition(folders[n], band) for n in names]


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


_POSE_HEADER = (f"{'condition':<16} {'hand':<5} {'pose':<10} {'n':>5} "
                f"{'sched':>6} {'det%':>6} {'loss_s':>7} {'reacq':>6} "
                f"{'med_cm':>7} {'band%':>6} {'face%':>6}")


def _pose_row(s: PoseStats) -> str:
    def num(value, fmt="{:6.1f}"):
        return "     -" if value is None else fmt.format(value)

    return (f"{s.condition[:16]:<16} {s.hand_side:<5} {s.pose[:10]:<10} "
            f"{s.frames:>5} {s.scheduled_s:>6.1f} "
            f"{s.detection_rate * 100:>6.1f} {s.longest_loss_s:>7.2f} "
            f"{s.reacquisitions:>6} {num(s.median_height_cm, '{:7.1f}'):>7} "
            f"{num(s.in_band_pct)} {num(s.facing_pct)}")


def pose_section(results: Sequence[ConditionResult], band=None) -> List[str]:
    """The per-pose table and the paired verdict, or why there is neither."""
    scheduled = [r for r in results if r.scheduled]
    if not scheduled:
        return ["", "Per pose",
                "  None of these takes was recorded under a --schedule, so "
                "there are no pose boundaries in",
                "  the files and no per-pose rows. The table above is the "
                "whole-condition measurement, which",
                "  mixes every pose the operator happened to hold together. "
                "Re-run the gate to get the",
                "  per-pose comparison: python scripts/leap/gate.py "
                "--conditions bare,glove"]

    plans = {r.schedule_text for r in scheduled if r.schedule_text}
    plan = plans.pop() if len(plans) == 1 else "several (see the rows)"
    lines = ["", f"Per pose (schedule: {plan}"
                 + (f", band {band_text(band)}" if band else "") + ")",
             _POSE_HEADER, "-" * len(_POSE_HEADER)]
    for r in results:
        for s in sorted(r.pose_stats, key=lambda s: (s.hand_side, s.pose)):
            lines.append(_pose_row(s))
    unscheduled = [r.condition for r in results
                   if r.saw_hand and not r.scheduled]
    lines.append("-" * len(_POSE_HEADER))
    lines.append("det% = frames present / (scheduled seconds x rate); "
                 "loss_s = longest stretch of that pose with no hand;")
    lines.append("band% = frames inside the height band; face% = frames with "
                 "the palm within 40 degrees of the lens.")
    if unscheduled:
        lines.append("no per-pose rows (recorded without a schedule): "
                     + ", ".join(unscheduled))

    verdicts = pose_verdicts(results)
    lines += ["", "Paired verdict per pose (glove vs bare, same hand)"]
    if not verdicts:
        lines.append("  No gloved condition has per-pose rows, so nothing "
                     "could be paired.")
        return lines
    for pv in verdicts:
        head = (f"  {pv.condition[:16]:<16} {pv.hand_side:<5} "
                f"{pv.pose[:10]:<10} {pv.status:<7}")
        lines.append(head + "; ".join(pv.reasons))
        if pv.passed is None:
            lines.append(f"  {'':<16} {'':<5} {'':<10} {'':<7}"
                         f"MISSING: no {pv.baseline_condition or BARE} "
                         f"{pv.pose} on the {pv.hand_side} hand in this run")
    missing = [pv for pv in verdicts if pv.passed is None]
    failed = [pv for pv in verdicts if pv.passed is False]
    lines.append(f"  {POSE_THRESHOLD_TEXT}.")
    lines.append(f"  {sum(1 for pv in verdicts if pv.passed)} passed, "
                 f"{len(failed)} failed, {len(missing)} not judged "
                 "(no bare pose to pair with).")
    if failed:
        lines.append("  Failing poses: " + ", ".join(
            f"{pv.condition}/{pv.hand_side}/{pv.pose}" for pv in failed))
    if missing:
        lines.append("  Re-run the bare condition with the same --schedule to "
                     "judge: " + ", ".join(
                         f"{pv.hand_side}/{pv.pose}" for pv in missing))
    return lines


def format_report(results: Sequence[ConditionResult], v: Verdict,
                  title: str = "", seconds: Optional[float] = None,
                  band=None) -> str:
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
    lines += rate_footnote([s for r in results for s in r.stats],
                           has_cadence_column=False)

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

    lines += pose_section(results, band=band)

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
