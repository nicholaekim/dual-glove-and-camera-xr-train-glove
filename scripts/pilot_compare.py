"""Camera (Ultraleap) against glove for the hand-on-object pilot. No hardware.

docs/pilot_hand_on_object.md, step 10, written before the pilot's data
exists. Reads the five run folders scripts/record_simultaneous.py writes
(--camera leap --hand left), each take as it is on disk:

    <run>/leap/<stem>.meta.json   the take's metadata (accepted takes only)
    <run>/leap/<stem>.jsonl       the camera's lines
    <run>/glove/<stem>.jsonl      the glove's lines, same name

Per take: ONE window for both sensors, the last --window seconds before the
window end, on the clock cam_hand.fusion.pairing_clock picks (capture_time
when every line has it, wall_time otherwise). The window end is the
recorder's `hold_end` from the meta (the REC hold as commanded) when the meta
has one; else the later of the two sensors' last stamps (the recorder stops
both together, so a camera that lost the hand early shows as missing
coverage, not as a shorter window). hold_end, wall_time and capture_time are
all time.time() seconds (capture_time is the sensor's capture instant on that
same clock), so hold_end ends the window on either clock.

Every camera frame of the hand in the window is paired with the nearest glove
frame within --max-dt, no lag shift (the hand is held still). The 8 flexion
angles (index, middle, ring, little; MCP and PIP) of both come from
xr_hand.joint_frames (wrist reference, positive toward the palm). Per pair
and joint, camera minus glove wrapped to [-180, 180) and its absolute value;
per take and joint the medians of both over the pairs, each sensor's own
median and MAD (held-pose stability), and each sensor's coverage
(leap_hand.protocol.coverage) and gaps over 100 ms (stream_health) in the
window, kept apart from the angles. The glove is a reference, not truth.

A take whose meta's hints_confirmed contradicts its run (true outside C, false
in C) is invalid: its row stays in takes.csv, numbers included; the summary
counts it and leaves its numbers out. A kept take whose meta says
hold_completed false (its hold was cut short) is unusable the same way.

Each take's acceptance as the recorder wrote it (amendment of 2026-10-23):
acceptance_policy ("strict", or "object_keep_all_tracking" for the object
runs, which keep every completed hold), hold_completed, the camera's tracked
fraction of the hold, its re-acquisitions and tracking_note, empty for a
meta written before them, which counts as strict.

Attempts that ended before their hold (amendment of 2026-11-17): no open hand
acquired in time, or the hand lost during the settle. They leave no take
file and no meta, so they are counted on their own (n_attempts_before_hold),
beside the attempts that reached the hold: from the recorder's
<run>/attempts_before_hold.log, and for takes recorded before the recorder
wrote it, from the attempt numbers in the take's metas (a floor, see
`before_hold_rows`).

The object gate (second amendment of 2026-10-23), per run and grasp: the raw
IR audit stills <run>/audit/<grasp>_*_L.png and the typed confirmations in
<run>/audit/confirmations.txt; a run without them counts 0.

Writes takes.csv, summary.csv and report.txt into --out (default
<root>\\comparison).

  .venv\\Scripts\\python.exe scripts\\pilot_compare.py --root recordings\\pilot_hoo
  .venv\\Scripts\\python.exe scripts\\pilot_compare.py --run C=recordings\\pilot_hoo\\dry_run --out recordings\\pilot_hoo\\dry_run_comparison
"""
import argparse
import csv
import datetime
import json
import sys
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from statistics import median

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cam_hand.fusion import WALL_CLOCK, _stamp, pairing_clock
from leap_hand.protocol import LOSS_GAP_S, coverage, stream_health
from xr_hand.joint_frames import detect_source, dof24, joint_table

# label, folder under --root, condition, the hints_confirmed its takes must have
RUNS = [("A", "A_mimed", "A", False), ("B1", "B1_object", "B", False),
        ("C", "C_object_hint", "C", True), ("B2", "B2_object", "B", False),
        ("A2", "A_mimed_again", "A", False)]
# The pre-registered grid: the pose names the protocol gives the recorder,
# and the takes per grasp it records in each run ("B" is B1 and B2 pooled).
# Every run x grasp of it is reported, recorded or not, so a grasp that is
# missing altogether shows as missing.
GRASPS = ["cylindrical_object", "spherical_object", "tip_object"]
EXPECTED = {"A": 5, "B1": 3, "C": 5, "B2": 2, "A2": 1, "B": 5}
JOINTS = ["I_MCP_fe", "I_PIP", "M_MCP_fe", "M_PIP",
          "R_MCP_fe", "R_PIP", "L_MCP_fe", "L_PIP"]
OK = "ok"
INVALID = "invalid:"
# Why the recorder refused an attempt (its meta's `why`), by the words of its
# messages; a refused attempt is counted once, under the first that matches.
REJECTED_BY = [("n_rejected_hand_lost", "was lost"),
               ("n_rejected_reacquired", "re-acquired"),
               ("n_rejected_low_coverage", "covered only")]
REJECTED_OTHER = "n_rejected_other"
# Where the recorder lists the attempts that ended before their hold
# (record_simultaneous.BEFORE_HOLD_LOG), one JSON line each.
BEFORE_HOLD_LOG = "attempts_before_hold.log"
META_END, LAST_STAMP = "meta hold_end", "later last stamp"
# The recorder's acceptance_policy (a meta without one was recorded strict);
# KEEP_ALL keeps every completed hold, its losses in tracking_note.
STRICT, KEEP_ALL = "strict", "object_keep_all_tracking"
HOLD_CUT = "unusable: hold not completed"
TITLE = ("Pilot comparison: camera (Ultraleap, desktop) against glove "
         "(StretchSense), flexion only, 8 joints")
TAKE_COLUMNS = (
    ["run", "condition", "grasp", "take", "file", "hand", "note",
     "hints_confirmed", "acceptance_policy", "hold_completed",
     "cam_tracked_fraction", "cam_reacquisitions", "tracking_note",
     "status", "clock", "window_start", "window_end",
     "window_end_source", "n_cam_frames_window", "n_glove_frames_window",
     "n_pairs", "n_cam_unpaired", "cam_coverage", "cam_max_gap_ms",
     "cam_gaps_over_100ms", "glove_coverage", "glove_max_gap_ms",
     "glove_gaps_over_100ms", "absdiff_med_all8", "absdiff_max_all8"]
    + [f"absdiff_med_{j}" for j in JOINTS] + [f"signed_med_{j}" for j in JOINTS]
    + [f"cam_med_{j}" for j in JOINTS] + [f"glove_med_{j}" for j in JOINTS]
    + [f"cam_mad_{j}" for j in JOINTS] + [f"glove_mad_{j}" for j in JOINTS]
    + ["cam_mad_med", "cam_mad_max", "glove_mad_med", "glove_mad_max"])
SUMMARY_COLUMNS = (
    ["level", "run", "condition", "grasp", "n_valid_takes", "n_expected_takes",
     "n_invalid_takes", "n_unusable_takes", "n_missing_takes", "n_attempts",
     "n_rejected", "n_rejected_hand_lost", "n_rejected_reacquired",
     "n_rejected_low_coverage", "n_rejected_other", "n_attempts_before_hold",
     "attempt_coverage_median",
     "attempt_coverage_min", "cam_tracked_fraction_median",
     "cam_tracked_fraction_min", "cam_reacquisitions_worst",
     "n_strict_policy_takes", "n_keep_all_takes", "n_audit_stills",
     "n_object_confirmations", "absdiff_median",
     "absdiff_largest", "paired_fraction_median", "paired_fraction_min",
     "cam_coverage_median", "cam_coverage_min", "cam_gaps_over_100ms_total",
     "cam_max_gap_ms_worst", "glove_coverage_median", "glove_coverage_min",
     "glove_max_gap_ms_worst", "cam_mad_max_median", "glove_mad_max_median",
     "n_wall_time_clock", "n_later_last_stamp"]
    + [f"absdiff_{j}" for j in JOINTS] + [f"signed_{j}" for j in JOINTS])


# --- one take ---------------------------------------------------------------------
def hand_lines(path: Path, hand: str) -> list:
    """The file's lines of `hand`; the other hand's lines are ignored."""
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(s) for s in f if s.strip()]
    return [d for d in lines if d.get("hand_side") == hand]


def angles8(line: dict, hand: str) -> list:
    d = dof24(joint_table(line, detect_source(line), reference="wrist",
                          hand=hand))
    return [d[j] for j in JOINTS]


def mad(values: list) -> float:
    """Median absolute deviation from the median."""
    m = median(values)
    return median(abs(v - m) for v in values)


def measure(row: dict, run_dir: Path, hold_end, window_s: float,
            max_dt: float) -> str:
    """Fill `row` with the take's numbers; return "ok" or why there are none."""
    hand = row["hand"]
    cam_path = run_dir / "leap" / row["file"]
    glove_path = run_dir / "glove" / row["file"]
    if not cam_path.is_file():
        return "no camera file"
    if not glove_path.is_file():
        return "no glove file"
    cam, glove = hand_lines(cam_path, hand), hand_lines(glove_path, hand)
    clock = pairing_clock(glove, cam)
    row["clock"] = clock
    glove.sort(key=lambda d: _stamp(d, clock))
    glove_t = [_stamp(d, clock) for d in glove]
    cam_t = [_stamp(d, clock) for d in cam]
    stamps = cam_t + glove_t
    if not stamps:
        return "no paired frames"
    if isinstance(hold_end, (int, float)) and not isinstance(hold_end, bool):
        w1, row["window_end_source"] = float(hold_end), META_END
    else:
        w1, row["window_end_source"] = max(stamps), LAST_STAMP
    span = w1 - min(stamps)
    if span < window_s:
        return f"take shorter than window ({span:.2f} s)"
    w0 = w1 - window_s

    cam_in = [(t, d) for t, d in zip(cam_t, cam) if w0 <= t <= w1]
    glove_in = [t for t in glove_t if w0 <= t <= w1]
    row.update(window_start=w0, window_end=w1,
               n_cam_frames_window=len(cam_in),
               n_glove_frames_window=len(glove_in))
    for name, times in (("cam", [t for t, _ in cam_in]), ("glove", glove_in)):
        health = stream_health(times, w0, w1)
        row[f"{name}_coverage"] = coverage(times, w0, w1)
        row[f"{name}_max_gap_ms"] = health["max_gap_ms"]
        row[f"{name}_gaps_over_100ms"] = health["gaps_over"]

    glove_angles: dict = {}                      # glove index -> its 8 angles
    cams, gloves = [], []
    for t, d in cam_in:
        i = bisect_left(glove_t, t)
        near = min((k for k in (i - 1, i) if 0 <= k < len(glove_t)),
                   key=lambda k: abs(glove_t[k] - t), default=None)
        if near is None or abs(glove_t[near] - t) > max_dt:
            continue
        if near not in glove_angles:
            glove_angles[near] = angles8(glove[near], hand)
        cams.append(angles8(d, hand))
        gloves.append(glove_angles[near])
    row["n_pairs"] = len(cams)
    row["n_cam_unpaired"] = len(cam_in) - len(cams)
    if not cams:
        return "no paired frames"
    for k, j in enumerate(JOINTS):
        c = [a[k] for a in cams]
        g = [a[k] for a in gloves]
        signed = [((x - y + 180.0) % 360.0) - 180.0 for x, y in zip(c, g)]
        row[f"absdiff_med_{j}"] = median(abs(v) for v in signed)
        row[f"signed_med_{j}"] = median(signed)
        row[f"cam_med_{j}"], row[f"glove_med_{j}"] = median(c), median(g)
        row[f"cam_mad_{j}"], row[f"glove_mad_{j}"] = mad(c), mad(g)
    per_joint = [row[f"absdiff_med_{j}"] for j in JOINTS]
    row["absdiff_med_all8"] = median(per_joint)
    row["absdiff_max_all8"] = max(per_joint)
    for name in ("cam", "glove"):
        mads = [row[f"{name}_mad_{j}"] for j in JOINTS]
        row[f"{name}_mad_med"], row[f"{name}_mad_max"] = median(mads), max(mads)
    return OK


def compare_take(run_dir: Path, label: str, condition: str, expected: bool,
                 meta: dict, window_s: float, max_dt: float) -> dict:
    hints = bool(meta.get("hints_confirmed"))
    row = {"run": label, "condition": condition, "grasp": str(meta["pose"]),
           "take": int(meta["take"]), "file": meta["file"],
           "hand": meta["hand"], "note": meta.get("note") or "",
           "hints_confirmed": hints,
           # the recorder's acceptance (amendment of 2026-10-23), None in a
           # meta written before it
           **{k: meta.get(k) for k in (
               "acceptance_policy", "hold_completed", "cam_tracked_fraction",
               "cam_reacquisitions", "tracking_note")}}
    data = measure(row, run_dir, meta.get("hold_end"), window_s, max_dt)
    problems = []
    if hints != expected:
        problems.append(f"{INVALID} hints_confirmed {hints} in run {label} "
                        f"(expected {expected})")
    if meta.get("hold_completed") is False:     # kept, but the hold was cut
        problems.append(HOLD_CUT)
    if data != OK:
        problems.append(data)
    row["status"] = "; ".join(problems) or OK
    return row


def accepted_metas(run_dir: Path) -> list:
    metas = []
    for path in sorted((run_dir / "leap").glob("*.meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("accepted"):
            metas.append(meta)
    return sorted(metas, key=lambda m: (str(m["pose"]), int(m["take"])))


def attempt_rows(run_dir: Path, label: str) -> list:
    """Every attempt of the run that reached its hold: the kept takes' metas
    in leap/ and the refused attempts' in rejected/leap/, any `accepted`
    value. (An attempt that ended before the hold has no meta: see
    `before_hold_rows`.) Under the strict policy the recorder keeps a take
    only while the hand stayed tracked, so the camera losing the hand shows
    HERE, not in the kept takes' coverage; under keep-all it shows in their
    tracked fraction."""
    rows = []
    for folder in (run_dir / "leap", run_dir / "rejected" / "leap"):
        for path in sorted(folder.glob("*.meta.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"run": label, "grasp": str(meta["pose"]),
                         "accepted": bool(meta.get("accepted")),
                         "why": meta.get("why") or "",
                         "coverage": meta.get("coverage"),
                         # which attempt at which take it was, for
                         # `before_hold_rows`
                         "take": meta.get("take"),
                         "number": meta.get("attempts")})
    return rows


def before_hold_rows(run_dir: Path, label: str, held: list) -> list:
    """One row per attempt of the run that ended before its hold (no open
    hand acquired in time, or the hand lost during the settle) and so left
    no take file and no meta; `held` is the run's `attempt_rows`.

    The recorder lists them in <run>/attempts_before_hold.log (amendment of
    2026-11-17), one JSON line each. A take recorded before that shows them
    only through the attempt numbers in its metas: every time the recorder
    starts a take it numbers the attempts from 1, and an attempt that reached
    its hold left one meta carrying its number. The fewest attempts those
    numbers allow, less the metas, ended before the hold. That is a floor:
    an attempt after the last meta of its take left nothing to count, nor did
    a take of which no attempt reached the hold."""
    path = run_dir / BEFORE_HOLD_LOG
    lines = (path.read_text(encoding="utf-8").splitlines()
             if path.is_file() else [])
    logged = [{"run": label, "grasp": str(d["pose"]), "take": d.get("take"),
               "number": d.get("attempts")}
              for d in (json.loads(s) for s in lines if s.strip())]
    numbers: dict = {}                # (grasp, take) -> how often each number
    for a in held + logged:
        if isinstance(a["number"], int):
            numbers.setdefault((a["grasp"], a["take"]),
                               Counter())[a["number"]] += 1
    unseen = []
    for (grasp, _), n in numbers.items():
        # a number seen c times: c starts of the take got at least that far
        least = sum(max(n[k] for k in range(j, max(n) + 1))
                    for j in range(1, max(n) + 1))
        unseen += [{"run": label, "grasp": grasp}] * (least - sum(n.values()))
    return logged + unseen


def audit_rows(run_dir: Path, label: str) -> list:
    """The object gate's record (second amendment of 2026-10-23), one row per
    raw IR audit still <run>/audit/<grasp>_*_L.png and per line of
    <run>/audit/confirmations.txt whose second field is the grasp."""
    audit = run_dir / "audit"
    rows = [{"run": label, "grasp": g, "kind": "still"}
            for g in GRASPS for _ in audit.glob(f"{g}_*_L.png")]
    path = audit / "confirmations.txt"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            words = line.split()
            if len(words) > 1 and words[1] in GRASPS:
                rows.append({"run": label, "grasp": words[1],
                             "kind": "confirmation"})
    return rows


# --- the summary --------------------------------------------------------------------
def summarise(level: str, run: str, condition: str, grasp: str, takes: list,
              expected: int, attempts: list, audits=(), before=()) -> dict:
    """One row over `takes` (any status): every take counted, only the valid
    ones measured; every attempt of the group that reached its hold, kept or
    refused, whatever its take's validity; its object gate's stills and
    confirmations; and how many attempts ended before the hold (`before`)."""
    valid = [t for t in takes if t["status"] == OK]
    invalid = sum(t["status"].startswith(INVALID) for t in takes)
    refused = [a["why"] for a in attempts if not a["accepted"]]
    reasons = [next((key for key, words in REJECTED_BY if words in why),
                    REJECTED_OTHER) for why in refused]
    covered = [float(a["coverage"]) for a in attempts
               if isinstance(a["coverage"], (int, float))]
    # the recorder's own camera numbers over the hold, valid takes only
    tracked = [float(t["cam_tracked_fraction"]) for t in valid
               if isinstance(t["cam_tracked_fraction"], (int, float))]
    reacq = [int(t["cam_reacquisitions"]) for t in valid
             if isinstance(t["cam_reacquisitions"], int)]
    policies = [t["acceptance_policy"] or STRICT for t in valid]
    row = {"level": level, "run": run, "condition": condition, "grasp": grasp,
           "n_valid_takes": len(valid), "n_expected_takes": expected,
           "n_invalid_takes": invalid,
           "n_unusable_takes": len(takes) - len(valid) - invalid,
           "n_missing_takes": max(0, expected - len(takes)),
           "n_attempts": len(attempts), "n_rejected": len(refused),
           **{key: reasons.count(key)
              for key in [k for k, _ in REJECTED_BY] + [REJECTED_OTHER]},
           "n_attempts_before_hold": len(before),
           "attempt_coverage_median": median(covered) if covered else None,
           "attempt_coverage_min": min(covered) if covered else None,
           "cam_tracked_fraction_median": median(tracked) if tracked else None,
           "cam_tracked_fraction_min": min(tracked) if tracked else None,
           "cam_reacquisitions_worst": max(reacq) if reacq else None,
           "n_strict_policy_takes": policies.count(STRICT),
           "n_keep_all_takes": policies.count(KEEP_ALL),
           "n_audit_stills": sum(a["kind"] == "still" for a in audits),
           "n_object_confirmations": sum(a["kind"] == "confirmation"
                                         for a in audits)}
    if not valid:
        return row
    per = [t[f"absdiff_med_{j}"] for t in valid for j in JOINTS]
    paired = [t["n_pairs"] / t["n_cam_frames_window"] for t in valid]
    row.update({
        "absdiff_median": median(per), "absdiff_largest": max(per),
        "paired_fraction_median": median(paired),
        "paired_fraction_min": min(paired),
        "cam_coverage_median": median(t["cam_coverage"] for t in valid),
        "cam_coverage_min": min(t["cam_coverage"] for t in valid),
        "cam_gaps_over_100ms_total": sum(t["cam_gaps_over_100ms"]
                                         for t in valid),
        "cam_max_gap_ms_worst": max(t["cam_max_gap_ms"] for t in valid),
        "glove_coverage_median": median(t["glove_coverage"] for t in valid),
        "glove_coverage_min": min(t["glove_coverage"] for t in valid),
        "glove_max_gap_ms_worst": max(t["glove_max_gap_ms"] for t in valid),
        "cam_mad_max_median": median(t["cam_mad_max"] for t in valid),
        "glove_mad_max_median": median(t["glove_mad_max"] for t in valid),
        "n_wall_time_clock": sum(t["clock"] == WALL_CLOCK for t in valid),
        "n_later_last_stamp": sum(t["window_end_source"] == LAST_STAMP
                                  for t in valid),
        **{f"absdiff_{j}": median(t[f"absdiff_med_{j}"] for t in valid)
           for j in JOINTS},
        **{f"signed_{j}": median(t[f"signed_med_{j}"] for t in valid)
           for j in JOINTS},
    })
    return row


def glove_drift(grasp: str, a: list, a2: list) -> dict:
    """A2 minus A on the glove's own medians, signed, degrees."""
    d = {j: median(t[f"glove_med_{j}"] for t in a2)
         - median(t[f"glove_med_{j}"] for t in a) for j in JOINTS}
    size = [abs(v) for v in d.values()]
    return {"level": "glove_drift_A2_minus_A", "run": "A2-A", "condition": "A",
            "grasp": grasp, "n_valid_takes": f"{len(a)}+{len(a2)}",
            "absdiff_median": median(size), "absdiff_largest": max(size),
            **{f"signed_{j}": v for j, v in d.items()}}


def contrasts(grasp: str, rows: list) -> list:
    """Differences of the PRIMARY quantity between runs; B is B1+B2 pooled."""
    v = {r["run"]: r.get("absdiff_median") for r in rows}
    out = {name: v[a] - v[b] for name, a, b in (
        ("Bpooled_minus_A", "B", "A"), ("C_minus_Bpooled", "C", "B"),
        ("B2_minus_B1", "B2", "B1"), ("A2_minus_A", "A2", "A"))
        if v.get(a) is not None and v.get(b) is not None}
    if "B2_minus_B1" in out and "C_minus_Bpooled" in out:
        out["abs_B2_minus_B1_minus_abs_C_minus_Bpooled"] = (
            abs(out["B2_minus_B1"]) - abs(out["C_minus_Bpooled"]))
    return [{"level": "contrast", "run": name, "condition": "", "grasp": grasp,
             "absdiff_median": value} for name, value in out.items()]


def grid_grasps(rows: list) -> list:
    """GRASPS, then any other grasp the takes or attempts hold (a dry
    run's)."""
    return GRASPS + sorted({t["grasp"] for t in rows} - set(GRASPS))


def summary_rows(takes: list, attempts: list, audits=(), before=()) -> list:
    """Every run x grasp of the grid, recorded or not: every take counted,
    only the valid ones measured, every attempt (those that reached the hold
    and, in `before`, those that did not) and audit row counted."""
    grasps = grid_grasps(takes + attempts + list(before))

    def tried(runs, grasp=None, rows=attempts):
        return [t for t in rows
                if t["run"] in runs and grasp in (None, t["grasp"])]

    def pick(runs, grasp=None, valid=False):
        return [t for t in takes
                if t["run"] in runs and grasp in (None, t["grasp"])
                and (t["status"] == OK or not valid)]

    def scope(grasp, level, pooled_level):
        # "all" pools every take of the run, extra grasps' included, but
        # expects only the protocol's grid: an extra grasp's takes never make
        # up for a missing protocol grasp. An extra grasp expects nothing.
        one = None if grasp == "all" else grasp
        factor = (len(GRASPS) if grasp == "all"
                  else 1 if grasp in GRASPS else 0)
        rows = [summarise(level, label, condition, grasp, pick({label}, one),
                          EXPECTED[label] * factor, tried({label}, one),
                          tried({label}, one, audits),
                          tried({label}, one, before))
                for label, _, condition, _ in RUNS]
        rows.append(summarise(pooled_level, "B", "B", grasp,
                              pick({"B1", "B2"}, one), EXPECTED["B"] * factor,
                              tried({"B1", "B2"}, one),
                              tried({"B1", "B2"}, one, audits),
                              tried({"B1", "B2"}, one, before)))
        return rows

    per_grasp = {g: scope(g, "run_grasp", "run_grasp_pooledB")
                 for g in grasps}
    pooled = scope("all", "run_all", "run_all_pooledB")
    rows = [r for g in grasps for r in per_grasp[g] if r["level"] == "run_grasp"]
    rows += [r for g in grasps for r in per_grasp[g]
             if r["level"] == "run_grasp_pooledB"]
    rows += pooled
    for g in grasps:
        a, a2 = pick({"A"}, g, valid=True), pick({"A2"}, g, valid=True)
        if a and a2:
            rows.append(glove_drift(g, a, a2))
    for g in grasps:
        rows += contrasts(g, per_grasp[g])
    rows += contrasts("all", pooled)
    return rows


# --- writing ----------------------------------------------------------------------------
def cell(key: str, value):
    """Degrees to 2 decimals, coverage, fractions and times to 3, "" when not
    computed."""
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 3 if ("coverage" in key or "fraction" in key
                                  or key.startswith("window_")) else 2)
    return value


def write_csv(path: Path, columns: list, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([cell(k, r.get(k)) for k in columns])


LABEL_W, COL_W = 54, 11
TABLE_ROWS = [  # (label, summary column, format) after the per-joint rows
    ("PRIMARY: median of take x joint |camera - glove|", "absdiff_median",
     "{:.1f}"),
    ("largest take x joint |camera - glove|", "absdiff_largest", "{:.1f}"),
    ("paired fraction median", "paired_fraction_median", "{:.2f}"),
    ("paired fraction min", "paired_fraction_min", "{:.2f}"),
    ("camera coverage median", "cam_coverage_median", "{:.2f}"),
    ("camera coverage min", "cam_coverage_min", "{:.2f}"),
    ("camera gaps >100 ms (sum)", "cam_gaps_over_100ms_total", "{}"),
    ("camera worst gap ms", "cam_max_gap_ms_worst", "{:.0f}"),
    ("glove coverage median", "glove_coverage_median", "{:.2f}"),
    ("glove coverage min", "glove_coverage_min", "{:.2f}"),
    ("glove worst gap ms", "glove_max_gap_ms_worst", "{:.0f}"),
    ("camera stability: worst-joint MAD, median over takes",
     "cam_mad_max_median", "{:.1f}"),
    ("glove stability: worst-joint MAD, median over takes",
     "glove_mad_max_median", "{:.1f}"),
    ("takes on wall_time clock", "n_wall_time_clock", "{}"),
    ("takes with window end from last stamp", "n_later_last_stamp", "{}"),
]


def table(cols: list) -> list:
    """Fixed-width rows, one column per (header, summary row); "-" where a
    column has no valid take."""
    lines = [" " * LABEL_W + "".join(f"{h:>{COL_W}}" for h, _ in cols)]

    def line(label, key, fmt):
        lines.append(f"{label:<{LABEL_W}}" + "".join(
            f"{'-' if r.get(key) is None else fmt.format(r[key]):>{COL_W}}"
            for _, r in cols))

    lines.append(f"{'takes valid / expected':<{LABEL_W}}" + "".join(
        f"{str(r['n_valid_takes']) + '/' + str(r['n_expected_takes']):>{COL_W}}"
        for _, r in cols))
    line("takes invalid", "n_invalid_takes", "{}")
    line("takes unusable", "n_unusable_takes", "{}")
    line("takes missing", "n_missing_takes", "{}")
    line("attempts (kept + rejected)", "n_attempts", "{}")
    line("rejected: hand lost", "n_rejected_hand_lost", "{}")
    line("rejected: re-acquired", "n_rejected_reacquired", "{}")
    line("rejected: low coverage", "n_rejected_low_coverage", "{}")
    line("rejected: other", "n_rejected_other", "{}")
    line("attempts ended before the hold (no file)", "n_attempts_before_hold",
         "{}")
    line("coverage over all attempts median", "attempt_coverage_median",
         "{:.2f}")
    line("coverage over all attempts min", "attempt_coverage_min", "{:.2f}")
    line("camera tracked fraction of hold median",
         "cam_tracked_fraction_median", "{:.2f}")
    line("camera tracked fraction of hold min", "cam_tracked_fraction_min",
         "{:.2f}")
    line("camera re-acquisitions worst", "cam_reacquisitions_worst", "{}")
    label = "takes under strict / keep-all policy"
    lines.append(f"{label:<{LABEL_W}}" + "".join(
        f"{r['n_strict_policy_takes']}/{r['n_keep_all_takes']}".rjust(COL_W)
        for _, r in cols))
    line("object audit stills (raw IR, per grasp)", "n_audit_stills", "{}")
    line("object confirmations typed", "n_object_confirmations", "{}")
    lines.append("per joint: median over takes of the per-take median "
                 "|camera - glove|, deg")
    for j in JOINTS:
        line(j, f"absdiff_{j}", "{:.1f}")
    lines.append("per joint: median over takes of the per-take median of "
                 "camera - glove, signed, deg")
    for j in JOINTS:
        line(j, f"signed_{j}", "{:+.1f}")
    for label, key, fmt in TABLE_ROWS:
        line(label, key, fmt)
    return lines


def contrast_lines(rows: list) -> list:
    return (["-- contrasts on the PRIMARY quantity, deg (numbers only) --"]
            + ([f"{r['run']:<{LABEL_W}}{r['absdiff_median']:>+{COL_W}.1f}"
                for r in rows] or ["none"]))


def report(root, window_s, max_dt, run_lines, takes, summary) -> str:
    def level(name, grasp=None):
        return [r for r in summary
                if r["level"] == name and grasp in (None, r["grasp"])]

    invalid = [f"{t['run']} {t['grasp']} take {t['take']}: {t['status']}"
               for t in takes if t["status"].startswith(INVALID)]
    lines = [
        TITLE,
        f"root: {root if root is not None else '(none: runs given by --run)'}",
        f"written: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"window: last {window_s:g} s before the window end, on the clock "
        "named per take",
        "window end: the recorder's hold_end from the take's meta; else the "
        "later of the two sensors' last stamps",
        f"pairing: each camera frame of the hand -> nearest glove frame "
        f"within {max_dt * 1000:g} ms, no lag shift",
        "angles: xr_hand.joint_frames, wrist reference, flexion positive "
        "toward the palm",
        "disagreement: camera minus glove per pair, wrapped to [-180, 180); "
        "absolute = |camera - glove|; the glove is a reference, not truth",
        f"coverage: leap_hand.protocol.coverage (holes longer than "
        f"{LOSS_GAP_S:g} s) and gaps over 100 ms, per sensor, inside the "
        "window",
        "attempts: every attempt that reached its hold, kept or rejected "
        "(rejected/leap), with the recorder's own coverage of each",
        "attempts ended before the hold: no open hand acquired in time, or "
        "the hand lost during the settle; no take file and no meta. Counted "
        f"from the recorder's {BEFORE_HOLD_LOG} (amendment of 2026-11-17) "
        "and, for a take recorded before it, from the attempt numbers in the "
        "take's metas, which is a floor: an attempt after the last meta of "
        "its take left nothing to count",
        f"policy: the recorder's acceptance_policy per valid take, {STRICT} "
        f"when the meta has none; {KEEP_ALL} (amendment of 2026-10-23) keeps "
        "every completed hold, so its losses show in the camera tracked "
        "fraction of the hold, not as rejected attempts",
        "", "runs:", *run_lines,
        "", "invalid takes:", *(invalid or ["none"]),
    ]
    grasps = list(dict.fromkeys(r["grasp"] for r in level("run_grasp")))
    for g in grasps:
        cols = ([(r["run"], r) for r in level("run_grasp", g)]
                + [("B(pooled)", r) for r in level("run_grasp_pooledB", g)])
        lines += ["", f"== {g} ==", *table(cols), "",
                  *contrast_lines(level("contrast", g))]

    lines += ["", "== glove drift, A2 minus A, glove alone (deg) =="]
    drift = level("glove_drift_A2_minus_A")
    if drift:
        lines.append(" " * LABEL_W + "".join(f"{r['grasp']:>20}" for r in drift))
        rows = ([("takes (A+A2)", "n_valid_takes", "{}")]
                + [(j, f"signed_{j}", "{:+.1f}") for j in JOINTS]
                + [("median of absolute values", "absdiff_median", "{:.1f}"),
                   ("largest absolute value", "absdiff_largest", "{:.1f}")])
        for label, key, fmt in rows:
            lines.append(f"{label:<{LABEL_W}}" + "".join(
                f"{fmt.format(r[key]):>20}" for r in drift))
    lines += [f"{g}: needs valid takes in both A and A2" for g in grasps
              if not level("glove_drift_A2_minus_A", g)]

    cols = ([(r["run"], r) for r in level("run_all")]
            + [("B(pooled)", r) for r in level("run_all_pooledB")])
    lines += ["", "== all grasps pooled (secondary) ==", *table(cols), "",
              *contrast_lines(level("contrast", "all"))]
    return "\n".join(lines) + "\n"


# --- the command ----------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Camera against glove, flexion of 8 joints, per take of "
                    "the hand-on-object pilot (docs/pilot_hand_on_object.md).")
    p.add_argument("--root", type=Path,
                   help="the pilot folder holding A_mimed, B1_object, "
                        "C_object_hint, B2_object, A_mimed_again")
    p.add_argument("--out", type=Path,
                   help="where takes.csv, summary.csv and report.txt go "
                        "(default <root>\\comparison)")
    p.add_argument("--window", type=float, default=3.0,
                   help="seconds before the window end (default 3.0)")
    p.add_argument("--max-dt", type=float, default=0.05,
                   help="largest camera-glove time difference of a pair, "
                        "seconds (default 0.05)")
    p.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                   help="read run LABEL (A, B1, C, B2, A2) from PATH instead "
                        "of its folder under --root; repeatable")
    a = p.parse_args(argv)

    labels = [r[0] for r in RUNS]
    given = {}
    for spec in a.run:
        label, sep, path = spec.partition("=")
        if not sep or not path or label not in labels:
            p.error(f"--run takes LABEL=PATH with LABEL one of "
                    f"{', '.join(labels)}, got {spec!r}")
        given[label] = Path(path)
    if a.root is None and not given:
        p.error("give --root, or every run with --run LABEL=PATH")
    out = a.out or ((a.root if a.root is not None
                     else next(iter(given.values())).parent) / "comparison")

    takes, attempts, before, audits, run_lines, unkept = [], [], [], [], [], []
    for label, folder, condition, expected in RUNS:
        if label in given:
            run_dir, name = given[label], str(given[label])
        else:
            run_dir = a.root / folder if a.root is not None else None
            name = folder
        metas = accepted_metas(run_dir) if run_dir is not None else []
        # counted even for a run with no take kept: its refusals are the data
        tried = attempt_rows(run_dir, label) if run_dir is not None else []
        early = (before_hold_rows(run_dir, label, tried)
                 if run_dir is not None else [])
        attempts += tried
        before += early
        if run_dir is not None:
            audits += audit_rows(run_dir, label)
        if not metas:
            # "absent" only when nothing was attempted: a run the camera lost
            # every time was recorded, and must not read as never recorded
            made = [f"{len(rows)} attempt{'' if len(rows) == 1 else 's'}{how}"
                    for rows, how in ((tried, ", all rejected"),
                                      (early, " ended before the hold"))
                    if rows]
            said = (f"{label} ({name}): 0 takes kept ({'; '.join(made)})"
                    if made else f"{label} ({name}): absent")
            run_lines.append(said)
            unkept.append(f"run {said}")
            continue
        got = [compare_take(run_dir, label, condition, expected, m,
                            a.window, a.max_dt) for m in metas]
        takes += got
        n_ok = sum(t["status"] == OK for t in got)
        run_lines.append(f"{label} ({name}): {len(got)} take"
                         f"{'' if len(got) == 1 else 's'} ({n_ok} valid)")

    # Nothing to write only when no run folder shows any attempt at all:
    # attempts with no take kept are a result (the camera lost the hand every
    # time), whether or not they got as far as the hold.
    if not takes and not attempts and not before:
        where = (a.root if a.root is not None
                 else ", ".join(str(v) for v in given.values()))
        print(f"nothing to compare under {where}")
        for s in unkept:
            print(s)
        return 1

    for t in takes:
        head = f"{t['run']} {t['grasp']} take {t['take']}: {t['status']}"
        if t["status"] != OK:
            print(head)
            continue
        print(f"{head}, clock {t['clock']}, {t['n_pairs']} pairs, camera "
              f"coverage {t['cam_coverage']:.2f}, median |camera - glove| "
              f"{t['absdiff_med_all8']:.1f} deg, largest "
              f"{t['absdiff_max_all8']:.1f} deg, window end "
              f"{t['window_end_source']}")
    for s in unkept:
        print(s)

    summary = summary_rows(takes, attempts, audits, before)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "takes.csv", TAKE_COLUMNS, takes)
    write_csv(out / "summary.csv", SUMMARY_COLUMNS, summary)
    (out / "report.txt").write_text(
        report(a.root, a.window, a.max_dt, run_lines, takes, summary),
        encoding="ascii", errors="replace")
    n_ok = sum(t["status"] == OK for t in takes)
    print(f"compared {n_ok} of {len(takes)} takes; wrote "
          f"{out / 'takes.csv'}, summary.csv, report.txt")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
