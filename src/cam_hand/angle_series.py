"""The glove's joint angles over a take: frame by frame, and at each held cue.

Sets B and C are movements, so one summary frame per take (the joint-frame
table of Set A) says nothing about them. What the work downstream of these
recordings needs is every joint's angle at every instant, with the cue that
was in force: the angles are checked against anatomical limits, and the
single-finger takes and the sequences are where one finger's bend shows in
its neighbours (the ring pulling the little finger along). Two tables come
from a take's glove file and its events file, and nothing else:

  frames   one row per glove frame of the recorded hand: the frame's time,
           the cue in force at that instant, and the paper's 24 named angles
           in degrees (`xr_hand.joint_frames.dof24`, the same table and the
           same signs as the grasp set's joint_angles_24.csv)
  cues     one row per HELD cue: the median of each angle over the cue's
           window. Set B: the hold and the rest of every cycle, over the
           whole phase (the bend and the straighten are movements, and have
           no row). Set C: every step, over the checker's own window, the
           last check_window_s of its hold (`protocol_check.step_windows`),
           so a row here and the step's row in sequence_check.csv describe
           the same frames.

A frame's cue is the last cue whose beep was at or before the frame's
`wall_time` (the events file's `t` is the same clock). A frame before the
first cue or at or after `take_end` has no cue: its cue columns are empty.
The glove's own delay is not removed: a cue's first frames still show the
pose before it (0.1 to 0.5 s on these gloves), which is why the Set C window
is the end of the hold.

WHAT THE GLOVE DOES NOT MEASURE is in `MODEL_ONLY`. The glove streams finger
flexion only, so the nine of the 24 that are not a joint's flexion are its
hand model's values: the four finger CMC, and the four MCP abductions and the
thumb's TMC abduction. They are written as streamed, the hand-in README names
them and prints the span each of the 24 took in the folder, and a limit
check on those nine checks the model. The other fifteen (`FLEXIONS`) are the
flexions the stream carries; the thumb's TMC flexion is one of them, though
the glove has no sensor for opposition, so a thumb crossing the palm reads as
a thumb bending.

ONE VALUE PER DIGIT (`one_value`). On the real glove the three flexions of a
digit are not three measurements: grouped by their sum, each joint sits
within about 0.1 degree of one value (the glove take of 2026-10-01, sums
moving 51 to 157 degrees). XR Trainer's hand model spreads one flexion per
digit over the joints, so these tables show how far one finger goes along
with another, and not how the joints of one finger depend on each other.
The hand-in README prints the measurement for every set, hand and digit.
"""
import csv
import json
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from xr_hand import joint_frames as jf

ANGLES: List[str] = list(jf.DOF24_NAMES)
# Of the 24: what the glove's hand model supplies without a sensor behind it.
MODEL_ONLY: Tuple[str, ...] = ("T_TMC_aa", "I_CMC", "I_MCP_aa", "M_CMC",
                               "M_MCP_aa", "R_CMC", "R_MCP_aa", "L_CMC",
                               "L_MCP_aa")
FLEXIONS: Tuple[str, ...] = tuple(a for a in ANGLES if a not in MODEL_ONLY)

# The flexions of each digit, base to tip.
DIGIT_FLEXIONS: Dict[str, Tuple[str, ...]] = {
    "thumb": ("T_TMC_fe", "T_MCP_fe", "T_IP"),
    "index": ("I_MCP_fe", "I_PIP", "I_DIP"),
    "middle": ("M_MCP_fe", "M_PIP", "M_DIP"),
    "ring": ("R_MCP_fe", "R_PIP", "R_DIP"),
    "little": ("L_MCP_fe", "L_PIP", "L_DIP")}
ONE_VALUE_STEP = 1.0         # degrees of summed flexion per group
ONE_VALUE_MIN_SPAN = 10.0    # the sum must move this far to be judged
ONE_VALUE_MIN_GROUP = 3      # frames a group needs to count

CUE_COLUMNS = ["cue_step", "cue_label", "cue_phase", "cue_cycle", "flexed"]
FRAME_COLUMNS = ["frame_id", "wall_time", "t_s"] + CUE_COLUMNS + ANGLES
CUE_ROW_COLUMNS = (["session", "take_name", "item", "take", "step", "label",
                    "phase", "cycle", "flexed", "cue_t_s", "window_start_s",
                    "window_end_s", "frames"] + ANGLES)

FLEXION = "finger_flexion"
SEQUENCES = "sequences"
# Set B phases that are held poses (the other two are the movements).
HELD_PHASES = ("hold", "rest")
ANGLE_DIGITS = 2
TIME_DIGITS = 4


def read_glove_lines(path, hand: str) -> Tuple[List[dict], int]:
    """The glove file's lines of `hand`, in file order, and the count of
    lines that could not be read (a take cut off mid-write)."""
    out, bad = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for text in f:
            if not text.strip():
                continue
            try:
                d = json.loads(text)
                float(d["wall_time"])
                d["joints"][0]["name"]
            except (ValueError, KeyError, TypeError, IndexError):
                bad += 1
                continue
            if d.get("hand_side") == hand:
                out.append(d)
    return out, bad


def cues_of(events: Sequence[dict]) -> List[dict]:
    """The take's cues in the order they sounded."""
    cues = [e for e in events if e.get("kind") == "cue" and "t" in e]
    return sorted(cues, key=lambda e: (float(e["t"]), e.get("step", 0)))


def _event_time(events: Sequence[dict], kind: str) -> Optional[float]:
    for e in events:
        if e.get("kind") == kind and "t" in e:
            return float(e["t"])
    return None


def take_origin(events: Sequence[dict], lines: Sequence[dict]
                ) -> Optional[float]:
    """Time zero of `t_s`: the take_start event, else the first cue, else
    the first frame."""
    t = _event_time(events, "take_start")
    if t is not None:
        return t
    cues = cues_of(events)
    if cues:
        return float(cues[0]["t"])
    return float(lines[0]["wall_time"]) if lines else None


def _flexed_text(cue: dict) -> str:
    return " ".join(str(f) for f in (cue.get("flexed") or []))


def _cue_cells(cue: Optional[dict]) -> Dict[str, object]:
    if cue is None:
        return {k: "" for k in CUE_COLUMNS}
    return {"cue_step": cue.get("step", ""),
            "cue_label": cue.get("label", ""),
            "cue_phase": cue.get("phase", ""),
            "cue_cycle": cue.get("cycle", ""),
            "flexed": _flexed_text(cue)}


def angles_of(line: dict, hand: str) -> Dict[str, float]:
    """The 24 named angles of one glove line, degrees."""
    return jf.dof24(jf.joint_table(line, "glove", hand=hand))


def frame_rows(lines: Sequence[dict], events: Sequence[dict], hand: str
               ) -> List[dict]:
    """One row per glove line: its time, its cue, its 24 angles (floats)."""
    cues = cues_of(events)
    end = _event_time(events, "take_end")
    origin = take_origin(events, lines)
    rows = []
    k = -1                                   # the cue in force
    for d in sorted(lines, key=lambda d: float(d["wall_time"])):
        t = float(d["wall_time"])
        while k + 1 < len(cues) and float(cues[k + 1]["t"]) <= t:
            k += 1
        cue = cues[k] if k >= 0 and (end is None or t < end) else None
        row = {"frame_id": d.get("frame_id", ""), "wall_time": t,
               "t_s": t - origin}
        row.update(_cue_cells(cue))
        row.update(angles_of(d, hand))
        rows.append(row)
    return rows


def held_windows(events: Sequence[dict], kind: str, hold_default: float,
                 window_s: float) -> List[dict]:
    """Every held cue of a take with the window its angles are read over.

    Set C: each step, the checker's window (the same share of the hold as
    `protocol_check.step_windows`, whatever the run's time scale). Set B:
    each hold and each rest, the whole phase.
    """
    out = []
    for k, e in enumerate(cues_of(events)):
        t = float(e["t"])
        hold = float(e.get("hold_s") or hold_default)
        if kind == SEQUENCES:
            share = (min(1.0, window_s / hold_default)
                     if hold_default > 0 else 1.0)
            w1 = t + hold
            w0 = w1 - share * hold
        elif e.get("phase") in HELD_PHASES:
            w0, w1 = t, t + hold
        else:
            continue
        out.append({"cue": e, "step": e.get("step", k), "t": t,
                    "window": (w0, w1)})
    return out


def cue_rows(rows: Sequence[dict], events: Sequence[dict], kind: str,
             hold_default: float, window_s: float) -> List[dict]:
    """One row per held cue: the median of each angle over the frames whose
    time is inside the cue's window (start included, end excluded). A window
    with no frame keeps its row, with 0 frames and empty angles."""
    origin = (rows[0]["wall_time"] - rows[0]["t_s"]) if rows else None
    out = []
    for w in held_windows(events, kind, hold_default, window_s):
        w0, w1 = w["window"]
        inside = [r for r in rows if w0 <= r["wall_time"] < w1]
        cue = w["cue"]
        base = origin if origin is not None else w["t"]
        row = {"step": w["step"], "label": cue.get("label", ""),
               "phase": cue.get("phase", ""), "cycle": cue.get("cycle", ""),
               "flexed": _flexed_text(cue), "cue_t_s": w["t"] - base,
               "window_start_s": w0 - base, "window_end_s": w1 - base,
               "frames": len(inside)}
        for a in ANGLES:
            row[a] = median(r[a] for r in inside) if inside else ""
        out.append(row)
    return out


def one_value(rows: Sequence[dict], digit: str
              ) -> Optional[Tuple[float, str, float, int]]:
    """Does one number per digit give all of that digit's flexions?

    The frames are grouped by the digit's summed flexion in steps of
    ONE_VALUE_STEP degrees. Inside a group of ONE_VALUE_MIN_GROUP frames or
    more, each joint's flexion is compared with the group's mean. Returns
    (the RMS of those differences in degrees for the joint where it is
    largest, that joint, the span of the sum in degrees, the frames that
    counted); None when the sum moved less than ONE_VALUE_MIN_SPAN degrees
    or under half of the frames are in a group that counts.

    When the RMS is a fraction of a degree, the digit's three flexions are
    one measurement spread over three joints by the glove's hand model:
    how its joints bend relative to each other is then the model's, and
    cannot be learned from the recording. Different digits are not tied
    this way.
    """
    import numpy as np
    cols = DIGIT_FLEXIONS[digit]
    if not rows:
        return None
    a = np.asarray([[r[c] for c in cols] for r in rows], dtype=float)
    total = a.sum(axis=1)
    span = float(np.ptp(total))
    if span < ONE_VALUE_MIN_SPAN:
        return None
    group = np.floor(total / ONE_VALUE_STEP).astype(int)
    squares = np.zeros(len(cols))
    counted = 0
    for g in np.unique(group):
        inside = a[group == g]
        if len(inside) < ONE_VALUE_MIN_GROUP:
            continue
        squares += ((inside - inside.mean(axis=0)) ** 2).sum(axis=0)
        counted += len(inside)
    if counted * 2 < len(rows):
        return None
    rms = np.sqrt(squares / counted)
    worst = int(np.argmax(rms))
    return float(rms[worst]), cols[worst], span, counted


def _cell(key: str, value) -> str:
    if value == "" or value is None:
        return ""
    if key in ANGLES:
        return f"{float(value):.{ANGLE_DIGITS}f}"
    if key == "wall_time":
        return f"{float(value):.6f}"
    if key in ("t_s", "cue_t_s", "window_start_s", "window_end_s"):
        return f"{float(value):.{TIME_DIGITS}f}"
    return str(value)


def write_csv(dest, columns: Sequence[str], rows: Sequence[dict]) -> int:
    with open(dest, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_cell(c, r.get(c, "")) for c in columns])
    return len(rows)


def take_rows(glove_path, events: Sequence[dict], hand: str) -> List[dict]:
    """The per-frame rows of one take's glove file."""
    lines, _bad = read_glove_lines(Path(glove_path), hand)
    return frame_rows(lines, events, hand)


def write_frames(dest, glove_path, events: Sequence[dict], hand: str) -> int:
    """`joint_angles/<take>.csv`: every frame of the take."""
    return write_csv(dest, FRAME_COLUMNS, take_rows(glove_path, events, hand))
