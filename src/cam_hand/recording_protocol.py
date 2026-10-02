"""The professor's finger-flexion and sequence protocols (Sets B and C), as values.

`scripts/record_protocol.py` runs these sessions with the gloves on and the
camera as the reference, and `scripts/check_protocol.py` and
`scripts/package_professor_set.py` read what it writes. The three are built
separately against `docs/protocol_formats.md`, so everything the recorder
decides that a later tool has to agree with lives here, as plain functions
with tests, rather than inside a loop that drives hardware:

  protocol files   `protocols/*.json`, loaded and validated with an error
                   message that names the file, the item and the field.
                   Sequences are data, not code: the professor's "try other
                   orders" is a new JSON entry, and a typo in one must stop
                   the session before the gloves go on, not halfway through.
  cue schedule     every cue of a take, in order, with its words, the set of
                   fingers that should be flexed and how long it lasts. Set B
                   cues every phase of every cycle (bend, hold, straighten,
                   rest) so a missed cue, fatigue or creep can be located to
                   the cycle; Set C cues every step.
  rounds           the order the takes are recorded in. Set C runs all seven
                   sequences once per round in a shuffled order, so the three
                   takes of a sequence are independent repetitions and
                   practice or fatigue cannot pass for a glove effect. The
                   order is a pure function of the seed, and the seed is saved.
  events           one JSON line per cue and per decision, stamped with
                   `time.time()`, the clock the frame files use.
  warm-up          each finger's own open and fist reading for this session,
                   as medians over the two cued windows, then each finger
                   bent on its own (`single`). A glove that barely moves
                   between open and fist has no range to measure fractions
                   against, and a finger that barely bends on its own has no
                   range for its own take, so either refuses the warm-up.
  fraction         `(open - value) / (open - end)`: 0 at the warm-up open
                   palm, 1 at the end of the finger's range. The end is the
                   finger's own single bend when it is usable (`_endpoints`)
                   for a finger the cue names, and the fist otherwise. Every
                   band below is in these units, so one number means the same
                   thing on every finger.
  quick check      the acquisition gates the recorder applies before it
                   accepts a take (plan, section 4). Set B: the cued finger
                   has to cover at least 60 % of its range, and has to bend
                   fully once per bend cue: its curl peaks (`count_peaks`,
                   the count `cam_hand.protocol_check` fails a take on, which
                   imports it from here) must number the take's bend cues,
                   so a take is redone while the gloves are on rather than
                   failed by the checker after the session. Set C: in the
                   last `check_window_s` of every step, a finger the step
                   says is flexed has to read above 0.6 and a finger the step
                   says is straight must not read flexed. A straight finger
                   between 0.3 and 0.6 is reported as coupling and does not
                   fail the take: the ring drags the middle and the little
                   finger along, and a hard cutoff would reject real motion.

The curl is the repo's one curl: `xr_hand.keypoints21.frame_to_keypoints21`
then `cam_hand.features.flexion_features`, fingertip-to-wrist over palm
length, higher = straighter, exactly as `leap_hand.pose_check` computes it.
The glove's index to pinky are measured by it. The glove's thumb is not
(since 2026-10-01): it is the sum of its three flexion angles in degrees,
larger = more bent (`glove_bends`), because the curl barely moves when the
thumb bends at its own joints. The camera keeps the curl for all five.
`read_bends` (and its old name `read_curls`) is the only function here that
touches a file.
"""
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from xr_hand.joint_frames import dof24, joint_table, line_from_hand_frame
from xr_hand.keypoints21 import frame_to_keypoints21

from .features import FLEXION_NAMES, flexion_features

FINGERS: Tuple[str, ...] = tuple(FLEXION_NAMES)   # thumb index middle ring pinky

# The words the operator reads for a finger. "little" is what the professor
# wrote; `pinky` stays the name in the data (`flexed`, `finger`).
FINGER_WORDS = {"thumb": "thumb", "index": "index", "middle": "middle",
                "ring": "ring", "pinky": "little finger"}

SETS = ("grasps", "finger_flexion", "sequences")
GRASPS, FLEXION, SEQUENCES = SETS

# Set B phases, in the order one cycle runs them.
BEND, HOLD, STRAIGHTEN, REST = "bend", "hold", "straighten", "rest"
PHASES = (BEND, HOLD, STRAIGHTEN, REST)

# --- the numbers the contract fixes (docs/protocol_formats.md) ---------------
# A glove span under this on any of these fingers refuses the session: a range
# that small turns every fraction into noise.
REFUSE_SPAN = 0.30
REFUSE_FINGERS = ("index", "middle", "ring", "pinky")
# Fractions are clipped here, so one wild frame cannot drag a median.
FRACTION_CLIP = (-0.5, 1.5)
# Set B acquisition gate: the cued finger spans at least this much of its range.
SPAN_GATE = 0.60
# Set C initial bands, in fractions of the finger's range.
FLEXED_ABOVE = 0.6
STRAIGHT_BELOW = 0.3
# A take's span is read between these percentiles, the same pair
# `scripts/leap/finger_sweep.py` uses for the glove, so one spike or one
# dropped frame does not make a finger look as if it moved.
SPAN_PERCENTILES = (5.0, 95.0)
# Peak counting (Set B, "cycles counted two ways"): a peak is a rise above
# PEAK_HIGH of the take's own range after a fall below PEAK_LOW of it. A
# finger whose range in the take is under PEAK_MIN_RANGE has no peaks. The
# checker imports these three and `count_peaks` from here, so the recorder
# and the checker count the same bends.
PEAK_LOW, PEAK_HIGH, PEAK_MIN_RANGE = 0.3, 0.6, 0.15
# A warm-up span this close to zero is not a range at all.
MIN_SPAN = 1e-6
# The single-finger part of the warm-up (each finger bent on its own). The
# fist is a different movement for the thumb (wrapped over the fingers) and
# lets a finger look fully bent with its neighbours' help, so a finger's own
# bend is the end of its range in the take rules wherever it is usable:
# index to pinky bent alone at least this share of their fist span ...
SINGLE_MIN_SHARE = 0.5
# ... and the thumb at least this many degrees (TMC_fe + MCP_fe + IP).
THUMB_MIN_DEG = 35.0
# The glove thumb's bend: these three of the paper's 24 angles, summed
# (`xr_hand.joint_frames.DOF24_SOURCE`).
THUMB_ANGLES = ("T_TMC_fe", "T_MCP_fe", "T_IP")
THUMB = "thumb"
# What the five glove numbers are, written to warmup.json as "units". A
# warm-up without it was recorded before the thumb was measured in degrees,
# and its glove files are read with the curl for all five.
GLOVE_UNITS = {
    "thumb": "degrees, TMC_fe + MCP_fe + IP, larger = more bent",
    "fingers": "curl, wrist to tip over palm length, smaller = more bent",
}

_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class ProtocolError(ValueError):
    """A protocol file that cannot be recorded, with what to fix."""


# --- protocol files ----------------------------------------------------------
@dataclass(frozen=True)
class Protocol:
    """One loaded, validated protocol file."""

    path: Path
    data: dict
    sha256: str

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def version(self) -> int:
        return int(self.data["version"])

    @property
    def items(self) -> List[dict]:
        return list(self.data["items"])

    @property
    def ids(self) -> List[str]:
        return [it["id"] for it in self.data["items"]]

    @property
    def takes_per_item(self) -> int:
        return int(self.data["takes_per_item"])

    @property
    def shuffle_rounds(self) -> bool:
        return bool(self.data.get("shuffle_rounds", False))

    def item(self, item_id: str) -> dict:
        for it in self.data["items"]:
            if it["id"] == item_id:
                return it
        raise ProtocolError(f"{self.path}: no item {item_id!r}; the items are "
                            f"{', '.join(self.ids)}")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def _positive(where: str, d: Mapping, key: str) -> None:
    if key not in d:
        raise ProtocolError(f"{where}: missing {key!r} (seconds, above 0)")
    v = d[key]
    if not _is_number(v) or float(v) <= 0:
        raise ProtocolError(f"{where}: {key!r} must be a number of seconds "
                            f"above 0, got {v!r}")


def _whole(where: str, d: Mapping, key: str, minimum: int = 1) -> None:
    if key not in d:
        raise ProtocolError(f"{where}: missing {key!r} (a whole number >= "
                            f"{minimum})")
    v = d[key]
    if not _is_int(v) or v < minimum:
        raise ProtocolError(f"{where}: {key!r} must be a whole number >= "
                            f"{minimum}, got {v!r}")


def _text(where: str, d: Mapping, key: str) -> None:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ProtocolError(f"{where}: {key!r} must be non-empty text, got "
                            f"{v!r}")


def _fingers(where: str, value, allow_empty: bool = True) -> None:
    if not isinstance(value, list):
        raise ProtocolError(f"{where}: must be a list of finger names, got "
                            f"{value!r}")
    if not value and not allow_empty:
        raise ProtocolError(f"{where}: must name at least one finger")
    for f in value:
        if f not in FINGERS:
            hint = " (the little finger is 'pinky' in the data)" \
                if f in ("little", "little finger") else ""
            raise ProtocolError(f"{where}: {f!r} is not a finger name{hint}; "
                                f"use {', '.join(FINGERS)}")
    if len(set(value)) != len(value):
        raise ProtocolError(f"{where}: names a finger twice: {value!r}")


def validate_protocol(data, where: str = "protocol") -> None:
    """Raise ProtocolError naming the first thing wrong with `data`.

    Checks what every tool downstream relies on: the shared header, unique
    file-name-safe ids, the per-set fields with their units, and finger names
    from the one list the data uses. A file that passes can be recorded,
    checked and packaged without any tool second-guessing it.
    """
    if not isinstance(data, dict):
        raise ProtocolError(f"{where}: the file must hold one JSON object")
    _text(where, data, "name")
    if data["name"] not in SETS:
        raise ProtocolError(f"{where}: 'name' must be one of {', '.join(SETS)} "
                            f"(it names the session folder), got "
                            f"{data['name']!r}")
    _whole(where, data, "version")
    if not isinstance(data.get("description"), str):
        raise ProtocolError(f"{where}: 'description' must be text")
    _whole(where, data, "takes_per_item")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ProtocolError(f"{where}: 'items' must be a non-empty list")
    kind = data["name"]
    if kind == SEQUENCES:
        _positive(where, data, "hold_s")
        _positive(where, data, "check_window_s")
        if float(data["check_window_s"]) > float(data["hold_s"]):
            raise ProtocolError(
                f"{where}: 'check_window_s' ({data['check_window_s']}) cannot "
                f"be longer than 'hold_s' ({data['hold_s']})")
        if not isinstance(data.get("shuffle_rounds"), bool):
            raise ProtocolError(f"{where}: 'shuffle_rounds' must be true or "
                                "false")
    if kind == GRASPS:
        _text(where, data, "status")
        _positive(where, data, "duration_s")
        _positive(where, data, "prep_s")

    seen = set()
    for i, it in enumerate(items):
        at = f"{where}: items[{i}]"
        if not isinstance(it, dict):
            raise ProtocolError(f"{at} must be an object")
        item_id = it.get("id")
        if not isinstance(item_id, str) or not _ID_RE.match(item_id):
            raise ProtocolError(
                f"{at}: 'id' must be file-name safe (lower-case letters, "
                f"digits and single underscores), got {item_id!r}")
        at = f"{at} ({item_id!r})"
        if item_id in seen:
            raise ProtocolError(f"{at}: the id is used twice")
        seen.add(item_id)
        _text(at, it, "label")
        if kind == FLEXION:
            if it.get("finger") not in FINGERS:
                raise ProtocolError(f"{at}: 'finger' must be one of "
                                    f"{', '.join(FINGERS)}, got "
                                    f"{it.get('finger')!r}")
            _whole(at, it, "cycles")
            for key in ("bend_s", "hold_s", "straighten_s", "rest_s"):
                _positive(at, it, key)
        elif kind == SEQUENCES:
            steps = it.get("steps")
            if not isinstance(steps, list) or not steps:
                raise ProtocolError(f"{at}: 'steps' must be a non-empty list")
            for k, step in enumerate(steps):
                sat = f"{at} step {k}"
                if not isinstance(step, dict):
                    raise ProtocolError(f"{sat} must be an object")
                _text(sat, step, "label")
                _fingers(f"{sat} 'flexed'", step.get("flexed"))
        else:
            for key in ("source", "figure"):
                if key not in it or not (it[key] is None
                                         or isinstance(it[key], str)):
                    raise ProtocolError(f"{at}: {key!r} must be text or null")
            _text(at, it, "shape")
            if not isinstance(it.get("object_implied"), bool):
                raise ProtocolError(f"{at}: 'object_implied' must be true or "
                                    "false")


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path) -> Protocol:
    """Read, parse and validate one protocol file. Raises ProtocolError."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ProtocolError(f"{path}: cannot be read ({e})") from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as e:
        raise ProtocolError(f"{path}: is not UTF-8 text ({e})") from None
    except json.JSONDecodeError as e:
        raise ProtocolError(f"{path}: is not valid JSON (line {e.lineno}, "
                            f"column {e.colno}: {e.msg})") from None
    validate_protocol(data, where=str(path))
    return Protocol(path=path, data=data,
                    sha256=hashlib.sha256(raw).hexdigest())


def select_items(protocol: Protocol,
                 wanted: Optional[Sequence[str]] = None) -> List[str]:
    """The item ids to record, in the protocol's order.

    `wanted` is the `--items a,b` subset. An id that is not in the file is an
    error that lists the ones that are: a silently empty session is worse
    than no session.
    """
    ids = protocol.ids
    if not wanted:
        return ids
    wanted = [w.strip() for w in wanted if w and w.strip()]
    unknown = [w for w in wanted if w not in ids]
    if unknown:
        raise ProtocolError(f"{protocol.path}: no item named "
                            f"{', '.join(unknown)}; the items are "
                            f"{', '.join(ids)}")
    return [i for i in ids if i in set(wanted)]


# --- the cue schedule --------------------------------------------------------
@dataclass(frozen=True)
class Cue:
    """One cue: a beep, words on the window, and what the hand should do.

    `hold_s` is how long this cue lasts until the next one (the contract's
    name for it, in both sets). `cycle` and `phase` are Set B only.
    """

    step: int
    label: str
    flexed: Tuple[str, ...]
    hold_s: float
    cycle: Optional[int] = None
    phase: Optional[str] = None

    def event_fields(self) -> dict:
        """The cue's fields as the events file writes them (contract 6)."""
        out = {"step": self.step, "label": self.label,
               "flexed": list(self.flexed), "hold_s": round(self.hold_s, 6)}
        if self.cycle is not None:
            out["cycle"] = self.cycle
            out["phase"] = self.phase
        return out


def _ordered(fingers: Iterable[str]) -> Tuple[str, ...]:
    """Finger names in the hand's order, whatever order they were given in."""
    s = set(fingers)
    return tuple(f for f in FINGERS if f in s)


def flexion_cues(item: Mapping, time_scale: float = 1.0) -> List[Cue]:
    """Set B: `cycles` x (bend, hold, straighten, rest), each its own cue."""
    finger = item["finger"]
    words = FINGER_WORDS[finger]
    durations = {BEND: item["bend_s"], HOLD: item["hold_s"],
                 STRAIGHTEN: item["straighten_s"], REST: item["rest_s"]}
    labels = {BEND: f"bend the {words}", HOLD: "hold",
              STRAIGHTEN: "straighten", REST: "rest"}
    out: List[Cue] = []
    for cycle in range(1, int(item["cycles"]) + 1):
        for phase in PHASES:
            flexed = (finger,) if phase in (BEND, HOLD) else ()
            out.append(Cue(step=len(out), label=labels[phase], flexed=flexed,
                           hold_s=float(durations[phase]) * time_scale,
                           cycle=cycle, phase=phase))
    return out


def sequence_cues(item: Mapping, hold_s: float,
                  time_scale: float = 1.0) -> List[Cue]:
    """Set C: one cue per step, each held the protocol's `hold_s`."""
    return [Cue(step=k, label=step["label"], flexed=_ordered(step["flexed"]),
                hold_s=float(hold_s) * time_scale)
            for k, step in enumerate(item["steps"])]


def build_schedule(protocol: Protocol, item_id: str,
                   time_scale: float = 1.0) -> List[Cue]:
    """Every cue of one take of `item_id`, in order.

    `time_scale` multiplies every duration. It exists for rehearsals and
    tests (`--time-scale 0.25`); a real session runs at 1.0, and the scale
    is written to session.json so a scaled folder is never mistaken for one.
    """
    if not (time_scale > 0):
        raise ValueError(f"time_scale must be above 0, got {time_scale!r}")
    item = protocol.item(item_id)
    if protocol.name == FLEXION:
        return flexion_cues(item, time_scale)
    if protocol.name == SEQUENCES:
        return sequence_cues(item, protocol.data["hold_s"], time_scale)
    raise ProtocolError(f"{protocol.path}: Set A ({protocol.name}) has no cue "
                        "schedule here; it is recorded by "
                        "scripts/leap/record_poses.py --protocol")


def schedule_seconds(cues: Sequence[Cue]) -> float:
    return float(sum(c.hold_s for c in cues))


def cue_offsets(cues: Sequence[Cue]) -> List[float]:
    """Seconds from the start of the take to each cue."""
    out, t = [], 0.0
    for c in cues:
        out.append(t)
        t += c.hold_s
    return out


# Beep pitches (Hz), as `scripts/leap/finger_sweep.py` cues them: higher to
# bend, lower to straighten. Hold and rest get their own softer tones so the
# four phases of a cycle can be told apart with the eyes closed.
PITCH_FLEX, PITCH_HOLD, PITCH_EXTEND, PITCH_REST = 1200, 1000, 800, 650


def cue_pitch(cue: Cue, previous: Optional[Cue] = None) -> int:
    """The beep for one cue.

    Set B by phase. Set C by what the step asks the hand to do relative to
    the step before it: any finger that has to flex gets the high tone (the
    movement to make), a step that only straightens gets the low one, and a
    step that changes nothing (the first "open hand", or "+little" followed
    by "full fist") gets the neutral one.
    """
    if cue.phase is not None:
        return {BEND: PITCH_FLEX, HOLD: PITCH_HOLD, STRAIGHTEN: PITCH_EXTEND,
                REST: PITCH_REST}[cue.phase]
    before = set(previous.flexed) if previous is not None else set()
    now = set(cue.flexed)
    if now - before:
        return PITCH_FLEX
    if before - now:
        return PITCH_EXTEND
    return PITCH_HOLD


def still_cue(cues: Sequence[Cue]) -> int:
    """Which cue the take's one still is taken in (its index).

    Set B: the hold of the middle cycle, so the still shows the cued finger
    bent. Set C: the middle step. Either way the hand is settled and the take
    is not yet over, which is when a picture says most about the take.
    """
    if not cues:
        raise ValueError("no cues")
    holds = [i for i, c in enumerate(cues) if c.phase == HOLD]
    if holds:
        return holds[(len(holds) - 1) // 2]
    return len(cues) // 2


# --- the order of the takes --------------------------------------------------
def rounds(items: Sequence[str], takes_per_item: int, seed=None,
           shuffle: bool = False) -> List[List[str]]:
    """One list of item ids per round; every round records every item once.

    Unshuffled, every round is the protocol's order. Shuffled, each round is
    a permutation drawn from `random.Random(seed)`, so the same seed always
    gives the same session plan; and no item closes one round and opens the
    next, because two takes of a sequence back to back are not independent
    repetitions.
    """
    items = list(items)
    if not items:
        raise ValueError("no items to record")
    if len(set(items)) != len(items):
        raise ValueError(f"an item is listed twice: {items}")
    n = int(takes_per_item)
    if n < 1:
        raise ValueError(f"takes_per_item must be at least 1, got {n}")
    if not shuffle:
        return [list(items) for _ in range(n)]
    rng = random.Random(seed)
    out: List[List[str]] = []
    for _ in range(n):
        order = list(items)
        rng.shuffle(order)
        if out and len(order) > 1 and order[0] == out[-1][-1]:
            k = rng.randrange(1, len(order))
            order[0], order[k] = order[k], order[0]
        out.append(order)
    return out


def take_name(item: str, hand: str, take: int, stamp: str) -> str:
    """`<item>_<hand>_take<N>_<YYYYMMDD_HHMMSS>` (contract section 1)."""
    return f"{item}_{hand}_take{int(take)}_{stamp}"


# --- events ------------------------------------------------------------------
class EventLog:
    """The take's events file: one JSON object per line, flushed per line.

    `t` is `time.time()` unless given, the same clock as `wall_time` in the
    frame files, so a cue can be laid over the glove trace with no offset.
    Opened for appending when `append` is set: the operator's redo is a
    decision written after the take's own.
    """

    def __init__(self, path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a" if append else "w", encoding="utf-8")

    def write(self, kind: str, t: Optional[float] = None, **fields) -> dict:
        if self._fh is None:
            raise RuntimeError(f"{self.path} is closed")
        row = {"t": round(float(time.time() if t is None else t), 6),
               "kind": kind}
        row.update(fields)
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        return row

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def append_event(path, kind: str, t: Optional[float] = None,
                 **fields) -> dict:
    """Append one event to an existing events file."""
    with EventLog(path, append=True) as log:
        return log.write(kind, t=t, **fields)


def read_events(path) -> List[dict]:
    """The events file back as a list of dicts. Raises ValueError on a bad
    line, naming it: a torn events file is not something to guess past."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {n}: not JSON ({e.msg})") from None
            if not isinstance(row, dict) or "t" not in row or "kind" not in row:
                raise ValueError(f"{path} line {n}: an event needs 't' and "
                                 "'kind'")
            out.append(row)
    return out


def cue_events(events: Sequence[Mapping]) -> List[Mapping]:
    return [e for e in events if e.get("kind") == "cue"]


def take_window(events: Sequence[Mapping]) -> Tuple[Optional[float],
                                                     Optional[float]]:
    """(take_start t, take_end t) from an events list; None where missing."""
    start = next((float(e["t"]) for e in events
                  if e.get("kind") == "take_start"), None)
    end = next((float(e["t"]) for e in reversed(list(events))
                if e.get("kind") == "take_end"), None)
    return start, end


# --- small statistics --------------------------------------------------------
def median(values: Iterable[float]) -> Optional[float]:
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy's default), None when empty."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q) / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _r(v: Optional[float], nd: int = 4) -> Optional[float]:
    return None if v is None else round(float(v), nd)


def _fmt(v: float) -> str:
    """Two decimals, and never "-0.00" for a value that rounds to zero."""
    return f"{round(float(v), 2) + 0.0:.2f}"


# --- what the glove measures -------------------------------------------------
def thumb_bend_deg(frame) -> float:
    """The glove thumb's bend in degrees: TMC_fe + MCP_fe + IP.

    The curl (wrist to tip over palm length) barely moves when the thumb
    bends at its own joints: its warm-up span was 0.34 on 2026-10-01 against
    1.0 to 1.3 for the fingers, and in the 16:55 thumb take it sat on the
    open value for four bends of five while the same glove's fingers moved.
    The thumb's three flexion angles are what the glove's thumb sensor
    drives, so their sum is the reading that moves when the thumb bends.
    Larger = more bent. The angles come from `xr_hand.joint_frames`
    (`joint_table`, then `dof24`) and not from a copy of its maths: the
    table costs about 0.7 ms a frame on the recording laptop, a small part
    of a 60 Hz glove frame.
    """
    angles = dof24(joint_table(line_from_hand_frame(frame), "glove"))
    return float(sum(angles[k] for k in THUMB_ANGLES))


def glove_bends(frame) -> List[float]:
    """Five readings of one glove frame, in FINGERS order.

    The thumb is `thumb_bend_deg` (degrees, larger = more bent); index,
    middle, ring and pinky are the curl (`flexion_features`, smaller = more
    bent), exactly as before. `fraction` works in either direction, so every
    rule downstream reads the five the same way. The recorder's warm-up and
    quick check and the checker's glove stream all measure through here, so
    the recorder and the checker measure one quantity.
    """
    out = [float(v) for v in flexion_features(frame_to_keypoints21(frame))]
    out[FINGERS.index(THUMB)] = thumb_bend_deg(frame)
    return out


# --- the warm-up -------------------------------------------------------------
def fraction(open_: float, fist: float, curl: float) -> Optional[float]:
    """How far bent, in the finger's own warm-up range, clipped.

    0 = the warm-up open palm, 1 = the end of the range (the warm-up fist,
    or the finger's own single bend), clipped to FRACTION_CLIP. Either
    direction works: a curl falls as the finger bends, the thumb's degrees
    rise. None when the two endpoints are equal: there is no range to be a
    fraction of.
    """
    span = float(open_) - float(fist)
    if abs(span) < MIN_SPAN:
        return None
    f = (float(open_) - float(curl)) / span
    lo, hi = FRACTION_CLIP
    return max(lo, min(hi, f))


def window_endpoints(samples: Iterable[Tuple[float, Sequence[float]]],
                     t_open: Sequence[float],
                     t_fist: Sequence[float]) -> Optional[dict]:
    """Per-finger medians over the open and the fist window, and the span.

    `samples` are (time, five curls) in the hand's finger order. A sample
    counts for a window when t0 <= t <= t1. Returns None when either window
    has no sample: an endpoint nobody measured is not an endpoint.
    """
    opened, fisted = [], []
    for t, curls in samples:
        t = float(t)
        if t_open[0] <= t <= t_open[1]:
            opened.append(list(curls))
        elif t_fist[0] <= t <= t_fist[1]:
            fisted.append(list(curls))
    if not opened or not fisted:
        return None
    out = {"open": {}, "fist": {}, "span": {},
           "frames": len(opened) + len(fisted)}
    for i, f in enumerate(FINGERS):
        o = median(c[i] for c in opened)
        k = median(c[i] for c in fisted)
        out["open"][f] = _r(o)
        out["fist"][f] = _r(k)
        out["span"][f] = _r(o - k)
    return out


def single_endpoints(samples: Iterable[Tuple[float, Sequence[float]]],
                     t_single: Mapping[str, Sequence[float]],
                     open_: Mapping[str, Optional[float]]) -> dict:
    """Each finger's own bend: the median over its window, and open - it.

    `t_single` is {finger: (t0, t1)}, the last WARMUP_SETTLE_S of the
    window in which that finger was bent alone. `open_` is the block's
    open palm. A finger whose window has no sample gets None, which no rule
    can use (`single_refusal` names it).
    """
    samples = [(float(t), list(c)) for t, c in samples]
    single, span, windows = {}, {}, {}
    for i, f in enumerate(FINGERS):
        win = t_single.get(f)
        if win is None:
            single[f] = span[f] = None
            continue
        windows[f] = [round(float(win[0]), 6), round(float(win[1]), 6)]
        s = median(c[i] for t, c in samples if win[0] <= t <= win[1])
        o = open_.get(f)
        single[f] = _r(s)
        span[f] = None if s is None or o is None else _r(float(o) - s)
    return {"single": single, "single_span": span, "t_single": windows}


def _single_amount(glove: Mapping, finger: str) -> Optional[float]:
    """How far `finger` bent on its own, in the unit its rule is in.

    The thumb in degrees (single - open: its degrees rise as it bends);
    index to pinky as a share of their fist span (single_span / span). None
    when the block has no such value or no fist span to share.
    """
    single = (glove.get("single") or {}).get(finger)
    opened = (glove.get("open") or {}).get(finger)
    sspan = (glove.get("single_span") or {}).get(finger)
    if finger == THUMB:
        if single is not None and opened is not None:
            return float(single) - float(opened)
        return None if sspan is None else -float(sspan)
    if sspan is None and single is not None and opened is not None:
        sspan = float(opened) - float(single)
    span = (glove.get("span") or {}).get(finger)
    if sspan is None or span is None or abs(float(span)) < MIN_SPAN:
        return None
    return float(sspan) / float(span)


def single_usable(glove: Optional[Mapping], finger: str) -> bool:
    """Is `finger`'s own bend in this warm-up a range the take rules can
    use? Index to pinky: at least SINGLE_MIN_SHARE of the fist span; the
    thumb: at least THUMB_MIN_DEG. A block without `single` (every warm-up
    recorded before 2026-10-01) has none."""
    if not glove or not isinstance(glove.get("single"), Mapping):
        return False
    amount = _single_amount(glove, finger)
    if amount is None:
        return False
    need = THUMB_MIN_DEG if finger == THUMB else SINGLE_MIN_SHARE
    return amount >= need


def _deg(v: float) -> str:
    """Whole degrees, unless rounding would make a short bend read as the
    threshold it missed."""
    return f"{v:.0f}" if round(v) < THUMB_MIN_DEG else f"{v:.1f}"


def single_refusal(glove: Optional[Mapping]) -> Optional[str]:
    """Why the single-finger part of this warm-up cannot be used, or None.

    `glove` is the glove block of warmup.json (`open`, `fist`, `span`,
    `single`, `single_span`). None when the block has no `single` (a
    warm-up recorded before 2026-10-01). Every finger whose own bend is not
    usable (`single_usable`) is named, with what to do about it: the take
    rules measure a finger against its own bend, so a bend that stopped
    short would make every take of it look fully bent.
    """
    if not glove or not isinstance(glove.get("single"), Mapping):
        return None
    parts = []
    for f in FINGERS:
        if single_usable(glove, f):
            continue
        words = FINGER_WORDS[f]
        amount = _single_amount(glove, f)
        if (glove.get("single") or {}).get(f) is None or amount is None:
            parts.append(f"the glove sent no frames of the {words} bent on "
                         "its own: bend it on the high beep and hold it bent "
                         "until the low beep")
        elif f == THUMB:
            parts.append(f"the thumb bent only {_deg(amount)} degrees (needs "
                         f"{THUMB_MIN_DEG:.0f}): fold it fully across the "
                         "palm, tip to the base of the little finger")
        else:
            parts.append(f"the {words} bent only {_fmt(amount)} of its fist "
                         f"span (needs {SINGLE_MIN_SHARE:.2f}): fold it fully "
                         "into the palm on its own; the others may follow "
                         "it a little")
    if not parts:
        return None
    return "in the single-finger warm-up " + "; ".join(parts)


def warmup_refusal(glove: Optional[Mapping]) -> Optional[str]:
    """Why the session cannot go on after this warm-up, or None.

    The contract's rule first: a glove span under REFUSE_SPAN on any of the
    four fingers refuses. The thumb is left out of it on purpose (the glove
    senses its flexion only, and its curl range was small on every session
    so far). Then, for a block that carries the single-finger part, every
    finger whose own bend is not usable (`single_refusal`).
    """
    if not glove:
        return ("the glove sent no frames of this hand during the open palm "
                "or the fist, so there are no endpoints to measure against")
    low = [f for f in REFUSE_FINGERS
           if glove["span"].get(f) is None or glove["span"][f] < REFUSE_SPAN]
    if not low:
        return single_refusal(glove)
    parts = ", ".join(f"{FINGER_WORDS[f]} {_fmt(glove['span'][f])}"
                      if glove["span"].get(f) is not None
                      else f"{FINGER_WORDS[f]} (no value)" for f in low)
    return (f"the glove barely moved between the open palm and the fist "
            f"(span {parts}; each needs at least {REFUSE_SPAN:.2f}). Open "
            "fully, then close into a full fist, and check the gloves are "
            "calibrated in XR Trainer")


def warmup_record(hand: str, t_open: Sequence[float], t_fist: Sequence[float],
                  glove_samples, camera_samples=None,
                  settle_s: Optional[float] = None,
                  t_single: Optional[Mapping[str, Sequence[float]]] = None,
                  units: Optional[Mapping[str, str]] = None) -> dict:
    """The warmup.json dict (contract section 4), refusal included.

    `camera_samples` None means the session ran with no camera, and the
    camera block is null; a camera that ran but saw nothing gives a null
    block too, which is what it measured. `t_single` ({finger: (t0, t1)},
    the single-finger medians' windows) adds `single`, `single_span` and
    `t_single` to each block (the camera's for information only), and
    `units` says what the glove numbers are (`GLOVE_UNITS`); a record
    without them is the one every session before 2026-10-01 wrote.
    """
    glove = window_endpoints(glove_samples, t_open, t_fist)
    camera = (None if camera_samples is None
              else window_endpoints(camera_samples, t_open, t_fist))
    if t_single is not None:
        for block, samples in ((glove, glove_samples),
                               (camera, camera_samples)):
            if block is not None:
                block.update(single_endpoints(samples, t_single,
                                              block["open"]))
    out = {"hand": hand,
           "t_open": [round(float(t_open[0]), 6), round(float(t_open[1]), 6)],
           "t_fist": [round(float(t_fist[0]), 6), round(float(t_fist[1]), 6)],
           "glove": glove, "camera": camera,
           "refused": warmup_refusal(glove)}
    if settle_s is not None:
        out["settle_s"] = round(float(settle_s), 6)
    if units is not None:
        out["units"] = dict(units)
    return out


# --- reading a take back -----------------------------------------------------
def _is_camera_line(row: Mapping) -> bool:
    """A line the Leap recorder wrote (it carries `abs26` and `source`)."""
    return bool(row.get("abs26")) or str(row.get("source", "")).lower() in (
        "leap", "camera")


def read_bends(path, hand: Optional[str] = None
               ) -> List[Tuple[float, List[float]]]:
    """A recorded glove or camera JSONL -> [(time, five readings)], in order.

    A glove line is measured by `glove_bends` (the thumb in degrees), a
    camera line by the curl for all five: the quantities the session's
    warm-up was measured in. Otherwise read as `leap_hand.pose_check`
    reads a take. The time is `capture_time` where the line has one (when
    the packet arrived) and `wall_time` otherwise.
    """
    from xr_hand.recorder import FrameRecorder

    path = Path(path)
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    out = []
    for row, (frame, wall) in zip(rows, FrameRecorder.load(path)):
        if hand is not None and frame.hand_side != hand:
            continue
        stamp = row.get("capture_time")
        t = float(stamp) if stamp is not None else float(wall)
        if _is_camera_line(row):
            values = [float(v) for v in
                      flexion_features(frame_to_keypoints21(frame))]
        else:
            values = glove_bends(frame)
        out.append((t, values))
    return out


# The name every caller used before the thumb was measured in degrees. It
# reads the same as `read_bends`: a glove file gives the thumb in degrees.
read_curls = read_bends


# --- the quick check ---------------------------------------------------------
@dataclass
class CheckResult:
    """The recorder's verdict on one take, and everything behind it.

    `reason` is empty when accepted. `notes` are information for the
    operator and the session file (coupling, events that do not match the
    protocol's cycle count), never a reason to reject. `details` is
    JSON-ready. `hint` is the one line the console prints under a take
    rejected for how the finger moved, so the retry a few seconds later is
    not the same movement again; it is advice, not a measurement, so
    `as_dict` (what session.json keeps) leaves it out.
    """

    accepted: bool
    reason: str = ""
    notes: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    hint: str = ""

    def as_dict(self) -> dict:
        return {"accepted": self.accepted, "reason": self.reason,
                "notes": list(self.notes), **self.details}


def _endpoints(glove: Mapping, finger: str,
               cued: bool = True) -> Tuple[float, float]:
    """(open, end) of `finger`'s range in a warm-up block.

    For a finger the cue names (`cued`), the end is its own single bend
    when the block has one and it is usable (`single_usable`), else the
    fist; for any other finger it is the fist. The fist is a different
    movement for the thumb (wrapped over the fingers) and lends every
    finger its neighbours' help, so a cued finger is measured against what
    it did alone; a finger the cue leaves straight keeps the fist, so
    coupling reads on the scale it always did. A block without `single`
    (every warm-up before 2026-10-01) gives (open, fist) for every finger.
    """
    opened = float(glove["open"][finger])
    if cued and single_usable(glove, finger):
        return opened, float(glove["single"][finger])
    return opened, float(glove["fist"][finger])


# The public name for the recorder's live bars.
range_ends = _endpoints


def fractions_of(curls: Sequence[Tuple[float, Sequence[float]]],
                 glove: Mapping, finger: str,
                 t0: Optional[float] = None,
                 t1: Optional[float] = None,
                 cued: bool = True) -> List[float]:
    """One finger's fraction per frame inside [t0, t1] (None = open end).

    `cued`: measured as the finger the cue names (`_endpoints`)."""
    i = FINGERS.index(finger)
    o, k = _endpoints(glove, finger, cued)
    out = []
    for t, c in curls:
        if t0 is not None and t < t0:
            continue
        if t1 is not None and t > t1:
            continue
        f = fraction(o, k, c[i])
        if f is not None:
            out.append(f)
    return out


def span_fraction(fracs: Sequence[float]) -> Optional[float]:
    """How much of its range a finger covered: p95 - p5 of its fractions."""
    if not fracs:
        return None
    return percentile(fracs, SPAN_PERCENTILES[1]) - percentile(
        fracs, SPAN_PERCENTILES[0])


def count_peaks(values, low_share: float = PEAK_LOW,
                high_share: float = PEAK_HIGH,
                min_range: float = PEAK_MIN_RANGE) -> int:
    """Bends in a fraction trace (higher = more flexed), Schmitt trigger.

    The one bend count: the recorder's quick check rejects on it and
    `cam_hand.protocol_check` imports it, so a take the recorder accepts
    cannot fail the checker on its count. A bend counts when the trace rises
    above `high_share` of its own range (5th to 95th percentile) after
    having been below `low_share` of it, so a take that starts with the
    finger already bent does not count that first bend and a wobble on a
    hold cannot count twice. The marks sit in the take's own range, not at
    fixed fractions: a glove that creeps, or reads too open on the day,
    still shows its cycles, and how far the finger got is the span gate's
    question, not this one's. Values that are None or not finite are
    skipped.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0
    lo, hi = (float(x) for x in np.percentile(v, list(SPAN_PERCENTILES)))
    if hi - lo < min_range:
        return 0
    low, high = lo + low_share * (hi - lo), lo + high_share * (hi - lo)
    armed, n = False, 0
    for x in v:
        if x <= low:
            armed = True
        elif x >= high and armed:
            n += 1
            armed = False
    return n


def bends_reason(finger: str, peaks: int, cues: int) -> str:
    """Why a take whose bend count is not its cue count is rejected.

    Says what the glove saw in the operator's words and states the counting
    rule in full, because "1 peak" alone does not tell anyone that a bend
    which stopped short of the take's top counts as no bend at all.
    """
    rule = (f"a bend counts when the curl rises above "
            f"{PEAK_HIGH * 100:.0f} percent of the take's range after being "
            f"below {PEAK_LOW * 100:.0f} percent")
    words = FINGER_WORDS[finger]
    if peaks < cues:
        times = "time" if peaks == 1 else "times"
        return (f"the {words} bent fully only {peaks} {times} of {cues} "
                f"({rule})")
    return (f"the {words} bent fully {peaks} times for {cues} bend cues "
            f"({rule})")


def bend_hint(finger: str) -> str:
    """What to do after a Set B take rejected for its movement, in one line.

    Generic per finger on purpose: the span gate and the bend count fail
    for the same cause (a fold that stopped short, or an opening that did),
    and the reason above already says which and by how much. The thumb
    crosses the palm; the others close into it.
    """
    where = "across the palm" if finger == "thumb" else "into the palm"
    return (f"fold the {FINGER_WORDS[finger]} fully {where} on every bend, "
            "then open it fully")


def check_flexion_take(curls: Sequence[Tuple[float, Sequence[float]]],
                       events: Sequence[Mapping], glove: Mapping,
                       finger: str, cycles: int,
                       span_gate: float = SPAN_GATE) -> CheckResult:
    """Set B's acquisition gate for one take.

    Rejects when the cued finger covered less than `span_gate` of its
    warm-up range, or when it did not bend fully once per bend cue: the
    curl peaks (`count_peaks`) must number the take's bend cues, the rule
    the checker fails a take on. The span alone let a thumb take through on
    2026-10-01 that folded fully on one cycle of five (span 0.78, 1 peak);
    the checker failed it after the session, when redoing it cost a new
    session. The other four fingers' spans are measured and reported, never
    failed on (the glove senses flexion only and the ring really does drag
    its neighbours). Events that disagree with the protocol's `cycles` are a
    note, not a reject: the recorder writes one bend cue per cycle, so the
    two differ only on a take cut short, which is rejected as interrupted.
    """
    t0, t1 = take_window(events)
    words = FINGER_WORDS[finger]
    fracs = fractions_of(curls, glove, finger, t0, t1)
    in_take = [c for t, c in curls if (t0 is None or t >= t0)
               and (t1 is None or t <= t1)]
    cycles_events = len({e.get("cycle") for e in cue_events(events)
                         if e.get("phase") == BEND})
    details = {"finger": finger, "frames": len(in_take),
               "cycles_expected": int(cycles),
               "cycles_from_events": cycles_events,
               "range_end": ("single" if single_usable(glove, finger)
                             else "fist")}
    o, k = _endpoints(glove, finger)
    if abs(o - k) < MIN_SPAN:
        return CheckResult(False, f"the {words} has no warm-up range to "
                           f"measure against (open {o:.3f}, fist {k:.3f})",
                           details=details)
    if not in_take:
        return CheckResult(False, "no glove frames of this hand during the "
                           "take", details=details)
    span = span_fraction(fracs)
    others = {}
    for f in FINGERS:
        if f == finger:
            continue
        s = span_fraction(fractions_of(curls, glove, f, t0, t1, cued=False))
        others[f] = _r(s, 3)
    peaks = count_peaks(fracs)
    details.update({"span_fraction": _r(span, 3), "other_spans": others,
                    "cycles_from_peaks": peaks})
    short = span is None or span < span_gate
    unbent = cycles_events > 0 and peaks != cycles_events
    notes = []
    # Said once: when the bend count is the reason, a note would repeat it.
    if cycles_events != int(cycles) or (unbent and short):
        notes.append(f"cycles: {cycles} cued in the protocol, "
                     f"{cycles_events} in the events, {peaks} peaks in the "
                     f"{words} curl")
    details["cycles_match"] = (peaks == cycles_events == int(cycles))
    moved = [f"{FINGER_WORDS[f]} {_fmt(v)}" for f, v in others.items()
             if v is not None and v >= span_gate]
    if moved:
        notes.append("other fingers that moved with it (span as a fraction "
                     "of their range): " + ", ".join(moved))
    if short:
        return CheckResult(False,
                           f"the {words} moved only {_fmt(span or 0.0)} of its "
                           f"warm-up range (needs {span_gate:.2f})",
                           notes=notes, details=details,
                           hint=bend_hint(finger))
    if unbent:
        return CheckResult(False, bends_reason(finger, peaks, cycles_events),
                           notes=notes, details=details,
                           hint=bend_hint(finger))
    return CheckResult(True, "", notes=notes, details=details)


def step_windows(events: Sequence[Mapping], window_s: float
                 ) -> List[Tuple[Mapping, float, float]]:
    """(cue event, t0, t1) for the check window at the end of each step.

    The window ends where the step really ended (the next cue, or the end of
    the take), not where the schedule said it would, and starts `window_s`
    before that, but never before the step's own cue.
    """
    cues = cue_events(events)
    _, t_end = take_window(events)
    out = []
    for k, cue in enumerate(cues):
        start = float(cue["t"])
        if k + 1 < len(cues):
            end = float(cues[k + 1]["t"])
        elif t_end is not None:
            end = t_end
        else:
            end = start + float(cue.get("hold_s", 0.0))
        out.append((cue, max(start, end - float(window_s)), end))
    return out


def check_sequence_take(curls: Sequence[Tuple[float, Sequence[float]]],
                        events: Sequence[Mapping], glove: Mapping,
                        window_s: float,
                        flexed_above: float = FLEXED_ABOVE,
                        straight_below: float = STRAIGHT_BELOW) -> CheckResult:
    """Set C's check: every step, all five fingers, in the hold window.

    For each step the median fraction over the last `window_s` of its hold:
    a finger the step says is flexed must read above `flexed_above`, a
    finger it says is straight below `straight_below`. All five fingers are
    checked on every step and every miss is reported, but the take is
    rejected only for the two misses that mean the step was not done: a
    flexed finger that is not flexed, or a straight finger that reads
    flexed. A straight finger in between is coupling, reported as a note.
    """
    steps = []
    failures: List[Tuple[Mapping, str]] = []
    coupling: List[str] = []
    for cue, t0, t1 in step_windows(events, window_s):
        want = set(cue.get("flexed") or [])
        values = {}
        fail, loose = [], []
        frames = sum(1 for t, _c in curls if t0 <= t <= t1)
        for f in FINGERS:
            v = median(fractions_of(curls, glove, f, t0, t1,
                                    cued=f in want))
            values[f] = _r(v, 3)
            words = FINGER_WORDS[f]
            if frames == 0:
                continue
            if v is None:
                fail.append(f"the {words} has no warm-up range to measure "
                            "against")
            elif f in want and not v > flexed_above:
                fail.append(f"{words} read {_fmt(v)} of its range, should be "
                            f"flexed (above {flexed_above:.2f})")
            elif f not in want and v > flexed_above:
                fail.append(f"{words} read flexed ({_fmt(v)}) but should be "
                            f"straight")
            elif f not in want and not v < straight_below:
                loose.append(f"{words} {_fmt(v)}")
        if frames == 0:
            fail.append("no glove frames in the hold window")
        tag = f"step {cue.get('step')} ({cue.get('label')})"
        if loose:
            coupling.append(f"{tag}: {', '.join(loose)}")
        for text in fail:
            failures.append((cue, f"{tag}: {text}"))
        steps.append({"step": cue.get("step"), "label": cue.get("label"),
                      "flexed": sorted(want, key=FINGERS.index),
                      "window": [round(t0, 6), round(t1, 6)],
                      "frames": frames, "fractions": values,
                      "pass": not fail, "fail": fail, "coupling": loose})
    details = {"steps_total": len(steps),
               "steps_pass": sum(1 for s in steps if s["pass"]),
               "steps": steps}
    notes = []
    if coupling:
        notes.append("straight fingers between the bands (coupling, not a "
                     "failure): " + "; ".join(coupling))
    if not steps:
        return CheckResult(False, "the events file has no cues", notes,
                           details)
    if failures:
        first = failures[0][1]
        more = len({c.get("step") for c, _ in failures}) - 1
        reason = first + (f"; {more} more step(s) failed" if more > 0 else "")
        return CheckResult(False, reason, notes, details)
    return CheckResult(True, "", notes, details)


def check_take(protocol: Protocol, item_id: str,
               curls: Sequence[Tuple[float, Sequence[float]]],
               events: Sequence[Mapping], glove: Mapping,
               time_scale: float = 1.0) -> CheckResult:
    """The quick check for one take of `item_id`, by the protocol's set."""
    item = protocol.item(item_id)
    if protocol.name == FLEXION:
        return check_flexion_take(curls, events, glove, item["finger"],
                                  int(item["cycles"]))
    if protocol.name == SEQUENCES:
        return check_sequence_take(
            curls, events, glove,
            float(protocol.data["check_window_s"]) * time_scale)
    raise ProtocolError(f"{protocol.path}: no quick check for {protocol.name}")
