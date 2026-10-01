"""Did each protocol take do what its cues asked for? Sets A, B and C, per take.

The quick check that runs between takes, so a bad take is redone while the
gloves are still on (plan section 4, contract `docs/protocol_formats.md`
section 8). `scripts/check_protocol.py` is the command line around it and
`scripts/package_professor_set.py` reads what it writes. Everything here reads
a session folder in the contract's layout and computes; the only function
that writes a file is `write_check`.

TWO KINDS OF NUMBER, KEPT APART ON PURPOSE
  Acquisition gates are fixed numbers that decide whether a take is usable:
  the cued finger's glove span of 0.60 of its own warm-up range (Set B), the
  0.6 / 0.3 bands of a sequence step (Set C), 90 % tracked for a grasp
  (Set A). They are screening numbers, never the definition of a correct
  movement, so every continuous value behind a verdict (each finger's span,
  each step's median fraction, each cycle's lag) is written next to it in
  `check.csv` and `check.txt`. A reader who disagrees with a gate can re-judge
  the take from the columns without re-running anything.

THE MEASUREMENT IS THE ONE THE REST OF THE REPO USES
  Curl is `cam_hand.features.flexion_features` on
  `xr_hand.keypoints21.frame_to_keypoints21`: fingertip-to-wrist over palm
  length, higher = straighter. The fraction of a finger's range is the
  contract's (section 4):

      fraction = (open - curl) / (open - fist), clipped to [-0.5, 1.5]

  with `open` and `fist` the finger's own medians from the session's warm-up
  (`warmup.json`), so 0 is that hand's open palm and 1 is its fist. The
  transfer curve, rail saturation, camera-range guard and lag are
  `leap_hand.diagnostics`, unchanged: the finger-sweep tool and this checker
  must say the same thing about the same bend.

SET B (finger_flexion)
  * cued span: 5th to 95th percentile of the cued finger's glove fraction over
    the take (the same percentiles `diagnostics.camera_range` uses, so one
    mistracked frame cannot invent range). Fails below 0.60.
  * the other four fingers' spans are reported, never failed: the ring drags
    the middle and pinky with it, and the glove senses flexion only.
  * cycles are counted twice: from the events (one per `phase == "bend"`
    cue) and from the glove curl peaks (a Schmitt trigger at 30 % and 60 % of
    the take's own range, so noise on a hold cannot count as a second bend).
    The two must agree, and when the protocol file is readable the event
    count must equal its `cycles` (the plan's "verify 5 repetitions rather
    than assume it"). The peaks are counted by
    `recording_protocol.count_peaks`, the function the recorder's quick
    check rejects on, so a take the recorder accepted is not failed here
    for a bend count it could have been redone for.
  * where the camera followed the finger (camera span >= 0.50 by
    `diagnostics.camera_range`, over the take and per cycle) the transfer
    curve, hysteresis and lag are reported, per cycle and per take, with the
    number of cycles and glove frames that met that rule. Lag is never one
    number per hand: `lag_ms_median` is the median of the per-cycle lags of
    this take, and the whole-take estimate sits beside it.

SET C (sequences)
  For every cued step the median fraction of every finger over the last
  `check_window_s` seconds of the step's hold (cue time + hold_s), glove and
  trusted camera separately. The rule is the plan's (section 4) and the
  recorder's live quick check (`cam_hand.recording_protocol`):

    * a finger the step names as flexed must read above 0.6;
    * a finger the step names as straight FAILS only when it reads flexed,
      0.6 or more;
    * a straight finger between 0.3 and 0.6 is COUPLING: reported per step
      and finger (the `coupling` entry of each step in `check.csv`, a count
      in the table), never failed. The ring pulls the middle and the pinky
      along, and the glove senses flexion only, so a hard 0.3 would reject
      real coupled motion.

    --bands fixed   the 0.6 and 0.3 are fractions of the warm-up range.
    --bands auto    the same two shares of THIS SESSION's own range, per
                    finger and per sensor:
                        flexed threshold   = open + 0.6 x (fist - open)
                        coupling threshold = open + 0.3 x (fist - open)
                    in curl units, where `open` is the median over the
                    session's accepted takes of that finger's step median on
                    every open-hand step (no finger flexed) and `fist` the
                    same over every full-fist step (all five flexed). A
                    session with no such step falls back to the warm-up for
                    that end. This is algebraically the fixed rule applied to
                    a fraction whose endpoints come from the session instead
                    of the warm-up, which is how the report prints it.

  A take passes when every step passes on the GLOVE. The camera's pass/fail
  is reported beside it and never fails a take: the glove is the instrument
  under test, and a step where the glove fails and the camera passes reads as
  the glove being wrong (the right glove's too-open readings), not the
  operator.

SET A (grasps), light pass
  The meta file of each take, tabulated, with the acquisition gate (tracked
  fraction >= 0.90 and no re-acquisition inside the static interval, read
  from `gate.reacquisitions_in_static_interval` when the meta has it and
  from `reacquisitions` otherwise) and a
  second-look flag on any take whose grab, pinch or curl sits far from its
  grasp's other takes. The acceptance itself is the operator's eye on the
  still, which the recorder has already timestamped.

CAMERA TRUST AND HAND LABEL
  A camera frame counts only when `cam_hand.fusion.camera_trusted` passes it
  with the default gates `scripts/fuse_poses.py` runs (held >= 300 ms, hand id
  stable, palm inside the central field, palm facing the module within 50
  degrees), after `flag_hand_id_stability` over the take. The tracker's
  left/right label is not trusted (it called a bare left hand "right" in 20
  of 21 poses), so the camera frames used are those labelled with the
  operator's hand when that label holds the most frames of the take, and
  otherwise the label that does, with a note saying so.

CLOCK
  One clock per take, chosen by `cam_hand.fusion.pairing_clock`:
  `capture_time` when every glove and camera line has one, else `wall_time`.
  Both are `time.time()` on the recording machine, the clock the events file
  is written on. The paired fraction is the share of glove frames with a
  camera frame of the same hand within 50 ms on that clock
  (`fusion.pair_by_time`).
"""
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# a per-grasp `"orientation": "...",` line on its own, as the Set A recorder
# adds them (the same pattern as record_poses._HINT_LINE)
_HINT_LINE = re.compile(
    rb'^[ \t]*"orientation"[ \t]*:[ \t]*"(?:[^"\\\r\n]|\\.)*"[ \t]*,[ \t]*\r?\n',
    re.MULTILINE)

from cam_hand.features import FLEXION_NAMES, flexion_features
from cam_hand.fusion import (
    CAPTURE_CLOCK,
    DEFAULT_GATES,
    WALL_CLOCK,
    camera_trusted,
    flag_hand_id_stability,
    pair_by_time,
    pairing_clock,
)
# Bends are counted by the recorder's own function and thresholds, imported
# rather than copied (see the module docstring, SET B). The names after
# each `as` are the ones this module has always used.
from cam_hand.recording_protocol import (
    PEAK_HIGH as PEAK_HIGH_SHARE,
    PEAK_LOW as PEAK_LOW_SHARE,
    PEAK_MIN_RANGE as MIN_PEAK_RANGE,
    count_peaks,
)
from leap_hand.diagnostics import (
    MIN_CAMERA_RANGE,
    RANGE_PERCENTILES,
    StreamLog,
    analyse_finger,
    camera_range,
    camera_range_verdict,
    estimate_lag,
    open_reference,
    transfer_curve,
)
from leap_hand.protocol import coverage, palm_normal_abs
from xr_hand.keypoints21 import frame_to_keypoints21
# The one-line parser behind `FrameRecorder.load`, used directly so a take is
# read once and a line cut off mid-write can be skipped instead of aborting
# the whole file (the same private import `leap_hand.recorder` makes).
from xr_hand.recorder import _dict_to_frame

REPO = Path(__file__).resolve().parents[2]

FINGERS: Tuple[str, ...] = tuple(FLEXION_NAMES)
# The protocol's labels say "little"; the data always says "pinky".
FINGER_ALIASES = {"little": "pinky"}

GRASPS, FLEXION, SEQUENCES = "grasps", "finger_flexion", "sequences"
SETS = (GRASPS, FLEXION, SEQUENCES)

PASS, FAIL = "pass", "fail"
FIXED, AUTO = "fixed", "auto"
BANDS = (FIXED, AUTO)
GLOVE, CAMERA = "glove", "camera"

# --- acquisition gates and measurement settings -------------------------------
PAIR_MAX_DT = 0.05              # s, nearest camera frame for a glove frame
FRACTION_CLIP = (-0.5, 1.5)     # contract section 4
MIN_CUED_SPAN = 0.60            # plan section 4, Set B screening number
# Bends in the glove trace are counted with PEAK_LOW_SHARE, PEAK_HIGH_SHARE
# and MIN_PEAK_RANGE, imported above with `count_peaks` from
# `recording_protocol`: one rule for the recorder and the checker.
FLEXED_ABOVE = 0.6              # plan section 4, Set C initial bands
STRAIGHT_BELOW = 0.3
DEFAULT_CHECK_WINDOW_S = 1.5    # plan D7, used when the protocol is unreadable
DEFAULT_HOLD_S = 2.5
# A step window counts for a sensor only when that sensor has frames over at
# least half of it (holes longer than `protocol.LOSS_GAP_S` are not coverage).
MIN_WINDOW_COVERAGE = 0.5
GRASP_MIN_TRACKED = 0.90        # plan section 4, Set A acquisition gate
WARMUP_MIN_SPAN = 0.30          # contract section 4, index..pinky glove span
# Second-look flag for grasp outliers: a take whose value is further from its
# grasp's median than this many times the session's typical take-to-median
# distance for that value, and at least OUTLIER_FLOOR away.
OUTLIER_FACTOR = 3.0
OUTLIER_FLOOR = 0.05

TAKE_RE = re.compile(r"^(?P<item>.+)_(?P<hand>left|right)_take(?P<take>\d+)"
                     r"_(?P<stamp>\d{8}_\d{6})$")

# Where each file of a take lives when session.json does not say (contract
# section 1). Rejected attempts use the same layout under rejected/.
LAYOUT = {
    "glove": "glove/{name}.jsonl",
    "leap": "leap/{name}.jsonl",
    "events": "events/{name}.events.jsonl",
    "still": "stills/{name}.png",
    "keypoints": "keypoints/{name}_keypoints.txt",
    "meta": "meta/{name}.json",
}

CSV_COLUMNS = [
    "set", "session", "item", "take", "name", "hand", "accepted", "reason",
    "frames_glove", "glove_hz", "glove_gap_max_ms", "glove_lost_packets",
    "frames_camera", "camera_trusted_fraction", "camera_label",
    "paired_fraction", "clock",
    "cued_finger", "cued_span_fraction", "other_spans",
    "cycles_expected", "cycles_from_events", "cycles_from_peaks",
    "camera_span", "cycles_camera_followed", "frames_camera_followed",
    "lag_ms_median", "lag_ms_take", "lag_ms_cycles", "hysteresis_median",
    "bands", "steps_total", "steps_pass_glove", "steps_judged_camera",
    "steps_pass_camera", "coupling_glove", "coupling_camera", "steps",
    "tracked_fraction", "reacquisitions", "grab_strength", "pinch_strength",
    "curls",
    "verdict", "failures", "notes",
]
JSON_COLUMNS = ("other_spans", "lag_ms_cycles", "steps", "curls")


def finger_name(name) -> Optional[str]:
    """'little' -> 'pinky'; anything that is not a finger -> None."""
    n = str(name).strip().lower()
    n = FINGER_ALIASES.get(n, n)
    return n if n in FINGERS else None


def flexed_set(names) -> Tuple[str, ...]:
    """A cue's `flexed` list as finger names in thumb..pinky order."""
    got = {finger_name(n) for n in (names or [])}
    return tuple(f for f in FINGERS if f in got)


# --- the session folder --------------------------------------------------------

@dataclass
class TakeRef:
    """One take of a session, accepted or not, and where its files are."""

    name: str
    item: str
    take: Optional[int]
    accepted: bool
    reason: str
    files: Dict[str, Path] = field(default_factory=dict)

    def path(self, kind: str) -> Optional[Path]:
        """The file of this kind if it exists on disk, else None."""
        p = self.files.get(kind)
        return p if p is not None and p.is_file() else None


@dataclass
class Session:
    path: Path
    meta: dict
    set: str
    hand: str
    warmup: Optional[dict]
    protocol: Optional[dict]
    protocol_note: str
    takes: List[TakeRef]

    @property
    def name(self) -> str:
        return self.path.name

    def item(self, item_id: str) -> Optional[dict]:
        """The protocol file's entry for this item, if the file was readable."""
        for it in (self.protocol or {}).get("items", []) or []:
            if it.get("id") == item_id:
                return it
        return None


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_protocol(session_dir: Path, meta: dict) -> Tuple[Optional[dict], str]:
    """The protocol file the session was recorded from, and a line about it.

    Looked for as written in session.json, relative to the working folder,
    the repository and every folder above the session. When session.json
    stores a sha256 and the file on disk no longer matches it, the file is
    NOT used: its cycle counts and windows would describe a protocol the
    session was not recorded under.
    """
    rel = meta.get("protocol_file")
    if not rel:
        return None, "session.json names no protocol file"
    p = Path(str(rel))
    candidates = [p] if p.is_absolute() else (
        [Path.cwd() / p, REPO / p, session_dir / p]
        + [parent / p for parent in session_dir.parents])
    for c in candidates:
        if not c.is_file():
            continue
        data = c.read_bytes()
        want = meta.get("protocol_sha256")
        hints_only = False
        if want and hashlib.sha256(data).hexdigest() != str(want).lower():
            # The recorder's --resume lets a session recorded before the
            # per-grasp orientation hints existed carry on with the hinted
            # file (record_poses.sha256_without_hints); the same file must
            # then count here, or a resumed session packages without its
            # labels. Any other change is still refused.
            if hashlib.sha256(_HINT_LINE.sub(b"", data)).hexdigest() == str(want).lower():
                hints_only = True
            else:
                return None, (f"{rel} has changed since this session was recorded "
                              "(sha256 differs), so its values were not used")
        try:
            proto = json.loads(data.decode("utf-8"))
        except ValueError:
            return None, f"{rel} is not valid JSON, so its values were not used"
        ver = proto.get("version", meta.get("protocol_version"))
        check = ("sha256 matches" if want and not hints_only
                 else "differs only by orientation hints" if hints_only
                 else "no sha256 in session.json")
        return proto, f"{proto.get('name', '?')} version {ver} ({rel}, {check})"
    return None, f"{rel} not found, so protocol values were not used"


def _take_fields(name: str) -> Tuple[str, Optional[int]]:
    m = TAKE_RE.match(name)
    if not m:
        return name, None
    return m.group("item"), int(m.group("take"))


def _resolve(session_dir: Path, name: str, accepted: bool,
             listed: Optional[dict]) -> Dict[str, Path]:
    base = session_dir if accepted else session_dir / "rejected"
    out = {k: base / v.format(name=name) for k, v in LAYOUT.items()}
    for k, v in (listed or {}).items():
        if v:
            out[k] = session_dir / str(v)
    return out


def _reason_file(session_dir: Path, name: str) -> str:
    p = session_dir / "rejected" / f"{name}.reason.txt"
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _scan_takes(session_dir: Path) -> List[TakeRef]:
    """Takes found on disk, for a session.json without a `takes` list."""
    out: List[TakeRef] = []
    for accepted, base in ((True, session_dir),
                           (False, session_dir / "rejected")):
        names = set()
        for sub, suffix in (("glove", ".jsonl"), ("leap", ".jsonl"),
                            ("events", ".events.jsonl"), ("meta", ".json")):
            folder = base / sub
            if not folder.is_dir():
                continue
            for p in folder.iterdir():
                if p.name.endswith(".settle.jsonl"):
                    continue
                if p.name.endswith(suffix):
                    names.add(p.name[:-len(suffix)])
        for name in sorted(names):
            item, take = _take_fields(name)
            out.append(TakeRef(
                name=name, item=item, take=take, accepted=accepted,
                reason="" if accepted else _reason_file(session_dir, name),
                files=_resolve(session_dir, name, accepted, None)))
    return out


def session_takes(session_dir: Path, meta: dict) -> List[TakeRef]:
    """Every take in the order session.json lists them (disk scan fallback)."""
    listed = meta.get("takes")
    if not isinstance(listed, list) or not listed:
        return _scan_takes(session_dir)
    out: List[TakeRef] = []
    for t in listed:
        name = str(t.get("name", ""))
        if not name:
            continue
        item, take = _take_fields(name)
        accepted = bool(t.get("accepted", False))
        reason = str(t.get("reason", "") or "")
        if not accepted and not reason:
            reason = _reason_file(session_dir, name)
        out.append(TakeRef(
            name=name, item=str(t.get("item") or item),
            take=t.get("take", take), accepted=accepted, reason=reason,
            files=_resolve(session_dir, name, accepted, t.get("files"))))
    return out


def mock_flags(meta: dict) -> List[str]:
    """Which mock flags a session.json sets; empty for a real session.

    Three shapes are in use: `"mock": true` (Set A, and Sets B and C after
    their fix), `"mock": {"glove": true, ...}` (Sets B and C before it) and
    `"mock_flags": {...}` (Sets B and C after it). Any true value counts.
    """
    out: List[str] = []
    for key in ("mock", "mock_flags"):
        v = meta.get(key)
        if isinstance(v, dict):
            out += [f"{key}.{k}" for k, flag in v.items() if flag]
        elif v:
            out.append(key)
    return out


def load_session(session_dir) -> Session:
    """A session folder -> `Session`. Raises FileNotFoundError without one."""
    session_dir = Path(session_dir)
    meta = read_json(session_dir / "session.json")
    if not isinstance(meta, dict):
        raise FileNotFoundError(f"no readable session.json in {session_dir}")
    kind = meta.get("set")
    if kind not in SETS:
        kind = next((s for s in SETS if s in session_dir.parts), str(kind))
    hand = str(meta.get("hand") or "")
    if hand not in ("left", "right"):
        hand = "right" if session_dir.name.endswith("_right") else "left"
    warmup = read_json(session_dir / "warmup.json")
    protocol, note = load_protocol(session_dir, meta)
    return Session(path=session_dir, meta=meta, set=kind, hand=hand,
                   warmup=warmup if isinstance(warmup, dict) else None,
                   protocol=protocol, protocol_note=note,
                   takes=session_takes(session_dir, meta))


def read_events(path: Optional[Path]) -> List[dict]:
    if path is None:
        return []
    out = []
    # errors="replace": a file that is not text at all yields no events
    # rather than an exception half way through a session.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    out.append(d)
    return out


def endpoints(warmup: Optional[dict], sensor: str
              ) -> Optional[Dict[str, Tuple[float, float]]]:
    """{finger: (open, fist)} from warmup.json for one sensor, or None."""
    if not warmup or warmup.get("refused"):
        return None
    block = warmup.get(sensor)
    if not isinstance(block, dict):
        return None
    opn, fst = block.get("open") or {}, block.get("fist") or {}
    out = {}
    for f in FINGERS:
        if opn.get(f) is not None and fst.get(f) is not None:
            out[f] = (float(opn[f]), float(fst[f]))
    return out or None


def fractions(curls, ends: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """(n, 5) curls -> (n, 5) warm-up fractions, NaN where not normalisable."""
    c = np.asarray(curls, dtype=float).reshape(-1, len(FINGERS))
    out = np.full(c.shape, np.nan)
    for i, f in enumerate(FINGERS):
        if f not in ends:
            continue
        opn, fst = ends[f]
        if abs(opn - fst) < 1e-9:
            continue
        out[:, i] = np.clip((opn - c[:, i]) / (opn - fst), *FRACTION_CLIP)
    return out


def fraction_of(curl: Optional[float], ends: Optional[Tuple[float, float]]
                ) -> Optional[float]:
    """One curl -> its clipped fraction of (open, fist), or None."""
    if curl is None or ends is None or abs(ends[0] - ends[1]) < 1e-9:
        return None
    return float(np.clip((ends[0] - curl) / (ends[0] - ends[1]),
                         *FRACTION_CLIP))


def span_of(values) -> Optional[float]:
    """5th-to-95th percentile range, the `diagnostics.camera_range` rule."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, list(RANGE_PERCENTILES))
    return float(hi - lo)


def choose_camera_label(counts: Dict[str, int], operator_hand: str
                        ) -> Tuple[Optional[str], str]:
    """Which tracker label holds the operator's hand in this file, and a note.

    The operator's own label when it has at least as many frames as any
    other, otherwise the label with the most frames: the hand held over the
    module is the one the tracker saw most, whatever it called it.
    """
    if not counts:
        return None, ""
    best = max(sorted(counts), key=lambda s: counts[s])
    label = operator_hand if counts.get(operator_hand, 0) >= counts[best] \
        else best
    others = {k: v for k, v in counts.items() if k != label}
    note = ""
    if label != operator_hand:
        note = (f"camera: the tracker labelled the hand '{label}' on "
                f"{counts[label]} frame(s) (operator's hand is "
                f"{operator_hand}); those frames were used")
    elif others:
        note = ("camera: ignored " + ", ".join(
            f"{n} frame(s) labelled {k}" for k, n in sorted(others.items())))
    return label, note


# --- one sensor's stream over one take ----------------------------------------

@dataclass
class Stream:
    """One sensor's lines of one take, reduced to what the checks read."""

    rows: List[dict] = field(default_factory=list)
    curls: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    counters: List[Optional[int]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    label: Optional[str] = None
    note: str = ""
    bad_lines: int = 0

    @property
    def n(self) -> int:
        return len(self.rows)


def read_frames(path: Path) -> Tuple[List[Tuple[dict, object]], int]:
    """[(line dict, HandFrame)] for every parseable line, and the bad count."""
    out = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                frame, _wall = _dict_to_frame(d)
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            out.append((d, frame))
    return out, bad


def load_stream(path: Optional[Path], hand: str, camera: bool) -> Stream:
    """A glove or camera JSONL take -> `Stream` of the operator's hand.

    The glove's label is its own device's and is trusted; the camera's is the
    tracker's guess and goes through `choose_camera_label`. Camera rows are
    re-labelled with the operator's hand so `pair_by_time` can match them to
    the glove, while the palm normal is still computed with the tracker's own
    label, because that is the chirality its skeleton was fitted as.
    A line that does not parse (a take cut off mid-write) is counted and
    skipped, not fatal.
    """
    if path is None:
        return Stream()
    raw, bad = read_frames(path)
    counts: Dict[str, int] = {}
    for d, _fr in raw:
        counts[d.get("hand_side")] = counts.get(d.get("hand_side"), 0) + 1
    if camera:
        label, note = choose_camera_label(counts, hand)
    else:
        label = hand
        others = {k: v for k, v in counts.items() if k != hand}
        note = ("" if not others else "glove: ignored " + ", ".join(
            f"{n} frame(s) of the {k} glove" for k, n in sorted(others.items())))
    rows, curls, counters = [], [], []
    for d, frame in raw:
        if d.get("hand_side") != label:
            continue
        row = {"wall_time": float(d["wall_time"]),
               CAPTURE_CLOCK: d.get(CAPTURE_CLOCK),
               "hand_side": hand}
        if camera:
            abs26 = d.get("abs26")
            row.update({
                "hand_id": d.get("hand_id"),
                "visible_time_us": d.get("visible_time_us"),
                "palm_abs": d.get("palm_abs"),
                "palm_normal_abs": (palm_normal_abs(abs26, label)
                                    if abs26 else None),
            })
        rows.append(row)
        curls.append(flexion_features(frame_to_keypoints21(frame)))
        counters.append(d.get("packet_counter"))
    return Stream(rows=rows,
                  curls=np.asarray(curls, dtype=float).reshape(-1, 5),
                  counters=counters, counts=counts, label=label, note=note,
                  bad_lines=bad)


def stamps(rows: Sequence[dict], clock: str) -> np.ndarray:
    """Each row's time on `clock`, wall_time where the row lacks it."""
    return np.asarray([float(r[clock]) if r.get(clock) is not None
                       else float(r["wall_time"]) for r in rows], dtype=float)


@dataclass
class TakeData:
    """Both sensors of one take on one clock, plus its events."""

    ref: TakeRef
    events: List[dict]
    t0: Optional[float]
    t1: Optional[float]
    clock: str
    glove_t: np.ndarray
    glove_c: np.ndarray
    cam_t: np.ndarray
    cam_c: np.ndarray
    cam_ok: np.ndarray
    fields: dict
    notes: List[str]

    def glove_in(self, a: float, b: float) -> np.ndarray:
        return (self.glove_t >= a) & (self.glove_t <= b)

    def cam_in(self, a: float, b: float) -> np.ndarray:
        """Trusted camera frames inside [a, b]."""
        return self.cam_ok & (self.cam_t >= a) & (self.cam_t <= b)


def _rnd(v, nd: int = 4):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else round(float(v), nd)
    return v


def load_take(session: Session, ref: TakeRef) -> TakeData:
    """Read one take's glove, camera and events and put them on one clock."""
    notes: List[str] = []
    glove = load_stream(ref.path("glove"), session.hand, camera=False)
    cam = load_stream(ref.path("leap"), session.hand, camera=True)
    if ref.path("glove") is None:
        notes.append("no glove file")
    for s in (glove, cam):
        if s.note:
            notes.append(s.note)
        if s.bad_lines:
            notes.append(f"{s.bad_lines} unreadable line(s) skipped")
    events = read_events(ref.path("events"))
    starts = [e["t"] for e in events
              if e.get("kind") == "take_start" and "t" in e]
    ends = [e["t"] for e in events if e.get("kind") == "take_end" and "t" in e]

    streams = [s.rows for s in (glove, cam) if s.n]
    clock = pairing_clock(*streams) if streams else WALL_CLOCK

    def ordered(s: Stream):
        t = stamps(s.rows, clock)
        order = np.argsort(t, kind="stable")
        return (t[order], s.curls[order] if s.n else s.curls,
                [s.rows[i] for i in order], [s.counters[i] for i in order])

    gt, gc, grows, gcount = ordered(glove)
    ct, cc, crows, _ = ordered(cam)
    t0 = float(starts[0]) if starts else (float(gt[0]) if gt.size else None)
    t1 = float(ends[-1]) if ends else (float(gt[-1]) if gt.size else None)

    cam_ok = np.zeros(ct.shape, dtype=bool)
    if crows:
        # Needs the whole take at once: "did the id change 0.25 s ago" is a
        # question about the frames around this one (as in fuse_poses.py).
        flag_hand_id_stability(crows, DEFAULT_GATES, clock=clock)
        cam_ok = np.asarray([camera_trusted(
            {"visible_time_us": r.get("visible_time_us"),
             "hand_id_stable": r.get("hand_id_stable"),
             "palm_abs": r.get("palm_abs"),
             "palm_normal_abs": r.get("palm_normal_abs")}, DEFAULT_GATES)
            for r in crows], dtype=bool)

    has_cam = ref.path("leap") is not None
    fields = {"frames_glove": glove.n,
              "frames_camera": cam.n if has_cam else None,
              "camera_label": cam.label, "clock": clock}
    if gt.size:
        log = StreamLog(gap_s=0.1)
        for t, c in zip(gt, gcount):
            log.add(float(t), c)
        a = min(t0, float(gt[0])) if t0 is not None else None
        b = max(t1, float(gt[-1])) if t1 is not None else None
        summ = log.summary(a, b)
        fields.update(glove_hz=summ.get("rate_hz"),
                      glove_gap_max_ms=summ.get("max_gap_ms"),
                      glove_lost_packets=summ.get("lost_packets"))
    if has_cam:
        fields["camera_trusted_fraction"] = (
            _rnd(float(cam_ok.mean())) if cam_ok.size else 0.0)
        if grows:
            pairs = pair_by_time(grows, crows, max_dt=PAIR_MAX_DT, clock=clock)
            fields["paired_fraction"] = _rnd(
                sum(1 for _g, c in pairs if c is not None) / len(pairs))
    if not events:
        notes.append("no events file" if ref.path("events") is None
                     else "the events file holds no events")
    return TakeData(ref=ref, events=events, t0=t0, t1=t1, clock=clock,
                    glove_t=gt, glove_c=gc, cam_t=ct, cam_c=cc, cam_ok=cam_ok,
                    fields=fields, notes=notes)


# --- Set B ---------------------------------------------------------------------

def cued_finger(bends: Sequence[dict], item_cfg: Optional[dict],
                item_id: str) -> Optional[str]:
    """The finger a flexion take cues: the events, then the protocol, then the id."""
    named = {f for e in bends for f in flexed_set(e.get("flexed"))}
    if len(named) == 1:
        return named.pop()
    if item_cfg and finger_name(item_cfg.get("finger", "")):
        return finger_name(item_cfg["finger"])
    for f in FINGERS:
        if item_id == f or item_id.startswith(f + "_"):
            return f
    return finger_name(item_id.split("_")[0])


def _median_or_none(values) -> Optional[float]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.median(v)) if v else None


def check_flexion_take(session: Session, data: TakeData
                       ) -> Tuple[dict, List[str], List[str]]:
    """Set B verdict for one take: (csv fields, failures, detail lines)."""
    fields: dict = {}
    failures: List[str] = []
    ref = data.ref
    item_cfg = session.item(ref.item)
    cues = sorted((e for e in data.events if e.get("kind") == "cue"),
                  key=lambda e: e.get("t", 0.0))
    bends = [e for e in cues if e.get("phase") == "bend" and "t" in e]
    finger = cued_finger(bends, item_cfg, ref.item)
    fields["cued_finger"] = finger
    cycles_ev = len({e.get("cycle", k) for k, e in enumerate(bends)})
    fields["cycles_from_events"] = cycles_ev
    expected = item_cfg.get("cycles") if item_cfg else None
    fields["cycles_expected"] = expected

    t0 = data.t0 if data.t0 is not None else -math.inf
    t1 = data.t1 if data.t1 is not None else math.inf
    gm = data.glove_in(t0, t1)
    gt, gc = data.glove_t[gm], data.glove_c[gm]
    ends = endpoints(session.warmup, GLOVE)

    if finger is None:
        failures.append("no cued finger in the events or the protocol")
    if gt.size == 0:
        failures.append("no glove frames in the take")
    if ends is None:
        failures.append("no glove warm-up endpoints, so the span rule cannot "
                        "be applied")
    if not bends:
        failures.append("no bend cues in the events")
    elif expected is not None and cycles_ev != int(expected):
        failures.append(f"{cycles_ev} bend cue(s), the protocol asks for "
                        f"{expected}")

    idx = FINGERS.index(finger) if finger else None
    frac = None
    if ends is not None and gt.size and idx is not None:
        frac = fractions(gc, ends)
        spans = {f: span_of(frac[:, i]) for i, f in enumerate(FINGERS)}
        cued = spans[finger]
        fields["cued_span_fraction"] = _rnd(cued)
        fields["other_spans"] = {f: _rnd(spans[f]) for f in FINGERS
                                 if f != finger}
        peaks = count_peaks(frac[:, idx])
        fields["cycles_from_peaks"] = peaks
        if cued is None:
            failures.append(f"{finger} cannot be put on a fraction of its "
                            "warm-up range")
        elif cued < MIN_CUED_SPAN:
            failures.append(f"{finger} spans {cued:.2f} of its warm-up range, "
                            f"need {MIN_CUED_SPAN:.2f}")
        if bends and peaks != cycles_ev:
            failures.append(f"{cycles_ev} bend cue(s) but {peaks} glove curl "
                            "peak(s)")

    detail: List[str] = []
    if idx is None or gt.size == 0:
        return fields, failures, detail
    bend_t = [float(e["t"]) for e in bends]
    last = t1 if math.isfinite(t1) else float(gt[-1])
    bounds = list(zip(bend_t, bend_t[1:] + [last]))

    if not data.cam_t.size:
        detail.append(f"{ref.item} ({ref.name}), cued finger {finger}: no "
                      "camera frames, glove only; no transfer curve or lag")
        detail.extend(_cycle_table(
            [_cycle_row(bends[k], k, gt, a, b, frac, idx)
             for k, (a, b) in enumerate(bounds)], camera=False))
        return fields, failures, detail

    # --- the camera as the reference -------------------------------------
    cm = data.cam_in(t0, t1)
    ct, cv = data.cam_t[cm], data.cam_c[cm, idx]
    rail = ends[finger][0] if ends and finger in ends else \
        open_reference(gc)[idx]
    sweep = analyse_finger(gt, gc[:, idx], ct, cv, rail, finger=finger)
    fields["camera_span"] = _rnd(sweep.span.span) if sweep.span.n else None
    cycles: List[dict] = []
    for k, (a, b) in enumerate(bounds):
        row = _cycle_row(bends[k], k, gt, a, b, frac, idx)
        g_in = (gt >= a) & (gt < b)
        c_in = (ct >= a) & (ct < b)
        rng = camera_range(cv[c_in], finger=finger)
        row["camera_span"] = _rnd(rng.span, 3) if rng.n else None
        row["followed"] = rng.followed
        if rng.followed:
            lag = estimate_lag(gt[g_in], gc[g_in, idx], ct[c_in], cv[c_in])
            shift = 0.0
            if lag is not None:
                row["lag_ms"] = _rnd(lag.ms, 1)
                shift = lag.seconds
            cam_at = np.interp(gt[g_in] - shift, ct[c_in], cv[c_in])
            row["hysteresis"] = _rnd(_median_or_none(
                b_.hysteresis for b_ in transfer_curve(cam_at, gc[g_in, idx])),
                3)
        cycles.append(row)
    followed = [c for c in cycles if c["followed"]]
    fields["cycles_camera_followed"] = len(followed)
    fields["frames_camera_followed"] = sum(c["frames"] for c in followed)
    fields["lag_ms_cycles"] = [c["lag_ms"] for c in cycles]
    fields["lag_ms_median"] = _rnd(_median_or_none(
        c["lag_ms"] for c in followed), 1)
    if sweep.measured:
        fields["lag_ms_take"] = (_rnd(sweep.lag.ms, 1)
                                 if sweep.lag is not None else None)
        fields["hysteresis_median"] = _rnd(_median_or_none(
            b_.hysteresis for b_ in sweep.bins), 3)
    detail.extend(_flexion_detail(ref, finger, sweep, cycles))
    return fields, failures, detail


def _cycle_row(bend: dict, k: int, gt: np.ndarray, a: float, b: float,
               frac: Optional[np.ndarray], idx: int) -> dict:
    g_in = (gt >= a) & (gt < b)
    peak = None
    if frac is not None and g_in.any():
        vals = frac[g_in, idx]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            peak = _rnd(float(np.percentile(vals, 95.0)), 3)
    return {"cycle": bend.get("cycle", k + 1), "frames": int(g_in.sum()),
            "glove_peak": peak, "camera_span": None, "followed": False,
            "lag_ms": None, "hysteresis": None}


def fmt(v, spec: str = ".2f", none: str = "-") -> str:
    if v is None:
        return none
    try:
        if isinstance(v, float) and not math.isfinite(v):
            return none
        text = format(v, spec)
    except (TypeError, ValueError):
        return str(v)
    # "-0.00" and "+0.000" are rounding artefacts, not signs worth reading
    if text[:1] in "+-" and text.strip("-+0.") == "":
        text = text[1:]
    return text


def _cycle_table(cycles: Sequence[dict], camera: bool = True) -> List[str]:
    if not camera:
        out = ["  cycle  glove frames  glove peak (95th pct fraction)"]
        for c in cycles:
            out.append(f"  {str(c['cycle']):>5}  {fmt(c['frames'], 'd'):>12}"
                       f"  {fmt(c['glove_peak']):>10}")
        return out
    out = ["  cycle  glove frames  glove peak  camera span  followed  "
           "lag ms  hysteresis"]
    for c in cycles:
        out.append(f"  {str(c['cycle']):>5}  {fmt(c['frames'], 'd'):>12}  "
                   f"{fmt(c['glove_peak']):>10}  {fmt(c['camera_span']):>11}"
                   f"  {'yes' if c['followed'] else 'no':>8}  "
                   f"{fmt(c['lag_ms'], '.0f'):>6}  "
                   f"{fmt(c['hysteresis'], '+.3f'):>10}")
    return out


def _flexion_detail(ref: TakeRef, finger: str, sweep, cycles) -> List[str]:
    """check.txt lines for one flexion take: range guard, lag, curve, cycles."""
    followed = [c for c in cycles if c["followed"]]
    out = [f"{ref.item} ({ref.name}), cued finger {finger}:"]
    if sweep.span.n:
        out.append(f"  camera span over the take {sweep.span.span:.2f} "
                   f"(camera curl {sweep.span.low:.2f} to {sweep.span.high:.2f},"
                   f" need {MIN_CAMERA_RANGE:.2f}); {len(followed)} of "
                   f"{len(cycles)} cycle(s) and "
                   f"{sum(c['frames'] for c in followed)} glove frame(s) met "
                   "the span rule")
    if not sweep.measured:
        out.extend("  " + line for line in camera_range_verdict(sweep.span))
        out.extend(_cycle_table(cycles))
        return out
    lag = sweep.lag
    per_cycle = _median_or_none(c["lag_ms"] for c in followed)
    out.append("  lag over the take " + (
        f"{lag.ms:.0f} ms (correlation {lag.correlation:.2f})"
        if lag is not None else "not measurable")
        + "; median of the cycles " + fmt(per_cycle, ".0f") + " ms"
        + " (positive = the glove trails the camera)")
    out.extend(_cycle_table(cycles))
    out.append("  transfer curve: glove curl binned by camera curl "
               "(hysteresis = bending minus straightening)")
    out.append("    camera       n   glove  bending  straightening  hysteresis")
    for b in sweep.bins:
        out.append(f"    {b.low:4.1f}-{b.high:3.1f}  {b.n:5d}  "
                   f"{fmt(b.glove, '.3f'):>6}  "
                   f"{fmt(b.glove_bending, '.3f'):>7}"
                   f"  {fmt(b.glove_straightening, '.3f'):>13}  "
                   f"{fmt(b.hysteresis, '+.3f'):>10}")
    out.append("  rail saturation: " + (
        f"the glove sits on its open reading {sweep.rail:.3f} from camera "
        f"curl {sweep.saturation:.2f} (bin centre) upward"
        if sweep.saturation is not None else
        "the glove never sits on its open reading in this take"))
    return out


# --- Set C ---------------------------------------------------------------------

def step_windows(data: TakeData, hold_default: float, window_s: float
                 ) -> List[dict]:
    """Each cued step with its check window: the last `window_s` of its hold.

    `hold_default` and `window_s` are the protocol's. A cue whose own
    `hold_s` differs (a recorder run with its timing scaled, a mock session)
    keeps the same SHARE of its hold, so the movement at the start of the
    hold stays out of the window whatever the timing.
    """
    cues = sorted((e for e in data.events
                   if e.get("kind") == "cue" and "t" in e),
                  key=lambda e: (e.get("t", 0.0), e.get("step", 0)))
    out = []
    for k, e in enumerate(cues):
        hold = float(e.get("hold_s") or hold_default)
        share = min(1.0, window_s / hold_default) if hold_default > 0 else 1.0
        w1 = float(e["t"]) + hold
        w0 = w1 - share * hold
        out.append({"step": e.get("step", k), "label": e.get("label", ""),
                    "flexed": list(flexed_set(e.get("flexed"))),
                    "window": [w0, w1]})
    return out


def measure_steps(data: TakeData, steps: List[dict]) -> List[dict]:
    """Per step, the median curl of each finger in its window, per sensor."""
    out = []
    for s in steps:
        w0, w1 = s["window"]
        m = dict(s)
        gm = data.glove_in(w0, w1)
        g_cov = coverage(data.glove_t[gm], w0, w1) if gm.any() else 0.0
        m["glove_frames"] = int(gm.sum())
        m["glove_curl"] = ({f: float(np.median(data.glove_c[gm, i]))
                            for i, f in enumerate(FINGERS)}
                           if g_cov >= MIN_WINDOW_COVERAGE else None)
        cm = data.cam_in(w0, w1)
        c_cov = coverage(data.cam_t[cm], w0, w1) if cm.any() else 0.0
        m["camera_frames"] = int(cm.sum())
        m["camera_curl"] = ({f: float(np.median(data.cam_c[cm, i]))
                             for i, f in enumerate(FINGERS)}
                            if c_cov >= MIN_WINDOW_COVERAGE else None)
        out.append(m)
    return out


def band_refs(session: Session, measured: Dict[str, List[dict]],
              accepted: Dict[str, bool], bands: str
              ) -> Tuple[Dict[str, Dict[str, Optional[Tuple[float, float]]]],
                         List[str]]:
    """The (open, fist) curl each finger's fraction is taken against, per sensor.

    `fixed`: the warm-up. `auto`: the session's own open-hand and full-fist
    steps (accepted takes), falling back to the warm-up per end. Returns the
    references and one line per sensor saying where they came from.
    """
    refs: Dict[str, Dict[str, Optional[Tuple[float, float]]]] = {}
    lines: List[str] = []
    for sensor in (GLOVE, CAMERA):
        warm = endpoints(session.warmup, sensor) or {}
        if bands == FIXED:
            refs[sensor] = {f: warm.get(f) for f in FINGERS}
            lines.append(f"{sensor} against the warm-up" if warm else
                         f"{sensor} has no warm-up endpoints")
            continue
        opens: Dict[str, list] = {f: [] for f in FINGERS}
        fists: Dict[str, list] = {f: [] for f in FINGERS}
        n_open = n_fist = 0
        for name, steps in measured.items():
            if not accepted.get(name):
                continue
            for s in steps:
                curl = s.get(f"{sensor}_curl")
                if curl is None:
                    continue
                if not s["flexed"]:
                    n_open += 1
                    for f in FINGERS:
                        opens[f].append(curl[f])
                elif len(s["flexed"]) == len(FINGERS):
                    n_fist += 1
                    for f in FINGERS:
                        fists[f].append(curl[f])
        ref = {}
        for f in FINGERS:
            o = float(np.median(opens[f])) if opens[f] else (
                warm[f][0] if f in warm else None)
            c = float(np.median(fists[f])) if fists[f] else (
                warm[f][1] if f in warm else None)
            ref[f] = (o, c) if o is not None and c is not None else None
        refs[sensor] = ref
        lines.append(
            f"{sensor} open from {n_open} open-hand step(s)"
            + ("" if n_open else " (none, warm-up used)")
            + f", fist from {n_fist} full-fist step(s)"
            + ("" if n_fist else " (none, warm-up used)"))
    return refs, lines


def judge_step(curl: Optional[Dict[str, float]], flexed: Sequence[str],
               ref: Dict[str, Optional[Tuple[float, float]]]
               ) -> Tuple[Optional[bool], Dict[str, Optional[float]],
                          List[str], Dict[str, float]]:
    """(pass, fractions, wrong fingers, coupling) for one step on one sensor.

    A flexed finger must read above FLEXED_ABOVE; a straight finger fails
    only at FLEXED_ABOVE or more, and between STRAIGHT_BELOW and that it is
    coupling: {finger: fraction}, reported, not failed. None for pass when
    there is no data in the window. A finger that cannot be put on a
    fraction is wrong: the step cannot be shown to have passed.
    """
    if curl is None:
        return None, {f: None for f in FINGERS}, [], {}
    fr = {f: fraction_of(curl[f], ref.get(f)) for f in FINGERS}
    wrong: List[str] = []
    coupling: Dict[str, float] = {}
    for f in FINGERS:
        v = fr[f]
        if v is None:
            wrong.append(f"{f} not normalisable")
        elif f in flexed and not v > FLEXED_ABOVE:
            wrong.append(f"{f} {fmt(v)} not flexed")
        elif f not in flexed and v >= FLEXED_ABOVE:
            wrong.append(f"{f} {fmt(v)} reads flexed, should be straight")
        elif f not in flexed and v >= STRAIGHT_BELOW:
            coupling[f] = _rnd(v, 3)
    return (not wrong), fr, wrong, coupling


def check_sequence_take(session: Session, data: TakeData, steps: List[dict],
                        refs, bands: str) -> Tuple[dict, List[str], List[str]]:
    """Set C verdict for one take from its measured steps."""
    fields: dict = {"bands": bands}
    failures: List[str] = []
    item_cfg = session.item(data.ref.item)
    if data.glove_t.size == 0:
        failures.append("no glove frames in the take")
    if not steps:
        failures.append("no cued steps in the events")
    elif item_cfg and item_cfg.get("steps") is not None and \
            len(steps) != len(item_cfg["steps"]):
        failures.append(f"{len(steps)} step(s) cued, the protocol has "
                        f"{len(item_cfg['steps'])}")
    glove_refs = any(v is not None for v in refs[GLOVE].values())
    camera_refs = any(v is not None for v in refs[CAMERA].values())
    if steps and not glove_refs:
        failures.append("no glove endpoints to put the steps on a fraction"
                        + (" (no warm-up; try --bands auto)" if bands == FIXED
                           else ""))
    judged = []
    for s in steps:
        g_ok, g_fr, g_wrong, g_cpl = judge_step(
            s["glove_curl"] if glove_refs else None, s["flexed"], refs[GLOVE])
        c_ok, c_fr, c_wrong, c_cpl = judge_step(
            s["camera_curl"] if camera_refs else None, s["flexed"],
            refs[CAMERA])
        if not glove_refs:
            g_wrong = ["no glove endpoints"]
        elif g_ok is None:
            g_wrong = ["no glove frames over the window"]
        judged.append({
            "step": s["step"], "label": s["label"], "flexed": s["flexed"],
            "window": [_rnd(s["window"][0], 3), _rnd(s["window"][1], 3)],
            "glove": {f: _rnd(v, 3) for f, v in g_fr.items()},
            "camera": {f: _rnd(v, 3) for f, v in c_fr.items()},
            "glove_curl": ({f: _rnd(v, 3) for f, v in s["glove_curl"].items()}
                           if s["glove_curl"] else None),
            "camera_curl": ({f: _rnd(v, 3)
                             for f, v in s["camera_curl"].items()}
                            if s["camera_curl"] else None),
            "glove_pass": bool(g_ok), "camera_pass": c_ok,
            "glove_wrong": g_wrong, "camera_wrong": c_wrong,
            # straight fingers between the bands: reported, not failed
            "coupling": {GLOVE: g_cpl, CAMERA: c_cpl},
            "glove_frames": s["glove_frames"],
            "camera_frames": s["camera_frames"]})
        if not g_ok and glove_refs:
            failures.append(f"step {s['step']} ({s['label']}): "
                            + "; ".join(g_wrong))
    fields["steps_total"] = len(judged)
    fields["steps_pass_glove"] = sum(1 for j in judged if j["glove_pass"])
    fields["steps_judged_camera"] = sum(1 for j in judged
                                        if j["camera_pass"] is not None)
    fields["steps_pass_camera"] = sum(1 for j in judged if j["camera_pass"])
    fields["coupling_glove"] = sum(len(j["coupling"][GLOVE]) for j in judged)
    fields["coupling_camera"] = sum(len(j["coupling"][CAMERA])
                                    for j in judged)
    fields["steps"] = judged
    return fields, failures, _sequence_detail(data.ref, judged, bands)


def _fr_text(fr: Dict[str, Optional[float]]) -> str:
    return " ".join(f"{fmt(fr.get(f)):>5}" for f in FINGERS)


def _sequence_detail(ref: TakeRef, judged: List[dict], bands: str
                     ) -> List[str]:
    out = [f"{ref.item} ({ref.name}), bands {bands}: median fraction per "
           "finger over each step's check window (thumb index middle ring "
           "pinky)"]
    out.append(f"  {'step':>4}  {'label':<22} {'flexed':<22} "
               f"{'glove':<29} {'camera':<29} glove camera")
    for j in judged:
        want = ("all five" if len(j["flexed"]) == len(FINGERS) else
                ",".join(j["flexed"]) or "none (open)")
        cam = ("  -  " if j["camera_pass"] is None else
               ("PASS " if j["camera_pass"] else "FAIL "))
        out.append(f"  {str(j['step']):>4}  {j['label'][:22]:<22} "
                   f"{want[:22]:<22} {_fr_text(j['glove']):<29} "
                   f"{_fr_text(j['camera']):<29} "
                   f"{'PASS ' if j['glove_pass'] else 'FAIL '} {cam}")
        if j["glove_wrong"]:
            out.append(f"        glove: {'; '.join(j['glove_wrong'])}")
        if j["camera_wrong"]:
            out.append(f"        camera: {'; '.join(j['camera_wrong'])}")
        for sensor in (GLOVE, CAMERA):
            if j["coupling"][sensor]:
                out.append(f"        {sensor} coupling (not a failure): "
                           + ", ".join(f"{f} {fmt(v)}" for f, v in
                                       j["coupling"][sensor].items()))
    return out


# --- Set A ---------------------------------------------------------------------

GRASP_VALUES = ("grab_strength", "pinch_strength") + tuple(
    f"curl_{f}" for f in FINGERS)


def check_grasp_take(ref: TakeRef) -> Tuple[dict, List[str], List[str]]:
    """Set A light pass for one take: its meta fields and the acquisition gate."""
    fields: dict = {}
    failures: List[str] = []
    meta = read_json(ref.path("meta")) if ref.path("meta") else None
    if not isinstance(meta, dict):
        return fields, ["no meta file"], []
    tracked = meta.get("tracked_fraction")
    reacq = meta.get("reacquisitions")
    # The gate is about the static interval. The contract's `reacquisitions`
    # does not say over which span it counts, and the Set A recorder counts
    # the whole take there and the interval under `gate`, so the interval's
    # own count wins wherever the meta has it.
    gate = meta.get("gate") if isinstance(meta.get("gate"), dict) else {}
    in_interval = gate.get("reacquisitions_in_static_interval", reacq)
    fields.update(frames_camera=meta.get("frames"),
                  tracked_fraction=_rnd(tracked),
                  reacquisitions=reacq,
                  grab_strength=_rnd(meta.get("grab_strength")),
                  pinch_strength=_rnd(meta.get("pinch_strength")),
                  curls={f: _rnd((meta.get("curls") or {}).get(f))
                         for f in FINGERS})
    if tracked is None or float(tracked) < GRASP_MIN_TRACKED:
        failures.append(f"tracked {fmt(tracked)} of the take, need "
                        f"{GRASP_MIN_TRACKED:.2f}")
    if in_interval is None or int(in_interval) > 0:
        failures.append(f"{in_interval} re-acquisition(s) inside the static "
                        "interval")
    # The tracker's label on the summary frame: a skeleton fitted as the
    # other hand is a mirrored model of the operator's, whatever was
    # measured from it.
    label, hand = meta.get("operator_hand_label"), meta.get("hand")
    if isinstance(label, str) and isinstance(hand, str) and label != hand:
        failures.append(f"summary frame is the tracker's {label} hand "
                        f"(operator's hand {hand})")
    notes = []
    others = meta.get("other_hand_ids")
    if others:
        ids = [str(i) for i in others] if isinstance(others, list) else [str(others)]
        notes.append(f"another hand in view ({'id' if len(ids) == 1 else 'ids'} "
                     f"{', '.join(ids)}), ignored")
    if reacq is not None and in_interval is not None and             int(reacq) > int(in_interval):
        notes.append(f"{int(reacq) - int(in_interval)} re-acquisition(s) "
                     "outside the static interval")
    if meta.get("static_interval"):
        a, b = meta["static_interval"][:2]
        notes.append(f"static interval {float(b) - float(a):.2f} s")
    if meta.get("orientation_note"):
        notes.append(f"orientation: {meta['orientation_note']}")
    return fields, failures, notes


def grasp_value(row: dict, key: str) -> Optional[float]:
    if key.startswith("curl_"):
        v = (row.get("curls") or {}).get(key[5:])
    else:
        v = row.get(key)
    return None if v is None else float(v)


def grasp_outliers(rows: List[dict]) -> Dict[str, List[str]]:
    """{take name: [flag, ...]} for takes far from their grasp's other takes.

    The band is the session's own: per value, the median distance of a take
    from its grasp's median, pooled over every grasp with at least three
    accepted takes; a take more than `OUTLIER_FACTOR` times that away (and at
    least `OUTLIER_FLOOR`) is flagged for a second look, never failed.
    """
    flags: Dict[str, List[str]] = {}
    ok = [r for r in rows if r.get("accepted")]
    for key in GRASP_VALUES:
        by_item: Dict[str, List[Tuple[str, float]]] = {}
        for r in ok:
            v = grasp_value(r, key)
            if v is not None:
                by_item.setdefault(r["item"], []).append((r["name"], v))
        dev: List[Tuple[str, float]] = []
        for vals in by_item.values():
            if len(vals) < 3:
                continue
            med = float(np.median([v for _n, v in vals]))
            dev.extend((n, abs(v - med)) for n, v in vals)
        if not dev:
            continue
        scale = float(np.median([d for _n, d in dev]))
        limit = max(OUTLIER_FACTOR * scale, OUTLIER_FLOOR)
        for n, d in dev:
            if d > limit:
                flags.setdefault(n, []).append(
                    f"{key} {d:.2f} from its grasp's median")
    return flags


# --- a whole session -------------------------------------------------------------

@dataclass
class SessionCheck:
    session: Session
    bands: str
    rows: List[dict]
    details: Dict[str, List[str]]
    header: List[str]

    @property
    def failed_accepted(self) -> List[dict]:
        return [r for r in self.rows if r["accepted"] and r["verdict"] == FAIL]

    @property
    def exit_code(self) -> int:
        """1 when an accepted take fails a rule; rejected takes never count."""
        return 1 if self.failed_accepted else 0


def _base_row(session: Session, ref: TakeRef) -> dict:
    row = {c: None for c in CSV_COLUMNS}
    row.update({"set": session.set, "session": session.name, "item": ref.item,
                "take": ref.take, "name": ref.name, "hand": session.hand,
                "accepted": ref.accepted, "reason": ref.reason})
    return row


def check_window_s(session: Session) -> Tuple[float, float]:
    """(hold_s, check_window_s) from the protocol, else the plan's defaults."""
    proto = session.protocol or {}
    return (float(proto.get("hold_s") or DEFAULT_HOLD_S),
            float(proto.get("check_window_s") or DEFAULT_CHECK_WINDOW_S))


def check_session(session_dir, bands: str = FIXED) -> SessionCheck:
    """Every take of one session, checked. See the module docstring."""
    if bands not in BANDS:
        raise ValueError(f"bands must be one of {BANDS}, not {bands!r}")
    session = session_dir if isinstance(session_dir, Session) else \
        load_session(session_dir)
    rows: List[dict] = []
    details: Dict[str, List[str]] = {}
    header: List[str] = [f"protocol  {session.protocol_note}"]
    mock = mock_flags(session.meta)
    if mock:
        header.append("MOCK      synthetic data, not a recording of a hand "
                      f"({', '.join(mock)})")

    if session.set == GRASPS:
        for ref in session.takes:
            row = _base_row(session, ref)
            fields, failures, notes = check_grasp_take(ref)
            row.update(fields)
            row["failures"] = failures
            row["notes"] = notes
            rows.append(row)
        for name, flags in grasp_outliers(rows).items():
            for r in rows:
                if r["name"] == name:
                    r["notes"] = r["notes"] + ["second look: " + f
                                               for f in flags]
    elif session.set in (FLEXION, SEQUENCES):
        header.append("warm-up   " + _warmup_text(session))
        loaded = [(ref, load_take(session, ref)) for ref in session.takes]
        clocks: Dict[str, int] = {}
        for _ref, data in loaded:
            clocks[data.clock] = clocks.get(data.clock, 0) + 1
        header.append("clock     " + (", ".join(
            f"{k} on {v} take(s)" for k, v in sorted(clocks.items()))
            or "no takes"))
        if session.set == FLEXION:
            for ref, data in loaded:
                row = _base_row(session, ref)
                row.update(data.fields)
                fields, failures, detail = check_flexion_take(session, data)
                row.update(fields)
                row["failures"] = failures
                row["notes"] = list(data.notes)
                rows.append(row)
                details[ref.name] = detail
        else:
            hold, window = check_window_s(session)
            header.append(f"window    last {window:g} s of each "
                          f"{hold:g} s hold (cue time + hold_s; the same "
                          "share of a cue whose hold_s differs)")
            measured = {ref.name: measure_steps(
                data, step_windows(data, hold, window))
                for ref, data in loaded}
            refs, ref_lines = band_refs(
                session, measured, {r.name: r.accepted for r, _d in loaded},
                bands)
            header.append(f"bands     {bands}: " + "; ".join(ref_lines))
            for ref, data in loaded:
                row = _base_row(session, ref)
                row.update(data.fields)
                fields, failures, detail = check_sequence_take(
                    session, data, measured[ref.name], refs, bands)
                row.update(fields)
                row["failures"] = failures
                row["notes"] = list(data.notes)
                rows.append(row)
                details[ref.name] = detail
    else:
        header.append(f"unknown set {session.set!r}: nothing checked")
    for r in rows:
        r["verdict"] = FAIL if r["failures"] else PASS
    return SessionCheck(session=session, bands=bands, rows=rows,
                        details=details, header=header)


def _warmup_text(session: Session) -> str:
    w = session.warmup
    if not w:
        return "none (no warmup.json)"
    if w.get("refused"):
        return f"REFUSED: {w['refused']}"
    parts = []
    for sensor in (GLOVE, CAMERA):
        e = endpoints(w, sensor)
        if e:
            parts.append(
                f"{sensor} open " + " ".join(fmt(e[f][0]) if f in e else "-"
                                             for f in FINGERS)
                + " fist " + " ".join(fmt(e[f][1]) if f in e else "-"
                                      for f in FINGERS))
        else:
            parts.append(f"{sensor} none")
    text = " | ".join(parts) + " (thumb..pinky, curl units)"
    ends = endpoints(w, GLOVE) or {}
    small = [f"{f} {ends[f][0] - ends[f][1]:.2f}" for f in FINGERS[1:]
             if f in ends and ends[f][0] - ends[f][1] < WARMUP_MIN_SPAN]
    if small:
        text += (f"; glove span under {WARMUP_MIN_SPAN:.2f} (contract "
                 "section 4 refuses such a session): " + ", ".join(small))
    return text


# --- output ----------------------------------------------------------------------

def csv_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"))
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return ""
        # Fixed point, at most 6 decimals: an epoch time keeps its digits
        # (".6g" would print 1.7902e+09) and a fraction keeps its precision.
        text = f"{float(v) + 0.0:.6f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text
    return str(v)


def csv_rows(check: SessionCheck) -> List[Dict[str, str]]:
    """The rows as check.csv holds them (strings; failures joined by '; ')."""
    out = []
    for r in check.rows:
        d = {}
        for c in CSV_COLUMNS:
            v = r.get(c)
            if c in ("failures", "notes"):
                v = "; ".join(v or [])
            d[c] = csv_value(v)
        out.append(d)
    return out


def read_check_csv(path) -> List[Dict[str, str]]:
    """check.csv back as string rows; the JSON_COLUMNS hold JSON text."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_is_current(session: Session, rows: Sequence[Dict[str, str]]
                     ) -> bool:
    """Does a check.csv cover exactly this session's takes, accepted or not?"""
    have = {(r.get("name"), r.get("accepted")) for r in rows}
    want = {(t.name, "true" if t.accepted else "false") for t in session.takes}
    return have == want


def _table(check: SessionCheck) -> List[str]:
    s = check.session
    out: List[str] = []
    if s.set == FLEXION:
        out.append(f"  {'take':<48} {'acc':<3} {'verdict':<7} {'glove':>6} "
                   f"{'Hz':>5} {'gap ms':>6} {'camera':>6} {'paired':>6} "
                   f"{'cued span':>15} {'cycles ev/pk/exp':>16} "
                   f"{'lag ms':>6}")
    elif s.set == SEQUENCES:
        out.append(f"  {'take':<48} {'acc':<3} {'verdict':<7} {'glove':>6} "
                   f"{'Hz':>5} {'gap ms':>6} {'camera':>6} {'paired':>6} "
                   f"{'steps':>5} {'glove ok':>8} {'camera ok':>10} "
                   f"{'coupling g/c':>12}")
    else:
        out.append(f"  {'take':<48} {'acc':<3} {'verdict':<7} {'frames':>6} "
                   f"{'tracked':>7} {'reacq':>5} {'grab':>5} {'pinch':>5}  "
                   "curls (thumb..pinky)")
    out.append("  " + "-" * 126)
    for r in check.rows:
        head = (f"  {r['name'][:48]:<48} "
                f"{'yes' if r['accepted'] else 'no':<3} "
                f"{r['verdict'].upper():<7}")
        if s.set in (FLEXION, SEQUENCES):
            head += (f" {fmt(r['frames_glove'], 'd'):>6} "
                     f"{fmt(r['glove_hz'], '.1f'):>5} "
                     f"{fmt(r['glove_gap_max_ms'], '.0f'):>6} "
                     f"{fmt(r['frames_camera'], 'd'):>6} "
                     f"{fmt(r['paired_fraction']):>6}")
        if s.set == FLEXION:
            cyc = (f"{fmt(r['cycles_from_events'], 'd')}/"
                   f"{fmt(r['cycles_from_peaks'], 'd')}/"
                   f"{fmt(r['cycles_expected'], 'd')}")
            head += (f" {str(r['cued_finger'] or '-'):>8} "
                     f"{fmt(r['cued_span_fraction']):>6} {cyc:>16} "
                     f"{fmt(r['lag_ms_median'], '.0f'):>6}")
            out.append(head)
            if r.get("other_spans"):
                out.append("      other fingers' spans (reported, not "
                           "failed): " + "  ".join(
                               f"{f} {fmt(v)}"
                               for f, v in r["other_spans"].items()))
        elif s.set == SEQUENCES:
            cam = (f"{fmt(r['steps_pass_camera'], 'd')}/"
                   f"{fmt(r['steps_judged_camera'], 'd')}")
            cpl = (f"{fmt(r.get('coupling_glove'), 'd')}/"
                   f"{fmt(r.get('coupling_camera'), 'd')}")
            head += (f" {fmt(r['steps_total'], 'd'):>5} "
                     f"{fmt(r['steps_pass_glove'], 'd'):>8} {cam:>10} "
                     f"{cpl:>12}")
            out.append(head)
        else:
            curls = r.get("curls") or {}
            head += (f" {fmt(r['frames_camera'], 'd'):>6} "
                     f"{fmt(r['tracked_fraction']):>7} "
                     f"{fmt(r['reacquisitions'], 'd'):>5} "
                     f"{fmt(r['grab_strength']):>5} "
                     f"{fmt(r['pinch_strength']):>5}  "
                     + " ".join(fmt(curls.get(f)) for f in FINGERS))
            out.append(head)
        if not r["accepted"] and r["reason"]:
            out.append(f"      rejected by the recorder: {r['reason']}")
        for f in r["failures"]:
            out.append(f"      fails: {f}")
        for n in r["notes"]:
            out.append(f"      note: {n}")
    return out


def rules_text(check: SessionCheck) -> List[str]:
    s = check.session
    out = ["Rules in force (fixed numbers are acquisition gates; the "
           "continuous values above are the measurement)"]
    if s.set in (FLEXION, SEQUENCES):
        out += [
            "  fraction        (open - curl) / (open - fist), clipped to "
            f"[{FRACTION_CLIP[0]}, {FRACTION_CLIP[1]}]; curl is tip-to-wrist "
            "over palm length",
            "  camera frames   cam_hand.fusion.camera_trusted, default gates "
            f"(held >= {DEFAULT_GATES.min_visible_time_us / 1000:.0f} ms, "
            "stable hand id, palm within "
            f"{DEFAULT_GATES.field_half_angle_deg:g} deg of the module axis, "
            f"view angle < {DEFAULT_GATES.view_gate_deg:g} deg)",
            "  paired          nearest camera frame within "
            f"{PAIR_MAX_DT * 1000:.0f} ms on the take's pairing clock",
        ]
    if s.set == FLEXION:
        out += [
            "  cued span       5th to 95th percentile of the cued finger's "
            f"glove fraction over the take, at least {MIN_CUED_SPAN:.2f}",
            "  other fingers   the same span, reported only",
            "  cycles          bend cues against glove curl peaks (a rise "
            f"above {PEAK_HIGH_SHARE:.0%} of the take's range after being "
            f"below {PEAK_LOW_SHARE:.0%}); they must match, and must equal "
            "the protocol's cycles when the protocol file is readable",
            f"  camera followed camera span >= {MIN_CAMERA_RANGE:.2f} curl "
            "(leap_hand.diagnostics.camera_range), per take and per cycle; "
            "only then are the transfer curve, hysteresis and lag reported",
        ]
    if s.set == SEQUENCES:
        hold, window = check_window_s(s)
        scale = ("the warm-up range" if check.bands == FIXED else
                 "the session's own open-hand to full-fist range")
        out += [
            f"  bands           {check.bands}, fractions of {scale}: a "
            f"flexed finger must read above {FLEXED_ABOVE:.2f}; a straight "
            f"finger fails only at {FLEXED_ABOVE:.2f} or more",
            f"  coupling        a straight finger from {STRAIGHT_BELOW:.2f} "
            f"to {FLEXED_ABOVE:.2f}: reported per step and finger "
            "(coupling g/c = glove/camera count), not failed",
            f"  window          median over the last {window:g} s of each "
            f"{hold:g} s hold",
            "  verdict         every step must pass on the glove; the camera "
            "is reported beside it and never fails a take",
        ]
    if s.set == GRASPS:
        out += [
            f"  gate            tracked >= {GRASP_MIN_TRACKED:.2f} and no "
            "re-acquisition inside the static interval",
            "  acceptance      the operator's eye on the still and the "
            "summary frame (decided in the recorder)",
            f"  second look     a value more than {OUTLIER_FACTOR:g} times "
            "the session's typical distance from its grasp's median",
        ]
    return out


def render_text(check: SessionCheck) -> str:
    """check.txt: header, the table, per item diagnostics, the rules."""
    s = check.session
    n_acc = sum(1 for r in check.rows if r["accepted"])
    lines = ["=" * 78,
             f"Protocol check: {s.set}, {s.hand} hand, session {s.name}",
             f"  folder    {s.path}"]
    lines += ["  " + h for h in check.header]
    lines.append(f"  takes     {n_acc} accepted, {len(check.rows) - n_acc} "
                 f"rejected; {len(check.failed_accepted)} accepted take(s) "
                 "fail a rule")
    lines.append("")
    lines += _table(check)
    if any(check.details.values()):
        lines.append("")
        if s.set == FLEXION:
            lines.append("Per item: transfer curve, hysteresis and lag where "
                         "the camera followed (span >= "
                         f"{MIN_CAMERA_RANGE:.2f})")
        else:
            lines.append("Per item: step by step, glove and camera")
        lines.append("-" * 78)
        order: List[str] = []
        for r in check.rows:
            if r["item"] not in order:
                order.append(r["item"])
        for item in order:
            for r in check.rows:
                block = check.details.get(r["name"])
                if r["item"] == item and block:
                    tag = "" if r["accepted"] else "  [rejected attempt]"
                    lines.append(block[0] + tag)
                    lines.extend(block[1:])
                    lines.append("")
    lines.append("")
    lines += rules_text(check)
    lines.append("")
    if check.failed_accepted:
        lines.append("REDO: " + ", ".join(r["name"]
                                          for r in check.failed_accepted))
    else:
        lines.append("Every accepted take passes.")
    return "\n".join(lines) + "\n"


def write_check(check: SessionCheck, out_dir=None) -> Tuple[Path, Path, str]:
    """Write check.csv and check.txt; returns their paths and the text."""
    out = Path(out_dir) if out_dir is not None else check.session.path
    out.mkdir(parents=True, exist_ok=True)
    csv_path, txt_path = out / "check.csv", out / "check.txt"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(csv_rows(check))
    text = render_text(check)
    txt_path.write_text(text, encoding="utf-8")
    return csv_path, txt_path, text
