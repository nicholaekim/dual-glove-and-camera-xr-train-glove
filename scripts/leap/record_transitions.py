"""Record a cued change from one grasp into another (camera only, bare hand).

The static grasp takes (`scripts/leap/record_poses.py --protocol`) record a
pose that is already formed, so they cannot say how long a hand takes to
change pose. Their forming file times one change, the open hand into the
grasp. This records the other kind, grasp A into grasp B, as a recording of
its own with everything a timing needs in one file:

  STARTING GRASP   the operator forms grasp A over the camera and holds it.
                   The hand nearest the module's axis is watched until it
                   has been still (mean joint speed over the camera under
                   20 mm/s) for `--start-hold` seconds; not within 30 s,
                   the attempt is rejected.
  THE WAIT         a further 0.5 to 1.5 s, drawn at random so the cue cannot
                   be counted in, the hand still watched: if it moves, back
                   to STARTING GRASP. No sound until the cue.
  THE CUE          a high beep: change to grasp B now. Its time is taken
                   just before the beep starts.
  THE CHANGE       the hand moves into grasp B.
  THE HOLD         once it has settled, grasp B is held `--hold` seconds,
                   and the recording ends (or `--timeout` seconds after the
                   cue, settled or not).

The file holds every frame, of every hand in view, from `--pre` seconds
before the cue to the end of the hold, in the line format of every other
camera recording (`leap_hand.recorder.LeapRecorder`), `pose` =
"<from>_to_<to>". Frames from before the cue are kept in memory until the
cue and written then, so their `wall_time` is the time of that write; their
`capture_time`, when the camera saw them, is what every timing reads.

Timing is `leap_hand.transition.transition_timing` on the hand that was
held still at the cue, followed by its tracker id: the start of the change
of pose, its settle, and transition_duration_s = settle - start, with the
reaction time beside it. It is read off the pose speed, the joints' speed
with the whole hand's movement taken out, so a hand carried across the
camera while it changes is timed on its change alone. A take is kept when
its timing is valid or questionable (the pose moved before its main
movement, or again after settling) and retried otherwise, up to `--retries`
times; every attempt stays on disk with what was measured and why it was
not kept.

Two speeds, two pairs of limits. Before the cue the hand as a whole has to
be still over the camera: that watch uses the camera-frame speed and
`transition.SETTLE_SPEED_MM_S` / `MOVE_SPEED_MM_S` (20 and 50 mm/s). The
timing uses the pose speed and the thresholds below (25 and 10 mm/s unless
given).

  recordings/protocol/transitions/<YYYYMMDD_HHMMSS>_<hand>/
    session.json                 rewritten after every attempt
    pose_transitions.csv         one row per attempt, the hand-in's columns
    leap/<from>_to_<to>_<hand>_take<N>_<stamp>.jsonl
    meta/<same name>.json        the cue, the hand followed, the timing

  .venv\\Scripts\\python.exe scripts\\leap\\record_transitions.py --hand left --pairs "p1_cylindrical>p1_spherical,p1_spherical>p1_tip" --takes 3 --operator "N Kim"
  .venv\\Scripts\\python.exe scripts\\leap\\record_transitions.py --mock --hand left --pairs "p1_hook>p1_tip" --takes 1

A pair is "<from>><to>" with grasp ids of the protocol file (or "open_hand"
on either side). The rebuild (`scripts/rebuild_grasp_set.py`, the
rebuild_professor_grasp_set.cmd launcher on the Desktop) finds every real
session in recordings/protocol/transitions by itself and puts its kept takes
into the professor's transition table; a --mock session only when it is
named with --transitions and --allow-mock (`rebuild_hint`, the last line a
session prints). The speed and dwell thresholds are options here and are
written into every meta; the rebuild times the recordings again with its own.

A --mock session goes to recordings/protocol_mock/transitions/ and acts the
change out on a synthetic hand (`TransitionActor`): the starting curl held,
then, a reaction time after the cue, an eased move to the target curl. It
has been run on synthetic hands only: as of 2026-10-05 no take has been
recorded with the camera.
"""
import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from leap_hand import grasp_handin as gh
from leap_hand import transition as tr
from leap_hand.recorder import LeapRecorder
from leap_hand.static_interval import LOSS_GAP_S
from leap_hand.stream import LeapUnavailable, open_stream

REPO = Path(__file__).resolve().parents[2]
SET_NAME = gh.TRANSITION_SET
OUT = Path("recordings") / "protocol" / SET_NAME
MOCK_OUT = Path("recordings") / "protocol_mock" / SET_NAME   # never beside real sessions
OPEN_HAND = gh.OPEN_HAND

START_HOLD_S = 1.5             # the starting grasp, still this long
CUE_WAIT_S = (0.5, 1.5)        # then this much longer, drawn at random
PRE_S = 1.0                    # kept from before the cue
HOLD_S = 1.5                   # the target grasp, held after it settled
TIMEOUT_S = 8.0                # after the cue, settled or not
START_TIMEOUT_S = 30.0         # no still starting grasp by then: rejected
EARLY_MOVES_MAX = 5            # moved during the wait this often: rejected
RETRIES = 2
RECENT_S = 0.2                 # a hand seen within this is in view
BEEP_CUE = (1400, 200)         # the high beep
BEEP_KEPT = (880, 120)
BEEP_REDO = (300, 450)


def beep(freq: int = 880, ms: int = 180) -> None:
    """Audible cue; falls back to the terminal bell off Windows."""
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


def drain_all(source) -> list:
    """Every hand the stream has queued, oldest first.

    `drain` hands out a bounded batch. After the loop has been held up (the
    cue's beep blocks for its 0.2 s, a loaded machine for longer) the queue
    holds more than one batch, and reading only one would leave the newest
    frames behind: the watched hand would look as if it had left.
    """
    out = []
    while True:
        batch = source.drain(256)
        if not batch:
            return out
        out.extend(batch)


def row_of(lh, wall: float) -> dict:
    """One tracked hand as the line the analysis reads: the keys of a
    recorded line that `leap_hand.transition` and `follow_hand` use."""
    return {"wall_time": float(wall),
            "capture_time": (None if lh.capture_time is None
                             else float(lh.capture_time)),
            "hand_side": lh.hand_side, "hand_id": int(lh.hand_id),
            "frame_id": int(lh.frame_id), "status": 1,
            "framerate": float(lh.framerate),
            "palm_abs": [float(v) for v in lh.palm_pos],
            "abs26": [[float(v) for v in p] for p in lh.abs26]}


class StillWatch:
    """The hands of the last few seconds, and whether one is being held
    still. Every hand is kept (the file gets them all); the one watched is
    the hand nearest the module's vertical axis."""

    def __init__(self, keep_s: float, still_speed_mm_s: float):
        self.keep_s = float(keep_s)
        self.still_speed = float(still_speed_mm_s)
        self.items: deque = deque()          # (frame time, LeapHand, row)

    def add(self, lh, now: float) -> None:
        row = row_of(lh, now)
        self.items.append((tr.frame_time(row), lh, row))
        while self.items and self.items[0][0] < now - self.keep_s:
            self.items.popleft()

    def rows(self, hand_id: Optional[int] = None,
             since: Optional[float] = None) -> List[dict]:
        return [r for t, _lh, r in self.items
                if (hand_id is None or r["hand_id"] == hand_id)
                and (since is None or t >= since)]

    def candidate(self, now: float) -> Optional[int]:
        """The id of the hand nearest the module's axis among those seen
        within `RECENT_S`, or None when no hand is in view."""
        latest: Dict[int, dict] = {}
        for t, _lh, r in self.items:
            if t >= now - RECENT_S:
                latest[r["hand_id"]] = r
        if not latest:
            return None
        return min(latest, key=lambda h: float(np.hypot(
            latest[h]["palm_abs"][0], latest[h]["palm_abs"][2])))

    def others(self, hand_id: int, now: float) -> List[int]:
        return sorted({r["hand_id"] for t, _lh, r in self.items
                       if t >= now - 0.5 and r["hand_id"] != hand_id})

    def still_for(self, hand_id: int, look_back: float) -> float:
        """How long the hand has been still, up to its newest frame: the
        length of the run of frames, counted back from the newest, that are
        all slower than the still speed with no hole between them. Only the
        last `look_back` seconds are looked at."""
        if not self.items:
            return 0.0
        rows = self.rows(hand_id, self.items[-1][0] - look_back)
        if len(rows) < 3:
            return 0.0
        t, speed = tr.joint_speeds(rows)
        k = len(t) - 1
        if not speed[k] < self.still_speed:          # NaN is not still
            return 0.0
        while k > 0 and (speed[k - 1] < self.still_speed
                         and t[k] - t[k - 1] <= LOSS_GAP_S):
            k -= 1
        return float(t[-1] - t[k])

    def newest_speed(self, hand_id: int, window: float = 0.15) -> float:
        """The fastest of the hand's frames of the last `window` seconds,
        mm/s; 0 with too few frames to tell."""
        if not self.items:
            return 0.0
        newest = self.items[-1][0]
        rows = self.rows(hand_id, newest - window - 2 * tr.SPEED_STEP_S)
        if len(rows) < 3:
            return 0.0
        t, speed = tr.joint_speeds(rows)
        recent = speed[t >= newest - window]
        recent = recent[~np.isnan(recent)]
        return float(recent.max()) if len(recent) else 0.0


class TransitionActor:
    """A `MockLeapStream` script that acts a cued change of pose out.

    The hand holds `from_curl` until the recorder says "cue"; `react_s`
    later it moves to `to_curl` over `move_s`, eased in and out, and holds
    it. `move_before_cue` moves it that many seconds BEFORE the cue instead
    (it starts on "wait"), the way a hand that anticipates would.
    """

    def __init__(self, hz: float = 90.0, from_curl: float = 0.15,
                 to_curl: float = 0.85, react_s: float = 0.25,
                 move_s: float = 0.5, move_before_cue: bool = False):
        self.hz = float(hz)
        self.from_curl, self.to_curl = float(from_curl), float(to_curl)
        self.react_s, self.move_s = float(react_s), float(move_s)
        self.move_before_cue = bool(move_before_cue)
        self.go: Optional[int] = None        # the frame the move starts at
        self.phases: List[str] = []

    def phase(self, name: Optional[str], i: int) -> None:
        self.phases.append(str(name))
        if name == "start":
            self.go = None
        elif name == ("wait" if self.move_before_cue else "cue"):
            self.go = int(i) + int(round(
                (0.0 if self.move_before_cue else self.react_s) * self.hz))

    def __call__(self, i: int) -> dict:
        if self.go is None or i < self.go:
            return {"curl": self.from_curl}
        w = min(1.0, (i - self.go) / self.hz / self.move_s) \
            if self.move_s > 0 else 1.0
        w = w * w * (3.0 - 2.0 * w)
        return {"curl": self.from_curl + (self.to_curl - self.from_curl) * w}


def parse_pairs(text: str, known: List[str]) -> List[Tuple[str, str]]:
    """"a>b,c>d" -> [("a", "b"), ("c", "d")], every id a grasp of the
    protocol or "open_hand". Raises ValueError naming what is wrong."""
    pairs = []
    for part in [x.strip() for x in str(text).split(",") if x.strip()]:
        if part.count(">") != 1:
            raise ValueError(f"\"{part}\" is not <from>><to>")
        frm, to = (x.strip() for x in part.split(">"))
        for name in (frm, to):
            if name != OPEN_HAND and name not in known:
                raise ValueError(f"\"{name}\" is not a grasp of the protocol "
                                 f"(nor \"{OPEN_HAND}\")")
        if frm == to:
            raise ValueError(f"\"{part}\" changes nothing")
        pairs.append((frm, to))
    if not pairs:
        raise ValueError("no pair given")
    return pairs


def thresholds_of(args) -> dict:
    """The timing's thresholds as `transition_timing` takes them: the
    options given, and nothing for the ones left at their defaults."""
    names = (("start_speed", "start_speed_mm_s"),
             ("start_sustain", "start_sustain_s"),
             ("settle_speed", "settle_speed_mm_s"),
             ("settle_window", "settle_window_s"),
             ("cue_still", "cue_still_s"))
    return {key: getattr(args, opt) for opt, key in names
            if getattr(args, opt, None) is not None}


def timing_dict(g: tr.Timing) -> dict:
    return {k: (None if v is None else round(float(v), 6)
                if isinstance(v, float) else v)
            for k, v in vars(g).items()}


def record_take(source, path: Path, hand: str, label: str, take: int,
                start_hold: float = START_HOLD_S,
                cue_wait: Tuple[float, float] = CUE_WAIT_S,
                pre: float = PRE_S, hold: float = HOLD_S,
                timeout: float = TIMEOUT_S,
                start_timeout: float = START_TIMEOUT_S,
                thresholds: Optional[dict] = None,
                beeper: Callable[..., None] = beep,
                say: Callable[[str], None] = print,
                rng: Optional[random.Random] = None) -> dict:
    """One attempt: wait for the starting grasp held still, cue, record
    until the target is held. Returns the attempt's meta (without names);
    `path` is written only once the cue has sounded."""
    thresholds = dict(thresholds or {})
    rng = rng or random.Random()
    # the hand as a whole, over the camera, before the cue: the camera-frame
    # speed and its own limits, not the pose speed's thresholds
    still_speed, move_speed = tr.SETTLE_SPEED_MM_S, tr.MOVE_SPEED_MM_S
    watch = StillWatch(pre + start_hold + cue_wait[1] + 1.0, still_speed)
    act = getattr(source, "act", lambda _phase: None)
    out = {"accepted": False, "reason": "", "cue_wall_time": None,
           "start_hand_id": None, "other_hand_ids": [], "start_palm": None,
           "frames": 0, "early_moves": 0, "cue_wait_s": None}

    def pump() -> float:
        now = time.time()
        for _side, lh in drain_all(source):
            watch.add(lh, now)
        return now

    t_begin = time.time()
    hid = None
    while True:
        # STARTING GRASP: held still for start_hold
        act("start")
        say("      hold the STARTING grasp still")
        while True:
            now = pump()
            hid = watch.candidate(now)
            if hid is not None and watch.still_for(
                    hid, start_hold + 0.5) >= start_hold:
                break
            if now - t_begin > start_timeout:
                out["reason"] = (f"the starting grasp was not held still "
                                 f"within {start_timeout:g} s")
                return out
            time.sleep(0.02)
        # THE WAIT: still watched, no sound
        act("wait")
        wait = rng.uniform(*cue_wait)
        say("      READY - change on the HIGH beep")
        t_cue = time.time() + wait
        moved = False
        while time.time() < t_cue:
            now = pump()
            if (watch.candidate(now) != hid
                    or watch.newest_speed(hid) >= move_speed):
                moved = True
                break
            time.sleep(0.02)
        if not moved:
            out["cue_wait_s"] = round(wait, 3)
            break
        out["early_moves"] += 1
        say("      MOVED BEFORE THE BEEP - hold the starting grasp again")
        if out["early_moves"] >= EARLY_MOVES_MAX:
            out["reason"] = (f"the hand moved before the cue "
                             f"{out['early_moves']} times")
            return out

    # THE CUE
    cue = time.time()
    act("cue")
    beeper(*BEEP_CUE)
    say("      CHANGE NOW")
    now = time.time()
    out["cue_wall_time"] = cue
    out["start_hand_id"] = int(hid)
    out["other_hand_ids"] = watch.others(hid, cue)
    before = [r for r in watch.rows(hid) if tr.frame_time(r) <= cue]
    out["start_palm"] = (before or watch.rows(hid))[-1]["palm_abs"]
    follow = (int(hid), out["other_hand_ids"], out["start_palm"])

    rec = LeapRecorder(hz=None, pose=label, take=take)
    rec.start(path)
    rows: List[dict] = []
    try:
        for t, lh, row in list(watch.items):
            if t >= cue - pre:
                rec.record(lh)
                rows.append(row)
        last_check = 0.0
        while True:
            now = time.time()
            for _side, lh in drain_all(source):
                rec.record(lh)
                rows.append(row_of(lh, now))
            if now - last_check >= 0.1:
                last_check = now
                g = tr.transition_timing(tr.measured_rows(rows, hand, follow),
                                         cue, **thresholds)
                if (g.movement_start_time_s is not None
                        and g.settled_time_s is not None
                        and now >= g.first_frame_time + g.settled_time_s
                        + hold):
                    break
            if now >= cue + timeout:
                break
            time.sleep(0.02)
    finally:
        rec.stop()
    act("done")
    g = tr.transition_timing(tr.measured_rows(rows, hand, follow), cue,
                             **thresholds)
    out["frames"] = len(rows)
    out["timing"] = timing_dict(g)
    out["accepted"] = g.validity in (tr.VALID, tr.QUESTIONABLE)
    out["reason"] = "" if out["accepted"] else g.notes
    out["_timing"] = g
    return out


def write_json(path: Path, data: dict) -> None:
    """Through a temporary file, retried briefly: OneDrive can hold a file
    for a moment."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def csv_row(participant: str, hand: str, frm: str, to: str, take: int,
            recording: str, g: tr.Timing, session: str) -> dict:
    """One attempt as a row of the hand-in's transition table."""
    move = gh.Move(from_grasp=frm, to_grasp=to, take=take, hand=hand,
                   source=Path(recording), dest=recording, timing=g,
                   session=session)
    shell = gh.GraspSet(session=None, hand=hand, participant=participant,
                        items=[], target=0, moves=[move])
    return gh.transition_rows(shell)[0]


def make_source(mock: bool, hand: str, mock_move_early: bool = False):
    if not mock:
        return open_stream(mock=False)
    from leap_hand.mock import MockLeapStream
    # noise_mm: the mock shakes every joint on its own, which is pose
    # movement; 0.05 mm gives the 1 mm/s of pose speed a held real hand
    # has (the default 0.35 mm would be 8 mm/s, and never settle)
    source = MockLeapStream(hz=90.0, pose="open_palm", sides=(hand,),
                            noise_mm=0.05,
                            dropout_every=0, reacquire_every=0,
                            script=TransitionActor(
                                move_before_cue=mock_move_early))
    source.start()
    return source


def run(args, source=None, beeper: Callable[..., None] = beep,
        say: Callable[[str], None] = print) -> int:
    proto_path = Path(args.protocol)
    if not proto_path.is_absolute():
        proto_path = REPO / proto_path
    try:
        protocol = json.loads(proto_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        say(f"cannot read the protocol file {proto_path}: {e}")
        return 2
    labels = {str(it["id"]): str(it.get("label") or it["id"])
              for it in protocol.get("items", []) if it.get("id")}
    labels[OPEN_HAND] = "Open hand, palm to the camera"
    try:
        pairs = parse_pairs(args.pairs, list(labels))
    except ValueError as e:
        say(f"--pairs: {e}")
        return 2
    thresholds = thresholds_of(args)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.out) if args.out else REPO / (MOCK_OUT if args.mock
                                                   else OUT)
    session = root / f"{stamp}_{args.hand}"
    (session / "leap").mkdir(parents=True, exist_ok=True)
    (session / "meta").mkdir(parents=True, exist_ok=True)
    own_source = source is None
    if own_source:
        try:
            source = make_source(args.mock, args.hand, args.mock_move_early)
        except LeapUnavailable as e:
            say(f"camera not available: {e}")
            return 2
    record = {
        "set": SET_NAME, "hand": args.hand, "operator": args.operator,
        "participant": args.participant, "camera": "leap",
        "mock": bool(args.mock), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ended": None, "protocol_file": str(args.protocol),
        "protocol_name": protocol.get("name"),
        "protocol_version": protocol.get("version"),
        "pairs": [list(p) for p in pairs], "takes_per_pair": args.takes,
        "start_hold_s": args.start_hold, "cue_wait_s": list(args.cue_wait),
        "pre_s": args.pre, "hold_s": args.hold, "timeout_s": args.timeout,
        "retries": args.retries, "thresholds": thresholds,
        "movement_start_method": tr.start_method(
            thresholds.get("start_speed_mm_s", tr.POSE_MOVE_SPEED_MM_S),
            thresholds.get("start_sustain_s", tr.START_SUSTAIN_S)),
        "settled_method": tr.settled_method(
            thresholds.get("settle_speed_mm_s", tr.POSE_SETTLE_SPEED_MM_S),
            thresholds.get("settle_window_s", tr.SETTLE_WINDOW_S)),
        "takes": []}
    rows: List[dict] = []
    rng = random.Random(args.seed)
    say(f"Transitions, {args.hand} hand, camera only, bare hand: "
        f"{len(pairs)} pair(s) x {args.takes} take(s) -> {session}")
    kept = 0
    try:
        for frm, to in pairs:
            say(f"\n{labels[frm]}  ->  {labels[to]}")
            for n in range(1, args.takes + 1):
                for attempt in range(1, args.retries + 2):
                    name = (f"{frm}_to_{to}_{args.hand}_take{n}_"
                            f"{time.strftime('%Y%m%d_%H%M%S')}")
                    leap = session / "leap" / f"{name}.jsonl"
                    say(f"  take {n}/{args.takes}"
                        + (f", attempt {attempt}" if attempt > 1 else "")
                        + f": form \"{labels[frm]}\"")
                    got = record_take(
                        source, leap, args.hand, f"{frm}_to_{to}", n,
                        start_hold=args.start_hold,
                        cue_wait=tuple(args.cue_wait), pre=args.pre,
                        hold=args.hold, timeout=args.timeout,
                        start_timeout=args.start_timeout,
                        thresholds=thresholds, beeper=beeper, say=say,
                        rng=rng)
                    g = got.pop("_timing", None) or tr.Timing(
                        notes=got["reason"])
                    files = {"leap": (f"leap/{name}.jsonl"
                                      if leap.is_file() else None),
                             "meta": f"meta/{name}.json"}
                    meta = {"set": SET_NAME, "name": name, "from_grasp": frm,
                            "to_grasp": to, "take": n, "attempt": attempt,
                            "hand": args.hand, "session": session.name,
                            "mock": bool(args.mock), "pre_s": args.pre,
                            "hold_s": args.hold, "timeout_s": args.timeout,
                            "start_hold_s": args.start_hold,
                            "thresholds": thresholds, "files": files, **got}
                    write_json(session / files["meta"], meta)
                    record["takes"].append({
                        "name": name, "from_grasp": frm, "to_grasp": to,
                        "take": n, "attempt": attempt,
                        "accepted": got["accepted"], "reason": got["reason"],
                        "files": files})
                    rows.append(csv_row(args.participant, args.hand, frm, to,
                                        n, files["leap"] or "", g,
                                        session.name))
                    gh.write_csv(session / gh.TRANSITIONS_FILE,
                                 gh.TRANSITION_COLUMNS, rows)
                    write_json(session / "session.json", record)
                    if got["accepted"]:
                        kept += 1
                        beeper(*BEEP_KEPT)
                        say(f"      kept ({g.validity}): started "
                            f"{g.reaction_time_s:.2f} s after the cue, "
                            f"{g.transition_duration_s:.2f} s to settle"
                            + (f"; {g.notes}" if g.notes else ""))
                        break
                    beeper(*BEEP_REDO)
                    say(f"      not kept: {got['reason']}")
    except KeyboardInterrupt:
        say("\nstopped; what was recorded is kept")
    finally:
        record["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(session / "session.json", record)
        if own_source:
            try:
                source.stop()
            except Exception:
                pass
    want = len(pairs) * args.takes
    say(f"\nkept {kept} of {want} take(s) in {session}")
    say(rebuild_hint(session, args.mock))
    return 0 if kept == want else 1


def rebuild_hint(session: Path, mock: bool) -> str:
    """The last line of a session: how its takes reach the professor's
    table. A real session in recordings/protocol/transitions is found by
    the rebuild by itself (its --transitions-dir default); a mock one only
    when named with --transitions and --allow-mock; a real one written
    elsewhere with --out only when named."""
    if mock:
        return ("this is a MOCK session (synthetic hand), never for the "
                "hand-in; to rehearse the rebuild with it, pass it with "
                "--transitions and --allow-mock: scripts\\rebuild_grasp_set.py "
                f"... --transitions \"{session}\" --allow-mock")
    if Path(session).resolve().parent == (REPO / OUT).resolve():
        return ("to put them in the professor's table: run the rebuild "
                "launcher (rebuild_professor_grasp_set.cmd on the Desktop); "
                "it picks this session up by itself from "
                "recordings\\protocol\\transitions")
    return ("to put them in the professor's table: scripts\\rebuild_grasp_set.py "
            f"... --transitions \"{session}\" (the rebuild launcher picks up "
            "by itself only the sessions in recordings\\protocol\\transitions)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Record cued changes from one grasp into another with "
                    "the Ultraleap camera (camera only, bare hand).")
    p.add_argument("--pairs", required=True,
                   help="\"<from>><to>,<from>><to>\": grasp ids of the "
                        "protocol, or open_hand")
    p.add_argument("--hand", choices=("left", "right"), required=True)
    p.add_argument("--takes", type=int, default=3, help="kept takes per pair")
    p.add_argument("--protocol", default="protocols/grasps.json",
                   help="the protocol file whose grasp ids the pairs name")
    p.add_argument("--operator", default=None)
    p.add_argument("--participant", default="P01")
    p.add_argument("--out", type=Path, default=None,
                   help="where the session folder goes (default: "
                        "recordings/protocol/transitions)")
    p.add_argument("--start-hold", type=float, default=START_HOLD_S,
                   help="seconds the starting grasp must be still")
    p.add_argument("--cue-wait", type=float, nargs=2, default=list(CUE_WAIT_S),
                   metavar=("MIN", "MAX"),
                   help="the further wait before the cue, drawn at random")
    p.add_argument("--pre", type=float, default=PRE_S,
                   help="seconds kept from before the cue")
    p.add_argument("--hold", type=float, default=HOLD_S,
                   help="seconds the target grasp is held after it settled")
    p.add_argument("--timeout", type=float, default=TIMEOUT_S,
                   help="the recording ends this long after the cue at most")
    p.add_argument("--start-timeout", type=float, default=START_TIMEOUT_S)
    p.add_argument("--retries", type=int, default=RETRIES)
    p.add_argument("--seed", type=int, default=None,
                   help="seed of the random wait (default: none)")
    p.add_argument("--start-speed", type=float, default=None,
                   help="mm/s of pose speed a change has started at "
                        "(default 25)")
    p.add_argument("--start-sustain", type=float, default=None,
                   help="seconds it must stay that fast (default 0.1)")
    p.add_argument("--settle-speed", type=float, default=None,
                   help="mm/s of pose speed a pose is settled below "
                        "(default 10)")
    p.add_argument("--settle-window", type=float, default=None,
                   help="seconds it must stay below it (default 0.3)")
    p.add_argument("--cue-still", type=float, default=None,
                   help="seconds around the cue the pose must be still for "
                        "(default 0.1)")
    p.add_argument("--mock", action="store_true",
                   help="a synthetic hand; the session goes to "
                        "recordings/protocol_mock/transitions")
    p.add_argument("--mock-move-early", action="store_true",
                   help="mock: the hand moves before the cue")
    p.add_argument("--quiet", action="store_true", help="no beeps")
    args = p.parse_args(argv)
    return run(args, beeper=(lambda *_a: None) if args.quiet else beep)


if __name__ == "__main__":
    raise SystemExit(main())
