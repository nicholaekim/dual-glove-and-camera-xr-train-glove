"""Guided pose-recording session with the Ultraleap camera - hands-free.

The camera twin of `scripts/glove/record_poses.py`: same pose list, same beep
protocol, same file naming, same labels. Announces the pose, counts down with
beeps, records a few seconds, moves on. Nothing to type once it starts, which
is the point - your hands are in front of the camera, and on Path A they are
also inside the glove.

Takes land in `recordings/leap/poses/` as labeled JSONL, named like
`fist_both_take2_20260916_154212.jsonl` - the glove's scheme, so the two
datasets line up folder for folder and every existing tool reads both:

  python scripts/glove/playback.py recordings/leap/poses/<file>.jsonl
  python scripts/glove/export_keypoints21.py recordings/leap/poses
  python scripts/glove/export_prof_format.py recordings/leap/poses

With --raw, each take also writes `<same name>.lmt`, LeapC's own recording of
the raw tracking stream. It is the only artefact that can be re-processed if
the joint mapping or the units later turn out to be wrong, which is worth the
disk space on the first real session.

Protocol (plan section 6): module flat on the table, lenses up, hand 20 to 50
cm above it, palm roughly facing the camera for the spread and thumb poses,
no sunlight and no other IR sources.

  python scripts/leap/record_poses.py                        # 6 poses x 3 takes x 5 s
  python scripts/leap/record_poses.py --poses pinch,fist --takes 2 --duration 4
  python scripts/leap/record_poses.py --hz 0                 # keep every frame (~90/s)
  python scripts/leap/record_poses.py --raw                  # also write .lmt
  python scripts/leap/record_poses.py --mock --takes 1 --duration 2 --prep 1

Ctrl+C at any point keeps the takes recorded so far.

The professor's grasp set (--protocol, Set A)
---------------------------------------------
`--protocol protocols/grasps.json --hand left` runs Set A of
docs/grasp_and_flexion_protocol_plan.md and writes the session folder that
docs/protocol_formats.md specifies (sections 1, 3, 5 and 7), so the checker
and the packager read it without knowing who wrote it:

    recordings/protocol/grasps/<YYYYMMDD_HHMMSS>_<hand>/
      session.json                      rewritten after every take
      leap/<take>.jsonl                 every frame, both hands if two were seen
      stills/<take>.png                 one hand-cropped IR still per take
      keypoints/<take>_keypoints.txt    the take's summary frame, his format
      meta/<take>.json                  the numbers and the decision
      rejected/...                      attempts not kept, same layout, plus
                                        rejected/<take>.reason.txt

What changes from the plain session, and why:

  The grasp list is data. Items, takes, seconds and preparation come from
  the protocol file, so a grasp named in the papers is a new JSON entry, not
  a code change; `--items`, `--takes`, `--duration`, `--prep` subset or
  override them for a redo.

  Every frame is kept. The professor wants the FINAL configuration, and the
  take's stillest two seconds (`leap_hand.static_interval`) are only
  findable in the full-rate stream. The summary frame is the medoid of that
  window, not of the whole take, so the reach into the pose cannot pull it.

  The operator's hand is `--hand`, and it is what the files are named after.
  The tracker's own left/right label is written on every line and counted
  in the meta, but it filters nothing: it called the left hand "right" in 20
  of 21 poses on 2026-09-23.

  The orientation is measured, not remembered. Plan D3 turns palm-down
  grasps only as far as the camera needs and asks for the rotation used, so
  each take's meta carries the summary frame's palm height and its angle to
  the lens; `--note` adds words to that.

  Two decisions per take. The acquisition gate is automatic: at least 90 %
  of the take's frames tracked and no re-acquisition inside the static
  interval, or the attempt moves to `rejected/` with its reason and is
  retried (`--retries`). The reason names the losses behind a rejection
  (`leap_hand.tracking_quality`): how many, and for the longest, where the
  hand was and the likely cause with its fix, for example "tracked 72
  percent: lost 3 times, longest 1.4 s with the hand at 49 cm (too high:
  keep the palm 25 to 35 cm above the module)". Every loss, kept take or
  not, is written to the meta as `gate.losses`. The gate is a minimum, not
  the acceptance: a take
  can be 95 % tracked with one fingertip wrong. So the recorder then shows
  the still with the numbers and the operator keeps it (Enter or space),
  redoes it (r) or ends the session (q). The decision and its time are
  written by the recorder, never by hand. `--auto-accept` skips the prompt.

  The picture to copy is on screen. The protocol's `images_dir` and each
  item's `image` name the panel cut from the paper's figure (made by
  `make_panels.py` beside the papers). Through the countdown and the take
  it fills a window titled COPY THIS, under the grasp's label, the take
  number and the countdown or HOLD STILL with the seconds left. The review
  then appears in that same window, the paper's picture beside the still,
  and the next countdown brings the next grasp's picture back. A missing
  picture is replaced by the grasp's shape text, and the console says so
  once. `--no-panel` turns the window off for headless runs.

  The take is coached, because the plain countdown failed. On 2026-10-01
  the operator formed each grasp during the 5 s GET READY exactly as the
  paper's photo shows it, which for the tip, fingertip, tripod and lateral
  grasps points the fingertips at the lens with the palm edge-on; the
  tracker lost the hand as the shape formed and could not pick up a hand
  that was already closed (45 of 69 attempts rejected, 15 with no hand at
  all), while kept takes of the same grasps tracked 99 to 100 %. So every
  attempt now goes, as `scripts/record_simultaneous.py` learned in
  September:
    OPEN HAND       "OPEN HAND, palm to the camera" until one hand has been
                    18 to 40 cm above the module, palm within 50 degrees of
                    the lens, for 0.5 s without a break; a beep. Not within
                    30 s: rejected, "no open hand acquired in 30 s".
    MAKE THE GRASP  a high beep, the paper's picture, the grasp's
                    orientation hint, 4 s (`--prep`) to close the hand while
                    it stays tracked. Gone for more than 0.3 s: a low beep,
                    "LOST YOU: open the hand, then close it slower", and back
                    to OPEN HAND; the third loss rejects the attempt, naming
                    the losses and where the hand was.
    HOLD STILL      the take, exactly as before: beep, file, gate, review.
  Each take's meta says how it went in `coaching` (`acquire_s`,
  `lost_while_forming`, and where every loss happened). An item's optional
  `orientation` text in the protocol file is its hint ("palm toward the
  camera" without one). `--no-coach` restores the plain countdown.

  One session per set (plan D9). A grasp still short of its takes at the
  end is recorded with `--resume <session folder>`, which adds the missing
  takes to that same folder and carries the take numbering on, instead of
  starting a second session that restarts at take 1. The end table prints
  the exact command.

  python scripts/leap/record_poses.py --protocol protocols/grasps.json --hand left
  python scripts/leap/record_poses.py --protocol protocols/grasps.json --hand left --items hook,lateral_key --takes 1
  python scripts/leap/record_poses.py --protocol protocols/grasps.json --hand left --resume recordings/protocol/grasps/<session>
  python scripts/leap/record_poses.py --mock --protocol protocols/grasps.json --hand left --auto-accept --takes 1 --duration 1 --prep 0.5 --no-open
  python scripts/leap/record_poses.py --mock --protocol protocols/grasps.json --hand left --items p1_tip --takes 1 --duration 2 --auto-accept --no-open --mock-lose-forming 1

A --mock session is written to recordings/protocol_mock/grasps/ instead, with
the same layout, so a rehearsal on synthetic hands can never be collected
with the real sessions. The mock acts the coached take out
(`leap_hand.mock.CoachedActor`): an open palm, then the hand closing, then
the grasp held; `--mock-lose-forming N` loses the hand while it closes on
the first N tries of every attempt, the way the camera did. The runbook is
docs/grasp_recording.md.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from leap_hand.protocol import (hand_view_angle_deg, palm_height_cm,
                                row_height_cm, row_view_angle_deg)
from leap_hand.recorder import LeapRecorder
from leap_hand.static_interval import (DEFAULT_STATIC_S, MIN_TRACKED_FRACTION,
                                       row_frame, summarise_take)
from leap_hand.stream import LeapUnavailable, open_stream
from leap_hand.tracking_quality import (LOSS_GAP_S, cause_label, causes_for,
                                        fix_for, state_from_leaphand,
                                        take_losses, take_reason)
from xr_hand.recorder import finalize_pose_name, hand_tag, pose_filename, slugify

# The same list the glove records, so the two datasets are comparable pose for
# pose. `three` is in POSE_HINTS but not the default set, exactly as in the
# glove script.
DEFAULT_POSES = ["open_palm", "fist", "index_point", "thumbs_up", "peace", "pinch"]

POSE_HINTS = {
    "open_palm": "all five fingers extended and spread",
    "fist": "all fingers curled into a tight fist",
    "index_point": "index finger extended, all others curled",
    "thumbs_up": "thumb extended up, all four fingers curled",
    "peace": "index + middle extended in a V, others curled",
    "pinch": "thumb and index fingertips touching, others relaxed",
    "three": "index + middle + ring extended, little and thumb curled",
}

STREAM_WAIT_TIMEOUT = 120.0   # s to wait for the first hands
STREAM_WAIT_HANDS = 10        # hands seen before the first take starts
MIN_VISIBLE_TIME_US = 300_000  # plan section 6: a hand counts after 0.3 s
MAX_DRAIN_ROUNDS = 8           # bound on the post-beep flush

# The plain session's defaults, applied when neither the command line nor a
# protocol file says otherwise.
LEGACY_TAKES = 3
LEGACY_DURATION = 5.0
LEGACY_PREP = 5.0
LEGACY_HZ = 5.0
LEGACY_OUT = Path("recordings") / "leap" / "poses"


def beep(freq: int = 880, ms: int = 180) -> None:
    """Audible cue; falls back to the terminal bell off Windows."""
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)


class Session:
    def __init__(self, source, hz, out_dir: Path, raw: bool = False):
        self.source = source
        self.hz = hz
        self.out_dir = out_dir
        self.raw = raw
        self.results = []
        self._ids = {}            # hand_side -> last hand_id, for re-acquisitions
        self._reacquired = 0
        self._skipped_young = 0

    # --- stream plumbing ------------------------------------------------
    def _consume(self, recorder=None, observe=None) -> int:
        """Drain pending hands: count everything, record if asked.

        Runs during the countdowns too (recorder=None) so the queue stays
        fresh and a stale hand from the previous pose never leaks into a take.
        `observe(hand)` sees every hand drained, young ones included (the
        coached take watches the hand this way). Returns how many hands were
        drained.
        """
        seen = 0
        for _side, lh in self.source.drain(64):
            seen += 1
            if observe is not None:
                observe(lh)
            previous = self._ids.get(lh.hand_side)
            if previous is not None and previous != lh.hand_id:
                self._reacquired += 1
            self._ids[lh.hand_side] = lh.hand_id
            # Gate on presence and settling time, never on confidence: LeapC
            # documents confidence as a constant 1.0 (plan section 6).
            if lh.visible_time_us < MIN_VISIBLE_TIME_US:
                self._skipped_young += 1
                continue
            if recorder is not None:
                recorder.record(lh)
        return seen

    def _discard_backlog(self) -> int:
        """Drop everything the tracker queued while the start beep blocked.

        `winsound.Beep` blocks for its whole duration and LeapC's polling
        thread keeps filling the queue behind it, so the first hands drained
        after a 250 ms beep are up to 250 ms old — and the recorder stamps
        each one with the time of the WRITE. Recording through that backdates
        the head of every take. Counters and hand ids still go through
        `_consume`, so the re-acquisition count stays honest.
        """
        dropped = 0
        for _ in range(MAX_DRAIN_ROUNDS):
            n = self._consume()
            dropped += n
            if not n:
                break
        return dropped

    def wait_for_stream(self) -> None:
        print(f"Waiting for hands (need {STREAM_WAIT_HANDS}, timeout "
              f"{int(STREAM_WAIT_TIMEOUT)} s)...")
        print("  Hold a hand 20 to 50 cm above the module, lenses up.")
        t0 = time.time()
        seen = 0
        sides = set()
        while time.time() - t0 < STREAM_WAIT_TIMEOUT:
            for side, _lh in self.source.drain(64):
                seen += 1
                sides.add(side)
            if seen >= STREAM_WAIT_HANDS:
                print(f"  OK - tracking: {', '.join(sorted(sides))}\n")
                return
            time.sleep(0.05)
        raise SystemExit(
            "No hands seen. Is the camera plugged into a direct USB port and "
            "listed in the Ultraleap Control Panel?\n"
            "  check with: python scripts/leap/check_setup.py"
        )

    # --- protocol -------------------------------------------------------
    def run_take(self, pose: str, take: int, n_takes: int,
                 pose_idx: int, n_poses: int, duration: float, prep: float) -> None:
        title = pose.replace("_", " ").upper()
        print(f"--- Pose {pose_idx}/{n_poses}: {title}  (take {take}/{n_takes}) ---")
        hint = POSE_HINTS.get(pose)
        if hint:
            print(f"    Hold: {hint}")
        # The mock can act out the pose, so a dry run looks like a real one.
        if hasattr(self.source, "set_pose"):
            self.source.set_pose(pose if pose in ("open_palm", "fist") else None)

        for s in range(int(round(prep)), 0, -1):
            if s <= 3:
                print(f"      {s}...")
                beep(660, 120)
            t_end = time.time() + 1.0
            while time.time() < t_end:
                self._consume()
                time.sleep(0.02)

        recorder = LeapRecorder(hz=self.hz, pose=pose, take=take)
        path = self.out_dir / pose_filename(pose, take)
        raw_path = path.with_suffix(".lmt")
        before = self._reacquired

        with self._raw_capture(raw_path):
            # Beep, throw away what queued behind the beep, and only then open
            # the file — see `_discard_backlog`.
            beep(1000, 250)
            self._discard_backlog()
            recorder.start(path)
            print(f"      REC {duration:g} s - hold it ", end="", flush=True)
            try:
                t_end = time.time() + duration
                next_dot = time.time() + 0.5
                while time.time() < t_end:
                    self._consume(recorder)
                    if time.time() >= next_dot:
                        print(".", end="", flush=True)
                        next_dot += 0.5
                    time.sleep(0.005)
            finally:
                # Runs on Ctrl+C too: close the file, then either finalize the
                # take (partial data is still labeled data) or drop it if empty.
                print(flush=True)
                recorder.stop()
                beep(500, 300)
                entry = {"pose": pose, "take": take, "frames": recorder.count,
                         "hands": hand_tag(recorder.hands_seen),
                         "ok": recorder.count > 0,
                         "reacquired": self._reacquired - before}
                if recorder.count == 0:
                    path.unlink(missing_ok=True)
                    raw_path.unlink(missing_ok=True)
                    print("      FAILED: no frames captured (did tracking stop?)\n")
                else:
                    final = finalize_pose_name(path, recorder.hands_seen)
                    entry["file"] = final.name
                    note = (f", {entry['reacquired']} re-acquisition(s)"
                            if entry["reacquired"] else "")
                    print(f"      saved {recorder.count} frames "
                          f"({entry['hands']}{note}) -> {final.name}\n")
                self.results.append(entry)

    def _raw_capture(self, path: Path):
        """LeapC's own .lmt recorder for this take, or nothing."""
        if not self.raw:
            return nullcontext()
        from leap_hand.replay import RawRecording
        return RawRecording(self.source, path)

    def print_summary(self) -> None:
        if not self.results:
            print("\nNothing recorded.")
            return
        ok = [r for r in self.results if r["ok"]]
        print("=" * 62)
        print(f"Session summary: {len(ok)}/{len(self.results)} takes captured")
        by_pose = {}
        for r in self.results:
            by_pose.setdefault(r["pose"], []).append(r)
        for pose, takes in by_pose.items():
            parts = []
            for r in takes:
                if r["ok"]:
                    note = f"take{r['take']}: {r['frames']}f/{r['hands']}"
                    if r["reacquired"]:
                        note += f" ({r['reacquired']} reacq)"
                else:
                    note = f"take{r['take']}: FAILED"
                parts.append(note)
            print(f"  {pose:<12} " + "   ".join(parts))
        if self._skipped_young:
            print(f"\n  {self._skipped_young} hands skipped: tracked for less "
                  f"than {MIN_VISIBLE_TIME_US / 1000:.0f} ms (settling)")
        if ok:
            print(f"\nFiles in {self.out_dir}")
            print(f"  stats:     python scripts/leap/stats.py {self.out_dir}")
            print("  playback:  python scripts/glove/playback.py <file>")
            print(f"  21 points: python scripts/glove/export_keypoints21.py {self.out_dir}")


# =============================================================================
# Set A: the professor's grasp protocol (--protocol). See the module docstring
# and docs/protocol_formats.md, which is the contract for every file below.
# =============================================================================
REPO = Path(__file__).resolve().parents[2]
SET_NAME = "grasps"
PROTOCOL_OUT = Path("recordings") / "protocol" / SET_NAME
# A rehearsal on synthetic hands goes beside the real sessions, never among
# them: whatever collects `recordings/protocol/grasps/` for the professor must
# not be able to pick up a mock session by accident. Same layout inside.
MOCK_PROTOCOL_OUT = Path("recordings") / "protocol_mock" / SET_NAME
# What a protocol file that leaves a field out gets: plan D4.
PROTOCOL_DEFAULTS = {"takes_per_item": 3, "duration_s": 5.0, "prep_s": 5.0}
PROTOCOL_RETRIES = 2
# Plan D3: not palm-to-lens for every grasp, an envelope every grasp can meet.
ENVELOPE_BAND_CM = (20.0, 40.0)
ENVELOPE = (
    "Orientation: hand 20 to 40 cm above the module, wrist inside the view,",
    "every finger chain visible to the lens, no finger edge-on. A palm-down",
    "grasp (lateral, hook, extension) is rotated only as much as that needs.",
)
STILL_MODES = ("hand", "full", "none")
STILL_OFF = "still_off"        # still_missing_reason with --still none
STILL_WAIT_S = 1.5             # how long the viewer gets to write the still
REVIEW_KEYS = "Enter or space = keep   r = redo   q = quit"
MIN_TRACKED = MIN_TRACKED_FRACTION   # plan section 4, the acquisition gate
LEAP_DIR, STILLS_DIR, KEYPOINTS_DIR, META_DIR = "leap", "stills", "keypoints", "meta"
REJECTED_DIR = "rejected"
# An item id goes into every file name between separators, so it must be
# file-name safe and must not contain the `_take` that the name is split on.
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# --- the coached take (see the module docstring for why) -----------------------
OPEN_HAND_TEXT = "OPEN HAND, palm to the camera"
MAKE_TEXT = "MAKE THE GRASP, keep the palm toward the camera"
LOST_TEXT = "LOST YOU: open the hand, then close it slower"
OPEN_STATUS, MAKE_STATUS, LOST_STATUS = "OPEN HAND", "MAKE THE GRASP", "LOST YOU"
# An item's `orientation` text in the protocol file, or this.
DEFAULT_ORIENTATION = "palm toward the camera"
OPEN_BAND_CM = (18.0, 40.0)    # palm height above the module that counts
OPEN_MAX_ANGLE_DEG = 50.0      # the palm within this of facing the lens
OPEN_HOLD_S = 0.5              # both, without a break, for this long
OPEN_GAP_S = LOSS_GAP_S        # a longer hole in the hand starts the 0.5 s again
OPEN_HAND_TIMEOUT_S = 30.0     # no open hand by then: the attempt is rejected
FORM_S = 4.0                   # MAKE THE GRASP, unless --prep says otherwise
FORM_LOST_S = 0.3              # gone longer than this while forming: LOST YOU
FORM_MAX_LOSSES = 3            # the third loss in one attempt rejects it
NOT_RECORDED = "not_recorded"  # still_missing_reason of an attempt never recorded
BEEP_ACQUIRED = (880, 120)
BEEP_MAKE = (1400, 200)        # the high beep
BEEP_LOST = (300, 450)         # the low beep
# A protocol file line holding nothing but an item's orientation hint.
_HINT_LINE = re.compile(
    rb'^[ \t]*"orientation"[ \t]*:[ \t]*"(?:[^"\\\r\n]|\\.)*"[ \t]*,[ \t]*\r?\n',
    re.MULTILINE)


class QuitSession(Exception):
    """The operator pressed q at a take's review."""


# --- the protocol file --------------------------------------------------------
def load_protocol(path: Path) -> Tuple[dict, str]:
    """(the protocol, sha256 of its bytes). Exits with the reason if unusable.

    Only what this recorder needs is checked: an `items` list whose entries
    have a file-name safe, unique `id` and a `label`. A Set B or C file is
    refused by name, because its items carry cycles or steps that this
    recorder would silently ignore.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise SystemExit(f"cannot read the protocol file {path}: {e}")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SystemExit(f"the protocol file {path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"the protocol file {path} is not a JSON object")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"the protocol file {path} has no items")
    seen = set()
    for n, item in enumerate(items, 1):
        if not isinstance(item, dict) or not item.get("id") or not item.get("label"):
            raise SystemExit(f"item {n} of {path} needs an id and a label")
        iid = str(item["id"])
        if not _ID_RE.fullmatch(iid) or "_take" in iid:
            raise SystemExit(f"item id {iid!r} in {path} is not file-name safe "
                             "(letters, digits, _ and -, no '_take')")
        if iid in seen:
            raise SystemExit(f"item id {iid!r} appears twice in {path}")
        seen.add(iid)
        if "steps" in item or "cycles" in item:
            raise SystemExit(
                f"{path} is a finger flexion or sequence protocol (item "
                f"{iid!r} has steps or cycles). Record it with "
                "scripts/record_protocol.py; this recorder is for grasps.")
        hint = item.get("orientation")
        if hint is not None and (not isinstance(hint, str) or not hint.strip()):
            raise SystemExit(f"item {iid!r} of {path}: 'orientation' must be "
                             "text (the hint shown while the grasp is made)")
    for key, default in PROTOCOL_DEFAULTS.items():
        value = data.get(key, default)
        try:
            data[key] = type(default)(value)
        except (TypeError, ValueError):
            raise SystemExit(f"{key} in {path} must be a number, not {value!r}")
    return data, hashlib.sha256(raw).hexdigest()


def sha256_without_hints(raw: bytes) -> str:
    """sha256 of a protocol file with every `"orientation": "...",` line
    taken out.

    What the file hashed to before the per-grasp hints were added, when
    adding them on lines of their own is all that changed: `--resume` uses
    it to carry on a session recorded before the hints existed, and refuses
    every other change as before.
    """
    return hashlib.sha256(_HINT_LINE.sub(b"", raw)).hexdigest()


def orientation_text(item: dict) -> str:
    """The grasp's orientation hint, or "palm toward the camera"."""
    hint = item.get("orientation")
    return hint.strip() if isinstance(hint, str) and hint.strip() else DEFAULT_ORIENTATION


def select_items(protocol: dict, only: Optional[str]) -> List[dict]:
    """The protocol's items, or the ones named in `--items`, in that order."""
    items = list(protocol["items"])
    if not only:
        return items
    by_id = {str(it["id"]): it for it in items}
    wanted = [x.strip() for x in only.split(",") if x.strip()]
    unknown = [x for x in wanted if x not in by_id]
    if unknown:
        raise SystemExit(f"not in the protocol: {', '.join(unknown)}\n"
                         f"  items are: {', '.join(by_id)}")
    return [by_id[x] for x in wanted]


def take_name(item: str, hand: str, n: int, stamp: str) -> str:
    """`<item>_<hand>_take<N>_<YYYYMMDD_HHMMSS>`, contract section 1."""
    return f"{item}_{hand}_take{n}_{stamp}"


def repo_relative(path: Path) -> str:
    """The path relative to the repo when it is inside it, forward slashes."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def tool_commit() -> Optional[str]:
    """The repo's short commit, or None when git cannot say."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def iso_now(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def _num(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


_EXPORTER = None


def prof_exporter():
    """`scripts/glove/export_prof_format.py`, imported by path, once.

    The professor's block layout is defined there and nowhere else, for the
    reason `scripts/leap/record_frame.py` gives: two copies would drift while
    both claimed to be his format.
    """
    global _EXPORTER
    if _EXPORTER is None:
        path = REPO / "scripts" / "glove" / "export_prof_format.py"
        spec = importlib.util.spec_from_file_location("glove_prof_format", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXPORTER = module
    return _EXPORTER


def write_json(path: Path, data: dict) -> None:
    """Write through a temporary file, so a reader never sees half of it.

    session.json is rewritten after every take while the checker may already
    be reading it. OneDrive can hold a file for a moment after it changes,
    so the swap is retried briefly rather than failing the session.
    """
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


def read_rows(path: Path) -> List[dict]:
    """A take's lines, parsed. The bytes actually written, not a copy of them."""
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- recording plumbing ---------------------------------------------------------
class _KeysWriter:
    """The recorder's file handle, with fixed keys added to every line.

    The contract's `session` and `item` on every frame, added where the line
    meets the file, the same trick `scripts/record_simultaneous.py` uses for
    `capture_time`: `LeapRecorder` is not changed and its `record()` is not
    copied. `take` is already on the line (LeapRecorder's own label).
    """

    def __init__(self, fh, keys: dict):
        self._fh = fh
        self._keys = dict(keys)

    def write(self, text: str) -> int:
        if text.strip():
            d = json.loads(text)
            d.update(self._keys)
            text = json.dumps(d) + "\n"
        return self._fh.write(text)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class _Tap:
    """The tracking source, remembering the newest hand of each label.

    The mock still is drawn from the hand the tracker reported at the
    moment the still is asked for, as the real viewer does; everything else
    passes straight through to the source.
    """

    def __init__(self, source):
        self._source = source
        self.last = {}

    def drain(self, max_items: int = 16):
        items = self._source.drain(max_items)
        for _side, lh in items:
            self.last[lh.hand_side] = lh
        return items

    def latest(self, prefer: Optional[str] = None):
        if prefer in self.last:
            return self.last[prefer]
        return next(iter(self.last.values()), None)

    def __getattr__(self, name):
        return getattr(self._source, name)


def _still_view_class():
    """`leap_hand.protocol.CameraView` that also passes `--still` to the viewer.

    The viewer (`scripts/leap/camera_view.py`) already knows how to cut a
    still to the hand or keep the whole frame; `CameraView.start` just never
    tells it which. Built on first use so the plain session never imports it.
    """
    from leap_hand.protocol import CameraView

    class StillView(CameraView):
        def __init__(self, still: str = "hand", **kwargs):
            super().__init__(**kwargs)
            self.still = still

        def start(self):
            if not self.enabled or self._proc is not None:
                return self
            import atexit
            import tempfile
            self._status = (Path(tempfile.gettempdir())
                            / f"leap_view_status_{os.getpid()}.txt")
            self.caption("starting")
            cmd = [sys.executable, str(self.SCRIPT), "--hand", self.hand,
                   "--status-file", str(self._status),
                   "--parent-pid", str(os.getpid()), "--still", self.still]
            if self.band:
                cmd += ["--band", f"{self.band[0]:g},{self.band[1]:g}"]
            try:
                self._proc = subprocess.Popen(cmd)
            except OSError:
                self.enabled = False
                return self
            atexit.register(self.close)
            return self

    return StillView


# Joint chains of the 26 OpenXR joints, wrist outward, for drawing.
_CHAINS = ([1, 2, 3, 4, 5],) + tuple([1] + list(range(6 + 5 * f, 11 + 5 * f))
                                     for f in range(4))


def write_mock_still(path: Path, lh, caption: str = "",
                     full: bool = False) -> Optional[str]:
    """A still for a mock take: the synthetic IR ramp with the hand drawn on it.

    There is no camera behind `--mock`, so there is no picture to cut. The
    still is drawn instead from `leap_hand.images.MockImageSampler`'s ramp,
    which is obviously synthetic on sight, and it says MOCK on it, so nobody
    can mistake it for evidence. It exists to exercise the rest of the path:
    the crop (`hand_crop_box`, the viewer's own rule), the PNG, the review.
    With no hand it writes the viewer's `.skipped.txt` note instead, the
    same as the real viewer. `full` keeps the whole frame, as `--still full`
    asks the viewer to. Returns the reason when there is no still.
    """
    import cv2
    import numpy as np

    from leap_hand.images import MockImageSampler
    from leap_hand.protocol import (NO_TRACKED_HAND, hand_crop_box,
                                    skipped_still_path, skipped_still_text)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 768
    pts = []
    if lh is not None:
        for x, y, z in lh.abs26:
            if y <= 0.005:
                pts.append(None)
                continue
            u = size / 2 + 300.0 * x / y
            v = size / 2 + 300.0 * z / y
            pts.append((int(round(size - 1 - u)), int(round(v))))   # mirrored, as the viewer is
    box = hand_crop_box([p for p in pts if p is not None], size) if pts else None
    if box is None:
        skipped_still_path(path).write_text(
            skipped_still_text(NO_TRACKED_HAND, time.time()), encoding="utf-8")
        return NO_TRACKED_HAND

    eye = MockImageSampler().latest().left
    frame = cv2.cvtColor(cv2.resize(eye, (size, size)), cv2.COLOR_GRAY2BGR)
    colour = (255, 220, 0) if lh.hand_side == "left" else (60, 60, 255)
    for chain in _CHAINS:
        for a, b in zip(chain, chain[1:]):
            if pts[a] is not None and pts[b] is not None:
                cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
    x0, y0, x1, y1 = (0, 0, size, size) if full else box
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    for k, text in enumerate(("MOCK, not evidence", caption)):
        if text:
            # Shrunk to fit the crop, as the viewer fits its caption.
            width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0][0] or 1
            scale = max(0.3, min(0.55, (crop.shape[1] - 16) / width))
            org = (8, 20 + 20 * k)
            cv2.putText(crop, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(crop, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), crop):
        return "write_failed"
    return None


# --- the operator's decision ------------------------------------------------------
class Reviewer:
    """Asks the operator to keep, redo or quit after a take passed the gate.

    The still and the numbers go in a review window beside the camera window,
    and the key is read from that window or from the console, whichever has
    the focus, so a bare hand on the keyboard is all it takes. `auto` keeps
    every take without asking (the mock rehearsal, hands-free runs). `keys`
    is a scripted key sequence for tests.
    """

    WINDOW = "Take review"
    ACTIONS = {"\r": "accept", "\n": "accept", " ": "accept",
               "r": "redo", "q": "quit"}

    def __init__(self, auto: bool = False, keys: Optional[Iterator[str]] = None):
        self.auto = bool(auto)
        self._keys = keys

    def decide(self, still: Optional[Path], lines: List[str],
               pump=None, screen=None, reference=None) -> Tuple[str, float, str]:
        """("accept" | "redo" | "quit", when, "auto" | "operator").

        `screen` is the session's COPY THIS window: when it is open the
        review is shown in it, with `reference` (the paper's picture of the
        grasp) beside the still, and the window stays open afterwards.
        """
        if self.auto:
            return "accept", time.time(), "auto"
        if self._keys is not None:
            key = next(self._keys, "q")
            return self.ACTIONS.get(key.lower(), "quit"), time.time(), "operator"
        print(f"      {REVIEW_KEYS}")
        try:
            if screen is not None and screen.enabled:
                return self._in_screen(screen, still, lines, pump, reference)
            return self._window(still, lines, pump)
        except Exception as e:                     # no GUI: fall back to the console
            print(f"      (review window unavailable: {e})")
        while True:
            answer = input("      keep (Enter), redo (r) or quit (q)? ")
            action = self.ACTIONS.get((answer[:1] or "\r").lower())
            if action:
                return action, time.time(), "operator"

    def _window(self, still, lines, pump) -> Tuple[str, float, str]:
        import cv2
        image = self.compose(still, lines)
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setWindowProperty(self.WINDOW, cv2.WND_PROP_TOPMOST, 1)
            cv2.moveWindow(self.WINDOW, 800, 40)
        except Exception:
            pass
        try:
            msvcrt = __import__("msvcrt")
        except ImportError:
            msvcrt = None
        try:
            while True:
                if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(self.WINDOW, image)
                code = cv2.waitKey(50)
                key = chr(code & 0xFF) if code != -1 else ""
                if not key and msvcrt is not None and msvcrt.kbhit():
                    key = msvcrt.getwch()
                action = self.ACTIONS.get(key.lower()) if key else None
                if action:
                    return action, time.time(), "operator"
                if pump is not None:
                    pump()
        finally:
            try:
                cv2.destroyWindow(self.WINDOW)
                cv2.waitKey(1)
            except Exception:
                pass

    def _in_screen(self, screen, still, lines, pump, reference) -> Tuple[str, float, str]:
        """The review in the COPY THIS window, read from it or the console."""
        image = self.compose(still, lines, reference)
        try:
            msvcrt = __import__("msvcrt")
        except ImportError:
            msvcrt = None
        # Keys typed into the console during the countdown or the take are
        # stale: one of them must not keep or redo this take unseen.
        while msvcrt is not None and msvcrt.kbhit():
            msvcrt.getwch()
        while True:
            code = screen.show(image)
            if not screen.enabled:
                raise RuntimeError("the COPY THIS window could not be shown")
            key = chr(code & 0xFF) if code != -1 else ""
            if not key and msvcrt is not None and msvcrt.kbhit():
                key = msvcrt.getwch()
            action = self.ACTIONS.get(key.lower()) if key else None
            if action:
                return action, time.time(), "operator"
            if pump is not None:
                pump()
            time.sleep(0.03)

    @staticmethod
    def compose(still: Optional[Path], lines: List[str], reference=None):
        """The still (or a note that there is none) above the numbers, and
        the paper's picture of the grasp beside the still when given."""
        import cv2
        import numpy as np

        width = 720
        img = cv2.imread(str(still)) if still is not None and Path(still).is_file() else None
        if reference is not None:
            ref_w = 250
            ref = _fit_into(reference, ref_w, 480)
            box_w = width - ref_w - 30
            if img is None:
                left = np.zeros((120, box_w, 3), np.uint8)
                cv2.putText(left, "no still for this take", (6, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2, cv2.LINE_AA)
            else:
                left = _fit_into(img, box_w, 480)
            top = np.zeros((max(left.shape[0], ref.shape[0]), width, 3), np.uint8)
            top[:left.shape[0], 10:10 + left.shape[1]] = left
            x = width - 10 - ref.shape[1]
            top[:ref.shape[0], x:x + ref.shape[1]] = ref
        elif img is None:
            top = np.zeros((120, width, 3), np.uint8)
            cv2.putText(top, "no still for this take", (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2, cv2.LINE_AA)
        else:
            scale = min(width / img.shape[1], 480 / img.shape[0])
            img = cv2.resize(img, (max(1, int(img.shape[1] * scale)),
                                   max(1, int(img.shape[0] * scale))))
            top = np.zeros((img.shape[0], width, 3), np.uint8)
            x = (width - img.shape[1]) // 2
            top[:, x:x + img.shape[1]] = img
        panel = np.zeros((34 * (len(lines) + 1), width, 3), np.uint8)
        for k, text in enumerate(lines + [REVIEW_KEYS]):
            colour = (0, 255, 255) if k == 0 or k == len(lines) else (230, 230, 230)
            cv2.putText(panel, text, (14, 26 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62 if k else 0.8, colour, 2 if k == 0 else 1, cv2.LINE_AA)
        return np.vstack([top, panel])


# --- the picture to copy ------------------------------------------------------------
# For every grasp the protocol names a picture: `images_dir` at the top of the
# file, `image` on each item, the panel cut from the paper's figure by
# `make_panels.py` beside the papers. It fills the COPY THIS window through the
# countdown and the take, so the operator never has to look for it on a sheet.
COPY_WINDOW = "COPY THIS"
COPY_W = 720                  # drawing width of the window
COPY_BAND_H = 170             # the label and the status line, above the picture
COPY_PANEL_H = 700            # the picture, drawn this tall when its width allows
COPY_H = COPY_BAND_H + COPY_PANEL_H + 10
COPY_REFRESH_S = 0.04         # redrawn at most 25 times a second
COPY_X = 800                  # beside the camera window, where the review was
_BG, _WHITE, _GREY = (32, 32, 32), (255, 255, 255), (200, 200, 200)
_HINT, _RED = (0, 200, 255), (60, 60, 255)
_STATUS_COLOURS = {"GET READY": (0, 220, 255), "HOLD STILL": (70, 70, 255),
                   OPEN_STATUS: (255, 210, 60), MAKE_STATUS: (90, 230, 90),
                   LOST_STATUS: _RED}
# Statuses that count down in whole seconds; HOLD STILL shows tenths.
_WHOLE_SECONDS = ("GET READY", MAKE_STATUS)


def panel_dir(protocol: dict, protocol_path=None) -> Optional[Path]:
    """The protocol's `images_dir`, a relative one taken from the protocol
    file's folder; None when the file has none."""
    folder = protocol.get("images_dir")
    if not folder:
        return None
    folder = Path(str(folder))
    if not folder.is_absolute() and protocol_path is not None:
        folder = Path(protocol_path).resolve().parent / folder
    return folder


def panel_path(protocol: dict, item: dict, protocol_path=None) -> Optional[Path]:
    """Where `item`'s picture should be, `images_dir` / `image`, or None when
    either key is absent. Whether the file exists is not checked here."""
    folder = panel_dir(protocol, protocol_path)
    name = item.get("image")
    if folder is None or not name:
        return None
    return folder / str(name)


def _fit_into(img, width: int, height: int):
    """`img` scaled to fit width x height, aspect kept; as is when it fits exactly."""
    import cv2

    h, w = img.shape[:2]
    f = min(width / w, height / h)
    size = (max(1, int(round(w * f))), max(1, int(round(h * f))))
    if size == (w, h):
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def _wrap_text(text: str, scale: float, thick: int, width: int) -> List[str]:
    """Words wrapped to `width` pixels in OpenCV's plain font at `scale`."""
    import cv2

    lines, line = [], ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if line and cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                    thick)[0][0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def _fit_text(text: str, width: int, max_lines: int, scales, thick: int):
    """(lines, scale): the largest of `scales` at which `text` takes at most
    `max_lines` lines, else the smallest, with every line."""
    for scale in scales:
        lines = _wrap_text(text, scale, thick, width)
        if len(lines) <= max_lines:
            break
    return lines, scale


def status_text(status: str, seconds: Optional[float]) -> str:
    """'GET READY  3', 'MAKE THE GRASP  3', 'HOLD STILL  2.4 s', or the
    status alone."""
    if seconds is None:
        return status
    if status in _WHOLE_SECONDS:
        return f"{status}  {max(0, math.ceil(seconds - 1e-9))}"
    return f"{status}  {max(0.0, seconds):.1f} s"


def _draw_band(put, item: dict, take: int, takes: int, status: str,
               seconds: Optional[float]):
    """The band above the picture: the label, the take, the status.

    Returns (label lines, status text, label_bottom, status_top), the rows
    where the label's letters end and the status line's begin.
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    # The label on one line as large as 1.2, else on two lines no larger
    # than 1.0, which is what leaves the status line room below it.
    label = str(item.get("label") or item.get("id"))
    lines, scale = _fit_text(label, COPY_W - 32, 1, (1.2, 1.1, 1.0), 2)
    if len(lines) > 1:
        lines, scale = _fit_text(label, COPY_W - 32, 2,
                                 (1.0, 0.9, 0.8, 0.7, 0.6), 2)
    label_lines = list(lines)
    label_bottom = 0
    for k, line in enumerate(lines):
        y = 14 + int(30 * scale) + k * int(42 * scale)
        put(line, (16, y), scale, _WHITE, 2)
        label_bottom = max(label_bottom,
                           y + cv2.getTextSize(line, font, scale, 2)[1] + 2)
    y_status = COPY_BAND_H - 20
    put(f"take {take}/{takes}", (16, y_status), 1.0, _GREY, 2)
    text = status_text(status, seconds)
    (width, height), _base = cv2.getTextSize(text, font, 1.4, 3)
    put(text, (COPY_W - 16 - width, y_status), 1.4,
        _STATUS_COLOURS.get(status, _GREY), 3)
    status_top = y_status - max(height, cv2.getTextSize(
        f"take {take}/{takes}", font, 1.0, 2)[0][1]) - 2
    return label_lines, text, label_bottom, status_top


def compose_copy(item: dict, panel, take: int, takes: int, status: str,
                 seconds: Optional[float] = None):
    """One frame of the COPY THIS window, and what is on it.

    The grasp's label in large letters, the take number and the countdown
    ("GET READY 3", "MAKE THE GRASP 3") or "HOLD STILL 2.4 s" above the
    paper's picture, drawn 700 px tall (narrower when the picture is wide).
    During MAKE THE GRASP the grasp's orientation hint (`orientation_text`)
    sits under the status line and the picture gives up the room it takes.
    Without a picture (`panel` None) the grasp's shape is written in its
    place, large, with where it is in the papers. Returns (image, info):
    info["lines"] is every text drawn, info["label"] the label's lines,
    info["status"] the status text, info["hint"] the hint's lines (empty
    outside MAKE THE GRASP), info["panel_box"] the picture's (x, y, w, h)
    or None, and info["label_bottom"] and info["status_top"] the rows where
    the label's letters end and the status line's begin.
    """
    import cv2
    import numpy as np

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.full((COPY_H, COPY_W, 3), _BG, np.uint8)
    drawn: List[str] = []

    def put(text, org, scale, colour, thick):
        cv2.putText(img, text, org, font, scale, colour, thick, cv2.LINE_AA)
        drawn.append(text)

    label_lines, text, label_bottom, status_top = _draw_band(
        put, item, take, takes, status, seconds)
    label = str(item.get("label") or item.get("id"))
    top = COPY_BAND_H
    hint_lines: List[str] = []
    if status == MAKE_STATUS:
        hint_lines, hscale = _fit_text(orientation_text(item), COPY_W - 32, 2,
                                       (0.9, 0.85, 0.8, 0.75, 0.7, 0.65), 2)
        y = top + 6
        for line in hint_lines:
            y += cv2.getTextSize(line, font, hscale, 2)[0][1] + 14
            put(line, (16, y), hscale, _HINT, 2)
        top = y + 18

    box = None
    if panel is not None:
        pic = _fit_into(panel, COPY_W - 20, min(COPY_PANEL_H, COPY_H - 10 - top))
        h, w = pic.shape[:2]
        x = (COPY_W - w) // 2
        img[top:top + h, x:x + w] = pic
        box = (x, top, w, h)
    else:
        y = top + 30
        put("no picture for this grasp: copy this shape", (16, y), 0.7,
            (0, 160, 255), 2)
        shape = str(item.get("shape") or label)
        lines, scale = _fit_text(shape, COPY_W - 32, 8,
                                 (1.3, 1.2, 1.1, 1.0, 0.9, 0.8), 2)
        y += 30
        for line in lines:
            y += cv2.getTextSize(line, font, scale, 2)[0][1] + 22
            put(line, (16, y), scale, _WHITE, 2)
        where = ", ".join(str(item[k]) for k in ("source", "figure") if item.get(k))
        if where:
            y += 40
            for line in _wrap_text(f"in the papers: {where}", 0.65, 1, COPY_W - 32):
                y += 30
                put(line, (16, y), 0.65, _GREY, 1)
    return img, {"lines": drawn, "label": label_lines, "status": text,
                 "hint": hint_lines, "panel_box": box,
                 "label_bottom": label_bottom, "status_top": status_top}


def compose_open(item: dict, take: int, takes: int, state: str,
                 seconds: Optional[float] = None, lost: bool = False,
                 status: str = OPEN_STATUS):
    """One frame of COPY THIS while the open hand is being acquired.

    The band as always (the grasp coming up, the take, OPEN HAND or LOST
    YOU), then "OPEN HAND, palm to the camera" in the largest letters that
    fit, what the camera sees of the hand now (`state`), and the seconds
    left. `lost` puts "LOST YOU: open the hand, then close it slower" in
    red above it, for the OPEN HAND that follows a loss. The paper's
    picture is deliberately NOT drawn: copying it before the hand is
    tracked is exactly what lost the hand on 2026-10-01; it comes with
    MAKE THE GRASP. Returns (image, info) like `compose_copy`, with
    info["big"] the large lines and info["lost"] the red ones.
    """
    import cv2
    import numpy as np

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.full((COPY_H, COPY_W, 3), _BG, np.uint8)
    drawn: List[str] = []

    def put(text, org, scale, colour, thick):
        cv2.putText(img, text, org, font, scale, colour, thick, cv2.LINE_AA)
        drawn.append(text)

    label_lines, text, label_bottom, status_top = _draw_band(
        put, item, take, takes, status, None)
    y = COPY_BAND_H + 20
    lost_lines: List[str] = []
    if lost:
        lost_lines, lscale = _fit_text(LOST_TEXT, COPY_W - 32, 2,
                                       (1.1, 1.0, 0.9, 0.8), 2)
        for line in lost_lines:
            y += cv2.getTextSize(line, font, lscale, 2)[0][1] + 18
            put(line, (16, y), lscale, _RED, 2)
        y += 20
    head, tail = OPEN_HAND_TEXT.split(", ", 1)
    big = [head + ",", tail]
    bscale = 1.2
    for s in (2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2):
        if max(cv2.getTextSize(b, font, s, 5)[0][0] for b in big) <= COPY_W - 32:
            bscale = s
            break
    y += 30
    for line in big:
        (w, h), _b = cv2.getTextSize(line, font, bscale, 5)
        y += h + 30
        put(line, ((COPY_W - w) // 2, y), bscale,
            _STATUS_COLOURS[OPEN_STATUS], 5)
    y += 40
    if state:
        for line in _wrap_text(state, 0.9, 2, COPY_W - 32):
            y += 42
            put(line, (16, y), 0.9, _WHITE, 2)
    y += 30
    notes = [f"{OPEN_BAND_CM[0]:g} to {OPEN_BAND_CM[1]:g} cm above the module, "
             f"palm within {OPEN_MAX_ANGLE_DEG:g} degrees of the lens, "
             f"held {OPEN_HOLD_S:g} s",
             "the grasp's picture comes after the beep"]
    if seconds is not None:
        notes.append(f"{max(0, math.ceil(seconds - 1e-9))} s left")
    for note in notes:
        for line in _wrap_text(note, 0.65, 1, COPY_W - 32):
            y += 30
            put(line, (16, y), 0.65, _GREY, 1)
    return img, {"lines": drawn, "label": label_lines, "status": text,
                 "big": big, "lost": lost_lines, "panel_box": None,
                 "label_bottom": label_bottom, "status_top": status_top}


def _screen_size() -> Optional[Tuple[int, int]]:
    """The primary screen's (width, height), or None off Windows."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return None


class CopyWindow:
    """The COPY THIS window: the paper's picture of the grasp being recorded.

    Shown through the preparation countdown and the take; the take's review
    then appears in the same window (`Reviewer.decide(screen=...)`), and the
    next countdown brings the next grasp's picture back. A grasp whose
    picture is missing gets its shape text instead, and the console says so
    once. A machine that cannot open a window turns it off with one line on
    the console and the session carries on.
    """

    def __init__(self, protocol: dict, protocol_path=None,
                 items: Optional[List[dict]] = None):
        self.enabled = True
        self.protocol = protocol
        self.protocol_path = protocol_path
        self.folder = panel_dir(protocol, protocol_path)
        self._panels = {}
        self._size = None          # the drawing size the window was fitted to
        self._open = False
        self._next = 0.0
        self.items = list(items if items is not None else protocol.get("items") or [])

    def describe(self) -> str:
        """The session header's line about the pictures."""
        if self.folder is None:
            return ("the protocol file has no images_dir: the COPY THIS window "
                    "shows each grasp's shape text instead of its picture")
        found = sum(1 for it in self.items
                    if (p := panel_path(self.protocol, it, self.protocol_path))
                    is not None and p.is_file())
        return (f"{found} of {len(self.items)} pictures found in {self.folder} "
                "(COPY THIS window)")

    def panel(self, item: dict):
        """The item's picture, fitted to the window once and kept; None when
        there is none, which the console is told the first time."""
        iid = str(item.get("id"))
        if iid not in self._panels:
            import cv2

            path = panel_path(self.protocol, item, self.protocol_path)
            img = None
            if path is not None and path.is_file():
                img = cv2.imread(str(path))
            if img is None and self.folder is not None:
                why = ("its item has no image" if path is None else
                       f"{path} is missing" if not path.is_file() else
                       f"{path} cannot be read")
                print(f"      no picture for {iid}: {why}; the COPY THIS window "
                      "shows its shape text instead")
            self._panels[iid] = (None if img is None
                                 else _fit_into(img, COPY_W - 20, COPY_PANEL_H))
        return self._panels[iid]

    def show_copy(self, item: dict, take: int, takes: int, status: str,
                  seconds: Optional[float] = None, force: bool = False) -> None:
        """Redraw the window for this moment of the take; at most 25 times a
        second unless `force`."""
        if not self.enabled:
            return
        now = time.time()
        if not force and now < self._next:
            return
        self._next = now + COPY_REFRESH_S
        img, _info = compose_copy(item, self.panel(item), take, takes, status, seconds)
        self.show(img)

    def show_open(self, item: dict, take: int, takes: int, state: str,
                  seconds: Optional[float] = None, lost: bool = False,
                  status: str = OPEN_STATUS, force: bool = False) -> None:
        """The OPEN HAND frame (`compose_open`), throttled like `show_copy`."""
        if not self.enabled:
            return
        now = time.time()
        if not force and now < self._next:
            return
        self._next = now + COPY_REFRESH_S
        img, _info = compose_open(item, take, takes, state, seconds, lost, status)
        self.show(img)

    def show(self, img) -> int:
        """Put `img` in the window, without waiting; the key pressed or -1.

        Smaller pictures (the review) are drawn on the window's own size, so
        the window does not jump between the countdown and the review.
        """
        if not self.enabled:
            return -1
        try:
            import cv2
            import numpy as np

            h, w = img.shape[:2]
            if h < COPY_H or w < COPY_W:
                canvas = np.full((max(h, COPY_H), max(w, COPY_W), 3), _BG, np.uint8)
                canvas[:h, (canvas.shape[1] - w) // 2:(canvas.shape[1] - w) // 2 + w] = img
                img = canvas
            if not self._visible():
                cv2.namedWindow(COPY_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                self._open, self._size = True, None
                try:
                    cv2.setWindowProperty(COPY_WINDOW, cv2.WND_PROP_TOPMOST, 1)
                except Exception:
                    pass
            if self._size != img.shape[:2]:
                self._size = img.shape[:2]
                self._place(*self._size)
            cv2.imshow(COPY_WINDOW, img)
            return cv2.pollKey()
        except Exception as e:
            self.enabled = False
            print(f"      (COPY THIS window unavailable: {e}; carrying on without it)")
            return -1

    def _visible(self) -> bool:
        if not self._open:
            return False
        import cv2
        try:
            return cv2.getWindowProperty(COPY_WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False

    @staticmethod
    def _place(h: int, w: int) -> None:
        """Sized to fit the screen, beside the camera window when there is room."""
        import cv2

        screen = _screen_size()
        f = 1.0 if screen is None else min(1.0, 0.95 * screen[0] / w,
                                           0.88 * screen[1] / h)
        cv2.resizeWindow(COPY_WINDOW, int(w * f), int(h * f))
        x = COPY_X if screen is None else max(0, min(COPY_X, screen[0] - int(w * f) - 10))
        try:
            cv2.moveWindow(COPY_WINDOW, x, 0)
        except Exception:
            pass

    def close(self) -> None:
        if self._open:
            self._open = False
            try:
                import cv2
                cv2.destroyWindow(COPY_WINDOW)
                cv2.waitKey(1)
            except Exception:
                pass


# --- the coached take: watching the hand -----------------------------------------
@dataclass
class HandSeen:
    """One tracked hand as the coached take judges it."""

    lh: object                 # the LeapHand
    wall: float                # when it was drained, time.time()
    ts: float                  # the tracker's own time of the frame, seconds
    height_cm: float
    angle_deg: Optional[float]  # 0 = palm square to the lens, 90 = edge-on


def hand_seen(lh, now: float) -> HandSeen:
    ts = getattr(lh, "timestamp_us", None)
    return HandSeen(lh=lh, wall=now, ts=now if ts is None else ts / 1e6,
                    height_cm=palm_height_cm(lh.palm_pos),
                    angle_deg=hand_view_angle_deg(lh))


class OpenHandWatch:
    """OPEN HAND: has a hand been open over the module long enough?

    A hand counts while its palm is `band` cm above the module and within
    `max_angle` degrees of facing the lens, and it is acquired once it has
    counted for `hold_s` without a hole longer than `gap_s`. Both are timed
    on the tracker's own clock, so a slow window redraw can neither cut the
    hold short nor stretch it. Every hand id is followed on its own; the
    operator's label wins a tie, but no label is required, because the
    tracker calls the left hand "right" often enough.
    """

    def __init__(self, band=OPEN_BAND_CM, max_angle: float = OPEN_MAX_ANGLE_DEG,
                 hold_s: float = OPEN_HOLD_S, gap_s: float = OPEN_GAP_S,
                 prefer: Optional[str] = None):
        self.band = tuple(band)
        self.max_angle = float(max_angle)
        self.hold_s = float(hold_s)
        self.gap_s = float(gap_s)
        self.prefer = prefer
        self.runs: Dict[int, List[float]] = {}    # hand id -> [first ts, last ts]
        self.latest: Dict[int, HandSeen] = {}

    def fits(self, h: HandSeen) -> bool:
        return (self.band[0] <= h.height_cm <= self.band[1]
                and h.angle_deg is not None and h.angle_deg <= self.max_angle)

    def add(self, lh, now: float) -> None:
        h = hand_seen(lh, now)
        hid = int(lh.hand_id)
        self.latest[hid] = h
        if not self.fits(h):
            self.runs.pop(hid, None)
            return
        run = self.runs.get(hid)
        if run is None or h.ts - run[1] > self.gap_s + 1e-6 or h.ts < run[1]:
            self.runs[hid] = [h.ts, h.ts]
        else:
            run[1] = h.ts

    def acquired(self) -> Optional[HandSeen]:
        """The newest frame of the acquired hand, or None yet."""
        done = [hid for hid, (a, b) in self.runs.items()
                if b - a >= self.hold_s - 1e-6]
        if not done:
            return None
        done.sort(key=lambda hid: (self.latest[hid].lh.hand_side != self.prefer,
                                   self.runs[hid][0] - self.runs[hid][1]))
        return self.latest[done[0]]

    def others(self, hid: int, now: float, window: float = 0.5) -> Set[int]:
        """Hand ids other than `hid` seen in the last `window` seconds."""
        return {k for k, h in self.latest.items()
                if k != hid and now - h.wall <= window}

    def _current(self, now: float) -> Optional[HandSeen]:
        fresh = [h for h in self.latest.values() if now - h.wall <= FORM_LOST_S]
        if not fresh:
            return None
        fresh.sort(key=lambda h: (h.lh.hand_side != self.prefer, -h.wall))
        return fresh[0]

    def status(self, now: float) -> str:
        """What the camera sees of the hand now, and what to change."""
        lo, hi = self.band
        h = self._current(now)
        if h is None:
            return f"no hand seen: hold the open hand {lo:g} to {hi:g} cm above the module"
        fixes = []
        if h.height_cm < lo:
            fixes.append(f"raise it to {lo:g} to {hi:g} cm (now {h.height_cm:.0f} cm)")
        elif h.height_cm > hi:
            fixes.append(f"lower it to {lo:g} to {hi:g} cm (now {h.height_cm:.0f} cm)")
        if h.angle_deg is None or h.angle_deg > self.max_angle:
            now_deg = "" if h.angle_deg is None else f" (now {h.angle_deg:.0f} degrees)"
            fixes.append(f"turn the palm to the camera{now_deg}")
        if fixes:
            return "; ".join(fixes)
        run = self.runs.get(int(h.lh.hand_id))
        held = 0.0 if run is None else run[1] - run[0]
        return f"hold it there: {min(held, self.hold_s):.1f} of {self.hold_s:g} s"

    def need(self, now: float) -> str:
        """`status` in a few words without numbers, for the camera window's
        caption, whose file is rewritten only when the words change."""
        h = self._current(now)
        if h is None:
            return "no hand"
        if h.height_cm < self.band[0]:
            return "raise it"
        if h.height_cm > self.band[1]:
            return "lower it"
        if h.angle_deg is None or h.angle_deg > self.max_angle:
            return "turn the palm to the camera"
        return "hold it there"


class FormWatch:
    """MAKE THE GRASP: is the acquired hand still tracked?

    The hand is the id acquired at OPEN HAND. An id the tracker gives it
    afterwards is followed too (a re-acquisition), but never one that was
    already in view at OPEN HAND as another hand. Lost means no frame of it
    for more than `lost_s`: between two of its frames on the tracker's own
    clock, or since the last drain that brought one. The queue is drained
    before each judgement, so a slow redraw or a blocking beep is not a
    loss: the frames that queued behind it arrive first.
    """

    def __init__(self, start: HandSeen, others: Set[int],
                 lost_s: float = FORM_LOST_S):
        self.hid = int(start.lh.hand_id)
        self.last = start
        self.others = set(others) - {self.hid}
        self.lost_s = float(lost_s)
        self.hole: Optional[HandSeen] = None
        self.new_ids = 0

    def add(self, lh, now: float) -> None:
        hid = int(lh.hand_id)
        if hid != self.hid:
            if hid in self.others:
                return
            self.hid = hid
            self.new_ids += 1
        h = hand_seen(lh, now)
        if self.hole is None and h.ts - self.last.ts > self.lost_s:
            self.hole = self.last
        self.last = h

    def lost(self, now: float) -> Optional[HandSeen]:
        """The hand's last frame before it was lost, or None if it was not."""
        if self.hole is not None:
            return self.hole
        if now - self.last.wall > self.lost_s:
            return self.last
        return None


def form_loss(h: HandSeen, t0: float) -> dict:
    """Where the hand was when it was lost while forming, and the likely cause."""
    cause = causes_for(state_from_leaphand(h.lh))[0]
    grab = getattr(h.lh, "grab_strength", None)
    return {"after_s": round(max(0.0, h.wall - t0), 2),
            "height_cm": round(h.height_cm, 1),
            "view_angle_deg": None if h.angle_deg is None else round(h.angle_deg, 1),
            "grab": None if grab is None else round(float(grab), 3),
            "hand_label": h.lh.hand_side,
            "cause": cause_label(cause), "fix": fix_for(cause)}


def form_loss_where(loss: dict) -> str:
    """`at 25 cm with the palm 73 degrees from the lens (cause: fix)`."""
    where = f"at {loss['height_cm']:.0f} cm"
    if loss.get("view_angle_deg") is not None:
        where += f" with the palm {loss['view_angle_deg']:.0f} degrees from the lens"
    return f"{where} ({loss['cause']}: {loss['fix']})"


def forming_reason(losses: List[dict]) -> str:
    """The reason an attempt is rejected after the hand was lost while forming."""
    n = len(losses)
    return (f"lost the hand {n} time{'' if n == 1 else 's'} while forming the "
            f"grasp, the last {form_loss_where(losses[-1])}")


# --- the session ------------------------------------------------------------------
@dataclass
class Attempt:
    """One recorded attempt, as the end-of-session table shows it."""

    item: str
    take: int
    attempt: int
    name: str
    accepted: bool
    reason: str
    tracked_fraction: float
    grab: Optional[float]
    pinch: Optional[float]
    by: str


def load_session(folder: Path, hand: str, sha256: str, mock: bool,
                 raw: Optional[bytes] = None) -> dict:
    """An earlier session's session.json, checked before `--resume` adds to it.

    Refused when it is not a grasp session, is another hand's, was a mock
    when this run is not (or the other way round), or was recorded against
    a protocol file whose bytes have since changed: one session, one hand,
    one version of the grasp list. The one change let through is per-grasp
    orientation hints added on lines of their own (`sha256_without_hints`
    of today's bytes, `raw`, is the session's sha256): the grasps, takes and
    timings are then byte for byte the ones the session was recorded with.
    """
    path = Path(folder) / "session.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot resume {folder}: no readable session.json ({e})")
    if old.get("set") != SET_NAME:
        raise SystemExit(f"cannot resume {folder}: it is a {old.get('set')!r} "
                         "session, not a grasp session")
    if old.get("hand") != hand:
        raise SystemExit(f"cannot resume {folder}: it is the {old.get('hand')} "
                         f"hand's session, not the {hand} hand's")
    if bool(old.get("mock")) != mock:
        raise SystemExit(f"cannot resume {folder}: it was "
                         f"{'a mock' if old.get('mock') else 'a camera'} session")
    if old.get("protocol_sha256") != sha256:
        if raw is None or sha256_without_hints(raw) != old.get("protocol_sha256"):
            raise SystemExit(f"cannot resume {folder}: the protocol file has "
                             "changed since that session was recorded; start a "
                             "new session")
        print(f"note: the protocol file differs from the one {Path(folder).name} "
              "was recorded with only by per-grasp orientation hints; resuming it")
    return old


def make_session_dir(root: Path, hand: str) -> Path:
    """`<root>/<YYYYMMDD_HHMMSS>_<hand>/`, new. Two runs in one second wait."""
    while True:
        folder = Path(root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{hand}"
        try:
            folder.mkdir(parents=True, exist_ok=False)
            return folder
        except FileExistsError:
            time.sleep(0.2)


class ProtocolSession(Session):
    """Set A: one guided session over a protocol file's grasps, one hand.

    Per take: announce the grasp and the orientation envelope, coach the
    hand into the grasp (OPEN HAND, then MAKE THE GRASP for `prep` seconds;
    with `coach` off, a plain `prep` second countdown), record every frame
    for the take's seconds, then read the file back and measure it
    (`leap_hand.static_interval.summarise_take`). An attempt the coaching
    gives up on (no open hand in `acquire_timeout` seconds, or the hand
    lost three times while forming) and a take that fails the acquisition
    gate move to `rejected/` with their reason and are retried up to
    `retries` times; a take that passes is shown to the operator, whose
    decision is final. session.json is rewritten after every decision, so
    a crash or a Ctrl+C never loses the record of what was already kept.
    """

    def __init__(self, source, out_root: Path, protocol: dict, sha256: str,
                 protocol_path: Path, hand: str, items: List[dict], takes: int,
                 duration: float, prep: float, static_s: float, retries: int,
                 still: str, reviewer: Reviewer, view=None, mock: bool = False,
                 raw: bool = False, note: str = "", operator: Optional[str] = None,
                 auto_accept: bool = False,
                 resume: Optional[Tuple[Path, dict]] = None, panel=None,
                 coach: bool = True,
                 acquire_timeout: float = OPEN_HAND_TIMEOUT_S):
        self.session_dir = (Path(resume[0]) if resume is not None
                            else make_session_dir(out_root, hand))
        super().__init__(_Tap(source), hz=None, out_dir=self.session_dir, raw=raw)
        self.protocol = protocol
        self.sha256 = sha256
        self.protocol_path = Path(protocol_path)
        self.hand = hand
        self.items = items
        self.takes = int(takes)
        self.duration = float(duration)
        self.prep = float(prep)
        self.static_s = float(static_s)
        self.retries = int(retries)
        self.still = still
        self.reviewer = reviewer
        self.view = view
        self.mock = bool(mock)
        self.note = note or ""
        self.operator = operator
        self.auto_accept = bool(auto_accept)
        self.panel = panel                     # the COPY THIS window, or None
        self.coach = bool(coach)
        self.acquire_timeout = float(acquire_timeout)
        self._coaching: Optional[dict] = None  # this attempt's, for its meta
        self.protocol_changes: List[dict] = []
        self._now: Optional[Tuple[dict, int]] = None   # (item, take) on screen
        self.entries: List[dict] = []          # session.json "takes"
        self.attempts: List[Attempt] = []
        self.accepted = {str(it["id"]): 0 for it in items}
        self.item_ids = [str(it["id"]) for it in items]   # session.json "items"
        self.started = iso_now()
        self.resumed: List[str] = []
        self.ended: Optional[str] = None
        self.commit = tool_commit()
        self._last_stamp = None
        if resume is not None:
            self._pick_up(resume[1])
        self.write_session()

    def _pick_up(self, old: dict) -> None:
        """Carry an earlier run's record of this session forward (`--resume`).

        Plan D9: one session per set is the deliverable, and the contract's
        take number counts the kept takes of an item within its session. So
        a redo continues the session it completes: the takes already kept
        are counted, the numbering carries on from them, and the earlier
        attempts stay in session.json exactly as they were written.
        """
        self.entries = list(old.get("takes") or [])
        self.started = old.get("started") or self.started
        # `load_session` let a hint-only change of the protocol file through:
        # the session now names today's bytes, and says what it was before.
        self.protocol_changes = list(old.get("protocol_changes") or [])
        if old.get("protocol_sha256") and old.get("protocol_sha256") != self.sha256:
            self.protocol_changes.append({
                "at": iso_now(), "from_sha256": old.get("protocol_sha256"),
                "to_sha256": self.sha256,
                "what": "per-grasp orientation hints added; nothing else changed"})
        earlier = [str(i) for i in (old.get("items") or [])]
        self.item_ids = earlier + [i for i in self.item_ids if i not in earlier]
        self.resumed = list(old.get("resumed") or []) + [iso_now()]
        if not self.operator:
            self.operator = old.get("operator")
        for e in self.entries:
            iid = str(e.get("item"))
            if e.get("accepted") and iid in self.accepted:
                self.accepted[iid] += 1
            self.attempts.append(Attempt(
                item=iid, take=int(e.get("take") or 0),
                attempt=int(e.get("attempt") or 0), name=str(e.get("name")),
                accepted=bool(e.get("accepted")), reason=str(e.get("reason") or ""),
                tracked_fraction=float(e.get("tracked_fraction") or 0.0),
                grab=e.get("grab_strength"), pinch=e.get("pinch_strength"),
                by=str(e.get("decided_by") or "")))

    # --- folders ---------------------------------------------------------
    def folder(self, kind: str, rejected: bool = False) -> Path:
        base = self.session_dir / REJECTED_DIR if rejected else self.session_dir
        return base / kind

    def rel(self, path: Optional[Path]) -> Optional[str]:
        return None if path is None else Path(path).relative_to(
            self.session_dir).as_posix()

    # --- session.json -------------------------------------------------------
    def session_dict(self) -> dict:
        """Contract section 3, plus how this session was run."""
        return {
            "set": SET_NAME,
            "protocol_file": repo_relative(self.protocol_path),
            "protocol_name": self.protocol.get("name"),
            "protocol_version": self.protocol.get("version"),
            "protocol_sha256": self.sha256,
            "protocol_changes": self.protocol_changes,
            "protocol_status": self.protocol.get("status"),
            "hand": self.hand,
            "operator": self.operator,
            "camera": "leap",
            "mock": self.mock,
            "glove": False,
            "started": self.started,
            "ended": self.ended,
            "resumed": self.resumed,
            "xr_trainer_calibrated_at": None,
            "seed": None,
            "rounds": None,
            "tool_commit": self.commit,
            "items": self.item_ids,
            "takes_per_item": self.takes,
            "duration_s": self.duration,
            # coached: the MAKE THE GRASP seconds; --no-coach: the countdown
            "prep_s": self.prep,
            "coach": self.coach,
            "coaching": None if not self.coach else {
                "open_band_cm": list(OPEN_BAND_CM),
                "open_max_angle_deg": OPEN_MAX_ANGLE_DEG,
                "open_hold_s": OPEN_HOLD_S,
                "open_timeout_s": self.acquire_timeout,
                "form_s": self.prep,
                "form_lost_s": FORM_LOST_S,
                "form_max_losses": FORM_MAX_LOSSES},
            "static_s": self.static_s,
            "retries": self.retries,
            "min_tracked_fraction": MIN_TRACKED,
            "still": self.still,
            "auto_accept": self.auto_accept,
            "orientation_note": self.note,
            "takes": self.entries,
        }

    def write_session(self) -> None:
        write_json(self.session_dir / "session.json", self.session_dict())

    def finish(self) -> None:
        self.ended = iso_now()
        self.write_session()

    # --- the camera window ------------------------------------------------
    def _caption(self, text: str) -> None:
        if self.view is not None:
            self.view.caption(text, band=ENVELOPE_BAND_CM)

    # --- the COPY THIS window -----------------------------------------------
    def _copy(self, status: str, seconds: Optional[float] = None,
              force: bool = False) -> None:
        """The current grasp's picture with this status, in COPY THIS."""
        if self.panel is not None and self._now is not None:
            item, n = self._now
            self.panel.show_copy(item, n, self.takes, status, seconds, force=force)

    def _copy_open(self, state: str, seconds: Optional[float] = None,
                   lost: bool = False, status: str = OPEN_STATUS,
                   force: bool = False) -> None:
        """The OPEN HAND frame for the current grasp, in COPY THIS."""
        if self.panel is not None and self._now is not None:
            item, n = self._now
            self.panel.show_open(item, n, self.takes, state, seconds, lost,
                                 status, force=force)

    # --- the coached take ---------------------------------------------------
    def _act(self, phase: Optional[str]) -> None:
        """Tell a mock that acts the take out (`CoachedActor`) where we are."""
        if hasattr(self.source, "act"):
            self.source.act(phase)

    def _watch(self, observe) -> None:
        """Drain everything queued, every hand shown to `observe` first."""
        for _ in range(MAX_DRAIN_ROUNDS):
            if self._consume(observe=observe) < 64:
                break

    def _coach(self, item: dict) -> Tuple[bool, str]:
        """OPEN HAND, then MAKE THE GRASP, until the hand has closed into
        the grasp without being lost. (True, "") or (False, the reason the
        attempt is rejected); what happened is left in `self._coaching`."""
        info = {"acquire_s": 0.0, "lost_while_forming": 0, "form_s": self.prep,
                "acquire_rounds_s": [], "acquired": [], "forming_losses": []}
        self._coaching = info
        while True:
            lost = bool(info["forming_losses"])
            t0 = time.time()
            got, others, missing = self._open_hand(lost)
            info["acquire_rounds_s"].append(round(time.time() - t0, 2))
            info["acquire_s"] = round(sum(info["acquire_rounds_s"]), 2)
            if got is None:
                info["acquire_missing"] = missing
                reason = f"no open hand acquired in {self.acquire_timeout:g} s"
                print(f"      REJECTED: {reason} ({missing})")
                return False, reason
            info["acquired"].append({
                "height_cm": round(got.height_cm, 1),
                "view_angle_deg": (None if got.angle_deg is None
                                   else round(got.angle_deg, 1)),
                "hand_label": got.lh.hand_side})
            loss = self._make_grasp(got, others)
            if loss is None:
                return True, ""
            info["forming_losses"].append(loss)
            k = info["lost_while_forming"] = len(info["forming_losses"])
            self._caption(LOST_TEXT)
            self._copy_open("", lost=True, status=LOST_STATUS, force=True)
            print(f"      {LOST_TEXT}")
            print(f"        loss {k} of {FORM_MAX_LOSSES}, "
                  f"{loss['after_s']:.1f} s into the grasp, "
                  f"{form_loss_where(loss)}")
            beep(*BEEP_LOST)
            if k >= FORM_MAX_LOSSES:
                reason = forming_reason(info["forming_losses"])
                print(f"      REJECTED: {reason}")
                return False, reason

    def _open_hand(self, lost: bool) -> Tuple[Optional[HandSeen], Set[int], str]:
        """OPEN HAND: wait for the open palm. (its newest frame, the other
        hands in view, "") once acquired, or (None, set(), what was still
        wrong) after `acquire_timeout` seconds."""
        self._act("open")
        lo, hi = OPEN_BAND_CM
        print(f"      {OPEN_HAND_TEXT}: {lo:g} to {hi:g} cm above the module, "
              "held still until the beep")
        watch = OpenHandWatch(prefer=self.hand)
        deadline = time.time() + self.acquire_timeout
        need = None
        while True:
            now = time.time()
            self._watch(lambda lh: watch.add(lh, now))
            got = watch.acquired()
            if got is not None:
                beep(*BEEP_ACQUIRED)
                return got, watch.others(int(got.lh.hand_id), now), ""
            if now >= deadline:
                return None, set(), watch.status(now)
            if watch.need(now) != need:
                need = watch.need(now)
                self._caption(f"{OPEN_HAND_TEXT}: {need}")
            self._copy_open(watch.status(now), deadline - now, lost)
            time.sleep(0.02)

    def _make_grasp(self, start: HandSeen, others: Set[int]) -> Optional[dict]:
        """MAKE THE GRASP: `prep` seconds to close the hand while it stays
        tracked. None when it did, else where it was lost (`form_loss`)."""
        label = str(self._now[0]["label"]) if self._now else ""
        # The seconds start with the high beep, the operator's cue to close.
        t0 = time.time()
        t_end = t0 + self.prep
        self._act("form")
        beep(*BEEP_MAKE)
        print(f"      {MAKE_TEXT}: {self.prep:g} s")
        watch = FormWatch(start, others)
        self._copy(MAKE_STATUS, self.prep, force=True)
        shown = None
        while True:
            now = time.time()
            self._watch(lambda lh: watch.add(lh, now))
            gone = watch.lost(now)
            if gone is not None:
                return form_loss(gone, t0)
            left = t_end - now
            if left <= 0:
                return None
            whole = math.ceil(left)
            if whole != shown:
                shown = whole
                self._caption(f"MAKE THE GRASP: {label}   {whole}s")
            self._copy(MAKE_STATUS, left)
            time.sleep(0.02)

    # --- one grasp ----------------------------------------------------------
    def announce(self, item: dict, index: int, count: int) -> None:
        print("=" * 62)
        print(f"Grasp {index}/{count}: {item['label']}   (id {item['id']})")
        if item.get("shape"):
            print(f"  shape:  {item['shape']}")
        print(f"  angle:  {orientation_text(item)}")
        source = ", ".join(str(item[k]) for k in ("source", "figure") if item.get(k))
        if source:
            print(f"  from:   {source}")
        if item.get("object_implied"):
            print("  mimed:  no object in the hand; hold the shape the object would give")
        for line in ENVELOPE:
            print(f"  {line}")
        if self.note:
            print(f"  note:   {self.note}")
        print("=" * 62)

    def run_item(self, item: dict, index: int, count: int) -> None:
        """One slot per take still missing: all of them in a new session,
        only the ones short of `takes` in a resumed one."""
        iid = str(item["id"])
        slots = self.takes - self.accepted[iid]
        if slots <= 0:
            print(f"Grasp {index}/{count}: {item['label']} already has "
                  f"{self.accepted[iid]} kept take(s); skipped\n")
            return
        self.announce(item, index, count)
        for slot in range(1, slots + 1):
            n = self.accepted[iid] + 1
            if self.run_slot(item, n, slot):
                self.accepted[iid] += 1

    def run_slot(self, item: dict, n: int, slot: int) -> bool:
        """Attempts at take `n` until one is kept or the retries run out.

        An operator's redo always gets another attempt: the retries bound
        what the automatic gate throws away, not the operator's judgement.
        """
        failed = 0
        attempt = 0
        while True:
            attempt += 1
            got = self.attempt(item, n, slot, attempt)
            if got.accepted:
                return True
            if got.by == "operator":
                print("      redo: recording the take again\n")
                continue
            failed += 1
            if failed > self.retries:
                print(f"      FAILED: take {n} of {item['id']} not kept after "
                      f"{attempt} attempt(s); moving on\n")
                return False
            print(f"      retrying ({failed}/{self.retries})\n")

    def _stamp(self) -> str:
        """This second's stamp, never the one the previous attempt used."""
        while True:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            if stamp != self._last_stamp:
                self._last_stamp = stamp
                return stamp
            time.sleep(0.05)

    def _countdown(self, label: str) -> None:
        """`prep` seconds to get into the grasp; beeps on the last three."""
        t_end = time.time() + self.prep
        shown = None
        while True:
            left = t_end - time.time()
            if left <= 0:
                break
            whole = math.ceil(left)
            if whole != shown:
                shown = whole
                self._caption(f"GET READY: {label}   {whole}s")
                if whole <= 3:
                    print(f"      {whole}...")
                    beep(660, 120)
            self._copy("GET READY", left)
            self._consume()
            time.sleep(0.02)

    def _request_still(self, path: Optional[Path], caption: str) -> None:
        if path is None:
            return
        if self.mock:
            write_mock_still(path, self.source.latest(self.hand), caption,
                             full=self.still == "full")
        elif self.view is not None:
            self.view.snapshot(path)

    def _still_result(self, path: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
        """(the still, None) or (None, why there is none), contract names."""
        if path is None:
            return None, STILL_OFF
        from leap_hand.protocol import STILL_UNKNOWN, still_status
        deadline = time.time() + (0.0 if self.mock else STILL_WAIT_S)
        while True:
            name, missing = still_status(path)
            if name or missing != STILL_UNKNOWN or time.time() >= deadline:
                break
            self._consume()
            time.sleep(0.05)
        return (path, None) if name else (None, missing)

    def attempt(self, item: dict, n: int, slot: int, attempt: int) -> Attempt:
        iid, label = str(item["id"]), str(item["label"])
        name = take_name(iid, self.hand, n, self._stamp())
        leap_path = self.folder(LEAP_DIR) / f"{name}.jsonl"
        still_path = (None if self.still == "none"
                      else self.folder(STILLS_DIR) / f"{name}.png")
        print(f"--- {label}: take {n}/{self.takes} (slot {slot}), attempt "
              f"{attempt}, {self.hand} hand ---")
        # The mock can act out a pose it knows; a grasp it does not know makes
        # it cycle, which still gives the static interval something to find.
        if hasattr(self.source, "set_pose"):
            self.source.set_pose(iid if iid in ("open_palm", "fist") else None)
        self._now = (item, n)
        self._coaching = None
        if self.coach:
            ok, why = self._coach(item)
            if not ok:
                # Nothing was recorded: the attempt is still written down,
                # under rejected/ with its reason and the coaching record.
                self._act("idle")
                now = time.time()
                return self._conclude(item, n, attempt, name, None, (now, now),
                                      NOT_RECORDED, accepted=False, reason=why,
                                      by="coach", decided_at=now)
            self._act("hold")
        else:
            self._copy("GET READY", self.prep, force=True)
            self._countdown(label)

        recorder = LeapRecorder(hz=None, pose=iid, take=n)
        t_start = t_stop = time.time()
        try:
            with self._raw_capture(leap_path.with_suffix(".lmt")):
                # Beep, drop what queued behind it, and only then open the file.
                beep(1000, 250)
                self._discard_backlog()
                recorder.start(leap_path)
                recorder._file = _KeysWriter(recorder._file, {
                    "session": self.session_dir.name, "item": iid})
                t_start = time.time()
                try:
                    self._record(recorder, t_start, label, still_path)
                finally:
                    recorder.stop()
                    t_stop = time.time()
                    if self.coach:
                        self._act("idle")
                    beep(500, 300)
                    self._copy("CHECKING THE TAKE", force=True)
        except KeyboardInterrupt:
            # Nothing recorded is deleted: an interrupted attempt is kept
            # under rejected/ with the reason, like any attempt not kept.
            self._conclude(item, n, attempt, name, None, (t_start, t_stop),
                           None, accepted=False,
                           reason="interrupted (Ctrl+C) during the take",
                           by="operator", decided_at=time.time())
            raise

        rows = read_rows(leap_path)
        summary = summarise_take(rows, t_stop - t_start, self.static_s,
                                 prefer=self.hand, min_tracked=MIN_TRACKED)
        # Every loss of the operator's hand in the take, with where the hand
        # was and why: the evidence a bare "tracked 72 %" does not carry.
        losses = take_losses(rows, summary.hand_label, t_start, t_stop)
        gate_text = take_reason(summary.gate_reason, summary.tracked_fraction,
                                MIN_TRACKED, summary.interval_losses,
                                summary.medoid_row is not None, losses.losses,
                                losses.head_s)
        still, missing = self._still_result(still_path)
        lines = self.review_lines(item, n, attempt, summary, t_start)
        for line in lines[1:]:
            print(f"      {line}")

        if not summary.passed:
            print(f"      REJECTED by the gate: {gate_text}")
            return self._conclude(item, n, attempt, name, summary,
                                  (t_start, t_stop), missing,
                                  accepted=False, reason=gate_text,
                                  by="gate", decided_at=time.time(),
                                  losses=losses, gate_text=gate_text)

        self._caption(f"REVIEW {label} take {n}: Enter keep, r redo, q quit")
        try:
            action, when, by = self.reviewer.decide(
                still, lines, pump=self._consume, screen=self.panel,
                reference=None if self.panel is None else self.panel.panel(item))
        except KeyboardInterrupt:
            self._conclude(item, n, attempt, name, summary, (t_start, t_stop),
                           missing, accepted=False,
                           reason="interrupted (Ctrl+C) at the review",
                           by="operator", decided_at=time.time(), losses=losses)
            raise
        if action == "accept":
            print(f"      KEPT ({by})\n")
            return self._conclude(item, n, attempt, name, summary,
                                  (t_start, t_stop), missing,
                                  accepted=True, reason="", by=by,
                                  decided_at=when, losses=losses)
        if action == "redo":
            return self._conclude(item, n, attempt, name, summary,
                                  (t_start, t_stop), missing,
                                  accepted=False, reason="operator redo",
                                  by="operator", decided_at=when, losses=losses)
        self._conclude(item, n, attempt, name, summary, (t_start, t_stop),
                       missing, accepted=False,
                       reason="the operator quit the session at the review",
                       by="operator", decided_at=when, losses=losses)
        raise QuitSession()

    def _record(self, recorder, t_start: float, label: str,
                still_path: Optional[Path]) -> None:
        print(f"      REC {self.duration:g} s - hold it ", end="", flush=True)
        t_end = t_start + self.duration
        midpoint = t_start + self.duration / 2.0
        next_dot = t_start + 0.5
        shown = None
        try:
            while True:
                now = time.time()
                if now >= t_end:
                    break
                self._consume(recorder)
                whole = math.ceil(t_end - now)
                if whole != shown:
                    shown = whole
                    self._caption(f"HOLD: {label}   REC {whole}s")
                self._copy("HOLD STILL", t_end - now)
                # The still at the midpoint: the hand is settled by then and
                # the take is not over, so it shows what the file holds.
                if still_path is not None and now >= midpoint:
                    self._request_still(still_path, f"{label}  REC")
                    still_path = None
                if now >= next_dot:
                    print(".", end="", flush=True)
                    next_dot += 0.5
                time.sleep(0.005)
        finally:
            print(flush=True)

    def review_lines(self, item: dict, n: int, attempt: int, s,
                     t_start: float) -> List[str]:
        """What the operator reads before deciding: the label and the numbers."""
        lines = [f"{item['label']}   take {n}/{self.takes}, attempt {attempt}"]
        lines.append(f"tracked {s.tracked_fraction * 100:.0f} %   "
                     f"id changes {s.reacquisitions} "
                     f"({len(s.interval_reacquisitions)} in the static interval, "
                     f"{len(s.interval_losses)} with the hand really gone)")
        if s.medoid_row is not None:
            lines.append(f"grab {_num(s.grab_strength)}   "
                         f"pinch {_num(s.pinch_strength)}")
            height, angle = row_height_cm(s.medoid_row), row_view_angle_deg(s.medoid_row)
            if height is not None and angle is not None:
                lines.append(f"palm {height:.0f} cm above the module, "
                             f"{angle:.0f} deg from facing the lens")
            lines.append("curls  " + "  ".join(f"{k} {v:.2f}"
                                               for k, v in s.curls.items()))
            t0, t1 = s.interval
            lines.append(f"static interval {t0 - t_start:.1f} to "
                         f"{t1 - t_start:.1f} s, summary frame at "
                         f"{s.medoid_wall_time - t_start:.1f} s")
        else:
            lines.append("no summary frame: no tracked frame in the static interval")
        return lines

    # --- the verdict ----------------------------------------------------------
    def _conclude(self, item: dict, n: int, attempt: int, name: str, summary,
                  span: Tuple[float, float], still_missing: Optional[str],
                  accepted: bool, reason: str, by: str,
                  decided_at: float, losses=None,
                  gate_text: Optional[str] = None) -> Attempt:
        """Write the keypoints and the meta, move a refused attempt, log it.

        `losses` is the take's `tracking_quality.TakeLosses` (None when the
        take never got that far) and `gate_text` the gate's reason with the
        losses named, when the gate is what refused it.
        """
        iid = str(item["id"])
        rejected = not accepted
        leap_src = self.folder(LEAP_DIR) / f"{name}.jsonl"
        # The contract's "files" keys for every set; a grasp take has no glove
        # file and no events file, and says so rather than leaving them out.
        files = {"glove": None, "events": None}

        def place(src: Path, kind: str, fname: str) -> Optional[Path]:
            if not src.is_file():
                return None
            dst = self.folder(kind, rejected) / fname
            if dst != src:
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
            return dst

        leap = place(leap_src, LEAP_DIR, f"{name}.jsonl")
        place(leap_src.with_suffix(".lmt"), LEAP_DIR, f"{name}.lmt")
        files["leap"] = self.rel(leap)
        # The still goes wherever its take goes, and so does the viewer's
        # note when it skipped the still. Whatever is on disk NOW decides,
        # so a still the viewer wrote after the wait is still found.
        from leap_hand.protocol import STILL_UNKNOWN, skipped_still_path
        still_src = self.folder(STILLS_DIR) / f"{name}.png"
        placed = place(still_src, STILLS_DIR, f"{name}.png")
        place(skipped_still_path(still_src), STILLS_DIR,
              skipped_still_path(still_src).name)
        files["still"] = self.rel(placed)
        if placed is not None:
            still_missing = None
        elif not still_missing:
            still_missing = STILL_UNKNOWN

        keypoints = None
        if summary is not None and summary.medoid_row is not None:
            keypoints = self.folder(KEYPOINTS_DIR, rejected) / f"{name}_keypoints.txt"
            keypoints.parent.mkdir(parents=True, exist_ok=True)
            block = prof_exporter().frame_block(row_frame(summary.medoid_row),
                                                summary.medoid_wall_time)
            keypoints.write_text(block + "\n", encoding="utf-8")
        files["keypoints"] = self.rel(keypoints)
        meta_path = self.folder(META_DIR, rejected) / f"{name}.json"
        files["meta"] = self.rel(meta_path)
        note = self.session_dir / REJECTED_DIR / f"{name}.reason.txt"
        if rejected:
            files["reason"] = self.rel(note)

        meta = self.meta_dict(item, n, attempt, name, summary, span, files,
                              still_missing, accepted, reason, by, decided_at,
                              losses=losses, gate_text=gate_text)
        write_json(meta_path, meta)
        if rejected:
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_text(
                f"{reason}\n"
                f"decided_by={by}\n"
                f"decided_at={decided_at:.3f}  ({iso_now(decided_at)})\n"
                f"item={iid} take={n} attempt={attempt}\n"
                f"tracked_fraction={meta['tracked_fraction']}\n",
                encoding="utf-8")

        self.entries.append({
            "item": iid, "take": n, "attempt": attempt, "name": name,
            "accepted": accepted, "reason": reason, "decided_at": decided_at,
            "decided_by": by, "tracked_fraction": meta["tracked_fraction"],
            "grab_strength": meta["grab_strength"],
            "pinch_strength": meta["pinch_strength"], "files": files})
        self.write_session()
        got = Attempt(item=iid, take=n, attempt=attempt, name=name,
                      accepted=accepted, reason=reason,
                      tracked_fraction=meta["tracked_fraction"],
                      grab=meta["grab_strength"], pinch=meta["pinch_strength"],
                      by=by)
        self.attempts.append(got)
        return got

    def meta_dict(self, item: dict, n: int, attempt: int, name: str, s,
                  span: Tuple[float, float], files: dict,
                  still_missing: Optional[str], accepted: bool, reason: str,
                  by: str, decided_at: float, losses=None,
                  gate_text: Optional[str] = None) -> dict:
        """Contract section 7, every field, then what else is known."""
        row = None if s is None else s.medoid_row
        interval = None if s is None or s.interval is None else [
            round(s.interval[0], 6), round(s.interval[1], 6)]
        height = None if row is None else row_height_cm(row)
        angle = None if row is None else row_view_angle_deg(row)
        return {
            "item": str(item["id"]),
            "take": n,
            "hand": self.hand,
            "frames": 0 if s is None else s.frames,
            "tracked_fraction": 0.0 if s is None else round(s.tracked_fraction, 4),
            "reacquisitions": 0 if s is None else s.reacquisitions,
            "tracker_hand_labels": {} if s is None else s.labels,
            "static_interval": interval,
            "medoid_wall_time": None if row is None else s.medoid_wall_time,
            "grab_strength": None if s is None else s.grab_strength,
            "pinch_strength": None if s is None else s.pinch_strength,
            "curls": None if s is None else s.curls,
            "orientation_note": self.note,
            "accepted": accepted,
            "reason": reason,
            "decided_at": decided_at,
            # --- beyond the contract's list
            "decided_by": by,
            "name": name,
            "session": self.session_dir.name,
            "attempt": attempt,
            "label": item.get("label"),
            "shape": item.get("shape"),
            "source": item.get("source"),
            "figure": item.get("figure"),
            "object_implied": item.get("object_implied"),
            "protocol_name": self.protocol.get("name"),
            "protocol_version": self.protocol.get("version"),
            "mock": self.mock,
            "take_start": round(span[0], 6),
            "take_end": round(span[1], 6),
            "duration_s": round(span[1] - span[0], 3),
            "static_s": self.static_s,
            "gate": {
                "passed": bool(s is not None and s.passed),
                "reason": (gate_text if gate_text is not None
                           else "" if s is None else s.gate_reason),
                "min_tracked_fraction": MIN_TRACKED,
                "tracked_frames": None if s is None else s.tracked_frames,
                "expected_frames": None if s is None else s.expected_frames,
                # real losses (hand gone longer than loss_gap_s) inside the
                # static interval: the number the gate judges on; id changes
                # with no hole in the data are counted beside it, not failed
                "reacquisitions_in_static_interval": (
                    None if s is None else len(s.interval_losses)),
                "id_changes_in_static_interval": (
                    None if s is None else len(s.interval_reacquisitions)),
                # Every loss of the operator's hand (gone longer than
                # loss_gap_s, or a new hand id), kept take or not, with where
                # the hand was, the likely causes and the fix; start_s is from
                # the start of the take. See leap_hand.tracking_quality.
                "loss_gap_s": LOSS_GAP_S if losses is None else losses.gap_s,
                "losses": [] if losses is None else losses.to_list(),
            },
            # The tracker label whose frames were taken as the operator's
            # hand (the one with the most frames), and the summary frame's
            # own identity: its line in the leap file, counted from 0.
            "operator_hand_label": None if s is None else s.hand_label,
            "medoid_line": None if s is None else s.medoid_index,
            "medoid_frame_id": None if row is None else row.get("frame_id"),
            "medoid_hand_id": None if row is None else row.get("hand_id"),
            # The orientation the summary frame was actually held at (plan
            # D3 asks for the rotation used): palm height above the module,
            # and the angle between the palm normal and the ray to the lens,
            # 0 = palm square to the lens, 90 = edge-on.
            "palm_height_cm": None if height is None else round(height, 1),
            "view_angle_deg": None if angle is None else round(angle, 1),
            # The orientation the operator was asked for (the item's hint),
            # and how the hand got into the grasp: seconds to acquire the
            # open hand (every OPEN HAND of the attempt summed), how often it
            # was lost while forming the grasp and where. None with --no-coach.
            "orientation_hint": orientation_text(item),
            "coaching": self._coaching,
            "still": files.get("still"),
            "still_missing_reason": still_missing,
            "files": dict(files),
        }

    # --- the end ----------------------------------------------------------------
    def print_table(self) -> None:
        print("=" * 62)
        if not self.attempts:
            print("Nothing recorded.")
        else:
            kept = sum(1 for a in self.attempts if a.accepted)
            print(f"Grasp session ({self.hand} hand): {kept} take(s) kept, "
                  f"{len(self.attempts) - kept} attempt(s) not kept")
            print(f"  {'item':<22} {'take':>4}  {'accepted':<8} {'tracked':>7} "
                  f"{'grab':>5} {'pinch':>5}")
            for a in self.attempts:
                grab, pinch = f"{_num(a.grab):>5}", f"{_num(a.pinch):>5}"
                print(f"  {a.item:<22} {a.take:>4}  {'yes' if a.accepted else 'no':<8} "
                      f"{a.tracked_fraction * 100:6.0f}% {grab} {pinch}"
                      + ("" if a.accepted else f"   {a.reason}"))
            short = {i: n for i, n in self.accepted.items() if n < self.takes}
            if short:
                print(f"\n  Short of {self.takes} kept takes: " + ", ".join(
                    f"{i} ({n}/{self.takes})" for i, n in short.items()))
                print("  Record the missing takes into this same session:")
                print(f"    {self.resume_command()}")
        print(f"\nFolder: {self.session_dir}")

    def resume_command(self) -> str:
        """The command that records what this session is still missing."""
        protocol = repo_relative(self.protocol_path).replace("/", os.sep)
        return (rf".venv\Scripts\python.exe scripts\leap\record_poses.py "
                f"{'--mock ' if self.mock else ''}--protocol {protocol} "
                f"--hand {self.hand} --resume \"{self.session_dir}\""
                f"{'' if self.coach else ' --no-coach'}")


def make_mock_source(dropout: float = 0.0, coach: bool = True,
                     lose_forming: int = 0):
    """The synthetic camera behind `--mock`, started.

    Coached, the hand acts the take out (`leap_hand.mock.CoachedActor`):
    an open palm at OPEN HAND, closing at MAKE THE GRASP (lost on the first
    `lose_forming` tries of every attempt), the grasp held at HOLD STILL;
    `dropout` then drops frames during the hold only, so a rehearsal of the
    gate's rejection still reaches the take. With `coach` off it is the
    plain mock it always was. Either way it is clean otherwise: the default
    mock drops 20 frames in 300 and changes hand id every 5 s, so a 1 s
    rehearsal take would fail the gate at random.
    """
    from leap_hand.mock import CoachedActor, MockLeapStream
    if coach:
        source = MockLeapStream(dropout_every=0, reacquire_every=0,
                                script=CoachedActor(lose_forming=lose_forming,
                                                    hold_dropout=dropout))
    elif dropout > 0:
        source = MockLeapStream(dropout_every=90, reacquire_every=0,
                                dropout_frames=max(1, min(89, round(90 * dropout))))
    else:
        source = MockLeapStream(dropout_every=0, reacquire_every=0)
    # Skip the mock hand past its first 0.3 s of visibility, which the
    # recorder drops as settling: a live hand has been in view that long
    # by the time the stream wait is over, and with --prep 0 the first
    # take would otherwise start on a hand too young to record.
    source.generate(int(math.ceil(MIN_VISIBLE_TIME_US / 1e6 * source.hz)) + 9)
    source.start()
    return source


def run_protocol(args, parser) -> None:
    """`--protocol`: Set A, one session folder per run."""
    if args.hand is None:
        parser.error("--protocol needs --hand left|right (the operator's hand)")
    if args.poses is not None:
        parser.error("--poses does not apply with --protocol; use --items")
    if args.hz:
        parser.error("a protocol take keeps every frame (plan D4); drop --hz")
    if args.raw and args.mock:
        raise SystemExit("--raw needs a live camera: there is no LeapC stream "
                         "behind --mock")
    retries = PROTOCOL_RETRIES if args.retries is None else args.retries
    if retries < 0:
        parser.error("--retries cannot be negative")
    if args.mock_dropout is not None and not args.mock:
        parser.error("--mock-dropout only applies with --mock")
    coach = not args.no_coach
    if args.mock_lose_forming is not None:
        if not args.mock:
            parser.error("--mock-lose-forming only applies with --mock")
        if not coach:
            parser.error("--mock-lose-forming acts out the coached take; "
                         "drop --no-coach")
        if args.mock_lose_forming < 0:
            parser.error("--mock-lose-forming cannot be negative")

    protocol, sha = load_protocol(args.protocol)
    resume = None
    # Each setting comes from the command line, else from the session being
    # resumed (a session keeps one timing throughout), else from the file.
    # Coached, the preparation is the MAKE THE GRASP countdown, FORM_S: the
    # file's prep_s is the plain countdown's, which also had to bring the
    # hand over the module, the job OPEN HAND now does.
    base = {"takes_per_item": protocol["takes_per_item"],
            "duration_s": protocol["duration_s"],
            "prep_s": FORM_S if coach else protocol["prep_s"],
            "static_s": protocol.get("static_s", DEFAULT_STATIC_S),
            "still": "hand", "items": None}
    if args.resume is not None:
        old = load_session(args.resume, args.hand, sha, bool(args.mock),
                           raw=Path(args.protocol).read_bytes())
        resume = (Path(args.resume), old)
        for key in base:
            # A plain countdown's seconds are not a forming time, nor the
            # other way round: prep_s carries over only within one mode.
            if key == "prep_s" and bool(old.get("coach")) != coach:
                continue
            if old.get(key) is not None:
                base[key] = old[key]
    items = select_items(protocol, args.items if args.items else (
        ",".join(base["items"]) if base["items"] else None))
    takes = int(base["takes_per_item"]) if args.takes is None else args.takes
    duration = float(base["duration_s"]) if args.duration is None else args.duration
    prep = float(base["prep_s"]) if args.prep is None else args.prep
    static_s = float(base["static_s"] if args.static_s is None else args.static_s)
    still = args.still or base["still"]
    if takes < 1 or duration <= 0 or prep < 0 or static_s <= 0:
        parser.error("takes must be at least 1, duration and static-s above 0, "
                     "prep 0 or more")

    if args.mock:
        # --mock-dropout puts a dropout back on purpose, to rehearse the
        # gate's rejection; --mock-lose-forming rehearses LOST YOU.
        source = make_mock_source(args.mock_dropout or 0.0, coach,
                                  args.mock_lose_forming or 0)
    else:
        try:
            source = open_stream(mode=args.mode)
        except LeapUnavailable as e:
            raise SystemExit(f"\nNo live tracking: {e}\n")

    view = None
    if not args.mock:
        view = _still_view_class()(still="full" if still == "full" else "hand",
                                   hand=None, band=ENVELOPE_BAND_CM).start()
    out_root = ((MOCK_PROTOCOL_OUT if args.mock else PROTOCOL_OUT)
                if args.out_dir is None else args.out_dir)
    reviewer = Reviewer(auto=args.auto_accept)
    panel = None if args.no_panel else CopyWindow(protocol, args.protocol, items)
    session = None
    try:
        session = ProtocolSession(
            source, out_root, protocol, sha, args.protocol, args.hand, items,
            takes, duration, prep, static_s, retries, still, reviewer,
            view=view, mock=args.mock, raw=args.raw, note=args.note or "",
            operator=args.operator, auto_accept=args.auto_accept,
            resume=resume, panel=panel, coach=coach,
            acquire_timeout=OPEN_HAND_TIMEOUT_S)

        # coached: about 2 s to show the open hand, plus the beeps
        eta = len(items) * takes * (prep + duration + (3.0 if coach else 1.0))
        print("=" * 62)
        print(f"Grasp protocol (Set A): {protocol.get('name')} v"
              f"{protocol.get('version')}, {len(items)} grasp(s) x {takes} "
              f"take(s) x {duration:g} s  (~{eta / 60:.1f} min)")
        print(f"  hand:    {args.hand} (the operator's; the tracker's own label "
              "is recorded, not trusted)")
        if coach:
            print(f"  coach:   OPEN HAND ({OPEN_BAND_CM[0]:g} to "
                  f"{OPEN_BAND_CM[1]:g} cm up, palm within "
                  f"{OPEN_MAX_ANGLE_DEG:g} degrees of the lens, "
                  f"{OPEN_HOLD_S:g} s; {OPEN_HAND_TIMEOUT_S:g} s to get "
                  f"there), MAKE THE GRASP {prep:g} s, HOLD STILL "
                  f"{duration:g} s")
        else:
            print(f"  coach:   off (--no-coach): a {prep:g} s GET READY "
                  "countdown")
        print(f"  gate:    {MIN_TRACKED * 100:.0f} % of frames tracked and no "
              f"re-acquisition in the {static_s:g} s static interval; "
              f"{retries} retries")
        print("  review:  " + ("every take that passes the gate is kept "
                               "(--auto-accept)" if args.auto_accept else REVIEW_KEYS))
        print("  picture: " + ("off (--no-panel)" if panel is None
                               else panel.describe()))
        print(f"  folder:  {session.session_dir}")
        if resume is not None:
            kept = sum(session.accepted.values())
            print(f"  resume:  {kept} take(s) already kept here; recording only "
                  "the missing ones, numbering carries on")
        if protocol.get("status"):
            print(f"  status:  {protocol['status']}")
        if args.mock:
            print("  Mock mode: synthetic hands, no camera; the stills say MOCK.")
        print("=" * 62 + "\n")

        if not args.mock:
            session.wait_for_stream()
        for i, item in enumerate(items, 1):
            session.run_item(item, i, len(items))
    except KeyboardInterrupt:
        print("\nInterrupted - the takes kept so far are in session.json.")
    except QuitSession:
        print("\nSession ended at the review (q).")
    finally:
        source.stop()
        if view is not None:
            view.close()
        if panel is not None:
            panel.close()
        if session is not None:
            session.finish()
            session.print_table()
            if not args.no_open:
                try:
                    os.startfile(str(session.session_dir))
                except (AttributeError, OSError):
                    pass


def main() -> None:
    p = argparse.ArgumentParser(
        description="Guided, hands-free pose recording with the Ultraleap camera. "
                    "With --protocol, the professor's grasp set (Set A).")
    p.add_argument("--poses", default=None,
                   help=f"comma-separated pose names (default: {','.join(DEFAULT_POSES)})")
    p.add_argument("--takes", type=int, default=None,
                   help=f"repetitions per pose (default: {LEGACY_TAKES}; with "
                        "--protocol, the file's takes_per_item)")
    p.add_argument("--duration", type=float, default=None,
                   help=f"seconds recorded per take (default: {LEGACY_DURATION:g}; "
                        "with --protocol, the file's duration_s)")
    p.add_argument("--prep", type=float, default=None,
                   help=f"seconds to get into the pose before each take (default: "
                        f"{LEGACY_PREP:g}; with --protocol, the MAKE THE GRASP "
                        f"countdown, {FORM_S:g}, or with --no-coach the file's "
                        "prep_s)")
    p.add_argument("--hz", type=float, default=None,
                   help=f"frames saved per second (default: {LEGACY_HZ:g}; 0 = keep "
                        "every frame). A protocol take always keeps every frame")
    p.add_argument("--out-dir", type=Path, default=None,
                   help=f"output folder (default: {LEGACY_OUT}; with --protocol, "
                        f"{PROTOCOL_OUT}, one session folder inside it per run; "
                        f"{MOCK_PROTOCOL_OUT} with --mock)")
    p.add_argument("--mock", action="store_true",
                   help="dry-run with synthetic hands; no camera needed")
    p.add_argument("--mode", default="desktop",
                   choices=("desktop", "hmd", "screentop"))
    p.add_argument("--raw", action="store_true",
                   help="also write LeapC's own .lmt recording beside each take")
    g = p.add_argument_group("the professor's grasp set (Set A)")
    g.add_argument("--protocol", type=Path, default=None,
                   help="protocol file, e.g. protocols/grasps.json")
    g.add_argument("--hand", choices=("left", "right"), default=None,
                   help="the operator's hand; required with --protocol")
    g.add_argument("--items", default=None,
                   help="comma-separated item ids to record, a subset of the file's")
    g.add_argument("--resume", type=Path, default=None,
                   help="an earlier session folder to add the missing takes to, "
                        "continuing its numbering (the end table prints this command)")
    g.add_argument("--still", choices=STILL_MODES, default=None,
                   help="per-take still: cropped to the hand (default), the whole "
                        "frame, or none")
    g.add_argument("--retries", type=int, default=None,
                   help=f"new attempts after the gate rejects one (default: "
                        f"{PROTOCOL_RETRIES})")
    g.add_argument("--static-s", type=float, default=None,
                   help="length of the static interval in seconds (default: the "
                        "file's static_s, else 2)")
    g.add_argument("--auto-accept", action="store_true",
                   help="keep every take that passes the gate without asking")
    g.add_argument("--no-panel", action="store_true",
                   help="no COPY THIS window with the paper's picture of each "
                        "grasp (headless runs); the review opens its own window")
    g.add_argument("--no-open", action="store_true",
                   help="do not open the session folder at the end")
    g.add_argument("--note", default=None,
                   help="orientation note stored in every take's meta, e.g. "
                        "'palm turned 30 degrees toward the lens'")
    g.add_argument("--operator", default=None,
                   help="who recorded the session, for session.json")
    g.add_argument("--no-coach", action="store_true",
                   help="the plain GET READY countdown instead of the coached "
                        "take (OPEN HAND, then MAKE THE GRASP, then HOLD STILL)")
    g.add_argument("--mock-dropout", type=float, default=None,
                   help="with --mock: fraction of frames the mock drops, to "
                        "rehearse the gate's rejection (e.g. 0.3); coached, "
                        "only during the take")
    g.add_argument("--mock-lose-forming", type=int, default=None, metavar="N",
                   help="with --mock: lose the hand while it closes into the "
                        "grasp on the first N tries of every attempt, to "
                        "rehearse LOST YOU (3 or more rejects every attempt)")
    args = p.parse_args()

    if args.protocol is not None:
        run_protocol(args, p)
        return
    protocol_only = [flag for flag, value in (
        ("--hand", args.hand), ("--items", args.items), ("--still", args.still),
        ("--resume", args.resume),
        ("--retries", args.retries), ("--static-s", args.static_s),
        ("--auto-accept", args.auto_accept), ("--no-open", args.no_open),
        ("--no-panel", args.no_panel),
        ("--note", args.note), ("--operator", args.operator),
        ("--no-coach", args.no_coach),
        ("--mock-dropout", args.mock_dropout),
        ("--mock-lose-forming", args.mock_lose_forming))
        if value not in (None, False)]
    if protocol_only:
        p.error(f"{', '.join(protocol_only)} only apply with --protocol")
    args.poses = ",".join(DEFAULT_POSES) if args.poses is None else args.poses
    args.takes = LEGACY_TAKES if args.takes is None else args.takes
    args.duration = LEGACY_DURATION if args.duration is None else args.duration
    args.prep = LEGACY_PREP if args.prep is None else args.prep
    args.hz = LEGACY_HZ if args.hz is None else args.hz
    args.out_dir = LEGACY_OUT if args.out_dir is None else args.out_dir

    poses = [slugify(x) for x in args.poses.split(",") if slugify(x)]
    if not poses:
        raise SystemExit("no poses given")
    if args.raw and args.mock:
        raise SystemExit("--raw needs a live camera: there is no LeapC stream "
                         "behind --mock")

    total = len(poses) * args.takes
    eta = total * (args.prep + args.duration)
    print("=" * 62)
    print(f"Leap pose session: {len(poses)} poses x {args.takes} takes "
          f"x {args.duration:g} s  (~{eta / 60:.1f} min)")
    print(f"  poses: {', '.join(poses)}")
    print(f"  rate:  {'every frame' if not args.hz else f'{args.hz:g} frames/s'}"
          f"   output: {args.out_dir}")
    print("  Once started it runs itself; beeps mark record start/stop.")
    print("=" * 62 + "\n")

    try:
        source = open_stream(mock=args.mock, mode=args.mode)
    except LeapUnavailable as e:
        raise SystemExit(f"\nNo live tracking: {e}\n")
    if args.mock:
        print("Mock mode: synthetic hands (no camera needed).\n")

    session = Session(source, hz=args.hz or None, out_dir=args.out_dir, raw=args.raw)
    try:
        if not args.mock:
            session.wait_for_stream()
        for i, pose in enumerate(poses, 1):
            for take in range(1, args.takes + 1):
                session.run_take(pose, take, args.takes, i, len(poses),
                                 args.duration, args.prep)
    except KeyboardInterrupt:
        print("\nInterrupted - keeping the takes recorded so far.")
    finally:
        source.stop()
        session.print_summary()


if __name__ == "__main__":
    main()
