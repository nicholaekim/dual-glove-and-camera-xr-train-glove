"""What the professor's hand-in folder says about Set A, the static grasps.

`scripts/package_professor_set.py` decides which files go into the folder
and writes them. This module works out what the folder says: one pass over
the session's accepted takes (`collect`), and from it the four tables and
the README.

  grasps\\grasps_summary.csv      one row per recording (`summary_rows`)
  grasps\\grasp_types_audit.csv   one row per grasp type of the protocol,
                                  recorded or not (`audit_rows`)
  grasps\\joint_angles_24.csv     one row per recording: the paper's 24
                                  angles and the two other readings of the
                                  same bones (`angle_rows`)
  transitions\\pose_transitions.csv  one row per recorded change of pose
                                  (`transition_rows`)
  README.txt                      twenty numbered sections (`readme_text`)

and what the folder says about each take's infrared still: where it is,
when it was written, whether the recorded hand's skeleton is drawn on it
(the build cuts it to that hand, `leap_hand.grasp_images`), and what was
read in it by eye (`protocols/grasps_still_notes.json`, loaded once per
build; its readings are notes, never measurements).

Three rules the text is written under, because the folder is read by someone
who was not there:

  Nothing is taken from a count of lines. Durations are differences of
  timestamps (`leap_hand.grasp_audit`).

  Nothing is said that the build did not measure. Every count, range and
  date in the README is computed here from the takes; a sentence that
  depends on the data (every grasp has a recording, another hand was in
  view, a transition was captured) is written from the data and changes
  with it.

  Nothing stands in for a measurement that was not made. A take without a
  recorded change into the grasp has no row in the transition table, and a
  recorded change that does not pass the timing rule has a row that says
  why and no duration (`leap_hand.transition.transition_timing`).
"""
import csv
import json
import math
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cam_hand import protocol_check as pc
from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_NAMES

from . import grasp_images as gi
from . import transition as tr
from .grasp_audit import (BACK_OF_HAND_NOTE, BY_LABEL, EDGE_S, FOLLOWED,
                          NOT_THE_POSE_MM, OTHER_HAND_MIN_SHARE,
                          SHAPE_CHANGE_MM, SHORT_BY_S, STATIC_MOVING_MM_S,
                          TakeFacts, operator_lines, spread, take_facts)
from .static_interval import (DEFAULT_STATIC_S, FOLLOW_RADIUS_M, LOSS_GAP_S,
                              MIN_LABEL_SHARE, MIN_TRACKED_FRACTION)

GRASPS_DIR = "grasps"
TRANSITIONS_DIR = "transitions"
JOINT_FRAMES_DIR = "joint_frames"          # scripts/joint_frames_view.py --session
SUMMARY_FILE = "grasps_summary.csv"
AUDIT_FILE = "grasp_types_audit.csv"
ANGLES_FILE = "joint_angles_24.csv"
TRANSITIONS_FILE = "pose_transitions.csv"
VALIDATION_FILE = "validation_report.txt"
IMAGES_DIR = "images"                      # grasps\images\<stem>.png
IMAGES_PDF = "grasp_images.pdf"            # grasps\images\, one page per grasp
KEY_CSV = "joint_key.csv"                  # grasps\joint_frames\ (xr_hand.joint_key)
KEY_PDF = "joint_key.pdf"
OPEN_HAND = "open_hand"                    # `from_grasp` of a coached take
# at the head of the notes of a mock transition take's row, and of its line in
# README section 17 (`grasp_validate` fails a folder with such a row unless
# it is told allow_mock)
MOCK_NOTE = "MOCK (synthetic hand)"
# The reason kinds of the recorder's rules of 2026-10-07 (`reason_kind`).
LOST_NOT_FOUND_KIND = "hand lost while forming the grasp, not found again in time"
RESET_LIMIT_KIND = "reset to OPEN HAND past the limit before the GO beep"
# A coached take kept after the hand was lost while the grasp was formed and
# found again (`coaching.forming_lost_count`, recorder rules of 2026-10-07):
# its note, in the summary's notes column and README section 15.
FORMING_LOST_NOTE = ("the hand was lost {n} {times} while forming the grasp "
                     "and found again before the hold; the forming file is "
                     "not timed")
READY, INCOMPLETE, COMPLETE = "READY", "INCOMPLETE", "COMPLETE"
WIDTH = 96
REPO_ROOT = Path(__file__).resolve().parents[2]    # protocols\grasps_still_notes.json
# The two hands' finger curl (`grasp_audit.other_hand_curl`) this close
# count as alike in README section 16.
CURL_ALIKE_DEG = 30.0

CURL_COLUMNS = [f"curl_{f}" for f in pc.FINGERS]
SUMMARY_COLUMNS = [
    "grasp_name", "grasp_label", "source_paper", "source_figure", "hand",
    "take", "raw_jsonl_file", "representative_keypoints_file",
    "all_frames_file", "joint_frames_file", "image_file",
    "transition_capture_file",
    "sensor", "recording_duration_s", "camera_clock_duration_s",
    "recorder_window_s", "unique_frame_count", "hand_frame_count",
    "jsonl_row_count", "frame_rate_hz", "tracked_percent", "largest_gap_s",
    "static_interval_start_s", "static_interval_end_s",
    "static_interval_duration_s", "static_interval_mean_speed_mm_s",
    "static_interval_pose_speed_mm_s",
    "static_interval_shape_rms_mm", "static_interval_shape_rms_max_mm",
    "static_interval_first_frame_id", "static_interval_last_frame_id",
    "representative_frame_index",
    "representative_frame_id", "representative_frame_time_s",
    "take_shape_rms_max_mm", "frames_within_5mm_percent",
    "shape_rms_start_mm", "shape_rms_end_mm",
    "other_hand_detected", "other_hand_frame_fraction",
    "other_hand_median_distance_cm", "other_hand_finger_curl_minus_own_deg",
    "hand_selection", "hand_ids",
    "tracker_label_on_representative_frame", "tracker_id_changes",
    "identity_flag", "status", "decided_by",
    "reacquisitions_in_static_interval", "reacquisitions", "grab_strength",
    "pinch_strength"] + CURL_COLUMNS + [
    "palm_height_cm", "view_angle_deg", "index_metacarpal_mm",
    "index_metacarpal_range_mm", "first_wall_time",
    "static_interval_start", "static_interval_end", "medoid_wall_time",
    "orientation_note", "session", "recorder_take_name", "notes"]
AUDIT_COLUMNS = [
    "grasp_name", "grasp_label", "source_paper", "source_figure",
    "same_configuration_as", "accepted_recordings", "takes_available",
    "target_takes", "missing_takes", "representative_keypoints_files",
    "all_frames_file", "all_frames_takes", "all_frames_frame_count",
    "joint_frame_xyz_files", "joint_angle_rows", "transition_captures",
    "valid_transitions", "questionable_transitions",
    "recordings_without_a_note", "recordings_with_an_image",
    "between_take_max_spread_deg",
    "between_take_max_spread_angle", "attempts_not_kept",
    "open_hand_timeouts", "rejected_recordings",
    "most_common_reject_reason",
    "static_pose_status", "multi_take_status"]
# The six of the paper's 24 a camera take does not give are empty in
# joint_angles_24.csv. Their numbers, which are real measurements of
# something else (where a metacarpal points, seen from the wrist joint or
# from the palm joint), are in columns named for what they are:
# column -> the key of `Take.angles` it is filled from.
NOT_GIVEN = jf.CAMERA_NOT_GIVEN
METACARPAL_COLUMNS: Dict[str, str] = {
    "T_metacarpal_flex_wrist_ref": "T_TMC_fe",
    "T_metacarpal_abd_wrist_ref": "T_TMC_aa",
    "I_metacarpal_flex_wrist_ref": "I_CMC",
    "M_metacarpal_flex_wrist_ref": "M_CMC",
    "R_metacarpal_flex_wrist_ref": "R_CMC",
    "L_metacarpal_flex_wrist_ref": "L_CMC",
    "T_metacarpal_flex_palm_ref": "T_TMC_fe_palm_ref",
    "T_metacarpal_abd_palm_ref": "T_TMC_aa_palm_ref",
    "I_metacarpal_flex_palm_ref": "I_CMC_palm_ref",
    "M_metacarpal_flex_palm_ref": "M_CMC_palm_ref",
    "R_metacarpal_flex_palm_ref": "R_CMC_palm_ref",
    "L_metacarpal_flex_palm_ref": "L_CMC_palm_ref",
}
GIVEN_NAMES = [n for n in jf.DOF24_NAMES if n not in NOT_GIVEN]
ANGLE_COLUMNS = (["grasp_name", "hand", "take", "joint_frames_file",
                  "representative_frame_index"] + jf.DOF24_NAMES
                 + ["forearm_to_palm_deg"] + list(METACARPAL_COLUMNS)
                 + jf.MCP_PAPER_NAMES)
TRANSITION_COLUMNS = [
    "participant_id", "hand", "from_grasp", "to_grasp", "take",
    "recording_file", "cue_wall_time", "cue_time_s",
    "movement_start_time_s",
    "settled_time_s", "transition_duration_s", "reaction_time_s",
    "cue_to_settled_s", "peak_speed_mm_s", "speed_at_cue_mm_s",
    "pose_change_mm", "hand_speed_at_cue_mm_s", "hand_peak_speed_mm_s",
    "last_movement_time_s", "movement_start_method", "settled_method",
    "validity", "notes", "frames", "hand_ids", "first_frame_capture_time",
    "analysed_until_s", "session", "static_take_file"]


# --- one pass over the session ----------------------------------------------
@dataclass
class Take:
    """One accepted take and everything the folder says about it."""

    ref: pc.TakeRef
    stem: str                              # <grasp>_<hand>_take<N>
    cfg: dict                              # the protocol's entry for the grasp
    meta: dict
    facts: TakeFacts
    keypoints: Optional[Path] = None       # the recorder's summary-frame file
    joint_frames: Optional[Path] = None    # <session>/joint_frames/<take>.csv
    joint_frames_match: Optional[bool] = None   # that CSV is the summary frame's table
    angles: Optional[Dict[str, float]] = None   # the 24, the 6 and the 8
    forearm_to_palm_deg: Optional[float] = None
    forming: Optional[Path] = None         # the recorded change into the grasp
    timing: Optional[tr.Timing] = None
    palm_cm: Optional[Tuple[float, float, float]] = None      # median, camera frame
    other_palm_cm: Optional[Tuple[float, float, float]] = None    # the other hand's
    other_palm_x_cm: Optional[float] = None
    wrist_minus_palm_z_cm: Optional[float] = None
    palm_back_up: Optional[bool] = None    # the palm joint's +y points up (+Y)
    # on the summary frame: the angle between the thumb metacarpal's own
    # flexion axis (+x) and the palm's, and the palm against the wrist joint
    # (flexion: the bend of the wrist)
    thumb_axis_deg: Optional[float] = None
    thumb_axis_wrist_deg: Optional[float] = None   # the same against the wrist's +x
    wrist_bend_deg: Optional[float] = None
    table: Optional[List[dict]] = None     # the 26 rows of the summary frame
    # the same frame's chained joints against its abs26, largest difference
    # along an axis in mm: the hand is in every line twice
    chain_gap_mm: Optional[float] = None
    # over the phalanx rows of the summary frame's table (every row but the
    # wrist, the palm, the metacarpals and the tips): the largest twist, and
    # the largest distance of a twist from -abd x tan(flex / 2)
    phalanx_twist_deg: Optional[float] = None
    phalanx_twist_residual_deg: Optional[float] = None
    bad_lines: int = 0
    # the recorder's infrared still of the take (`files.still`), the time
    # its file was written after the recorder's take_start (file times),
    # whether the recorded hand's skeleton is drawn on it (the build cuts it
    # to that hand), what the still notes read in it by eye, and whether
    # they say it was read at all (`still_read`, set in `collect`)
    still: Optional[Path] = None
    still_offset_s: Optional[float] = None
    still_has_hand: bool = False
    still_note_hand: str = ""
    still_note_other: str = ""
    still_unclear: bool = False
    still_read: bool = False
    picture_notes: Dict[str, str] = field(default_factory=dict)  # `picture_notes`
    # the hand lost while forming and found again (`forming_lost_notes`)
    forming_notes: List[str] = field(default_factory=list)

    @property
    def item(self) -> str:
        return self.ref.item

    @property
    def waited_rules(self) -> bool:
        """Recorded under the rules in which a hand lost while forming is
        waited for (`waited_for_rules`)."""
        return waited_for_rules(self.meta)

    @property
    def forming_lost(self) -> int:
        """Times the hand was lost while forming the grasp and found again
        before the hold (0 under the earlier rules, where that rejected)."""
        return forming_lost_count(self.meta)

    @property
    def n(self) -> int:
        return int(self.ref.take or 0)

    @property
    def coached(self) -> bool:
        return isinstance(self.meta.get("coaching"), dict)

    @property
    def recorded_day(self) -> Optional[str]:
        t = self.facts.first_wall_time
        return None if t is None else time.strftime("%Y-%m-%d",
                                                    time.localtime(t))


@dataclass
class Move:
    """One recorded change of pose, as the transition table lists it: a
    coached take's forming file (from the open hand into the grasp), or a
    take of `scripts/leap/record_transitions.py` (between two grasps)."""

    from_grasp: str
    to_grasp: str
    take: Optional[int]
    hand: str
    source: Path                           # the recording, in its session
    dest: str                              # its place in the hand-in folder
    timing: tr.Timing
    session: str
    static_take_file: Optional[str] = None     # the take it led into
    coached: bool = False                  # a coached grasp take's forming file
    ready_step: bool = False               # held still before the cue, watched
    context: str = ""                      # what a reader must know about how it came about
    # the same recording timed at twice and at half the two speed
    # thresholds: (start speed, settle speed, timing)
    alternatives: List[Tuple[float, float, tr.Timing]] = field(
        default_factory=list)
    # and with the hand laid on the palm's joints without the WRIST
    no_wrist: Optional[tr.Timing] = None
    # a take of a mock transition session (synthetic hand, recorded with
    # --mock), in the folder only because the build was told --allow-mock
    mock: bool = False

    @property
    def own_notes(self) -> str:
        """The timing's own note and the context."""
        return "; ".join(x for x in (self.timing.notes, self.context) if x)

    @property
    def notes(self) -> str:
        """The notes column of the table: `own_notes`, after `MOCK_NOTE`
        for a mock take, so a synthetic row never reads like a real one."""
        if not self.mock:
            return self.own_notes
        return (f"{MOCK_NOTE}: {self.own_notes}" if self.own_notes
                else MOCK_NOTE)

    @property
    def timed(self) -> bool:
        return self.timing.validity in (tr.VALID, tr.QUESTIONABLE)

    @property
    def forming_lost(self) -> bool:
        """The hand was lost while the grasp was formed: the recording has
        a gap inside the change and is not read (INVALID, no duration)."""
        return self.timing.forming_lost_count > 0

    @property
    def changed(self) -> bool:
        """The recording holds a change of pose after the cue (whatever its
        validity): a movement start was found, or the pose was already
        changing at the cue."""
        g = self.timing
        return (g.movement_start_time_s is not None
                or g.notes.startswith("the pose was already changing"))


@dataclass
class GraspSet:
    """A Set A session as the hand-in folder presents it."""

    session: pc.Session
    hand: str
    participant: str
    items: List[dict]                      # the protocol's grasps, in order
    target: int                            # kept takes wanted per grasp
    takes: List[Take] = field(default_factory=list)
    moves: List[Move] = field(default_factory=list)    # recorded changes of pose
    missing: List[str] = field(default_factory=list)   # takes left out, and why
    problems: List[str] = field(default_factory=list)  # things the build found wrong
    rejected: Dict[str, int] = field(default_factory=dict)   # reason kind -> count
    # the same per grasp: grasp -> reason kind -> count
    rejected_by_item: Dict[str, Dict[str, int]] = field(default_factory=dict)
    has_joint_frames: bool = False
    joint_frames_pdf: Optional[Path] = None
    source_notes: List[str] = field(default_factory=list)   # <protocol>_source_notes.txt
    # the waits for an open hand that ran out: how many with no hand seen at
    # all, how many with a hand the coach did not accept
    waits_no_hand: int = 0
    waits_other: int = 0
    # the still notes of this session (`load_notes`), {} when there are none
    still_notes: Dict[str, object] = field(default_factory=dict)
    # the mock transition sessions added with allow_mock
    # (`add_transition_session`), by folder name: the README says MOCK
    mock_transition_sessions: List[str] = field(default_factory=list)

    def of(self, item: str) -> List[Take]:
        return sorted((t for t in self.takes if t.item == item),
                      key=lambda t: t.n)

    @property
    def item_ids(self) -> List[str]:
        return [str(it.get("id")) for it in self.items]

    @property
    def with_forming(self) -> List[Take]:
        return [t for t in self.takes if t.forming is not None]

    @property
    def from_open(self) -> List[Move]:
        """The coached grasp takes' own changes, open hand into the grasp."""
        return [m for m in self.moves if m.coached]

    @property
    def recorded_moves(self) -> List[Move]:
        """The takes of `scripts/leap/record_transitions.py`."""
        return [m for m in self.moves if not m.coached]

    @property
    def between_grasps(self) -> List[Move]:
        """The recorded changes from one grasp type into another."""
        return [m for m in self.moves
                if OPEN_HAND not in (m.from_grasp, m.to_grasp)]

    @property
    def target_total(self) -> int:
        return self.target * len(self.items)

    @property
    def missing_total(self) -> int:
        return sum(max(0, self.target - len(self.of(i))) for i in self.item_ids)

    @property
    def imaged(self) -> List[Take]:
        """The takes whose still the build cuts to the recorded hand."""
        return [t for t in self.takes if t.still_has_hand]

    @property
    def unread(self) -> List[Take]:
        """The takes with a picture that the still notes do not say was read
        by eye; [] when there are no still notes at all (section 1 then says
        that no picture was read)."""
        return ([t for t in self.imaged if not t.still_read]
                if self.still_notes else [])

    @property
    def notes_read_on(self) -> str:
        return str(self.still_notes.get("read_on") or "a day not recorded")

    @property
    def notes_confirmed(self) -> bool:
        return self.still_notes.get("confirmed_by_operator") is True


def long_path(path) -> Path:
    """`path` in a form Windows opens past 260 characters.

    The hand-in folder sits under a long OneDrive path and its files have
    long names, so a deep folder crosses MAX_PATH; without the
    extended-length prefix `is_file` on such a path says False and a file
    that is there reads as missing. The prefix is added to every path on
    Windows, short ones included, so that everything derived from a folder
    (`folder / name`, `rglob`) carries it. Elsewhere the path is only made
    absolute. (`scripts/package_professor_set.py` writes through the same.)
    """
    s = os.path.abspath(str(path))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = ("\\\\?\\UNC\\" + s[2:]) if s.startswith("\\\\") else \
            ("\\\\?\\" + s)
    return Path(s)


def read_lines(path: Path) -> Tuple[List[dict], int]:
    """(every parseable JSON line of a take in file order, the number that
    did not parse)."""
    out, bad = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for text in f:
            if not text.strip():
                continue
            try:
                d = json.loads(text)
            except ValueError:
                bad += 1
                continue
            if isinstance(d, dict):
                out.append(d)
            else:
                bad += 1
    return out, bad


def is_wait(kind: str) -> bool:
    """A reason kind that is a wait for an open hand that ran out: no
    recording was made, so it is not a rejected recording."""
    return kind.lower().startswith("no open hand acquired")


def alternative_timings(time_it, thresholds: dict
                        ) -> List[Tuple[float, float, tr.Timing]]:
    """The recording `time_it(**thresholds)` times, timed again at twice
    and at half the two speed thresholds: (start speed, settle speed,
    timing). A duration read off speed thresholds depends on them, and the
    README shows by how much."""
    start = thresholds.get("start_speed_mm_s", tr.POSE_MOVE_SPEED_MM_S)
    settle = thresholds.get("settle_speed_mm_s", tr.POSE_SETTLE_SPEED_MM_S)
    out = []
    for k in (2.0, 0.5):
        th = dict(thresholds, start_speed_mm_s=start * k,
                  settle_speed_mm_s=settle * k)
        out.append((start * k, settle * k, time_it(**th)))
    return out


def forming_file(session: pc.Session, t: pc.TakeRef,
                 meta: dict) -> Optional[Path]:
    """The take's forming file (`forming/<take>.jsonl`), or None.

    session.json's `files.forming` first, then the meta's; None for a take
    recorded before the recorder kept one, or a --no-coach take.
    """
    found = t.path("forming")
    if found is not None:
        return found
    rel = (meta.get("files") or {}).get("forming") if isinstance(
        meta.get("files"), dict) else None
    p = session.path / str(rel) if rel else None
    return p if p is not None and p.is_file() else None


def _coaching(meta: dict) -> dict:
    c = meta.get("coaching") if isinstance(meta, dict) else None
    return c if isinstance(c, dict) else {}


def waited_for_rules(meta: dict) -> bool:
    """A coached take recorded under the recorder's rules of 2026-10-07 or
    later, read from its meta: its `coaching` has `forming_lost_count`
    (written by every coached take since, 0 or more; absent before)."""
    return "forming_lost_count" in _coaching(meta)


def forming_lost_count(meta: dict) -> int:
    """How many times the hand was lost while forming the grasp of this
    take (`coaching.forming_lost_count`); 0 without the key."""
    try:
        return int(_coaching(meta).get("forming_lost_count") or 0)
    except (TypeError, ValueError):
        return 0


def forming_lost_notes(meta: dict) -> List[str]:
    """The note of an accepted take kept after the hand was lost while
    forming and found again (`FORMING_LOST_NOTE`), or []."""
    n = forming_lost_count(meta)
    if not n:
        return []
    return [FORMING_LOST_NOTE.format(n=n, times="time" if n == 1 else "times")]


def reason_kind(reason: str) -> str:
    """What kind of rejection a recorder's reason is, without its numbers.

    The reasons name the take's own numbers ("tracked 63 percent: lost 1
    time, for 1.9 s (not back by the end) with the hand at 23 cm ..."), so
    no two are equal; this groups them so the README can say which kind a
    grasp was rejected for most often. The recorder's rules of 2026-10-07
    (`recorder_rules` in a meta) have reasons of their own, kept apart from
    the earlier ones they replace: a hand lost while forming is waited for,
    and rejects only when "not found again"; a close before the GO beep
    resets the attempt, and rejects only past "the limit that ends a stuck
    loop".
    """
    r = " ".join(str(reason or "").split())
    low = r.lower()
    if not low:
        return "no reason recorded"
    if low.startswith("no hand was tracked"):
        return "no hand tracked during the take"
    if low.startswith("no open hand acquired"):
        return r.split(" (")[0]
    if "while forming the grasp" in low and "not found again" in low:
        return LOST_NOT_FOUND_KIND
    if "the limit that ends a stuck loop" in low:
        return RESET_LIMIT_KIND
    if "while forming the grasp" in low:
        return "hand lost while forming the grasp"
    if low.startswith("closed before the cue"):
        return "closed before the cue"
    if low == "operator redo":
        return "operator redo"
    if "operator quit" in low:
        return "operator quit at the review"
    if low.startswith("interrupted"):
        return "interrupted (Ctrl+C)"
    if "fitted the hand as" in low:
        return "tracker fitted it as the other hand"
    if "not back by the end" in low:
        return "hand lost during the take, not back by the end"
    if "inside the static interval" in low:
        return "hand lost inside the static interval"
    if "no loss inside the take" in low and "first tracked" in low:
        return "hand first tracked late in the take"
    if "new id" in low:
        return "tracker gave the hand a new id"
    if low.startswith("tracked") and " lost " in low:
        return "hand lost during the take"
    return re.sub(r"\d+(?:\.\d+)?", "N", r.split(": ")[0])


def _protocol_items(session: pc.Session) -> List[dict]:
    """The grasps the session was to record, in the protocol's order: the
    protocol file's entries, then any grasp a take names that it does not."""
    items = [dict(it) for it in (session.protocol or {}).get("items", []) or []
             if it.get("id")]
    listed = [str(i) for i in (session.meta.get("items") or [])]
    if listed and items:
        keep = [it for it in items if str(it["id"]) in listed]
        items = keep or items
    known = {str(it["id"]) for it in items}
    for name in listed:
        if name not in known:
            items.append({"id": name})
            known.add(name)
    for t in session.takes:
        if t.item not in known:
            items.append({"id": t.item})
            known.add(t.item)
    return items


def _angles(row: dict) -> Tuple[Dict[str, float], float, List[dict]]:
    """(the 24 with the 6 palm-referenced and the 8 paper-order values,
    the forearm-to-palm angle, the 26-row table) of a summary frame."""
    table = jf.joint_table(row, "camera", "wrist")
    values = dict(jf.dof24(table))
    values.update(jf.palm_reference_angles(row, "camera"))
    values.update(jf.mcp_paper_order(row, "camera"))
    return values, jf.wrist_palm_deg(row, "camera"), table


def phalanx_twist(table: Sequence[dict]) -> Tuple[float, float]:
    """(largest |twist|, largest |twist + abd tan(flex / 2)|) in degrees over
    the phalanx rows of a joint-frame table.

    The tracker builds a finger's PROXIMAL, INTERMEDIATE and DISTAL joints
    (and the thumb's PROXIMAL and DISTAL) as swings, with no turn about the
    bone. Split in the table's order (flexion, abduction, twist) such a
    joint still shows a twist, -abd tan(flex / 2), which is arithmetic and
    not the finger; the second number says how exactly that holds.
    """
    worst_twist = worst_residual = 0.0
    for r in table:
        name = r["joint"]
        if name in ("WRIST", "PALM") or name.endswith(("_METACARPAL", "_TIP")):
            continue
        flex, abd, twist = r["flex_deg"], r["abd_deg"], r["twist_deg"]
        worst_twist = max(worst_twist, abs(twist))
        worst_residual = max(worst_residual, abs(
            twist + abd * math.tan(math.radians(flex) / 2.0)))
    return worst_twist, worst_residual


def _csv_cells(path: Path) -> Optional[List[List[str]]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [r for r in csv.reader(f)]
    except OSError:
        return None


def source_notes(session: pc.Session) -> List[str]:
    """The lines of `<protocol file's stem>_source_notes.txt` beside the
    session's protocol file, when there is one: what we read into the
    papers that their own words do not say (which panel of a figure is
    which grasp, which grasps are the same configuration). Kept beside the
    protocol and not in it, because a session pins its protocol file by
    hash."""
    name = session.meta.get("protocol_file")
    if not name:
        return []
    path = Path(str(name))
    if not path.is_absolute():
        path = pc.REPO / path
    notes = path.with_name(path.stem + "_source_notes.txt")
    try:
        text = notes.read_text(encoding="utf-8")
    except OSError:
        return []
    return [" ".join(x.split()) for x in text.splitlines() if x.strip()]


def citations(session: pc.Session) -> List[str]:
    """The papers as the protocol file's description cites them: its
    sentences of the form "p1 = ... (...)."."""
    desc = str((session.protocol or {}).get("description") or "")
    return re.findall(r"\bp\d+ = .*?\)\.", desc)


def configurations(items: Sequence[dict]) -> Tuple[int, int]:
    """(grasps the protocol links to another by `same_as`, distinct hand
    configurations once linked grasps count as one)."""
    ids = [str(it.get("id")) for it in items]
    root = {i: i for i in ids}

    def find(x: str) -> str:
        while root[x] != x:
            x = root[x]
        return x
    for it in items:
        for other in it.get("same_as") or []:
            if other in root:
                root[find(str(other))] = find(str(it.get("id")))
    return (sum(1 for it in items if it.get("same_as")),
            len({find(i) for i in ids}))


def load_notes(session: pc.Session) -> Dict[str, object]:
    """The still notes of `session`: what was read by eye in its takes'
    infrared stills (`grasp_images.load_still_notes`, the repository's
    protocols\\grasps_still_notes.json). {} when there are none, or when
    they are another session's: a reading of one session's pictures says
    nothing about another's."""
    try:
        # a ValueError: the notes are of another session
        notes = gi.load_still_notes(REPO_ROOT, session.name)
    except ValueError:
        return {}
    if not isinstance(notes, dict) or not notes:
        return {}
    if str(notes.get("session") or "") != session.name:
        return {}
    return notes


def still_offset(still: Path, meta: dict) -> Optional[float]:
    """When the still's file was written, in seconds after the recorder's
    take_start: the file's modification time minus that start, both on the
    recording computer's clock. None without a start in the meta."""
    try:
        return os.stat(long_path(still)).st_mtime - float(meta["take_start"])
    except (KeyError, TypeError, ValueError, OSError):
        return None


def picture_notes(t: "Take", notes: Dict[str, object]) -> Dict[str, str]:
    """The notes a take carries for what its infrared picture shows, read
    by eye, by kind: "hand" (something in or against the recorded hand),
    "other" (something in the other hand), "unclear" (the picture does not
    let one tell), and "unread" (the take has a picture and there are still
    notes, but they do not say it was read: nothing about it is known, and
    it is not counted as showing nothing). Section 15 lists them under
    these four kinds. The reading was made on the recorder's uncut still:
    the cut picture of the folder shows the recorded hand only, so the
    "other" note says that hand is not in it."""
    when = (f"read by eye on {notes.get('read_on') or 'a day not recorded'}"
            + (", confirmed by the operator"
               if notes.get("confirmed_by_operator") is True
               else ", not confirmed by the operator"))
    out: Dict[str, str] = {}
    if t.still_note_hand:
        out["hand"] = ("the infrared picture shows in or against the "
                       f"recorded hand: {t.still_note_hand} ({when})")
    if t.still_note_other:
        out["other"] = ("the recorder's uncut still shows in the other "
                        f"hand: {t.still_note_other} ({when}); the other "
                        "hand is not in the cut picture of this folder")
    if t.still_unclear:
        out["unclear"] = ("the infrared picture does not settle whether "
                          f"something was in the hand ({when})")
    if notes and t.still_has_hand and not t.still_read:
        out["unread"] = "the infrared picture has not been read by eye"
    return out


def collect(session: pc.Session, participant: str = "P01",
            still_notes: Optional[Dict[str, object]] = None,
            **thresholds) -> GraspSet:
    """Read every accepted take of a Set A session once and measure it.

    `thresholds` are `leap_hand.transition.transition_timing`'s keyword
    arguments, for the takes that have a forming file. `still_notes` are
    the readings of the stills to use, `load_notes(session)` when None.
    """
    hand = session.hand
    proto = session.protocol or {}
    gs = GraspSet(session=session, hand=hand, participant=participant,
                  items=_protocol_items(session),
                  target=int(session.meta.get("takes_per_item")
                             or proto.get("takes_per_item") or 3))
    gs.still_notes = dict(load_notes(session) if still_notes is None
                          else still_notes)
    unclear = set(gi.unclear(gs.still_notes))
    # the pictures read by eye: the notes' "read" list; notes written before
    # they had one say only that the takes they name were read
    listed = "read" in gs.still_notes
    read = set(gi.read_by_eye(gs.still_notes))
    frames_dir = session.path / JOINT_FRAMES_DIR
    gs.has_joint_frames = frames_dir.is_dir()
    gs.source_notes = source_notes(session)
    pdf = frames_dir / "joint_frames.pdf"
    gs.joint_frames_pdf = pdf if pdf.is_file() else None
    for t in session.takes:
        if not t.accepted:
            kind = reason_kind(t.reason)
            gs.rejected[kind] = gs.rejected.get(kind, 0) + 1
            mine = gs.rejected_by_item.setdefault(t.item, {})
            mine[kind] = mine.get(kind, 0) + 1
            if is_wait(kind):
                # what the recorder itself wrote about the wait
                m = pc.read_json(t.path("meta")) if t.path("meta") else None
                coaching = (m.get("coaching") if isinstance(m, dict) else
                            None) or {}
                seen = str(coaching.get("acquire_missing") or "")
                if seen.lower().startswith("no hand seen"):
                    gs.waits_no_hand += 1
                else:
                    gs.waits_other += 1
            continue
        leap = t.path("leap")
        if leap is None:
            gs.missing.append(f"{t.name}: no camera file, take not handed in")
            continue
        meta = pc.read_json(t.path("meta")) if t.path("meta") else None
        meta = meta if isinstance(meta, dict) else {}
        lines, bad = read_lines(leap)
        facts = take_facts(lines, meta, hand)
        take = Take(ref=t, stem=f"{t.item}_{hand}_take{t.take}",
                    cfg=session.item(t.item) or {}, meta=meta, facts=facts,
                    keypoints=t.path("keypoints"), bad_lines=bad)
        if take.keypoints is None:
            gs.missing.append(f"{t.name}: no keypoints file from the recorder")
        take.still = t.path("still")
        if take.still is not None:
            take.still_offset_s = still_offset(take.still, meta)
            take.still_has_hand = gi.has_hand(take.still, hand)
        (take.still_note_hand,
         take.still_note_other) = gi.looked_at_note(gs.still_notes, take.stem)
        take.still_unclear = take.stem in unclear
        named = bool(take.still_note_hand or take.still_note_other
                     or take.still_unclear)
        take.still_read = take.stem in read if listed else named
        if named and not take.still_read:
            gs.problems.append(
                f"{take.stem}: the still notes say what its picture shows "
                "but do not list it under \"read\"")
        take.picture_notes = picture_notes(take, gs.still_notes)
        facts.picture_notes = list(take.picture_notes.values())
        facts.notes += facts.picture_notes
        # kept after the hand was lost while forming and found again (the
        # recorder's rules of 2026-10-07): the static take is whole, its
        # forming file is not timed, and its notes say so
        take.forming_notes = forming_lost_notes(meta)
        facts.notes += take.forming_notes
        if bad:
            gs.problems.append(f"{t.name}: {bad} line(s) of the JSONL do not "
                               "parse and were skipped")

        mine, _how = operator_lines(lines, meta, hand)
        if mine:
            palms = np.array([p for p in (np.asarray(d["palm_abs"], float)
                                          if d.get("palm_abs") else None
                                          for _i, d in mine) if p is not None])
            if len(palms):
                take.palm_cm = tuple(float(v) * 100.0
                                     for v in np.median(palms, axis=0))
            both = [d for _i, d in mine if d.get("abs26")]
            if both:
                arr = np.asarray([d["abs26"] for d in both], dtype=float)
                take.wrist_minus_palm_z_cm = float(np.median(
                    arr[:, jf.WRIST, 2] - arr[:, jf.PALM, 2]) * 100.0)
            own = {i for i, _d in mine}
            other = np.array([d["palm_abs"] for i, d in enumerate(lines)
                              if i not in own and d.get("palm_abs")],
                             dtype=float)
            if len(other):
                take.other_palm_cm = tuple(
                    float(v) * 100.0 for v in np.median(other, axis=0))
                take.other_palm_x_cm = take.other_palm_cm[0]

        index = facts.representative_frame_index
        row = None if index is None else lines[index]
        csv_path = frames_dir / f"{t.name}.csv"
        take.joint_frames = csv_path if csv_path.is_file() else None
        if row is not None:
            try:
                take.angles, take.forearm_to_palm_deg, table = _angles(row)
                (take.phalanx_twist_deg,
                 take.phalanx_twist_residual_deg) = phalanx_twist(table)
                pos, rots, _side = jf.world_pose(row, "camera")
                take.palm_back_up = bool(rots[jf.PALM][1, 1] > 0.0)
                if np.asarray(row.get("abs26", [])).shape == (26, 3):
                    take.chain_gap_mm = float(np.abs(
                        pos - np.asarray(row["abs26"], dtype=float)
                    ).max()) * 1000.0
                thumb = rots[JOINT_NAMES.index("THUMB_METACARPAL")]
                take.thumb_axis_deg = float(np.degrees(np.arccos(np.clip(
                    float(rots[jf.PALM][:, 0] @ thumb[:, 0]), -1.0, 1.0))))
                take.thumb_axis_wrist_deg = float(np.degrees(np.arccos(
                    np.clip(float(rots[jf.WRIST][:, 0] @ thumb[:, 0]),
                            -1.0, 1.0))))
                take.wrist_bend_deg = float(table[jf.PALM]["flex_deg"])
                take.table = table
            except (ValueError, KeyError, TypeError) as e:
                gs.problems.append(f"{t.name}: the representative frame's "
                                   f"angles could not be computed ({e})")
                table = None
            if take.joint_frames is not None and table is not None:
                want = [jf.CSV_COLUMNS] + jf.csv_rows(table)
                take.joint_frames_match = _csv_cells(take.joint_frames) == want
                if not take.joint_frames_match:
                    gs.problems.append(
                        f"{t.name}: joint_frames/{t.name}.csv is not the "
                        "table of the take's representative frame; run "
                        "scripts/joint_frames_view.py --session again")
        if gs.has_joint_frames and take.joint_frames is None:
            gs.missing.append(f"{t.name}: no joint-frame CSV in the session's "
                              "joint_frames folder")

        take.forming = forming_file(session, t, meta)
        if take.forming is not None:
            rows, _bad = read_lines(take.forming)
            take.timing = tr.forming_timing(rows, meta, **thresholds)
            again = alternative_timings(
                lambda rows=rows, meta=meta, **th: tr.forming_timing(
                    rows, meta, **th), thresholds)
            gs.moves.append(Move(
                alternatives=again,
                no_wrist=tr.forming_timing(rows, meta, **dict(
                    thresholds, palm_joints=tr.PALM_JOINTS_NO_WRIST)),
                from_grasp=OPEN_HAND, to_grasp=t.item, take=t.take, hand=hand,
                source=take.forming,
                dest=rel(TRANSITIONS_DIR, f"{OPEN_HAND}_to_{take.stem}.jsonl"),
                timing=take.timing, session=session.name,
                static_take_file=rel(GRASPS_DIR, f"{take.stem}.jsonl"),
                coached=True,
                ready_step=bool((meta.get("coaching") or {}).get(
                    "ready_pause_s")),
                context=forming_context(meta)))
        gs.takes.append(take)
    return gs


def forming_context(meta: dict) -> str:
    """What a coached take's meta says came before the close that was kept:
    under the rules before 2026-10-07 a close the tracker lost sent the
    operator back with "close it slower", so the close in the forming file
    is then a slowed one. Under the later rules (`waited_for_rules`) a lost
    hand is waited for instead and nobody is told to close slower: "" (the
    forming file of such a take says why it is not timed itself)."""
    if waited_for_rules(meta):
        return ""
    coaching = meta.get("coaching") if isinstance(meta.get("coaching"),
                                                  dict) else {}
    lost = int(coaching.get("lost_while_forming") or 0)
    if not lost:
        return ""
    return (f"the tracker lost the hand on {_plural(lost, 'earlier close')} "
            "of this attempt and the operator was told to close slower: "
            "this is the slower close")


TRANSITION_SET = "transitions"             # scripts/leap/record_transitions.py


def add_transition_session(gs: GraspSet, folder: Path,
                           allow_mock: bool = False,
                           **thresholds) -> List[str]:
    """Add the accepted takes of a transition session to `gs.moves`; returns
    lines about what was left out.

    Each take is timed again here from its recording and its meta
    (`leap_hand.transition.recorded_timing`), with this build's thresholds,
    so the table has one rule for every row. A mock session (synthetic
    hands) is left out unless `allow_mock`, as a mock grasp session is;
    with it, the session is named in `gs.mock_transition_sessions` (the
    README's MOCK DATA banner) and each of its takes is a `Move` with
    `mock` set (MOCK in its row's notes and its line of section 17).
    """
    folder = Path(folder)
    meta = pc.read_json(folder / "session.json")
    if not isinstance(meta, dict) or meta.get("set") != TRANSITION_SET:
        return [f"{folder}: not a transition session (no session.json with "
                f"set \"{TRANSITION_SET}\")"]
    mock = bool(pc.mock_flags(meta))
    if mock and not allow_mock:
        return [f"{folder.name}: mock transition session, not handed in"]
    if mock and folder.name not in gs.mock_transition_sessions:
        gs.mock_transition_sessions.append(folder.name)
    notes = []
    for entry in meta.get("takes", []) or []:
        if not entry.get("accepted"):
            continue
        files = entry.get("files") or {}
        leap = folder / str(files.get("leap") or "")
        take_meta = pc.read_json(folder / str(files.get("meta") or ""))
        if not leap.is_file() or not isinstance(take_meta, dict):
            notes.append(f"{folder.name}/{entry.get('name')}: recording or "
                         "meta missing, not handed in")
            continue
        rows, _bad = read_lines(leap)
        hand = str(take_meta.get("hand") or meta.get("hand") or gs.hand)
        frm, to = str(take_meta["from_grasp"]), str(take_meta["to_grasp"])
        n = take_meta.get("take")
        gs.moves.append(Move(
            alternatives=alternative_timings(
                lambda rows=rows, take_meta=take_meta, **th:
                tr.recorded_timing(rows, take_meta, **th), thresholds),
            no_wrist=tr.recorded_timing(rows, take_meta, **dict(
                thresholds, palm_joints=tr.PALM_JOINTS_NO_WRIST)),
            from_grasp=frm, to_grasp=to, take=n, hand=hand, source=leap,
            dest=rel(TRANSITIONS_DIR, f"{frm}_to_{to}_{hand}_take{n}.jsonl"),
            timing=tr.recorded_timing(rows, take_meta, **thresholds),
            session=folder.name, ready_step=True, mock=mock))
    return notes


# --- the tables ---------------------------------------------------------------
def _r(v, nd: int = 3):
    return None if v is None else round(float(v), nd)


def rel(*parts: str) -> str:
    """A file's place in the hand-in folder, as the tables write it."""
    return "/".join(parts)


def image_file(t: Take) -> str:
    """The place of a take's cut picture in the hand-in folder."""
    return rel(GRASPS_DIR, IMAGES_DIR, f"{t.stem}.png")


def take_files(gs: GraspSet, t: Take) -> Dict[str, Optional[str]]:
    """Where a take's files are in the hand-in folder (None: not there)."""
    return {
        "raw": rel(GRASPS_DIR, f"{t.stem}.jsonl"),
        "keypoints": (rel(GRASPS_DIR, f"{t.stem}_keypoints.txt")
                      if t.keypoints is not None else None),
        "all_frames": rel(GRASPS_DIR, f"{t.item}_{gs.hand}_all_frames.txt"),
        "joint_frames": (rel(GRASPS_DIR, JOINT_FRAMES_DIR, f"{t.stem}.csv")
                         if t.joint_frames is not None else None),
        "image": (image_file(t) if t.still_has_hand else None),
        "transition": (rel(TRANSITIONS_DIR,
                           f"{OPEN_HAND}_to_{t.stem}.jsonl")
                       if t.forming is not None else None)}


def static_take(m: Move) -> Optional[str]:
    return m.static_take_file


def take_sensor(meta_files: Optional[dict], t: pc.TakeRef) -> str:
    """Which sensors a take has: "camera", "glove", "glove+camera" or "".

    Read from the meta's `files` (contract section 7: a key with null means
    the take has no such file), and for a key the meta does not have, from
    the take's files on disk. Set A today is "camera": glove null, leap
    present.
    """
    files = meta_files if isinstance(meta_files, dict) else {}

    def has(kind: str) -> bool:
        if kind in files:
            return bool(files[kind])
        return t.path(kind) is not None

    glove, camera = has("glove"), has("leap")
    return ("glove+camera" if glove and camera else "camera" if camera
            else "glove" if glove else "")


def summary_rows(gs: GraspSet) -> List[dict]:
    """`SUMMARY_COLUMNS` for every take, in the session's order."""
    out = []
    for t in gs.takes:
        m, f, files = t.meta, t.facts, take_files(gs, t)
        gate = m.get("gate") if isinstance(m.get("gate"), dict) else {}
        tracked = m.get("tracked_fraction")
        interval = m.get("static_interval") or [None, None]
        out.append({
            "grasp_name": t.item, "grasp_label": t.cfg.get("label", ""),
            "source_paper": t.cfg.get("source") or "",
            "source_figure": t.cfg.get("figure") or "",
            "hand": gs.hand, "take": t.ref.take,
            "raw_jsonl_file": files["raw"],
            "representative_keypoints_file": files["keypoints"],
            "all_frames_file": files["all_frames"],
            "joint_frames_file": files["joint_frames"],
            "image_file": files["image"],
            "transition_capture_file": files["transition"],
            "sensor": take_sensor(m.get("files"), t.ref),
            "recording_duration_s": _r(f.recording_duration_s),
            "camera_clock_duration_s": _r(f.camera_clock_duration_s),
            "recorder_window_s": _r(f.recorder_window_s),
            "unique_frame_count": f.unique_frame_count,
            "hand_frame_count": f.hand_frame_count,
            "jsonl_row_count": f.jsonl_row_count,
            "frame_rate_hz": _r(f.frame_rate_hz, 1),
            "tracked_percent": (None if tracked is None
                                else round(100.0 * float(tracked), 1)),
            "largest_gap_s": _r(f.largest_gap_s),
            "static_interval_start_s": _r(f.static_interval_start_s),
            "static_interval_end_s": _r(f.static_interval_end_s),
            "static_interval_duration_s": _r(f.static_interval_duration_s),
            "static_interval_mean_speed_mm_s":
                _r(f.static_interval_mean_speed_mm_s, 1),
            "static_interval_pose_speed_mm_s":
                _r(f.static_interval_pose_speed_mm_s, 1),
            "static_interval_shape_rms_mm":
                _r(f.static_interval_shape_rms_mm, 2),
            "static_interval_shape_rms_max_mm":
                _r(f.static_interval_shape_rms_max_mm, 2),
            "static_interval_first_frame_id":
                f.static_interval_first_frame_id,
            "static_interval_last_frame_id": f.static_interval_last_frame_id,
            "representative_frame_index": f.representative_frame_index,
            "representative_frame_id": f.representative_frame_id,
            "representative_frame_time_s": _r(f.representative_frame_time_s),
            "take_shape_rms_max_mm": _r(f.take_shape_rms_max_mm, 2),
            "frames_within_5mm_percent": _r(f.frames_within_5mm_percent, 1),
            "shape_rms_start_mm": _r(f.shape_rms_start_mm, 2),
            "shape_rms_end_mm": _r(f.shape_rms_end_mm, 2),
            "other_hand_detected": f.other_hand_detected,
            "other_hand_frame_fraction": _r(f.other_hand_frame_fraction, 4),
            "other_hand_median_distance_cm":
                _r(f.other_hand_median_distance_cm, 1),
            "other_hand_finger_curl_minus_own_deg":
                _r(f.other_hand_finger_curl_minus_own_deg, 1),
            "hand_selection": f.hand_selection,
            "hand_ids": " ".join(str(i) for i in f.hand_ids),
            "tracker_label_on_representative_frame": f.representative_label,
            "tracker_id_changes": f.tracker_id_changes,
            "identity_flag": f.identity_flag,
            "status": "accepted",
            "decided_by": m.get("decided_by", ""),
            # the gate's count (protocol_check.check_grasp_take): the static
            # interval's own when the meta has it, else the take's
            "reacquisitions_in_static_interval": gate.get(
                "reacquisitions_in_static_interval", m.get("reacquisitions")),
            "reacquisitions": m.get("reacquisitions"),
            "grab_strength": m.get("grab_strength"),
            "pinch_strength": m.get("pinch_strength"),
            **{f"curl_{k}": (m.get("curls") or {}).get(k)
               for k in pc.FINGERS},
            "palm_height_cm": m.get("palm_height_cm"),
            "view_angle_deg": m.get("view_angle_deg"),
            "index_metacarpal_mm": _r(f.index_metacarpal_mm, 2),
            "index_metacarpal_range_mm": _r(f.index_metacarpal_range_mm, 2),
            "first_wall_time": f.first_wall_time,
            "static_interval_start": interval[0],
            "static_interval_end": interval[1] if len(interval) > 1 else None,
            "medoid_wall_time": m.get("medoid_wall_time"),
            "orientation_note": m.get("orientation_note", ""),
            "session": gs.session.name, "recorder_take_name": t.ref.name,
            "notes": "; ".join(f.notes)})
    return out


def static_status(takes: Sequence[Take]) -> str:
    """READY when one take of the grasp has all four datasets: the summary
    frame's keypoints, the frame-by-frame file (written for every grasp that
    has a take), the joint-frame XYZ table and the angles."""
    ok = any(t.keypoints is not None and t.joint_frames is not None
             and t.angles is not None for t in takes)
    return READY if ok else INCOMPLETE


def all_noted(gs: GraspSet) -> List[str]:
    """The grasp types that have recordings and a note on every one of them:
    READY by their files, with no recording a reader can use without
    reading its note first."""
    return [i for i in gs.item_ids
            if gs.of(i) and all(t.facts.abnormal for t in gs.of(i))]


def same_configuration(items: Sequence[dict]) -> Dict[str, List[str]]:
    """grasp id -> the other grasps of its configuration, in protocol order:
    the protocol's `same_as` links followed both ways (it lists some of
    them from one side only)."""
    ids = [str(it.get("id")) for it in items]
    root = {i: i for i in ids}

    def find(x: str) -> str:
        while root[x] != x:
            x = root[x]
        return x
    for it in items:
        for other in it.get("same_as") or []:
            if other in root:
                root[find(str(other))] = find(str(it.get("id")))
    return {i: [j for j in ids if j != i and find(j) == find(i)] for i in ids}


def in_angle_table(t: Take) -> bool:
    """The take has a row in the angle table: its angles were computed and
    its joint-frame table is handed in (`angle_rows`)."""
    return bool(t.angles) and t.joint_frames is not None


def between_take_spread(takes: Sequence[Take]
                        ) -> Optional[Tuple[float, str, Dict[str, float]]]:
    """How far the takes of one grasp are from each other: (the largest
    spread, the angle it belongs to, every angle's spread), a spread being
    the largest minus the smallest value of one of the given angles over
    the takes' representative frames. None with fewer than two takes that
    have their angles."""
    have = [t for t in takes if in_angle_table(t)]
    if len(have) < 2:
        return None
    spreads = {n: max(t.angles[n] for t in have) - min(t.angles[n]
                                                      for t in have)
               for n in GIVEN_NAMES}
    worst = max(GIVEN_NAMES, key=lambda n: spreads[n])
    return spreads[worst], worst, spreads


def angle_rms(a: Take, b: Take) -> float:
    """RMS difference in degrees of the given angles of two takes."""
    return float(np.sqrt(np.mean([(a.angles[n] - b.angles[n]) ** 2
                                  for n in GIVEN_NAMES])))


def most_common_reject(kinds: Dict[str, int], attempted: bool) -> str:
    """"<kind> (n of m)" for the kind a grasp's recordings were rejected for
    most often, every kind tied for it named, the waits for an open hand
    left out (no recording was made in them); "" with no rejected
    recording, "not attempted" for a grasp with no attempt at all."""
    waited = any(is_wait(k) for k in kinds)
    kinds = {k: n for k, n in kinds.items() if not is_wait(k)}
    total = sum(kinds.values())
    if not total:
        return "" if attempted or waited else "not attempted"
    most = max(kinds.values())
    return "; ".join(f"{k} ({n} of {total})" for k, n in kinds.items()
                     if n == most)


def audit_rows(gs: GraspSet) -> List[dict]:
    """`AUDIT_COLUMNS` for every grasp type of the protocol, recorded or not."""
    out = []
    same = same_configuration(gs.items)
    for it in gs.items:
        gid = str(it.get("id"))
        takes = gs.of(gid)
        have = [t.n for t in takes]
        missing = max(0, gs.target - len(takes))
        captures = [m for m in gs.moves if m.to_grasp == gid]
        spread_ = between_take_spread(takes)
        kinds = gs.rejected_by_item.get(gid, {})
        waits = sum(n for k, n in kinds.items() if is_wait(k))
        out.append({
            "grasp_name": gid, "grasp_label": it.get("label", ""),
            "source_paper": it.get("source") or "",
            "source_figure": it.get("figure") or "",
            "same_configuration_as": " ".join(same[gid]),
            "accepted_recordings": len(takes),
            "takes_available": " ".join(str(n) for n in have),
            "target_takes": gs.target, "missing_takes": missing,
            "representative_keypoints_files":
                sum(1 for t in takes if t.keypoints is not None),
            "all_frames_file": (rel(GRASPS_DIR,
                                    f"{gid}_{gs.hand}_all_frames.txt")
                                if takes else None),
            "all_frames_takes": " ".join(str(n) for n in have),
            "all_frames_frame_count":
                sum(t.facts.hand_frame_count for t in takes),
            "joint_frame_xyz_files":
                sum(1 for t in takes if t.joint_frames is not None),
            "joint_angle_rows":
                sum(1 for t in takes
                    if t.joint_frames is not None and t.angles is not None),
            "transition_captures": len(captures),
            "valid_transitions":
                sum(1 for m in captures if m.timing.validity == tr.VALID),
            "questionable_transitions":
                sum(1 for m in captures
                    if m.timing.validity == tr.QUESTIONABLE),
            "recordings_without_a_note":
                sum(1 for t in takes if not t.facts.abnormal),
            "recordings_with_an_image":
                sum(1 for t in takes if t.still_has_hand),
            "between_take_max_spread_deg":
                None if spread_ is None else jf.fmt_num(spread_[0], 1),
            "between_take_max_spread_angle":
                None if spread_ is None else spread_[1],
            "attempts_not_kept": sum(kinds.values()),
            "open_hand_timeouts": waits,
            "rejected_recordings": sum(kinds.values()) - waits,
            "most_common_reject_reason": most_common_reject(
                kinds, bool(takes)),
            "static_pose_status": static_status(takes),
            "multi_take_status": COMPLETE if not missing else INCOMPLETE})
    return out


def angle_rows(gs: GraspSet) -> List[dict]:
    """`ANGLE_COLUMNS` for every take that has a joint-frame table, the
    numbers rounded as that table rounds them."""
    out = []
    for t in gs.takes:
        if t.joint_frames is None or t.angles is None:
            continue
        row = {"grasp_name": t.item, "hand": gs.hand, "take": t.ref.take,
               "joint_frames_file": take_files(gs, t)["joint_frames"],
               "representative_frame_index":
                   t.facts.representative_frame_index,
               "forearm_to_palm_deg": jf.fmt_num(t.forearm_to_palm_deg, 2)}
        for name in jf.DOF24_NAMES:
            # the six this tracker does not give stay empty under the
            # paper's names (README section 13)
            row[name] = (None if name in NOT_GIVEN
                         else jf.fmt_num(t.angles[name], 2))
        for column, key in METACARPAL_COLUMNS.items():
            row[column] = jf.fmt_num(t.angles[key], 2)
        for name in jf.MCP_PAPER_NAMES:
            row[name] = jf.fmt_num(t.angles[name], 2)
        out.append(row)
    return out


def transition_rows(gs: GraspSet) -> List[dict]:
    """`TRANSITION_COLUMNS` for every recorded change of pose (`gs.moves`);
    none for a take without one."""
    out = []
    for m in gs.moves:
        g = m.timing
        ok = m.timed
        out.append({
            "participant_id": gs.participant, "hand": m.hand,
            "from_grasp": m.from_grasp, "to_grasp": m.to_grasp,
            "take": m.take, "recording_file": m.dest,
            "cue_wall_time": (None if g.cue_time_s is None
                              or g.first_frame_time is None
                              else g.first_frame_time + g.cue_time_s),
            "cue_time_s": _r(g.cue_time_s),
            "movement_start_time_s": _r(g.movement_start_time_s),
            "settled_time_s": _r(g.settled_time_s),
            "transition_duration_s": _r(g.transition_duration_s) if ok else None,
            "reaction_time_s": _r(g.reaction_time_s),
            "cue_to_settled_s": _r(g.cue_to_settled_s),
            "peak_speed_mm_s": _r(g.peak_speed_mm_s, 1),
            "speed_at_cue_mm_s": _r(g.speed_at_cue_mm_s, 1),
            "pose_change_mm": _r(g.pose_change_mm, 1),
            "hand_speed_at_cue_mm_s": _r(g.hand_speed_at_cue_mm_s, 1),
            "hand_peak_speed_mm_s": _r(g.hand_peak_speed_mm_s, 1),
            "last_movement_time_s": _r(g.last_move_time_s),
            "movement_start_method": g.movement_start_method,
            "settled_method": g.settled_method,
            "validity": g.validity, "notes": m.notes, "frames": g.frames,
            "hand_ids": " ".join(str(i) for i in g.hand_ids),
            "first_frame_capture_time": g.first_frame_time,
            "analysed_until_s": _r(g.end_time_s, 6),
            "session": m.session, "static_take_file": static_take(m)})
    return out


# --- the PDF of the pictures ----------------------------------------------------
def images_pdf_footer(gs: GraspSet) -> str:
    """The footer of every page of grasp_images.pdf; no time of the build in
    it, so the same session gives the same bytes."""
    return ("Infrared pictures cut to the recorded hand; see README.txt "
            f"sections 1 and 5. Built from session {gs.session.name}.")


def image_pdf_pages(gs: GraspSet) -> List[dict]:
    """The pages of grasp_images.pdf (`grasp_images.write_images_pdf`): one
    per grasp type with a picture, in the protocol's order; the takes of
    the grasp side by side, each captioned with its name, take and day, and
    what was seen by eye when the still notes say. The notes were read off
    the recorder's uncut still, so a caption says "uncut still"; "(above)"
    points at the picture over it, and only where that picture shows what
    the caption names: something in the other hand is "not in this
    picture", which is cut to the recorded hand. The texts go in whole:
    `grasp_images.page_layout` wraps every line to the page."""
    pages = []
    for it in gs.items:
        gid = str(it.get("id"))
        takes = [t for t in gs.of(gid) if t.still_has_hand]
        if not takes:
            continue
        where = ", ".join(x for x in (str(it.get("source") or ""),
                                      str(it.get("figure") or "")) if x)
        lines: List[str] = []
        if it.get("shape"):
            lines.append(f"Shape asked for: {it['shape']}")
        seen = [t for t in takes if t.picture_notes]
        if seen:
            lines.append(f"What was seen, {notes_who(gs)}:")
            for t in seen:
                for kind, text in (("hand", "in or against the recorded "
                                    f"hand: {t.still_note_hand}"),
                                   ("other", "in the other hand, which is "
                                    "not in the picture below: "
                                    f"{t.still_note_other}"),
                                   ("unclear", "the picture does not settle "
                                    "whether something was in the hand"),
                                   ("unread", "the picture has not been read "
                                    "by eye")):
                    if kind in t.picture_notes:
                        lines.append(f"take {t.n}: {text}")
        pictures = []
        for t in takes:
            day = t.recorded_day or "day not recorded"
            caption = [t.stem, f"take {t.n}, {day}"]
            if t.picture_notes:
                caption.append(
                    "by eye, uncut still: something in the hand (above)"
                    if t.still_note_hand else
                    "by eye, uncut still: something in the other hand (not "
                    "in this picture)" if t.still_note_other else
                    "by eye, uncut still: cannot tell" if t.still_unclear
                    else "not yet read by eye (above)")
            pictures.append((caption, t.still, gs.hand))
        pages.append({"title": gid, "label": str(it.get("label") or ""),
                      "source": where, "lines": lines,
                      "pictures": pictures})
    return pages


def write_csv(dest: Path, columns: Sequence[str], rows: Sequence[dict]) -> None:
    with open(dest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({k: pc.csv_value(r.get(k)) for k in columns})


# --- the README -----------------------------------------------------------------
def wrapped(text: str, first: str = "  ", rest: str = "  ") -> List[str]:
    return textwrap.wrap(text, width=WIDTH, initial_indent=first,
                         subsequent_indent=rest, break_on_hyphens=False,
                         break_long_words=False)


def wrapped_path(text: str, first: str = "  ", rest: str = "  "
                 ) -> List[str]:
    """`wrapped`, and a word still longer than a line (a long path in a
    command) broken after a path separator, so no line passes WIDTH."""
    out = []
    for line in wrapped(text, first, rest):
        while len(line) > WIDTH:
            cut = max(line.rfind("\\", 0, WIDTH), line.rfind("/", 0, WIDTH))
            if cut <= len(rest):
                cut = WIDTH - 1
            out.append(line[:cut + 1])
            line = rest + line[cut + 1:]
        out.append(line)
    return out


def _head(n: int, title: str) -> List[str]:
    return [f"{n}. {title}", "-" * 78]


def _fmt(v, nd: int = 2, none: str = "-") -> str:
    """`v` to `nd` decimals, never "-0.00" (a rounded zero has no sign)."""
    return none if v is None else jf.fmt_num(float(v), nd)


def _range(values, nd: int = 2, unit: str = "") -> str:
    s = spread(values)
    if s is None:
        return "not measured"
    return (f"minimum {_fmt(s[0], nd)}{unit}, median {_fmt(s[1], nd)}{unit}, "
            f"maximum {_fmt(s[2], nd)}{unit}")


def _between(lo: float, hi: float, nd: int = 0) -> str:
    """"7" when both ends round to it, else "7 to 9"."""
    a, b = _fmt(lo, nd), _fmt(hi, nd)
    return a if a == b else f"{a} to {b}"


def _plural(n: int, one: str, many: Optional[str] = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def duration_stats(gs: GraspSet) -> Optional[Tuple[float, float, float]]:
    return spread([t.facts.recording_duration_s for t in gs.takes])


def static_complete(gs: GraspSet) -> bool:
    """The task's own requirement: one usable static configuration for
    every grasp type of the protocol."""
    return bool(gs.items) and all(static_status(gs.of(i)) == READY
                                  for i in gs.item_ids)


def recorded_everywhere(gs: GraspSet) -> bool:
    return bool(gs.items) and all(gs.of(i) for i in gs.item_ids)


def rigid_palm(gs: GraspSet) -> Optional[Dict[str, float]]:
    """The four finger CMC angles against the palm joint, when they are the
    same in every recording (to 0.05 degree): the tracker's palm is then
    rigid, and those four are constants of its hand model, not
    measurements. None when they vary, or there is nothing to tell from."""
    names = [n for n in jf.PALM_REF_NAMES if "_CMC_" in n]
    values = {n: [t.angles[n] for t in gs.takes if t.angles] for n in names}
    if not all(len(v) >= 2 for v in values.values()):
        return None
    if any(max(v) - min(v) > 0.05 for v in values.values()):
        return None
    return {n: float(np.median(v)) for n, v in values.items()}


def transition_status(gs: GraspSet) -> str:
    """PARTIAL with a valid timing of any change of pose, MISSING without.
    Never more: no number of them was asked for."""
    return ("PARTIAL" if any(m.timing.validity == tr.VALID for m in gs.moves)
            else "MISSING")


def moves_summary(moves: Sequence[Move]) -> str:
    """"6 recordings, 4 containing a change of pose: 2 valid, 2
    questionable" for a list of recorded changes, and how many were not
    looked at because the hand was lost while forming, when any were."""
    n = {v: sum(1 for m in moves if m.timing.validity == v)
         for v in (tr.VALID, tr.QUESTIONABLE)}
    lost = sum(1 for m in moves if m.forming_lost)
    return (f"{_plural(len(moves), 'recording')}, "
            f"{sum(1 for m in moves if m.changed)} containing a change of "
            f"pose: {n[tr.VALID]} valid, {n[tr.QUESTIONABLE]} questionable"
            + (f"; {lost} not read, the hand lost while forming the grasp"
               if lost else ""))


def take_pairs(gs: GraspSet) -> List[Tuple[Take, Take]]:
    """Every pair of takes of one grasp that both have their angles."""
    out = []
    for gid in gs.item_ids:
        have = [t for t in gs.of(gid) if in_angle_table(t)]
        out += [(a, b) for i, a in enumerate(have) for b in have[i + 1:]]
    return out


def back_side(t: Take) -> bool:
    """The tracker's fit of the take has the back of the hand to the camera."""
    return any(n.startswith(BACK_OF_HAND_NOTE) for n in t.facts.pose_notes)


def opening_statement(gs: GraspSet) -> str:
    """The first paragraph, written from what the folder holds."""
    n = len(gs.items)
    have = sum(1 for i in gs.item_ids if gs.of(i))
    if recorded_everywhere(gs):
        first = (f"This folder contains Leap Motion / Ultraleap recordings of "
                 f"{n} grasp types taken from the reference grasp papers. Each "
                 "grasp has at least one accepted static-pose recording.")
    else:
        lacking = [i for i in gs.item_ids if not gs.of(i)]
        first = (f"This folder contains Leap Motion / Ultraleap recordings of "
                 f"{have} of the {n} grasp types taken from the reference "
                 "grasp papers. No accepted recording yet for: "
                 + ", ".join(lacking) + ".")
    parts = ["raw tracked joint data"]
    if all(t.keypoints is not None for t in gs.takes) and gs.takes:
        parts.append("representative static hand keypoints")
    parts.append("complete frame-level pose data")
    if gs.takes and all(t.joint_frames is not None and t.angles is not None
                        for t in gs.takes):
        parts += ["XYZ joint coordinates", "joint-local coordinate frames",
                  "and derived joint angles"]
        second = "The package includes " + ", ".join(parts) + "."
    else:
        second = ("The package includes " + ", ".join(parts)
                  + ". The joint-local coordinate frames and joint angles "
                  "are not in this build for every recording (section 18).")
    third = ("Transition time between different grasp poses is a separate "
             "measurement and is not inferred from these static recordings.")
    return " ".join([first, second, third])


# What the hand-in was asked to make easy to find, in the words it was asked
# in; the README opens with where each one is (`_where_asked`).
ASKED_TITLE = "WHERE EACH THING ASKED FOR IS"
ASKED_FOR = ("Recording duration",
             "Time to change from one pose to another",
             "XYZ coordinates",
             "Joint angles",
             "Mathematical definition / derivation of the angles",
             "Dataset for each individual pose",
             "XYZ coordinate plane / coordinate convention",
             "Representative final / static hand configuration",
             "Images of the grasp types",
             "Which joint each angle corresponds to")


def imaged_items(gs: GraspSet) -> List[str]:
    """The grasp types with at least one cut picture, in protocol order:
    the pages of grasp_images.pdf."""
    return [i for i in gs.item_ids if any(t.still_has_hand
                                          for t in gs.of(i))]


def images_where(gs: GraspSet) -> str:
    """Where the pictures are, as the map at the README's top says it."""
    n, total = len(gs.imaged), len(gs.takes)
    if not n:
        return ("Not in this build: no recording has a still with the "
                "recorded hand drawn on it (section 5).")
    pages = len(imaged_items(gs))
    return (f"{GRASPS_DIR}\\{IMAGES_DIR}\\<grasp>_{gs.hand}_take<N>.png: one "
            "infrared picture per recording"
            + ("" if n == total else
               f" whose still shows the hand ({n} of {total})")
            + f", cut to the recorded hand, and {GRASPS_DIR}\\{IMAGES_DIR}"
            f"\\{IMAGES_PDF}, one page per grasp type"
            + ("" if pages == len(gs.items) else
               f" with a picture ({pages} of {len(gs.items)})")
            + " (section 5).")


def images_header(gs: GraspSet) -> str:
    """The line under the README's title on pictures."""
    n, total = len(gs.imaged), len(gs.takes)
    if not n:
        return "No photographs are included."
    return ("Images: one infrared picture per recording"
            + ("" if n == total else f" whose still shows the hand ({n} of "
               f"{total})")
            + f", cut to the recorded hand, {GRASPS_DIR}\\{IMAGES_DIR} "
            "(section 5). No other photographs are included.")


def _where_asked(gs: GraspSet) -> List[str]:
    """Right under the opening statement: each thing that was asked for,
    the file that holds it and the section that explains it. Written from
    what the folder holds, like the opening statement."""
    h = gs.hand
    take = f"{GRASPS_DIR}\\<grasp>_{h}_take<N>"
    table = f"{GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\<grasp>_{h}_take<N>.csv"
    framed = bool(gs.takes) and all(
        t.joint_frames is not None and t.angles is not None
        for t in gs.takes)
    timed = sum(1 for m in gs.moves if m.timed)
    between = sum(1 for m in gs.moves
                  if OPEN_HAND not in (m.from_grasp, m.to_grasp))
    poses = sum(1 for i in gs.item_ids if gs.of(i))
    where = [
        f"{GRASPS_DIR}\\{SUMMARY_FILE}, column recording_duration_s: one "
        "row per recording, from the timestamps in it (section 6).",
        (f"{TRANSITIONS_DIR}\\{TRANSITIONS_FILE}, column "
         "transition_duration_s. Recorded separately, never taken from the "
         f"static recordings: {_plural(len(gs.moves), 'recording')} of a "
         f"change of pose, {timed} of them timed, "
         + (f"{between} between two grasp types" if between else
            "all from the open hand into a grasp and none between two "
            "grasp types")
         + " (section 17)."
         if gs.moves else
         "Not recorded. None is taken from the static recordings "
         "(section 17)."),
        f"Every frame: {take}.jsonl, 26 joints, metres, camera frame "
        f"(section 8). The representative frame, millimetres: "
        f"{take}_keypoints.txt, 21 landmarks, camera frame (section 9)"
        + (f", and {table}, 26 joints, wrist frame (section 11)."
           if framed else "."),
        (f"{table}, columns flex_deg abd_deg twist_deg of each of the 26 "
         f"joints (section 12). {GRASPS_DIR}\\{ANGLES_FILE}: the same "
         "under the 24 angle names of the reference paper, "
         f"{len(GIVEN_NAMES)} given and {len(NOT_GIVEN)} that this "
         "tracker does not measure left empty (section 13)."
         if framed else
         "Not in this build for every recording (section 18)."),
        "This file, section 12; against the reference paper's "
        "definitions, section 13.",
        f"{GRASPS_DIR}\\<grasp>_{h}_all_frames.txt: one file per grasp "
        f"type, {_plural(poses, 'file')}, every frame of every take of "
        "the grasp (section 14).",
        "This file, sections 10 and 11"
        + (f"; drawn on every page of {GRASPS_DIR}\\{JOINT_FRAMES_DIR}"
           "\\joint_frames.pdf." if gs.joint_frames_pdf is not None
           else "."),
        f"{take}_keypoints.txt: one frame per recording, taken from the "
        "stillest part of it, not the last frame (section 7).",
        images_where(gs),
        (f"{GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\{KEY_CSV} and {KEY_PDF}, and "
         "the table in section 11: for each of the 26 rows of a "
         "joint-frame table its joint, the bones before and after it and "
         "the paper's angle names it carries (sections 11 and 13)."
         if gs.has_joint_frames else
         "Not in this build: the session has no joint-frame tables "
         "(section 11)."),
    ]
    L = [ASKED_TITLE, "-" * 78]
    for name, text in zip(ASKED_FOR, where):
        L.append(f"  {name}")
        L += wrapped(text, "      ", "      ")
    return L + [""]


def _days(gs: GraspSet) -> List[str]:
    return sorted({d for d in (t.recorded_day for t in gs.takes) if d})


def _confirmed(gs: GraspSet) -> str:
    return ("confirmed by the operator" if gs.notes_confirmed
            else "not confirmed by the operator")


def notes_who(gs: GraspSet) -> str:
    """How the still notes were read, as the README and the PDF say it each
    time. The reading was made on the recorder's uncut stills (both hands
    and the room, in the session folder), not on the cut pictures of the
    folder (section 5)."""
    return (f"read by eye on {gs.notes_read_on} from the recorder's uncut "
            f"stills; {_confirmed(gs)}")


def objects_seen(gs: GraspSet) -> List[str]:
    """Section 1's paragraph on what was seen in the hands, from the still
    notes (`load_notes`); every count is of the notes and the takes. The
    notes were read off the recorder's uncut stills, which show both hands
    and are not handed in; the folder's pictures are cut to the recorded
    hand, so the lead says that what is reported for the other hand cannot
    be seen in them. "Nothing seen" counts only the pictures the notes say
    were read and name under none of the three kinds; a picture the notes
    do not say was read (`GraspSet.unread`, a take recorded after the
    reading) is on a line of its own, never among those that showed
    nothing. Every picture read: the same lines as before the notes listed
    what was read."""
    if not gs.imaged:
        return []
    if not gs.still_notes:
        return wrapped(
            "What was in or against the hand was not read from the "
            "infrared pictures (section 5); this folder says nothing about "
            "it.")
    hand_ = [t for t in gs.takes if t.still_note_hand]
    other = [t for t in gs.takes if t.still_note_other]
    unclear = [t for t in gs.takes if t.still_unclear]
    named = {t.stem for t in hand_ + other + unclear}
    rest = [t for t in gs.imaged if t.stem not in named and t.still_read]
    unread = gs.unread
    bare = [t for t in gs.takes if not t.still_has_hand
            and t.stem not in named]
    L = wrapped(
        f"What was seen in the hands, read by eye on {gs.notes_read_on} "
        f"({_confirmed(gs)}) from the recorder's uncut stills, which show "
        "both hands and the room and are kept in the session folder, not "
        "in this folder; the pictures in this folder (section 5) are cut to "
        "the recorded hand, so what is reported for the other hand cannot "
        "be seen in them:")
    L += wrapped(f"{_plural(len(hand_), 'recording')} with something in "
                 "or against the recorded hand" + (":" if hand_ else "."),
                 "    ", "      ")
    for t in hand_:
        L += wrapped(f"{t.stem}: {t.still_note_hand}", "      ", "        ")
    L += wrapped(f"{len(other)} with something in the other hand"
                 + (":" if other else "."), "    ", "      ")
    for t in other:
        L += wrapped(f"{t.stem}: {t.still_note_other}", "      ", "        ")
    L += wrapped(f"{len(unclear)} where the picture does not let one tell"
                 + (": " + ", ".join(t.stem for t in unclear) + "."
                    if unclear else "."), "    ", "      ")
    L += wrapped(f"the other {len(rest)} with a picture: nothing seen in "
                 "the hand.", "    ", "      ")
    if unread:
        L += wrapped(f"{len(unread)} with a picture not yet read by eye: "
                     + ", ".join(t.stem for t in unread) + ".", "    ",
                     "      ")
    if bare:
        L += wrapped(f"{len(bare)} without a picture of the hand: "
                     + ", ".join(t.stem for t in bare) + ".", "    ",
                     "      ")
    L += wrapped(
        "Which recordings were made with a real object in the hand is not "
        "known to this folder: a picture is one moment of a recording, "
        "read by eye.")
    return L


def _sec_purpose(gs: GraspSet) -> List[str]:
    L = _head(1, "PURPOSE")
    L += wrapped(
        "The task: using the Leap Motion, record the static hand pose of "
        "every grasp type shown in the reference papers. For each grasp, "
        "form the pose shown in the paper, hold it steadily for a few "
        "seconds, record the hand data, and label the recording with the "
        "grasp type. The task is mainly interested in the final hand "
        "configuration; it does not require the whole movement from the "
        "open hand into the grasp to be recorded.")
    L.append("")
    implied = [it for it in gs.items if it.get("object_implied")]
    desc = " ".join(str((gs.session.protocol or {}).get("description")
                        or "").lower().split())
    mimed = "mimed in the air" in desc and "no object" in desc
    L += wrapped(
        f"What is here: recordings of the operator's bare {gs.hand} hand "
        "held over the camera in each grasp, one file per recording that "
        "was kept (a \"take\"), each named after its grasp type."
        + (" The protocol asked for every grasp to be mimed in the air, "
           "with no object (its description says so)." if mimed else "")
        + (" The papers' pictures show the hand on an object in "
           + ("every grasp." if len(implied) == len(gs.items) else
              f"{len(implied)} of the {len(gs.items)} grasps.")
           if implied else ""))
    seen = objects_seen(gs)
    L += ([""] + seen + [""]) if seen else []
    L += wrapped(
        "What a label means: the grasp the operator was asked to make and "
        "judged, by eye against the paper's picture, to have made. No "
        "measurement in this folder checks a hand shape against a paper "
        "(section 15).")
    return L + [""]


def _sec_glance(gs: GraspSet) -> List[str]:
    L = _head(2, "DATASET AT A GLANCE")
    takes = gs.takes
    n_items = len(gs.items)
    have = sum(1 for i in gs.item_ids if gs.of(i))
    per = sorted({len(gs.of(i)) for i in gs.item_ids})
    rates = spread([t.facts.frame_rate_hz for t in takes])
    dur = duration_stats(gs)
    others = sum(1 for t in takes if t.facts.other_hand_detected)
    flagged = sum(1 for t in takes if t.facts.abnormal)
    moves = gs.moves
    speed = spread([t.facts.static_interval_mean_speed_mm_s for t in takes])
    shape = spread([t.facts.static_interval_shape_rms_mm for t in takes])
    size = spread([t.facts.index_metacarpal_mm for t in takes])
    static_s = gs.session.meta.get("static_s") or DEFAULT_STATIC_S
    spreads = [x[0] for x in (between_take_spread(gs.of(i))
                              for i in gs.item_ids) if x is not None]

    def row(name: str, value: str) -> None:
        L.extend(wrapped(f"{name:<34}{value}", "  ", " " * 36))

    row("Grasp types in the protocol", f"{n_items}; with an accepted "
        f"recording: {have}")
    row("Accepted recordings (takes)", f"{len(takes)}; per grasp: "
        + (", ".join(str(p) for p in per) if per else "none")
        + f" (section 18); target {gs.target} per grasp")
    row("Hand", f"{gs.hand}, bare, no glove; what the pictures show in the "
        "hand: section 1")
    row("Sensor", "Ultraleap Stereo IR 170 camera module (the \"Leap "
        "Motion\" of the task), desktop tracking mode")
    row("Tracking rate", "not measured" if rates is None
        else f"{rates[1]:.0f} frames per second (the tracker's own figure)")
    row("Recording duration", "not measured" if dur is None
        else f"{dur[0]:.2f} to {dur[2]:.2f} s, median {dur[1]:.2f} s, from "
             "timestamps (section 6)")
    row("Data per frame", "26 joints: XYZ position and orientation "
        "(sections 8 and 10)")
    row("Per recording", "one representative static frame: 21 keypoints, "
        f"26 joint frames, {len(GIVEN_NAMES)} of the paper's 24 angles "
        "(sections 7, 9, 11 to 13; section 13 says why "
        f"{len(NOT_GIVEN)} are not given)")
    row("Per grasp type", "every frame of every take in one file "
        "(section 14)")
    row("Pictures", f"{len(gs.imaged)} of {len(takes)} recordings: one "
        "infrared picture each, cut to the recorded hand (section 5)"
        if gs.imaged else "none (section 5)")
    if speed and shape:
        row("Stillness of the static frame",
            f"the stillest {float(static_s):g} s moved at {speed[0]:.0f} "
            f"to {speed[2]:.0f} "
            f"mm/s (median {speed[1]:.0f}); the hand's shape in it held to "
            f"{shape[1]:.1f} mm RMS (median) (section 7)")
    if spreads:
        row("Agreement between takes",
            "the takes of one grasp differ by up to "
            f"{max(spreads):.0f} degrees in a joint angle, by more than "
            f"{SPREAD_NOTE_DEG:g} in {sum(1 for v in spreads if v > SPREAD_NOTE_DEG)}"
            f" of the {len(spreads)} grasps with two or more takes "
            "(section 15)")
    if size:
        row("Hand size as tracked",
            f"index metacarpal {size[0]:.0f} to {size[2]:.0f} mm across "
            "recordings of the same hand (section 10)")
    row("Another hand in the file", f"{others} of {len(takes)} recordings; "
        "its joints are in no derived file (section 16)")
    row("Recordings with a note", f"{flagged} (section 15)")
    row("Transition timing", "no change of pose was recorded" if not moves
        else moves_summary(moves) + "; "
        + (f"{len(gs.between_grasps)} of them" if gs.between_grasps
           else "none") + " between two grasp types (section 17)")
    row("Recorded on", ", ".join(_days(gs)) or "not recorded")
    return L + [""]


def _sec_types(gs: GraspSet) -> List[str]:
    L = _head(3, "GRASP TYPES AND PAPER SOURCES")
    L += wrapped(
        "grasp_name is the label every file of the grasp carries. The same "
        f"list as a table, with the datasets present for each grasp, is "
        f"{GRASPS_DIR}\\{AUDIT_FILE}.")
    for it in gs.items:
        gid = str(it.get("id"))
        src = str(it.get("source") or "")
        fig = str(it.get("figure") or "")
        where = ": ".join(x for x in (src, fig) if x) or "source not recorded"
        L.append(f"  {gid}   ({_plural(len(gs.of(gid)), 'take')})")
        L += wrapped(". ".join(x for x in (str(it.get("label") or ""), where)
                               if x), "      ", "      ")
    cited = citations(gs.session)
    if cited:
        L.append("")
        L.append("  The papers, as the protocol file cites them:")
        for c in cited:
            L += wrapped(c, "    ", "      ")
    linked, distinct = configurations(gs.items)
    if linked:
        L.append("")
        L += wrapped(
            f"{linked} of the {len(gs.items)} grasps are, by our reading of "
            "the papers' pictures, the same hand configuration as another "
            f"grasp under another paper's name: {distinct} distinct "
            "configurations. Each was still recorded separately under its "
            f"own label; same_configuration_as in {AUDIT_FILE} lists which.")
    if gs.source_notes:
        L.append("")
        L.append("  Notes on the sources:")
        for note in gs.source_notes:
            L += wrapped(note, "    ", "      ")
    return L + [""]


# What every column of grasps_summary.csv holds, for the README's list. A
# column without an entry fails `column_help`: nothing is handed in
# undefined.
COLUMN_HELP = {
    "grasp_name": "the label: the grasp type (section 3)",
    "grasp_label": "the grasp's name in words",
    "source_paper": "the paper it is from",
    "source_figure": "where in that paper",
    "hand": "the operator's hand",
    "take": "the take number of this grasp",
    "raw_jsonl_file": "the recording (section 8)",
    "representative_keypoints_file": "its representative frame as 21 "
                                     "landmarks (sections 7, 9)",
    "all_frames_file": "the grasp's frame-by-frame dataset (section 14)",
    "joint_frames_file": "the representative frame's 26-row table "
                         "(section 11)",
    "image_file": "the recording's infrared picture, cut to the recorded "
                  "hand (section 5); empty when its still does not show "
                  "the hand",
    "transition_capture_file": "the recording of the change into the "
                               "grasp, when one was made (section 17)",
    "sensor": "what recorded it: \"camera\"",
    "recording_duration_s": "last minus first wall_time of the labelled "
                            "hand's frames (section 6)",
    "camera_clock_duration_s": "the same span on the tracker's frame clock",
    "recorder_window_s": "how long the recorder kept the file open; copied "
                         "from the recorder",
    "unique_frame_count": "camera frames in the file",
    "hand_frame_count": "frames in which the labelled hand was tracked",
    "jsonl_row_count": "lines in the file, every hand",
    "frame_rate_hz": "the tracker's own frames per second (median)",
    "tracked_percent": "the gate's figure (section 15); copied from the "
                       "recorder",
    "largest_gap_s": "longest step between two frames of the labelled hand",
    "static_interval_start_s": "start of the stillest window, s from the "
                               "hand's first frame (section 7)",
    "static_interval_end_s": "its end",
    "static_interval_duration_s": "its length",
    "static_interval_mean_speed_mm_s": "mean speed of the 26 joints over "
                                       "it, camera frame (section 7)",
    "static_interval_shape_rms_mm": "how far the hand's shape was from the "
                                    "representative frame inside it: RMS "
                                    "over 26 joints after a rigid fit, "
                                    "median over the window",
    "static_interval_shape_rms_max_mm": "the same, the largest in the "
                                        "window",
    "static_interval_pose_speed_mm_s": "median over it of the pose speed, "
                                       "the joints' speed with the whole "
                                       "hand's movement taken out (section "
                                       "17)",
    "static_interval_first_frame_id": "the tracker's number of the static "
                                      "interval's first frame (the 'Frame' "
                                      "number of the text files)",
    "static_interval_last_frame_id": "and of its last",
    "take_shape_rms_max_mm": "the furthest any frame of the whole "
                             "recording is from the representative "
                             "frame's shape, measured as the two columns "
                             "above (section 14)",
    "frames_within_5mm_percent": "share of the recording's frames within "
                                 f"{SHAPE_CHANGE_MM:g} mm RMS of that shape",
    "shape_rms_start_mm": "the same distance, mean over the recording's "
                          f"first {EDGE_S:g} s",
    "shape_rms_end_mm": f"and over its last {EDGE_S:g} s",
    "representative_frame_index": "line of the representative frame in "
                                  "the .jsonl, from 0",
    "representative_frame_id": "the tracker's number for that frame",
    "representative_frame_time_s": "its time, s from the hand's first frame",
    "other_hand_detected": "true when another hand is in the file "
                           "(section 16)",
    "other_hand_frame_fraction": "share of the frames in which it appears",
    "other_hand_median_distance_cm": "how far its palm was from the "
                                     "labelled hand's",
    "other_hand_finger_curl_minus_own_deg": "the sum of the four fingers' "
                                            "MCP + PIP + DIP flexion of the "
                                            "other hand minus that of the "
                                            "labelled hand, each at its "
                                            "frame nearest the middle of "
                                            "the recording, degrees; empty "
                                            "when the other hand is on "
                                            "fewer than "
                                            f"{100 * OTHER_HAND_MIN_SHARE:.0f}"
                                            " % of the frames (section 16)",
    "hand_selection": "how the labelled hand was told apart (section 16)",
    "hand_ids": "the tracker ids of the labelled hand's lines",
    "tracker_label_on_representative_frame": "left or right, as the "
                                             "tracker fitted that frame",
    "tracker_id_changes": "times the tracker gave the hand a new id",
    "identity_flag": "why the hand's identity needs a note, or empty",
    "status": "\"accepted\": only kept recordings are here",
    "decided_by": "who kept it; copied from the recorder",
    "reacquisitions_in_static_interval": "times the hand was lost and "
                                         "found inside the static interval "
                                         "(the gate wants 0); copied",
    "reacquisitions": "the same over the whole recording; copied",
    "grab_strength": "the tracker's 0 to 1 measure of a fist on the "
                     "representative frame; copied",
    "pinch_strength": "its 0 to 1 measure of a pinch; copied",
    "curl_thumb": "wrist-to-fingertip distance over wrist-to-middle-knuckle "
                  "distance on the representative frame: SMALLER = more "
                  "curled; copied",
    "curl_index": "the same for the index finger",
    "curl_middle": "the middle finger",
    "curl_ring": "the ring finger",
    "curl_pinky": "the little finger",
    "palm_height_cm": "the palm's height above the camera on the "
                      "representative frame; copied",
    "view_angle_deg": "angle between the palm's normal and the line from "
                      "the palm to the camera on that frame: 0 = palm "
                      "square to the lens, over 90 = the back of the hand "
                      "toward it; copied",
    "index_metacarpal_mm": "length the tracker gave the index metacarpal, "
                           "median over the recording: its idea of the "
                           "hand's size (section 10)",
    "index_metacarpal_range_mm": "largest minus smallest of that length "
                                 "inside the recording",
    "first_wall_time": "wall_time of the labelled hand's first frame "
                       "(seconds since 1970)",
    "static_interval_start": "the static interval's start as a wall_time; "
                             "copied",
    "static_interval_end": "its end as a wall_time; copied",
    "medoid_wall_time": "the representative frame's wall_time; copied",
    "orientation_note": "the operator's note on how the hand was turned",
    "session": "the recording session",
    "recorder_take_name": "the take's name in the session folder",
    "notes": "everything a reader should know before using the recording "
             "(section 15)",
}


def column_help(columns: Sequence[str]) -> List[str]:
    """README lines "name   meaning" for `columns`, in order."""
    out = []
    for c in columns:
        out += wrapped(f"{c}: {COLUMN_HELP[c]}", "    ", "        ")
    return out


def _sec_files(gs: GraspSet) -> List[str]:
    h = gs.hand
    L = _head(4, "FOLDER / FILE STRUCTURE")
    rows = [
        ("README.txt", "this file"),
        (VALIDATION_FILE, "the check of this folder, written by the "
         "rebuild (section 20)"),
        (f"{GRASPS_DIR}\\{SUMMARY_FILE}", "one row per recording; its "
         "columns are listed below"),
        (f"{GRASPS_DIR}\\{AUDIT_FILE}", "one row per grasp type: paper and "
         "figure, recordings against the target, which datasets exist, how "
         "many of its recordings carry no note "
         "(recordings_without_a_note) and how many have a picture "
         "(recordings_with_an_image), and "
         "two statuses kept apart: static_pose_status (READY or "
         "INCOMPLETE) and multi_take_status (COMPLETE or INCOMPLETE) "
         "(section 18)"),
        (f"{GRASPS_DIR}\\<grasp>_{h}_take<N>.jsonl", "the recording: every "
         "frame, 26 joints (section 8)"),
        (f"{GRASPS_DIR}\\<grasp>_{h}_take<N>_keypoints.txt", "the "
         "representative static hand configuration of the recording: one "
         "frame, 21 landmarks, millimetres (sections 7 and 9)"),
        (f"{GRASPS_DIR}\\<grasp>_{h}_all_frames.txt", "the dataset of the "
         "pose: every frame of every take of the grasp, 21 landmarks, "
         "millimetres (section 14)"),
    ]
    if gs.has_joint_frames:
        rows += [
            (f"{GRASPS_DIR}\\{ANGLES_FILE}", "one row per recording: the "
             "paper's 24 angle names for the representative frame, 18 "
             "filled and 6 empty because this tracker does not give them, "
             "and what it does measure of those joints under other names "
             "(section 13)"),
            (f"{GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\<grasp>_{h}_take<N>.csv",
             "the representative frame as 26 rows: XYZ position, "
             "orientation, the three axes and the angles of every joint "
             "(sections 10 to 12)"),
            (f"{GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\joint_frames.pdf",
             "one page per recording: the hand drawn with every joint's "
             "+x +y +z axes, the paper's 24 angle names with the 18 "
             "values this tracker gives, and the 26-row table"),
            (f"{GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\{KEY_CSV}",
             "the joint key: one row per row of a joint-frame table, its "
             "joint, the bones before and after it, and the paper's angle "
             "names it carries (section 11)"),
            (f"{GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\{KEY_PDF}",
             "the same key on one page, with a flat left hand and its 26 "
             "joints numbered (section 11)"),
        ]
    if gs.imaged:
        rows += [
            (f"{GRASPS_DIR}\\{IMAGES_DIR}\\<grasp>_{h}_take<N>.png",
             "the recording's infrared picture, cut to the recorded hand "
             "(section 5)"),
            (f"{GRASPS_DIR}\\{IMAGES_DIR}\\{IMAGES_PDF}",
             "the same pictures, one page per grasp type"
             + ("" if len(imaged_items(gs)) == len(gs.items) else
                " with a picture")
             + ", each captioned with its recording (section 5)"),
        ]
    rows.append((f"{TRANSITIONS_DIR}\\{TRANSITIONS_FILE}",
                 "the transition-timing dataset: one row per recorded "
                 "change of pose (section 17)"))
    if gs.from_open:
        rows.append((f"{TRANSITIONS_DIR}\\{OPEN_HAND}_to_<grasp>_{h}_take<N>"
                     ".jsonl", "the recorded change from the open hand "
                     "into the grasp, from the cue to the start of the "
                     "static recording, same line format"))
    if gs.recorded_moves:
        rows.append((f"{TRANSITIONS_DIR}\\<from>_to_<to>_<hand>_take<N>"
                     ".jsonl", "a recorded change from one held pose into "
                     "another, from before the cue to the end of the hold, "
                     "same line format"))
    for name, what in rows:
        L.append(f"  {name}")
        L += wrapped(what, "      ", "      ")
    L += wrapped(
        "File names in the CSV tables are relative to this folder, with "
        "forward slashes. <grasp> is a grasp_name of section 3, <N> the "
        "take number. A file in this folder that is not listed here was not "
        "written by the build.")
    L.append("")
    L += wrapped(
        f"Columns of {SUMMARY_FILE}, in order. \"copied\" marks a value "
        "taken from the recorder's own record of the take (its meta file) "
        "and not measured again by the rebuild. The static interval and "
        "the representative frame are the recorder's choice too: the "
        "rebuild finds them in the recording, and the check runs the rule "
        "on the recording again (section 15). Every other number is "
        "measured from the recording at each rebuild.")
    L += column_help(SUMMARY_COLUMNS)
    return L + [""]


def _sec_capture(gs: GraspSet) -> List[str]:
    s, meta = gs.session, gs.session.meta
    proto = s.protocol or {}
    dur = meta.get("duration_s") or proto.get("duration_s") or 5.0
    static_s = meta.get("static_s") or DEFAULT_STATIC_S
    min_tracked = meta.get("min_tracked_fraction") or MIN_TRACKED_FRACTION
    heights = spread([t.palm_cm[1] for t in gs.takes if t.palm_cm])
    aside = spread([math.hypot(t.palm_cm[0], t.palm_cm[2])
                    for t in gs.takes if t.palm_cm])
    aside_coached = spread([math.hypot(t.palm_cm[0], t.palm_cm[2])
                            for t in gs.takes if t.palm_cm and t.coached])
    rates = spread([t.facts.frame_rate_hz for t in gs.takes])
    coached = [t for t in gs.takes if t.coached]
    plain = [t for t in gs.takes if not t.coached]
    ready = [t for t in coached
             if (t.meta.get("coaching") or {}).get("ready_pause_s")]
    redone = gs.rejected.get("operator redo", 0)
    ended = (gs.rejected.get("operator quit at the review", 0)
             + gs.rejected.get("interrupted (Ctrl+C)", 0))
    L = _head(5, "HOW EACH RECORDING WAS CAPTURED")
    L += wrapped(
        "Sensor: camera only. An Ultraleap Stereo IR 170 module (the \"Leap "
        "Motion\" of the task) lay flat on the table, lenses up, in the "
        "tracker's desktop mode. The Ultraleap tracking service fitted a "
        "hand skeleton to its two infrared images "
        + ("many" if rates is None else f"about {rates[1]:.0f}")
        + " times a second; "
        "what is recorded is that skeleton, not the images. The operator's "
        f"bare {gs.hand} hand was held over the table above the module"
        + ("" if heights is None or aside is None else
           f": its palm {heights[0]:.0f} to {heights[2]:.0f} cm above the "
           f"module's level (median {heights[1]:.1f} cm, each recording's "
           f"own median) and {aside[0]:.0f} to {aside[2]:.0f} cm to the "
           f"side of its vertical axis (median {aside[1]:.0f} cm"
           + ("" if aside_coached is None else
              f"; {aside_coached[1]:.0f} cm in the coached recordings")
           + ")")
        + ". No glove. What was in or against the hand: section 1.")
    L.append("")
    L += wrapped(
        f"One recording: the grasp was held and the recorder kept every "
        f"frame the tracker produced for {float(dur):g} s. The paper's "
        "picture of the grasp was on screen throughout. A recording is "
        "kept only if it passes two decisions:")
    L += wrapped(
        f"the acquisition gate, automatic: the hand tracked on at least "
        f"{100 * float(min_tracked):.0f} % of the frames the tracker "
        "produced over the recording, not lost and found again inside the "
        f"static interval (section 7), and, in the coached procedure, "
        f"fitted as a {gs.hand} hand on at least "
        f"{100 * MIN_LABEL_SHARE:.0f} % of that interval's frames. The "
        "gate is about tracking. It sets no limit on how much the hand "
        "moved;", "    a. ", "       ")
    L += wrapped(
        "the operator's review: a still infrared image from the camera "
        + ("(in this folder cut to the recorded hand, THE PICTURES below) "
           if gs.imaged else "(kept in the session folder, not handed in) ")
        + "and the recording's numbers were shown beside the paper's "
        "picture, and the operator kept the recording or asked for it "
        "again.", "    b. ", "       ")
    L += wrapped(
        "An attempt that failed the gate was recorded again; failed "
        "attempts are kept in the session folder with the reason and are "
        "not in this folder. "
        + ("At the review the operator asked for no recording again"
           if not redone else
           "At the review the operator asked for "
           f"{_plural(redone, 'recording')} again")
        + (f"; {_plural(ended, 'attempt')} that had passed the gate "
           f"{'was' if ended == 1 else 'were'} dropped because the sitting "
           "was ended there (the operator quit, or interrupted it)"
           if ended else "")
        + ". "
        + "That a kept hand shape matches the paper's picture is the "
        "operator's judgement at the time of recording; nothing in this "
        "folder measures it.")
    L.append("")
    if coached:
        L += wrapped(
            f"Coached procedure ({_plural(len(coached), 'recording')}): on "
            "the first day the tracker lost hands that closed edge-on to "
            "the lens and did not find a hand that was already closed, so "
            "each attempt went: OPEN HAND, the open palm held to the camera "
            "until the tracker had it; MAKE THE GRASP, a high beep and a "
            "few seconds to close the hand into the grasp while it stayed "
            f"tracked; HOLD STILL, the {float(dur):g} s recording. The hand "
            "the tracker acquired at OPEN HAND is followed by its tracker "
            "id through the recording, by the rule of id and distance in "
            "section 16, so the frames measured are the ones that rule "
            "gives that hand, whatever else was in view.")
        if ready:
            pause = _span_text(_coaching(t.meta).get("ready_pause_s")
                               for t in ready)
            L += wrapped(
                f"In {len(ready)} of them a READY step held the open hand "
                f"still for {pause} between the two beeps, so that the hand "
                "was at rest when the high beep sounded (section 17).")
        elif gs.with_forming:
            L += wrapped(
                "The READY step of the present recorder (the open hand "
                "held still for 1 s before the high beep) was added after "
                "these recordings: none of them had it (section 17).")
        L += _rules_change(gs, coached)
    if plain:
        L += wrapped(
            f"Plain countdown ({_plural(len(plain), 'recording')}, made "
            "before the coached procedure): a countdown, then the "
            f"{float(dur):g} s recording of a hand already in the grasp. "
            "The hand measured is the one carrying the tracker's "
            f"{gs.hand} label (section 16).")
    L.append("")
    n_rej = sum(gs.rejected.values())
    waits = sum(n for k, n in gs.rejected.items()
                if k.lower().startswith("no open hand acquired"))
    resumed = meta.get("resumed") or []
    L += wrapped(
        f"Session {s.name}: started {meta.get('started', '?')}, last "
        f"written {meta.get('ended', '?')}"
        + (f", continued {len(resumed)} time(s) into the same folder"
           if resumed else "")
        + f"; operator {meta.get('operator') or 'not recorded'}; protocol "
        f"{meta.get('protocol_name', 'grasps')} version "
        f"{meta.get('protocol_version', '?')}. Accepted {len(gs.takes)}, "
        f"not kept {n_rej}"
        + (f" ({waits} of those are waits for an open hand that ran out, "
           "not recordings that failed: by the recorder's own record of "
           f"each, it saw no hand at all in {gs.waits_no_hand} of them"
           + (f" and a hand it did not accept in {gs.waits_other}"
              if gs.waits_other else "") + ")"
           if waits else "") + ".")
    L += wrapped(
        f"The static interval looked for in each recording is "
        f"{float(static_s):g} s long (section 7).")
    L += _pictures(gs, float(dur))
    return L + [""]


def _span_text(values, unit: str = "s") -> Optional[str]:
    """"4 s", or "3 to 4 s" when the recordings differ; None without one."""
    got = sorted({float(v) for v in values if v is not None})
    if not got:
        return None
    if len(got) == 1:
        return f"{got[0]:g} {unit}".strip()
    return f"{got[0]:g} to {got[-1]:g} {unit}".strip()


def _rules_change(gs: GraspSet, coached: Sequence[Take]) -> List[str]:
    """Section 5's paragraph on the recorder's rules of 2026-10-07, written
    only when some coached recording here was made under them, and from
    the recordings' own metas (`waited_for_rules`, `recorder_rules` and the
    limits in `coaching`): since then a hand lost while forming is waited
    for, and a close before the GO beep resets the attempt."""
    waited = [t for t in coached if t.waited_rules]
    if not waited:
        return []
    dates = sorted({str(t.meta["recorder_rules"]) for t in waited
                    if t.meta.get("recorder_rules")})
    c = [_coaching(t.meta) for t in waited]
    wait = _span_text(x.get("lost_wait_s") for x in c)
    window = _span_text(x.get("form_window_s") for x in c)
    relock = _span_text(x.get("relock_s") for x in c)
    limit = _span_text((x.get("reset_limit") for x in c), "")
    kept = [t for t in gs.takes if t.forming_lost]
    # the rules' date as the metas give it (never a date written here); a
    # meta with the new coaching keys and no recorder_rules is named by them
    when, ref = ((f"of {' and '.join(dates)}", "recorder_rules")
                 if dates else ("that wait for a hand lost while forming",
                                "coaching.forming_lost_count"))
    how = "; ".join(x for x in (
        f"up to {wait} each time" if wait else "",
        f"no later than {window} after the GO beep" if window else "") if x)
    text = (
        (f"The recorder's rules changed during these recordings: "
         f"{len(waited)} of the {len(coached)} coached recordings were made "
         f"under its rules {when} ({ref} in each recording's meta), the "
         "rest before them. "
         if len(waited) < len(coached) else
         f"All {len(coached)} coached recordings were made under the "
         f"recorder's rules {when} ({ref} in each recording's meta), which "
         "changed the coached procedure. ")
        + "Since that change a hand lost while the grasp was being "
        "formed no longer rejects the attempt: the recorder waits for the "
        "tracker to find the hand again"
        + (f" ({how})" if how else "")
        + (f", holds it tracked for {relock}" if relock else "")
        + ", and then records the static hold under the same acquisition "
        "gate (a. above), which did not change. "
        # the recorder's FormWatch.add (scripts/leap/record_poses.py): which
        # id it follows after a loss, stated as the rule it is
        "The tracker's next hand is treated as the same hand when it has "
        "the same tracker id, or a new id seen in a frame without the hand, "
        f"within {FOLLOW_RADIUS_M * 100:.0f} cm of where the hand was lost "
        "and not set aside before as another hand (in view at OPEN HAND, "
        "seen beside it, or further away): a rule of id and distance, not a "
        "proof of identity. "
        + (f"{_plural(len(kept), 'recording')} here "
           f"{'was' if len(kept) == 1 else 'were'} kept that way: "
           "the hand was lost while forming and found again before the "
           "hold (section 15); "
           f"{'its forming file is' if len(kept) == 1 else 'their forming files are'}"
           " not timed (section 17). "
           if kept else "In none of the recordings here was the hand lost "
           "while forming. ")
        # the recorder's own words (record_poses.py): "a close before the GO
        # cue, or a loss during READY, resets the attempt"
        + "A hand that closed before the GO beep, or was lost during READY, "
        "goes back to OPEN HAND"
        + (f" up to {limit} times" if limit else "")
        + " instead of the attempt being dropped at the third time"
        + _cues_text(gs) + ".")
    return wrapped(text)


def _cues_text(gs: GraspSet) -> str:
    """", the READY cue two short 520 Hz beeps ..." from session.json's
    `coaching.cues`, or "" when the session does not record its cues."""
    cues = (gs.session.meta.get("coaching") or {}).get("cues") if isinstance(
        gs.session.meta.get("coaching"), dict) else None
    if not isinstance(cues, dict):
        return ""
    ready, go = cues.get("ready") or {}, cues.get("go") or {}
    try:
        return (f"; the READY cue is {int(ready['beeps'])} short beeps "
                f"({float(ready['hz']):g} Hz, {float(ready['ms']):g} ms, at "
                f"{100 * float(ready['volume']):.0f} % of full scale) with "
                "KEEP HAND OPEN on screen, the GO cue one "
                f"{float(go['ms']):g} ms beep at {float(go['hz']):g} Hz and "
                f"{100 * float(go['volume']):.0f} % of full scale")
    except (KeyError, TypeError, ValueError):
        return ""


# A still is said to be written about the middle of its recording when it
# was written this far through it (whole percent, from the recorder's
# take_start to its take_end); a still outside is named with its share.
STILL_MIDDLE_PERCENT = (40, 60)


def still_shares(gs: GraspSet) -> List[Tuple[Take, int]]:
    """(take, how far through its recording its still was written, whole
    percent) for each take whose still has a file time and whose meta a
    start and an end: the still's offset (`still_offset`) over the meta's
    take_end minus take_start. Measured, never assumed."""
    out = []
    for t in gs.takes:
        if t.still is None or t.still_offset_s is None:
            continue
        try:
            length = float(t.meta["take_end"]) - float(t.meta["take_start"])
        except (KeyError, TypeError, ValueError):
            continue
        if length > 0:
            out.append((t, int(f"{100.0 * t.still_offset_s / length:.0f}")))
    return out


def _still_share_text(gs: GraspSet) -> str:
    """"; that is 50 to 66 % of the way through its recording ...: about
    the middle (40 to 60 %) in 114 of the 115, later in 1: <take> (66 %)",
    from `still_shares`; "" when no still has a share."""
    shares = still_shares(gs)
    if not shares:
        return ""
    lo, hi = STILL_MIDDLE_PERCENT
    pcs = [p for _t, p in shares]
    early = [(t, p) for t, p in shares if p < lo]
    late = [(t, p) for t, p in shares if p > hi]
    text = (f"; that is {min(pcs)} to {max(pcs)} % of the way through its "
            "recording (from the recorder's start of it to its end): about "
            f"the middle ({lo} to {hi} %) in ")
    if not early and not late:
        return text + f"all {len(shares)}"
    inside = len(shares) - len(early) - len(late)

    def named(group) -> str:
        return ", ".join(f"{t.item} take {t.ref.take} ({p} %)"
                         for t, p in group)
    parts = [f"{word} in {len(group)}: {named(group)}"
             for word, group in (("earlier", early), ("later", late))
             if group]
    return text + f"{inside} of the {len(shares)}, " + "; ".join(parts)


def _pictures(gs: GraspSet, dur: float) -> List[str]:
    """Section 5's paragraph on the pictures: what a still is, when it was
    written (file times), and how this build cuts it (`grasp_images`)."""
    if not gs.takes:
        return []
    L = [""]
    without = [t for t in gs.takes if not t.still_has_hand]
    if not gs.imaged:
        return L + wrapped(
            "THE PICTURES. None in this build: no recording has a still "
            f"with the {gs.hand} hand's skeleton drawn on it.")
    offs = spread([t.still_offset_s for t in gs.takes
                   if t.still is not None])
    colour = "red" if gs.hand == "right" else "cyan"
    when = ""
    if offs is not None:
        when = (f" {offs[0]:.2f} to {offs[2]:.2f} s after the recording "
                "began (the time of the still's file minus the recorder's "
                "start of the recording, both on the recording computer's "
                f"clock), median {offs[1]:.2f} s" + _still_share_text(gs))
    side = "right" if gs.hand == "right" else "left"

    def band(b) -> str:
        return ", ".join(f"{name} {lo} to {hi}"
                         for name, (lo, hi) in zip("RGB", b))
    L += wrapped(
        "THE PICTURES. One infrared frame per recording from the camera's "
        "left lens, as the viewer showed it at the review: mirrored, with "
        "the tracker's skeleton drawn on it, cyan on a hand it fitted as "
        f"left and red on one fitted as right (the recorded {gs.hand} hand "
        f"is the {colour} one). The viewer wrote it to the session folder"
        + when + ". For this folder each is cut to the recorded hand by "
        "one rule, in this order:")
    steps = [
        "The skeleton: the pixels whose R, G and B (0 to 255) lie in "
        f"{band(gi.SKELETON_BAND[side])}, a band around the viewer's "
        f"{colour} {tuple(gi.SKELETON_COLOUR[side])}, wide because the "
        "lines are drawn anti-aliased.",
        "The hand's radius: the median distance of the skeleton's pixels "
        "from their median point.",
        f"Kept: a pixel within {gi.CORE:g} radii of the nearest skeleton "
        f"pixel, or within {gi.REACH:g} radii and at least {gi.GATE:g} "
        "times as bright as the hand. A pixel's brightness is the mean of "
        "its R, G and B; the hand's is the median brightness of the "
        f"pixels within {gi.LEVEL_PX} pixels of the skeleton that are "
        "drawn neither in the skeleton's band nor in the caption's yellow "
        f"({band(gi.CAPTION_BAND)}). The distances are OpenCV's distance "
        f"transform (L2, a {gi.DISTANCE_MASK} x {gi.DISTANCE_MASK} mask).",
        f"Closed: the kept set is closed with a {gi.CLOSE_PX} by "
        f"{gi.CLOSE_PX} pixel window (dilated, then eroded). A hole or gap "
        "narrower than that between kept pixels is filled, so a few "
        "pixels the two tests of step 3 alone would drop stay.",
        "Connected: of that, only the part connected to the skeleton "
        "(each pixel to its 8 neighbours) is kept.",
        "Every other pixel is set to black, and the picture is cut to the "
        "bounding box of what is kept.",
    ]
    for k, text in enumerate(steps, 1):
        L += wrapped(text, f"    {k}. ", "       ")
    L += wrapped(
        "This rule, applied to the still, gives the picture pixel for "
        "pixel. What is left is the hand, a strip of the forearm under the "
        "drawn arm line, and whatever was in or against the hand. The "
        "operator's face and the room, further from the skeleton, are cut "
        "away; the cut is by distance and brightness, not a recognition of "
        "faces.")
    L += wrapped(
        "The pictures are for the eye only: no number in this folder is "
        "measured from them. The skeleton on a picture is the tracker's "
        "fit at the moment the still was written, not the representative "
        "frame of section 7. What was seen in the hands, read by eye"
        + (", and which pictures have not been read yet" if gs.unread
           else "")
        + ": section 1."
        # the reading was not made on these pictures: they show one hand
        + (" The reading by eye of section 1 was made on the uncut stills, "
           "which show both hands and the room and are kept in the session "
           "folder, not in this folder." if gs.still_notes else ""))
    if without:
        L += wrapped(
            f"{_plural(len(without), 'recording has', 'recordings have')} "
            "no picture: "
            + "; ".join(f"{t.stem} ("
                        + ("no still" if t.still is None else
                           f"no {gs.hand} skeleton drawn on its still")
                        + ")" for t in without) + ".")
    return L


def _sec_duration(gs: GraspSet) -> List[str]:
    L = _head(6, "RECORDING DURATION")
    takes = gs.takes
    L += wrapped(
        "recording_duration_s = the wall_time of the last frame of the "
        "labelled hand minus the wall_time of its first frame. wall_time is "
        "the recording computer's clock, in seconds, written on every line. "
        "It is computed from the timestamps in the recording and is in "
        f"{GRASPS_DIR}\\{SUMMARY_FILE} for every recording.")
    L.append("")
    L += wrapped(
        "The number of lines in a .jsonl file is NOT used to calculate "
        "recording time, and must not be. The tracker writes one line per "
        "tracked hand per frame, so a recording with a second hand in view "
        "has up to twice the lines of one without, for the same seconds.")
    rows = spread([t.facts.jsonl_row_count for t in takes])
    frames = spread([t.facts.hand_frame_count for t in takes])
    if rows and frames:
        L += wrapped(
            f"In this folder the files have {rows[0]:.0f} to {rows[2]:.0f} "
            f"lines; the labelled hand has {frames[0]:.0f} to "
            f"{frames[2]:.0f} frames per recording.")
    L.append("")
    L += wrapped(f"Over the {len(takes)} recordings:")
    L.append("  recording_duration_s      "
             + _range([t.facts.recording_duration_s for t in takes], 3, " s"))
    L.append("  camera_clock_duration_s   "
             + _range([t.facts.camera_clock_duration_s for t in takes], 3,
                      " s"))
    L.append("  recorder_window_s         "
             + _range([t.facts.recorder_window_s for t in takes], 3, " s"))
    L.append("")
    both = [(t.facts.recording_duration_s, t.facts.camera_clock_duration_s,
             t.facts.recorder_window_s) for t in takes]
    clocks = [abs(a - b) for a, b, _w in both
              if a is not None and b is not None]
    short = spread([w - a for a, _b, w in both
                    if a is not None and w is not None])
    over = [b - w for _a, b, w in both if b is not None and w is not None]
    L += wrapped(
        "The first two columns are one span on two clocks. "
        "recording_duration_s is on wall_time, stamped when each line was "
        "written. camera_clock_duration_s is the same first-to-last span "
        "on the tracker's own frame clock (timestamp_us, microseconds), "
        "the cleaner clock for intervals inside one recording, because the "
        "tracker's frames reach the recorder in small bursts."
        + (f" The two differ by up to {max(clocks):.3f} s here."
           if clocks else ""))
    L += wrapped(
        "recorder_window_s is a different quantity: how long the recorder "
        "kept the file open (take_end minus take_start in its own record "
        "of the take, on the computer's clock)."
        + ("" if short is None else
           f" The wall_time span is shorter than that window by "
           f"{short[0]:.3f} to {short[2]:.3f} s (median {short[1]:.3f}); "
           "the larger differences are recordings in which the hand was "
           "not tracked at one end.")
        + (f" The camera-clock span is longer than the window in "
           f"{_plural(sum(1 for v in over if v > 0), 'recording')}, by "
           f"{'' if sum(1 for v in over if v > 0) == 1 else 'up to '}"
           f"{max(over):.3f} s. The two are different clocks, the "
           "camera's for when a frame was captured and the computer's for "
           "when the file was open, and nothing in this folder depends on "
           "comparing them."
           if over and max(over) > 0 else ""))
    L.append("")
    odd = [t for t in takes if t.facts.timing_notes]
    if odd:
        L += wrapped(
            f"Recordings whose timing is not the ordinary one "
            f"({len(odd)}; a span more than {SHORT_BY_S:g} s short of the "
            f"recorder's window, or a hole longer than {LOSS_GAP_S:g} s). "
            "They passed the acquisition gate and the review, and are kept "
            "as recorded:")
        for t in odd:
            L += wrapped(f"{t.stem}: " + "; ".join(t.facts.timing_notes),
                         "    ", "      ")
    else:
        L += wrapped(
            "No recording has a span more than "
            f"{SHORT_BY_S:g} s short of the recorder's window or a hole "
            f"longer than {LOSS_GAP_S:g} s.")
    return L + [""]


def _sec_static(gs: GraspSet) -> List[str]:
    static_s = gs.session.meta.get("static_s") or DEFAULT_STATIC_S
    takes = gs.takes
    L = _head(7, "STATIC-POSE / REPRESENTATIVE-FRAME SELECTION")
    L += wrapped(
        f"<grasp>_{gs.hand}_take<N>_keypoints.txt holds the "
        "REPRESENTATIVE STATIC HAND CONFIGURATION of a recording: one frame "
        "that was actually recorded, chosen from the stillest part of the "
        "recording. It is not the last frame and not an average. It is "
        "chosen in two steps, by the recorder, at the moment of recording.")
    L.append("")
    L += wrapped(
        f"Step 1, the static interval: the {float(static_s):g} s window "
        "with the least joint motion. Every frame of the labelled hand "
        f"starts a candidate window that runs {float(static_s):g} s from "
        "that frame's time and must lie inside the recording. A window's "
        "score is its mean joint speed: the distance each of the 26 joints "
        "moved from one frame to the next (in the camera's XYZ frame, "
        "section 10), averaged over the joints, summed over the window, "
        "and divided by the time between the window's first and last "
        "frame. The window with the lowest score is the static interval; "
        "of two with the same score, the later one. Its start, end and "
        "length are static_interval_start_s, static_interval_end_s and "
        "static_interval_duration_s, in seconds from the labelled hand's "
        "first frame.")
    L.append("")
    L += wrapped(
        "Step 2, the representative frame: the frame of the static "
        "interval nearest its mean hand. (The recorder's files and the "
        "column medoid_wall_time call it the medoid. Strictly a medoid is "
        "the frame nearest all the other frames, which need not be the "
        "same one; what is done is what is described here.) "
        "For each frame, 21 points of the hand (the "
        "wrist, four joints of the thumb, four joints of each finger) are "
        "taken relative to its wrist. Their mean over the interval's frames "
        "is the mean hand. Each frame is then moved onto the mean hand by "
        "the rotation and translation that best fit its wrist and four "
        "knuckles to the mean hand's (least squares, no scaling, no "
        "mirroring), and the frame with the smallest sum of squared "
        "distances to the mean hand over all 21 points is the "
        "representative frame. (These 21 points are the wrist and the 20 "
        "finger joints. The 21 landmarks of the text files, section 9, "
        "have the palm centre in the wrist's place and give the wrist on a "
        "line of its own.) representative_frame_index is its line number "
        "in the .jsonl file counted from 0, representative_frame_id the "
        "tracker's frame number (the 'Frame' number in the keypoints "
        "file), representative_frame_time_s its time from the first frame.")
    L.append("")
    L += wrapped(
        "Why not simply the last frame. A hand held for five seconds is "
        "not equally still throughout, and any one frame picked by its "
        "position in the file carries whatever the tracker's fit was at "
        "that instant. The stillest window is where the pose was most "
        "nearly being held, and the frame nearest that window's average is "
        "the most typical one in it. A real frame is used rather than the "
        "average itself because averaged joint positions shorten the "
        "bones: the average hand is a hand nobody made.")
    L.append("")
    speed = spread([t.facts.static_interval_mean_speed_mm_s for t in takes])
    shape = spread([t.facts.static_interval_shape_rms_mm for t in takes])
    worst = spread([t.facts.static_interval_shape_rms_max_mm for t in takes])
    fast = sum(1 for t in takes
               if (t.facts.static_interval_mean_speed_mm_s or 0.0)
               > STATIC_MOVING_MM_S)
    changed = sum(1 for t in takes
                  if (t.facts.static_interval_shape_rms_max_mm or 0.0)
                  > SHAPE_CHANGE_MM)
    L += wrapped(
        "HOW STILL IT WAS. \"Stillest\" is not \"still\": the static "
        "interval is the least-moving part of each recording, and nothing "
        "required it to be motionless. So each recording says how still "
        "its own was, in two ways, because a hand can drift over the "
        "camera without changing shape:")
    L += wrapped(
        "static_interval_mean_speed_mm_s: the score of step 1 itself, the "
        "whole hand's movement in the camera's frame."
        + ("" if speed is None else
           f" Here {speed[0]:.1f} to {speed[2]:.1f} mm/s, median "
           f"{speed[1]:.1f}; {fast} of the {len(takes)} are above "
           f"{STATIC_MOVING_MM_S:g} mm/s."), "    ", "      ")
    L += wrapped(
        "static_interval_shape_rms_mm and ..._max_mm: the shape alone. "
        "Every frame of the interval is fitted onto the representative "
        "frame by a rotation and a translation (least squares over the 26 "
        "joints, no scaling), and the RMS distance of the 26 joints after "
        "the fit is taken: its median over the interval, and its largest "
        "value."
        + ("" if shape is None or worst is None else
           f" Here the median is {shape[0]:.2f} to {shape[2]:.2f} mm "
           f"(median over recordings {shape[1]:.2f}), and the largest "
           f"value reaches {worst[2]:.1f} mm; it is above "
           f"{SHAPE_CHANGE_MM:g} mm in {_plural(changed, 'recording')}."),
        "    ", "      ")
    L += wrapped(
        f"The recordings above either limit are named in section 15. "
        f"{STATIC_MOVING_MM_S:g} mm/s and {SHAPE_CHANGE_MM:g} mm are "
        "limits for a note in this README, not rules a recording was "
        "accepted by.")
    starts = spread([t.facts.static_interval_start_s for t in takes])
    reps = spread([t.facts.representative_frame_time_s for t in takes])
    if starts and reps:
        L.append("")
        L += wrapped(
            f"In these recordings the static interval is "
            f"{float(static_s):g} s long by construction and starts "
            f"{_fmt(starts[0])} to {_fmt(starts[2])} s into the recording "
            f"(median {_fmt(starts[1])} s); the representative frame is "
            f"{_fmt(reps[0])} to {_fmt(reps[2])} s in (median "
            f"{_fmt(reps[1])} s).")
    L.append("")
    L += wrapped(
        "The joint-frame table and the paper's angles of a recording "
        "(sections 11 to 13) are of this same frame. The rebuild checks "
        "that for every recording.")
    return L + [""]


RAW_FIELDS = [
    ("wall_time", "the recording computer's clock when the line was "
     "written, seconds since 1970. The clock of the durations and of the "
     "static interval."),
    ("timestamp_us, timestamp", "the tracker's own clock for the frame, in "
     "microseconds and in seconds. Its zero is arbitrary; use it for "
     "intervals inside one recording."),
    ("capture_time", "when the camera saw the hand, on the computer's "
     "clock: the time the frame was received minus the frame's age."),
    ("frame_id, packet_counter", "the tracker's frame number. Two lines "
     "with the same frame_id are two hands seen in the same frame."),
    ("hand_side", "the tracker's own left / right label for the hand it "
     "fitted (section 16)."),
    ("hand_id", "the tracker's identity number for a hand, constant while "
     "it keeps tracking it."),
    ("framerate", "the tracker's frames per second."),
    ("visible_time_us, frame_age_us", "how long the tracker had been "
     "seeing this hand, and the frame's age when received, microseconds."),
    ("grab_strength, pinch_strength", "the tracker's own 0 to 1 measures "
     "of a fist and of a pinch."),
    ("abs26", "the 26 joint positions [x, y, z] in the camera frame, "
     "metres, in the joint order below."),
    ("palm_abs", "the palm centre in the camera frame, metres."),
    ("joints", "the 26 joints as a chain: name, x y z (the joint's "
     "position in its parent joint's frame, metres) and qw qx qy qz (its "
     "rotation relative to the parent, a unit quaternion). The WRIST is the "
     "root: its x y z is its position in the camera frame and its "
     "quaternion its orientation in the camera frame."),
    ("pose, item, take, session", "the grasp_name, the take number and the "
     "session the line belongs to: the label, on every line."),
    ("source, space, units, status", "\"leap\", \"leap_desktop\" (the "
     "camera frame of section 10), \"m\" (metres), 1 (tracked)."),
]


def _sec_raw(gs: GraspSet) -> List[str]:
    L = _head(8, "RAW LEAP / OPENXR JOINT DATA")
    L += wrapped(
        f"<grasp>_{gs.hand}_take<N>.jsonl is the recording as the recorder "
        "wrote it, copied unchanged: one line of JSON per tracked hand per "
        "frame, in time order. The fields of a line:")
    for name, what in RAW_FIELDS:
        L += wrapped(f"{name}: {what}", "    ", "        ")
    L.append("")
    L += wrapped(
        "The 26 joints are the OpenXR hand-joint set, in this order: "
        + ", ".join(JOINT_NAMES) + ". Each joint's parent is the joint "
        "before it on its own digit; the PALM and the five METACARPAL "
        "joints have the WRIST as parent; the WRIST has none.")
    L.append("")
    L += wrapped(
        "Where each joint comes from in the tracker's skeleton: a joint is "
        "the near end of a bone (METACARPAL, PROXIMAL, INTERMEDIATE and "
        "DISTAL are the near ends of those four bones, TIP the far end of "
        "the distal bone), the PALM is the tracker's palm centre and the "
        "WRIST the far end of its forearm bone. The thumb has no "
        "INTERMEDIATE joint: the tracker gives it four bones of which the "
        "first has no length, and the other three are named here by "
        "anatomy, THUMB_METACARPAL (the bone the tracker itself calls the "
        "thumb's proximal), THUMB_PROXIMAL and THUMB_DISTAL. A joint's "
        "orientation is the orientation of the bone that starts at it, as "
        "the tracker reports it; a TIP repeats its distal bone's, and the "
        "WRIST's is the FOREARM bone's, because the tracker has no wrist "
        "frame of its own (this matters for the angles: section 13).")
    return L + [""]


def _sec_keypoints(gs: GraspSet) -> List[str]:
    labels: Dict[str, int] = {}
    for t in gs.takes:
        for k, n in t.facts.tracker_labels.items():
            labels[k] = labels.get(k, 0) + n
    L = _head(9, "21-POINT KEYPOINT FORMAT")
    L += wrapped(
        "The _keypoints.txt and _all_frames.txt files use the 21-landmark "
        "text layout of the earlier hand-ins. One block per frame, blocks "
        "separated by an empty line:")
    L += ["    Frame <tracker frame number> | Hand ID: <left|right> | "
          "Time: <local time, ms>",
          "    Wrist: (x, y, z)",
          "    0 (Palm): (x, y, z)",
          "    1 (THUMB): (x, y, z)   ... through ...   20 (Pinky): (x, y, z)"]
    L += wrapped(
        "Units: millimetres, written to two decimals. That is the file's "
        "resolution, not the tracker's accuracy, which was not measured in "
        "this work. Frame: the camera's XYZ frame of section 10; the Wrist "
        "line is the wrist joint's measured position, not zero. The points "
        "are joints of the 26, by index:")
    L += ["    Wrist     WRIST",
          "    0         PALM (the palm centre)",
          "    1 to 4    thumb: METACARPAL, PROXIMAL, DISTAL, TIP",
          "    5 to 8    index: PROXIMAL (the knuckle), INTERMEDIATE, DISTAL,"
          " TIP",
          "    9 to 12   middle, 13 to 16 ring, 17 to 20 little finger: the "
          "same four"]
    L += wrapped(
        "The four finger METACARPAL joints of the 26 are not in this "
        "layout; they are in the .jsonl and in the joint-frame tables.")
    L += wrapped(
        "Hand ID is the tracker's own label for the hand in that frame, "
        "written as it came. "
        + ("In this folder every frame of the labelled hand carries "
           f"\"{gs.hand}\"." if list(labels) == [gs.hand] else
           "In this folder the labelled hand's frames carry: "
           + ", ".join(f"{k} on {n}" for k, n in sorted(labels.items()))
           + " (section 16).")
        + " The hand recorded is the one in the file name.")
    return L + [""]


def convention_evidence(gs: GraspSet) -> Dict[str, object]:
    """What the recordings themselves say about the camera's axes."""
    takes = gs.takes
    y = [t.palm_cm[1] for t in takes if t.palm_cm]
    z = [t.wrist_minus_palm_z_cm for t in takes
         if t.wrist_minus_palm_z_cm is not None]
    two = [(t.palm_cm[0], t.other_palm_x_cm) for t in takes
           if t.palm_cm and t.other_palm_x_cm is not None
           and (t.facts.other_hand_median_distance_cm or 0.0)
           >= FOLLOW_RADIUS_M * 100.0]
    side = "smaller" if gs.hand == "left" else "larger"
    agree = sum(1 for a, b in two if (a < b) == (gs.hand == "left"))
    return {"y": spread(y), "y_positive": sum(1 for v in y if v > 0),
            "y_n": len(y), "z": spread(z),
            "z_positive": sum(1 for v in z if v > 0), "z_n": len(z),
            "x_pairs": len(two), "x_agree": agree, "x_side": side,
            "x_mine": spread([a for a, _b in two]),
            "x_other": spread([b for _a, b in two])}


def _sec_xyz(gs: GraspSet) -> List[str]:
    e = convention_evidence(gs)
    takes = gs.takes
    gaps = [t.chain_gap_mm for t in takes if t.chain_gap_mm is not None]
    L = _head(10, "XYZ COORDINATE CONVENTIONS")
    L += wrapped(
        "Three coordinate frames appear in this folder. All are "
        "right-handed and metric.")
    L.append("")
    L += wrapped(
        "A. The camera frame (the tracker's \"desktop\" frame). Used by "
        "abs26, palm_abs and the WRIST joint in the .jsonl files (metres) "
        "and by every point of the _keypoints.txt and _all_frames.txt "
        "files (millimetres).")
    L += ["      origin  the centre of the camera module, midway between "
          "its two lenses",
          "      +X      along the module's long side, to the operator's "
          "right",
          "      +Y      straight up out of the module, toward the hand",
          "      +Z      horizontal, toward the operator"]
    L += wrapped(
        "It is fixed to the module, not to the hand: moving the hand over "
        "the camera changes every coordinate. The origin and the axes are "
        "the tracker maker's definition and were not measured here. The "
        "three directions were checked on these recordings:", "    ", "    ")
    if e["y"] is not None:
        L += wrapped(
            f"+Y up: the palm is {e['y'][0]:.0f} to {e['y'][2]:.0f} cm "
            f"above the origin, positive in {e['y_positive']} of "
            f"{e['y_n']} recordings.", "      ", "      ")
    if e["z"] is not None:
        L += wrapped(
            f"+Z toward the operator: the wrist, where the forearm comes "
            f"from, is {e['z'][0]:.1f} to {e['z'][2]:.1f} cm further along "
            f"+Z than the palm centre, positive in {e['z_positive']} of "
            f"{e['z_n']} recordings.", "      ", "      ")
    if e["x_pairs"]:
        ghosts = sum(1 for t in takes if t.facts.other_hand_detected) \
            - e["x_pairs"]
        L += wrapped(
            f"+X to the operator's right: in {e['x_agree']} of the "
            f"{e['x_pairs']} recordings with the other hand in view, the "
            f"{gs.hand} hand has the {e['x_side']} X (X of the {gs.hand} "
            f"hand {e['x_mine'][0]:.0f} to {e['x_mine'][2]:.0f} cm, of the "
            f"other hand {e['x_other'][0]:.0f} to {e['x_other'][2]:.0f} "
            "cm)."
            + (f" (In {_plural(ghosts, 'more recording')} the second "
               "\"hand\" is a second fit on the same hand, section 16; "
               "not counted here.)" if ghosts > 0 else ""),
            "      ", "      ")
    L.append("")
    L += wrapped(
        "B. The wrist frame. Used by the joint-frame tables "
        f"({GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\*.csv and the PDF): x_mm, y_mm, "
        "z_mm and the axis and quaternion columns, millimetres.")
    L += ["      origin  the WRIST joint of that frame",
          "      axes    the WRIST joint's own axes (section 11): +z along "
          "the forearm toward the elbow,",
          "              +y out of the back of the forearm and hand, "
          "+x = y cross z (on a left hand",
          "              toward the thumb side, on a right hand toward "
          "the little-finger side)"]
    L += wrapped(
        "It moves with the hand, so the hand's position over the camera "
        "drops out. A point p in the camera frame becomes R_wrist^T (p - "
        "p_wrist) in the wrist frame, where p_wrist is the WRIST joint's "
        "position and R_wrist the rotation matrix of its orientation "
        "(section 12, step 1), both from the same frame.", "    ", "    ")
    bends = spread([t.wrist_bend_deg for t in takes
                    if t.wrist_bend_deg is not None])
    L += wrapped(
        "Its axes are the forearm's, as the tracker estimates them, so "
        "these coordinates still carry the tracked bend of the wrist"
        + ("" if bends is None else
           f" (the PALM row's flex_deg is {bends[0]:.0f} to {bends[2]:.0f} "
           "degrees across these recordings)")
        + ": the same hand shape on a differently bent wrist has "
        "different x_mm y_mm z_mm. To lay the shapes of two recordings "
        "side by side, re-express them in the PALM joint's frame, "
        "R_palm^T (p - p_palm) with the PALM row's position and "
        "quaternion, or compare joint angles.", "    ", "    ")
    L.append("")
    L += wrapped(
        "C. Joint-local frames. Each entry of 'joints' in the .jsonl gives "
        "a joint's position and rotation in its PARENT joint's frame "
        "(metres): the same hand as abs26, stored as a chain. Chaining "
        "from the WRIST (position = parent position + parent rotation x "
        "local position; rotation = parent rotation x local rotation) "
        "gives the camera-frame positions of abs26"
        + ("." if not gaps else
           f": to within {max(gaps):.4f} mm on the representative frames "
           "here. The two are stored separately and are not identical to "
           "the last digit; the joint-frame tables are made from the "
           "chained positions."))
    L.append("")
    L += wrapped(
        "Handedness: all three frames are right-handed for both hands. "
        "What differs between a left and a right hand is where +x of a "
        "finger's joint points, because x = y cross z with y out of the "
        "back of the hand and z toward the wrist: toward the thumb on a "
        "left hand, toward the little finger on a right hand (the thumb's "
        "own joints are turned: section 12). The angle signs of "
        "section 12 are corrected for that, so they read the same on both "
        "hands.")
    size = spread([t.facts.index_metacarpal_mm for t in takes])
    drift = spread([t.facts.index_metacarpal_range_mm for t in takes])
    if size and drift:
        L.append("")
        L += wrapped(
            "SCALE: MILLIMETRES ARE NOT COMPARABLE BETWEEN RECORDINGS AS "
            "THEY STAND. The tracker fits the size of the hand as well as "
            "its pose, fits it again each time it finds the hand, and goes "
            "on adjusting it while it tracks. The "
            "same hand's index metacarpal (the bone from INDEX_METACARPAL "
            f"to INDEX_PROXIMAL) is {size[0]:.1f} to {size[2]:.1f} mm long "
            f"across these {len(takes)} recordings, "
            f"{(size[2] / size[0] - 1) * 100:.0f} % apart, and inside one "
            f"recording it changes by {drift[1]:.1f} mm at the median and "
            f"by up to {drift[2]:.1f} mm. index_metacarpal_mm and "
            "index_metacarpal_range_mm in the summary give both per "
            "recording, and bone_len_mm in each joint-frame table gives "
            "every bone. To compare positions between recordings, divide "
            "them by a bone length of the same recording first. The angle "
            "formulas use no bone length; whether the fitted angles shift "
            "when the size estimate does was not examined.")
    return L + [""]


def _key(gs: GraspSet) -> List[str]:
    """Section 11's joint key (`xr_hand.joint_key`): which joint, which
    bones and which of the paper's angles each row of a table is, and the
    two files that hold the same key."""
    from xr_hand import joint_key as jk

    L = ["", "  WHICH JOINT EACH ROW IS"]
    key = list(jk.readme_lines())
    rule = " ".join(jk.NAMING_RULE.split())
    if rule not in " ".join(" ".join(key).split()):
        L += wrapped(jk.NAMING_RULE)
    L += key
    L += wrapped(
        f"The same key is {GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\{KEY_CSV} (one "
        f"row per joint, columns {', '.join(jk.KEY_COLUMNS)}) and, drawn on "
        f"one page with a flat left hand and its 26 joints numbered, "
        f"{GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\{KEY_PDF}.")
    return L + [""]


def _sec_frames(gs: GraspSet) -> List[str]:
    L = _head(11, "JOINT LOCAL FRAMES")
    if not gs.has_joint_frames:
        L += wrapped(
            "The joint-frame tables are not in this build: the session has "
            "no joint_frames folder. Run the rebuild of section 20, which "
            "makes them.")
        return L + [""]
    L += wrapped(
        "Every one of the 26 joints carries its own x y z frame: an origin "
        "at the joint and three perpendicular unit axes that turn with the "
        "bone that starts there.")
    L += ["      +z  along that bone, pointing BACK toward the wrist (the "
          "bone itself runs along -z)",
          "      +y  out of the back of the hand or finger (dorsal)",
          "      +x  = y cross z: across the finger, the axis a finger "
          "joint bends about"]
    L += wrapped(
        "These are the tracker's bone frames, unchanged. The PALM joint's "
        "frame is the tracker's palm frame with the same meaning (+z toward "
        "the wrist, +y out of the back of the hand); the WRIST joint's is "
        "the forearm bone's; a TIP has its distal bone's frame.")
    L.append("")
    L += wrapped(
        f"Each {GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\<grasp>_{gs.hand}_take<N>"
        ".csv describes the recording's representative frame (section 7), "
        "one row per joint, 26 rows. Columns:")
    L += ["    joint, parent           the joint and the joint it is "
          "measured from (empty for the WRIST)",
          "    x_mm y_mm z_mm          the joint's position in the wrist "
          "frame (section 10, frame B)",
          "    qx qy qz qw             the joint's orientation in the wrist "
          "frame, unit quaternion, qw >= 0",
          "    axis_x_x axis_x_y axis_x_z   the joint's +x axis as a unit "
          "vector in the wrist frame",
          "    axis_y_*, axis_z_*      its +y and +z axes, the same way",
          "    bone_len_mm             distance from the parent joint to "
          "this joint",
          "    flex_deg abd_deg twist_deg   the joint angles (section 12)"]
    L += _key(gs)
    L += wrapped(
        "The three axis vectors are the columns of the rotation matrix of "
        "qx qy qz qw. A check a reader can make: for a joint whose parent "
        "is on the same digit, the vector from the joint back to its "
        "parent, divided by bone_len_mm, equals the PARENT's axis_z.")
    L.append("")
    L += wrapped(
        f"{GRASPS_DIR}\\{JOINT_FRAMES_DIR}\\joint_frames.pdf draws it: one "
        "page per recording, named at the top. The line under the name "
        "says which frame is drawn, \"the recorder's static-interval "
        "medoid\": that is the representative frame of section 7, under "
        "the recorder's own name for it. The hand is shown twice, "
        "from the palm side and from the side, with a 10 mm arrow for each "
        "axis at every joint: +x red, +y green, +z blue, the arrowhead at "
        "the positive end. Each digit is marked at its tip (T thumb, I "
        "index, M middle, R ring, L little finger) and each joint with the "
        "paper's name for it (TMC, MCP, IP on the thumb; CMC, MCP, PIP, "
        "DIP on a finger). Below are the paper's 24 angle names, 18 with "
        "their values and 6 marked n/a (section 13), and the 26-row table. "
        "In a tightly closed hand the labels at the fingertips crowd each "
        "other; the table under the drawing names every joint.")
    return L + [""]


def abd_directions(gs: GraspSet) -> Dict[str, Tuple[float, float, float]]:
    """Which way a positive abd_deg leans a bone, measured on the takes'
    tables: abd_deg = s x asin(z_c . x_p) and the bone runs along -z_c, so
    a positive value leans it toward -s x x_p (s = +1 for a hand fitted as
    right, -1 for left; x_p the parent's +x). Returned as (minimum,
    median, maximum) over the tables of that direction dotted with
      "thumb"   the palm's face (minus the PALM joint's +y), on the
                THUMB_PROXIMAL and THUMB_DISTAL rows together;
      "fingers" the line from the little finger's MCP to the index
                finger's (the thumb side), on the four fingers' PROXIMAL
                rows.
    A key is missing when no table has the rows."""
    thumb, fingers = [], []
    for t in gs.takes:
        if not t.table:
            continue
        by = {r["joint"]: r for r in t.table}
        s = 1.0 if t.facts.representative_label == "right" else -1.0
        face = -np.asarray(by["PALM"]["axis_y"], dtype=float)
        side = np.array([by["INDEX_PROXIMAL"][k] - by["LITTLE_PROXIMAL"][k]
                         for k in ("x_mm", "y_mm", "z_mm")], dtype=float)
        side /= float(np.linalg.norm(side)) or 1.0
        for row in ("THUMB_PROXIMAL", "THUMB_DISTAL"):
            x_p = np.asarray(by[by[row]["parent"]]["axis_x"], dtype=float)
            thumb.append(float((-s * x_p) @ face))
        for f in ("INDEX", "MIDDLE", "RING", "LITTLE"):
            x_p = np.asarray(by[f"{f}_METACARPAL"]["axis_x"], dtype=float)
            fingers.append(float((-s * x_p) @ side))
    out = {}
    for key, values in (("thumb", thumb), ("fingers", fingers)):
        found = spread(values)
        if found is not None:
            out[key] = found
    return out


def abd_sign_text(gs: GraspSet) -> str:
    """Section 12's sentence on the sign of abd_deg, from `abd_directions`."""
    d = abd_directions(gs)
    n = sum(1 for t in gs.takes if t.table)
    text = ("The bone runs along -z_c, so a positive abd_deg leans it toward "
            "-s x x_p (s as in step 4). ")
    if "fingers" in d:
        f = d["fingers"]
        text += ("On the index, middle, ring and little fingers that is "
                 "toward the thumb side of the hand: on their PROXIMAL rows "
                 "it is at a dot product of "
                 f"{_between(f[0], f[2], 2)} with the line from the little "
                 "finger's MCP to the index finger's. ")
    else:
        text += ("On the four fingers it is meant to be toward the thumb "
                 "side; no table of this build checks it. ")
    if "thumb" in d:
        th = d["thumb"]
        text += ("On the thumb's PROXIMAL and DISTAL rows (its MCP and IP) "
                 "it is not the thumb side but toward the palm's face: "
                 f"dotted with the palm's face (minus the PALM joint's +y) "
                 f"it is {_between(th[0], th[2], 2)} (median "
                 f"{_fmt(th[1], 2)}; 1 = straight toward the palm's face, 0 "
                 f"= across it), over the {_plural(n, 'table')} of this "
                 "folder.")
    else:
        text += ("On the thumb the sign was not checked in this build: it "
                 "has no joint-frame table to check it on.")
    return text


def _sec_angles(gs: GraspSet) -> List[str]:
    L = _head(12, "JOINT ANGLES")
    if not gs.has_joint_frames:
        L += wrapped("Not in this build (section 11).")
        return L + [""]
    abd = _abd_stats(gs)
    L += wrapped(
        "flex_deg, abd_deg and twist_deg of a row are the rotation that "
        "takes the PARENT joint's frame to this joint's frame, split into "
        "three angles. They are computed as follows; nothing else enters.")
    L.append("")
    L += wrapped(
        "Step 1, quaternion to rotation matrix. For a unit quaternion "
        "(qx, qy, qz, qw) = (x, y, z, w):")
    L += ["      R = | 1-2(yy+zz)   2(xy-wz)   2(xz+wy) |",
          "          |  2(xy+wz)  1-2(xx+zz)   2(yz-wx) |",
          "          |  2(xz-wy)   2(yz+wx)  1-2(xx+yy) |"]
    L += wrapped(
        "Column 1, 2, 3 of R is the frame's +x, +y, +z axis written in the "
        "frame the quaternion is relative to (vectors are columns, and R "
        "turns the reference frame's axes into the joint's).", "    ",
        "    ")
    L += wrapped(
        "Step 2, the parent-to-child rotation. R_rel = R_parent^T R_child, "
        "with both orientations in the same frame. In the .jsonl this is "
        "simply the joint's own qw qx qy qz, which is stored relative to "
        "its parent. Its entries are dot products of axes: entry (i, j), "
        "written Rij, is the parent's axis i dotted with the child's axis "
        "j. Its third column, (R13, R23, R33), is the child's +z axis, the "
        "child's bone line, seen from the parent.")
    L += wrapped(
        "Step 3, three angles. R_rel is written as a turn about x, then "
        "about the new y, then about the new z: R_rel = Rx(a) Ry(b) Rz(c), "
        "each a right-handed turn about that axis. Then:")
    L += ["      b = asin(R13)            R13 first limited to [-1, +1]",
          "      a = atan2(-R23, R33)",
          "      c = atan2(-R12, R11)",
          "      if |R13| >= 1 - 1e-9 (the bone exactly sideways, which no "
          "finger reaches):",
          "          a = atan2(R32, R22), c = 0"]
    L += wrapped(
        "Step 4, signs and degrees, with s = +1 for a right hand and -1 "
        "for a left hand (the hand_side of the frame):")
    L += ["      flex_deg  = -a x 180/pi",
          "      abd_deg   =  s x b x 180/pi",
          "      twist_deg = -s x c x 180/pi"]
    L += wrapped(
        "The tables are rounded to 0.01 degree and 0.01 mm. No other "
        "numerical step is used: no acos, no smoothing, no limits on the "
        "result.")
    L.append("")
    L += wrapped("What the three angles mean, and their signs, on both hands:")
    L += wrapped(
        "flex_deg: the angle of the child bone inside the parent's "
        "sagittal plane (the plane of the parent's +y and +z, which cuts "
        "the finger lengthways), = atan2(z_c . y_p, z_c . z_p) with z_c "
        "the child's +z axis and y_p, z_p the parent's. Positive = bent "
        "toward the palm; 0 = in line with the parent bone; negative = "
        "bent backward.", "    ", "        ")
    L += wrapped(
        "abd_deg: the angle of the child bone out of that plane, "
        "= s x asin(z_c . x_p) with x_p the parent's +x. "
        + abd_sign_text(gs), "    ", "        ")
    L += wrapped(
        "twist_deg: the turn about the child bone's own axis that is left "
        "over. Positive = the back of the bone turning toward the "
        "little-finger side.", "    ", "        ")
    L.append("")
    L += wrapped(
        "The order (flexion, then abduction, then twist) is chosen so that "
        "flexion and abduction depend on the direction of the bone only. "
        "The split fails only when abduction reaches 90 degrees, where "
        "flexion and twist can no longer be told apart"
        + ("" if not abd.get("METACARPAL") or not abd.get("PROXIMAL") else
           ". The largest abduction in this folder is "
           f"{max(abs(v) for v in abd['METACARPAL']):.1f} degrees on a "
           "METACARPAL row (a metacarpal seen from the forearm's frame) "
           f"and {max(abs(v) for v in abd['PROXIMAL']):.1f} on a finger's "
           "PROXIMAL row")
        + ". Taken the other way round (abduction "
        "first) it fails at 90 degrees of flexion, which every fist "
        "reaches.")
    L += wrapped(
        "Rows: the WRIST row has no parent and all three angles 0. A TIP "
        "has its distal bone's orientation, so its angles are 0. The PALM "
        "row is the palm against the WRIST joint, which is the forearm: "
        "its flex_deg and abd_deg are the bend of the wrist.")
    twists = [t.phalanx_twist_deg for t in gs.takes
              if t.phalanx_twist_deg is not None]
    residuals = [t.phalanx_twist_residual_deg for t in gs.takes
                 if t.phalanx_twist_residual_deg is not None]
    L += wrapped(
        "Reading twist_deg. The tracker builds the joints along a digit "
        "(a finger's PROXIMAL, INTERMEDIATE and DISTAL rows, the thumb's "
        "PROXIMAL and DISTAL) without any turn about the bone. Split in "
        "this order such a joint still shows a twist, fixed by the other "
        "two angles: tan(twist / 2) = -tan(flex / 2) x tan(abd / 2), for a "
        "small abduction twist_deg = -abd_deg x tan(flex_deg / 2), about 2 "
        "degrees at 45 of flexion and 5 of abduction. It is arithmetic, "
        "not the finger turning."
        + (f" In this folder the small-abduction form holds on every such "
           f"row to within {max(residuals):.2f} degree, and the largest "
           f"such twist is {max(twists):.1f} degrees." if twists else "")
        + " On a METACARPAL row the twist is the turn of that bone against "
        "the forearm frame and can be large; on the thumb's it is the "
        "thumb's turn about its own length.")
    return L + [""]


def paper_order_example(flex_deg: float, abd_deg: float
                        ) -> Tuple[float, float]:
    """What a bone at `flex_deg` of flexion and `abd_deg` of abduction in
    this folder's order reads as in the paper's order, through the code
    that fills the *_paper_order columns: the README's examples are what
    the code does."""
    return jf.paper_order_angles(
        jf.rotation_from_angles(flex_deg, abd_deg, 0.0, "right"), "right")


PAPER_TABLE = [
    ("T_TMC_fe, T_TMC_aa", "thumb trapeziometacarpal flexion, abduction",
     "NOT GIVEN (item 3); the columns are empty"),
    ("T_MCP_fe", "thumb metacarpophalangeal flexion",
     "flex_deg of THUMB_PROXIMAL"),
    ("T_IP", "thumb interphalangeal", "flex_deg of THUMB_DISTAL"),
    ("I_CMC, M_CMC, R_CMC, L_CMC", "finger carpometacarpal (the palm arc)",
     "NOT GIVEN (item 2); the columns are empty"),
    ("*_MCP_fe, *_MCP_aa", "finger metacarpophalangeal flexion, abduction",
     "flex_deg, abd_deg of the finger's PROXIMAL row"),
    ("*_PIP", "proximal interphalangeal", "flex_deg of INTERMEDIATE"),
    ("*_DIP", "distal interphalangeal", "flex_deg of DISTAL"),
]


def _abd_stats(gs: GraspSet) -> Dict[str, List[float]]:
    """The abd_deg cells of the representative frames' tables by kind of
    row (METACARPAL, and the fingers' PROXIMAL, INTERMEDIATE and DISTAL):
    how large abduction gets, and how far the tracker's PIP and DIP rows
    are from hinges."""
    by: Dict[str, List[float]] = {}
    for t in gs.takes:
        for r in t.table or []:
            name = r["joint"]
            kind = ("INTERMEDIATE" if name.endswith("_INTERMEDIATE") else
                    "DISTAL" if name.endswith("_DISTAL")
                    and not name.startswith("THUMB") else
                    "PROXIMAL" if name.endswith("_PROXIMAL")
                    and not name.startswith("THUMB") else
                    "METACARPAL" if name.endswith("_METACARPAL") else None)
            if kind:
                by.setdefault(kind, []).append(float(r["abd_deg"]))
    return by


def _sec_paper(gs: GraspSet) -> List[str]:
    takes = [t for t in gs.takes if t.angles]
    L = _head(13, "MAPPING TO THE PAPER'S 24 ANGLES")
    if not gs.has_joint_frames:
        L += wrapped("Not in this build (section 11).")
        return L + [""]
    L += wrapped(
        "Cobos et al. 2009 (\"Human Hand Descriptions and Gesture "
        "Recognition for Object Manipulation\", its Figure 1 and Table 1) "
        "describe the hand by 24 joint angles: 5 for each of the index, "
        "middle, ring and little finger (I, M, R, L) and 4 for the thumb "
        f"(T). This folder gives {len(GIVEN_NAMES)} of them, each one cell "
        "of a recording's joint-frame table, and does not give "
        f"{len(NOT_GIVEN)}, because this tracker does not measure them:")
    w = max(len(a) for a, _b, _c in PAPER_TABLE)
    for name, meaning, cell in PAPER_TABLE:
        L += wrapped(f"{name:<{w}}  {meaning}: {cell}", "    ",
                     " " * (w + 8))
    L += wrapped(
        f"{GRASPS_DIR}\\{ANGLES_FILE} has one row per recording with the "
        "paper's 24 names in the paper's order (degrees): "
        f"{len(GIVEN_NAMES)} filled, {len(NOT_GIVEN)} empty. Each page of "
        f"the PDF prints the {len(GIVEN_NAMES)} and \"n/a\" for the "
        f"{len(NOT_GIVEN)}. An empty cell there means \"not given\", never "
        "zero. What the tracker does measure at those joints is something "
        "else, and is in the same file under names that say what it is:")
    L += wrapped(
        "T_metacarpal_flex_wrist_ref, T_metacarpal_abd_wrist_ref and "
        "I_, M_, R_, L_metacarpal_flex_wrist_ref: flex_deg and abd_deg of "
        "the THUMB_METACARPAL row and flex_deg of the four finger "
        "METACARPAL rows, that is, where each metacarpal points as seen "
        "from the WRIST joint, whose axes are the forearm's.", "    ",
        "      ")
    L += wrapped(
        "the same six with _palm_ref: where each metacarpal points as seen "
        "from the PALM joint (the angles of section 12 with R_rel = "
        "R_palm^T R_metacarpal).", "    ", "      ")
    L += wrapped(
        "forearm_to_palm_deg: the angle of the one rotation that takes "
        "the WRIST joint's frame to the PALM joint's, acos((trace("
        "R_wrist^T R_palm) - 1) / 2): how far the wrist is bent, as "
        "tracked.", "    ", "      ")
    L.append("")
    L += wrapped(
        "WHERE THESE NUMBERS ARE NOT THE PAPER'S, item by item:")
    open_hand, closed = (paper_order_example(60.0, 10.0),
                         paper_order_example(85.0, 2.0))
    L += wrapped(
        "Order of the two turns at the MCP. The paper's kinematic chain "
        "(its Table 2) turns the joint about its abduction axis first and "
        "its flexion axis second. The tables here turn about flexion "
        "first (section 12). The bone points the same way in both; the "
        "two numbers that say where differ as soon as both are non-zero:",
        "    1. ", "       ")
    L += ["           sin(fe_paper) = sin(fe) x cos(aa)",
          "           tan(aa_paper) = tan(aa) / cos(fe)"]
    L += wrapped(
        "For example 60 degrees of flexion with 10 of abduction here is "
        f"{open_hand[0]:.1f} with {open_hand[1]:.1f} in the paper's order. "
        "For small abduction or small "
        "flexion the two agree. The columns *_MCP_fe_paper_order and "
        f"*_MCP_aa_paper_order of {ANGLES_FILE} give the four finger MCP "
        "joints in the paper's order, computed from the same bone "
        "direction (z_c of section 12): fe_paper = atan2(z_c . y_p, "
        "k x hypot(z_c . x_p, z_c . z_p)), aa_paper = s x atan2(k x "
        "z_c . x_p, k x z_c . z_p), k = the sign of z_c . z_p. Use "
        "*_MCP_aa_paper_order with care in a closed hand: the paper's "
        "order divides by cos(fe), so near 90 degrees of flexion a small "
        "sideways lean of the finger becomes a large abduction (2 degrees "
        f"at 85 of flexion reads as {closed[1]:.0f}).", "       ",
        "       ")
    rigid = rigid_palm(gs)
    cmc = spread([t.angles[n] for t in takes
                  for n in ("I_CMC", "M_CMC", "R_CMC", "L_CMC")])
    bend = spread([t.forearm_to_palm_deg for t in takes
                   if t.forearm_to_palm_deg is not None])
    if rigid is not None:
        L += wrapped(
            "The four CMC angles are not measured, and are not given. The "
            "paper's CMC joint is the bending of the palm itself (it "
            "\"allows simulation of the palm arc\"; its Table 5 gives it "
            "5 to 15 degrees of range). This tracker's palm is rigid: in "
            "these recordings its four finger metacarpals do not move "
            "against its PALM joint. Seen from the PALM joint they are "
            f"the same numbers in every one of the {len(takes)} recordings "
            "("
            + ", ".join(f"{n[0]} {v:.2f}" for n, v in rigid.items())
            + " degrees: the columns *_metacarpal_flex_palm_ref), which "
            "are constants of the tracker's hand model. Seen from the "
            "WRIST joint (*_metacarpal_flex_wrist_ref) they are that rigid "
            "palm on a bent wrist: they change with the bend of the wrist "
            "and with nothing else"
            + ("" if cmc is None else
               f", {cmc[0]:.0f} to {cmc[2]:.0f} degrees across the "
               "recordings")
            + ("" if bend is None else
               f" (forearm_to_palm_deg {bend[0]:.0f} to {bend[2]:.0f})")
            + ". Neither is the arch of the palm, and the wrist's bend "
            "itself rests on where the tracker puts the forearm, which it "
            "estimates and which was not checked here.",
            "    2. ", "       ")
    else:
        L += wrapped(
            "The four CMC angles are not given. The paper's CMC is the "
            "arch of the palm, measured from a reference frame in the "
            "hand. What the tracker has is where each metacarpal points "
            "as seen from the WRIST joint, whose axes are the forearm's "
            "(*_metacarpal_flex_wrist_ref), which includes the bend of "
            "the wrist, or as seen from the PALM joint "
            "(*_metacarpal_flex_palm_ref).", "    2. ", "       ")
    axis = spread([t.thumb_axis_deg for t in takes
                   if t.thumb_axis_deg is not None])
    axis_w = spread([t.thumb_axis_wrist_deg for t in takes
                     if t.thumb_axis_wrist_deg is not None])
    L += wrapped(
        "The thumb's two TMC angles are not given. In the paper (its "
        "Table 3) the TMC flexion axis is parallel to the thumb's own MCP "
        "and IP flexion axes, the TMC abduction axis is at a right angle "
        "to it, and the pair is measured from a thumb reference frame "
        "that the paper shows in its Figure 1 and does not give in "
        "numbers. What the tracker has is where the thumb's metacarpal "
        "points as seen from the WRIST joint or from the PALM joint "
        "(T_metacarpal_flex_*, T_metacarpal_abd_*): about the hand's "
        "axes, not the thumb's. The thumb's own flexion axis (the +x of "
        "THUMB_METACARPAL, about which T_MCP_fe and T_IP are measured) is "
        "far from the hand's"
        + ("" if axis is None or axis_w is None else
           f": {axis_w[0]:.0f} to {axis_w[2]:.0f} degrees from the WRIST "
           f"joint's +x in these recordings (median {axis_w[1]:.0f}), "
           f"{axis[0]:.0f} to {axis[2]:.0f} from the PALM joint's (median "
           f"{axis[1]:.0f})")
        + ". So these two numbers are not the paper's two, and what "
        "exactly relates them cannot be stated without the paper's thumb "
        "frame. T_MCP_fe and T_IP are turns about the thumb's own flexion "
        "axis and are not affected.", "    3. ", "       ")
    abd = _abd_stats(gs)
    if abd.get("INTERMEDIATE") and abd.get("DISTAL"):
        pip, dip = np.array(abd["INTERMEDIATE"]), np.array(abd["DISTAL"])
        L += wrapped(
            "PIP and DIP. The paper's are hinges: one angle each. The "
            "tracker's are not quite: its INTERMEDIATE rows carry an "
            f"abd_deg of {pip.min():.1f} to {pip.max():.1f} degrees "
            f"(beyond 5 in {100.0 * float((np.abs(pip) > 5).mean()):.0f} % "
            f"of them) and its finger DISTAL rows {dip.min():.1f} to "
            f"{dip.max():.1f} "
            f"({100.0 * float((np.abs(dip) > 5).mean()):.0f} %). *_PIP and "
            "*_DIP are the flex_deg alone; that sideways part is in the "
            "joint-frame tables and in no paper angle.", "    4. ",
            "       ")
    L += wrapped(
        "Twist. The paper's finger bones do not turn about their own "
        "axis. twist_deg is reported for every joint and is not one of "
        "the 24.", "    5. ", "       ")
    thumb_checked = "thumb" in abd_directions(gs)
    L += wrapped(
        "Signs and zeros. Here flexion is positive toward the palm; "
        "abduction is positive toward the thumb side on the four fingers "
        + ("and toward the palm's face on the thumb's MCP and IP rows "
           "(section 12 gives the measured direction)" if thumb_checked else
           "(on the thumb the sign was not checked in this build, section "
           "12)")
        + "; an angle is zero when the bone is in line with its parent bone. "
        "The paper says its angles can be positive or negative about a "
        "reference line; it does not say which direction of abduction is "
        "positive, nor give its zero posture in numbers. Either may "
        "differ from these.", "    6. ", "       ")
    back = sum(1 for t in gs.takes if back_side(t))
    L += wrapped(
        "What is measured. The paper's angles came from a CyberGlove data "
        "glove. These come from a skeleton a camera tracker fitted to two "
        "infrared images: a finger the camera cannot see is placed by the "
        "tracker's hand model, not seen"
        + (f" (in {_plural(back, 'recording')} the fit has the back of "
           "the hand toward the camera, and may be upside down: section "
           "15)" if back else "")
        + ", the size of the hand is the tracker's estimate (section "
        "10), and the takes of one grasp do not always agree (section "
        "15). No accuracy figure for these angles was measured in this "
        "work.", "    7. ", "       ")
    return L + [""]


def _sec_perpose(gs: GraspSet) -> List[str]:
    L = _head(14, "PER-POSE DATASETS")
    L += wrapped(
        f"<grasp>_{gs.hand}_all_frames.txt is the dataset of one grasp "
        "type: every frame of the labelled hand from every accepted take "
        "of that grasp, in the 21-landmark layout of section 9 (22 lines "
        "of coordinates per frame, millimetres, the camera frame of "
        "section 10), take after take in take order, the frames of a take "
        "in time order. The 'Frame' number is the tracker's own counter and "
        "does not run on from one take to the next; the 'Time' on each "
        "block tells the takes apart.")
    L += wrapped(
        "Only the labelled hand is in it. A second hand in the recording "
        "(section 16) is in the take's .jsonl and nowhere else. Only "
        "accepted takes of that grasp are in it: no rejected attempt and "
        "no other grasp. Its takes were fitted with different hand sizes "
        "(section 10) and are not always the same hand shape (section 15).")
    total = sum(t.facts.shape_frames for t in gs.takes)
    over5 = sum(t.facts.frames_over_5mm for t in gs.takes)
    over10 = sum(t.facts.frames_over_10mm for t in gs.takes)
    edges = sum(1 for t in gs.takes if t.facts.extent_notes)
    if total:
        L += wrapped(
            "NOT EVERY FRAME IS THE POSE. A file holds each take whole, and "
            "a take is five seconds in which the pose was held more or "
            "less: a hand still closing when the recording began, a fit "
            "still settling, a drift at the end. Measured against each "
            "take's representative frame (shape distance: RMS over the 26 "
            "joints after a rigid fit, as in section 7), "
            f"{over5} of the {total} frames ({100.0 * over5 / total:.1f} %) "
            f"are more than {SHAPE_CHANGE_MM:g} mm away and {over10} "
            f"({100.0 * over10 / total:.1f} %) more than "
            f"{NOT_THE_POSE_MM:g} mm; {_plural(edges, 'take')} "
            f"{'begins or ends' if edges == 1 else 'begin or end'} more "
            f"than {NOT_THE_POSE_MM:g} mm away (section 15). The summary "
            "says it per take: take_shape_rms_max_mm, "
            "frames_within_5mm_percent, shape_rms_start_mm and "
            "shape_rms_end_mm.")
        L += wrapped(
            "To keep the pose alone, cut each take to its static interval: "
            "static_interval_first_frame_id and static_interval_last_frame_"
            "id in the summary are the tracker frame numbers it runs "
            "between, and the 'Frame' number heads every block of the "
            "file.")
    w = max([len("grasp_name")] + [len(i) for i in gs.item_ids])
    L += wrapped(
        "Per grasp, the takes in its file, its frames, and how many of "
        f"those frames are more than {SHAPE_CHANGE_MM:g} mm from their "
        "take's representative frame:")
    L.append(f"  {'grasp_name':<{w}}  takes included        frames  "
             f"over {SHAPE_CHANGE_MM:g} mm")
    for gid in gs.item_ids:
        takes = gs.of(gid)
        ns = " ".join(str(t.n) for t in takes) or "none"
        L.append(f"  {gid:<{w}}  {ns:<20}  "
                 f"{sum(t.facts.hand_frame_count for t in takes):>6}  "
                 f"{sum(t.facts.frames_over_5mm for t in takes):>6}")
    L.append(f"  {'total':<{w}}  {len(gs.takes):<20}  "
             f"{sum(t.facts.hand_frame_count for t in gs.takes):>6}  "
             f"{over5:>6}")
    return L + [""]


def _noted(gs: GraspSet, pick) -> List[Tuple[Take, List[str]]]:
    out = []
    for t in gs.takes:
        notes = pick(t)
        if notes:
            out.append((t, notes))
    return out


def _sec_quality(gs: GraspSet) -> List[str]:
    meta = gs.session.meta
    min_tracked = meta.get("min_tracked_fraction") or MIN_TRACKED_FRACTION
    L = _head(15, "QUALITY CONTROL")
    L += wrapped(
        "At recording (section 5): the acquisition gate, which is about "
        "tracking, and the operator's review against the paper's picture, "
        "which is a judgement by eye. tracked_percent in the summary is "
        "the gate's figure: the labelled hand's frames as a percentage of "
        "the frames the tracker produced over the recording (its rate "
        f"times the window); the gate needs "
        f"{100 * float(min_tracked):.0f}. "
        "reacquisitions_in_static_interval must be 0.")
    tracked = spread([100.0 * float(t.meta["tracked_fraction"])
                      for t in gs.takes
                      if t.meta.get("tracked_fraction") is not None])
    if tracked:
        L += wrapped(
            f"Over the accepted recordings tracked_percent is "
            f"{tracked[0]:.1f} to {tracked[2]:.1f}, median {tracked[1]:.1f}.")
    n_rej = sum(gs.rejected.values())
    if n_rej:
        top = sorted(gs.rejected.items(), key=lambda kv: (-kv[1], kv[0]))
        waits = sum(n for k, n in gs.rejected.items() if is_wait(k))
        L += wrapped(
            f"Attempts not kept: {n_rej}, in the session folder with their "
            "reasons, none in this folder. By kind: "
            + "; ".join(f"{k} {n}" for k, n in top) + "."
            + ("" if not waits else
               f" The {waits} waits for an open hand are not recordings "
               "that failed: no recording was started. "
               f"{AUDIT_FILE} counts them apart (open_hand_timeouts, "
               "rejected_recordings)."))
    L.append("")
    L += wrapped(
        "At the rebuild (section 20): the durations, the frame counts, the "
        "stillness of the static interval, the shape of the whole "
        "recording, the hand's size, the other hand and the hand's "
        "identity are measured from the recordings each "
        "time; the columns marked \"copied\" in section 4 are taken from "
        "the recorder's own record of each take. The representative frame "
        "of each recording is found again in its file, and each "
        f"joint-frame table is checked to be that frame's. {VALIDATION_FILE} "
        "is a second pass over the finished folder: it reads the files "
        "back and recounts. The positions, axes and angles of every "
        "joint-frame table, every column of the angle table, the "
        "stillness of every static interval, the shape of every whole "
        "recording and the hand's size it computes with code of its own, "
        "written from sections 7, 10, 12 and 13 of this README and "
        "sharing nothing with the code that made the tables. For the "
        "choice of the static interval, the choice of the representative "
        "frame and the transition rule it runs the same library code "
        "again: there it shows that the folder agrees with the code, not "
        "that the code is right.")
    L.append("")
    L += wrapped(
        "What is NOT checked by anything: that a hand shape is the grasp "
        "its label names (section 5), what was in the hand (the pictures "
        "are read by eye only, section 1), and how accurate the tracker's "
        "positions and angles are.")
    L.append("")
    L += _between_takes(gs)
    flagged = [t for t in gs.takes if t.facts.abnormal]
    if not flagged:
        L += wrapped("No accepted recording carries a note.")
    else:
        L += wrapped(
            f"Recordings with a note: {len(flagged)} of {len(gs.takes)}. "
            "Each is an accepted recording, kept as recorded; a note says "
            "what a reader should know before using it, and is not a "
            "rejection. The same text is in the notes column of the "
            "summary. By kind (a recording can be under more than one):")
        groups = [
            ("Timing (section 6)", lambda t: t.facts.timing_notes),
            ("The static interval was not still (section 7)",
             lambda t: [n for n in t.facts.pose_notes
                        if not n.startswith(BACK_OF_HAND_NOTE)]),
            ("The recording begins or ends in another shape (section 14)",
             lambda t: t.facts.extent_notes),
            ("The back of the hand toward the camera, as tracked",
             lambda t: [n for n in t.facts.pose_notes
                        if n.startswith(BACK_OF_HAND_NOTE)]),
            ("The hand's identity (section 16)",
             lambda t: [t.facts.identity_flag] if t.facts.identity_flag
             else []),
            ("The representative frame",
             lambda t: [n for n in t.facts.notes
                        if n not in t.facts.timing_notes
                        and n not in t.facts.pose_notes
                        and n not in t.facts.extent_notes
                        and n not in t.facts.picture_notes
                        and n not in t.forming_notes
                        and n != t.facts.identity_flag]),
            # kept under the recorder's rules of 2026-10-07 (section 5)
            ("Lost while forming the grasp, found again before the hold "
             "(sections 5, 17)",
             lambda t: t.forming_notes),
            ("Something in the recorded hand in the picture (section 5)",
             lambda t: [n for k, n in t.picture_notes.items()
                        if k == "hand"]),
            # read off the uncut still: the other hand is not in the picture
            ("Something in the other hand in the uncut still, not in the "
             "cut picture (section 5)",
             lambda t: [n for k, n in t.picture_notes.items()
                        if k == "other"]),
            ("The picture does not settle whether something was in the "
             "hand (section 5)",
             lambda t: [n for k, n in t.picture_notes.items()
                        if k == "unclear"]),
            ("The picture has not been read by eye (section 5)",
             lambda t: [n for k, n in t.picture_notes.items()
                        if k == "unread"]),
        ]
        for title, pick in groups:
            found = _noted(gs, pick)
            if not found:
                continue
            L.append(f"  {title}: {len(found)}")
            for t, notes in found:
                L += wrapped(f"{t.stem}: " + "; ".join(notes), "    ",
                             "      ")
        L += _back_side(gs)
    if gs.problems or gs.missing:
        L.append("")
        L += wrapped("Found wrong or missing by this build:")
        for p in gs.problems + gs.missing:
            L += wrapped(p, "    ", "      ")
    return L + [""]


SPREAD_NOTE_DEG = 30.0     # a joint angle this far apart between takes is listed


def _between_takes(gs: GraspSet) -> List[str]:
    """Section 15's paragraph on how far the takes of one grasp are from
    each other."""
    found = [(gid, between_take_spread(gs.of(gid))) for gid in gs.item_ids]
    found = [(gid, x) for gid, x in found if x is not None]
    if not found:
        return []
    worst = spread([x[0] for _g, x in found])
    pairs = spread([angle_rms(a, b) for a, b in take_pairs(gs)])
    far = [(gid, x) for gid, x in found if x[0] > SPREAD_NOTE_DEG]
    L = wrapped(
        "AGREEMENT BETWEEN THE TAKES OF ONE GRASP. The takes of a grasp are "
        "repeats by label: the same instruction, the same picture. Their "
        "representative frames are not always the same hand shape. For "
        f"each of the {len(found)} grasps with two or more takes, the "
        f"spread of each of the {len(GIVEN_NAMES)} given angles over its "
        "takes (largest minus smallest) was taken; the largest of them is "
        f"between_take_max_spread_deg in {AUDIT_FILE}, with the angle it "
        f"belongs to. It is {worst[0]:.0f} to {worst[2]:.0f} degrees, "
        f"median {worst[1]:.0f}"
        + ("" if pairs is None else
           f"; two takes of one grasp differ by {pairs[1]:.0f} degrees RMS "
           f"over the {len(GIVEN_NAMES)} angles at the median, by up to "
           f"{pairs[2]:.0f}")
        + f". Above {SPREAD_NOTE_DEG:g} degrees in {len(far)}:")
    for gid, (_w, name, _all) in sorted(far, key=lambda x: -x[1][0]):
        vals = " / ".join(jf.fmt_num(t.angles[name], 1)
                          for t in gs.of(gid) if in_angle_table(t))
        L += wrapped(f"{gid}: {name} {vals} (takes "
                     + " ".join(str(t.n) for t in gs.of(gid)
                                if in_angle_table(t))
                     + ")", "    ", "      ")
    L += wrapped(
        "Why takes differ was not examined. A pose made without an object "
        "to hold (section 1) leaves the fingers free, and the tracker "
        "places a finger it "
        "cannot see by its hand model. No take is singled out here as the "
        "grasp: a reader who needs one configuration per grasp should "
        "look at its takes side by side (the PDF has a page for each).")
    return L + [""]


def _back_side(gs: GraspSet) -> List[str]:
    """Section 15's paragraph on the takes fitted from the back of the
    hand: what the folder's own numbers say about whether the fit is real."""
    back = [t for t in gs.takes if back_side(t)]
    if not back:
        return []
    plain = sum(1 for t in back if not t.coached)
    same_side = spread([angle_rms(a, b) for a, b in take_pairs(gs)
                        if back_side(a) == back_side(b)])
    L = wrapped(
        f"About the {_plural(len(back), 'recording')} fitted from the back "
        "of the hand. The tracker's fit is all that is measured, and "
        "nothing in the data excludes a fit that is upside down. "
        f"{plain} of the {len(back)} "
        f"{'is a recording' if plain == 1 else 'are recordings'} made with "
        "the plain countdown, a hand the tracker had to find already "
        "closed. For each, the other takes of its grasp:")
    for t in back:
        others = [o for o in gs.of(t.item) if o is not t
                  and in_angle_table(o) and not back_side(o)]
        if not others or not in_angle_table(t):
            L += wrapped(f"{t.stem}: no take of this grasp fitted from the "
                         "palm side to compare with", "    ", "      ")
            continue
        d = [angle_rms(t, o) for o in others]
        L += wrapped(
            f"{t.stem}: take"
            f"{'' if len(others) == 1 else 's'} "
            + " ".join(str(o.n) for o in others)
            + " of the same grasp "
            f"{'was' if len(others) == 1 else 'were'} fitted from the palm "
            f"side; its {len(GIVEN_NAMES)} angles are "
            f"{_between(min(d), max(d))} degrees RMS from "
            f"{'it' if len(others) == 1 else 'theirs'}", "    ", "      ")
    across = [angle_rms(t, o) for t in back if in_angle_table(t)
              for o in gs.of(t.item)
              if o is not t and in_angle_table(o) and not back_side(o)]
    L += wrapped(
        "So the same grasp was fitted from both sides. Either the hand "
        "was really turned over between those takes, or one of the fits "
        "is upside down. Which was not settled here: the infrared "
        + ("pictures of section 5 show the hand and can settle it; they "
           "were not read for it."
           if gs.imaged else
           "stills kept at the review (in the session folder, not in this "
           "one) show the hand and can settle it.")
        + ("" if same_side is None or not across else
           f" For comparison, by the same {len(GIVEN_NAMES)} angles: a "
           "take fitted from the back is "
           f"{_between(min(across), max(across))} degrees RMS from a "
           "palm-side take of its grasp, and two takes of one grasp fitted "
           f"from the same side differ by {same_side[1]:.0f} degrees RMS "
           f"at the median and by up to {same_side[2]:.0f}."))
    return L


def _other_shape(gs: GraspSet) -> List[str]:
    """Section 16's paragraph on what shape the other hand was in: the
    measurement of `grasp_audit.other_hand_curl`, counted over the takes,
    and what the still notes say of it by eye, when they say anything."""
    share = f"{100 * OTHER_HAND_MIN_SHARE:.0f}"
    diffs = [t.facts.other_hand_finger_curl_minus_own_deg for t in gs.takes
             if t.facts.other_hand_finger_curl_minus_own_deg is not None]
    if not diffs:
        L = wrapped(
            "What shape the other hand was in was not measured: in no "
            f"recording was it in view on {share} % of the frames or more.")
    else:
        alike = sum(1 for d in diffs if abs(d) <= CURL_ALIKE_DEG)
        other = sum(1 for d in diffs if d > CURL_ALIKE_DEG)
        own = sum(1 for d in diffs if d < -CURL_ALIKE_DEG)
        s = spread(diffs)
        L = wrapped(
            "What shape the other hand was in, measured: in the "
            f"{_plural(len(diffs), 'recording')} with another hand on "
            f"{share} % of the frames or more, both hands were measured at "
            "the frame of each nearest the middle of the recording: the "
            "sum of the flexion of the MCP, PIP and DIP joints of the "
            "index, middle, ring and little fingers (flex_deg of their "
            "PROXIMAL, INTERMEDIATE and DISTAL rows, section 12; the thumb "
            "is left out), the other hand's minus the labelled hand's "
            "(other_hand_finger_curl_minus_own_deg in the summary). The two "
            f"are within {CURL_ALIKE_DEG:g} degrees of each other in "
            f"{alike}; the other hand is more curled by more than "
            f"{CURL_ALIKE_DEG:g} in {other}, the labelled hand in {own}. "
            f"The difference runs from {_fmt(s[0], 0)} to {_fmt(s[2], 0)} "
            f"degrees, median {_fmt(s[1], 0)}. It says how alike the two "
            "hands were curled at one moment, not whether they made the "
            "same grasp.")
    by_eye = gs.still_notes.get("other_hand_by_eye")
    if isinstance(by_eye, str) and by_eye.strip():
        L += wrapped(
            f"The other hand, by eye ({notes_who(gs)}): "
            f"{by_eye.strip().rstrip('.')}.")
    elif diffs:
        L += wrapped("The other hand's shape was not read by eye.")
    return L


def _sec_other(gs: GraspSet) -> List[str]:
    takes = gs.takes
    others = [t for t in takes if t.facts.other_hand_detected]
    whole = [t for t in others if t.facts.other_hand_frame_fraction >= 0.9]
    brief = [t for t in others if t.facts.other_hand_frame_fraction < 0.9]
    followed = [t for t in takes if t.facts.hand_selection == FOLLOWED]
    by_label = [t for t in takes if t.facts.hand_selection == BY_LABEL]
    flagged = [t for t in takes if t.facts.identity_flag]
    L = _head(16, "OTHER-HAND DETECTION")
    L += wrapped(
        "A second tracked hand, or a second fit of the same hand, is in the "
        f"file of {len(others)} of the {len(takes)} recordings: on 90 % of "
        f"the frames or more in {len(whole)}, on fewer in {len(brief)}. "
        "other_hand_detected says so per recording, "
        "other_hand_frame_fraction is the share of the recording's frames "
        "in which a second hand appears, and "
        "other_hand_median_distance_cm how far its palm was from the "
        "labelled hand's.")
    L += _other_shape(gs)
    L += wrapped(
        "A second hand does not make a "
        "recording invalid, and its joints are in no derived file: the "
        "keypoints, the per-pose dataset, the joint frames and the angles "
        "use the labelled hand's frames only. Of the other hand only the "
        "palm's position is used, to tell the two apart (here and in "
        "section 10), and its finger flexion at one frame, for the "
        "measurement above. Its "
        "frames stay in the recording's .jsonl, under its own hand_id. "
        "hand_ids in the summary lists the tracker ids of the labelled "
        "hand: the lines of a .jsonl with those hand_id values are the "
        "labelled hand's.")
    for t in brief:
        f = t.facts
        near = (f.other_hand_median_distance_cm is not None
                and f.other_hand_median_distance_cm < FOLLOW_RADIUS_M * 100.0)
        L += wrapped(
            f"{t.stem}: a second hand on {f.other_hand_frames} frame(s), "
            f"{100 * f.other_hand_frame_fraction:.0f} % of the recording"
            + ("" if f.other_hand_median_distance_cm is None else
               f", {f.other_hand_median_distance_cm:.1f} cm from the "
               "labelled hand")
            + (": a second fit on the same hand, not another hand." if near
               else ": the other hand, in view for part of the recording."),
            "    ", "      ")
    L.append("")
    L += wrapped(
        "Which frames are the labelled hand's (hand_selection in the "
        "summary):")
    # the recorder's rule (static_interval.follow_hand), stated as a rule
    radius = f"{FOLLOW_RADIUS_M * 100:.0f} cm"
    L += wrapped(
        f"\"{FOLLOWED}\", {_plural(len(followed), 'recording')}: the "
        "recorder followed the hand it acquired at OPEN HAND by the "
        "tracker's hand_id. A new id appearing where that hand was, in a "
        "frame without it, is treated as the same hand found again and "
        f"followed: a rule of id and distance (within {radius}, the "
        "recorder's FOLLOW_RADIUS_M), not a proof of identity. An id seen "
        f"in the same frame, or more than {radius} away, is treated as "
        "another hand.", "    ", "      ")
    L += wrapped(
        f"\"{BY_LABEL}\", {_plural(len(by_label), 'recording')}, made "
        "before the recorder followed a hand: the hand is the one carrying "
        f"the tracker's {gs.hand} label (the label with the most frames).",
        "    ", "      ")
    # Where the label alone decided with a real second hand in view, what
    # the positions say: the hand being recorded is the one over the camera.
    both = [t for t in by_label if t.palm_cm and t.other_palm_cm
            and (t.facts.other_hand_median_distance_cm or 0.0)
            >= FOLLOW_RADIUS_M * 100.0]
    if both:
        def off_axis(palm) -> float:
            return math.hypot(palm[0], palm[2])
        nearer = sum(1 for t in both
                     if off_axis(t.palm_cm) < off_axis(t.other_palm_cm))
        mine = spread([off_axis(t.palm_cm) for t in both])
        theirs = spread([off_axis(t.other_palm_cm) for t in both])
        L += wrapped(
            f"In {len(both)} of these {len(by_label)} a second hand was in "
            "view, so the tracker's label alone decided. The positions "
            f"agree with it in {nearer} of the {len(both)}: the labelled "
            "hand was the one over the camera, "
            f"{_between(mine[0], mine[2])} cm from the module's vertical "
            f"axis, the other hand {_between(theirs[0], theirs[2])} cm.",
            "      ", "      ")
    L.append("")
    if flagged:
        L += wrapped(
            f"Hand identity needing a note ({len(flagged)}; identity_flag "
            "in the summary):")
        for t in flagged:
            L += wrapped(f"{t.stem}: {t.facts.identity_flag}", "    ",
                         "      ")
    else:
        L += wrapped(
            "Hand identity: in no recording did the tracker change the "
            "labelled hand's id, label it as the other hand, or fit a "
            "second model on it (identity_flag is empty everywhere).")
    return L + [""]


def _move_line(m: Move) -> str:
    """One recorded change of pose, as the README lists it; a mock take's
    line begins with `MOCK_NOTE`."""
    g = m.timing
    notes = m.own_notes
    if m.timed:
        text = (f"{g.validity}: the pose began to change "
                f"{_fmt(g.reaction_time_s)} s after the cue and took "
                f"{_fmt(g.transition_duration_s)} s to settle, a change of "
                f"{_fmt(g.pose_change_mm, 0)} mm RMS"
                + (f" ({notes})" if notes else ""))
    else:
        text = f"{g.validity}: {notes}"
        if g.cue_to_settled_s is not None:
            text += f" (settled {_fmt(g.cue_to_settled_s)} s after the cue)"
    return ((f"{MOCK_NOTE}: " if m.mock else "")
            + f"{m.from_grasp} to {m.to_grasp}, take {m.take}: {text}")


def _validity_counts(moves: Sequence[Move]) -> str:
    """"2 valid, 2 questionable (these have a transition_duration_s), 2
    invalid (these have none)" for a list of recorded changes."""
    n = {v: sum(1 for m in moves if m.timing.validity == v)
         for v in (tr.VALID, tr.QUESTIONABLE, tr.INVALID)}
    return (f"{n[tr.VALID]} valid, {n[tr.QUESTIONABLE]} questionable (these "
            f"have a transition_duration_s), {n[tr.INVALID]} invalid (these "
            "have none)")


def _move_day(gs: GraspSet, m: Move) -> Optional[str]:
    """The day a coached take's forming file was recorded: its take's
    (`Take.recorded_day`, on the recording computer's clock). None for a
    take of a transition session, or a take not in `gs`."""
    if not m.coached:
        return None
    for t in gs.takes:
        if t.item == m.to_grasp and t.ref.take == m.take:
            return t.recorded_day
    return None


def _after_thresholds(gs: GraspSet, moves: Sequence[Move]) -> List[Move]:
    """The coached recordings of `moves` made after the day the pose-speed
    thresholds were read (`tr.THRESHOLDS_READ_ON`): not among the
    recordings they were read from."""
    return [m for m in moves if m.coached
            and (_move_day(gs, m) or "") > tr.THRESHOLDS_READ_ON]


def _thresholds_origin(gs: GraspSet, opened: Sequence[Move]) -> str:
    """Section 17's limit on where the thresholds come from: which of these
    recordings they were read from and which came after, by the day each
    was recorded (`_move_day`) against `tr.THRESHOLDS_READ_ON`."""
    after = _after_thresholds(gs, opened)
    if not after:
        return ("The thresholds were read from these same recordings and "
                "have been tried on no others.")
    late = {id(m) for m in after}
    before = [m for m in opened if id(m) not in late]
    kept = [t for t in gs.takes
            if t.recorded_day and t.recorded_day <= tr.THRESHOLDS_READ_ON]

    def days(moves) -> str:
        return ", ".join(sorted({d for d in (_move_day(gs, m)
                                             for m in moves) if d}))
    source = " and ".join(x for x in (
        f"{_plural(len(before), 'recording')} of these ({days(before)})"
        if before else "",
        f"the held poses of the {_plural(len(kept), 'take')} kept by then"
        if kept else "") if x)
    return (f"The thresholds were read on {tr.THRESHOLDS_READ_ON} "
            + (f"from the recordings made by then: {source}. "
               if source else "from recordings not in this folder. ")
            + f"The other {len(after)} here ({days(after)}) "
            f"{'was' if len(after) == 1 else 'were'} recorded after that; "
            "the thresholds have been tried on no recordings besides "
            "these.")


def _carried_text(gs: GraspSet, opened: Sequence[Move],
                  timed: Sequence[Move], hand_moving: int) -> str:
    """Section 17's limit on hands carried over the camera at the cue, and
    how many of the timed changes were made with the READY step (the open
    hand held still before the cue, `Move.ready_step`), counted; the pause
    as the takes' metas give it (`coaching.ready_pause_s`)."""
    carried = (f"In {hand_moving} of them the hand as a whole was being "
               "moved over the camera at the cue, its pose still "
               "(hand_speed_at_cue_mm_s): the operator was carrying the "
               "hand, not yet closing it." if hand_moving else "")
    if not any(m.ready_step for m in opened):
        return ((carried + " ") if carried else "") + (
            "The recorder now holds the open hand still for 1 s before the "
            "cue; no recording has been made with that yet.")
    keys = {(m.to_grasp, m.take) for m in opened if m.ready_step}
    pause = _span_text(_coaching(t.meta).get("ready_pause_s")
                       for t in gs.takes if (t.item, t.ref.take) in keys)
    ready = [m for m in timed if m.ready_step]
    text = (f"{len(ready)} of the {len(timed)} were recorded with the READY "
            f"step, the open hand held still for {pause} before the cue "
            "(section 5)")
    if not hand_moving:
        return text + "."
    moving = [m for m in timed
              if (m.timing.hand_speed_at_cue_mm_s or 0.0)
              >= tr.MOVE_SPEED_MM_S]
    n = sum(1 for m in moving if m.ready_step)
    return (f"{carried} {text}; {n} of the {len(moving)} carried "
            f"{'was' if n == 1 else 'were'} among them.")


def _alternatives(m: Move) -> str:
    """"at 50 / 20 mm/s 0.46 s; at 12.5 / 5 mm/s 1.51 s" for one timed
    recording: what the same recording reads at other thresholds."""
    parts = []
    for start, settle, g in m.alternatives:
        parts.append(
            f"at {start:g} / {settle:g} mm/s "
            + (f"{g.transition_duration_s:.2f} s"
               if g.validity in (tr.VALID, tr.QUESTIONABLE)
               and g.transition_duration_s is not None
               else "not timed (" + g.notes.split(":")[0].split(" (")[0]
               + ")"))
    return "; ".join(parts)


def palm_choice_text(moves: Sequence[Move]) -> str:
    """One sentence on what leaving the WRIST out of the joints the hand is
    laid on does to the recordings of the table; "" when none was timed
    either way."""
    both = [(m.timing, m.no_wrist) for m in moves if m.no_wrist is not None]
    timed = [(a, b) for a, b in both
             if a.transition_duration_s is not None
             and b.transition_duration_s is not None]
    flipped = sum(1 for a, b in both if a.validity != b.validity)
    if not timed and not flipped:
        return ""
    worst = max((abs(a.transition_duration_s - b.transition_duration_s)
                 for a, b in timed), default=0.0)
    parts = []
    if timed:
        parts.append(f"the {_plural(len(timed), 'duration')} of the table "
                     f"{'changes' if len(timed) == 1 else 'change'} by at "
                     f"most {worst:.2f} s")
    parts.append("no validity changes" if not flipped else
                 f"the validity of {_plural(flipped, 'recording')} changes")
    little = not flipped and worst <= tr.SPEED_STEP_S / 2.0
    return (f"Which joints the hand is laid on matters "
            f"{'little' if little else 'here'}: with the WRIST left out of "
            "the fit, " + " and ".join(parts) + ".")


KNOWN_MOVE_S = 0.5
KNOWN_MOVE_TRAVELS_MM = (10.0, 20.0, 40.0, 80.0, 160.0)
KNOWN_CARRY_MM = 150.0


def known_move_text(thresholds: Optional[dict] = None) -> str:
    """What the timing rule, with these thresholds, reads off synthetic
    changes of pose of one known length and several sizes
    (`transition.known_move_reading`): the README's sentence on it."""
    th = thresholds or {}
    seen, unseen, carried_same = [], [], True
    for travel in KNOWN_MOVE_TRAVELS_MM:
        g = tr.known_move_reading(travel, KNOWN_MOVE_S, **th)
        c = tr.known_move_reading(travel, KNOWN_MOVE_S,
                                  carry_mm=KNOWN_CARRY_MM, **th)
        if (g.transition_duration_s is None) != (
                c.transition_duration_s is None) or (
                g.transition_duration_s is not None and abs(
                    g.transition_duration_s - c.transition_duration_s)
                > 0.012):
            carried_same = False
        if g.transition_duration_s is None:
            unseen.append(f"{g.pose_change_mm:.0f} mm")
        else:
            seen.append(f"{g.transition_duration_s:.2f} s for a change of "
                        f"{g.pose_change_mm:.0f} mm RMS")
    text = ("Run on a synthetic hand whose palm stays still and whose "
            "other 15 joints all travel the same distance in exactly "
            f"{KNOWN_MOVE_S:.2f} s, eased at both ends, it reads "
            + ", ".join(seen) + ".") if seen else ""
    if unseen:
        text += (" A change of " + " or ".join(unseen) + " RMS is not seen "
                 "as a movement at all.")
    if carried_same:
        text += (f" Carrying the whole hand {KNOWN_CARRY_MM:g} mm across "
                 "the camera during the change alters none of these "
                 "readings.")
    return (text + " A small change of pose is read short, or missed; a "
            "duration from this rule is comparable only between changes "
            "of similar size (pose_change_mm).").strip()


def _sec_transition(gs: GraspSet, thresholds: dict) -> List[str]:
    opened, between = gs.from_open, gs.recorded_moves
    start_speed = thresholds.get("start_speed_mm_s", tr.POSE_MOVE_SPEED_MM_S)
    sustain = thresholds.get("start_sustain_s", tr.START_SUSTAIN_S)
    settle_speed = thresholds.get("settle_speed_mm_s",
                                  tr.POSE_SETTLE_SPEED_MM_S)
    window = thresholds.get("settle_window_s", tr.SETTLE_WINDOW_S)
    still = thresholds.get("cue_still_s", tr.CUE_STILL_S)
    step = thresholds.get("step_s", tr.SPEED_STEP_S)
    L = _head(17, "POSE TRANSITION TIMING")
    L += wrapped(
        "The time a hand takes to change from one pose to another is NOT "
        "in the static recordings and is not derived from them. A static "
        "recording is of a hand asked to hold one pose; where one caught "
        "the end of the hand closing (section 14), that is not a cued "
        "change with a known start. The time between two files, a file's "
        "creation time, the gap between two sessions, the operator's "
        "set-up time and rejected attempts all measure the recording "
        "procedure, not the hand, and none of them is used anywhere in "
        "this folder.")
    L += wrapped(
        "A transition time needs its own recording: a starting pose held "
        "still, a cue, the change of pose, and the new pose held until it "
        f"is stable, with every frame kept. {TRANSITIONS_DIR}\\"
        f"{TRANSITIONS_FILE} is the table for recordings of a change of "
        "pose, one row each; a take with no such recording has no row. "
        "Not every recording in it meets that description, and the "
        "validity column says which do.")
    L.append("")
    L += wrapped(
        "WHAT IS TIMED IS THE POSE, NOT THE HAND OVER THE CAMERA. A hand "
        "can be carried across the camera without changing shape, and in "
        "these recordings it was. So every speed the rule reads is the "
        "POSE SPEED: the movement of the joints that is left once the "
        "whole hand's movement has been taken out (defined below). The "
        "speed over the camera, the hand as a whole included, is kept "
        "beside it in the hand_* columns and decides nothing.")
    L.append("")
    if opened:
        hand_moving = sum(
            1 for m in opened if m.timed
            and (m.timing.hand_speed_at_cue_mm_s or 0.0)
            >= tr.MOVE_SPEED_MM_S)
        # the forming files of takes kept after the hand was lost while
        # forming (recorder rules of 2026-10-07): marked, not read
        lost = [m for m in opened if m.forming_lost]
        unchanged = [m.timing.pose_change_mm for m in opened
                     if not m.changed and not m.forming_lost
                     and m.timing.pose_change_mm is not None]
        L += wrapped(
            f"What exists, open hand into a grasp: "
            f"{_plural(len(opened), 'recording')}, made by the coached "
            "procedure (section 5): every frame from the high beep, which "
            "is the cue to close the hand, until the static recording "
            f"starts. They are {TRANSITIONS_DIR}\\{OPEN_HAND}_to_<grasp>_"
            f"{gs.hand}_take<N>.jsonl. "
            f"{sum(1 for m in opened if m.changed)} of them contain a "
            "change of pose after the cue"
            + ("" if all(m.changed or m.forming_lost for m in opened) else
               ("; in the rest the rule found none" if not lost else
                "; in " + str(sum(1 for m in opened if not m.changed
                                  and not m.forming_lost))
                + " the rule found none")
               + ("" if not unchanged else
                  ": their pose at the end of the file is "
                  f"{_between(min(unchanged), max(unchanged))} mm RMS from "
                  "their pose at the cue (pose_change_mm), the hand already "
                  "in or near the grasp when the cue came"))
            + ("" if not lost else
               f"; {len(lost)} {'was' if len(lost) == 1 else 'were'} not "
               "read at all: the hand was lost while forming the grasp and "
               "found again before the hold, which the recorder allows "
               "since its rules changed (section 5), so the change of pose "
               "has a gap in the tracking in it")
            + f". By the rule below, of the {len(opened)}: "
            f"{_validity_counts(opened)}"
            + ("" if not lost else
               f"; {len(lost)} of the invalid because the hand was lost "
               "while forming the grasp (notes: \"the hand was lost while "
               "forming the grasp\")")
            + ".")
        for m in opened:
            L += wrapped(_move_line(m), "    ", "      ")
        timed = [m for m in opened if m.timed]
        if timed:
            slowed = sum(1 for m in timed if m.context)
            days = sorted({d for d in (_move_day(gs, m) for m in timed)
                           if d})
            L += wrapped("How far these numbers go:")
            limits = [
                f"They are {_plural(len(timed), 'change')} into "
                f"{_plural(len({m.to_grasp for m in timed}), 'grasp type')}"
                ", by one operator"
                + ("" if not days else
                   f" on one day ({days[0]})" if len(days) == 1 else
                   f" on {len(days)} days ({', '.join(days)})") + ".",
                "They are coached closes, not fast ones: the procedure "
                "gave the operator a few seconds to close, and asked for "
                "no speed"
                + (f"; {slowed} of them "
                   f"{'is a close' if slowed == 1 else 'are closes'} the "
                   "operator had just been told to make slower, after the "
                   "tracker lost a faster one" if slowed else "")
                + ". reaction_time_s is the time until the pose began to "
                "change, not a reaction time in the experimental sense.",
                "These files begin at the cue, so what the hand did before "
                "the cue is not on record. \"Still at the cue\" means only "
                f"a pose speed under {start_speed:g} mm/s in the "
                f"{still:g} s after it (speed_at_cue_mm_s is the fastest "
                "frame there).",
                _carried_text(gs, opened, timed, hand_moving),
                _thresholds_origin(gs, opened),
            ]
            for text in limits:
                if text:
                    L += wrapped(text, "    - ", "      ")
            L += wrapped(
                "The length depends on where \"settled\" is put, and by a "
                "lot. The same recordings read at twice and at half the "
                "two speed thresholds:", "    - ", "      ")
            for m in timed:
                L += wrapped(
                    f"{m.to_grasp} take {m.take}: "
                    f"{_fmt(m.timing.transition_duration_s)} s as "
                    f"tabulated ({start_speed:g} / {settle_speed:g} "
                    f"mm/s); {_alternatives(m)}", "        ", "          ")
            L += wrapped(
                "At twice the thresholds only the fast part of the close "
                "is counted. The pose begins to move before that part and "
                "goes on creeping after it, and lower thresholds take in "
                "more of both ends. No one of these is the time the change "
                "took: a duration is a number together with its "
                "thresholds.", "      ", "      ")
    else:
        L += wrapped(
            "What exists, open hand into a grasp: nothing. No recording in "
            "this folder has a recorded change from the open hand.")
    if between:
        mocked = [m for m in between if m.mock]
        L += wrapped(
            f"What exists, one held pose into another: "
            f"{_plural(len(between), 'recording')} made with "
            "scripts\\leap\\record_transitions.py: the starting pose held "
            "still, the cue, the change, and the target pose held, from "
            f"before the cue to the end of the hold. They are "
            f"{TRANSITIONS_DIR}\\<from>_to_<to>_<hand>_take<N>.jsonl. Of "
            f"the {len(between)}: {_validity_counts(between)}."
            + ("" if not mocked else
               f" {len(mocked)} of them "
               f"{'is' if len(mocked) == 1 else 'are'} MOCK: a "
               "synthetic hand acting the change out, recorded with --mock "
               "(session " + ", ".join(sorted({m.session for m in mocked}))
               + "), not a hand; the build was told --allow-mock. Their "
               f"lines below and their rows' notes begin \"{MOCK_NOTE}\"."))
        for m in between:
            L += wrapped(_move_line(m), "    ", "      ")
    if not gs.between_grasps:
        L += wrapped(
            "What does not exist: a recording of the change from one GRASP "
            "type to another grasp type. None has been made. No such time "
            "is given or estimated.")
    if not gs.moves:
        L += wrapped("The table has its header row only.")
    L.append("")
    L += wrapped("Columns of " + TRANSITIONS_FILE + ":")
    L += ["    participant_id, hand, from_grasp, to_grasp, take",
          "                            who, and which change (\"open_hand\" "
          "= the open palm)",
          "    recording_file          the recording the row was measured "
          "from",
          "    cue_wall_time           the cue on the computer's clock "
          "(seconds since 1970)",
          "    cue_time_s              the cue",
          "    movement_start_time_s   when the pose started to change",
          "    settled_time_s          when it had settled in the new pose",
          "    transition_duration_s   = settled_time_s - "
          "movement_start_time_s",
          "    reaction_time_s         = movement_start_time_s - cue_time_s",
          "    cue_to_settled_s        = settled_time_s - cue_time_s",
          "    peak_speed_mm_s         the fastest frame of the movement "
          "(pose speed)",
          f"    speed_at_cue_mm_s       the fastest pose speed within "
          f"{still:g} s of the cue",
          "    pose_change_mm          how far the pose at the settle (at "
          "the end, without one) is",
          "                            from the pose at the cue: RMS over "
          "the 26 joints, palm fit",
          f"    hand_speed_at_cue_mm_s  the fastest speed over the camera "
          f"within {still:g} s of the cue",
          "    hand_peak_speed_mm_s    the fastest speed over the camera "
          "from the cue to the settle",
          "    last_movement_time_s    the last frame at the start speed "
          "or faster (pose speed)",
          "    movement_start_method, settled_method   the rule and the "
          "numbers it was run with",
          "    validity, notes         valid / questionable / invalid, and "
          "why",
          "    frames, hand_ids, first_frame_capture_time, analysed_until_s",
          "                            which lines of the recording were "
          "measured: those hand_id",
          "                            values, from that capture_time for "
          "that long",
          "    session, static_take_file   the recording session, and the "
          "static take the change led into"]
    L += wrapped(
        "The *_time_s columns are in seconds from the first measured frame "
        "of the recording file. A recording that begins at the cue has a "
        "cue_time_s within a few milliseconds of 0.", "    ", "    ")
    L.append("")
    L += wrapped("How the two moments are found. Every number below is a "
                 "setting of the analysis, not a constant of it:")
    L += wrapped(
        f"Pose speed, for every frame: take the earliest and the latest "
        f"frame within {step / 2:g} s of it (at an end of the file, the "
        "frames on the one side there is). Lay the later hand on the "
        "earlier by the rotation and translation that best fit the "
        f"palm's {len(tr.PALM_JOINTS)} joints (the WRIST, the PALM, the "
        "five METACARPAL joints and the four finger PROXIMAL joints: the "
        "wrist, the palm centre, the bases of the digits and the "
        "knuckles; least squares, no scaling). The "
        "mean over the 26 joints of the distance left between the two "
        "hands, divided by the time between the two frames, is the pose "
        "speed (mm/s; the frame's capture_time). The speed over the "
        "camera is the same without the laying on. Those "
        f"{len(tr.PALM_JOINTS)} are the part of the hand that bending the "
        "fingers does not move, but they are not a rigid body: the palm "
        "itself does not bend (section 13), yet the WRIST joint shifts "
        "against it when the wrist bends, and the tracker's changing "
        "estimate of the hand's size (section 10) moves them all. Both "
        "are read as pose movement."
        + ("" if not palm_choice_text(gs.moves) else
           " " + palm_choice_text(gs.moves)), "    ", "      ")
    L += wrapped(
        f"THE MOVEMENT: the stretch of frames whose pose speed is at or "
        f"above {start_speed:g} mm/s, {sustain:g} s or longer without a "
        "break, that holds the fastest frame after the cue. MOVEMENT "
        "START is its first frame.", "    ", "      ")
    L += wrapped(
        f"SETTLED: after the movement's fastest frame, the first "
        f"frame from which the pose speed stays below {settle_speed:g} "
        f"mm/s for {window:g} s without a break (a hole in the tracking "
        f"longer than {LOSS_GAP_S:g} s breaks either run).", "    ",
        "      ")
    L += wrapped(
        f"VALID: the pose was still at the cue (no frame at "
        f"{start_speed:g} mm/s or faster within {still:g} s of it; a "
        "pose changing that soon after a beep is taken to have begun "
        "before it, reacting to a sound being slower than that: a general "
        "figure, not one measured here), it changed in one "
        "such movement, and it settled and did not move again before the "
        "recording ended. QUESTIONABLE: the same, but the pose also moved "
        "before its main movement or again after settling (a grip "
        "adjusted); the main movement and its first settle are timed and "
        "the note says what else happened. INVALID: anything else, with "
        "the reason, and no transition_duration_s.", "    ", "      ")
    stills = spread([t.facts.static_interval_mean_speed_mm_s
                     for t in gs.takes])
    held = spread([t.facts.static_interval_pose_speed_mm_s
                   for t in gs.takes])
    peaks = spread([m.timing.peak_speed_mm_s for m in gs.moves
                    if m.timed and m.timing.peak_speed_mm_s is not None])
    if held is not None and stills is not None:
        L += wrapped(
            f"Why {start_speed:g} and {settle_speed:g} mm/s. In the held "
            "poses of the static recordings the pose speed is "
            f"{held[1]:.1f} mm/s at the median over recordings "
            f"({held[2]:.1f} in the least steady; "
            "static_interval_pose_speed_mm_s in the summary), against "
            f"{stills[1]:.0f} mm/s over the camera, most of which is the "
            "hand as a whole. "
            + ("" if peaks is None else
               f"The timed changes peak at {_between(peaks[0], peaks[2])} "
               "mm/s of pose speed. ")
            + f"{settle_speed:g} is several times a held pose and "
            f"{start_speed:g} well under a change's peak. "
            + ("Both were read off these recordings."
               if not _after_thresholds(gs, [m for m in gs.moves
                                             if m.from_grasp == OPEN_HAND])
               else
               f"Both were read on {tr.THRESHOLDS_READ_ON} off the "
               "recordings made by then (How far these numbers go, "
               "above); the figures here are of every recording in this "
               "folder."), "    ", "      ")
    L += wrapped(
        f"How fine: a frame about every 1/90 s, and a speed taken over "
        f"{step:g} s, so neither moment can be placed more finely than "
        f"about {step / 2:g} s. That is the method's grain; how accurate "
        "the timing is against a reference was not measured. Both "
        "moments are crossings of a speed, and the speed is an average "
        f"over {step:g} s, so what the rule reads off a change depends "
        "on how big the change is. " + known_move_text(thresholds)
        + " The cue is the instant the beep was "
        "started on the recording computer; the delay until it is heard "
        "was not measured, and it is part of reaction_time_s, not of "
        "transition_duration_s.", "    ", "      ")
    L.append("")
    L += wrapped(
        "To record more. Open hand into a grasp: every recording made "
        "with the coached procedure from now on has one, with the open "
        "hand held still before the cue. Between two grasps: "
        "scripts\\leap\\record_transitions.py "
        "records the starting grasp held still (kept from 1 s before the "
        "cue), the cue, the change and the target grasp held until stable, "
        "into a session of its own"
        + ("" if any(not m.mock for m in between) else
           "; it has been run on synthetic hands only, not yet on the "
           "camera")
        + ". The rebuild of section 20 adds whatever was recorded to the "
        "table.")
    return L + [""]


def _sec_complete(gs: GraspSet) -> List[str]:
    n = len(gs.items)
    ready = sum(1 for i in gs.item_ids if static_status(gs.of(i)) == READY)
    have = sum(1 for i in gs.item_ids if gs.of(i))
    done = sum(1 for i in gs.item_ids if len(gs.of(i)) >= gs.target)
    L = _head(18, "CURRENT DATASET COMPLETENESS")
    L += wrapped("Three different things, with three different answers.")
    L.append("")
    L += wrapped(
        "A. THE TASK'S REQUIREMENT: at least one labelled static "
        "configuration for every grasp type of the papers.")
    L += wrapped(
        f"STATIC-POSE TASK: {COMPLETE if static_complete(gs) else INCOMPLETE}"
        f". {have} of {n} grasp types have an accepted recording; {ready} "
        f"of {n} are READY, meaning one recording with all four datasets "
        "(representative keypoints, frame-by-frame file, joint-frame XYZ "
        "table, angles). READY is about the files being there and "
        "agreeing with each other; whether a hand shape is the grasp it "
        "is labelled as was the operator's judgement (section 5).",
        "    ", "    ")
    if not static_complete(gs):
        lacking = [i for i in gs.item_ids
                   if static_status(gs.of(i)) != READY]
        L += wrapped("Not READY: " + ", ".join(lacking) + ".", "    ",
                     "    ")
    every = all_noted(gs)
    if gs.takes:
        L += wrapped(
            f"{have - len(every)} of the {n} grasp types have at least one "
            "recording that carries no note (section 15)."
            + ("" if not every else
               " Every recording of " + ", ".join(every) + " carries one "
               f"(recordings_without_a_note in {AUDIT_FILE} is 0 there): "
               "READY says its files are complete, the notes say what a "
               "reader has to know before using them."), "    ", "    ")
    L.append("")
    L += wrapped(
        f"B. OUR REPLICATION TARGET: {gs.target} accepted takes of every "
        f"grasp type, {gs.target_total} recordings. It is the number of "
        "takes set for the collection; the task as written is the one "
        "under A. The takes of a grasp are repeats by label, and do not "
        "always agree in shape (section 15).")
    L += wrapped(
        "FULL MULTI-TAKE DATA COLLECTION: "
        f"{COMPLETE if gs.missing_total == 0 else INCOMPLETE}. "
        f"{len(gs.takes)} of {gs.target_total} recordings; {done} of {n} "
        f"grasp types have their {gs.target}; {gs.missing_total} "
        "recordings still to make (section 19).", "    ", "    ")
    L.append("")
    L += wrapped(
        "C. THE TIME TO CHANGE POSE, asked for in addition to the task "
        "(the task itself says the movement into the grasp need not be "
        "recorded).")
    valid = [m for m in gs.moves if m.timing.validity == tr.VALID]
    doubt = [m for m in gs.moves if m.timing.validity == tr.QUESTIONABLE]
    timed = sorted({m.to_grasp for m in valid + doubt
                    if m.from_grasp == OPEN_HAND})
    L += wrapped(
        f"TRANSITION TIMING: {transition_status(gs)}. "
        + (f"{len(valid)} valid and {len(doubt)} questionable out of "
           f"{_plural(len(gs.moves), 'recording')} of a change"
           if gs.moves else "No change of pose has been recorded")
        + (f"; open hand into a grasp is timed for {len(timed)} of the "
           f"{n} grasp types, by coached closes whose length depends on "
           "the thresholds it is read with" if gs.moves else "")
        + ("; a change between two grasp types has not been recorded"
           if not gs.between_grasps else "")
        + " (section 17).", "    ", "    ")
    L.append("")
    counts: Dict[int, List[str]] = {}
    for gid in gs.item_ids:
        counts.setdefault(len(gs.of(gid)), []).append(gid)
    for k in sorted(counts, reverse=True):
        L += wrapped(f"{_plural(k, 'take')}: {len(counts[k])} grasp "
                     f"type(s): " + ", ".join(counts[k]), "    ", "      ")
    return L + [""]


def _sec_required(gs: GraspSet) -> List[str]:
    L = _head(19, "RECORDINGS STILL REQUIRED")
    need = [(gid, len(gs.of(gid))) for gid in gs.item_ids
            if len(gs.of(gid)) < gs.target]
    unrecorded = [gid for gid in gs.item_ids if not gs.of(gid)]
    L += wrapped(
        "For the task as set (section 18 A): "
        + ("none. Every grasp type has its recording."
           if not unrecorded else
           "a first recording of " + ", ".join(unrecorded) + "."))
    if need:
        L += wrapped(
            f"For our {gs.target}-take replication target (section 18 B), "
            f"{gs.missing_total} more accepted recordings:")
        w = max(len(g) for g, _k in need)
        L.append(f"  {'grasp_name':<{w}}  have  still needed  takes to "
                 "record")
        for gid, k in need:
            have = {t.n for t in gs.of(gid)}
            todo = [n for n in range(1, gs.target + 1) if n not in have]
            L.append(f"  {gid:<{w}}  {k:>4}  {gs.target - k:>12}  "
                     + " ".join(str(n) for n in todo))
    else:
        L += wrapped(f"For our {gs.target}-take replication target (section "
                     "18 B): none, every grasp type has its takes.")
    every = all_noted(gs)
    if every:
        L += wrapped(
            "Worth making all the same: a recording of " + ", ".join(every)
            + " that needs no note. Every one so far carries a note "
            "(section 15).")
    L.append("")
    L += wrapped(
        "They are recorded into the same session with the same procedure "
        "(camera only, bare hand): on the recording computer, "
        "resume_set_a.cmd on the Desktop, which runs")
    L += wrapped(
        f"scripts\\leap\\record_poses.py --protocol protocols\\grasps.json "
        f"--hand {gs.hand} --resume <this session's folder> --takes "
        f"{gs.target}", "    ", "      ")
    # the grasp types no timed change from the open hand leads into, as
    # section 18 C counts them (valid or questionable)
    timed_into = {m.to_grasp for m in gs.moves if m.from_grasp == OPEN_HAND
                  and m.timing.validity in (tr.VALID, tr.QUESTIONABLE)}
    untimed = [gid for gid in gs.item_ids if gid not in timed_into]
    L += wrapped(
        "and records only the missing takes. Each new recording also "
        "captures the change from the open hand into the grasp, with the "
        "open hand held still before the cue (section 17), "
        + ("which the transition timing (section 18 C) already has for "
           "every grasp type" if not untimed else
           "which the transition timing (section 18 C) lacks for every "
           "grasp type" if len(untimed) == len(gs.item_ids) else
           "which the transition timing (section 18 C) lacks for "
           f"{_plural(len(untimed), 'grasp type')}: {', '.join(untimed)}")
        + ". Then the rebuild of section 20 brings this folder up to "
        "date.")
    L += wrapped(
        f"How often a recording of each grasp was rejected, and for what "
        f"most often, is in {AUDIT_FILE} (rejected_recordings, "
        "most_common_reject_reason; open_hand_timeouts counts the waits "
        "that ran out, which are not rejected recordings).")
    L += wrapped(
        "Not recorded, and a question for whoever uses the set: a change "
        "from one grasp type to another (section 17), and the grasps with "
        "a real object in the hand on purpose, as a pass of their own "
        "(section 1 says what the pictures of these recordings show).")
    return L + [""]


def _sec_rebuild(gs: GraspSet, command: str, tool: str) -> List[str]:
    L = _head(20, "REPRODUCING / REBUILDING THE EXPORT")
    L += wrapped(
        "Every file listed in section 4 is written by one command from the "
        "session folder, where the raw recordings are; none of them is "
        "typed or edited by hand:")
    L += wrapped_path(command, "    ", "      ")
    L += wrapped(
        "(on the recording computer: rebuild_professor_grasp_set.cmd on "
        "the Desktop). It")
    steps = [
        "reads session.json and takes the accepted recordings only;",
        "checks each recording's label against the protocol's grasp list;",
        "measures each recording from its own lines: duration from "
        "timestamps, frame counts, the stillness of the static interval, "
        "the shape of the whole recording, the hand's size, the other "
        "hand, the hand's identity;",
        "finds each recording's representative frame again in its file;",
        "writes the joint-frame table and the PDF page of that frame, and "
        "its row of the paper's angles, and the joint key;",
        "cuts each recording's infrared still to the recorded hand and "
        "writes it, and the PDF of the pictures;",
        "copies the recording and its keypoints file, and writes each "
        "grasp's frame-by-frame dataset from the labelled hand's frames;",
        "times every recorded change of pose and writes the transition "
        "table;",
        "writes the summary, the grasp audit and this README with the "
        "counts of that moment;",
        "checks the finished folder and writes " + VALIDATION_FILE
        + " (only when its text has changed).",
    ]
    for k, text in enumerate(steps, 1):
        L += wrapped(text, f"    {k}. ", "       ")
    L += wrapped(
        "The raw recordings are only read, never written. A file of this "
        "folder that the rebuild changes or no longer makes is first moved "
        "to a dated folder beside this one (_superseded), not deleted. "
        "After new recordings, run the same command again.")
    L += wrapped(
        f"Built with: {tool}. Until the changes that are not yet committed "
        "are, the build cannot be repeated from that commit alone."
        if "uncommitted" in tool else f"Built with: {tool}.")
    return L + [""]


def readme_text(gs: GraspSet, operator: str, built: str, command: str,
                tool: str = "not recorded", mock: Sequence[str] = (),
                thresholds: Optional[dict] = None) -> str:
    """The README of a grasp-only hand-in folder. `mock` names the synthetic
    sessions the build was allowed; the mock transition sessions of `gs`
    are named with them whether the caller gave them or not."""
    L: List[str] = []
    L += ["Static grasp poses recorded with the Leap Motion (Ultraleap) "
          "hand tracker", "=" * 78]
    mock = sorted(set(mock) | set(gs.mock_transition_sessions))
    if mock:
        L += wrapped("MOCK DATA: this folder was built with --allow-mock "
                     "and holds synthetic sessions, not recordings of a "
                     "hand: " + ", ".join(sorted(mock)) + ". It is not a "
                     "hand-in.", "", "")
    # wrapped at the page's width when the recording days do not fit one
    # line, its continuation indented like the "Folder built:" line's
    L += [*wrapped(f"Operator: {operator or 'not recorded'} (participant "
                   f"{gs.participant}). Hand: {gs.hand}. Recorded: "
                   f"{', '.join(_days(gs)) or 'not recorded'}.", "", "    "),
          *wrapped(f"Folder built: {built}", "", "    "),
          *wrapped(images_header(gs), "", ""), ""]
    L += wrapped(opening_statement(gs), "", "")
    L.append("")
    L += _where_asked(gs)
    L += ["CONTENTS", "-" * 78]
    titles = ["PURPOSE", "DATASET AT A GLANCE",
              "GRASP TYPES AND PAPER SOURCES", "FOLDER / FILE STRUCTURE",
              "HOW EACH RECORDING WAS CAPTURED", "RECORDING DURATION",
              "STATIC-POSE / REPRESENTATIVE-FRAME SELECTION",
              "RAW LEAP / OPENXR JOINT DATA", "21-POINT KEYPOINT FORMAT",
              "XYZ COORDINATE CONVENTIONS", "JOINT LOCAL FRAMES",
              "JOINT ANGLES", "MAPPING TO THE PAPER'S 24 ANGLES",
              "PER-POSE DATASETS", "QUALITY CONTROL", "OTHER-HAND DETECTION",
              "POSE TRANSITION TIMING", "CURRENT DATASET COMPLETENESS",
              "RECORDINGS STILL REQUIRED",
              "REPRODUCING / REBUILDING THE EXPORT"]
    L += [f"  {k:>2}. {t}" for k, t in enumerate(titles, 1)]
    L.append("")
    L += _sec_purpose(gs)
    L += _sec_glance(gs)
    L += _sec_types(gs)
    L += _sec_files(gs)
    L += _sec_capture(gs)
    L += _sec_duration(gs)
    L += _sec_static(gs)
    L += _sec_raw(gs)
    L += _sec_keypoints(gs)
    L += _sec_xyz(gs)
    L += _sec_frames(gs)
    L += _sec_angles(gs)
    L += _sec_paper(gs)
    L += _sec_perpose(gs)
    L += _sec_quality(gs)
    L += _sec_other(gs)
    L += _sec_transition(gs, thresholds or {})
    L += _sec_complete(gs)
    L += _sec_required(gs)
    L += _sec_rebuild(gs, command, tool)
    return "\n".join(L).rstrip("\n") + "\n"
