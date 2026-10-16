"""Build the hand-in folder for the professor: grasps, finger flexion, sequences.

Plan section 5 and contract `docs/protocol_formats.md` section 9 fix what goes
in; this script is the only thing that writes it, so the folder is rebuilt
from the session folders every time instead of being assembled by hand:

    <out>\\README.txt
    <out>\\grasps\\<item>_<hand>_take<N>.jsonl, ..._keypoints.txt,
                  <item>_<hand>_all_frames.txt, grasps_summary.csv,
                  grasp_types_audit.csv, joint_angles_24.csv,
                  joint_frames\\<item>_<hand>_take<N>.csv, joint_frames.pdf,
                  joint_frames\\joint_key.csv, joint_key.pdf,
                  images\\<item>_<hand>_take<N>.png, grasp_images.pdf
    <out>\\transitions\\pose_transitions.csv,
                  open_hand_to_<item>_<hand>_take<N>.jsonl (takes since
                  2026-10-02)
    <out>\\finger_flexion\\<hand>\\<take>.jsonl, camera_<take>.jsonl (a
                  session that ran the camera), <take>.events.jsonl,
                  <take>.txt, joint_angles\\<take>.csv, cue_angles.csv,
                  flexion_report.txt, <session>_warmup*.json
    <out>\\sequences\\<hand>\\ the same layout, with sequence_check.csv in
                  place of flexion_report.txt

  python scripts/package_professor_set.py --out "..\\xr trainer\\grasp and flexion set for professor 2026-09-28" ^
      --grasps recordings/protocol/grasps/20260928_101500_left ^
      --flexion recordings/protocol/finger_flexion/20260928_140501_left recordings/protocol/finger_flexion/20260928_150210_right ^
      --sequences recordings/protocol/sequences/20260928_143012_left recordings/protocol/sequences/20260928_153340_right ^
      --operator "N Kim"

For the grasp set alone, `scripts/rebuild_grasp_set.py` is the one command:
it makes the session's joint frames, runs this, and checks the result.

WHAT GOES IN, AND WHAT DOES NOT
  Accepted takes only: session.json says which, and a rejected attempt stays
  in `recordings/` with its reason where it can still be counted. The JSONL
  is copied as recorded, because it is the richer source; the professor's
  21-landmark text files are derived from it with the exact block layout of
  `scripts/glove/export_prof_format.py` (imported by path, the way
  `scripts/leap/record_frame.py` does, so the format is defined in one place):

    glove takes   every frame of the operator's glove, wrist at the origin, mm
    grasp takes   the recorder's own medoid file (`keypoints/`) copied, plus
                  every frame of the item's takes in `<item>_<hand>_all_frames`
                  in camera millimetres, the Wrist line giving the measured
                  wrist position; the operator's hand is the one
                  `leap_hand.grasp_audit.operator_lines` picks: the hand ids
                  the recorder followed (`operator_hand_ids` in the meta),
                  else, for a take recorded before it followed one, the
                  tracker label `protocol_check.choose_camera_label` picks

  SET A (the grasps) is described by `leap_hand.grasp_handin`, which reads
  every accepted take once and measures it from its own lines
  (`leap_hand.grasp_audit`): the duration from timestamps, never from a
  count of lines; the frames; the other hand in view; the hand's identity.
  From that one pass come grasps_summary.csv (one row per take),
  grasp_types_audit.csv (one row per grasp type of the protocol, recorded or
  not), joint_angles_24.csv (the paper's 24 angles per take) and the README's
  twenty sections.

  The change into the grasp is a dataset of its own, `transitions\\`: a
  take's forming file, every frame from the close cue to the take, is copied
  there as `open_hand_to_<item>_<hand>_take<N>.jsonl`, and
  pose_transitions.csv has one row per such file with the movement start,
  the settle and the time between (`leap_hand.transition.transition_timing`;
  its thresholds are `--start-speed`, `--start-sustain`, `--settle-speed`,
  `--settle-window` and `--cue-still` here). A take without a forming file,
  every take before 2026-10-02, has no row there, and no static recording
  is ever timed.

  Warm-ups (Sets B and C): every `warmup*.json` of a session, renamed
  `<session>_<file>`, beside its takes: a session resumed after a break has
  one warm-up per run, and each take was judged against its own (session
  section of the README, and `takes[i].warmup` in its session.json). They
  hold the open, fist and single-finger ends every fraction is taken
  between, and say what the glove numbers are.

  Items not recorded (Sets B and C): a protocol item a hand has no accepted
  take of is named in SETUP ("Not recorded with ..."), and under each
  session with what became of its attempts at it (rejected, stopped or
  skipped by the operator, or not tried) and how the session ended; the
  Set B item table says "not recorded" beside it. `--not-recorded-reason`
  adds why, in the README's words after the facts.

  Joint angles (Sets B and C, `cam_hand.angle_series`): the movement sets
  are handed in as angles over time, because one summary frame says nothing
  about a movement. `joint_angles\\<take>.csv` has every glove frame of the
  take with the cue in force and the paper's 24 angles; `cue_angles.csv`,
  one per hand, has the median of each angle at every held cue (Set B: each
  hold and rest; Set C: each step, over the checker's window). The README
  prints the span each angle took and names the nine that are the glove's
  hand model and not measurements.

  With or without a camera: a session recorded with `--camera none` is the
  glove alone (session.json "camera": "none"). The README, the flexion
  report and the known limits then say so and claim nothing a camera would
  have measured (`ran_camera`); a folder that mixes both kinds of session
  names the glove-only ones.

  Joint frames, when `scripts/joint_frames_view.py --session` has been run
  on a session: its `joint_frames/<take>.csv` (and `camera_<take>.csv`) for
  the accepted takes, renamed like the take's other files, and the one
  `joint_frames.pdf`, into a `joint_frames` folder beside the set's files.
  Never its PNGs: the PDF is the drawing's hand-in form. Without that
  folder the set packages without them, and the grasp README says so.

  Pictures: one per grasp take, and nothing else. A take's infrared still
  shows the operator's face, so it never goes in as it is; the build cuts
  it to the recorded hand (`leap_hand.grasp_images.write_hand_image`) and
  writes that as `grasps\\images\\<item>_<hand>_take<N>.png`, with
  `grasp_images.pdf` beside it (one page per grasp type). Those PNGs are
  the only images the folder may hold (`allowed_images`: planned under
  `grasps\\images\\`, written from a still, marked `image_ok` by
  `plan_grasps` alone). Before anything is written, every file the plan
  would put in the folder is checked by name and, for copies, by its first
  bytes; one image outside that set and nothing is written and the exit
  code is 1. So is a folder that already holds an image outside it, and
  after writing the whole folder is scanned the same way.

  The joint key (`xr_hand.joint_key`): `joint_key.csv` and `joint_key.pdf`
  beside the grasp set's joint_frames.pdf, written by this build.

NOTHING IS DESTROYED
  The session folders are only read. In the hand-in folder a file the build
  would write with the same bytes is left alone; one whose bytes change, and
  one in `grasps\\`, `transitions\\`, `finger_flexion\\` or `sequences\\` that
  this build no longer makes, is moved first to
  `<out's parent>\\_superseded\\<out's name>\\<time of the build>\\` (or
  `--superseded`), keeping its place under it. A file at the top of the
  folder that the build does not make is left where it is and named.

REPORTS COME FROM THE CHECKER
  `flexion_report.txt` and `sequence_check.csv` are built from each session's
  `check.csv` / `check.txt` (`scripts/check_protocol.py`) when those cover
  exactly the session's takes and hold every column the checker writes
  today. When they are missing, older than a retake or written before a
  column existed (the bend times, 2026-10-13), the checker's functions are
  run here in memory (fixed bands) and a line says so; nothing is written
  back into the session folder.

  How fast the Set B bends were (`recording_protocol.bend_moves`, through
  the checker): each take's median bend and straighten time over the
  seconds its cue gave, in two columns of flexion_report.txt, and under
  each session whether its bends were judged on the bend-speed rule
  (session.json `speed_check`). The README's QUALITY RULES state the rule
  with the sessions' own numbers and say which sessions were under it; a
  session recorded before the rule is handed in with its times as they are.

A session folder that does not exist, or holds no session.json, is skipped
with a printed line, not an error. So is a MOCK session (session.json with
`"mock": true`, a `"mock"` dict with any flag set, or a `"mock_flags"` dict
with any flag set; see `protocol_check.mock_flags`): synthetic hands are for
rehearsing the tools, never for the professor, and `--allow-mock` is the
only way one gets in (for rehearsing this script, and the README then says
so at the top). The same holds for a transition session given with
`--transitions`: a mock one is left out without `--allow-mock`; with it, the
README's MOCK DATA line names it, its rows' notes in pose_transitions.csv
and its takes' lines in section 17 begin "MOCK (synthetic hand)", and
`scripts/validate_grasp_set.py` fails the folder unless it is given
`--allow-mock` too. The exit code is 1 for an image outside the planned
pictures, or when not one session could be packaged.
"""
import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cam_hand import angle_series as asr
from cam_hand import protocol_check as pc
from leap_hand import grasp_handin as gh
from leap_hand import grasp_images as gi
from leap_hand.grasp_audit import operator_lines
from xr_hand.joints import HandFrame

REPO = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".mp4")
# First bytes of the same formats, so a renamed image is still caught.
IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM")

SET_TITLES = {pc.GRASPS: "Set A, static grasp-type poses",
              pc.FLEXION: "Set B, single-finger flexion",
              pc.SEQUENCES: "Set C, sequential movements"}

KNOWN_LIMITS = [
    "The thumb has flexion only: the glove has no sensor for opposition, so "
    "a thumb crossing the palm reads as a thumb bending.",
    "Finger spread is not measured by the glove: its hand model sets each "
    "finger's spread from that finger's bend (a fixed value with the hand "
    "open that, for most fingers, shifts by a few degrees as the finger "
    "bends). Abduction is measured only in the camera files.",
    "The right glove sometimes reads too open, on a session-dependent "
    "basis; compare its fractions with the camera beside it.",
    "Curl creeps during holds: a held finger's glove reading drifts slowly "
    "while the camera's does not.",
    "Lag differs by hand: the left glove trails the camera by about 0.10 s, "
    "the right by about 0.47 s (measured per cycle and per take in "
    "flexion_report.txt, never assumed).",
]
# The same five for a folder none of whose glove sessions ran a camera: what
# a camera would have shown is said to be absent, never implied.
KNOWN_LIMITS_GLOVE_ONLY = [
    KNOWN_LIMITS[0],
    "Finger spread is not measured by the glove: its hand model sets each "
    "finger's spread from that finger's bend (a fixed value with the hand "
    "open that, for most fingers, shifts by a few degrees as the finger "
    "bends), so no abduction was measured in these recordings.",
    "A glove can read too open, on a session-dependent basis (seen on the "
    "right glove in earlier sessions with a camera beside it). Here the "
    "session's own warm-up (the open, fist and single-finger ends) is the "
    "only reference for the fractions.",
    "Curl creeps during holds: a held finger's glove reading drifts slowly "
    "(measured against a camera in earlier sessions).",
    "The glove trails the hand. Earlier sessions with a camera beside the "
    "gloves measured about 0.1 s on one glove and about 0.5 s on the other, "
    "and which glove was the slow one changed between sessions. No camera "
    "ran here, so the lag of these recordings is not measured: a cue's time "
    "in the events file is the beep, not the movement.",
]

ANGLES_DIR = "joint_angles"            # <hand>\\joint_angles\\<take>.csv
CUE_ANGLES_FILE = "cue_angles.csv"     # <hand>\\cue_angles.csv


def ran_camera(session: pc.Session) -> bool:
    """Whether the session ran a camera beside the glove
    (`protocol_check.ran_camera` on its session.json)."""
    return pc.ran_camera(session.meta)


def camera_split(sessions: Sequence[pc.Session]
                 ) -> Tuple[List[pc.Session], List[pc.Session]]:
    """(the sessions that ran a camera, the glove-only ones)."""
    return ([s for s in sessions if ran_camera(s)],
            [s for s in sessions if not ran_camera(s)])


class AngleTables:
    """The per-frame angle rows of each take, worked out once per build:
    the take's own table, its hand's cue table and the README's spans all
    read the same rows."""

    def __init__(self) -> None:
        self._rows: Dict[str, List[dict]] = {}
        self._events: Dict[str, List[dict]] = {}

    def events(self, take: pc.TakeRef) -> List[dict]:
        key = str(take.files.get("events") or take.name)
        if key not in self._events:
            self._events[key] = pc.read_events(take.path("events"))
        return self._events[key]

    def rows(self, take: pc.TakeRef, hand: str) -> List[dict]:
        key = f"{take.path('glove')}|{hand}"
        if key not in self._rows:
            self._rows[key] = asr.take_rows(take.path("glove"),
                                            self.events(take), hand)
        return self._rows[key]


def prof_exporter():
    """`scripts/glove/export_prof_format.py`, imported by path.

    The block layout (the `Frame` header, the `Wrist:` line, landmarks 0 to
    20, millimetres) is defined there and must stay defined in one place, or
    this folder's text files and the earlier hand-ins drift apart while both
    claim to be his format. `scripts/glove/` is not a package; its `main()`
    is behind `if __name__ == "__main__"`, so importing runs nothing.
    """
    path = REPO / "scripts" / "glove" / "export_prof_format.py"
    spec = importlib.util.spec_from_file_location("glove_prof_format", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrist_at_origin(frame: HandFrame) -> HandFrame:
    """The same frame with its root (the wrist) moved to (0, 0, 0).

    Every joint is parent-relative and the wrist is the root, so its
    translation is the whole hand's offset: zeroing it moves every point by
    minus the wrist and leaves the hand's shape and orientation alone. The
    glove already streams its wrist at the origin; this makes the text files
    say so even for a stream that did not.
    """
    joints = [replace(j, x=0.0, y=0.0, z=0.0) if j.name == "WRIST" else j
              for j in frame.joints]
    return replace(frame, joints=joints)


def write_blocks(dest: Path, blocks) -> int:
    """Blocks separated by a blank line, as `export_prof_format.py` writes."""
    n = 0
    with open(dest, "w", encoding="utf-8") as out:
        for block in blocks:
            out.write(("\n\n" if n else "") + block)
            n += 1
        if n:
            out.write("\n")
    return n


def glove_blocks(src: Path, hand: str, exporter):
    """Every frame of the operator's glove in `src`, wrist-origin mm blocks."""
    frames, _bad = pc.read_frames(src)
    for d, frame in frames:
        if d.get("hand_side") == hand:
            yield exporter.frame_block(wrist_at_origin(frame),
                                       float(d["wall_time"]))


def camera_blocks(srcs: Sequence[Tuple[Path, Optional[dict]]],
                  hand: str, exporter):
    """Every frame of the operator's hand in camera mm, take after take.

    `srcs` are (leap file, the take's meta or None). Whose lines they are
    is `leap_hand.grasp_audit.operator_lines`'s call, the same one the
    summary and the checks make: with `operator_hand_ids` in the meta, the
    lines of those ids, whatever the tracker called them (the recorder
    followed that hand by id, and the other hand in view is in the same
    file under other ids); without (a take recorded before the recorder
    followed one), the tracker label `protocol_check.choose_camera_label`
    picks, the same rule the checker uses.
    """
    for src, meta in srcs:
        frames, _bad = pc.read_frames(src)
        mine, _how = operator_lines([d for d, _f in frames], meta, hand)
        for i, d in mine:
            yield exporter.frame_block(frames[i][1], float(d["wall_time"]))


# --- the plan: every file the folder will hold -----------------------------------

@dataclass
class Output:
    """One file of the hand-in folder: copied from `source` or written."""

    dest: Path                                   # relative to the out folder
    source: Optional[Path] = None
    write: Optional[Callable[[Path], object]] = None
    # a picture this build cuts from a still: set by `plan_grasps` alone,
    # and only for a .png under grasps\images (`allowed_images`)
    image_ok: bool = False


IMAGES_DIR = Path(gh.GRASPS_DIR) / gh.IMAGES_DIR


def image_key(dest) -> str:
    """A file's place in the folder as the image checks compare it."""
    return Path(dest).as_posix().lower()


def allowed_images(plan: Sequence[Output]) -> set:
    """The only images the folder may hold: the planned PNGs under
    grasps\\images that are written (not copied) and marked `image_ok`."""
    return {image_key(o.dest) for o in plan
            if o.image_ok and o.write is not None and o.source is None
            and o.dest.suffix.lower() == ".png"
            and o.dest.parent == IMAGES_DIR}


def is_image(path: Path, check_bytes: bool = True) -> bool:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return True
    if not check_bytes or not path.is_file():
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return head.startswith(IMAGE_MAGIC) or head[4:8] == b"ftyp"


def image_violations(plan: Sequence[Output]) -> List[str]:
    """Every planned file that would be an image outside `allowed_images`:
    by its name, or for a copy by its source's first bytes (no image is
    ever copied in)."""
    allowed = allowed_images(plan)
    bad = []
    for o in plan:
        if o.source is not None and is_image(o.source):
            bad.append(f"{o.dest} (copied from the image {o.source})")
        elif (is_image(o.dest, check_bytes=False)
              and image_key(o.dest) not in allowed):
            bad.append(f"{o.dest} (an image name)")
    return bad


def images_in(folder: Path) -> List[Path]:
    """Every image in the folder, by name or by first bytes, however deep."""
    return [p for p in sorted(os_path(folder).rglob("*")) if p.is_file()
            and is_image(p)]


def stray_images(folder: Path, allowed: set) -> List[Path]:
    """The images in the folder that are not among `allowed` (places in the
    folder, `image_key`)."""
    root = Path(plain(os_path(folder)))
    return [p for p in images_in(folder)
            if image_key(Path(plain(p)).relative_to(root)) not in allowed]


def via_suffix(write: Callable[[Path], object], suffix: str
               ) -> Callable[[Path], object]:
    """`write` made to write to a path ending in `suffix` and the result
    moved to the path it is given: `write_plan` builds a file beside its
    place under a name ending in .building, and a writer may choose its
    format by the name's ending."""
    def run(dest: Path) -> None:
        tmp = Path(str(dest) + suffix)
        write(tmp)
        move_into_place(tmp, dest)
    return run


def os_path(path) -> Path:
    """`path` in a form Windows opens past 260 characters.

    The hand-in folder sits under a long OneDrive path and a take name like
    `camera_seq6_configurations_right_take3_20260928_143012.jsonl` is 60
    characters on its own, so a deep `--out` crosses MAX_PATH; the
    extended-length prefix lifts that limit. It is added to every path on
    Windows, short ones included, because a folder scanned from a short root
    still has to see the long files inside it (`is_file` on a 270-character
    path without the prefix says False, which would let an image hide from
    the final scan). Elsewhere the path is only made absolute.
    """
    return gh.long_path(path)


def plain(path) -> str:
    """A path as the operator typed it, without the extended-length prefix."""
    s = str(path)
    if s.startswith("\\\\?\\UNC\\"):
        return "\\\\" + s[8:]
    return s[4:] if s.startswith("\\\\?\\") else s


def load_sessions(dirs: Sequence[Path], kind: str,
                  allow_mock: bool = False) -> List[pc.Session]:
    """The sessions of one set that exist; a line for each one that does not.

    A mock session is skipped the same way unless `allow_mock`.
    """
    out, seen = [], set()
    for d in dirs or []:
        d = Path(d)
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        if not d.is_dir():
            print(f"skipped {kind}: {d} does not exist")
            continue
        try:
            session = pc.load_session(d)
        except FileNotFoundError:
            print(f"skipped {kind}: no session.json in {d}")
            continue
        if session.set != kind:
            print(f"skipped {kind}: {d} is a {session.set} session")
            continue
        if pc.mock_flags(session.meta) and not allow_mock:
            print(f"mock session, not handed in: {d}")
            continue
        out.append(session)
    return out


def accepted(session: pc.Session) -> List[pc.TakeRef]:
    return [t for t in session.takes if t.accepted]


# --- Set A ---------------------------------------------------------------------------
# What the folder says about Set A is worked out in `leap_hand.grasp_handin`
# (one pass over the accepted takes); here its files are planned. The names
# below are kept for the callers that knew them here.
SUMMARY_COLUMNS = gh.SUMMARY_COLUMNS
take_sensor = gh.take_sensor
forming_file = gh.forming_file
reason_kind = gh.reason_kind

JOINT_FRAMES_DIR = gh.JOINT_FRAMES_DIR      # scripts/joint_frames_view.py --session
JOINT_FRAMES_PDF = "joint_frames.pdf"
# `leap_hand.transition.transition_timing`'s thresholds, as options here.
THRESHOLD_OPTIONS = (("start_speed", "start_speed_mm_s"),
                     ("start_sustain", "start_sustain_s"),
                     ("settle_speed", "settle_speed_mm_s"),
                     ("settle_window", "settle_window_s"),
                     ("cue_still", "cue_still_s"))


KEY_FOOTER = ("The rows of every joint-frame table of this folder; see "
              "README.txt sections 11 to 13.")


def plan_joint_frames(session: pc.Session, folder: Path,
                      names: Sequence[Tuple[str, str]],
                      pdf_name: str = JOINT_FRAMES_PDF, key: bool = False
                      ) -> Tuple[List[Output], List[str]]:
    """The session's joint-frame exports, when it has them.

    `names` pairs each handed-in take's session name with its name in the
    hand-in folder. A CSV of a take that is not handed in stays out, with a
    line saying so; a PNG is never looked at. Returns ([], []) for a session
    without a joint_frames folder, which then packages as before. With
    `key` (the grasp set, whose README carries it) the joint key is written
    beside them: joint_key.csv and joint_key.pdf (`xr_hand.joint_key`).
    """
    src = session.path / JOINT_FRAMES_DIR
    if not src.is_dir():
        return [], []
    wanted: Dict[str, str] = {}
    for name, stem in names:
        wanted[f"{name}.csv"] = f"{stem}.csv"
        wanted[f"camera_{name}.csv"] = f"camera_{stem}.csv"
    plan: List[Output] = []
    notes: List[str] = []
    for p in sorted(src.glob("*.csv")):
        dest = wanted.get(p.name)
        if dest is None:
            notes.append(f"{session.name}/{JOINT_FRAMES_DIR}/{p.name}: not an "
                         "accepted take, not handed in")
            continue
        plan.append(Output(folder / JOINT_FRAMES_DIR / dest, source=p))
    pdf = src / JOINT_FRAMES_PDF
    if pdf.is_file():
        plan.append(Output(folder / JOINT_FRAMES_DIR / pdf_name, source=pdf))
    if key:
        from xr_hand import joint_key as jk
        plan.append(Output(folder / JOINT_FRAMES_DIR / gh.KEY_CSV,
                           write=jk.write_key_csv))
        plan.append(Output(folder / JOINT_FRAMES_DIR / gh.KEY_PDF,
                           write=via_suffix(lambda dest: jk.write_key_pdf(
                               dest, KEY_FOOTER), ".pdf")))
    return plan, notes


KEY_FILES = (gh.KEY_CSV, gh.KEY_PDF)


def joint_frame_counts(plan: Sequence[Output]) -> Tuple[int, int]:
    """(joint-frame CSVs, joint-frame PDFs) among planned outputs; the joint
    key's two files are not tables of a take."""
    frames = [o for o in plan if o.dest.parent.name == JOINT_FRAMES_DIR
              and o.dest.name not in KEY_FILES]
    return (sum(1 for o in frames if o.dest.suffix == ".csv"),
            sum(1 for o in frames if o.dest.suffix == ".pdf"))


def plan_grasps(session: pc.Session, exporter, participant: str = "P01",
                thresholds: Optional[dict] = None,
                transitions: Sequence[Path] = (), allow_mock: bool = False
                ) -> Tuple[List[Output], dict]:
    """Every file of Set A, and the set as `leap_hand.grasp_handin` read it
    (`info["set"]`), which the README is written from. `transitions` are
    session folders of `scripts/leap/record_transitions.py`, whose accepted
    takes join the transition table."""
    hand = session.hand
    gs = gh.collect(session, participant, **(thresholds or {}))
    unread: List[str] = []
    for extra in transitions:
        unread += gh.add_transition_session(gs, extra, allow_mock,
                                            **(thresholds or {}))
    folder = Path(gh.GRASPS_DIR)
    moves = Path(gh.TRANSITIONS_DIR)
    plan: List[Output] = []
    by_item: Dict[str, List[Tuple[int, Path, dict]]] = {}
    names: List[Tuple[str, str]] = []
    for t in gs.takes:
        leap = t.ref.path("leap")
        names.append((t.ref.name, t.stem))
        plan.append(Output(folder / f"{t.stem}.jsonl", source=leap))
        by_item.setdefault(t.item, []).append((t.n, leap, t.meta))
        if t.keypoints is not None:
            plan.append(Output(folder / f"{t.stem}_keypoints.txt",
                               source=t.keypoints))
    # Every recorded change of pose, as recorded: a coached take's forming
    # file (every frame from the close cue to the hold), and the takes of
    # the transition sessions.
    for m in gs.moves:
        plan.append(Output(Path(m.dest), source=m.source))
    for item, takes in sorted(by_item.items()):
        srcs = [(p, meta) for _n, p, meta in sorted(takes,
                                                    key=lambda x: x[:2])]
        plan.append(Output(
            folder / f"{item}_{hand}_all_frames.txt",
            write=lambda dest, srcs=srcs: write_blocks(
                dest, camera_blocks(srcs, hand, exporter))))
    plan.append(Output(folder / gh.SUMMARY_FILE,
                       write=lambda dest: gh.write_csv(
                           dest, gh.SUMMARY_COLUMNS, gh.summary_rows(gs))))
    plan.append(Output(folder / gh.AUDIT_FILE,
                       write=lambda dest: gh.write_csv(
                           dest, gh.AUDIT_COLUMNS, gh.audit_rows(gs))))
    # One row per recorded change of pose; the header alone when there is
    # none, so the folder says what the table would hold.
    plan.append(Output(moves / gh.TRANSITIONS_FILE,
                       write=lambda dest: gh.write_csv(
                           dest, gh.TRANSITION_COLUMNS,
                           gh.transition_rows(gs))))
    frames, notes = plan_joint_frames(session, folder, names, key=True)
    plan += frames
    csvs, pdfs = joint_frame_counts(frames)
    if gs.has_joint_frames:
        plan.append(Output(folder / gh.ANGLES_FILE,
                           write=lambda dest: gh.write_csv(
                               dest, gh.ANGLE_COLUMNS, gh.angle_rows(gs))))
    # The pictures: each take's still cut to the recorded hand, the only
    # images the folder may hold (`allowed_images`), and the PDF of them.
    for t in gs.imaged:
        plan.append(Output(IMAGES_DIR / f"{t.stem}.png",
                           write=lambda dest, s=t.still, h=hand:
                           gi.write_hand_image(s, dest, h), image_ok=True))
    if gs.imaged:
        plan.append(Output(IMAGES_DIR / gh.IMAGES_PDF,
                           write=via_suffix(lambda dest: gi.write_images_pdf(
                               dest, gh.image_pdf_pages(gs),
                               gh.images_pdf_footer(gs)), ".pdf")))
    return plan, {"takes": len(gs.takes), "items": len(by_item),
                  "missing": gs.missing + notes + unread,
                  "problems": gs.problems,
                  "joint_frames": csvs, "joint_frames_pdf": pdfs,
                  "images": len(gs.imaged), "set": gs}


# --- Sets B and C --------------------------------------------------------------------

def check_for(session: pc.Session) -> Tuple[List[Dict[str, str]], str, str]:
    """(check.csv rows, check.txt text, where they came from) for a session."""
    csv_path, txt_path = session.path / "check.csv", session.path / "check.txt"
    if csv_path.is_file() and txt_path.is_file():
        rows = pc.read_check_csv(csv_path)
        if pc.check_is_current(session, rows):
            return (rows, txt_path.read_text(encoding="utf-8"),
                    "check.csv and check.txt in the session folder")
        why = ("are not current (other takes than the session's, or "
               "written before a column the checker now has)")
    else:
        why = "are missing"
    print(f"  {session.set} {session.name}: check.csv/check.txt {why}; "
          "running the checker in memory (fixed bands)")
    check = pc.check_session(session)
    return (pc.csv_rows(check), pc.render_text(check),
            "checked while packaging (fixed bands)")


def warmup_files(session: pc.Session) -> List[Path]:
    """Every warm-up file of a session folder: warmup.json, then a resumed
    session's warmup_<HHMMSS>.json files in the order they were written."""
    found = [p for p in session.path.glob("warmup*.json") if p.is_file()]
    return sorted(found, key=lambda p: (p.name != pc.WARMUP_FILE, p.name))


def handed_in(session: pc.Session) -> List[pc.TakeRef]:
    """The accepted takes that have a glove file: the ones handed in."""
    return [t for t in accepted(session) if t.path("glove") is not None]


def plan_glove_takes(session: pc.Session, kind: str, exporter,
                     pdf_name: str = JOINT_FRAMES_PDF,
                     tables: Optional[AngleTables] = None
                     ) -> Tuple[List[Output], List[str]]:
    folder = Path(kind) / session.hand
    tables = tables if tables is not None else AngleTables()
    plan: List[Output] = []
    missing: List[str] = []
    names: List[Tuple[str, str]] = []
    for t in accepted(session):
        glove = t.path("glove")
        if glove is None:
            missing.append(f"{t.name}: no glove file, take not handed in")
            continue
        names.append((t.name, t.name))
        plan.append(Output(folder / f"{t.name}.jsonl", source=glove))
        plan.append(Output(
            folder / f"{t.name}.txt",
            write=lambda dest, src=glove, hand=session.hand: write_blocks(
                dest, glove_blocks(src, hand, exporter))))
        plan.append(Output(
            folder / ANGLES_DIR / f"{t.name}.csv",
            write=lambda dest, t=t, hand=session.hand: asr.write_csv(
                dest, asr.FRAME_COLUMNS, tables.rows(t, hand))))
        if t.path("leap"):
            plan.append(Output(folder / f"camera_{t.name}.jsonl",
                               source=t.path("leap")))
        if t.path("events"):
            plan.append(Output(folder / f"{t.name}.events.jsonl",
                               source=t.path("events")))
        else:
            missing.append(f"{t.name}: no events file")
    # Every warm-up the takes were judged against, named by session (two
    # sessions of one hand share this folder).
    for p in warmup_files(session):
        plan.append(Output(folder / f"{session.name}_{p.name}", source=p))
    frames, notes = plan_joint_frames(session, folder, names, pdf_name)
    return plan + frames, missing + notes


def _j(row: Dict[str, str], key: str):
    try:
        return json.loads(row.get(key) or "null")
    except ValueError:
        return None


# The report's key to its span and cycles columns. A fraction runs from the
# finger's open-palm reading to its own end of the warm-up: the cued finger's
# bend on its own when that was usable, else the fist (the report's "range
# end" rule); the other fingers' end is the fist.
SPAN_HEADER = ("span = 5th to 95th percentile of the glove fraction over the "
               "take, 0 = warm-up open palm, 1 = the finger's own end of the "
               "warm-up: for the cued finger its bend on its own when that "
               "was usable, else the fist, for the other fingers the fist "
               "(range end, under Rules in force at the end). cycles "
               "ev/pk/exp = bend cues / glove curl peaks / the protocol's "
               "count.")


def flexion_report(hand: str, parts: List[Tuple[pc.Session, list, str, str]]
                   ) -> str:
    cam, glove_only = camera_split([part[0] for part in parts])
    lines = [f"Finger flexion (Set B), {hand} hand", "=" * 78]
    if cam:
        lines += [
            "Glove at full rate, Ultraleap camera as the reference. One row "
            "per handed-in take; the",
            "checker's full output for each session follows, rejected "
            "attempts included and marked.",
            ] + wrapped(SPAN_HEADER, "", "") + [
            "Lag is per cycle (median shown) and per take, positive = the "
            "glove trails the camera, only",
            "where the camera followed the finger (camera span >= 0.50)."]
        if glove_only:
            lines += wrapped(
                "No camera ran in " + ", ".join(s.name for s in glove_only)
                + ": its rows say no cam and have no camera span, lag or "
                "hysteresis.", "", "")
    else:
        lines += [
            "Glove at full rate, no camera: the glove is the only sensor of "
            "these recordings. One row",
            "per handed-in take; the checker's full output for each session "
            "follows, rejected attempts",
            "included and marked."] + wrapped(SPAN_HEADER, "", "") + [
            "cam span, followed, lag and hyst need a camera beside the glove "
            "and are empty here (no cam)."]
    marks = pc.rule_marks(None)
    lines += wrapped(
        "bend s and open s = how long the glove's reading took to cross "
        f"from {marks[0] * 100:.0f} % to {marks[1] * 100:.0f} % of a bend's "
        "own rise, and to fall back across it: the median over the take's "
        "bends, in seconds, then the seconds the cue gave for the movement "
        "(0.17/4 = 0.17 s of a 4 s cue). Every bend's own time is in the "
        "checker's output below. Whether a session's bends were judged on "
        "it is said under the session.", "", "")
    lines.append("")
    head = (f"  {'take':<38} {'finger':<7} {'span':>5} {'cycles':>9} "
            f"{'bend s':>9} {'open s':>9} "
            f"{'cam span':>8} {'followed':>8} {'lag med':>7} {'lag take':>8} "
            f"{'hyst':>6}  others")
    for session, rows, text, source in parts:
        lines.append(f"Session {session.name} ({session.meta.get('started')} "
                     f"to {session.meta.get('ended')}), {source}")
        lines += wrapped(pc.speed_rule_text(pc.speed_rule_of(session.meta))
                         + ".", "  ", "  ")
        lines.append(head)
        for r in rows:
            if r.get("accepted") != "true":
                continue
            others = _j(r, "other_spans") or {}
            cyc = (f"{r.get('cycles_from_events') or '-'}/"
                   f"{r.get('cycles_from_peaks') or '-'}/"
                   f"{r.get('cycles_expected') or '-'}")
            followed = (f"{r.get('cycles_camera_followed') or '0'}/"
                        f"{r.get('cycles_from_events') or '-'}"
                        if r.get("frames_camera") else "no cam")
            lines.append(
                f"  {r['name'][:38]:<38} {r.get('cued_finger') or '-':<7} "
                f"{_num(r.get('cued_span_fraction')):>5} {cyc:>9} "
                f"{_over_cue(r, 'bend'):>9} {_over_cue(r, 'straighten'):>9} "
                f"{_num(r.get('camera_span')):>8} {followed:>8} "
                f"{_num(r.get('lag_ms_median'), '.0f'):>7} "
                f"{_num(r.get('lag_ms_take'), '.0f'):>8} "
                f"{_num(r.get('hysteresis_median'), '+.3f'):>6}  "
                + " ".join(f"{f} {_num(v)}" for f, v in others.items()))
        lines.append("")
        lines.append(text.rstrip("\n"))
        lines.append("")
    return "\n".join(lines) + "\n"


def _over_cue(row: Dict[str, str], move: str) -> str:
    """'0.17/4': a take's median bend (or straighten) time over the seconds
    its cue gave; '-' when the take has no such time."""
    median = row.get(f"{move}_time_median_s")
    cue = row.get(f"{move}_cue_s")
    if median in (None, "") or cue in (None, ""):
        return "-"
    return f"{_num(median)}/{float(cue):g}"


def _num(v, spec: str = ".2f") -> str:
    if v in (None, ""):
        return "-"
    try:
        return pc.fmt(float(v), spec)
    except ValueError:
        return str(v)


SEQUENCE_COLUMNS = (["session", "take_name", "item", "take", "take_verdict",
                     "bands", "step", "label", "flexed"]
                    + [f"glove_{f}" for f in pc.FINGERS]
                    + ["glove_pass", "glove_wrong", "glove_coupling"]
                    + [f"camera_{f}" for f in pc.FINGERS]
                    + ["camera_pass", "camera_wrong", "camera_coupling"])


def sequence_rows(parts: List[Tuple[pc.Session, list, str, str]]
                  ) -> List[dict]:
    out = []
    for session, rows, _text, _source in parts:
        for r in rows:
            if r.get("accepted") != "true":
                continue
            for s in _j(r, "steps") or []:
                g, c = s.get("glove") or {}, s.get("camera") or {}
                cam_pass = s.get("camera_pass")
                out.append({
                    "session": session.name, "take_name": r["name"],
                    "item": r["item"], "take": r["take"],
                    "take_verdict": r["verdict"], "bands": r.get("bands"),
                    "step": s.get("step"), "label": s.get("label"),
                    "flexed": " ".join(s.get("flexed") or []),
                    **{f"glove_{f}": g.get(f) for f in pc.FINGERS},
                    "glove_pass": "pass" if s.get("glove_pass") else "fail",
                    "glove_wrong": "; ".join(s.get("glove_wrong") or []),
                    **{f"camera_{f}": c.get(f) for f in pc.FINGERS},
                    "camera_pass": ("" if cam_pass is None else
                                    "pass" if cam_pass else "fail"),
                    "camera_wrong": "; ".join(s.get("camera_wrong") or []),
                    "glove_coupling": _coupling_text(s, pc.GLOVE),
                    "camera_coupling": _coupling_text(s, pc.CAMERA)})
    return out


def _coupling_text(step: dict, sensor: str) -> str:
    """'middle 0.45; pinky 0.32': straight fingers between the bands."""
    cpl = (step.get("coupling") or {}).get(sensor) or {}
    return "; ".join(f"{f} {pc.fmt(v)}" for f, v in cpl.items())


def cue_angle_rows(sessions: Sequence[pc.Session], kind: str,
                   tables: AngleTables) -> List[dict]:
    """`cue_angles.csv` of one hand: every held cue of every handed-in take
    (`angle_series.cue_rows`), session after session."""
    out = []
    for session in sessions:
        hold, window = pc.check_window_s(session)
        for t in handed_in(session):
            for r in asr.cue_rows(tables.rows(t, session.hand),
                                  tables.events(t), kind, hold, window):
                out.append({"session": session.name, "take_name": t.name,
                            "item": t.item, "take": t.take, **r})
    return out


def angle_spans(sessions: Sequence[pc.Session], tables: AngleTables
                ) -> Dict[str, Tuple[float, float, int]]:
    """Each of the 24 angles: (lowest, highest, frames) over every frame of
    the handed-in takes of these sessions; {} when they hold no frame."""
    low: Dict[str, float] = {}
    high: Dict[str, float] = {}
    n = 0
    for session in sessions:
        for t in handed_in(session):
            for r in tables.rows(t, session.hand):
                n += 1
                for a in asr.ANGLES:
                    v = r[a]
                    if a not in low or v < low[a]:
                        low[a] = v
                    if a not in high or v > high[a]:
                        high[a] = v
    return {a: (low[a], high[a], n) for a in asr.ANGLES if a in low}


def digits_one_value(sessions: Sequence[pc.Session], tables: AngleTables
                     ) -> Dict[str, Optional[Tuple[float, str, float, int]]]:
    """`angle_series.one_value` for each digit, over every frame of the
    handed-in takes of these sessions."""
    rows = [r for session in sessions for t in handed_in(session)
            for r in tables.rows(t, session.hand)]
    return {d: asr.one_value(rows, d) for d in asr.DIGIT_FLEXIONS}


def plan_glove_set(sessions: List[pc.Session], kind: str, exporter
                   ) -> Tuple[List[Output], dict]:
    plan: List[Output] = []
    tables = AngleTables()
    info = {"takes": 0, "missing": [], "hands": {}, "joint_frames": 0,
            "joint_frames_pdf": 0, "angle_spans": {}, "one_value": {}}
    by_hand: Dict[str, List[Tuple[pc.Session, list, str, str]]] = {}
    per_hand: Dict[str, int] = {}
    for session in sessions:
        per_hand[session.hand] = per_hand.get(session.hand, 0) + 1
    for session in sessions:
        # two sessions of one hand share a folder: their PDFs keep apart
        pdf_name = (JOINT_FRAMES_PDF if per_hand[session.hand] == 1 else
                    f"joint_frames_{session.name}.pdf")
        takes, missing = plan_glove_takes(session, kind, exporter, pdf_name,
                                          tables)
        csvs, pdfs = joint_frame_counts(takes)
        info["joint_frames"] += csvs
        info["joint_frames_pdf"] += pdfs
        plan.extend(takes)
        info["missing"].extend(missing)
        n = sum(1 for o in takes if o.dest.suffix == ".txt")
        info["takes"] += n
        info["hands"][session.hand] = info["hands"].get(session.hand, 0) + n
        rows, text, source = check_for(session)
        by_hand.setdefault(session.hand, []).append(
            (session, rows, text, source))
    for hand, parts in sorted(by_hand.items()):
        folder = Path(kind) / hand
        of_hand = [part[0] for part in parts]
        info["angle_spans"][hand] = angle_spans(of_hand, tables)
        info["one_value"][hand] = digits_one_value(of_hand, tables)
        plan.append(Output(
            folder / CUE_ANGLES_FILE,
            write=lambda dest, of_hand=of_hand: asr.write_csv(
                dest, asr.CUE_ROW_COLUMNS,
                cue_angle_rows(of_hand, kind, tables))))
        if kind == pc.FLEXION:
            plan.append(Output(
                folder / "flexion_report.txt",
                write=lambda dest, hand=hand, parts=parts: dest.write_text(
                    flexion_report(hand, parts), encoding="utf-8")))
        else:
            def write_csv(dest: Path, parts=parts) -> None:
                with open(dest, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=SEQUENCE_COLUMNS)
                    w.writeheader()
                    for r in sequence_rows(parts):
                        w.writerow({k: pc.csv_value(v) for k, v in r.items()})
            plan.append(Output(folder / "sequence_check.csv", write=write_csv))
    return plan, info


# --- README ----------------------------------------------------------------------------

def _endpoints_lines(session: pc.Session) -> List[str]:
    """README lines: each warm-up file's glove open, fist and (when it has
    them) single-finger values, in the units its takes were measured in.

    The open palm and the fist are read as measured (`cued=False`): the
    checker's default puts a cued finger's single bend at the end of its
    range, which printed under "fist" would misname it.
    """
    names = session.warmup_names or [pc.WARMUP_FILE]
    out = []
    for n in names:
        w = session.warmup_named(n)
        where = "" if len(names) == 1 else f" {n}"
        ends = pc.endpoints(w, pc.GLOVE, cued=False)
        if not ends:
            out.append(f"    warm-up{where}: none recorded")
            continue
        units = ("glove (thumb in degrees, the rest curl)"
                 if pc.thumb_in_degrees(w) else "glove curl")
        text = (f"    warm-up{where} {units}, thumb index middle ring "
                "pinky: open "
                + " ".join(pc.fmt(ends[f][0]) if f in ends else "-"
                           for f in pc.FINGERS)
                + ", fist " + " ".join(pc.fmt(ends[f][1]) if f in ends
                                       else "-" for f in pc.FINGERS))
        single = (w.get(pc.GLOVE) or {}).get("single")
        if isinstance(single, dict):
            text += ", single " + " ".join(pc.fmt(single.get(f))
                                           for f in pc.FINGERS)
        out.append(text)
    return out


def _item_counts(session: pc.Session) -> str:
    counts: Dict[str, int] = {}
    for t in accepted(session):
        counts[t.item] = counts.get(t.item, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"


def wrapped(text: str, first: str = "  ", rest: str = "    ") -> List[str]:
    return textwrap.wrap(text, width=96, initial_indent=first,
                         subsequent_indent=rest, break_on_hyphens=False)


GLOVE_SETS = (pc.FLEXION, pc.SEQUENCES)


def protocol_of(sessions: Sequence[pc.Session]) -> dict:
    """The protocol the set's sessions were recorded under (the first one
    that could be read), or {}."""
    return next((s.protocol for s in sessions if s.protocol), None) or {}


def _and(words: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    words = list(words)
    return (", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1
            else "".join(words))


def not_recorded(sessions: Sequence[pc.Session], proto: dict
                 ) -> List[Tuple[str, List[str]]]:
    """The protocol's items, in its order, that some hand of the set has no
    accepted take of, each with those hands: what the folder lacks."""
    hands = sorted({s.hand for s in sessions})
    got = {(t.item, s.hand) for s in sessions for t in accepted(s)}
    out = []
    for it in proto.get("items", []):
        lacking = [h for h in hands if (it["id"], h) not in got]
        if lacking:
            out.append((it["id"], lacking))
    return out


def not_recorded_lines(sessions: Sequence[pc.Session], proto: dict,
                       reason: Optional[str] = None) -> List[str]:
    """SETUP: the items of the set no take of which is in the folder, by
    hand, and why, when the build was told (`--not-recorded-reason`)."""
    missing = not_recorded(sessions, proto)
    if not missing:
        return []
    hands = sorted({s.hand for s in sessions})
    groups: Dict[Tuple[str, ...], List[str]] = {}
    for item, lacking in missing:
        groups.setdefault(tuple(lacking), []).append(item)
    parts = [("Not recorded with either hand: " if len(hands) > 1
              and list(lacking) == hands else
              f"Not recorded with the {_and(lacking)} hand: ") + _and(items)
             + "." for lacking, items in groups.items()]
    text = (" ".join(parts) + " How each session's attempts at "
            + ("it" if len(missing) == 1 else "them")
            + " ended is under SESSIONS.")
    if reason:
        text += " Why: " + reason.strip().rstrip(".") + "."
    return wrapped(text, "  ", "    ")


def _how_ended(take: pc.TakeRef) -> str:
    """How a rejected attempt ended, in a few words."""
    reason = take.reason or ""
    if reason.startswith("interrupted by the operator"):
        return "stopped by the operator during the take"
    if reason.startswith("skipped by the operator"):
        return "skipped by the operator during the take"
    return "rejected"


def session_not_recorded_lines(session: pc.Session) -> List[str]:
    """SESSIONS: the protocol's items this session has no accepted take of,
    with what became of its attempts at each, and how the session ended."""
    proto = session.protocol or {}
    have = {t.item for t in accepted(session)}
    lacking = [it["id"] for it in proto.get("items", [])
               if it["id"] not in have]
    if not lacking:
        return []
    skipped = {x.get("item") for x in (session.meta.get("skipped") or [])
               if isinstance(x, dict)}
    parts = []
    for item in lacking:
        tries = [t for t in session.takes if t.item == item]
        if not tries:
            parts.append(f"{item} (" + ("skipped by the operator"
                                        if item in skipped else "not tried")
                         + ")")
            continue
        ends: Dict[str, int] = {}
        for t in tries:
            ends[_how_ended(t)] = ends.get(_how_ended(t), 0) + 1
        told = f"{len(tries)} attempt" + ("s" if len(tries) > 1 else "")
        if len(ends) == 1:
            told += ", " + ("" if len(tries) == 1 else "both "
                            if len(tries) == 2 else "all ") + next(iter(ends))
        else:
            told += ": " + ", ".join(f"{n} {what}" for what, n in ends.items())
        parts.append(f"{item} ({told})")
    text = "not recorded: " + ", ".join(parts)
    if session.meta.get("stopped"):
        text += f"; the session ended: {session.meta['stopped']}"
    return wrapped(text, "    ", "      ")


def flexion_item_lines(sessions: Sequence[pc.Session], proto: dict
                       ) -> List[str]:
    """Set B's items as the protocol has them, one line each, with the takes
    handed in per hand, or "not recorded"."""
    hands = sorted({s.hand for s in sessions})
    count: Dict[Tuple[str, str], int] = {}
    for s in sessions:
        for t in accepted(s):
            count[(t.item, s.hand)] = count.get((t.item, s.hand), 0) + 1
    items = [it for it in proto.get("items", [])
             if it.get("bend_s") is not None]
    if not items:
        return []
    each = proto.get("takes_per_item")
    lines = wrapped(
        "Items, as the protocol has them (seconds of each phase of a cycle, "
        "every phase cued by a beep), and the takes handed in"
        + (f" (the protocol asks for {each} of each per hand)" if each else "")
        + ":", "  ", "  ")
    width = max(len("item"), *(len(it["id"]) for it in items)) + 2
    lines.append(f"    {'item':<{width}}{'cycles':>6}{'bend':>7}{'hold':>7}"
                 f"{'straighten':>12}{'rest':>7}   handed in")
    for it in items:
        got = [f"{h} {count.get((it['id'], h), 0)}" for h in hands]
        done = any(count.get((it["id"], h)) for h in hands)
        lines.append(
            f"    {it['id']:<{width}}{it.get('cycles', '-'):>6}"
            f"{it['bend_s']:>7g}{it.get('hold_s', 0):>7g}"
            f"{it.get('straighten_s', 0):>12g}{it.get('rest_s', 0):>7g}   "
            + (", ".join(got) if done else "not recorded"))
    return lines


def setup_lines(kind: str, sessions: List[pc.Session],
                info: Optional[dict] = None,
                not_recorded_reason: Optional[str] = None) -> List[str]:
    """SETUP lines of one glove set (Set A's are the grasp README's)."""
    hands = ", ".join(sorted({s.hand for s in sessions})) or "-"
    proto = protocol_of(sessions)
    cam, glove_only = camera_split(sessions)
    only_note = (wrapped("Glove only, no camera: session "
                         + ", ".join(s.name for s in glove_only) + ".",
                         "  ", "    ") if cam and glove_only else [])
    missing = not_recorded_lines(sessions, proto, not_recorded_reason)
    if kind == pc.FLEXION:
        out = [
            f"  Hardware: gloves on + camera ({hands} hand, one hand at a "
            "time, the other resting on the",
            "  table). StretchSense gloves streamed through XR Trainer at "
            "their full rate, calibrated in",
            "  XR Trainer at the start of the session; the Ultraleap Stereo "
            "IR 170 ran as the reference,",
            "  palm to the lens 18 to 28 cm above the module. Every phase of "
            "every cycle was cued by a",
            "  beep and is timestamped in the .events.jsonl file.",
        ] if cam else wrapped(
            f"Hardware: gloves on, no camera ({hands} hand, one hand at a "
            "time, the other resting on the table). StretchSense gloves "
            "streamed through XR Trainer at their full rate, calibrated in "
            "XR Trainer at the start of the session. No camera ran: the "
            "glove is the only sensor of these recordings. Every phase of "
            "every cycle was cued by a beep and is timestamped in the "
            ".events.jsonl file.", "  ", "  ")
        out += only_note
        out += flexion_item_lines(sessions, proto)
        return out + missing
    hold = proto.get("hold_s", 2.5)
    window = proto.get("check_window_s", 1.5)
    return [
        f"  Hardware: gloves on{' + camera' if cam else ', no camera'} "
        f"({hands} hand, one hand at a time), the same setup as Set B.",
    ] + only_note + [
        f"  Seven sequences; each step cued by a beep and held {hold:g} s "
        "(the first second is the",
        f"  movement; the check reads the last {window:g} s). Three takes "
        "per sequence, recorded as three",
        "  rounds in a shuffled order (the seed is in the session's own "
        "metadata).",
    ] + missing


def joint_frame_contents(kind: str, info: dict) -> List[str]:
    """CONTENTS lines for a glove set's joint_frames folder, when it has one."""
    if not (info.get("joint_frames") or info.get("joint_frames_pdf")):
        return []
    where = "<hand>\\"
    return [f"      {where}joint_frames\\<take>.csv   the take's summary "
            "frame: every joint's position (mm),",
            "          orientation, axes and angles in the wrist frame "
            "(JOINT FRAMES below)",
            f"      {where}joint_frames\\camera_<take>.csv   the same "
            "from the camera file of the take",
            f"      {where}joint_frames\\joint_frames.pdf   one page per "
            "summary frame: the hand drawn with every",
            "          joint's x y z axes, the paper's 24 angles and the "
            "26-row table"]


JOINT_FRAMES_SECTION = (
    ["JOINT FRAMES AND THE PAPER'S 24 ANGLES", "-" * 78]
    + wrapped(
        "Each joint_frames CSV holds one frame of a take, its summary frame "
        "(Set A: the medoid of the static interval, as in _keypoints.txt; "
        "Sets B and C: the medoid of the whole take), one row per OpenXR "
        "joint, 26 rows. x_mm y_mm z_mm is the joint's position in the "
        "wrist frame, in millimetres: the origin at the wrist joint and the "
        "axes those of the wrist joint's own orientation. For the camera "
        "that orientation is the tracker's forearm bone, so a bent wrist "
        "turns the whole hand in this frame; the PDF prints the "
        "forearm-to-palm angle on every camera page. qx qy qz qw and the "
        "axis_x_*, axis_y_*, axis_z_* columns are the joint's orientation "
        "and its three unit axes in the same frame: z runs along the bone "
        "back toward the wrist, y out of the back of the hand, x across it "
        "(x = y cross z). bone_len_mm is the distance to the parent joint.",
        "  ", "  ")
    + [""]
    + wrapped(
        "flex_deg, abd_deg and twist_deg are the rotation from the parent "
        "joint's frame to this joint's, taken about x, then the new y, then "
        "the new z. Flexion is positive toward the palm, abduction positive "
        "toward the thumb, twist positive when the back of the bone turns "
        "toward the little finger; the signs are the same for both hands "
        "and both sensors.", "  ", "  ")
    + [""]
    + wrapped(
        "The 24 angles of Cobos et al. 2009 (Table 1) are cells of this "
        "table. Fingers I, M, R, L (index, middle, ring, little): CMC = "
        "flex_deg of the METACARPAL row, MCP_fe and MCP_aa = flex_deg and "
        "abd_deg of the PROXIMAL row, PIP = flex_deg of INTERMEDIATE, DIP = "
        "flex_deg of DISTAL. Thumb T: TMC_fe and TMC_aa = flex_deg and "
        "abd_deg of THUMB_METACARPAL (they include the thumb's resting "
        "position, so they are not zero with the hand open), MCP_fe = "
        "flex_deg of THUMB_PROXIMAL, IP = flex_deg of THUMB_DISTAL. "
        "joint_frames.pdf lists the 24 for every summary frame. The glove "
        "senses flexion only: its CMC, abduction and TMC values come from "
        "its own hand model; the camera measures them.", "  ", "  ")
    + [""])


def speed_rule_lines(packaged: Dict[str, List[pc.Session]]) -> List[str]:
    """QUALITY RULES lines on how fast the Set B bends were, when the
    folder has Set B: what is timed, what the rule is and which sessions
    were recorded under it (the rule's numbers are the sessions' own)."""
    sessions = packaged.get(pc.FLEXION) or []
    if not sessions:
        return []
    ruled = [s for s in sessions if pc.speed_rule_of(s.meta)]
    marks = pc.rule_marks(pc.speed_rule_of(ruled[0].meta) if ruled else None)
    text = (
        "Set B, bend speed: the protocol asks for a slow bend and a slow "
        "straighten. Every bend the cycle count finds is timed from "
        f"{marks[0] * 100:.0f} % to {marks[1] * 100:.0f} % of its own rise "
        "(from the lowest reading since the finger was last open to the "
        "highest before it opens again), and every straighten the same way "
        "down; flexion_report.txt has each take's median over the seconds "
        "its cue gave, and every bend's own time. ")
    if ruled:
        rule = pc.speed_rule_of(ruled[0].meta)
        text += (
            "Under the bend-speed rule a bend the cue gives "
            f"{float(rule['slow_from_s']):g} s or more for must take at "
            f"least {float(rule['min_bend_share']) * 100:.0f} % of the "
            "cue's time, or the take was recorded again; the fast index "
            "item is not judged. The bar is an acquisition gate set between "
            "what a snapped bend and a paced slow bend read on this glove, "
            "not the definition of a slow bend. ")
        others = [s.name for s in sessions if s not in ruled]
        text += ("Every Set B session of this folder was recorded under it. "
                 if not others else
                 "Recorded under it: " + ", ".join(s.name for s in ruled)
                 + "; not under it: " + ", ".join(others)
                 + " (their times are reported as they are, not judged). ")
    else:
        text += ("No Set B session of this folder was recorded under the "
                 "bend-speed rule (it was added on " + pc.SPEED_RULE_SINCE
                 + "): the times are reported as they are, and a cue that "
                 "gave 4 s for a bend does not mean the bend took 4 s. ")
    text += ("The straighten is timed and reported, never judged: in the "
             "one earlier recording of a finger opened slowly on purpose "
             "(a paced sweep of one finger, 2026-09-20), the glove's "
             "reading fell no slower than in snapped straightens, so the "
             "straighten's time is not taken as a measure of the "
             "movement.")
    return wrapped(text, "  ", "    ")


ANGLE_SECTION_TITLE = "JOINT ANGLES, FRAME BY FRAME AND AT EACH HELD CUE"
MODEL_MARK = "hand model, not a measurement"


def angle_section(packaged: Dict[str, List[pc.Session]],
                  infos: Dict[str, dict]) -> List[str]:
    """README lines for joint_angles\\<take>.csv and cue_angles.csv: the
    columns, which cue a frame belongs to, the windows, the angles'
    definition, the nine the glove does not measure, and the span each
    angle took in this folder (from the same rows the files are written
    from)."""
    L = [ANGLE_SECTION_TITLE, "-" * 78]
    L += wrapped(
        f"<hand>\\{ANGLES_DIR}\\<take>.csv has one row per glove frame of the "
        "recorded hand: frame_id (copied from the glove line; it is 0 in the "
        "glove lines XR Trainer sends, so it does not number the frames), "
        "wall_time (when the recorder wrote the frame, seconds since 1970 on "
        "the recording computer, the clock of the .jsonl and .events.jsonl "
        "files; the .jsonl line also has capture_time, when the packet "
        "arrived, which can be earlier), t_s (seconds since the take_start "
        "event), the cue in force at that instant (cue_step, cue_label, for "
        "Set B cue_phase and cue_cycle, and flexed = the fingers the cue "
        "flexes, empty = none), then the 24 angles in degrees. A frame's "
        "cue is the last cue whose beep was at or before the frame's time; "
        "a frame before the first cue or after the end of the take has "
        "empty cue columns. The glove's delay is not removed, so the first "
        "frames of a cue still show the pose before it.", "  ", "  ")
    L.append("")
    L += wrapped(
        "The 24 angles are those of Cobos et al. 2009 (Table 1), read from "
        "the glove's 26 joints. Each is the rotation from the parent "
        "joint's frame to the joint's own, taken about x (flexion, positive "
        "toward the palm), then the new y (abduction, positive toward the "
        "thumb), then the new z; the signs are the same for both hands. "
        "Thumb: T_TMC_fe and T_TMC_aa = flexion and abduction of "
        "THUMB_METACARPAL (they include the thumb's resting position, so "
        "they are not zero with the hand open), T_MCP_fe = flexion of "
        "THUMB_PROXIMAL, T_IP = flexion of THUMB_DISTAL. Fingers I, M, R, L "
        "(index, middle, ring, little): CMC = flexion of the METACARPAL "
        "joint, MCP_fe and MCP_aa = flexion and abduction of PROXIMAL, PIP "
        "= flexion of INTERMEDIATE, DIP = flexion of DISTAL. The joints and "
        "their rotations are the ones in the take's .jsonl file.",
        "  ", "  ")
    L.append("")
    L += wrapped(
        "What the glove measures: it streams finger flexion only. Fifteen "
        "of the 24 are flexions the stream carries (T_TMC_fe, T_MCP_fe, "
        "T_IP, and MCP_fe, PIP and DIP of each finger). The other nine are "
        "values of the glove's hand model, not measurements: "
        + ", ".join(asr.MODEL_ONLY) + ". They are written as streamed, and "
        "a joint-limit check on them checks the model, not the hand. The "
        "glove has no sensor for thumb opposition, so T_TMC_fe rises when "
        "the thumb crosses the palm.", "  ", "  ")
    L.append("")
    hold, window = pc.DEFAULT_HOLD_S, pc.DEFAULT_CHECK_WINDOW_S
    for s in packaged.get(pc.SEQUENCES, []):
        hold, window = pc.check_window_s(s)
        break
    L += wrapped(
        f"<hand>\\{CUE_ANGLES_FILE} has one row per held cue of every take: "
        "session, take_name, item, take, step, label, phase and cycle (Set "
        "B), flexed, cue_t_s (the beep, seconds since take_start), "
        "window_start_s and window_end_s, frames (the glove frames inside "
        "the window, start included, end excluded) and the median of each "
        "of the 24 angles over those frames. Set B: one row for the hold "
        "and one for the rest of every cycle, over the whole phase; the "
        "bend and the straighten are movements and have no row. Set C: one "
        f"row per step, over the last {window:g} s of its {hold:g} s hold "
        "(the same share of the hold when a run's timing was scaled), the "
        "window sequence_check.csv is read over. A finger the cue does not "
        "name is in the row too: its angles at a hold against its angles "
        "at the rest, or at the open hand, are how far it went along with "
        "the cued finger.", "  ", "  ")
    L.append("")
    L.append("  Span of each angle over every frame of the handed-in takes, "
             "degrees:")
    for kind in GLOVE_SETS:
        for hand, spans in sorted(
                (infos.get(kind, {}).get("angle_spans") or {}).items()):
            if not spans:
                L.append(f"    {kind}, {hand} hand: no frame")
                continue
            n = next(iter(spans.values()))[2]
            L.append(f"    {kind}, {hand} hand, {n} frames")
            for a in asr.ANGLES:
                lo, hi, _n = spans[a]
                L.append(f"      {a:<9}{lo:9.2f} to {hi:8.2f}"
                         + (f"   {MODEL_MARK}" if a in asr.MODEL_ONLY
                            else ""))
    L.append("")
    L += wrapped(
        "One value per digit. For each digit the frames of the handed-in "
        "takes are grouped by the digit's summed flexion (thumb: T_TMC_fe + "
        "T_MCP_fe + T_IP; a finger: MCP_fe + PIP + DIP) in steps of "
        f"{asr.ONE_VALUE_STEP:g} degree, and inside each group of "
        f"{asr.ONE_VALUE_MIN_GROUP} frames or more every joint's flexion is "
        "compared with the group's mean. The line gives how far the sum "
        "moved and the RMS of those differences for the joint where it is "
        "largest. Where that is a fraction of a degree, the digit's three "
        "flexions are one measurement spread over three joints by the "
        "glove's hand model: how the joints of that digit bend relative to "
        "each other is then the model's, and a dependency between them "
        "cannot be learned from these files. Different digits are not tied "
        "this way: how far one finger goes along with another is measured. "
        "(The glove's raw sensor values are not exposed.) A digit whose sum "
        f"moved less than {asr.ONE_VALUE_MIN_SPAN:g} degrees is not judged.",
        "  ", "  ")
    for kind in GLOVE_SETS:
        for hand, digits in sorted(
                (infos.get(kind, {}).get("one_value") or {}).items()):
            L.append(f"    {kind}, {hand} hand")
            for digit in asr.DIGIT_FLEXIONS:
                got = digits.get(digit)
                L.append(f"      {digit:<7}" + (
                    "not judged" if got is None else
                    f"sum moved {got[2]:6.1f} deg; joints within "
                    f"{got[0]:5.2f} deg RMS of one value (largest: "
                    f"{got[1]}, {got[3]} frames)"))
    L.append("")
    return L


def glove_readme_text(packaged: Dict[str, List[pc.Session]],
                      infos: Dict[str, dict], operator: str, built: str,
                      standalone: bool = True,
                      not_recorded_reason: Optional[str] = None) -> str:
    """The README of Sets B and C, the glove recordings.

    `standalone` is the whole README of a folder without Set A; otherwise
    the text is the second part of the folder's README, after the grasp
    set's own twenty sections (`leap_hand.grasp_handin.readme_text`).
    `not_recorded_reason`: why the items the folder has no take of were not
    recorded (`--not-recorded-reason`), printed after them in SETUP.
    """
    L: List[str] = []
    sessions = [s for k in GLOVE_SETS for s in packaged.get(k, [])]
    dates = sorted({str(s.meta.get("started", ""))[:10]
                    for s in sessions if s.meta.get("started")})
    mock = {s.name: pc.mock_flags(s.meta) for s in sessions
            if pc.mock_flags(s.meta)}
    if standalone:
        L += ["Grasp and finger-flexion recordings", "=" * 78]
    else:
        L += ["=" * 78,
              "SETS B AND C: THE GLOVE RECORDINGS (single-finger flexion, "
              "sequential movements)", "=" * 78]
    if mock:
        L += wrapped("MOCK DATA: this folder was built with --allow-mock and "
                     "holds synthetic sessions, not recordings of a hand: "
                     + ", ".join(sorted(mock)) + ". It is not a hand-in.",
                     "", "")
    if standalone:
        L += [f"Operator: {operator or 'not recorded'}",
              f"Recorded: {', '.join(dates) or 'not recorded'}",
              f"Folder built: {built} by scripts/package_professor_set.py "
              "from the session folders below.",
              "No photographs are included.", ""]
    else:
        L += [f"Recorded: {', '.join(dates) or 'not recorded'}", ""]

    cam_of = {k: any(ran_camera(x) for x in packaged.get(k, []))
              for k in GLOVE_SETS}
    any_cam = any(cam_of.values())
    # The rules of both sets are printed whichever is in the folder. A set
    # that is not in it says about the camera what the folder's own
    # sessions did, so a glove-only folder never describes a camera check.
    rule_cam = {k: cam_of[k] if packaged.get(k) else any_cam
                for k in GLOVE_SETS}
    L += ["CONTENTS", "-" * 78]
    if standalone:
        L.append(f"  {pc.GRASPS}\\  {SET_TITLES[pc.GRASPS]}: not included in "
                 "this folder")
    for kind in GLOVE_SETS:
        n = infos.get(kind, {}).get("takes", 0)
        if not packaged.get(kind):
            L.append(f"  {kind}\\  {SET_TITLES[kind]}: not included in this "
                     "folder")
            continue
        L.append(f"  {kind}\\  {SET_TITLES[kind]}: {n} take(s)")
        with_cam = cam_of[kind]
        L += [f"      <hand>\\<take>.jsonl             glove, full rate,"
              " 26 joints"]
        if with_cam:
            L += [f"      <hand>\\camera_<take>.jsonl      the camera "
                  "reference for the same take (when it ran)"]
        L += [f"      <hand>\\<take>.events.jsonl      every cue and "
              "decision, same clock as the frames",
              f"      <hand>\\<take>.txt               glove, every "
              "frame, 21 landmarks, wrist at the origin, mm",
              f"      <hand>\\{ANGLES_DIR}\\<take>.csv  glove, every "
              "frame: its cue and the 24 joint angles, degrees",
              f"      <hand>\\{CUE_ANGLES_FILE}           the 24 angles "
              "at every held cue of every take (medians)",
              f"      <hand>\\<session>_warmup*.json   each session's "
              "warm-up(s): the open, fist and single-finger ends"]
        if kind == pc.FLEXION:
            L.append("      <hand>\\flexion_report.txt       cycles and "
                     "range" + (", hysteresis, lag" if with_cam else "")
                     + ", per finger and speed")
        else:
            L.append("      <hand>\\sequence_check.csv       step by step "
                     "fractions and pass/fail, "
                     + ("glove and camera" if with_cam else
                        "glove (the camera columns are empty: no camera "
                        "ran)")
                     + "; flexed = the fingers the step flexes, empty = "
                     "open hand")
        L += joint_frame_contents(kind, infos.get(kind, {}))
    L.append("")

    L += ["SETUP", "-" * 78]
    for kind in GLOVE_SETS:
        if packaged.get(kind):
            L.append(f"{SET_TITLES[kind]}:")
            L += setup_lines(kind, packaged[kind], infos.get(kind),
                             not_recorded_reason)
            L.append("")
    missing_any = any(not_recorded(packaged[k], protocol_of(packaged[k]))
                      for k in GLOVE_SETS if packaged.get(k))

    L += ["SESSIONS", "-" * 78]
    for kind in GLOVE_SETS:
        for s in packaged.get(kind, []):
            n_rej = sum(1 for t in s.takes if not t.accepted)
            L.append(f"  {kind}, {s.hand} hand, session {s.name}")
            L.append(f"    recorded {s.meta.get('started', '?')} to "
                     f"{s.meta.get('ended', '?')}, operator "
                     f"{s.meta.get('operator') or '-'}")
            L.append(f"    protocol {s.meta.get('protocol_name', kind)} "
                     f"version {s.meta.get('protocol_version', '?')}")
            L.append("    sensors: " + ("glove and Ultraleap camera"
                                           if ran_camera(s) else
                                           "glove only, no camera"))
            if kind == pc.FLEXION:
                L += wrapped(pc.speed_rule_text(pc.speed_rule_of(s.meta))
                             + ".", "    ", "      ")
            L.append("    XR Trainer calibrated at "
                     f"{s.meta.get('xr_trainer_calibrated_at') or '-'}")
            L += _endpoints_lines(s)
            if s.name in mock:
                L.append("    MOCK session (synthetic data): "
                         + ", ".join(mock[s.name]))
            L.append(f"    handed in: {len(accepted(s))} take(s) "
                     f"({_item_counts(s)}); {n_rej} rejected attempt(s) kept "
                     "back with their reasons")
            L += session_not_recorded_lines(s)
    for kind in GLOVE_SETS:
        for m in infos.get(kind, {}).get("missing", []):
            L.append(f"  not handed in: {m}")
    L.append("")

    L += ["QUALITY RULES", "-" * 78,
          "Fixed numbers are acquisition gates that decide on the spot "
          "whether a take is usable; the",
          "continuous values are always kept in the reports, and no fixed "
          "number is the definition of",
          "a correct movement.",
          *wrapped("Set B: the cued finger's glove reading spans at least "
                   "60 % of its own range from the session warm-up (open "
                   "palm to the finger bent on its own, for sessions recorded "
                   "since 2026-10-01; open palm to fist before); the other "
                   "four fingers' spans are reported, not failed (a "
                   "finger's neighbours can bend with it). Cycles are "
                   "counted from the cues and from the glove curl peaks and "
                   "must agree with each other and with the protocol.",
                   "  ", "    ")] + ([
              "    Where the camera followed the finger (camera span at "
              "least 0.50), the transfer curve,",
              "    hysteresis and lag are reported per cycle and per take."]
              if rule_cam[pc.FLEXION] else [
              "    No camera ran, so no transfer curve, hysteresis or lag "
              "is reported."]) + [
          ] + speed_rule_lines(packaged) + [
          "  Set C: for each step, the median glove fraction over the last "
          "part of the hold must be above",
          "    0.6 for the fingers the step flexes; a finger the step leaves "
          "straight fails only when it",
          "    reads flexed (0.6 or more), and between 0.3 and 0.6 it is "
          "reported as coupling (a finger's",
          "    neighbours can bend with it), not failed. A take passes when "
          "every step does."] + ([
              "    The same check on the camera where it is trusted is "
              "reported beside it, so a glove fault",
              "    reads as the glove being wrong, not the operator. "
              "sequence_check.csv has every fraction,",
              "    the pass/fail and the coupling per step, glove and "
              "camera."]
              if rule_cam[pc.SEQUENCES] else [
              "    No camera ran, so a glove fault cannot be told from the "
              "operator's movement by a second",
              "    sensor. sequence_check.csv has every fraction, the "
              "pass/fail and the coupling per step."]) + [
          *wrapped("Failed takes were redone in the same session"
                   + (", except for the items SETUP lists as not recorded"
                      if missing_any else "")
                   + "; the recordings of the failed attempts are not in "
                   "this folder.", "  ", "    "),
          "  fraction = (open - curl) / (open - fist), with open and fist "
          "the finger's own medians from",
          "    the session warm-up (3 s open palm, 3 s fist); 0 = open palm, "
          "1 = fist. curl = fingertip-",
          "    to-wrist distance over palm length.",
          "  Since 2026-10-01 the warm-up also has each finger bent on its "
          "own for 3 s (single), and a",
          "    finger the cue names is measured against that (0 = open "
          "palm, 1 = its own bend) when it",
          "    bent far enough (index to little finger at least half of the "
          "fist span, thumb at least 35",
          "    degrees). The glove thumb is then its bend in degrees "
          "(TMC_fe + MCP_fe + IP, larger = more",
          "    bent) instead of the curl, which barely moves when the thumb "
          "bends; the warm-up file's",
          "    units say which. A session resumed after a break has one "
          "warm-up per run.",
          ""]

    if any(infos.get(k, {}).get("joint_frames") or
           infos.get(k, {}).get("joint_frames_pdf") for k in GLOVE_SETS):
        L += JOINT_FRAMES_SECTION

    L += angle_section(packaged, infos)

    L += ["KNOWN GLOVE LIMITS (reported as such, not hidden)", "-" * 78]
    for x in (KNOWN_LIMITS if any_cam else KNOWN_LIMITS_GLOVE_ONLY):
        L += wrapped(x, "  - ", "    ")
    L.append("")

    # wall_time is when the recorder wrote a line; capture_time, where a
    # file has it, when the glove's packet arrived (QueueItem.recv_time) or
    # when the tracker captured a camera frame. A folder with no camera says
    # nothing about one.
    if any_cam:
        capture = ("capture_time (glove: when the packet reached the "
                   "recording computer; camera: when the tracker captured "
                   "the frame; the same clock)")
    else:
        capture = ("capture_time (when the packet reached the recording "
                   "computer, the same clock)")
    jsonl_line = (
        ".jsonl  one JSON object per frame: wall_time (when the recorder "
        "wrote the line, seconds since 1970 on the recording computer) and, "
        f"where the file has it, {capture}, then the 26 OpenXR joints (PALM, "
        "WRIST, then thumb metacarpal to tip and each finger metacarpal, "
        "proximal, intermediate, distal, tip). Each joint's x y z is a "
        "translation in its parent's frame in metres and qx qy qz qw its "
        "rotation relative to the parent; chaining them from the wrist gives "
        "the hand. Every line also carries session, item and take.")
    L += ["COORDINATE CONVENTIONS", "-" * 78,
          *wrapped(jsonl_line, "  ", "    "),
          "    Glove: the wrist is the root at the glove's own origin (the "
          "glove senses no position in",
          "    the room)."] + ([
              "    Camera: the wrist carries its position in the "
              "Ultraleap's desktop space (origin at the",
              "    module, +x along the camera baseline, +y up, +z toward "
              "the operator, metres), and each",
              "    line also holds abs26 (all 26 joint positions in that "
              "space) and palm_abs."] if any_cam else []) + [
          "  .txt  the 21-landmark format of the earlier hand-ins: per frame "
          "a line 'Frame <n> | Hand",
          "    ID: <left|right> | Time: <local time>', a 'Wrist:' line, then "
          "0 (Palm) to 20 (Pinky) in",
          "    millimetres: 0 palm centre; 1-4 thumb metacarpal, proximal, "
          "distal, tip; 5-8 index knuckle,",
          "    middle joint, end joint, tip; 9-12 middle; 13-16 ring; 17-20 "
          "little finger.",
          "    Glove text files: wrist at the origin, every point relative to"
          " the wrist, as in the earlier",
          "    hand-ins. In the glove files the Hand ID is the glove's own "
          "side, which is reliable.",
          "  .events.jsonl: t is the same clock as wall_time; kind is "
          "take_start, cue, take_end or",
          "    decision; a cue names the step, its label, the fingers that "
          "should be flexed and, for",
          "    Set B, the cycle and phase (bend, hold, straighten, rest).",
          ""]
    return "\n".join(L) + "\n"


def tool_version() -> str:
    """The repository commit the build ran from, and whether the working
    tree had uncommitted changes; words, for the README."""
    import subprocess

    def git(*args: str) -> Optional[str]:
        try:
            done = subprocess.run(["git", *args], cwd=str(REPO), timeout=20,
                                  capture_output=True, text=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return "the tools of this repository (commit not readable)"
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return (f"repository commit {commit}"
            + (" with uncommitted changes" if dirty else ""))


def readme_text(packaged: Dict[str, List[pc.Session]], infos: Dict[str, dict],
                operator: str, built: str, command: str = "",
                thresholds: Optional[dict] = None,
                not_recorded_reason: Optional[str] = None) -> str:
    """The folder's README: the grasp set's twenty sections when Set A is in
    the folder, then the glove sets' part when Sets B or C are."""
    grasp_info = infos.get(pc.GRASPS)
    gloves = any(packaged.get(k) for k in GLOVE_SETS)
    if grasp_info is None or grasp_info.get("set") is None:
        return glove_readme_text(packaged, infos, operator, built,
                                 not_recorded_reason=not_recorded_reason)
    gs = grasp_info["set"]
    # MOCK DATA at the top when the grasp session OR any transition session
    # of the build is mock (`grasp_handin.add_transition_session` names
    # those it was allowed to add)
    mock = (([gs.session.name] if pc.mock_flags(gs.session.meta) else [])
            + list(gs.mock_transition_sessions))
    text = gh.readme_text(
        gs, operator,
        f"{built} from the session folder {gs.session.name} (section 20 "
        "has the command).",
        command or "python scripts/rebuild_grasp_set.py",
        tool=tool_version(), mock=mock, thresholds=thresholds)
    if gloves:
        text += "\n" + glove_readme_text(
            packaged, infos, operator, built, standalone=False,
            not_recorded_reason=not_recorded_reason)
    return text


# --- main ----------------------------------------------------------------------------------

def build_plan(grasps: List[pc.Session], flexion: List[pc.Session],
               sequences: List[pc.Session], exporter,
               participant: str = "P01", thresholds: Optional[dict] = None,
               transitions: Sequence[Path] = (), allow_mock: bool = False
               ) -> Tuple[List[Output], Dict[str, dict]]:
    plan: List[Output] = []
    infos: Dict[str, dict] = {}
    if grasps:
        p, info = plan_grasps(grasps[0], exporter, participant, thresholds,
                              transitions, allow_mock)
        plan += p
        infos[pc.GRASPS] = info
    if flexion:
        p, info = plan_glove_set(flexion, pc.FLEXION, exporter)
        plan += p
        infos[pc.FLEXION] = info
    if sequences:
        p, info = plan_glove_set(sequences, pc.SEQUENCES, exporter)
        plan += p
        infos[pc.SEQUENCES] = info
    return plan, infos


# --- writing, without destroying what was there ------------------------------------------

BUILD_LINE = "Folder built:"


def without_build_line(text: str) -> str:
    """A README without the one line that says when it was built."""
    return "\n".join(x for x in text.splitlines()
                     if not x.startswith(BUILD_LINE))


def keep_unchanged_readme(out: Path, readme: str) -> str:
    """The README to write: the one already in the folder when the new one
    differs from it only in the time of the build. A rebuild that finds
    nothing new then changes no file at all, and the time in the README
    stays the time the folder was last different. (When the build does
    change another file, `main` writes the new README after all: the time
    in it is then this build's.)"""
    existing = os_path(out / "README.txt")
    try:
        old = existing.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return readme
    return old if without_build_line(old) == without_build_line(readme) \
        else readme


def same_bytes(a: Path, b: Path) -> bool:
    """Two files hold the same bytes (sizes first, then the content)."""
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                x, y = fa.read(1 << 20), fb.read(1 << 20)
                if x != y:
                    return False
                if not x:
                    return True
    except OSError:
        return False


def move_into_place(src: Path, dst: Path) -> None:
    """`os.replace`, retried briefly: OneDrive and a viewer can hold a file
    for a moment (the recorder's `write_json` waits the same way)."""
    for attempt in range(40):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def set_aside(path: Path, out: Path, superseded: Path) -> Path:
    """Move a file of the hand-in folder into `superseded`, keeping its
    place under it; returns where it went. Never deletes."""
    target = os_path(superseded / Path(plain(path)).relative_to(
        Path(plain(os_path(out)))))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        move_into_place(path, target)
    except PermissionError:
        raise
    except OSError:                      # another volume
        shutil.move(str(path), str(target))
    return target


def write_plan(plan: Sequence[Output], out: Path, superseded: Path
               ) -> Dict[str, List[str]]:
    """Write every planned file; say what happened to each.

    A file already there with the same bytes is left untouched ("same"); a
    new one is written ("new"); one whose bytes differ goes to `superseded`
    first and is then written ("changed"). A written file is built beside
    its place and moved in, so a build that stops half-way leaves no
    half-written file under a final name.
    """
    done: Dict[str, List[str]] = {"new": [], "changed": [], "same": []}
    for o in plan:
        dest = os_path(out / o.dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".building")
        if o.source is not None:
            if dest.is_file() and same_bytes(os_path(o.source), dest):
                done["same"].append(str(o.dest))
                continue
            shutil.copyfile(os_path(o.source), tmp)
        else:
            o.write(tmp)
            if dest.is_file() and same_bytes(tmp, dest):
                tmp.unlink()
                done["same"].append(str(o.dest))
                continue
        if dest.is_file():
            set_aside(dest, out, superseded)
            done["changed"].append(str(o.dest))
        else:
            done["new"].append(str(o.dest))
        move_into_place(tmp, dest)
    return done


def prune(plan: Sequence[Output], out: Path, superseded: Path,
          owned: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(files moved to `superseded`, files left in place).

    Inside the folders this build owns (`owned`, the set folders it wrote),
    a file that is not in the plan is one an earlier build made and this one
    does not: it is moved to `superseded`. Anywhere else in the hand-in
    folder a file the build did not make was put there by hand; it is left
    alone and named.
    """
    wanted = {str(os_path(out / o.dest)).lower() for o in plan}
    root = os_path(out)
    moved, left = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or str(p).lower() in wanted:
            continue
        rel = Path(plain(p)).relative_to(Path(plain(root)))
        if rel.parts and rel.parts[0] in owned:
            set_aside(p, out, superseded)
            moved.append(str(rel))
        elif str(rel) != gh.VALIDATION_FILE:      # scripts/rebuild_grasp_set.py's
            left.append(str(rel))
    # an owned folder emptied by that goes too (it holds nothing to keep)
    for d in sorted((p for p in root.rglob("*") if p.is_dir()),
                    key=lambda p: len(str(p)), reverse=True):
        rel = Path(plain(d)).relative_to(Path(plain(root)))
        if rel.parts and rel.parts[0] in owned and not any(d.iterdir()):
            d.rmdir()
    return moved, left


def rebuild_arguments(argv: Sequence[str]) -> List[str]:
    """This run's arguments as the README prints them for a rebuild: all of
    them but where the superseded files of this one run went."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "--superseded":
            skip = True
        elif not str(a).startswith("--superseded="):
            out.append(str(a))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build the professor's hand-in folder from protocol "
                    "session folders (accepted takes only; no image but "
                    "the grasp takes' pictures cut to the hand).")
    p.add_argument("--out", type=Path, required=True,
                   help="the hand-in folder, e.g. \"..\\xr trainer\\grasp and "
                        "flexion set for professor 2026-09-28\"")
    p.add_argument("--grasps", type=Path, default=None,
                   help="Set A session folder")
    p.add_argument("--flexion", type=Path, nargs="+", action="extend",
                   default=[], help="Set B session folder(s), one per hand")
    p.add_argument("--sequences", type=Path, nargs="+", action="extend",
                   default=[], help="Set C session folder(s), one per hand")
    p.add_argument("--transitions", type=Path, nargs="+", action="extend",
                   default=[],
                   help="transition session folder(s) of scripts/leap/"
                        "record_transitions.py, added to the grasp set's "
                        "transition table (needs --grasps)")
    p.add_argument("--operator", default=None,
                   help="operator name for the README (default: from "
                        "session.json)")
    p.add_argument("--not-recorded-reason", default=None,
                   help="why the protocol items of Sets B and C that have no "
                        "take in the folder were not recorded; the README "
                        "prints it after naming them (they are named either "
                        "way)")
    p.add_argument("--participant", default="P01",
                   help="participant id written in the transition table "
                        "(default: P01)")
    p.add_argument("--superseded", type=Path, default=None,
                   help="where files this build replaces or no longer makes "
                        "are moved (default: _superseded\\<out's name>\\"
                        "<time> beside --out)")
    p.add_argument("--command", default=None,
                   help="the rebuild command, as the README's last section "
                        "prints it (default: this run's own)")
    p.add_argument("--start-speed", type=float, default=None,
                   help="transition timing: pose speed, mm/s, a movement has "
                        "started at")
    p.add_argument("--start-sustain", type=float, default=None,
                   help="transition timing: seconds it must stay that fast")
    p.add_argument("--settle-speed", type=float, default=None,
                   help="transition timing: pose speed, mm/s, a pose is "
                        "settled below")
    p.add_argument("--settle-window", type=float, default=None,
                   help="transition timing: seconds it must stay below it")
    p.add_argument("--cue-still", type=float, default=None,
                   help="transition timing: seconds around the cue the pose "
                        "must be still for")
    p.add_argument("--allow-mock", action="store_true",
                   help="package mock (synthetic) sessions too; only for "
                        "rehearsing this script, never for the hand-in")
    args = p.parse_args(argv)
    thresholds = {key: getattr(args, opt) for opt, key in THRESHOLD_OPTIONS
                  if getattr(args, opt) is not None}

    grasps = load_sessions([args.grasps] if args.grasps else [], pc.GRASPS,
                           args.allow_mock)
    flexion = load_sessions(args.flexion, pc.FLEXION, args.allow_mock)
    sequences = load_sessions(args.sequences, pc.SEQUENCES, args.allow_mock)
    packaged = {pc.GRASPS: grasps, pc.FLEXION: flexion,
                pc.SEQUENCES: sequences}
    if not any(packaged.values()):
        print("nothing to package: no session folder could be read")
        return 1

    if args.transitions and not grasps:
        print("skipped transitions: they go into the grasp set's table and "
              "no grasp session is being packaged")
    exporter = prof_exporter()
    plan, infos = build_plan(grasps, flexion, sequences, exporter,
                             args.participant, thresholds, args.transitions,
                             args.allow_mock)
    operator = args.operator or ", ".join(sorted({
        str(s.meta.get("operator")) for v in packaged.values() for s in v
        if s.meta.get("operator")}))
    built = time.strftime("%Y-%m-%d %H:%M")
    command = args.command or " ".join(
        ["python", "scripts/package_professor_set.py"]
        + [f'"{a}"' if " " in a else a for a in rebuild_arguments(
            argv if argv is not None else sys.argv[1:])])
    fresh = readme_text(packaged, infos, operator, built, command,
                        thresholds, args.not_recorded_reason)
    readme = keep_unchanged_readme(args.out, fresh)
    plan.append(Output(Path("README.txt"),
                       write=lambda dest: dest.write_text(readme,
                                                          encoding="utf-8")))

    bad = image_violations(plan)
    if bad:
        print("REFUSED: these would put an image other than a planned "
              "picture in the hand-in folder; nothing was written:")
        for b in bad:
            print(f"  {b}")
        return 1
    allowed = allowed_images(plan)
    dests = [str(o.dest).lower() for o in plan]
    if len(set(dests)) != len(dests):
        dup = sorted({d for d in dests if dests.count(d) > 1})
        print("REFUSED: two takes map to the same file name; nothing was "
              "written:\n  " + "\n  ".join(dup))
        return 1

    out = args.out
    # An image already in the folder that this build does not plan was put
    # there by hand (an earlier build's pictures are planned again): say so
    # before anything is written or moved, and stop.
    if os_path(out).is_dir():
        images = stray_images(out, allowed)
        if images:
            print("REFUSED: the hand-in folder holds image files this build "
                  "does not make; remove them:")
            for i in images:
                print(f"  {plain(i)}")
            return 1
    superseded = args.superseded or (
        out.parent / "_superseded" / out.name / time.strftime("%Y%m%d_%H%M%S"))
    owned = [k for k in pc.SETS if packaged.get(k)]
    if grasps:
        owned.append(gh.TRANSITIONS_DIR)
    try:
        done = write_plan(plan, out, superseded)
        moved, left = prune(plan, out, superseded, owned)
        if readme != fresh and (done["new"] or done["changed"] or moved):
            # the README was kept because only its build time differed, but
            # this build changed another file: the time is this build's
            again = write_plan(
                [Output(Path("README.txt"),
                        write=lambda dest: dest.write_text(
                            fresh, encoding="utf-8"))], out, superseded)
            done["same"] = [x for x in done["same"] if x != "README.txt"]
            done["changed"] += again["changed"]
    except OSError as e:
        print(f"FAILED writing {out}: {e}\n  the folder is incomplete; fix "
              "the cause and run this again")
        return 1

    images = stray_images(out, allowed)
    if images:
        print("REFUSED: the hand-in folder holds image files this build "
              "does not make; remove them:")
        for i in images:
            print(f"  {plain(i)}")
        return 1

    for kind in pc.SETS:
        info = infos.get(kind)
        if info is None:
            continue
        hands = info.get("hands")
        per_hand = (" (" + ", ".join(f"{h} {n}" for h, n in
                                     sorted(hands.items())) + ")"
                    if hands else "")
        frames = (f", joint frames {info['joint_frames']} CSV and "
                  f"{info['joint_frames_pdf']} PDF"
                  if info.get("joint_frames") or info.get("joint_frames_pdf")
                  else "")
        if info.get("images") is not None:
            frames += f", {info['images']} picture(s)"
        print(f"{kind}: {info['takes']} take(s){per_hand}{frames}")
        for m in info.get("missing", []):
            print(f"  not handed in: {m}")
        for m in info.get("problems", []):
            print(f"  PROBLEM: {m}")
    print(f"wrote {len(plan)} file(s) to {out}: {len(done['new'])} new, "
          f"{len(done['changed'])} changed, {len(done['same'])} unchanged")
    if done["changed"] or moved:
        print(f"  {len(done['changed'])} replaced and {len(moved)} no longer "
              f"made: the earlier files are in {superseded}")
    for name in left:
        print(f"  left in place (not made by this build): {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
