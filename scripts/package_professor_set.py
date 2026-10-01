"""Build the hand-in folder for the professor: grasps, finger flexion, sequences.

Plan section 5 and contract `docs/protocol_formats.md` section 9 fix what goes
in; this script is the only thing that writes it, so the folder is rebuilt
from the session folders every time instead of being assembled by hand:

    <out>\\README.txt
    <out>\\grasps\\<item>_<hand>_take<N>.jsonl, ..._keypoints.txt,
                  <item>_<hand>_all_frames.txt, grasps_summary.csv
    <out>\\finger_flexion\\<hand>\\<take>.jsonl, camera_<take>.jsonl,
                  <take>.events.jsonl, <take>.txt, flexion_report.txt
    <out>\\sequences\\<hand>\\ the same layout, plus sequence_check.csv

  python scripts/package_professor_set.py --out "..\\xr trainer\\grasp and flexion set for professor 2026-09-28" ^
      --grasps recordings/protocol/grasps/20260928_101500_left ^
      --flexion recordings/protocol/finger_flexion/20260928_140501_left recordings/protocol/finger_flexion/20260928_150210_right ^
      --sequences recordings/protocol/sequences/20260928_143012_left recordings/protocol/sequences/20260928_153340_right ^
      --operator "N Kim"

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
                  wrist position; the operator's hand is the hand ids the
                  recorder followed (`operator_hand_ids` in the meta), else,
                  for a take recorded before it followed one, the tracker
                  label `protocol_check.choose_camera_label` picks

  Joint frames, when `scripts/joint_frames_view.py --session` has been run
  on a session: its `joint_frames/<take>.csv` (and `camera_<take>.csv`) for
  the accepted takes, renamed like the take's other files, and the one
  `joint_frames.pdf`, into a `joint_frames` folder beside the set's files.
  Never its PNGs: the PDF is the drawing's hand-in form. Without that
  folder the set packages exactly as before.

  No photograph goes in: the hand-cropped stills are for our own checking.
  Before anything is written, every file the plan would put in the folder is
  checked by name and, for copies, by its first bytes; one image and nothing
  is written and the exit code is 1. After writing, the whole folder is
  scanned the same way, so an image left there by hand also fails the run.

REPORTS COME FROM THE CHECKER
  `flexion_report.txt` and `sequence_check.csv` are built from each session's
  `check.csv` / `check.txt` (`scripts/check_protocol.py`) when those cover
  exactly the session's takes. When they are missing or older than a retake,
  the checker's functions are run here in memory (fixed bands) and a line
  says so; nothing is written back into the session folder.

A session folder that does not exist, or holds no session.json, is skipped
with a printed line, not an error. So is a MOCK session (session.json with
`"mock": true`, a `"mock"` dict with any flag set, or a `"mock_flags"` dict
with any flag set; see `protocol_check.mock_flags`): synthetic hands are for
rehearsing the tools, never for the professor, and `--allow-mock` is the
only way one gets in (for rehearsing this script, and the README then says
so at the top). The exit code is 1 for an image, or when not one session
could be packaged.
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cam_hand import protocol_check as pc
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
    "Finger spread is a template constant in the glove data; abduction is "
    "only in the camera files.",
    "The right glove sometimes reads too open, on a session-dependent "
    "basis; compare its fractions with the camera beside it.",
    "Curl creeps during holds: a held finger's glove reading drifts slowly "
    "while the camera's does not.",
    "Lag differs by hand: the left glove trails the camera by about 0.10 s, "
    "the right by about 0.47 s (measured per cycle and per take in "
    "flexion_report.txt, never assumed).",
]


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


def camera_blocks(srcs: Sequence[Tuple[Path, Optional[Sequence[int]]]],
                  hand: str, exporter):
    """Every frame of the operator's hand in camera mm, take after take.

    `srcs` are (leap file, the hand ids its meta says were the operator's,
    or None). With ids, the operator's frames are the lines of those ids,
    whatever the tracker called them: the recorder followed that hand by
    id, and the other hand in view is in the same file under other ids.
    Without (a take recorded before the recorder followed a hand), which
    tracker label holds the operator's hand is chosen per take by
    `protocol_check.choose_camera_label`, the same rule the checker uses.
    """
    for src, ids in srcs:
        frames, _bad = pc.read_frames(src)
        if ids:
            wanted = {int(i) for i in ids}
            for d, frame in frames:
                if d.get("hand_id") is not None and int(d["hand_id"]) in wanted:
                    yield exporter.frame_block(frame, float(d["wall_time"]))
            continue
        counts: Dict[str, int] = {}
        for d, _f in frames:
            counts[d.get("hand_side")] = counts.get(d.get("hand_side"), 0) + 1
        label, _note = pc.choose_camera_label(counts, hand)
        for d, frame in frames:
            if d.get("hand_side") == label:
                yield exporter.frame_block(frame, float(d["wall_time"]))


# --- the plan: every file the folder will hold -----------------------------------

@dataclass
class Output:
    """One file of the hand-in folder: copied from `source` or written."""

    dest: Path                                   # relative to the out folder
    source: Optional[Path] = None
    write: Optional[Callable[[Path], object]] = None


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
    bad = []
    for o in plan:
        if is_image(o.dest, check_bytes=False):
            bad.append(f"{o.dest} (an image name)")
        elif o.source is not None and is_image(o.source):
            bad.append(f"{o.dest} (copied from the image {o.source})")
    return bad


def images_in(folder: Path) -> List[Path]:
    """Every image in the folder, by name or by first bytes, however deep."""
    return [p for p in sorted(os_path(folder).rglob("*")) if p.is_file()
            and is_image(p)]


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
    s = os.path.abspath(str(path))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = ("\\\\?\\UNC\\" + s[2:]) if s.startswith("\\\\") else \
            ("\\\\?\\" + s)
    return Path(s)


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

SUMMARY_COLUMNS = ["grasp", "label", "source", "figure", "take", "hand",
                   "name", "frames", "tracked_percent",
                   "reacquisitions_in_static_interval", "reacquisitions",
                   "grab_strength", "pinch_strength"] + [
    f"curl_{f}" for f in pc.FINGERS] + [
    "static_interval_start", "static_interval_end", "medoid_wall_time",
    "operator_hand_label", "other_hand_in_view", "orientation_note", "session"]


JOINT_FRAMES_DIR = "joint_frames"          # scripts/joint_frames_view.py --session
JOINT_FRAMES_PDF = "joint_frames.pdf"


def plan_joint_frames(session: pc.Session, folder: Path,
                      names: Sequence[Tuple[str, str]],
                      pdf_name: str = JOINT_FRAMES_PDF
                      ) -> Tuple[List[Output], List[str]]:
    """The session's joint-frame exports, when it has them.

    `names` pairs each handed-in take's session name with its name in the
    hand-in folder. A CSV of a take that is not handed in stays out, with a
    line saying so; a PNG is never looked at. Returns ([], []) for a session
    without a joint_frames folder, which then packages as before.
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
    return plan, notes


def joint_frame_counts(plan: Sequence[Output]) -> Tuple[int, int]:
    """(joint-frame CSVs, joint-frame PDFs) among planned outputs."""
    frames = [o for o in plan if o.dest.parent.name == JOINT_FRAMES_DIR]
    return (sum(1 for o in frames if o.dest.suffix == ".csv"),
            sum(1 for o in frames if o.dest.suffix == ".pdf"))


def plan_grasps(session: pc.Session, exporter) -> Tuple[List[Output], dict]:
    hand = session.hand
    plan: List[Output] = []
    rows: List[dict] = []
    by_item: Dict[str, List[Tuple[int, Path, Optional[List[int]]]]] = {}
    missing: List[str] = []
    static_s: List[float] = []
    names: List[Tuple[str, str]] = []
    for t in accepted(session):
        stem = f"{t.item}_{hand}_take{t.take}"
        leap = t.path("leap")
        if leap is None:
            missing.append(f"{t.name}: no camera file, take not handed in")
            continue
        names.append((t.name, stem))
        plan.append(Output(Path("grasps") / f"{stem}.jsonl", source=leap))
        meta = pc.read_json(t.path("meta")) if t.path("meta") else None
        meta = meta if isinstance(meta, dict) else {}
        ids = meta.get("operator_hand_ids")
        ids = [int(i) for i in ids] if isinstance(ids, list) and ids else None
        by_item.setdefault(t.item, []).append((int(t.take or 0), leap, ids))
        if t.path("keypoints"):
            plan.append(Output(Path("grasps") / f"{stem}_keypoints.txt",
                               source=t.path("keypoints")))
        else:
            missing.append(f"{t.name}: no keypoints file from the recorder")
        cfg = session.item(t.item) or {}
        interval = meta.get("static_interval") or [None, None]
        if len(interval) > 1 and None not in interval[:2]:
            static_s.append(float(interval[1]) - float(interval[0]))
        tracked = meta.get("tracked_fraction")
        rows.append({
            "grasp": t.item, "label": cfg.get("label", ""),
            "source": cfg.get("source") or "", "figure": cfg.get("figure") or "",
            "take": t.take, "hand": hand, "name": t.name,
            "frames": meta.get("frames"),
            "tracked_percent": (None if tracked is None
                                else round(100.0 * float(tracked), 1)),
            # the gate's count (protocol_check.check_grasp_take): the
            # static interval's own when the meta has it, else the take's
            "reacquisitions_in_static_interval": (
                (meta.get("gate") or {}).get(
                    "reacquisitions_in_static_interval",
                    meta.get("reacquisitions"))
                if isinstance(meta.get("gate") or {}, dict)
                else meta.get("reacquisitions")),
            "reacquisitions": meta.get("reacquisitions"),
            "grab_strength": meta.get("grab_strength"),
            "pinch_strength": meta.get("pinch_strength"),
            **{f"curl_{f}": (meta.get("curls") or {}).get(f)
               for f in pc.FINGERS},
            "static_interval_start": interval[0],
            "static_interval_end": interval[1] if len(interval) > 1 else None,
            "medoid_wall_time": meta.get("medoid_wall_time"),
            # the tracker's label on the summary frame, and whether another
            # hand was in view (its frames are in the take's JSONL, never in
            # the keypoints or the all-frames file)
            "operator_hand_label": meta.get("operator_hand_label"),
            "other_hand_in_view": bool(meta.get("other_hand_ids")),
            "orientation_note": meta.get("orientation_note", ""),
            "session": session.name})
    for item, takes in sorted(by_item.items()):
        srcs = [(p, ids) for _n, p, ids in sorted(takes, key=lambda x: x[:2])]
        plan.append(Output(
            Path("grasps") / f"{item}_{hand}_all_frames.txt",
            write=lambda dest, srcs=srcs: write_blocks(
                dest, camera_blocks(srcs, hand, exporter))))

    def write_summary(dest: Path, rows=rows) -> None:
        with open(dest, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: pc.csv_value(v) for k, v in r.items()})
    plan.append(Output(Path("grasps") / "grasps_summary.csv",
                       write=write_summary))
    frames, notes = plan_joint_frames(session, Path("grasps"), names)
    plan += frames
    missing += notes
    csvs, pdfs = joint_frame_counts(frames)
    static = sorted(static_s)[len(static_s) // 2] if static_s else None
    return plan, {"takes": len(rows), "items": len(by_item),
                  "missing": missing, "static_s": static,
                  "joint_frames": csvs, "joint_frames_pdf": pdfs}


# --- Sets B and C --------------------------------------------------------------------

def check_for(session: pc.Session) -> Tuple[List[Dict[str, str]], str, str]:
    """(check.csv rows, check.txt text, where they came from) for a session."""
    csv_path, txt_path = session.path / "check.csv", session.path / "check.txt"
    if csv_path.is_file() and txt_path.is_file():
        rows = pc.read_check_csv(csv_path)
        if pc.check_is_current(session, rows):
            return (rows, txt_path.read_text(encoding="utf-8"),
                    "check.csv and check.txt in the session folder")
        why = "do not match the session's takes"
    else:
        why = "are missing"
    print(f"  {session.set} {session.name}: check.csv/check.txt {why}; "
          "running the checker in memory (fixed bands)")
    check = pc.check_session(session)
    return (pc.csv_rows(check), pc.render_text(check),
            "checked while packaging (fixed bands)")


def plan_glove_takes(session: pc.Session, kind: str, exporter,
                     pdf_name: str = JOINT_FRAMES_PDF
                     ) -> Tuple[List[Output], List[str]]:
    folder = Path(kind) / session.hand
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
        if t.path("leap"):
            plan.append(Output(folder / f"camera_{t.name}.jsonl",
                               source=t.path("leap")))
        if t.path("events"):
            plan.append(Output(folder / f"{t.name}.events.jsonl",
                               source=t.path("events")))
        else:
            missing.append(f"{t.name}: no events file")
    frames, notes = plan_joint_frames(session, folder, names, pdf_name)
    return plan + frames, missing + notes


def _j(row: Dict[str, str], key: str):
    try:
        return json.loads(row.get(key) or "null")
    except ValueError:
        return None


def flexion_report(hand: str, parts: List[Tuple[pc.Session, list, str, str]]
                   ) -> str:
    lines = [f"Finger flexion (Set B), {hand} hand", "=" * 78,
             "Glove at full rate, Ultraleap camera as the reference. One row "
             "per handed-in take; the",
             "checker's full output for each session follows, rejected "
             "attempts included and marked.",
             "span = 5th to 95th percentile of the glove fraction over the "
             "take, 0 = warm-up open palm,",
             "1 = warm-up fist. cycles ev/pk/exp = bend cues / glove curl "
             "peaks / the protocol's count.",
             "Lag is per cycle (median shown) and per take, positive = the "
             "glove trails the camera, only",
             "where the camera followed the finger (camera span >= 0.50).",
             ""]
    head = (f"  {'take':<38} {'finger':<7} {'span':>5} {'cycles':>9} "
            f"{'cam span':>8} {'followed':>8} {'lag med':>7} {'lag take':>8} "
            f"{'hyst':>6}  others")
    for session, rows, text, source in parts:
        lines.append(f"Session {session.name} ({session.meta.get('started')} "
                     f"to {session.meta.get('ended')}), {source}")
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
                f"{_num(r.get('camera_span')):>8} {followed:>8} "
                f"{_num(r.get('lag_ms_median'), '.0f'):>7} "
                f"{_num(r.get('lag_ms_take'), '.0f'):>8} "
                f"{_num(r.get('hysteresis_median'), '+.3f'):>6}  "
                + " ".join(f"{f} {_num(v)}" for f, v in others.items()))
        lines.append("")
        lines.append(text.rstrip("\n"))
        lines.append("")
    return "\n".join(lines) + "\n"


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


def plan_glove_set(sessions: List[pc.Session], kind: str, exporter
                   ) -> Tuple[List[Output], dict]:
    plan: List[Output] = []
    info = {"takes": 0, "missing": [], "hands": {}, "joint_frames": 0,
            "joint_frames_pdf": 0}
    by_hand: Dict[str, List[Tuple[pc.Session, list, str, str]]] = {}
    per_hand: Dict[str, int] = {}
    for session in sessions:
        per_hand[session.hand] = per_hand.get(session.hand, 0) + 1
    for session in sessions:
        # two sessions of one hand share a folder: their PDFs keep apart
        pdf_name = (JOINT_FRAMES_PDF if per_hand[session.hand] == 1 else
                    f"joint_frames_{session.name}.pdf")
        takes, missing = plan_glove_takes(session, kind, exporter, pdf_name)
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
    ends = pc.endpoints(session.warmup, pc.GLOVE)
    if not ends:
        return ["    warm-up: none recorded"]
    return ["    warm-up glove curl, thumb index middle ring pinky: open "
            + " ".join(pc.fmt(ends[f][0]) if f in ends else "-"
                       for f in pc.FINGERS)
            + ", fist " + " ".join(pc.fmt(ends[f][1]) if f in ends else "-"
                                   for f in pc.FINGERS)]


def _item_counts(session: pc.Session) -> str:
    counts: Dict[str, int] = {}
    for t in accepted(session):
        counts[t.item] = counts.get(t.item, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"


def reason_kind(reason: str) -> str:
    """What kind of rejection a recorder's reason is, without its numbers.

    The reasons name the take's own numbers ("tracked 63 percent: lost 1
    time, for 1.9 s (not back by the end) with the hand at 23 cm ..."), so
    no two are equal; this groups them so the README can say which kind a
    grasp was rejected for most often.
    """
    r = " ".join(str(reason or "").split())
    low = r.lower()
    if not low:
        return "no reason recorded"
    if low.startswith("no hand was tracked"):
        return "no hand tracked during the take"
    if low.startswith("no open hand acquired"):
        return r.split(" (")[0]
    if "while forming the grasp" in low:
        return "hand lost while forming the grasp"
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


def short_grasp_lines(session: pc.Session) -> List[str]:
    """The README table of the grasps short of their kept takes.

    One row per grasp of the session (session.json's item list, else the
    protocol's) with fewer kept takes than takes_per_item: kept, rejected
    attempts, and the most common kind of rejection (`reason_kind`) with
    how many of the rejections it was; kinds tied for most common are all
    named. A grasp never attempted says so.
    """
    meta = session.meta
    proto = session.protocol or {}
    target = int(meta.get("takes_per_item") or proto.get("takes_per_item") or 3)
    items = [str(i) for i in (meta.get("items") or [])] or [
        str(it.get("id")) for it in proto.get("items", []) or []]
    for t in session.takes:
        if t.item not in items:
            items.append(t.item)
    title = f"Grasps short of {target} kept takes"
    rows = []
    for item in items:
        mine = [t for t in session.takes if t.item == item]
        kept = sum(1 for t in mine if t.accepted)
        if kept >= target:
            continue
        kinds: Dict[str, int] = {}
        for t in mine:
            if not t.accepted:
                k = reason_kind(t.reason)
                kinds[k] = kinds.get(k, 0) + 1
        rejected = sum(kinds.values())
        if rejected:
            most = max(kinds.values())
            why = "; ".join(f"{k} ({n} of {rejected})"
                            for k, n in kinds.items() if n == most)
        else:
            why = "not attempted"
        rows.append((item, kept, rejected, why))
    if not rows:
        return [f"    {title}: none, every grasp has {target}."]
    w = max(len("grasp id"), *(len(r[0]) for r in rows))
    out = [f"    {title}:",
           f"      {'grasp id':<{w}}  kept  rejected  most common reject reason"]
    out += [f"      {item:<{w}}  {kept:>4}  {rejected:>8}  {why}"
            for item, kept, rejected, why in rows]
    return out


def wrapped(text: str, first: str = "  ", rest: str = "    ") -> List[str]:
    return textwrap.wrap(text, width=96, initial_indent=first,
                         subsequent_indent=rest, break_on_hyphens=False)


def setup_lines(kind: str, sessions: List[pc.Session],
                info: Optional[dict] = None) -> List[str]:
    hands = ", ".join(sorted({s.hand for s in sessions})) or "-"
    proto = next((s.protocol for s in sessions if s.protocol), None) or {}
    if kind == pc.GRASPS:
        per = proto.get("takes_per_item", 3)
        dur = proto.get("duration_s", 5.0)
        return [
            f"  Hardware: camera only, bare hand ({hands}). Ultraleap Stereo "
            "IR 170 (the \"Leap Motion\" in",
            "  the request) flat on the table, lenses up; hand 20 to 40 cm "
            "above it, wrist in the field of view,",
            "  every finger chain visible. Mimed in the air, no object. "
            f"{per} takes per grasp, {dur:g} s each.",
        ] + wrapped(
            "Each take keeps all frames; the summary frame (_keypoints.txt) "
            "is the medoid of the take's static interval (the "
            + (f"{(info or {}).get('static_s'):.1f} s"
               if (info or {}).get("static_s") else "2 s")
            + " window with the least joint motion, its start and end in "
            "grasps_summary.csv): the final configuration.", "  ", "  ")
    if kind == pc.FLEXION:
        items = [f"{it['id']} ({it.get('cycles')} cycles, bend "
                 f"{it.get('bend_s'):g} s, hold {it.get('hold_s'):g} s, "
                 f"straighten {it.get('straighten_s'):g} s, rest "
                 f"{it.get('rest_s'):g} s)"
                 for it in proto.get("items", [])
                 if it.get("bend_s") is not None]
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
        ]
        if items:
            out += wrapped("Items: " + "; ".join(items) + ".", "  ", "    ")
        return out
    hold = proto.get("hold_s", 2.5)
    window = proto.get("check_window_s", 1.5)
    return [
        f"  Hardware: gloves on + camera ({hands} hand, one hand at a time), "
        "the same setup as Set B.",
        f"  Seven sequences; each step cued by a beep and held {hold:g} s "
        "(the first second is the",
        f"  movement; the check reads the last {window:g} s). Three takes "
        "per sequence, recorded as three",
        "  rounds in a shuffled order (the seed is in the session's own "
        "metadata).",
    ]


def joint_frame_contents(kind: str, info: dict) -> List[str]:
    """CONTENTS lines for a set's joint_frames folder, when it has one."""
    if not (info.get("joint_frames") or info.get("joint_frames_pdf")):
        return []
    where = "" if kind == pc.GRASPS else "<hand>\\"
    out = [f"      {where}joint_frames\\<take>.csv   the take's summary "
           "frame: every joint's position (mm),",
           "          orientation, axes and angles in the wrist frame "
           "(JOINT FRAMES below)"]
    if kind != pc.GRASPS:
        out.append(f"      {where}joint_frames\\camera_<take>.csv   the same "
                   "from the camera file of the take")
    out.append(f"      {where}joint_frames\\joint_frames.pdf   one page per "
               "summary frame: the hand drawn with every")
    out.append("          joint's x y z axes, the paper's 24 angles and the "
               "26-row table")
    return out


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


def readme_text(packaged: Dict[str, List[pc.Session]], infos: Dict[str, dict],
                operator: str, built: str) -> str:
    L: List[str] = []
    all_sessions = [s for v in packaged.values() for s in v]
    dates = sorted({str(s.meta.get("started", ""))[:10]
                    for s in all_sessions if s.meta.get("started")})
    mock = {s.name: pc.mock_flags(s.meta) for s in all_sessions
            if pc.mock_flags(s.meta)}
    L += ["Grasp and finger-flexion recordings", "=" * 78]
    if mock:
        L += wrapped("MOCK DATA: this folder was built with --allow-mock and "
                     "holds synthetic sessions, not recordings of a hand: "
                     + ", ".join(sorted(mock)) + ". It is not a hand-in.",
                     "", "")
    L += [f"Operator: {operator or 'not recorded'}",
          f"Recorded: {', '.join(dates) or 'not recorded'}",
          f"Folder built: {built} by scripts/package_professor_set.py from "
          "the session folders below.",
          "No photographs are included.", ""]

    L += ["CONTENTS", "-" * 78]
    for kind in pc.SETS:
        n = infos.get(kind, {}).get("takes", 0)
        if not packaged.get(kind):
            L.append(f"  {kind}\\  {SET_TITLES[kind]}: not included in this "
                     "folder")
            continue
        L.append(f"  {kind}\\  {SET_TITLES[kind]}: {n} take(s)")
        if kind == pc.GRASPS:
            L += ["      <grasp>_<hand>_take<N>.jsonl        every frame of "
                  "the take, 26 joints, camera",
                  "      <grasp>_<hand>_take<N>_keypoints.txt the summary "
                  "frame, 21 landmarks, camera mm",
                  "      <grasp>_<hand>_all_frames.txt       every frame of "
                  "the grasp's takes, 21 landmarks, camera mm",
                  "      grasps_summary.csv                  grasp, take, "
                  "frames, tracked %, grab, pinch, curls"]
        else:
            L += [f"      <hand>\\<take>.jsonl             glove, full rate,"
                  " 26 joints",
                  f"      <hand>\\camera_<take>.jsonl      the camera "
                  "reference for the same take (when it ran)",
                  f"      <hand>\\<take>.events.jsonl      every cue and "
                  "decision, same clock as the frames",
                  f"      <hand>\\<take>.txt               glove, every "
                  "frame, 21 landmarks, wrist at the origin, mm"]
            L.append("      <hand>\\flexion_report.txt       cycles, range, "
                     "hysteresis, lag, per finger and speed"
                     if kind == pc.FLEXION else
                     "      <hand>\\sequence_check.csv       step by step "
                     "fractions and pass/fail, glove and camera; "
                     "flexed = the fingers the step flexes, empty = "
                     "open hand")
        L += joint_frame_contents(kind, infos.get(kind, {}))
    L.append("")

    L += ["SETUP", "-" * 78]
    for kind in pc.SETS:
        if packaged.get(kind):
            L.append(f"{SET_TITLES[kind]}:")
            L += setup_lines(kind, packaged[kind], infos.get(kind))
            L.append("")

    L += ["SESSIONS", "-" * 78]
    for kind in pc.SETS:
        for s in packaged.get(kind, []):
            n_rej = sum(1 for t in s.takes if not t.accepted)
            L.append(f"  {kind}, {s.hand} hand, session {s.name}")
            L.append(f"    recorded {s.meta.get('started', '?')} to "
                     f"{s.meta.get('ended', '?')}, operator "
                     f"{s.meta.get('operator') or '-'}")
            L.append(f"    protocol {s.meta.get('protocol_name', kind)} "
                     f"version {s.meta.get('protocol_version', '?')}")
            if kind != pc.GRASPS:
                L.append("    XR Trainer calibrated at "
                         f"{s.meta.get('xr_trainer_calibrated_at') or '-'}")
                L += _endpoints_lines(s)
            if s.name in mock:
                L.append("    MOCK session (synthetic data): "
                         + ", ".join(mock[s.name]))
            L.append(f"    handed in: {len(accepted(s))} take(s) "
                     f"({_item_counts(s)}); {n_rej} rejected attempt(s) kept "
                     "back with their reasons")
            if kind == pc.GRASPS:
                # which grasps the camera could not hold, and why
                L += short_grasp_lines(s)
    for kind in pc.SETS:
        for m in infos.get(kind, {}).get("missing", []):
            L.append(f"  not handed in: {m}")
    L.append("")

    L += ["QUALITY RULES", "-" * 78,
          "Fixed numbers are acquisition gates that decide on the spot "
          "whether a take is usable; the",
          "continuous values are always kept in the reports, and no fixed "
          "number is the definition of",
          "a correct movement.",
          "  Set A: at least 90 % of the take's frames tracked and no "
          "re-acquisition inside the static",
          "    interval (the minimum, not the acceptance); acceptance is the "
          "still and the summary frame",
          "    matching the paper's figure by eye, fingertip by fingertip, "
          "decided on the spot and",
          "    timestamped by the recorder. grab, pinch and curls are in "
          "grasps_summary.csv.",
          "    The re-acquisition gate the grasp takes were judged on is "
          "reacquisitions_in_static_interval",
          "    (a column of grasps_summary.csv, from each take's meta); the "
          "reacquisitions column counts",
          "    the whole take and was not a gate. A take whose meta has no "
          "separate count was judged on",
          "    its reacquisitions figure, and that figure is repeated in "
          "both columns.",
          "  Set B: the cued finger's glove curl spans at least 60 % of its "
          "own open-to-fist range from",
          "    the session warm-up; the other four fingers' spans are "
          "reported, not failed (the ring drags",
          "    the middle and little finger along). Cycles are counted from "
          "the cues and from the glove",
          "    curl peaks and must agree with each other and with the "
          "protocol. Where the camera followed",
          "    the finger (camera span at least 0.50), the transfer curve, "
          "hysteresis and lag are reported",
          "    per cycle and per take.",
          "  Set C: for each step, the median glove fraction over the last "
          "part of the hold must be above",
          "    0.6 for the fingers the step flexes; a finger the step leaves "
          "straight fails only when it",
          "    reads flexed (0.6 or more), and between 0.3 and 0.6 it is "
          "reported as coupling (the ring",
          "    pulls its neighbours along), not failed. A take passes when "
          "every step does. The same",
          "    check on the camera where it is trusted is reported beside it, "
          "so a glove fault reads as",
          "    the glove being wrong, not the operator. sequence_check.csv "
          "has every fraction, the",
          "    pass/fail and the coupling per step, glove and camera.",
          "  Failed takes were redone in the same session; the failed "
          "attempts are not in this folder.",
          "  fraction = (open - curl) / (open - fist), with open and fist "
          "the finger's own medians from",
          "    the session warm-up (3 s open palm, 3 s fist); 0 = open palm, "
          "1 = fist. curl = fingertip-",
          "    to-wrist distance over palm length.",
          ""]

    if any(infos.get(k, {}).get("joint_frames") or
           infos.get(k, {}).get("joint_frames_pdf") for k in pc.SETS):
        L += JOINT_FRAMES_SECTION

    L += ["KNOWN GLOVE LIMITS (reported as such, not hidden)", "-" * 78]
    for x in KNOWN_LIMITS:
        L += wrapped(x, "  - ", "    ")
    L.append("")

    L += ["COORDINATE CONVENTIONS", "-" * 78,
          "  .jsonl  one JSON object per frame: wall_time (seconds since 1970"
          ", recording machine), the",
          "    26 OpenXR joints (PALM, WRIST, then thumb metacarpal to tip and"
          " each finger metacarpal,",
          "    proximal, intermediate, distal, tip). Each joint's x y z is a "
          "translation in its parent's",
          "    frame in metres and qx qy qz qw its rotation relative to the "
          "parent; chaining them from",
          "    the wrist gives the hand. Every line also carries session, "
          "item and take.",
          "    Glove: the wrist is the root at the glove's own origin (the "
          "glove senses no position in",
          "    the room). Camera: the wrist carries its position in the "
          "Ultraleap's desktop space (origin",
          "    at the module, +x along the camera baseline, +y up, +z toward "
          "the operator, metres), and",
          "    each line also holds abs26 (all 26 joint positions in that "
          "space) and palm_abs.",
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
          "    hand-ins. Camera text files (grasps): the camera frame in "
          "millimetres, the Wrist line giving",
          "    the measured wrist position (not zero).",
          ] + wrapped(
          "Hand ID: in the grasp keypoint files (_keypoints.txt and "
          "_all_frames.txt) the 'Hand ID' line is the tracker's own "
          "left/right label, written untouched. It is often wrong: in the "
          "September hand-in the Ultraleap labelled the operator's left hand "
          "\"right\" in 20 of 21 poses. The hand actually recorded is the one "
          "in the file name and in session.json (listed under SESSIONS "
          "above). In the glove files the Hand ID is the glove's own side, "
          "which is reliable.", "  ", "    ") + [
          "  .events.jsonl: t is the same clock as wall_time; kind is "
          "take_start, cue, take_end or",
          "    decision; a cue names the step, its label, the fingers that "
          "should be flexed and, for",
          "    Set B, the cycle and phase (bend, hold, straighten, rest).",
          ""]
    return "\n".join(L) + "\n"


# --- main ----------------------------------------------------------------------------------

def build_plan(grasps: List[pc.Session], flexion: List[pc.Session],
               sequences: List[pc.Session], exporter
               ) -> Tuple[List[Output], Dict[str, dict]]:
    plan: List[Output] = []
    infos: Dict[str, dict] = {}
    if grasps:
        p, info = plan_grasps(grasps[0], exporter)
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build the professor's hand-in folder from protocol "
                    "session folders (accepted takes only, no images).")
    p.add_argument("--out", type=Path, required=True,
                   help="the hand-in folder, e.g. \"..\\xr trainer\\grasp and "
                        "flexion set for professor 2026-09-28\"")
    p.add_argument("--grasps", type=Path, default=None,
                   help="Set A session folder")
    p.add_argument("--flexion", type=Path, nargs="+", action="extend",
                   default=[], help="Set B session folder(s), one per hand")
    p.add_argument("--sequences", type=Path, nargs="+", action="extend",
                   default=[], help="Set C session folder(s), one per hand")
    p.add_argument("--operator", default=None,
                   help="operator name for the README (default: from "
                        "session.json)")
    p.add_argument("--allow-mock", action="store_true",
                   help="package mock (synthetic) sessions too; only for "
                        "rehearsing this script, never for the hand-in")
    args = p.parse_args(argv)

    grasps = load_sessions([args.grasps] if args.grasps else [], pc.GRASPS,
                           args.allow_mock)
    flexion = load_sessions(args.flexion, pc.FLEXION, args.allow_mock)
    sequences = load_sessions(args.sequences, pc.SEQUENCES, args.allow_mock)
    packaged = {pc.GRASPS: grasps, pc.FLEXION: flexion,
                pc.SEQUENCES: sequences}
    if not any(packaged.values()):
        print("nothing to package: no session folder could be read")
        return 1

    exporter = prof_exporter()
    plan, infos = build_plan(grasps, flexion, sequences, exporter)
    operator = args.operator or ", ".join(sorted({
        str(s.meta.get("operator")) for v in packaged.values() for s in v
        if s.meta.get("operator")}))
    built = time.strftime("%Y-%m-%d %H:%M")
    readme = readme_text(packaged, infos, operator, built)
    plan.append(Output(Path("README.txt"),
                       write=lambda dest: dest.write_text(readme,
                                                          encoding="utf-8")))

    bad = image_violations(plan)
    if bad:
        print("REFUSED: these would put an image in the hand-in folder; "
              "nothing was written:")
        for b in bad:
            print(f"  {b}")
        return 1
    dests = [str(o.dest).lower() for o in plan]
    if len(set(dests)) != len(dests):
        dup = sorted({d for d in dests if dests.count(d) > 1})
        print("REFUSED: two takes map to the same file name; nothing was "
              "written:\n  " + "\n  ".join(dup))
        return 1

    out = args.out
    if out.is_dir() and any(out.iterdir()):
        print(f"note: {out} already has files; files from an earlier build "
              "that are not in this one stay there")
    for o in plan:
        dest = os_path(out / o.dest)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if o.source is not None:
                shutil.copyfile(os_path(o.source), dest)
            else:
                o.write(dest)
        except OSError as e:
            print(f"FAILED writing {out / o.dest}: {e}\n  the folder is "
                  "incomplete; fix the cause and run this again")
            return 1

    images = images_in(os_path(out))
    if images:
        print("REFUSED: the hand-in folder holds image files; remove them:")
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
        print(f"{kind}: {info['takes']} take(s){per_hand}{frames}")
        for m in info.get("missing", []):
            print(f"  not handed in: {m}")
    print(f"wrote {len(plan)} file(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
