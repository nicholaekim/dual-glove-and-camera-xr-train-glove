"""See every joint's x y z frame and the paper's angles: live, per take, per session.

The numbers come from `xr_hand.joint_frames` (read its docstring for the
axis and sign conventions); this script only shows and exports them.

LIVE (camera only, bare hand; or gloves on + camera with --glove)

  .venv\\Scripts\\python.exe scripts\\joint_frames_view.py --live
  .venv\\Scripts\\python.exe scripts\\joint_frames_view.py --live --glove

  The IR image with the tracked hand, a 1 cm axis triad at each of the 26
  joints (x red, y green, z blue) and, beside it, the 26 rows (joint, x y z
  in mm in the wrist frame, flexion, abduction) and the paper's 24 angles.
  Keys: g switches the table between the camera hand and the glove hand,
  n highlights the next finger, s saves a PNG of the window into
  recordings\\joint_frames\\, q or Esc quits. The glove has no position of
  its own, so its hand is drawn at the camera hand's wrist (or above the
  module when no hand is tracked).

  It is a read-only client of the tracking service, like
  scripts\\leap\\camera_view.py, so it runs next to any recorder. `--glove`
  listens on the glove's OSC port, which a glove recorder also needs: do
  not add --glove while one is running.

ONE TAKE (no hardware)

  .venv\\Scripts\\python.exe scripts\\joint_frames_view.py --take recordings\\sync_day2\\glove\\fist_left_take1_20260920_195141.jsonl

  Picks the take's medoid frame (the rule of scripts\\leap\\record_frame.py:
  the real frame closest to the take's mean after a rigid alignment), or
  `--frame N` (the file's line N, from 0), and writes <take>.csv (26 rows)
  and <take>.png (the hand drawn in the wrist frame on a plain background,
  no IR pixels) into --out (default: beside the take), then prints the 24
  angles. Glove or camera is read from the line format (`--source`
  overrides).

A SESSION (no hardware)

  .venv\\Scripts\\python.exe scripts\\joint_frames_view.py --session recordings\\protocol\\grasps\\20260928_101500_left

  Every accepted take's summary frame into <session>\\joint_frames\\:
  Set A grasps use the static-interval medoid the recorder stored in
  meta\\<take>.json; Sets B and C the medoid of the whole take, for the
  glove file (<take>.csv) and the camera file (camera_<take>.csv). Also one
  joint_frames.pdf with a page per summary frame, the drawing and the
  26-row table: the PDF is what goes to the professor, because the hand-in
  folder takes no image files (scripts\\package_professor_set.py copies the
  CSVs and the PDF, never the PNGs).

`--frame-ref palm` expresses everything against the palm instead of the
wrist joint; on the camera the wrist joint's axes are the forearm's (see
xr_hand.joint_frames).
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):                    # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from xr_hand import joint_frames as jf
from xr_hand.joints import BONES

REPO = Path(__file__).resolve().parents[1]
SNAP_DIR = REPO / "recordings" / "joint_frames"
OUT_SUBDIR = "joint_frames"
PDF_NAME = "joint_frames.pdf"

AXIS_LEN_M = 0.010                               # the 1 cm triad
AXIS_RGB = [(230, 30, 30), (20, 170, 40), (30, 90, 240)]      # x red, y green, z blue
HAND_RGB = {"left": (0, 150, 190), "right": (210, 30, 30)}    # cyan left, red right
LIVE_HAND_BGR = {"left": (255, 220, 0), "right": (60, 60, 255)}   # camera_view.py
HIGHLIGHT_BGR = (0, 230, 255)
FINGER_CYCLE = [None, "THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"]
CONVENTION_LINES = [
    "Wrist frame: origin at the wrist joint, axes of the wrist joint (camera: the forearm), mm.",
    "Angles are from the parent joint's frame, order x (flexion), y (abduction), z (twist):",
    "flex + toward the palm, abd + toward the thumb, twist + back of the bone toward the little finger.",
]


# --- reading takes ----------------------------------------------------------------
def read_lines(path: Path) -> List[dict]:
    """Every parseable JSON line of a take, in file order (bad lines skipped)."""
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for text in f:
            if not text.strip():
                continue
            try:
                d = json.loads(text)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def medoid_line(lines: Sequence[dict], label: Optional[str] = None,
                prefer: Optional[str] = None) -> Tuple[dict, int]:
    """(line, index in `lines`) of the whole take's medoid for one hand label.

    `label` None means the label with the most tracked lines (`prefer`
    breaks a tie), the rule of `leap_hand.static_interval`.
    """
    from leap_hand.static_interval import (hand_rows, medoid_index,
                                           operator_label, row_frame)
    side = label or operator_label(lines, prefer)
    picked = hand_rows(lines, side)
    if not picked:
        raise ValueError(f"no tracked line of the {side!r} hand in this take")
    k = medoid_index([row_frame(r) for _i, r in picked])
    index, line = picked[k]
    return line, index


def static_medoid_line(lines: Sequence[dict], meta: dict,
                       prefer: Optional[str]) -> Tuple[dict, int, str]:
    """Set A: the summary frame the recorder chose, found again in the take.

    The meta stores `medoid_wall_time`; the line of the operator's label
    with that wall_time is the one. When the meta has no usable time the
    static interval and its medoid are recomputed with the recorder's own
    functions.
    """
    from leap_hand.static_interval import (DEFAULT_STATIC_S, medoid_in_window,
                                           operator_label, static_interval)
    side = operator_label(lines, prefer)
    t = meta.get("medoid_wall_time")
    if t is not None:
        for i, line in enumerate(lines):
            if str(line.get("hand_side")) == side and \
                    abs(float(line.get("wall_time", -1e18)) - float(t)) < 1e-6:
                return line, i, "the recorder's static-interval medoid (meta)"
    interval = meta.get("static_interval")
    if not (isinstance(interval, list) and len(interval) == 2
            and None not in interval):
        interval = static_interval(lines, DEFAULT_STATIC_S, side)
    if interval is None:
        raise ValueError("no tracked line in this take")
    found = medoid_in_window(lines, float(interval[0]), float(interval[1]),
                             side)
    if found is None:
        raise ValueError("no tracked line inside the static interval")
    line, index = found
    return line, index, "static-interval medoid (recomputed)"


# --- the drawing (shared by the PNG and the PDF) ------------------------------------
CANVAS_W, CANVAS_H = 1200, 680


def _fit(points: np.ndarray, box: Tuple[float, float, float, float]):
    """Uniform scale + offset that fits 2-D points (y up) into a pixel box."""
    x0, y0, x1, y1 = box
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    s = min((x1 - x0) / span[0], (y1 - y0) / span[1]) * 0.9
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    mid = (lo + hi) / 2.0

    def to_px(p):
        return (cx + s * (p[0] - mid[0]), cy - s * (p[1] - mid[1]))
    return to_px


def _view_points(rows: Sequence[dict], view: str):
    """2-D (right, up) coordinates of every joint and axis tip, in mm.

    palm: seen from the palm side (the viewer on -y, fingers up): right = -x,
          up = -z. On a left hand the thumb is then on the left, on a right
          hand on the right, as when you look at your own palm.
    side: seen from +x, fingers to the right: right = -z, up = +y (dorsal).
    """
    def project(v):
        if view == "palm":
            return np.array([-v[0], -v[2]])
        return np.array([-v[2], v[1]])
    joints, tips = [], []
    for r in rows:
        p = np.array([r["x_mm"], r["y_mm"], r["z_mm"]])
        joints.append(project(p))
        tips.append([project(p + 10.0 * np.asarray(r[a]))
                     for a in ("axis_x", "axis_y", "axis_z")])
    return np.array(joints), tips


def _dof_lines(dof: Dict[str, float]) -> List[str]:
    out = []
    for letter, finger in jf.FINGERS:
        names = [n for n in jf.DOF24_NAMES if n.startswith(letter + "_")]
        cells = [f"{n[2:]} {jf.fmt_num(dof[n], 1):>6}" for n in names]
        out.append(f"{letter} ({finger.lower()})".ljust(11) + "   ".join(cells))
    return out


def layout(rows: Sequence[dict], dof: Dict[str, float], title: str,
           subtitle: str, hand: str) -> list:
    """The drawing as primitives on a CANVAS_W x CANVAS_H canvas (y down).

    ("line", x1, y1, x2, y2, rgb, width), ("dot", x, y, r, rgb),
    ("text", x, y, text, rgb, size, mono), ("rect", x0, y0, x1, y1, rgb).
    One list, two renderers: `render_png` (OpenCV) and `render_pdf_page`
    (reportlab, vector, no image embedded).
    """
    prims: list = []
    colour = HAND_RGB.get(hand, (80, 80, 80))
    prims.append(("text", 24, 36, title, (0, 0, 0), 22, False))
    prims.append(("text", 24, 62, subtitle, (60, 60, 60), 15, False))
    panels = {"palm": (24, 84, 640, 620), "side": (660, 84, 1176, 400)}
    names = {"palm": "palm side view (looking at the palm, fingers up)",
             "side": "side view from +x (fingers right, back of the hand up)"}
    for view, box in panels.items():
        prims.append(("rect",) + box + ((200, 200, 200),))
        prims.append(("text", box[0] + 8, box[1] + 20, names[view],
                      (90, 90, 90), 13, False))
        pts, tips = _view_points(rows, view)
        allpts = np.vstack([pts] + [np.array(t) for t in tips])
        inner = (box[0] + 20, box[1] + 34, box[2] - 20, box[3] - 14)
        to_px = _fit(allpts, inner)
        px = [to_px(p) for p in pts]
        for a, b in BONES:
            prims.append(("line", *px[a], *px[b], colour, 3))
        for i, r in enumerate(rows):
            for k in range(3):
                prims.append(("line", *px[i], *to_px(tips[i][k]),
                              AXIS_RGB[k], 2))
            prims.append(("dot", *px[i], 3.5, (30, 30, 30)))
        if view == "palm":
            for i, r in enumerate(rows):
                label = jf.PAPER_JOINT.get(r["joint"])
                if label:
                    prims.append(("text", px[i][0] + 7, px[i][1] - 5, label,
                                  (0, 0, 0), 12, False))
    # legend: the 24 angles, one digit per line, one angle per column
    x0, y0 = 660, 430
    prims.append(("text", x0, y0, "The paper's 24 angles (degrees)",
                  (0, 0, 0), 16, False))
    for k, (letter, finger) in enumerate(jf.FINGERS):
        yy = y0 + 30 + 24 * k
        prims.append(("text", x0, yy, f"{letter} {finger.lower()}",
                      (0, 0, 0), 13, False))
        names = [n for n in jf.DOF24_NAMES if n.startswith(letter + "_")]
        for m, n in enumerate(names):
            prims.append(("text", x0 + 86 + 86 * m, yy,
                          f"{n[2:]} {jf.fmt_num(dof[n], 1)}", (0, 0, 0), 13,
                          False))
    y = y0 + 30 + 24 * 5 + 16
    for k, (name, rgb) in enumerate(zip("xyz", AXIS_RGB)):
        prims.append(("line", x0 + 150 * k, y, x0 + 150 * k + 30, y, rgb, 3))
        prims.append(("text", x0 + 150 * k + 38, y + 5, f"{name} axis, 10 mm",
                      (0, 0, 0), 13, False))
    for k, text in enumerate(CONVENTION_LINES):
        prims.append(("text", x0, y + 36 + 20 * k, text, (60, 60, 60), 10,
                      False))
    prims.append(("text", 24, 650,
                  "Labels: TMC, MCP, IP (thumb); CMC, MCP, PIP, DIP (fingers), "
                  "at the joint whose angles they name.", (60, 60, 60), 13,
                  False))
    return prims


def render_png(prims: list, path: Path) -> None:
    import cv2
    img = np.full((CANVAS_H, CANVAS_W, 3), 255, np.uint8)
    for p in prims:
        kind = p[0]
        if kind == "line":
            _k, x1, y1, x2, y2, rgb, w = p
            cv2.line(img, (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))), rgb[::-1], int(w),
                     cv2.LINE_AA)
        elif kind == "dot":
            _k, x, y, r, rgb = p
            cv2.circle(img, (int(round(x)), int(round(y))), int(round(r)),
                       rgb[::-1], -1, cv2.LINE_AA)
        elif kind == "rect":
            _k, x0, y0, x1, y1, rgb = p
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                          rgb[::-1], 1)
        elif kind == "text":
            _k, x, y, text, rgb, size, mono = p
            font = cv2.FONT_HERSHEY_PLAIN if mono else cv2.FONT_HERSHEY_SIMPLEX
            scale = size / (15.0 if mono else 30.0)
            cv2.putText(img, text, (int(x), int(y)), font, scale, rgb[::-1],
                        1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")


TABLE_COLUMNS = [("joint", "joint", 0), ("x mm", "x_mm", 1),
                 ("y mm", "y_mm", 1), ("z mm", "z_mm", 1),
                 ("bone mm", "bone_len_mm", 1), ("flex", "flex_deg", 1),
                 ("abd", "abd_deg", 1), ("twist", "twist_deg", 1),
                 ("qx", "qx", 3), ("qy", "qy", 3), ("qz", "qz", 3),
                 ("qw", "qw", 3)]


def render_pdf_page(c, prims: list, rows: Sequence[dict]) -> None:
    """One portrait A4 page: the drawing on top, the 26-row table below."""
    from reportlab.lib.pagesizes import A4
    page_w, page_h = A4
    margin = 28.0
    k = (page_w - 2 * margin) / CANVAS_W
    top = page_h - margin

    def X(x):
        return margin + x * k

    def Y(y):
        return top - y * k

    for p in prims:
        kind = p[0]
        if kind == "line":
            _k, x1, y1, x2, y2, rgb, w = p
            c.setStrokeColorRGB(*(v / 255.0 for v in rgb))
            c.setLineWidth(max(0.3, w * k))
            c.line(X(x1), Y(y1), X(x2), Y(y2))
        elif kind == "dot":
            _k, x, y, r, rgb = p
            c.setFillColorRGB(*(v / 255.0 for v in rgb))
            c.circle(X(x), Y(y), r * k, stroke=0, fill=1)
        elif kind == "rect":
            _k, x0, y0, x1, y1, rgb = p
            c.setStrokeColorRGB(*(v / 255.0 for v in rgb))
            c.setLineWidth(0.4)
            c.rect(X(x0), Y(y1), (x1 - x0) * k, (y1 - y0) * k, stroke=1,
                   fill=0)
        elif kind == "text":
            _k, x, y, text, rgb, size, mono = p
            c.setFillColorRGB(*(v / 255.0 for v in rgb))
            c.setFont("Courier" if mono else "Helvetica",
                      max(4.0, size * k * (1.0 if mono else 1.05)))
            c.drawString(X(x), Y(y), text)

    # the table
    y = Y(CANVAS_H) - 18
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(margin, y, "The 26 joints in the wrist frame (mm, degrees; "
                 "orientation as a quaternion in the wrist frame)")
    y -= 14
    widths = [98] + [38] * 7 + [38] * 4
    xs = [margin]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    c.setFont("Courier-Bold", 7.2)
    for (head, _key, _nd), x, w in zip(TABLE_COLUMNS, xs, widths):
        if _key == "joint":
            c.drawString(x, y, head)
        else:
            c.drawRightString(x + w - 4, y, head)
    c.setFont("Courier", 7.2)
    for r in rows:
        y -= 10.2
        for (_head, key, nd), x, w in zip(TABLE_COLUMNS, xs, widths):
            if key == "joint":
                c.drawString(x, y, r["joint"])
            else:
                c.drawRightString(x + w - 4, y, jf.fmt_num(float(r[key]), nd))
    c.setFont("Helvetica", 6.8)
    y -= 14
    c.drawString(margin, y, "Every row, with the joint's three unit axes, is "
                 "in the CSV of the same name; conventions in "
                 "docs/joint_frames.md.")


def write_pdf(pages: Sequence[dict], path: Path, title: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(title)
    c.setAuthor("scripts/joint_frames_view.py")
    for page in pages:
        render_pdf_page(c, page["prims"], page["rows"])
        c.showPage()
    c.save()


# --- one take -----------------------------------------------------------------------
def write_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(jf.CSV_COLUMNS)
        w.writerows(jf.csv_rows(rows))


def summarise(line: dict, source: str, reference: str, title: str,
              how: str) -> dict:
    """The table, the 24 angles and the drawing of one summary frame."""
    hand = str(line.get("hand_side", ""))
    rows = jf.joint_table(line, source, reference)
    dof = jf.dof24(rows)
    bend = jf.wrist_palm_deg(line, source)
    ref_text = ("wrist frame (the wrist joint's axes)" if reference == "wrist"
                else "palm frame (the palm joint's axes)")
    subtitle = (f"{source}, {hand} hand label, {how}; {ref_text}; "
                f"forearm to palm {bend:.0f} deg" if source == "camera" else
                f"{source}, {hand} hand, {how}; {ref_text}")
    prims = layout(rows, dof, title, subtitle, hand)
    return {"rows": rows, "dof": dof, "prims": prims, "hand": hand,
            "source": source, "title": title, "subtitle": subtitle,
            "wrist_palm_deg": bend}


def export_take(path: Path, out_dir: Optional[Path] = None,
                frame: str = "medoid", source: Optional[str] = None,
                reference: str = "wrist", stem: Optional[str] = None,
                line: Optional[dict] = None, how: Optional[str] = None,
                label: Optional[str] = None, prefer: Optional[str] = None,
                title: Optional[str] = None, png: bool = True) -> dict:
    """Write <stem>.csv and <stem>.png for one take's summary frame.

    `line` given: that line is the summary frame (the session mode passes
    the one it chose). Otherwise `frame` is "medoid" or a line number.
    """
    path = Path(path)
    lines = read_lines(path)
    if not lines:
        raise ValueError(f"{path} holds no frame lines")
    source = source or jf.detect_source(lines[0])
    index = None
    if line is None:
        if str(frame) == "medoid":
            line, index = medoid_line(lines, label, prefer)
            how = how or f"medoid of the whole take (line {index})"
        else:
            index = int(frame)
            if not 0 <= index < len(lines):
                raise ValueError(f"--frame {index}: the take has lines 0 to "
                                 f"{len(lines) - 1}")
            line = lines[index]
            how = how or f"line {index}"
    stem = stem or path.stem
    out_dir = Path(out_dir) if out_dir is not None else path.parent
    info = summarise(line, source, reference, title or stem,
                     how or "chosen frame")
    csv_path = out_dir / f"{stem}.csv"
    write_csv(info["rows"], csv_path)
    info.update({"stem": stem, "csv": csv_path, "take": path,
                 "index": index, "wall_time": line.get("wall_time")})
    if png:
        png_path = out_dir / f"{stem}.png"
        render_png(info["prims"], png_path)
        info["png"] = png_path
    return info


def print_dof(info: dict) -> None:
    print(f"{info['stem']}: {info['subtitle']}")
    for text in _dof_lines(info["dof"]):
        print("  " + text)


def run_take(args) -> int:
    try:
        info = export_take(Path(args.take), args.out, args.frame, args.source,
                           args.frame_ref, label=args.hand)
    except (OSError, ValueError) as e:
        print(f"could not export {args.take}: {e}")
        return 1
    print_dof(info)
    print(f"wrote {info['csv']}")
    print(f"wrote {info['png']}")
    return 0


# --- a session ----------------------------------------------------------------------
def export_session(session_dir: Path, reference: str = "wrist") -> dict:
    """Every accepted take's summary frame into <session>/joint_frames/."""
    from cam_hand import protocol_check as pc
    session = pc.load_session(session_dir)
    out = Path(session_dir) / OUT_SUBDIR
    mock = pc.mock_flags(session.meta)
    pages, skipped = [], []
    for t in session.takes:
        if not t.accepted:
            continue
        head = f"{session.set} {t.name}" + ("  (MOCK session)" if mock else "")
        try:
            if session.set == pc.GRASPS:
                leap = t.path("leap")
                if leap is None:
                    skipped.append(f"{t.name}: no camera file")
                    continue
                lines = read_lines(leap)
                meta = pc.read_json(t.path("meta")) if t.path("meta") else None
                line, _i, how = static_medoid_line(
                    lines, meta if isinstance(meta, dict) else {},
                    session.hand)
                pages.append(export_take(leap, out, source="camera",
                                         reference=reference, stem=t.name,
                                         line=line, how=how, title=head))
                continue
            glove = t.path("glove")
            if glove is not None:
                lines = read_lines(glove)
                sides = {str(d.get("hand_side")) for d in lines}
                label = session.hand if session.hand in sides else None
                pages.append(export_take(
                    glove, out, source="glove", reference=reference,
                    stem=t.name, label=label, prefer=session.hand,
                    how="medoid of the whole take", title=head + ", glove"))
            else:
                skipped.append(f"{t.name}: no glove file")
            leap = t.path("leap")
            if leap is not None:
                lines = read_lines(leap)
                counts: Dict[str, int] = {}
                for d in lines:
                    s = str(d.get("hand_side"))
                    counts[s] = counts.get(s, 0) + 1
                label, _note = pc.choose_camera_label(counts, session.hand)
                pages.append(export_take(
                    leap, out, source="camera", reference=reference,
                    stem=f"camera_{t.name}", label=label,
                    how="medoid of the whole take", title=head + ", camera"))
        except (OSError, ValueError) as e:
            skipped.append(f"{t.name}: {e}")
    pdf = None
    if pages:
        pdf = out / PDF_NAME
        write_pdf(pages, pdf, f"Joint frames, {session.set} {session.name}")
    return {"session": session, "pages": pages, "skipped": skipped,
            "pdf": pdf, "out": out}


def run_session(args) -> int:
    try:
        result = export_session(Path(args.session), args.frame_ref)
    except FileNotFoundError as e:
        print(f"not a session folder: {e}")
        return 1
    for info in result["pages"]:
        print_dof(info)
    for s in result["skipped"]:
        print(f"skipped {s}")
    if result["pdf"] is None:
        print("no accepted take could be exported; nothing written")
        return 1
    print(f"wrote {len(result['pages'])} CSV and PNG pair(s) and "
          f"{result['pdf']}")
    return 0


# --- live ---------------------------------------------------------------------------
SCALE = 2                         # 384 px sensor image -> 768 px window (camera_view.py)
IMAGE_SIZE = 384 * SCALE
BASELINE_HALF_MM = 32.0           # camera_view.py: the left lens sits at x = +32 mm
LEFT_CAMERA = 1                   # eLeapPerspectiveType_stereo_left
DEFAULT_WRIST_M = (0.0, 0.25, 0.05)   # the glove hand with no camera hand: above the module
PANEL_W = 600


class Projector:
    """Tracking-space millimetres -> pixel in the unmirrored, upscaled image.

    The same call as `to_px` in scripts/leap/camera_view.py (checked there
    against saved IR stills): the rectilinear ray (-(x - 32) / y, z / y)
    through LeapRectilinearToPixel on the left camera. With no LeapC
    connection (the mock) a plain pinhole stands in, so the drawing code is
    the same.
    """

    def __init__(self, connection=None):
        self.ptr = None
        if connection is not None:
            try:
                from leapc_cffi import ffi, libleapc
                self.ffi, self.lib = ffi, libleapc
                self.ptr = connection.get_connection_ptr()
            except Exception:
                self.ptr = None

    def __call__(self, p_mm):
        x, y, z = (float(v) for v in p_mm)
        if y <= 5.0:
            return None
        rx, ry = -(x - BASELINE_HALF_MM) / y, z / y
        if self.ptr is None:
            u, v = 192.0 + 260.0 * rx, 192.0 + 260.0 * ry
        else:
            vec = self.ffi.new("LEAP_VECTOR*")
            vec.x, vec.y, vec.z = rx, ry, 1.0
            r = self.lib.LeapRectilinearToPixel(self.ptr, LEFT_CAMERA, vec[0])
            u, v = r.x, r.y
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        return int(u * SCALE), int(v * SCALE)


def put(img, text, org, scale=0.45, bgr=(235, 235, 235), thick=1):
    """Text over the IR picture: cam_hand.draw's drift-free dark outline."""
    from cam_hand.draw import draw_text
    draw_text(img, text, org, scale, bgr, thick)


def panel_text(img, text, org, scale=0.4, bgr=(235, 235, 235), right=False):
    """Text on the dark panel, left- or right-aligned at `org`."""
    import cv2
    x, y = int(org[0]), int(org[1])
    if right:
        x -= cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, bgr, 1,
                cv2.LINE_AA)


# The panel's table: (heading, row key, right edge of the column in px).
PANEL_COLUMNS = [("x mm", "x_mm", 262), ("y mm", "y_mm", 322),
                 ("z mm", "z_mm", 382), ("flex", "flex_deg", 452),
                 ("abd", "abd_deg", 522)]


class LiveView:
    """State and drawing of the live window; `compose()` is one frame."""

    def __init__(self, reference: str = "wrist", prefer: str = "both",
                 glove_on: bool = False):
        self.reference = reference
        self.prefer = prefer
        self.glove_on = glove_on
        self.source = "camera"
        self.finger = 0                     # index into FINGER_CYCLE
        self.cam: Dict[str, Tuple[float, object]] = {}     # side -> (t, LeapHand)
        self.glove: Dict[str, Tuple[float, object]] = {}   # side -> (t, HandFrame)
        self.image = None
        self.drawn = 0
        self.fps = 0.0

    # --- inputs
    def add_camera(self, lh) -> None:
        if getattr(lh, "abs26", None) and getattr(lh, "quat26", None):
            self.cam[lh.hand_side] = (time.time(), lh)
            self.fps = float(getattr(lh, "framerate", 0.0) or 0.0)

    def add_glove(self, frame) -> None:
        self.glove[frame.hand_side] = (time.time(), frame)

    def fresh(self, store: dict, max_age: float) -> dict:
        now = time.time()
        return {s: v for s, (t, v) in store.items() if now - t < max_age}

    def camera_hand(self):
        hands = self.fresh(self.cam, 0.25)
        if not hands:
            return None
        if self.prefer in hands:
            return hands[self.prefer]
        return max(hands.values(), key=lambda lh: lh.visible_time_us)

    def glove_frame(self, side: Optional[str]):
        frames = self.fresh(self.glove, 1.0)
        if not frames:
            return None
        if side in frames:
            return frames[side]
        return next(iter(frames.values()))

    def keys(self, key: int) -> bool:
        """Apply one key; False means quit."""
        if key in (27, ord("q")):
            return False
        if key == ord("g"):
            self.source = "glove" if self.source == "camera" else "camera"
        elif key == ord("n"):
            self.finger = (self.finger + 1) % len(FINGER_CYCLE)
        return True

    # --- drawing
    def _hand_world(self):
        """(camera LeapHand or None, camera rows or None, glove rows or None,
        glove world pose (pos, axes) or None)."""
        lh = self.camera_hand()
        cam_rows = None
        if lh is not None:
            cam_rows = jf.joint_table(jf.line_from_leap_hand(lh), "camera",
                                      self.reference)
        glove_rows = glove_world = None
        if self.glove_on:
            frame = self.glove_frame(lh.hand_side if lh is not None else None)
            if frame is not None:
                glove_rows = jf.joint_table(jf.line_from_hand_frame(frame),
                                            "glove", self.reference)
                if lh is not None:
                    pos, rots, _ = jf.world_pose(jf.line_from_leap_hand(lh),
                                                 "camera")
                    ref = jf.PALM if self.reference == "palm" else jf.WRIST
                    origin, rot = pos[jf.WRIST], rots[ref]
                else:
                    origin, rot = np.array(DEFAULT_WRIST_M), np.eye(3)
                glove_world = jf.to_world(glove_rows, origin, rot)
        return lh, cam_rows, glove_rows, glove_world

    def compose(self, project) -> np.ndarray:
        import cv2
        lh, cam_rows, glove_rows, glove_world = self._hand_world()
        size = IMAGE_SIZE
        if self.image is None:
            frame = np.zeros((size, size, 3), np.uint8)
        else:
            img = cv2.resize(self.image, (size, size),
                             interpolation=cv2.INTER_LINEAR)
            img = cv2.convertScaleAbs(img, alpha=2.2, beta=8)
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        highlight = FINGER_CYCLE[self.finger]
        lit = set(jf.finger_joints(highlight)) if highlight else set()
        show = self.source
        if show == "glove" and glove_rows is None:
            show = "camera"
        # both skeletons; axes on the one the table shows
        drawn = []
        if lh is not None:
            pos = np.asarray(lh.abs26, dtype=float)
            _p, rots, _h = jf.world_pose(jf.line_from_leap_hand(lh), "camera")
            axes = np.array(rots)
            drawn.append(("camera", lh.hand_side, pos, axes))
        if glove_world is not None:
            side = lh.hand_side if lh is not None else next(iter(
                self.fresh(self.glove, 1.0)), "left")
            drawn.append(("glove", side, glove_world[0], glove_world[1]))
        for src, side, pos, axes in drawn:
            col = LIVE_HAND_BGR.get(side, (200, 200, 200))
            if src != show:
                col = tuple(int(v * 0.45) for v in col)
            px = [project(p * 1000.0) for p in pos]
            for a, b in BONES:
                if px[a] and px[b]:
                    cv2.line(frame, px[a], px[b], col, 2 if src == show else 1,
                             cv2.LINE_AA)
            if src != show:
                continue
            for i, p in enumerate(pos):
                if px[i] is None:
                    continue
                for k in range(3):
                    tip = project((p + AXIS_LEN_M * axes[i][:, k]) * 1000.0)
                    if tip:
                        bgr = AXIS_RGB[k][::-1]
                        cv2.line(frame, px[i], tip, bgr,
                                 3 if i in lit else 1, cv2.LINE_AA)
                cv2.circle(frame, px[i], 4 if i in lit else 2,
                           HIGHLIGHT_BGR if i in lit else (255, 255, 255), -1,
                           cv2.LINE_AA)
        frame = cv2.flip(frame, 1)          # mirror, as camera_view.py
        put(frame, f"table: {show} hand" + (
            "" if show == self.source else " (no glove frame yet)"),
            (12, 24), 0.6, (0, 255, 255), 2)
        put(frame, f"tracking {self.fps:4.1f} Hz   g glove/camera   n finger"
                   "   s save PNG   q quit", (12, size - 14), 0.45)
        rows = cam_rows if show == "camera" else glove_rows
        panel = self.panel(rows, show, lh, highlight)
        self.drawn += 1
        return np.hstack([frame, panel])

    def panel(self, rows, show, lh, highlight) -> np.ndarray:
        panel = np.full((IMAGE_SIZE, PANEL_W, 3), 24, np.uint8)
        grey, white = (175, 175, 175), (235, 235, 235)
        y = 22
        panel_text(panel, f"{show.upper()} hand, {self.reference} frame "
                          "(mm, degrees)", (10, y), 0.55, (255, 255, 255))
        y += 26
        if rows is None:
            panel_text(panel, "no hand tracked" if show == "camera" else
                       "no glove frame", (10, y + 10), 0.6, (60, 60, 255))
            return panel
        panel_text(panel, "joint", (10, y), 0.4, grey)
        for head, _key, right in PANEL_COLUMNS:
            panel_text(panel, head, (right, y), 0.4, grey, right=True)
        for r in rows:
            y += 15
            lit = bool(highlight) and r["joint"].startswith(highlight + "_")
            colour = HIGHLIGHT_BGR if lit else white
            panel_text(panel, r["joint"], (10, y), 0.4, colour)
            for _head, key, right in PANEL_COLUMNS:
                panel_text(panel, jf.fmt_num(r[key], 1), (right, y), 0.4,
                           colour, right=True)
        y += 28
        panel_text(panel, "the paper's 24 angles (degrees)", (10, y), 0.5,
                   (255, 255, 255))
        dof = jf.dof24(rows)
        for letter, finger in jf.FINGERS:
            y += 18
            lit = highlight == finger
            colour = HIGHLIGHT_BGR if lit else white
            panel_text(panel, f"{letter} {finger.lower()}", (10, y), 0.4,
                       colour)
            names = [n for n in jf.DOF24_NAMES if n.startswith(letter + "_")]
            for m, n in enumerate(names):
                panel_text(panel, f"{n[2:]} {jf.fmt_num(dof[n], 1)}",
                           (92 + 100 * m, y), 0.4, colour)
        y += 26
        if show == "camera" and lh is not None:
            bend = jf.wrist_palm_deg(jf.line_from_leap_hand(lh), "camera")
            panel_text(panel, f"hand label {lh.hand_side} (the tracker's), "
                              f"forearm to palm {bend:.0f} deg", (10, y),
                       0.4, grey)
            y += 17
        panel_text(panel, "flex + toward the palm, abd + toward the thumb",
                   (10, y), 0.4, grey)
        panel_text(panel, "axes: x red, y green, z blue, 1 cm", (10, y + 17),
                   0.4, grey)
        return panel


def run_live(args) -> int:
    from leap_hand.images import open_sampler
    from leap_hand.stream import LeapUnavailable, open_stream
    from xr_hand.parser import parse_hand_message

    try:
        stream = open_stream(mock=args.mock)
    except LeapUnavailable as e:
        print(e)
        return 1
    sampler = None
    try:
        sampler = open_sampler(stream, mock=args.mock)
    except Exception as e:                 # the picture is optional
        print(f"no IR image (the axes are still drawn): {e}")
    project = Projector(None if args.mock else stream.connection)
    glove = None
    if args.glove:
        try:
            if args.mock:
                from leap_hand.live import MockGlove
                glove = MockGlove(args.hand if args.hand != "both" else "left")
            else:
                from xr_hand.receiver import OSCHandReceiver
                glove = OSCHandReceiver()
            glove.start()
        except OSError as e:
            print(f"the glove's OSC port could not be opened ({e}); is a glove "
                  "recorder running? Showing the camera only.")
            glove = None
    view = LiveView(args.frame_ref, args.hand, glove is not None)
    win = "Joint frames"
    if not args.no_window:
        import cv2
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, IMAGE_SIZE + PANEL_W, IMAGE_SIZE)
    t0 = time.time()
    try:
        while True:
            for _side, lh in stream.drain(256):
                view.add_camera(lh)
            if glove is not None:
                for item in glove.drain(256):
                    hand, raw = item
                    try:
                        view.add_glove(parse_hand_message(
                            raw, hand_side_hint=hand))
                    except Exception:
                        continue
            if sampler is not None:
                pair = sampler.latest()
                if pair is not None:
                    view.image = pair.left
            shown = view.compose(project)
            if not args.no_window:
                import cv2
                cv2.imshow(win, shown)
                key = cv2.waitKey(15) & 0xFF
                if key == ord("s"):
                    SNAP_DIR.mkdir(parents=True, exist_ok=True)
                    path = SNAP_DIR / time.strftime(
                        "joint_frames_%Y%m%d_%H%M%S.png")
                    cv2.imwrite(str(path), shown)
                    print(f"saved {path}")
                elif key != 255 and not view.keys(key):
                    break
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(0.03)
            if args.seconds and time.time() - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for closer in (getattr(glove, "stop", None),
                       (lambda: sampler.detach(stream.connection))
                       if sampler is not None and not args.mock else None,
                       getattr(stream, "stop", None)):
            if closer is None:
                continue
            try:
                closer()
            except Exception:
                pass
        if not args.no_window:
            import cv2
            cv2.destroyAllWindows()
        print(f"frames drawn: {view.drawn}")
    return 0


# --- main ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Each joint's x y z frame and the paper's 24 angles: "
                    "live on the IR image, for one take, or for a session.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true",
                      help="live window (the default when no other mode is given)")
    mode.add_argument("--take", type=Path, help="a glove or camera JSONL take")
    mode.add_argument("--session", type=Path,
                      help="a protocol session folder (Set A, B or C)")
    p.add_argument("--glove", action="store_true",
                   help="live: also the glove over OSC (g switches the table)")
    p.add_argument("--mock", action="store_true",
                   help="live: synthetic camera (and glove with --glove)")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="live: stop after this long (0 = until q)")
    p.add_argument("--no-window", action="store_true",
                   help="live: no window, for tests; prints frames drawn")
    p.add_argument("--hand", default=None,
                   help="live: left, right or both (default both); take: the "
                        "tracker label to read (default: the label with the "
                        "most frames)")
    p.add_argument("--frame", default="medoid",
                   help="take: 'medoid' (default) or a line number from 0")
    p.add_argument("--source", choices=jf.SOURCES, default=None,
                   help="take: glove or camera (default: from the line format)")
    p.add_argument("--out", type=Path, default=None,
                   help="take: output folder (default: beside the take)")
    p.add_argument("--frame-ref", choices=jf.REFERENCES, default="wrist",
                   help="axes of the reference frame: the wrist joint "
                        "(default) or the palm joint")
    args = p.parse_args(argv)
    if args.take:
        return run_take(args)
    if args.session:
        return run_session(args)
    args.hand = args.hand or "both"
    if args.hand not in ("left", "right", "both"):
        p.error("--hand must be left, right or both")
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
