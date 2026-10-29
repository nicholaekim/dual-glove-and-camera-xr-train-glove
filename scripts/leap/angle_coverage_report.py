"""Which joint angles the grasp set gives, joint by joint: report.html and report.pdf.

The answer to the professor's note on the 2026-10-08 grasp set ("some angles
are not calculated for some joint like TMC and MCP. Also, some angles for
abduction and adduction."), as a paper-style report. Writes into --out:

  angle_coverage.csv   leap_hand.angle_coverage.write_coverage_csv (one row per
                       COVERAGE entry)
  report.html          one self-contained file (CSS inline, figures as base64 PNG)
  report.pdf           printed from it by headless Edge or Chrome

Every count and table row is read at run time from `leap_hand.angle_coverage`
(COVERAGE, STATUS_LABELS, STATUS_MEANINGS) and from section 2 of
docs/angle_coverage.md. Section 4 recomputes the dataset's own angle table:
for each row of metadata/joint_angles_24.csv it loads the take
(`replay_core.find_static`), rebuilds the frame whose frame_id is the take's
`representative_frame_id` in grasps_summary.csv (`replay_core.reconstruct`),
computes `angle_coverage.all_angles` and compares the 18 calculated Paper 3
names and the 8 paper-order MCP columns with the dataset's values; it counts
the blank cells of those 26 and of the six unavailable names. The HTML
helpers and the page CSS are the sibling report's
(scripts/leap/camera_clock_erratum_report.py), loaded from beside this file.
The dataset is only read; an output folder inside it is refused.

  .venv\\Scripts\\python.exe scripts\\leap\\angle_coverage_report.py --dataset "<root>" --out "<folder>"

Prints `COVERAGE REPORT OK: entries=N given=G unavailable=U max_abs_diff_deg=X
pdf=yes|no` (exit 0); with no Edge or Chrome, `PDF NOT MADE: ...` before it and
no PDF. Exit 1: joint_angles_24.csv or one of its columns is missing,
docs/angle_coverage.md has no section 2, or the browser did not print the PDF.
Exit 2: bad arguments or an output folder inside the dataset.
"""
import argparse
import csv
import html
import importlib.util
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402

from leap_hand import angle_coverage as ac  # noqa: E402
from leap_hand import replay_core as rc  # noqa: E402
from xr_hand import joint_frames as jf  # noqa: E402
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DOC = REPO / "docs" / "angle_coverage.md"
_spec = importlib.util.spec_from_file_location("camera_clock_erratum_report",
                                               HERE / "camera_clock_erratum_report.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
brk, inline, th = base.brk, base.inline, base.th

TITLE = "Which joint angles the grasp set gives, joint by joint"
AUTHOR = "Nicholas Kim"
FOOTER = ("Derived from the repository's angle coverage table and the dataset's own angle "
          "table; the dataset itself is unchanged.")
BROWSERS = base.BROWSERS
ANGLES_CSV = Path("metadata") / "joint_angles_24.csv"

# --- what the coverage table says, counted ---------------------------------------------------
PAPER24 = list(jf.DOF24_NAMES)
GIVEN = [e.paper3_name for e in ac.COVERAGE if e.status == "calculated" and e.paper3_name]
SIX = [e.paper3_name for e in ac.COVERAGE if e.status == "unavailable" and e.paper3_name]
OUTSIDE = [e for e in ac.COVERAGE if e.status == "outside_paper3"]
ESTIMATED = [e for e in ac.COVERAGE if e.status == "estimated"]
EVALUATED = [e for e in ac.COVERAGE if e.status == "evaluated"]
# the 26 columns the dataset's angle table shares with the calculated entries
COMPARED = [c for e in ac.COVERAGE if e.status == "calculated" for c in e.output_columns]
assert set(SIX) == set(ac.UNAVAILABLE_COLUMNS) and set(GIVEN) | set(SIX) == set(PAPER24)
assert set(COMPARED) == set(GIVEN) | set(jf.MCP_PAPER_NAMES)

DIGITS = (("thumb", "Thumb"), ("index", "Index finger"), ("middle", "Middle finger"),
          ("ring", "Ring finger"), ("little", "Little finger"),
          ("wrist", "Wrist, and the twist of every bone"))
LETTER_DIGIT = {"T": "thumb", "I": "index", "M": "middle", "R": "ring", "L": "little"}


def digit_of(entry):
    word = entry.joint.split()[0]
    return word if word in LETTER_DIGIT.values() else "wrist"


def by_digit(entries):
    """[(digit title, [entries])] in DIGITS order, entries in COVERAGE order."""
    return [(title, [e for e in entries if digit_of(e) == key]) for key, title in DIGITS
            if any(digit_of(e) == key for e in entries)]


def movement_kind(entry):
    """'fe', 'aa' or None (the twist): the column of the status grid."""
    if entry.movement.startswith("flexion"):
        return "fe"
    if entry.movement.startswith("abduction") or "deviation" in entry.movement:
        return "aa"
    return None


def compact_columns(cols):
    """Column names for a table cell: many names with one suffix are written
    `<JOINT>suffix` plus the list of JOINTs."""
    if len(cols) <= 4:
        return ", ".join(inline(f"`{c}`") for c in cols)
    suffix = os.path.commonprefix([c[::-1] for c in cols])[::-1]
    suffix = suffix[suffix.index("_"):] if "_" in suffix else ""
    if not suffix:
        return ", ".join(inline(f"`{c}`") for c in cols)
    heads = [c[:-len(suffix)] for c in cols]
    return (inline(f"`<JOINT>{suffix}`") + f", one per joint ({len(cols)}): "
            + ", ".join(brk(h) for h in heads))


def code(name):
    return inline(f"`{name}`")


def names(seq):
    seq = [code(s) for s in seq]
    return seq[0] if len(seq) == 1 else ", ".join(seq[:-1]) + " and " + seq[-1]


def fmt(x, digits=4):
    return "" if x is None else f"{round(x, digits) + 0.0:.{digits}f}"


# --- the dataset's angle table, recomputed ----------------------------------------------------
class DatasetError(Exception):
    pass


def agreement(root):
    """Recompute every row of joint_angles_24.csv at its representative frame.

    Returns a dict: rows (the table's row count), compared (takes recomputed),
    failed [(grasp, take, why)], diffs {column: [|dataset - recomputed|]},
    blanks {column: blank cells} for COMPARED + SIX, decimals (the most
    decimals any compared cell is written with), palm {metacarpal joint: the
    largest angle, over the compared frames of one hand, between its
    orientation in the PALM frame and that hand's first compared frame's}."""
    path = Path(root) / ANGLES_CSV
    if not path.is_file():
        raise DatasetError(f"{path} does not exist")
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ["grasp_name", "take"] + COMPARED + SIX
                   if c not in (reader.fieldnames or [])]
        rows = list(reader)
    if missing:
        raise DatasetError(f"{path} has no column {', '.join(missing)}")
    meta = {(r["grasp_name"], int(r["take"])): r for r in rc.list_static(root)}
    diffs = {c: [] for c in COMPARED}
    blanks = {c: sum(1 for r in rows if not r[c].strip()) for c in COMPARED + SIX}
    decimals = max([len(r[c].strip().partition(".")[2]) for r in rows for c in COMPARED
                    if r[c].strip()] or [0])
    metacarpals = [n for n in JOINT_NAMES if n.endswith("_METACARPAL")]
    first, palm = {}, {n: 0.0 for n in metacarpals}
    failed, compared = [], 0
    for r in rows:
        name, take = r["grasp_name"], int(r["take"])
        m = meta.get((name, take))
        if m is None:
            failed.append((name, take, "no row in metadata/grasps_summary.csv"))
            continue
        rid = (m.get("representative_frame_id") or "").strip()
        if not rid:
            failed.append((name, take, "no representative_frame_id in grasps_summary.csv"))
            continue
        try:
            rec = rc.find_static(root, name, take)
        except (OSError, ValueError, LookupError, KeyError) as e:
            failed.append((name, take, f"could not be loaded ({e})"))
            continue
        hits = [f for f in rec.frames if f.frame_id == int(float(rid))]
        if not hits:
            failed.append((name, take, f"frame {rid} is not among the selected frames"))
            continue
        pose = rc.reconstruct(hits[0])
        got = ac.all_angles(pose.pos, pose.rot, rec.hand_side)
        compared += 1
        for c in COMPARED:
            if r[c].strip():
                diffs[c].append(abs(float(r[c]) - got[c]))
        rel = {n: pose.rot[JOINT_INDEX["PALM"]].T @ pose.rot[JOINT_INDEX[n]] for n in metacarpals}
        ref = first.setdefault(rec.hand_side, rel)
        for n in metacarpals:
            palm[n] = max(palm[n], float(rc.rotation_angle_deg(ref[n], rel[n])))
    return {"rows": len(rows), "compared": compared, "failed": failed, "diffs": diffs,
            "blanks": blanks, "decimals": decimals, "palm": palm}


def max_of(values):
    return max(values) if values else None


# --- figures ---------------------------------------------------------------------------------
FIG_W = base.FIG_W
INK, INK2, MUTED = base.INK, base.INK2, base.MUTED
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"    # categorical slots 1, 3, 2
NONE_FILL, NONE_BAR = "#f0efec", "#898781"              # neutral: nothing measured
STATUS_COLOR = {"calculated": BLUE, "outside_paper3": AQUA, "estimated": ORANGE,
                "unavailable": NONE_BAR}
GRID_STATUSES = ("calculated", "outside_paper3", "estimated", "unavailable")


def tint(hex_color, a=0.2):
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(1.0 - a * (1.0 - rgb))


def fill_of(status):
    return NONE_FILL if status == "unavailable" else tint(STATUS_COLOR[status])


def _arrow(ax, start, end, color=INK, lw=0.9):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=8,
                                shrinkA=0, shrinkB=0))


def _out_of_page(ax, xy, r=0.11):
    ax.add_patch(Circle(xy, r, fill=False, ec=INK, lw=0.9, zorder=5))
    ax.add_patch(Circle(xy, r * 0.28, color=INK, zorder=6))


def fig_conventions():
    """F1: one joint in its parent bone's frame. (a) side view, x (hinge) out of
    the page: flexion toward the palm. (b) back of a right hand, y (dorsal)
    out of the page, x toward the little finger: abduction toward the thumb.
    In both, z runs along the parent bone toward the wrist (to the left) and
    the child bone's zero is the parent's line continued (dashed)."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.95), layout="constrained")
    bone, child_len, axis_len, r = 2.3, 2.2, 1.2, 1.45
    for ax, panel in zip(axes, ("fe", "aa")):
        fe = panel == "fe"
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 3.3)
        ax.set_ylim(-2.0, 2.0)
        ax.axis("off")
        ax.plot([-bone, 0], [0, 0], color="#c9c7bf", lw=7, solid_capstyle="round", zorder=1)
        ax.text(-bone, 0.2, "parent bone", ha="left", va="bottom", fontsize=6.5, color=INK2)
        # the zero, labelled on the side the child bone does not turn to
        ax.plot([0, 2.9], [0, 0], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(2.9, 0.1 if fe else -0.1, "zero: in line\nwith the parent", ha="right",
                va="bottom" if fe else "top", fontsize=6.3, color=INK2)
        ang = -35.0 if fe else 25.0
        tip = (child_len * np.cos(np.radians(ang)), child_len * np.sin(np.radians(ang)))
        ax.plot([0, tip[0]], [0, tip[1]], color=BLUE, lw=4.5, solid_capstyle="round", zorder=2)
        ax.text(tip[0] + 0.1, tip[1] + (-0.05 if fe else 0.05), "child bone", ha="left",
                va="top" if fe else "bottom", fontsize=6.5, color=INK)
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=min(ang, 0.0), theta2=max(ang, 0.0),
                         color=INK, lw=0.9, zorder=3))
        end, before = np.radians(ang), np.radians(ang * 0.8)
        _arrow(ax, (r * np.cos(before), r * np.sin(before)), (r * np.cos(end), r * np.sin(end)))
        mid = np.radians(ang / 2)
        ax.text(1.62 * np.cos(mid), 1.62 * np.sin(mid),
                "+ flexion\ntoward the palm" if fe else "+ abduction\ntoward the thumb",
                ha="left", va="center", fontsize=6.8, color=INK, fontweight="bold")
        # the parent bone's frame at the joint: z drawn along the bone, the
        # axis in the page drawn as an arrow, the axis out of the page as a dot
        _arrow(ax, (0, 0), (-axis_len, 0))
        ax.text(-axis_len, -0.16, "z", ha="center", va="top", fontsize=7.5, color=INK,
                fontweight="bold")
        _arrow(ax, (0, 0), (0, axis_len if fe else -axis_len))
        ax.text(0.12, axis_len if fe else -axis_len, "y" if fe else "x", ha="left",
                va="center", fontsize=7.5, color=INK, fontweight="bold")
        _out_of_page(ax, (0, 0))
        ax.text(-0.18, 0.16, "x" if fe else "y", ha="right", va="bottom", fontsize=7.5,
                color=INK, fontweight="bold")
        top, bottom = (("back of the hand", "palm") if fe else
                       ("thumb side", "little-finger side"))
        ax.text(0.35, 2.0, top, ha="center", va="top", fontsize=6.5, color=MUTED,
                style="italic")
        ax.text(0.35, -2.0, bottom, ha="center", va="bottom", fontsize=6.5, color=MUTED,
                style="italic")
        ax.set_title("(a) flexion/extension, seen from the side" if fe else
                     "(b) abduction/adduction, seen from the back of a right hand",
                     fontsize=7.5, color=INK, loc="left")
    fig.supxlabel("Parent bone frame: x the hinge axis, y dorsal (the back of the hand), z along "
                  "the parent bone toward the wrist.\nA dot in a circle is an axis pointing out of "
                  "the page.", fontsize=6.6, color=INK2)
    return base.png(fig)


def grid_rows():
    """[(digit, joint, {'fe': entries, 'aa': entries})] in COVERAGE order, the
    twist entry left out."""
    rows = []
    for e in ac.COVERAGE:
        kind = movement_kind(e)
        if kind is None:
            continue
        if not rows or rows[-1][1] != e.joint:
            rows.append((digit_of(e), e.joint, {"fe": [], "aa": []}))
        rows[-1][2][kind].append(e)
    return rows


def cell_parts(entries):
    """[(status, second line)] for one grid cell: the entry that is not an
    estimate (Paper 3's coordinate, or the tracker's), then the estimates as
    one part. The reference-evaluated entries are not drawn (caption)."""
    main = [e for e in entries if e.status not in ("estimated", "evaluated")]
    assert len(main) == 1, [e.output_columns for e in entries]
    e = main[0]
    line = e.paper3_name or (", ".join(e.output_columns) if e.output_columns
                             else "not in Paper 3's model")
    parts = [(e.status, line)]
    est = [c for x in entries if x.status == "estimated" for c in x.output_columns]
    if est:
        parts.append(("estimated", os.path.commonprefix(est) + "*" if len(est) > 1 else est[0]))
    return parts


def fig_status_grid():
    """F2: one row per joint, one column per movement, each cell filled and
    worded by its status, with the column name under the word."""
    rows = grid_rows()
    label_w, gap, row_h, group_gap = 1.0, 0.07, 0.31, 0.1
    legend_h, header_h = 0.34, 0.27
    col_w = (FIG_W - label_w - 2 * gap) / 2
    groups = len({d for d, _, _ in rows})
    height = legend_h + header_h + len(rows) * row_h + (groups - 1) * group_gap + 0.04
    fig = plt.figure(figsize=(FIG_W, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(height, 0)
    ax.axis("off")

    def swatch(x, y, w, h, status):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill_of(status), edgecolor="none"))
        ax.add_patch(Rectangle((x, y), 0.05, h, facecolor=STATUS_COLOR[status],
                               edgecolor="none"))

    slot = FIG_W / len(GRID_STATUSES)
    for k, s in enumerate(GRID_STATUSES):
        swatch(k * slot + 0.02, 0.07, 0.3, 0.17, s)
        ax.text(k * slot + 0.4, 0.155, ac.STATUS_LABELS[s], va="center", ha="left",
                fontsize=7.2, color=INK)
    y = legend_h
    for k, text in enumerate(("flexion / extension", "abduction / adduction")):
        x = label_w + gap + k * (col_w + gap)
        ax.text(x + col_w / 2, y + header_h / 2, text, ha="center", va="center",
                fontsize=7.6, color=INK, fontweight="bold")
    y += header_h
    previous = None
    for digit, joint, cells in rows:
        if previous is not None and digit != previous:
            y += group_gap
        previous = digit
        ax.text(label_w - 0.06, y + row_h / 2, joint, ha="right", va="center", fontsize=7.4,
                color=INK)
        for k, kind in enumerate(("fe", "aa")):
            x = label_w + gap + k * (col_w + gap)
            parts = cell_parts(cells[kind])
            widths = [col_w] if len(parts) == 1 else [col_w * 0.5 - 0.02, col_w * 0.5 - 0.02]
            for (status, line), w in zip(parts, widths):
                swatch(x, y + 0.025, w, row_h - 0.05, status)
                ax.text(x + 0.11, y + row_h * 0.35, ac.STATUS_LABELS[status], ha="left",
                        va="center", fontsize=7, color=INK, fontweight="bold")
                ax.text(x + 0.11, y + row_h * 0.7, line, ha="left", va="center",
                        fontsize=6.3, color=INK2, family="DejaVu Sans Mono")
                x += w + 0.04
        y += row_h
    return base.png(fig)


# --- the report ------------------------------------------------------------------------------
CSS_EXTRA = """
table.coverage { font-size: 8pt; line-height: 1.22; break-before: page; page-break-before: always; }
table.coverage th, table.coverage td { padding: 1.2pt 3pt; }
table.coverage tr.grp th { font-size: 8.5pt; text-align: left; padding-top: 5pt;
  border-bottom: 0.4pt solid #000; }
/* headless Edge ended pages up to 110 mm early wherever break-after: avoid (the
   base CSS's headings) or break-inside: avoid on table groups was in play; so no
   break-after: avoid here, and a heading is kept with what follows it by one
   .keep block around both (Report.heading) */
h2, h3, h4 { break-after: auto; page-break-after: auto; }
table.why td { font-size: 8.5pt; }
.entry { break-inside: avoid; page-break-inside: avoid; margin: 0 0 7pt; }
.entry p.head { margin: 0 0 1.5pt; font-size: 9.5pt; }
.entry dl.fields { display: grid; grid-template-columns: 25mm 1fr; column-gap: 5pt;
  row-gap: 0.8pt; margin: 0; padding-bottom: 2pt; border-bottom: 0.5pt solid #999;
  font-size: 8.5pt; line-height: 1.25; }
.entry dl.fields dt { color: #333; padding-left: 3pt; }
.entry dl.fields dd { margin: 0; }
.none { color: #777; font-style: italic; }
dl.req dt { font-weight: bold; margin-top: 4pt; }
dl.req dd { margin: 1pt 0 4pt 14pt; }
"""


class Report(base.Report):
    """base.Report whose headings are held back and added in one .keep block
    with the next part, so a heading never ends a page."""
    held = ""

    def heading(self, s):
        self.held += s

    def add(self, s):
        if self.held:
            s, self.held = f'<div class="keep">{self.held}{s}</div>', ""
        super().add(s)


def doc_section(number):
    """Section `number` of docs/angle_coverage.md, without its heading."""
    text = DOC.read_text(encoding="utf-8")
    m = re.search(rf"^## {number}\. [^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not m:
        raise DatasetError(f"{DOC} has no section {number}")
    return m.group(1).strip()


def coverage_table(rep):
    t = rep.next_table()
    head = ("<thead><tr>" + th("Joint") + th("Movement") + th("Paper 3 name") + th("Status")
            + th("Output column(s)") + th("In the dataset") + "</tr></thead>")
    bodies = []
    for title, entries in by_digit(ac.COVERAGE):
        rows = [f'<tr class="grp"><th colspan="6">{html.escape(title)}</th></tr>']
        for e in entries:
            rows.append(
                '<tr class="entry">'
                f"<td>{html.escape(e.joint)}</td><td>{html.escape(e.movement)}</td>"
                f"<td>{code(e.paper3_name) if e.paper3_name else ''}</td>"
                f"<td>{html.escape(ac.STATUS_LABELS[e.status])}</td>"
                f"<td>{compact_columns(e.output_columns)}</td>"
                f"<td>{'; '.join(code(c.strip()) for c in e.dataset_column.split(';')) if e.dataset_column else ''}</td>"
                "</tr>")
        bodies.append("<tbody>" + "\n".join(rows) + "</tbody>")
    widths = ("19mm", "30mm", "17mm", "24mm", None, "37mm")
    cols = "<colgroup>" + "".join(f'<col style="width:{w}">' if w else "<col>"
                                  for w in widths) + "</colgroup>"
    caption = (f"<strong>Table {t}.</strong> All {len(ac.COVERAGE)} entries of the coverage table "
               f"(<code>leap_hand.angle_coverage.COVERAGE</code>; the same rows are "
               f"<code>angle_coverage.csv</code> beside this report), grouped by digit. Output "
               f"column(s): the name(s) under which the repository's exports give the value; empty "
               f"when nothing is measured. In the dataset: the dataset's column holding the same "
               f"number, either a column of <code>metadata/joint_angles_24.csv</code> or, written "
               f"<code>JOINT.angle</code>, a row and column of the take's joint-frames file. "
               f"Appendix A gives each entry's frame, zero, sign, method, note and requirement.")
    rep.add(f'<table id="coverage" class="coverage"><caption>{caption}</caption>{cols}{head}'
            + "\n".join(bodies) + "</table>")
    return t


def build(args, ag):
    rep = Report()
    n_entries = len(ac.COVERAGE)
    out_cols = [c for e in OUTSIDE for c in e.output_columns]
    all_d = [d for c in COMPARED for d in ag["diffs"][c]]
    worst = max_of(all_d)
    worst_col = max(COMPARED, key=lambda c: max_of(ag["diffs"][c]) or -1.0) if all_d else None
    step = 10.0 ** -ag["decimals"]
    step_txt = f"{step:.{ag['decimals']}f}"
    half_txt = f"{step / 2:.{ag['decimals'] + 1}f}"
    within = worst is not None and worst <= step / 2 + 1e-9
    finger_mcp = [c for c in GIVEN if "_MCP_" in c and not c.startswith("T")]
    mcp_present = all(ag["blanks"][c] == 0 for c in finger_mcp + list(jf.MCP_PAPER_NAMES))

    # --- title and abstract
    rep.add(f'<header class="title"><h1>{html.escape(TITLE)}</h1><p class="author">{AUTHOR}</p>'
            f'<p class="date">Report generated {datetime.now():%Y-%m-%d}</p></header>')
    a = [f"Some angles at the thumb's TMC joint and the fingers' CMC joints, and some "
         f"abduction and adduction angles, are not in the grasp set's tables. This report says, "
         f"joint by joint, which angles are there, which are not, and why.",
         f"Of the {len(PAPER24)} joint angles of Paper 3, the grasp set gives {len(GIVEN)}, "
         f"calculated from the tracker's bone orientations.",
         f"{len(SIX)} are unavailable: {names(SIX)}. Their cells are blank in the dataset, never "
         f"zero.",
         f"The tracker also reports {len(OUTSIDE)} movements that are not in Paper 3's model, "
         f"such as sideways movement at the PIP and DIP joints, the wrist and the twist of each "
         f"bone. They are kept under {len(out_cols)} separate column names ending in "
         f"<code>_tracker</code>."]
    if all_d:
        a.append(f"The finger MCP flexion and abduction angles are calculated and "
                 f"{'present in every row of' if mcp_present else 'in'} "
                 f"<code>metadata/joint_angles_24.csv</code>. Recomputed from the raw recordings "
                 f"at each take's representative frame, all {len(all_d)} values of the "
                 f"{len(COMPARED)} calculated columns agree with that table to within "
                 f"{fmt(worst)} degrees; the table is rounded to {step_txt} degrees.")
    rep.add(f'<section class="abstract"><h2>Abstract</h2><p>{" ".join(a)}</p></section>')

    # --- 1 Background
    rep.heading("<h2>1 Background</h2>")
    rep.add(f"<p><strong>Paper 3's angles.</strong> Paper 3 (Cobos et al. 2009, “Human Hand "
            f"Descriptions and Gesture Recognition for Object Manipulation”) describes the hand "
            f"with {len(PAPER24)} joint angles, listed in its Table 1 (p.4). Each joint is a chain "
            f"link, and each angle turns one bone against the bone before it:</p>")
    by_name = {e.paper3_name: e for e in ac.COVERAGE if e.paper3_name}
    items = []
    for letter, digit in LETTER_DIGIT.items():
        mine = [n for n in PAPER24 if n.startswith(letter + "_")]
        parts = [f"{code(n)} {html.escape(by_name[n].joint.split(' ', 1)[1])} "
                 f"{html.escape(by_name[n].movement)}" for n in mine]
        items.append(f"<li><strong>{digit.capitalize()}</strong> ({len(mine)}): "
                     f"{'; '.join(parts)}.</li>")
    rep.add("<ul>" + "".join(items) + "</ul>")
    rep.add(f"<p>The finger CMC is the arching of the palm. The thumb TMC sits on the trapezium, "
            f"a wrist bone. The paper measured its angles with a Cyberglove.</p>")
    rep.add(f"<p><strong>What the camera gives.</strong> The grasp set was recorded with a "
            f"camera, not a glove. In every frame the camera's software fits a hand skeleton to "
            f"its two infrared images. For each of {len(JOINT_NAMES)} joints it reports a position "
            f"and an orientation. Every angle in the dataset is read off these bone "
            f"orientations. Two limits follow. The skeleton's palm is one rigid piece, so the "
            f"metacarpals never move against each other. And the skeleton has no trapezium, so "
            f"there is no bone to measure the thumb's TMC against in the way Paper 3 defines "
            f"it.</p>")

    # --- 2 How each angle is read
    rep.heading("<h2>2 How each angle is read</h2>")
    rep.add("<p>A calculated angle is the rotation of one bone against its parent bone, split "
            "into named parts. The repository's angle coverage page "
            "(<code>docs/angle_coverage.md</code>, section 2) states the frames, the zero, the "
            "signs and the two rotation orders in these words:</p>")
    rep.add(base.md_to_html(doc_section(2)))
    f1 = rep.figure(fig_conventions(),
                    "One joint in its parent bone's frame (schematic). (a) Seen from the side, "
                    "with the hinge axis x pointing out of the page: flexion turns the child bone "
                    "toward the palm. (b) Seen from the back of a right hand, with the dorsal "
                    "axis y pointing out of the page: abduction turns it toward the thumb. In both, "
                    "z runs along the parent bone toward the wrist, and the zero is the child "
                    "bone in line with the parent (dashed). On a left hand x points toward the "
                    "thumb instead; the signs stay the same on both hands.")
    mcp = next(e for e in ac.COVERAGE if e.paper3_name == finger_mcp[0])
    rep.add(f"<p>Figure {f1} shows the xyz order's first two angles. The finger MCP is the one "
            f"joint with two Paper 3 angles that the camera gives, so it is given in both "
            f"orders: the xyz order under the Paper 3 name (for example {code(mcp.paper3_name)}) "
            f"and the paper's own order under the same name ending in "
            f"<code>_paper_order</code>. In the coverage table's words, these are "
            f"{html.escape(mcp.note.split(':')[0])}.</p>")

    # --- 3 The table
    rep.heading("<h2>3 The table</h2>")
    present = [s for s in ac.STATUS if any(e.status == s for e in ac.COVERAGE)]
    rep.add(f"<p>The repository keeps one entry per joint and movement, {n_entries} in all. Each "
            f"has one status:</p>")
    rep.add("<ul>" + "".join(
        f"<li><strong>{html.escape(ac.STATUS_LABELS[s])}</strong> "
        f"({sum(1 for e in ac.COVERAGE if e.status == s)}): "
        f"{html.escape(ac.STATUS_MEANINGS[s])}.</li>" for s in present) + "</ul>")
    t1 = rep.n_tab + 1
    f2 = rep.n_fig + 1
    rep.add(f"<p>Figure {f2} shows the status of each joint's two movements at a glance. Table "
            f"{t1} lists every entry with its column names.</p>")
    mapped = [c for e in EVALUATED for c in e.output_columns]
    rep.figure(fig_status_grid(),
               f"Status of each joint's flexion/extension and abduction/adduction. The fill and "
               f"the word give the status; the second line is the Paper 3 name, or the column the "
               f"tracker's value is kept in. Thumb TMC: Paper 3's coordinate is unavailable; the "
               f"right half of the cell is the unvalidated estimate "
               f"({len(ESTIMATED)} columns, not part of this report's numbers). Table {t1} also "
               f"lists a reference-evaluated pair, {names(mapped)}, which stays empty until a "
               f"reference device has recorded the same hand. The wrist's abduction/adduction "
               f"cell is its radial/ulnar deviation. The twist about each bone's own axis is in "
               f"Table {t1} and not drawn.")
    coverage_table(rep)

    # --- 4 The finger MCP angles are present
    rep.heading("<h2>4 The finger MCP angles are present</h2>")
    rep.add(f"<p>The dataset's own angle table, <code>metadata/joint_angles_24.csv</code>, has one "
            f"row per static take ({ag['rows']} rows). For each row the report reads the raw "
            f"recording, takes the frame named by <code>representative_frame_id</code> in "
            f"<code>metadata/grasps_summary.csv</code>, rebuilds the hand from it and computes "
            f"every angle as the repository does (<code>angle_coverage.all_angles</code>). It "
            f"then compares the {len(GIVEN)} calculated Paper 3 angles and the "
            f"{len(jf.MCP_PAPER_NAMES)} paper-order MCP columns with the dataset's values.</p>")
    res = [f"{ag['compared']} of the {ag['rows']} takes were recomputed."]
    if ag["failed"]:
        res.append("Not recomputed: " + "; ".join(
            f"{html.escape(g)} take {k}: {html.escape(w)}" for g, k, w in ag["failed"]) + ".")
    if all_d:
        res.append(f"Over the {len(COMPARED)} columns, {len(all_d)} values were compared. The "
                   f"largest difference is {fmt(worst)} degrees ({code(worst_col)}).")
        res.append(f"The dataset writes these values with {ag['decimals']} decimals, a step of "
                   f"{step_txt} degrees, so rounding alone gives differences up to half a step, "
                   f"{half_txt} degrees. "
                   + ("Every difference is within that." if within else
                      "Some differences are larger than that."))
        groups = (("thumb MCP and IP flexion", [c for c in GIVEN if c.startswith("T_")]),
                  ("finger MCP, xyz order", finger_mcp),
                  ("finger MCP, paper order", list(jf.MCP_PAPER_NAMES)),
                  ("finger PIP and DIP", [c for c in GIVEN if c.endswith(("_PIP", "_DIP"))]))
        res.append("Largest difference by group: " + "; ".join(
            f"{label} ({len(cols)} columns) {fmt(max_of([d for c in cols for d in ag['diffs'][c]]))}"
            for label, cols in groups) + " degrees.")
    zero_blank = [c for c in COMPARED if ag["blanks"][c] == 0]
    res.append(f"{len(zero_blank)} of the {len(COMPARED)} columns have no blank cell "
               f"(Table {rep.n_tab + 1}).")
    rep.add(f"<p>{' '.join(res)}</p>")
    t = rep.next_table()
    body = []
    for c in COMPARED:
        d = ag["diffs"][c]
        worst_c = max_of(d)
        body.append(f'<tr class="cmp" data-column="{c}" data-n="{len(d)}" '
                    f'data-max-abs-diff="{"" if worst_c is None else repr(worst_c)}" '
                    f'data-blank="{ag["blanks"][c]}">'
                    f"<td>{code(c)}</td><td>{html.escape(ac.STATUS_LABELS[ac.status_of(c)])}</td>"
                    f'<td class="num">{len(d)}</td><td class="num">{fmt(worst_c)}</td>'
                    f'<td class="num">{ag["blanks"][c]}</td></tr>')
    # each short table in a .keep block: caption and rows on one page
    rep.add(f'<div class="keep"><table id="agreement" class="small"><caption><strong>Table {t}.</strong> The '
            f"{len(COMPARED)} calculated columns of <code>joint_angles_24.csv</code> against the "
            f"angles recomputed from the raw recordings at each take's representative frame. n: "
            f"takes compared (a blank dataset cell is not compared). Max |difference|: the "
            f"largest absolute difference, degrees. Blank: empty cells in the dataset's column, "
            f"out of {ag['rows']}.</caption>"
            f'<colgroup><col style="width:46mm"><col><col style="width:14mm">'
            f'<col style="width:34mm"><col style="width:18mm"></colgroup>'
            f"<thead><tr>{th('Column')}{th('Status')}{th('n', True)}"
            f"{th('Max |difference| (deg)', True)}{th('Blank', True)}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")
    all_blank = all(ag["blanks"][c] == ag["rows"] for c in SIX)
    rep.add(f"<p>The {len(SIX)} unavailable Paper 3 names are columns of the same table. "
            + (f"Every one of their cells is blank: {ag['rows']} of {ag['rows']} each "
               if all_blank else "Their blank cells are counted ")
            + f"(Table {rep.n_tab + 1}). The reason each is blank is the coverage table's "
              f"note.</p>")
    t = rep.next_table()
    by_six = {e.paper3_name: e for e in ac.COVERAGE if e.paper3_name in SIX}
    body = [f'<tr class="six" data-column="{c}" data-blank="{ag["blanks"][c]}">'
            f"<td>{code(c)}</td><td class=\"num\">{ag['blanks'][c]}</td>"
            f"<td class=\"num\">{ag['rows']}</td><td>{html.escape(by_six[c].note)}</td></tr>"
            for c in SIX]
    rep.add(f'<div class="keep"><table id="unavailable" class="small why"><caption><strong>Table '
            f"{t}.</strong> The {len(SIX)} unavailable Paper 3 names in "
            f"<code>joint_angles_24.csv</code>: blank cells out of the rows, and why the cell is "
            f"blank (the coverage table's note).</caption>"
            f'<colgroup><col style="width:20mm"><col style="width:12mm"><col style="width:12mm">'
            f"<col></colgroup><thead><tr>{th('Column')}{th('Blank', True)}{th('Rows', True)}"
            f"{th('Why it is blank')}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")

    # --- 5 What is not given and what it would take
    rep.heading("<h2>5 What is not given and what it would take</h2>")
    groups = {}
    for c in SIX:
        groups.setdefault(by_six[c].requirement, []).append(c)
    # the sentence and its list kept together, so the heading never ends a page
    rep.add(f'<div class="keep"><p>The {len(SIX)} unavailable angles fall into {len(groups)} '
            f"groups, each with one requirement in the coverage table:</p>")

    def joint_phrase(cols):
        joints = list(dict.fromkeys(by_six[c].joint for c in cols))
        if len(joints) == 1:
            return joints[0]
        words = [j.split() for j in joints]
        digits = [w[0] for w in words]
        return f"{words[0][1]} of the {', '.join(digits[:-1])} and {digits[-1]} fingers"

    rep.add('<dl class="req">' + "".join(
        f'<div class="keep"><dt>{html.escape(joint_phrase(cols)[:1].upper() + joint_phrase(cols)[1:])}: '
        f"{names(cols)}</dt>"
        f"<dd>{html.escape(req)}.</dd></div>" for req, cols in groups.items()) + "</dl></div>")
    fingers = [n for n in ag["palm"] if not n.startswith("THUMB")]
    if ag["compared"]:
        rep.add(f"<p><strong>The rigid palm, measured.</strong> At the {ag['compared']} "
                f"representative frames, each finger metacarpal's orientation in the palm frame "
                f"differs from the first frame's by at most "
                f"{max(ag['palm'][n] for n in fingers):.6f} degrees. The thumb metacarpal, which "
                f"the tracker does move, differs by up to {ag['palm']['THUMB_METACARPAL']:.2f} "
                f"degrees. So the palm arc is not in the recordings at all, while the thumb's "
                f"movement is; what is missing for the thumb is a frame that ties it to Paper "
                f"3's TMC.</p>")
    open_cols = [c for e in ESTIMATED if e.zero_position == ac.ZERO_OPEN_HAND
                 for c in e.output_columns]
    rep.add(f"<p><strong>The open-hand TMC estimate.</strong> The repository can compute the "
            f"thumb metacarpal's movement against the open hand recorded in the same session "
            f"({names(open_cols)}; <code>docs/tmc_estimate.md</code>). It is an unvalidated "
            f"estimate: it is not one of Paper 3's coordinates, a change to Paper 3's reference "
            f"would not shift it by a fixed amount, and no independent sensor has checked it. "
            f"It is not part of this report, and no number from it appears here.</p>")

    # --- 6 Abduction and adduction
    rep.heading("<h2>6 Abduction and adduction</h2>")
    paper_aa = [e for e in ac.COVERAGE if e.paper3_name and movement_kind(e) == "aa"]
    given_aa = [e.paper3_name for e in paper_aa if e.status == "calculated"]
    missing_aa = [e.paper3_name for e in paper_aa if e.status != "calculated"]
    every = all(ag["blanks"][c] == 0 for c in given_aa)
    rep.add(f"<p><strong>In Paper 3's model.</strong> The model has {len(paper_aa)} abduction "
            f"angles: {names([e.paper3_name for e in paper_aa])}. The grasp set gives "
            f"{len(given_aa)} of them, {names(given_aa)}"
            f"{', in every row' if every else ''} (Table {t1 + 1}). "
            + (f"{names(missing_aa)} is among the unavailable angles (section 5)."
               if missing_aa else "") + "</p>")
    side = [e for e in OUTSIDE if movement_kind(e) == "aa"]
    joint_side = [e for e in side if digit_of(e) != "wrist"]
    rep.add(f"<p><strong>Not in Paper 3's model.</strong> In the paper, the thumb MCP and IP and "
            f"the finger PIP and DIP bend one way only: each has one degree of freedom (the "
            f"coverage table's note, for example: “{html.escape(joint_side[0].note)}”). The "
            f"tracker's bones can still lean a little sideways at these joints, and it reports "
            f"that lean. The report keeps it under {len(joint_side)} names ending in "
            f"<code>_tracker</code>: {names([c for e in joint_side for c in e.output_columns])}. "
            f"The wrist's sideways bend (radial/ulnar deviation) is kept the same way, as "
            f"{names([c for e in side if digit_of(e) == 'wrist' for c in e.output_columns])}. "
            f"The finger CMC has no abduction in the paper's model and is not measured either "
            f"(rigid palm).</p>")
    rep.add("<p><strong>Why not under Paper 3's names.</strong> Paper 3 has no coordinate for "
            "these movements, so there is no name to put them under. Writing them under a "
            "Paper 3 name, or in the 24-angle table, would make a tracker number look like one of "
            "the paper's angles. A separate <code>_tracker</code> name keeps them available "
            "without that confusion.</p>")

    # --- 7 Reproduce
    rep.heading("<h2>7 Reproduce</h2>")
    cmd = (f'.venv\\Scripts\\python.exe scripts\\leap\\angle_coverage_report.py --dataset '
           f'"{Path(args.dataset).resolve()}" --out "{Path(args.out).resolve()}"'
           + (" --no-pdf" if args.no_pdf else ""))
    rep.heading("<p>The command below reads the coverage table from the code, section 2 from "
                "<code>docs/angle_coverage.md</code> and the dataset's tables and recordings, and "
                "writes <code>angle_coverage.csv</code>, <code>report.html</code> and "
                "<code>report.pdf</code> into the output folder. It refuses an output folder "
                "inside the dataset. Run it from the repository root "
                "(<code>non glove xr trainer</code>):</p>")
    rep.add(f"<pre>{html.escape(cmd)}</pre>")

    # --- Appendix A
    rep.add('<div class="newpage"></div>')
    rep.heading('<h2>Appendix A: every entry in full</h2>')
    rep.heading(f"<p>The {n_entries} entries of Table {t1}, with every field of the coverage "
                f"table, grouped by digit.</p>")
    fields = (("Reference frame", "reference_frame"), ("Zero", "zero_position"),
              ("Sign", "sign_convention"), ("Method", "method"), ("Note", "note"),
              ("Requirement", "requirement"))
    k = 0
    for title, entries in by_digit(ac.COVERAGE):
        rep.heading(f"<h3>{html.escape(title)}</h3>")
        for e in entries:
            k += 1
            label = (code(e.paper3_name) if e.paper3_name else
                     compact_columns(e.output_columns) if e.output_columns else "no column")
            rows = "".join(
                f"<dt>{name}</dt><dd>"
                + (html.escape(getattr(e, attr)) if getattr(e, attr)
                   else '<span class="none">none</span>') + "</dd>"
                for name, attr in fields)
            rep.add(f'<div class="entry"><p class="head"><strong>A{k}. '
                    f"{html.escape(e.joint)}, {html.escape(e.movement)}</strong>: "
                    f"{html.escape(ac.STATUS_LABELS[e.status])}; {label}</p>"
                    f'<dl class="fields">{rows}</dl></div>')
    rep.add(f'<p class="footer">{html.escape(FOOTER)}</p>')

    page = (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(TITLE)}</title><style>{base.CSS}{CSS_EXTRA}</style></head>\n"
            f"<body><main>\n" + "\n".join(rep.parts) + "\n</main></body></html>\n")
    return page, worst


# --- the run ---------------------------------------------------------------------------------
def print_pdf(html_path, pdf_path):
    """Print with the first browser found; the browser's path, or None when none exists."""
    exe = next((b for b in BROWSERS if b.is_file()), None)
    if exe is None:
        return None
    pdf_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run([str(exe), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--no-first-run", f"--user-data-dir={profile}",
                        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                       capture_output=True, timeout=300)
    return exe


def run(args):
    out = Path(args.out)
    if base._is_under(out, args.dataset):
        print(f"refusing to write inside the dataset: {out} is inside {args.dataset}")
        return 2
    try:
        ag = agreement(args.dataset)
        plt.rcParams.update(base.RC)
        page, worst = build(args, ag)
    except DatasetError as e:
        print(f"COVERAGE REPORT FAIL: {e}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    print(f"wrote {ac.write_coverage_csv(out / 'angle_coverage.csv')}")
    html_path = out / "report.html"
    html_path.write_text(page, encoding="utf-8")
    print(f"wrote {html_path}")
    pdf = "no"
    if not args.no_pdf:
        pdf_path = out / "report.pdf"
        exe = print_pdf(html_path, pdf_path)
        if exe is None:
            print(f"PDF NOT MADE: neither {' nor '.join(str(b) for b in BROWSERS)} exists; "
                  f"report.html only")
        elif not pdf_path.is_file() or pdf_path.read_bytes()[:4] != b"%PDF":
            print(f"COVERAGE REPORT FAIL: {exe} did not print {pdf_path}")
            return 1
        else:
            print(f"wrote {pdf_path} ({exe.name})")
            pdf = "yes"
    print(f"COVERAGE REPORT OK: entries={len(ac.COVERAGE)} given={len(GIVEN)} "
          f"unavailable={len(SIX)} max_abs_diff_deg={'none' if worst is None else f'{worst:.6g}'} "
          f"pdf={pdf}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Write which joint angles the grasp set gives, joint by joint, as a "
                    "paper-style report (report.html, report.pdf) with angle_coverage.csv.")
    p.add_argument("--dataset", type=Path, required=True,
                   help="the organized dataset root (read only): metadata/joint_angles_24.csv, "
                        "metadata/grasps_summary.csv and the static takes' JSONL")
    p.add_argument("--out", type=Path, required=True,
                   help="output folder, outside the dataset")
    p.add_argument("--no-pdf", action="store_true", help="write report.html only")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
