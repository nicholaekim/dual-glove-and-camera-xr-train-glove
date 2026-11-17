"""The camera-clock erratum as a paper-style report: report.html and report.pdf.

Reads an erratum folder written by `scripts/leap/camera_clock_erratum.py`
(README.md, static_takes.csv, transitions.csv, summary.md, and a hand-written
VERIFICATION.md when there is one) and writes beside it `report.html`, one
self-contained file (CSS inline, figures as base64 PNG), and `report.pdf`,
printed from it by headless Edge or Chrome. Every number in the report is
read from those files at run time; nothing is recomputed from the dataset
except the one figure that needs the raw lines: with `--dataset`, the writer
and camera clocks across the hold of the take with the largest difference,
read through `leap_hand.replay_core.find_static`. When that take cannot be
loaded the report says the figure was skipped, and why. The dataset is only
read; an output folder inside it is refused.

  .venv\\Scripts\\python.exe scripts\\leap\\camera_clock_erratum_report.py --erratum "<folder>" --dataset "<root>"

Prints `REPORT OK: static=N transitions=M over_one_period=K figures=F pdf=yes|no`
(exit 0); with no Edge or Chrome, `PDF NOT MADE: ...` before it and the HTML
only. Exit 1: an erratum file is missing, summary.md disagrees with the CSVs,
or the browser did not print the PDF. Exit 2: bad arguments or an output
folder inside the dataset.
"""
import argparse
import base64
import csv
import html
import io
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from leap_hand import replay_core as rc  # noqa: E402

TITLE = "Camera-clock erratum for the grasp set recorded on 2026-10-08"
AUTHOR = "Nicholas Kim"
FOOTER = "Derived from the erratum tables; the dataset itself is unchanged."
FILES = ("README.md", "static_takes.csv", "transitions.csv", "summary.md")
BROWSERS = (Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))

# figures: drawn at the printed size (A4 text width 174 mm), so font sizes are true points
DPI = 200
FIG_W = 6.7
BIN_MS = 0.5
FAR_MS = 100.0          # a value further than this from zero gets its own panel
INK, INK2, MUTED, AXIS, GRID = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 (validated pair)
REST = "#c9c7bf"                         # de-emphasised fill
BAND = "#f0efec"
RC = {"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": AXIS,
      "axes.labelcolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
      "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
      "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
      "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7}


# --- reading the erratum ------------------------------------------------------------------
def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    return None if v in (None, "") else float(v)


def fx(x, digits):
    """Fixed digits, blank for None; + 0.0 turns a rounded -0.0 into 0.0."""
    return "" if x is None else f"{round(x, digits) + 0.0:.{digits}f}"


def ms(seconds):
    return None if seconds is None else seconds * 1e3


def summary_tables(text):
    """{first header cell: [(label, n, min, median, p95, max), ...]} for every
    table in summary.md. The label is all but the last five cells, since a
    label may itself hold a '|'."""
    lines, tables = text.splitlines(), {}
    for i, line in enumerate(lines[:-1]):
        if line.startswith("| ") and lines[i + 1].startswith("|---"):
            title = line.strip()[1:-1].split("|")[0].strip()
            body = []
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                cells = [c.strip() for c in row.strip()[1:-1].split("|")]
                body.append(("|".join(cells[:-5]), *cells[-5:]))
            tables[title] = body
    return tables


def readme_block(text, opening):
    """The README from the paragraph that opens with `opening` up to the next
    paragraph that opens with a bold run-in heading."""
    i = text.find(opening)
    if i < 0:
        raise LookupError(f"README.md has no {opening!r} section")
    rest = text[i + len(opening):]
    m = re.search(r"\n\n\*\*[^*\n]+\*\*", rest)
    return (rest[:m.start()] if m else rest).strip()


# --- HTML pieces -----------------------------------------------------------------------------
def brk(s):
    """Escaped text with break opportunities after underscores and path separators."""
    return re.sub(r"([_\\/])", r"\1<wbr>", html.escape(s))


def inline(text):
    """Markdown inline: `code` and **bold**; everything else escaped."""
    out = []
    for k, part in enumerate(text.split("`")):
        if k % 2:
            out.append(f"<code>{brk(part)}</code>")
        else:
            out.append(re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(part)))
    return "".join(out)


def table(head_rows, body_rows, num_cols=(), tid=None, cls=None, caption=None, widths=()):
    """`widths`: CSS widths of the leading columns (a grasp name, a column
    name), so the browser does not squeeze them to their narrowest wrap."""
    attrs = (f' id="{tid}"' if tid else "") + (f' class="{cls}"' if cls else "")
    cap = f"<caption>{caption}</caption>" if caption else ""
    cols = ("<colgroup>" + "".join(f'<col style="width:{w}">' for w in widths) + "</colgroup>"
            if widths else "")
    head = "".join(f"<tr>{''.join(r)}</tr>" for r in head_rows)
    body = "".join(
        "<tr>" + "".join(f'<td class="num">{c}</td>' if i in num_cols else f"<td>{c}</td>"
                         for i, c in enumerate(r)) + "</tr>\n" for r in body_rows)
    return f"<table{attrs}>{cap}{cols}<thead>{head}</thead><tbody>\n{body}</tbody></table>"


def th(text, numeric=False, colspan=1, rowspan=1):
    a = (' class="num"' if numeric else "") + (f' colspan="{colspan}"' if colspan > 1 else "") \
        + (f' rowspan="{rowspan}"' if rowspan > 1 else "")
    return f"<th{a}>{text}</th>"


def md_table_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_html(text, shift=0):
    """The Markdown the erratum's files use: headings, paragraphs, - and 1.
    lists with indented continuation lines, | tables, ``` blocks."""
    lines, out, para, i = text.splitlines(), [], [], 0

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(f"<pre>{html.escape(chr(10).join(lines[i + 1:j]))}</pre>")
            i = j + 1
            continue
        m = re.match(r"(#{1,6}) (.*)", line)
        if m:
            flush()
            level = min(len(m.group(1)) + shift, 6)
            out.append(f'<h{level} class="md">{inline(m.group(2))}</h{level}>')
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"\|[-:| ]+$", lines[i + 1].strip()):
            flush()
            j, body = i + 2, []
            while j < len(lines) and lines[j].startswith("|"):
                body.append([inline(c) for c in md_table_row(lines[j])])
                j += 1
            out.append(table([[th(inline(c)) for c in md_table_row(line)]], body))
            i = j
            continue
        m = re.match(r"(-|\d+\.) ", line)
        if m:
            flush()
            ordered, items = m.group(1) != "-", []
            while i < len(lines):
                m = re.match(r"(-|\d+\.) (.*)", lines[i])
                if m and (m.group(1) != "-") == ordered:
                    items.append(m.group(2).strip())
                elif items and lines[i].startswith(" ") and lines[i].strip():
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        if line.strip():
            para.append(line.strip())
        else:
            flush()
        i += 1
    flush()
    return "\n".join(out)


# --- figures ---------------------------------------------------------------------------------
def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def windows(values, keep=()):
    """Axis windows for a broken axis: values further than FAR_MS from zero
    get a narrow window of their own on their side, the rest (with `keep`)
    one main window. [(lo, hi, is_main)], left to right."""
    near = [v for v in values if abs(v) <= FAR_MS]
    if not near:                       # nothing to break away from
        near = list(values)
    far_lo = [v for v in values if v not in near and v < 0]
    far_hi = [v for v in values if v not in near and v > 0]

    def win(vs, pad):
        lo, hi = min(vs) - pad, max(vs) + pad
        return np.floor(lo / BIN_MS) * BIN_MS, np.ceil(hi / BIN_MS) * BIN_MS

    out = [(*win(far_lo, 4.0), False)] if far_lo else []
    out.append((*win(near + list(keep), 1.5), True))
    if far_hi:
        out.append((*win(far_hi, 4.0), False))
    return out


def broken_axes(fig_h, wins, sharey=True):
    """One panel per window (far panels narrow), with the break marked."""
    ratios = [5 if main else 1 for _, _, main in wins]
    fig, axes = plt.subplots(1, len(wins), figsize=(FIG_W, fig_h), sharey=sharey,
                             width_ratios=ratios, layout="constrained")
    axes = list(np.atleast_1d(axes))
    for k, (ax, (lo, hi, _)) in enumerate(zip(axes, wins)):
        ax.set_xlim(lo, hi)
        if k > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False)
        if k < len(axes) - 1:      # mark the break with two slashes, sized in points
            slash = dict(marker=[(-1, -2), (1, 2)], markersize=7, linestyle="none", color=INK2,
                         mec=INK2, mew=0.7, clip_on=False)
            ax.plot([1], [0], transform=ax.transAxes, **slash)
            axes[k + 1].plot([0], [0], transform=axes[k + 1].transAxes, **slash)
    if len(axes) > 1:
        fig.get_layout_engine().set(wspace=0.02)
    return fig, axes


def fig_histogram(static, period_ms, top):
    """F1: camera minus declared for every take, ms; over one period in blue."""
    over = [ms(num(r["static_interval_difference_s"])) for r in static
            if r["static_interval_over_one_period"] == "true"]
    within = [ms(num(r["static_interval_difference_s"])) for r in static
              if r["static_interval_over_one_period"] == "false"]
    wins = windows(over + within, keep=(-period_ms, period_ms))
    fig, axes = broken_axes(2.4, wins)
    tallest = 0
    for ax, (lo, hi, main) in zip(axes, wins):
        edges = np.arange(lo, hi + BIN_MS / 2, BIN_MS)
        inside = [v for v in within + over if lo <= v <= hi]
        counts = np.histogram(inside, bins=edges)[0]
        tallest = max([tallest, *counts])
        ax.hist([[v for v in within if lo <= v <= hi], [v for v in over if lo <= v <= hi]],
                bins=edges, stacked=True, color=[REST, BLUE], edgecolor="white", linewidth=0.5,
                label=["within one frame period", "over one frame period"])
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        if main:
            for sign in (-1, 1):
                ax.axvline(sign * period_ms, color=INK2, lw=0.7, ls=(0, (3, 2)))
                ax.text(sign * period_ms - 0.4, 0.98,
                        f"{'−' if sign < 0 else '+'}1 frame period\n"
                        f"({'−' if sign < 0 else '+'}{period_ms:.3f} ms)",
                        transform=ax.get_xaxis_transform(), ha="right", va="top",
                        fontsize=6.5, color=INK2)
            main_ax = ax
        else:
            ax.set_xticks([round((lo + hi) / 2 / 5) * 5])
            for v in inside:             # call the far value out with its number
                ax.annotate(f"{'outlier' if top and v == top[1] else ''}\n{v:.3f} ms"
                            .replace("-", "−").strip(),
                            xy=(v, counts[min(np.searchsorted(edges, v, side="right") - 1,
                                              len(counts) - 1)]), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.5, color=INK)
    axes[0].set_ylim(0, tallest * 1.45)
    axes[0].set_ylabel("static takes")
    axes[0].yaxis.get_major_locator().set_params(integer=True)
    fig.supxlabel("camera-clock minus declared hold duration (ms)", fontsize=7.5, color=INK2)
    main_ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    return png(fig)


def fig_bars(flagged):
    """F2: the takes over one period, largest |difference| first, ms."""
    labels = [f"{r['grasp_name']} · take {r['take']}" for r in flagged]
    vals = [ms(num(r["static_interval_difference_s"])) for r in flagged]
    wins = windows(vals, keep=(0.0,))
    y = np.arange(len(vals))
    fig, axes = broken_axes(max(1.5, 0.125 * len(vals) + 0.75), wins)
    for ax, (lo, hi, main) in zip(axes, wins):
        ax.barh(y, vals, height=0.62, color=BLUE)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, right=False)
        if main:
            ax.axvline(0, color=AXIS, lw=0.8)
            ax.yaxis.tick_right()
            ax.set_yticks(y, labels, fontsize=6.5, color=INK)
            ax.tick_params(axis="y", right=False, pad=3)
        else:
            ax.set_xticks([round((lo + hi) / 2 / 5) * 5])
    axes[0].set_ylim(len(vals) - 0.5, -0.5)
    fig.supxlabel("camera-clock minus declared hold duration (ms)", fontsize=7.5, color=INK2)
    return png(fig)


def fig_clocks(ids, writer, camera, k, jump_s, camera_ms, name):
    """F3: wall_time and timestamp_us from the hold's first frame, per frame."""
    fig, ax = plt.subplots(figsize=(FIG_W, 3.1), layout="constrained")
    ax.plot(ids, camera, color=BLUE, lw=2.2, solid_capstyle="round",
            label="camera clock (timestamp_us)")
    ax.plot(ids, writer, color=ORANGE, lw=1.2, solid_capstyle="round",
            label="writer clock (wall_time)")
    span = ids[-1] - ids[0]
    ax.set_xlim(ids[0] - 0.02 * span, ids[-1] + 0.16 * span)
    ax.set_ylim(-0.05, max(writer[-1], camera[-1]) * 1.12)
    for series, text in ((writer, "writer"), (camera, "camera")):
        ax.annotate(f"{text} {series[-1]:.3f} s", xy=(ids[-1], series[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5,
                    color=INK)
    mid = (writer[k] + writer[k + 1]) / 2
    ax.annotate(f"writer clock steps {jump_s * 1e3:.3f} ms between frames {ids[k]} and "
                f"{ids[k + 1]};\ncamera clock steps {camera_ms:.3f} ms over the same pair",
                xy=(ids[k] + 0.5, mid), xytext=(0.04, 0.80), textcoords="axes fraction",
                fontsize=6.5, color=INK, va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=7,
                                connectionstyle="arc3,rad=-0.1", shrinkB=3))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"frame_id ({name})")
    ax.set_ylabel("seconds from the hold's first frame")
    ax.legend(loc="lower right")
    return png(fig)


def fig_residuals(trans):
    """F4: movement start and settle residuals on the two clocks, one axis."""
    fig, ax = plt.subplots(figsize=(FIG_W, 2.3), layout="constrained")
    rng = np.random.default_rng(0)
    every = []
    for event, color, off, label in (("movement_start", BLUE, 0.13, "movement start"),
                                     ("settled", ORANGE, -0.13, "settle")):
        for row_y, clock in ((1, "capture"), (0, "wall")):
            v = [num(r[f"{event}_{clock}_residual_ms"]) for r in trans
                 if num(r[f"{event}_{clock}_residual_ms"]) is not None]
            every += v
            ax.scatter(v, row_y + off + rng.uniform(-0.08, 0.08, len(v)), s=14, color=color,
                       edgecolors="white", linewidths=0.6, zorder=3,
                       label=label if clock == "capture" else None)
    m = max([abs(x) for x in every] + [1.0]) * 1.08
    ax.set_xlim(-m, m)
    ax.set_ylim(-0.45, 1.45)
    ax.axvspan(-0.5, 0.5, color=BAND, zorder=0)
    ax.text(0.6, 1.42, "±0.5 ms", ha="left", va="top", fontsize=6.5, color=INK2)
    ax.set_yticks([1, 0], ["by capture_time", "by wall_time"], fontsize=7, color=INK)
    ax.tick_params(axis="y", left=False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("nearest selected frame minus the metadata instant (ms)")
    ax.legend(loc="upper left")
    return png(fig)


# --- the outlier's raw lines -----------------------------------------------------------------
def outlier_lines(root, row):
    """(frame ids, writer s, camera s, index of the stall pair, jump s, camera ms)
    for the hold of `row`, read from the JSONL; or (None, why)."""
    name, take = row["grasp_name"], int(row["take"])
    try:
        rec = rc.find_static(root, name, take)
    except (OSError, ValueError, LookupError, KeyError) as e:
        return None, f"{name} take {take} could not be loaded from the dataset ({e})"
    first, last = int(row["static_interval_first_frame_id"]), int(row["static_interval_last_frame_id"])
    hold = [f for f in rec.frames if first <= f.frame_id <= last]
    if len(hold) < 2 or any(f.wall_time is None or f.timestamp_us is None for f in hold):
        return None, (f"{name} take {take}: the hold frames {first} to {last} do not all carry "
                      f"wall_time and timestamp_us")
    ids = [f.frame_id for f in hold]
    try:
        k = ids.index(int(row["stall_frame_id_before"]))
    except ValueError:
        return None, f"{name} take {take}: frame {row['stall_frame_id_before']} is not in the hold"
    writer = [f.wall_time - hold[0].wall_time for f in hold]
    camera = [(f.timestamp_us - hold[0].timestamp_us) / 1e6 for f in hold]
    return (ids, writer, camera, k, writer[k + 1] - writer[k],
            (camera[k + 1] - camera[k]) * 1e3), None


# --- the report ------------------------------------------------------------------------------
CSS = """
@page { size: A4; margin: 20mm 18mm;
  @bottom-center { content: counter(page) " of " counter(pages);
    font-family: Georgia, "Times New Roman", Times, serif; font-size: 8.5pt; color: #444; } }
html, body { background: #fff; }
body { margin: 0; color: #111; font-family: Georgia, "Times New Roman", Times, serif;
  font-size: 10.5pt; line-height: 1.35; }
main { max-width: 174mm; margin: 0 auto; }
@media screen { main { padding: 24px 16px; } }
header.title { text-align: center; margin: 4mm 0 8mm; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 8pt; font-weight: bold; }
header.title p { margin: 2pt 0; }
.author { font-size: 11.5pt; }
.date { font-size: 9.5pt; color: #333; }
.abstract { margin: 0 8mm 6mm; font-size: 10pt; }
.abstract h2 { font-size: 10.5pt; text-align: center; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 14pt 0 5pt; }
h3 { font-size: 11.5pt; margin: 11pt 0 4pt; }
h4, h5, h6 { font-size: 10.5pt; margin: 9pt 0 3pt; }
h2, h3, h4, h5, h6 { break-after: avoid; page-break-after: avoid; break-inside: avoid; }
h3.md { font-size: 11pt; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 6pt; padding-left: 20pt; }
li { margin-bottom: 2pt; }
code, pre { font-family: Consolas, "Courier New", monospace; }
code { font-size: 0.88em; }
pre { font-size: 8.5pt; line-height: 1.3; white-space: pre-wrap; overflow-wrap: anywhere;
  border-top: 0.5pt solid #000; border-bottom: 0.5pt solid #000; padding: 3pt 0;
  margin: 2pt 0 8pt; break-inside: avoid; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 2pt 0 10pt; font-size: 9pt;
  line-height: 1.25; }
caption { caption-side: top; text-align: left; font-size: 9.5pt; padding: 0 0 4pt;
  line-height: 1.3; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th, td { padding: 1.6pt 4pt; text-align: left; vertical-align: top; }
thead tr:first-child th { border-top: 0.9pt solid #000; }
thead tr:last-child th { border-bottom: 0.5pt solid #000; }
thead th { font-weight: bold; vertical-align: bottom; }
th.group { text-align: center; border-bottom: 0.4pt solid #000; }
tbody tr:last-child td { border-bottom: 0.9pt solid #000; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
td.num { white-space: nowrap; }
table.small { font-size: 8pt; line-height: 1.2; }
table.small th, table.small td { padding: 0.9pt 2.5pt; }
table.small thead th { font-size: 7.5pt; }
figure { margin: 6pt 0 12pt; break-inside: avoid; page-break-inside: avoid; }
figure img { display: block; margin: 0 auto; max-width: 100%; height: auto; }
figcaption, .skipped { font-size: 9.5pt; line-height: 1.3; margin-top: 4pt; }
.skipped { font-style: italic; }
.keep { break-inside: avoid; page-break-inside: avoid; }
.newpage { break-before: page; page-break-before: always; }
.footer { margin-top: 16pt; border-top: 0.5pt solid #000; padding-top: 4pt; font-size: 9pt;
  font-style: italic; }
"""


class Report:
    """The page, assembled in reading order; figures and tables numbered as they come."""

    def __init__(self):
        self.parts, self.n_fig, self.n_tab = [], 0, 0

    def add(self, s):
        self.parts.append(s)

    def figure(self, b64, caption):
        self.n_fig += 1
        self.add(f'<figure><img alt="Figure {self.n_fig}" style="width:{FIG_W}in" '
                 f'src="data:image/png;base64,{b64}"><figcaption><strong>Figure {self.n_fig}.'
                 f'</strong> {caption}</figcaption></figure>')
        return self.n_fig

    def next_table(self):
        self.n_tab += 1
        return self.n_tab


def flagged_rows(static):
    over = [r for r in static if r["static_interval_over_one_period"] == "true"]
    return sorted(over, key=lambda r: -abs(num(r["static_interval_difference_s"])))


def dist_table(rep, rows, caption, to_ms, first_col):
    """A distribution table of summary.md, in ms (`to_ms`: summary.md gives seconds)."""
    t = rep.next_table()
    body = []
    for label, n, *vals in rows:
        cells = [fx(num(v) * 1e3, 3) if (to_ms and v) else v for v in vals]
        body.append([inline(label), n, *cells])
    rep.add(table([[th(first_col), th("n", True), th("min", True), th("median", True),
                    th("p95", True), th("max", True)]], body,
                  num_cols={1, 2, 3, 4, 5}, caption=f"<strong>Table {t}.</strong> {caption}"))
    return t


def build(args, static, trans, readme, summary, verification):
    tables = summary_tables(summary)
    verdict = re.search(r"ERRATUM (?:OK|FAIL): static=(\d+) transitions=(\d+) "
                        r"static_interval_over_one_period=(\d+).*", summary)
    generated = re.search(r"Generated (\d{4}-\d\d-\d\d \d\d:\d\d)", summary)
    flagged = flagged_rows(static)
    n, nt, k = len(static), len(trans), len(flagged)
    periods = sorted(num(r["frame_period_s"]) * 1e3 for r in static if num(r["frame_period_s"]))
    period_ms = float(np.median(periods)) if periods else 1e3 / 90
    prange = (f"{periods[0]:.3f} to {periods[-1]:.3f} ms" if periods and periods[0] != periods[-1]
              else f"{period_ms:.3f} ms")
    declared = sorted({num(r["static_interval_duration_metadata_s"]) for r in static
                       if num(r["static_interval_duration_metadata_s"]) is not None})
    top = flagged[0] if flagged else None
    others = flagged[1:]
    timed = [r for r in trans if num(r["transition_duration_camera_s"]) is not None]
    cap_res = [abs(num(r[f"{e}_capture_residual_ms"])) for r in trans
               for e in ("movement_start", "settled") if num(r[f"{e}_capture_residual_ms"]) is not None]
    wall_res = [abs(num(r[f"{e}_wall_residual_ms"])) for r in trans
                for e in ("movement_start", "settled") if num(r[f"{e}_wall_residual_ms"]) is not None]
    inst_diff = [abs(num(r[f"{e}_difference_s"])) * 1e3 for r in trans
                 for e in ("movement_start", "settled") if num(r[f"{e}_difference_s"]) is not None]
    on_ts = sum(1 for r in static if r["clock_source"] == "timestamp_us"
                and r["clock_fallback"] == "false")

    rep = Report()
    # --- title and abstract
    rep.add(f'<header class="title"><h1>{html.escape(TITLE)}</h1><p class="author">{AUTHOR}</p>'
            f'<p class="date">Report generated {datetime.now():%Y-%m-%d}'
            + (f"; erratum tables generated {generated.group(1)}" if generated else "")
            + "</p></header>")
    a = [f"The dataset's metadata gives each static take's hold duration on the recorder's "
         f"writer clock (<code>wall_time</code>), which stalls. Recomputed on the camera's own "
         f"clock (<code>timestamp_us</code>), {k} of the {n} static takes differ from the declared "
         f"hold duration by more than one frame period ({prange})."]
    if top:
        if others:
            a.append(f"Of these, {len(others)} differ by at most "
                     f"{max(abs(num(r['static_interval_difference_s'])) for r in others) * 1e3:.3f} ms.")
        a.append(f"One, {html.escape(top['grasp_name'])} take {top['take']}, differs by "
                 f"{abs(num(top['static_interval_difference_s'])) * 1e3:.3f} ms: declared "
                 f"{num(top['static_interval_duration_metadata_s']):.3f} s, "
                 f"{num(top['static_interval_camera_s']):.6f} s on the camera clock, because the "
                 f"writer clock stepped {num(top['static_interval_largest_writer_stall_s']) * 1e3:.3f} ms "
                 f"between frames {top['stall_frame_id_before']} and {top['stall_frame_id_after']}.")
    a.append(f"The {nt} transition captures are unaffected: their times were already counted on "
             f"the capture clock"
             + (f", and every one of their {len(cap_res)} movement starts and settles lies within "
                f"{max(cap_res):.3f} ms of a recorded frame by that clock." if cap_res else ".")
             + " The recordings, the frames of each hold and the joint angles are unchanged; "
               "only durations and instants change.")
    rep.add(f'<section class="abstract"><h2>Abstract</h2><p>{" ".join(a)}</p></section>')

    # --- 1 Background
    rep.add("<h2>1 Background</h2>")
    rep.add("<p>Every line the recorder wrote carries three clocks. The erratum's README "
            "describes them as follows.</p>")
    rep.add(md_to_html(readme_block(readme, "**The three clocks on every JSONL line.**")))
    decl = (f"Every one of the {n} takes declares {declared[0]:.3f} s." if len(declared) == 1 else
            f"The declared lengths run from {declared[0]:.3f} to {declared[-1]:.3f} s."
            if declared else "")
    rep.add(f"<p><strong>The static interval.</strong> Each static take's row in "
            f"<code>metadata/grasps_summary.csv</code> names a static interval: the frames from "
            f"{inline('`static_interval_first_frame_id`')} to "
            f"{inline('`static_interval_last_frame_id`')}, while the hand held the grasp. Its "
            f"declared length, {inline('`static_interval_duration_s`')}, was cut on the writer "
            f"clock: the interval's end minus its start, where the start is the first interval "
            f"frame's <code>wall_time</code> and the end is an instant, not a frame. {decl} When "
            f"the writer stalls inside the interval, the writer clock runs ahead of the camera, "
            f"and the declared length covers less camera time than it says.</p>")

    # --- 2 Method
    rep.add("<h2>2 Method</h2>")
    rep.add("<p><strong>Selected frames.</strong> A take's frames are the lines of the hand its "
            "metadata row names (side and tracker ids) with <code>status != 0</code>; a line of the "
            "other hand or of a stray tracker id is never counted. The hold's frames are the "
            "selected frames whose <code>frame_id</code> lies between the interval's first and "
            "last frame ids.</p>")
    rep.add(f"<p><strong>Camera-clock durations.</strong> The hold's camera-clock duration "
            f"({inline('`static_interval_camera_s`')}) is the last hold frame's "
            f"<code>timestamp_us</code> minus the first's, in seconds. {on_ts} of the {n} takes were "
            f"timed on <code>timestamp_us</code> with no fallback. The same frames' "
            f"<code>wall_time</code> span ({inline('`static_interval_writer_span_s`')}) is given "
            f"beside it, so that the clock and the convention can be told apart: the declared end "
            f"is an instant up to one frame period after the last hold frame, so a difference of "
            f"up to one period can come from the convention alone. Every difference in this "
            f"report is camera minus declared.</p>")
    rep.add(f"<p><strong>One frame period.</strong> A take is flagged "
            f"({inline('`static_interval_over_one_period`')}) when the absolute difference exceeds "
            f"one nominal frame period, 1 / the median <code>framerate</code> of the take's lines: "
            f"{prange} in this set.</p>")
    rep.add("<p><strong>Which clock the transitions used.</strong> "
            "<code>metadata/pose_transitions.csv</code> gives each capture's movement start and "
            "settle as seconds from the capture time of its first frame, rounded to 1 ms. An "
            "instant cut on a given clock therefore lies within 0.5 ms of a frame on that clock. "
            "Each instant was matched to the nearest selected frame twice, once by "
            "<code>capture_time</code> and once by <code>wall_time</code>, and both residuals "
            "(frame minus instant) were written "
            f"({inline('`*_capture_residual_ms`')}, {inline('`*_wall_residual_ms`')}). The clock "
            "whose residuals stay within 0.5 ms is the clock the instants were cut on (Section "
            "3.3); each instant then takes that frame's camera time.</p>")
    rep.add("<p><strong>Columns.</strong> The README beside the tables defines every column (its "
            "parts <em>Static takes</em> and <em>Transitions</em>). The run that wrote the tables "
            "ended with this line (<code>summary.md</code>):</p>")
    rep.add(f"<pre>{html.escape(verdict.group(0))}</pre>")

    # --- 3 Results
    rep.add("<h2>3 Results</h2>")
    rep.add("<h3>3.1 Static takes</h3>")
    t1 = rep.n_tab + 1
    f1 = rep.n_fig + 1
    rep.add(f"<p>Table {t1} gives the distribution over the {n} takes of each difference "
            f"<code>summary.md</code> reports, in milliseconds. {k} of the {n} static takes differ "
            f"by more than one frame period (Figure {f1}); Figure {f1 + 1} and Table {t1 + 1} list "
            f"them, largest first.</p>")
    static_dist = tables.get("Static takes, seconds", [])
    dist_table(rep, static_dist, f"Distribution over the {n} static takes (milliseconds; from "
               f"<code>summary.md</code>, which gives seconds). Differences are camera minus "
               f"metadata; a writer stall is the largest <code>wall_time</code> step between "
               f"consecutive frames.", True, "Quantity")
    top_tuple = None
    if top:
        top_tuple = (f"{top['grasp_name']} take {top['take']}",
                     ms(num(top["static_interval_difference_s"])))
    far = top and abs(top_tuple[1]) > FAR_MS and any(
        abs(ms(num(r["static_interval_difference_s"]))) <= FAR_MS for r in static
        if num(r["static_interval_difference_s"]) is not None)
    rep.figure(fig_histogram(static, period_ms, top_tuple if far else None),
               f"Camera-clock minus declared hold duration of the {n} static takes (ms, bins of "
               f"{BIN_MS} ms); blue: the {k} over one frame period. Dashed: plus and minus one "
               f"frame period ({period_ms:.3f} ms, the median of the takes' periods)."
               + (f" Axis broken: {html.escape(top_tuple[0])} ({top_tuple[1]:.3f} ms) is in the "
                  f"left panel." if far else ""))
    if flagged:
        rep.figure(fig_bars(flagged),
                   f"The {k} takes whose difference exceeds one frame period, largest first; "
                   f"bars run from zero to the difference (ms)."
                   + (f" The axis is broken: the first bar, {html.escape(top_tuple[0])}, ends at "
                      f"{top_tuple[1]:.3f} ms in the left panel." if far else ""))
    t = rep.next_table()
    rep.add(table(
        [[th("Grasp"), th("Take", True), th("Declared (s)", True), th("Writer span (s)", True),
          th("Camera (s)", True), th("Difference (ms)", True), th("Largest stall (ms)", True),
          th("between frames", True)]],
        [[brk(r["grasp_name"]), r["take"], fx(num(r["static_interval_duration_metadata_s"]), 3),
          fx(num(r["static_interval_writer_span_s"]), 6), fx(num(r["static_interval_camera_s"]), 6),
          fx(ms(num(r["static_interval_difference_s"])), 3),
          fx(ms(num(r["static_interval_largest_writer_stall_s"])), 3),
          f"{r['stall_frame_id_before']}–{r['stall_frame_id_after']}"] for r in flagged],
        # a long table starts on its own page, so its caption and header are never
        # left at the foot of the figure page without a row
        num_cols={1, 2, 3, 4, 5, 6, 7}, tid="over-one-period",
        cls="small newpage" if k > 20 else "small", widths=("58mm",),
        caption=f"<strong>Table {t}.</strong> The {k} takes over one frame period, largest "
                f"difference first. Writer span: the <code>wall_time</code> span of the same hold "
                f"frames. Largest stall: the largest <code>wall_time</code> step between "
                f"consecutive hold frames, and the frame pair it falls between."))

    # 3.2 the outlier
    rep.add("<h3>3.2 The outlier</h3>")
    drawn = False
    if top:
        d_ms = abs(num(top["static_interval_difference_s"])) * 1e3
        txt = (f"The largest difference is {html.escape(top['grasp_name'])} take {top['take']}. It "
               f"declares a {num(top['static_interval_duration_metadata_s']):.3f} s hold. Its "
               f"{top['static_interval_frames']} hold frames ({top['static_interval_first_frame_id']} "
               f"to {top['static_interval_last_frame_id']}) span "
               f"{num(top['static_interval_writer_span_s']):.6f} s on the writer clock and "
               f"{num(top['static_interval_camera_s']):.6f} s on the camera clock, {d_ms:.3f} ms "
               f"less than declared. Between frames {top['stall_frame_id_before']} and "
               f"{top['stall_frame_id_after']} the writer clock steps "
               f"{num(top['static_interval_largest_writer_stall_s']) * 1e3:.3f} ms while the camera "
               f"clock steps {num(top['stall_camera_ms']):.3f} ms: the writer paused, the camera "
               f"kept its pace. The camera-clock value, "
               f"{num(top['static_interval_camera_s']):.6f} s, is the hold's length.")
        if others:
            o = others[0]
            txt += (f" The next largest difference is "
                    f"{abs(num(o['static_interval_difference_s'])) * 1e3:.3f} ms "
                    f"({html.escape(o['grasp_name'])} take {o['take']}).")
        rep.add(f"<p>{txt}</p>")
        if args.dataset is None:
            why = "no --dataset was given"
        else:
            got, why = outlier_lines(args.dataset, top)
            if got:
                ids, writer, camera, i, jump_s, cam_ms = got
                rep.figure(fig_clocks(ids, writer, camera, i, jump_s, cam_ms,
                                      f"{top['grasp_name']} take {top['take']}"),
                           f"The writer clock (<code>wall_time</code>) and the camera clock "
                           f"(<code>timestamp_us</code>) across the hold of "
                           f"{html.escape(top['grasp_name'])} take {top['take']}, frames {ids[0]} "
                           f"to {ids[-1]}, each in seconds from the hold's first frame, read from "
                           f"the raw JSONL lines. The writer clock steps {jump_s * 1e3:.3f} ms "
                           f"between frames {ids[i]} and {ids[i + 1]} "
                           f"(<code>static_takes.csv</code>: "
                           f"{num(top['static_interval_largest_writer_stall_s']) * 1e3:.3f} ms); "
                           f"the camera clock steps {cam_ms:.3f} ms.")
                drawn = True
        if not drawn:
            rep.add(f'<p class="skipped">The figure of the two clocks across this hold was '
                    f"skipped: {html.escape(why)}.</p>")
    else:
        rep.add("<p>No take differs by more than one frame period.</p>")

    # 3.3 transitions
    rep.add("<h3>3.3 Transitions</h3>")
    rep.add(f"<p>The erratum holds {nt} transition captures; {len(timed)} have a movement start "
            f"and a settle, the other {nt - len(timed)} are blank in the metadata (invalid "
            f"timing). "
            + (f"By <code>capture_time</code>, every one of the {len(cap_res)} instants lies within "
               f"{max(cap_res):.3f} ms of a selected frame. By <code>wall_time</code> the nearest "
               f"frame is up to {max(wall_res):.3f} ms away (median "
               f"{float(np.median(wall_res)):.3f} ms) (Figure {rep.n_fig + 1}, Table "
               f"{rep.n_tab + 1}). " if cap_res and wall_res else "")
            + "The transition times therefore already count on the capture clock and need no "
              "correction"
            + (f": on the camera clock they differ from the metadata by at most "
               f"{max(inst_diff):.3f} ms (Table {rep.n_tab + 2}), and the metadata itself is "
               f"rounded to 1 ms.</p>" if inst_diff else ".</p>"))
    if cap_res:
        rep.figure(fig_residuals(trans),
                   f"Residual of each transition instant (nearest selected frame minus the "
                   f"metadata instant, ms) when the instant is placed by <code>capture_time</code> "
                   f"(top) and by <code>wall_time</code> (bottom); {len(timed)} movement starts and "
                   f"{len(timed)} settles. Shaded: ±0.5 ms, half the metadata's 1 ms rounding step.")
    dist_table(rep, tables.get("Transitions, ms", []),
               "Which clock the transition instants were cut on (milliseconds; from "
               "<code>summary.md</code>): the nearest selected frame minus the instant.",
               False, "Residual")
    dist_table(rep, [r for r in tables.get("Transitions, seconds", []) if "camera - metadata" in r[0]],
               "Transition instants and durations, camera clock minus metadata (milliseconds; "
               "from <code>summary.md</code>, which gives seconds).", True, "Quantity")

    # --- 4 How to apply
    block = readme_block(readme, "**How to apply.**")
    intro, items, closing, state = [], [], [], "intro"
    for line in block.splitlines():
        m = re.match(r"(\d+)\. (.*)", line)
        if m and state != "closing":
            state = "items"
            items.append([m.group(1), m.group(2).strip()])
        elif state == "items" and line.startswith(" ") and line.strip():
            items[-1][1] += " " + line.strip()
        elif state == "items" and not line.strip():
            state = "closing"
        elif line.strip():
            (intro if state == "intro" else closing).append(line.strip())
    rep.add("<h2>4 How to apply</h2>")
    rep.add(f"<p>In the README's words: {inline(' '.join(intro))}</p>")
    t = rep.next_table()
    body = []
    for number, text in items:
        if "in place of" in text:
            before, after = text.split("in place of", 1)
            use = [c for c in re.findall(r"`([^`]+)`", before) if not c.endswith(".csv")]
            old = [c for c in re.findall(r"`([^`]+)`", after) if not c.endswith(".csv")]
            cells = ["<br>".join(inline(f"`{c}`") for c in use),
                     "<br>".join(inline(f"`{c}`") for c in old)]
        else:
            cells = ["no change", ""]
        body.append([number, inline(text), *cells])
    rep.add(table([[th("#"), th("README wording"), th("Use (erratum)"), th("In place of (metadata)")]],
                  body, tid="how-to-apply", widths=("4%", "50%", "23%", "23%"),
                  caption=f"<strong>Table {t}.</strong> Which column replaces which. Join on "
                          f"<code>grasp_name</code> + <code>take</code> (static takes) or "
                          f"<code>to_grasp</code> + <code>take</code> (transitions)."))
    if closing:
        rep.add(f"<p>{inline(' '.join(closing))}</p>")

    # --- 5 Verification
    rep.add("<h2>5 Verification</h2>")
    if verification is None:
        rep.add("<p>No verification note (<code>VERIFICATION.md</code>) was found in the erratum "
                "folder.</p>")
    else:
        rep.add("<p>The erratum folder holds a hand-written verification note, "
                "<code>VERIFICATION.md</code>, reproduced here in full.</p>")
        rep.add(md_to_html(verification, shift=2))

    # --- 6 Reproduce
    rep.add("<h2>6 Reproduce</h2>")
    code = re.search(r"```\n(.*?)\n```", readme_block(readme, "**Reproduce.**"), re.S)
    rep.add("<p>From the repository root (<code>non glove xr trainer</code>), the erratum tables "
            "(the command from the README):</p>")
    rep.add(f"<pre>{html.escape(code.group(1) if code else '(the README gives no command)')}</pre>")
    cmd = (f'.venv\\Scripts\\python.exe scripts\\leap\\camera_clock_erratum_report.py --erratum '
           f'"{Path(args.erratum).resolve()}"'
           + (f' --dataset "{Path(args.dataset).resolve()}"' if args.dataset else "")
           + (f' --out "{Path(args.out).resolve()}"' if args.out else "")
           + (" --no-pdf" if args.no_pdf else ""))
    rep.add("<p>and this report, from those tables:</p>")
    rep.add(f"<pre>{html.escape(cmd)}</pre>")

    # --- appendices
    rep.add(f'<h2 class="newpage">Appendix A: all {n} static takes</h2>')
    t = rep.next_table()
    srt = sorted(static, key=lambda r: (r["grasp_name"], int(r["take"])))
    rep.add(table(
        [[th("Grasp"), th("Take", True), th("Declared (s)", True), th("Camera (s)", True),
          th("Difference (ms)", True), th("Over one period"), th("Largest stall (ms)", True)]],
        [[brk(r["grasp_name"]), r["take"], fx(num(r["static_interval_duration_metadata_s"]), 3),
          fx(num(r["static_interval_camera_s"]), 6),
          fx(ms(num(r["static_interval_difference_s"])), 3),
          {"true": "yes", "false": "no"}.get(r["static_interval_over_one_period"], ""),
          fx(ms(num(r["static_interval_largest_writer_stall_s"])), 3)] for r in srt],
        num_cols={1, 2, 3, 4, 6}, tid="appendix-a", cls="small", widths=("58mm",),
        caption=f"<strong>Table {t}.</strong> All {n} static takes, by grasp then take. Declared: "
                f"<code>static_interval_duration_metadata_s</code>; camera: "
                f"<code>static_interval_camera_s</code>; difference: camera minus declared; "
                f"largest stall: the largest <code>wall_time</code> step between consecutive hold "
                f"frames."))
    rep.add(f'<h2 class="newpage">Appendix B: all {nt} transitions</h2>')
    t = rep.next_table()
    srt = sorted(trans, key=lambda r: (r["to_grasp"], int(r["take"])))
    rep.add(table(
        [[th("To grasp", rowspan=2), th("Take", True, rowspan=2), th("Validity", rowspan=2),
          '<th class="group" colspan="3">Movement start</th>',
          '<th class="group" colspan="3">Settled</th>',
          th("Duration difference (ms)", True, rowspan=2)],
         [th("metadata (s)", True), th("camera (s)", True), th("residual (ms)", True),
          th("metadata (s)", True), th("camera (s)", True), th("residual (ms)", True)]],
        [[brk(r["to_grasp"]), r["take"], html.escape(r["validity"]),
          fx(num(r["movement_start_metadata_s"]), 3), fx(num(r["movement_start_camera_s"]), 6),
          fx(num(r["movement_start_capture_residual_ms"]), 3),
          fx(num(r["settled_metadata_s"]), 3), fx(num(r["settled_camera_s"]), 6),
          fx(num(r["settled_capture_residual_ms"]), 3),
          fx(ms(num(r["transition_duration_difference_s"])), 3)] for r in srt],
        num_cols={1, 3, 4, 5, 6, 7, 8, 9}, tid="appendix-b", cls="small",
        widths=("56mm", "7mm", "17mm"),
        caption=f"<strong>Table {t}.</strong> All {nt} transition captures, by target grasp then "
                f"take. Metadata: seconds from the first frame's capture time; camera: the "
                f"camera-clock time of the frame the instant falls on; residual: that frame's "
                f"<code>capture_time</code> minus the instant; duration difference: camera minus "
                f"metadata. Blank: the metadata gives no movement start or settle."))
    rep.add(f'<p class="footer">{html.escape(FOOTER)}</p>')

    page = (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(TITLE)}</title><style>{CSS}</style></head>\n<body><main>\n"
            + "\n".join(rep.parts) + "\n</main></body></html>\n")
    return page, rep.n_fig


# --- the run ---------------------------------------------------------------------------------
def _path_key(path):
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path)))).casefold()


def _is_under(path, root):
    p, r = _path_key(path), _path_key(root)
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


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
        # Edge 156 returns from the launcher before the PDF is written (about a
        # second later on this machine), so wait for the file and a stable size.
        deadline, size = time.monotonic() + 60, -1
        while time.monotonic() < deadline:
            now = pdf_path.stat().st_size if pdf_path.is_file() else -1
            if now > 0 and now == size:
                break
            size = now
            time.sleep(0.5)
    return exe


def run(args):
    folder = Path(args.erratum)
    out = Path(args.out) if args.out else folder
    if args.dataset is not None and _is_under(out, args.dataset):
        print(f"refusing to write inside the dataset: {out} is inside {args.dataset}")
        return 2
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        print(f"REPORT FAIL: {folder} has no {', '.join(missing)}")
        return 1
    static, trans = read_csv(folder / "static_takes.csv"), read_csv(folder / "transitions.csv")
    readme = (folder / "README.md").read_text(encoding="utf-8")
    summary = (folder / "summary.md").read_text(encoding="utf-8")
    vpath = folder / "VERIFICATION.md"
    verification = vpath.read_text(encoding="utf-8") if vpath.is_file() else None
    verdict = re.search(r"ERRATUM (?:OK|FAIL): static=(\d+) transitions=(\d+) "
                        r"static_interval_over_one_period=(\d+)", summary)
    k = len(flagged_rows(static))
    if not verdict or tuple(map(int, verdict.groups())) != (len(static), len(trans), k):
        print(f"REPORT FAIL: summary.md's verdict line "
              f"({verdict.group(0) if verdict else 'missing'}) disagrees with the CSVs "
              f"(static={len(static)} transitions={len(trans)} over one period={k})")
        return 1
    plt.rcParams.update(RC)
    try:
        page, figures = build(args, static, trans, readme, summary, verification)
    except LookupError as e:
        print(f"REPORT FAIL: {e}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
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
            print(f"REPORT FAIL: {exe} did not print {pdf_path}")
            return 1
        else:
            print(f"wrote {pdf_path} ({exe.name})")
            pdf = "yes"
    print(f"REPORT OK: static={len(static)} transitions={len(trans)} over_one_period={k} "
          f"figures={figures} pdf={pdf}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Write the camera-clock erratum as a paper-style report (report.html and "
                    "report.pdf); every number is read from the erratum's files.")
    p.add_argument("--erratum", type=Path, required=True,
                   help="the erratum folder (README.md, static_takes.csv, transitions.csv, "
                        "summary.md, optional VERIFICATION.md)")
    p.add_argument("--dataset", type=Path,
                   help="the organized dataset root (read only); adds the outlier figure")
    p.add_argument("--out", type=Path, help="output folder (default: the erratum folder)")
    p.add_argument("--no-pdf", action="store_true", help="write report.html only")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
