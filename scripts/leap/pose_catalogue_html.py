r"""One HTML page of the 100-pose catalogue with the MyoHand renders beside each pose.

Reads docs/pose_catalogue_100.md (the note the generator wrote) and
results/pose_catalogue/renders/ (the images pose_catalogue_render.py wrote)
and writes results/pose_catalogue/pose_catalogue_100.html, which references
the renders by relative path, so the folder can be served or opened as is.

    .venv\Scripts\python.exe scripts\leap\pose_catalogue_html.py

The images are renders of MyoSuite's MyoHand at the catalogue's template
angles, not photographs; the page says so above the table.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "pose_catalogue_100.md"
RENDERS = REPO / "results" / "pose_catalogue" / "renders"
OUT = REPO / "results" / "pose_catalogue" / "pose_catalogue_100.html"

CSS = """
body{font-family:Arial,Helvetica,sans-serif;font-size:14px;margin:24px;color:#111}
h1{font-size:22px} h2{font-size:17px;margin-top:28px} p{max-width:900px;line-height:1.45}
code{background:#f1f1f1;padding:0 3px;border-radius:3px}
table{border-collapse:collapse;margin:8px 0} th,td{border:1px solid #bbb;padding:3px 6px;vertical-align:top;text-align:left}
th{background:#eee;position:sticky;top:0} .small td,.small th{font-size:13px}
.poses{font-size:11.5px} .poses td:nth-child(3){min-width:200px} .poses td{min-width:110px}
.poses tbody tr:nth-child(even){background:#fafafa} .wrap{overflow-x:auto}
.pics{white-space:nowrap} .pics img{width:170px;height:170px;border:1px solid #ddd;background:#fff;margin-right:4px}
.note{background:#fff7e0;border:1px solid #e6c766;padding:8px 12px;max-width:900px}
"""


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def table_html(lines, with_pictures: bool) -> str:
    rows = [r for r in lines if not re.fullmatch(r"\|[-| ]+\|", r.strip())]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, rest = cells[0], cells[1:]
    if with_pictures:
        head = head[:1] + ["MyoHand render (palm view, side view)"] + head[1:]
    h = "<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead>"
    body = []
    for r in rest:
        tds = [f"<td>{inline(c)}</td>" for c in r]
        if with_pictures:
            pid = r[0]
            pics = "".join(
                f'<img src="renders/{pid}_{v}.png" alt="{pid} {v} view" title="{pid} {v} view">'
                for v in ("palm", "side") if (RENDERS / f"{pid}_{v}.png").is_file())
            tds.insert(1, f'<td class="pics">{pics or "no render"}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    cls = "poses" if with_pictures else "small"
    return f'<div class="wrap"><table class="{cls}">{h}<tbody>{"".join(body)}</tbody></table></div>'


def build(doc_text: str) -> str:
    out, table, para = [], [], []
    in_poses = False

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    def flush_table():
        nonlocal in_poses
        if table:
            out.append(table_html(table, with_pictures=in_poses))
            table.clear()

    for ln in doc_text.splitlines():
        if ln.startswith("|"):
            flush_para()
            table.append(ln)
            continue
        flush_table()
        if ln.startswith("#"):
            flush_para()
            level = len(ln) - len(ln.lstrip("#"))
            title = ln.strip("# ").strip()
            out.append(f"<h{level}>{inline(title)}</h{level}>")
            if re.match(r"the [0-9]+ poses", title.lower()):
                in_poses = True
                out.append('<p class="note">The pictures are not photographs and not recorded takes. '
                           'Each is MyoSuite\'s MyoHand model posed at the midpoint of '
                           'the pose\'s template ranges (joint values clamped to the model\'s own limits, '
                           'so a full curl shows PIP 90; the finger MCP alone is let back to -40 for the '
                           'picture, so a finger pressed back is drawn pressed back), with the thumb base '
                           'placed by eye from the pose\'s words, seen from '
                           'the palm side as the Leap would see it and from the thumb side. Rendered by '
                           '<code>scripts/leap/pose_catalogue_render.py</code>.</p>')
        elif ln.strip() == "":
            flush_para()
        else:
            para.append(ln.strip())
    flush_para()
    flush_table()
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>100 poses</title>"
            f"<style>{CSS}</style></head><body>{''.join(out)}</body></html>")


def main() -> int:
    page = build(DOC.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    n_img = len(list(RENDERS.glob("*_palm.png"))) if RENDERS.is_dir() else 0
    print(f"HTML OK: {OUT} renders={n_img}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
