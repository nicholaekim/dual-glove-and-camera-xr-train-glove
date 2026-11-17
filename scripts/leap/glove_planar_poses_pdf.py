r"""The 50 single-plane glove poses as one PDF, a MyoHand render beside each pose.

The glove list's twin of results/pose_catalogue/pose_catalogue_100.pdf, built
from the same parts so the two read alike:

  1. scripts/leap/pose_catalogue_render.py renders every glove pose on
     MyoSuite's MyoHand (palm view and thumb-side view) into
     results/glove_planar_poses/renders/;
  2. scripts/leap/pose_catalogue_html.py lays the generator's note and the
     renders into results/glove_planar_poses/glove_planar_poses_50.html;
  3. the same page with a print stylesheet (A3 landscape) is printed by
     headless Edge or Chrome to results/glove_planar_poses/glove_planar_poses_50.pdf.

The pictures are renders of a model, not photographs: no glove pose has been
recorded yet. MyoHand is a right hand; a left-hand recording is its mirror
image. Runs in the MyoSuite venv (mujoco, myo_sim, PIL):

    .venv-myo\Scripts\python.exe scripts\leap\glove_planar_poses_pdf.py
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "glove_planar_poses"
BROWSERS = (Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
PRINT_CSS = ("@page{size:A3 landscape;margin:10mm} body{margin:0;font-size:12px} th{position:static} "
             "tr{page-break-inside:avoid} .pics img{width:150px;height:150px} .poses{font-size:10px} "
             "h2{page-break-before:auto} .wrap{overflow:visible}")
RIGHT_HAND = (" The model is a right hand; a left-hand recording is its mirror image. The glove poses "
              "keep the fingers side by side and the wrist straight, so the two views differ only in "
              "what each digit's bend looks like.")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "leap" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def recording_only(doc: str, ids) -> str:
    """The note with its pose table cut to the recording set: the sheet to
    have open while recording (`--recording-set`)."""
    heading = "## The 50 poses"
    if doc.count(heading) != 1:
        raise ValueError(f"the note has {doc.count(heading)} headings {heading!r}, not 1")
    head, table = doc.split(heading, 1)
    title, rest = head.split("\n", 1)
    if not title.startswith("# 50 single-plane poses"):
        raise ValueError(f"unexpected first line of the note: {title!r}")
    head = (f"# The {len(ids)} glove poses to record, from the 50 single-plane poses, "
            "with reference joint ranges\n" + rest)
    kept = [ln for ln in table.splitlines()
            if not (m := re.match(r"\| (p[a-z]\d\d) \|", ln)) or m.group(1) in ids]
    return head + f"## The {len(ids)} poses to record" + "\n".join(kept) + "\n"


def print_pdf(html_path: Path, pdf_path: Path):
    """Print with the first browser found; its path, or None when none exists."""
    exe = next((b for b in BROWSERS if b.is_file()), None)
    if exe is None:
        return None
    pdf_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run([str(exe), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--no-first-run", f"--user-data-dir={profile}",
                        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                       capture_output=True, timeout=600)
        # Edge 156 returns before the PDF is written: wait for the file and a stable size.
        deadline, size = time.monotonic() + 120, -1
        while time.monotonic() < deadline:
            now = pdf_path.stat().st_size if pdf_path.is_file() else -1
            if now > 0 and now == size:
                break
            size = now
            time.sleep(1.0)
    return exe


def page_count(pdf_path: Path) -> int:
    try:
        import pymupdf
        with pymupdf.open(pdf_path) as doc:
            return doc.page_count
    except ImportError:
        return len(re.findall(rb"/Type\s*/Page(?!s)", pdf_path.read_bytes()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--no-render", action="store_true", help="reuse the renders already in --out")
    ap.add_argument("--recording-set", action="store_true",
                    help="only the poses that are recorded: glove_planar_poses_recording_set.pdf")
    args = ap.parse_args(argv)
    gp = _load("glove_planar_poses")
    plist = gp.poses()
    doc, stem = gp.document(plist), "glove_planar_poses_50"
    if args.recording_set:
        doc = recording_only(doc, gp.RECORDING_SET)
        plist = [p for p in plist if p.id in gp.RECORDING_SET]
        stem = gp.RECORDING_NAME
    renders = args.out / "renders"

    # 1. the renders, through the catalogue's render script with the glove poses
    if not args.no_render:
        render = _load("pose_catalogue_render")

        class GlovePoses:
            poses = staticmethod(gp.poses)
            ranges = staticmethod(gp.pc.ranges)

        render.load_catalogue = lambda: GlovePoses
        if render.main(["--out", str(renders), "--size", str(args.size)]) != 0:
            print("GLOVE POSES PDF FAIL: the render step failed")
            return 1
    have = [p.id for p in plist if all((renders / f"{p.id}_{v}.png").is_file() for v in ("palm", "side"))]
    if len(have) != len(plist):
        missing = [p.id for p in plist if p.id not in have]
        print(f"GLOVE POSES PDF FAIL: no render for {', '.join(missing)}")
        return 1

    # 2. the page, through the catalogue's HTML builder
    page = _load("pose_catalogue_html")
    page.RENDERS = renders
    html = page.build(doc)
    html = html.replace("<title>100 poses</title>", f"<title>{len(plist)} glove poses</title>")
    marker = "<code>scripts/leap/pose_catalogue_render.py</code>.</p>"
    if html.count(marker) != 1 or html.count('<td class="pics">') != len(plist):
        print("GLOVE POSES PDF FAIL: the page builder did not place the pose table's pictures")
        return 1
    html = html.replace(marker, marker[:-4] + RIGHT_HAND + "</p>")
    # the builder's note is the camera catalogue's; this list is recorded with no camera
    html = html.replace("seen from the palm side as the Leap would see it", "seen from the palm side")
    args.out.mkdir(parents=True, exist_ok=True)
    html_path = args.out / f"{stem}.html"
    html_path.write_text(html, encoding="utf-8")
    print_path = args.out / f"{stem}_print.html"
    print_path.write_text(html.replace("</style>", PRINT_CSS + "</style>", 1), encoding="utf-8")

    # 3. the PDF
    pdf_path = args.out / f"{stem}.pdf"
    exe = print_pdf(print_path, pdf_path)
    if exe is None:
        print(f"GLOVE POSES PDF FAIL: none of {', '.join(str(b) for b in BROWSERS)} exists")
        return 1
    if not pdf_path.is_file() or pdf_path.read_bytes()[:4] != b"%PDF":
        print(f"GLOVE POSES PDF FAIL: {exe} did not print {pdf_path}")
        return 1
    print(f"wrote {html_path}")
    print(f"wrote {pdf_path} ({exe.name}, {pdf_path.stat().st_size} bytes)")
    print(f"GLOVE POSES PDF OK: poses={len(plist)} renders={2 * len(have)} pages={page_count(pdf_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
