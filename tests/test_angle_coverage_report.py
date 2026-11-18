"""scripts/leap/angle_coverage_report.py: the readable report of which joint
angles the grasp set gives (angle_coverage.csv, report.html, report.pdf).

The dataset is test_replay_verify's synthetic one (three static takes),
given a metadata/joint_angles_24.csv whose values are computed here with
angle_coverage.all_angles on the very poses the JSONL lines were written
from (test_replay_verify.hand_pose), not read back through the script's
path: so the recomputed and the tabled angles agree to float precision, the
six unavailable cells are blank, and a value moved by 1 degree must show up.
"""
import base64
import contextlib
import csv
import html
import importlib.util
import io
import re
import shutil
from pathlib import Path

import pytest

import test_replay_verify as rv_tests
from leap_hand import angle_coverage as ac
from leap_hand import paper3 as p3

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "angle_coverage_report", REPO / "scripts" / "leap" / "angle_coverage_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

HEADINGS = ("Abstract", "1 Background", "2 How each angle is read", "3 The table",
            "4 The finger MCP angles are present",
            "5 What is not given and what it would take", "6 Abduction and adduction",
            "7 Reproduce", "Appendix A")
SIX = ("T_TMC_fe", "T_TMC_aa", "I_CMC", "M_CMC", "R_CMC", "L_CMC")
# The representative frame of each static take of make_dataset and the k its
# line was written with: circular take 1 frames 1000+i at k=i, interval from
# 1003; circular take 2 frames 1100+i at k=6+i, from 1102; prismatic take 1
# frames 2000+i at k=i, from 2002 (representative_frame_id = interval start).
REPRESENTATIVE = {("p3_circular_power", 1): ("right", 3),
                  ("p3_circular_power", 2): ("right", 8),
                  ("p3_prismatic_precision", 1): ("left", 2)}
ANGLE_COLUMNS = ["grasp_name", "hand", "take"] + list(p3.ANGLE_NAMES)


def write_angle_table(root, nudge=None):
    """metadata/joint_angles_24.csv for the synthetic takes; `nudge` =
    (grasp, take, column, degrees) moves one value."""
    rows = []
    for (grasp, take), (side, k) in REPRESENTATIVE.items():
        values = ac.all_angles(*rv_tests.hand_pose(side, k), side)
        row = {"grasp_name": grasp, "hand": side, "take": take}
        for c in p3.ANGLE_NAMES:
            v = values[c]
            if nudge and nudge[:3] == (grasp, take, c):
                v += nudge[3]
            row[c] = "" if v is None else f"{v:.10f}"
        rows.append(row)
    rv_tests.write_table(Path(root) / "metadata" / "joint_angles_24.csv", ANGLE_COLUMNS, rows)
    return root


def report(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = rep.main([str(a) for a in argv])
    return code, buf.getvalue()


def last_line(text):
    return text.strip().splitlines()[-1]


def max_diff(text):
    return float(re.search(r"max_abs_diff_deg=(\S+)", last_line(text)).group(1))


def table_attrs(page, cls):
    """[{attribute: value}] of every <tr class=cls>."""
    return [dict(re.findall(r'data-([a-z-]+)="([^"]*)"', m))
            for m in re.findall(rf'<tr class="{cls}"([^>]*)>', page)]


def images(page):
    return [base64.b64decode(m) for m in re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", page)]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = rv_tests.make_dataset(tmp_path_factory.mktemp("coverage") / "dataset")
    return write_angle_table(root)


def test_the_report_holds_every_section_table_row_and_figure(dataset, tmp_path):
    out = tmp_path / "out"
    code, text = report(["--dataset", dataset, "--out", out, "--no-pdf"])
    assert code == 0, text
    assert re.fullmatch(rf"COVERAGE REPORT OK: entries={len(ac.COVERAGE)} given=18 unavailable=6 "
                        r"max_abs_diff_deg=\S+ pdf=no", last_line(text)), text
    assert max_diff(text) < 1e-6

    with open(out / "angle_coverage.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(ac.COVERAGE) == 47
    assert [r["joint"] for r in rows] == [e.joint for e in ac.COVERAGE]

    page = (out / "report.html").read_text(encoding="utf-8")
    assert rep.TITLE in page and "Nicholas Kim" in page and html.escape(rep.FOOTER) in page
    for heading in HEADINGS:
        assert re.search(rf"<h[1-4][^>]*>{re.escape(heading)}\b", page), heading
    assert page.count('<tr class="entry">') == len(ac.COVERAGE)
    assert page.count('<div class="entry">') == len(ac.COVERAGE)      # Appendix A
    pngs = images(page)
    assert len(pngs) == 2 and all(p.startswith(b"\x89PNG") for p in pngs)
    assert "Of the 24 joint angles of Paper 3, the grasp set gives 18" in page
    # section 2 in the coverage page's own words
    assert "x the hinge axis" in page and "Rx(a) Ry(b) Rz(c)" in page

    compared = table_attrs(page, "cmp")
    assert [a["column"] for a in compared] == rep.COMPARED and len(compared) == 26
    for a in compared:
        assert int(a["n"]) == 3 and int(a["blank"]) == 0, a
        assert float(a["max-abs-diff"]) < 1e-6, a
    six = table_attrs(page, "six")
    assert [a["column"] for a in six] == list(SIX)
    assert all(int(a["blank"]) == len(REPRESENTATIVE) for a in six)
    assert "3 of the 3 takes were recomputed." in page


def test_a_value_moved_by_one_degree_shows_in_its_column(dataset, tmp_path):
    root = tmp_path / "dataset"
    shutil.copytree(dataset, root)
    write_angle_table(root, nudge=("p3_circular_power", 2, "I_MCP_aa", 1.0))
    code, text = report(["--dataset", root, "--out", tmp_path / "out", "--no-pdf"])
    assert code == 0, text
    assert abs(max_diff(text) - 1.0) < 1e-6
    page = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    for a in table_attrs(page, "cmp"):
        if a["column"] == "I_MCP_aa":
            assert abs(float(a["max-abs-diff"]) - 1.0) < 1e-6, a
        else:
            assert float(a["max-abs-diff"]) < 1e-6, a
    assert "Some differences are larger than that." in page


def test_output_inside_the_dataset_is_refused(dataset):
    code, text = report(["--dataset", dataset, "--out", dataset / "report", "--no-pdf"])
    assert code == 2 and "inside the dataset" in text
    assert not (dataset / "report").exists()


def test_a_missing_angle_table_fails(dataset, tmp_path):
    root = tmp_path / "dataset"
    shutil.copytree(dataset, root)
    (root / "metadata" / "joint_angles_24.csv").unlink()
    code, text = report(["--dataset", root, "--out", tmp_path / "out", "--no-pdf"])
    assert code == 1 and "joint_angles_24.csv does not exist" in text
    assert not (tmp_path / "out" / "report.html").exists()


def test_no_browser_leaves_the_html_and_says_so(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(rep, "BROWSERS", (tmp_path / "no-such-browser.exe",))
    code, text = report(["--dataset", dataset, "--out", tmp_path / "out"])
    assert code == 0, text
    assert "PDF NOT MADE" in text and last_line(text).endswith("pdf=no")
    assert (tmp_path / "out" / "report.html").is_file()
    assert not (tmp_path / "out" / "report.pdf").exists()


def test_the_pdf_is_printed_when_a_browser_exists(dataset, tmp_path):
    if next((b for b in rep.BROWSERS if b.is_file()), None) is None:
        pytest.skip("no Edge or Chrome on this machine to print the PDF")
    code, text = report(["--dataset", dataset, "--out", tmp_path / "out"])
    assert code == 0, text
    pdf = tmp_path / "out" / "report.pdf"
    assert pdf.is_file() and pdf.read_bytes()[:4] == b"%PDF"
    assert last_line(text).endswith("pdf=yes")


def test_pdf_is_written_when_out_is_given_as_a_relative_path(dataset, tmp_path, monkeypatch):
    if next((b for b in rep.BROWSERS if b.is_file()), None) is None:
        pytest.skip("no Edge or Chrome on this machine to print the PDF")
    monkeypatch.chdir(tmp_path)
    code, text = report(["--dataset", dataset, "--out", "out"])
    assert code == 0, text
    pdf = tmp_path / "out" / "report.pdf"
    assert pdf.is_file() and pdf.read_bytes()[:4] == b"%PDF"
    assert last_line(text).endswith("pdf=yes")
