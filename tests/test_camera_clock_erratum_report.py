"""scripts/leap/camera_clock_erratum_report.py: the human-readable report
(report.html, report.pdf) of an erratum folder.

The erratum folder is built here from test_camera_clock_erratum's synthetic
dataset by the erratum script itself, so the report is checked against the
files it reads: the counts are taken from summary.md's verdict line and the
row counts from the two CSVs, never typed in.
"""
import base64
import contextlib
import csv
import importlib.util
import io
import re
import shutil
from pathlib import Path

import pytest

import test_camera_clock_erratum as ce_tests

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "camera_clock_erratum_report", REPO / "scripts" / "leap" / "camera_clock_erratum_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

HEADINGS = ("Abstract", "1 Background", "2 Method", "3 Results", "3.1 Static takes",
            "3.2 The outlier", "3.3 Transitions", "4 How to apply", "5 Verification",
            "6 Reproduce", "Appendix A", "Appendix B")
PNG = "data:image/png;base64,"


def report(argv):
    """(exit code, printed text) of the report script's main()."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = rep.main([str(a) for a in argv])
    return code, buf.getvalue()


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def table_rows(page, table_id):
    """Body rows of the table with this id."""
    body = re.search(rf'<table id="{table_id}".*?<tbody>(.*?)</tbody>', page, re.S)
    assert body, table_id
    return body.group(1).count("<tr")


def images(page):
    return [base64.b64decode(m) for m in re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", page)]


@pytest.fixture(scope="module")
def erratum(tmp_path_factory):
    """(dataset root, erratum folder) written by the erratum script."""
    base = tmp_path_factory.mktemp("report")
    root, _ = ce_tests.make_dataset(base / "dataset")
    code, text = ce_tests.run(["--dataset", root, "--out", base / "erratum"])
    assert code == 0, text
    return root, base / "erratum"


def copy_of(erratum_dir, tmp_path):
    dst = tmp_path / "erratum"
    shutil.copytree(erratum_dir, dst)
    return dst


def test_html_holds_the_report(erratum, tmp_path):
    _, folder = erratum
    folder = copy_of(folder, tmp_path)
    code, text = report(["--erratum", folder, "--no-pdf"])
    assert code == 0, text
    page = (folder / "report.html").read_text(encoding="utf-8")

    summary = (folder / "summary.md").read_text(encoding="utf-8")
    verdict = re.search(r"ERRATUM OK: static=(\d+) transitions=(\d+) "
                        r"static_interval_over_one_period=(\d+)", summary)
    n_static, n_trans, n_over = verdict.groups()
    assert rep.TITLE in page and "Nicholas Kim" in page
    assert verdict.group(0) in page
    assert f"{n_over} of the {n_static} static takes" in page
    assert f"{n_trans} transition captures" in page
    assert text.strip().splitlines()[-1] == (
        f"REPORT OK: static={n_static} transitions={n_trans} over_one_period={n_over} "
        f"figures=3 pdf=no")
    for heading in HEADINGS:
        assert re.search(rf"<h[1-4][^>]*>{re.escape(heading)}\b", page), heading

    # F1, F2, F4 embedded as PNG; F3 needs --dataset, and the report says so
    pngs = images(page)
    assert len(pngs) == 3 and all(p.startswith(b"\x89PNG") for p in pngs)
    assert "was skipped: no --dataset was given" in page

    assert table_rows(page, "appendix-a") == len(rows(folder / "static_takes.csv"))
    assert table_rows(page, "appendix-b") == len(rows(folder / "transitions.csv"))
    assert table_rows(page, "over-one-period") == int(n_over)

    # the How-to-apply wording comes from the README
    assert "Keep the dataset&#x27;s tables as they are" in page
    assert "in place of" in page and "static_interval_camera_s" in page.replace("<wbr>", "")
    assert "No verification note" in page
    assert rep.FOOTER in page


def test_verification_note_is_rendered(erratum, tmp_path):
    _, folder = erratum
    folder = copy_of(folder, tmp_path)
    (folder / "VERIFICATION.md").write_text(
        "# Verification before sending\n\nChecked by hand.\n\n## 1. Claims\n\n"
        "| Claim | Recomputed |\n|---|---|\n| takes over one period | 7 of 9 checked |\n\n"
        "- every file compared\n", encoding="utf-8")
    code, text = report(["--erratum", folder, "--no-pdf"])
    assert code == 0, text
    page = (folder / "report.html").read_text(encoding="utf-8")
    assert "Verification before sending" in page and "Checked by hand." in page
    assert "<td>7 of 9 checked</td>" in page
    assert "<li>every file compared</li>" in page
    assert "No verification note" not in page


def test_out_folder_and_pdf(erratum, tmp_path):
    browser = next((b for b in rep.BROWSERS if b.is_file()), None)
    if browser is None:
        pytest.skip("no Edge or Chrome on this machine to print the PDF")
    _, folder = erratum
    code, text = report(["--erratum", folder, "--out", tmp_path / "out"])
    assert code == 0, text
    pdf = tmp_path / "out" / "report.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert pdf.read_bytes()[:4] == b"%PDF"
    assert text.strip().splitlines()[-1].endswith("pdf=yes")
    assert not (folder / "report.html").exists()      # --out, not the erratum folder


def test_no_browser_leaves_the_html_and_says_so(erratum, tmp_path, monkeypatch):
    _, folder = erratum
    monkeypatch.setattr(rep, "BROWSERS", (tmp_path / "no-such-browser.exe",))
    code, text = report(["--erratum", folder, "--out", tmp_path / "out"])
    assert code == 0, text
    assert "PDF NOT MADE" in text and text.strip().splitlines()[-1].endswith("pdf=no")
    assert (tmp_path / "out" / "report.html").is_file()
    assert not (tmp_path / "out" / "report.pdf").exists()


def test_dataset_draws_the_outlier_figure_when_its_take_loads(erratum, tmp_path):
    """The take with the largest difference (p3_circular_power take 1 here)
    is in the synthetic dataset: the figure is drawn from its JSONL lines."""
    root, folder = erratum
    code, text = report(["--erratum", folder, "--dataset", root, "--out", tmp_path, "--no-pdf"])
    assert code == 0, text
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert len(images(page)) == 4
    assert "was skipped" not in page
    assert "p3_circular_power take 1" in page
    # the stall the caption names is the CSV's frame pair, measured from the lines
    (circ,) = [r for r in rows(folder / "static_takes.csv") if r["grasp_name"] == "p3_circular_power"]
    assert f"between frames {circ['stall_frame_id_before']} and {circ['stall_frame_id_after']}" in page
    assert text.strip().splitlines()[-1].endswith("figures=4 pdf=no")


def test_dataset_without_the_take_skips_the_figure(erratum, tmp_path):
    root, folder = erratum
    dataset = tmp_path / "dataset"
    shutil.copytree(root, dataset)
    (dataset / ce_tests.fx.CIRC1).unlink()
    code, text = report(["--erratum", folder, "--dataset", dataset, "--out", tmp_path / "out",
                         "--no-pdf"])
    assert code == 0, text
    page = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert len(images(page)) == 3
    assert "was skipped: p3_circular_power take 1 could not be loaded from the dataset" in page
    assert text.strip().splitlines()[-1].endswith("figures=3 pdf=no")


def test_output_inside_the_dataset_is_refused(erratum):
    root, folder = erratum
    code, text = report(["--erratum", folder, "--dataset", root, "--out", root / "report",
                         "--no-pdf"])
    assert code == 2 and "inside the dataset" in text
    assert not (root / "report").exists()


def test_a_missing_erratum_file_fails(erratum, tmp_path):
    _, folder = erratum
    folder = copy_of(folder, tmp_path)
    (folder / "transitions.csv").unlink()
    code, text = report(["--erratum", folder, "--no-pdf"])
    assert code == 1 and "transitions.csv" in text
    assert not (folder / "report.html").exists()
