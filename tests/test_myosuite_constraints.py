"""The MyoSuite constraint comparison: its two CSVs and its note.

The CSVs are written by scripts/leap/myosuite_hand_limits.py in the MyoSuite
venv (.venv-myo); these tests read what it wrote and need neither mujoco nor
myosuite. The note is docs/myosuite_constraints.md.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from leap_hand.paper3 import TABLE5
from xr_hand.joint_frames import CAMERA_NOT_GIVEN, DOF24_NAMES

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "myosuite_constraints"
LIMITS = OUT / "myohand_joint_limits.csv"
COMPARISON = OUT / "comparison_24_angles.csv"
DOC = REPO / "docs" / "myosuite_constraints.md"

FINGER_CMC = tuple(n for n in CAMERA_NOT_GIVEN if n.endswith("_CMC"))


def _rows(path: Path) -> list[dict]:
    assert path.is_file(), f"{path} missing: run scripts/leap/myosuite_hand_limits.py in .venv-myo"
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- csv -------------------------------------------------------------------------------
def test_csv_joint_limits_one_row_per_joint_in_degrees():
    rows = _rows(LIMITS)
    assert list(rows[0]) == ["joint", "type", "limited", "min_deg", "max_deg",
                             "span_deg", "axis", "body"]
    assert len(rows) >= 20, "the MyoSuite hand has at least 20 joints"
    names = [r["joint"] for r in rows]
    assert len(set(names)) == len(names)
    for r in rows:
        if r["limited"] == "yes":
            lo, hi, span = (float(r[k]) for k in ("min_deg", "max_deg", "span_deg"))
            assert lo < hi, r
            assert span == pytest.approx(hi - lo, abs=0.02)
            if r["type"] == "hinge":
                assert -360 < lo < hi < 360, "a hinge range in degrees, not radians"
        else:
            assert r["min_deg"] == r["max_deg"] == r["span_deg"] == ""
        assert len(r["axis"].split()) == 3


def test_csv_comparison_has_the_24_angles_in_paper_order():
    comp = _rows(COMPARISON)
    assert [r["angle"] for r in comp] == DOF24_NAMES
    assert list(comp[0]) == [
        "angle", "myosuite_joint", "myosuite_min_deg", "myosuite_max_deg",
        "myosuite_span_deg", "paper3_table5_range", "paper3_table5_hyperextension",
        "glove_row", "glove_min_deg", "glove_max_deg"]


def test_csv_comparison_myosuite_cells_come_from_the_limits_csv():
    limits = {r["joint"]: r for r in _rows(LIMITS)}
    for r in _rows(COMPARISON):
        if r["angle"] in FINGER_CMC:
            assert r["myosuite_joint"] == "", "MyoSuite has no finger CMC joint"
            assert r["myosuite_min_deg"] == r["myosuite_max_deg"] == ""
            continue
        j = limits[r["myosuite_joint"]]
        assert j["limited"] == "yes"
        assert (r["myosuite_min_deg"], r["myosuite_max_deg"], r["myosuite_span_deg"]) == (
            j["min_deg"], j["max_deg"], j["span_deg"])
        assert float(r["myosuite_min_deg"]) < float(r["myosuite_max_deg"])


def test_csv_comparison_paper3_cells_are_table5_as_printed():
    for r in _rows(COMPARISON):
        t5 = TABLE5[r["angle"]]
        own = t5["abduction_range"] if r["angle"].endswith("_aa") else t5["flexion_range"]
        assert r["paper3_table5_range"] == (own or "")
        assert r["paper3_table5_hyperextension"] == (t5["hyperextension"] or "")


def test_csv_comparison_glove_cells_are_numbers_or_empty():
    for r in _rows(COMPARISON):
        assert re.fullmatch(r"[A-Z_]+\.(flex|abd|twist)_deg", r["glove_row"])
        for k in ("glove_min_deg", "glove_max_deg"):
            if r[k] != "":
                float(r[k])
        assert (r["glove_min_deg"] == "") == (r["glove_max_deg"] == "")


# --- document --------------------------------------------------------------------------
def _doc() -> str:
    assert DOC.is_file()
    return DOC.read_text(encoding="utf-8")


def test_document_has_the_required_sections():
    headings = [ln.strip("# ").strip().lower()
                for ln in _doc().splitlines() if ln.startswith("#")]
    for needle in ("what the two tools are", "what we can do", "what we cannot",
                   "what the joint comparison shows", "how to reproduce", "next step"):
        assert any(needle in h for h in headings), f"no heading about '{needle}'"


def test_document_reproduce_section_has_the_one_command():
    text = _doc()
    sec = text.lower().split("how to reproduce", 1)[1]
    assert "myosuite_hand_limits.py" in sec
    assert ".venv-myo" in sec


def test_document_table_matches_the_comparison_csv():
    text = _doc()
    comp = {r["angle"]: r for r in _rows(COMPARISON)}
    seen = 0
    for ln in text.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] in comp:
            r = comp[cells[0]]
            myo = cells[1]
            if r["myosuite_joint"]:
                assert f"{float(r['myosuite_min_deg']):.0f}" in myo and \
                    f"{float(r['myosuite_max_deg']):.0f}" in myo, (cells[0], myo)
            else:
                assert myo in ("", "–", "-", "none", "no joint"), (cells[0], myo)
            seen += 1
    assert seen == 24, "the note tabulates every one of the 24 angles"
