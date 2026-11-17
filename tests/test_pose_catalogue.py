"""The 100-pose catalogue: its CSV, its exclusions and its note."""
from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "protocols" / "pose_catalogue_100.csv"
DOC_PATH = REPO / "docs" / "pose_catalogue_100.md"


def _module():
    spec = importlib.util.spec_from_file_location(
        "pose_catalogue", REPO / "scripts" / "leap" / "pose_catalogue.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pose_catalogue"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows():
    assert CSV_PATH.is_file(), "run scripts/leap/pose_catalogue.py first"
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- csv --------------------------------------------------------------------------------------
def test_csv_has_100_unique_poses():
    rows = _rows()
    assert len(rows) == 100
    for key in ("id", "name", "signature"):
        values = [r[key] for r in rows]
        assert len(set(values)) == 100, f"duplicate {key}"
    assert all(r["setup"].startswith("camera only, bare hand") for r in rows)
    assert all(r["leap_note"].startswith(("good", "fair")) for r in rows)
    assert all(r["description"] for r in rows)


def test_csv_excludes_every_database_pose():
    mod = _module()
    db = mod.DATABASE_SIGNATURES
    for r in _rows():
        assert r["signature"] not in db, r["name"]
    # the exclusion list itself holds the database poses it is meant to hold
    E, F = "ext", "flex"
    assert "|".join([E, E, E, E, "spread", "out", "", "neutral"]) in db          # open palm
    assert "|".join([F, F, F, F, "together", "across", "", "neutral"]) in db     # fist
    assert "|".join([E, F, F, F, "neutral", "across", "", "neutral"]) in db      # index point
    assert "|".join([F, F, F, F, "together", "up", "", "neutral"]) in db         # thumbs up
    assert "|".join([E, E, F, F, "spread", "across", "", "neutral"]) in db       # peace
    assert "|".join([E, E, E, F, "neutral", "across", "", "neutral"]) in db      # three
    assert "|".join(["half", E, E, E, "neutral", "oppose", "index", "neutral"]) in db  # pinch
    assert "|".join([E, E, E, E, "spread", "out", "", "flexed"]) in db           # bent wrist
    assert "|".join([E, F, E, E, "neutral", "out", "", "neutral"]) in db         # sequence: middle alone
    assert "|".join([F, F, F, E, "neutral", "across", "", "neutral"]) in db      # sequence: thumb+index+middle+ring
    assert "|".join([E, E, F, F, "spread", "out", "", "neutral"]) in db          # sequence: ring+little (ASL 3)
    assert "|".join(["hook", "hook", "hook", "hook", "neutral", "along", "", "neutral"]) in db  # hook grasp


def test_csv_ranges_are_the_templates_and_lie_in_the_envelope():
    mod = _module()
    by_id = {p.id: p for p in mod.poses()}
    for r in _rows():
        pose = by_id[r["id"]]
        expect = mod.ranges(pose)
        for col in mod.ANGLE_COLUMNS:
            lo, hi = float(r[f"{col}_min"]), float(r[f"{col}_max"])
            assert (lo, hi) == tuple(float(v) for v in expect[col]), (r["id"], col)
            e = mod.ENVELOPE[mod.envelope_key(col)]
            assert e[0] <= lo < hi <= e[1], (r["id"], col, lo, hi, e)


def test_csv_states_match_the_signature():
    for r in _rows():
        sig = "|".join([r["index_state"], r["middle_state"], r["ring_state"], r["little_state"],
                        r["spread_state"], r["thumb_state"], r["thumb_target"], r["wrist_state"]])
        assert sig == r["signature"], r["id"]


def test_csv_covers_every_template_state_at_least_once():
    mod = _module()
    rows = _rows()
    used_flex = {r[k] for r in rows for k in ("index_state", "middle_state", "ring_state", "little_state")}
    assert used_flex == set(mod.FLEX)
    assert {r["thumb_state"] for r in rows} == set(mod.THUMB)
    assert {r["wrist_state"] for r in rows} == set(mod.WRIST)
    assert {r["spread_state"] for r in rows} == set(mod.ABD)


# --- document ---------------------------------------------------------------------------------
def _doc() -> str:
    assert DOC_PATH.is_file()
    return DOC_PATH.read_text(encoding="utf-8")


def test_document_has_the_required_sections():
    headings = [ln.strip("# ").strip().lower() for ln in _doc().splitlines() if ln.startswith("#")]
    for needle in ("what this is", "how to read the ranges", "what is excluded", "recording setup", "the 100 poses"):
        assert any(needle in h for h in headings), needle


def test_document_table_lists_every_id_once():
    text = _doc()
    ids = [r["id"] for r in _rows()]
    seen = re.findall(r"^\| ([A-G]\d\d) \|", text, flags=re.M)
    assert seen == ids


def test_document_names_the_envelope_sources_and_the_setup():
    text = _doc()
    for name in ("Dragusanu", "Isaac", "Ingram", "Chen", "Cobos", "Jia", "MyoHand"):
        assert name in text
    assert "camera only, bare hand" in text
    assert "positive toward the thumb" in text


# --- both hands -------------------------------------------------------------------------------
def test_the_both_hands_protocol_has_every_pose_and_its_picture():
    import json
    path = REPO / "protocols" / "pose_catalogue_100_both.json"
    proto = json.loads(path.read_text(encoding="utf-8"))
    assert [i["id"] for i in proto["items"]] == [r["id"] for r in _rows()]
    folder = (path.parent / proto["images_dir"]).resolve()
    if not folder.is_dir():
        # the pictures are rendered into results/, which is not in the repository
        pytest.skip("the pose pictures are not here: run scripts/leap/pose_catalogue_render.py")
    assert all((folder / i["image"]).is_file() for i in proto["items"])
