"""The 50 single-plane glove poses: their CSV, their recorder pose list and their note."""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "protocols" / "glove_planar_poses_50.csv"
JSON_PATH = REPO / "protocols" / "glove_planar_poses_50.json"
DOC_PATH = REPO / "docs" / "glove_planar_poses_50.md"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _module():
    return _load("glove_planar_poses", REPO / "scripts" / "leap" / "glove_planar_poses.py")


def _rows():
    assert CSV_PATH.is_file(), "run scripts/leap/glove_planar_poses.py first"
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- csv --------------------------------------------------------------------------------------
def test_csv_has_50_unique_poses_with_setup_tests_and_glove_note():
    rows = _rows()
    assert len(rows) == 50
    for key in ("id", "name", "signature"):
        assert len({r[key] for r in rows}) == 50, f"duplicate {key}"
    for r in rows:
        assert r["setup"].startswith("gloves on, no camera"), r["id"]
        assert r["plane"].startswith("each digit moves only in its own flexion-extension plane")
        assert r["tests"] and r["glove_note"] and r["description"], r["id"]
        assert re.fullmatch(r"p[sdfrcmt]\d\d", r["id"]), r["id"]


def test_csv_every_pose_is_single_plane():
    mod = _module()
    for r in _rows():
        assert r["spread_state"] == "together", r["id"]
        assert r["wrist_state"] == "neutral", r["id"]
        assert r["thumb_state"] in mod.PLANAR_THUMB, r["id"]
        assert r["thumb_target"] == "", r["id"]
        for k in ("index_state", "middle_state", "ring_state", "little_state"):
            assert r[k] not in mod.GLOVE_FLOOR_STATES, (r["id"], k)


def test_csv_excludes_database_and_camera_catalogue_poses():
    mod = _module()
    excluded = mod.EXCLUDED_SIGNATURES
    for r in _rows():
        assert r["signature"] not in excluded, r["name"]
    # the exclusion set holds both kinds
    E, F, H = "ext", "flex", "half"
    assert "|".join([F, F, F, F, "together", "up", "", "neutral"]) in excluded        # thumbs up (recorded)
    assert "|".join([E, E, E, E, "together", "across", "", "neutral"]) in excluded    # Set C thumb step
    assert "|".join([H, H, H, H, "together", "up", "", "neutral"]) in excluded        # catalogue C11
    assert "|".join(["tabletop"] * 4 + ["together", "along", "", "neutral"]) in excluded  # catalogue C04
    assert "|".join(["hook"] * 4 + ["together", "up", "", "neutral"]) in excluded     # hook grasp (recorded)


def test_csv_ranges_are_the_catalogue_templates_inside_the_envelope():
    mod = _module()
    pc = mod.pc
    by_id = {p.id: p for p in mod.poses()}
    for r in _rows():
        expect = pc.ranges(by_id[r["id"]])
        for col in pc.ANGLE_COLUMNS:
            lo, hi = float(r[f"{col}_min"]), float(r[f"{col}_max"])
            assert (lo, hi) == tuple(float(v) for v in expect[col]), (r["id"], col)
            e = pc.ENVELOPE[pc.envelope_key(col)]
            assert e[0] <= lo < hi <= e[1], (r["id"], col)


def test_csv_states_match_the_signature():
    for r in _rows():
        sig = "|".join([r["index_state"], r["middle_state"], r["ring_state"], r["little_state"],
                        r["spread_state"], r["thumb_state"], r["thumb_target"], r["wrist_state"]])
        assert sig == r["signature"], r["id"]
        assert r["finger_shape"] == "|".join(sig.split("|")[:4]), r["id"]
        assert "camera_twin" not in r


def test_csv_no_finger_shape_is_in_the_leap_list_or_the_database():
    """The strict rule: the four finger states alone, read from the two CSVs and
    from the catalogue's database list, independently of the generator's check."""
    mod = _module()
    F = ("index_state", "middle_state", "ring_state", "little_state")
    leap_csv = REPO / "protocols" / "pose_catalogue_100.csv"
    with leap_csv.open(newline="", encoding="utf-8") as fh:
        leap = {tuple(r[k] for k in F) for r in csv.DictReader(fh)}
    assert len(leap) > 50
    database = {tuple(s.split("|")[:4]) for s in mod.pc.DATABASE_SIGNATURES}
    shapes = [tuple(r[k] for k in F) for r in _rows()]
    assert len(set(shapes)) == 50, "two glove poses share a finger shape"
    for r, shape in zip(_rows(), shapes):
        assert shape not in leap, (r["id"], r["name"], "finger shape is in the Leap list")
        assert shape not in database, (r["id"], r["name"], "finger shape is a recorded pose's")
    # positive controls: shapes the rule must catch
    E, T, F4 = "ext", "tabletop", ("flex",) * 4
    assert (T, E, E, E) in leap            # Leap B03, index tabletop
    assert (E, E, E, E) in leap and F4 in leap
    assert F4 in database and ("tipcurl", E, E, E) in database   # fist; pinch family
    assert mod.EXCLUDED_FINGER_SHAPES == leap | database


def test_csv_groups_cover_the_planned_tests():
    rows = _rows()
    groups = {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    assert groups == {"single finger": 2, "pair": 12, "three fingers": 7, "split": 9,
                      "coupling": 8, "mixed depth": 6, "thumb": 6}
    assert all("coupling" in r["tests"] for r in rows if r["group"] == "coupling")
    thumbs = {r["thumb_state"] for r in rows}
    assert {"up", "ip_bent", "mcp_bent", "across"} <= thumbs


# --- json -------------------------------------------------------------------------------------
def test_json_pose_list_matches_the_csv_and_loads_in_the_recorder():
    assert JSON_PATH.is_file()
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    ids = [r["id"] for r in _rows()]
    assert [it["id"] for it in data["items"]] == ids
    assert all(it["label"] and it["shape"] and it["orientation"] for it in data["items"])
    assert data["takes_per_item"] == 3 and data["setup"].startswith("gloves on")
    rp = _load("record_poses_for_planar_test", REPO / "scripts" / "leap" / "record_poses.py")
    protocol, digest = rp.load_protocol(JSON_PATH)
    assert [it["id"] for it in protocol["items"]] == ids
    assert len(digest) == 64


def test_json_recording_set_is_10_of_the_50_and_keeps_every_kind_of_test():
    """13 Nov 2026: only 10 are recorded (25 earlier that day). They are poses of
    the list, in its order, with every group, every finger state and the three
    thumb-joint tests, and each has a stated reason for being in the set."""
    mod = _module()
    rows = _rows()
    chosen = [r for r in rows if r["recording_set"] == "yes"]
    assert len(chosen) == 10 and {r["recording_set"] for r in rows} == {"yes", "no"}
    assert [r["id"] for r in chosen] == list(mod.RECORDING_SET)
    assert set(mod.RECORDING_REASON) == set(mod.RECORDING_SET)
    assert all(mod.RECORDING_REASON[i] for i in mod.RECORDING_SET)
    states = ("index_state", "middle_state", "ring_state", "little_state")
    assert {r["group"] for r in chosen} == {r["group"] for r in rows}
    assert {r[k] for r in chosen for k in states} == {r[k] for r in rows for k in states}
    assert {"up", "ip_bent", "mcp_bent", "across"} <= {r["thumb_state"] for r in chosen}
    data = json.loads((REPO / "protocols" / "glove_planar_poses_recording_set.json")
                      .read_text(encoding="utf-8"))
    assert data["name"] == "glove_planar_poses_recording_set"
    assert "10 static" in data["description"]
    # the count is in no file name, and the 25-pose files of earlier that day are gone
    assert not (REPO / "protocols" / "glove_planar_poses_25.json").exists()
    assert not (REPO / "launchers" / "record_glove_25_poses.cmd").exists()
    assert not (REPO.parent / "record_glove_25_poses.cmd").exists()
    assert [it["id"] for it in data["items"]] == [r["id"] for r in chosen]
    full = {it["id"]: it for it in json.loads(JSON_PATH.read_text(encoding="utf-8"))["items"]}
    assert all(it == full[it["id"]] for it in data["items"]), "the same items as in the 50"


def test_no_camera_in_the_setup_of_the_note_the_csv_and_both_pose_lists():
    """13 Nov 2026, the user's correction: the glove recording uses no camera.
    The Leap (camera) catalogue is still named as the other, separate list."""
    for r in _rows():
        assert r["setup"].startswith("gloves on, no camera"), r["id"]
        assert "only the camera" not in r["glove_note"], r["id"]
    for name in ("glove_planar_poses_50.json", "glove_planar_poses_recording_set.json"):
        data = json.loads((REPO / "protocols" / name).read_text(encoding="utf-8"))
        assert data["setup"].startswith("gloves on, no camera"), name
        assert "gloves on, no camera" in data["description"], name
        assert "scripts/glove/record_poses.py" in data["description"], name
        for it in data["items"]:
            assert it["orientation"].startswith("gloves on, no camera"), it["id"]
    text = DOC_PATH.read_text(encoding="utf-8")
    setup = text.split("## Recording setup", 1)[1].split("## The 10 to record", 1)[0]
    assert "Setup for every step: gloves on, no camera" in setup
    assert "uses no camera at all" in setup
    for needle in ("record_simultaneous", "--camera", "Ultraleap", "camera running",
                   "toward the camera", "camera still"):
        assert needle not in text, needle


# --- document ---------------------------------------------------------------------------------
def _doc() -> str:
    assert DOC_PATH.is_file()
    return DOC_PATH.read_text(encoding="utf-8")


def test_document_has_the_required_sections():
    headings = [ln.strip("# ").strip().lower() for ln in _doc().splitlines() if ln.startswith("#")]
    for needle in ("what this is", "why one plane", "how to read the ranges", "what is excluded",
                   "recording setup", "the 50 poses"):
        assert any(needle in h for h in headings), needle


def test_document_table_lists_every_id_once_and_names_setup_and_sources():
    text = _doc()
    ids = [r["id"] for r in _rows()]
    assert re.findall(r"^\| (p[sdfrcmt]\d\d) \|", text, flags=re.M) == ids
    assert "finger shape" in text and "twin" not in text.lower()
    assert "gloves on, no camera" in text
    recorded = [r["id"] for r in _rows() if r["recording_set"] == "yes"]
    assert "scripts\\glove\\record_poses.py" in text and ",".join(recorded) in text
    assert "## The 10 to record" in text and "glove_planar_poses_25" not in text
    assert re.findall(r"^\| \d+ \| `(p[sdfrcmt]\d\d)` \|", text, flags=re.M) == recorded
    # the recording command is the launcher's, with the XR Train screenshot
    assert "launchers\\record_glove_poses.cmd" in text
    assert "--pose-list protocols\\glove_planar_poses_recording_set.json" in text
    assert '--app-screenshot "XR Train"' in text and "xr_train" in text
    launcher = (REPO / "launchers" / "record_glove_poses.cmd").read_text(encoding="utf-8")
    command = next(ln for ln in text.splitlines() if "glove\\record_poses.py --hand" in ln).strip()
    head, tail = command.split(" --takes ", 1)
    assert head in launcher and "--takes " + tail.split(" --app-screenshot")[0] in launcher
    assert '--app-screenshot "XR Train"' in launcher and "%START%" in launcher
    for name in ("Dragusanu", "Isaac", "Ingram", "Chen", "Cobos", "Jia", "MyoHand"):
        assert name in text


# --- pdf --------------------------------------------------------------------------------------
def test_the_pdf_is_written_when_its_path_is_relative(tmp_path, monkeypatch):
    mod = _load("glove_planar_poses_pdf", REPO / "scripts" / "leap" / "glove_planar_poses_pdf.py")
    if next((b for b in mod.BROWSERS if b.is_file()), None) is None:
        pytest.skip("no Edge or Chrome on this machine to print the PDF")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "page.html").write_text(
        "<!doctype html><html><head><title>one line</title></head><body><p>one line</p></body></html>",
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert mod.print_pdf(Path("out") / "page.html", Path("out") / "page.pdf") is not None
    pdf = tmp_path / "out" / "page.pdf"
    assert pdf.is_file() and pdf.read_bytes()[:4] == b"%PDF"
