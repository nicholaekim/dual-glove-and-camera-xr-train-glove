"""The hand-in folder: layout for all three sets, the professor's text format,
accepted takes only, no images, and missing sessions skipped.

The sessions are the synthetic ones `test_protocol_check.py` builds in the
contract's layout, stills included, so "no photograph goes in" is tested
against a session that has photographs to leave out.
"""
import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

import test_protocol_check as fx
from cam_hand import protocol_check as pc
from cam_hand.prof_format import parse_text

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "package_professor_set", REPO / "scripts" / "package_professor_set.py")
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)

LANDMARK_RE = re.compile(r"^(\d+) \((Palm|THUMB|Index|Middle|Ring|Pinky)\): "
                         r"\((-?\d+\.\d\d), (-?\d+\.\d\d), (-?\d+\.\d\d)\)$")


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("sessions")
    grasps = fx.make_grasp_session(root / "a")
    flexion = fx.make_flexion_session(root / "b", [
        fx.FlexTake("index"),
        fx.FlexTake("middle", peaks=[0.3] * 5, accepted=False,
                    reason="span too small"),
        fx.FlexTake("middle")])
    flexion_right = fx.make_flexion_session(root / "b2", [
        fx.FlexTake("ring")], hand="right", camera=False, stills=False)
    sequences = fx.make_sequence_session(root / "c", [
        fx.SeqTake(), fx.SeqTake(glove_override={3: {"ring": 0.45}},
                                 accepted=False, reason="operator redo"),
        fx.SeqTake()])
    # Set B has a current check.csv; Set C has none, so the packager runs
    # the checker itself.
    pc.write_check(pc.check_session(flexion))
    return {"grasps": grasps, "flexion": flexion,
            "flexion_right": flexion_right, "sequences": sequences}


@pytest.fixture(scope="module")
def package(sessions, tmp_path_factory):
    out = tmp_path_factory.mktemp("handin") / "grasp and flexion set for professor 2026-09-28"
    rc = pkg.main(["--out", str(out),
                   "--grasps", str(sessions["grasps"]),
                   "--flexion", str(sessions["flexion"]),
                   str(sessions["flexion_right"]),
                   "--sequences", str(sessions["sequences"]),
                   "--operator", "N Kim"])
    return rc, out


def blocks_of(path: Path):
    text = path.read_text(encoding="utf-8")
    return [b.splitlines() for b in text.strip("\n").split("\n\n")]


def check_block_shape(block):
    assert block[0].startswith("Frame ") and "| Hand ID: " in block[0]
    assert re.match(r"^Wrist: \(-?\d+\.\d\d, -?\d+\.\d\d, -?\d+\.\d\d\)$",
                    block[1]), block[1]
    marks = [LANDMARK_RE.match(line) for line in block[2:]]
    assert len(block) == 23 and all(marks), block
    assert [int(m.group(1)) for m in marks] == list(range(21))


def test_package_runs(package):
    rc, out = package
    assert rc == 0
    assert (out / "README.txt").is_file()


def test_grasps_layout(package, sessions):
    _rc, out = package
    g = out / "grasps"
    names = sorted(p.name for p in g.iterdir())
    expected = set()
    for item in ("cylindrical", "tip_pinch"):
        for n in (1, 2, 3):
            expected |= {f"{item}_left_take{n}.jsonl",
                         f"{item}_left_take{n}_keypoints.txt"}
        expected.add(f"{item}_left_all_frames.txt")
    expected.add("grasps_summary.csv")
    assert set(names) == expected
    # the medoid file is the recorder's own, copied unchanged
    src = sessions["grasps"] / "keypoints"
    (kp,) = [p for p in src.iterdir() if p.name.startswith(
        "cylindrical_left_take1_")]
    assert (g / "cylindrical_left_take1_keypoints.txt").read_text(
        encoding="utf-8") == kp.read_text(encoding="utf-8")
    with open(g / "grasps_summary.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert {"grasp", "take", "frames", "tracked_percent", "grab_strength",
            "pinch_strength", "curl_index"} <= set(rows[0])
    assert rows[0]["tracked_percent"] == "98"
    # no `gate` block in these metas: the take's own count is the gate's
    assert rows[0]["reacquisitions_in_static_interval"] == "0"
    assert {r["label"] for r in rows} == {"cylindrical grasp", "tip pinch"}
    # times keep their digits, straight from the meta file
    meta = json.loads(next((sessions["grasps"] / "meta").glob(
        "cylindrical_left_take1_*.json")).read_text(encoding="utf-8"))
    assert float(rows[0]["static_interval_start"]) == pytest.approx(
        meta["static_interval"][0], abs=1e-6)
    assert float(rows[0]["medoid_wall_time"]) == pytest.approx(
        meta["medoid_wall_time"], abs=1e-6)


def test_grasp_all_frames_are_camera_millimetres(package):
    _rc, out = package
    path = out / "grasps" / "cylindrical_left_all_frames.txt"
    blocks = blocks_of(path)
    assert len(blocks) == 3 * 90        # three takes, the left hand only
    for b in blocks[:5] + blocks[-5:]:
        check_block_shape(b)
        assert "Hand ID: left" in b[0]
    # the wrist is where the camera measured it, not forced to zero
    wrist = [float(v) for v in re.findall(r"-?\d+\.\d\d", blocks[0][1])]
    assert abs(wrist[1]) > 100.0
    assert len(list(parse_text(path.read_text(encoding="utf-8")))) == 270


def test_rejected_takes_and_stills_stay_out(package):
    _rc, out = package
    files = [p for p in out.rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix.lower() in
                pkg.IMAGE_EXTENSIONS]
    assert pkg.images_in(out) == []
    names = " ".join(p.name for p in files)
    # the rejected flexion attempt and the rejected sequence attempt
    assert "middle_left_take1_20260928_140001" not in names
    assert "seq4_pairs_left_take2_20260928_140001" not in names
    assert "tip_pinch_left_take2_20260928_140004" not in names


def test_flexion_layout_and_text_export(package, sessions):
    _rc, out = package
    left = out / "finger_flexion" / "left"
    takes = [t for t in pc.load_session(sessions["flexion"]).takes
             if t.accepted]
    assert len(takes) == 2
    for t in takes:
        for name in (f"{t.name}.jsonl", f"camera_{t.name}.jsonl",
                     f"{t.name}.events.jsonl", f"{t.name}.txt"):
            assert (left / name).is_file(), name
        # the JSONL is copied as recorded
        assert (left / f"{t.name}.jsonl").read_bytes() == \
            t.path("glove").read_bytes()
        blocks = blocks_of(left / f"{t.name}.txt")
        n_frames = sum(1 for line in t.path("glove").read_text(
            encoding="utf-8").splitlines() if line.strip())
        assert len(blocks) == n_frames
        for b in blocks[:3] + blocks[-3:]:
            check_block_shape(b)
            # glove text files: wrist at the origin, even though the mock
            # glove streams its wrist 12 cm off it
            assert b[1] == "Wrist: (0.00, 0.00, 0.00)"
    assert (left / "flexion_report.txt").is_file()
    report = (left / "flexion_report.txt").read_text(encoding="utf-8")
    assert "check.csv and check.txt in the session folder" in report
    assert "transfer curve" in report and "lag" in report
    # glove-only right hand: no camera file, still a report
    right = out / "finger_flexion" / "right"
    assert not list(right.glob("camera_*"))
    assert (right / "flexion_report.txt").is_file()
    assert len(list(right.glob("*.events.jsonl"))) == 1


def test_sequences_layout_and_check_csv(package, sessions):
    _rc, out = package
    left = out / "sequences" / "left"
    assert len(list(left.glob("camera_*.jsonl"))) == 2
    assert len(list(left.glob("*.events.jsonl"))) == 2
    with open(left / "sequence_check.csv", encoding="utf-8",
              newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 7                       # accepted takes only
    assert {r["glove_pass"] for r in rows} == {"pass"}
    assert {r["camera_pass"] for r in rows} == {"pass"}
    assert rows[5]["flexed"] == "thumb index middle ring pinky"
    assert float(rows[5]["glove_index"]) > 0.9
    # coupling sits beside pass/fail; the clean fixture has none
    assert {"glove_coupling", "camera_coupling"} <= set(rows[0])
    assert {r["glove_coupling"] for r in rows} == {""}
    # the checker ran in memory: nothing was written into the session
    assert not (sessions["sequences"] / "check.csv").exists()


def test_sequence_check_csv_carries_coupling(tmp_path):
    session = fx.make_sequence_session(tmp_path / "s", [fx.SeqTake(
        glove_override={1: {"middle": 0.45}})], camera=False)
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--sequences", str(session)]) == 0
    with open(out / "sequences" / "left" / "sequence_check.csv",
              encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["glove_pass"] == "pass"
    assert re.fullmatch(r"middle 0\.4\d", rows[1]["glove_coupling"])
    assert rows[1]["camera_pass"] == "" and rows[1]["camera_coupling"] == ""


def test_readme_names_the_setup_and_the_conventions(package):
    _rc, out = package
    text = (out / "README.txt").read_text(encoding="utf-8")
    for needle in ("Operator: N Kim", "camera only, bare hand",
                   "gloves on + camera", "Ultraleap Stereo IR 170",
                   "StretchSense", "wrist at the origin",
                   "the Wrist line giving", "KNOWN GLOVE LIMITS",
                   "0.47 s", "60 %", "No photographs",
                   "2026-09-28", "finger_flexion, right hand",
                   "tracker's own", "20 of 21 poses", "written untouched",
                   "reacquisitions_in_static_interval",
                   "reported as coupling"):
        assert needle in text, needle
    assert "MOCK" not in text
    assert "\u2014" not in text


def _mocked(src: Path, dst: Path, flags: dict) -> Path:
    """A copy of a session whose session.json carries these mock flags."""
    import shutil
    shutil.copytree(src, dst)
    meta = json.loads((dst / "session.json").read_text(encoding="utf-8"))
    meta.update(flags)
    (dst / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    return dst


MOCK_SHAPES = [
    {"mock": True},                                      # Set A, and B/C later
    {"mock": {"glove": True, "glove_follows_cues": True,
              "leap": False}},                          # Sets B and C today
    {"mock": False, "mock_flags": {"glove": False,
                                   "leap": True}},      # Sets B and C later
]


@pytest.mark.parametrize("flags", MOCK_SHAPES)
def test_mock_session_is_refused_unless_allowed(sessions, tmp_path, capsys,
                                                flags):
    mock = _mocked(sessions["flexion"], tmp_path / "20260928_140000_left",
                   flags)
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(mock)]) == 1
    text = capsys.readouterr().out
    assert f"mock session, not handed in: {mock}" in text
    assert not out.exists()
    assert pkg.main(["--out", str(out), "--flexion", str(mock),
                     "--allow-mock"]) == 0
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "MOCK DATA" in readme and "It is not a hand-in." in readme
    assert "MOCK session (synthetic data)" in readme
    assert len(list((out / "finger_flexion" / "left").glob(
        "*.events.jsonl"))) == 2


def test_mock_session_beside_real_ones(sessions, tmp_path, capsys):
    mock = _mocked(sessions["grasps"], tmp_path / "grasps_mock",
                   {"mock": True})
    out = tmp_path / "out"
    rc = pkg.main(["--out", str(out), "--grasps", str(mock),
                   "--flexion", str(sessions["flexion"])])
    assert rc == 0
    assert f"mock session, not handed in: {mock}" in capsys.readouterr().out
    assert not (out / "grasps").exists()
    assert (out / "finger_flexion" / "left" / "flexion_report.txt").is_file()
    assert "MOCK" not in (out / "README.txt").read_text(encoding="utf-8")


def test_mock_flags_that_are_all_false_are_a_real_session(sessions, tmp_path):
    real = _mocked(sessions["flexion"], tmp_path / "20260928_140000_left",
                   {"mock": {"glove": False, "leap": False},
                    "mock_flags": {}})
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(real)]) == 0
    assert "MOCK" not in (out / "README.txt").read_text(encoding="utf-8")


def test_missing_session_is_skipped(sessions, tmp_path, capsys):
    out = tmp_path / "out"
    rc = pkg.main(["--out", str(out),
                   "--flexion", str(sessions["flexion"]),
                   "--sequences", str(tmp_path / "not_recorded_yet"),
                   "--grasps", str(tmp_path)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "skipped sequences:" in text and "does not exist" in text
    assert "skipped grasps: no session.json" in text
    assert (out / "finger_flexion" / "left").is_dir()
    assert not (out / "sequences").exists()
    assert "not included in this folder" in (out / "README.txt").read_text(
        encoding="utf-8")
    assert pkg.main(["--out", str(tmp_path / "none"),
                     "--flexion", str(tmp_path / "nope")]) == 1


def test_an_image_in_the_plan_refuses_the_whole_folder(tmp_path, capsys):
    session = fx.make_flexion_session(tmp_path / "s", [fx.FlexTake("index")])
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    # a recorder bug that points the events entry at the take's still
    meta["takes"][0]["files"]["events"] = meta["takes"][0]["files"]["still"]
    (session / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "out"
    assert pkg.main(["--out", str(out), "--flexion", str(session)]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert not out.exists()


def test_an_image_left_in_the_folder_fails_the_run(sessions, tmp_path,
                                                   capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "photo.JPG").write_bytes(b"\xff\xd8\xff\xe0 fake")
    assert pkg.main(["--out", str(out), "--flexion",
                     str(sessions["flexion"])]) == 1
    assert "photo.JPG" in capsys.readouterr().out
    # and a renamed image is caught by its first bytes
    renamed = tmp_path / "x.txt"
    renamed.write_bytes(b"\x89PNG\r\n\x1a\n...")
    assert pkg.is_image(renamed)
    assert not pkg.is_image(out.parent / "absent.txt")


def test_a_deep_out_folder_past_max_path(sessions, tmp_path):
    """The hand-in folder sits under a long OneDrive path; a take file name
    is 60 characters on its own. Past 260 characters it must still work."""
    out = tmp_path / ("x" * 110) / ("y" * 110) / "handin"
    assert len(str(out)) > 240
    rc = pkg.main(["--out", str(out), "--flexion", str(sessions["flexion"])])
    assert rc == 0
    left = pkg.os_path(out / "finger_flexion" / "left")
    assert len(list(left.glob("camera_*.jsonl"))) == 2
    assert (left / "flexion_report.txt").is_file()
    # the final scan sees that deep too: an image hidden at a long path fails
    (left / ("z" * 40 + ".png")).write_bytes(b"\x89PNG fake")
    assert len(pkg.images_in(out)) == 1
    assert pkg.main(["--out", str(out), "--flexion",
                     str(sessions["flexion"])]) == 1
