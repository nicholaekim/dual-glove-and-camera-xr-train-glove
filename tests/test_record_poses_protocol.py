"""`scripts/leap/record_poses.py --protocol`, end to end on the mock.

The session folder is a contract (docs/protocol_formats.md, sections 1, 3,
5 and 7) that the checker and the packager are written against without
reading this recorder, so these tests read the folder the way they will:
file names, the keys on every frame line, every meta field, the keypoint
block, and session.json as it grows take by take. The rejection path and
the operator's decisions are driven through the same main() the operator
runs, with a scripted key sequence standing in for the keyboard.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from cam_hand.prof_format import load_file

REPO = Path(__file__).resolve().parents[1]
ITEMS = ["cylindrical", "tip_pinch", "hook"]
META_FIELDS = {
    "item", "take", "hand", "frames", "tracked_fraction", "reacquisitions",
    "tracker_hand_labels", "static_interval", "medoid_wall_time",
    "grab_strength", "pinch_strength", "curls", "orientation_note",
    "accepted", "reason", "decided_at",
}
SESSION_FIELDS = {
    "set", "protocol_file", "protocol_name", "protocol_version",
    "protocol_sha256", "hand", "operator", "camera", "glove", "started",
    "ended", "xr_trainer_calibrated_at", "seed", "rounds", "tool_commit",
    "takes",
}
TAKE_RE = r"(?P<item>[a-z_]+)_(?P<hand>left|right)_take(?P<n>\d+)_\d{8}_\d{6}"


def load_script():
    path = REPO / "scripts" / "leap" / "record_poses.py"
    spec = importlib.util.spec_from_file_location("leap_script_record_poses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rp(monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "beep", lambda *a, **k: None)      # quiet, fast
    monkeypatch.setattr(module.os, "startfile", lambda *a, **k: None,
                        raising=False)
    return module


@pytest.fixture
def protocol_file(tmp_path: Path) -> Path:
    """A small grasps.json with the contract's schema: three items."""
    path = tmp_path / "grasps_fixture.json"
    path.write_text(json.dumps({
        "name": "grasps", "version": 1,
        "status": "placeholder until the reference papers are in hand",
        "description": "test fixture",
        "takes_per_item": 3, "duration_s": 5.0, "prep_s": 5.0,
        "items": [
            {"id": "cylindrical", "label": "cylindrical grasp", "source": None,
             "figure": None, "shape": "all fingers wrap a vertical cylinder",
             "object_implied": True},
            {"id": "tip_pinch", "label": "tip pinch", "source": None,
             "figure": None, "shape": "thumb tip to index tip",
             "object_implied": False},
            {"id": "hook", "label": "hook grasp", "source": None,
             "figure": None, "shape": "fingers bent at the middle joints",
             "object_implied": True},
        ]}), encoding="utf-8")
    return path


def run(rp, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["record_poses.py", *map(str, argv)])
    rp.main()


def session_dir(out: Path) -> Path:
    folders = [p for p in out.iterdir() if p.is_dir()]
    assert len(folders) == 1, folders
    return folders[0]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def mock_args(protocol_file, out, *extra):
    return ["--mock", "--protocol", protocol_file, "--hand", "left",
            "--duration", "1", "--prep", "0", "--no-open", "--out-dir", out,
            *extra]


# --- the accepted path ------------------------------------------------------------
def test_mock_session_writes_the_contract_folder(rp, monkeypatch, tmp_path,
                                                  protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--note", "palm down, no turn"))
    folder = session_dir(out)
    assert re.fullmatch(r"\d{8}_\d{6}_left", folder.name)

    session = read_json(folder / "session.json")
    assert SESSION_FIELDS <= set(session)
    assert session["set"] == "grasps" and session["hand"] == "left"
    assert session["camera"] == "leap" and session["glove"] is False
    assert session["protocol_name"] == "grasps" and session["protocol_version"] == 1
    assert len(session["protocol_sha256"]) == 64
    assert session["ended"] is not None
    assert [t["item"] for t in session["takes"]] == ITEMS
    assert all(t["accepted"] and t["reason"] == "" for t in session["takes"])

    for entry in session["takes"]:
        name = entry["name"]
        m = re.fullmatch(TAKE_RE, name)
        assert m and m["item"] == entry["item"] and m["hand"] == "left" and m["n"] == "1"
        files = entry["files"]
        assert files["leap"] == f"leap/{name}.jsonl"
        assert files["still"] == f"stills/{name}.png"
        assert files["keypoints"] == f"keypoints/{name}_keypoints.txt"
        assert files["meta"] == f"meta/{name}.json"
        assert files["glove"] is None and files["events"] is None   # camera only
        for rel in files.values():
            assert rel is None or (folder / rel).is_file(), rel

        # every frame line: the coached recorder's keys plus session/item/take
        lines = read_lines(folder / files["leap"])
        assert lines
        for line in lines:
            assert line["session"] == folder.name
            assert line["item"] == entry["item"]
            assert line["take"] == 1
            assert {"wall_time", "timestamp", "packet_counter", "hand_side",
                    "frame_id", "status", "joints", "hand_id",
                    "grab_strength", "pinch_strength", "abs26"} <= set(line)
        # the mock shows both hands and the file keeps both
        assert {line["hand_side"] for line in lines} == {"left", "right"}

        meta = read_json(folder / files["meta"])
        assert META_FIELDS <= set(meta)
        assert meta["item"] == entry["item"] and meta["take"] == 1
        assert meta["hand"] == "left" and meta["accepted"] is True
        assert meta["frames"] == len(lines)
        assert meta["tracked_fraction"] >= 0.9
        assert meta["tracker_hand_labels"] == {
            side: sum(1 for x in lines if x["hand_side"] == side)
            for side in ("left", "right")}
        assert set(meta["curls"]) == {"thumb", "index", "middle", "ring", "pinky"}
        assert 0.0 <= meta["grab_strength"] <= 1.0
        assert 0.0 <= meta["pinch_strength"] <= 1.0
        assert meta["orientation_note"] == "palm down, no turn"
        assert meta["decided_at"] == entry["decided_at"]
        assert meta["decided_by"] == "auto"
        assert meta["still_missing_reason"] is None
        # the orientation actually held: the mock hovers 25 cm up
        assert meta["palm_height_cm"] == pytest.approx(25.0, abs=1.0)
        assert 0.0 <= meta["view_angle_deg"] <= 180.0


def test_the_static_interval_and_the_summary_lie_inside_the_take(
        rp, monkeypatch, tmp_path, protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook"))
    folder = session_dir(out)
    meta = read_json(next((folder / "meta").glob("*.json")))
    lines = read_lines(next((folder / "leap").glob("*.jsonl")))
    times = [x["wall_time"] for x in lines]
    t0, t1 = meta["static_interval"]
    assert min(times) - 1e-6 <= t0 <= t1 <= max(times) + 1e-6
    assert meta["take_start"] <= t0 and t1 <= meta["take_end"]
    assert t0 <= meta["medoid_wall_time"] <= t1
    medoid = lines[meta["medoid_line"]]
    assert medoid["wall_time"] == meta["medoid_wall_time"]
    assert medoid["grab_strength"] == pytest.approx(meta["grab_strength"], abs=1e-4)


def test_the_keypoint_file_is_one_block_with_the_measured_wrist(
        rp, monkeypatch, tmp_path, protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "cylindrical"))
    folder = session_dir(out)
    meta = read_json(next((folder / "meta").glob("*.json")))
    kp = folder / meta["files"]["keypoints"]
    text = kp.read_text(encoding="utf-8")

    blocks = load_file(kp)                  # the reader of his own files
    assert len(blocks) == 1 and len(blocks[0].points) == 21
    assert blocks[0].frame == meta["medoid_frame_id"]
    landmark_lines = [x for x in text.splitlines() if re.match(r"^\d+ \(", x)]
    assert len(landmark_lines) == 21
    wrist_lines = [x for x in text.splitlines() if x.startswith("Wrist:")]
    assert len(wrist_lines) == 1

    # camera millimetres, and the wrist line is where the wrist actually was
    # (the glove's files force it to zero; these must not)
    wrist = [float(v) for v in re.findall(r"-?\d+\.\d+", wrist_lines[0])]
    lines = read_lines(folder / meta["files"]["leap"])
    measured = [c * 1000.0 for c in lines[meta["medoid_line"]]["abs26"][1]]
    assert wrist == pytest.approx(measured, abs=0.02)
    assert abs(wrist[1]) > 100.0            # 25 cm above the module, not 0


def test_session_json_is_written_after_every_take(rp, monkeypatch, tmp_path,
                                                  protocol_file):
    out = tmp_path / "grasps"
    seen = []
    original = rp.ProtocolSession.write_session

    def spy(self):
        original(self)
        on_disk = read_json(self.session_dir / "session.json")
        seen.append((len(on_disk["takes"]), on_disk["ended"] is not None))

    monkeypatch.setattr(rp.ProtocolSession, "write_session", spy)
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1"))
    # on disk at the start, after each of the three takes, and at the end
    assert seen == [(0, False), (1, False), (2, False), (3, False), (3, True)]


def test_items_and_takes_come_from_the_file_and_the_flags_subset_them(
        rp, monkeypatch, tmp_path, protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--items", "hook", "--takes", "2"))
    session = read_json(session_dir(out) / "session.json")
    assert session["items"] == ["hook"]
    assert session["takes_per_item"] == 2 and session["duration_s"] == 1.0
    assert [(t["item"], t["take"]) for t in session["takes"]] == [("hook", 1), ("hook", 2)]
    names = [t["name"] for t in session["takes"]]
    assert len(set(names)) == 2


def test_still_none_writes_no_still_and_says_why(rp, monkeypatch, tmp_path,
                                                 protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--still", "none"))
    folder = session_dir(out)
    meta = read_json(next((folder / "meta").glob("*.json")))
    assert meta["still"] is None and meta["still_missing_reason"] == rp.STILL_OFF
    assert not list(folder.glob("stills/*.png"))


def test_the_mock_still_is_cropped_to_the_hand(rp, monkeypatch, tmp_path,
                                               protocol_file):
    import cv2

    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook"))
    still = next((session_dir(out) / "stills").glob("*.png"))
    img = cv2.imread(str(still))
    assert img is not None
    assert img.shape[0] < 768 or img.shape[1] < 768, "cropped, not the whole frame"
    assert min(img.shape[:2]) >= 192


# --- the rejection path ------------------------------------------------------------
def test_a_take_under_90_percent_tracked_is_rejected_with_its_reason(
        rp, monkeypatch, tmp_path, protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "cylindrical",
                                    "--retries", "1", "--mock-dropout", "0.3"))
    folder = session_dir(out)
    session = read_json(folder / "session.json")
    takes = session["takes"]
    assert len(takes) == 2, "one attempt plus one retry"
    for entry in takes:
        assert entry["accepted"] is False
        assert entry["decided_by"] == "gate"
        assert "tracked" in entry["reason"] and "90 %" in entry["reason"]
        assert entry["tracked_fraction"] < 0.9
        name = entry["name"]
        # a rejected attempt keeps the number it was trying to become
        assert re.fullmatch(TAKE_RE, name)["n"] == "1"
        assert entry["files"]["leap"] == f"rejected/leap/{name}.jsonl"
        assert entry["files"]["meta"] == f"rejected/meta/{name}.json"
        assert entry["files"]["still"] == f"rejected/stills/{name}.png"
        assert entry["files"]["reason"] == f"rejected/{name}.reason.txt"
        for rel in entry["files"].values():
            if rel:
                assert (folder / rel).is_file(), rel
        reason = (folder / "rejected" / f"{name}.reason.txt").read_text(encoding="utf-8")
        assert entry["reason"] in reason
        meta = read_json(folder / entry["files"]["meta"])
        assert meta["accepted"] is False and meta["reason"] == entry["reason"]
        assert META_FIELDS <= set(meta)
    # nothing left behind in the accepted folders
    for kind in ("leap", "stills", "keypoints", "meta"):
        d = folder / kind
        assert not d.is_dir() or not any(d.iterdir()), kind


def test_the_operator_can_redo_a_take_and_the_decision_is_timed(
        rp, monkeypatch, tmp_path, protocol_file):
    real = rp.Reviewer
    monkeypatch.setattr(rp, "Reviewer",
                        lambda auto=False: real(keys=iter(["r", "\r"])))
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--takes", "1",
                                    "--items", "tip_pinch", "--retries", "0"))
    folder = session_dir(out)
    takes = read_json(folder / "session.json")["takes"]
    assert [(t["accepted"], t["reason"], t["decided_by"]) for t in takes] == [
        (False, "operator redo", "operator"), (True, "", "operator")]
    assert takes[0]["decided_at"] < takes[1]["decided_at"]
    assert takes[0]["files"]["leap"].startswith("rejected/leap/")
    assert takes[1]["files"]["leap"].startswith("leap/")
    # a redo is not a gate failure: --retries 0 did not stop the second attempt
    meta = read_json(folder / takes[1]["files"]["meta"])
    assert meta["accepted"] is True and meta["decided_by"] == "operator"
    assert meta["attempt"] == 2


def test_q_at_the_review_ends_the_session_and_keeps_the_record(
        rp, monkeypatch, tmp_path, protocol_file):
    real = rp.Reviewer
    monkeypatch.setattr(rp, "Reviewer", lambda auto=False: real(keys=iter(["q"])))
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--takes", "1"))
    session = read_json(session_dir(out) / "session.json")
    assert len(session["takes"]) == 1, "nothing after the quit"
    assert session["takes"][0]["accepted"] is False
    assert "quit" in session["takes"][0]["reason"]
    assert session["ended"] is not None


def test_the_review_window_maps_the_keys(rp, monkeypatch, tmp_path):
    """Enter or space keeps, r redoes, q quits, read from the review window."""
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a, **k: 1.0)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "moveWindow", lambda *a, **k: None)
    for code, action in ((13, "accept"), (32, "accept"), (ord("r"), "redo"),
                         (ord("q"), "quit")):
        keys = iter([-1, -1, code])         # nothing pressed, then the key
        monkeypatch.setattr(cv2, "waitKey", lambda *a, _k=keys: next(_k, -1))
        pumped = []
        got, when, by = rp.Reviewer().decide(None, ["label", "numbers"],
                                             pump=lambda: pumped.append(1))
        assert (got, by) == (action, "operator")
        assert pumped, "the tracker queue is drained while waiting"


def test_the_review_falls_back_to_the_console_without_a_window(rp, monkeypatch):
    import builtins

    import cv2

    def no_gui(*a, **k):
        raise cv2.error("no display")

    monkeypatch.setattr(cv2, "namedWindow", no_gui)
    answers = iter(["x", "r"])              # an unknown key is asked again
    monkeypatch.setattr(builtins, "input", lambda *_a: next(answers))
    got, _when, by = rp.Reviewer().decide(None, ["label"])
    assert (got, by) == ("redo", "operator")


def test_resume_adds_the_missing_takes_to_the_same_session(
        rp, monkeypatch, tmp_path, protocol_file):
    """Plan D9: one session per set, and take N counts kept takes within it."""
    real = rp.Reviewer
    monkeypatch.setattr(rp, "Reviewer",
                        lambda auto=False: real(auto=auto, keys=iter(["\r", "q"])))
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--items", "hook",
                                    "--takes", "2"))
    folder = session_dir(out)
    first = read_json(folder / "session.json")
    assert [(t["take"], t["accepted"]) for t in first["takes"]] == [(1, True), (2, False)]

    # the same folder, the session's own items and timing, numbering carries on
    run(rp, monkeypatch, "--mock", "--protocol", protocol_file, "--hand", "left",
        "--resume", folder, "--auto-accept", "--no-open")
    assert session_dir(out) == folder, "no second session folder"
    session = read_json(folder / "session.json")
    assert [(t["item"], t["take"], t["accepted"]) for t in session["takes"]] == [
        ("hook", 1, True), ("hook", 2, False), ("hook", 2, True)]
    assert session["takes"][:2] == first["takes"], "earlier entries untouched"
    assert session["started"] == first["started"] and len(session["resumed"]) == 1
    assert session["items"] == ["hook"] and session["duration_s"] == 1.0
    kept = [t for t in session["takes"] if t["accepted"]]
    assert all((folder / t["files"]["leap"]).is_file() for t in kept)


def test_resume_refuses_another_hands_session(rp, monkeypatch, tmp_path,
                                              protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--items", "hook", "--takes", "1"))
    folder = session_dir(out)
    with pytest.raises(SystemExit, match="left hand's session"):
        run(rp, monkeypatch, "--mock", "--protocol", protocol_file, "--hand",
            "right", "--resume", folder, "--no-open")


# --- the protocol file and the flags -------------------------------------------------
def test_protocol_needs_the_operators_hand(rp, monkeypatch, tmp_path, protocol_file):
    with pytest.raises(SystemExit):
        run(rp, monkeypatch, "--mock", "--protocol", protocol_file)


def test_protocol_flags_are_refused_without_a_protocol(rp, monkeypatch):
    with pytest.raises(SystemExit):
        run(rp, monkeypatch, "--mock", "--hand", "left")


def test_an_unknown_item_is_refused_by_name(rp, monkeypatch, tmp_path, protocol_file):
    with pytest.raises(SystemExit, match="not in the protocol: nope"):
        run(rp, monkeypatch, *mock_args(protocol_file, tmp_path, "--items", "nope"))


def test_a_sequence_protocol_is_sent_to_its_own_recorder(rp, tmp_path):
    path = tmp_path / "sequences.json"
    path.write_text(json.dumps({"name": "sequences", "version": 1, "items": [
        {"id": "seq1", "label": "Sequence 1", "steps": []}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="record_protocol.py"):
        rp.load_protocol(path)


def test_the_protocol_hash_is_of_the_file_bytes(rp, protocol_file):
    import hashlib

    data, sha = rp.load_protocol(protocol_file)
    assert sha == hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    assert data["takes_per_item"] == 3 and data["duration_s"] == 5.0


# --- the plain session is unchanged --------------------------------------------------
def test_without_protocol_the_plain_session_is_unchanged(rp, monkeypatch, tmp_path):
    out = tmp_path / "poses"
    run(rp, monkeypatch, "--mock", "--poses", "fist", "--takes", "1",
        "--duration", "1", "--prep", "0", "--out-dir", out)
    files = list(out.glob("*.jsonl"))
    assert len(files) == 1
    assert re.fullmatch(r"fist_both_take1_\d{8}_\d{6}\.jsonl", files[0].name)
    lines = read_lines(files[0])
    assert all(x["pose"] == "fist" and x["take"] == 1 for x in lines)
    assert not any("session" in x or "item" in x for x in lines)
    # the 5 frames/s default per hand, not every frame
    assert len(lines) <= 2 * 8
    assert not (out / "session.json").exists()
