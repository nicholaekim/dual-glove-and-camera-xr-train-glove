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
import math
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


class Gui:
    """OpenCV's window calls, recorded instead of opening windows."""

    def __init__(self):
        self.calls = []          # (call, window name)
        self.shown = []          # the window name of every imshow
        self.last = {}           # window name -> the last image shown in it
        self.keys = []           # codes pollKey and waitKey return, then -1


@pytest.fixture
def gui(monkeypatch):
    """No test opens a real window: the COPY THIS window, on by default,
    is drawn into this recorder instead."""
    import cv2

    g = Gui()

    def record(name):
        def call(*a, **_k):
            g.calls.append((name, a[0] if a else None))
        return call

    for name in ("namedWindow", "setWindowProperty", "moveWindow",
                 "resizeWindow", "destroyWindow"):
        monkeypatch.setattr(cv2, name, record(name))

    def imshow(win, img):
        g.shown.append(win)
        g.last[win] = img

    def key(*_a):
        return g.keys.pop(0) if g.keys else -1

    monkeypatch.setattr(cv2, "imshow", imshow)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "pollKey", key)
    monkeypatch.setattr(cv2, "waitKey", key)
    return g


@pytest.fixture
def rp(monkeypatch, gui):
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
    assert session["recorder_rules"] == rp.RECORDER_RULES
    assert session["recorder_rules_changes"] == []
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
        # the rules it was recorded under, so a reader can tell these takes
        # from those recorded before 2026-10-07
        assert meta["recorder_rules"] == rp.RECORDER_RULES == "2026-10-07"
        assert meta["coaching"]["forming_lost_count"] == 0
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
        assert "tracked" in entry["reason"] and "90 percent" in entry["reason"]
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


LOSS_REASON_RE = re.compile(
    r"tracked (?P<pct>\d+) percent: lost (?P<n>\d+) times?, "
    r"(?:for|longest) (?P<secs>\d+\.\d) s(?: \(not back by the end\))? "
    r"with the hand at (?P<cm>\d+) cm \((?P<cause>[a-z ]+(?: \([^)]*\))?): (?P<fix>[^)]+)\); "
    r"the gate needs 90 percent")


def test_a_mock_dropout_reject_reason_names_the_losses_and_the_meta_lists_them(
        rp, monkeypatch, tmp_path, protocol_file):
    """The reason says how many losses, the longest, where the hand was and
    why; `gate.losses` in the meta holds every one of them."""
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--retries", "0", "--mock-dropout", "0.3"))
    folder = session_dir(out)
    entry = read_json(folder / "session.json")["takes"][0]
    assert entry["accepted"] is False and entry["decided_by"] == "gate"
    m = LOSS_REASON_RE.fullmatch(entry["reason"])
    assert m, entry["reason"]
    assert int(m["pct"]) == round(entry["tracked_fraction"] * 100)
    # the mock hand hovers 25 cm up, over the module, palm to the lens: none
    # of the measured causes fits, and the reason says so rather than guessing
    assert m["cm"] == "25" and m["cause"] == "unexplained"

    meta = read_json(folder / entry["files"]["meta"])
    gate = meta["gate"]
    assert gate["reason"] == entry["reason"] == meta["reason"]
    assert gate["loss_gap_s"] == pytest.approx(0.1)
    losses = gate["losses"]
    assert len(losses) == int(m["n"]) >= 1
    longest = max(losses, key=lambda d: d["duration_s"])
    assert f"{longest['duration_s']:.1f}" == m["secs"]
    # --mock-dropout 0.3 drops 27 frames of every 90: a 0.3 s hole
    assert longest["duration_s"] == pytest.approx(0.3, abs=0.03) or not longest["recovered"]
    for d in losses:
        assert d["height_cm"] == pytest.approx(25.0, abs=1.0)
        assert d["cause"] == d["causes"][0] and d["fix"]
        assert 0.0 <= d["start_s"] <= meta["duration_s"] + 0.1
    reason_txt = (folder / entry["files"]["reason"]).read_text(encoding="utf-8")
    assert reason_txt.splitlines()[0] == entry["reason"]


def test_a_kept_take_lists_no_losses_in_the_meta(rp, monkeypatch, tmp_path,
                                                 protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook"))
    meta = read_json(next((session_dir(out) / "meta").glob("*.json")))
    assert meta["accepted"] is True and meta["gate"]["reason"] == ""
    assert meta["gate"]["losses"] == []


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


# --- the picture to copy: the COPY THIS window ---------------------------------------
GREEN = (40, 200, 60)                       # BGR of the fixture picture
RED = (0, 0, 255)


def fixture_picture(path: Path, size=(300, 450)):
    """A stand-in for a paper's panel: green, a red square near its top left."""
    import cv2
    import numpy as np

    img = np.full((size[1], size[0], 3), GREEN, np.uint8)
    cv2.rectangle(img, (20, 20), (120, 120), RED, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)
    return img


@pytest.fixture
def with_pictures(tmp_path, protocol_file):
    """The fixture protocol with `images_dir` and an `image` per item, and a
    picture for each in a temporary folder (never the private papers folder)."""
    data = read_json(protocol_file)
    folder = tmp_path / "panels"
    data["images_dir"] = str(folder)
    for it in data["items"]:
        it["image"] = f"{it['id']}.png"
        fixture_picture(folder / it["image"])
    path = tmp_path / "grasps_with_pictures.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def spy_frames(rp, monkeypatch):
    """(item id, status, picture given) for every COPY THIS frame drawn."""
    seen = []
    real = rp.compose_copy

    def spy(item, panel, take, takes, status, seconds=None):
        seen.append((item["id"], status, panel is not None))
        return real(item, panel, take, takes, status, seconds)

    monkeypatch.setattr(rp, "compose_copy", spy)
    return seen


def test_the_picture_is_images_dir_joined_with_the_items_image(rp, tmp_path):
    item = {"id": "hook", "label": "Hook", "image": "hook.png"}
    folder = tmp_path / "pictures"
    assert rp.panel_path({"images_dir": str(folder)}, item) == folder / "hook.png"
    # a relative images_dir is read from the protocol file's folder, not the
    # folder the recorder happens to be started in
    proto = tmp_path / "protocols" / "grasps.json"
    assert rp.panel_path({"images_dir": "pictures"}, item, proto) == (
        (tmp_path / "protocols").resolve() / "pictures" / "hook.png")
    # either key absent: no picture, and no guess
    assert rp.panel_path({}, item, proto) is None
    assert rp.panel_path({"images_dir": str(folder)}, {"id": "hook", "label": "Hook"}) is None


def test_the_grasp_list_names_a_picture_for_every_grasp(rp):
    data, _sha = rp.load_protocol(REPO / "protocols" / "grasps.json")
    assert Path(data["images_dir"]).is_absolute()
    for it in data["items"]:
        assert rp.panel_path(data, it) == Path(data["images_dir"]) / f"{it['id']}.png"


def test_the_copy_frame_shows_the_label_the_take_the_countdown_and_the_picture(
        rp, tmp_path):
    import numpy as np

    pic = fixture_picture(tmp_path / "hook.png")             # 300 x 450, tall
    item = {"id": "hook", "label": "Hook (paper 1, Schlesinger)",
            "shape": "four fingers curled into a hook"}
    img, info = rp.compose_copy(item, pic, 2, 3, "GET READY", 2.3)
    assert img.shape[:2] == (rp.COPY_H, rp.COPY_W)
    assert " ".join(info["label"]) == item["label"]
    assert info["lines"][:len(info["label"])] == info["label"]   # drawn first, on top
    assert "take 2/3" in info["lines"]
    assert info["status"] == "GET READY  3" and "GET READY  3" in info["lines"]
    # the picture, 700 px tall, below the band, unchanged but for the scaling
    x, y, w, h = info["panel_box"]
    assert h == rp.COPY_PANEL_H == 700 and w == round(300 * 700 / 450)
    assert y == rp.COPY_BAND_H and 0 <= x and x + w <= rp.COPY_W
    placed = img[y:y + h, x:x + w].astype(int)
    assert np.abs(placed[300:, :] - GREEN).max() <= 2         # below the square
    assert np.abs(placed[60:150, 60:150] - RED).max() <= 2    # the square, top left
    # the label's letters are in the band above the picture, white on dark
    band = img[:rp.COPY_BAND_H]
    assert int((band == 255).all(axis=2).sum()) > 300
    assert not (band == GREEN).all(axis=2).any()
    # during the take: HOLD STILL and the seconds left
    _img, info = rp.compose_copy(item, pic, 1, 3, "HOLD STILL", 2.44)
    assert info["status"] == "HOLD STILL  2.4 s" and "take 1/3" in info["lines"]


def test_no_grasp_label_runs_into_the_status_line(rp):
    """Every real label, long ones on two lines, ends above the take number
    and the countdown, and is drawn whole."""
    data, _sha = rp.load_protocol(REPO / "protocols" / "grasps.json")
    for it in data["items"]:
        for status, secs in (("GET READY", 5.0), ("HOLD STILL", 4.96),
                             ("CHECKING THE TAKE", None)):
            _img, info = rp.compose_copy(it, None, 3, 3, status, secs)
            assert " ".join(info["label"]) == it["label"], it["id"]
            assert len(info["label"]) <= 2, it["id"]
            assert info["label_bottom"] < info["status_top"], (it["id"], status)
            assert info["status_top"] > 0 and info["status_top"] < rp.COPY_BAND_H


def test_a_missing_picture_shows_the_shape_text_and_is_reported_once(
        rp, tmp_path, capsys, gui):
    item = {"id": "hook", "label": "Hook", "image": "hook.png",
            "shape": "four fingers curled into a hook carrying a handle, "
                     "thumb straight and out of the way",
            "source": "p1 Heumer et al. 2007", "figure": "Figure 1, panel 2"}
    window = rp.CopyWindow({"images_dir": str(tmp_path / "nowhere"), "items": [item]})
    assert window.describe().startswith("0 of 1 pictures found")
    for _ in range(3):
        window.show_copy(item, 1, 3, "GET READY", 3.0, force=True)
    out = capsys.readouterr().out
    assert out.count("no picture for hook") == 1
    assert "hook.png is missing" in out
    assert gui.shown == [rp.COPY_WINDOW] * 3

    _img, info = rp.compose_copy(item, window.panel(item), 1, 3, "GET READY", 3.0)
    assert info["panel_box"] is None
    assert info["label"] == ["Hook"]
    assert item["shape"] in " ".join(info["lines"])        # every word, in order
    assert any("Figure 1, panel 2" in line for line in info["lines"])

    # no images_dir at all: said once in the header, not once per grasp
    bare = rp.CopyWindow({"items": [item]})
    assert "no images_dir" in bare.describe()
    assert bare.panel(item) is None
    assert "no picture for" not in capsys.readouterr().out


def test_a_mock_session_shows_each_grasps_picture_in_copy_this(
        rp, monkeypatch, tmp_path, with_pictures, gui, capsys):
    """The plain countdown (--no-coach); the coached take has its own test,
    test_a_coached_session_shows_open_hand_then_the_picture_with_the_hint."""
    import numpy as np

    frames = spy_frames(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(with_pictures, out, "--auto-accept",
                                    "--takes", "1", "--prep", "0.3",
                                    "--no-coach"))
    session = read_json(session_dir(out) / "session.json")
    assert [(t["item"], t["accepted"]) for t in session["takes"]] == [
        (i, True) for i in ITEMS]
    printed = capsys.readouterr().out
    assert "3 of 3 pictures found" in printed and "no picture for" not in printed

    # every grasp: its picture through the countdown and the take, in order
    assert all(found for _iid, _status, found in frames)
    for iid in ITEMS:
        statuses = [s for i, s, _f in frames if i == iid]
        assert statuses[0] == "GET READY" and "HOLD STILL" in statuses
        assert statuses.index("HOLD STILL") > statuses.index("GET READY")
    assert [i for i, _s, _f in frames] == sorted(
        (i for i, _s, _f in frames), key=ITEMS.index)
    # one window, COPY THIS, holding the picture; closed at the end
    assert set(gui.shown) == {rp.COPY_WINDOW}
    assert ("namedWindow", rp.COPY_WINDOW) in gui.calls
    assert gui.calls[-1] == ("destroyWindow", rp.COPY_WINDOW)
    last = gui.last[rp.COPY_WINDOW]
    assert (np.abs(last.astype(int) - GREEN).max(axis=2) <= 2).sum() > 100_000


def test_a_mock_session_runs_without_the_pictures_folder(
        rp, monkeypatch, tmp_path, protocol_file, gui, capsys):
    """No images_dir in the file, then an images_dir whose folder is gone:
    both sessions run, with the shape text in place of the pictures."""
    frames = spy_frames(rp, monkeypatch)
    out = tmp_path / "no_dir"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept", "--takes", "1"))
    assert all(t["accepted"] for t in read_json(session_dir(out) / "session.json")["takes"])
    printed = capsys.readouterr().out
    assert "no images_dir" in printed and "no picture for" not in printed
    assert frames and not any(found for _i, _s, found in frames)
    assert gui.shown and set(gui.shown) == {rp.COPY_WINDOW}

    data = read_json(protocol_file)
    data["images_dir"] = str(tmp_path / "deleted_panels")
    for it in data["items"]:
        it["image"] = f"{it['id']}.png"
    gone = tmp_path / "gone.json"
    gone.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "gone"
    run(rp, monkeypatch, *mock_args(gone, out, "--auto-accept", "--takes", "1"))
    assert all(t["accepted"] for t in read_json(session_dir(out) / "session.json")["takes"])
    printed = capsys.readouterr().out
    assert "0 of 3 pictures found" in printed
    for iid in ITEMS:                         # once per grasp, not once per frame
        assert printed.count(f"no picture for {iid}:") == 1


def test_no_panel_opens_no_window(rp, monkeypatch, tmp_path, with_pictures, gui,
                                  capsys):
    frames = spy_frames(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(with_pictures, out, "--auto-accept",
                                    "--takes", "1", "--no-panel"))
    assert all(t["accepted"] for t in read_json(session_dir(out) / "session.json")["takes"])
    assert "picture: off (--no-panel)" in capsys.readouterr().out
    assert frames == [] and gui.shown == [] and gui.calls == []


def test_no_panel_only_applies_with_a_protocol(rp, monkeypatch):
    with pytest.raises(SystemExit):
        run(rp, monkeypatch, "--mock", "--no-panel")


def test_the_review_appears_in_copy_this_beside_the_paper_and_it_stays_open(
        rp, tmp_path, gui):
    import numpy as np

    fixture_picture(tmp_path / "pics" / "hook.png")
    item = {"id": "hook", "label": "Hook", "image": "hook.png"}
    window = rp.CopyWindow({"images_dir": str(tmp_path / "pics"), "items": [item]})
    window.show_copy(item, 1, 3, "HOLD STILL", 0.5, force=True)
    countdown = gui.last[rp.COPY_WINDOW]

    gui.keys = [-1, -1, ord("r")]              # nothing pressed, then r
    pumped = []
    got, _when, by = rp.Reviewer().decide(
        None, ["Hook   take 1/3, attempt 1", "tracked 100 %"],
        pump=lambda: pumped.append(1), screen=window,
        reference=window.panel(item))
    assert (got, by) == ("redo", "operator")
    assert pumped, "the tracker queue is drained while waiting"
    assert set(gui.shown) == {rp.COPY_WINDOW}, "no second window"
    assert ("destroyWindow", rp.COPY_WINDOW) not in gui.calls
    review = gui.last[rp.COPY_WINDOW]
    assert review.shape == countdown.shape, "the window keeps its size"
    assert not np.array_equal(review, countdown)
    # the paper's picture in the top right, beside where the still goes
    right = review[:400, rp.COPY_W - 260:].astype(int)
    assert (np.abs(right - GREEN).max(axis=2) <= 2).sum() > 20_000
    # and back to the picture for the next take
    window.show_copy(item, 2, 3, "GET READY", 3.0, force=True)
    assert gui.last[rp.COPY_WINDOW].shape == countdown.shape


# --- the coached take: OPEN HAND, MAKE THE GRASP, HOLD STILL --------------------------
HINT = ("turn the forearm so the palm and the fingertips face the camera; "
        "same fingers, different angle")
FINGERTIPS_DOWN = {
    "p1_tip", "p1_lateral", "p2_s04_fingertip_grasp",
    "p2_s05_fingertip_grasp_side_support", "p2_s06_tripod_grasp",
    "p3_circular_precision_thumb_1_finger",
    "p3_circular_precision_thumb_2_fingers", "p3_prismatic_precision",
    "p1_palmar", "p3_circular_precision_thumb_4_fingers"}
# the reason under the rules of 2026-10-07: a hand lost while forming is
# waited for, and rejects the attempt only when it is not found again in time
NOT_FOUND_REASON_RE = re.compile(
    r"lost the hand while forming the grasp and it was not found again "
    r"(?P<why>within the (?P<wait>[\d.]+) s wait for it|before the forming "
    r"window ended, (?P<window>[\d.]+) s after the GO beep): at "
    r"(?P<cm>\d+) cm with the palm (?P<deg>\d+) degrees from the lens "
    r"\((?P<cause>[a-z ]+): (?P<fix>.+)\)")


def actor_with(monkeypatch, **options):
    """Every `CoachedActor` the mock builds gets these options on top of the
    recorder's; the actors built are returned, to read what they acted."""
    import leap_hand.mock as mock

    real = mock.CoachedActor
    made = []

    def build(**kw):
        actor = real(**{**kw, **options})
        made.append(actor)
        return actor

    monkeypatch.setattr(mock, "CoachedActor", build)
    return made


def spy_timeline(rp, monkeypatch):
    """Every COPY THIS frame in order: ("copy", item, status, picture given,
    info) from compose_copy, ("open", item, status, lost, info) from
    compose_open."""
    seen = []
    real_copy, real_open = rp.compose_copy, rp.compose_open

    def copy(item, panel, take, takes, status, seconds=None):
        img, info = real_copy(item, panel, take, takes, status, seconds)
        seen.append(("copy", item["id"], status, panel is not None, info))
        return img, info

    def open_(item, take, takes, state, seconds=None, lost=False,
              status=rp.OPEN_STATUS):
        img, info = real_open(item, take, takes, state, seconds, lost, status)
        seen.append(("open", item["id"], status, lost, info))
        return img, info

    monkeypatch.setattr(rp, "compose_copy", copy)
    monkeypatch.setattr(rp, "compose_open", open_)
    return seen


def phases(timeline):
    """The statuses in order, repeats collapsed: the flow as the operator saw it."""
    out = []
    for kind, _iid, status, _x, info in timeline:
        name = status + (" (lost)" if kind == "open" and info["lost"]
                         and status == "OPEN HAND" else "")
        if not out or out[-1] != name:
            out.append(name)
    return out


def test_the_grasp_list_hints_the_fingertips_down_grasps(rp):
    data, _sha = rp.load_protocol(REPO / "protocols" / "grasps.json")
    hinted = {it["id"] for it in data["items"] if "orientation" in it}
    assert hinted == FINGERTIPS_DOWN
    for it in data["items"]:
        want = HINT if it["id"] in FINGERTIPS_DOWN else "palm toward the camera"
        assert rp.orientation_text(it) == want, it["id"]


def test_an_orientation_that_is_not_text_is_refused(rp, tmp_path, protocol_file):
    data = read_json(protocol_file)
    data["items"][0]["orientation"] = 12
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="'orientation' must be text"):
        rp.load_protocol(bad)


def test_the_make_the_grasp_frame_has_the_hint_under_the_status_line(rp, tmp_path):
    pic = fixture_picture(tmp_path / "tip.png")
    hinted = {"id": "p1_tip", "label": "Tip (paper 1, Schlesinger)",
              "orientation": HINT}
    plain = {"id": "p1_hook", "label": "Hook (paper 1, Schlesinger)"}
    _img, info = rp.compose_copy(hinted, pic, 1, 3, "MAKE THE GRASP", 3.2)
    assert info["status"] == "MAKE THE GRASP  4"
    assert " ".join(info["hint"]) == HINT and len(info["hint"]) <= 2
    assert info["label_bottom"] < info["status_top"]
    # the hint sits below the status line, the picture below the hint
    x, y, w, h = info["panel_box"]
    assert y > rp.COPY_BAND_H + 40 and y + h <= rp.COPY_H
    _img, info = rp.compose_copy(plain, pic, 1, 3, "MAKE THE GRASP", 1.0)
    assert info["hint"] == ["palm toward the camera"]
    # outside MAKE THE GRASP, no hint and the picture where it always was
    _img, info = rp.compose_copy(hinted, pic, 1, 3, "HOLD STILL", 2.0)
    assert info["hint"] == [] and info["panel_box"][1] == rp.COPY_BAND_H


def test_the_open_hand_frame_says_open_hand_and_hides_the_picture(rp):
    data, _sha = rp.load_protocol(REPO / "protocols" / "grasps.json")
    for it in data["items"]:
        _img, info = rp.compose_open(it, 2, 3, "no hand seen", 29.5)
        assert " ".join(info["big"]) == "OPEN HAND, palm to the camera"
        assert info["status"] == "OPEN HAND" and info["panel_box"] is None
        assert info["lost"] == [] and "no hand seen" in info["lines"]
        assert " ".join(info["label"]) == it["label"]
        assert info["label_bottom"] < info["status_top"], it["id"]
        # the OPEN HAND after a loss during READY
        _img, info = rp.compose_open(it, 2, 3, "", lost=True, status="LOST YOU")
        assert " ".join(info["lost"]) == ("LOST YOU: show the open hand to the "
                                          "camera again")
        assert info["status"] == "LOST YOU"
        _img, info = rp.compose_copy(it, None, 3, 3, "MAKE THE GRASP", 4.0)
        assert info["label_bottom"] < info["status_top"], it["id"]


def test_coached_open_hand_waits_for_the_hand(rp, monkeypatch, tmp_path,
                                              protocol_file, capsys):
    """No hand for the first second: OPEN HAND waits, says so, and the meta
    has the time it took; then the grasp, then the take."""
    actors = actor_with(monkeypatch, absent_open_s=1.0)
    timeline = spy_timeline(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "0.5"))
    folder = session_dir(out)
    session = read_json(folder / "session.json")
    (entry,) = session["takes"]
    assert entry["accepted"] is True
    assert session["coach"] is True and session["prep_s"] == 0.5
    assert session["coaching"]["open_band_cm"] == [18.0, 40.0]
    assert session["coaching"]["open_max_angle_deg"] == 50.0
    meta = read_json(folder / entry["files"]["meta"])
    coaching = meta["coaching"]
    # 1 s with no hand, then 0.5 s of open palm in the band
    assert 1.4 <= coaching["acquire_s"] <= 3.0, coaching
    assert coaching["forming_lost_count"] == 0
    # the earlier rules' count of send-backs is not written any more
    assert "lost_while_forming" not in coaching
    assert coaching["acquire_rounds_s"] == [coaching["acquire_s"]]
    (acq,) = coaching["acquired"]
    assert 18.0 <= acq["height_cm"] <= 40.0 and acq["view_angle_deg"] <= 50.0
    assert meta["orientation_hint"] == "palm toward the camera"
    # the mock was walked through the take, phase by phase
    (actor,) = actors
    assert actor.phases == ["open", "ready", "form", "hold", "idle"]
    # COPY THIS: OPEN HAND (no picture yet), READY, then the grasp, then the take
    assert phases(timeline)[:4] == ["OPEN HAND", "READY", "MAKE THE GRASP",
                                    "HOLD STILL"]
    opens = [x for x in timeline if x[0] == "open"]
    assert opens and all(x[4]["panel_box"] is None for x in opens)
    assert any(line.startswith("no hand seen") for x in opens
               for line in x[4]["lines"])
    printed = capsys.readouterr().out
    assert printed.count("OPEN HAND, palm to the camera") == 1
    assert "MAKE THE GRASP, keep the palm toward the camera" in printed


def test_coached_open_hand_waits_even_when_the_recorder_is_slow_to_start(
        rp, monkeypatch, tmp_path, protocol_file):
    """The recorder stalls 1 s before its first OPEN HAND (a slow `git
    rev-parse` on a loaded machine; the full suite hit this): the mock's
    frames for that second are made at the first drain, all at once, and
    they must not be an open hand already, or OPEN HAND is acquired at once
    (acquire_s 0.06 before the fix, with or without the forming changes)."""
    import time

    real = rp.tool_commit
    monkeypatch.setattr(rp, "tool_commit", lambda: (time.sleep(1.0), real())[1])
    actor_with(monkeypatch, absent_open_s=1.0)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "0.5"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    coaching = read_json(folder / entry["files"]["meta"])["coaching"]
    # 1 s with no hand, then 0.5 s of open palm, counted from OPEN HAND
    assert 1.4 <= coaching["acquire_s"] <= 3.0, coaching


def test_coached_open_hand_times_out_with_the_reason(rp, monkeypatch, tmp_path,
                                                     protocol_file):
    """No open hand at all: after the timeout the attempt is rejected with
    "no open hand acquired in 30 s" (here 1 s) and retried as usual."""
    assert (f"no open hand acquired in {rp.OPEN_HAND_TIMEOUT_S:g} s"
            == "no open hand acquired in 30 s")
    monkeypatch.setattr(rp, "OPEN_HAND_TIMEOUT_S", 1.0)
    actor_with(monkeypatch, absent_open_s=99.0)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--retries", "1"))
    folder = session_dir(out)
    takes = read_json(folder / "session.json")["takes"]
    assert len(takes) == 2, "one attempt plus the usual retry"
    for entry in takes:
        assert entry["accepted"] is False and entry["decided_by"] == "coach"
        assert entry["reason"] == "no open hand acquired in 1 s"
        assert entry["tracked_fraction"] == 0.0
        assert entry["files"]["leap"] is None and entry["files"]["still"] is None
        assert entry["files"]["meta"].startswith("rejected/meta/")
        meta = read_json(folder / entry["files"]["meta"])
        assert META_FIELDS <= set(meta)
        assert meta["reason"] == entry["reason"]
        assert meta["still_missing_reason"] == "not_recorded"
        coaching = meta["coaching"]
        assert coaching["acquire_s"] == pytest.approx(1.0, abs=0.3)
        assert coaching["forming_lost_count"] == 0
        assert coaching["acquire_missing"].startswith("no hand seen")
        reason = (folder / entry["files"]["reason"]).read_text(encoding="utf-8")
        assert reason.splitlines()[0] == entry["reason"]
    assert not list(folder.glob("leap/*")) and not list(folder.glob("rejected/leap/*"))


def test_a_hand_lost_while_forming_and_found_again_keeps_the_take(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """The rules of 2026-10-07. The mock loses the hand as it closes (palm
    turned edge-on, then gone, 0.6 s after the GO beep) and it is back 0.5 s
    later under a new id: LOST YOU, the recorder waits, finds it, holds it
    1 s, and records the static take under the unchanged gate. The take is
    kept at its first attempt with forming_lost_count 1; the forming file
    is the one MAKE THE GRASP, the gap in it, every line marked, and the
    transition code calls it invalid with no duration."""
    from leap_hand import transition as tr

    actors = actor_with(monkeypatch)
    timeline = spy_timeline(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.5", "--mock-lose-forming", "1",
                                    "--mock-back-after", "0.5"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is True and entry["attempt"] == 1, entry["reason"]
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["recorder_rules"] == rp.RECORDER_RULES == "2026-10-07"
    # the static take itself: the same gate as ever, passed
    assert meta["tracked_fraction"] >= 0.9 and meta["gate"]["passed"] is True
    assert meta["gate"]["reacquisitions_in_static_interval"] == 0
    coaching = meta["coaching"]
    assert coaching["forming_lost_count"] == 1
    assert "lost_while_forming" not in coaching   # no "close it slower" any more
    (loss,) = coaching["forming_losses"]
    assert loss["view_angle_deg"] > 60.0 and loss["cause"] == "palm turned away"
    assert loss["height_cm"] == pytest.approx(25.0, abs=1.5)
    assert 0.5 <= loss["after_s"] < 1.0 and loss["fix"]
    assert loss["gone_s"] == pytest.approx(0.5, abs=0.1)
    assert loss["found_after_s"] == pytest.approx(1.1, abs=0.15)
    # one OPEN HAND, no reset: the loss was waited out
    assert len(coaching["acquire_rounds_s"]) == 1 and coaching["resets"] == 0
    assert coaching["lost_wait_s"] == rp.LOST_WAIT_S == 4.0
    assert coaching["form_window_s"] == rp.FORM_WINDOW_S == 8.0
    assert coaching["relock_s"] == rp.RELOCK_S == 1.0
    # the hold began once the hand found again had been held 1 s
    started = meta["take_start"] - coaching["form_start"]
    assert loss["found_after_s"] + rp.RELOCK_S - 0.05 <= started
    assert started <= loss["found_after_s"] + rp.RELOCK_S + 0.4
    (actor,) = actors
    assert actor.lost == 1
    assert actor.phases == ["open", "ready", "form", "hold", "idle"]
    assert phases(timeline) == ["OPEN HAND", "READY", "MAKE THE GRASP",
                                "LOST YOU", "MAKE THE GRASP", "HOLD STILL",
                                "CHECKING THE TAKE"]
    lost = [x for x in timeline if x[0] == "copy" and x[2] == "LOST YOU"]
    assert lost and all(" ".join(x[4]["hint"]) == rp.FORM_LOST_HINT for x in lost)
    printed = capsys.readouterr().out
    assert printed.count(rp.FORM_LOST_TEXT) == 1
    assert "found again after 0.5 s: hold the grasp there 1 s before the take" \
        in printed
    assert printed.count("OPEN HAND, palm to the camera:") == 1
    # the take follows the hand found again, and is the hold only
    first, = coaching["acquired"]
    assert meta["operator_hand_ids"] == [first["hand_id"] + 1]
    lines = read_lines(folder / entry["files"]["leap"])
    assert min(x["grab_strength"] for x in lines) == pytest.approx(0.6, abs=0.01)
    # the forming file: both ids, the gap, every line marked
    forming = read_lines(folder / entry["files"]["forming"])
    assert coaching["forming_frames"] == len(forming)
    assert {x["hand_id"] for x in forming if x["hand_side"] == "left"} == {
        first["hand_id"], first["hand_id"] + 1}
    assert all(x[tr.FORMING_LOST_KEY] == 1 for x in forming)
    t = sorted(x["capture_time"] for x in forming if x["hand_side"] == "left")
    assert max(b - a for a, b in zip(t, t[1:])) == pytest.approx(0.5, abs=0.1)
    # and no rule times it: the meta's transition, the hand-in's timing
    assert coaching["transition_s"] is None
    assert coaching["transition_note"] == tr.lost_note(1)
    g = tr.forming_timing(forming, meta)
    assert g.validity == tr.INVALID and g.notes == tr.lost_note(1)
    assert g.transition_duration_s is None and g.movement_start_time_s is None


def test_a_hand_lost_while_forming_and_not_found_again_is_rejected(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """Lost, and gone for good: the recorder waits `--lost-wait` (here 0.6 s)
    from its last frame and rejects the attempt with the reason, nothing
    recorded; the forming file goes to rejected/ marked. With a forming
    window that ends first, the reason names the window instead."""
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.5", "--retries", "0",
                                    "--mock-lose-forming", "1",
                                    "--lost-wait", "0.6"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is False and entry["decided_by"] == "coach"
    m = NOT_FOUND_REASON_RE.fullmatch(entry["reason"])
    assert m, entry["reason"]
    assert m["wait"] == "0.6" and m["cause"] == "palm turned away"
    assert m["cm"] == "25" and int(m["deg"]) > 60
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["reason"] == entry["reason"] and meta["accepted"] is False
    coaching = meta["coaching"]
    assert coaching["forming_lost_count"] == 1
    (loss,) = coaching["forming_losses"]
    assert loss["found_after_s"] is None
    assert loss["gone_s"] == pytest.approx(0.6, abs=0.15)
    assert len(coaching["acquire_rounds_s"]) == 1 and coaching["resets"] == 0
    assert entry["files"]["leap"] is None, "nothing was recorded"
    assert entry["files"]["forming"] == f"rejected/forming/{entry['name']}.jsonl"
    reason = (folder / entry["files"]["reason"]).read_text(encoding="utf-8")
    assert reason.splitlines()[0] == entry["reason"]
    printed = capsys.readouterr().out
    assert printed.count(rp.FORM_LOST_TEXT) == 1
    assert "REJECTED: lost the hand while forming the grasp and it was not " \
           "found again within the 0.6 s wait for it" in printed

    # the forming window, 1.7 s after the GO beep, ends before the 4 s wait
    out = tmp_path / "window"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.5", "--retries", "0",
                                    "--mock-lose-forming", "1",
                                    "--form-window", "1.7"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    m = NOT_FOUND_REASON_RE.fullmatch(entry["reason"])
    assert m and m["window"] == "1.7", entry["reason"]
    coaching = read_json(folder / entry["files"]["meta"])["coaching"]
    assert coaching["form_window_s"] == 1.7
    assert coaching["forming_losses"][0]["after_s"] + \
        coaching["forming_losses"][0]["gone_s"] == pytest.approx(1.7, abs=0.2)
    assert read_json(folder / "session.json")["coaching"]["form_window_s"] == 1.7


def test_the_form_window_and_the_wait_are_options_of_the_coached_take(
        rp, monkeypatch, tmp_path, protocol_file):
    for extra in (["--lost-wait", "0"], ["--form-window", "-1"],
                  ["--form-window", "0.5", "--prep", "1.0"],
                  ["--no-coach", "--lost-wait", "2"],
                  ["--mock-back-after", "0.5"],                # nothing to bring back
                  ["--mock-lose-forming", "1", "--mock-back-after", "-1"]):
        with pytest.raises(SystemExit):
            run(rp, monkeypatch, *mock_args(protocol_file, tmp_path / "x",
                                            *extra))
    with pytest.raises(SystemExit):                    # protocol options
        run(rp, monkeypatch, "--mock", "--lost-wait", "2")
    assert not (tmp_path / "x").exists()


# --- READY, and the hand closed before the cue ---------------------------------------
# the reason under the rules of 2026-10-07: a close before the GO beep resets
# the attempt, and only past the limit rejects it
RESET_REASON_RE = re.compile(
    r"closed before the GO beep (?P<n>\d+) times?; reset to OPEN HAND "
    r"(?P<limit>\d+) times, the limit that ends a stuck loop; the last "
    r"(?P<where>.+)")


def spy_beeps(rp, monkeypatch):
    """Every beep, (wall clock, frequency, ms, volume), in order."""
    import time

    heard = []
    monkeypatch.setattr(rp, "beep", lambda freq=880, ms=180, volume=None:
                        heard.append((time.time(), freq, ms, volume)))
    return heard


def test_the_ready_pause_frame_says_keep_hand_open_until_the_go_beep(rp):
    item = {"id": "hook", "label": "Hook (paper 1, Schlesinger)"}
    _img, info = rp.compose_open(item, 1, 3, "hand acquired", 0.6,
                                 status=rp.READY_STATUS)
    assert info["big"] == ["KEEP HAND OPEN", "close on the GO beep"]
    assert info["status"] == "READY" and info["panel_box"] is None
    assert info["lost"] == []
    assert rp.READY_TEXT == "KEEP HAND OPEN - close on the GO beep"
    assert any("until the GO beep" in line for line in info["lines"])
    # the OPEN HAND after an early close says why, in red: the hint
    _img, info = rp.compose_open(item, 1, 3, "", lost=rp.CLOSED_TEXT,
                                 status=rp.CLOSED_STATUS)
    assert " ".join(info["lost"]) == ("CLOSED TOO SOON - keep the hand open "
                                      "until the GO beep")
    assert info["status"] == "CLOSED EARLY"
    assert info["big"] == ["OPEN HAND,", "palm to the camera"]
    # every word drawable by OpenCV's font
    assert all(ord(c) < 128 for c in rp.READY_TEXT + rp.CLOSED_TEXT
               + rp.LOST_TEXT + rp.FORM_LOST_TEXT)


def test_the_ready_and_go_cues_are_two_different_sounds(rp, monkeypatch):
    """READY: two short, quiet, low beeps; GO: one long, loud, high beep.
    Written as tones of those loudnesses (`tone_wav`), so the difference is
    in the samples, not only in the numbers asked for."""
    import io
    import wave

    import numpy as np

    heard = spy_beeps(rp, monkeypatch)
    rp.cue_ready()
    rp.cue_go()
    (_t1, f1, ms1, v1), (_t2, f2, ms2, v2), (_t3, fg, msg, vg) = heard
    assert (f1, ms1, v1) == (f2, ms2, v2) == rp.BEEP_READY
    assert (fg, msg, vg) == rp.BEEP_GO
    assert fg >= 3 * f1 and msg >= 5 * ms1 and vg >= 3 * v1
    assert rp.BEEP_LOST[0] < fg      # the low beep of a loss is not the GO

    def samples(freq, ms, volume):
        with wave.open(io.BytesIO(rp.tone_wav(freq, ms, volume))) as w:
            assert (w.getnchannels(), w.getsampwidth(),
                    w.getframerate()) == (1, 2, rp.TONE_RATE)
            return np.frombuffer(w.readframes(w.getnframes()), "<i2")

    ready, go = samples(*rp.BEEP_READY), samples(*rp.BEEP_GO)
    assert len(ready) == round(rp.TONE_RATE * 0.070)
    assert len(go) == round(rp.TONE_RATE * 0.450)
    assert np.abs(go).max() > 32000 and np.abs(ready).max() < 0.35 * 32767
    assert np.abs(go).max() >= 3 * np.abs(ready).max()
    # zero crossings: the pitch of each
    def hz(s):
        return np.count_nonzero(np.diff(np.signbit(s))) / 2 / (len(s) / rp.TONE_RATE)
    assert hz(ready) == pytest.approx(rp.BEEP_READY[0], rel=0.1)
    assert hz(go) == pytest.approx(rp.BEEP_GO[0], rel=0.05)
    # session.json says what the cues were
    cues = rp.cue_sounds()
    assert cues["ready"]["beeps"] == 2 and cues["go"]["beeps"] == 1
    assert cues["go"]["volume"] > cues["ready"]["volume"]


def test_ready_pause_watch_sees_a_close_but_not_a_still_open_hand(rp):
    """READY's watch, on the mock acting READY: an open palm held still is
    never closing; the same palm closing on the acquisition beep is, at
    50 mm/s or more, and a frame of the other hand never counts."""
    from leap_hand.mock import CoachedActor, MockLeapStream

    def ready_hands(early):
        actor = CoachedActor(operator="left", close_before_cue=int(early),
                             close_s=0.5)
        s = MockLeapStream(dropout_every=0, reacquire_every=0, script=actor)
        s.start()
        actor.phase("open", 0)
        actor.phase("ready", 0)
        return s.generate(90)                         # READY_PAUSE_S of frames

    for early in (False, True):
        hands = ready_hands(early)
        left = [lh for side, lh in hands if side == "left"]
        right = {lh.hand_id for side, lh in hands if side == "right"}
        w = rp.ReadyWatch(rp.hand_seen(left[0], 50.0), others=right,
                          label="left")
        got = None
        for k, (_side, lh) in enumerate(hands[2:]):
            w.add(lh, 50.0 + k / 180)
            got = got or w.closing()
        assert w.lost(50.0 + 89 / 90) is None
        if early:
            assert got is not None and got[0] >= rp.MOVE_SPEED_MM_S
            assert left[0].capture_time <= got[1] <= left[0].capture_time + 0.5
        else:
            assert got is None
            assert w.fastest is not None and w.fastest[0] < 25.0


def test_ready_pause_comes_between_the_ready_cue_and_the_go_cue(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """The two beeps on 2026-10-02 sounded back to back and alike. Now the
    acquired hand gets the READY cue (two short low beeps), is held open
    READY_PAUSE_S (1 s) from the end of it, KEEP HAND OPEN on the window
    and the console, and only then the GO cue (one long high beep): never
    sooner than before. The mock's close waits for the GO cue, and the
    forming file starts at it."""
    heard = spy_beeps(rp, monkeypatch)
    actors = actor_with(monkeypatch)
    timeline = spy_timeline(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "1.0"))
    folder = session_dir(out)
    session = read_json(folder / "session.json")
    (entry,) = session["takes"]
    assert entry["accepted"] is True
    assert session["coaching"]["ready_pause_s"] == rp.READY_PAUSE_S == 1.0
    assert session["coaching"]["move_speed_mm_s"] == 50.0
    assert session["coaching"]["cues"] == rp.cue_sounds()
    ready_beeps = [t for t, f, ms, v in heard if (f, ms, v) == rp.BEEP_READY]
    go = [t for t, f, ms, v in heard if (f, ms, v) == rp.BEEP_GO]
    assert len(ready_beeps) == 2 and len(go) == 1
    assert ready_beeps[1] - ready_beeps[0] >= rp.BEEP_READY_GAP_S - 0.01
    # the GO cue no sooner than a full READY_PAUSE_S after the READY cue
    assert go[0] - ready_beeps[1] >= rp.READY_PAUSE_S
    assert go[0] - ready_beeps[1] < rp.READY_PAUSE_S + 0.5
    # the cues in order: READY, READY, GO, then the take's own two beeps
    assert [(f, ms) for _t, f, ms, _v in heard[:3]] == [
        rp.BEEP_READY[:2], rp.BEEP_READY[:2], rp.BEEP_GO[:2]]
    (actor,) = actors
    assert actor.phases == ["open", "ready", "form", "hold", "idle"]
    assert phases(timeline)[:4] == ["OPEN HAND", "READY", "MAKE THE GRASP",
                                    "HOLD STILL"]
    ready = [x for x in timeline if x[2] == "READY"]
    assert all(x[0] == "open" and x[4]["big"] == ["KEEP HAND OPEN",
                                                  "close on the GO beep"]
               and x[4]["panel_box"] is None for x in ready)
    printed = capsys.readouterr().out
    assert printed.count("KEEP HAND OPEN - close on the GO beep: 1 s") == 1
    assert "close only on the GO beep" in printed         # the session header
    assert "READY = two short low beeps, KEEP HAND OPEN; GO = one long high " \
           "beep" in printed
    meta = read_json(folder / entry["files"]["meta"])
    coaching = meta["coaching"]
    assert coaching["ready_pause_s"] == 1.0
    assert coaching["closed_before_cue"] == 0 and coaching["early_closes"] == []
    # the cue is the GO beep: the forming file starts there, after READY,
    # and the open palm held through READY is not in it
    assert coaching["form_start"] >= ready_beeps[1] + rp.READY_PAUSE_S
    assert coaching["form_start"] == pytest.approx(go[0], abs=0.05)
    lines = read_lines(folder / entry["files"]["forming"])
    assert min(x["capture_time"] for x in lines) >= coaching["form_start"]
    assert coaching["transition_s"] == pytest.approx(0.25, abs=0.1)
    assert coaching["peak_speed_mm_s"] >= rp.MOVE_SPEED_MM_S


def test_a_mock_closing_before_the_cue_in_ready_is_reset_and_counted(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """--mock-close-early 1: the hand closes on the READY cue, during
    READY; the attempt is reset to OPEN HAND with CLOSED TOO SOON, counted,
    and the next round closes on the GO beep and is kept."""
    actors = actor_with(monkeypatch)
    timeline = spy_timeline(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.0", "--mock-close-early", "1"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is True and entry["attempt"] == 1
    meta = read_json(folder / entry["files"]["meta"])
    coaching = meta["coaching"]
    assert coaching["closed_before_cue"] == 1
    assert coaching["forming_lost_count"] == 0
    (early,) = coaching["early_closes"]
    assert early["during"] == "ready" and early["speed_mm_s"] >= 50.0
    assert 0.0 <= early["after_s"] < rp.READY_PAUSE_S
    assert len(coaching["acquire_rounds_s"]) == 2 and len(coaching["acquired"]) == 2
    (actor,) = actors
    assert actor.closed_early == 1
    assert actor.phases == ["open", "ready", "open", "ready", "form", "hold",
                            "idle"]
    assert phases(timeline) == ["OPEN HAND", "READY", "CLOSED EARLY",
                                "OPEN HAND (lost)", "READY", "MAKE THE GRASP",
                                "HOLD STILL", "CHECKING THE TAKE"]
    assert coaching["resets"] == 1 and coaching["ready_losses"] == []
    printed = capsys.readouterr().out
    assert printed.count(rp.CLOSED_TEXT) == 1
    assert "reset 1 of 10: closed before the GO beep, during READY" in printed
    # the kept take's forming file is the round that closed on the cue
    assert coaching["transition_s"] is not None
    assert coaching["transition_note"] == ""


def test_a_hand_closed_before_the_cue_has_no_close_after_the_go_beep(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """The hand already closed at OPEN HAND (acquired all the same) and
    still through READY: nothing to see until MAKE THE GRASP ends with no
    frame at 50 mm/s since the GO beep. CLOSED TOO SOON, reset to OPEN
    HAND, counted; the next round closes on the cue and is kept."""
    actor_with(monkeypatch, close_before_cue=1, early_close_at="open")
    timeline = spy_timeline(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "1.0"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is True
    coaching = read_json(folder / entry["files"]["meta"])["coaching"]
    assert coaching["closed_before_cue"] == 1
    (early,) = coaching["early_closes"]
    assert early["during"] == "make"
    assert early["peak_speed_mm_s"] < rp.MOVE_SPEED_MM_S
    assert early["after_s"] == pytest.approx(1.0, abs=0.2)
    assert phases(timeline) == ["OPEN HAND", "READY", "MAKE THE GRASP",
                                "CLOSED EARLY", "OPEN HAND (lost)", "READY",
                                "MAKE THE GRASP", "HOLD STILL",
                                "CHECKING THE TAKE"]
    printed = capsys.readouterr().out
    assert "reset 1 of 10: closed before the GO beep, at the GO beep" in printed
    # the forming file is the second round's, which closed on the cue
    assert coaching["transition_s"] is not None
    assert coaching["peak_speed_mm_s"] >= rp.MOVE_SPEED_MM_S
    assert not list(folder.glob("rejected/forming/*"))


def test_early_closes_reset_the_attempt_and_the_eleventh_drops_it(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """The rules of 2026-10-07: a close before the GO beep is not a
    rejection. Three of them reset the attempt three times and its fourth
    round is kept, at attempt 1 (under the earlier rules the third dropped
    it). Only the 11th, past the limit of 10 that ends a stuck loop, drops
    the attempt, with a reason that says so."""
    assert rp.RESET_LIMIT == 10
    out = tmp_path / "three"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.0", "--retries", "0",
                                    "--mock-close-early", "3"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is True and entry["attempt"] == 1, entry["reason"]
    coaching = read_json(folder / entry["files"]["meta"])["coaching"]
    assert coaching["closed_before_cue"] == 3 == coaching["resets"]
    assert all(e["during"] == "ready" for e in coaching["early_closes"])
    assert len(coaching["acquire_rounds_s"]) == 4
    assert coaching["forming_lost_count"] == 0
    printed = capsys.readouterr().out
    assert printed.count(rp.CLOSED_TEXT) == 3
    for k in (1, 2, 3):
        assert f"reset {k} of 10: closed before the GO beep, during READY" in printed
    assert "REJECTED" not in printed

    out = tmp_path / "eleven"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.0", "--retries", "0",
                                    "--mock-close-early", "11"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is False and entry["decided_by"] == "coach"
    m = RESET_REASON_RE.fullmatch(entry["reason"])
    assert m, entry["reason"]
    assert m["n"] == "11" and m["limit"] == "10"
    assert m["where"].startswith("during READY")
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["reason"] == entry["reason"]
    coaching = meta["coaching"]
    assert coaching["closed_before_cue"] == 11 and len(coaching["early_closes"]) == 11
    assert coaching["resets"] == 10 and coaching["reset_limit"] == 10
    assert len(coaching["acquire_rounds_s"]) == 11
    assert entry["files"]["leap"] is None, "nothing was recorded"
    reason = (folder / entry["files"]["reason"]).read_text(encoding="utf-8")
    assert reason.splitlines()[0] == entry["reason"]
    printed = capsys.readouterr().out
    assert printed.count(rp.CLOSED_TEXT) == 11
    assert "reset 10 of 10:" in printed and "reset 11 of" not in printed
    assert "REJECTED: closed before the GO beep 11 times; reset to OPEN HAND " \
           "10 times, the limit that ends a stuck loop" in printed


def test_early_closes_and_a_loss_while_forming_do_not_count_together(
        rp, monkeypatch, tmp_path, protocol_file):
    """Two early closes reset the attempt; the third round's hand is lost
    while forming and found again: one attempt, kept, its meta counting
    each apart (under the earlier rules the three were three send-backs
    and dropped the attempt)."""
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.5", "--retries", "0",
                                    "--mock-close-early", "2",
                                    "--mock-lose-forming", "1",
                                    "--mock-back-after", "0.4"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is True and entry["attempt"] == 1, entry["reason"]
    coaching = read_json(folder / entry["files"]["meta"])["coaching"]
    assert coaching["closed_before_cue"] == 2 and coaching["resets"] == 2
    assert coaching["forming_lost_count"] == 1
    assert coaching["forming_losses"][0]["found_after_s"] is not None
    assert len(coaching["acquire_rounds_s"]) == 3


def test_a_loss_during_ready_resets_the_attempt(rp, monkeypatch):
    """READY watches the open hand: lost there, the attempt goes back to
    OPEN HAND with LOST YOU, counted in `ready_losses` and `resets`, and
    the next round goes on. Driven on the recorder's own `_coach` with a
    READY that loses the hand once."""
    session = object.__new__(rp.ProtocolSession)
    session.prep, session.lost_wait, session.form_window = 1.0, 4.0, 8.0
    session.acquire_timeout, session.hand = 30.0, "left"
    session._now, session.view, session.panel = None, None, None
    from types import SimpleNamespace as NS
    hand = NS(height_cm=25.0, angle_deg=10.0, offset_cm=2.0,
              lh=NS(hand_side="left", hand_id=7))
    session._open_hand = lambda back=False: (hand, set(), "")
    lost = {"after_s": 0.4, "height_cm": 25.0, "view_angle_deg": 80.0,
            "grab": 0.0, "hand_label": "left", "cause": "palm turned away",
            "fix": "turn it back", "during": "ready"}
    calls = iter([(("lost", lost), hand, set()), (None, hand, set())])
    session._ready = lambda got, others: next(calls)
    session._make_grasp = lambda got, others: None
    ok, why = rp.ProtocolSession._coach(session, {"id": "x"})
    assert ok and why == ""
    c = session._coaching
    assert c["ready_losses"] == [lost] and c["resets"] == 1
    assert c["forming_lost_count"] == 0 and c["closed_before_cue"] == 0
    assert len(c["acquired"]) == 2
    assert rp.ready_loss_where(lost).startswith(
        "lost during READY, 0.4 s after the READY beep, at 25 cm")
    # both kinds past the limit: the reason names both, and the last
    early = {"during": "ready", "after_s": 0.2, "speed_mm_s": 96.0}
    reason = rp.reset_reason([early] * 4, [lost] * 7, "lost")
    assert reason == (
        "closed before the GO beep 4 times and lost the hand during READY 7 "
        "times; reset to OPEN HAND 10 times, the limit that ends a stuck "
        "loop; the last " + rp.ready_loss_where(lost))


def test_a_short_make_the_grasp_is_not_judged_closed_before_the_cue(
        rp, monkeypatch, tmp_path, protocol_file):
    """--prep under CUE_CHECK_MIN_S (a rehearsal's 0.3 s) has too few frames
    to tell a close from none: a hand already closed is not sent back."""
    assert rp.CUE_CHECK_MIN_S == 0.5
    actor_with(monkeypatch, close_before_cue=1, early_close_at="open")
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "0.3"))
    (entry,) = read_json(session_dir(out) / "session.json")["takes"]
    coaching = read_json(session_dir(out) / entry["files"]["meta"])["coaching"]
    assert entry["accepted"] is True and coaching["closed_before_cue"] == 0
    # the meta still says what the forming file shows: no close after the cue
    assert coaching["transition_s"] is None
    assert coaching["transition_note"].startswith("no close after the cue")


def test_mock_close_early_before_the_cue_needs_the_coached_mock(
        rp, monkeypatch, tmp_path, protocol_file):
    for extra in (["--no-coach"], []):
        argv = mock_args(protocol_file, tmp_path / "x", "--mock-close-early",
                         "1", *extra)
        if not extra:
            argv.remove("--mock")
        with pytest.raises(SystemExit):
            run(rp, monkeypatch, *argv)
    with pytest.raises(SystemExit):
        run(rp, monkeypatch, *mock_args(protocol_file, tmp_path / "y",
                                        "--mock-close-early", "-1"))


def test_no_coach_keeps_the_plain_countdown(rp, monkeypatch, tmp_path,
                                            with_pictures, capsys):
    actors = actor_with(monkeypatch)
    timeline = spy_timeline(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(with_pictures, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "0.3", "--no-coach"))
    folder = session_dir(out)
    session = read_json(folder / "session.json")
    (entry,) = session["takes"]
    assert entry["accepted"] is True
    assert session["coach"] is False and session["coaching"] is None
    assert session["prep_s"] == 0.3
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["coaching"] is None
    assert actors == [], "the plain mock, not the actor"
    assert phases(timeline) == ["GET READY", "HOLD STILL", "CHECKING THE TAKE"]
    assert not [x for x in timeline if x[0] == "open"]
    printed = capsys.readouterr().out
    assert "OPEN HAND" not in printed and "MAKE THE GRASP" not in printed
    assert "coach:   off (--no-coach)" in printed
    # the LOST YOU rehearsal is the coached take's
    with pytest.raises(SystemExit):
        run(rp, monkeypatch, *mock_args(with_pictures, tmp_path / "x",
                                        "--no-coach", "--mock-lose-forming", "1"))
    # the printed resume command keeps the session's mode
    from types import SimpleNamespace
    plain = SimpleNamespace(protocol_path=with_pictures, mock=True, hand="left",
                            session_dir=folder, coach=False)
    assert rp.ProtocolSession.resume_command(plain).endswith(" --no-coach")
    plain.coach = True
    assert "--no-coach" not in rp.ProtocolSession.resume_command(plain)


def test_a_coached_session_shows_open_hand_then_the_picture_with_the_hint(
        rp, monkeypatch, tmp_path, with_pictures, gui, capsys):
    data = read_json(with_pictures)
    data["items"][1]["orientation"] = HINT                 # tip_pinch
    hinted = tmp_path / "hinted.json"
    hinted.write_text(json.dumps(data), encoding="utf-8")
    timeline = spy_timeline(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(hinted, out, "--auto-accept", "--takes", "2",
                                    "--items", "hook,tip_pinch", "--prep", "0.3"))
    session = read_json(session_dir(out) / "session.json")
    assert [t["accepted"] for t in session["takes"]] == [True] * 4
    for iid in ("hook", "tip_pinch"):
        mine = [x for x in timeline if x[1] == iid]
        assert phases(mine)[:4] == ["OPEN HAND", "READY", "MAKE THE GRASP",
                                    "HOLD STILL"]
        # the picture only once the grasp is asked for
        assert all(x[3] for x in mine if x[0] == "copy")
        assert all(x[4]["panel_box"] is None for x in mine if x[0] == "open")
        hints = {" ".join(x[4]["hint"]) for x in mine
                 if x[0] == "copy" and x[2] == "MAKE THE GRASP"}
        assert hints == ({HINT} if iid == "tip_pinch" else {"palm toward the camera"})
    assert set(gui.shown) == {rp.COPY_WINDOW}
    printed = capsys.readouterr().out
    # the hint on the console once per grasp, not once per take
    assert printed.count(HINT) == 1
    assert printed.count("angle:  palm toward the camera") == 1
    metas = [read_json(session_dir(out) / t["files"]["meta"])
             for t in session["takes"]]
    assert [m["orientation_hint"] for m in metas] == (
        ["palm toward the camera"] * 2 + [HINT] * 2)


def test_the_open_hand_watch_needs_half_a_second_in_the_band(rp):
    """Height, palm angle and an unbroken half second, on the tracker's clock."""
    from leap_hand.mock import MockLeapStream

    def hands(script, n, side="left"):
        s = MockLeapStream(dropout_every=0, reacquire_every=0, script=script)
        return [lh for sd, lh in s.generate(n) if sd == side]

    w = rp.OpenHandWatch(prefer="left")
    for k, lh in enumerate(hands(None, 44)):             # 0.48 s of open palm
        w.add(lh, 100.0 + k / 90)
    assert w.acquired() is None
    assert w.status(100.5).startswith("hold it there")
    for k, lh in enumerate(hands(None, 50)[44:]):
        w.add(lh, 100.5 + k / 90)
    assert w.acquired() is not None and w.acquired().lh.hand_side == "left"
    # too high, then palm edge-on: never acquired, and the status says why
    for script, words in ((lambda i: {"origin_mm": (0.0, 450.0, 40.0)}, "lower it"),
                          (lambda i: {"roll_deg": 85.0}, "turn the palm")):
        w = rp.OpenHandWatch(prefer="left")
        for k, lh in enumerate(hands(script, 120)):
            w.add(lh, 200.0 + k / 90)
        assert w.acquired() is None
        assert w.status(200.0 + 119 / 90).startswith(words)
    # a hole longer than 0.1 s starts the half second again
    w = rp.OpenHandWatch(prefer="left")
    gap = lambda i: {"drop": True} if 30 <= i < 45 else None
    for k, lh in enumerate(hands(gap, 80)):
        w.add(lh, 300.0 + k / 90)
    assert w.acquired() is None                         # 35 frames since the hole
    assert rp.OpenHandWatch().status(0.0).startswith("no hand seen")


def test_the_form_watch_calls_a_hole_a_loss_but_not_a_slow_drain(rp):
    from leap_hand.mock import MockLeapStream

    s = MockLeapStream(dropout_every=0, reacquire_every=0, sides=("left",))
    lhs = [lh for _sd, lh in s.generate(200)]
    start = rp.hand_seen(lhs[0], 10.0)
    w = rp.FormWatch(start, others=set())
    # 0.5 s with no drain at all, then the backlog: not a loss
    for lh in lhs[1:46]:
        w.add(lh, 10.5)
    assert w.lost(10.5) is None
    # nothing for 0.31 s of wall clock: lost, at its last frame
    assert w.lost(10.81).lh.frame_id == lhs[45].frame_id
    # a hole of 0.33 s inside the data, even with the hand back: lost
    w = rp.FormWatch(rp.hand_seen(lhs[0], 20.0), others=set())
    for lh in lhs[1:10] + lhs[40:60]:
        w.add(lh, 20.7)
    assert w.lost(20.7).lh.frame_id == lhs[9].frame_id
    # another hand already in view at OPEN HAND does not stand in for ours
    w = rp.FormWatch(rp.hand_seen(lhs[0], 30.0), others={999})
    other = MockLeapStream(dropout_every=0, reacquire_every=0, sides=("right",),
                           script=lambda i: {"id_offset": 999 - 1001})
    for _sd, lh in other.generate(60):
        w.add(lh, 30.5)
    assert w.lost(30.5) is not None


def test_the_form_watch_lists_every_hole_and_says_when_the_hand_is_gone(rp):
    """MAKE THE GRASP waits for a lost hand instead of giving up at once:
    `gone` while no frame of it has come for more than 0.3 s, and every
    hole on the tracker's clock in `holes`, (last frame before, first
    after), whichever id the hand comes back under nearby."""
    import dataclasses

    from leap_hand.mock import MockLeapStream

    s = MockLeapStream(dropout_every=0, reacquire_every=0, sides=("left",))
    lhs = [lh for _sd, lh in s.generate(200)]
    w = rp.FormWatch(rp.hand_seen(lhs[0], 10.0), others=set())
    for lh in lhs[1:30]:
        w.add(lh, 10.3)
    assert w.gone(10.3) is None and w.holes == []
    assert w.gone(10.61).lh.frame_id == lhs[29].frame_id      # 0.31 s, no frame
    # back after a hole of 0.4 s, under a new id where it was: found again
    back = dataclasses.replace(lhs[66], hand_id=lhs[0].hand_id + 1)
    w.add(back, 10.8)
    assert w.gone(10.8) is None
    ((before, after),) = w.holes
    assert before.lh.frame_id == lhs[29].frame_id and after.lh is back
    assert after.ts - before.ts == pytest.approx(37 / 90, abs=0.01)
    assert w.hid == back.hand_id and w.new_ids == 1
    # a second loss is a second hole; `lost` still says the first
    for lh in lhs[67:80]:
        w.add(dataclasses.replace(lh, hand_id=back.hand_id), 10.9)
    w.add(dataclasses.replace(lhs[120], hand_id=back.hand_id), 11.5)
    assert len(w.holes) == 2
    assert w.lost(11.5).lh.frame_id == lhs[29].frame_id


def test_the_open_hand_watch_never_acquires_a_hand_off_to_the_side(rp):
    """2026-10-01 13:40:28: OPEN HAND acquired the idle right hand, held off
    to the side 21 cm up. A palm 24 cm to the side at 21 cm is in the band
    and within the palm angle, and is still never acquired; the same hand
    5 cm to the side is."""
    from leap_hand.mock import MockLeapStream

    def hands(side_cm, n=90):
        # the mock's left palm centre is 2 mm right of the wrist and 42 mm
        # further from the user: put the PALM at side_cm, 21 cm up, z = 0
        x = -side_cm * 10.0 - 2.0
        s = MockLeapStream(dropout_every=0, reacquire_every=0, noise_mm=0.0,
                           sides=("left",),
                           script=lambda i: {"origin_mm": (x, 210.0, 42.0),
                                             "curl": 0.0})
        return [lh for _sd, lh in s.generate(n)]

    off = hands(24.0)
    seen = rp.hand_seen(off[0], 0.0)
    assert seen.height_cm == pytest.approx(21.0, abs=0.1)
    assert seen.offset_cm == pytest.approx(24.0, abs=0.1)
    assert seen.angle_deg <= rp.OPEN_MAX_ANGLE_DEG, "only the centring fails"
    w = rp.OpenHandWatch(prefer="left")
    for k, lh in enumerate(off):
        w.add(lh, 100.0 + k / 90)
    assert w.acquired() is None
    now = 100.0 + 89 / 90
    assert w.status(now) == "centre it over the module (now 24 cm to the side)"
    assert w.need(now) == "centre it over the module"

    near = hands(5.0)
    assert rp.hand_seen(near[0], 0.0).offset_cm == pytest.approx(5.0, abs=0.1)
    w = rp.OpenHandWatch(prefer="left")
    for k, lh in enumerate(near):
        w.add(lh, 200.0 + k / 90)
    assert w.acquired() is not None
    assert w.status(200.0 + 89 / 90).startswith("hold it there")


def test_the_form_watch_follows_one_hand_by_id_and_place(rp):
    import dataclasses

    from leap_hand.mock import MockLeapStream

    s = MockLeapStream(dropout_every=0, reacquire_every=0, sides=("left",))
    lhs = [lh for _sd, lh in s.generate(20)]
    w = rp.FormWatch(rp.hand_seen(lhs[0], 10.0), others=set())
    assert w.mine == {lhs[0].hand_id}

    def as_id(lh, hid, dx=0.0):
        palm = [lh.palm_pos[0] + dx, lh.palm_pos[1], lh.palm_pos[2]]
        return dataclasses.replace(lh, hand_id=hid, palm_pos=palm)

    # a second id in the same tracking frame is another hand, however near
    w.add(lhs[1], 10.0)
    w.add(as_id(lhs[1], 777), 10.0)
    assert 777 in w.others and w.hid == lhs[0].hand_id
    w.add(as_id(lhs[2], 777), 10.0)                # and stays one
    assert w.hid == lhs[0].hand_id and w.last.lh.frame_id == lhs[1].frame_id
    # a new id near where the hand was, without it in that frame: the same
    # hand under a new id (a flip, a re-acquisition)
    w.add(as_id(lhs[3], 778, dx=0.03), 10.0)
    assert w.hid == 778 and 778 in w.mine and w.new_ids == 1
    # a new id far from it: another hand
    w.add(as_id(lhs[4], 779, dx=0.30), 10.0)
    assert 779 in w.others and w.hid == 778
    # a line of an id already followed is always taken, even beside another
    # of ours in the same frame and far away
    w.add(as_id(lhs[5], 778), 10.0)
    w.add(as_id(lhs[5], lhs[0].hand_id, dx=0.40), 10.0)
    assert w.hid == lhs[0].hand_id and w.new_ids == 1
    hid, others, palm = w.follow_start
    assert hid == lhs[0].hand_id and others == [777, 779]
    assert palm == pytest.approx([lhs[5].palm_pos[0] + 0.40, *lhs[5].palm_pos[1:]])
    assert w.lost(10.0) is None


def test_a_mock_session_with_the_other_hand_open_to_the_side(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """The idle hand open in view 25 cm to the side: OPEN HAND acquires the
    operator's hand only, the take follows it, and the idle hand's frames
    stay in the file, listed and never measured."""
    actors = actor_with(monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--mock-other-hand"))
    (actor,) = actors
    assert actor.operator == "left" and actor.other_hand == "open"
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is True, entry["reason"]
    meta = read_json(folder / entry["files"]["meta"])
    lines = read_lines(folder / entry["files"]["leap"])
    ours = meta["operator_hand_ids"]
    assert len(ours) == 1
    assert {x["hand_side"] for x in lines if x["hand_id"] in ours} == {"left"}
    others = meta["other_hand_ids"]
    assert others and not set(others) & set(ours)
    idle = [x for x in lines if x["hand_id"] in others]
    assert idle and {x["hand_side"] for x in idle} == {"right"}
    assert all(math.hypot(x["palm_abs"][0], x["palm_abs"][2]) > 0.20 for x in idle)
    assert max(x["grab_strength"] for x in idle) == 0.0
    assert set(meta["tracker_hand_labels"]) == {"left", "right"}
    assert meta["operator_hand_label"] == "left"
    assert meta["operator_hand_labels"] == {
        "left": sum(1 for x in lines if x["hand_id"] in ours)}
    assert meta["medoid_hand_id"] in ours
    assert meta["grab_strength"] == pytest.approx(0.6, abs=0.01)
    gate = meta["gate"]
    assert gate["interval_label_frames"] >= 0.5 * gate["interval_expected_frames"]
    coaching = meta["coaching"]
    assert coaching["other_hands"] and set(coaching["other_hands"]) == set(others)
    assert coaching["followed_ids"] == ours
    (acq,) = coaching["acquired"]
    assert acq["hand_label"] == "left" and acq["hand_id"] in ours
    assert acq["offset_cm"] < 0.6 * acq["height_cm"]
    printed = capsys.readouterr().out
    assert (f"hand: id {ours[0]}, tracker label left; another hand in view "
            f"(id {others[0]}) ignored") in printed


def test_the_plain_mock_follows_the_left_hand_beside_the_right(
        rp, monkeypatch, tmp_path, protocol_file):
    """The plain mock shows both hands 11 cm apart: OPEN HAND takes the left
    one by the operator's label, and the right one is the other hand."""
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    meta = read_json(folder / entry["files"]["meta"])
    lines = read_lines(folder / entry["files"]["leap"])
    by_id = {}
    for x in lines:
        by_id.setdefault(x["hand_id"], set()).add(x["hand_side"])
    assert [by_id[h] for h in meta["operator_hand_ids"]] == [{"left"}]
    assert [by_id[h] for h in meta["other_hand_ids"]] == [{"right"}]
    assert meta["coaching"]["acquired"][0]["hand_label"] == "left"
    assert meta["operator_hand_label"] == "left"


def test_mock_other_hand_needs_the_mock_and_the_coached_take(
        rp, monkeypatch, tmp_path, protocol_file):
    for argv in (["--protocol", protocol_file, "--hand", "left",
                  "--mock-other-hand", "--out-dir", tmp_path / "a"],
                 mock_args(protocol_file, tmp_path / "b", "--no-coach",
                           "--mock-other-hand"),
                 ["--mock", "--mock-other-hand"]):
        with pytest.raises(SystemExit):
            run(rp, monkeypatch, *argv)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_resume_says_when_the_rules_changed_and_keeps_the_sessions_wait(
        rp, monkeypatch, tmp_path, protocol_file):
    """A session recorded before the rules of 2026-10-07 (no recorder_rules
    in its session.json), resumed: session.json says when it changed to
    them and what changed, the old takes' metas stay as they were (without
    recorder_rules), the new take's names its rules. A session that set
    its own wait and window keeps them when resumed."""
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--lost-wait", "2.5", "--form-window", "6"))
    folder = session_dir(out)
    data = read_json(folder / "session.json")
    assert data["coaching"]["lost_wait_s"] == 2.5
    assert data["coaching"]["form_window_s"] == 6.0
    # as the recorder before 2026-10-07 left it
    del data["recorder_rules"], data["recorder_rules_changes"]
    (folder / "session.json").write_text(json.dumps(data), encoding="utf-8")
    old_meta = folder / data["takes"][0]["files"]["meta"]
    m = read_json(old_meta)
    del m["recorder_rules"]
    old_meta.write_text(json.dumps(m), encoding="utf-8")
    before = old_meta.read_bytes()
    run(rp, monkeypatch, "--mock", "--protocol", protocol_file, "--hand", "left",
        "--resume", folder, "--items", "hook", "--takes", "2", "--duration", "1",
        "--auto-accept", "--no-open")
    data = read_json(folder / "session.json")
    (change,) = data["recorder_rules_changes"]
    assert change["from"] is None and change["to"] == rp.RECORDER_RULES
    assert "lost while forming is waited for" in change["what"]
    assert data["recorder_rules"] == rp.RECORDER_RULES
    assert data["coaching"]["lost_wait_s"] == 2.5          # the session's own
    assert data["coaching"]["form_window_s"] == 6.0
    assert old_meta.read_bytes() == before
    new = read_json(folder / data["takes"][-1]["files"]["meta"])
    assert data["takes"][-1]["take"] == 2
    assert new["recorder_rules"] == rp.RECORDER_RULES
    assert new["coaching"]["lost_wait_s"] == 2.5


def test_resume_carries_on_across_a_hint_only_protocol_change(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """Adding orientation hints changes the file's bytes; a session recorded
    before them can still be resumed, and says so. Any other change cannot."""
    data = read_json(protocol_file)
    proto = tmp_path / "grasps_lines.json"
    proto.write_text(json.dumps(data, indent=2), encoding="utf-8")
    real = rp.Reviewer
    monkeypatch.setattr(rp, "Reviewer",
                        lambda auto=False: real(auto=auto, keys=iter(["\r", "q"])))
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(proto, out, "--items", "hook", "--takes", "2"))
    folder = session_dir(out)
    first = read_json(folder / "session.json")
    old_sha = first["protocol_sha256"]

    text = proto.read_text(encoding="utf-8")
    hinted = text.replace('"id": "hook",\n', '"id": "hook",\n      "orientation": '
                          + json.dumps(HINT) + ',\n')
    assert hinted != text
    proto.write_text(hinted, encoding="utf-8")
    assert rp.sha256_without_hints(proto.read_bytes()) == old_sha
    run(rp, monkeypatch, "--mock", "--protocol", proto, "--hand", "left",
        "--resume", folder, "--auto-accept", "--no-open")
    session = read_json(folder / "session.json")
    assert [(t["take"], t["accepted"]) for t in session["takes"]] == [
        (1, True), (2, False), (2, True)]
    import hashlib
    new_sha = hashlib.sha256(proto.read_bytes()).hexdigest()
    assert session["protocol_sha256"] == new_sha
    (change,) = session["protocol_changes"]
    assert change["from_sha256"] == old_sha and change["to_sha256"] == new_sha
    assert "only by per-grasp orientation hints" in capsys.readouterr().out
    meta = read_json(folder / session["takes"][2]["files"]["meta"])
    assert meta["orientation_hint"] == HINT

    # a label changed as well: refused, as any other change always was
    proto.write_text(hinted.replace("hook grasp", "hook grip"), encoding="utf-8")
    with pytest.raises(SystemExit, match="protocol file has changed"):
        run(rp, monkeypatch, "--mock", "--protocol", proto, "--hand", "left",
            "--resume", folder, "--auto-accept", "--no-open")


# --- the change into the grasp: forming/<take>.jsonl ----------------------------------
FORMING_KEYS = {"form_start", "forming_frames", "forming_tracked_fraction",
                "transition_s", "peak_speed_mm_s", "peak_at_s", "last_move_s",
                "transition_note", "transition_rule"}


def test_a_coached_take_writes_its_forming_file_and_transition(
        rp, monkeypatch, tmp_path, protocol_file):
    """Every frame from the high beep until the take starts, in the take's
    own line format, and the meta's transition_s timed from that beep: the
    mock closes in a quarter of MAKE THE GRASP (0.25 s here), then holds."""
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "1.0"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    name = entry["name"]
    assert entry["accepted"] is True
    assert entry["files"]["forming"] == f"forming/{name}.jsonl"
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["files"]["forming"] == entry["files"]["forming"]
    coaching = meta["coaching"]
    assert FORMING_KEYS <= set(coaching)

    lines = read_lines(folder / entry["files"]["forming"])
    take = read_lines(folder / entry["files"]["leap"])
    assert coaching["forming_frames"] == len(lines) >= 1
    # the same line format as the take: same keys, same session/item/take
    assert set(lines[0]) == set(take[0])
    assert all(x["session"] == folder.name and x["item"] == "hook"
               and x["take"] == 1 for x in lines)
    # from the cue on, and over before the take's first line was seen
    t0 = coaching["form_start"]
    assert all(x["capture_time"] >= t0 for x in lines)
    assert max(x["capture_time"] for x in lines) <= min(
        x["capture_time"] for x in take) + 1e-6
    # MAKE THE GRASP took the prep's second; the file runs to the take
    span = max(x["capture_time"] for x in lines) - t0
    assert 0.9 <= span <= meta["take_start"] - t0 + 0.05
    # the open palm closing into the grasp, both hands in the file
    ours = [x for x in lines if x["hand_id"] in coaching["followed_ids"]]
    assert {x["hand_side"] for x in lines} == {"left", "right"}
    assert ours[0]["grab_strength"] < 0.2
    assert ours[-1]["grab_strength"] == pytest.approx(0.6, abs=0.01)
    assert coaching["forming_tracked_fraction"] >= 0.9
    # closed by 0.25 s, settled at its end (the window ends well before the take)
    assert coaching["transition_s"] == pytest.approx(0.25, abs=0.1)
    assert coaching["transition_rule"] == {"speed_mm_s": 20.0, "window_s": 0.3,
                                           "step_s": 0.1, "move_speed_mm_s": 50.0}
    # the close is the main movement: its peak, when, and nothing after it
    assert coaching["peak_speed_mm_s"] >= 50.0
    assert 0.0 <= coaching["peak_at_s"] < coaching["transition_s"]
    assert coaching["last_move_s"] < coaching["transition_s"]
    assert coaching["transition_note"] == ""
    assert coaching["ready_pause_s"] == rp.READY_PAUSE_S == 1.0
    assert coaching["closed_before_cue"] == 0


def test_a_reset_restarts_the_forming_file_and_a_loss_does_not(
        rp, monkeypatch, tmp_path, protocol_file):
    """A reset (here a close before the GO beep that MAKE THE GRASP finds:
    no close after the cue) sends the operator back to OPEN HAND, and the
    next MAKE THE GRASP starts the forming file again: it is the round that
    led to the take. A loss while forming, waited out, does not: the file
    is that one MAKE THE GRASP, the gap and both hand ids in it."""
    from leap_hand import transition as tr

    actor_with(monkeypatch, close_before_cue=1, early_close_at="open")
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "1.0"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    meta = read_json(folder / entry["files"]["meta"])
    coaching = meta["coaching"]
    assert coaching["resets"] == 1 and coaching["forming_lost_count"] == 0
    lines = read_lines(folder / entry["files"]["forming"])
    assert coaching["forming_frames"] == len(lines)
    # form_start is the second round's cue: the file begins there
    assert coaching["form_start"] > meta["take_start"] - 1.0 - 0.5
    assert min(x["capture_time"] for x in lines) >= coaching["form_start"]
    assert coaching["transition_s"] is not None
    assert not any(tr.FORMING_LOST_KEY in x for x in lines)
    assert not list(folder.glob("rejected/forming/*"))


def test_a_rejected_attempt_keeps_its_forming_file_under_rejected(
        rp, monkeypatch, tmp_path, protocol_file):
    """A hand lost while forming and not found again rejects the attempt
    before anything is recorded; its MAKE THE GRASP is kept with it in
    rejected/, marked, with no transition."""
    from leap_hand import transition as tr

    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.5", "--retries", "0",
                                    "--mock-lose-forming", "1",
                                    "--lost-wait", "0.5"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    name = entry["name"]
    assert entry["accepted"] is False and entry["files"]["leap"] is None
    assert entry["files"]["forming"] == f"rejected/forming/{name}.jsonl"
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["files"]["forming"] == entry["files"]["forming"]
    lines = read_lines(folder / entry["files"]["forming"])
    assert meta["coaching"]["forming_frames"] == len(lines) >= 1
    assert all(x[tr.FORMING_LOST_KEY] == 1 for x in lines)
    # the hand turned away and was gone: not timed, and the note says why
    assert "transition_s" in meta["coaching"]
    assert meta["coaching"]["transition_s"] is None
    assert meta["coaching"]["transition_note"] == tr.lost_note(1)
    assert not (folder / "forming").is_dir() or not any((folder / "forming").iterdir())


def test_a_gate_rejected_take_takes_its_forming_file_to_rejected(
        rp, monkeypatch, tmp_path, protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "0.5", "--retries", "0",
                                    "--mock-dropout", "0.3"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is False and entry["decided_by"] == "gate"
    assert entry["files"]["forming"] == f"rejected/forming/{entry['name']}.jsonl"
    assert (folder / entry["files"]["forming"]).is_file()
    # the dropout is the hold's: the forming phase was whole
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["coaching"]["forming_tracked_fraction"] >= 0.9


def test_a_no_coach_take_has_no_forming_file(rp, monkeypatch, tmp_path,
                                             protocol_file):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "0.3", "--no-coach"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["files"]["forming"] is None
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["coaching"] is None and meta["files"]["forming"] is None
    assert not (folder / "forming").exists()


def test_the_e2e_mock_rehearsal_leaves_forming_files(rp, monkeypatch, tmp_path,
                                                     protocol_file):
    """The runbook's rehearsal command (step 2), on the fixture's grasps:
    every take has its forming file and a transition."""
    out = tmp_path / "grasps"
    run(rp, monkeypatch, "--mock", "--protocol", protocol_file, "--hand", "left",
        "--auto-accept", "--takes", "1", "--duration", "1", "--prep", "0.5",
        "--no-open", "--out-dir", out)
    folder = session_dir(out)
    takes = read_json(folder / "session.json")["takes"]
    assert [t["accepted"] for t in takes] == [True] * len(ITEMS)
    assert len(list((folder / "forming").glob("*.jsonl"))) == len(ITEMS)
    for t in takes:
        coaching = read_json(folder / t["files"]["meta"])["coaching"]
        assert coaching["forming_frames"] >= 1
        assert coaching["transition_s"] == pytest.approx(0.15, abs=0.1)


def _recompute_cli():
    spec = importlib.util.spec_from_file_location(
        "recompute_transition_cli", REPO / "scripts" / "leap" / "recompute_transition.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _snapshot(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_recompute_transition_rewrites_only_the_transition_keys(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """A mock session with a kept take and a rejected attempt that both have
    forming files, put back the way the first rule wrote them (no peak,
    no note, the old rule, a wrong transition_s). The recompute rewrites
    those keys to what today's recorder writes, in the recorder's place and
    format, and leaves every other byte of the session as it was."""
    import shutil

    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "hook",
                                    "--prep", "1.0"))
    folder = session_dir(out)
    run(rp, monkeypatch, "--mock", "--protocol", protocol_file, "--hand", "left",
        "--resume", folder, "--items", "tip_pinch", "--takes", "1",
        "--duration", "1", "--prep", "1.0", "--retries", "0",
        "--mock-lose-forming", "1", "--lost-wait", "0.5", "--auto-accept",
        "--no-open")
    copy = tmp_path / "copy" / folder.name
    shutil.copytree(folder, copy)
    cli = _recompute_cli()
    metas = cli.forming_metas(copy)
    assert [p.parent.relative_to(copy).as_posix() for p in metas] == [
        "meta", "rejected/meta"]
    recorded = {p.name: read_json(p) for p in metas}
    capsys.readouterr()

    # today's recorder and the recompute agree: nothing to change
    before = _snapshot(copy)
    assert cli.main([str(copy), "--dry-run"]) == 0
    assert cli.main([str(copy)]) == 0
    printed = capsys.readouterr().out
    assert printed.count("(unchanged)") == 4
    assert printed.rstrip().endswith("recompute ok")
    assert _snapshot(copy) == before

    # the first rule's metas, written as the recorder wrote them
    for p in metas:
        meta = read_json(p)
        c = meta["coaching"]
        for key in ("peak_speed_mm_s", "peak_at_s", "last_move_s",
                    "transition_note"):
            del c[key]
        c["transition_s"] = 0.18
        c["transition_rule"] = {"speed_mm_s": 20.0, "window_s": 0.3, "step_s": 0.1}
        rp.write_json(p, meta)
    old = _snapshot(copy)

    assert cli.main([str(copy), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "transition_s 0.18 -> " in printed
    assert printed.rstrip().splitlines()[-1] == "recompute dry-run ok"
    assert _snapshot(copy) == old, "a dry run writes nothing"

    assert cli.main([str(copy)]) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == "recompute ok"
    new = _snapshot(copy)
    changed = {k for k in new if new[k] != old[k]}
    assert changed == {p.relative_to(copy).as_posix() for p in metas}
    assert set(new) == set(old)
    for p in metas:
        meta, was = read_json(p), recorded[p.name]
        # the numbers, the keys and their order are what the recorder wrote
        assert meta == was
        assert list(meta["coaching"]) == list(was["coaching"])
        # and the bytes are the recorder's write_json of it
        rp.write_json(tmp_path / "again.json", meta)
        assert p.read_bytes() == (tmp_path / "again.json").read_bytes()
    kept, rejected = (read_json(p)["coaching"] for p in metas)
    assert kept["transition_s"] is not None and kept["transition_note"] == ""
    # the hand lost while forming: the recompute reads the mark on its
    # forming file's lines, as the recorder did, and times nothing
    assert rejected["transition_s"] is None
    assert rejected["transition_note"].startswith(
        "the hand was lost while forming the grasp (1 time)")

    # a meta not in the recorder's form is named and left alone
    odd = metas[0]
    odd.write_text(json.dumps(read_json(odd), indent=1), encoding="utf-8")
    kept_bytes = odd.read_bytes()
    assert cli.main([str(copy)]) == 1
    printed = capsys.readouterr().out
    assert "not in the recorder's json form" in printed
    assert printed.rstrip().endswith("recompute: 1 take(s) not rewritten")
    assert odd.read_bytes() == kept_bytes


def _check_cli():
    spec = importlib.util.spec_from_file_location(
        "check_protocol_cli_rp", REPO / "scripts" / "check_protocol.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _as_before_forming(folder: Path) -> None:
    """The session as the recorder wrote it before 2026-10-02: no forming
    folder, no `files.forming`, no forming keys in `coaching`."""
    import shutil

    shutil.rmtree(folder / "forming", ignore_errors=True)
    session = read_json(folder / "session.json")
    for t in session["takes"]:
        t["files"].pop("forming", None)
        meta_path = folder / t["files"]["meta"]
        meta = read_json(meta_path)
        meta["files"].pop("forming", None)
        for key in FORMING_KEYS:
            meta["coaching"].pop(key, None)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (folder / "session.json").write_text(json.dumps(session, indent=2),
                                         encoding="utf-8")


def test_resume_with_takes_5_records_only_takes_4_and_5(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """The professor's 5 takes per grasp: a copy of a session with 3 kept
    takes of each grasp (as recorded before the forming files), resumed
    with --takes 5, gets takes 4 and 5 of each and nothing else, and still
    passes the checker, the old takes as before and the new ones with
    their transitions."""
    import shutil

    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--items", "hook,tip_pinch", "--takes", "3"))
    original = session_dir(out)
    copy = tmp_path / "copy" / original.name
    shutil.copytree(original, copy)
    _as_before_forming(copy)
    first = read_json(copy / "session.json")
    assert [(t["item"], t["take"]) for t in first["takes"]] == [
        ("hook", 1), ("hook", 2), ("hook", 3),
        ("tip_pinch", 1), ("tip_pinch", 2), ("tip_pinch", 3)]
    before = read_json(original / "session.json")

    run(rp, monkeypatch, "--mock", "--protocol", protocol_file, "--hand", "left",
        "--resume", copy, "--takes", "5", "--prep", "0.5", "--auto-accept",
        "--no-open")
    session = read_json(copy / "session.json")
    assert session["takes"][:6] == first["takes"], "earlier entries untouched"
    added = session["takes"][6:]
    assert [(t["item"], t["take"], t["accepted"]) for t in added] == [
        ("hook", 4, True), ("hook", 5, True),
        ("tip_pinch", 4, True), ("tip_pinch", 5, True)]
    assert session["takes_per_item"] == 5 and len(session["resumed"]) == 1
    assert session["items"] == ["hook", "tip_pinch"]
    printed = capsys.readouterr().out
    assert "6 take(s) already kept here" in printed
    assert "Short of" not in printed
    # the new takes have their forming files; the original is as it was
    for t in added:
        assert (copy / t["files"]["forming"]).is_file()
    assert read_json(original / "session.json") == before

    # a session resumed this way still passes the checker, old and new takes
    cli = _check_cli()
    assert cli.main([str(copy)]) == 0
    text = capsys.readouterr().out
    assert "10 accepted, 0 rejected; 0 accepted take(s) fail a rule" in text
    assert text.count("close cue to settled") == 4


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
