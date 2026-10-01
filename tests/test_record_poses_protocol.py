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
FORMING_REASON_RE = re.compile(
    r"lost the hand (?P<n>\d+) times? while forming the grasp, the last at "
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
        _img, info = rp.compose_open(it, 2, 3, "", lost=True, status="LOST YOU")
        assert " ".join(info["lost"]) == ("LOST YOU: open the hand, then close "
                                          "it slower")
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
    assert coaching["lost_while_forming"] == 0
    assert coaching["acquire_rounds_s"] == [coaching["acquire_s"]]
    (acq,) = coaching["acquired"]
    assert 18.0 <= acq["height_cm"] <= 40.0 and acq["view_angle_deg"] <= 50.0
    assert meta["orientation_hint"] == "palm toward the camera"
    # the mock was walked through the take, phase by phase
    (actor,) = actors
    assert actor.phases == ["open", "form", "hold", "idle"]
    # COPY THIS: OPEN HAND (no picture yet), then the grasp, then the take
    assert phases(timeline)[:3] == ["OPEN HAND", "MAKE THE GRASP", "HOLD STILL"]
    opens = [x for x in timeline if x[0] == "open"]
    assert opens and all(x[4]["panel_box"] is None for x in opens)
    assert any(line.startswith("no hand seen") for x in opens
               for line in x[4]["lines"])
    printed = capsys.readouterr().out
    assert printed.count("OPEN HAND, palm to the camera") == 1
    assert "MAKE THE GRASP, keep the palm toward the camera" in printed


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
        assert coaching["lost_while_forming"] == 0
        assert coaching["acquire_missing"].startswith("no hand seen")
        reason = (folder / entry["files"]["reason"]).read_text(encoding="utf-8")
        assert reason.splitlines()[0] == entry["reason"]
    assert not list(folder.glob("leap/*")) and not list(folder.glob("rejected/leap/*"))


def test_coached_loss_while_forming_goes_back_to_open_hand_and_is_counted(
        rp, monkeypatch, tmp_path, protocol_file, capsys):
    """The mock loses the hand as it closes on the first try (palm turned
    edge-on, then gone) and keeps it on the second: LOST YOU, back to OPEN
    HAND, and the kept take's meta counts the loss and says where it was."""
    actors = actor_with(monkeypatch)
    timeline = spy_timeline(rp, monkeypatch)
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.5", "--mock-lose-forming", "1"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is True and entry["attempt"] == 1
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["tracked_fraction"] >= 0.9
    coaching = meta["coaching"]
    assert coaching["lost_while_forming"] == 1
    assert len(coaching["acquire_rounds_s"]) == 2 and len(coaching["acquired"]) == 2
    assert coaching["acquire_s"] == pytest.approx(sum(coaching["acquire_rounds_s"]),
                                                  abs=0.02)
    (loss,) = coaching["forming_losses"]
    assert loss["view_angle_deg"] > 60.0 and loss["cause"] == "palm turned away"
    assert loss["height_cm"] == pytest.approx(25.0, abs=1.5)
    assert 0.0 < loss["after_s"] < 1.5 and loss["fix"]
    (actor,) = actors
    assert actor.lost == 1
    assert actor.phases == ["open", "form", "open", "form", "hold", "idle"]
    # what the operator saw: the loss, then OPEN HAND again with LOST YOU on it
    assert phases(timeline) == ["OPEN HAND", "MAKE THE GRASP", "LOST YOU",
                                "OPEN HAND (lost)", "MAKE THE GRASP",
                                "HOLD STILL", "CHECKING THE TAKE"]
    printed = capsys.readouterr().out
    assert printed.count("LOST YOU: open the hand, then close it slower") == 1
    assert printed.count("OPEN HAND, palm to the camera:") == 2
    # the take is the hold only: the open palm and the loss are not in it
    lines = read_lines(folder / entry["files"]["leap"])
    assert min(x["grab_strength"] for x in lines) == pytest.approx(0.6, abs=0.01)


def test_coached_three_losses_reject_with_the_reason(rp, monkeypatch, tmp_path,
                                                     protocol_file, capsys):
    out = tmp_path / "grasps"
    run(rp, monkeypatch, *mock_args(protocol_file, out, "--auto-accept",
                                    "--takes", "1", "--items", "tip_pinch",
                                    "--prep", "1.5", "--retries", "0",
                                    "--mock-lose-forming", "3"))
    folder = session_dir(out)
    (entry,) = read_json(folder / "session.json")["takes"]
    assert entry["accepted"] is False and entry["decided_by"] == "coach"
    m = FORMING_REASON_RE.fullmatch(entry["reason"])
    assert m, entry["reason"]
    assert m["n"] == "3" and m["cause"] == "palm turned away"
    assert m["cm"] == "25" and int(m["deg"]) > 60
    meta = read_json(folder / entry["files"]["meta"])
    assert meta["reason"] == entry["reason"] and meta["accepted"] is False
    coaching = meta["coaching"]
    assert coaching["lost_while_forming"] == 3
    assert len(coaching["forming_losses"]) == 3
    assert len(coaching["acquire_rounds_s"]) == 3
    assert entry["files"]["leap"] is None, "nothing was recorded"
    reason = (folder / entry["files"]["reason"]).read_text(encoding="utf-8")
    assert reason.splitlines()[0] == entry["reason"]
    printed = capsys.readouterr().out
    assert printed.count("LOST YOU: open the hand, then close it slower") == 3
    assert "REJECTED: lost the hand 3 times while forming the grasp" in printed


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
        assert phases(mine)[:3] == ["OPEN HAND", "MAKE THE GRASP", "HOLD STILL"]
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
