"""scripts/record_protocol.py end to end, with the mock glove and camera.

The two runbook rehearsal commands are run exactly as documented (plus an
--out-dir under tmp_path and --no-beep), and the folder they leave behind is
checked against docs/protocol_formats.md: layout, session.json, warmup.json,
the keys added to every frame line, and the events of every take. The
reject path runs on `--mock-glove-ignore-cues`; the operator's redo, the
stop key and the warm-up refusal run in-process with the keys faked.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cam_hand import recording_protocol as rp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_protocol.py"
CONTRACT_SESSION_KEYS = {
    "set", "protocol_file", "protocol_name", "protocol_version",
    "protocol_sha256", "hand", "operator", "camera", "glove", "started",
    "ended", "xr_trainer_calibrated_at", "seed", "rounds", "tool_commit",
    "takes"}
CONTRACT_TAKE_KEYS = {"item", "take", "name", "accepted", "reason",
                      "decided_at", "files"}
SEQ3 = [[], ["thumb"], ["middle"], ["index"], ["pinky"], ["ring"], []]


def run_cli(*args, out_dir, timeout=120):
    cmd = [sys.executable, str(SCRIPT), *args, "--out-dir", str(out_dir),
           "--no-beep"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, stdin=subprocess.DEVNULL)
    return proc, time.time() - t0


def only_session(out_dir, set_name):
    folders = [p for p in (out_dir / set_name).iterdir() if p.is_dir()]
    assert len(folders) == 1, folders
    return folders[0]


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()
            if x.strip()]


def load_script():
    spec = importlib.util.spec_from_file_location("record_protocol_script",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_frames(folder, name, item, take, camera=True):
    glove = lines(folder / "glove" / f"{name}.jsonl")
    assert len(glove) > 50
    for row in glove:
        assert row["session"] == folder.name
        assert row["item"] == item and row["take"] == take
        assert row["hand_side"] == "left"
        assert len(row["joints"]) == 26 and "capture_time" in row
        assert {"wall_time", "timestamp", "packet_counter", "frame_id",
                "status"} <= set(row)
    if camera:
        leap = lines(folder / "leap" / f"{name}.jsonl")
        assert len(leap) > 50
        assert {r["hand_side"] for r in leap} == {"left", "right"}
        for row in leap:
            assert row["session"] == folder.name
            assert row["item"] == item and row["take"] == take
            assert row["source"] == "leap" and len(row["abs26"]) == 26


# --- the two rehearsal commands of the runbook --------------------------------
def test_finger_flexion_rehearsal(tmp_path):
    proc, took = run_cli(
        "--set", "finger_flexion", "--hand", "left", "--items", "index_fast",
        "--mock-glove", "--mock-leap", "--no-view", "--no-open",
        "--calibrated-at", "2026-09-28T14:00:00", "--time-scale", "0.25",
        out_dir=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert took < 90
    folder = only_session(tmp_path, "finger_flexion")
    assert folder.name.endswith("_left")

    session = json.loads((folder / "session.json").read_text())
    assert CONTRACT_SESSION_KEYS <= set(session)
    assert session["set"] == "finger_flexion"
    assert session["protocol_file"] == "protocols/finger_flexion.json"
    assert session["protocol_name"] == "finger_flexion"
    assert session["protocol_version"] == 1
    assert session["protocol_sha256"] == hashlib.sha256(
        (ROOT / "protocols" / "finger_flexion.json").read_bytes()).hexdigest()
    assert session["hand"] == "left" and session["camera"] == "leap"
    assert session["glove"] is True
    assert session["xr_trainer_calibrated_at"] == "2026-09-28T14:00:00"
    assert session["seed"] is None and session["rounds"] == [["index_fast"]]
    assert session["started"] and session["ended"]
    assert session["time_scale"] == 0.25
    assert session["mock"] is True
    assert session["mock_flags"] == {"glove": True,
                                     "glove_follows_cues": True, "leap": True}
    [take] = session["takes"]
    assert CONTRACT_TAKE_KEYS <= set(take)
    name = take["name"]
    assert name.startswith("index_fast_left_take1_")
    assert take["accepted"] is True and take["reason"] == ""
    assert take["files"] == {
        "glove": f"glove/{name}.jsonl", "leap": f"leap/{name}.jsonl",
        "events": f"events/{name}.events.jsonl", "still": f"stills/{name}.png"}
    for rel in take["files"].values():
        assert (folder / rel).is_file(), rel
    assert take["check"]["cycles_from_peaks"] == 8
    assert take["check"]["span_fraction"] >= 0.6
    assert not (folder / "rejected").exists()

    warm = json.loads((folder / "warmup.json").read_text())
    assert warm["hand"] == "left" and warm["refused"] is None
    assert warm["t_open"][1] <= warm["t_fist"][0]
    for sensor in ("glove", "camera"):
        block = warm[sensor]
        assert set(block) == {"open", "fist", "span", "frames"}
        assert set(block["open"]) == set(rp.FINGERS)
        for f in rp.FINGERS:
            assert block["span"][f] == pytest.approx(
                block["open"][f] - block["fist"][f], abs=2e-4)
        assert block["frames"] > 10
    assert min(warm["glove"]["span"][f] for f in rp.REFUSE_FINGERS) >= 0.3

    check_frames(folder, name, "index_fast", 1)

    events = rp.read_events(folder / take["files"]["events"])
    kinds = [e["kind"] for e in events]
    assert kinds == ["take_start"] + ["cue"] * 32 + ["take_end", "decision"]
    assert events[0]["item"] == "index_fast" and events[0]["take"] == 1
    cues = rp.cue_events(events)
    assert [c["step"] for c in cues] == list(range(32))
    assert [c["phase"] for c in cues] == list(rp.PHASES) * 8
    assert [c["cycle"] for c in cues] == [k for k in range(1, 9) for _ in range(4)]
    assert cues[0]["label"] == "bend the index" and cues[0]["flexed"] == ["index"]
    assert cues[2]["flexed"] == [] and cues[0]["hold_s"] == 0.25
    ts = [e["t"] for e in events]
    assert ts == sorted(ts)
    assert events[-1] == {"t": events[-1]["t"], "kind": "decision",
                          "accepted": True, "reason": "", "by": "auto"}
    assert (folder / "stills" / f"{name}.png").stat().st_size > 100


def test_sequences_rehearsal(tmp_path):
    proc, took = run_cli(
        "--set", "sequences", "--hand", "left", "--items", "seq3_alternating",
        "--takes", "1", "--mock-glove", "--mock-leap", "--no-view",
        "--no-open", "--calibrated-at", "2026-09-28T14:00:00",
        "--time-scale", "0.25", out_dir=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert took < 90
    folder = only_session(tmp_path, "sequences")
    session = json.loads((folder / "session.json").read_text())
    assert CONTRACT_SESSION_KEYS <= set(session)
    assert session["set"] == "sequences"
    assert isinstance(session["seed"], int)
    assert session["rounds"] == [["seq3_alternating"]]
    [take] = session["takes"]
    assert take["accepted"] is True and take["take"] == 1
    assert take["check"]["steps_total"] == 7
    assert take["check"]["steps_pass"] == 7
    name = take["name"]
    check_frames(folder, name, "seq3_alternating", 1)

    events = rp.read_events(folder / take["files"]["events"])
    assert [e["kind"] for e in events] == (["take_start"] + ["cue"] * 7
                                           + ["take_end", "decision"])
    cues = rp.cue_events(events)
    assert [c["flexed"] for c in cues] == SEQ3
    assert [c["label"] for c in cues] == [
        "open hand", "thumb flexion", "middle flexion", "index flexion",
        "little flexion", "ring flexion", "open hand"]
    for c in cues:
        assert c["hold_s"] == 0.625
        assert "cycle" not in c and "phase" not in c
    assert events[-1]["by"] == "auto" and events[-1]["accepted"] is True
    assert (folder / "warmup.json").is_file()
    assert (folder / "stills" / f"{name}.png").is_file()


# --- the reject path -------------------------------------------------------
def test_ignored_cues_are_rejected_and_kept(tmp_path):
    proc, _took = run_cli(
        "--set", "finger_flexion", "--hand", "left", "--items", "index_fast",
        "--mock-glove-ignore-cues", "--mock-leap", "--no-view", "--no-open",
        "--calibrated-at", "2026-09-28T14:00:00", "--time-scale", "0.1",
        "--retries", "1", out_dir=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAILED: index_fast take 1 after 2 attempt(s)" in proc.stdout
    folder = only_session(tmp_path, "finger_flexion")
    session = json.loads((folder / "session.json").read_text())
    assert session["mock"] is True
    assert session["mock_flags"]["glove_follows_cues"] is False
    assert [t["attempt"] for t in session["takes"]] == [1, 2]
    names = [t["name"] for t in session["takes"]]
    assert len(set(names)) == 2
    for t in session["takes"]:
        assert t["accepted"] is False and t["take"] == 1
        assert t["reason"].startswith("the index moved only 0.")
        assert t["reason"].endswith("of its warm-up range (needs 0.60)")
        name = t["name"]
        assert t["files"] == {
            "glove": f"rejected/glove/{name}.jsonl",
            "leap": f"rejected/leap/{name}.jsonl",
            "events": f"rejected/events/{name}.events.jsonl",
            "still": f"rejected/stills/{name}.png"}
        for rel in t["files"].values():
            assert (folder / rel).is_file(), rel
        reason = (folder / "rejected" / f"{name}.reason.txt").read_text()
        assert reason.splitlines()[0] == t["reason"]
        assert "decided by: auto" in reason
        events = rp.read_events(folder / t["files"]["events"])
        assert [e["kind"] for e in events][-2:] == ["take_end", "decision"]
        assert events[-1]["accepted"] is False
        assert events[-1]["by"] == "auto"
        assert events[-1]["reason"] == t["reason"]
    assert list((folder / "glove").iterdir()) == []
    assert list((folder / "events").iterdir()) == []


def test_ignored_cues_on_a_sequence_name_the_step(tmp_path):
    proc, _took = run_cli(
        "--set", "sequences", "--hand", "left", "--items", "seq3_alternating",
        "--takes", "1", "--retries", "0", "--mock-glove-ignore-cues",
        "--camera", "none", "--no-view", "--no-open",
        "--calibrated-at", "2026-09-28T14:00:00", "--time-scale", "0.15",
        out_dir=tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    folder = only_session(tmp_path, "sequences")
    [take] = json.loads((folder / "session.json").read_text())["takes"]
    assert take["reason"].startswith("step 1 (thumb flexion): thumb read ")
    assert "should be flexed (above 0.60); 4 more step(s) failed" in take["reason"]
    assert take["files"]["leap"] is None and take["files"]["still"] is None
    assert not (folder / "leap").exists() and not (folder / "stills").exists()
    warm = json.loads((folder / "warmup.json").read_text())
    assert warm["camera"] is None and warm["glove"] is not None


# --- in-process: the operator's keys and the warm-up refusal ------------------
def _in_process(module, monkeypatch, tmp_path, keys, *extra):
    presses = list(keys)

    def fake_key():
        return presses.pop(0) if presses else ""

    monkeypatch.setattr(module, "console_key", fake_key)
    monkeypatch.setattr(module, "flush_console_keys", lambda: None)
    args = module.parse_args([
        "--set", "finger_flexion", "--hand", "left", "--items", "index_fast",
        "--camera", "none", "--mock-glove", "--no-view", "--no-open",
        "--no-beep", "--calibrated-at", "2026-09-28T14:00:00",
        "--time-scale", "0.1", "--out-dir", str(tmp_path), *extra])
    return module.run(args)


def test_operator_redo_moves_an_accepted_take(tmp_path, monkeypatch):
    module = load_script()
    code = _in_process(module, monkeypatch, tmp_path, ["r"])
    assert code == 0
    folder = only_session(tmp_path, "finger_flexion")
    first, second = json.loads((folder / "session.json").read_text())["takes"]
    assert first["accepted"] is False
    assert first["reason"] == "redo asked by the operator"
    assert first["decided_by"] == "operator"
    assert first["files"]["glove"].startswith("rejected/glove/")
    events = rp.read_events(folder / first["files"]["events"])
    decisions = [e for e in events if e["kind"] == "decision"]
    assert [(d["accepted"], d["by"]) for d in decisions] == [(True, "auto"),
                                                            (False, "operator")]
    assert "decided by: operator" in (
        folder / "rejected" / f"{first['name']}.reason.txt").read_text()
    assert second["accepted"] is True and second["take"] == 1
    assert second["attempt"] == 2
    assert (folder / second["files"]["glove"]).is_file()


def test_q_stops_the_session_and_keeps_what_was_done(tmp_path, monkeypatch):
    module = load_script()
    code = _in_process(module, monkeypatch, tmp_path, ["q"], "--items",
                       "index_fast,index")
    assert code == 1
    folder = only_session(tmp_path, "finger_flexion")
    session = json.loads((folder / "session.json").read_text())
    assert session["stopped"] == "stopped by the operator (q)"
    assert session["ended"]
    assert [t["item"] for t in session["takes"]] == ["index"]
    assert session["takes"][0]["accepted"] is True


def test_rounds_follow_the_saved_seed_and_number_the_takes(tmp_path,
                                                           monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "console_key", lambda: "")
    monkeypatch.setattr(module, "flush_console_keys", lambda: None)
    items = ["seq3_alternating", "seq6_configurations"]
    args = module.parse_args([
        "--set", "sequences", "--hand", "right", "--items", ",".join(items),
        "--takes", "2", "--seed", "7", "--camera", "none", "--mock-glove",
        "--no-view", "--no-open", "--no-beep",
        "--calibrated-at", "2026-09-28T14:00:00", "--time-scale", "0.05",
        "--out-dir", str(tmp_path)])
    assert module.run(args) == 0
    folder = only_session(tmp_path, "sequences")
    session = json.loads((folder / "session.json").read_text())
    assert session["seed"] == 7
    assert session["rounds"] == rp.rounds(items, 2, seed=7, shuffle=True)
    order = [t["item"] for t in session["takes"]]
    assert order == [i for r in session["rounds"] for i in r]
    assert all(t["accepted"] for t in session["takes"])
    for item in items:
        mine = [t for t in session["takes"] if t["item"] == item]
        assert [t["take"] for t in mine] == [1, 2]
        assert [t["round"] for t in mine] == [1, 2]
        assert mine[1]["name"].startswith(f"{item}_right_take2_")
        assert {r["take"] for r in lines(folder / mine[1]["files"]["glove"])} \
            == {2}
    assert session["camera"] == "none"
    assert all(t["files"]["leap"] is None for t in session["takes"])


def test_warmup_refusal_stops_before_any_take(tmp_path, monkeypatch):
    module = load_script()
    monkeypatch.setattr(module.CueFollowingGlove, "MAX_ANGLE", 0.3)
    code = _in_process(module, monkeypatch, tmp_path, [])
    assert code == 2
    folder = only_session(tmp_path, "finger_flexion")
    warm = json.loads((folder / "warmup.json").read_text())
    assert warm["refused"].startswith("the glove barely moved")
    session = json.loads((folder / "session.json").read_text())
    assert session["stopped"].startswith("warm-up refused: the glove barely")
    assert session["takes"] == []


# --- pieces ------------------------------------------------------------------
def test_mock_glove_follows_and_ignores_the_cues():
    module = load_script()
    follows = module.CueFollowingGlove("left", time_scale=1.0)
    ignores = module.CueFollowingGlove("left", follow_cues=False)
    for g in (follows, ignores):
        g.follow(("index",), 2.0, "bend", t=100.0)
        g.follow(("index",), 1.0, "hold", t=102.0)
        g.follow((), 2.0, "straighten", t=103.0)
        g.follow(rp.FINGERS, 3.0, t=105.0, warmup=True)
    lag = module.CueFollowingGlove.LAG_S["left"]
    assert follows.bend_at(101.0 + lag)["index"] == pytest.approx(0.5)
    assert follows.bend_at(102.5 + lag)["index"] == pytest.approx(1.0)
    assert follows.bend_at(104.0 + lag)["index"] == pytest.approx(0.5)
    assert follows.bend_at(102.5 + lag)["middle"] == 0.0
    assert ignores.bend_at(102.5 + lag)["index"] == 0.0
    assert ignores.bend_at(107.0)["ring"] == pytest.approx(1.0)   # warm-up
    assert follows.bend_at(50.0)["index"] == 0.0


def test_mock_sessions_default_to_their_own_tree(tmp_path, monkeypatch):
    module = load_script()
    base = ["--set", "finger_flexion", "--hand", "left"]
    real = module.parse_args(base)
    assert real.mock is False
    assert real.out_dir == Path("recordings") / "protocol"
    for flags in (["--mock-glove"], ["--mock-leap"],
                  ["--mock-glove-ignore-cues"]):
        args = module.parse_args(base + flags)
        assert args.mock is True
        assert args.out_dir == Path("recordings") / "protocol_mock"
    chosen = module.parse_args(base + ["--mock-glove", "--out-dir", "x"])
    assert chosen.out_dir == Path("x")                  # --out-dir still wins

    # End to end with no --out-dir: the session lands in the mock tree and
    # says so (the tree is redirected into tmp_path for the test).
    monkeypatch.setattr(module, "MOCK_OUT_DIR", tmp_path / "protocol_mock")
    monkeypatch.setattr(module, "console_key", lambda: "")
    monkeypatch.setattr(module, "flush_console_keys", lambda: None)
    args = module.parse_args(base + [
        "--items", "index_fast", "--camera", "none", "--mock-glove",
        "--no-view", "--no-open", "--no-beep",
        "--calibrated-at", "2026-09-28T14:00:00", "--time-scale", "0.1"])
    assert module.run(args) == 0
    folder = only_session(tmp_path / "protocol_mock", "finger_flexion")
    session = json.loads((folder / "session.json").read_text())
    assert session["mock"] is True
    assert session["mock_flags"] == {"glove": True,
                                     "glove_follows_cues": True,
                                     "leap": False}


def test_parse_args_guards(capsys):
    module = load_script()
    base = ["--set", "sequences", "--hand", "left"]
    with pytest.raises(SystemExit):
        module.parse_args(base + ["--camera", "none", "--mock-leap"])
    with pytest.raises(SystemExit):
        module.parse_args(base + ["--time-scale", "0"])
    with pytest.raises(SystemExit):
        module.parse_args(base + ["--calibrated-at", "yesterday"])
    with pytest.raises(SystemExit):
        module.parse_args(["--hand", "left"])                  # no set
    args = module.parse_args(base + ["--mock-glove-ignore-cues",
                                     "--calibrated-at", "2026-09-28 14:00"])
    assert args.mock_glove is True
    assert args.calibrated_at == "2026-09-28T14:00:00"
    assert args.band == (18.0, 28.0)


def test_protocol_problems_refuse_before_any_hardware(tmp_path):
    module = load_script()
    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "sequences"}', encoding="utf-8")
    args = module.parse_args(["--protocol", str(bad), "--hand", "left",
                              "--mock-glove", "--out-dir", str(tmp_path)])
    assert module.run(args) == 2
    args = module.parse_args(["--protocol", str(ROOT / "protocols" /
                                                 "grasps.json"),
                              "--hand", "left", "--mock-glove"])
    assert module.run(args) == 2
    args = module.parse_args(["--set", "sequences", "--hand", "left",
                              "--items", "seq9", "--mock-glove"])
    assert module.run(args) == 2
    assert not (tmp_path / "sequences").exists()


def test_keys_writer_adds_the_contract_keys(tmp_path):
    module = load_script()
    path = tmp_path / "x.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        w = module.KeysWriter(fh, {"session": "s1", "item": "index"})
        w.write(json.dumps({"take": 1, "wall_time": 1.0}) + "\n")
        w.write("\n")
    assert lines(path) == [{"take": 1, "wall_time": 1.0, "session": "s1",
                            "item": "index"}]
