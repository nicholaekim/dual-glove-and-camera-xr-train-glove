"""scripts/record_protocol.py end to end, with the mock glove and camera.

The two runbook rehearsal commands are run exactly as documented (plus an
--out-dir under tmp_path and --no-beep), and the folder they leave behind is
checked against docs/protocol_formats.md: layout, session.json, warmup.json,
the keys added to every frame line, and the events of every take. The
reject path runs on `--mock-glove-ignore-cues`; the operator's redo, the
stop key and the warm-up refusal run in-process with the keys faked, and
so do every Set B item on the cue-following mock and a mock hand that
folds fully on one bend of five.
"""
import hashlib
import importlib.util
import json
import re
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
        # since 2026-10-01 each finger is also bent on its own
        assert set(block) == {"open", "fist", "span", "frames", "single",
                              "single_span", "t_single"}
        assert set(block["open"]) == set(rp.FINGERS)
        for f in rp.FINGERS:
            assert block["span"][f] == pytest.approx(
                block["open"][f] - block["fist"][f], abs=2e-4)
            assert block["single_span"][f] == pytest.approx(
                block["open"][f] - block["single"][f], abs=2e-4)
        assert block["frames"] > 10
    assert min(warm["glove"]["span"][f] for f in rp.REFUSE_FINGERS) >= 0.3
    assert warm["units"] == rp.GLOVE_UNITS

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
    assert proc.stdout.count("      what to do: fold the index fully into "
                             "the palm on every bend, then open it "
                             "fully\n") == 2
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


def test_warmup_refusal_stops_before_any_take(tmp_path, monkeypatch, capsys):
    module = load_script()
    monkeypatch.setattr(module.CueFollowingGlove, "MAX_ANGLE", 0.3)
    code = _in_process(module, monkeypatch, tmp_path, [])
    assert code == 2
    out = capsys.readouterr().out
    # done again, whole, before the session is refused
    assert "warm-up again (2 of 3)" in out and "warm-up again (3 of 3)" in out
    folder = only_session(tmp_path, "finger_flexion")
    warm = json.loads((folder / "warmup.json").read_text())
    assert warm["refused"].startswith("the glove barely moved")
    assert warm["try"] == 3 and len(warm["earlier_refusals"]) == 2
    session = json.loads((folder / "session.json").read_text())
    assert session["stopped"].startswith("warm-up refused: the glove barely")
    assert session["takes"] == []


# --- in-process: the bend count of a Set B take ------------------------------
def test_the_mock_glove_passes_every_set_b_item(tmp_path, monkeypatch,
                                                capsys):
    """The mock hand that follows the cues folds fully on every bend, so
    every Set B item, the slow and the fast index included, passes the quick
    check (span and bend count) on its first attempt. Were the bend count to
    reject it, a rehearsal could only ever rehearse the reject path."""
    module = load_script()
    protocol = rp.load_protocol(ROOT / "protocols" / "finger_flexion.json")
    items = protocol.ids
    assert {"index_slow", "index_fast"} <= set(items)
    code = _in_process(module, monkeypatch, tmp_path, [], "--items",
                       ",".join(items), "--retries", "0")
    out = capsys.readouterr().out
    assert code == 0, out
    folder = only_session(tmp_path, "finger_flexion")
    takes = json.loads((folder / "session.json").read_text())["takes"]
    assert [t["item"] for t in takes] == items
    for t in takes:
        cycles = protocol.item(t["item"])["cycles"]
        assert t["accepted"] is True and t["attempt"] == 1, t
        assert t["check"]["cycles_from_events"] == cycles, t
        assert t["check"]["cycles_from_peaks"] == cycles, t
        assert t["check"]["span_fraction"] >= rp.SPAN_GATE, t
    assert not (folder / "rejected").exists()
    assert "what to do:" not in out


def _folds_fully_once(module):
    """The mock hand of the thumb take of 2026-10-01. On the first attempt
    it folds the cued finger only 60 % of the way on four bends of five and
    fully on the fifth (fast, and staying folded: the stock mock clamps a
    bend at 1.05, so 1.5 reaches it early in the bend), which is enough
    for the span gate on that one bend. From the sixth bend on (the retry)
    it folds fully every time, like the stock mock."""

    class Glove(module.CueFollowingGlove):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.bends = []                 # (cue time, how far it folds)

        def follow(self, flexed, duration, phase=None, t=None,
                   warmup=False):
            super().follow(flexed, duration, phase, t=t, warmup=warmup)
            if phase == rp.BEND:
                n = len(self.bends)
                self.bends.append((self._t0s[-1],
                                   0.6 if n < 4 else 1.5 if n == 4 else 1.0))

        def bend_at(self, t):
            amount = 1.0
            for t0, a in self.bends:
                if t0 <= t - self.lag:
                    amount = a
            return {f: v * amount for f, v in super().bend_at(t).items()}

    return Glove


def test_a_take_bent_fully_once_is_rejected_and_recorded_again(
        tmp_path, monkeypatch, capsys):
    """2026-10-01: a thumb take covered 0.78 of its range, so the span gate
    accepted it, but the thumb folded fully on one cycle of five; the
    checker failed it after the session, which cost a whole new session.
    The quick check now counts the bends with the checker's own function,
    rejects the take as soon as it is recorded, says what to do, and the
    retry records it again."""
    module = load_script()
    monkeypatch.setattr(module, "CueFollowingGlove",
                        _folds_fully_once(module))
    code = _in_process(module, monkeypatch, tmp_path, [], "--items",
                       "thumb", "--retries", "1")
    out = capsys.readouterr().out
    assert code == 0, out
    folder = only_session(tmp_path, "finger_flexion")
    first, second = json.loads((folder / "session.json").read_text())["takes"]
    reason = ("the thumb bent fully only 1 time of 5 (a bend counts when the "
              "curl rises above 60 percent of the take's range after being "
              "below 30 percent)")
    assert first["accepted"] is False and first["reason"] == reason
    assert first["decided_by"] == "auto" and first["attempt"] == 1
    assert first["check"]["span_fraction"] >= rp.SPAN_GATE   # span: a pass
    assert first["check"]["cycles_from_events"] == 5
    assert first["check"]["cycles_from_peaks"] == 1
    name = first["name"]
    assert first["files"]["glove"] == f"rejected/glove/{name}.jsonl"
    note = (folder / "rejected" / f"{name}.reason.txt").read_text()
    assert note.splitlines()[0] == reason
    assert "decided by: auto" in note
    events = rp.read_events(folder / first["files"]["events"])
    assert events[-1]["kind"] == "decision"
    assert events[-1]["accepted"] is False and events[-1]["reason"] == reason
    assert (f"      REJECTED: {reason}\n"
            "      what to do: fold the thumb fully across the palm on every "
            "bend, then open it fully\n") in out
    assert "retrying (1/1)" in out
    assert second["accepted"] is True and second["attempt"] == 2
    assert second["take"] == 1 and second["check"]["cycles_from_peaks"] == 5
    assert (folder / second["files"]["glove"]).is_file()


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


def test_the_camera_window_is_shown_unless_hidden(monkeypatch):
    """Set B and C show the camera window by default: on 2026-10-01 the
    operator could not place the hand or see how far the thumb bent without
    the picture, and two sessions ended on the first take. `--hide-camera`
    keeps the plan's cue-only screen, the viewer run with `--no-window`."""
    import subprocess as sp

    module = load_script()
    launched = []

    class Proc:
        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(sp, "Popen", lambda cmd, **kw: (launched.append(cmd), Proc())[1])
    shown = module.ViewerStills("left", (18.0, 28.0))
    hidden = module.ViewerStills("left", (18.0, 28.0), window=False)
    try:
        assert len(launched) == 2
        assert "--no-window" not in launched[0]
        assert "--no-window" in launched[1]
        for cmd in launched:
            assert cmd[cmd.index("--hand") + 1] == "left"
            assert cmd[cmd.index("--band") + 1] == "18,28"
    finally:
        shown.close()
        hidden.close()
    base = ["--set", "finger_flexion", "--hand", "left"]
    assert module.parse_args(base).hide_camera is False
    assert module.parse_args(base + ["--hide-camera"]).hide_camera is True


# --- 2026-10-01: live bars, the single-finger warm-up, resume, skip ----------
def _bar_pixels(img, module, finger, colour):
    """Rows of the bottom third, in the middle column of `finger`'s bar,
    that are exactly `colour` (BGR)."""
    import numpy as np
    win = module.CueWindow
    slot = win.W // len(rp.FINGERS)
    x = rp.FINGERS.index(finger) * slot + slot // 2
    col = img[win.H - win.H // 3:, x, :]
    return int(np.all(col == np.array(colour, dtype=col.dtype), axis=1).sum())


def test_cue_window_renders_the_bars():
    """The bars are drawn by `render` alone (no window): the cued finger's
    fill is its fraction of the bar, green when a bend phase has it at 0.7,
    amber at 0.4; the others grey; nothing in the bottom third without
    bars."""
    module = load_script()
    win = module.CueWindow
    height = (win.H - win.LABEL_H) - (win.H - win.H // 3 + 8)

    def bars(index, phase="bend"):
        return {"fractions": {"thumb": 0.1, "index": index, "middle": 0.2,
                              "ring": None, "pinky": 0.05},
                "cued": ["index"], "phase": phase}

    plain = win.render("index_left_take1", "BEND THE INDEX", None, "")
    assert plain.shape == (win.H, win.W, 3)
    assert plain[win.H - win.H // 3:].max() == 0           # no bars at all
    green = win.render("t", "BEND THE INDEX", 2.0, "cycle 1/5", bars(0.7))
    assert _bar_pixels(green, module, "index", win.GREEN) == pytest.approx(
        0.7 * height, abs=1.5)
    assert _bar_pixels(green, module, "index", win.AMBER) == 0
    amber = win.render("t", "BEND THE INDEX", 2.0, "cycle 1/5", bars(0.4))
    assert _bar_pixels(amber, module, "index", win.AMBER) == pytest.approx(
        0.4 * height, abs=1.5)
    assert _bar_pixels(amber, module, "index", win.GREEN) == 0
    # the others are grey; the fill is clipped to the bar
    assert _bar_pixels(green, module, "middle", win.GREY) == pytest.approx(
        0.2 * height, abs=1.5)
    full = win.render("t", "HOLD", 1.0, "", bars(1.4, "hold"))
    assert _bar_pixels(full, module, "index", win.GREEN) == pytest.approx(
        height, abs=1.5)
    # straightening wants the finger low
    assert win.bar_colour(0.2, "straighten") == win.GREEN
    assert win.bar_colour(0.5, "rest") == win.AMBER
    assert win.bar_colour(0.9, "flex") == win.GREEN
    assert win.bar_colour(0.1, "extend") == win.GREEN
    assert win.bar_colour(0.9, None) == win.NEUTRAL
    # the instance method is the same drawing
    same = module.CueWindow(enabled=False).render(
        "t", "BEND THE INDEX", 2.0, "cycle 1/5", bars(0.7))
    assert (same == green).all()
    args = module.parse_args(["--set", "finger_flexion", "--hand", "left"])
    assert args.no_bars is False
    assert module.parse_args(["--set", "finger_flexion", "--hand", "left",
                              "--no-bars"]).no_bars is True


def test_glove_bends_measure_the_thumb_in_degrees():
    """On the cue-following mock, asking for the thumb on its own moves the
    glove thumb by far more than 40 degrees (TMC_fe + MCP_fe + IP); the
    other four readings are exactly the old curls."""
    from cam_hand.features import flexion_features
    from xr_hand.keypoints21 import frame_to_keypoints21
    from xr_hand.parser import parse_hand_message

    module = load_script()
    glove = module.CueFollowingGlove("left", time_scale=1.0)
    glove.NOISE = 0.0
    glove.follow(("thumb",), 3.0, t=100.0, warmup=True)

    def frame_at(t):
        raw = glove.gens["left"].next_frame()
        glove._shape(raw, glove.bend_at(t))
        return parse_hand_message(raw, "left")

    opened, bent = frame_at(50.0), frame_at(102.5 + glove.lag)
    a, b = rp.glove_bends(opened), rp.glove_bends(bent)
    assert b[0] - a[0] >= 40.0
    assert b[0] - a[0] == pytest.approx(2 * 1.1 * 180.0 / 3.141592653589793,
                                        abs=0.5)        # MCP and IP, 1.1 rad
    for frame, values in ((opened, a), (bent, b)):
        old = [float(v) for v in flexion_features(frame_to_keypoints21(frame))]
        assert values[1:] == old[1:]
    assert rp.thumb_bend_deg(bent) == b[0]


def test_set_b_mock_run_measures_each_finger_on_its_own(tmp_path):
    """A mock Set B run (thumb and the fast index, cue-following glove, mock
    camera, a quarter of the durations) records the single-finger warm-up
    with its units, accepts every take against it, and the checker passes
    every accepted take with the warm-up named in check.csv."""
    proc, _took = run_cli(
        "--set", "finger_flexion", "--hand", "left", "--items",
        "thumb,index_fast", "--mock-glove", "--mock-leap", "--no-view",
        "--no-open", "--calibrated-at", "2026-10-01T17:00:00",
        "--time-scale", "0.25", "--retries", "0", out_dir=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    folder = only_session(tmp_path, "finger_flexion")
    warm = json.loads((folder / "warmup.json").read_text())
    glove = warm["glove"]
    assert warm["refused"] is None and warm["units"] == rp.GLOVE_UNITS
    assert warm["try"] == 1 and warm["earlier_refusals"] == []
    assert set(glove["single"]) == set(rp.FINGERS)
    assert all(glove["single"][f] is not None for f in rp.FINGERS)
    assert set(glove["t_single"]) == set(rp.FINGERS)
    assert glove["single"]["thumb"] > 35.0
    assert glove["single"]["thumb"] - glove["open"]["thumb"] > 35.0
    assert rp.single_refusal(glove) is None
    session = json.loads((folder / "session.json").read_text())
    assert session["warmups"] == ["warmup.json"]
    assert session["resumed"] == [] and session["skipped"] == []
    assert [t["item"] for t in session["takes"]] == ["thumb", "index_fast"]
    for t in session["takes"]:
        assert t["accepted"] is True, t
        assert t["warmup"] == "warmup.json"
        assert t["check"]["range_end"] == "single"
    out = tmp_path / "check"
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_protocol.py"),
         str(folder), "--out", str(out)], cwd=ROOT, capture_output=True,
        text=True, timeout=300)
    assert check.returncode == 0, check.stdout + check.stderr
    assert "Every accepted take passes." in check.stdout
    assert "glove thumb in degrees" in check.stdout
    import csv
    with open(out / "check.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["warmup"] for r in rows] == ["warmup.json"] * 2
    assert {r["verdict"] for r in rows} == {"pass"}


def _thumb_lazy_on_first_warmup(module):
    """A mock hand that does not bend the thumb on its own in the first
    warm-up (it stays open), and does in every warm-up after it."""

    class Glove(module.CueFollowingGlove):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.thumb_singles = 0

        def follow(self, flexed, duration, phase=None, t=None,
                   warmup=False):
            if warmup and tuple(flexed) == ("thumb",):
                self.thumb_singles += 1
                if self.thumb_singles == 1:
                    flexed = ()
            super().follow(flexed, duration, phase, t=t, warmup=warmup)

    return Glove


def test_a_short_single_bend_repeats_the_warm_up(tmp_path, monkeypatch,
                                                 capsys):
    """The thumb that stays open in the single-finger part refuses the
    warm-up, naming the thumb and what to do; the warm-up is done again
    and the second one is used, with the first refusal kept in it."""
    module = load_script()
    monkeypatch.setattr(module, "CueFollowingGlove",
                        _thumb_lazy_on_first_warmup(module))
    code = _in_process(module, monkeypatch, tmp_path, [], "--items", "thumb")
    out = capsys.readouterr().out
    assert code == 0, out
    assert "warm-up again (2 of 3)" in out
    assert "in the single-finger warm-up the thumb bent only" in out
    folder = only_session(tmp_path, "finger_flexion")
    warm = json.loads((folder / "warmup.json").read_text())
    assert warm["refused"] is None and warm["try"] == 2
    [why] = warm["earlier_refusals"]
    assert why.startswith("in the single-finger warm-up the thumb bent only ")
    assert "(needs 35): fold it fully across the palm, tip to the base of " \
           "the little finger" in why
    [take] = json.loads((folder / "session.json").read_text())["takes"]
    assert take["accepted"] is True


def test_resume_carries_a_session_on_in_its_own_folder(tmp_path,
                                                       monkeypatch, capsys):
    """Every take of the first run rejected (the mock hand ignores the
    cues, no retries); `--resume latest` with a hand that follows them
    records both items into the same folder, against a warm-up of its own,
    without touching the first run's warmup.json or round order."""
    module = load_script()
    monkeypatch.setattr(module, "console_key", lambda: "")
    monkeypatch.setattr(module, "flush_console_keys", lambda: None)
    base = ["--set", "finger_flexion", "--hand", "left", "--camera", "none",
            "--no-view", "--no-open", "--no-beep", "--time-scale", "0.1",
            "--calibrated-at", "2026-10-01T17:00:00",
            "--out-dir", str(tmp_path)]
    first = module.parse_args(base + ["--mock-glove-ignore-cues",
                                      "--retries", "0", "--items",
                                      "thumb,index", "--takes", "1"])
    assert module.run(first) == 1
    folder = only_session(tmp_path, "finger_flexion")
    before = json.loads((folder / "session.json").read_text())
    assert [t["accepted"] for t in before["takes"]] == [False, False]
    warm_before = (folder / "warmup.json").read_bytes()
    assert module.latest_session(tmp_path, "finger_flexion", "left") == folder

    second = module.parse_args(base + ["--mock-glove", "--resume", "latest",
                                       "--calibrated-at",
                                       "2026-10-01T17:30:00"])
    assert second.time_scale_given is True
    assert module.run(second) == 0, capsys.readouterr().out
    assert only_session(tmp_path, "finger_flexion") == folder
    after = json.loads((folder / "session.json").read_text())
    assert after["rounds"] == before["rounds"] == [["thumb", "index"]]
    assert after["items"] == ["thumb", "index"]
    assert len(after["resumed"]) == 1
    assert after["warmups"][0] == "warmup.json"
    [new_warm] = after["warmups"][1:]
    assert re.fullmatch(r"warmup_\d{6}\.json", new_warm)
    assert (folder / new_warm).is_file()
    assert (folder / "warmup.json").read_bytes() == warm_before
    assert after["xr_trainer_calibrated_at"] == "2026-10-01T17:00:00"
    assert after["calibrated"] == ["2026-10-01T17:00:00",
                                   "2026-10-01T17:30:00"]
    takes = after["takes"]
    assert [t["accepted"] for t in takes] == [False, False, True, True]
    assert [t["warmup"] for t in takes] == ["warmup.json"] * 2 + [new_warm] * 2
    assert [(t["item"], t["take"]) for t in takes[2:]] == [("thumb", 1),
                                                           ("index", 1)]
    for t in takes[2:]:
        assert (folder / t["files"]["glove"]).is_file()

    # everything recorded: a third --resume has nothing left to do
    assert module.run(module.parse_args(base + ["--mock-glove", "--resume",
                                                str(folder)])) == 0
    assert len(json.loads((folder / "session.json").read_text())[
        "resumed"]) == 1


def test_resume_refusals(tmp_path, monkeypatch, capsys):
    module = load_script()
    left = tmp_path / "finger_flexion" / "20261001_170000_left"
    right = tmp_path / "finger_flexion" / "20261001_170500_right"
    protocol = rp.load_protocol(ROOT / "protocols" / "finger_flexion.json")
    meta = {"set": "finger_flexion", "hand": "left", "mock": True,
            "protocol_sha256": protocol.sha256, "camera": "none",
            "time_scale": 0.1, "rounds": [["thumb"]], "items": ["thumb"],
            "takes_per_item": 1, "takes": []}
    for folder, hand in ((left, "left"), (right, "right")):
        folder.mkdir(parents=True)
        (folder / "session.json").write_text(json.dumps(dict(meta,
                                                             hand=hand)))
    base = ["--set", "finger_flexion", "--camera", "none", "--mock-glove",
            "--no-view", "--no-open", "--no-beep", "--out-dir", str(tmp_path)]
    # the right hand's folder with --hand left
    args = module.parse_args(base + ["--hand", "left", "--resume", str(right)])
    assert module.run(args) == 2
    assert "it is the right hand's session and this run is --hand left" in \
        capsys.readouterr().out
    # latest finds the hand's own folder only
    assert module.latest_session(tmp_path, "finger_flexion", "left") == left
    assert module.latest_session(tmp_path, "finger_flexion", "right") == right
    assert module.latest_session(tmp_path, "sequences", "left") is None
    # a real run cannot resume a mock session, nor another time scale
    args = module.parse_args(["--set", "finger_flexion", "--hand", "left",
                              "--camera", "none", "--resume", str(left),
                              "--time-scale", "0.5", "--out-dir",
                              str(tmp_path)])
    assert module.run(args) == 2
    out = capsys.readouterr().out
    assert "it is a mock session and this run is real" in out
    assert "it ran at --time-scale 0.1" in out
    # a session recorded before the thumb was measured in degrees
    (left / "warmup.json").write_text(json.dumps({"glove": {}, "refused":
                                                  None}))
    assert module.run(module.parse_args(base + ["--hand", "left", "--resume",
                                                "latest"])) == 2
    assert "before the glove thumb was measured in degrees" in \
        capsys.readouterr().out
    # the session already fixes its items, takes and seed
    for extra in (["--items", "thumb"], ["--takes", "2"], ["--seed", "3"]):
        with pytest.raises(SystemExit):
            module.parse_args(base + ["--hand", "left", "--resume", "latest"]
                              + extra)
    assert not (tmp_path / "finger_flexion" / "20261001_170000_left"
                / "glove").exists()


def test_s_in_the_pause_skips_the_rest_of_the_item(tmp_path, monkeypatch,
                                                   capsys):
    module = load_script()
    code = _in_process(module, monkeypatch, tmp_path, ["s"], "--items",
                       "index_fast,index", "--takes", "2")
    out = capsys.readouterr().out
    assert code == 0, out           # the skipped item has an accepted take
    folder = only_session(tmp_path, "finger_flexion")
    session = json.loads((folder / "session.json").read_text())
    assert [(t["item"], t["take"], t["accepted"]) for t in session["takes"]] \
        == [("index", 1, True), ("index_fast", 1, True),
            ("index_fast", 2, True)]
    [skip] = session["skipped"]
    assert skip["item"] == "index" and skip["take"] == 1 and skip["at"]
    assert "skipped by the operator" in out


def test_s_during_a_take_rejects_it_and_skips_the_item(tmp_path,
                                                       monkeypatch, capsys):
    module = load_script()

    def show(self, title, words, seconds_left, sub="", force=False,
             bars=None):
        return "s" if title.startswith("index_left_take1") and \
            words == "HOLD" else ""

    monkeypatch.setattr(module.CueWindow, "show", show)
    code = _in_process(module, monkeypatch, tmp_path, [], "--items",
                       "index_fast,index")
    out = capsys.readouterr().out
    assert code == 1, out           # skipped with no accepted take
    folder = only_session(tmp_path, "finger_flexion")
    session = json.loads((folder / "session.json").read_text())
    first, second = session["takes"]
    assert first["item"] == "index" and first["accepted"] is False
    assert first["reason"] == "skipped by the operator (s) during the take"
    assert first["decided_by"] == "operator"
    name = first["name"]
    assert first["files"]["glove"] == f"rejected/glove/{name}.jsonl"
    assert (folder / first["files"]["glove"]).is_file()
    note = (folder / "rejected" / f"{name}.reason.txt").read_text()
    assert note.splitlines()[0] == first["reason"]
    events = rp.read_events(folder / first["files"]["events"])
    assert events[-1]["kind"] == "decision" and events[-1]["by"] == "operator"
    assert len(rp.cue_events(events)) < 20          # it ended at once
    assert second["item"] == "index_fast" and second["accepted"] is True
    assert session["skipped"] == [{"item": "index", "at": session["skipped"][
        0]["at"], "take": 1}]
