"""leap_hand.tmc_calibration_take and `scripts/leap/record_poses.py --calibration`.

Synthetic tmc_neutral takes are written the way the recorder writes them
(camera line format through `absolute_to_relative`, the thumb metacarpal at
a known rest frame against the palm, one `.meta.json` sidecar per take) and
read back: the rest frame is recovered exactly from identical holds, a take
shifted by 15 degrees shows in the across-take agreement at the chordal-mean
value worked out here, the hold is the span of the frames' timestamps, short
takes and short holds are skipped with reasons, mixed sessions and missing
or foreign sidecars are refused, and known abduction-then-flexion angles are
recovered against the result by `tmc_estimate.estimate` and under the
`*_est_neutral` names. The recorder's mode is driven through its own main()
on the mock, with the rehearsal command of the protocol document. The
document itself is checked to read one step at a time. Expected numbers are
worked out here, not read back from the code.
"""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from leap_hand import tmc_calibration_take as tct
from leap_hand import tmc_estimate as te
from xr_hand import joint_frames as jf

from test_tmc_estimate import (PERIOD_US, R_REST, T0_US, cal_with, hand_pose, line,
                               rest_for, write_jsonl)

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "tmc_calibration_protocol.md"
SETUPS = ("No hardware", "Camera only, no hand over it", "Camera only, bare left hand")
N_HOLD = 200                      # 199 periods of 11.111 ms: a 2.21 s hold
PHI = math.degrees(math.atan2(math.sin(math.radians(15.0)),
                              2.0 + math.cos(math.radians(15.0))))


def span(n: int) -> float:
    """The hold of n frames at PERIOD_US, first to last frame."""
    return (n - 1) * PERIOD_US / 1e6


def write_take(folder: Path, take: int, n: int, *, rest=None, hand="left",
               fe=0.0, aa=0.0, meta_over=None) -> Path:
    """One take of n identical frames at PERIOD_US (the thumb at (fe, aa)
    from `rest`, the hand's measured rest by default) and its sidecar, as the
    recorder writes them. Returns the JSONL path."""
    rest = rest_for(hand) if rest is None else rest
    t0 = T0_US + take * 100_000_000
    pose = hand_pose(hand, fe, aa, rest=rest)
    lines = [line(hand, 40 + take, 7000 + 1000 * take + k, t0 + k * PERIOD_US, pose)
             for k in range(n)]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"tmc_neutral_{hand}_take{take}_20990101_00000{take}.jsonl"
    write_jsonl(path, lines)
    start = lines[0]["capture_time"]
    meta = {"kind": tct.CALIBRATION_KIND, "calibration": tct.POSE, "pose": tct.POSE,
            "pose_definition": dict(tct.POSE_DEFINITION), "hand": hand,
            "operator": "test", "goniometer": None, "mock": False, "hz": 0.0,
            "recorder_rules": "2026-10-07", "tool_commit": None,
            "session": folder.name, "take": take, "takes": 3, "duration_s": 5.0,
            "prep_s": 5.0, "hold_start": start, "hold_end": round(start + 5.0, 6),
            "frames": n, "hands": hand, "reacquired": 0, "file": path.name,
            "recorded": "2099-01-01T00:00:00"}
    meta.update(meta_over or {})
    tct.meta_path(path).write_text(json.dumps(meta), encoding="utf-8")
    return path


# --- the loader -----------------------------------------------------------------------
def test_single_take_rest_frame_and_hold_from_timestamps(tmp_path):
    folder = tmp_path / "20990101_000000_left"
    write_take(folder, 1, N_HOLD)
    cal = tct.calibrate(folder)
    assert te.rotation_angle_deg(R_REST, cal.R_rest) < 1e-6
    assert cal.n_transitions == 1 and cal.n_frames == N_HOLD and cal.skipped == []
    assert cal.session == folder.name
    (p,) = cal.per_transition
    assert p["name"] == tct.POSE and p["take"] == 1 and p["n_frames"] == N_HOLD
    assert p["file"].endswith("take1_20990101_000001.jsonl")
    assert p["hold_s"] == pytest.approx(span(N_HOLD), abs=1e-9)       # 2.211 s
    assert p["hold_commanded_s"] == pytest.approx(5.0, abs=1e-6)       # kept apart
    assert p["clock"] == "timestamp_us"
    assert p["spread_max_deg"] < 1e-6 and p["angle_to_session_mean_deg"] < 1e-6
    assert p["id_changes"] == 0 and p["gaps"] == 0
    assert p["rest_xyz_deg"] == pytest.approx([-28.0, -41.0, -79.0], abs=1e-6)
    assert cal.agreement["max_deg"] < 1e-9
    assert cal.agreement["worst"] == {"name": tct.POSE, "take": 1}
    assert cal.kind == "neutral" and cal.hand == "left"
    assert cal.provenance.startswith(tct.PROVENANCE_PREFIX)
    for text in ("left hand", "1 take(s) (200 frames", f"holds {span(N_HOLD):.2f} to",
                 ">= 20 frames and >= 2 s per take", "zero = tmc_neutral_v1: ",
                 tct.POSE_DEFINITION["text"], "single take: the agreement is not",
                 "not the ISB bone-based neutral",
                 "not validated against an independent sensor"):
        assert text in cal.provenance, text
    assert "goniometer" not in cal.provenance
    assert tct.summary_line(cal) == (
        f"TMC NEUTRAL CALIBRATION OK: takes=1 frames={N_HOLD} "
        f"hold_s={span(N_HOLD):.2f}..{span(N_HOLD):.2f} agreement_max_deg=0.00 skipped=0")


def test_three_takes_one_shifted_shows_in_the_agreement(tmp_path):
    folder = tmp_path / "20990101_000000_left"
    write_take(folder, 1, N_HOLD)
    write_take(folder, 2, N_HOLD)
    write_take(folder, 3, N_HOLD, rest=R_REST @ jf.rot_x(15.0))
    cal = tct.calibrate(folder)
    # the chordal mean of {R, R, R Rx(15)} is R Rx(phi), tan(phi) = sin 15 / (2 + cos 15)
    assert te.rotation_angle_deg(R_REST @ jf.rot_x(PHI), cal.R_rest) < 1e-6
    angles = [p["angle_to_session_mean_deg"] for p in cal.per_transition]
    assert angles == pytest.approx([PHI, PHI, 15.0 - PHI], abs=1e-6)
    assert cal.agreement["worst"] == {"name": tct.POSE, "take": 3}
    assert cal.agreement["max_deg"] == pytest.approx(15.0 - PHI, abs=1e-6)
    assert cal.agreement["median_deg"] == pytest.approx(PHI, abs=1e-6)
    assert cal.n_transitions == 3 and cal.n_frames == 3 * N_HOLD
    assert "3 take(s)" in cal.provenance and "single take" not in cal.provenance
    assert f"max {15.0 - PHI:.1f} deg (take 3)" in cal.provenance


def test_short_take_and_short_hold_are_skipped_with_reasons(tmp_path):
    folder = tmp_path / "20990101_000000_left"
    write_take(folder, 1, N_HOLD)
    write_take(folder, 2, 10)                 # under 20 frames
    write_take(folder, 3, 40)                 # 20 frames or more, but a 0.43 s hold
    cal = tct.calibrate(folder)
    assert [p["take"] for p in cal.per_transition] == [1]
    skipped = {s["take"]: s["reason"] for s in cal.skipped}
    assert skipped[2] == "10 frames < 20"
    assert skipped[3] == f"hold {span(40):.2f} s < 2 s (from the frames' timestamp_us)"
    assert all(s["name"] == tct.POSE and s["file"] for s in cal.skipped)
    assert tct.summary_line(cal).endswith("skipped=2")
    # with the hold rule off, the 40-frame take counts
    assert [p["take"] for p in tct.calibrate(folder, min_hold_s=0.0).per_transition] == [1, 3]
    # nothing usable: refused, every reason named
    short = tmp_path / "20990101_000001_left"
    write_take(short, 1, 10)
    with pytest.raises(ValueError, match=r"no usable take .*take 1: 10 frames < 20"):
        tct.calibrate(short)


def test_mixed_sessions_and_missing_or_foreign_sidecars_are_refused(tmp_path):
    both = tmp_path / "both"
    write_take(both, 1, N_HOLD, hand="left")
    write_take(both, 2, N_HOLD, hand="right")
    with pytest.raises(ValueError, match="takes of both hands"):
        tct.calibrate(both)

    defs = tmp_path / "defs"
    write_take(defs, 1, N_HOLD)
    write_take(defs, 2, N_HOLD, meta_over={"pose_definition": {"id": "tmc_neutral_v2",
                                                                "text": "another"}})
    with pytest.raises(ValueError, match="different posture definitions"):
        tct.calibrate(defs)

    missing = tmp_path / "missing"
    path = write_take(missing, 1, N_HOLD)
    tct.meta_path(path).unlink()
    with pytest.raises(ValueError, match=r"without a \.meta\.json sidecar"):
        tct.calibrate(missing)
    with pytest.raises(ValueError, match="no sidecar"):
        tct.load_take(path)

    foreign = tmp_path / "foreign"
    write_take(foreign, 1, N_HOLD, meta_over={"kind": "grasp_take"})
    with pytest.raises(ValueError, match="not a tmc_neutral_calibration_take"):
        tct.calibrate(foreign)

    hands = tmp_path / "hands"
    write_take(hands, 1, N_HOLD, meta_over={"hand": "both"})
    with pytest.raises(ValueError, match="expected one of"):
        tct.calibrate(hands)

    with pytest.raises(ValueError, match="no .jsonl take"):
        tct.calibrate(tmp_path / "empty")


def test_the_sidecars_hand_selects_the_frames(tmp_path):
    folder = tmp_path / "20990101_000000_left"
    path = write_take(folder, 1, N_HOLD, hand="left")
    # a left-hand file whose sidecar claims the right hand has no held frames
    meta = json.loads(tct.meta_path(path).read_text(encoding="utf-8"))
    meta["hand"] = "right"
    tct.meta_path(path).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="no right hand line"):
        tct.load_take(path)


def test_sidecar_without_hold_times_keeps_the_timestamp_hold(tmp_path):
    folder = tmp_path / "20990101_000000_left"
    path = write_take(folder, 1, N_HOLD, meta_over={"hold_start": None, "hold_end": None})
    t = tct.load_take(path)
    assert t.hold_commanded_s is None
    assert t.hold_s == pytest.approx(span(N_HOLD), abs=1e-9)


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("fe,aa", [(20.0, 0.0), (-10.0, 25.0), (60.0, 40.0)])
def test_known_angles_recovered_under_the_neutral_names(tmp_path, hand, fe, aa):
    folder = tmp_path / f"20990101_000000_{hand}"
    write_take(folder, 1, N_HOLD, hand=hand)
    write_take(folder, 2, N_HOLD, hand=hand)
    cal = tct.calibrate(folder)
    assert tct.is_neutral(cal)
    pos, rot = hand_pose(hand, fe, aa)                 # from the same rest frame
    e = tct.estimate(pos, rot, hand, cal)
    assert set(e) == set(tct.KEYS) | {"calibration", "kind"}
    assert e[tct.FE] == pytest.approx(fe, abs=1e-6)
    assert e[tct.AA] == pytest.approx(aa, abs=1e-6)
    assert e[tct.TWIST] == pytest.approx(0.0, abs=1e-6)
    assert e[tct.RESIDUAL] == pytest.approx(0.0, abs=1e-6)
    assert e["calibration"] == cal.provenance and e["kind"] == "neutral"
    # the estimate module names its outputs by the calibration's kind: the
    # same call gives the same dict, never an `_est_openhand` name
    plain = te.estimate(pos, rot, hand, cal)
    assert plain == e
    assert te.FE not in e and te.FE not in plain and tct.FE in plain


def test_neutral_estimate_refuses_an_open_hand_calibration():
    open_hand = cal_with(R_REST)
    assert not tct.is_neutral(open_hand)
    with pytest.raises(ValueError, match="not a tmc_neutral calibration"):
        tct.estimate(*hand_pose("left"), "left", open_hand)


def test_calibration_json_round_trip_and_goniometer_note(tmp_path):
    folder = tmp_path / "20990101_000000_left"
    note = "radial abduction 4 deg, palmar abduction 1 deg; plastic goniometer, read by X"
    write_take(folder, 1, N_HOLD, meta_over={"goniometer": note})
    write_take(folder, 2, N_HOLD, meta_over={"goniometer": note})
    cal = tct.calibrate(folder)
    assert f"; goniometer: {note};" in cal.provenance
    doc = json.loads(json.dumps(cal.to_json()))
    assert doc["kind"] == "neutral" and doc["hand"] == "left"
    back = te.Calibration.from_json(doc)
    assert tct.is_neutral(back) and back.provenance == cal.provenance
    assert back.kind == "neutral" and back.hand == "left"
    # a calibration.json written before `kind` existed is still read as neutral
    older = {k: v for k, v in doc.items() if k not in ("kind", "hand")}
    assert tct.is_neutral(te.Calibration.from_json(older))
    assert np.allclose(back.R_rest, cal.R_rest)
    assert back.per_transition == cal.per_transition and back.skipped == []
    pos, rot = hand_pose("left", 35.0, -12.0)
    assert tct.estimate(pos, rot, "left", back) == tct.estimate(pos, rot, "left", cal)


# --- the recorder, through its main() on the mock -----------------------------------------
def load_script():
    path = REPO / "scripts" / "leap" / "record_poses.py"
    spec = importlib.util.spec_from_file_location("leap_script_record_poses_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rp(monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "beep", lambda *a, **k: None)      # quiet, fast
    return module


def run(rp, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["record_poses.py", *map(str, argv)])
    rp.main()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def doc_steps():
    """(number, title, body) of every `### Step N.` of the protocol document;
    a body ends at the next step or the next section."""
    text = DOC.read_text(encoding="utf-8")
    heads = list(re.finditer(r"^### Step (\d+)\. (.+)$", text, flags=re.M))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end():end]
        cut = re.search(r"^## ", body, flags=re.M)
        out.append((int(m.group(1)), m.group(2), body[:cut.start()] if cut else body))
    return out


def doc_command(step: int):
    """The one command of a step, split, without the interpreter and script."""
    body = next(b for n, _t, b in doc_steps() if n == step)
    block = re.search(r"```\n(.+?)\n```", body, flags=re.S).group(1).strip()
    words = block.split()
    assert words[:2] == [r".venv\Scripts\python.exe", r"scripts\leap\record_poses.py"]
    return words[2:]


def test_mock_calibration_session_writes_takes_sidecars_and_calibration(
        rp, monkeypatch, tmp_path, capsys):
    out = tmp_path / "tmc_neutral"
    argv = doc_command(2) + ["--out-dir", out, "--operator", "N Kim",
                             "--goniometer", "none taken"]
    assert {"--mock", "--calibration", "tmc_neutral", "--hand", "left"} <= set(argv)
    run(rp, monkeypatch, *argv)

    folders = [p for p in out.iterdir() if p.is_dir()]
    assert len(folders) == 1
    folder = folders[0]
    assert re.fullmatch(r"\d{8}_\d{6}_left", folder.name)
    jsonls = sorted(folder.glob("*.jsonl"))
    assert len(jsonls) == 2
    assert sorted(folder.glob("*.meta.json")) == [tct.meta_path(p) for p in jsonls]
    assert (folder / "calibration.json").is_file()

    held = []
    for i, path in enumerate(jsonls, 1):
        assert re.fullmatch(rf"tmc_neutral_(left|both)_take{i}_\d{{8}}_\d{{6}}\.jsonl",
                            path.name)
        lines = read_lines(path)
        assert lines
        for d in lines:
            assert d["pose"] == tct.POSE and d["take"] == i
            assert {"timestamp_us", "capture_time", "hand_id", "joints", "abs26",
                    "hand_side", "frame_id", "status"} <= set(d)
        left = [d for d in lines if d["hand_side"] == "left"]
        assert len(left) >= tct.MIN_FRAMES
        held.append(len(left))

        meta = read_json(tct.meta_path(path))
        assert meta["kind"] == tct.CALIBRATION_KIND
        assert meta["calibration"] == tct.POSE and meta["pose"] == tct.POSE
        assert meta["pose_definition"] == tct.POSE_DEFINITION
        assert meta["hand"] == "left" and meta["take"] == i and meta["takes"] == 2
        assert meta["duration_s"] == 3.0 and meta["prep_s"] == 1.0 and meta["hz"] == 0.0
        assert meta["mock"] is True
        assert meta["operator"] == "N Kim" and meta["goniometer"] == "none taken"
        assert meta["frames"] == len(lines) and meta["file"] == path.name
        assert meta["session"] == folder.name
        assert meta["recorder_rules"] == rp.RECORDER_RULES and "tool_commit" in meta
        assert meta["hands"] in ("left", "both") and meta["reacquired"] >= 0
        assert 2.9 <= meta["hold_end"] - meta["hold_start"] <= 3.8   # the 3 s hold, plus the loop's slack
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", meta["recorded"])

    cal = tct.calibrate(folder)
    assert cal.n_transitions == 2 and cal.skipped == []
    assert cal.n_frames == sum(held)
    for p in cal.per_transition:
        assert 2.0 <= p["hold_s"] <= 3.8 and p["clock"] == "timestamp_us"
        assert p["hold_commanded_s"] >= p["hold_s"] - 0.1
    assert cal.agreement["max_deg"] < 0.5         # the mock's thumb has no orientation noise
    assert "goniometer: none taken" in cal.provenance
    saved = te.Calibration.from_json(read_json(folder / "calibration.json"))
    assert saved.provenance == cal.provenance and np.allclose(saved.R_rest, cal.R_rest)

    text = capsys.readouterr().out
    assert "synthetic hands hold open_palm, not the posture" in text
    assert tct.POSE_DEFINITION["text"] in text
    assert "TMC NEUTRAL CALIBRATION OK: takes=2 " in text
    assert text.rstrip().splitlines()[-3] == tct.summary_line(cal)


def test_mock_calibration_with_a_hold_under_two_seconds_fails_after_recording(
        rp, monkeypatch, tmp_path, capsys):
    out = tmp_path / "tmc_neutral"
    with pytest.raises(SystemExit) as e:
        run(rp, monkeypatch, "--mock", "--calibration", "tmc_neutral", "--hand", "left",
            "--takes", "1", "--duration", "0.6", "--prep", "1", "--out-dir", out)
    message = str(e.value)
    assert message.startswith("TMC NEUTRAL CALIBRATION FAILED: no usable take")
    assert "take 1: hold 0." in message and "s < 2 s" in message
    folder = next(p for p in out.iterdir() if p.is_dir())
    assert len(list(folder.glob("*.jsonl"))) == 1 and len(list(folder.glob("*.meta.json"))) == 1
    assert not (folder / "calibration.json").exists()


@pytest.mark.parametrize("argv,message", [
    (["--mock", "--calibration", "tmc_neutral"], "--calibration needs --hand"),
    (["--mock", "--calibration", "tmc_neutral", "--hand", "left", "--poses", "fist"],
     "--poses does not apply"),
    (["--mock", "--calibration", "tmc_neutral", "--hand", "left",
      "--protocol", "protocols/grasps.json"], "two different sessions"),
    (["--mock", "--goniometer", "4 deg"], "--goniometer only applies with --calibration"),
    (["--mock", "--calibration", "tmc_neutral", "--hand", "left", "--items", "p1_tip",
      "--auto-accept"], "--items, --auto-accept only apply with --protocol"),
    (["--mock", "--calibration", "tmc_neutral", "--hand", "left", "--poses", "fist",
      "--protocol", "protocols/grasps.json"], "two different sessions"),
])
def test_mock_calibration_refusals_record_nothing(rp, monkeypatch, tmp_path, capsys,
                                                  argv, message):
    out = tmp_path / "tmc_neutral"
    with pytest.raises(SystemExit) as e:
        run(rp, monkeypatch, *argv, "--out-dir", out)
    assert e.value.code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


# --- the protocol document ----------------------------------------------------------------
def test_protocol_document_reads_one_step_at_a_time():
    steps = doc_steps()
    assert [n for n, _t, _b in steps] == list(range(1, len(steps) + 1))
    assert len(steps) >= 8
    for n, title, body in steps:
        assert sum(title.startswith(s + ":") for s in SETUPS) == 1, f"step {n}: {title}"
        assert body.count("```") in (0, 2), f"step {n} holds more than one command"
        assert " or " not in title, f"step {n} offers an alternative: {title}"
    text = DOC.read_text(encoding="utf-8")
    assert "ChatGPT thread" in text
    unwrapped = " ".join(text.split()).lower()         # the page wraps its lines
    assert tct.POSE_DEFINITION["text"].lower() in unwrapped   # page and sidecar agree
    assert "4.4.1" in text and "Cooney" in text and "Cheze" in text
    assert "goniometer" in text.lower() and "timestamp" in text.lower()


def test_protocol_document_rehearsal_is_the_recorded_one():
    args = doc_command(2)
    assert args == ["--mock", "--calibration", "tmc_neutral", "--hand", "left",
                    "--takes", "2", "--duration", "3", "--prep", "1"]
    real = doc_command(7)
    assert real == ["--calibration", "tmc_neutral", "--hand", "left", "--operator", '"N', 'Kim"']
    assert "--mock" not in real
