"""scripts/leap/record_validation.py on the mock, and the capture protocol
document's commands.

The mock cannot make the movements (its synthetic hands do what the mock
does), so this checks the files and their chain of custody only: every item
of leap_hand.reference_validation.capture_sequence recorded in order, a
sidecar per take naming the participant, hand, session and item, the manifest
with each take's sha256, the folder read back by the comparison's own loader,
and a changed take refused. It never stands in for a live capture.
"""
import importlib.util
import json
import re
import shlex
from pathlib import Path

import pytest

from leap_hand import reference_validation as rv

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "validation_capture_protocol.md"


def load_script():
    path = REPO / "scripts" / "leap" / "record_validation.py"
    spec = importlib.util.spec_from_file_location("record_validation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(monkeypatch):
    module = load_script()
    monkeypatch.setattr(module.rp, "beep", lambda *a, **k: None)     # quiet, fast
    return module


def test_capture_rehearsal_on_the_mock_writes_a_session_the_comparison_reads(
        script, tmp_path, capsys):
    rc_ = script.main(["--participant", "REHEARSAL", "--hand", "left",
                       "--reference-device", "none (mock rehearsal)", "--mock",
                       "--trials", "1", "--duration", "0.4", "--prep", "0",
                       "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc_ == 0, out
    seq = rv.capture_sequence(1)
    m = re.search(r"VALIDATION CAPTURE DONE: participant=REHEARSAL hand=left "
                  r"session=(\S+) takes=(\d+)/(\d+) mock=yes", out)
    assert m and int(m.group(2)) == int(m.group(3)) == len(seq), out
    folder = tmp_path / m.group(1)
    manifest = json.loads((folder / rv.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["format"] == rv.SESSION_FORMAT and manifest["protocol"] == rv.PROTOCOL
    assert manifest["mock"] is True and manifest["missing"] == []
    assert [(t["item"], t["take"]) for t in manifest["takes"]] == \
        [(i.id, k) for i, k in seq]
    for t in manifest["takes"]:
        side = json.loads((folder / t["file"]).with_suffix(rv.META_SUFFIX)
                          .read_text(encoding="utf-8"))
        assert (side["participant"], side["hand"], side["session"], side["item"]) == \
            ("REHEARSAL", "left", folder.name, t["item"])
        assert side["instruction"] == rv.CAPTURE_ITEMS[t["item"]].instruction
        assert side["reference_device"] == "none (mock rehearsal)"
        assert side["hold_end"] > side["hold_start"]
        assert len(t["sha256"]) == 64
    trk = rv.load_tracker_session(folder)
    assert trk.participant == "REHEARSAL" and len(trk.takes) == len(seq)
    rec = trk.takes[1].rec
    assert rv.recording_scope(rec) == {"session": folder.name, "participant": "REHEARSAL"}
    # a take changed after the session was recorded is refused
    first = folder / manifest["takes"][1]["file"]
    first.write_text(first.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(rv.ReferenceError, match="changed after the session was recorded"):
        rv.load_tracker_session(folder)


def test_capture_sequence_puts_both_isolated_sweeps_in_every_trial():
    seq = rv.capture_sequence(3)
    assert seq[0][0].id == "sync_start" and seq[-1][0].id == "sync_end"
    for trial in (1, 2, 3):
        isolated = {i.isolated for i, k in seq if k == trial and i.joints == ("thumb_tmc",)}
        assert {"flexion_extension", "abduction_adduction"} <= isolated
    assert len(seq) == 2 + 3 * len(rv.TRIAL_ITEMS)


def doc_commands():
    text = DOC.read_text(encoding="utf-8")
    return re.findall(r"```(?:powershell)?\n(\.venv\\Scripts\\python\.exe .+?)\n```", text)


def test_capture_protocol_document_commands_parse_and_name_every_item():
    text = DOC.read_text(encoding="utf-8")
    for item in rv.CAPTURE_ITEMS.values():
        assert f"`{item.id}`" in text, item.id
    cmds = doc_commands()
    rec = [c for c in cmds if "record_validation.py" in c]
    assert rec, "no record_validation.py command in the protocol"
    parser = load_script().build_parser()
    for c in rec:
        words = shlex.split(c, posix=False)[2:]
        args = parser.parse_args([w.strip('"') for w in words])
        assert args.participant and args.reference_device
    names = {c.split()[2] for c in cmds if "reference_validation.py" in c}
    assert {"taps", "check", "compare"} <= names, names
    for heading in ("## Required equipment", "## Setup and calibration",
                    "## Synchronization checks", "## Quality checks and rejection reasons",
                    "## Expected output files", "## What remains outstanding"):
        assert heading in text, heading
