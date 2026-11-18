"""Known rotations and paired holds through the equipment-check CLI."""
import importlib.util
import json
import math
from pathlib import Path
import socket

import pytest
from xr_hand.joints import JOINT_NAMES

spec = importlib.util.spec_from_file_location(
    "glove_camera_pilot_check", Path(__file__).parents[1] / "scripts/glove_camera_pilot_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def frame(fe, t, source="glove"):
    joints = []
    for name in JOINT_NAMES:
        a = math.radians(-fe / 2) if name == "INDEX_PROXIMAL" else 0.0
        joints.append(dict(name=name, x=0.0, y=0.0, z=-0.03,
                           qx=math.sin(a), qy=0.0, qz=0.0, qw=math.cos(a)))
    return dict(hand_side="left", pose="open_palm", take=1, status=1,
                source=source, units="m", joints=joints, capture_time=100.0 + t,
                wall_time=100.1 + t, frame_id=int(t * 10), packet_counter=int(t * 10))


def paired_session(tmp_path):
    root = tmp_path / "source"
    for sensor, fe in (("glove", 10), ("leap", 22)):
        folder = root / sensor
        folder.mkdir(parents=True)
        rows = [frame(fe, k * 0.1, sensor) for k in range(41)]
        (folder / "open_palm_left_take1_test.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return root


def test_known_rotation_range():
    ranges = check.angle_ranges([frame(10, 0), frame(40, 1)], "left")
    index = next(r for r in ranges if r["joint"] == "INDEX_PROXIMAL"
                 and r["component"] == "flex_deg")
    assert index["min_deg"] == pytest.approx(10)
    assert index["max_deg"] == pytest.approx(40)
    assert index["span_deg"] == pytest.approx(30)
    fixed = [r for r in ranges if r["joint"] == "INDEX_METACARPAL"]
    assert all(r["span_deg"] == 0 for r in fixed)


def test_cli_reports_known_disagreement_and_preserves_sources(tmp_path):
    root = paired_session(tmp_path)
    before = {str(p): check.sha256(p) for p in root.rglob("*.jsonl")}
    out = tmp_path / "result"
    assert check.main(["audit", "--session", str(root), "--poses", "open_palm",
                       "--out", str(out), "--data-origin", "synthetic"]) == 0
    report = json.loads((out / "report.json").read_text())
    row = report["takes"][0]
    assert row["signed_med_I_MCP_fe"] == pytest.approx(12)
    assert row["absdiff_med_I_MCP_fe"] == pytest.approx(12)
    assert row["clock"] == "capture_time"
    assert report["physical_raw_sensor_count"] is None
    assert report["data_origin"] == "synthetic"
    assert before == {str(p): check.sha256(p) for p in root.rglob("*.jsonl")}
    assert check.main(["audit", "--session", str(root), "--poses", "open_palm",
                       "--out", str(out)]) == 2  # never overwrites a report


def test_other_hand_and_invalid_tracking_refused(tmp_path):
    root = paired_session(tmp_path)
    path = next((root / "glove").glob("*.jsonl"))
    with pytest.raises(ValueError, match="no right-hand"):
        check.load_hand(path, "right")
    rows = [frame(10, 0), {**frame(20, 1), "status": 0}]
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid tracking"):
        check.load_hand(path, "left")


def test_missing_pair_and_source_output_refused(tmp_path):
    root = paired_session(tmp_path)
    with pytest.raises(ValueError, match="outside the source"):
        check.audit(root, root / "derived", poses=("open_palm",))
    next((root / "leap").glob("*.jsonl")).unlink()
    assert check.main(["audit", "--session", str(root), "--poses", "open_palm",
                       "--out", str(tmp_path / "result")]) == 2


def test_probe_is_bounded(tmp_path):
    with pytest.raises(ValueError, match="at most 60"):
        check.probe(float("inf"), tmp_path / "probe")


def test_recorder_settling_segment_is_not_a_second_hold(tmp_path):
    root = paired_session(tmp_path)
    for folder in (root / "glove", root / "leap"):
        path = next(folder.glob("*.jsonl"))
        path.with_suffix(".settle.jsonl").write_bytes(path.read_bytes())
    report = check.audit(root, tmp_path / "result", poses=("open_palm",))
    assert len(report["takes"]) == 1
    assert len(report["sources"]) == 2


def test_probe_preserves_device_labels_and_does_not_count_joints_as_sensors(tmp_path, monkeypatch):
    from pythonosc.osc_message_builder import OscMessageBuilder
    builder = OscMessageBuilder(address="/v1/animation/kinematic/all")
    for value in (1, 2, 1, "Reality Glove (L)", "Reality Glove", *([0.0] * 182)):
        builder.add_arg(value)
    packet = builder.build().dgram

    class FakeSocket:
        sent = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 9002)

        def settimeout(self, seconds):
            assert seconds > 0

        def recv(self, size):
            if self.sent:
                raise socket.timeout()
            self.sent = True
            return packet

    monkeypatch.setattr(check.socket, "socket", lambda *args: FakeSocket())
    out = tmp_path / "probe"
    assert check.probe(0.01, out)
    doc = json.loads((out / "osc_probe.json").read_text())
    assert doc["device_labels"] == ["Reality Glove", "Reality Glove (L)"]
    assert doc["addresses"]["/v1/animation/kinematic/all"] == {
        "messages": 1, "argument_counts": [187]}
    assert doc["physical_sensor_mapping"] == "Not inferred from OSC argument counts."


def test_timestamp_fallback_is_disclosed(tmp_path):
    root = paired_session(tmp_path)
    for path in root.rglob("*.jsonl"):
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        for row in rows:
            del row["capture_time"]
        path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    report = check.audit(root, tmp_path / "result", poses=("open_palm",))
    assert report["takes"][0]["clock"] == "wall_time"
    assert "Neither establishes device latency" in report["timing"]
