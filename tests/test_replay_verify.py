"""scripts/leap/replay_verify.py: the batch fidelity run, the scenario checker
and the dataset-unchanged check.

The synthetic dataset is shaped like `grasp set 2026-10-08 organized`: its
metadata tables are written for Excel (utf-8 with a BOM), every JSONL line is
one hand of one tracking frame with the 26 parent-relative joints from
`xr_hand.kinematics.absolute_to_relative` and `abs26` rounded to 1e-6 m, as
`leap_hand.recorder` writes them. Hands are chained in joint_frames'
convention (each child along -z of its parent). Expected counts are worked
out here from the fixture, not read back from the script. The last test
runs the four named scenarios on the real dataset and skips on a machine
that does not have it.
"""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from leap_hand import paper3 as p3
from xr_hand import joint_frames as jf
from xr_hand.joints import JOINT_INDEX, JOINT_NAMES
from xr_hand.kinematics import PARENT, absolute_to_relative, mat3_to_quat

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "replay_verify", REPO / "scripts" / "leap" / "replay_verify.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

DATASET = Path(r"C:\Users\nkim2\OneDrive\Desktop\xr trainer"
               r"\grasp set 2026-10-08 organized")
needs_dataset = pytest.mark.skipif(not DATASET.exists(),
                                   reason="real dataset not on this machine")
SIX = ("T_TMC_fe", "T_TMC_aa", "I_CMC", "M_CMC", "R_CMC", "L_CMC")
OUTPUTS = ("fidelity.json", "fidelity_per_recording.csv", "events.csv",
           "paper3_residuals.csv", "paper3_residuals.json",
           "paper3_pooled_by_class.csv", "paper3_fingertip_error.csv",
           "summary.md")


# --- a synthetic hand -----------------------------------------------------------------
BONE_M = {"PROXIMAL": 0.045, "INTERMEDIATE": 0.040, "DISTAL": 0.025, "TIP": 0.020}
SPREAD_M = {"THUMB": -0.030, "INDEX": -0.020, "MIDDLE": 0.0, "RING": 0.018,
            "LITTLE": 0.034}
FLEX_DEG = {"PROXIMAL": 20.0, "INTERMEDIATE": 30.0, "DISTAL": 15.0}


def hand_pose(side, k):
    """World (pos (26, 3) m, rot (26, 3, 3)): every child at a fixed offset
    along -z of its parent, turned by jf.rotation_from_angles. The fingers
    flex and spread a little more with every frame k, so each relation has
    a spread to fit a line to."""
    pos, rot = np.zeros((26, 3)), np.zeros((26, 3, 3))
    w = JOINT_INDEX["WRIST"]
    pos[w] = (0.03 + 0.001 * k, 0.18, -0.05)
    rot[w] = jf.rot_z(25.0) @ jf.rot_x(-30.0)
    for i in [JOINT_INDEX["PALM"]] + list(range(2, 26)):    # parents first
        name = JOINT_NAMES[i]
        digit, _, part = name.partition("_")
        off = ((0.0, 0.0, -0.045) if name == "PALM" else
               (SPREAD_M[digit], 0.0, -0.010) if part == "METACARPAL" else
               (0.0, 0.0, -BONE_M[part]))
        flex = FLEX_DEG.get(part, 0.0) * (1.0 + 0.08 * k)
        abd = 3.0 + 0.4 * k if part == "PROXIMAL" else 0.0
        par = PARENT[i]
        rot[i] = rot[par] @ jf.rotation_from_angles(flex, abd, 0.0, side)
        pos[i] = pos[par] + rot[par] @ np.asarray(off)
    return pos, rot


T0_US = 5_000_000_000
PERIOD_US = 11_111


def line(side, hand_id, frame_id, t_us, k, status=1):
    """One JSONL line as the recorder writes it."""
    pos, rot = hand_pose(side, k)
    joints = absolute_to_relative(pos, [mat3_to_quat(r) for r in rot])
    capture = round(1_790_000_000.0 + (t_us - T0_US) / 1e6 + 0.0003, 6)
    return {"wall_time": round(capture + 0.015 + 0.001 * (k % 3), 6),
            "timestamp": t_us / 1e6, "timestamp_us": t_us,
            "packet_counter": frame_id, "hand_side": side, "frame_id": frame_id,
            "status": status, "hand_id": hand_id, "framerate": 90.0,
            "capture_time": capture, "source": "leap", "space": "leap_desktop",
            "units": "m",
            "joints": [{"name": j.name, "x": j.x, "y": j.y, "z": j.z, "qw": j.qw,
                        "qx": j.qx, "qy": j.qy, "qz": j.qz} for j in joints],
            "abs26": [[round(float(c), 6) for c in p] for p in pos]}


# --- a synthetic dataset directory ------------------------------------------------------
CIRC1 = "grasps/p3_circular_power/take_01/p3_circular_power_right_take1.jsonl"
CIRC2 = "grasps/p3_circular_power/take_02/p3_circular_power_right_take2.jsonl"
PRIS1 = "grasps/p3_prismatic_precision/take_01/p3_prismatic_precision_left_take1.jsonl"
TRANS = ("transitions/p3_circular_power/take_01/"
         "open_hand_to_p3_circular_power_right_take1.jsonl")
STATIC_COLUMNS = ["grasp_name", "take", "hand", "hand_ids", "raw_jsonl_file",
                  "hand_frame_count", "frame_rate_hz",
                  "static_interval_first_frame_id", "static_interval_last_frame_id",
                  "representative_frame_index", "representative_frame_id"]
TRANSITION_COLUMNS = ["participant_id", "hand", "from_grasp", "to_grasp", "take",
                      "recording_file", "validity", "frames", "hand_ids",
                      "static_take_file"]


def write_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def write_table(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # with a BOM
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def make_dataset(root, nudge_um=0.0):
    """Three static takes and one transition.

    circular take 1: right hand id 7, 12 frames (1000-1011), one left-hand
        line of the same tracking frame and one status-0 line of the selected
        hand: 14 lines. Static interval 1003-1008 (6 frames).
    circular take 2: right hand id 7, 8 frames (1100-1107), interval
        1102-1104 (3 frames), flexed further than take 1.
    prismatic take 1: left hand id 11, 10 frames (2000-2009), interval
        2002-2007.
    transition to circular, take 1: right hand, id 21 for 6 frames
        (3000-3005), 0.4 s with no hand, id 22 for 6 frames (3041-3046), and
        one right-hand line of id 99 in between: 13 lines.
    nudge_um moves one recorded abs26 coordinate of circular take 1.
    """
    circ = [line("right", 7, 1000 + i, T0_US + i * PERIOD_US, i) for i in range(12)]
    if nudge_um:
        circ[5]["abs26"][10][0] = round(circ[5]["abs26"][10][0] + nudge_um * 1e-6, 6)
    other_hand = line("left", 8, 1003, T0_US + 3 * PERIOD_US, 3)
    invalid = line("right", 7, 1012, T0_US + 12 * PERIOD_US, 12, status=0)
    write_jsonl(root / CIRC1, circ[:4] + [other_hand] + circ[4:] + [invalid])
    t2 = T0_US + 500_000
    write_jsonl(root / CIRC2, [line("right", 7, 1100 + i, t2 + i * PERIOD_US, 6 + i)
                               for i in range(8)])
    t3 = T0_US + 1_000_000
    write_jsonl(root / PRIS1, [line("left", 11, 2000 + i, t3 + i * PERIOD_US, i)
                               for i in range(10)])
    t4 = T0_US + 2_000_000
    first = [line("right", 21, 3000 + i, t4 + i * PERIOD_US, i) for i in range(6)]
    t5 = t4 + 5 * PERIOD_US + 400_000                  # the camera's 0.4 s gap
    second = [line("right", 22, 3041 + i, t5 + i * PERIOD_US, 6 + i) for i in range(6)]
    stray = line("right", 99, 3043, t5 + 2 * PERIOD_US, 8)
    write_jsonl(root / TRANS, first + second[:3] + [stray] + second[3:])

    def static_row(grasp, take, hand, ids, path, n, a, b):
        return {"grasp_name": grasp, "take": take, "hand": hand, "hand_ids": ids,
                "raw_jsonl_file": path, "hand_frame_count": n, "frame_rate_hz": 90,
                "static_interval_first_frame_id": a,
                "static_interval_last_frame_id": b,
                "representative_frame_index": 0, "representative_frame_id": a}

    write_table(root / "metadata" / "grasps_summary.csv", STATIC_COLUMNS, [
        static_row("p3_circular_power", 1, "right", "7", CIRC1, 12, 1003, 1008),
        static_row("p3_circular_power", 2, "right", "7", CIRC2, 8, 1102, 1104),
        static_row("p3_prismatic_precision", 1, "left", "11", PRIS1, 10, 2002, 2007),
    ])
    write_table(root / "metadata" / "pose_transitions.csv", TRANSITION_COLUMNS, [
        {"participant_id": "P01", "hand": "right", "from_grasp": "open_hand",
         "to_grasp": "p3_circular_power", "take": 1, "recording_file": TRANS,
         "validity": "valid", "frames": 12, "hand_ids": "21 22",
         "static_take_file": CIRC1},
    ])
    return root


def run(argv):
    """(exit code, printed text) of the script's main()."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = rv.main([str(a) for a in argv])
    return code, buf.getvalue()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """The synthetic dataset and one --all run over it."""
    base = tmp_path_factory.mktemp("verify")
    root = make_dataset(base / "dataset")
    out = base / "out"
    code, text = run(["--dataset", root, "--out", out, "--all"])
    return root, out, code, text


# --- the batch run ------------------------------------------------------------------------
def test_batch_summary_line_and_outputs(batch):
    _, out, code, text = batch
    assert code == 0, text
    expected = "FIDELITY OK: static=3 transitions=1 frames=42 within_tolerance=4/4"
    assert text.strip().splitlines()[-1] == expected
    for name in OUTPUTS:
        assert (out / name).is_file(), name
    assert expected in (out / "summary.md").read_text(encoding="utf-8")


def test_per_recording_rows_count_every_excluded_line(batch):
    _, out, _, _ = batch
    rows = read_csv(out / "fidelity_per_recording.csv")
    assert [(r["kind"], r["name"], r["take"]) for r in rows] == [
        ("static", "p3_circular_power", "1"), ("static", "p3_circular_power", "2"),
        ("static", "p3_prismatic_precision", "1"),
        ("transition", "p3_circular_power", "1")]
    circ, _, pris, trans = rows
    assert (circ["lines_total"], circ["frames_selected"], circ["excluded_other_hand"],
            circ["excluded_other_id"], circ["excluded_invalid_status"]) == (
        "14", "12", "1", "0", "1")
    assert (trans["lines_total"], trans["frames_selected"],
            trans["excluded_other_id"], trans["hand_ids"]) == ("13", "12", "1", "21 22")
    assert (trans["segments"], trans["n_id_change"], trans["n_gap"]) == ("2", "1", "1")
    assert trans["frame_ids_missing"] == "35"                 # 3006-3040
    assert circ["n_invalid_status"] == "1" and circ["segments"] == "1"
    assert pris["hand_side"] == "left" and pris["paper3_class"] == "prismatic"
    for r in rows:
        assert r["within_tolerance"] == "true"
        assert r["clock_source"] == "timestamp_us" and r["clock_fallback"] == "false"
        assert float(r["pos_residual_um_max"]) <= 0.87       # sqrt(3)/2 um
        assert float(r["chain_deg_max"]) <= 1e-6
    assert float(trans["max_dt_s"]) == pytest.approx(0.4, abs=1e-9)


def test_fidelity_json_totals(batch):
    _, out, _, _ = batch
    doc = json.loads((out / "fidelity.json").read_text(encoding="utf-8"))
    t = doc["totals"]
    assert (t["static"], t["transitions"], t["frames_selected"], t["lines_total"]) == (
        3, 1, 42, 45)
    assert t["lines_excluded"] == {"other_hand": 1, "other_id": 1, "invalid_status": 1}
    assert t["within_tolerance"] == 4 and t["clock_fallbacks"] == []
    assert t["events_by_kind"] == {"gap": 1, "id_change": 1, "invalid_status": 1}
    assert doc["gap_threshold_s"] == 0.05
    assert doc["tolerances"]["position_m"]["value"] == 1e-6
    assert len(doc["per_recording"]) == 4
    gap = doc["worst"]["longest_gap_s"]
    assert gap["value"] == pytest.approx(0.4, abs=1e-9)
    assert (gap["kind"], gap["frame_id_before"], gap["frame_id_after"]) == (
        "transition", 3005, 3041)
    for key in ("position_residual_um", "chain_deg", "chain_m", "bone_direction_deg",
                "capture_vs_timestamp_interval_ms", "writer_delay_ms"):
        assert doc["worst"][key]["recording"], key


def test_events_csv_has_the_identity_change_and_the_gap(batch):
    _, out, _, _ = batch
    events = read_csv(out / "events.csv")
    assert sorted((e["kind"], e["recording_kind"], e["name"]) for e in events) == [
        ("gap", "transition", "p3_circular_power"),
        ("id_change", "transition", "p3_circular_power"),
        ("invalid_status", "static", "p3_circular_power")]
    for e in events:
        if e["kind"] in ("gap", "id_change"):
            assert (e["frame_id_before"], e["frame_id_after"], e["index"]) == (
                "3005", "3041", "6")
            assert float(e["dt_s"]) == pytest.approx(0.4, abs=1e-9)
        else:
            assert e["frame_id_before"] == e["frame_id_after"] == "1012"


def test_paper3_rows_statuses_and_duplicates(batch):
    _, out, _, _ = batch
    rows = read_csv(out / "paper3_residuals.csv")
    assert len(rows) == 4 * len(p3.RELATIONS)
    by = {(r["kind"], r["name"], r["take"], r["relation"]): r for r in rows}
    circ = lambda rel: by[("static", "p3_circular_power", "1", rel)]
    pris = lambda rel: by[("static", "p3_prismatic_precision", "1", rel)]
    trans = lambda rel: by[("transition", "p3_circular_power", "1", rel)]
    applied = circ("T6:I_MCP_fe:I_PIP")
    assert applied["status"] == "applied" and applied["class"] == "circular"
    assert (applied["all_n"], applied["hold_n"]) == ("12", "6")
    assert applied["mcp_order"] == "paper_order" and applied["all_main_n"] == "12"
    assert trans("T6:I_MCP_fe:I_PIP")["all_n"] == "12"
    assert trans("T6:I_MCP_fe:I_PIP")["hold_n"] == ""        # no static interval
    # eq. 8 repeats Table 6's PIP/DIP relation where both apply (circular)
    assert circ("EQ8:I_PIP:I_DIP")["duplicate_of"] == "T6:I_PIP:I_DIP"
    assert circ("EQ8:I_PIP:I_DIP")["all_mean"] == circ("T6:I_PIP:I_DIP")["all_mean"]
    assert pris("EQ8:I_PIP:I_DIP")["status"] == "not_applied"
    assert pris("EQ8:I_PIP:I_DIP")["duplicate_of"] == ""
    assert pris("T7:I_PIP:I_DIP")["status"] == "applied"
    for rel in ("T6:T_TMC_fe:T_MCP_fe", "T6:I_CMC:M_CMC"):
        assert circ(rel)["status"] == "unavailable" and circ(rel)["all_n"] == ""
    doc = json.loads((out / "paper3_residuals.json").read_text(encoding="utf-8"))
    assert len(doc["per_recording"]) == 4
    assert "|rp|" in doc["fingertip_note"]


def test_pooled_statistics_concatenate_frames(batch):
    """Pooled = one series over every frame of the class: the frame-weighted
    mean and RMSE of the two takes, not the mean of their means."""
    _, out, _, _ = batch
    rel = "T6:I_MCP_fe:I_PIP"
    takes = [r for r in read_csv(out / "paper3_residuals.csv")
             if (r["kind"], r["class"], r["relation"]) == ("static", "circular", rel)]
    assert [r["take"] for r in takes] == ["1", "2"]
    n = [int(r["all_n"]) for r in takes]
    mean = [float(r["all_mean"]) for r in takes]
    rmse = [float(r["all_rmse"]) for r in takes]
    pooled = read_csv(out / "paper3_pooled_by_class.csv")
    assert len(pooled) == 3 * len(p3.RELATIONS)   # static circular, static prismatic, transition circular
    row = next(r for r in pooled
               if (r["kind"], r["class"], r["relation"]) == ("static", "circular", rel))
    assert (row["recordings"], row["frames"], row["hold_frames"]) == ("2", "20", "9")
    assert int(row["all_n"]) == n[0] + n[1] == 20 and row["hold_n"] == "9"
    weighted = (n[0] * mean[0] + n[1] * mean[1]) / 20
    assert float(row["all_mean"]) == pytest.approx(weighted, rel=1e-9)
    assert abs(weighted - (mean[0] + mean[1]) / 2) > 0.1     # the test can tell them apart
    pooled_rmse = math.sqrt((n[0] * rmse[0] ** 2 + n[1] * rmse[1] ** 2) / 20)
    assert float(row["all_rmse"]) == pytest.approx(pooled_rmse, rel=1e-9)


def test_fingertip_error_rows(batch):
    _, out, _, _ = batch
    rows = read_csv(out / "paper3_fingertip_error.csv")
    assert len(rows) == 4
    for r in rows:
        assert r["status"] == "applied" and r["thumb"] == "unavailable"
        assert float(r["gesture_error_pct"]) >= 0.0 and r["I_n"] != "0"
        assert "|rp|" in r["note"]
    assert {r["paper_table8_pct"] for r in rows} == {"7.67", "5.9"}
    assert rows[0]["I_mean_pct_hold"] != "" and rows[3]["I_mean_pct_hold"] == ""


def test_a_residual_over_tolerance_fails_the_run(tmp_path):
    """Negative control: one recorded abs26 coordinate moved by 3 um."""
    root = make_dataset(tmp_path / "dataset", nudge_um=3.0)
    code, text = run(["--dataset", root, "--out", tmp_path / "out", "--all"])
    assert code == 1
    assert text.strip().splitlines()[-1].startswith(
        "FIDELITY FAIL: static=3 transitions=1 frames=42 within_tolerance=3/4")
    rows = read_csv(tmp_path / "out" / "fidelity_per_recording.csv")
    assert [r["within_tolerance"] for r in rows] == ["false", "true", "true", "true"]
    assert float(rows[0]["pos_residual_um_max"]) == pytest.approx(3.0, abs=0.9)


def test_a_recording_that_cannot_be_loaded_fails_the_run(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    (root / TRANS).unlink()
    code, text = run(["--dataset", root, "--out", tmp_path / "out", "--all"])
    assert code == 1
    assert "LOAD FAILED: transition p3_circular_power take 1" in text
    last = text.strip().splitlines()[-1]
    assert last.startswith("FIDELITY FAIL: static=3 transitions=0 frames=30 "
                           "within_tolerance=3/4") and "load_failures=1" in last


def test_subsets_by_name_and_take(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    code, text = run(["--dataset", root, "--out", tmp_path / "out",
                      "--static", "p3_prismatic_precision", 1,
                      "--transition", "p3_circular_power", 1])
    assert code == 0, text
    assert text.strip().splitlines()[-1] == (
        "FIDELITY OK: static=1 transitions=1 frames=22 within_tolerance=2/2")


def test_output_inside_the_dataset_is_refused(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    code, text = run(["--dataset", root, "--out", root / "results", "--all"])
    assert code == 2 and "inside the dataset" in text
    assert not (root / "results").exists()


# --- the scenario reports and their checker ------------------------------------------------
def test_scenarios_written_for_the_names_the_dataset_has(tmp_path):
    """The synthetic set carries circular and prismatic take 1 but neither
    the writer-stall take nor the identity-change take: two reports are
    written and pass, the other two are named as failures."""
    root = make_dataset(tmp_path / "dataset")
    out = tmp_path / "out"
    code, text = run(["--dataset", root, "--out", out, "--scenarios"])
    assert code == 1
    assert "SCENARIO FAILED: timestamp_anomaly" in text
    assert "SCENARIO FAILED: identity_change" in text
    assert "FIDELITY" not in text                             # no batch was asked for
    circ = json.loads((out / "scenarios" / "circular.json").read_text(encoding="utf-8"))
    assert circ["recording"]["name"] == "p3_circular_power"
    assert sorted(circ["unavailable_coordinates"]) == sorted(SIX)
    assert circ["fidelity"]["within_tolerance"] is True
    assert circ["paper3"]["summary"]["applied"] > 0
    failures = rv.check_scenarios(out)
    assert [f.split(":")[0] for f in failures] == ["timestamp_anomaly", "identity_change"]


def good_scenarios():
    """Hand-written reports holding the known facts (the schema the script
    writes, cut to what the checker reads)."""
    def grasp(name):
        status = {n: "unavailable" for n in SIX}
        status.update(I_PIP="available", I_DIP="available")
        return {"recording": {"kind": "static", "name": name, "take": 1},
                "paper3": {"class": "circular", "relations": [
                    {"id": "T6:I_CMC:M_CMC", "status": "unavailable", "stats_all": None},
                    {"id": "T6:I_PIP:I_DIP", "status": "applied",
                     "stats_all": {"n": 449, "mean": 2.5}}]},
                "coordinate_status": status}
    return {
        "circular": grasp("p3_circular_power"),
        "prismatic": grasp("p3_prismatic_precision"),
        "timestamp_anomaly": {
            "recording": {"kind": "static", "take": 4,
                          "name": "p3_circular_precision_thumb_2_fingers"},
            "window": {"writer_span_s": 1.9993, "camera_span_s": 1.178128,
                       "clock_source": "timestamp_us", "clock_fallback": False}},
        "identity_change": {
            "recording": {"kind": "static", "name": "p2_s03_power_grasp_finger_side",
                          "take": 1},
            "events": [{"kind": "id_change", "index": 61, "frame_id_before": 88967,
                        "frame_id_after": 89008, "dt_s": 0.455302},
                       {"kind": "gap", "index": 61, "frame_id_before": 88967,
                        "frame_id_after": 89008, "dt_s": 0.455302}],
            "segments": [{"start": 0, "end": 60, "hand_id": 573},
                         {"start": 61, "end": 408, "hand_id": 574}],
            "midpoint_pose": {"source": "held", "frame_id": 88967}},
    }


def write_scenarios(directory, reports):
    d = directory / "scenarios"
    d.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (d / f"{name}.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


def test_checker_passes_the_known_facts(tmp_path):
    write_scenarios(tmp_path, good_scenarios())
    assert rv.check_scenarios(tmp_path) == []
    assert rv.check_scenarios(tmp_path / "scenarios") == []  # the folder itself
    code, text = run(["--check-scenarios", tmp_path])
    assert code == 0 and text.strip() == "SCENARIOS OK"


@pytest.mark.parametrize("which,breaks", [
    ("timestamp_anomaly", lambda s: s["window"].update(writer_span_s=1.178128)),
    ("timestamp_anomaly", lambda s: s["window"].update(camera_span_s=1.999)),
    ("timestamp_anomaly", lambda s: s["window"].update(clock_source="wall_time",
                                                       clock_fallback=True)),
    ("identity_change", lambda s: s["midpoint_pose"].update(source="interpolated")),
    ("identity_change", lambda s: s["segments"].pop()),
    ("identity_change", lambda s: s["events"][0].update(dt_s=0.40)),
    ("identity_change", lambda s: s["events"][0].update(frame_id_after=89009)),
    ("circular", lambda s: s["coordinate_status"].update(T_TMC_aa="available")),
    ("prismatic", lambda s: s["paper3"]["relations"][1].update(status="not_applied")),
    ("prismatic", lambda s: s["recording"].update(take=2)),
])
def test_checker_names_what_is_wrong(tmp_path, which, breaks):
    reports = good_scenarios()
    breaks(reports[which])
    write_scenarios(tmp_path, reports)
    failures = rv.check_scenarios(tmp_path)
    assert [f.split(":")[0] for f in failures] == [which]
    code, text = run(["--check-scenarios", tmp_path])
    assert code == 1 and text.startswith(f"SCENARIOS FAIL: {which}")
    assert "SCENARIOS OK" not in text


def test_checker_fails_on_a_missing_report(tmp_path):
    reports = good_scenarios()
    del reports["identity_change"]
    write_scenarios(tmp_path, reports)
    assert [f.split(":")[0] for f in rv.check_scenarios(tmp_path)] == ["identity_change"]
    code, text = run(["--check-scenarios", tmp_path / "nowhere"])
    assert code == 1 and text.count("cannot read") == 4


# --- the dataset-unchanged check -------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset_unchanged_then_changed(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    assert len(files) == 6
    sibling = tmp_path / "dataset_copy" / "x.txt"            # shares the prefix only
    sibling.parent.mkdir()
    sibling.write_text("x", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({str(p): sha256(p) for p in files + [sibling]}),
                        encoding="utf-8")
    root_arg = str(root).upper() if os.name == "nt" else str(root)  # case-blind match
    code, text = run(["--check-dataset-unchanged", baseline, root_arg])
    assert code == 0 and "DATASET UNCHANGED (6 files, 4 JSONL)" in text

    with open(root / CIRC1, "ab") as f:                       # negative control
        f.write(b" ")
    (root / "metadata" / "pose_transitions.csv").unlink()
    code, text = run(["--check-dataset-unchanged", baseline, root])
    assert code == 1 and "UNCHANGED" not in text
    assert "DATASET CHANGED: 2 files differ/missing" in text
    assert "p3_circular_power_right_take1.jsonl" in text and "pose_transitions.csv" in text


def test_dataset_check_refuses_an_empty_comparison(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({str(p): sha256(p) for p in root.rglob("*")
                                    if p.is_file()}), encoding="utf-8")
    code, text = run(["--check-dataset-unchanged", baseline, elsewhere])
    assert code == 1 and "UNCHANGED" not in text
    assert "no baseline entry lies under" in text


# --- the real dataset --------------------------------------------------------------------------
@needs_dataset
def test_real_scenarios_hold_the_known_facts(tmp_path):
    code, text = run(["--dataset", DATASET, "--out", tmp_path, "--scenarios"])
    assert code == 0, text
    assert "SCENARIOS WRITTEN: 4" in text
    anomaly = json.loads((tmp_path / "scenarios" / "timestamp_anomaly.json")
                         .read_text(encoding="utf-8"))
    w = anomaly["window"]
    assert w["frames"] == 107
    assert w["camera_span_s"] == pytest.approx(1.178128, abs=1e-9)
    jump = w["largest_writer_jump"]
    assert (jump["frame_id_before"], jump["frame_id_after"]) == (151852, 151853)
    assert jump["camera_ms"] == pytest.approx(10.809, abs=1e-9)
    assert jump["writer_s"] > 0.8
    code, text = run(["--check-scenarios", tmp_path])
    assert code == 0 and text.strip() == "SCENARIOS OK"
