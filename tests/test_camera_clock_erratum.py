"""scripts/leap/camera_clock_erratum.py: the camera-clock erratum for the
timing columns the dataset's metadata gives on the writer clock.

The synthetic dataset reuses test_replay_verify's hand, line format and
tables and adds the timing columns `grasp set 2026-10-08 organized`
carries, written the way the dataset's own code cut them
(leap_hand.grasp_audit: the static columns from `wall_time`;
leap_hand.transition: the transition instants from `capture_time`; both
rounded to 1 ms). The writer clock is then made to stall where the real
one does. Expected values are worked out here from the fixture's own lines,
never read back from the script. The last two tests run on the real dataset
and skip on a machine that does not have it.
"""
import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

import test_replay_verify as fx

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "camera_clock_erratum", REPO / "scripts" / "leap" / "camera_clock_erratum.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)

PERIOD_S = fx.PERIOD_US / 1e6
TRANS2 = ("transitions/p3_prismatic_precision/take_02/"
          "open_hand_to_p3_prismatic_precision_left_take2.jsonl")
STATIC_COLUMNS = fx.STATIC_COLUMNS + [
    "recording_duration_s", "camera_clock_duration_s", "static_interval_start_s",
    "static_interval_end_s", "static_interval_duration_s", "first_wall_time",
    "static_interval_start", "static_interval_end"]
TRANSITION_COLUMNS = fx.TRANSITION_COLUMNS + [
    "cue_wall_time", "cue_time_s", "movement_start_time_s", "settled_time_s",
    "transition_duration_s", "first_frame_capture_time"]
OUTPUTS = ("static_takes.csv", "transitions.csv", "README.md", "summary.md")


def delayed(lines, delays):
    """wall_time = capture_time + delay, per line: the writer clock as the
    recorder stamps it, rounded as the recorder rounds."""
    for d, delay in zip(lines, delays):
        d["wall_time"] = round(d["capture_time"] + delay, 6)
    return lines


def static_meta(lines, first, last, declared_s):
    """The writer-clock columns as leap_hand.grasp_audit cuts them: the
    interval starts at frame `first`'s wall_time and is declared `declared_s`
    long; seconds from the hand's first wall_time, rounded to 1 ms."""
    wall = [d["wall_time"] for d in lines]
    start = next(d["wall_time"] for d in lines if d["frame_id"] == first)
    end = round(start + declared_s, 6)
    camera = (lines[-1]["timestamp_us"] - lines[0]["timestamp_us"]) / 1e6
    return {"recording_duration_s": round(wall[-1] - wall[0], 3),
            "camera_clock_duration_s": round(camera, 3),
            "static_interval_start_s": round(start - wall[0], 3),
            "static_interval_end_s": round(end - wall[0], 3),
            "static_interval_duration_s": declared_s, "first_wall_time": wall[0],
            "static_interval_start": start, "static_interval_end": end,
            "static_interval_first_frame_id": first, "static_interval_last_frame_id": last}


def make_dataset(root):
    """Two static takes and two transitions.

    circular take 1: right hand id 7, frames 1000-1011 one period apart.
        The writer stalls between 1005 and 1006: frames 1006-1008 are written
        0.5 s late in a burst, 1009-1011 at the usual 15 ms again. Static
        interval 1003-1008 (6 frames, 5 periods of camera time), declared as
        0.6 s of writer time from frame 1003's wall_time.
    prismatic take 1: left hand id 11, frames 2000-2009, no stall (15 to
        17 ms delay), interval 2002-2007 declared 0.06 s.
    transition to circular take 1: right hand id 21, frames 3000-3019.
        The writer stalls between 3013 and 3014 and stays 0.3 s behind.
        Movement start at frame 3006, settled at 3015 (both as capture
        times rounded to 1 ms), the cue 2 ms before frame 3000's capture.
    transition to prismatic take 2: left hand id 31, frames 4000-4007,
        invalid: no movement start, settle or duration.
    """
    circ = [fx.line("right", 7, 1000 + i, fx.T0_US + i * fx.PERIOD_US, i)
            for i in range(12)]
    cap = [d["capture_time"] for d in circ]
    burst = [cap[6] - cap[i] + 0.515 + 0.0002 * (i - 6) for i in range(6, 9)]
    delayed(circ, [0.015] * 6 + burst + [0.015] * 3)
    fx.write_jsonl(root / fx.CIRC1, circ)
    pris = [fx.line("left", 11, 2000 + i, fx.T0_US + 1_000_000 + i * fx.PERIOD_US, i)
            for i in range(10)]
    fx.write_jsonl(root / fx.PRIS1, pris)
    t4 = fx.T0_US + 2_000_000
    trans = [fx.line("right", 21, 3000 + i, t4 + i * fx.PERIOD_US, i) for i in range(20)]
    delayed(trans, [0.015] * 14 + [0.315] * 6)
    fx.write_jsonl(root / fx.TRANS, trans)
    t5 = fx.T0_US + 3_000_000
    trans2 = [fx.line("left", 31, 4000 + i, t5 + i * fx.PERIOD_US, i) for i in range(8)]
    fx.write_jsonl(root / TRANS2, trans2)

    def static_row(grasp, take, hand, ids, path, lines, first, last, declared_s):
        return {"grasp_name": grasp, "take": take, "hand": hand, "hand_ids": ids,
                "raw_jsonl_file": path, "hand_frame_count": len(lines),
                "frame_rate_hz": 90, "representative_frame_index": 0,
                "representative_frame_id": first,
                **static_meta(lines, first, last, declared_s)}

    fx.write_table(root / "metadata" / "grasps_summary.csv", STATIC_COLUMNS, [
        static_row("p3_circular_power", 1, "right", "7", fx.CIRC1, circ, 1003, 1008, 0.6),
        static_row("p3_prismatic_precision", 1, "left", "11", fx.PRIS1, pris,
                   2002, 2007, 0.06),
    ])

    def transition_row(to_grasp, take, hand, ids, path, lines, validity, static_file,
                       start_index=None, settled_index=None, cue_before_s=0.002):
        ffc = lines[0]["capture_time"]
        blank = {"movement_start_time_s": "", "settled_time_s": "",
                 "transition_duration_s": ""}
        if start_index is not None:
            a = round(lines[start_index]["capture_time"] - ffc, 3)
            b = round(lines[settled_index]["capture_time"] - ffc, 3)
            blank = {"movement_start_time_s": a, "settled_time_s": b,
                     "transition_duration_s": round(b - a, 3)}
        return {"participant_id": "P01", "hand": hand, "from_grasp": "open_hand",
                "to_grasp": to_grasp, "take": take, "recording_file": path,
                "validity": validity, "frames": len(lines), "hand_ids": ids,
                "static_take_file": static_file,
                "cue_wall_time": round(ffc - cue_before_s, 6), "cue_time_s": -cue_before_s,
                "first_frame_capture_time": ffc, **blank}

    fx.write_table(root / "metadata" / "pose_transitions.csv", TRANSITION_COLUMNS, [
        transition_row("p3_circular_power", 1, "right", "21", fx.TRANS, trans, "valid",
                       fx.CIRC1, start_index=6, settled_index=15),
        transition_row("p3_prismatic_precision", 2, "left", "31", TRANS2, trans2,
                       "invalid", fx.PRIS1, cue_before_s=0.003),
    ])
    return root, {"circ": circ, "pris": pris, "trans": trans, "trans2": trans2}


def run(argv):
    """(exit code, printed text) of the script's main()."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = ce.main([str(a) for a in argv])
    return code, buf.getvalue()


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    base = tmp_path_factory.mktemp("erratum")
    root, lines = make_dataset(base / "dataset")
    out = base / "out"
    code, text = run(["--dataset", root, "--out", out])
    return root, lines, out, code, text


def wall_jumps(lines):
    return [b["wall_time"] - a["wall_time"] for a, b in zip(lines, lines[1:])]


# --- the run and its outputs --------------------------------------------------------------
def test_run_writes_the_outputs_and_the_summary_line(batch):
    _, _, out, code, text = batch
    assert code == 0, text
    assert text.strip().splitlines()[-1] == (
        "ERRATUM OK: static=2 transitions=2 static_interval_over_one_period=1")
    for name in OUTPUTS:
        assert (out / name).is_file(), name


def test_static_rows(batch):
    _, lines, out, _, _ = batch
    rows = fx.read_csv(out / "static_takes.csv")
    assert [(r["grasp_name"], r["take"]) for r in rows] == [
        ("p3_circular_power", "1"), ("p3_prismatic_precision", "1")]
    circ, pris = rows
    for r in rows:
        assert r["clock_source"] == "timestamp_us" and r["clock_fallback"] == "false"
        assert float(r["frame_period_s"]) == pytest.approx(PERIOD_S, abs=1e-6)
        assert float(r["first_frame_wall_minus_first_wall_time_ms"]) == 0.0

    # circular take 1: the writer caught up by the end of the take, so the
    # whole-take durations agree; inside the interval it was 0.5 s behind
    c = lines["circ"]
    assert circ["frames_selected"] == "12" and circ["static_interval_frames"] == "6"
    assert float(circ["recording_duration_metadata_s"]) == 0.122
    assert float(circ["recording_duration_camera_s"]) == pytest.approx(11 * PERIOD_S, abs=1e-6)
    assert float(circ["recording_duration_difference_s"]) == pytest.approx(
        11 * PERIOD_S - 0.122, abs=1e-6)
    assert float(circ["metadata_camera_clock_duration_s"]) == 0.122
    assert float(circ["recording_largest_writer_stall_s"]) == pytest.approx(
        max(wall_jumps(c)), abs=1e-6)
    assert (circ["static_interval_first_frame_id"], circ["static_interval_last_frame_id"]) == (
        "1003", "1008")
    assert float(circ["static_interval_start_metadata_s"]) == 0.033
    assert float(circ["static_interval_start_camera_s"]) == pytest.approx(3 * PERIOD_S, abs=1e-6)
    assert float(circ["static_interval_end_metadata_s"]) == 0.633
    assert float(circ["static_interval_end_camera_s"]) == pytest.approx(8 * PERIOD_S, abs=1e-6)
    assert float(circ["static_interval_end_difference_s"]) == pytest.approx(
        8 * PERIOD_S - 0.633, abs=1e-6)
    assert float(circ["static_interval_duration_metadata_s"]) == 0.6
    writer_span = c[8]["wall_time"] - c[3]["wall_time"]
    assert writer_span == pytest.approx(3 * PERIOD_S + 0.5004, abs=1e-6)
    assert float(circ["static_interval_writer_span_s"]) == pytest.approx(writer_span, abs=1e-6)
    assert float(circ["static_interval_camera_s"]) == pytest.approx(5 * PERIOD_S, abs=1e-6)
    assert float(circ["static_interval_difference_s"]) == pytest.approx(
        5 * PERIOD_S - 0.6, abs=1e-6)
    assert float(circ["static_interval_camera_minus_writer_span_s"]) == pytest.approx(
        5 * PERIOD_S - writer_span, abs=1e-6)
    assert circ["static_interval_over_one_period"] == "true"
    assert float(circ["static_interval_largest_writer_stall_s"]) == pytest.approx(
        PERIOD_S + 0.5, abs=1e-6)
    assert (circ["stall_frame_id_before"], circ["stall_frame_id_after"]) == ("1005", "1006")
    assert float(circ["stall_camera_ms"]) == pytest.approx(PERIOD_S * 1e3, abs=1e-3)

    # prismatic take 1: no stall; the declared 0.06 s is 4.4 ms over the
    # frames' camera span, under one period, so it is not flagged
    p = lines["pris"]
    assert float(pris["static_interval_duration_metadata_s"]) == 0.06
    assert float(pris["static_interval_camera_s"]) == pytest.approx(5 * PERIOD_S, abs=1e-6)
    assert float(pris["static_interval_writer_span_s"]) == pytest.approx(
        p[7]["wall_time"] - p[2]["wall_time"], abs=1e-6)
    assert float(pris["static_interval_difference_s"]) == pytest.approx(
        5 * PERIOD_S - 0.06, abs=1e-6)
    assert pris["static_interval_over_one_period"] == "false"
    assert float(pris["static_interval_largest_writer_stall_s"]) == pytest.approx(
        max(wall_jumps(p[2:8])), abs=1e-6)
    assert abs(float(pris["recording_duration_difference_s"])) < 0.003


def test_transition_rows(batch):
    _, lines, out, _, _ = batch
    rows = fx.read_csv(out / "transitions.csv")
    assert [(r["to_grasp"], r["take"], r["validity"]) for r in rows] == [
        ("p3_circular_power", "1", "valid"), ("p3_prismatic_precision", "2", "invalid")]
    valid, invalid = rows
    t = lines["trans"]
    assert valid["frames_selected"] == "20" and valid["clock_source"] == "timestamp_us"
    assert float(valid["first_frame_capture_minus_first_frame_capture_time_ms"]) == 0.0
    assert float(valid["recording_writer_span_s"]) == pytest.approx(
        t[-1]["wall_time"] - t[0]["wall_time"], abs=1e-6)
    assert float(valid["recording_camera_s"]) == pytest.approx(19 * PERIOD_S, abs=1e-6)
    assert float(valid["recording_largest_writer_stall_s"]) == pytest.approx(
        PERIOD_S + 0.3, abs=1e-6)
    # the cue (cue_wall_time) falls 2 ms before the first frame: carried onto
    # the camera timeline by that frame, it keeps its value
    assert float(valid["cue_time_metadata_s"]) == -0.002
    assert valid["cue_frame_id"] == "3000"
    assert float(valid["cue_time_camera_s"]) == pytest.approx(-0.002, abs=1e-6)
    assert float(valid["cue_time_difference_s"]) == pytest.approx(0.0, abs=1e-6)
    # the instants land on frames 3006 and 3015 by capture_time, within the
    # metadata's 1 ms rounding; by wall_time they miss by the writer delay
    for key, index in (("movement_start", 6), ("settled", 15)):
        assert valid[f"{key}_frame_id"] == str(3000 + index)
        residual = float(valid[f"{key}_capture_residual_ms"])
        assert abs(residual) <= 0.5
        assert abs(float(valid[f"{key}_wall_residual_ms"])) > 2.0
        assert float(valid[f"{key}_camera_s"]) == pytest.approx(index * PERIOD_S, abs=1e-6)
        meta = float(valid[f"{key}_metadata_s"])
        assert float(valid[f"{key}_difference_s"]) == pytest.approx(
            index * PERIOD_S - meta, abs=1e-6)
    assert float(valid["transition_duration_metadata_s"]) == 0.1
    assert float(valid["transition_duration_camera_s"]) == pytest.approx(9 * PERIOD_S, abs=1e-6)
    assert float(valid["transition_duration_difference_s"]) == pytest.approx(
        9 * PERIOD_S - 0.1, abs=1e-6)
    assert float(valid["transition_duration_writer_span_s"]) == pytest.approx(
        t[15]["wall_time"] - t[6]["wall_time"], abs=1e-6)
    assert t[15]["wall_time"] - t[6]["wall_time"] == pytest.approx(9 * PERIOD_S + 0.3, abs=1e-6)
    assert float(valid["transition_largest_writer_stall_s"]) == pytest.approx(
        PERIOD_S + 0.3, abs=1e-6)
    assert (valid["stall_frame_id_before"], valid["stall_frame_id_after"]) == ("3013", "3014")

    # invalid: the metadata has no start, settle or duration, so neither does
    # the erratum; the cue and the recording spans are still given
    assert float(invalid["cue_time_metadata_s"]) == -0.003
    assert float(invalid["cue_time_camera_s"]) == pytest.approx(-0.003, abs=1e-6)
    assert float(invalid["recording_camera_s"]) == pytest.approx(7 * PERIOD_S, abs=1e-6)
    for key in ("movement_start_metadata_s", "movement_start_frame_id",
                "movement_start_camera_s", "settled_camera_s",
                "transition_duration_metadata_s", "transition_duration_camera_s",
                "transition_duration_difference_s", "transition_duration_writer_span_s",
                "transition_largest_writer_stall_s", "stall_frame_id_before"):
        assert invalid[key] == "", key


def test_summary_and_readme(batch):
    _, _, out, _, _ = batch
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "ERRATUM OK: static=2 transitions=2 static_interval_over_one_period=1" in summary
    assert "p3_circular_power take 1" in summary
    assert "p3_prismatic_precision take 1" not in summary.split("more than one frame period")[1].split("\n\n")[0]
    readme = (out / "README.md").read_text(encoding="utf-8")
    for phrase in ("timestamp_us", "capture_time", "wall_time", "unchanged",
                   "static_takes.csv", "transitions.csv", "--check-dataset-unchanged",
                   "nearest selected frame by `capture_time`", "**How to apply.**",
                   "static_interval_camera_s", "in place of `static_interval_duration_s`"):
        assert phrase in readme, phrase


def test_subsets_by_name_and_take(tmp_path):
    root, _ = make_dataset(tmp_path / "dataset")
    code, text = run(["--dataset", root, "--out", tmp_path / "out",
                      "--static", "p3_prismatic_precision", 1])
    assert code == 0, text
    assert text.strip().splitlines()[-1] == (
        "ERRATUM OK: static=1 transitions=0 static_interval_over_one_period=0")
    assert len(fx.read_csv(tmp_path / "out" / "static_takes.csv")) == 1
    assert fx.read_csv(tmp_path / "out" / "transitions.csv") == []


def test_output_inside_the_dataset_is_refused(tmp_path):
    root, _ = make_dataset(tmp_path / "dataset")
    code, text = run(["--dataset", root, "--out", root / "results"])
    assert code == 2 and "inside the dataset" in text
    assert not (root / "results").exists()


def test_a_recording_that_cannot_be_loaded_fails_the_run(tmp_path):
    root, _ = make_dataset(tmp_path / "dataset")
    (root / TRANS2).unlink()
    code, text = run(["--dataset", root, "--out", tmp_path / "out"])
    assert code == 1
    assert "LOAD FAILED: transition p3_prismatic_precision take 2" in text
    last = text.strip().splitlines()[-1]
    assert last.startswith("ERRATUM FAIL: static=2 transitions=1 "
                           "static_interval_over_one_period=1") and "load_failures=1" in last
    assert len(fx.read_csv(tmp_path / "out" / "static_takes.csv")) == 2
    assert len(fx.read_csv(tmp_path / "out" / "transitions.csv")) == 1
    assert "LOAD FAILED" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")


# --- the real dataset --------------------------------------------------------------------------
@fx.needs_dataset
def test_real_data_thumb_2_fingers_take_4(tmp_path):
    """The known writer stall: 2.000 s declared, 1.999 s of writer time over
    the interval's frames, 1.178 s of camera time."""
    code, text = run(["--dataset", fx.DATASET, "--out", tmp_path,
                      "--static", "p3_circular_precision_thumb_2_fingers", 4])
    assert code == 0, text
    (row,) = fx.read_csv(tmp_path / "static_takes.csv")
    assert (row["static_interval_first_frame_id"], row["static_interval_last_frame_id"]) == (
        "151749", "151855")
    assert row["static_interval_frames"] == "107"
    assert row["clock_source"] == "timestamp_us" and row["clock_fallback"] == "false"
    assert float(row["static_interval_duration_metadata_s"]) == 2.0
    assert float(row["static_interval_writer_span_s"]) == pytest.approx(1.999, abs=1e-3)
    assert float(row["static_interval_camera_s"]) == pytest.approx(1.178128, abs=1e-6)
    assert float(row["static_interval_difference_s"]) == pytest.approx(-0.821872, abs=1e-6)
    assert row["static_interval_over_one_period"] == "true"
    assert float(row["static_interval_largest_writer_stall_s"]) == pytest.approx(0.8074, abs=1e-3)
    assert (row["stall_frame_id_before"], row["stall_frame_id_after"]) == ("151852", "151853")
    assert float(row["stall_camera_ms"]) == pytest.approx(10.809, abs=1e-3)
    assert float(row["recording_duration_metadata_s"]) == 4.972
    assert float(row["recording_duration_camera_s"]) == pytest.approx(4.977819, abs=1e-6)
    assert float(row["metadata_camera_clock_duration_s"]) == 4.978


@fx.needs_dataset
def test_real_data_transition_times_are_on_the_capture_clock(tmp_path):
    """p1_cylindrical transition take 4: movement start 0.755 s and settle
    1.555 s land on frames 871 and 943 by capture_time within the 1 ms
    rounding; on the camera clock the change lasts 0.800093 s."""
    code, text = run(["--dataset", fx.DATASET, "--out", tmp_path,
                      "--transition", "p1_cylindrical", 4])
    assert code == 0, text
    (row,) = fx.read_csv(tmp_path / "transitions.csv")
    assert (row["movement_start_frame_id"], row["settled_frame_id"]) == ("871", "943")
    assert abs(float(row["movement_start_capture_residual_ms"])) <= 0.5
    assert abs(float(row["settled_capture_residual_ms"])) <= 0.5
    assert float(row["movement_start_camera_s"]) == pytest.approx(0.754881, abs=1e-6)
    assert float(row["settled_camera_s"]) == pytest.approx(1.554974, abs=1e-6)
    assert float(row["transition_duration_metadata_s"]) == 0.8
    assert float(row["transition_duration_camera_s"]) == pytest.approx(0.800093, abs=1e-6)
    # the cue came 0.753 ms before the first frame's capture (metadata: -0.001
    # after rounding); carried onto the camera timeline by that frame
    assert float(row["cue_time_metadata_s"]) == -0.001
    assert float(row["cue_time_camera_s"]) == pytest.approx(-0.000753, abs=1e-6)
    assert float(row["recording_camera_s"]) == pytest.approx(4.266214, abs=1e-6)
    assert float(row["recording_writer_span_s"]) == pytest.approx(4.045806, abs=1e-6)
