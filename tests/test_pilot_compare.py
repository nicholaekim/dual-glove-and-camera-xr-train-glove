"""scripts/pilot_compare.py: the pilot's camera-against-glove comparison
(docs/pilot_hand_on_object.md, step 10), on synthetic takes in the exact
layout scripts/record_simultaneous.py writes:

    <run>/leap/<stem>.jsonl, <run>/leap/<stem>.meta.json, <run>/glove/<stem>.jsonl

The hands are built with test_joint_frames.made_line: each of the eight
flexion joints (index, middle, ring, little; MCP and PIP) gets the angle a
test prescribes as a function of the time into the take, so every number the
script reports is known before it runs. Camera at 90 Hz, glove at 60 Hz,
capture_time 6 ms before wall_time on both, as the recorders stamp them.
"""
import csv
import functools
import importlib.util
import json
import re
from pathlib import Path

import pytest

from test_joint_frames import made_line
from xr_hand import joint_frames as jf

REPO = Path(__file__).resolve().parents[1]
J8 = ["I_MCP_fe", "I_PIP", "M_MCP_fe", "M_PIP",
      "R_MCP_fe", "R_PIP", "L_MCP_fe", "L_PIP"]
BASE = dict(zip(J8, (20.0, 30.0, 25.0, 35.0, 30.0, 40.0, 35.0, 45.0)))
T0 = 1_792_000_000.0
CAPTURE_LEAD = 0.006                  # capture_time = wall_time - 6 ms
CAM_HZ, GLOVE_HZ = 90, 60
FOLDER = {"A": "A_mimed", "B1": "B1_object", "C": "C_object_hint",
          "B2": "B2_object", "A2": "A_mimed_again"}
TOL = 0.05
TAKE_COLUMNS = (
    ["run", "condition", "grasp", "take", "file", "hand", "note",
     "hints_confirmed", "acceptance_policy", "hold_completed",
     "cam_tracked_fraction", "cam_reacquisitions", "tracking_note",
     "status", "clock", "window_start", "window_end",
     "window_end_source", "n_cam_frames_window", "n_glove_frames_window",
     "n_pairs", "n_cam_unpaired", "cam_coverage", "cam_max_gap_ms",
     "cam_gaps_over_100ms", "glove_coverage", "glove_max_gap_ms",
     "glove_gaps_over_100ms", "absdiff_med_all8", "absdiff_max_all8"]
    + [f"absdiff_med_{j}" for j in J8] + [f"signed_med_{j}" for j in J8]
    + [f"cam_med_{j}" for j in J8] + [f"glove_med_{j}" for j in J8]
    + [f"cam_mad_{j}" for j in J8] + [f"glove_mad_{j}" for j in J8]
    + ["cam_mad_med", "cam_mad_max", "glove_mad_med", "glove_mad_max"])


def load_compare():
    spec = importlib.util.spec_from_file_location(
        "repo_script_pilot_compare", REPO / "scripts" / "pilot_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- synthetic takes ----------------------------------------------------------------
def shifted(angles, by):
    return {j: a + by for j, a in angles.items()}


@functools.lru_cache(maxsize=None)
def _joints(hand, angles):
    """The 26 joints of a hand whose eight flexion joints read `angles`."""
    rel = {jf.DOF24_SOURCE[j][0]: jf.rotation_from_angles(a, 0.0, 0.0, hand)
           for j, a in zip(J8, angles)}
    return made_line(hand, rel)["joints"]


def frame_line(hand, angles, k, wall, pose, take, camera):
    line = {"wall_time": wall, "timestamp": float(k), "packet_counter": k,
            "hand_side": hand, "frame_id": k, "status": 1,
            "joints": _joints(hand, tuple(angles[j] for j in J8)),
            "pose": pose, "take": take, "capture_time": wall - CAPTURE_LEAD}
    if camera:
        line.update(source="leap", units="m", hand_id=1, framerate=90.0)
    return line


def write_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d) + "\n" for d in lines),
                    encoding="utf-8")


def make_take(run_dir, pose="cylindrical_object", take=1, hand="left",
              length=5.0, glove=lambda t: BASE, cam=None, offset=10.0,
              hints_confirmed=False, edit=None, meta_extra=None):
    """One accepted take as the recorder leaves it. `glove(t)` and `cam(t)`
    give the eight angles t seconds into the take (the camera's default: the
    glove's plus `offset` at every joint); `edit(cam_lines, glove_lines)` may
    change the lines in place before they are written; `meta_extra` is added
    to the meta (no hold_start / hold_end unless given there)."""
    cam = cam or (lambda t: shifted(glove(t), offset))
    stem = f"{pose}_{hand}_take{take}_20261021_120000"
    cam_lines = [frame_line(hand, cam(k / CAM_HZ), k, T0 + k / CAM_HZ,
                            pose, take, True)
                 for k in range(round(length * CAM_HZ))]
    glove_lines = [frame_line(hand, glove(k / GLOVE_HZ), k, T0 + k / GLOVE_HZ,
                              pose, take, False)
                   for k in range(round(length * GLOVE_HZ))]
    if edit:
        edit(cam_lines, glove_lines)
    run_dir = Path(run_dir)
    write_jsonl(run_dir / "leap" / f"{stem}.jsonl", cam_lines)
    write_jsonl(run_dir / "glove" / f"{stem}.jsonl", glove_lines)
    meta = {"accepted": True, "hand": hand, "pose": pose, "take": take,
            "coverage": 1.0, "cam_frames": len(cam_lines),
            "glove_frames": len(glove_lines), "file": f"{stem}.jsonl",
            "tracking_hints": ["hand_on_object"] if hints_confirmed else [],
            "hints_confirmed": hints_confirmed,
            "license_hyperion": True if hints_confirmed else None,
            "note": "synthetic take", **(meta_extra or {})}
    (run_dir / "leap" / f"{stem}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return stem


def rejected_meta(run_dir, pose, take, attempt, why, coverage, hand="left"):
    """A refused attempt's meta where the recorder moves it:
    <run>/rejected/leap/<stem>_attempt<N>.meta.json, accepted false."""
    stem = f"{pose}_{hand}_take{take}_20261021_120000_attempt{attempt}"
    folder = Path(run_dir) / "rejected" / "leap"
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"accepted": False, "hand": hand, "pose": pose, "take": take,
            "coverage": coverage, "why": why, "file": f"{stem}.jsonl",
            "tracking_hints": [], "hints_confirmed": False,
            "license_hyperion": None, "note": "synthetic attempt"}
    (folder / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2),
                                              encoding="utf-8")


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compare(tmp_path, runs, *extra):
    """main() over `runs` {label: folder} -> (return code, takes.csv rows, out)."""
    out = tmp_path / "out"
    argv = [a for label, path in runs.items()
            for a in ("--run", f"{label}={path}")]
    code = load_compare().main([*argv, "--out", str(out), *map(str, extra)])
    return code, read_rows(out / "takes.csv"), out


def num(row, key):
    return float(row[key])


def assert_diff_everywhere(row, value):
    for j in J8:
        assert num(row, f"absdiff_med_{j}") == pytest.approx(value, abs=TOL), j
        assert num(row, f"signed_med_{j}") == pytest.approx(value, abs=TOL), j
    assert num(row, "absdiff_med_all8") == pytest.approx(value, abs=TOL)
    assert num(row, "absdiff_max_all8") == pytest.approx(value, abs=TOL)


# --- one take -----------------------------------------------------------------------
def test_the_eight_angles_read_back_from_both_sensors(tmp_path):
    run = tmp_path / FOLDER["A"]
    make_take(run, offset=10.0)
    code, rows, out = compare(tmp_path, {"A": run})
    assert code == 0
    with open(out / "takes.csv", encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == TAKE_COLUMNS
    (row,) = rows
    assert row["status"] == "ok"
    assert row["clock"] == "capture_time"
    assert (row["run"], row["condition"], row["grasp"], row["take"],
            row["hand"]) == ("A", "A", "cylindrical_object", "1", "left")
    assert_diff_everywhere(row, 10.0)
    for j in J8:
        assert num(row, f"glove_med_{j}") == pytest.approx(BASE[j], abs=TOL)
        assert num(row, f"cam_med_{j}") == pytest.approx(BASE[j] + 10, abs=TOL)
        assert num(row, f"cam_mad_{j}") == pytest.approx(0.0, abs=TOL)
    assert abs(int(row["n_pairs"]) - 270) <= 2
    assert int(row["n_pairs"]) == int(row["n_cam_frames_window"])
    assert int(row["n_cam_unpaired"]) == 0
    assert num(row, "cam_coverage") == 1.0
    assert num(row, "glove_coverage") == 1.0


def test_the_window_is_the_last_3_s_on_the_shared_clock(tmp_path):
    run = tmp_path / FOLDER["A"]
    make_take(run, cam=lambda t: shifted(BASE, 40.0 if t < 2.0 else 10.0))
    code, (row,), _ = compare(tmp_path, {"A": run})
    assert code == 0 and row["status"] == "ok"
    assert_diff_everywhere(row, 10.0)                  # not 22, not 40
    last_cam = T0 + (5 * CAM_HZ - 1) / CAM_HZ - CAPTURE_LEAD
    assert num(row, "window_end") == pytest.approx(last_cam, abs=0.001)
    assert num(row, "window_start") == pytest.approx(
        num(row, "window_end") - 3.0, abs=0.002)


def test_the_window_ends_at_the_later_of_the_two_last_stamps(tmp_path):
    run = tmp_path / FOLDER["A"]

    def camera_stops_early(cam, glove):
        cam[:] = [d for d in cam
                  if d["capture_time"] <= glove[-1]["capture_time"] - 1.0]

    make_take(run, edit=camera_stops_early)            # meta: no hold_end
    code, (row,), _ = compare(tmp_path, {"A": run})
    assert code == 0 and row["status"] == "ok"
    assert row["window_end_source"] == "later last stamp"
    last_glove = T0 + (5 * GLOVE_HZ - 1) / GLOVE_HZ - CAPTURE_LEAD
    assert num(row, "window_end") == pytest.approx(last_glove, abs=0.001)
    assert num(row, "cam_coverage") == pytest.approx(2 / 3, abs=0.02)
    assert num(row, "cam_max_gap_ms") == pytest.approx(1000, abs=30)
    assert int(row["cam_gaps_over_100ms"]) == 1
    assert abs(int(row["n_pairs"]) - 180) <= 2
    assert num(row, "glove_coverage") == 1.0
    assert_diff_everywhere(row, 10.0)


def test_the_window_ends_at_the_hold_end_the_meta_gives(tmp_path):
    run = tmp_path / FOLDER["A"]
    # The camera reads 50 deg further after the hold's end: inside a window
    # that ended at the last stamp (1.98 s .. 4.98 s) those frames would be
    # the majority.
    hold_end = T0 + 3.2
    make_take(run, cam=lambda t: shifted(BASE, 10.0 if t <= 3.2 else 60.0),
              meta_extra={"hold_start": T0 - 1.8, "hold_end": hold_end})
    code, (row,), _ = compare(tmp_path, {"A": run})
    assert code == 0 and row["status"] == "ok"
    assert row["window_end_source"] == "meta hold_end"
    assert num(row, "window_end") == pytest.approx(hold_end, abs=0.001)
    assert num(row, "window_start") == pytest.approx(hold_end - 3.0, abs=0.001)
    assert_diff_everywhere(row, 10.0)


def test_pairing_is_camera_to_nearest_glove_within_max_dt(tmp_path):
    run = tmp_path / FOLDER["A"]

    def glove_hole(cam, glove):        # glove frames 2.5 s <= t < 3.0 s gone
        glove[:] = [d for d in glove if not 150 <= d["frame_id"] < 180]

    make_take(run, edit=glove_hole)
    code, (row,), _ = compare(tmp_path, {"A": run})
    assert code == 0 and row["status"] == "ok"
    # The hole runs from the glove frame at 149/60 s to the one at 3.0 s. A
    # camera frame is unpaired when both are more than 50 ms away from it.
    hole = 3.0 - 149 / GLOVE_HZ
    unpaired = int(row["n_cam_unpaired"])
    assert abs(unpaired - (hole - 2 * 0.05) * CAM_HZ) <= 2
    assert int(row["n_pairs"]) == int(row["n_cam_frames_window"]) - unpaired
    assert num(row, "glove_coverage") < 1.0
    assert int(row["glove_gaps_over_100ms"]) == 1
    assert num(row, "cam_coverage") == 1.0
    assert_diff_everywhere(row, 10.0)

    # Half the hole is under 300 ms: every camera frame finds a glove frame.
    code, (row,), _ = compare(tmp_path, {"A": run}, "--max-dt", "0.3")
    assert int(row["n_cam_unpaired"]) == 0
    assert int(row["n_pairs"]) == int(row["n_cam_frames_window"])


def test_the_signed_disagreement_is_camera_minus_glove_wrapped(tmp_path):
    run = tmp_path / FOLDER["A"]
    glove = dict(BASE, I_MCP_fe=175.0)
    cam = dict(shifted(BASE, -10.0), I_MCP_fe=-175.0)  # 10 deg past 180
    make_take(run, length=1.2, glove=lambda t: glove, cam=lambda t: cam)
    code, (row,), _ = compare(tmp_path, {"A": run}, "--window", "1.0")
    assert code == 0 and row["status"] == "ok"
    for j in J8:
        assert num(row, f"absdiff_med_{j}") == pytest.approx(10.0, abs=TOL), j
        assert num(row, f"signed_med_{j}") == pytest.approx(
            10.0 if j == "I_MCP_fe" else -10.0, abs=TOL), j
    assert num(row, "absdiff_max_all8") == pytest.approx(10.0, abs=TOL)


def test_held_pose_stability_is_the_mad_of_each_sensor(tmp_path):
    run = tmp_path / FOLDER["A"]

    def cam(t):                         # M_PIP +5, -5, +5, ... frame to frame
        k = round(t * CAM_HZ)
        return dict(shifted(BASE, 10.0),
                    M_PIP=BASE["M_PIP"] + 10.0 + (5.0 if k % 2 else -5.0))

    # The window's end halfway between two camera frames, so it holds an even
    # number of them, as many +5 as -5 (an odd count would put the median on
    # one side and the MAD at 0).
    hold_end = T0 + 400.5 / CAM_HZ - CAPTURE_LEAD
    make_take(run, cam=cam, meta_extra={"hold_end": hold_end})
    code, (row,), _ = compare(tmp_path, {"A": run})
    assert code == 0 and row["status"] == "ok"
    assert int(row["n_pairs"]) == 270
    assert num(row, "cam_mad_M_PIP") == pytest.approx(5.0, abs=TOL)
    assert num(row, "glove_mad_M_PIP") == pytest.approx(0.0, abs=TOL)
    assert num(row, "cam_mad_max") == pytest.approx(5.0, abs=TOL)
    assert num(row, "cam_mad_med") == pytest.approx(0.0, abs=TOL)
    assert num(row, "glove_mad_max") == pytest.approx(0.0, abs=TOL)


def test_the_clock_falls_back_to_wall_time_when_a_line_lacks_capture_time(
        tmp_path):
    run = tmp_path / FOLDER["A"]

    def one_line_without(cam, glove):
        del glove[100]["capture_time"]

    make_take(run, edit=one_line_without)
    code, (row,), _ = compare(tmp_path, {"A": run})
    assert code == 0
    assert row["status"] == "ok"
    assert row["clock"] == "wall_time"
    assert_diff_everywhere(row, 10.0)


def test_a_line_of_the_other_hand_is_ignored(tmp_path):
    run = tmp_path / FOLDER["A"]
    wild = {j: 170.0 for j in J8}

    def other_hand(cam, glove):
        for lines, camera in ((cam, True), (glove, False)):
            lines.append(frame_line("right", wild, 9999, T0 + 1.0,
                                    "cylindrical_object", 2, camera))

    make_take(run, take=1, length=1.2)
    make_take(run, take=2, length=1.2, edit=other_hand)
    code, (plain, mixed), _ = compare(tmp_path, {"A": run}, "--window", "1.0")
    assert code == 0
    assert plain["status"] == mixed["status"] == "ok"
    for key in ("n_cam_frames_window", "n_glove_frames_window", "n_pairs"):
        assert mixed[key] == plain[key], key
    assert_diff_everywhere(mixed, 10.0)


# --- runs -----------------------------------------------------------------------------
def test_a_take_whose_hints_contradict_its_run_is_listed_and_left_out(
        tmp_path):
    root = tmp_path / "pilot"
    for label, take, hints in (("A", 1, False), ("A", 2, True), ("C", 1, False)):
        make_take(root / FOLDER[label], take=take, length=1.2,
                  hints_confirmed=hints)
    out = tmp_path / "out"
    assert load_compare().main(["--root", str(root), "--out", str(out),
                                "--window", "1.0"]) == 0

    rows = {(r["run"], r["take"]): r for r in read_rows(out / "takes.csv")}
    assert rows[("A", "1")]["status"] == "ok"
    for key in (("A", "2"), ("C", "1")):
        assert rows[key]["status"].startswith("invalid: hints_confirmed"), key
        assert rows[key]["absdiff_med_all8"] != ""   # nothing hidden
    assert "expected True" in rows[("C", "1")]["status"]
    assert "expected False" in rows[("A", "2")]["status"]

    # Counted in the summary, never measured in it.
    summary = {r["run"]: r for r in read_rows(out / "summary.csv")
               if r["level"] == "run_grasp"
               and r["grasp"] == "cylindrical_object"}
    a, c = summary["A"], summary["C"]
    assert (a["n_valid_takes"], a["n_invalid_takes"]) == ("1", "1")
    assert (c["n_valid_takes"], c["n_invalid_takes"]) == ("0", "1")
    assert c["absdiff_median"] == "" and c["absdiff_I_MCP_fe"] == ""

    report = (out / "report.txt").read_text(encoding="ascii")
    invalid = report.split("invalid takes:")[1].split("\n\n")[0]
    assert "A cylindrical_object take 2: invalid: hints_confirmed" in invalid
    assert "C cylindrical_object take 1: invalid: hints_confirmed" in invalid
    assert "take 1: ok" not in invalid


OFFSETS = {"A": 5.0, "B1": 20.0, "C": 12.0, "B2": 22.0, "A2": 6.0}
N_TAKES = {"A": 2, "B1": 2, "C": 2, "B2": 1, "A2": 1}
EXPECTED = {"A": 5, "B1": 3, "C": 5, "B2": 2, "A2": 1}
RUN_ALL_EXPECTED = {"A": 15, "B1": 9, "C": 15, "B2": 6, "A2": 3}
GRASPS = ["cylindrical_object", "spherical_object", "tip_object"]
CONDITION = {"A": "A", "B1": "B", "C": "C", "B2": "B", "A2": "A"}
CONTRASTS = {"Bpooled_minus_A": 15.0, "C_minus_Bpooled": -8.0,
             "B2_minus_B1": 2.0, "A2_minus_A": 1.0,
             "abs_B2_minus_B1_minus_abs_C_minus_Bpooled": -6.0}


def test_the_summary_keeps_runs_apart_pools_b_and_reads_glove_drift(
        tmp_path):
    root = tmp_path / "pilot"
    for label, n in N_TAKES.items():
        angles = shifted(BASE, 3.0) if label == "A2" else BASE
        for grasp in GRASPS:
            for take in range(1, n + 1):
                make_take(root / FOLDER[label], grasp, take, length=0.7,
                          glove=lambda t, g=angles: g, offset=OFFSETS[label],
                          hints_confirmed=(label == "C"))
    # one invalid take (hint confirmed outside C) with a far-off camera, and
    # one too short to compare: both counted, neither measured
    make_take(root / FOLDER["B2"], "tip_object", 2, length=0.7, offset=90.0,
              hints_confirmed=True)
    make_take(root / FOLDER["A"], "spherical_object", 3, length=0.3)
    out = tmp_path / "out"
    assert load_compare().main(["--root", str(root), "--out", str(out),
                                "--window", "0.5"]) == 0
    summary = read_rows(out / "summary.csv")

    def rows(level):
        return [r for r in summary if r["level"] == level]

    by_run_grasp = {(r["grasp"], r["run"]): r for r in rows("run_grasp")}
    assert set(by_run_grasp) == {(g, l) for g in GRASPS for l in OFFSETS}
    for (grasp, label), r in by_run_grasp.items():
        assert r["condition"] == CONDITION[label]
        assert int(r["n_valid_takes"]) == N_TAKES[label]
        assert int(r["n_expected_takes"]) == EXPECTED[label]
        for key in (["absdiff_median", "absdiff_largest"]
                    + [f"absdiff_{j}" for j in J8]
                    + [f"signed_{j}" for j in J8]):
            assert num(r, key) == pytest.approx(OFFSETS[label], abs=TOL), key
        assert num(r, "cam_coverage_min") == 1.0
        assert int(r["cam_gaps_over_100ms_total"]) == 0
        assert num(r, "paired_fraction_median") == 1.0
        assert num(r, "paired_fraction_min") == 1.0
        assert int(r["n_wall_time_clock"]) == 0
        assert int(r["n_later_last_stamp"]) == N_TAKES[label]   # no hold_end
        assert num(r, "cam_mad_max_median") == pytest.approx(0.0, abs=TOL)
    assert by_run_grasp[("tip_object", "B2")]["n_invalid_takes"] == "1"
    assert by_run_grasp[("cylindrical_object", "B2")]["n_invalid_takes"] == "0"
    assert by_run_grasp[("spherical_object", "A")]["n_unusable_takes"] == "1"
    assert by_run_grasp[("tip_object", "A")]["n_unusable_takes"] == "0"
    # missing = expected - every take recorded, invalid and unusable included
    for (grasp, label), r in by_run_grasp.items():
        extra = {("tip_object", "B2"), ("spherical_object", "A")}
        assert int(r["n_missing_takes"]) == (
            EXPECTED[label] - N_TAKES[label] - ((grasp, label) in extra)),             (grasp, label)
    assert by_run_grasp[("cylindrical_object", "A")]["n_missing_takes"] == "3"

    pooled = {r["grasp"]: r for r in rows("run_grasp_pooledB")}
    assert set(pooled) == set(GRASPS)
    for r in pooled.values():
        assert (r["run"], r["condition"]) == ("B", "B")
        assert (r["n_valid_takes"], r["n_expected_takes"]) == ("3", "5")
        assert num(r, "absdiff_median") == pytest.approx(20.0, abs=TOL)
        assert num(r, "absdiff_largest") == pytest.approx(22.0, abs=TOL)
    assert pooled["tip_object"]["n_invalid_takes"] == "1"
    assert pooled["tip_object"]["n_missing_takes"] == "1"
    assert pooled["cylindrical_object"]["n_missing_takes"] == "2"

    pooled_all = {r["run"]: r for r in rows("run_all")}
    assert set(pooled_all) == set(OFFSETS)
    # 9 and 3 missing for A and B2 by the plain 2/1 takes per grasp, less the
    # unusable A take and the invalid B2 take, which were recorded
    missing = {"A": 8, "B1": 3, "C": 9, "B2": 2, "A2": 0}
    for label, r in pooled_all.items():
        assert int(r["n_valid_takes"]) == 3 * N_TAKES[label]
        assert int(r["n_expected_takes"]) == RUN_ALL_EXPECTED[label]
        assert int(r["n_missing_takes"]) == missing[label], label
        assert num(r, "absdiff_median") == pytest.approx(OFFSETS[label],
                                                         abs=TOL)
    (b_all,) = rows("run_all_pooledB")
    assert (b_all["n_valid_takes"], b_all["n_expected_takes"],
            b_all["n_invalid_takes"], b_all["n_missing_takes"]) == (
                "9", "15", "1", "5")

    drift = {r["grasp"]: r for r in rows("glove_drift_A2_minus_A")}
    assert set(drift) == set(GRASPS)
    for r in drift.values():
        assert r["n_valid_takes"] == "2+1"
        for j in J8:
            assert num(r, f"signed_{j}") == pytest.approx(3.0, abs=TOL), j
        assert num(r, "absdiff_median") == pytest.approx(3.0, abs=TOL)
        assert num(r, "absdiff_largest") == pytest.approx(3.0, abs=TOL)
        assert r["cam_coverage_median"] == r["glove_coverage_min"] == ""

    contrasts = {(r["grasp"], r["run"]): r for r in rows("contrast")}
    assert set(contrasts) == {(g, name) for g in GRASPS + ["all"]
                              for name in CONTRASTS}
    for (grasp, name), r in contrasts.items():
        assert num(r, "absdiff_median") == pytest.approx(CONTRASTS[name],
                                                         abs=TOL), (grasp, name)
        assert r["condition"] == r["n_valid_takes"] == ""

    report = (out / "report.txt").read_text(encoding="ascii")
    report_lines = report.splitlines()
    for grasp in GRASPS:
        at = report_lines.index(f"== {grasp} ==")
        header = next(s for s in report_lines[at + 1:] if s.startswith(" "))
        assert header.split() == ["A", "B1", "C", "B2", "A2", "B(pooled)"]
    assert "== glove drift, A2 minus A, glove alone (deg) ==" in report_lines
    assert "== all grasps pooled (secondary) ==" in report_lines
    assert report.count("-- contrasts on the PRIMARY quantity, deg "
                        "(numbers only) --") == len(GRASPS) + 1
    assert "PRIMARY: median of take x joint |camera - glove|" in report
    assert not re.search(r"\b(better|worse|good|bad|error)\b", report.lower())


def test_a_protocol_grasp_with_no_takes_still_has_its_rows_and_block(
        tmp_path):
    """The grid is the protocol's, never the grasps seen: tip_object, never
    recorded, shows as missing; open_palm, recorded but not in the protocol,
    follows the three with nothing expected of it."""
    root = tmp_path / "pilot"
    for grasp in ("cylindrical_object", "spherical_object", "open_palm"):
        make_take(root / FOLDER["A"], grasp, 1, length=0.7, offset=5.0)
    make_take(root / FOLDER["C"], "cylindrical_object", 1, length=0.7,
              offset=12.0, hints_confirmed=True)
    out = tmp_path / "out"
    assert load_compare().main(["--root", str(root), "--out", str(out),
                                "--window", "0.5"]) == 0
    summary = read_rows(out / "summary.csv")
    grid = {(r["grasp"], r["run"]): r for r in summary
            if r["level"] in ("run_grasp", "run_grasp_pooledB")}
    assert set(grid) == {(g, l) for g in GRASPS + ["open_palm"]
                         for l in list(OFFSETS) + ["B"]}
    for label in list(OFFSETS) + ["B"]:
        r = grid[("tip_object", label)]
        expected = EXPECTED.get(label, 5)
        assert (r["n_valid_takes"], r["n_invalid_takes"],
                r["n_unusable_takes"]) == ("0", "0", "0"), label
        assert int(r["n_expected_takes"]) == expected
        assert int(r["n_missing_takes"]) == expected
        assert r["absdiff_median"] == r["cam_coverage_min"] == ""
        extra = grid[("open_palm", label)]
        assert (extra["n_expected_takes"], extra["n_missing_takes"]) == (
            "0", "0")
    assert grid[("open_palm", "A")]["n_valid_takes"] == "1"
    assert grid[("cylindrical_object", "B1")]["n_missing_takes"] == "3"

    run_all = {r["run"]: r for r in summary
               if r["level"] in ("run_all", "run_all_pooledB")}
    assert {k: int(r["n_expected_takes"]) for k, r in run_all.items()} == {
        **RUN_ALL_EXPECTED, "B": 15}
    # the extra grasp's take is measured in run_all, but expected of nobody
    assert run_all["A"]["n_valid_takes"] == "3"
    assert run_all["A"]["n_missing_takes"] == "12"
    assert run_all["B1"]["absdiff_median"] == ""

    lines = (out / "report.txt").read_text(encoding="ascii").splitlines()
    at = [lines.index(f"== {g} ==") for g in GRASPS + ["open_palm"]]
    assert at == sorted(at)
    block = lines[at[2]:at[3]]                          # tip_object
    assert block[1].split() == ["A", "B1", "C", "B2", "A2", "B(pooled)"]
    (primary,) = [s for s in block if s.startswith("PRIMARY:")]
    assert primary.split()[-6:] == ["-"] * 6
    (missing,) = [s for s in block if s.startswith("takes missing")]
    assert missing.split()[-6:] == ["5", "3", "5", "2", "1", "5"]


def test_rejected_attempts_are_counted_by_reason_beside_the_kept_takes(
        tmp_path):
    """In the object conditions the camera losing the hand shows as refused
    attempts under rejected/leap/, not as low coverage in the kept takes."""
    root = tmp_path / "pilot"
    run_a = root / FOLDER["A"]
    for take in (1, 2):
        make_take(run_a, "cylindrical_object", take, length=0.7)
    for n, (why, cov) in enumerate([
            ("the left hand was lost for 0.5 s during the take", 0.0),
            ("the tracker let go and re-acquired the hand (id 26 -> 27) "
             "during the take", 0.95),
            ("the left hand covered only 73 % of the take (need 90 %)", 0.73),
    ], start=1):
        rejected_meta(run_a, "cylindrical_object", 3, n, why, cov)
    make_take(run_a, "spherical_object", 1, length=0.7)
    rejected_meta(run_a, "spherical_object", 2, 1,
                  "both sensors saw a FIST, expected OPEN PALM", 1.0)
    for take in (1, 2):                              # no rejected folder
        make_take(root / FOLDER["B1"], "cylindrical_object", take, length=0.7)
    out = tmp_path / "out"
    assert load_compare().main(["--root", str(root), "--out", str(out),
                                "--window", "0.5"]) == 0
    rows = {(r["level"], r["run"], r["grasp"]): r
            for r in read_rows(out / "summary.csv")}

    def counts(r):
        return tuple(int(r[k]) for k in (
            "n_attempts", "n_rejected", "n_rejected_hand_lost",
            "n_rejected_reacquired", "n_rejected_low_coverage",
            "n_rejected_other"))

    a = rows[("run_grasp", "A", "cylindrical_object")]
    assert counts(a) == (5, 3, 1, 1, 1, 0)
    assert a["n_valid_takes"] == "2"                 # attempts are not takes
    assert num(a, "attempt_coverage_median") == 0.95  # 1, 1, 0, 0.95, 0.73
    assert num(a, "attempt_coverage_min") == 0.0
    assert counts(rows[("run_grasp", "A", "spherical_object")]) == (
        2, 1, 0, 0, 0, 1)
    b1 = rows[("run_grasp", "B1", "cylindrical_object")]
    assert counts(b1) == (2, 0, 0, 0, 0, 0)
    assert num(b1, "attempt_coverage_median") == num(
        b1, "attempt_coverage_min") == 1.0
    assert counts(rows[("run_grasp_pooledB", "B", "cylindrical_object")]) == (
        2, 0, 0, 0, 0, 0)
    assert counts(rows[("run_all", "A", "all")]) == (7, 4, 1, 1, 1, 1)
    tip = rows[("run_grasp", "A", "tip_object")]
    assert counts(tip) == (0, 0, 0, 0, 0, 0)
    assert tip["attempt_coverage_median"] == tip["attempt_coverage_min"] == ""

    report = (out / "report.txt").read_text(encoding="ascii")
    assert ("attempts: every attempt the recorder made, kept or rejected "
            "(rejected/leap), with the recorder's own coverage of each"
            in report.splitlines())
    lines = report.splitlines()
    block = lines[lines.index("== cylindrical_object =="):
                  lines.index("== spherical_object ==")]

    def cells(label):
        (line,) = [s for s in block if s.startswith(label)]
        return line.split()[-6:]                     # A B1 C B2 A2 B(pooled)

    assert cells("attempts (kept + rejected)") == ["5", "2", "0", "0", "0",
                                                   "2"]
    assert cells("rejected: hand lost") == ["1", "0", "0", "0", "0", "0"]
    assert cells("rejected: re-acquired") == ["1", "0", "0", "0", "0", "0"]
    assert cells("rejected: low coverage") == ["1", "0", "0", "0", "0", "0"]
    assert cells("rejected: other") == ["0", "0", "0", "0", "0", "0"]
    assert cells("coverage over all attempts median") == [
        "0.95", "1.00", "-", "-", "-", "1.00"]
    assert cells("coverage over all attempts min") == [
        "0.00", "1.00", "-", "-", "-", "1.00"]
    takes_missing = next(i for i, s in enumerate(block)
                         if s.startswith("takes missing"))
    assert block[takes_missing + 1].startswith("attempts (kept + rejected)")


def test_a_run_whose_every_attempt_was_rejected_is_not_called_absent(
        tmp_path, capsys):
    """The object run the camera lost every time is recorded, not absent."""
    root = tmp_path / "pilot"
    make_take(root / FOLDER["A"], length=0.7)
    for n in (1, 2):
        rejected_meta(root / FOLDER["B1"], "cylindrical_object", 1, n,
                      "the left hand was lost for 0.5 s during the take", 0.1)
    out = tmp_path / "out"
    assert load_compare().main(["--root", str(root), "--out", str(out),
                                "--window", "0.5"]) == 0
    said = "B1 (B1_object): 0 takes kept (2 attempts, all rejected)"
    text = capsys.readouterr().out
    assert f"run {said}" in text.splitlines()
    assert "run C (C_object_hint): absent" in text.splitlines()
    report = (out / "report.txt").read_text(encoding="ascii").splitlines()
    assert said in report
    assert "C (C_object_hint): absent" in report
    rows = {(r["level"], r["run"], r["grasp"]): r
            for r in read_rows(out / "summary.csv")}
    for key in (("run_grasp", "B1", "cylindrical_object"),
                ("run_all", "B1", "all")):
        r = rows[key]
        assert (r["n_valid_takes"], r["n_attempts"], r["n_rejected"]) == (
            "0", "2", "2"), key


def test_attempts_alone_are_still_written_out(tmp_path, capsys):
    """No take kept anywhere, but attempts were made: the counts are the
    result, so the three files are written; the return is 1 (nothing
    compared)."""
    root = tmp_path / "pilot"
    for n in (1, 2):
        rejected_meta(root / FOLDER["B1"], "cylindrical_object", 1, n,
                      "the left hand was lost for 0.5 s during the take", 0.1)
    out = tmp_path / "out"
    assert load_compare().main(["--root", str(root), "--out", str(out)]) == 1
    assert capsys.readouterr().out.splitlines()[-1].startswith(
        "compared 0 of 0 takes; wrote ")
    assert read_rows(out / "takes.csv") == []
    rows = {(r["level"], r["run"], r["grasp"]): r
            for r in read_rows(out / "summary.csv")}
    b1 = rows[("run_grasp", "B1", "cylindrical_object")]
    assert (b1["n_attempts"], b1["n_rejected"], b1["absdiff_median"]) == (
        "2", "2", "")
    report = (out / "report.txt").read_text(encoding="ascii").splitlines()
    assert "B1 (B1_object): 0 takes kept (2 attempts, all rejected)" in report


def test_the_command_line_end_to_end(tmp_path, capsys):
    root = tmp_path / "pilot"
    for label in ("A", "B1", "C", "A2"):                 # B2 never recorded
        make_take(root / FOLDER[label], length=1.2, offset=OFFSETS[label],
                  hints_confirmed=(label == "C"))
    out = tmp_path / "out"
    compare_script = load_compare()
    assert compare_script.main(["--root", str(root), "--out", str(out),
                                "--window", "1.0"]) == 0
    text = capsys.readouterr().out
    for name in ("takes.csv", "summary.csv", "report.txt"):
        assert (out / name).is_file()
    assert "A cylindrical_object take 1: ok, clock capture_time, " in text
    assert "window end later last stamp" in text
    assert "B2 (B2_object): absent" in text
    assert "compared 4 of 4 takes; wrote " in text
    assert "B2 (B2_object): absent" in (out / "report.txt").read_text(
        encoding="ascii")

    # --run A=<path> replaces A_mimed; with no --out the result goes to
    # <root>\comparison.
    other = tmp_path / "elsewhere" / "A_retake"
    make_take(other, pose="tip_object", length=1.2, offset=7.0)
    assert compare_script.main(["--root", str(root), "--run", f"A={other}",
                                "--window", "1.0"]) == 0
    rows = read_rows(root / "comparison" / "takes.csv")
    (a,) = [r for r in rows if r["run"] == "A"]
    assert a["grasp"] == "tip_object"
    assert num(a, "absdiff_med_all8") == pytest.approx(7.0, abs=TOL)
    capsys.readouterr()

    empty = tmp_path / "empty"
    empty.mkdir()
    nothing = tmp_path / "nothing_out"
    assert compare_script.main(["--root", str(empty), "--out",
                                str(nothing)]) == 1
    text = capsys.readouterr().out
    assert "nothing to compare" in text
    assert "C (C_object_hint): absent" in text
    assert not nothing.exists()


def test_a_short_take_and_a_take_with_no_glove_file_get_a_status_and_no_numbers(
        tmp_path):
    run = tmp_path / FOLDER["A"]
    make_take(run, take=1, length=2.0)
    stem = make_take(run, take=2)
    (run / "glove" / f"{stem}.jsonl").unlink()
    code, (short, lone), _ = compare(tmp_path, {"A": run})
    assert code == 1                                   # nothing compared
    assert short["status"].startswith("take shorter than window")
    assert lone["status"] == "no glove file"
    for row in (short, lone):
        for key in ("n_pairs", "cam_coverage", "absdiff_med_all8",
                    "absdiff_med_I_MCP_fe", "signed_med_I_MCP_fe",
                    "cam_med_L_PIP", "glove_med_L_PIP", "cam_mad_max"):
            assert row[key] == "", key


def test_the_recorders_acceptance_policy_is_read_per_take(tmp_path):
    """The amendment of 2026-10-23: object takes recorded with
    --keep-all-tracking say so in their meta, with the camera's tracking over
    the hold. takes.csv carries it; the summary and the report give its
    median, minimum and worst re-acquisition count over the valid takes, and
    how many were kept under each policy. A meta without the keys (recorded
    before them) leaves the columns empty and counts as strict."""
    root = tmp_path / "pilot"
    for take in (1, 2):                              # recorded before the keys
        make_take(root / FOLDER["A"], "cylindrical_object", take, length=0.7)
    note = "the left hand was lost for 0.7 s during the take"
    keep_all = {"acceptance_policy": "object_keep_all_tracking",
                "protocol_amendment_date": "2026-10-23",
                "hold_completed": True}
    run_b1 = root / FOLDER["B1"]
    make_take(run_b1, "cylindrical_object", 1, length=0.7, meta_extra={
        **keep_all, "cam_tracked_fraction": 0.8, "cam_reacquisitions": 1,
        "tracking_note": note})
    make_take(run_b1, "cylindrical_object", 2, length=0.7, meta_extra={
        **keep_all, "cam_tracked_fraction": 0.6, "cam_reacquisitions": 2,
        "tracking_note": note})
    make_take(run_b1, "cylindrical_object", 3, length=0.7, meta_extra={
        "acceptance_policy": "strict", "protocol_amendment_date": None,
        "hold_completed": True, "cam_tracked_fraction": 1.0,
        "cam_reacquisitions": 0, "tracking_note": ""})
    # kept although its hold was cut short: should not exist; not usable
    make_take(run_b1, "spherical_object", 1, length=0.7, meta_extra={
        "acceptance_policy": "strict", "hold_completed": False,
        "cam_tracked_fraction": 0.3, "cam_reacquisitions": 5})
    out = tmp_path / "out"
    assert load_compare().main(["--root", str(root), "--out", str(out),
                                "--window", "0.5"]) == 0

    takes = {(r["run"], r["grasp"], r["take"]): r
             for r in read_rows(out / "takes.csv")}
    first = takes[("B1", "cylindrical_object", "1")]
    assert (first["acceptance_policy"], first["hold_completed"],
            first["cam_tracked_fraction"], first["cam_reacquisitions"],
            first["tracking_note"], first["status"]) == (
                "object_keep_all_tracking", "True", "0.8", "1", note, "ok")
    old = takes[("A", "cylindrical_object", "1")]
    assert [old[k] for k in ("acceptance_policy", "hold_completed",
                             "cam_tracked_fraction", "cam_reacquisitions",
                             "tracking_note")] == [""] * 5
    assert old["status"] == "ok"
    cut = takes[("B1", "spherical_object", "1")]
    assert cut["status"] == "unusable: hold not completed"
    assert cut["absdiff_med_all8"] != "", "measured, but left out"

    summary = read_rows(out / "summary.csv")
    with open(out / "summary.csv", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    at = header.index("attempt_coverage_min")
    assert header[at + 1:at + 6] == [
        "cam_tracked_fraction_median", "cam_tracked_fraction_min",
        "cam_reacquisitions_worst", "n_strict_policy_takes",
        "n_keep_all_takes"]
    rows = {(r["level"], r["run"], r["grasp"]): r for r in summary}

    def policy(r):
        return (r["cam_tracked_fraction_median"], r["cam_tracked_fraction_min"],
                r["cam_reacquisitions_worst"], r["n_strict_policy_takes"],
                r["n_keep_all_takes"])

    for key in (("run_grasp", "B1", "cylindrical_object"),
                ("run_grasp_pooledB", "B", "cylindrical_object"),
                ("run_all", "B1", "all"), ("run_all_pooledB", "B", "all")):
        assert policy(rows[key]) == ("0.8", "0.6", "2", "1", "2"), key
    assert policy(rows[("run_grasp", "A", "cylindrical_object")]) == (
        "", "", "", "2", "0")
    spherical = rows[("run_grasp", "B1", "spherical_object")]
    assert policy(spherical) == ("", "", "", "0", "0")
    assert spherical["n_unusable_takes"] == "1"
    assert policy(rows[("run_grasp", "C", "cylindrical_object")]) == (
        "", "", "", "0", "0")

    lines = (out / "report.txt").read_text(encoding="ascii").splitlines()
    assert any(s.startswith("policy: ") for s in lines)
    block = lines[lines.index("== cylindrical_object =="):
                  lines.index("== spherical_object ==")]

    def cells(label):
        (line,) = [s for s in block if s.startswith(label)]
        return line[len(label):].split()             # A B1 C B2 A2 B(pooled)

    labels = ["camera tracked fraction of hold median",
              "camera tracked fraction of hold min",
              "camera re-acquisitions worst",
              "takes under strict / keep-all policy"]
    after = next(i for i, s in enumerate(block)
                 if s.startswith("coverage over all attempts min"))
    assert [s[:len(t)] for s, t in zip(block[after + 1:], labels)] == labels
    assert cells(labels[0]) == ["-", "0.80", "-", "-", "-", "0.80"]
    assert cells(labels[1]) == ["-", "0.60", "-", "-", "-", "0.60"]
    assert cells(labels[2]) == ["-", "2", "-", "-", "-", "2"]
    assert cells(labels[3]) == ["2/0", "1/2", "0/0", "0/0", "0/0", "1/2"]
